/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Build vertex->face CSR adjacency, unique edge list, and
 * edge->face CSR adjacency from alive faces.
 */
#include "pamo/pamo_mesh.h"

#include <stdlib.h>
#include <string.h>

/* ── Internal edge sorting helper ────────────────────────────────── */

static int edge_cmp(const void *a, const void *b) {
    const pamo_edge *ea = (const pamo_edge *)a;
    const pamo_edge *eb = (const pamo_edge *)b;
    if (ea->u != eb->u) return (ea->u < eb->u) ? -1 : 1;
    if (ea->v != eb->v) return (ea->v < eb->v) ? -1 : 1;
    return 0;
}

/* ── Build adjacency ─────────────────────────────────────────────── */

pamo_error pamo_mesh_build_adjacency(pamo_mesh *m) {
    if (!m) return PAMO_ERR_INVALID_ARG;
    pamo_allocator *a = &m->alloc;

    /* Free any previous adjacency. */
    pamo_mesh_free_adjacency(m);

    size_t nv = m->n_verts;
    size_t nf = m->n_faces;

    /* ── 1. Vertex->face CSR ─────────────────────────────────────── */
    m->vert_face_offset = PAMO_ALLOC_ARRAY(a, int32_t, nv + 1);
    if (!m->vert_face_offset) return PAMO_ERR_ALLOC;

    /* Count faces per vertex. */
    for (size_t fi = 0; fi < nf; fi++) {
        if (!m->face_alive[fi]) continue;
        for (int k = 0; k < 3; k++) {
            int32_t vi = m->faces[fi].v[k];
            PAMO_ASSERT(vi >= 0 && (size_t)vi < nv);
            m->vert_face_offset[vi]++;
        }
    }

    /* Prefix sum. */
    {
        int32_t sum = 0;
        for (size_t i = 0; i < nv; i++) {
            int32_t cnt = m->vert_face_offset[i];
            m->vert_face_offset[i] = sum;
            sum += cnt;
        }
        m->vert_face_offset[nv] = sum;
    }

    size_t total_valence = (size_t)m->vert_face_offset[nv];
    m->vert_face_list = PAMO_ALLOC_ARRAY(a, int32_t, total_valence > 0 ? total_valence : 1);
    if (!m->vert_face_list && total_valence > 0) return PAMO_ERR_ALLOC;

    /* Temporary write cursors. */
    int32_t *cursor = PAMO_ALLOC_ARRAY(a, int32_t, nv);
    if (!cursor && nv > 0) return PAMO_ERR_ALLOC;
    memcpy(cursor, m->vert_face_offset, nv * sizeof(int32_t));

    for (size_t fi = 0; fi < nf; fi++) {
        if (!m->face_alive[fi]) continue;
        for (int k = 0; k < 3; k++) {
            int32_t vi = m->faces[fi].v[k];
            m->vert_face_list[cursor[vi]++] = (int32_t)fi;
        }
    }
    PAMO_FREE_ARRAY(a, cursor, int32_t, nv);

    /* ── 2. Unique edge list ─────────────────────────────────────── */
    /* Each alive face contributes 3 half-edges.  We canonicalize
     * (u < v) and deduplicate by sorting. */
    size_t n_half = 0;
    for (size_t fi = 0; fi < nf; fi++) {
        if (m->face_alive[fi]) n_half += 3;
    }

    pamo_edge *half_edges = PAMO_ALLOC_ARRAY(a, pamo_edge,
                                             n_half > 0 ? n_half : 1);
    if (!half_edges && n_half > 0) return PAMO_ERR_ALLOC;

    size_t ei = 0;
    for (size_t fi = 0; fi < nf; fi++) {
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t u = fv[k], v = fv[(k + 1) % 3];
            if (u > v) { int32_t tmp = u; u = v; v = tmp; }
            half_edges[ei++] = (pamo_edge){u, v};
        }
    }
    PAMO_ASSERT(ei == n_half);

    qsort(half_edges, n_half, sizeof(pamo_edge), edge_cmp);

    /* Count unique edges. */
    size_t n_unique = 0;
    for (size_t i = 0; i < n_half; i++) {
        if (i == 0 || edge_cmp(&half_edges[i], &half_edges[i - 1]) != 0) {
            n_unique++;
        }
    }

    m->edges = PAMO_ALLOC_ARRAY(a, pamo_edge, n_unique > 0 ? n_unique : 1);
    if (!m->edges && n_unique > 0) {
        PAMO_FREE_ARRAY(a, half_edges, pamo_edge, n_half > 0 ? n_half : 1);
        return PAMO_ERR_ALLOC;
    }

    size_t eidx = 0;
    for (size_t i = 0; i < n_half; i++) {
        if (i == 0 || edge_cmp(&half_edges[i], &half_edges[i - 1]) != 0) {
            m->edges[eidx++] = half_edges[i];
        }
    }
    m->n_edges = n_unique;

    PAMO_FREE_ARRAY(a, half_edges, pamo_edge, n_half > 0 ? n_half : 1);

    /* ── 3. Edge->face CSR ───────────────────────────────────────── */
    m->edge_face_offset = PAMO_ALLOC_ARRAY(a, int32_t, n_unique + 1);
    if (!m->edge_face_offset && n_unique > 0) return PAMO_ERR_ALLOC;

    /* For each alive face, find its 3 edges in the sorted edge list
     * and count.  Use binary search. */
    for (size_t fi = 0; fi < nf; fi++) {
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t u = fv[k], v = fv[(k + 1) % 3];
            if (u > v) { int32_t tmp = u; u = v; v = tmp; }
            /* Binary search for (u, v). */
            pamo_edge key = {u, v};
            size_t lo = 0, hi = n_unique;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (edge_cmp(&m->edges[mid], &key) < 0) lo = mid + 1;
                else hi = mid;
            }
            PAMO_ASSERT(lo < n_unique &&
                        m->edges[lo].u == u && m->edges[lo].v == v);
            m->edge_face_offset[lo]++;
        }
    }

    /* Prefix sum. */
    {
        int32_t sum = 0;
        for (size_t i = 0; i < n_unique; i++) {
            int32_t cnt = m->edge_face_offset[i];
            m->edge_face_offset[i] = sum;
            sum += cnt;
        }
        m->edge_face_offset[n_unique] = sum;
    }

    size_t total_ef = (size_t)m->edge_face_offset[n_unique];
    m->edge_face_list = PAMO_ALLOC_ARRAY(a, int32_t, total_ef > 0 ? total_ef : 1);
    if (!m->edge_face_list && total_ef > 0) return PAMO_ERR_ALLOC;

    int32_t *ef_cursor = PAMO_ALLOC_ARRAY(a, int32_t, n_unique > 0 ? n_unique : 1);
    if (!ef_cursor && n_unique > 0) return PAMO_ERR_ALLOC;
    memcpy(ef_cursor, m->edge_face_offset,
           n_unique * sizeof(int32_t));

    for (size_t fi = 0; fi < nf; fi++) {
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t u = fv[k], v = fv[(k + 1) % 3];
            if (u > v) { int32_t tmp = u; u = v; v = tmp; }
            pamo_edge key = {u, v};
            size_t lo = 0, hi = n_unique;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (edge_cmp(&m->edges[mid], &key) < 0) lo = mid + 1;
                else hi = mid;
            }
            m->edge_face_list[ef_cursor[lo]++] = (int32_t)fi;
        }
    }
    PAMO_FREE_ARRAY(a, ef_cursor, int32_t, n_unique > 0 ? n_unique : 1);

    return PAMO_OK;
}

void pamo_mesh_free_adjacency(pamo_mesh *m) {
    if (!m) return;
    pamo_allocator *a = &m->alloc;

    /* Read sizes before freeing offset arrays. */
    size_t vf_total = 0;
    if (m->vert_face_offset && m->n_verts > 0) {
        vf_total = (size_t)m->vert_face_offset[m->n_verts];
    }
    size_t ef_total = 0;
    if (m->edge_face_offset && m->n_edges > 0) {
        ef_total = (size_t)m->edge_face_offset[m->n_edges];
    }

    if (m->vert_face_list) {
        PAMO_FREE_ARRAY(a, m->vert_face_list, int32_t,
                        vf_total > 0 ? vf_total : 1);
        m->vert_face_list = NULL;
    }
    if (m->vert_face_offset) {
        PAMO_FREE_ARRAY(a, m->vert_face_offset, int32_t, m->n_verts + 1);
        m->vert_face_offset = NULL;
    }
    if (m->edge_face_list) {
        PAMO_FREE_ARRAY(a, m->edge_face_list, int32_t,
                        ef_total > 0 ? ef_total : 1);
        m->edge_face_list = NULL;
    }
    if (m->edge_face_offset) {
        PAMO_FREE_ARRAY(a, m->edge_face_offset, int32_t, m->n_edges + 1);
        m->edge_face_offset = NULL;
    }
    if (m->edges) {
        PAMO_FREE_ARRAY(a, m->edges, pamo_edge,
                        m->n_edges > 0 ? m->n_edges : 1);
        m->edges = NULL;
    }
    m->n_edges = 0;
}
