/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 2 collapse predicates + apply. The simplifier proper
 * (simplify.c) computes per-edge costs and picks candidates; this TU
 * answers "is collapsing (u,v) safe?" and, when it is, performs the
 * collapse. Splitting these out keeps the cost/round/driver code
 * focused on the algorithm rather than the topology bookkeeping.
 */

#include "pamo/pamo_mesh.h"
#include "pamo/pamo_internal.h"

#include <math.h>

/* ── Normal-flip / skinny check ───────────────────────────────────── */

bool pamo_collapse_creates_problem(const pamo_mesh *m,
                                   int32_t endpoint, int32_t other,
                                   pamo_vec3d mid,
                                   double skinny_thresh,
                                   double flip_threshold) {
    int32_t s = m->vert_face_offset[endpoint];
    int32_t e = m->vert_face_offset[endpoint + 1];
    for (int32_t i = s; i < e; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        if (fv[0] == other || fv[1] == other || fv[2] == other) continue;

        pamo_vec3d old_n = pamo_face_unit_normal(m, fi);
        pamo_vec3d verts[3];
        for (int k = 0; k < 3; k++)
            verts[k] = (fv[k] == endpoint) ? mid : m->verts[fv[k]];

        pamo_vec3d new_n_raw = pamo_v3_cross(pamo_v3_sub(verts[1], verts[0]),
                                             pamo_v3_sub(verts[2], verts[0]));
        double new_len = sqrt(pamo_v3_length_sq(new_n_raw));
        if (new_len < 1e-30) return true;
        pamo_vec3d new_n = pamo_v3_scale(new_n_raw, 1.0 / new_len);
        if (pamo_v3_dot(old_n, new_n) < flip_threshold) return true;
        if (pamo_triangle_is_skinny(verts[0], verts[1], verts[2], skinny_thresh))
            return true;
    }
    return false;
}

/* ── Link condition ──────────────────────────────────────────────── */

static int32_t count_shared_faces(const pamo_mesh *m, int32_t u, int32_t v) {
    int32_t count = 0;
    int32_t u_s = m->vert_face_offset[u];
    int32_t u_e = m->vert_face_offset[u + 1];
    for (int32_t i = u_s; i < u_e; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        if (fv[0] == v || fv[1] == v || fv[2] == v) count++;
    }
    return count;
}

bool pamo_collapse_violates_link(const pamo_mesh *m, int32_t u, int32_t v) {
    /* Vertices adjacent to u (excluding v). */
    int32_t u_adj[PAMO_MAX_VALENCE], n_u = 0;
    int32_t u_s = m->vert_face_offset[u];
    int32_t u_e = m->vert_face_offset[u + 1];
    for (int32_t i = u_s; i < u_e; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t w = fv[k];
            if (w == u || w == v) continue;
            bool dup = false;
            for (int32_t j = 0; j < n_u; j++) {
                if (u_adj[j] == w) { dup = true; break; }
            }
            if (dup) continue;
            /* Conservative on overflow: report unsafe rather than
             * truncate (a truncated neighbour list could mask a
             * non-manifold collapse). */
            if (n_u >= PAMO_MAX_VALENCE) return true;
            u_adj[n_u++] = w;
        }
    }

    /* Vertices adjacent to v (excluding u). */
    int32_t v_adj[PAMO_MAX_VALENCE], n_v = 0;
    int32_t v_s = m->vert_face_offset[v];
    int32_t v_e = m->vert_face_offset[v + 1];
    for (int32_t i = v_s; i < v_e; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t w = fv[k];
            if (w == u || w == v) continue;
            bool dup = false;
            for (int32_t j = 0; j < n_v; j++) {
                if (v_adj[j] == w) { dup = true; break; }
            }
            if (dup) continue;
            if (n_v >= PAMO_MAX_VALENCE) return true;
            v_adj[n_v++] = w;
        }
    }

    int32_t n_shared = 0;
    for (int32_t i = 0; i < n_u; i++) {
        for (int32_t j = 0; j < n_v; j++) {
            if (u_adj[i] == v_adj[j]) { n_shared++; break; }
        }
    }

    int32_t n_shared_faces = count_shared_faces(m, u, v);
    if (n_shared_faces == 2) return n_shared != 2;   /* interior edge */
    if (n_shared_faces == 1) return n_shared != 1;   /* boundary edge */
    return true;                                     /* 0 or >2: unsafe */
}

/* ── Predictive 1-ring self-intersection ─────────────────────────── */

bool pamo_collapse_self_intersects(const pamo_mesh *m,
                                   int32_t u, int32_t v,
                                   pamo_vec3d mid) {
    /* Enumerate post-collapse triangles (faces incident to u or v
     * minus the shared faces that die, with v→u remapped). */
    pamo_vec3d tri_v[PAMO_MAX_VALENCE * 2][3];
    int32_t    tri_orig[PAMO_MAX_VALENCE * 2][3];
    int32_t    n_tri = 0;

    for (int pass = 0; pass < 2; pass++) {
        int32_t endpoint = (pass == 0) ? u : v;
        int32_t other    = (pass == 0) ? v : u;
        int32_t s = m->vert_face_offset[endpoint];
        int32_t e = m->vert_face_offset[endpoint + 1];
        for (int32_t i = s; i < e; i++) {
            int32_t fi = m->vert_face_list[i];
            if (!m->face_alive[fi]) continue;
            const int32_t *fv = m->faces[fi].v;
            if (fv[0] == other || fv[1] == other || fv[2] == other) continue;
            if (pass == 1) {
                /* Skip faces already enumerated under u's pass. */
                bool seen = false;
                for (int32_t k = 0; k < n_tri && !seen; k++) {
                    if (tri_orig[k][0] == fv[0] &&
                        tri_orig[k][1] == fv[1] &&
                        tri_orig[k][2] == fv[2]) seen = true;
                }
                if (seen) continue;
            }
            if (n_tri >= PAMO_MAX_VALENCE * 2) return true;  /* conservative */
            for (int k = 0; k < 3; k++) {
                tri_orig[n_tri][k] = fv[k];
                int32_t vid = (fv[k] == endpoint) ? endpoint
                            : (fv[k] == v        ? u
                                                 : fv[k]);
                tri_v[n_tri][k] = (fv[k] == endpoint) ? mid : m->verts[vid];
            }
            if (pamo_v3_length_sq(pamo_v3_sub(tri_v[n_tri][1], tri_v[n_tri][0])) < 1e-30 ||
                pamo_v3_length_sq(pamo_v3_sub(tri_v[n_tri][2], tri_v[n_tri][1])) < 1e-30 ||
                pamo_v3_length_sq(pamo_v3_sub(tri_v[n_tri][0], tri_v[n_tri][2])) < 1e-30) {
                continue;
            }
            n_tri++;
        }
    }

    /* Pairwise test. Skip pairs that share a (post-remap) vertex —
     * adjacent triangles touch along a real shared edge and the
     * tri-tri test would flag that as an "intersection". */
    for (int32_t i = 0; i < n_tri; i++) {
        for (int32_t j = i + 1; j < n_tri; j++) {
            int32_t ai[3], bi[3];
            for (int k = 0; k < 3; k++) {
                ai[k] = (tri_orig[i][k] == v) ? u : tri_orig[i][k];
                bi[k] = (tri_orig[j][k] == v) ? u : tri_orig[j][k];
            }
            bool share = false;
            for (int a = 0; a < 3 && !share; a++)
                for (int b = 0; b < 3 && !share; b++)
                    if (ai[a] == bi[b]) share = true;
            if (share) continue;
            if (pamo_tri_tri_intersect(tri_v[i][0], tri_v[i][1], tri_v[i][2],
                                       tri_v[j][0], tri_v[j][1], tri_v[j][2])) {
                return true;
            }
        }
    }
    return false;
}

/* ── Apply collapse ──────────────────────────────────────────────── */

void pamo_apply_collapse(pamo_mesh *m, int32_t u, int32_t v,
                         pamo_vec3d mid) {
    m->verts[u] = mid;
    m->vert_alive[v] = false;

    int32_t v_start = m->vert_face_offset[v];
    int32_t v_end   = m->vert_face_offset[v + 1];
    for (int32_t fi_idx = v_start; fi_idx < v_end; fi_idx++) {
        int32_t fi = m->vert_face_list[fi_idx];
        if (!m->face_alive[fi]) continue;
        int32_t *fv = m->faces[fi].v;
        bool has_u = (fv[0] == u || fv[1] == u || fv[2] == u);
        if (has_u) {
            /* Shared face: would become degenerate. */
            m->face_alive[fi] = false;
        } else {
            for (int k = 0; k < 3; k++) {
                if (fv[k] == v) fv[k] = u;
            }
            /* Defensive: link condition should have prevented this. */
            if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0]) {
                m->face_alive[fi] = false;
            }
        }
    }

    /* After v→u remap, two faces incident to u may be the same
     * triangle. Remove duplicates. */
    int32_t u_start = m->vert_face_offset[u];
    int32_t u_end   = m->vert_face_offset[u + 1];
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        for (int32_t j = v_start; j < v_end; j++) {
            int32_t fj = m->vert_face_list[j];
            if (!m->face_alive[fj] || fj == fi) continue;
            const int32_t *gv = m->faces[fj].v;
            int32_t fa[3] = {fv[0], fv[1], fv[2]};
            int32_t ga[3] = {gv[0], gv[1], gv[2]};
            for (int a = 0; a < 2; a++) for (int b = a + 1; b < 3; b++) {
                if (fa[a] > fa[b]) { int32_t t=fa[a]; fa[a]=fa[b]; fa[b]=t; }
                if (ga[a] > ga[b]) { int32_t t=ga[a]; ga[a]=ga[b]; ga[b]=t; }
            }
            if (fa[0]==ga[0] && fa[1]==ga[1] && fa[2]==ga[2]) {
                m->face_alive[fj] = false;
            }
        }
    }
}
