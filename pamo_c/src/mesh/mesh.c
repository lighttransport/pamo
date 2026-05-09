/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_internal.h"

#include <string.h>

pamo_error pamo_mesh_create(pamo_mesh *m, size_t n_verts, size_t n_faces,
                            const pamo_allocator *alloc) {
    if (!m || !alloc) return PAMO_ERR_INVALID_ARG;

    memset(m, 0, sizeof(*m));
    m->alloc = *alloc;
    m->n_verts = n_verts;
    m->n_faces = n_faces;
    m->n_verts_cap = n_verts;
    m->n_faces_cap = n_faces;

    if (n_verts > 0) {
        m->verts = PAMO_ALLOC_ARRAY(&m->alloc, pamo_vec3d, n_verts);
        m->vert_alive = PAMO_ALLOC_ARRAY(&m->alloc, bool, n_verts);
        if (!m->verts || !m->vert_alive) goto fail;
        for (size_t i = 0; i < n_verts; i++) m->vert_alive[i] = true;
    }
    if (n_faces > 0) {
        m->faces = PAMO_ALLOC_ARRAY(&m->alloc, pamo_tri, n_faces);
        m->face_alive = PAMO_ALLOC_ARRAY(&m->alloc, bool, n_faces);
        if (!m->faces || !m->face_alive) goto fail;
        for (size_t i = 0; i < n_faces; i++) m->face_alive[i] = true;
    }
    return PAMO_OK;

fail:
    pamo_mesh_destroy(m);
    return PAMO_ERR_ALLOC;
}

void pamo_mesh_destroy(pamo_mesh *m) {
    if (!m) return;
    pamo_allocator *a = &m->alloc;

    if (m->verts)      PAMO_FREE_ARRAY(a, m->verts, pamo_vec3d, m->n_verts_cap);
    if (m->faces)      PAMO_FREE_ARRAY(a, m->faces, pamo_tri, m->n_faces_cap);
    if (m->vert_alive) PAMO_FREE_ARRAY(a, m->vert_alive, bool, m->n_verts_cap);
    if (m->face_alive) PAMO_FREE_ARRAY(a, m->face_alive, bool, m->n_faces_cap);

    pamo_mesh_free_adjacency(m);
    memset(m, 0, sizeof(*m));
}

pamo_error pamo_mesh_deep_copy(pamo_mesh *dst, const pamo_mesh *src) {
    if (!dst || !src) return PAMO_ERR_INVALID_ARG;

    pamo_error err = pamo_mesh_create(dst, src->n_verts, src->n_faces,
                                      &src->alloc);
    if (err != PAMO_OK) return err;

    memcpy(dst->verts, src->verts, src->n_verts * sizeof(pamo_vec3d));
    memcpy(dst->faces, src->faces, src->n_faces * sizeof(pamo_tri));
    memcpy(dst->vert_alive, src->vert_alive, src->n_verts * sizeof(bool));
    memcpy(dst->face_alive, src->face_alive, src->n_faces * sizeof(bool));

    return PAMO_OK;
}

size_t pamo_mesh_count_alive_faces(const pamo_mesh *m) {
    if (!m) return 0;
    size_t count = 0;
    for (size_t i = 0; i < m->n_faces; i++) {
        if (m->face_alive[i]) count++;
    }
    return count;
}

size_t pamo_mesh_count_alive_verts(const pamo_mesh *m) {
    if (!m) return 0;
    size_t count = 0;
    for (size_t i = 0; i < m->n_verts; i++) {
        if (m->vert_alive[i]) count++;
    }
    return count;
}

/* ── Capacity-growing helpers (internal API) ─────────────────────── */

pamo_error pamo_mesh_reserve_verts(pamo_mesh *m, size_t n) {
    if (m->n_verts_cap >= n) return PAMO_OK;
    size_t new_cap = m->n_verts_cap ? m->n_verts_cap : 16;
    while (new_cap < n) new_cap *= 2;
    pamo_vec3d *nv = (pamo_vec3d *)PAMO_REALLOC(
        &m->alloc, m->verts,
        m->n_verts_cap * sizeof(pamo_vec3d),
        new_cap * sizeof(pamo_vec3d));
    bool *na = (bool *)PAMO_REALLOC(
        &m->alloc, m->vert_alive,
        m->n_verts_cap * sizeof(bool),
        new_cap * sizeof(bool));
    if (!nv || !na) return PAMO_ERR_ALLOC;
    m->verts = nv;
    m->vert_alive = na;
    m->n_verts_cap = new_cap;
    return PAMO_OK;
}

pamo_error pamo_mesh_reserve_faces(pamo_mesh *m, size_t n) {
    if (m->n_faces_cap >= n) return PAMO_OK;
    size_t new_cap = m->n_faces_cap ? m->n_faces_cap : 16;
    while (new_cap < n) new_cap *= 2;
    pamo_tri *nf = (pamo_tri *)PAMO_REALLOC(
        &m->alloc, m->faces,
        m->n_faces_cap * sizeof(pamo_tri),
        new_cap * sizeof(pamo_tri));
    bool *nfa = (bool *)PAMO_REALLOC(
        &m->alloc, m->face_alive,
        m->n_faces_cap * sizeof(bool),
        new_cap * sizeof(bool));
    if (!nf || !nfa) return PAMO_ERR_ALLOC;
    m->faces = nf;
    m->face_alive = nfa;
    m->n_faces_cap = new_cap;
    return PAMO_OK;
}

pamo_error pamo_mesh_append_vert(pamo_mesh *m, pamo_vec3d p, int32_t *out) {
    pamo_error e = pamo_mesh_reserve_verts(m, m->n_verts + 1);
    if (e != PAMO_OK) return e;
    int32_t i = (int32_t)m->n_verts++;
    m->verts[i] = p;
    m->vert_alive[i] = true;
    if (out) *out = i;
    return PAMO_OK;
}

pamo_error pamo_mesh_append_face(pamo_mesh *m,
                                 int32_t a, int32_t b, int32_t c) {
    pamo_error e = pamo_mesh_reserve_faces(m, m->n_faces + 1);
    if (e != PAMO_OK) return e;
    int32_t i = (int32_t)m->n_faces++;
    m->faces[i].v[0] = a;
    m->faces[i].v[1] = b;
    m->faces[i].v[2] = c;
    m->face_alive[i] = true;
    return PAMO_OK;
}

pamo_aabb pamo_mesh_bounds(const pamo_mesh *m) {
    pamo_aabb bb;
    bb.lo = (pamo_vec3d){ 1e30,  1e30,  1e30};
    bb.hi = (pamo_vec3d){-1e30, -1e30, -1e30};
    if (!m) return bb;
    for (size_t i = 0; i < m->n_verts; i++) {
        if (!m->vert_alive[i]) continue;
        pamo_vec3d v = m->verts[i];
        if (v.x < bb.lo.x) bb.lo.x = v.x;
        if (v.y < bb.lo.y) bb.lo.y = v.y;
        if (v.z < bb.lo.z) bb.lo.z = v.z;
        if (v.x > bb.hi.x) bb.hi.x = v.x;
        if (v.y > bb.hi.y) bb.hi.y = v.y;
        if (v.z > bb.hi.z) bb.hi.z = v.z;
    }
    return bb;
}
