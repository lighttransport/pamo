/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_mesh.h"

#include <string.h>

pamo_error pamo_mesh_create(pamo_mesh *m, size_t n_verts, size_t n_faces,
                            const pamo_allocator *alloc) {
    if (!m || !alloc) return PAMO_ERR_INVALID_ARG;
    if (n_verts > (size_t)INT32_MAX || n_faces > (size_t)INT32_MAX)
        return PAMO_ERR_INVALID_ARG;
    if (n_verts > SIZE_MAX / sizeof(pamo_vec3d) ||
        n_verts > SIZE_MAX / sizeof(bool) ||
        n_faces > SIZE_MAX / sizeof(pamo_tri) ||
        n_faces > SIZE_MAX / sizeof(bool)) {
        return PAMO_ERR_INVALID_ARG;
    }

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

    if (src->n_verts > 0) {
        if (!src->verts || !src->vert_alive) {
            pamo_mesh_destroy(dst);
            return PAMO_ERR_INVALID_ARG;
        }
        memcpy(dst->verts, src->verts, src->n_verts * sizeof(pamo_vec3d));
        memcpy(dst->vert_alive, src->vert_alive, src->n_verts * sizeof(bool));
    }
    if (src->n_faces > 0) {
        if (!src->faces || !src->face_alive) {
            pamo_mesh_destroy(dst);
            return PAMO_ERR_INVALID_ARG;
        }
        memcpy(dst->faces, src->faces, src->n_faces * sizeof(pamo_tri));
        memcpy(dst->face_alive, src->face_alive, src->n_faces * sizeof(bool));
    }

    return PAMO_OK;
}

size_t pamo_mesh_count_alive_faces(const pamo_mesh *m) {
    if (!m || !m->face_alive) return 0;
    size_t count = 0;
    for (size_t i = 0; i < m->n_faces; i++) {
        if (m->face_alive[i]) count++;
    }
    return count;
}

size_t pamo_mesh_count_alive_verts(const pamo_mesh *m) {
    if (!m || !m->vert_alive) return 0;
    size_t count = 0;
    for (size_t i = 0; i < m->n_verts; i++) {
        if (m->vert_alive[i]) count++;
    }
    return count;
}

bool pamo_mesh_face_is_valid(const pamo_mesh *m, size_t face_id) {
    if (!m || !m->faces || !m->face_alive ||
        !m->verts || !m->vert_alive || face_id >= m->n_faces ||
        !m->face_alive[face_id]) {
        return false;
    }

    const pamo_tri f = m->faces[face_id];
    if (f.v[0] == f.v[1] || f.v[1] == f.v[2] || f.v[2] == f.v[0])
        return false;

    for (int k = 0; k < 3; k++) {
        int32_t vi = f.v[k];
        if (vi < 0 || (size_t)vi >= m->n_verts) return false;
        if (!m->vert_alive[vi]) return false;
    }
    return true;
}

pamo_aabb pamo_mesh_bounds(const pamo_mesh *m) {
    pamo_aabb bb;
    bb.lo = (pamo_vec3d){ 1e30,  1e30,  1e30};
    bb.hi = (pamo_vec3d){-1e30, -1e30, -1e30};
    if (!m || !m->verts || !m->vert_alive) return bb;
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
