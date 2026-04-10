/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_mesh.h"

#include <string.h>

pamo_error pamo_mesh_compact(pamo_mesh *m) {
    if (!m) return PAMO_ERR_INVALID_ARG;
    pamo_allocator *a = &m->alloc;

    /* Free adjacency since it will be invalid after compaction. */
    pamo_mesh_free_adjacency(m);

    size_t old_nv = m->n_verts;
    size_t old_nf = m->n_faces;

    /* ── 1. Build vertex remap (old index -> new index) ──────────── */
    size_t map_alloc = old_nv > 0 ? old_nv : 1;
    int32_t *vert_map = PAMO_ALLOC_ARRAY(a, int32_t, map_alloc);
    if (!vert_map && old_nv > 0) return PAMO_ERR_ALLOC;

    int32_t new_nv = 0;
    for (size_t i = 0; i < old_nv; i++) {
        if (m->vert_alive[i]) {
            vert_map[i] = new_nv;
            m->verts[new_nv] = m->verts[i];
            m->vert_alive[new_nv] = true;
            new_nv++;
        } else {
            vert_map[i] = -1;
        }
    }

    for (size_t i = (size_t)new_nv; i < old_nv; i++) {
        m->vert_alive[i] = false;
    }
    m->n_verts = (size_t)new_nv;

    /* ── 2. Compact faces and remap vertex indices ───────────────── */
    int32_t new_nf = 0;
    for (size_t i = 0; i < old_nf; i++) {
        if (!m->face_alive[i]) continue;
        pamo_tri f = m->faces[i];
        int32_t a0 = vert_map[f.v[0]];
        int32_t a1 = vert_map[f.v[1]];
        int32_t a2 = vert_map[f.v[2]];
        if (a0 < 0 || a1 < 0 || a2 < 0) continue;
        if (a0 == a1 || a1 == a2 || a2 == a0) continue;
        m->faces[new_nf] = (pamo_tri){{a0, a1, a2}};
        m->face_alive[new_nf] = true;
        new_nf++;
    }

    for (size_t i = (size_t)new_nf; i < old_nf; i++) {
        m->face_alive[i] = false;
    }
    m->n_faces = (size_t)new_nf;

    PAMO_FREE_ARRAY(a, vert_map, int32_t, map_alloc);

    return PAMO_OK;
}
