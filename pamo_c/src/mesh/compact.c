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

    /* ── 1. Compact faces, removing dead/degenerate ones ─────────── */
    int32_t new_nf = 0;
    for (size_t i = 0; i < old_nf; i++) {
        if (!m->face_alive[i]) continue;
        pamo_tri f = m->faces[i];
        /* Skip faces with dead vertices. */
        if (!m->vert_alive[f.v[0]] || !m->vert_alive[f.v[1]] ||
            !m->vert_alive[f.v[2]]) continue;
        /* Skip degenerate. */
        if (f.v[0] == f.v[1] || f.v[1] == f.v[2] || f.v[2] == f.v[0])
            continue;
        m->faces[new_nf] = f;
        m->face_alive[new_nf] = true;
        new_nf++;
    }
    for (size_t i = (size_t)new_nf; i < old_nf; i++) {
        m->face_alive[i] = false;
    }
    m->n_faces = (size_t)new_nf;

    /* ── 2. Mark unreferenced vertices as dead ───────────────────── */
    for (size_t i = 0; i < old_nv; i++) m->vert_alive[i] = false;
    for (int32_t i = 0; i < new_nf; i++) {
        m->vert_alive[m->faces[i].v[0]] = true;
        m->vert_alive[m->faces[i].v[1]] = true;
        m->vert_alive[m->faces[i].v[2]] = true;
    }

    /* ── 3. Build vertex remap and compact vertices ──────────────── */
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

    /* ── 4. Remap face indices ───────────────────────────────────── */
    for (int32_t i = 0; i < new_nf; i++) {
        m->faces[i].v[0] = vert_map[m->faces[i].v[0]];
        m->faces[i].v[1] = vert_map[m->faces[i].v[1]];
        m->faces[i].v[2] = vert_map[m->faces[i].v[2]];
    }

    PAMO_FREE_ARRAY(a, vert_map, int32_t, map_alloc);
    return PAMO_OK;
}
