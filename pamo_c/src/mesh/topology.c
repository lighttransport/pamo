/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_mesh.h"

int32_t pamo_shared_neighbor_count(const pamo_mesh *m,
                                   int32_t u, int32_t v) {
    if (!m || !m->vert_face_offset || !m->vert_face_list)
        return 0;
    if (u < 0 || v < 0 || (size_t)u >= m->n_verts ||
        (size_t)v >= m->n_verts) {
        return 0;
    }

    int32_t u_start = m->vert_face_offset[u];
    int32_t u_end   = m->vert_face_offset[u + 1];
    int32_t v_start = m->vert_face_offset[v];
    int32_t v_end   = m->vert_face_offset[v + 1];

    /* Collect unique third-vertex IDs from u's neighborhood
     * (excluding u and v themselves).  Use a small stack buffer
     * since valence is typically low. */
    int32_t u_neighbors[1024];
    int32_t n_un = 0;

    for (int32_t ui = u_start; ui < u_end; ui++) {
        int32_t fi = m->vert_face_list[ui];
        if (fi < 0 || (size_t)fi >= m->n_faces || !m->face_alive[fi])
            continue;
        const int32_t *fv = m->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t w = fv[k];
            if (w == u || w == v) continue;
            /* Check if already in list. */
            bool found = false;
            for (int32_t j = 0; j < n_un; j++) {
                if (u_neighbors[j] == w) { found = true; break; }
            }
            if (!found && n_un < 1024) {
                u_neighbors[n_un++] = w;
            }
        }
    }

    /* Count how many of u's unique neighbors also appear
     * in v's neighborhood. */
    int32_t count = 0;
    for (int32_t i = 0; i < n_un; i++) {
        int32_t w = u_neighbors[i];
        for (int32_t vi = v_start; vi < v_end; vi++) {
            int32_t fi = m->vert_face_list[vi];
            if (fi < 0 || (size_t)fi >= m->n_faces || !m->face_alive[fi])
                continue;
            const int32_t *fv = m->faces[fi].v;
            if (fv[0] == w || fv[1] == w || fv[2] == w) {
                count++;
                break;
            }
        }
    }

    return count;
}
