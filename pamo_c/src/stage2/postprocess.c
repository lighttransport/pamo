/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 2 post-process: bowtie-vertex split followed by boundary-loop
 * hole-fill (min-area DP triangulation, with a centroid-fan fallback
 * for loops larger than the DP cap). Run after the main collapse loop
 * in pamo_simplify so the output is watertight + 2-manifold for
 * collision-detection use. No-ops on already-clean meshes.
 */

#include "pamo/pamo_mesh.h"
#include "pamo/pamo_internal.h"

#include <math.h>

/* Upper bound on vertex valence handled on the stack. The collapse
 * code in simplify.c carries its own copy of this — they don't need
 * to match (each is sized for its own use). */
#define PAMO_PP_MAX_VALENCE 256

/* Cap for O(L^3) min-area DP. Loops longer than this fall back to a
 * centroid fan; pure DP would still finish but the cubic scaling
 * starts to hurt. */
#define PAMO_PP_DP_MAX 96

static bool mesh_has_boundary(pamo_mesh *mesh) {
    if (pamo_mesh_build_adjacency(mesh) != PAMO_OK) return false;
    bool has = false;
    if (mesh->edge_face_offset) {
        for (size_t ei = 0; ei < mesh->n_edges && !has; ei++) {
            if (mesh->edge_face_offset[ei + 1]
                - mesh->edge_face_offset[ei] == 1) has = true;
        }
    }
    pamo_mesh_free_adjacency(mesh);
    return has;
}

/* ── Bowtie split: convert non-manifold (multi-fan) vertices into N
 * coincident vertex copies, one per fan. Replaces the older
 * face-deletion strategy that used to tear holes. */
static pamo_error split_bowties(pamo_mesh *mesh) {
    for (int cleanup = 0; cleanup < 10; cleanup++) {
        pamo_error err = pamo_mesh_build_adjacency(mesh);
        if (err != PAMO_OK) return err;

        bool found_any = false;
        /* Pin the scan to the pre-split count; new vertex copies
         * appended below have only one fan by construction. */
        const size_t scan_limit = mesh->n_verts;
        for (size_t vi = 0; vi < scan_limit; vi++) {
            if (!mesh->vert_alive[vi]) continue;
            int32_t s = mesh->vert_face_offset[vi];
            int32_t e = mesh->vert_face_offset[vi + 1];
            int32_t n_inc = 0;
            int32_t inc_faces[PAMO_PP_MAX_VALENCE];
            bool inc_overflow = false;
            for (int32_t j = s; j < e; j++) {
                int32_t fi = mesh->vert_face_list[j];
                if (!mesh->face_alive[fi]) continue;
                if (n_inc >= PAMO_PP_MAX_VALENCE) {
                    inc_overflow = true; break;
                }
                inc_faces[n_inc++] = fi;
            }
            /* Skip extreme-valence vertices: a truncated component
             * count could misclassify a manifold vertex as a bowtie. */
            if (inc_overflow) continue;
            if (n_inc <= 1) continue;

            /* Group incident faces into 1-ring fans by walking face-
             * face adjacency around vi (two faces are adjacent iff
             * they share a non-vi vertex). One fan = manifold; two or
             * more = bowtie. */
            int8_t comp_id[PAMO_PP_MAX_VALENCE];
            for (int32_t i = 0; i < n_inc; i++) comp_id[i] = -1;
            int32_t n_comps = 0;
            int32_t queue[PAMO_PP_MAX_VALENCE];
            for (int32_t seed = 0; seed < n_inc; seed++) {
                if (comp_id[seed] >= 0) continue;
                int32_t qh = 0, qt = 0;
                queue[qt++] = seed;
                comp_id[seed] = (int8_t)n_comps;
                while (qh < qt) {
                    int32_t ci = queue[qh++];
                    int32_t fi = inc_faces[ci];
                    const int32_t *fv = mesh->faces[fi].v;
                    int32_t others[2], no = 0;
                    for (int k = 0; k < 3; k++)
                        if (fv[k] != (int32_t)vi && no < 2)
                            others[no++] = fv[k];
                    for (int32_t ci2 = 0; ci2 < n_inc; ci2++) {
                        if (comp_id[ci2] >= 0) continue;
                        int32_t fi2 = inc_faces[ci2];
                        const int32_t *gv = mesh->faces[fi2].v;
                        bool adj = false;
                        for (int a = 0; a < no && !adj; a++)
                            for (int b = 0; b < 3 && !adj; b++)
                                if (gv[b] == others[a]
                                    && gv[b] != (int32_t)vi)
                                    adj = true;
                        if (adj) {
                            comp_id[ci2] = (int8_t)n_comps;
                            queue[qt++] = ci2;
                        }
                    }
                }
                n_comps++;
            }
            if (n_comps <= 1) continue;

            /* Bowtie. Keep the largest fan on vi and rebind every
             * other fan to a fresh coincident copy. All geometry
             * survives (face deletion would tear the watertight
             * surface this pass exists to maintain). */
            found_any = true;
            int32_t comp_size[PAMO_PP_MAX_VALENCE] = {0};
            for (int32_t i = 0; i < n_inc; i++) comp_size[comp_id[i]]++;
            int8_t keep = 0;
            for (int8_t k = 1; k < (int8_t)n_comps; k++) {
                if (comp_size[k] > comp_size[keep]) keep = k;
            }

            for (int8_t k = 0; k < (int8_t)n_comps; k++) {
                if (k == keep) continue;
                int32_t new_vi = -1;
                pamo_error ae = pamo_mesh_append_vert(mesh,
                                                      mesh->verts[vi],
                                                      &new_vi);
                if (ae != PAMO_OK) {
                    pamo_mesh_free_adjacency(mesh);
                    return ae;
                }
                for (int32_t i = 0; i < n_inc; i++) {
                    if (comp_id[i] != k) continue;
                    int32_t fi = inc_faces[i];
                    int32_t *fv = mesh->faces[fi].v;
                    for (int j = 0; j < 3; j++)
                        if (fv[j] == (int32_t)vi) fv[j] = new_vi;
                }
            }
        }

        pamo_mesh_free_adjacency(mesh);

        if (!found_any) break;

        err = pamo_mesh_compact(mesh);
        if (err != PAMO_OK) return err;
    }
    return PAMO_OK;
}

/* ── Hole-fill: walk every closed boundary loop and triangulate it.
 * Uses a min-area DP for L ≤ DP_MAX (avoids the centroid spike that
 * a fan introduces, especially on long thin cracks); falls back to a
 * centroid fan above the cap. Triangle winding is the reverse of the
 * boundary half-edge direction so each fill faces outward. */

/* Min-area DP: for each (i, k, j) sub-polygon find the split index k
 * that minimises sum of triangle areas. Reconstruct the triangulation
 * via an explicit stack so deep loops never recurse. Allocates from
 * mesh->alloc; caller must check the returned error. */
static pamo_error fill_loop_min_area(pamo_mesh *mesh,
                                     const int32_t *loop, size_t L) {
    double  *T = PAMO_ALLOC_ARRAY(&mesh->alloc, double,  L * L);
    int32_t *K = PAMO_ALLOC_ARRAY(&mesh->alloc, int32_t, L * L);
    if (!T || !K) {
        if (T) PAMO_FREE_ARRAY(&mesh->alloc, T, double,  L * L);
        if (K) PAMO_FREE_ARRAY(&mesh->alloc, K, int32_t, L * L);
        return PAMO_ERR_ALLOC;
    }
    for (size_t i = 0; i < L * L; i++) { T[i] = 0.0; K[i] = -1; }
    for (size_t span = 2; span < L; span++) {
        for (size_t i = 0; i + span < L; i++) {
            size_t j = i + span;
            double best = 1e300;
            int32_t best_k = -1;
            for (size_t k = i + 1; k < j; k++) {
                pamo_vec3d a  = mesh->verts[loop[i]];
                pamo_vec3d b  = mesh->verts[loop[k]];
                pamo_vec3d c2 = mesh->verts[loop[j]];
                pamo_vec3d cr = pamo_v3_cross(
                    pamo_v3_sub(b, a), pamo_v3_sub(c2, a));
                double area = 0.5 * sqrt(pamo_v3_length_sq(cr));
                double cand = T[i * L + k] + T[k * L + j] + area;
                if (cand < best) { best = cand; best_k = (int32_t)k; }
            }
            T[i * L + j] = best;
            K[i * L + j] = best_k;
        }
    }
    int32_t *stk = PAMO_ALLOC_ARRAY(&mesh->alloc, int32_t, 2 * L);
    if (!stk) {
        PAMO_FREE_ARRAY(&mesh->alloc, T, double,  L * L);
        PAMO_FREE_ARRAY(&mesh->alloc, K, int32_t, L * L);
        return PAMO_ERR_ALLOC;
    }
    size_t sp = 0;
    stk[sp++] = 0;
    stk[sp++] = (int32_t)L - 1;
    pamo_error err = PAMO_OK;
    while (sp > 0 && err == PAMO_OK) {
        int32_t j = stk[--sp];
        int32_t i = stk[--sp];
        if (j - i < 2) continue;
        int32_t k = K[(size_t)i * L + (size_t)j];
        if (k < 0) continue;
        err = pamo_mesh_append_face(mesh,
            loop[(size_t)j], loop[(size_t)k], loop[(size_t)i]);
        if (err != PAMO_OK) break;
        stk[sp++] = i; stk[sp++] = k;
        stk[sp++] = k; stk[sp++] = j;
    }
    PAMO_FREE_ARRAY(&mesh->alloc, stk, int32_t, 2 * L);
    PAMO_FREE_ARRAY(&mesh->alloc, T,   double,  L * L);
    PAMO_FREE_ARRAY(&mesh->alloc, K,   int32_t, L * L);
    return err;
}

/* Centroid-fan fallback: append a centroid vertex and emit L
 * triangles (cv, b, a) for each boundary edge a→b. Always succeeds
 * structurally; the only failure mode is allocation. */
static pamo_error fill_loop_fan(pamo_mesh *mesh,
                                const int32_t *loop, size_t L) {
    pamo_vec3d cc = {0, 0, 0};
    for (size_t i = 0; i < L; i++) {
        cc.x += mesh->verts[loop[i]].x;
        cc.y += mesh->verts[loop[i]].y;
        cc.z += mesh->verts[loop[i]].z;
    }
    cc.x /= (double)L; cc.y /= (double)L; cc.z /= (double)L;
    int32_t cv = -1;
    pamo_error err = pamo_mesh_append_vert(mesh, cc, &cv);
    if (err != PAMO_OK) return err;
    for (size_t i = 0; i < L; i++) {
        int32_t a = loop[i];
        int32_t b = loop[(i + 1) % L];
        err = pamo_mesh_append_face(mesh, cv, b, a);
        if (err != PAMO_OK) return err;
    }
    return PAMO_OK;
}

static pamo_error fill_holes(pamo_mesh *mesh) {
    pamo_error err = pamo_mesh_build_adjacency(mesh);
    if (err != PAMO_OK) return err;

    size_t n_bdy = 0;
    for (size_t ei = 0; ei < mesh->n_edges; ei++) {
        if (mesh->edge_face_offset[ei + 1]
            - mesh->edge_face_offset[ei] == 1) n_bdy++;
    }
    if (n_bdy == 0) {
        pamo_mesh_free_adjacency(mesh);
        return PAMO_OK;
    }

    /* Lift each boundary edge to a directed half-edge using the
     * surviving face's vertex order so loop walks are consistent. */
    int32_t *be_from = PAMO_ALLOC_ARRAY(&mesh->alloc, int32_t, n_bdy);
    int32_t *be_to   = PAMO_ALLOC_ARRAY(&mesh->alloc, int32_t, n_bdy);
    bool    *be_used = PAMO_ALLOC_ARRAY(&mesh->alloc, bool,    n_bdy);
    int32_t *loop    = PAMO_ALLOC_ARRAY(&mesh->alloc, int32_t, n_bdy);
    if (!be_from || !be_to || !be_used || !loop) {
        if (be_from) PAMO_FREE_ARRAY(&mesh->alloc, be_from, int32_t, n_bdy);
        if (be_to)   PAMO_FREE_ARRAY(&mesh->alloc, be_to,   int32_t, n_bdy);
        if (be_used) PAMO_FREE_ARRAY(&mesh->alloc, be_used, bool,    n_bdy);
        if (loop)    PAMO_FREE_ARRAY(&mesh->alloc, loop,    int32_t, n_bdy);
        pamo_mesh_free_adjacency(mesh);
        return PAMO_ERR_ALLOC;
    }
    for (size_t i = 0; i < n_bdy; i++) be_used[i] = false;

    size_t nb = 0;
    for (size_t ei = 0; ei < mesh->n_edges; ei++) {
        if (mesh->edge_face_offset[ei + 1]
            - mesh->edge_face_offset[ei] != 1) continue;
        int32_t u = mesh->edges[ei].u, v = mesh->edges[ei].v;
        int32_t fi = mesh->edge_face_list[mesh->edge_face_offset[ei]];
        const int32_t *fv = mesh->faces[fi].v;
        for (int k = 0; k < 3; k++) {
            int32_t a = fv[k], b = fv[(k + 1) % 3];
            if ((a == u && b == v) || (a == v && b == u)) {
                be_from[nb] = a;
                be_to[nb]   = b;
                nb++;
                break;
            }
        }
    }

    pamo_mesh_free_adjacency(mesh);

    /* Walk loops by chaining (to == next.from). Linear scan to find
     * the next match — n_bdy is tiny in practice. */
    pamo_error fill_err = PAMO_OK;
    for (size_t seed = 0; seed < n_bdy && fill_err == PAMO_OK; seed++) {
        if (be_used[seed]) continue;
        size_t L = 0;
        int32_t start = be_from[seed];
        size_t ci = seed;
        bool closed = false;
        for (size_t step = 0; step < n_bdy + 1; step++) {
            if (be_used[ci]) break;
            be_used[ci] = true;
            loop[L++] = be_from[ci];
            int32_t next_v = be_to[ci];
            if (next_v == start) { closed = true; break; }
            size_t found = SIZE_MAX;
            for (size_t k = 0; k < n_bdy; k++) {
                if (!be_used[k] && be_from[k] == next_v) {
                    found = k; break;
                }
            }
            if (found == SIZE_MAX) break;
            ci = found;
        }
        if (!closed || L < 3) continue;

        if (L <= PAMO_PP_DP_MAX) fill_err = fill_loop_min_area(mesh, loop, L);
        else                     fill_err = fill_loop_fan(mesh, loop, L);
    }

    PAMO_FREE_ARRAY(&mesh->alloc, loop,    int32_t, n_bdy);
    PAMO_FREE_ARRAY(&mesh->alloc, be_from, int32_t, n_bdy);
    PAMO_FREE_ARRAY(&mesh->alloc, be_to,   int32_t, n_bdy);
    PAMO_FREE_ARRAY(&mesh->alloc, be_used, bool,    n_bdy);
    return fill_err;
}

pamo_error pamo_simplify_postprocess(pamo_mesh *mesh) {
    if (!mesh) return PAMO_ERR_INVALID_ARG;

    /* A watertight (no-boundary-edge) input — typically Stage 1 DMC
     * output that the collapse loop didn't perforate — needs neither
     * bowtie split nor hole-fill. Skip both rather than risk turning
     * benign T-junctions into boundary edges. */
    if (!mesh_has_boundary(mesh)) return PAMO_OK;

    pamo_error err = split_bowties(mesh);
    if (err != PAMO_OK) return err;
    return fill_holes(mesh);
}
