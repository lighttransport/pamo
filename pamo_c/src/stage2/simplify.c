/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 2: Iterative mesh simplification.
 *
 * Algorithm (per round):
 *   1. Build adjacency (vertex->face, unique edges, edge->face)
 *   2. Compute per-vertex quadric matrices Q = sum(plane outer plane)
 *   3. Score each edge: quadric error + edge length penalty + skinny penalty
 *   4. Validity checks: manifold (2 shared neighbors), normal flip, skinny
 *   5. Sort edges by cost, greedily select non-conflicting collapses
 *   6. Apply collapses: midpoint, delete degenerate faces, remap indices
 *   7. Self-intersection detection via BVH + tri-tri test
 *   8. Undo collapses touching intersecting triangles
 *   9. Compact mesh
 */
#include "pamo/pamo_stage2.h"
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_internal.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

pamo_simplify_opts pamo_simplify_opts_default(void) {
    return (pamo_simplify_opts){
        .target_faces            = 100,
        .min_faces               = 10,
        .max_iters               = 100,
        .tolerance               = 4,
        .max_undo_retries        = 5,
        .skinny_area_threshold   = 1e-6,
        .skinny_penalty_weight   = 5.0,
        .cost_range              = 10.0,
        .check_self_intersection = true,
    };
}

/* ── Per-vertex quadric computation ──────────────────────────────── */

static pamo_error compute_quadrics(const pamo_mesh *m, pamo_quadric *Q) {
    for (size_t i = 0; i < m->n_verts; i++) Q[i] = pamo_quadric_zero();

    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi]) continue;
        pamo_vec3d n = pamo_face_normal(m, (int32_t)fi);
        double len = sqrt(pamo_v3_length_sq(n));
        if (len < 1e-30) continue;
        n = pamo_v3_scale(n, 1.0 / len);
        double d = -pamo_v3_dot(n, m->verts[m->faces[fi].v[0]]);
        pamo_quadric qf = pamo_quadric_from_plane(n.x, n.y, n.z, d);
        for (int k = 0; k < 3; k++) {
            Q[m->faces[fi].v[k]] = pamo_quadric_add(Q[m->faces[fi].v[k]], qf);
        }
    }
    return PAMO_OK;
}

/* ── Edge cost computation ───────────────────────────────────────── */

static const double COST_INF = 1e20;

static double compute_edge_cost(const pamo_mesh *m, const pamo_quadric *Q,
                                size_t edge_idx,
                                double scale, double threshold,
                                double skinny_weight, double skinny_thresh,
                                const bool *vert_locked) {
    pamo_edge e = m->edges[edge_idx];
    int32_t u = e.u, v = e.v;

    if (!m->vert_alive[u] || !m->vert_alive[v]) return COST_INF;
    if (vert_locked && (vert_locked[u] || vert_locked[v])) return COST_INF;

    /* Manifold check: exactly 2 shared neighbors. */
    int32_t sn = pamo_shared_neighbor_count(m, u, v);
    if (sn != 2) return COST_INF;

    pamo_vec3d pu = m->verts[u];
    pamo_vec3d pv = m->verts[v];
    pamo_vec3d mid = pamo_v3_scale(pamo_v3_add(pu, pv), 0.5);

    /* Normal flip check + skinny penalty. */
    double skinny_sum = 0.0;
    int32_t tri_count = 0;

    /* Check triangles incident to u. */
    int32_t u_start = m->vert_face_offset[u];
    int32_t u_end   = m->vert_face_offset[u + 1];
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        /* Skip shared triangles (those containing both u and v). */
        if ((fv[0] == v || fv[1] == v || fv[2] == v)) continue;

        pamo_vec3d old_n = pamo_face_unit_normal(m, fi);

        /* Compute new normal after replacing u with mid. */
        pamo_vec3d verts[3];
        for (int k = 0; k < 3; k++) {
            verts[k] = (fv[k] == u) ? mid : m->verts[fv[k]];
        }
        pamo_vec3d new_n_raw = pamo_v3_cross(pamo_v3_sub(verts[1], verts[0]),
                                             pamo_v3_sub(verts[2], verts[0]));
        double new_len = sqrt(pamo_v3_length_sq(new_n_raw));
        if (new_len < 1e-30) return COST_INF;
        pamo_vec3d new_n = pamo_v3_scale(new_n_raw, 1.0 / new_len);

        if (pamo_v3_dot(old_n, new_n) < 0.0) return COST_INF;

        double q = pamo_triangle_quality(verts[0], verts[1], verts[2]);
        skinny_sum += 1.0 - (q > 1.0 ? 1.0 : q);
        tri_count++;

        if (pamo_triangle_is_skinny(verts[0], verts[1], verts[2], skinny_thresh))
            return COST_INF;
    }

    /* Same for triangles incident to v. */
    int32_t v_start = m->vert_face_offset[v];
    int32_t v_end   = m->vert_face_offset[v + 1];
    for (int32_t i = v_start; i < v_end; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        if ((fv[0] == u || fv[1] == u || fv[2] == u)) continue;

        pamo_vec3d old_n = pamo_face_unit_normal(m, fi);

        pamo_vec3d verts[3];
        for (int k = 0; k < 3; k++) {
            verts[k] = (fv[k] == v) ? mid : m->verts[fv[k]];
        }
        pamo_vec3d new_n_raw = pamo_v3_cross(pamo_v3_sub(verts[1], verts[0]),
                                             pamo_v3_sub(verts[2], verts[0]));
        double new_len = sqrt(pamo_v3_length_sq(new_n_raw));
        if (new_len < 1e-30) return COST_INF;
        pamo_vec3d new_n = pamo_v3_scale(new_n_raw, 1.0 / new_len);

        if (pamo_v3_dot(old_n, new_n) < 0.0) return COST_INF;

        double q = pamo_triangle_quality(verts[0], verts[1], verts[2]);
        skinny_sum += 1.0 - (q > 1.0 ? 1.0 : q);
        tri_count++;

        if (pamo_triangle_is_skinny(verts[0], verts[1], verts[2], skinny_thresh))
            return COST_INF;
    }

    /* Quadric error at midpoint. */
    pamo_quadric Qsum = pamo_quadric_add(Q[u], Q[v]);
    double qerr = pamo_quadric_eval(&Qsum, mid);
    if (scale > 1e-30) qerr /= (scale * scale);

    /* Edge length penalty. */
    double edge_len = sqrt(pamo_v3_length_sq(pamo_v3_sub(pu, pv)));
    double edge_cost = 0.0;
    if (scale > 1e-30) edge_cost = edge_len / scale * threshold;

    /* Skinny penalty. */
    double skinny_cost = 0.0;
    if (tri_count > 0) {
        skinny_cost = skinny_weight * skinny_sum / (double)tri_count * threshold;
    }

    double total = qerr + edge_cost + skinny_cost;
    if (total < 0.0) total = 0.0;
    return total;
}

/* ── Self-intersection detection ─────────────────────────────────── */

static size_t count_self_intersections(const pamo_mesh *m,
                                       const pamo_allocator *alloc,
                                       int32_t **out_isect_faces,
                                       size_t *out_n_isect,
                                       size_t *out_isect_cap) {
    *out_isect_faces = NULL;
    *out_n_isect = 0;
    *out_isect_cap = 0;

    pamo_bvh bvh;
    if (pamo_bvh_build_triangles(&bvh, m, alloc) != PAMO_OK) return 0;

    pamo_overlap_result ores;
    pamo_overlap_result_init(&ores, alloc);

    /* For each face, query BVH for overlapping AABBs, then exact test. */
    size_t n_isect = 0;
    size_t isect_cap = 64;
    int32_t *isect = PAMO_ALLOC_ARRAY(alloc, int32_t, isect_cap);

    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi]) continue;
        const pamo_tri *f = &m->faces[fi];
        pamo_vec3d v0 = m->verts[f->v[0]];
        pamo_vec3d v1 = m->verts[f->v[1]];
        pamo_vec3d v2 = m->verts[f->v[2]];

        /* Build face AABB. */
        pamo_aabb fbox;
        fbox.lo.x = fmin(v0.x, fmin(v1.x, v2.x)) - 1e-10;
        fbox.lo.y = fmin(v0.y, fmin(v1.y, v2.y)) - 1e-10;
        fbox.lo.z = fmin(v0.z, fmin(v1.z, v2.z)) - 1e-10;
        fbox.hi.x = fmax(v0.x, fmax(v1.x, v2.x)) + 1e-10;
        fbox.hi.y = fmax(v0.y, fmax(v1.y, v2.y)) + 1e-10;
        fbox.hi.z = fmax(v0.z, fmax(v1.z, v2.z)) + 1e-10;

        ores.n_hits = 0;
        pamo_bvh_overlap(&bvh, fbox, &ores);

        for (size_t h = 0; h < ores.n_hits; h++) {
            int32_t gi = ores.hits[h];
            if (gi <= (int32_t)fi) continue;  /* avoid duplicates */
            if (!m->face_alive[gi]) continue;

            const pamo_tri *g = &m->faces[gi];
            /* Skip adjacent triangles (shared vertex). */
            bool shared = false;
            for (int a = 0; a < 3 && !shared; a++)
                for (int b = 0; b < 3 && !shared; b++)
                    if (f->v[a] == g->v[b]) shared = true;
            if (shared) continue;

            pamo_vec3d w0 = m->verts[g->v[0]];
            pamo_vec3d w1 = m->verts[g->v[1]];
            pamo_vec3d w2 = m->verts[g->v[2]];

            if (pamo_tri_tri_intersect(v0, v1, v2, w0, w1, w2)) {
                /* Record both faces. */
                if (n_isect + 2 > isect_cap) {
                    size_t new_cap = isect_cap * 2;
                    isect = (int32_t *)pamo_realloc(alloc, isect,
                        isect_cap * sizeof(int32_t),
                        new_cap * sizeof(int32_t));
                    isect_cap = new_cap;
                }
                isect[n_isect++] = (int32_t)fi;
                isect[n_isect++] = gi;
            }
        }
    }

    pamo_overlap_result_destroy(&ores);
    pamo_bvh_destroy(&bvh);

    *out_isect_faces = isect;
    *out_n_isect = n_isect;
    *out_isect_cap = isect_cap;
    return n_isect / 2;
}

/* ── Collapse record for undo ────────────────────────────────────── */

typedef struct {
    int32_t    edge_idx;
    int32_t    u, v;
    pamo_vec3d orig_u, orig_v;
} pamo_collapse_record;

/* ── One simplification round ────────────────────────────────────── */

static pamo_error simplify_round(pamo_mesh *m,
                                 const pamo_simplify_opts *opts,
                                 const bool *vert_locked,
                                 size_t *n_collapsed_out) {
    pamo_allocator *alloc = &m->alloc;
    pamo_error err;

    *n_collapsed_out = 0;

    /* 1. Build adjacency. */
    err = pamo_mesh_build_adjacency(m);
    if (err != PAMO_OK) return err;

    if (m->n_edges == 0) return PAMO_OK;

    /* 2. Compute quadrics. */
    pamo_quadric *Q = PAMO_ALLOC_ARRAY(alloc, pamo_quadric, m->n_verts);
    if (!Q) return PAMO_ERR_ALLOC;
    compute_quadrics(m, Q);

    /* Compute scale (average edge length). */
    double scale_sum = 0.0;
    size_t scale_cnt = 0;
    for (size_t i = 0; i < m->n_edges; i++) {
        pamo_vec3d d = pamo_v3_sub(m->verts[m->edges[i].u],
                                   m->verts[m->edges[i].v]);
        scale_sum += sqrt(pamo_v3_length_sq(d));
        scale_cnt++;
    }
    double scale = (scale_cnt > 0) ? scale_sum / (double)scale_cnt : 1.0;
    double threshold = 1.0;

    /* 3. Score edges. */
    pamo_edge_cost_entry *costs = PAMO_ALLOC_ARRAY(alloc, pamo_edge_cost_entry,
                                                   m->n_edges);
    if (!costs) {
        PAMO_FREE_ARRAY(alloc, Q, pamo_quadric, m->n_verts);
        return PAMO_ERR_ALLOC;
    }

    size_t n_valid = 0;
    for (size_t i = 0; i < m->n_edges; i++) {
        double c = compute_edge_cost(m, Q, i, scale, threshold,
                                     opts->skinny_penalty_weight,
                                     opts->skinny_area_threshold,
                                     vert_locked);
        if (c < opts->cost_range) {
            costs[n_valid].cost = c;
            costs[n_valid].edge_idx = (int32_t)i;
            n_valid++;
        }
    }

    /* 4. Sort by cost. */
    pamo_sort_edge_costs(costs, n_valid);

    /* 5. Greedy selection: lock both endpoints. */
    bool *locked = PAMO_ALLOC_ARRAY(alloc, bool, m->n_verts);
    if (!locked) {
        PAMO_FREE_ARRAY(alloc, costs, pamo_edge_cost_entry, m->n_edges);
        PAMO_FREE_ARRAY(alloc, Q, pamo_quadric, m->n_verts);
        return PAMO_ERR_ALLOC;
    }

    size_t max_collapses = n_valid;
    pamo_collapse_record *records = PAMO_ALLOC_ARRAY(alloc, pamo_collapse_record,
                                                     max_collapses > 0 ? max_collapses : 1);
    size_t n_accepted = 0;

    for (size_t i = 0; i < n_valid; i++) {
        int32_t ei = costs[i].edge_idx;
        int32_t u = m->edges[ei].u;
        int32_t v = m->edges[ei].v;

        if (locked[u] || locked[v]) continue;

        locked[u] = true;
        locked[v] = true;

        records[n_accepted].edge_idx = ei;
        records[n_accepted].u = u;
        records[n_accepted].v = v;
        records[n_accepted].orig_u = m->verts[u];
        records[n_accepted].orig_v = m->verts[v];
        n_accepted++;
    }

    /* 6. Apply collapses. */
    for (size_t i = 0; i < n_accepted; i++) {
        int32_t u = records[i].u;
        int32_t v = records[i].v;
        pamo_vec3d mid = pamo_v3_scale(
            pamo_v3_add(m->verts[u], m->verts[v]), 0.5);

        m->verts[u] = mid;
        m->vert_alive[v] = false;

        /* Remap v -> u in all faces incident to v, delete shared faces. */
        int32_t v_start = m->vert_face_offset[v];
        int32_t v_end   = m->vert_face_offset[v + 1];
        for (int32_t fi_idx = v_start; fi_idx < v_end; fi_idx++) {
            int32_t fi = m->vert_face_list[fi_idx];
            if (!m->face_alive[fi]) continue;
            int32_t *fv = m->faces[fi].v;

            bool has_u = (fv[0] == u || fv[1] == u || fv[2] == u);
            if (has_u) {
                /* Shared triangle: delete. */
                m->face_alive[fi] = false;
            } else {
                /* Remap v -> u. */
                for (int k = 0; k < 3; k++) {
                    if (fv[k] == v) fv[k] = u;
                }
                /* Check for degenerate face. */
                if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0]) {
                    m->face_alive[fi] = false;
                }
            }
        }
    }

    /* 7. Self-intersection rollback. */
    if (opts->check_self_intersection && n_accepted > 0) {
        for (int retry = 0; retry < opts->max_undo_retries; retry++) {
            int32_t *isect_faces = NULL;
            size_t n_isect = 0;
            size_t isect_cap = 0;
            count_self_intersections(m, alloc, &isect_faces, &n_isect, &isect_cap);

            if (n_isect == 0) {
                if (isect_faces) pamo_free(alloc, isect_faces,
                    isect_cap * sizeof(int32_t));
                break;
            }

            /* Mark intersecting face IDs. */
            bool *isect_set = PAMO_ALLOC_ARRAY(alloc, bool, m->n_faces);
            for (size_t i = 0; i < n_isect; i++) {
                if (isect_faces[i] >= 0 && (size_t)isect_faces[i] < m->n_faces) {
                    isect_set[isect_faces[i]] = true;
                }
            }

            /* Find collapses touching intersecting faces and undo them. */
            size_t n_undone = 0;
            for (size_t ci = 0; ci < n_accepted; ci++) {
                if (records[ci].u < 0) continue; /* already undone */
                int32_t u = records[ci].u;
                bool touches = false;

                int32_t u_start = m->vert_face_offset[u];
                int32_t u_end   = m->vert_face_offset[u + 1];
                for (int32_t j = u_start; j < u_end && !touches; j++) {
                    int32_t fi = m->vert_face_list[j];
                    if (fi >= 0 && (size_t)fi < m->n_faces && isect_set[fi])
                        touches = true;
                }

                /* Also check v's original faces. */
                int32_t v = records[ci].v;
                int32_t v_start = m->vert_face_offset[v];
                int32_t v_end   = m->vert_face_offset[v + 1];
                for (int32_t j = v_start; j < v_end && !touches; j++) {
                    int32_t fi = m->vert_face_list[j];
                    if (fi >= 0 && (size_t)fi < m->n_faces && isect_set[fi])
                        touches = true;
                }

                if (touches) {
                    /* Undo: restore original positions, revive v. */
                    m->verts[u] = records[ci].orig_u;
                    m->verts[v] = records[ci].orig_v;
                    m->vert_alive[v] = true;

                    /* Restore faces incident to v. */
                    /* We can't perfectly restore deleted faces without
                     * saving them, so we re-enable them and fix indices.
                     * This is a simplification -- we just revive all
                     * faces that were incident to v before collapse. */
                    for (int32_t j = v_start; j < v_end; j++) {
                        int32_t fi = m->vert_face_list[j];
                        if ((size_t)fi < m->n_faces) {
                            m->face_alive[fi] = true;
                        }
                    }

                    records[ci].u = -1; /* mark undone */
                    n_undone++;
                }
            }

            PAMO_FREE_ARRAY(alloc, isect_set, bool, m->n_faces);
            pamo_free(alloc, isect_faces, isect_cap * sizeof(int32_t));

            if (n_undone == 0) break;

            /* Rebuild adjacency for next self-intersection check. */
            pamo_mesh_free_adjacency(m);
            pamo_mesh_build_adjacency(m);
        }
        n_accepted -= 0; /* count final collapses after undo */
    }

    /* Count actual collapses (not undone). */
    size_t final_count = 0;
    for (size_t i = 0; i < n_accepted; i++) {
        if (records[i].u >= 0) final_count++;
    }
    *n_collapsed_out = final_count;

    /* Cleanup. */
    PAMO_FREE_ARRAY(alloc, records, pamo_collapse_record,
                    max_collapses > 0 ? max_collapses : 1);
    PAMO_FREE_ARRAY(alloc, locked, bool, m->n_verts);
    PAMO_FREE_ARRAY(alloc, costs, pamo_edge_cost_entry, m->n_edges);
    PAMO_FREE_ARRAY(alloc, Q, pamo_quadric, m->n_verts);
    pamo_mesh_free_adjacency(m);

    return PAMO_OK;
}

/* ── Main simplification driver ──────────────────────────────────── */

pamo_error pamo_simplify(pamo_mesh *mesh, const pamo_simplify_opts *opts) {
    if (!mesh || !opts) return PAMO_ERR_INVALID_ARG;

    int32_t stuck_counter = 0;
    bool *vert_locked = NULL;  /* for stuck-mode vertex exclusion */

    for (int32_t iter = 0; iter < opts->max_iters; iter++) {
        size_t alive_faces = pamo_mesh_count_alive_faces(mesh);

        if ((int32_t)alive_faces <= opts->target_faces) break;
        if ((int32_t)alive_faces <= opts->min_faces) break;

        size_t n_collapsed = 0;
        pamo_error err = simplify_round(mesh, opts, vert_locked, &n_collapsed);
        if (err != PAMO_OK) {
            if (vert_locked) {
                PAMO_FREE_ARRAY(&mesh->alloc, vert_locked, bool, mesh->n_verts);
            }
            return err;
        }

        /* Compact to remove dead vertices/faces. */
        err = pamo_mesh_compact(mesh);
        if (err != PAMO_OK) {
            if (vert_locked) {
                PAMO_FREE_ARRAY(&mesh->alloc, vert_locked, bool, mesh->n_verts);
                vert_locked = NULL;
            }
            return err;
        }

        /* Free old vert_locked since indices changed after compact. */
        if (vert_locked) {
            /* Size may have been from previous n_verts. Free conservatively. */
            /* We'll just free and re-create if needed. */
        }

        size_t new_alive = pamo_mesh_count_alive_faces(mesh);

        if (new_alive >= alive_faces) {
            stuck_counter++;
        } else {
            stuck_counter = 0;
        }

        if (stuck_counter >= opts->tolerance) {
            break;
        }
    }

    if (vert_locked) {
        /* Nothing to free since we didn't allocate in this version. */
    }

    return PAMO_OK;
}
