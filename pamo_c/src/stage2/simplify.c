/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 2: Iterative mesh simplification.
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
        .tolerance               = 8,
        .max_undo_retries        = 5,
        .skinny_area_threshold   = 1e-6,
        .skinny_penalty_weight   = 5.0,
        .cost_range              = 10.0,
        .check_self_intersection = false,
    };
}

/* Upper bound on 1-ring valence we handle on the stack. Real meshes
 * almost never have vertices with more than ~30 incident triangles; we
 * allow 8× that to stay safe with only 2 KiB of stack per call. When a
 * vertex exceeds this we take the conservative path — report the
 * collapse as unsafe or skip the cleanup pass for that vertex — instead
 * of silently truncating the neighbour list. */
#define PAMO_MAX_VALENCE 256

/* ── Per-vertex quadric computation ──────────────────────────────── */

static void compute_quadrics(const pamo_mesh *m, pamo_quadric *Q) {
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
            int32_t vi = m->faces[fi].v[k];
            Q[vi] = pamo_quadric_add(Q[vi], qf);
        }
    }
}

/* ── Edge cost computation ───────────────────────────────────────── */

static const double COST_INF = 1e20;

/* Check if collapsing endpoint→mid flips any adjacent normal or
 * creates a skinny triangle. Returns true if INVALID. */
static bool collapse_creates_problem(const pamo_mesh *m,
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

/* Count how many alive faces are shared between u and v
 * (i.e., faces that contain both u and v). */
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

/* Check the "link condition" for edge collapse safety.
 *
 * For a manifold interior edge (u,v) with exactly 2 shared faces:
 *   The collapse is safe if the two "tip" vertices (the third vertex
 *   of each shared triangle) are NOT connected by any face that doesn't
 *   also contain u or v. This prevents creating non-manifold vertices.
 *
 * For boundary edges (1 shared face): similar but simpler check.
 *
 * Returns true if the collapse would create a topology problem. */
static bool collapse_violates_link(const pamo_mesh *m,
                                   int32_t u, int32_t v) {
    /* Collect vertices adjacent to u (excluding v). */
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
            /* Overflow: conservatively report as unsafe rather than
             * silently drop neighbours, which could cause us to accept
             * a collapse that would actually create a non-manifold
             * vertex. */
            if (n_u >= PAMO_MAX_VALENCE) return true;
            u_adj[n_u++] = w;
        }
    }

    /* Collect vertices adjacent to v (excluding u). */
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

    /* Count vertices in the intersection of u_adj and v_adj. */
    int32_t n_shared = 0;
    for (int32_t i = 0; i < n_u; i++) {
        for (int32_t j = 0; j < n_v; j++) {
            if (u_adj[i] == v_adj[j]) { n_shared++; break; }
        }
    }

    /* For an interior edge: exactly 2 shared neighbors expected
     * (the two triangle tips). More means non-manifold result.
     * For a boundary edge: exactly 1 shared neighbor expected.
     * If 0 shared neighbors, u and v are in disconnected components. */
    int32_t n_shared_faces = count_shared_faces(m, u, v);

    if (n_shared_faces == 2) {
        /* Interior edge: must have exactly 2 shared vertex neighbors. */
        return n_shared != 2;
    } else if (n_shared_faces == 1) {
        /* Boundary edge: must have exactly 1 shared vertex neighbor. */
        return n_shared != 1;
    } else {
        /* Edge with 0 or >2 shared faces: not safe to collapse. */
        return true;
    }
}

static double compute_edge_cost(const pamo_mesh *m, const pamo_quadric *Q,
                                size_t edge_idx, double scale,
                                double skinny_weight, double skinny_thresh,
                                double flip_threshold,
                                const bool *vert_locked) {
    pamo_edge e = m->edges[edge_idx];
    int32_t u = e.u, v = e.v;

    if (!m->vert_alive[u] || !m->vert_alive[v]) return COST_INF;
    if (vert_locked && (vert_locked[u] || vert_locked[v])) return COST_INF;

    /* Link condition: ensures collapse preserves manifold topology. */
    if (collapse_violates_link(m, u, v)) return COST_INF;

    pamo_vec3d pu = m->verts[u];
    pamo_vec3d pv = m->verts[v];
    pamo_vec3d mid = pamo_v3_scale(pamo_v3_add(pu, pv), 0.5);

    /* Normal flip / skinny check for both endpoints. */
    if (collapse_creates_problem(m, u, v, mid, skinny_thresh, flip_threshold))
        return COST_INF;
    if (collapse_creates_problem(m, v, u, mid, skinny_thresh, flip_threshold))
        return COST_INF;

    /* Quadric error at midpoint. */
    pamo_quadric Qsum = pamo_quadric_add(Q[u], Q[v]);
    double qerr = pamo_quadric_eval(&Qsum, mid);
    if (qerr < 0.0) qerr = 0.0;
    double s2 = scale * scale;
    if (s2 > 1e-30) qerr /= s2;

    /* Edge length penalty. */
    double edge_len = sqrt(pamo_v3_length_sq(pamo_v3_sub(pu, pv)));
    double edge_cost = (scale > 1e-30) ? edge_len / scale : 0.0;

    /* Skinny penalty. */
    double skinny_cost = 0.0;
    int32_t tri_count = 0;
    double skinny_sum = 0.0;

    for (int pass = 0; pass < 2; pass++) {
        int32_t endpoint = (pass == 0) ? u : v;
        int32_t other    = (pass == 0) ? v : u;
        int32_t s = m->vert_face_offset[endpoint];
        int32_t e2 = m->vert_face_offset[endpoint + 1];
        for (int32_t i = s; i < e2; i++) {
            int32_t fi = m->vert_face_list[i];
            if (!m->face_alive[fi]) continue;
            const int32_t *fv = m->faces[fi].v;
            if (fv[0] == other || fv[1] == other || fv[2] == other) continue;
            pamo_vec3d verts[3];
            for (int k = 0; k < 3; k++)
                verts[k] = (fv[k] == endpoint) ? mid : m->verts[fv[k]];
            double q = pamo_triangle_quality(verts[0], verts[1], verts[2]);
            if (q > 1.0) q = 1.0;
            skinny_sum += 1.0 - q;
            tri_count++;
        }
    }
    if (tri_count > 0) {
        skinny_cost = skinny_weight * skinny_sum / (double)tri_count;
    }

    return qerr + edge_cost + skinny_cost;
}

/* ── Collapse record for undo ────────────────────────────────────── */

typedef struct {
    int32_t    u, v;
    pamo_vec3d orig_u, orig_v;
    bool       undone;
} pamo_collapse_record;

/* ── Apply collapse ──────────────────────────────────────────────── */

static void apply_collapse(pamo_mesh *m, int32_t u, int32_t v,
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
            /* Shared face: delete (it would become degenerate). */
            m->face_alive[fi] = false;
        } else {
            /* Remap v -> u. */
            for (int k = 0; k < 3; k++) {
                if (fv[k] == v) fv[k] = u;
            }
            /* Check for degenerate (shouldn't happen with link condition). */
            if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0]) {
                m->face_alive[fi] = false;
            }
        }
    }

    /* Also check u's faces for duplicates: after remap, two faces
     * incident to u may now be identical. Remove duplicates. */
    int32_t u_start = m->vert_face_offset[u];
    int32_t u_end   = m->vert_face_offset[u + 1];
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t fi = m->vert_face_list[i];
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        /* Check against v's remapped faces. */
        for (int32_t j = v_start; j < v_end; j++) {
            int32_t fj = m->vert_face_list[j];
            if (!m->face_alive[fj] || fj == fi) continue;
            const int32_t *gv = m->faces[fj].v;
            /* Same triangle if same sorted vertex set. */
            int32_t fa[3] = {fv[0], fv[1], fv[2]};
            int32_t ga[3] = {gv[0], gv[1], gv[2]};
            /* Sort both. */
            for (int a = 0; a < 2; a++) for (int b = a+1; b < 3; b++) {
                if (fa[a] > fa[b]) { int32_t t=fa[a]; fa[a]=fa[b]; fa[b]=t; }
                if (ga[a] > ga[b]) { int32_t t=ga[a]; ga[a]=ga[b]; ga[b]=t; }
            }
            if (fa[0]==ga[0] && fa[1]==ga[1] && fa[2]==ga[2]) {
                m->face_alive[fj] = false;
            }
        }
    }
}

/* ── One simplification round ────────────────────────────────────── */

static pamo_error simplify_round(pamo_mesh *m,
                                 const pamo_simplify_opts *opts,
                                 const bool *vert_locked,
                                 double cost_limit,
                                 bool relax_normals,
                                 size_t *n_collapsed_out) {
    pamo_allocator *alloc = &m->alloc;
    *n_collapsed_out = 0;

    pamo_error err = pamo_mesh_build_adjacency(m);
    if (err != PAMO_OK) return err;
    if (m->n_edges == 0) {
        pamo_mesh_free_adjacency(m);
        return PAMO_OK;
    }

    /* Compute quadrics. */
    pamo_quadric *Q = PAMO_ALLOC_ARRAY(alloc, pamo_quadric, m->n_verts);
    if (!Q) { pamo_mesh_free_adjacency(m); return PAMO_ERR_ALLOC; }
    compute_quadrics(m, Q);

    /* Compute scale. */
    double scale_sum = 0.0;
    for (size_t i = 0; i < m->n_edges; i++) {
        pamo_vec3d d = pamo_v3_sub(m->verts[m->edges[i].u],
                                   m->verts[m->edges[i].v]);
        scale_sum += sqrt(pamo_v3_length_sq(d));
    }
    double scale = (m->n_edges > 0) ? scale_sum / (double)m->n_edges : 1.0;

    /* Score edges. */
    pamo_edge_cost_entry *costs = PAMO_ALLOC_ARRAY(alloc, pamo_edge_cost_entry,
                                                   m->n_edges);
    if (!costs) {
        PAMO_FREE_ARRAY(alloc, Q, pamo_quadric, m->n_verts);
        pamo_mesh_free_adjacency(m);
        return PAMO_ERR_ALLOC;
    }

    /* Normal flip threshold: 0 = strict (no flip allowed),
     * negative = allow slight flips when relaxing constraints. */
    double flip_thresh = relax_normals ? -0.5 : 0.0;

    size_t n_valid = 0;
    for (size_t i = 0; i < m->n_edges; i++) {
        double c = compute_edge_cost(m, Q, i, scale,
                                     opts->skinny_penalty_weight,
                                     opts->skinny_area_threshold,
                                     flip_thresh,
                                     vert_locked);
        if (c < cost_limit) {
            costs[n_valid].cost = c;
            costs[n_valid].edge_idx = (int32_t)i;
            n_valid++;
        }
    }

    pamo_sort_edge_costs(costs, n_valid);

    /* Greedy selection. */
    bool *locked = PAMO_ALLOC_ARRAY(alloc, bool, m->n_verts);
    if (!locked) goto cleanup;

    size_t records_cap = n_valid > 0 ? n_valid : 1;
    pamo_collapse_record *records = PAMO_ALLOC_ARRAY(alloc, pamo_collapse_record,
                                                     records_cap);
    if (!records) goto cleanup;

    size_t n_accepted = 0;
    for (size_t i = 0; i < n_valid; i++) {
        int32_t ei = costs[i].edge_idx;
        int32_t u = m->edges[ei].u;
        int32_t v = m->edges[ei].v;
        if (locked[u] || locked[v]) continue;

        /* Lock 1-ring neighborhood of both endpoints.
         * This prevents adjacent edges from collapsing in the same round.
         * The runtime link-condition re-check (below) catches any
         * remaining topology issues from stale adjacency. */
        locked[u] = true;
        locked[v] = true;
        for (int pass = 0; pass < 2; pass++) {
            int32_t ep = (pass == 0) ? u : v;
            int32_t ep_s = m->vert_face_offset[ep];
            int32_t ep_e = m->vert_face_offset[ep + 1];
            for (int32_t j = ep_s; j < ep_e; j++) {
                int32_t fi = m->vert_face_list[j];
                if (!m->face_alive[fi]) continue;
                for (int k = 0; k < 3; k++)
                    locked[m->faces[fi].v[k]] = true;
            }
        }

        records[n_accepted].u = u;
        records[n_accepted].v = v;
        records[n_accepted].orig_u = m->verts[u];
        records[n_accepted].orig_v = m->verts[v];
        records[n_accepted].undone = false;
        n_accepted++;
    }

    /* Apply collapses with runtime validation.
     * Re-check the link condition before each collapse since
     * earlier collapses in this round may have changed topology. */
    size_t actually_collapsed = 0;
    for (size_t i = 0; i < n_accepted; i++) {
        int32_t u = records[i].u, v = records[i].v;
        if (!m->vert_alive[u] || !m->vert_alive[v]) continue;

        /* Re-verify link condition with current face state. */
        /* Count shared faces (faces containing both u and v). */
        int32_t shared = 0;
        int32_t u_s2 = m->vert_face_offset[u];
        int32_t u_e2 = m->vert_face_offset[u + 1];
        for (int32_t j = u_s2; j < u_e2; j++) {
            int32_t fi = m->vert_face_list[j];
            if (!m->face_alive[fi]) continue;
            const int32_t *fv = m->faces[fi].v;
            if (fv[0] == v || fv[1] == v || fv[2] == v) shared++;
        }
        if (shared == 0) continue;  /* disconnected: skip */

        /* Count shared vertex neighbors. */
        int32_t u_nb[PAMO_MAX_VALENCE], n_un = 0;
        bool nb_overflow = false;
        for (int32_t j = u_s2; j < u_e2 && !nb_overflow; j++) {
            int32_t fi = m->vert_face_list[j];
            if (!m->face_alive[fi]) continue;
            const int32_t *fv = m->faces[fi].v;
            for (int k = 0; k < 3; k++) {
                int32_t w = fv[k];
                if (w == u || w == v) continue;
                bool dup = false;
                for (int32_t d = 0; d < n_un; d++)
                    if (u_nb[d] == w) { dup = true; break; }
                if (dup) continue;
                if (n_un >= PAMO_MAX_VALENCE) { nb_overflow = true; break; }
                u_nb[n_un++] = w;
            }
        }
        /* Valence exceeded stack buffer — skip this collapse rather
         * than silently truncating the neighbour set. */
        if (nb_overflow) continue;
        int32_t v_s2 = m->vert_face_offset[v];
        int32_t v_e2 = m->vert_face_offset[v + 1];
        int32_t n_shared_nb = 0;
        for (int32_t j = v_s2; j < v_e2; j++) {
            int32_t fi = m->vert_face_list[j];
            if (!m->face_alive[fi]) continue;
            const int32_t *fv = m->faces[fi].v;
            for (int k = 0; k < 3; k++) {
                int32_t w = fv[k];
                if (w == u || w == v) continue;
                for (int32_t d = 0; d < n_un; d++) {
                    if (u_nb[d] == w) { n_shared_nb++; u_nb[d] = -1; break; }
                }
            }
        }

        bool safe;
        if (shared == 2) safe = (n_shared_nb == 2);
        else if (shared == 1) safe = (n_shared_nb == 1);
        else safe = false;

        if (!safe) continue;

        pamo_vec3d mid = pamo_v3_scale(
            pamo_v3_add(m->verts[u], m->verts[v]), 0.5);
        apply_collapse(m, u, v, mid);
        actually_collapsed++;
    }

    *n_collapsed_out = actually_collapsed;

    PAMO_FREE_ARRAY(alloc, records, pamo_collapse_record, records_cap);
cleanup:
    if (locked) PAMO_FREE_ARRAY(alloc, locked, bool, m->n_verts);
    PAMO_FREE_ARRAY(alloc, costs, pamo_edge_cost_entry, m->n_edges);
    PAMO_FREE_ARRAY(alloc, Q, pamo_quadric, m->n_verts);
    pamo_mesh_free_adjacency(m);
    return PAMO_OK;
}

/* ── Main simplification driver ──────────────────────────────────── */

pamo_error pamo_simplify(pamo_mesh *mesh, const pamo_simplify_opts *opts) {
    if (!mesh || !opts) return PAMO_ERR_INVALID_ARG;

    int32_t stuck_counter = 0;
    double cost_limit = opts->cost_range;

    for (int32_t iter = 0; iter < opts->max_iters; iter++) {
        size_t alive_faces = pamo_mesh_count_alive_faces(mesh);
        if ((int32_t)alive_faces <= opts->target_faces) break;
        if ((int32_t)alive_faces <= opts->min_faces) break;

        size_t n_collapsed = 0;
        bool relax = (stuck_counter >= 2); /* relax normals when deeply stuck */
        pamo_error err = simplify_round(mesh, opts, NULL,
                                        cost_limit, relax, &n_collapsed);
        if (err != PAMO_OK) return err;

        err = pamo_mesh_compact(mesh);
        if (err != PAMO_OK) return err;

        size_t new_alive = pamo_mesh_count_alive_faces(mesh);

        if (new_alive >= alive_faces) {
            stuck_counter++;
            cost_limit *= 2.0;
        } else {
            stuck_counter = 0;
            cost_limit = opts->cost_range;
        }

        if (stuck_counter >= opts->tolerance) break;
    }

    /* ── Post-process: remove T-junction non-manifold vertices ────── */
    /* Only run this cleanup for non-watertight input meshes.
     * For watertight input (from Stage 1), T-junction removal creates
     * boundary edges which is worse than keeping the T-junctions.
     * The 2-ring locking + runtime link condition prevents most
     * T-junctions from forming during collapse. */
    /* Check if mesh has boundary edges. */
    pamo_error adj_err = pamo_mesh_build_adjacency(mesh);
    bool has_boundary = false;
    if (adj_err == PAMO_OK && mesh->edge_face_offset) {
        for (size_t ei = 0; ei < mesh->n_edges && !has_boundary; ei++) {
            if (mesh->edge_face_offset[ei+1] - mesh->edge_face_offset[ei] == 1)
                has_boundary = true;
        }
        pamo_mesh_free_adjacency(mesh);
    }
    if (!has_boundary) return PAMO_OK; /* watertight: skip cleanup */

    for (int cleanup = 0; cleanup < 10; cleanup++) {
        pamo_error err = pamo_mesh_build_adjacency(mesh);
        if (err != PAMO_OK) return err;

        bool found_any = false;
        for (size_t vi = 0; vi < mesh->n_verts; vi++) {
            if (!mesh->vert_alive[vi]) continue;
            int32_t s = mesh->vert_face_offset[vi];
            int32_t e = mesh->vert_face_offset[vi + 1];
            int32_t n_inc = 0;
            int32_t inc_faces[PAMO_MAX_VALENCE];
            bool inc_overflow = false;
            for (int32_t j = s; j < e; j++) {
                int32_t fi = mesh->vert_face_list[j];
                if (!mesh->face_alive[fi]) continue;
                if (n_inc >= PAMO_MAX_VALENCE) { inc_overflow = true; break; }
                inc_faces[n_inc++] = fi;
            }
            /* Skip extreme-valence vertices rather than truncating their
             * incident-face list, which would misreport the component
             * count and could delete real geometry. */
            if (inc_overflow) continue;
            if (n_inc <= 1) continue;

            /* BFS over face adjacency around this vertex. */
            bool visited[PAMO_MAX_VALENCE] = {false};
            visited[0] = true;
            int32_t queue[PAMO_MAX_VALENCE];
            queue[0] = 0;
            int32_t qhead = 0, qtail = 1;
            while (qhead < qtail) {
                int32_t ci = queue[qhead++];
                int32_t fi = inc_faces[ci];
                const int32_t *fv = mesh->faces[fi].v;
                int32_t others[2], no = 0;
                for (int k = 0; k < 3; k++)
                    if (fv[k] != (int32_t)vi && no < 2) others[no++] = fv[k];

                for (int32_t ci2 = 0; ci2 < n_inc; ci2++) {
                    if (visited[ci2]) continue;
                    int32_t fi2 = inc_faces[ci2];
                    const int32_t *gv = mesh->faces[fi2].v;
                    bool adj = false;
                    for (int a = 0; a < no && !adj; a++)
                        for (int b = 0; b < 3 && !adj; b++)
                            if (gv[b] == others[a] && gv[b] != (int32_t)vi)
                                adj = true;
                    if (adj) {
                        visited[ci2] = true;
                        queue[qtail++] = ci2;
                    }
                }
            }

            if (qtail < n_inc) {
                /* Multiple components. Kill faces NOT in the largest. */
                /* First component is visited[0..qtail-1]. Count others. */
                /* Simple: just kill all faces in smaller components. */
                found_any = true;
                for (int32_t ci = 0; ci < n_inc; ci++) {
                    if (!visited[ci]) {
                        mesh->face_alive[inc_faces[ci]] = false;
                    }
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
