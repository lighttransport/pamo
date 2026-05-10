/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 2: Iterative mesh simplification.
 */
#include "pamo/pamo_stage2.h"
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_internal.h"
#include "pamo/pamo_progress.h"

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
        .fold_guard_min_dot      = -0.2,
        .check_self_intersection = true,
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
        if (!pamo_mesh_face_is_valid(m, fi)) continue;
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

static bool solve_symmetric_3x3(double m[3][3], double rhs[3],
                                pamo_vec3d *x) {
    for (int col = 0; col < 3; col++) {
        int pivot = col;
        double best = fabs(m[col][col]);
        for (int row = col + 1; row < 3; row++) {
            double v = fabs(m[row][col]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < 1.0e-14) return false;
        if (pivot != col) {
            for (int k = col; k < 3; k++) {
                double t = m[col][k];
                m[col][k] = m[pivot][k];
                m[pivot][k] = t;
            }
            double tb = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = tb;
        }

        double inv = 1.0 / m[col][col];
        for (int row = col + 1; row < 3; row++) {
            double f = m[row][col] * inv;
            m[row][col] = 0.0;
            for (int k = col + 1; k < 3; k++) {
                m[row][k] -= f * m[col][k];
            }
            rhs[row] -= f * rhs[col];
        }
    }

    double z = rhs[2] / m[2][2];
    double y = (rhs[1] - m[1][2] * z) / m[1][1];
    double xx = (rhs[0] - m[0][1] * y - m[0][2] * z) / m[0][0];
    *x = (pamo_vec3d){xx, y, z};
    return true;
}

static bool point_in_loose_edge_bounds(pamo_vec3d p,
                                       pamo_vec3d a,
                                       pamo_vec3d b) {
    double ex = fabs(a.x - b.x);
    double ey = fabs(a.y - b.y);
    double ez = fabs(a.z - b.z);
    double eps = 0.25 * sqrt(ex * ex + ey * ey + ez * ez) + 1.0e-9;
    double lo_x = a.x < b.x ? a.x : b.x;
    double lo_y = a.y < b.y ? a.y : b.y;
    double lo_z = a.z < b.z ? a.z : b.z;
    double hi_x = a.x > b.x ? a.x : b.x;
    double hi_y = a.y > b.y ? a.y : b.y;
    double hi_z = a.z > b.z ? a.z : b.z;
    return p.x >= lo_x - eps && p.x <= hi_x + eps &&
           p.y >= lo_y - eps && p.y <= hi_y + eps &&
           p.z >= lo_z - eps && p.z <= hi_z + eps;
}

static pamo_vec3d compute_collapse_point(const pamo_mesh *m,
                                         const pamo_quadric *Q,
                                         int32_t u, int32_t v) {
    pamo_vec3d pu = m->verts[u];
    pamo_vec3d pv = m->verts[v];
    pamo_vec3d mid = pamo_v3_scale(pamo_v3_add(pu, pv), 0.5);
    pamo_quadric q = pamo_quadric_add(Q[u], Q[v]);

    pamo_vec3d candidates[4];
    int n_candidates = 0;
    candidates[n_candidates++] = mid;
    candidates[n_candidates++] = pu;
    candidates[n_candidates++] = pv;

    double a[3][3] = {
        {q.m[0], q.m[1], q.m[2]},
        {q.m[1], q.m[4], q.m[5]},
        {q.m[2], q.m[5], q.m[7]},
    };
    double edge_len2 = pamo_v3_length_sq(pamo_v3_sub(pu, pv));
    double lambda = 1.0e-10;
    if (edge_len2 > 1.0) lambda *= edge_len2;
    a[0][0] += lambda;
    a[1][1] += lambda;
    a[2][2] += lambda;
    double rhs[3] = {
        -q.m[3] + lambda * mid.x,
        -q.m[6] + lambda * mid.y,
        -q.m[8] + lambda * mid.z,
    };
    pamo_vec3d opt;
    if (solve_symmetric_3x3(a, rhs, &opt) &&
        point_in_loose_edge_bounds(opt, pu, pv)) {
        candidates[n_candidates++] = opt;
    }

    pamo_vec3d best = candidates[0];
    double best_cost = pamo_quadric_eval(&q, best);
    for (int i = 1; i < n_candidates; i++) {
        double cost = pamo_quadric_eval(&q, candidates[i]);
        if (cost < best_cost) {
            best_cost = cost;
            best = candidates[i];
        }
    }
    return best;
}

/* Check if collapsing endpoint→mid flips any adjacent normal or
 * creates a skinny triangle. Returns true if INVALID. */
static bool collapse_creates_problem(const pamo_mesh *m,
                                     int32_t endpoint, int32_t other,
                                     pamo_vec3d point,
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
            verts[k] = (fv[k] == endpoint) ? point : m->verts[fv[k]];

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

static bool face_normal_after_collapse(const pamo_mesh *m,
                                       int32_t fi,
                                       int32_t u,
                                       int32_t v,
                                       pamo_vec3d point,
                                       pamo_vec3d *normal_out) {
    if (fi < 0 || (size_t)fi >= m->n_faces || !m->face_alive[fi])
        return false;
    const int32_t *fv = m->faces[fi].v;
    bool has_u = (fv[0] == u || fv[1] == u || fv[2] == u);
    bool has_v = (fv[0] == v || fv[1] == v || fv[2] == v);
    if (has_u && has_v) return false;

    pamo_vec3d verts[3];
    for (int k = 0; k < 3; k++) {
        verts[k] = (fv[k] == u || fv[k] == v) ? point : m->verts[fv[k]];
    }

    pamo_vec3d n = pamo_v3_cross(pamo_v3_sub(verts[1], verts[0]),
                                 pamo_v3_sub(verts[2], verts[0]));
    double len2 = pamo_v3_length_sq(n);
    if (len2 <= 1e-30) return false;
    *normal_out = pamo_v3_scale(n, 1.0 / sqrt(len2));
    return true;
}

static bool face_has_mapped_edge(const pamo_mesh *m,
                                 int32_t fi,
                                 int32_t u,
                                 int32_t v,
                                 int32_t a,
                                 int32_t b) {
    if (fi < 0 || (size_t)fi >= m->n_faces || !m->face_alive[fi])
        return false;
    int32_t mapped[3];
    const int32_t *fv = m->faces[fi].v;
    bool has_u = false;
    bool has_v = false;
    for (int k = 0; k < 3; k++) {
        if (fv[k] == u) has_u = true;
        if (fv[k] == v) has_v = true;
        mapped[k] = (fv[k] == v) ? u : fv[k];
    }
    if (has_u && has_v) return false;
    bool found_a = false;
    bool found_b = false;
    for (int k = 0; k < 3; k++) {
        if (mapped[k] == a) found_a = true;
        if (mapped[k] == b) found_b = true;
    }
    return found_a && found_b;
}

static bool collapse_creates_fold(const pamo_mesh *m,
                                  int32_t u,
                                  int32_t v,
                                  pamo_vec3d point,
                                  double min_adjacent_dot) {
    for (int pass = 0; pass < 2; pass++) {
        int32_t endpoint = pass == 0 ? u : v;
        int32_t s = m->vert_face_offset[endpoint];
        int32_t e = m->vert_face_offset[endpoint + 1];
        for (int32_t it = s; it < e; it++) {
            int32_t fi = m->vert_face_list[it];
            if (!m->face_alive[fi]) continue;
            const int32_t *fv = m->faces[fi].v;
            if ((fv[0] == u || fv[1] == u || fv[2] == u) &&
                (fv[0] == v || fv[1] == v || fv[2] == v)) {
                continue;
            }

            pamo_vec3d n0;
            if (!face_normal_after_collapse(m, fi, u, v, point, &n0))
                return true;

            int32_t mapped[3];
            for (int k = 0; k < 3; k++)
                mapped[k] = (fv[k] == v) ? u : fv[k];

            for (int edge = 0; edge < 3; edge++) {
                int32_t a = mapped[edge];
                int32_t b = mapped[(edge + 1) % 3];
                if (a == b) return true;

                int32_t scan = a;
                if (scan < 0 || (size_t)scan >= m->n_verts) return true;
                int32_t ns = m->vert_face_offset[scan];
                int32_t ne = m->vert_face_offset[scan + 1];
                for (int32_t jt = ns; jt < ne; jt++) {
                    int32_t gj = m->vert_face_list[jt];
                    if (gj == fi || !m->face_alive[gj]) continue;
                    if (!face_has_mapped_edge(m, gj, u, v, a, b)) continue;
                    pamo_vec3d n1;
                    if (!face_normal_after_collapse(m, gj, u, v, point, &n1))
                        continue;
                    if (pamo_v3_dot(n0, n1) < min_adjacent_dot)
                        return true;
                }
            }
        }
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
                                double fold_guard_min_dot,
                                const bool *vert_locked) {
    pamo_edge e = m->edges[edge_idx];
    int32_t u = e.u, v = e.v;

    if (!m->vert_alive[u] || !m->vert_alive[v]) return COST_INF;
    if (vert_locked && (vert_locked[u] || vert_locked[v])) return COST_INF;

    /* Link condition: ensures collapse preserves manifold topology. */
    if (collapse_violates_link(m, u, v)) return COST_INF;

    pamo_vec3d pu = m->verts[u];
    pamo_vec3d pv = m->verts[v];
    pamo_quadric Qsum = pamo_quadric_add(Q[u], Q[v]);
    pamo_vec3d point = compute_collapse_point(m, Q, u, v);

    /* Normal flip / skinny check for both endpoints. */
    if (collapse_creates_problem(m, u, v, point, skinny_thresh, flip_threshold))
        return COST_INF;
    if (collapse_creates_problem(m, v, u, point, skinny_thresh, flip_threshold))
        return COST_INF;
    if (fold_guard_min_dot > -1.0 &&
        collapse_creates_fold(m, u, v, point, fold_guard_min_dot))
        return COST_INF;

    /* Quadric error at the selected collapse point. */
    double qerr = pamo_quadric_eval(&Qsum, point);
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
                verts[k] = (fv[k] == endpoint) ? point : m->verts[fv[k]];
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

/* Match CUDA PaMO's batch selection more closely: an edge may collapse only
 * when it is the cheapest candidate for every incident triangle. */
static void update_tri_min_edges(const pamo_mesh *m,
                                 size_t edge_idx,
                                 double cost,
                                 double *tri_min_cost,
                                 int32_t *tri_min_edge) {
    const pamo_edge edge = m->edges[edge_idx];
    for (int pass = 0; pass < 2; pass++) {
        int32_t v = (pass == 0) ? edge.u : edge.v;
        int32_t s = m->vert_face_offset[v];
        int32_t e = m->vert_face_offset[v + 1];
        for (int32_t i = s; i < e; i++) {
            int32_t fi = m->vert_face_list[i];
            if (!m->face_alive[fi]) continue;
            if (cost < tri_min_cost[fi] ||
                (cost == tri_min_cost[fi] &&
                 (tri_min_edge[fi] < 0 ||
                  (int32_t)edge_idx < tri_min_edge[fi]))) {
                tri_min_cost[fi] = cost;
                tri_min_edge[fi] = (int32_t)edge_idx;
            }
        }
    }
}

static bool edge_is_local_triangle_min(const pamo_mesh *m,
                                       size_t edge_idx,
                                       const int32_t *tri_min_edge) {
    const pamo_edge edge = m->edges[edge_idx];
    for (int pass = 0; pass < 2; pass++) {
        int32_t v = (pass == 0) ? edge.u : edge.v;
        int32_t s = m->vert_face_offset[v];
        int32_t e = m->vert_face_offset[v + 1];
        for (int32_t i = s; i < e; i++) {
            int32_t fi = m->vert_face_list[i];
            if (!m->face_alive[fi]) continue;
            if (tri_min_edge[fi] != (int32_t)edge_idx) return false;
        }
    }
    return true;
}

static pamo_aabb face_bounds_expanded(const pamo_mesh *m,
                                      int32_t fi,
                                      double eps) {
    pamo_aabb box = {
        .lo = { 1e30,  1e30,  1e30},
        .hi = {-1e30, -1e30, -1e30},
    };
    if (!pamo_mesh_face_is_valid(m, (size_t)fi)) return box;
    const int32_t *fv = m->faces[fi].v;
    for (int k = 0; k < 3; k++) {
        pamo_vec3d p = m->verts[fv[k]];
        if (p.x < box.lo.x) box.lo.x = p.x;
        if (p.y < box.lo.y) box.lo.y = p.y;
        if (p.z < box.lo.z) box.lo.z = p.z;
        if (p.x > box.hi.x) box.hi.x = p.x;
        if (p.y > box.hi.y) box.hi.y = p.y;
        if (p.z > box.hi.z) box.hi.z = p.z;
    }
    box.lo.x -= eps; box.lo.y -= eps; box.lo.z -= eps;
    box.hi.x += eps; box.hi.y += eps; box.hi.z += eps;
    return box;
}

static bool faces_share_vertex(const pamo_mesh *m, int32_t a, int32_t b) {
    if (!pamo_mesh_face_is_valid(m, (size_t)a) ||
        !pamo_mesh_face_is_valid(m, (size_t)b)) {
        return false;
    }
    for (int i = 0; i < 3; i++) {
        int32_t av = m->faces[a].v[i];
        for (int j = 0; j < 3; j++) {
            if (av == m->faces[b].v[j]) return true;
        }
    }
    return false;
}

static pamo_error mark_self_intersections(const pamo_mesh *m,
                                          bool *hit_faces,
                                          size_t *count_out) {
    if (!m || !count_out) return PAMO_ERR_INVALID_ARG;
    *count_out = 0;
    if (hit_faces) memset(hit_faces, 0, m->n_faces * sizeof(bool));
    if (pamo_mesh_count_alive_faces(m) == 0) return PAMO_OK;

    pamo_allocator alloc = m->alloc;
    pamo_bvh bvh;
    pamo_error err = pamo_bvh_build_triangles(&bvh, m, &alloc);
    if (err != PAMO_OK) return err;
    if (bvh.n_nodes == 0) {
        pamo_bvh_destroy(&bvh);
        return PAMO_OK;
    }

    pamo_aabb bounds = pamo_mesh_bounds(m);
    double dx = bounds.hi.x - bounds.lo.x;
    double dy = bounds.hi.y - bounds.lo.y;
    double dz = bounds.hi.z - bounds.lo.z;
    double eps = sqrt(dx * dx + dy * dy + dz * dz) * 1.0e-9 + 1.0e-12;

    pamo_overlap_result ov;
    pamo_overlap_result_init(&ov, &alloc);
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!pamo_mesh_face_is_valid(m, fi)) continue;
        pamo_aabb box = face_bounds_expanded(m, (int32_t)fi, eps);
        err = pamo_bvh_overlap(&bvh, box, &ov);
        if (err != PAMO_OK) break;
        for (size_t hit_i = 0; hit_i < ov.n_hits; hit_i++) {
            int32_t fj = ov.hits[hit_i];
            if (fj <= (int32_t)fi) continue;
            if (!pamo_mesh_face_is_valid(m, (size_t)fj)) continue;
            if (faces_share_vertex(m, (int32_t)fi, fj)) continue;
            pamo_vec3d a0 = m->verts[m->faces[fi].v[0]];
            pamo_vec3d a1 = m->verts[m->faces[fi].v[1]];
            pamo_vec3d a2 = m->verts[m->faces[fi].v[2]];
            pamo_vec3d b0 = m->verts[m->faces[fj].v[0]];
            pamo_vec3d b1 = m->verts[m->faces[fj].v[1]];
            pamo_vec3d b2 = m->verts[m->faces[fj].v[2]];
            if (pamo_tri_tri_intersect(a0, a1, a2, b0, b1, b2)) {
                (*count_out)++;
                if (hit_faces) {
                    hit_faces[fi] = true;
                    hit_faces[fj] = true;
                }
            }
        }
    }

    pamo_overlap_result_destroy(&ov);
    pamo_bvh_destroy(&bvh);
    return err;
}

/* ── Collapse record for undo ────────────────────────────────────── */

typedef struct {
    int32_t    u, v;
    bool       applied;
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

static bool record_touches_hit_face(const pamo_mesh *m,
                                    const pamo_collapse_record *record,
                                    const bool *hit_faces) {
    for (int pass = 0; pass < 2; pass++) {
        int32_t v = (pass == 0) ? record->u : record->v;
        int32_t s = m->vert_face_offset[v];
        int32_t e = m->vert_face_offset[v + 1];
        for (int32_t i = s; i < e; i++) {
            int32_t fi = m->vert_face_list[i];
            if (fi >= 0 && (size_t)fi < m->n_faces && hit_faces[fi])
                return true;
        }
    }
    return false;
}

static void restore_record_from_snapshot(pamo_mesh *m,
                                         const pamo_collapse_record *record,
                                         const pamo_vec3d *orig_verts,
                                         const bool *orig_vert_alive,
                                         const pamo_tri *orig_faces,
                                         const bool *orig_face_alive) {
    const int32_t endpoints[2] = {record->u, record->v};
    for (int pass = 0; pass < 2; pass++) {
        int32_t v = endpoints[pass];
        if (v < 0 || (size_t)v >= m->n_verts) continue;
        m->verts[v] = orig_verts[v];
        m->vert_alive[v] = orig_vert_alive[v];
        int32_t s = m->vert_face_offset[v];
        int32_t e = m->vert_face_offset[v + 1];
        for (int32_t i = s; i < e; i++) {
            int32_t fi = m->vert_face_list[i];
            if (fi < 0 || (size_t)fi >= m->n_faces) continue;
            m->faces[fi] = orig_faces[fi];
            m->face_alive[fi] = orig_face_alive[fi];
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
    double *tri_min_cost = NULL;
    int32_t *tri_min_edge = NULL;
    pamo_vec3d *orig_verts = NULL;
    bool *orig_vert_alive = NULL;
    pamo_tri *orig_faces = NULL;
    bool *orig_face_alive = NULL;
    bool *locked = NULL;
    pamo_collapse_record *records = NULL;
    size_t records_cap = 1;
    size_t self_intersections_before = 0;

    tri_min_cost = PAMO_ALLOC_ARRAY(alloc, double, m->n_faces);
    tri_min_edge = PAMO_ALLOC_ARRAY(alloc, int32_t, m->n_faces);
    if ((m->n_faces > 0) && (!tri_min_cost || !tri_min_edge)) {
        err = PAMO_ERR_ALLOC;
        goto cleanup;
    }
    for (size_t i = 0; i < m->n_faces; i++) {
        tri_min_cost[i] = COST_INF;
        tri_min_edge[i] = -1;
    }

    if (opts->check_self_intersection) {
        orig_verts = PAMO_ALLOC_ARRAY(alloc, pamo_vec3d, m->n_verts);
        orig_vert_alive = PAMO_ALLOC_ARRAY(alloc, bool, m->n_verts);
        orig_faces = PAMO_ALLOC_ARRAY(alloc, pamo_tri, m->n_faces);
        orig_face_alive = PAMO_ALLOC_ARRAY(alloc, bool, m->n_faces);
        if ((m->n_verts > 0 && (!orig_verts || !orig_vert_alive)) ||
            (m->n_faces > 0 && (!orig_faces || !orig_face_alive))) {
            err = PAMO_ERR_ALLOC;
            goto cleanup;
        }
        if (m->n_verts > 0) {
            memcpy(orig_verts, m->verts, m->n_verts * sizeof(pamo_vec3d));
            memcpy(orig_vert_alive, m->vert_alive, m->n_verts * sizeof(bool));
        }
        if (m->n_faces > 0) {
            memcpy(orig_faces, m->faces, m->n_faces * sizeof(pamo_tri));
            memcpy(orig_face_alive, m->face_alive, m->n_faces * sizeof(bool));
        }
        err = mark_self_intersections(m, NULL, &self_intersections_before);
        if (err != PAMO_OK) goto cleanup;
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
                                     opts->fold_guard_min_dot,
                                     vert_locked);
        if (c < COST_INF) {
            update_tri_min_edges(m, i, c, tri_min_cost, tri_min_edge);
        }
        if (c < cost_limit) {
            costs[n_valid].cost = c;
            costs[n_valid].edge_idx = (int32_t)i;
            n_valid++;
        }
    }

    pamo_sort_edge_costs(costs, n_valid);

    /* Greedy selection. */
    records_cap = n_valid > 0 ? n_valid : 1;

    locked = PAMO_ALLOC_ARRAY(alloc, bool, m->n_verts);
    if (!locked) {
        err = PAMO_ERR_ALLOC;
        goto cleanup;
    }
    memset(locked, 0, m->n_verts * sizeof(bool));

    records = PAMO_ALLOC_ARRAY(alloc, pamo_collapse_record, records_cap);
    if (!records) {
        err = PAMO_ERR_ALLOC;
        goto cleanup;
    }

    size_t n_accepted = 0;
    for (size_t i = 0; i < n_valid; i++) {
        int32_t ei = costs[i].edge_idx;
        int32_t u = m->edges[ei].u;
        int32_t v = m->edges[ei].v;
        if (locked[u] || locked[v]) continue;
        if (!edge_is_local_triangle_min(m, (size_t)ei, tri_min_edge))
            continue;

        /* CUDA PaMO uses the per-triangle local-min rule to prevent
         * conflicting collapse batches. Keep only endpoint locks here so
         * an edge cannot be selected twice in the same CPU batch. */
        locked[u] = true;
        locked[v] = true;

        records[n_accepted].u = u;
        records[n_accepted].v = v;
        records[n_accepted].applied = false;
        records[n_accepted].undone = false;
        n_accepted++;
    }

    /* Apply collapses with runtime validation.
     * Re-check the link condition before each collapse since
     * earlier collapses in this round may have changed topology. */
    size_t actually_collapsed = 0;
    /* Sub-iter progress emit cadence — every PROGRESS_CADENCE iterations
     * of this loop. 256 keeps emit overhead negligible while making the
     * Blender bar tick visibly during multi-second outer rounds.        */
    const size_t PROGRESS_CADENCE = 256;
    pamo_emit_progress("simplify_collapse", 0, (int64_t)n_accepted);
    for (size_t i = 0; i < n_accepted; i++) {
        if ((i % PROGRESS_CADENCE) == 0 && i > 0) {
            pamo_emit_progress("simplify_collapse", (int64_t)i,
                               (int64_t)n_accepted);
        }
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

        pamo_vec3d point = compute_collapse_point(m, Q, u, v);
        if (collapse_creates_problem(m, u, v, point,
                                     opts->skinny_area_threshold, 0.0) ||
            collapse_creates_problem(m, v, u, point,
                                     opts->skinny_area_threshold, 0.0) ||
            (opts->fold_guard_min_dot > -1.0 &&
             collapse_creates_fold(m, u, v, point,
                                   opts->fold_guard_min_dot))) {
            continue;
        }
        apply_collapse(m, u, v, point);
        records[i].applied = true;
        actually_collapsed++;
    }

    if (opts->check_self_intersection && actually_collapsed > 0) {
        bool *hit_faces = PAMO_ALLOC_ARRAY(alloc, bool, m->n_faces);
        if (!hit_faces && m->n_faces > 0) {
            err = PAMO_ERR_ALLOC;
            goto cleanup;
        }

        size_t self_intersections_after = 0;
        err = mark_self_intersections(m, hit_faces, &self_intersections_after);
        if (err != PAMO_OK) {
            if (hit_faces) PAMO_FREE_ARRAY(alloc, hit_faces, bool, m->n_faces);
            goto cleanup;
        }

        int32_t undo_retry = 0;
        while (self_intersections_after > self_intersections_before &&
               undo_retry < opts->max_undo_retries) {
            size_t undone_this_pass = 0;
            for (size_t i = 0; i < n_accepted; i++) {
                if (!records[i].applied || records[i].undone) continue;
                if (!record_touches_hit_face(m, &records[i], hit_faces))
                    continue;
                restore_record_from_snapshot(m, &records[i],
                                             orig_verts, orig_vert_alive,
                                             orig_faces, orig_face_alive);
                records[i].undone = true;
                undone_this_pass++;
            }

            if (undone_this_pass == 0) break;
            if (undone_this_pass >= actually_collapsed) actually_collapsed = 0;
            else actually_collapsed -= undone_this_pass;
            undo_retry++;

            err = mark_self_intersections(m, hit_faces,
                                          &self_intersections_after);
            if (err != PAMO_OK) {
                PAMO_FREE_ARRAY(alloc, hit_faces, bool, m->n_faces);
                goto cleanup;
            }
        }

        if (self_intersections_after > self_intersections_before) {
            for (size_t i = 0; i < n_accepted; i++) {
                if (!records[i].applied || records[i].undone) continue;
                restore_record_from_snapshot(m, &records[i],
                                             orig_verts, orig_vert_alive,
                                             orig_faces, orig_face_alive);
                records[i].undone = true;
            }
            actually_collapsed = 0;
        }

        if (hit_faces) PAMO_FREE_ARRAY(alloc, hit_faces, bool, m->n_faces);
    }

    *n_collapsed_out = actually_collapsed;

cleanup:
    if (orig_face_alive)
        PAMO_FREE_ARRAY(alloc, orig_face_alive, bool, m->n_faces);
    if (orig_faces) PAMO_FREE_ARRAY(alloc, orig_faces, pamo_tri, m->n_faces);
    if (orig_vert_alive)
        PAMO_FREE_ARRAY(alloc, orig_vert_alive, bool, m->n_verts);
    if (orig_verts) PAMO_FREE_ARRAY(alloc, orig_verts, pamo_vec3d, m->n_verts);
    if (tri_min_edge) PAMO_FREE_ARRAY(alloc, tri_min_edge, int32_t, m->n_faces);
    if (tri_min_cost) PAMO_FREE_ARRAY(alloc, tri_min_cost, double, m->n_faces);
    if (records)
        PAMO_FREE_ARRAY(alloc, records, pamo_collapse_record, records_cap);
    if (locked) PAMO_FREE_ARRAY(alloc, locked, bool, m->n_verts);
    PAMO_FREE_ARRAY(alloc, costs, pamo_edge_cost_entry, m->n_edges);
    PAMO_FREE_ARRAY(alloc, Q, pamo_quadric, m->n_verts);
    pamo_mesh_free_adjacency(m);
    return err;
}

/* ── Main simplification driver ──────────────────────────────────── */

pamo_error pamo_simplify(pamo_mesh *mesh, const pamo_simplify_opts *opts) {
    if (!mesh || !opts) return PAMO_ERR_INVALID_ARG;

    pamo_error err = pamo_mesh_compact(mesh);
    if (err != PAMO_OK) return err;

    int32_t stuck_counter = 0;
    double cost_limit = opts->cost_range;
    pamo_emit_progress("simplify_start", 0,
                       (int64_t)pamo_mesh_count_alive_faces(mesh));

    for (int32_t iter = 0; iter < opts->max_iters; iter++) {
        size_t alive_faces = pamo_mesh_count_alive_faces(mesh);
        /* Report progress as faces collapsed against (initial - target).
         * `iter` flows in the `current` slot too — useful when the bar
         * needs to advance while we're still far from the target. */
        pamo_emit_progress("simplify_iter",
                           (int64_t)alive_faces,
                           (int64_t)opts->target_faces);
        if ((int32_t)alive_faces <= opts->target_faces) break;
        if ((int32_t)alive_faces <= opts->min_faces) break;

        size_t n_collapsed = 0;
        bool relax = (stuck_counter >= 2); /* relax normals when deeply stuck */
        err = simplify_round(mesh, opts, NULL, cost_limit, relax, &n_collapsed);
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
