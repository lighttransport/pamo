/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Continuous collision detection (CCD) for step size limiting.
 *
 * For each vertex moving along direction p, find the first time t
 * where the vertex hits a non-adjacent triangle. Return alpha = t * slackness.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"

#include <math.h>

/* Simple CCD: for each vertex, check if the move q -> q + alpha*p
 * would bring it within thickness of any non-adjacent triangle.
 * Binary search for the safe step size. */
double pamo_ccd_step(const pamo_mesh *m, const pamo_vec3d *q,
                     const double *p, double slackness, double thickness,
                     int max_iters) {
    /* Build BVH for current mesh. */
    pamo_allocator alloc = pamo_default_allocator();
    pamo_bvh bvh;
    if (pamo_bvh_build_triangles(&bvh, m, &alloc) != PAMO_OK)
        return slackness;

    double alpha = 1.0;

    /* For each vertex, check the endpoint position. */
    for (size_t vi = 0; vi < m->n_verts; vi++) {
        if (!m->vert_alive[vi]) continue;

        /* Endpoint after full step. */
        pamo_vec3d endpoint = {
            q[vi].x + alpha * p[vi * 3 + 0],
            q[vi].y + alpha * p[vi * 3 + 1],
            q[vi].z + alpha * p[vi * 3 + 2],
        };

        pamo_nearest_result res;
        pamo_bvh_nearest(&bvh, m, endpoint, &res);
        if (res.prim_id < 0) continue;

        /* Skip self. */
        const int32_t *fv = m->faces[res.prim_id].v;
        if (fv[0] == (int32_t)vi || fv[1] == (int32_t)vi ||
            fv[2] == (int32_t)vi) continue;

        double dist = sqrt(res.dist_sq);
        if (dist < thickness) {
            /* Binary search for safe alpha. */
            double lo = 0.0, hi = alpha;
            for (int iter = 0; iter < max_iters && hi - lo > 1e-8; iter++) {
                double mid_alpha = (lo + hi) * 0.5;
                pamo_vec3d mid_pos = {
                    q[vi].x + mid_alpha * p[vi * 3 + 0],
                    q[vi].y + mid_alpha * p[vi * 3 + 1],
                    q[vi].z + mid_alpha * p[vi * 3 + 2],
                };
                pamo_nearest_result mid_res;
                pamo_bvh_nearest(&bvh, m, mid_pos, &mid_res);
                double mid_dist = sqrt(mid_res.dist_sq);
                if (mid_dist < thickness) {
                    hi = mid_alpha;
                } else {
                    lo = mid_alpha;
                }
            }
            if (lo < alpha) alpha = lo;
        }
    }

    pamo_bvh_destroy(&bvh);
    return alpha * slackness;
}
