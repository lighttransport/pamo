/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mesh-to-GT distance energy: for each vertex of the current mesh,
 * find closest point on GT mesh (via BVH) and penalize squared distance.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"

#include <math.h>

/* Compute mesh-to-GT distance energy.
 * E = stiffness * sum_i ||q_i - target_i||^2
 * grad_i = 2 * stiffness * (q_i - target_i)
 * hess_diag_i = 2 * stiffness
 */
double pamo_distance_energy(const pamo_vec3d *q, size_t n_verts,
                            const pamo_vec3d *targets,
                            double stiffness) {
    double E = 0.0;
    for (size_t i = 0; i < n_verts; i++) {
        pamo_vec3d d = pamo_v3_sub(q[i], targets[i]);
        E += stiffness * pamo_v3_length_sq(d);
    }
    return E;
}

void pamo_distance_gradient(const pamo_vec3d *q, size_t n_verts,
                            const pamo_vec3d *targets,
                            double stiffness,
                            double *grad, double *hess_diag) {
    for (size_t i = 0; i < n_verts; i++) {
        pamo_vec3d d = pamo_v3_sub(q[i], targets[i]);
        double s2 = 2.0 * stiffness;
        grad[i * 3 + 0] += s2 * d.x;
        grad[i * 3 + 1] += s2 * d.y;
        grad[i * 3 + 2] += s2 * d.z;
        hess_diag[i * 3 + 0] += s2;
        hess_diag[i * 3 + 1] += s2;
        hess_diag[i * 3 + 2] += s2;
    }
}

void pamo_distance_hess_vec(const pamo_vec3d *q, size_t n_verts,
                            const double *dx, double *out,
                            double stiffness) {
    double s2 = 2.0 * stiffness;
    for (size_t i = 0; i < n_verts; i++) {
        out[i * 3 + 0] += s2 * dx[i * 3 + 0];
        out[i * 3 + 1] += s2 * dx[i * 3 + 1];
        out[i * 3 + 2] += s2 * dx[i * 3 + 2];
    }
    (void)q;
}

/* Update targets: find closest point on GT mesh for each vertex. */
pamo_error pamo_distance_update_targets(const pamo_vec3d *q, size_t n_verts,
                                        const pamo_mesh *gt_mesh,
                                        const pamo_bvh *gt_bvh,
                                        pamo_vec3d *targets) {
    for (size_t i = 0; i < n_verts; i++) {
        pamo_nearest_result res;
        pamo_error err = pamo_bvh_nearest(gt_bvh, gt_mesh, q[i], &res);
        if (err != PAMO_OK) return err;
        targets[i] = res.point;
    }
    return PAMO_OK;
}
