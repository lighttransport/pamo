/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Collision barrier energy (stub for initial integration).
 * Full implementation requires point-triangle and edge-edge
 * contact detection, barrier functions, and CCD.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"

#include <math.h>

/* Barrier function: b(d, d_hat) = -(d - d_hat)^2 * ln(d / d_hat) */
double pamo_barrier(double d, double d_hat) {
    if (d >= d_hat) return 0.0;
    if (d <= 0.0) return 1e20;
    double t = d - d_hat;
    return -t * t * log(d / d_hat);
}

/* Barrier derivative: db/dd */
double pamo_barrier_grad(double d, double d_hat) {
    if (d >= d_hat) return 0.0;
    if (d <= 0.0) return -1e20;
    double t = d - d_hat;
    return -2.0 * t * log(d / d_hat) - t * t / d;
}

/* Collision energy is a stub for now -- full contact detection
 * (BVH-based PT + EE) will be implemented later. */
double pamo_collision_energy(const pamo_mesh *m, const pamo_vec3d *q,
                             double d_hat, double stiffness) {
    (void)m; (void)q; (void)d_hat; (void)stiffness;
    return 0.0;
}
