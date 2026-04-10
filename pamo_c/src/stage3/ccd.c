/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Continuous collision detection (CCD) for step size limiting.
 * Stub implementation -- returns full step (alpha = 1.0).
 * Full CCD requires detecting the first time of contact along
 * the trajectory q -> q + alpha * p.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"

double pamo_ccd_step(const pamo_mesh *m, const pamo_vec3d *q,
                     const double *p, double slackness, double thickness,
                     int max_iters) {
    (void)m; (void)q; (void)p; (void)thickness; (void)max_iters;
    return slackness;  /* conservative: return slackness factor */
}
