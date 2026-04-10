/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_STAGE3_H
#define PAMO_STAGE3_H

#include "pamo_types.h"
#include "pamo_alloc.h"
#include "pamo_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t n_outer_iters;          /* default 5 */
    int32_t n_newton_iters;         /* default 10 */
    int32_t n_cg_iters;             /* default 40 */
    int32_t n_line_search_iters;    /* default 10 */

    double  d_hat;                  /* collision barrier threshold (default 1e-3) */
    double  coll_stiffness;         /* collision stiffness (default 1e2) */
    double  elas_young_modulus;     /* Young's modulus (default 1e-1) */
    double  elas_poisson_ratio;     /* Poisson's ratio (default 0.0) */
    double  hinge_stiffness;        /* bending stiffness (default 1e-2) */
    double  dist_stiffness;         /* mesh-to-GT distance weight (default 1e3) */

    double  ccd_slackness;          /* CCD safety margin (default 0.7) */
    double  ccd_thickness;          /* CCD thickness (default 1e-6) */
    int32_t ccd_max_iters;          /* CCD max iterations (default 100) */
    int32_t spd_max_iters;          /* Jacobi SPD projection iterations (default 3) */

    bool    enable_collision;       /* enable collision barrier (default true) */
    bool    enable_hinge;           /* enable hinge bending (default true) */
} pamo_safe_opts;

pamo_safe_opts pamo_safe_opts_default(void);

/* Stage 3: SAFE projection.
 * Projects the simplified mesh back toward the ground-truth mesh
 * using optimization with elastic, bending, distance, and collision
 * energies.  Modifies mesh->verts in place. */
pamo_error pamo_safe_project(pamo_mesh *mesh, const pamo_mesh *gt_mesh,
                             const pamo_safe_opts *opts,
                             const pamo_allocator *alloc);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_STAGE3_H */
