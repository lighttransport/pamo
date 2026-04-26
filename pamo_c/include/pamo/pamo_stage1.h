/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_STAGE1_H
#define PAMO_STAGE1_H

#include "pamo_types.h"
#include "pamo_alloc.h"
#include "pamo_mesh.h"
#include "pamo_bvh.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute a signed distance field (double, R^3) of `m` on a uniform
 * cubic grid (origin = `grid_origin`, voxel size = `voxel_size`,
 * resolution = R). `bvh` must be a triangle-BVH built over `m` (use
 * pamo_bvh_build_triangles); pass NULL only if `m` is small enough
 * that brute-force suffices.
 *
 * Used internally by pamo_remesh; exposed here so external callers
 * (e.g. lgphys's offline SDF builder) can reuse the lightrt-accelerated
 * path without pulling pamo_remesh's marching-cubes machinery. */
pamo_error pamo_compute_sdf(double *grid_out, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh);

typedef struct {
    int32_t resolution;   /* SDF grid resolution (default 256) */
    double  band;         /* SDF band width (default 3.0/resolution) */
} pamo_remesh_opts;

pamo_remesh_opts pamo_remesh_opts_default(void);

/* Stage 1: Volumetric remeshing.
 * Rasterizes input mesh into an SDF, then extracts a watertight
 * mesh via dual marching cubes. */
pamo_error pamo_remesh(pamo_mesh *out, const pamo_mesh *in,
                       const pamo_remesh_opts *opts,
                       const pamo_allocator *alloc);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_STAGE1_H */
