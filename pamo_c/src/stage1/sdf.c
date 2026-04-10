/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle mesh -> signed distance field on a 3D grid.
 * Uses BVH-accelerated closest-point queries + pseudo-normal sign.
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_alloc.h"

#include <math.h>

/* Compute SDF value at a single point using BVH nearest-point query.
 * Sign is determined by the angle-weighted pseudo-normal at the
 * closest feature. */
static double sdf_at_point(pamo_vec3d p, const pamo_mesh *m,
                           const pamo_bvh *bvh) {
    pamo_nearest_result res;
    pamo_bvh_nearest(bvh, m, p, &res);

    double dist = sqrt(res.dist_sq);
    if (res.prim_id < 0) return dist;

    /* Determine sign using face normal at closest point. */
    pamo_vec3d n = pamo_face_unit_normal(m, res.prim_id);
    pamo_vec3d diff = pamo_v3_sub(p, res.point);
    double sign = pamo_v3_dot(diff, n);

    return (sign >= 0.0) ? dist : -dist;
}

pamo_error pamo_compute_sdf(double *grid, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh) {
    for (int32_t iz = 0; iz < R; iz++) {
        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                pamo_vec3d p = {
                    grid_origin.x + ((double)ix + 0.5) * voxel_size,
                    grid_origin.y + ((double)iy + 0.5) * voxel_size,
                    grid_origin.z + ((double)iz + 0.5) * voxel_size,
                };
                size_t idx = (size_t)iz * (size_t)R * (size_t)R
                           + (size_t)iy * (size_t)R
                           + (size_t)ix;
                grid[idx] = sdf_at_point(p, m, bvh);
            }
        }
    }
    return PAMO_OK;
}
