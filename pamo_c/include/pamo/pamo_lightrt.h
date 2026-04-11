/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C bridge to lightrt TriangleBVH for accelerated ray traversal.
 * Only available when PAMO_USE_LIGHTRT is defined.
 */
#ifndef PAMO_LIGHTRT_H
#define PAMO_LIGHTRT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lightrt_bvh lightrt_bvh;

/* Build a TriangleBVH from mesh data (float32 vertices). */
lightrt_bvh *lightrt_bvh_build(const float *verts, int32_t n_verts,
                                const int32_t *faces, int32_t n_faces);

/* Destroy. */
void lightrt_bvh_destroy(lightrt_bvh *bvh);

/* Closest-hit ray traversal.  Returns tri index or -1. */
int32_t lightrt_bvh_traverse(const lightrt_bvh *bvh,
                              const float *ray_origin,
                              const float *ray_dir,
                              float tmax,
                              float *hit_t);

/* Any-hit traversal.  Returns 1 if any hit, 0 otherwise. */
int lightrt_bvh_any_hit(const lightrt_bvh *bvh,
                         const float *ray_origin,
                         const float *ray_dir,
                         float tmax);

/* Count all intersections along a ray (for parity sign). */
int32_t lightrt_bvh_count_hits(const lightrt_bvh *bvh,
                                const float *ray_origin,
                                const float *ray_dir,
                                float tmax);

/* Get sorted hit t-values along a ray.  Returns count written. */
int32_t lightrt_bvh_multi_hit(const lightrt_bvh *bvh,
                               const float *ray_origin,
                               const float *ray_dir,
                               float tmax,
                               float *out_t, int32_t max_hits);

/* Nearest distance query.  Returns squared distance to closest triangle. */
float lightrt_bvh_nearest_dist_sq(const lightrt_bvh *bvh,
                                   const float *point);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_LIGHTRT_H */
