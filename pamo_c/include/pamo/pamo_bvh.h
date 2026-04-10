/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_BVH_H
#define PAMO_BVH_H

#include "pamo_types.h"
#include "pamo_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── BVH node ────────────────────────────────────────────────────── */

typedef struct {
    pamo_aabb box;
    int32_t   left;      /* child index, or -1 if leaf */
    int32_t   right;     /* child index, or -1 if leaf */
    int32_t   prim_id;   /* primitive index (valid only if leaf) */
} pamo_bvh_node;

/* ── BVH structure ───────────────────────────────────────────────── */

typedef struct {
    pamo_bvh_node *nodes;
    size_t         n_nodes;
    size_t         n_prims;    /* number of leaf primitives */
    pamo_allocator alloc;
} pamo_bvh;

/* Forward declaration */
struct pamo_mesh;

/* Build a BVH over alive triangles in the mesh. */
pamo_error pamo_bvh_build_triangles(pamo_bvh *bvh,
                                    const struct pamo_mesh *m,
                                    const pamo_allocator *alloc);

/* Free BVH memory. */
void pamo_bvh_destroy(pamo_bvh *bvh);

/* ── Nearest-point query ─────────────────────────────────────────── */

typedef struct {
    pamo_vec3d point;     /* closest point on the primitive */
    double     dist_sq;   /* squared distance */
    int32_t    prim_id;   /* face index in the mesh */
} pamo_nearest_result;

/* Find the nearest point on the mesh to a query point. */
pamo_error pamo_bvh_nearest(const pamo_bvh *bvh,
                            const struct pamo_mesh *m,
                            pamo_vec3d query,
                            pamo_nearest_result *result);

/* ── Overlap query ───────────────────────────────────────────────── */

typedef struct {
    int32_t *hits;        /* array of primitive IDs */
    size_t   n_hits;
    size_t   n_hits_cap;
    pamo_allocator alloc;
} pamo_overlap_result;

/* Initialize an overlap result buffer. */
void pamo_overlap_result_init(pamo_overlap_result *r,
                              const pamo_allocator *alloc);

/* Free an overlap result buffer. */
void pamo_overlap_result_destroy(pamo_overlap_result *r);

/* Find all primitives whose AABB overlaps the query box. */
pamo_error pamo_bvh_overlap(const pamo_bvh *bvh,
                            pamo_aabb query_box,
                            pamo_overlap_result *result);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_BVH_H */
