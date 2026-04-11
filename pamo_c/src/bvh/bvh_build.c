/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Top-down median-split BVH for triangles.
 */
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_mesh.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── AABB helpers ────────────────────────────────────────────────── */

static pamo_aabb aabb_empty(void) {
    return (pamo_aabb){
        .lo = { 1e30,  1e30,  1e30},
        .hi = {-1e30, -1e30, -1e30},
    };
}

static pamo_aabb aabb_union_point(pamo_aabb b, pamo_vec3d p) {
    if (p.x < b.lo.x) b.lo.x = p.x;
    if (p.y < b.lo.y) b.lo.y = p.y;
    if (p.z < b.lo.z) b.lo.z = p.z;
    if (p.x > b.hi.x) b.hi.x = p.x;
    if (p.y > b.hi.y) b.hi.y = p.y;
    if (p.z > b.hi.z) b.hi.z = p.z;
    return b;
}

static pamo_aabb aabb_union(pamo_aabb a, pamo_aabb b) {
    a = aabb_union_point(a, b.lo);
    a = aabb_union_point(a, b.hi);
    return a;
}

static double aabb_min_dist_sq(pamo_aabb b, pamo_vec3d p) {
    double dx = (p.x < b.lo.x) ? (b.lo.x - p.x) :
                (p.x > b.hi.x) ? (p.x - b.hi.x) : 0.0;
    double dy = (p.y < b.lo.y) ? (b.lo.y - p.y) :
                (p.y > b.hi.y) ? (p.y - b.hi.y) : 0.0;
    double dz = (p.z < b.lo.z) ? (b.lo.z - p.z) :
                (p.z > b.hi.z) ? (p.z - b.hi.z) : 0.0;
    return dx * dx + dy * dy + dz * dz;
}

static bool aabb_overlaps(pamo_aabb a, pamo_aabb b) {
    return a.lo.x <= b.hi.x && a.hi.x >= b.lo.x &&
           a.lo.y <= b.hi.y && a.hi.y >= b.lo.y &&
           a.lo.z <= b.hi.z && a.hi.z >= b.lo.z;
}

/* ── Build data ──────────────────────────────────────────────────── */

typedef struct {
    pamo_aabb    box;
    pamo_vec3d   centroid;
    int32_t      face_id;
} pamo_bvh_prim;

/* Partition primitives [lo..hi) around median on given axis. */
static int prim_cmp_x(const void *a, const void *b) {
    double ca = ((const pamo_bvh_prim *)a)->centroid.x;
    double cb = ((const pamo_bvh_prim *)b)->centroid.x;
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}
static int prim_cmp_y(const void *a, const void *b) {
    double ca = ((const pamo_bvh_prim *)a)->centroid.y;
    double cb = ((const pamo_bvh_prim *)b)->centroid.y;
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}
static int prim_cmp_z(const void *a, const void *b) {
    double ca = ((const pamo_bvh_prim *)a)->centroid.z;
    double cb = ((const pamo_bvh_prim *)b)->centroid.z;
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}

typedef int (*cmp_fn)(const void *, const void *);
static const cmp_fn axis_cmp[3] = { prim_cmp_x, prim_cmp_y, prim_cmp_z };

#define PAMO_BVH_MAX_DEPTH 64

/* Recursive build.  Returns node index in bvh->nodes. */
static int32_t build_recursive(pamo_bvh *bvh,
                               pamo_bvh_prim *prims,
                               size_t lo, size_t hi,
                               size_t *next_node, int depth) {
    PAMO_ASSERT(lo < hi);
    int32_t node_idx = (int32_t)(*next_node);
    (*next_node)++;

    pamo_bvh_node *node = &bvh->nodes[node_idx];

    /* Compute bounds. */
    pamo_aabb bounds = aabb_empty();
    for (size_t i = lo; i < hi; i++) {
        bounds = aabb_union(bounds, prims[i].box);
    }
    node->box = bounds;

    if (hi - lo == 1 || depth >= PAMO_BVH_MAX_DEPTH) {
        /* Leaf (or depth limit reached -- store first prim). */
        node->left    = -1;
        node->right   = -1;
        node->prim_id = prims[lo].face_id;
        return node_idx;
    }

    /* Choose split axis: longest extent. */
    double ex = bounds.hi.x - bounds.lo.x;
    double ey = bounds.hi.y - bounds.lo.y;
    double ez = bounds.hi.z - bounds.lo.z;
    int axis = 0;
    if (ey > ex && ey >= ez) axis = 1;
    else if (ez > ex && ez > ey) axis = 2;

    /* Sort on axis and split at median. */
    size_t n = hi - lo;
    qsort(prims + lo, n, sizeof(pamo_bvh_prim), axis_cmp[axis]);
    size_t mid = lo + n / 2;

    node->prim_id = -1;
    node->left  = build_recursive(bvh, prims, lo, mid, next_node, depth + 1);
    node->right = build_recursive(bvh, prims, mid, hi, next_node, depth + 1);

    return node_idx;
}

/* ── Public API ──────────────────────────────────────────────────── */

pamo_error pamo_bvh_build_triangles(pamo_bvh *bvh,
                                    const struct pamo_mesh *m,
                                    const pamo_allocator *alloc) {
    if (!bvh || !m || !alloc) return PAMO_ERR_INVALID_ARG;

    memset(bvh, 0, sizeof(*bvh));
    bvh->alloc = *alloc;

    /* Count alive faces. */
    size_t n_alive = 0;
    for (size_t i = 0; i < m->n_faces; i++) {
        if (m->face_alive[i]) n_alive++;
    }
    if (n_alive == 0) {
        bvh->n_prims = 0;
        bvh->n_nodes = 0;
        bvh->nodes = NULL;
        return PAMO_OK;
    }

    /* Build primitives array. */
    pamo_bvh_prim *prims = PAMO_ALLOC_ARRAY(alloc, pamo_bvh_prim, n_alive);
    if (!prims) return PAMO_ERR_ALLOC;

    size_t pi = 0;
    for (size_t i = 0; i < m->n_faces; i++) {
        if (!m->face_alive[i]) continue;
        pamo_vec3d v0 = m->verts[m->faces[i].v[0]];
        pamo_vec3d v1 = m->verts[m->faces[i].v[1]];
        pamo_vec3d v2 = m->verts[m->faces[i].v[2]];
        pamo_aabb box = aabb_empty();
        box = aabb_union_point(box, v0);
        box = aabb_union_point(box, v1);
        box = aabb_union_point(box, v2);
        prims[pi].box = box;
        prims[pi].centroid = pamo_v3_scale(
            pamo_v3_add(v0, pamo_v3_add(v1, v2)), 1.0 / 3.0);
        prims[pi].face_id = (int32_t)i;
        pi++;
    }

    /* Allocate nodes.  A binary tree with n leaves has at most 2n-1 nodes. */
    size_t max_nodes = 2 * n_alive;
    bvh->nodes = PAMO_ALLOC_ARRAY(alloc, pamo_bvh_node, max_nodes);
    if (!bvh->nodes) {
        PAMO_FREE_ARRAY(alloc, prims, pamo_bvh_prim, n_alive);
        return PAMO_ERR_ALLOC;
    }

    size_t next_node = 0;
    build_recursive(bvh, prims, 0, n_alive, &next_node, 0);
    bvh->n_nodes = next_node;
    bvh->n_prims = n_alive;

    PAMO_FREE_ARRAY(alloc, prims, pamo_bvh_prim, n_alive);
    return PAMO_OK;
}

void pamo_bvh_destroy(pamo_bvh *bvh) {
    if (!bvh) return;
    if (bvh->nodes) {
        size_t max_nodes = 2 * bvh->n_prims;
        if (max_nodes == 0) max_nodes = 1;
        PAMO_FREE(&bvh->alloc, bvh->nodes,
                  max_nodes * sizeof(pamo_bvh_node));
        bvh->nodes = NULL;
    }
    bvh->n_nodes = 0;
    bvh->n_prims = 0;
}

/* ── Nearest-point query ─────────────────────────────────────────── */

static void nearest_recurse(const pamo_bvh *bvh, const pamo_mesh *m,
                            pamo_vec3d query, int32_t node_idx,
                            pamo_nearest_result *best) {
    if (node_idx < 0) return;
    const pamo_bvh_node *node = &bvh->nodes[node_idx];

    double box_dist = aabb_min_dist_sq(node->box, query);
    if (box_dist >= best->dist_sq) return;

    if (node->left < 0 && node->right < 0) {
        /* Leaf. */
        int32_t fi = node->prim_id;
        pamo_vec3d v0 = m->verts[m->faces[fi].v[0]];
        pamo_vec3d v1 = m->verts[m->faces[fi].v[1]];
        pamo_vec3d v2 = m->verts[m->faces[fi].v[2]];
        double dsq;
        pamo_vec3d cp = pamo_closest_point_on_tri(query, v0, v1, v2, &dsq);
        if (dsq < best->dist_sq) {
            best->dist_sq = dsq;
            best->point   = cp;
            best->prim_id = fi;
        }
        return;
    }

    /* Visit nearer child first. */
    double d_left  = aabb_min_dist_sq(bvh->nodes[node->left].box, query);
    double d_right = aabb_min_dist_sq(bvh->nodes[node->right].box, query);
    if (d_left <= d_right) {
        nearest_recurse(bvh, m, query, node->left, best);
        nearest_recurse(bvh, m, query, node->right, best);
    } else {
        nearest_recurse(bvh, m, query, node->right, best);
        nearest_recurse(bvh, m, query, node->left, best);
    }
}

pamo_error pamo_bvh_nearest(const pamo_bvh *bvh,
                            const pamo_mesh *m,
                            pamo_vec3d query,
                            pamo_nearest_result *result) {
    if (!bvh || !m || !result) return PAMO_ERR_INVALID_ARG;
    if (bvh->n_nodes == 0) return PAMO_ERR_INVALID_ARG;

    result->dist_sq = 1e30;
    result->prim_id = -1;
    result->point   = pamo_v3_zero();

    nearest_recurse(bvh, m, query, 0, result);
    return PAMO_OK;
}

/* ── Overlap query ───────────────────────────────────────────────── */

void pamo_overlap_result_init(pamo_overlap_result *r,
                              const pamo_allocator *alloc) {
    if (!r) return;
    memset(r, 0, sizeof(*r));
    r->alloc = *alloc;
}

void pamo_overlap_result_destroy(pamo_overlap_result *r) {
    if (!r) return;
    if (r->hits) {
        pamo_free(&r->alloc, r->hits, r->n_hits_cap * sizeof(int32_t));
        r->hits = NULL;
    }
    r->n_hits = 0;
    r->n_hits_cap = 0;
}

static pamo_error overlap_push(pamo_overlap_result *r, int32_t prim_id) {
    if (r->n_hits >= r->n_hits_cap) {
        size_t new_cap = r->n_hits_cap == 0 ? 64 : r->n_hits_cap * 2;
        int32_t *new_hits = (int32_t *)pamo_realloc(
            &r->alloc, r->hits,
            r->n_hits_cap * sizeof(int32_t),
            new_cap * sizeof(int32_t));
        if (!new_hits) return PAMO_ERR_ALLOC;
        r->hits = new_hits;
        r->n_hits_cap = new_cap;
    }
    r->hits[r->n_hits++] = prim_id;
    return PAMO_OK;
}

static pamo_error overlap_recurse(const pamo_bvh *bvh,
                                  pamo_aabb query_box,
                                  int32_t node_idx,
                                  pamo_overlap_result *result) {
    if (node_idx < 0) return PAMO_OK;
    const pamo_bvh_node *node = &bvh->nodes[node_idx];

    if (!aabb_overlaps(node->box, query_box)) return PAMO_OK;

    if (node->left < 0 && node->right < 0) {
        return overlap_push(result, node->prim_id);
    }

    pamo_error err = overlap_recurse(bvh, query_box, node->left, result);
    if (err != PAMO_OK) return err;
    return overlap_recurse(bvh, query_box, node->right, result);
}

pamo_error pamo_bvh_overlap(const pamo_bvh *bvh,
                            pamo_aabb query_box,
                            pamo_overlap_result *result) {
    if (!bvh || !result) return PAMO_ERR_INVALID_ARG;
    if (bvh->n_nodes == 0) return PAMO_OK;
    result->n_hits = 0;
    return overlap_recurse(bvh, query_box, 0, result);
}
