/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_MESH_H
#define PAMO_MESH_H

#include "pamo_types.h"
#include "pamo_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Mesh container ──────────────────────────────────────────────── */

typedef struct pamo_mesh {
    pamo_vec3d  *verts;           /* [n_verts_cap] vertex positions */
    pamo_tri    *faces;           /* [n_faces_cap] triangle indices */
    bool        *vert_alive;      /* [n_verts_cap] */
    bool        *face_alive;      /* [n_faces_cap] */
    size_t       n_verts;         /* number of logical vertices */
    size_t       n_faces;         /* number of logical faces */
    size_t       n_verts_cap;     /* allocated capacity */
    size_t       n_faces_cap;

    /* Adjacency (valid after pamo_mesh_build_adjacency) */
    int32_t     *vert_face_offset; /* [n_verts + 1] CSR row pointers */
    int32_t     *vert_face_list;   /* [total_valence] face IDs */
    pamo_edge   *edges;            /* [n_edges] canonical (u < v) */
    int32_t     *edge_face_offset; /* [n_edges + 1] CSR row pointers */
    int32_t     *edge_face_list;   /* face IDs per edge */
    size_t       n_edges;

    pamo_allocator alloc;
} pamo_mesh;

/* Create a mesh with given vertex/face counts.  All verts/faces
 * are marked alive.  Adjacency is NOT built. */
pamo_error pamo_mesh_create(pamo_mesh *m, size_t n_verts, size_t n_faces,
                            const pamo_allocator *alloc);

/* Free all memory owned by the mesh. */
void pamo_mesh_destroy(pamo_mesh *m);

/* Deep copy src -> dst.  dst must be uninitialized or destroyed. */
pamo_error pamo_mesh_deep_copy(pamo_mesh *dst, const pamo_mesh *src);

/* Build vertex->face adjacency (CSR), unique edge list, and
 * edge->face adjacency from the current alive verts/faces. */
pamo_error pamo_mesh_build_adjacency(pamo_mesh *m);

/* Free only the adjacency arrays (vert_face_*, edges, edge_face_*). */
void pamo_mesh_free_adjacency(pamo_mesh *m);

/* Remove dead verts/faces, remap face indices, shrink arrays.
 * Invalidates adjacency. */
pamo_error pamo_mesh_compact(pamo_mesh *m);

/* Count currently alive faces. */
size_t pamo_mesh_count_alive_faces(const pamo_mesh *m);

/* Count currently alive vertices. */
size_t pamo_mesh_count_alive_verts(const pamo_mesh *m);

/* ── Geometry utilities ──────────────────────────────────────────── */

/* Compute face normal (unnormalized: magnitude = 2 * area). */
pamo_vec3d pamo_face_normal(const pamo_mesh *m, int32_t face_id);

/* Compute face area. */
double pamo_face_area(const pamo_mesh *m, int32_t face_id);

/* Compute unit-length face normal.  Returns zero vector for
 * degenerate faces. */
pamo_vec3d pamo_face_unit_normal(const pamo_mesh *m, int32_t face_id);

/* Closest point on triangle (v0,v1,v2) to point p.
 * Returns the closest point and sets *dist_sq if non-NULL. */
pamo_vec3d pamo_closest_point_on_tri(pamo_vec3d p,
                                     pamo_vec3d v0, pamo_vec3d v1,
                                     pamo_vec3d v2,
                                     double *dist_sq);

/* Triangle quality metric: 4*sqrt(3)*area / (e0^2+e1^2+e2^2).
 * Returns 1.0 for equilateral, approaches 0 for degenerate. */
double pamo_triangle_quality(pamo_vec3d v0, pamo_vec3d v1, pamo_vec3d v2);

/* Check if a triangle is degenerate (area < threshold * max_edge^2). */
bool pamo_triangle_is_skinny(pamo_vec3d v0, pamo_vec3d v1, pamo_vec3d v2,
                             double threshold);

/* ── Topology utilities ──────────────────────────────────────────── */

/* Count the number of shared neighboring vertices between two
 * vertices u and v using the adjacency structure.  For a manifold
 * edge, this should be exactly 2. */
int32_t pamo_shared_neighbor_count(const pamo_mesh *m,
                                   int32_t u, int32_t v);

/* ── Mesh bounding box ───────────────────────────────────────────── */

pamo_aabb pamo_mesh_bounds(const pamo_mesh *m);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_MESH_H */
