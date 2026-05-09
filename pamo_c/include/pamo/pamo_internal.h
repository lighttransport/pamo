/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Internal declarations shared across library source files.
 * NOT part of the public API.
 */
#ifndef PAMO_INTERNAL_H
#define PAMO_INTERNAL_H

#include "pamo_types.h"
#include "pamo_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Triangle-triangle intersection ──────────────────────────────── */

bool pamo_tri_tri_intersect(pamo_vec3d a0, pamo_vec3d a1, pamo_vec3d a2,
                            pamo_vec3d b0, pamo_vec3d b1, pamo_vec3d b2);

/* ── Edge cost sorting ───────────────────────────────────────────── */

typedef struct {
    double  cost;
    int32_t edge_idx;
} pamo_edge_cost_entry;

void pamo_sort_edge_costs(pamo_edge_cost_entry *arr, size_t n);

/* ── Mesh growth helpers (defined in src/mesh/mesh.c) ─────────────
 * Reserve a capacity ≥ n; double-and-grow with PAMO_REALLOC. Append
 * helpers grow as needed, mark the new slot alive, and (for verts)
 * return the new index via *out. */
pamo_error pamo_mesh_reserve_verts(pamo_mesh *m, size_t n);
pamo_error pamo_mesh_reserve_faces(pamo_mesh *m, size_t n);
pamo_error pamo_mesh_append_vert(pamo_mesh *m, pamo_vec3d p, int32_t *out);
pamo_error pamo_mesh_append_face(pamo_mesh *m,
                                 int32_t a, int32_t b, int32_t c);

/* ── Stage 2 post-process (defined in src/stage2/postprocess.c) ───
 * Bowtie-vertex split + boundary-loop hole-fill. Run after the main
 * simplification loop to make the output watertight + 2-manifold for
 * collision use. Idempotent on already-clean meshes. */
pamo_error pamo_simplify_postprocess(pamo_mesh *mesh);

/* ── Stage 2 collapse predicates + apply (src/stage2/collapse.c) ─── */

/* Upper bound on 1-ring valence handled on the stack. ~30 is typical;
 * 256 stays safe (2 KiB stack per call). When a vertex exceeds this
 * the predicates conservatively report "unsafe" instead of truncating
 * the neighbour list. */
#define PAMO_MAX_VALENCE 256

/* Returns true if collapsing endpoint→mid would flip an adjacent
 * triangle's normal past `flip_threshold` (dot product) or produce a
 * skinny triangle. Tests faces incident to `endpoint` only; call
 * twice (u→mid, v→mid) for a symmetric check. */
bool pamo_collapse_creates_problem(const pamo_mesh *m,
                                   int32_t endpoint, int32_t other,
                                   pamo_vec3d mid,
                                   double skinny_thresh,
                                   double flip_threshold);

/* Topological link condition. Returns true if collapsing edge (u,v)
 * would create a non-manifold vertex (interior edge: shared-neighbour
 * count != 2; boundary edge: != 1; otherwise unsafe). */
bool pamo_collapse_violates_link(const pamo_mesh *m, int32_t u, int32_t v);

/* Predictive 1-ring tri-tri test. Returns true if collapsing (u,v)
 * onto `mid` would produce self-intersecting triangles in the
 * resulting 1-ring. O(K^2) tri-tri tests, K = local valence. */
bool pamo_collapse_self_intersects(const pamo_mesh *m,
                                   int32_t u, int32_t v,
                                   pamo_vec3d mid);

/* Perform the collapse: u absorbs v, mid becomes u's new position,
 * v's faces are remapped to u (or marked dead if shared with u or
 * degenerate after remap). Caller must have validated link condition
 * etc. first. */
void pamo_apply_collapse(pamo_mesh *m, int32_t u, int32_t v,
                         pamo_vec3d mid);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_INTERNAL_H */
