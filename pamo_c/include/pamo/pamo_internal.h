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

#ifdef __cplusplus
}
#endif

#endif /* PAMO_INTERNAL_H */
