/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_STAGE2_H
#define PAMO_STAGE2_H

#include "pamo_types.h"
#include "pamo_alloc.h"
#include "pamo_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t target_faces;           /* stop when face_count <= target */
    int32_t min_faces;              /* absolute minimum (default 10) */
    int32_t max_iters;              /* max simplification rounds (default 100) */
    int32_t tolerance;              /* stuck iterations before quit (default 4) */
    int32_t max_undo_retries;       /* self-intersection undo retries (default 5) */
    double  skinny_area_threshold;  /* skinny triangle area threshold (default 1e-6) */
    double  skinny_penalty_weight;  /* weight for skinny cost term (default 5.0) */
    double  cost_range;             /* cost clamp range (default 10.0) */
    bool    check_self_intersection; /* enable self-intersection rollback (default true) */
    bool    preserve_boundary;       /* lock vertices on mesh boundary edges
                                       (an edge with only 1 incident face) so
                                       cracks/seams in the input are not
                                       widened by collapses near them.
                                       Default: true. */
} pamo_simplify_opts;

pamo_simplify_opts pamo_simplify_opts_default(void);

/* Stage 2: Iterative mesh simplification.
 * Modifies mesh in place.  Adjacency is rebuilt internally. */
pamo_error pamo_simplify(pamo_mesh *mesh, const pamo_simplify_opts *opts);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_STAGE2_H */
