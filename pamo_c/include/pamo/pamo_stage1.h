/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_STAGE1_H
#define PAMO_STAGE1_H

#include "pamo_types.h"
#include "pamo_alloc.h"
#include "pamo_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

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
