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

#ifdef __cplusplus
}
#endif

#endif /* PAMO_INTERNAL_H */
