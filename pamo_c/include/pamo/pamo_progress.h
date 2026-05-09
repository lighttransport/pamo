/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Progress callback for long-running pipeline stages.
 * Called synchronously from PaMO's thread (no callback marshalling).
 * Cheap when no callback is installed.
 */
#ifndef PAMO_PROGRESS_H
#define PAMO_PROGRESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* `stage` is a short ASCII tag — "remesh", "simplify", "safe", plus
 * sub-stages like "remesh_sdf", "simplify_iter", "safe_outer".
 * `current`/`total` are integer counters; `total <= 0` means open-ended. */
typedef void (*pamo_progress_fn)(const char *stage,
                                 int64_t current,
                                 int64_t total,
                                 void *userdata);

void pamo_set_progress_callback(pamo_progress_fn cb, void *userdata);

/* Internal: emit one event to whatever callback is currently installed.
 * Safe to call from any PaMO source; defined in pamo_progress.c. */
void pamo_emit_progress(const char *stage, int64_t current, int64_t total);

#ifdef __cplusplus
}
#endif

#endif /* PAMO_PROGRESS_H */
