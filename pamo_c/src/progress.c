/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_progress.h"

#include <stddef.h>

/* PaMO's stages are single-threaded at the outer level — the SDF /
 * dual-MC paths use OpenMP internally, but the progress emit points
 * we add are at sequential outer-iteration boundaries, so no mutex is
 * needed. */
static pamo_progress_fn g_cb = NULL;
static void *g_userdata = NULL;

void pamo_set_progress_callback(pamo_progress_fn cb, void *userdata) {
    g_cb = cb;
    g_userdata = userdata;
}

void pamo_emit_progress(const char *stage, int64_t current, int64_t total) {
    if (g_cb) {
        g_cb(stage, current, total, g_userdata);
    }
}
