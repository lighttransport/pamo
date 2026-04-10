/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_ALLOC_H
#define PAMO_ALLOC_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Allocator interface ─────────────────────────────────────────── */

typedef struct pamo_allocator {
    void *(*alloc)(size_t size, void *ctx);
    void *(*realloc)(void *ptr, size_t old_size, size_t new_size, void *ctx);
    void  (*free)(void *ptr, size_t size, void *ctx);
    void *ctx;
} pamo_allocator;

/* ── Tracking allocator ──────────────────────────────────────────── */

/* Creates a tracking allocator that wraps malloc/free and counts
 * allocations.  The returned allocator's ctx points to a heap-allocated
 * tracking context that must be released with
 * pamo_tracking_allocator_destroy(). */
pamo_allocator pamo_tracking_allocator_create(void);

/* Print allocation statistics to stderr. */
void pamo_tracking_allocator_report(const pamo_allocator *a);

/* Returns true if there are outstanding (leaked) allocations. */
bool pamo_tracking_allocator_has_leaks(const pamo_allocator *a);

/* Returns total bytes currently outstanding. */
size_t pamo_tracking_allocator_outstanding(const pamo_allocator *a);

/* Free the tracking context itself.  Must be called after all
 * allocations through this allocator have been freed. */
void pamo_tracking_allocator_destroy(pamo_allocator *a);

/* ── Default allocator (plain malloc/free, no tracking) ──────────── */

pamo_allocator pamo_default_allocator(void);

/* ── Convenience inline functions ─────────────────────────────────── */

static inline void *pamo_alloc(const pamo_allocator *a, size_t size) {
    return a->alloc(size, a->ctx);
}

static inline void *pamo_realloc(const pamo_allocator *a, void *ptr,
                                 size_t old_size, size_t new_size) {
    return a->realloc(ptr, old_size, new_size, a->ctx);
}

static inline void pamo_free(const pamo_allocator *a, void *ptr,
                             size_t size) {
    a->free(ptr, size, a->ctx);
}

/* Convenience macros using the inline functions. */
#define PAMO_ALLOC(a, size)             pamo_alloc((a), (size))
#define PAMO_REALLOC(a, p, os, ns)      pamo_realloc((a), (p), (os), (ns))
#define PAMO_FREE(a, p, size)           pamo_free((a), (p), (size))

#define PAMO_ALLOC_ARRAY(a, type, count) \
    ((type *)pamo_alloc((a), sizeof(type) * (count)))

#define PAMO_FREE_ARRAY(a, ptr, type, count) \
    pamo_free((a), (ptr), sizeof(type) * (count))

#ifdef __cplusplus
}
#endif

#endif /* PAMO_ALLOC_H */
