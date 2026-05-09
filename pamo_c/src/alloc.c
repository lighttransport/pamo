/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_alloc.h"
#include "pamo/pamo_types.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Default allocator (plain malloc/free) ───────────────────────── */

static void *default_alloc(size_t size, void *ctx) {
    (void)ctx;
    if (size == 0) return NULL;
    void *p = malloc(size);
    if (p) memset(p, 0, size);
    return p;
}

static void *default_realloc(void *ptr, size_t old_size, size_t new_size,
                             void *ctx) {
    (void)ctx;
    (void)old_size;
    if (new_size == 0) {
        free(ptr);
        return NULL;
    }
    void *p = realloc(ptr, new_size);
    if (p && new_size > old_size) {
        memset((char *)p + old_size, 0, new_size - old_size);
    }
    return p;
}

static void default_free(void *ptr, size_t size, void *ctx) {
    (void)ctx;
    (void)size;
    free(ptr);
}

pamo_allocator pamo_default_allocator(void) {
    return (pamo_allocator){
        .alloc   = default_alloc,
        .realloc = default_realloc,
        .free    = default_free,
        .ctx     = NULL,
    };
}

/* ── Tracking allocator ──────────────────────────────────────────── */

/* Track each live allocation by pointer so we can detect real leaks
 * regardless of whether callers pass the exact size to free. */

typedef struct pamo_alloc_record {
    void   *ptr;
    size_t  size;
    struct pamo_alloc_record *next;
} pamo_alloc_record;

typedef struct {
    pamo_alloc_record *live;     /* linked list of live allocations */
    size_t n_allocs;
    size_t n_frees;
    size_t total_allocated;
    size_t peak;
    size_t current;
} pamo_tracking_ctx;

static void record_add(pamo_tracking_ctx *t, void *ptr, size_t size) {
    pamo_alloc_record *r = (pamo_alloc_record *)malloc(sizeof(*r));
    if (!r) return;
    r->ptr  = ptr;
    r->size = size;
    r->next = t->live;
    t->live = r;
}

static size_t record_remove(pamo_tracking_ctx *t, void *ptr) {
    pamo_alloc_record **pp = &t->live;
    while (*pp) {
        if ((*pp)->ptr == ptr) {
            pamo_alloc_record *r = *pp;
            size_t sz = r->size;
            *pp = r->next;
            free(r);
            return sz;
        }
        pp = &(*pp)->next;
    }
    return 0; /* not found */
}

static void *tracking_alloc(size_t size, void *ctx) {
    pamo_tracking_ctx *t = (pamo_tracking_ctx *)ctx;
    if (size == 0) return NULL;
    void *p = malloc(size);
    if (!p) return NULL;
    memset(p, 0, size);
    record_add(t, p, size);
    t->total_allocated += size;
    t->current += size;
    t->n_allocs++;
    if (t->current > t->peak) t->peak = t->current;
    return p;
}

static void *tracking_realloc(void *ptr, size_t old_size, size_t new_size,
                              void *ctx) {
    pamo_tracking_ctx *t = (pamo_tracking_ctx *)ctx;
    (void)old_size;
    if (new_size == 0) {
        if (ptr) {
            size_t actual = record_remove(t, ptr);
            free(ptr);
            t->current -= actual;
            t->n_frees++;
        }
        return NULL;
    }
    if (!ptr) {
        return tracking_alloc(new_size, ctx);
    }
    /* Locate the record up-front and keep a direct handle. realloc
     * may invalidate ptr, so doing the lookup afterwards (with an
     * old_ptr key) is technically a use-after-free even though we
     * only compare the pointer value — silence the warning and avoid
     * a second list walk by writing the new ptr/size into the record
     * we already found. */
    size_t actual_old = 0;
    pamo_alloc_record *rec = t->live;
    while (rec) {
        if (rec->ptr == ptr) { actual_old = rec->size; break; }
        rec = rec->next;
    }
    void *p = realloc(ptr, new_size);
    if (!p) return NULL;
    if (new_size > actual_old) {
        memset((char *)p + actual_old, 0, new_size - actual_old);
    }
    if (rec) {
        rec->ptr = p;
        rec->size = new_size;
    } else {
        record_add(t, p, new_size);
    }
    if (new_size > actual_old) {
        t->total_allocated += (new_size - actual_old);
        t->current += (new_size - actual_old);
    } else {
        t->current -= (actual_old - new_size);
    }
    if (t->current > t->peak) t->peak = t->current;
    return p;
}

static void tracking_free(void *ptr, size_t size, void *ctx) {
    pamo_tracking_ctx *t = (pamo_tracking_ctx *)ctx;
    (void)size;
    if (!ptr) return;
    size_t actual = record_remove(t, ptr);
    free(ptr);
    t->current -= actual;
    t->n_frees++;
}

pamo_allocator pamo_tracking_allocator_create(void) {
    pamo_tracking_ctx *t = (pamo_tracking_ctx *)calloc(1, sizeof(*t));
    return (pamo_allocator){
        .alloc   = tracking_alloc,
        .realloc = tracking_realloc,
        .free    = tracking_free,
        .ctx     = t,
    };
}

void pamo_tracking_allocator_destroy(pamo_allocator *a) {
    if (a && a->ctx) {
        pamo_tracking_ctx *t = (pamo_tracking_ctx *)a->ctx;
        /* Free any remaining records (leak list). */
        pamo_alloc_record *r = t->live;
        while (r) {
            pamo_alloc_record *next = r->next;
            free(r);
            r = next;
        }
        free(t);
        a->ctx = NULL;
    }
}

bool pamo_tracking_allocator_has_leaks(const pamo_allocator *a) {
    if (!a || !a->ctx) return false;
    const pamo_tracking_ctx *t = (const pamo_tracking_ctx *)a->ctx;
    return t->live != NULL;
}

size_t pamo_tracking_allocator_outstanding(const pamo_allocator *a) {
    if (!a || !a->ctx) return 0;
    const pamo_tracking_ctx *t = (const pamo_tracking_ctx *)a->ctx;
    return t->current;
}

void pamo_tracking_allocator_report(const pamo_allocator *a) {
    if (!a || !a->ctx) return;
    const pamo_tracking_ctx *t = (const pamo_tracking_ctx *)a->ctx;
    fprintf(stderr, "pamo allocator: allocs=%zu frees=%zu "
            "outstanding=%zu peak=%zu\n",
            t->n_allocs, t->n_frees,
            t->current, t->peak);
    if (t->live) {
        size_t n = 0;
        const pamo_alloc_record *r = t->live;
        while (r) { n++; r = r->next; }
        fprintf(stderr, "  %zu live allocations:\n", n);
        r = t->live;
        for (size_t i = 0; i < 10 && r; i++, r = r->next) {
            fprintf(stderr, "    ptr=%p size=%zu\n", r->ptr, r->size);
        }
        if (n > 10) fprintf(stderr, "    ... and %zu more\n", n - 10);
    }
}

/* ── Error strings ───────────────────────────────────────────────── */

const char *pamo_error_string(pamo_error err) {
    switch (err) {
        case PAMO_OK:              return "ok";
        case PAMO_ERR_ALLOC:       return "allocation failed";
        case PAMO_ERR_INVALID_ARG: return "invalid argument";
        case PAMO_ERR_DEGENERATE:  return "degenerate mesh";
        case PAMO_ERR_NOT_MANIFOLD: return "non-manifold mesh";
        case PAMO_ERR_CONVERGENCE: return "convergence failure";
        case PAMO_ERR_IO:          return "I/O error";
    }
    return "unknown error";
}
