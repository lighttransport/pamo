/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_alloc.h"
#include "pamo/pamo_types.h"

#include <stdio.h>
#include <assert.h>

static void test_default_allocator(void) {
    pamo_allocator a = pamo_default_allocator();
    int *p = (int *)PAMO_ALLOC(&a, sizeof(int) * 10);
    assert(p != NULL);
    for (int i = 0; i < 10; i++) assert(p[i] == 0); /* zero-init */
    p[0] = 42;
    PAMO_FREE(&a, p, sizeof(int) * 10);
    printf("  default_allocator: PASS\n");
}

static void test_tracking_allocator(void) {
    pamo_allocator a = pamo_tracking_allocator_create();

    assert(!pamo_tracking_allocator_has_leaks(&a));

    void *p1 = PAMO_ALLOC(&a, 100);
    assert(p1 != NULL);
    assert(pamo_tracking_allocator_has_leaks(&a));
    assert(pamo_tracking_allocator_outstanding(&a) == 100);

    void *p2 = PAMO_ALLOC(&a, 200);
    assert(p2 != NULL);
    assert(pamo_tracking_allocator_outstanding(&a) == 300);

    PAMO_FREE(&a, p1, 100);
    assert(pamo_tracking_allocator_outstanding(&a) == 200);

    PAMO_FREE(&a, p2, 200);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    assert(pamo_tracking_allocator_outstanding(&a) == 0);

    pamo_tracking_allocator_destroy(&a);
    printf("  tracking_allocator: PASS\n");
}

static void test_tracking_realloc(void) {
    pamo_allocator a = pamo_tracking_allocator_create();

    void *p = PAMO_ALLOC(&a, 64);
    assert(p != NULL);
    assert(pamo_tracking_allocator_outstanding(&a) == 64);

    p = PAMO_REALLOC(&a, p, 64, 128);
    assert(p != NULL);
    assert(pamo_tracking_allocator_outstanding(&a) == 128);

    PAMO_FREE(&a, p, 128);
    assert(!pamo_tracking_allocator_has_leaks(&a));

    pamo_tracking_allocator_destroy(&a);
    printf("  tracking_realloc: PASS\n");
}

int main(void) {
    printf("test_alloc:\n");
    test_default_allocator();
    test_tracking_allocator();
    test_tracking_realloc();
    printf("All alloc tests passed.\n");
    return 0;
}
