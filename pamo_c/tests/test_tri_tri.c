/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_internal.h"

#include <stdio.h>
#include <assert.h>

static void test_non_intersecting(void) {
    /* Two triangles on parallel planes. */
    pamo_vec3d a0 = {0,0,0}, a1 = {1,0,0}, a2 = {0,1,0};
    pamo_vec3d b0 = {0,0,1}, b1 = {1,0,1}, b2 = {0,1,1};
    assert(!pamo_tri_tri_intersect(a0,a1,a2, b0,b1,b2));
    printf("  non_intersecting: PASS\n");
}

static void test_intersecting(void) {
    /* Two triangles that cross each other. */
    pamo_vec3d a0 = {0,0,-1}, a1 = {1,0,-1}, a2 = {0.5,0,1};
    pamo_vec3d b0 = {0.5,-1,0}, b1 = {0.5,1,0}, b2 = {0.5,0,0.5};

    /* Actually let me make a clearer crossing case. */
    pamo_vec3d c0 = {-1, 0, 0}, c1 = {1, 0, 0}, c2 = {0, 1, 0};
    pamo_vec3d d0 = {0, 0.5, -1}, d1 = {0, 0.5, 1}, d2 = {0, -0.5, 0};
    assert(pamo_tri_tri_intersect(c0,c1,c2, d0,d1,d2));
    printf("  intersecting: PASS\n");
    (void)a0; (void)a1; (void)a2; (void)b0; (void)b1; (void)b2;
}

static void test_same_plane(void) {
    /* Coplanar triangles -- we conservatively say no intersection. */
    pamo_vec3d a0 = {0,0,0}, a1 = {1,0,0}, a2 = {0,1,0};
    pamo_vec3d b0 = {0.5,0,0}, b1 = {1.5,0,0}, b2 = {0.5,1,0};
    assert(!pamo_tri_tri_intersect(a0,a1,a2, b0,b1,b2));
    printf("  coplanar: PASS\n");
}

static void test_separated(void) {
    /* Far apart. */
    pamo_vec3d a0 = {0,0,0}, a1 = {1,0,0}, a2 = {0,1,0};
    pamo_vec3d b0 = {10,10,10}, b1 = {11,10,10}, b2 = {10,11,10};
    assert(!pamo_tri_tri_intersect(a0,a1,a2, b0,b1,b2));
    printf("  separated: PASS\n");
}

int main(void) {
    printf("test_tri_tri:\n");
    test_non_intersecting();
    test_intersecting();
    test_same_plane();
    test_separated();
    printf("All tri-tri tests passed.\n");
    return 0;
}
