/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo.h"

#include <stdio.h>
#include <assert.h>
#include <math.h>

static void test_bvh_build_and_nearest(void) {
    pamo_allocator a = pamo_tracking_allocator_create();

    /* Create a two-triangle mesh (a unit square split diagonally). */
    pamo_mesh m;
    pamo_error err = pamo_mesh_create(&m, 4, 2, &a);
    assert(err == PAMO_OK);
    m.verts[0] = (pamo_vec3d){0, 0, 0};
    m.verts[1] = (pamo_vec3d){1, 0, 0};
    m.verts[2] = (pamo_vec3d){1, 1, 0};
    m.verts[3] = (pamo_vec3d){0, 1, 0};
    m.faces[0] = (pamo_tri){{0, 1, 2}};
    m.faces[1] = (pamo_tri){{0, 2, 3}};

    pamo_bvh bvh;
    err = pamo_bvh_build_triangles(&bvh, &m, &a);
    assert(err == PAMO_OK);
    assert(bvh.n_prims == 2);

    /* Query: point above center of square. */
    pamo_nearest_result res;
    err = pamo_bvh_nearest(&bvh, &m, (pamo_vec3d){0.5, 0.5, 1.0}, &res);
    assert(err == PAMO_OK);
    assert(fabs(res.point.z) < 1e-10);
    assert(fabs(res.dist_sq - 1.0) < 1e-10);

    /* Query: point on the mesh. */
    err = pamo_bvh_nearest(&bvh, &m, (pamo_vec3d){0.25, 0.25, 0.0}, &res);
    assert(err == PAMO_OK);
    assert(res.dist_sq < 1e-20);

    pamo_bvh_destroy(&bvh);
    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  bvh_build_and_nearest: PASS\n");
}

static void test_bvh_overlap(void) {
    pamo_allocator a = pamo_tracking_allocator_create();

    pamo_mesh m;
    pamo_error err = pamo_mesh_create(&m, 4, 2, &a);
    assert(err == PAMO_OK);
    m.verts[0] = (pamo_vec3d){0, 0, 0};
    m.verts[1] = (pamo_vec3d){1, 0, 0};
    m.verts[2] = (pamo_vec3d){1, 1, 0};
    m.verts[3] = (pamo_vec3d){0, 1, 0};
    m.faces[0] = (pamo_tri){{0, 1, 2}};
    m.faces[1] = (pamo_tri){{0, 2, 3}};

    pamo_bvh bvh;
    err = pamo_bvh_build_triangles(&bvh, &m, &a);
    assert(err == PAMO_OK);

    pamo_overlap_result ores;
    pamo_overlap_result_init(&ores, &a);

    /* Query box covering the whole mesh. */
    pamo_aabb qbox = {{-0.1, -0.1, -0.1}, {1.1, 1.1, 0.1}};
    err = pamo_bvh_overlap(&bvh, qbox, &ores);
    assert(err == PAMO_OK);
    assert(ores.n_hits == 2);

    /* Query box missing everything. */
    ores.n_hits = 0;
    qbox = (pamo_aabb){{5.0, 5.0, 5.0}, {6.0, 6.0, 6.0}};
    err = pamo_bvh_overlap(&bvh, qbox, &ores);
    assert(err == PAMO_OK);
    assert(ores.n_hits == 0);

    pamo_overlap_result_destroy(&ores);
    pamo_bvh_destroy(&bvh);
    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  bvh_overlap: PASS\n");
}

int main(void) {
    printf("test_bvh:\n");
    test_bvh_build_and_nearest();
    test_bvh_overlap();
    printf("All BVH tests passed.\n");
    return 0;
}
