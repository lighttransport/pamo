/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo.h"

#include <stdio.h>
#include <assert.h>
#include <math.h>

/* Helper: create a single-triangle mesh. */
static void make_single_triangle(pamo_mesh *m, pamo_allocator *a) {
    pamo_error err = pamo_mesh_create(m, 3, 1, a);
    assert(err == PAMO_OK);
    m->verts[0] = (pamo_vec3d){0.0, 0.0, 0.0};
    m->verts[1] = (pamo_vec3d){1.0, 0.0, 0.0};
    m->verts[2] = (pamo_vec3d){0.0, 1.0, 0.0};
    m->faces[0] = (pamo_tri){{0, 1, 2}};
}

/* Helper: create a tetrahedron (4 verts, 4 faces). */
static void make_tetrahedron(pamo_mesh *m, pamo_allocator *a) {
    pamo_error err = pamo_mesh_create(m, 4, 4, a);
    assert(err == PAMO_OK);
    m->verts[0] = (pamo_vec3d){0.0, 0.0, 0.0};
    m->verts[1] = (pamo_vec3d){1.0, 0.0, 0.0};
    m->verts[2] = (pamo_vec3d){0.5, 1.0, 0.0};
    m->verts[3] = (pamo_vec3d){0.5, 0.5, 1.0};
    m->faces[0] = (pamo_tri){{0, 1, 2}};
    m->faces[1] = (pamo_tri){{0, 1, 3}};
    m->faces[2] = (pamo_tri){{1, 2, 3}};
    m->faces[3] = (pamo_tri){{0, 2, 3}};
}

static void test_create_destroy(void) {
    pamo_allocator a = pamo_tracking_allocator_create();
    pamo_mesh m;
    make_single_triangle(&m, &a);

    assert(m.n_verts == 3);
    assert(m.n_faces == 1);
    assert(m.vert_alive[0] && m.vert_alive[1] && m.vert_alive[2]);
    assert(m.face_alive[0]);

    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  create_destroy: PASS\n");
}

static void test_deep_copy(void) {
    pamo_allocator a = pamo_tracking_allocator_create();
    pamo_mesh m1, m2;
    make_single_triangle(&m1, &a);

    pamo_error err = pamo_mesh_deep_copy(&m2, &m1);
    assert(err == PAMO_OK);
    assert(m2.n_verts == 3);
    assert(m2.verts[1].x == 1.0);

    pamo_mesh_destroy(&m1);
    pamo_mesh_destroy(&m2);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  deep_copy: PASS\n");
}

static void test_geometry(void) {
    pamo_allocator a = pamo_tracking_allocator_create();
    pamo_mesh m;
    make_single_triangle(&m, &a);

    double area = pamo_face_area(&m, 0);
    assert(fabs(area - 0.5) < 1e-10);

    pamo_vec3d n = pamo_face_unit_normal(&m, 0);
    assert(fabs(n.z - 1.0) < 1e-10);

    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  geometry: PASS\n");
}

static void test_closest_point(void) {
    pamo_vec3d v0 = {0, 0, 0}, v1 = {1, 0, 0}, v2 = {0, 1, 0};
    double dsq;

    /* Point above center of triangle. */
    pamo_vec3d p = {0.25, 0.25, 1.0};
    pamo_vec3d cp = pamo_closest_point_on_tri(p, v0, v1, v2, &dsq);
    assert(fabs(cp.x - 0.25) < 1e-10);
    assert(fabs(cp.y - 0.25) < 1e-10);
    assert(fabs(cp.z) < 1e-10);
    assert(fabs(dsq - 1.0) < 1e-10);

    /* Point at vertex. */
    p = (pamo_vec3d){-0.5, -0.5, 0.0};
    cp = pamo_closest_point_on_tri(p, v0, v1, v2, &dsq);
    assert(fabs(cp.x) < 1e-10 && fabs(cp.y) < 1e-10 && fabs(cp.z) < 1e-10);

    printf("  closest_point: PASS\n");
}

static void test_adjacency(void) {
    pamo_allocator a = pamo_tracking_allocator_create();
    pamo_mesh m;
    make_tetrahedron(&m, &a);

    pamo_error err = pamo_mesh_build_adjacency(&m);
    assert(err == PAMO_OK);

    /* Tetrahedron: 4 faces, 6 edges. */
    assert(m.n_edges == 6);

    /* Each vertex is incident to 3 faces. */
    for (int i = 0; i < 4; i++) {
        int32_t cnt = m.vert_face_offset[i + 1] - m.vert_face_offset[i];
        assert(cnt == 3);
    }

    /* Each edge has exactly 2 incident faces (manifold). */
    for (size_t i = 0; i < m.n_edges; i++) {
        int32_t cnt = m.edge_face_offset[i + 1] - m.edge_face_offset[i];
        assert(cnt == 2);
    }

    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  adjacency: PASS\n");
}

static void test_compact(void) {
    pamo_allocator a = pamo_tracking_allocator_create();
    pamo_mesh m;
    make_tetrahedron(&m, &a);

    /* Kill vertex 3 and its incident faces (faces 1, 2, 3). */
    m.vert_alive[3] = false;
    m.face_alive[1] = false;
    m.face_alive[2] = false;
    m.face_alive[3] = false;

    pamo_error err = pamo_mesh_compact(&m);
    assert(err == PAMO_OK);
    assert(m.n_verts == 3);
    assert(m.n_faces == 1);
    assert(m.faces[0].v[0] == 0);
    assert(m.faces[0].v[1] == 1);
    assert(m.faces[0].v[2] == 2);

    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  compact: PASS\n");
}

static void test_shared_neighbors(void) {
    pamo_allocator a = pamo_tracking_allocator_create();
    pamo_mesh m;
    make_tetrahedron(&m, &a);

    pamo_error err = pamo_mesh_build_adjacency(&m);
    assert(err == PAMO_OK);

    /* For a tetrahedron, any edge shares exactly 2 neighbor vertices. */
    int32_t sn = pamo_shared_neighbor_count(&m, 0, 1);
    assert(sn == 2);

    pamo_mesh_destroy(&m);
    assert(!pamo_tracking_allocator_has_leaks(&a));
    pamo_tracking_allocator_destroy(&a);
    printf("  shared_neighbors: PASS\n");
}

static void test_quadric(void) {
    /* Q from plane z=0: n=(0,0,1), d=0 */
    pamo_quadric q = pamo_quadric_from_plane(0, 0, 1, 0);
    /* Eval at (1, 2, 3, 1): should be 3^2 = 9 */
    double val = pamo_quadric_eval(&q, (pamo_vec3d){1, 2, 3});
    assert(fabs(val - 9.0) < 1e-10);

    pamo_quadric q2 = pamo_quadric_add(q, q);
    val = pamo_quadric_eval(&q2, (pamo_vec3d){1, 2, 3});
    assert(fabs(val - 18.0) < 1e-10);

    printf("  quadric: PASS\n");
}

static void test_triangle_quality(void) {
    /* Equilateral-ish. */
    pamo_vec3d v0 = {0, 0, 0};
    pamo_vec3d v1 = {1, 0, 0};
    pamo_vec3d v2 = {0.5, 0.866025, 0};
    double q = pamo_triangle_quality(v0, v1, v2);
    assert(fabs(q - 1.0) < 0.01);

    /* Degenerate. */
    pamo_vec3d d0 = {0, 0, 0};
    pamo_vec3d d1 = {1, 0, 0};
    pamo_vec3d d2 = {0.5, 1e-10, 0};
    double qd = pamo_triangle_quality(d0, d1, d2);
    assert(qd < 0.001);

    printf("  triangle_quality: PASS\n");
}

int main(void) {
    printf("test_mesh:\n");
    test_create_destroy();
    test_deep_copy();
    test_geometry();
    test_closest_point();
    test_adjacency();
    test_compact();
    test_shared_neighbors();
    test_quadric();
    test_triangle_quality();
    printf("All mesh tests passed.\n");
    return 0;
}
