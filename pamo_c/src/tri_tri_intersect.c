/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle-triangle intersection test.
 * Based on Moeller's "A Fast Triangle-Triangle Intersection Test" (1997).
 * Written from scratch using the geometric approach described in the paper.
 */
#include "pamo/pamo_types.h"

#include <math.h>

/* Internal: orient3d sign -- sign of dot(n, p - v0) where n = (v1-v0)x(v2-v0). */
static double orient_tri(pamo_vec3d v0, pamo_vec3d v1, pamo_vec3d v2,
                         pamo_vec3d p) {
    pamo_vec3d n = pamo_v3_cross(pamo_v3_sub(v1, v0), pamo_v3_sub(v2, v0));
    return pamo_v3_dot(n, pamo_v3_sub(p, v0));
}

/* Project interval of triangle onto the line direction D.
 * Given signed distances d0,d1,d2 of vertices to the other triangle's plane,
 * and projections p0,p1,p2 onto the intersection line,
 * compute the interval [t0,t1] where the triangle crosses the line.
 * Assumes d0 is on the opposite side from d1,d2 (or adjust caller). */
static void compute_interval(double p0, double p1, double p2,
                             double d0, double d1, double d2,
                             double *t0, double *t1) {
    double denom01 = d0 - d1;
    double denom02 = d0 - d2;
    double a = (fabs(denom01) > 1e-15) ? p0 + (p1 - p0) * d0 / denom01 : p0;
    double b = (fabs(denom02) > 1e-15) ? p0 + (p2 - p0) * d0 / denom02 : p0;
    if (a > b) { double tmp = a; a = b; b = tmp; }
    *t0 = a;
    *t1 = b;
}

/*
 * Test if two triangles intersect in 3D.
 *
 * Triangles sharing an edge or vertex are NOT considered intersecting
 * (the caller should skip pairs with shared vertices if desired).
 *
 * Returns true if the triangles intersect.
 */
bool pamo_tri_tri_intersect(pamo_vec3d a0, pamo_vec3d a1, pamo_vec3d a2,
                            pamo_vec3d b0, pamo_vec3d b1, pamo_vec3d b2) {
    /* Signed distances of b's vertices to a's plane. */
    double db0 = orient_tri(a0, a1, a2, b0);
    double db1 = orient_tri(a0, a1, a2, b1);
    double db2 = orient_tri(a0, a1, a2, b2);

    /* Snap near-zero to zero. */
    double eps = 1e-15;
    if (fabs(db0) < eps) db0 = 0.0;
    if (fabs(db1) < eps) db1 = 0.0;
    if (fabs(db2) < eps) db2 = 0.0;

    /* If all on the same side, no intersection. */
    if (db0 > 0 && db1 > 0 && db2 > 0) return false;
    if (db0 < 0 && db1 < 0 && db2 < 0) return false;

    /* Signed distances of a's vertices to b's plane. */
    double da0 = orient_tri(b0, b1, b2, a0);
    double da1 = orient_tri(b0, b1, b2, a1);
    double da2 = orient_tri(b0, b1, b2, a2);

    if (fabs(da0) < eps) da0 = 0.0;
    if (fabs(da1) < eps) da1 = 0.0;
    if (fabs(da2) < eps) da2 = 0.0;

    if (da0 > 0 && da1 > 0 && da2 > 0) return false;
    if (da0 < 0 && da1 < 0 && da2 < 0) return false;

    /* Coplanar case: skip (conservative -- treat as non-intersecting). */
    if (db0 == 0 && db1 == 0 && db2 == 0) return false;

    /* Compute intersection line direction. */
    pamo_vec3d na = pamo_v3_cross(pamo_v3_sub(a1, a0), pamo_v3_sub(a2, a0));
    pamo_vec3d nb = pamo_v3_cross(pamo_v3_sub(b1, b0), pamo_v3_sub(b2, b0));
    pamo_vec3d D = pamo_v3_cross(na, nb);

    /* Project vertices onto the intersection line. */
    /* Use the axis with largest |D| component. */
    int axis = 0;
    double mx = fabs(D.x);
    if (fabs(D.y) > mx) { mx = fabs(D.y); axis = 1; }
    if (fabs(D.z) > mx) { axis = 2; }

    double pa0, pa1, pa2, pb0, pb1, pb2;
    switch (axis) {
        case 0: pa0=a0.x; pa1=a1.x; pa2=a2.x; pb0=b0.x; pb1=b1.x; pb2=b2.x; break;
        case 1: pa0=a0.y; pa1=a1.y; pa2=a2.y; pb0=b0.y; pb1=b1.y; pb2=b2.y; break;
        default:pa0=a0.z; pa1=a1.z; pa2=a2.z; pb0=b0.z; pb1=b1.z; pb2=b2.z; break;
    }

    /* Rearrange triangle A so vertex 0 is on the opposite side. */
    double ta0, ta1;
    if (da0 * da1 > 0) {
        /* da2 is the loner. */
        compute_interval(pa2, pa0, pa1, da2, da0, da1, &ta0, &ta1);
    } else if (da0 * da2 > 0) {
        /* da1 is the loner. */
        compute_interval(pa1, pa0, pa2, da1, da0, da2, &ta0, &ta1);
    } else {
        /* da0 is the loner (or zero cases). */
        compute_interval(pa0, pa1, pa2, da0, da1, da2, &ta0, &ta1);
    }

    /* Same for triangle B. */
    double tb0, tb1;
    if (db0 * db1 > 0) {
        compute_interval(pb2, pb0, pb1, db2, db0, db1, &tb0, &tb1);
    } else if (db0 * db2 > 0) {
        compute_interval(pb1, pb0, pb2, db1, db0, db2, &tb0, &tb1);
    } else {
        compute_interval(pb0, pb1, pb2, db0, db1, db2, &tb0, &tb1);
    }

    /* Check if intervals overlap. */
    if (ta0 > ta1) { double tmp = ta0; ta0 = ta1; ta1 = tmp; }
    if (tb0 > tb1) { double tmp = tb0; tb0 = tb1; tb1 = tmp; }

    return ta0 < tb1 && tb0 < ta1;
}
