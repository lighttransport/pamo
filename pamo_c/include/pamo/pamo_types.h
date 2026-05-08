/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PAMO_TYPES_H
#define PAMO_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes ─────────────────────────────────────────────────── */

typedef enum {
    PAMO_OK = 0,
    PAMO_ERR_ALLOC,
    PAMO_ERR_INVALID_ARG,
    PAMO_ERR_DEGENERATE,
    PAMO_ERR_NOT_MANIFOLD,
    PAMO_ERR_CONVERGENCE,
    PAMO_ERR_IO,
} pamo_error;

const char *pamo_error_string(pamo_error err);

/* ── Vector / index types ────────────────────────────────────────── */

typedef struct { double x, y, z; } pamo_vec3d;

typedef struct { int32_t v[3]; } pamo_tri;

typedef struct { int32_t u, v; } pamo_edge;

/* ── AABB ────────────────────────────────────────────────────────── */

typedef struct { pamo_vec3d lo, hi; } pamo_aabb;

/* ── Quadric (symmetric 4x4 stored as upper triangle, 10 elements) ─
 *
 * Layout (row-major upper triangle):
 *   m[0]  m[1]  m[2]  m[3]
 *         m[4]  m[5]  m[6]
 *               m[7]  m[8]
 *                     m[9]
 *
 * Index map: Q(i,j) with i<=j  =>  m[ i*(7-i)/2 + j ]
 *   (0,0)=0 (0,1)=1 (0,2)=2 (0,3)=3
 *            (1,1)=4 (1,2)=5 (1,3)=6
 *                     (2,2)=7 (2,3)=8
 *                              (3,3)=9
 */
typedef struct { double m[10]; } pamo_quadric;

/* ── Vec3 helpers (inline) ───────────────────────────────────────── */

static inline pamo_vec3d pamo_v3_add(pamo_vec3d a, pamo_vec3d b) {
    return (pamo_vec3d){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline pamo_vec3d pamo_v3_sub(pamo_vec3d a, pamo_vec3d b) {
    return (pamo_vec3d){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline pamo_vec3d pamo_v3_scale(pamo_vec3d a, double s) {
    return (pamo_vec3d){a.x * s, a.y * s, a.z * s};
}

static inline double pamo_v3_dot(pamo_vec3d a, pamo_vec3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline pamo_vec3d pamo_v3_cross(pamo_vec3d a, pamo_vec3d b) {
    return (pamo_vec3d){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static inline double pamo_v3_length_sq(pamo_vec3d a) {
    return pamo_v3_dot(a, a);
}

static inline pamo_vec3d pamo_v3_zero(void) {
    return (pamo_vec3d){0.0, 0.0, 0.0};
}

/* ── Quadric helpers (inline) ────────────────────────────────────── */

static inline pamo_quadric pamo_quadric_zero(void) {
    pamo_quadric q = {{0}};
    return q;
}

/* Add two quadrics. */
static inline pamo_quadric pamo_quadric_add(pamo_quadric a, pamo_quadric b) {
    pamo_quadric r;
    for (int i = 0; i < 10; i++) r.m[i] = a.m[i] + b.m[i];
    return r;
}

/* Build a quadric from a plane equation (nx, ny, nz, d).
 * Q = p * p^T  where p = [nx, ny, nz, d]. */
static inline pamo_quadric pamo_quadric_from_plane(double nx, double ny,
                                                   double nz, double d) {
    pamo_quadric q;
    q.m[0] = nx * nx;  q.m[1] = nx * ny;  q.m[2] = nx * nz;  q.m[3] = nx * d;
                        q.m[4] = ny * ny;  q.m[5] = ny * nz;  q.m[6] = ny * d;
                                            q.m[7] = nz * nz;  q.m[8] = nz * d;
                                                                q.m[9] = d * d;
    return q;
}

/* Evaluate v^T Q v  for v = (x, y, z, 1). */
static inline double pamo_quadric_eval(const pamo_quadric *q,
                                       pamo_vec3d v) {
    double x = v.x, y = v.y, z = v.z;
    return q->m[0]*x*x + 2.0*q->m[1]*x*y + 2.0*q->m[2]*x*z + 2.0*q->m[3]*x
         + q->m[4]*y*y + 2.0*q->m[5]*y*z + 2.0*q->m[6]*y
         + q->m[7]*z*z + 2.0*q->m[8]*z
         + q->m[9];
}

/* Solve for the QEM-optimal placement v* that minimises v^T Q v.
 *   A v* = -b, where A = top-left 3x3 of Q, b = (m[3], m[6], m[8]).
 * Returns true on success; false if A is (near-)singular (e.g. on
 * planar regions where Q is rank-1) and the caller should fall back
 * to e.g. midpoint.
 *
 * `cond_eps` is a unit-less conditioning threshold: |det(A)| must be
 * >= cond_eps * trace(A)^3 / 27. The trace^3 normalisation makes the
 * test scale-invariant under area-weighted quadrics, so a flat panel
 * (rank-1 A, det=0, finite trace) fails uniformly regardless of how
 * many faces feed into it. 1e-9 is a reasonable default. */
static inline bool pamo_quadric_optimal(const pamo_quadric *q,
                                        double cond_eps,
                                        pamo_vec3d *out) {
    /* A is symmetric: |A00 A01 A02; A01 A11 A12; A02 A12 A22|. */
    double a00 = q->m[0], a01 = q->m[1], a02 = q->m[2];
    double a11 = q->m[4], a12 = q->m[5];
    double a22 = q->m[7];
    double b0  = q->m[3], b1  = q->m[6], b2  = q->m[8];

    double c00 = a11*a22 - a12*a12;
    double c01 = a02*a12 - a01*a22;
    double c02 = a01*a12 - a02*a11;
    double det = a00*c00 + a01*c01 + a02*c02;

    /* Scale-invariant conditioning test. trace/3 is the average
     * eigenvalue; (trace/3)^3 is its cube. det/(trace/3)^3 = product
     * of eigenvalues / mean-cubed; near-zero iff at least one
     * eigenvalue is much smaller than the others (= near-singular). */
    double trace = a00 + a11 + a22;
    double mean = trace / 3.0;
    double scale_cube = mean * mean * mean;
    double abs_det = det < 0 ? -det : det;
    if (scale_cube <= 0.0) return false;
    if (abs_det < cond_eps * scale_cube) return false;

    double c11 = a00*a22 - a02*a02;
    double c12 = a01*a02 - a00*a12;
    double c22 = a00*a11 - a01*a01;
    double inv = 1.0 / det;
    /* v* = -A^-1 b. */
    out->x = -inv * (c00*b0 + c01*b1 + c02*b2);
    out->y = -inv * (c01*b0 + c11*b1 + c12*b2);
    out->z = -inv * (c02*b0 + c12*b1 + c22*b2);
    return true;
}

/* ── Debug assertion macro ───────────────────────────────────────── */

#ifndef NDEBUG
#include <assert.h>
#define PAMO_ASSERT(cond) assert(cond)
#else
#define PAMO_ASSERT(cond) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* PAMO_TYPES_H */
