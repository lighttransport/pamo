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
