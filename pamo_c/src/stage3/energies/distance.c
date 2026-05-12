/* POSIX feature macro: required for sysconf on glibc. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

/* On macOS, _POSIX_C_SOURCE alone hides BSD extensions used below
 * (sysconf(_SC_NPROCESSORS_ONLN)). _DARWIN_C_SOURCE re-exposes them. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mesh-to-GT distance energy: for each vertex of the current mesh,
 * find closest point on GT mesh (via BVH) and penalize squared distance.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"

#include <math.h>
#include <stdlib.h>
#ifdef PAMO_USE_PTHREADS
#include <pthread.h>
#include <unistd.h>
#endif

/* Compute mesh-to-GT distance energy.
 * E = stiffness * sum_i ||q_i - target_i||^2
 * grad_i = 2 * stiffness * (q_i - target_i)
 * hess_diag_i = 2 * stiffness
 */
double pamo_distance_energy(const pamo_vec3d *q, size_t n_verts,
                            const pamo_vec3d *targets,
                            double stiffness) {
    double E = 0.0;
    for (size_t i = 0; i < n_verts; i++) {
        pamo_vec3d d = pamo_v3_sub(q[i], targets[i]);
        E += stiffness * pamo_v3_length_sq(d);
    }
    return E;
}

void pamo_distance_gradient(const pamo_vec3d *q, size_t n_verts,
                            const pamo_vec3d *targets,
                            double stiffness,
                            double *grad, double *hess_diag) {
    for (size_t i = 0; i < n_verts; i++) {
        pamo_vec3d d = pamo_v3_sub(q[i], targets[i]);
        double s2 = 2.0 * stiffness;
        grad[i * 3 + 0] += s2 * d.x;
        grad[i * 3 + 1] += s2 * d.y;
        grad[i * 3 + 2] += s2 * d.z;
        hess_diag[i * 3 + 0] += s2;
        hess_diag[i * 3 + 1] += s2;
        hess_diag[i * 3 + 2] += s2;
    }
}

void pamo_distance_hess_vec(const pamo_vec3d *q, size_t n_verts,
                            const double *dx, double *out,
                            double stiffness) {
    double s2 = 2.0 * stiffness;
    for (size_t i = 0; i < n_verts; i++) {
        out[i * 3 + 0] += s2 * dx[i * 3 + 0];
        out[i * 3 + 1] += s2 * dx[i * 3 + 1];
        out[i * 3 + 2] += s2 * dx[i * 3 + 2];
    }
    (void)q;
}

#ifdef PAMO_USE_PTHREADS
static int pamo_distance_thread_count(size_t max_tasks) {
    int nt = 0;
    const char *ev = getenv("PAMO_NUM_THREADS");
    if (ev && ev[0]) nt = atoi(ev);
    if (nt <= 0) {
        long on = sysconf(_SC_NPROCESSORS_ONLN);
        nt = (on > 0) ? (int)on : 8;
    }
    if (max_tasks > 0 && (size_t)nt > max_tasks) nt = (int)max_tasks;
    if (nt < 1) nt = 1;
    return nt;
}

typedef struct {
    const pamo_vec3d *q;
    size_t start;
    size_t end;
    const pamo_mesh *gt_mesh;
    const pamo_bvh *gt_bvh;
    pamo_vec3d *targets;
    pamo_error err;
} pamo_distance_target_arg;

static void *pamo_distance_target_worker(void *ptr) {
    pamo_distance_target_arg *a = (pamo_distance_target_arg *)ptr;
    a->err = PAMO_OK;
    for (size_t i = a->start; i < a->end; i++) {
        pamo_nearest_result res;
        pamo_error err = pamo_bvh_nearest(a->gt_bvh, a->gt_mesh, a->q[i], &res);
        if (err != PAMO_OK) {
            a->err = err;
            return NULL;
        }
        a->targets[i] = res.point;
    }
    return NULL;
}
#endif

/* Update targets: find closest point on GT mesh for each vertex. */
pamo_error pamo_distance_update_targets(const pamo_vec3d *q, size_t n_verts,
                                        const pamo_mesh *gt_mesh,
                                        const pamo_bvh *gt_bvh,
                                        pamo_vec3d *targets) {
    if (!q || !gt_mesh || !gt_bvh || !targets) return PAMO_ERR_INVALID_ARG;

#ifdef PAMO_USE_PTHREADS
    int nt = pamo_distance_thread_count(n_verts);
    if (nt > 1 && n_verts >= 1024) {
        pthread_t *threads = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        pamo_distance_target_arg *args =
            (pamo_distance_target_arg *)malloc((size_t)nt * sizeof(*args));
        if (threads && args) {
            int created = 0;
            for (int t = 0; t < nt; t++) {
                size_t start = ((size_t)t * n_verts) / (size_t)nt;
                size_t end = ((size_t)(t + 1) * n_verts) / (size_t)nt;
                args[t] = (pamo_distance_target_arg){
                    .q = q,
                    .start = start,
                    .end = end,
                    .gt_mesh = gt_mesh,
                    .gt_bvh = gt_bvh,
                    .targets = targets,
                    .err = PAMO_OK,
                };
                if (pthread_create(&threads[t], NULL,
                                   pamo_distance_target_worker,
                                   &args[t]) != 0) {
                    break;
                }
                created++;
            }

            if (created == nt) {
                pamo_error err = PAMO_OK;
                for (int t = 0; t < nt; t++) {
                    pthread_join(threads[t], NULL);
                    if (args[t].err != PAMO_OK && err == PAMO_OK) {
                        err = args[t].err;
                    }
                }
                free(args);
                free(threads);
                return err;
            }

            for (int t = 0; t < created; t++) {
                pthread_join(threads[t], NULL);
            }
        }
        free(args);
        free(threads);
    }
#endif

    for (size_t i = 0; i < n_verts; i++) {
        pamo_nearest_result res;
        pamo_error err = pamo_bvh_nearest(gt_bvh, gt_mesh, q[i], &res);
        if (err != PAMO_OK) return err;
        targets[i] = res.point;
    }
    return PAMO_OK;
}
