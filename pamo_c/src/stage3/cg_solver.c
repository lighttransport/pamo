/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Jacobi-preconditioned conjugate gradient solver (matrix-free).
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_alloc.h"

#include <math.h>
#include <string.h>

/*
 * Solve H * p = -grad using preconditioned CG.
 *
 * hess_vec: callback that computes out = H * dx
 * hess_diag: diagonal of H (for Jacobi preconditioner)
 * grad: gradient vector (3*n_verts doubles, xyz interleaved)
 * p: output search direction (3*n_verts doubles)
 * n: number of vertices
 * max_iters: maximum CG iterations
 * ctx: opaque context passed to hess_vec
 */
typedef void (*pamo_hess_vec_fn)(const double *dx, double *out,
                                 size_t n, void *ctx);

pamo_error pamo_cg_solve(pamo_hess_vec_fn hess_vec,
                         const double *hess_diag,
                         const double *grad,
                         double *p_out,
                         size_t n,
                         int max_iters,
                         void *ctx,
                         const pamo_allocator *alloc) {
    size_t dim = n * 3;

    double *r  = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *z  = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *v  = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *Av = (double *)pamo_alloc(alloc, dim * sizeof(double));
    if (!r || !z || !v || !Av) {
        if (r)  pamo_free(alloc, r,  dim * sizeof(double));
        if (z)  pamo_free(alloc, z,  dim * sizeof(double));
        if (v)  pamo_free(alloc, v,  dim * sizeof(double));
        if (Av) pamo_free(alloc, Av, dim * sizeof(double));
        return PAMO_ERR_ALLOC;
    }

    /* p = 0 */
    memset(p_out, 0, dim * sizeof(double));

    /* r = grad (we solve H*p = -grad, so r = -(-grad) = grad initially,
     * but CG convention: r = b - A*x0 = -grad - H*0 = -grad) */
    for (size_t i = 0; i < dim; i++) r[i] = -grad[i];

    /* z = M^{-1} r (Jacobi preconditioner: M = diag(H)) */
    for (size_t i = 0; i < dim; i++) {
        double d = hess_diag[i];
        z[i] = (fabs(d) > 1e-30) ? r[i] / d : r[i];
    }

    /* v = z */
    memcpy(v, z, dim * sizeof(double));

    double rz = 0.0;
    for (size_t i = 0; i < dim; i++) rz += r[i] * z[i];

    for (int iter = 0; iter < max_iters; iter++) {
        if (fabs(rz) < 1e-30) break;

        hess_vec(v, Av, n, ctx);

        double vAv = 0.0;
        for (size_t i = 0; i < dim; i++) vAv += v[i] * Av[i];
        if (fabs(vAv) < 1e-30) break;

        double alpha = rz / vAv;

        for (size_t i = 0; i < dim; i++) p_out[i] += alpha * v[i];
        for (size_t i = 0; i < dim; i++) r[i] -= alpha * Av[i];

        /* z_new = M^{-1} r */
        for (size_t i = 0; i < dim; i++) {
            double d = hess_diag[i];
            z[i] = (fabs(d) > 1e-30) ? r[i] / d : r[i];
        }

        double rz_new = 0.0;
        for (size_t i = 0; i < dim; i++) rz_new += r[i] * z[i];

        double beta = rz_new / rz;
        for (size_t i = 0; i < dim; i++) v[i] = z[i] + beta * v[i];
        rz = rz_new;
    }

    pamo_free(alloc, r,  dim * sizeof(double));
    pamo_free(alloc, z,  dim * sizeof(double));
    pamo_free(alloc, v,  dim * sizeof(double));
    pamo_free(alloc, Av, dim * sizeof(double));

    return PAMO_OK;
}
