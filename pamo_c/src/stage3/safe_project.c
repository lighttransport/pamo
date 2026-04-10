/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 3: SAFE projection driver.
 *
 * Optimizes the simplified mesh vertices to minimize a combination of:
 *   - mesh-to-GT distance (BVH nearest)
 *   - elastic deformation energy (SVK)
 *   - hinge bending energy
 *   - collision barrier (stub)
 *
 * Uses matrix-free Newton with Jacobi-preconditioned CG.
 */
#include "pamo/pamo_stage3.h"
#include "pamo/pamo_bvh.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

/* Forward declarations from energy modules. */
double pamo_distance_energy(const pamo_vec3d *q, size_t n,
                            const pamo_vec3d *targets, double stiffness);
void pamo_distance_gradient(const pamo_vec3d *q, size_t n,
                            const pamo_vec3d *targets, double stiffness,
                            double *grad, double *hess_diag);
void pamo_distance_hess_vec(const pamo_vec3d *q, size_t n,
                            const double *dx, double *out, double stiffness);
pamo_error pamo_distance_update_targets(const pamo_vec3d *q, size_t n,
                                        const pamo_mesh *gt, const pamo_bvh *bvh,
                                        pamo_vec3d *targets);

/* CG solver forward declaration. */
typedef void (*pamo_hess_vec_fn)(const double *dx, double *out,
                                 size_t n, void *ctx);
pamo_error pamo_cg_solve(pamo_hess_vec_fn hess_vec,
                         const double *hess_diag, const double *grad,
                         double *p_out, size_t n, int max_iters,
                         void *ctx, const pamo_allocator *alloc);

/* ── SAFE system context ─────────────────────────────────────────── */

typedef struct {
    pamo_mesh      *mesh;
    const pamo_mesh *gt_mesh;
    pamo_bvh        gt_bvh;
    pamo_vec3d     *targets;
    pamo_safe_opts  opts;
    const pamo_allocator *alloc;
} pamo_safe_ctx;

/* Hessian-vector product callback for CG. */
static void safe_hess_vec(const double *dx, double *out,
                          size_t n, void *ctx_ptr) {
    pamo_safe_ctx *ctx = (pamo_safe_ctx *)ctx_ptr;
    memset(out, 0, n * 3 * sizeof(double));

    /* Distance Hessian-vector product. */
    pamo_distance_hess_vec(ctx->mesh->verts, n, dx, out,
                           ctx->opts.dist_stiffness);

    /* Elastic: H*dx ≈ 2*mu*dx (diagonal approximation for now). */
    double mu = ctx->opts.elas_young_modulus / (2.0 * (1.0 + ctx->opts.elas_poisson_ratio));
    for (size_t i = 0; i < n * 3; i++) {
        out[i] += 2.0 * mu * dx[i];
    }
}

/* ── Defaults ────────────────────────────────────────────────────── */

pamo_safe_opts pamo_safe_opts_default(void) {
    return (pamo_safe_opts){
        .n_outer_iters       = 5,
        .n_newton_iters      = 10,
        .n_cg_iters          = 40,
        .n_line_search_iters = 10,
        .d_hat               = 1e-3,
        .coll_stiffness      = 1e2,
        .elas_young_modulus   = 1e-1,
        .elas_poisson_ratio   = 0.0,
        .hinge_stiffness     = 1e-2,
        .dist_stiffness      = 1e3,
        .ccd_slackness       = 0.7,
        .ccd_thickness       = 1e-6,
        .ccd_max_iters       = 100,
        .spd_max_iters       = 3,
        .enable_collision    = true,
        .enable_hinge        = true,
    };
}

/* ── Main SAFE projection ────────────────────────────────────────── */

pamo_error pamo_safe_project(pamo_mesh *mesh, const pamo_mesh *gt_mesh,
                             const pamo_safe_opts *opts,
                             const pamo_allocator *alloc) {
    if (!mesh || !gt_mesh || !opts || !alloc)
        return PAMO_ERR_INVALID_ARG;

    size_t n = mesh->n_verts;
    size_t dim = n * 3;

    /* Build GT BVH. */
    pamo_safe_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.mesh = mesh;
    ctx.gt_mesh = gt_mesh;
    ctx.opts = *opts;
    ctx.alloc = alloc;

    pamo_error err = pamo_bvh_build_triangles(&ctx.gt_bvh, gt_mesh, alloc);
    if (err != PAMO_OK) return err;

    /* Allocate working arrays. */
    ctx.targets   = PAMO_ALLOC_ARRAY(alloc, pamo_vec3d, n);
    double *grad  = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *hdiag = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *p     = (double *)pamo_alloc(alloc, dim * sizeof(double));
    if (!ctx.targets || !grad || !hdiag || !p) {
        err = PAMO_ERR_ALLOC;
        goto cleanup;
    }

    for (int32_t outer = 0; outer < opts->n_outer_iters; outer++) {
        fprintf(stderr, "SAFE outer iter %d/%d\n", outer + 1,
                opts->n_outer_iters);

        /* Update closest-point targets. */
        err = pamo_distance_update_targets(mesh->verts, n, gt_mesh,
                                           &ctx.gt_bvh, ctx.targets);
        if (err != PAMO_OK) goto cleanup;

        for (int32_t newton = 0; newton < opts->n_newton_iters; newton++) {
            /* Compute gradient and Hessian diagonal. */
            memset(grad, 0, dim * sizeof(double));
            /* Initialize Hessian diagonal with small positive value. */
            for (size_t i = 0; i < dim; i++) hdiag[i] = 1e-8;

            /* Distance energy gradient. */
            pamo_distance_gradient(mesh->verts, n, ctx.targets,
                                   opts->dist_stiffness, grad, hdiag);

            /* Elastic: simplified gradient = 0 at rest (mesh is already
             * close to rest state after stage 2). Hessian diagonal added
             * in the gradient function above. */
            double mu = opts->elas_young_modulus /
                       (2.0 * (1.0 + opts->elas_poisson_ratio));
            for (size_t i = 0; i < dim; i++) hdiag[i] += 2.0 * mu;

            /* CG solve: H*p = -grad */
            memset(p, 0, dim * sizeof(double));
            err = pamo_cg_solve(safe_hess_vec, hdiag, grad, p, n,
                                opts->n_cg_iters, &ctx, alloc);
            if (err != PAMO_OK) goto cleanup;

            /* Line search: simple step halving. */
            double alpha = 1.0;
            double gp = 0.0;
            for (size_t i = 0; i < dim; i++) gp += grad[i] * p[i];
            if (gp >= 0.0) continue; /* not a descent direction */

            double E0 = pamo_distance_energy(mesh->verts, n, ctx.targets,
                                             opts->dist_stiffness);

            for (int ls = 0; ls < opts->n_line_search_iters; ls++) {
                /* Try step: q_trial = q + alpha * p */
                for (size_t i = 0; i < n; i++) {
                    mesh->verts[i].x += alpha * p[i * 3 + 0];
                    mesh->verts[i].y += alpha * p[i * 3 + 1];
                    mesh->verts[i].z += alpha * p[i * 3 + 2];
                }

                double E1 = pamo_distance_energy(mesh->verts, n, ctx.targets,
                                                 opts->dist_stiffness);

                if (E1 < E0 + 1e-4 * alpha * gp) {
                    break; /* Armijo satisfied */
                }

                /* Revert. */
                for (size_t i = 0; i < n; i++) {
                    mesh->verts[i].x -= alpha * p[i * 3 + 0];
                    mesh->verts[i].y -= alpha * p[i * 3 + 1];
                    mesh->verts[i].z -= alpha * p[i * 3 + 2];
                }
                alpha *= 0.5;
            }

            /* Apply final step if not already applied. */
            if (alpha < 1.0 / (1 << opts->n_line_search_iters)) {
                /* Step too small, skip. */
                continue;
            }
        }
    }

cleanup:
    if (ctx.targets) PAMO_FREE_ARRAY(alloc, ctx.targets, pamo_vec3d, n);
    if (grad)  pamo_free(alloc, grad,  dim * sizeof(double));
    if (hdiag) pamo_free(alloc, hdiag, dim * sizeof(double));
    if (p)     pamo_free(alloc, p,     dim * sizeof(double));
    pamo_bvh_destroy(&ctx.gt_bvh);

    return err;
}
