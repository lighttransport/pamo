/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 3: SAFE projection driver.
 */
#include "pamo/pamo_stage3.h"
#include "pamo/pamo_bvh.h"

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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

/* Elastic energy forward declarations. */
typedef struct { double m[2][2]; } pamo_elastic_rest_mat2;
typedef struct { pamo_elastic_rest_mat2 inv_DtD; double area; } pamo_elastic_rest;

void   pamo_elastic_precompute(const pamo_mesh *m, const pamo_vec3d *rest,
                               pamo_elastic_rest *state);
double pamo_elastic_energy(const pamo_mesh *m, const pamo_vec3d *q,
                           const pamo_elastic_rest *state,
                           double young, double poisson);
void   pamo_elastic_gradient_fd(const pamo_mesh *m, pamo_vec3d *q,
                                const pamo_elastic_rest *state,
                                double young, double poisson, double *grad);
void   pamo_elastic_hess_diag(const pamo_mesh *m, const pamo_vec3d *q,
                              const pamo_elastic_rest *state,
                              double young, double poisson, double *hdiag);
void   pamo_elastic_hess_vec_fd(const pamo_mesh *m, pamo_vec3d *q,
                                const pamo_elastic_rest *state,
                                double young, double poisson,
                                const double *dx, double *out);

/* CG solver. */
typedef void (*pamo_hess_vec_fn)(const double *dx, double *out,
                                 size_t n, void *ctx);
pamo_error pamo_cg_solve(pamo_hess_vec_fn hess_vec,
                         const double *hess_diag, const double *grad,
                         double *p_out, size_t n, int max_iters,
                         void *ctx, const pamo_allocator *alloc);

/* ── SAFE system context ─────────────────────────────────────────── */

typedef struct {
    pamo_mesh         *mesh;
    const pamo_mesh   *gt_mesh;
    pamo_bvh           gt_bvh;
    pamo_vec3d        *targets;
    pamo_vec3d        *rest_verts;
    pamo_elastic_rest *elastic_rest;
    pamo_safe_opts     opts;
    const pamo_allocator *alloc;
} pamo_safe_ctx;

static void safe_hess_vec(const double *dx, double *out,
                          size_t n, void *ctx_ptr) {
    pamo_safe_ctx *ctx = (pamo_safe_ctx *)ctx_ptr;
    memset(out, 0, n * 3 * sizeof(double));

    /* Distance Hessian-vector product. */
    pamo_distance_hess_vec(ctx->mesh->verts, n, dx, out,
                           ctx->opts.dist_stiffness);

    /* Elastic Hessian-vector product (finite differences). */
    pamo_elastic_hess_vec_fd(ctx->mesh, ctx->mesh->verts,
                             ctx->elastic_rest,
                             ctx->opts.elas_young_modulus,
                             ctx->opts.elas_poisson_ratio,
                             dx, out);
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

/* Compute total energy. */
static double total_energy(pamo_safe_ctx *ctx) {
    double E = 0.0;
    E += pamo_distance_energy(ctx->mesh->verts, ctx->mesh->n_verts,
                              ctx->targets, ctx->opts.dist_stiffness);
    E += pamo_elastic_energy(ctx->mesh, ctx->mesh->verts,
                             ctx->elastic_rest,
                             ctx->opts.elas_young_modulus,
                             ctx->opts.elas_poisson_ratio);
    return E;
}

/* ── Main SAFE projection ────────────────────────────────────────── */

pamo_error pamo_safe_project(pamo_mesh *mesh, const pamo_mesh *gt_mesh,
                             const pamo_safe_opts *opts,
                             const pamo_allocator *alloc) {
    if (!mesh || !gt_mesh || !opts || !alloc) return PAMO_ERR_INVALID_ARG;

    size_t n = mesh->n_verts;
    size_t dim = n * 3;

    pamo_safe_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.mesh = mesh;
    ctx.gt_mesh = gt_mesh;
    ctx.opts = *opts;
    ctx.alloc = alloc;

    pamo_error err = pamo_bvh_build_triangles(&ctx.gt_bvh, gt_mesh, alloc);
    if (err != PAMO_OK) return err;

    /* Allocate. */
    ctx.targets      = PAMO_ALLOC_ARRAY(alloc, pamo_vec3d, n);
    ctx.rest_verts   = PAMO_ALLOC_ARRAY(alloc, pamo_vec3d, n);
    ctx.elastic_rest = (pamo_elastic_rest *)pamo_alloc(alloc,
                        mesh->n_faces * sizeof(pamo_elastic_rest));
    double *grad  = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *hdiag = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *p     = (double *)pamo_alloc(alloc, dim * sizeof(double));

    if (!ctx.targets || !ctx.rest_verts || !ctx.elastic_rest ||
        !grad || !hdiag || !p) {
        err = PAMO_ERR_ALLOC;
        goto cleanup;
    }

    /* Save rest positions and precompute elastic rest state. */
    memcpy(ctx.rest_verts, mesh->verts, n * sizeof(pamo_vec3d));
    pamo_elastic_precompute(mesh, ctx.rest_verts, ctx.elastic_rest);

    for (int32_t outer = 0; outer < opts->n_outer_iters; outer++) {
        /* Update closest-point targets. */
        err = pamo_distance_update_targets(mesh->verts, n, gt_mesh,
                                           &ctx.gt_bvh, ctx.targets);
        if (err != PAMO_OK) goto cleanup;

        for (int32_t newton = 0; newton < opts->n_newton_iters; newton++) {
            memset(grad, 0, dim * sizeof(double));
            for (size_t i = 0; i < dim; i++) hdiag[i] = 1e-8;

            /* Distance energy gradient. */
            pamo_distance_gradient(mesh->verts, n, ctx.targets,
                                   opts->dist_stiffness, grad, hdiag);

            /* Elastic gradient (finite differences). */
            pamo_elastic_gradient_fd(mesh, mesh->verts, ctx.elastic_rest,
                                     opts->elas_young_modulus,
                                     opts->elas_poisson_ratio, grad);
            pamo_elastic_hess_diag(mesh, mesh->verts, ctx.elastic_rest,
                                   opts->elas_young_modulus,
                                   opts->elas_poisson_ratio, hdiag);

            /* CG solve: H * p = -grad */
            memset(p, 0, dim * sizeof(double));
            err = pamo_cg_solve(safe_hess_vec, hdiag, grad, p, n,
                                opts->n_cg_iters, &ctx, alloc);
            if (err != PAMO_OK) goto cleanup;

            /* Line search. */
            double gp = 0.0;
            for (size_t i = 0; i < dim; i++) gp += grad[i] * p[i];
            if (gp >= 0.0) continue;

            double E0 = total_energy(&ctx);
            double alpha = 1.0;
            bool stepped = false;

            for (int ls = 0; ls < opts->n_line_search_iters; ls++) {
                for (size_t i = 0; i < n; i++) {
                    mesh->verts[i].x += alpha * p[i * 3 + 0];
                    mesh->verts[i].y += alpha * p[i * 3 + 1];
                    mesh->verts[i].z += alpha * p[i * 3 + 2];
                }

                double E1 = total_energy(&ctx);
                if (E1 < E0 + 1e-4 * alpha * gp) {
                    stepped = true;
                    break;
                }

                /* Revert. */
                for (size_t i = 0; i < n; i++) {
                    mesh->verts[i].x -= alpha * p[i * 3 + 0];
                    mesh->verts[i].y -= alpha * p[i * 3 + 1];
                    mesh->verts[i].z -= alpha * p[i * 3 + 2];
                }
                alpha *= 0.5;
            }

            if (!stepped && alpha > 1e-10) {
                /* Apply with smallest alpha anyway. */
                for (size_t i = 0; i < n; i++) {
                    mesh->verts[i].x += alpha * p[i * 3 + 0];
                    mesh->verts[i].y += alpha * p[i * 3 + 1];
                    mesh->verts[i].z += alpha * p[i * 3 + 2];
                }
            }
        }

        double E = total_energy(&ctx);
        fprintf(stderr, "  SAFE outer %d: energy=%.6e\n", outer + 1, E);
    }

cleanup:
    if (ctx.targets) PAMO_FREE_ARRAY(alloc, ctx.targets, pamo_vec3d, n);
    if (ctx.rest_verts) PAMO_FREE_ARRAY(alloc, ctx.rest_verts, pamo_vec3d, n);
    if (ctx.elastic_rest)
        pamo_free(alloc, ctx.elastic_rest,
                  mesh->n_faces * sizeof(pamo_elastic_rest));
    if (grad)  pamo_free(alloc, grad,  dim * sizeof(double));
    if (hdiag) pamo_free(alloc, hdiag, dim * sizeof(double));
    if (p)     pamo_free(alloc, p,     dim * sizeof(double));
    pamo_bvh_destroy(&ctx.gt_bvh);

    return err;
}
