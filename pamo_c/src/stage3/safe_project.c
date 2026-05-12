/* POSIX feature macro: required for clock_gettime / CLOCK_MONOTONIC on glibc. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

/* On macOS, _POSIX_C_SOURCE=199309L narrows the libc surface and hides
 * snprintf / BSD extensions. _DARWIN_C_SOURCE restores them. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

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
#include <time.h>

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
void   pamo_elastic_hess_vec_fd_with_baseline(const pamo_mesh *m, pamo_vec3d *q,
                                              const pamo_elastic_rest *state,
                                              double young, double poisson,
                                              const double *grad0,
                                              double *grad1,
                                              const double *dx, double *out);

/* CG solver. */
typedef void (*pamo_hess_vec_fn)(const double *dx, double *out,
                                 size_t n, void *ctx);
pamo_error pamo_cg_solve(pamo_hess_vec_fn hess_vec,
                         const double *hess_diag, const double *grad,
                         double *p_out, size_t n, int max_iters,
                         void *ctx, const pamo_allocator *alloc);

/* ── Profiling helpers ───────────────────────────────────────────── */

static int stage3_profile_enabled(void) {
    const char *ev = getenv("PAMO_STAGE3_PROFILE");
    if (ev && ev[0]) return atoi(ev) != 0;
    ev = getenv("PAMO_SDF_PROFILE");
    return (ev && ev[0]) ? (atoi(ev) != 0) : 0;
}

static double stage3_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void stage3_profile_log(const char *label, double seconds) {
    fprintf(stderr, "[stage3-prof] %-28s %.3f s\n", label, seconds);
}

/* ── SAFE system context ─────────────────────────────────────────── */

typedef struct {
    pamo_mesh         *mesh;
    const pamo_mesh   *gt_mesh;
    pamo_bvh           gt_bvh;
    pamo_vec3d        *targets;
    pamo_vec3d        *rest_verts;
    pamo_elastic_rest *elastic_rest;
    double            *elastic_grad0;
    double            *elastic_grad1;
    pamo_safe_opts     opts;
    const pamo_allocator *alloc;
    int                profile;
    double             hess_vec_seconds;
    size_t             hess_vec_calls;
} pamo_safe_ctx;

static void safe_hess_vec(const double *dx, double *out,
                          size_t n, void *ctx_ptr) {
    pamo_safe_ctx *ctx = (pamo_safe_ctx *)ctx_ptr;
    double t0 = ctx->profile ? stage3_now_seconds() : 0.0;
    memset(out, 0, n * 3 * sizeof(double));

    /* Distance Hessian-vector product. */
    pamo_distance_hess_vec(ctx->mesh->verts, n, dx, out,
                           ctx->opts.dist_stiffness);

    /* Elastic Hessian-vector product (finite differences). */
    if (ctx->elastic_grad0 && ctx->elastic_grad1) {
        pamo_elastic_hess_vec_fd_with_baseline(ctx->mesh, ctx->mesh->verts,
                                               ctx->elastic_rest,
                                               ctx->opts.elas_young_modulus,
                                               ctx->opts.elas_poisson_ratio,
                                               ctx->elastic_grad0,
                                               ctx->elastic_grad1,
                                               dx, out);
    } else {
        pamo_elastic_hess_vec_fd(ctx->mesh, ctx->mesh->verts,
                                 ctx->elastic_rest,
                                 ctx->opts.elas_young_modulus,
                                 ctx->opts.elas_poisson_ratio,
                                 dx, out);
    }
    if (ctx->profile) {
        ctx->hess_vec_seconds += stage3_now_seconds() - t0;
        ctx->hess_vec_calls++;
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
    if (!mesh->verts || !mesh->faces || !mesh->vert_alive ||
        !mesh->face_alive) {
        return PAMO_ERR_INVALID_ARG;
    }
    if (mesh->n_verts == 0 || mesh->n_faces == 0)
        return PAMO_ERR_INVALID_ARG;
    for (size_t fi = 0; fi < mesh->n_faces; fi++) {
        if (mesh->face_alive[fi] && !pamo_mesh_face_is_valid(mesh, fi))
            return PAMO_ERR_INVALID_ARG;
    }

    size_t n = mesh->n_verts;
    if (n > SIZE_MAX / 3u ||
        n * 3u > SIZE_MAX / sizeof(double) ||
        mesh->n_faces > SIZE_MAX / sizeof(pamo_elastic_rest)) {
        return PAMO_ERR_INVALID_ARG;
    }
    size_t dim = n * 3;
    int profile = stage3_profile_enabled();
    double total_t0 = profile ? stage3_now_seconds() : 0.0;

    pamo_safe_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.mesh = mesh;
    ctx.gt_mesh = gt_mesh;
    ctx.opts = *opts;
    ctx.alloc = alloc;
    ctx.profile = profile;

    double t0 = profile ? stage3_now_seconds() : 0.0;
    pamo_error err = pamo_bvh_build_triangles(&ctx.gt_bvh, gt_mesh, alloc);
    if (err != PAMO_OK) return err;
    if (profile) stage3_profile_log("gt-bvh-build", stage3_now_seconds() - t0);

    /* Allocate. */
    ctx.targets      = PAMO_ALLOC_ARRAY(alloc, pamo_vec3d, n);
    ctx.rest_verts   = PAMO_ALLOC_ARRAY(alloc, pamo_vec3d, n);
    ctx.elastic_rest = (pamo_elastic_rest *)pamo_alloc(alloc,
                        mesh->n_faces * sizeof(pamo_elastic_rest));
    ctx.elastic_grad0 = (double *)pamo_alloc(alloc, dim * sizeof(double));
    ctx.elastic_grad1 = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *grad  = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *hdiag = (double *)pamo_alloc(alloc, dim * sizeof(double));
    double *p     = (double *)pamo_alloc(alloc, dim * sizeof(double));

    if (!ctx.targets || !ctx.rest_verts || !ctx.elastic_rest ||
        !ctx.elastic_grad0 || !ctx.elastic_grad1 || !grad || !hdiag || !p) {
        err = PAMO_ERR_ALLOC;
        goto cleanup;
    }

    /* Save rest positions and precompute elastic rest state. */
    memcpy(ctx.rest_verts, mesh->verts, n * sizeof(pamo_vec3d));
    pamo_elastic_precompute(mesh, ctx.rest_verts, ctx.elastic_rest);

    for (int32_t outer = 0; outer < opts->n_outer_iters; outer++) {
        double outer_t0 = profile ? stage3_now_seconds() : 0.0;

        /* Update closest-point targets. */
        t0 = profile ? stage3_now_seconds() : 0.0;
        err = pamo_distance_update_targets(mesh->verts, n, gt_mesh,
                                           &ctx.gt_bvh, ctx.targets);
        if (err != PAMO_OK) goto cleanup;
        if (profile) {
            char label[64];
            snprintf(label, sizeof(label), "outer %d target-update", outer + 1);
            stage3_profile_log(label, stage3_now_seconds() - t0);
        }

        for (int32_t newton = 0; newton < opts->n_newton_iters; newton++) {
            double newton_t0 = profile ? stage3_now_seconds() : 0.0;
            memset(grad, 0, dim * sizeof(double));
            for (size_t i = 0; i < dim; i++) hdiag[i] = 1e-8;

            /* Distance energy gradient. */
            t0 = profile ? stage3_now_seconds() : 0.0;
            pamo_distance_gradient(mesh->verts, n, ctx.targets,
                                   opts->dist_stiffness, grad, hdiag);

            /* Elastic gradient. Also cache it as the baseline used by the
             * finite-difference Hessian-vector products during this CG solve.
             */
            memset(ctx.elastic_grad0, 0, dim * sizeof(double));
            pamo_elastic_gradient_fd(mesh, mesh->verts, ctx.elastic_rest,
                                     opts->elas_young_modulus,
                                     opts->elas_poisson_ratio,
                                     ctx.elastic_grad0);
            for (size_t i = 0; i < dim; i++) {
                grad[i] += ctx.elastic_grad0[i];
            }
            pamo_elastic_hess_diag(mesh, mesh->verts, ctx.elastic_rest,
                                   opts->elas_young_modulus,
                                   opts->elas_poisson_ratio, hdiag);
            if (profile) {
                char label[80];
                snprintf(label, sizeof(label), "outer %d newton %d gradient",
                         outer + 1, newton + 1);
                stage3_profile_log(label, stage3_now_seconds() - t0);
            }

            /* CG solve: H * p = -grad */
            memset(p, 0, dim * sizeof(double));
            ctx.hess_vec_seconds = 0.0;
            ctx.hess_vec_calls = 0;
            t0 = profile ? stage3_now_seconds() : 0.0;
            err = pamo_cg_solve(safe_hess_vec, hdiag, grad, p, n,
                                opts->n_cg_iters, &ctx, alloc);
            if (err != PAMO_OK) goto cleanup;
            if (profile) {
                fprintf(stderr,
                        "[stage3-prof] outer %d newton %d cg          %.3f s"
                        " (hess_vec %.3f s, %zu calls)\n",
                        outer + 1, newton + 1,
                        stage3_now_seconds() - t0,
                        ctx.hess_vec_seconds,
                        ctx.hess_vec_calls);
            }

            /* Line search. */
            t0 = profile ? stage3_now_seconds() : 0.0;
            double gp = 0.0;
            for (size_t i = 0; i < dim; i++) gp += grad[i] * p[i];
            if (gp >= 0.0) {
                if (profile) {
                    fprintf(stderr,
                            "[stage3-prof] outer %d newton %d line-search %.3f s"
                            " (skipped)\n",
                            outer + 1, newton + 1,
                            stage3_now_seconds() - t0);
                    fprintf(stderr,
                            "[stage3-prof] outer %d newton %d total       %.3f s\n",
                            outer + 1, newton + 1,
                            stage3_now_seconds() - newton_t0);
                }
                continue;
            }

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
            if (profile) {
                fprintf(stderr,
                        "[stage3-prof] outer %d newton %d line-search %.3f s\n",
                        outer + 1, newton + 1,
                        stage3_now_seconds() - t0);
                fprintf(stderr,
                        "[stage3-prof] outer %d newton %d total       %.3f s\n",
                        outer + 1, newton + 1,
                        stage3_now_seconds() - newton_t0);
            }
        }

        double E = total_energy(&ctx);
        fprintf(stderr, "  SAFE outer %d: energy=%.6e\n", outer + 1, E);
        if (profile) {
            char label[64];
            snprintf(label, sizeof(label), "outer %d total", outer + 1);
            stage3_profile_log(label, stage3_now_seconds() - outer_t0);
        }
    }
    if (profile) stage3_profile_log("total", stage3_now_seconds() - total_t0);

cleanup:
    if (ctx.targets) PAMO_FREE_ARRAY(alloc, ctx.targets, pamo_vec3d, n);
    if (ctx.rest_verts) PAMO_FREE_ARRAY(alloc, ctx.rest_verts, pamo_vec3d, n);
    if (ctx.elastic_rest)
        pamo_free(alloc, ctx.elastic_rest,
                  mesh->n_faces * sizeof(pamo_elastic_rest));
    if (ctx.elastic_grad0) pamo_free(alloc, ctx.elastic_grad0,
                                     dim * sizeof(double));
    if (ctx.elastic_grad1) pamo_free(alloc, ctx.elastic_grad1,
                                     dim * sizeof(double));
    if (grad)  pamo_free(alloc, grad,  dim * sizeof(double));
    if (hdiag) pamo_free(alloc, hdiag, dim * sizeof(double));
    if (p)     pamo_free(alloc, p,     dim * sizeof(double));
    pamo_bvh_destroy(&ctx.gt_bvh);

    return err;
}
