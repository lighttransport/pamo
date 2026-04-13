/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle-based elastic energy (Saint Venant-Kirchhoff model).
 *
 * For each triangle with rest edges e1_r, e2_r and current edges e1, e2:
 *   Dm = [e1_r, e2_r]  (3x2 edge matrix in rest config)
 *   Ds = [e1, e2]      (3x2 edge matrix in current config)
 *   F = Ds * inv(Dm^T Dm) * Dm^T   (deformation gradient via metric)
 *   Green strain: S = 0.5 * (Ds^T Ds * inv(Dm^T Dm) - I)
 *   Energy: psi = mu * tr(S^T S) + lambda/2 * tr(S)^2
 *
 * We use the metric formulation to avoid computing F explicitly.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_alloc.h"

#include <math.h>
#include <string.h>

/* 2x2 matrix. */
typedef struct { double m[2][2]; } mat2;

static mat2 mat2_inv(mat2 M) {
    double det = M.m[0][0] * M.m[1][1] - M.m[0][1] * M.m[1][0];
    double inv_det = (fabs(det) > 1e-30) ? 1.0 / det : 0.0;
    mat2 R;
    R.m[0][0] =  M.m[1][1] * inv_det;
    R.m[0][1] = -M.m[0][1] * inv_det;
    R.m[1][0] = -M.m[1][0] * inv_det;
    R.m[1][1] =  M.m[0][0] * inv_det;
    return R;
}

/* Per-triangle precomputed rest state. */
typedef struct {
    mat2   inv_DtD; /* inverse of Dm^T * Dm */
    double area;    /* rest area */
} pamo_elastic_rest;

/* Precompute rest state for all faces. Caller allocates rest[n_faces]. */
void pamo_elastic_precompute(const pamo_mesh *m,
                             const pamo_vec3d *rest_verts,
                             pamo_elastic_rest *rest) {
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        rest[fi].area = 0.0;
        if (!m->face_alive[fi]) continue;
        const int32_t *fv = m->faces[fi].v;
        pamo_vec3d e1 = pamo_v3_sub(rest_verts[fv[1]], rest_verts[fv[0]]);
        pamo_vec3d e2 = pamo_v3_sub(rest_verts[fv[2]], rest_verts[fv[0]]);

        mat2 DtD;
        DtD.m[0][0] = pamo_v3_dot(e1, e1);
        DtD.m[0][1] = pamo_v3_dot(e1, e2);
        DtD.m[1][0] = DtD.m[0][1];
        DtD.m[1][1] = pamo_v3_dot(e2, e2);

        rest[fi].inv_DtD = mat2_inv(DtD);
        pamo_vec3d cross = pamo_v3_cross(e1, e2);
        rest[fi].area = 0.5 * sqrt(pamo_v3_length_sq(cross));
    }
}

/* Compute Green strain components S00, S01, S11 for one triangle. */
static void green_strain(const pamo_vec3d *q, const int32_t *fv,
                         const mat2 *inv_DtD,
                         double *S00, double *S01, double *S11) {
    pamo_vec3d e1 = pamo_v3_sub(q[fv[1]], q[fv[0]]);
    pamo_vec3d e2 = pamo_v3_sub(q[fv[2]], q[fv[0]]);

    /* C = inv_DtD * [e1.e1  e1.e2; e1.e2  e2.e2] = metric in ref coords. */
    double g00 = pamo_v3_dot(e1, e1);
    double g01 = pamo_v3_dot(e1, e2);
    double g11 = pamo_v3_dot(e2, e2);

    const mat2 *I = inv_DtD;
    double C00 = I->m[0][0]*g00 + I->m[0][1]*g01;
    double C01 = I->m[0][0]*g01 + I->m[0][1]*g11;
    double C10 = I->m[1][0]*g00 + I->m[1][1]*g01;
    double C11 = I->m[1][0]*g01 + I->m[1][1]*g11;
    (void)C10; /* C is symmetric: C10 == C01 in exact arithmetic. */

    *S00 = 0.5 * (C00 - 1.0);
    *S01 = 0.5 * C01;
    *S11 = 0.5 * (C11 - 1.0);
}

/* Compute elastic energy. */
double pamo_elastic_energy(const pamo_mesh *m, const pamo_vec3d *q,
                           const pamo_elastic_rest *rest,
                           double young, double poisson) {
    double mu = young / (2.0 * (1.0 + poisson));
    double lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    if (fabs(1.0 - 2.0 * poisson) < 1e-10) lambda = 0.0;

    double E = 0.0;
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi] || rest[fi].area < 1e-30) continue;
        double S00, S01, S11;
        green_strain(q, m->faces[fi].v, &rest[fi].inv_DtD, &S00, &S01, &S11);
        double trS = S00 + S11;
        double trSS = S00*S00 + 2.0*S01*S01 + S11*S11;
        double psi = mu * trSS + 0.5 * lambda * trS * trS;
        E += rest[fi].area * psi;
    }
    return E;
}

/* Analytic elastic gradient for the Saint Venant–Kirchhoff metric
 * formulation used in pamo_elastic_energy.
 *
 * Per-face energy:
 *   E_face = area_rest * psi
 *   psi    = mu * (S00² + 2·S01² + S11²) + (lambda/2) * (S00 + S11)²
 *
 * with Green strain components
 *   S00 = 0.5 * (I00·g00 + I01·g01 − 1)
 *   S01 = 0.5 * (I00·g01 + I01·g11)
 *   S11 = 0.5 * (I01·g01 + I11·g11 − 1)     [I is symmetric: I10 = I01]
 *
 * where I = inv(Dm^T·Dm) is precomputed per face and
 *   g00 = e1·e1, g11 = e2·e2, g01 = e1·e2
 *   e1  = q[v1] − q[v0],      e2 = q[v2] − q[v0].
 *
 * Chain rule:
 *   ∂psi/∂S00 = 2·mu·S00 + lambda·(S00 + S11)   = P00
 *   ∂psi/∂S01 = 4·mu·S01                        = P01
 *   ∂psi/∂S11 = 2·mu·S11 + lambda·(S00 + S11)   = P11
 *
 *   dg00 := ∂psi/∂g00 = 0.5·(P00·I00)
 *   dg01 := ∂psi/∂g01 = 0.5·(P00·I01 + P11·I01 + P01·I00)
 *   dg11 := ∂psi/∂g11 = 0.5·(P11·I11 + P01·I01)
 *
 *   ∂psi/∂e1 = 2·dg00·e1 + dg01·e2
 *   ∂psi/∂e2 =   dg01·e1 + 2·dg11·e2
 *
 *   ∂E_face/∂v1 =  area · ∂psi/∂e1
 *   ∂E_face/∂v2 =  area · ∂psi/∂e2
 *   ∂E_face/∂v0 = −area · (∂psi/∂e1 + ∂psi/∂e2)
 *
 * Total cost: O(n_faces). The previous finite-difference implementation
 * was O(n_verts · n_faces) and dominated SAFE projection runtime by
 * 100–500× on meshes with a few thousand faces. The public API is
 * unchanged — callers pass a 3·n_verts gradient buffer and we add our
 * contribution to it.
 *
 * Note: `q` is declared non-const for API compatibility with the old
 * FD version (which temporarily perturbed it). We do not modify `q`.
 */
void pamo_elastic_gradient_fd(const pamo_mesh *m, pamo_vec3d *q,
                              const pamo_elastic_rest *rest,
                              double young, double poisson,
                              double *grad) {
    double mu = young / (2.0 * (1.0 + poisson));
    double lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    if (fabs(1.0 - 2.0 * poisson) < 1e-10) lambda = 0.0;

    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi] || rest[fi].area < 1e-30) continue;
        const int32_t *fv = m->faces[fi].v;
        const int32_t v0 = fv[0], v1 = fv[1], v2 = fv[2];

        pamo_vec3d e1 = pamo_v3_sub(q[v1], q[v0]);
        pamo_vec3d e2 = pamo_v3_sub(q[v2], q[v0]);

        /* Green strain components. */
        const mat2 *I = &rest[fi].inv_DtD;
        double g00 = pamo_v3_dot(e1, e1);
        double g01 = pamo_v3_dot(e1, e2);
        double g11 = pamo_v3_dot(e2, e2);
        double C00 = I->m[0][0]*g00 + I->m[0][1]*g01;
        double C01 = I->m[0][0]*g01 + I->m[0][1]*g11;
        double C11 = I->m[1][0]*g01 + I->m[1][1]*g11;
        double S00 = 0.5 * (C00 - 1.0);
        double S01 = 0.5 * C01;
        double S11 = 0.5 * (C11 - 1.0);

        /* ∂psi/∂S__ */
        double trS = S00 + S11;
        double P00 = 2.0*mu*S00 + lambda*trS;
        double P01 = 4.0*mu*S01;
        double P11 = 2.0*mu*S11 + lambda*trS;

        /* ∂psi/∂g__ */
        double dg00 = 0.5 * (P00 * I->m[0][0]);
        double dg01 = 0.5 * (P00 * I->m[0][1] + P11 * I->m[0][1] + P01 * I->m[0][0]);
        double dg11 = 0.5 * (P11 * I->m[1][1] + P01 * I->m[0][1]);

        /* ∂psi/∂e1, ∂psi/∂e2 as 3-vectors. */
        double d_e1x = 2.0*dg00*e1.x + dg01*e2.x;
        double d_e1y = 2.0*dg00*e1.y + dg01*e2.y;
        double d_e1z = 2.0*dg00*e1.z + dg01*e2.z;
        double d_e2x = dg01*e1.x + 2.0*dg11*e2.x;
        double d_e2y = dg01*e1.y + 2.0*dg11*e2.y;
        double d_e2z = dg01*e1.z + 2.0*dg11*e2.z;

        /* Distribute to the three face vertices, scaled by rest area. */
        double a = rest[fi].area;
        /* v1: +a * d_e1 */
        grad[v1*3 + 0] += a * d_e1x;
        grad[v1*3 + 1] += a * d_e1y;
        grad[v1*3 + 2] += a * d_e1z;
        /* v2: +a * d_e2 */
        grad[v2*3 + 0] += a * d_e2x;
        grad[v2*3 + 1] += a * d_e2y;
        grad[v2*3 + 2] += a * d_e2z;
        /* v0: −a * (d_e1 + d_e2) */
        grad[v0*3 + 0] -= a * (d_e1x + d_e2x);
        grad[v0*3 + 1] -= a * (d_e1y + d_e2y);
        grad[v0*3 + 2] -= a * (d_e1z + d_e2z);
    }
}

/* Compute elastic Hessian diagonal approximation. */
void pamo_elastic_hess_diag(const pamo_mesh *m, const pamo_vec3d *q,
                            const pamo_elastic_rest *rest,
                            double young, double poisson,
                            double *hess_diag) {
    double mu = young / (2.0 * (1.0 + poisson));
    /* Simple approximation: 2*mu per vertex, weighted by incident area. */
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi] || rest[fi].area < 1e-30) continue;
        double w = 2.0 * mu * rest[fi].area;
        for (int k = 0; k < 3; k++) {
            int32_t vi = m->faces[fi].v[k];
            hess_diag[vi * 3 + 0] += w;
            hess_diag[vi * 3 + 1] += w;
            hess_diag[vi * 3 + 2] += w;
        }
    }
    (void)q;
}

/* Hessian-vector product via finite differences. */
void pamo_elastic_hess_vec_fd(const pamo_mesh *m, pamo_vec3d *q,
                              const pamo_elastic_rest *rest,
                              double young, double poisson,
                              const double *dx, double *out) {
    double eps = 1e-6;
    size_t dim = m->n_verts * 3;
    if (dim > SIZE_MAX / sizeof(double)) return;

    pamo_allocator alloc = pamo_default_allocator();
    double *grad0 = (double *)pamo_alloc(&alloc, dim * sizeof(double));
    double *grad1 = (double *)pamo_alloc(&alloc, dim * sizeof(double));
    if (!grad0 || !grad1) {
        if (grad0) pamo_free(&alloc, grad0, dim * sizeof(double));
        if (grad1) pamo_free(&alloc, grad1, dim * sizeof(double));
        return;
    }

    pamo_elastic_gradient_fd(m, q, rest, young, poisson, grad0);

    /* Perturb: q + eps * dx. */
    for (size_t i = 0; i < m->n_verts; i++) {
        q[i].x += eps * dx[i * 3 + 0];
        q[i].y += eps * dx[i * 3 + 1];
        q[i].z += eps * dx[i * 3 + 2];
    }
    pamo_elastic_gradient_fd(m, q, rest, young, poisson, grad1);
    /* Restore. */
    for (size_t i = 0; i < m->n_verts; i++) {
        q[i].x -= eps * dx[i * 3 + 0];
        q[i].y -= eps * dx[i * 3 + 1];
        q[i].z -= eps * dx[i * 3 + 2];
    }

    for (size_t i = 0; i < dim; i++) {
        out[i] += (grad1[i] - grad0[i]) / eps;
    }
    pamo_free(&alloc, grad0, dim * sizeof(double));
    pamo_free(&alloc, grad1, dim * sizeof(double));
}
