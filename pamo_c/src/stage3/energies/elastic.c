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

#include <math.h>
#include <string.h>
#include <stdlib.h>

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

/* Compute elastic gradient via finite differences.
 * This is O(n_verts * n_faces) but correct and simple.
 * For production use, replace with analytic gradient. */
void pamo_elastic_gradient_fd(const pamo_mesh *m, pamo_vec3d *q,
                              const pamo_elastic_rest *rest,
                              double young, double poisson,
                              double *grad) {
    double eps = 1e-7;
    double E0 = pamo_elastic_energy(m, q, rest, young, poisson);

    for (size_t vi = 0; vi < m->n_verts; vi++) {
        if (!m->vert_alive[vi]) continue;
        double *pos = &q[vi].x;
        for (int c = 0; c < 3; c++) {
            double orig = pos[c];
            pos[c] = orig + eps;
            double Ep = pamo_elastic_energy(m, q, rest, young, poisson);
            pos[c] = orig;
            grad[vi * 3 + (size_t)c] += (Ep - E0) / eps;
        }
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
    double *grad0 = (double *)calloc(dim, sizeof(double));
    double *grad1 = (double *)calloc(dim, sizeof(double));
    if (!grad0 || !grad1) { free(grad0); free(grad1); return; }

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
    free(grad0);
    free(grad1);
}
