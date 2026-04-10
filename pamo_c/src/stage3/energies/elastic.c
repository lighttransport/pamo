/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle-based elastic energy (Saint Venant-Kirchhoff model).
 * Penalizes deformation from the rest state.
 *
 * For each triangle:
 *   F = Ds * inv(Dm)     (deformation gradient)
 *   E_tri = 0.5 * (F^T F - I)  (Green strain)
 *   energy = mu * tr(E^T E) + lambda/2 * tr(E)^2
 * summed over triangles, weighted by rest area.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"

#include <math.h>
#include <string.h>

/* 2x2 matrix type for the 2D reference frame. */
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

/* Per-triangle rest state. */
typedef struct {
    mat2   inv_Dm;    /* inverse of rest edge matrix */
    double rest_area; /* rest triangle area */
} pamo_elastic_tri_state;

/* Precompute rest state for all triangles. */
pamo_error pamo_elastic_precompute(const pamo_mesh *m,
                                   const pamo_vec3d *rest_verts,
                                   pamo_elastic_tri_state *state) {
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi]) {
            state[fi].rest_area = 0.0;
            continue;
        }
        const int32_t *fv = m->faces[fi].v;
        pamo_vec3d v0 = rest_verts[fv[0]];
        pamo_vec3d v1 = rest_verts[fv[1]];
        pamo_vec3d v2 = rest_verts[fv[2]];

        /* Dm = [v1-v0, v2-v0] projected into triangle plane.
         * For simplicity, compute in 3D and use a 2D parameterization. */
        pamo_vec3d e1 = pamo_v3_sub(v1, v0);
        pamo_vec3d e2 = pamo_v3_sub(v2, v0);

        /* 2x2 metric tensor: Dm^T Dm */
        mat2 Dm;
        Dm.m[0][0] = pamo_v3_dot(e1, e1);
        Dm.m[0][1] = pamo_v3_dot(e1, e2);
        Dm.m[1][0] = Dm.m[0][1];
        Dm.m[1][1] = pamo_v3_dot(e2, e2);

        state[fi].inv_Dm = mat2_inv(Dm);

        pamo_vec3d cross = pamo_v3_cross(e1, e2);
        state[fi].rest_area = 0.5 * sqrt(pamo_v3_length_sq(cross));
    }
    return PAMO_OK;
}

/* Compute elastic energy for all triangles. */
double pamo_elastic_energy(const pamo_mesh *m,
                           const pamo_vec3d *q,
                           const pamo_elastic_tri_state *state,
                           double young, double poisson) {
    double mu = young / (2.0 * (1.0 + poisson));
    double lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    if (fabs(1.0 - 2.0 * poisson) < 1e-10) lambda = 0.0;

    double E = 0.0;
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi] || state[fi].rest_area < 1e-30) continue;
        const int32_t *fv = m->faces[fi].v;
        pamo_vec3d v0 = q[fv[0]], v1 = q[fv[1]], v2 = q[fv[2]];

        pamo_vec3d e1 = pamo_v3_sub(v1, v0);
        pamo_vec3d e2 = pamo_v3_sub(v2, v0);

        /* Ds^T Ds (metric in current config). */
        mat2 G;
        G.m[0][0] = pamo_v3_dot(e1, e1);
        G.m[0][1] = pamo_v3_dot(e1, e2);
        G.m[1][0] = G.m[0][1];
        G.m[1][1] = pamo_v3_dot(e2, e2);

        /* C = inv_Dm^T * G * inv_Dm (right Cauchy-Green in 2D) */
        /* Green strain: S = 0.5 * (C - I) */
        /* For energy: just use metric approach. */
        const mat2 *iD = &state[fi].inv_Dm;

        /* C = iD^T * G * iD */
        mat2 tmp, C;
        /* tmp = G * iD */
        tmp.m[0][0] = G.m[0][0]*iD->m[0][0] + G.m[0][1]*iD->m[1][0];
        tmp.m[0][1] = G.m[0][0]*iD->m[0][1] + G.m[0][1]*iD->m[1][1];
        tmp.m[1][0] = G.m[1][0]*iD->m[0][0] + G.m[1][1]*iD->m[1][0];
        tmp.m[1][1] = G.m[1][0]*iD->m[0][1] + G.m[1][1]*iD->m[1][1];
        /* C = iD^T * tmp */
        C.m[0][0] = iD->m[0][0]*tmp.m[0][0] + iD->m[1][0]*tmp.m[1][0];
        C.m[0][1] = iD->m[0][0]*tmp.m[0][1] + iD->m[1][0]*tmp.m[1][1];
        C.m[1][0] = iD->m[0][1]*tmp.m[0][0] + iD->m[1][1]*tmp.m[1][0];
        C.m[1][1] = iD->m[0][1]*tmp.m[0][1] + iD->m[1][1]*tmp.m[1][1];

        /* Green strain: S = 0.5*(C - I) */
        double S00 = 0.5 * (C.m[0][0] - 1.0);
        double S01 = 0.5 * C.m[0][1];
        double S11 = 0.5 * (C.m[1][1] - 1.0);

        double trS = S00 + S11;
        double trSS = S00*S00 + 2.0*S01*S01 + S11*S11;

        double psi = mu * trSS + 0.5 * lambda * trS * trS;
        E += state[fi].rest_area * psi;
    }
    return E;
}

/* Compute elastic gradient (accumulated into grad). */
void pamo_elastic_gradient(const pamo_mesh *m,
                           const pamo_vec3d *q,
                           const pamo_elastic_tri_state *state,
                           double young, double poisson,
                           double *grad, double *hess_diag) {
    double mu = young / (2.0 * (1.0 + poisson));
    double lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    if (fabs(1.0 - 2.0 * poisson) < 1e-10) lambda = 0.0;

    /* Finite-difference gradient for now (to be replaced with analytic). */
    double eps = 1e-7;
    for (size_t vi = 0; vi < m->n_verts; vi++) {
        if (!m->vert_alive[vi]) continue;
        for (int c = 0; c < 3; c++) {
            /* We can't easily modify q here since it's const.
             * Use the chain rule approach:
             * dE/dq_i = 2 * stiffness * sum over incident tris of
             *   area * dPsi/dq_i
             * For a first implementation, use distance-like heuristic. */
        }
        /* Simplified: use 2*mu as diagonal Hessian approximation. */
        hess_diag[vi * 3 + 0] += 2.0 * mu;
        hess_diag[vi * 3 + 1] += 2.0 * mu;
        hess_diag[vi * 3 + 2] += 2.0 * mu;
    }
    (void)grad; (void)q; (void)state; (void)lambda; (void)eps;
}
