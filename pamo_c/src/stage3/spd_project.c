/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * 9x9 SPD (symmetric positive semi-definite) matrix projection
 * via Jacobi eigenvalue iteration.
 *
 * Algorithm: eigendecompose, clamp negative eigenvalues to 0, reconstruct.
 * Based on Burkardt's Jacobi eigenvalue implementation (GNU LGPL).
 */
#include "pamo/pamo_types.h"

#include <math.h>
#include <string.h>

void pamo_spd_project_9x9(double a[9][9], int max_iters) {
    double d[9], v[9][9], bw[9], zw[9];

    /* Initialize V = identity. */
    memset(v, 0, sizeof(v));
    for (int i = 0; i < 9; i++) v[i][i] = 1.0;

    /* D = diagonal of A. */
    for (int i = 0; i < 9; i++) {
        d[i] = a[i][i];
        bw[i] = d[i];
        zw[i] = 0.0;
    }

    for (int it = 0; it < max_iters; it++) {
        /* Convergence check: upper triangle norm. */
        double thresh = 0.0;
        for (int j = 0; j < 9; j++)
            for (int i = 0; i < j; i++)
                thresh += a[i][j] * a[i][j];
        thresh = sqrt(thresh) / 36.0;
        if (thresh == 0.0) break;

        for (int p = 0; p < 9; p++) {
            for (int q = p + 1; q < 9; q++) {
                double gapq = 10.0 * fabs(a[p][q]);
                double termp = gapq + fabs(d[p]);
                double termq = gapq + fabs(d[q]);

                if (it > 3 && termp == fabs(d[p]) && termq == fabs(d[q])) {
                    a[p][q] = 0.0;
                } else if (thresh <= fabs(a[p][q])) {
                    double h = d[q] - d[p];
                    double term = fabs(h) + gapq;
                    double t;
                    if (term == fabs(h)) {
                        t = a[p][q] / h;
                    } else {
                        double theta = 0.5 * h / a[p][q];
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }

                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    h = t * a[p][q];

                    zw[p] -= h;
                    zw[q] += h;
                    d[p] -= h;
                    d[q] += h;
                    a[p][q] = 0.0;

                    for (int j = 0; j < p; j++) {
                        double g = a[j][p], hh = a[j][q];
                        a[j][p] = g - s * (hh + g * tau);
                        a[j][q] = hh + s * (g - hh * tau);
                    }
                    for (int j = p + 1; j < q; j++) {
                        double g = a[p][j], hh = a[j][q];
                        a[p][j] = g - s * (hh + g * tau);
                        a[j][q] = hh + s * (g - hh * tau);
                    }
                    for (int j = q + 1; j < 9; j++) {
                        double g = a[p][j], hh = a[q][j];
                        a[p][j] = g - s * (hh + g * tau);
                        a[q][j] = hh + s * (g - hh * tau);
                    }
                    for (int j = 0; j < 9; j++) {
                        double g = v[j][p], hh = v[j][q];
                        v[j][p] = g - s * (hh + g * tau);
                        v[j][q] = hh + s * (g - hh * tau);
                    }
                }
            }
        }

        for (int i = 0; i < 9; i++) {
            bw[i] += zw[i];
            d[i] = bw[i];
            zw[i] = 0.0;
        }
    }

    /* Reconstruct A = V * max(D, 0) * V^T */
    memset(a, 0, 9 * 9 * sizeof(double));
    for (int k = 0; k < 9; k++) {
        if (d[k] <= 0.0) continue;
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                a[i][j] += d[k] * v[i][k] * v[j][k];
    }

    /* Enforce symmetry. */
    for (int i = 0; i < 9; i++)
        for (int j = i + 1; j < 9; j++) {
            double avg = (a[i][j] + a[j][i]) * 0.5;
            a[i][j] = avg;
            a[j][i] = avg;
        }
}
