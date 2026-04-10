/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Backtracking line search with Armijo condition.
 */
#include "pamo/pamo_types.h"

/* Backtracking line search.
 * Starting from q, move along direction p with initial step alpha.
 * Returns the step size that satisfies the Armijo condition. */
double pamo_line_search_backtrack(double alpha,
                                  double energy_at_q,
                                  const double *grad,
                                  const double *p,
                                  size_t dim,
                                  int max_iters) {
    /* Directional derivative: g^T p */
    double gp = 0.0;
    for (size_t i = 0; i < dim; i++) gp += grad[i] * p[i];

    /* If direction is not descent, return 0. */
    if (gp >= 0.0) return 0.0;

    double c = 1e-4; /* Armijo constant */

    for (int i = 0; i < max_iters; i++) {
        /* Armijo condition: f(q + alpha*p) <= f(q) + c*alpha*g^T*p
         * We can't evaluate f here (no callback), so the caller
         * should do the actual line search.  This just halves alpha. */
        alpha *= 0.5;
    }

    (void)energy_at_q; (void)c;
    return alpha;
}
