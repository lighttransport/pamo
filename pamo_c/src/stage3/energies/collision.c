/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Collision barrier energy via BVH-based point-triangle contact detection.
 *
 * For each vertex that is within d_hat distance of a non-adjacent triangle,
 * add a log-barrier energy: kappa * area * b(d, d_hat)
 * where b(d, d_hat) = -(d - d_hat)^2 * ln(d / d_hat).
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"

#include <math.h>
#include <string.h>

/* ── Barrier functions ───────────────────────────────────────────── */

double pamo_barrier(double d, double d_hat) {
    if (d >= d_hat || d <= 0.0) return 0.0;
    double t = d - d_hat;
    return -t * t * log(d / d_hat);
}

double pamo_barrier_grad(double d, double d_hat) {
    if (d >= d_hat || d <= 0.0) return 0.0;
    double t = d - d_hat;
    return -2.0 * t * log(d / d_hat) - t * t / d;
}

double pamo_barrier_hess(double d, double d_hat) {
    if (d >= d_hat || d <= 0.0) return 0.0;
    return -2.0 * log(d / d_hat) - 2.0 * (d - d_hat) / d
           - 1.0 + (d_hat * d_hat) / (d * d);
}

/* ── Contact detection ───────────────────────────────────────────── */

typedef struct {
    int32_t vert_id;
    int32_t face_id;
    double  dist;
    pamo_vec3d closest;
} pamo_contact;

/* Detect point-triangle contacts within d_hat distance.
 * Returns number of contacts found. */
size_t pamo_detect_pt_contacts(const pamo_mesh *m, const pamo_bvh *bvh,
                               double d_hat,
                               pamo_contact *contacts, size_t max_contacts) {
    size_t n = 0;

    for (size_t vi = 0; vi < m->n_verts && n < max_contacts; vi++) {
        if (!m->vert_alive[vi]) continue;
        pamo_vec3d p = m->verts[vi];

        /* Find nearest triangle. */
        pamo_nearest_result res;
        pamo_bvh_nearest(bvh, m, p, &res);
        if (res.prim_id < 0) continue;

        double dist = sqrt(res.dist_sq);
        if (dist >= d_hat || dist < 1e-15) continue;

        /* Check if vertex is part of this face (skip self-contact). */
        const int32_t *fv = m->faces[res.prim_id].v;
        if (fv[0] == (int32_t)vi || fv[1] == (int32_t)vi ||
            fv[2] == (int32_t)vi) continue;

        contacts[n].vert_id = (int32_t)vi;
        contacts[n].face_id = res.prim_id;
        contacts[n].dist    = dist;
        contacts[n].closest = res.point;
        n++;
    }
    return n;
}

/* ── Collision energy ────────────────────────────────────────────── */

double pamo_collision_energy(const pamo_mesh *m, const pamo_vec3d *q,
                             double d_hat, double stiffness) {
    (void)q;
    /* Build BVH for self-collision detection. */
    pamo_allocator alloc = pamo_default_allocator();
    pamo_bvh bvh;
    if (pamo_bvh_build_triangles(&bvh, m, &alloc) != PAMO_OK) return 0.0;

    size_t max_contacts = m->n_verts;
    pamo_contact *contacts = (pamo_contact *)pamo_alloc(&alloc,
        max_contacts * sizeof(pamo_contact));
    if (!contacts) { pamo_bvh_destroy(&bvh); return 0.0; }

    size_t nc = pamo_detect_pt_contacts(m, &bvh, d_hat,
                                        contacts, max_contacts);

    double E = 0.0;
    for (size_t i = 0; i < nc; i++) {
        E += stiffness * pamo_barrier(contacts[i].dist, d_hat);
    }

    pamo_free(&alloc, contacts, max_contacts * sizeof(pamo_contact));
    pamo_bvh_destroy(&bvh);
    return E;
}

/* Collision gradient: for each contact, push vertex away from triangle. */
void pamo_collision_gradient(const pamo_mesh *m, const pamo_vec3d *q,
                             double d_hat, double stiffness,
                             double *grad, double *hess_diag) {
    (void)q;
    pamo_allocator alloc = pamo_default_allocator();
    pamo_bvh bvh;
    if (pamo_bvh_build_triangles(&bvh, m, &alloc) != PAMO_OK) return;

    size_t max_contacts = m->n_verts;
    pamo_contact *contacts = (pamo_contact *)pamo_alloc(&alloc,
        max_contacts * sizeof(pamo_contact));
    if (!contacts) { pamo_bvh_destroy(&bvh); return; }

    size_t nc = pamo_detect_pt_contacts(m, &bvh, d_hat,
                                        contacts, max_contacts);

    for (size_t i = 0; i < nc; i++) {
        int32_t vi = contacts[i].vert_id;
        double d = contacts[i].dist;
        if (d < 1e-15) continue;

        /* Gradient of distance w.r.t. vertex position:
         * d(p) = ||p - closest||, grad_p(d) = (p - closest) / d */
        pamo_vec3d diff = pamo_v3_sub(m->verts[vi], contacts[i].closest);
        pamo_vec3d grad_d = pamo_v3_scale(diff, 1.0 / d);

        double db = stiffness * pamo_barrier_grad(d, d_hat);
        grad[vi * 3 + 0] += db * grad_d.x;
        grad[vi * 3 + 1] += db * grad_d.y;
        grad[vi * 3 + 2] += db * grad_d.z;

        double d2b = stiffness * pamo_barrier_hess(d, d_hat);
        double hd = fabs(d2b) + fabs(db) / d;
        hess_diag[vi * 3 + 0] += hd;
        hess_diag[vi * 3 + 1] += hd;
        hess_diag[vi * 3 + 2] += hd;
    }

    pamo_free(&alloc, contacts, max_contacts * sizeof(pamo_contact));
    pamo_bvh_destroy(&bvh);
}
