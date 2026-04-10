/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hinge bending energy: penalizes deviation of dihedral angles
 * from rest-state angles at interior edges.
 */
#include "pamo/pamo_types.h"
#include "pamo/pamo_mesh.h"

#include <math.h>

/* Compute dihedral angle at an edge shared by two triangles.
 * Returns angle in radians [0, pi]. */
static double dihedral_angle(pamo_vec3d n1, pamo_vec3d n2) {
    double dot = pamo_v3_dot(n1, n2);
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;
    return acos(dot);
}

/* Per-edge hinge state. */
typedef struct {
    int32_t f0, f1;       /* two adjacent faces */
    double rest_angle;     /* rest dihedral angle */
    double rest_edge_len;  /* rest edge length */
} pamo_hinge_edge_state;

/* Precompute hinge state for all interior edges. */
pamo_error pamo_hinge_precompute(const pamo_mesh *m,
                                 const pamo_vec3d *rest_verts,
                                 pamo_hinge_edge_state *state,
                                 size_t *n_hinges) {
    *n_hinges = 0;

    if (!m->edge_face_offset || !m->edge_face_list) return PAMO_ERR_INVALID_ARG;

    for (size_t ei = 0; ei < m->n_edges; ei++) {
        int32_t start = m->edge_face_offset[ei];
        int32_t end = m->edge_face_offset[ei + 1];
        if (end - start != 2) continue; /* not interior edge */

        int32_t f0 = m->edge_face_list[start];
        int32_t f1 = m->edge_face_list[start + 1];
        if (!m->face_alive[f0] || !m->face_alive[f1]) continue;

        pamo_vec3d n0 = pamo_face_unit_normal(m, f0);
        pamo_vec3d n1 = pamo_face_unit_normal(m, f1);

        pamo_hinge_edge_state *s = &state[*n_hinges];
        s->f0 = f0;
        s->f1 = f1;
        s->rest_angle = dihedral_angle(n0, n1);

        pamo_vec3d eu = rest_verts[m->edges[ei].u];
        pamo_vec3d ev = rest_verts[m->edges[ei].v];
        s->rest_edge_len = sqrt(pamo_v3_length_sq(pamo_v3_sub(eu, ev)));

        (*n_hinges)++;
    }
    return PAMO_OK;
}

/* Compute hinge energy. */
double pamo_hinge_energy(const pamo_mesh *m,
                         const pamo_vec3d *q,
                         const pamo_hinge_edge_state *state,
                         size_t n_hinges,
                         double stiffness) {
    double E = 0.0;
    for (size_t i = 0; i < n_hinges; i++) {
        pamo_vec3d n0 = pamo_face_unit_normal(m, state[i].f0);
        pamo_vec3d n1 = pamo_face_unit_normal(m, state[i].f1);
        double angle = dihedral_angle(n0, n1);
        double diff = angle - state[i].rest_angle;
        E += stiffness * state[i].rest_edge_len * diff * diff;
    }
    (void)q;
    return E;
}
