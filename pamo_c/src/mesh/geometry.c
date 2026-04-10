/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "pamo/pamo_mesh.h"

#include <math.h>

/* ── Face geometry ───────────────────────────────────────────────── */

pamo_vec3d pamo_face_normal(const pamo_mesh *m, int32_t face_id) {
    PAMO_ASSERT(m && face_id >= 0 && (size_t)face_id < m->n_faces);
    const pamo_tri *f = &m->faces[face_id];
    pamo_vec3d v0 = m->verts[f->v[0]];
    pamo_vec3d v1 = m->verts[f->v[1]];
    pamo_vec3d v2 = m->verts[f->v[2]];
    return pamo_v3_cross(pamo_v3_sub(v1, v0), pamo_v3_sub(v2, v0));
}

double pamo_face_area(const pamo_mesh *m, int32_t face_id) {
    pamo_vec3d n = pamo_face_normal(m, face_id);
    return 0.5 * sqrt(pamo_v3_length_sq(n));
}

pamo_vec3d pamo_face_unit_normal(const pamo_mesh *m, int32_t face_id) {
    pamo_vec3d n = pamo_face_normal(m, face_id);
    double len = sqrt(pamo_v3_length_sq(n));
    if (len < 1e-30) return pamo_v3_zero();
    return pamo_v3_scale(n, 1.0 / len);
}

/* ── Closest point on triangle ───────────────────────────────────── */

pamo_vec3d pamo_closest_point_on_tri(pamo_vec3d p,
                                     pamo_vec3d v0, pamo_vec3d v1,
                                     pamo_vec3d v2,
                                     double *dist_sq) {
    pamo_vec3d ab = pamo_v3_sub(v1, v0);
    pamo_vec3d ac = pamo_v3_sub(v2, v0);
    pamo_vec3d ap = pamo_v3_sub(p, v0);

    double d1 = pamo_v3_dot(ab, ap);
    double d2 = pamo_v3_dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, v0));
        return v0;
    }

    pamo_vec3d bp = pamo_v3_sub(p, v1);
    double d3 = pamo_v3_dot(ab, bp);
    double d4 = pamo_v3_dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, v1));
        return v1;
    }

    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double w = d1 / (d1 - d3);
        pamo_vec3d r = pamo_v3_add(v0, pamo_v3_scale(ab, w));
        if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, r));
        return r;
    }

    pamo_vec3d cp = pamo_v3_sub(p, v2);
    double d5 = pamo_v3_dot(ab, cp);
    double d6 = pamo_v3_dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, v2));
        return v2;
    }

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double w = d2 / (d2 - d6);
        pamo_vec3d r = pamo_v3_add(v0, pamo_v3_scale(ac, w));
        if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, r));
        return r;
    }

    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        pamo_vec3d r = pamo_v3_add(v1, pamo_v3_scale(pamo_v3_sub(v2, v1), w));
        if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, r));
        return r;
    }

    double denom = 1.0 / (va + vb + vc);
    double sv = vb * denom;
    double tv = vc * denom;
    pamo_vec3d r = pamo_v3_add(v0, pamo_v3_add(pamo_v3_scale(ab, sv),
                                                pamo_v3_scale(ac, tv)));
    if (dist_sq) *dist_sq = pamo_v3_length_sq(pamo_v3_sub(p, r));
    return r;
}

/* ── Triangle quality ────────────────────────────────────────────── */

double pamo_triangle_quality(pamo_vec3d v0, pamo_vec3d v1, pamo_vec3d v2) {
    pamo_vec3d e0 = pamo_v3_sub(v1, v0);
    pamo_vec3d e1 = pamo_v3_sub(v2, v1);
    pamo_vec3d e2 = pamo_v3_sub(v0, v2);
    double sum_sq = pamo_v3_length_sq(e0) + pamo_v3_length_sq(e1)
                  + pamo_v3_length_sq(e2);
    if (sum_sq < 1e-30) return 0.0;
    pamo_vec3d cross = pamo_v3_cross(e0, pamo_v3_sub(v2, v0));
    double area = 0.5 * sqrt(pamo_v3_length_sq(cross));
    /* 4*sqrt(3)*area / sum_sq.  For equilateral: ratio = 1. */
    return 4.0 * 1.7320508075688772 * area / sum_sq;
}

bool pamo_triangle_is_skinny(pamo_vec3d v0, pamo_vec3d v1, pamo_vec3d v2,
                             double threshold) {
    pamo_vec3d e0 = pamo_v3_sub(v1, v0);
    pamo_vec3d e1 = pamo_v3_sub(v2, v1);
    pamo_vec3d e2 = pamo_v3_sub(v0, v2);
    double max_sq = pamo_v3_length_sq(e0);
    double tmp = pamo_v3_length_sq(e1);
    if (tmp > max_sq) max_sq = tmp;
    tmp = pamo_v3_length_sq(e2);
    if (tmp > max_sq) max_sq = tmp;
    pamo_vec3d cross = pamo_v3_cross(e0, pamo_v3_sub(v2, v0));
    double area = 0.5 * sqrt(pamo_v3_length_sq(cross));
    return area < threshold * max_sq;
}
