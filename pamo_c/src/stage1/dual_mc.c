/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dual Marching Cubes: extract a triangle mesh from an SDF grid.
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_alloc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SDF_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

#define CELL_IDX(ix, iy, iz, C) \
    ((size_t)(iz) * (size_t)(C) * (size_t)(C) + \
     (size_t)(iy) * (size_t)(C) + (size_t)(ix))

/* ── Edge usage tracking to prevent non-manifold edges ───────────── */

typedef struct {
    int32_t u, v;
    int16_t count;
} dmc_edge_slot;

typedef struct {
    dmc_edge_slot *slots;
    size_t capacity;
    const pamo_allocator *alloc;  /* used for free in emap_destroy */
} dmc_edge_map;

static void emap_init(dmc_edge_map *em, size_t cap,
                      const pamo_allocator *alloc) {
    em->capacity = cap;
    em->alloc = alloc;
    em->slots = NULL;
    if (cap == 0 || cap > SIZE_MAX / sizeof(dmc_edge_slot)) return;
    /* Route through the mesh allocator so tracking allocators (leak
     * detection, budget tracking) see this buffer. */
    em->slots = (dmc_edge_slot *)pamo_alloc(
        alloc, cap * sizeof(dmc_edge_slot));
    if (em->slots) {
        for (size_t i = 0; i < cap; i++) em->slots[i].u = -1;
    }
}

static void emap_destroy(dmc_edge_map *em) {
    if (em->slots && em->alloc) {
        pamo_free(em->alloc, em->slots,
                  em->capacity * sizeof(dmc_edge_slot));
    }
    em->slots = NULL;
    em->capacity = 0;
}

static size_t emap_hash(int32_t u, int32_t v, size_t cap) {
    if (cap == 0) return 0;
    uint32_t h = (uint32_t)u * 2654435761u ^ (uint32_t)v * 40503u;
    return h % cap;
}

/* Returns current count for edge (u,v), or 0 if not found. */
static int emap_get(const dmc_edge_map *em, int32_t u, int32_t v) {
    if (!em || !em->slots || em->capacity == 0) return 0;
    if (u > v) { int32_t t = u; u = v; v = t; }
    size_t h = emap_hash(u, v, em->capacity);
    for (size_t probe = 0; probe < 64; probe++) {
        size_t idx = (h + probe) % em->capacity;
        if (em->slots[idx].u < 0) return 0;
        if (em->slots[idx].u == u && em->slots[idx].v == v)
            return em->slots[idx].count;
    }
    return 0;
}

/* Increment count for edge (u,v). */
static void emap_add(dmc_edge_map *em, int32_t u, int32_t v) {
    if (!em || !em->slots || em->capacity == 0) return;
    if (u > v) { int32_t t = u; u = v; v = t; }
    size_t h = emap_hash(u, v, em->capacity);
    for (size_t probe = 0; probe < 64; probe++) {
        size_t idx = (h + probe) % em->capacity;
        if (em->slots[idx].u < 0) {
            em->slots[idx].u = u;
            em->slots[idx].v = v;
            em->slots[idx].count = 1;
            return;
        }
        if (em->slots[idx].u == u && em->slots[idx].v == v) {
            em->slots[idx].count++;
            return;
        }
    }
}

/* Check if a triangle (a,b,c) can be added without creating >2-face edges. */
static int tri_can_add(const dmc_edge_map *em, int32_t a, int32_t b, int32_t c) {
    return emap_get(em, a, b) < 2 &&
           emap_get(em, b, c) < 2 &&
           emap_get(em, a, c) < 2;
}

/* Record a triangle's edges. */
static void tri_record(dmc_edge_map *em, int32_t a, int32_t b, int32_t c) {
    emap_add(em, a, b);
    emap_add(em, b, c);
    emap_add(em, a, c);
}

/* ── DMC patch extraction ────────────────────────────────────────── */

static const int k_dmc_corner_offset[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
};

static const int k_dmc_edge_corners[12][2] = {
    {0, 1}, {2, 3}, {4, 5}, {6, 7},
    {0, 2}, {1, 3}, {4, 6}, {5, 7},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

static const int k_dmc_face_corners[6][4] = {
    {0, 1, 3, 2},  /* z = 0 */
    {4, 6, 7, 5},  /* z = 1 */
    {0, 4, 5, 1},  /* y = 0 */
    {2, 3, 7, 6},  /* y = 1 */
    {0, 2, 6, 4},  /* x = 0 */
    {1, 5, 7, 3},  /* x = 1 */
};

static const int k_dmc_face_edges[6][4] = {
    {0, 5, 1, 4},
    {6, 3, 7, 2},
    {8, 2, 9, 0},
    {1, 11, 3, 10},
    {4, 10, 6, 8},
    {9, 7, 11, 5},
};

typedef struct {
    int patch_count;
    int edge_patch[12];
    pamo_vec3d patch_pos[12];
} dmc_cell_patches;

static pamo_vec3d grid_point(pamo_vec3d origin, double voxel_size,
                             int32_t ix, int32_t iy, int32_t iz) {
    return (pamo_vec3d){
        origin.x + ((double)ix + 0.5) * voxel_size,
        origin.y + ((double)iy + 0.5) * voxel_size,
        origin.z + ((double)iz + 0.5) * voxel_size,
    };
}

static int edge_crosses(double s0, double s1) {
    return (s0 < 0.0 && s1 >= 0.0) || (s1 < 0.0 && s0 >= 0.0);
}

static pamo_vec3d edge_crossing_point(pamo_vec3d p0, pamo_vec3d p1,
                                      double s0, double s1) {
    double t = s0 / (s0 - s1);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return pamo_v3_add(p0, pamo_v3_scale(pamo_v3_sub(p1, p0), t));
}

static pamo_vec3d v3_normalize_or_zero(pamo_vec3d v) {
    double len2 = pamo_v3_dot(v, v);
    if (len2 <= 1.0e-30) return pamo_v3_zero();
    return pamo_v3_scale(v, 1.0 / sqrt(len2));
}

static double dmc_qef_blend(void) {
    static int checked = 0;
    /* Keep the default aligned with the reference CUDA PaMO path, which uses
     * the DMC vertex placement directly. QEF pull is useful for experiments,
     * but can open thin sheet features on production meshes. */
    static double blend = 0.0;
    if (!checked) {
        const char *e = getenv("PAMO_DMC_QEF_BLEND");
        if (e && e[0]) {
            blend = strtod(e, NULL);
            if (blend < 0.0) blend = 0.0;
            if (blend > 1.0) blend = 1.0;
        }
        checked = 1;
    }
    return blend;
}

static double sdf_at_grid(const double *sdf, int32_t R,
                          int32_t ix, int32_t iy, int32_t iz) {
    return sdf[SDF_IDX(ix, iy, iz, R)];
}

static pamo_vec3d sdf_grid_gradient(const double *sdf, int32_t R,
                                    int32_t ix, int32_t iy, int32_t iz,
                                    double voxel_size) {
    int32_t xm = ix > 0 ? ix - 1 : ix;
    int32_t xp = ix + 1 < R ? ix + 1 : ix;
    int32_t ym = iy > 0 ? iy - 1 : iy;
    int32_t yp = iy + 1 < R ? iy + 1 : iy;
    int32_t zm = iz > 0 ? iz - 1 : iz;
    int32_t zp = iz + 1 < R ? iz + 1 : iz;

    double dx_den = (double)(xp - xm) * voxel_size;
    double dy_den = (double)(yp - ym) * voxel_size;
    double dz_den = (double)(zp - zm) * voxel_size;
    if (dx_den == 0.0) dx_den = 1.0;
    if (dy_den == 0.0) dy_den = 1.0;
    if (dz_den == 0.0) dz_den = 1.0;

    return (pamo_vec3d){
        (sdf_at_grid(sdf, R, xp, iy, iz) -
         sdf_at_grid(sdf, R, xm, iy, iz)) / dx_den,
        (sdf_at_grid(sdf, R, ix, yp, iz) -
         sdf_at_grid(sdf, R, ix, ym, iz)) / dy_den,
        (sdf_at_grid(sdf, R, ix, iy, zp) -
         sdf_at_grid(sdf, R, ix, iy, zm)) / dz_den,
    };
}

static pamo_vec3d edge_crossing_normal(const double *sdf, int32_t R,
                                       int32_t ax, int32_t ay, int32_t az,
                                       int32_t bx, int32_t by, int32_t bz,
                                       pamo_vec3d p0, pamo_vec3d p1,
                                       double s0, double s1,
                                       double voxel_size) {
    double t = s0 / (s0 - s1);
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    pamo_vec3d g0 = sdf_grid_gradient(sdf, R, ax, ay, az, voxel_size);
    pamo_vec3d g1 = sdf_grid_gradient(sdf, R, bx, by, bz, voxel_size);
    pamo_vec3d n = pamo_v3_add(pamo_v3_scale(g0, 1.0 - t),
                               pamo_v3_scale(g1, t));
    n = v3_normalize_or_zero(n);
    if (pamo_v3_dot(n, n) > 0.0) return n;

    pamo_vec3d dir = pamo_v3_sub(p1, p0);
    if (s0 >= 0.0 && s1 < 0.0) dir = pamo_v3_scale(dir, -1.0);
    return v3_normalize_or_zero(dir);
}

static void qef_add(double a[6], double b[3],
                    pamo_vec3d p, pamo_vec3d n) {
    double len2 = pamo_v3_dot(n, n);
    if (len2 <= 0.0) return;
    double d = pamo_v3_dot(n, p);
    a[0] += n.x * n.x;
    a[1] += n.x * n.y;
    a[2] += n.x * n.z;
    a[3] += n.y * n.y;
    a[4] += n.y * n.z;
    a[5] += n.z * n.z;
    b[0] += n.x * d;
    b[1] += n.y * d;
    b[2] += n.z * d;
}

static int solve_3x3(double m[3][3], double rhs[3], pamo_vec3d *x) {
    for (int col = 0; col < 3; col++) {
        int pivot = col;
        double best = fabs(m[col][col]);
        for (int row = col + 1; row < 3; row++) {
            double v = fabs(m[row][col]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < 1.0e-14) return 0;
        if (pivot != col) {
            for (int k = col; k < 3; k++) {
                double t = m[col][k];
                m[col][k] = m[pivot][k];
                m[pivot][k] = t;
            }
            double tb = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = tb;
        }

        double inv = 1.0 / m[col][col];
        for (int row = col + 1; row < 3; row++) {
            double f = m[row][col] * inv;
            m[row][col] = 0.0;
            for (int k = col + 1; k < 3; k++) {
                m[row][k] -= f * m[col][k];
            }
            rhs[row] -= f * rhs[col];
        }
    }

    double z = rhs[2] / m[2][2];
    double y = (rhs[1] - m[1][2] * z) / m[1][1];
    double xx = (rhs[0] - m[0][1] * y - m[0][2] * z) / m[0][0];
    *x = (pamo_vec3d){xx, y, z};
    return 1;
}

static pamo_vec3d clamp_to_cell(pamo_vec3d p,
                                pamo_vec3d lo, pamo_vec3d hi) {
    if (p.x < lo.x) p.x = lo.x;
    if (p.y < lo.y) p.y = lo.y;
    if (p.z < lo.z) p.z = lo.z;
    if (p.x > hi.x) p.x = hi.x;
    if (p.y > hi.y) p.y = hi.y;
    if (p.z > hi.z) p.z = hi.z;
    return p;
}

static int uf_find(int parent[12], int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static void uf_union(int parent[12], int a, int b) {
    int ra = uf_find(parent, a);
    int rb = uf_find(parent, b);
    if (ra != rb) parent[rb] = ra;
}

static int build_cell_patches(const double *sdf, int32_t R,
                              int32_t ix, int32_t iy, int32_t iz,
                              pamo_vec3d origin, double voxel_size,
                              double iso,
                              dmc_cell_patches *out) {
    if (!out) return 0;
    out->patch_count = 0;
    for (int i = 0; i < 12; i++) {
        out->edge_patch[i] = -1;
        out->patch_pos[i] = pamo_v3_zero();
    }

    pamo_vec3d p[8];
    double s[8];
    for (int i = 0; i < 8; i++) {
        int32_t gx = ix + k_dmc_corner_offset[i][0];
        int32_t gy = iy + k_dmc_corner_offset[i][1];
        int32_t gz = iz + k_dmc_corner_offset[i][2];
        p[i] = grid_point(origin, voxel_size, gx, gy, gz);
        s[i] = sdf[SDF_IDX(gx, gy, gz, R)] - iso;
    }

    int has_pos = 0, has_neg = 0;
    for (int i = 0; i < 8; i++) {
        if (s[i] >= 0.0) has_pos = 1;
        else has_neg = 1;
    }
    if (!has_pos || !has_neg) return 0;

    int crosses[12];
    int parent[12];
    for (int e = 0; e < 12; e++) {
        int a = k_dmc_edge_corners[e][0];
        int b = k_dmc_edge_corners[e][1];
        crosses[e] = edge_crosses(s[a], s[b]);
        parent[e] = e;
    }

    for (int f = 0; f < 6; f++) {
        int face_cross[4];
        int n = 0;
        for (int i = 0; i < 4; i++) {
            int e = k_dmc_face_edges[f][i];
            if (crosses[e]) face_cross[n++] = i;
        }
        if (n == 2) {
            uf_union(parent,
                     k_dmc_face_edges[f][face_cross[0]],
                     k_dmc_face_edges[f][face_cross[1]]);
        } else if (n == 4) {
            int c0 = k_dmc_face_corners[f][0];
            int c1 = k_dmc_face_corners[f][1];
            int c2 = k_dmc_face_corners[f][2];
            int c3 = k_dmc_face_corners[f][3];
            double center = 0.25 * (s[c0] + s[c1] + s[c2] + s[c3]);
            int center_same_as_c0 = ((center >= 0.0) == (s[c0] >= 0.0));
            if (center_same_as_c0) {
                uf_union(parent, k_dmc_face_edges[f][0],
                                  k_dmc_face_edges[f][1]);
                uf_union(parent, k_dmc_face_edges[f][2],
                                  k_dmc_face_edges[f][3]);
            } else {
                uf_union(parent, k_dmc_face_edges[f][3],
                                  k_dmc_face_edges[f][0]);
                uf_union(parent, k_dmc_face_edges[f][1],
                                  k_dmc_face_edges[f][2]);
            }
        } else if (n > 2) {
            for (int i = 1; i < n; i++) {
                uf_union(parent,
                         k_dmc_face_edges[f][face_cross[0]],
                         k_dmc_face_edges[f][face_cross[i]]);
            }
        }
    }

    int root_to_patch[12];
    int patch_samples[12];
    pamo_vec3d patch_sum[12];
    double patch_a[12][6];
    double patch_b[12][3];
    double qef_blend = dmc_qef_blend();
    for (int i = 0; i < 12; i++) {
        root_to_patch[i] = -1;
        patch_samples[i] = 0;
        patch_sum[i] = pamo_v3_zero();
        for (int j = 0; j < 6; j++) patch_a[i][j] = 0.0;
        for (int j = 0; j < 3; j++) patch_b[i][j] = 0.0;
    }

    for (int e = 0; e < 12; e++) {
        if (!crosses[e]) continue;
        int root = uf_find(parent, e);
        int patch = root_to_patch[root];
        if (patch < 0) {
            patch = out->patch_count++;
            root_to_patch[root] = patch;
        }
        out->edge_patch[e] = patch;

        int a = k_dmc_edge_corners[e][0];
        int b = k_dmc_edge_corners[e][1];
        pamo_vec3d cp = edge_crossing_point(p[a], p[b], s[a], s[b]);
        patch_sum[patch] = pamo_v3_add(patch_sum[patch], cp);
        if (qef_blend > 0.0) {
            int32_t ax = ix + k_dmc_corner_offset[a][0];
            int32_t ay = iy + k_dmc_corner_offset[a][1];
            int32_t az = iz + k_dmc_corner_offset[a][2];
            int32_t bx = ix + k_dmc_corner_offset[b][0];
            int32_t by = iy + k_dmc_corner_offset[b][1];
            int32_t bz = iz + k_dmc_corner_offset[b][2];
            pamo_vec3d n = edge_crossing_normal(
                sdf, R, ax, ay, az, bx, by, bz,
                p[a], p[b], s[a], s[b], voxel_size);
            qef_add(patch_a[patch], patch_b[patch], cp, n);
        }
        patch_samples[patch]++;
    }

    pamo_vec3d cell_lo = grid_point(origin, voxel_size, ix, iy, iz);
    pamo_vec3d cell_hi = grid_point(origin, voxel_size, ix + 1, iy + 1, iz + 1);
    for (int i = 0; i < out->patch_count; i++) {
        if (patch_samples[i] > 0) {
            pamo_vec3d avg =
                pamo_v3_scale(patch_sum[i], 1.0 / (double)patch_samples[i]);
            out->patch_pos[i] = avg;
            if (qef_blend <= 0.0) continue;

            double lambda = 1.0e-8 * (double)patch_samples[i];
            double m[3][3] = {
                {patch_a[i][0] + lambda, patch_a[i][1], patch_a[i][2]},
                {patch_a[i][1], patch_a[i][3] + lambda, patch_a[i][4]},
                {patch_a[i][2], patch_a[i][4], patch_a[i][5] + lambda},
            };
            double rhs[3] = {
                patch_b[i][0] + lambda * avg.x,
                patch_b[i][1] + lambda * avg.y,
                patch_b[i][2] + lambda * avg.z,
            };
            pamo_vec3d qef_pos = avg;
            if (solve_3x3(m, rhs, &qef_pos)) {
                qef_pos = clamp_to_cell(qef_pos, cell_lo, cell_hi);
                out->patch_pos[i] = pamo_v3_add(
                    avg, pamo_v3_scale(pamo_v3_sub(qef_pos, avg), qef_blend));
            }
        }
    }

    return out->patch_count;
}

/* ── DMC face emission ───────────────────────────────────────────── */

static double tri_quality_score(pamo_vec3d a, pamo_vec3d b, pamo_vec3d c) {
    pamo_vec3d ab = pamo_v3_sub(b, a);
    pamo_vec3d bc = pamo_v3_sub(c, b);
    pamo_vec3d ca = pamo_v3_sub(a, c);
    double l0 = pamo_v3_dot(ab, ab);
    double l1 = pamo_v3_dot(bc, bc);
    double l2 = pamo_v3_dot(ca, ca);
    double denom = l0 + l1 + l2;
    if (denom <= 0.0) return 0.0;
    pamo_vec3d cr = pamo_v3_cross(ab, pamo_v3_sub(c, a));
    return pamo_v3_dot(cr, cr) / (denom * denom);
}

static double split_quality(const pamo_mesh *mesh, const int32_t q[4],
                            int split) {
    pamo_vec3d v0 = mesh->verts[q[0]];
    pamo_vec3d v1 = mesh->verts[q[1]];
    pamo_vec3d v2 = mesh->verts[q[2]];
    pamo_vec3d v3 = mesh->verts[q[3]];
    if (split == 0) {
        double a = tri_quality_score(v0, v1, v2);
        double b = tri_quality_score(v0, v2, v3);
        return a < b ? a : b;
    }
    double a = tri_quality_score(v0, v1, v3);
    double b = tri_quality_score(v1, v2, v3);
    return a < b ? a : b;
}

static int split_can_add(const dmc_edge_map *em, const int32_t q[4],
                         int split) {
    if (split == 0) {
        return tri_can_add(em, q[0], q[1], q[2]) &&
               tri_can_add(em, q[0], q[2], q[3]);
    }
    return tri_can_add(em, q[0], q[1], q[3]) &&
           tri_can_add(em, q[1], q[2], q[3]);
}

static size_t emit_split(pamo_tri *faces, size_t fi, size_t max_fi,
                         const int32_t q[4], int split,
                         dmc_edge_map *em) {
    if (fi + 2 > max_fi) return fi;
    if (split == 0) {
        faces[fi] = (pamo_tri){{q[0], q[1], q[2]}}; fi++;
        tri_record(em, q[0], q[1], q[2]);
        faces[fi] = (pamo_tri){{q[0], q[2], q[3]}}; fi++;
        tri_record(em, q[0], q[2], q[3]);
    } else {
        faces[fi] = (pamo_tri){{q[0], q[1], q[3]}}; fi++;
        tri_record(em, q[0], q[1], q[3]);
        faces[fi] = (pamo_tri){{q[1], q[2], q[3]}}; fi++;
        tri_record(em, q[1], q[2], q[3]);
    }
    return fi;
}

static size_t emit_quad(pamo_mesh *mesh, size_t fi, size_t max_fi,
                        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                        bool flip, dmc_edge_map *em) {
    if (fi + 2 > max_fi) return fi;

    int32_t q[4];
    if (flip) {
        q[0] = c0; q[1] = c3; q[2] = c2; q[3] = c1;
    } else {
        q[0] = c0; q[1] = c1; q[2] = c2; q[3] = c3;
    }

    double qa = split_quality(mesh, q, 0);
    double qb = split_quality(mesh, q, 1);
    int preferred = qa >= qb ? 0 : 1;
    int alternate = 1 - preferred;

    if (split_can_add(em, q, preferred)) {
        return emit_split(mesh->faces, fi, max_fi, q, preferred, em);
    }
    if (split_can_add(em, q, alternate)) {
        return emit_split(mesh->faces, fi, max_fi, q, alternate, em);
    }
    return emit_split(mesh->faces, fi, max_fi, q, preferred, em);
}

static size_t emit_triangle(pamo_mesh *mesh, size_t fi, size_t max_fi,
                            int32_t a, int32_t b, int32_t c,
                            bool flip, dmc_edge_map *em) {
    if (fi >= max_fi) return fi;
    if (flip) {
        int32_t t = b;
        b = c;
        c = t;
    }
    if (a == b || b == c || a == c) return fi;
    if (!tri_can_add(em, a, b, c)) return fi;
    mesh->faces[fi++] = (pamo_tri){{a, b, c}};
    tri_record(em, a, b, c);
    return fi;
}

static int32_t get_cell_edge_vertex(const int32_t *cell_active_index,
                                    const int32_t *cell_edge_vert,
                                    int32_t C,
                                    int32_t cx, int32_t cy, int32_t cz,
                                    int local_edge) {
    if (cx < 0 || cy < 0 || cz < 0 || cx >= C || cy >= C || cz >= C)
        return -1;
    int32_t active = cell_active_index[CELL_IDX(cx, cy, cz, C)];
    if (active < 0) return -1;
    return cell_edge_vert[(size_t)active * 12u + (size_t)local_edge];
}

/* ── Main DMC ────────────────────────────────────────────────────── */

pamo_error pamo_dual_mc(pamo_mesh *out, const double *sdf, int32_t R,
                        pamo_vec3d origin, double voxel_size, double iso,
                        const pamo_allocator *alloc) {
    if (!out || !sdf || R < 2) return PAMO_ERR_INVALID_ARG;

    int32_t C = R - 1;
    size_t n_cells = (size_t)C * (size_t)C * (size_t)C;

    int32_t *cell_active_index = (int32_t *)pamo_alloc(alloc,
        n_cells * sizeof(int32_t));
    if (!cell_active_index) return PAMO_ERR_ALLOC;
    memset(cell_active_index, -1, n_cells * sizeof(int32_t));

    size_t n_surf = 0;
    size_t total_patches = 0;
    for (int32_t iz = 0; iz < C; iz++)
        for (int32_t iy = 0; iy < C; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                dmc_cell_patches patches;
                int patch_count = build_cell_patches(
                    sdf, R, ix, iy, iz, origin, voxel_size, iso, &patches);
                if (patch_count > 0) {
                    if (n_surf > (size_t)0x7fffffff ||
                        total_patches + (size_t)patch_count >
                            (size_t)0x7fffffff) {
                        pamo_free(alloc, cell_active_index,
                                  n_cells * sizeof(int32_t));
                        return PAMO_ERR_INVALID_ARG;
                    }
                    cell_active_index[CELL_IDX(ix, iy, iz, C)] =
                        (int32_t)n_surf;
                    n_surf++;
                    total_patches += (size_t)patch_count;
                }
            }

    if (n_surf == 0) {
        pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
        return pamo_mesh_create(out, 0, 0, alloc);
    }

    if (n_surf > SIZE_MAX / (12u * sizeof(int32_t))) {
        pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
        return PAMO_ERR_INVALID_ARG;
    }
    int32_t *cell_edge_vert = (int32_t *)pamo_alloc(
        alloc, n_surf * 12u * sizeof(int32_t));
    if (!cell_edge_vert) {
        pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
        return PAMO_ERR_ALLOC;
    }
    memset(cell_edge_vert, -1, n_surf * 12u * sizeof(int32_t));

    if (total_patches > (((size_t)INT32_MAX - 64u) / 8u)) {
        pamo_free(alloc, cell_edge_vert, n_surf * 12u * sizeof(int32_t));
        pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
        return PAMO_ERR_INVALID_ARG;
    }
    size_t max_faces = total_patches * 8u + 64u;
    pamo_error err = pamo_mesh_create(out, total_patches, max_faces, alloc);
    if (err != PAMO_OK) {
        pamo_free(alloc, cell_edge_vert, n_surf * 12u * sizeof(int32_t));
        pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
        return err;
    }

    size_t patch_base = 0;
    for (int32_t iz = 0; iz < C; iz++)
        for (int32_t iy = 0; iy < C; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                int32_t active =
                    cell_active_index[CELL_IDX(ix, iy, iz, C)];
                if (active < 0) continue;

                dmc_cell_patches patches;
                int patch_count = build_cell_patches(
                    sdf, R, ix, iy, iz, origin, voxel_size, iso, &patches);
                if (patch_base + (size_t)patch_count > total_patches) {
                    pamo_free(alloc, cell_edge_vert,
                              n_surf * 12u * sizeof(int32_t));
                    pamo_free(alloc, cell_active_index,
                              n_cells * sizeof(int32_t));
                    pamo_mesh_destroy(out);
                    return PAMO_ERR_INVALID_ARG;
                }
                for (int p = 0; p < patch_count; p++) {
                    out->verts[patch_base + (size_t)p] =
                        patches.patch_pos[p];
                }
                for (int e = 0; e < 12; e++) {
                    int patch = patches.edge_patch[e];
                    if (patch >= 0) {
                        cell_edge_vert[(size_t)active * 12u + (size_t)e] =
                            (int32_t)(patch_base + (size_t)patch);
                    }
                }
                patch_base += (size_t)patch_count;
            }

    /* Edge map to track usage. Size ~4× expected edges for low collision. */
    dmc_edge_map em;
    emap_init(&em, max_faces * 4u + 1024u, alloc);
    if (!em.slots) {
        pamo_free(alloc, cell_edge_vert, n_surf * 12u * sizeof(int32_t));
        pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
        pamo_mesh_destroy(out);
        return PAMO_ERR_ALLOC;
    }

    size_t fi = 0;

    /* Emit quads around each grid edge that crosses the iso-surface.
     *
     * The cells surrounding a grid edge are traversed in the order
     * (-,-), (+,-), (+,+), (-,+) in the 2D plane perpendicular to the
     * edge. For an X-edge that maps to the YZ plane, and using
     * (c1-c0) × (c2-c0) as the triangle normal convention, the
     * non-flipped triangulation (c0,c1,c2) + (c0,c2,c3) has a normal
     * of +X. For a Y-edge in the XZ plane the non-flipped normal is
     * -Y, and for a Z-edge in the XY plane it is +Z.
     *
     * The outward surface normal points from negative SDF (inside) to
     * positive SDF (outside), i.e. in the direction where v increases.
     * So the per-axis flip conditions are:
     *   X: flip when surface normal points -X → when v0 > v1
     *   Y: flip when surface normal points +Y → when v0 < v1
     *      (because the non-flipped quad already faces -Y)
     *   Z: flip when surface normal points -Z → when v0 > v1
     *
     * Getting this wrong on any axis leaves neighbouring quads with
     * inconsistent winding — the mesh is still closed but trimesh
     * reports is_winding_consistent = False.
     */

    /* X-edges */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix+1, iy, iz, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;
                int32_t cy[4] = {iy-1, iy,   iy,   iy-1};
                int32_t cz[4] = {iz-1, iz-1, iz,   iz  };
                int local_edge[4] = {3, 2, 0, 1};
                int32_t c[4]; int nc = 0;
                for (int k = 0; k < 4; k++) {
                    int32_t cv = get_cell_edge_vertex(
                        cell_active_index, cell_edge_vert, C,
                        ix, cy[k], cz[k], local_edge[k]);
                    if (cv >= 0) c[nc++] = cv;
                }
                if (nc == 4) {
                    fi = emit_quad(out, fi, max_faces,
                                   c[0], c[1], c[2], c[3], v0 > v1, &em);
                } else if (nc == 3) {
                    fi = emit_triangle(out, fi, max_faces,
                                       c[0], c[1], c[2], v0 > v1, &em);
                }
            }

    /* Y-edges */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < C; iy++)
            for (int32_t ix = 0; ix < R; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix, iy+1, iz, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;
                int32_t cx[4] = {ix-1, ix,   ix,   ix-1};
                int32_t cz[4] = {iz-1, iz-1, iz,   iz  };
                int local_edge[4] = {7, 6, 4, 5};
                int32_t c[4]; int nc = 0;
                for (int k = 0; k < 4; k++) {
                    int32_t cv = get_cell_edge_vertex(
                        cell_active_index, cell_edge_vert, C,
                        cx[k], iy, cz[k], local_edge[k]);
                    if (cv >= 0) c[nc++] = cv;
                }
                if (nc == 4) {
                    fi = emit_quad(out, fi, max_faces,
                                   c[0], c[1], c[2], c[3], v0 < v1, &em);
                } else if (nc == 3) {
                    fi = emit_triangle(out, fi, max_faces,
                                       c[0], c[1], c[2], v0 < v1, &em);
                }
            }

    /* Z-edges */
    for (int32_t iz = 0; iz < C; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < R; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix, iy, iz+1, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;
                int32_t cx[4] = {ix-1, ix,   ix,   ix-1};
                int32_t cy[4] = {iy-1, iy-1, iy,   iy  };
                int local_edge[4] = {11, 10, 8, 9};
                int32_t c[4]; int nc = 0;
                for (int k = 0; k < 4; k++) {
                    int32_t cv = get_cell_edge_vertex(
                        cell_active_index, cell_edge_vert, C,
                        cx[k], cy[k], iz, local_edge[k]);
                    if (cv >= 0) c[nc++] = cv;
                }
                if (nc == 4) {
                    fi = emit_quad(out, fi, max_faces,
                                   c[0], c[1], c[2], c[3], v0 > v1, &em);
                } else if (nc == 3) {
                    fi = emit_triangle(out, fi, max_faces,
                                       c[0], c[1], c[2], v0 > v1, &em);
                }
            }

    emap_destroy(&em);

    out->n_faces = fi;
    for (size_t i = fi; i < max_faces; i++)
        out->face_alive[i] = false;

    pamo_free(alloc, cell_edge_vert, n_surf * 12u * sizeof(int32_t));
    pamo_free(alloc, cell_active_index, n_cells * sizeof(int32_t));
    return pamo_mesh_compact(out);
}
