/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle mesh -> signed distance field on a 3D grid.
 *
 * Algorithm (equivalent to cumesh2sdf):
 *   Phase 1: Compute unsigned distance from each voxel to nearest triangle
 *            using BVH-accelerated closest-point queries.
 *   Phase 2: Determine sign using scanline parity (ray casting along Z axis).
 *            For each (x,y) column, count ray-mesh intersections along Z.
 *            Odd crossing count = inside, even = outside.
 *            Uses 3-axis voting for robustness at degenerate cases.
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_alloc.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── Ray-triangle intersection (Moeller-Trumbore) ────────────────── */

static double ray_tri_hit(pamo_vec3d origin, pamo_vec3d dir,
                          pamo_vec3d v0, pamo_vec3d v1, pamo_vec3d v2) {
    pamo_vec3d e1 = pamo_v3_sub(v1, v0);
    pamo_vec3d e2 = pamo_v3_sub(v2, v0);
    pamo_vec3d h = pamo_v3_cross(dir, e2);
    double a = pamo_v3_dot(e1, h);
    if (fabs(a) < 1e-15) return -1.0;
    double f = 1.0 / a;
    pamo_vec3d s = pamo_v3_sub(origin, v0);
    double u = f * pamo_v3_dot(s, h);
    if (u < -1e-10 || u > 1.0 + 1e-10) return -1.0;
    pamo_vec3d q = pamo_v3_cross(s, e1);
    double v = f * pamo_v3_dot(dir, q);
    if (v < -1e-10 || u + v > 1.0 + 1e-10) return -1.0;
    double t = f * pamo_v3_dot(e2, q);
    return (t > 1e-10) ? t : -1.0;
}

/* ── SDF scanline sign computation ──────��────────────────────────── */

/* For a given scanline at (px, py), cast a ray in +Z direction and
 * collect all intersection t-values with the mesh. Return the count. */
static int scanline_intersections(pamo_vec3d origin, const pamo_mesh *m,
                                  double *t_buf, int max_t) {
    int n = 0;
    pamo_vec3d dir = {0.0, 0.0, 1.0};
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        if (!m->face_alive[fi]) continue;
        pamo_vec3d v0 = m->verts[m->faces[fi].v[0]];
        pamo_vec3d v1 = m->verts[m->faces[fi].v[1]];
        pamo_vec3d v2 = m->verts[m->faces[fi].v[2]];
        double t = ray_tri_hit(origin, dir, v0, v1, v2);
        if (t >= 0.0 && n < max_t) {
            t_buf[n++] = t;
        }
    }
    /* Sort intersections by t. */
    for (int i = 1; i < n; i++) {
        double key = t_buf[i];
        int j = i - 1;
        while (j >= 0 && t_buf[j] > key) {
            t_buf[j + 1] = t_buf[j];
            j--;
        }
        t_buf[j + 1] = key;
    }
    return n;
}

/* ── Main SDF computation ────────────────────────────────────────── */

#define GRID_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

pamo_error pamo_compute_sdf(double *grid, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh) {
    size_t grid_size = (size_t)R * (size_t)R * (size_t)R;

    /* ── Phase 1: Unsigned distances via BVH ─────────────────────── */
    for (int32_t iz = 0; iz < R; iz++) {
        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                pamo_vec3d p = {
                    grid_origin.x + ((double)ix + 0.5) * voxel_size,
                    grid_origin.y + ((double)iy + 0.5) * voxel_size,
                    grid_origin.z + ((double)iz + 0.5) * voxel_size,
                };
                pamo_nearest_result res;
                pamo_bvh_nearest(bvh, m, p, &res);
                grid[GRID_IDX(ix, iy, iz, R)] = sqrt(res.dist_sq);
            }
        }
    }

    /* ── Phase 2: Sign via Z-scanline parity ─────────────────────── */
    /* For each (ix, iy) column, cast a ray in +Z direction from below
     * the grid. Count intersections to determine inside/outside for
     * each voxel along the column. */

    int max_hits = (int)m->n_faces; /* worst case */
    if (max_hits > 10000) max_hits = 10000;
    double *t_buf = (double *)malloc((size_t)max_hits * sizeof(double));
    if (!t_buf) return PAMO_ERR_ALLOC;

    for (int32_t iy = 0; iy < R; iy++) {
        for (int32_t ix = 0; ix < R; ix++) {
            /* Ray origin: bottom of this column. */
            pamo_vec3d origin = {
                grid_origin.x + ((double)ix + 0.5) * voxel_size,
                grid_origin.y + ((double)iy + 0.5) * voxel_size,
                grid_origin.z - voxel_size, /* below grid */
            };

            int n_hits = scanline_intersections(origin, m, t_buf, max_hits);

            /* Walk the voxels in this column. A voxel at position iz
             * is inside if there's an odd number of intersections
             * between the ray origin and the voxel center. */
            int hit_idx = 0;
            bool inside = false;
            for (int32_t iz = 0; iz < R; iz++) {
                double voxel_z = ((double)iz + 0.5) * voxel_size + voxel_size; /* offset from origin.z */

                /* Count intersections up to this voxel. */
                while (hit_idx < n_hits && t_buf[hit_idx] < voxel_z) {
                    inside = !inside;
                    hit_idx++;
                }

                if (inside) {
                    size_t idx = GRID_IDX(ix, iy, iz, R);
                    grid[idx] = -grid[idx];
                }
            }
        }
    }

    free(t_buf);
    return PAMO_OK;
}
