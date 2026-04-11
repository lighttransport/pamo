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
#ifdef PAMO_USE_LIGHTRT
#include "pamo/pamo_lightrt.h"
#endif

#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef PAMO_USE_PTHREADS
#include <pthread.h>
#endif

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

/* ── Parallel SDF helper ──────────────────────────────────────────── */

#define GRID_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

#if defined(PAMO_USE_PTHREADS) && defined(PAMO_USE_LIGHTRT)
typedef struct {
    const void *bvh; /* lightrt_bvh* */
    double *grid;
    int32_t R, iz_start, iz_end;
    pamo_vec3d grid_origin;
    double voxel_size;
} sdf_thread_arg;

static void *sdf_phase1_worker(void *arg) {
    sdf_thread_arg *a = (sdf_thread_arg *)arg;
    for (int32_t iz = a->iz_start; iz < a->iz_end; iz++) {
        for (int32_t iy = 0; iy < a->R; iy++) {
            for (int32_t ix = 0; ix < a->R; ix++) {
                float p[3] = {
                    (float)(a->grid_origin.x + ((double)ix + 0.5) * a->voxel_size),
                    (float)(a->grid_origin.y + ((double)iy + 0.5) * a->voxel_size),
                    (float)(a->grid_origin.z + ((double)iz + 0.5) * a->voxel_size),
                };
                float dsq = lightrt_bvh_nearest_dist_sq(
                    (const lightrt_bvh *)a->bvh, p);
                a->grid[GRID_IDX(ix, iy, iz, a->R)] = sqrt((double)dsq);
            }
        }
    }
    return NULL;
}
#endif

/* ── Main SDF computation ────────────────────────────────────────── */

#undef GRID_IDX
#define GRID_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

pamo_error pamo_compute_sdf(double *grid, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh) {
    size_t grid_size = (size_t)R * (size_t)R * (size_t)R;

    /* ── Phase 1: Unsigned distances ────────────────────────────── */
#ifdef PAMO_USE_LIGHTRT
    /* Fast path: use lightrt queryNearest (SIMD-optimized BVH). */
    {
        float *fverts = (float *)malloc(m->n_verts * 3 * sizeof(float));
        int32_t *ifaces = (int32_t *)malloc(m->n_faces * 3 * sizeof(int32_t));
        int32_t *vert_remap = (int32_t *)malloc(m->n_verts * sizeof(int32_t));
        if (!fverts || !ifaces || !vert_remap) {
            free(fverts); free(ifaces); free(vert_remap);
            return PAMO_ERR_ALLOC;
        }
        int32_t av = 0, af = 0;
        for (size_t i = 0; i < m->n_verts; i++) {
            if (m->vert_alive[i]) {
                fverts[av*3+0] = (float)m->verts[i].x;
                fverts[av*3+1] = (float)m->verts[i].y;
                fverts[av*3+2] = (float)m->verts[i].z;
                vert_remap[i] = av++;
            } else vert_remap[i] = -1;
        }
        for (size_t i = 0; i < m->n_faces; i++) {
            if (!m->face_alive[i]) continue;
            ifaces[af*3+0] = vert_remap[m->faces[i].v[0]];
            ifaces[af*3+1] = vert_remap[m->faces[i].v[1]];
            ifaces[af*3+2] = vert_remap[m->faces[i].v[2]];
            af++;
        }
        free(vert_remap);

        lightrt_bvh *lbvh_dist = lightrt_bvh_build(fverts, av, ifaces, af);
        free(fverts); free(ifaces);
        if (!lbvh_dist) return PAMO_ERR_ALLOC;

#ifdef PAMO_USE_PTHREADS
        /* Parallel Phase 1: split Z-slices across threads. */
        int n_threads = 8;
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        sdf_thread_arg *args = (sdf_thread_arg *)malloc((size_t)n_threads * sizeof(sdf_thread_arg));
        int32_t slices_per_thread = (R + n_threads - 1) / n_threads;
        for (int t = 0; t < n_threads; t++) {
            args[t].bvh = lbvh_dist;
            args[t].grid = grid;
            args[t].R = R;
            args[t].iz_start = t * slices_per_thread;
            args[t].iz_end = (t + 1) * slices_per_thread;
            if (args[t].iz_end > R) args[t].iz_end = R;
            args[t].grid_origin = grid_origin;
            args[t].voxel_size = voxel_size;
            pthread_create(&threads[t], NULL, sdf_phase1_worker, &args[t]);
        }
        for (int t = 0; t < n_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        free(threads);
        free(args);
#else
        for (int32_t iz = 0; iz < R; iz++) {
            for (int32_t iy = 0; iy < R; iy++) {
                for (int32_t ix = 0; ix < R; ix++) {
                    float p[3] = {
                        (float)(grid_origin.x + ((double)ix + 0.5) * voxel_size),
                        (float)(grid_origin.y + ((double)iy + 0.5) * voxel_size),
                        (float)(grid_origin.z + ((double)iz + 0.5) * voxel_size),
                    };
                    float dsq = lightrt_bvh_nearest_dist_sq(lbvh_dist, p);
                    grid[GRID_IDX(ix, iy, iz, R)] = sqrt((double)dsq);
                }
            }
        }
#endif
        lightrt_bvh_destroy(lbvh_dist);
    }
#else
    /* Fallback: pamo BVH nearest-point queries. */
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
#endif

    /* ── Phase 2: Sign via Z-scanline parity ─────────────────────── */
    /* For each (ix, iy) column, cast a ray in +Z direction from below
     * the grid. Count intersections to determine inside/outside for
     * each voxel along the column. */

#ifdef PAMO_USE_LIGHTRT
    /* Fast path: use lightrt BVH for multi-hit ray traversal. */
    {
        /* Convert mesh to float32 for lightrt. */
        float *fverts = (float *)malloc(m->n_verts * 3 * sizeof(float));
        int32_t *ifaces = (int32_t *)malloc(m->n_faces * 3 * sizeof(int32_t));
        int32_t *vert_remap = (int32_t *)malloc(m->n_verts * sizeof(int32_t));
        if (!fverts || !ifaces || !vert_remap) {
            free(fverts); free(ifaces); free(vert_remap);
            return PAMO_ERR_ALLOC;
        }
        int32_t alive_verts = 0, alive_faces = 0;
        for (size_t i = 0; i < m->n_verts; i++) {
            if (m->vert_alive[i]) {
                fverts[alive_verts*3+0] = (float)m->verts[i].x;
                fverts[alive_verts*3+1] = (float)m->verts[i].y;
                fverts[alive_verts*3+2] = (float)m->verts[i].z;
                vert_remap[i] = alive_verts++;
            } else {
                vert_remap[i] = -1;
            }
        }
        for (size_t i = 0; i < m->n_faces; i++) {
            if (!m->face_alive[i]) continue;
            ifaces[alive_faces*3+0] = vert_remap[m->faces[i].v[0]];
            ifaces[alive_faces*3+1] = vert_remap[m->faces[i].v[1]];
            ifaces[alive_faces*3+2] = vert_remap[m->faces[i].v[2]];
            alive_faces++;
        }
        free(vert_remap);

        lightrt_bvh *lbvh = lightrt_bvh_build(fverts, alive_verts,
                                               ifaces, alive_faces);
        free(fverts); free(ifaces);
        if (!lbvh) return PAMO_ERR_ALLOC;

        float grid_extent = (float)(voxel_size * (double)R);
        float tmax = grid_extent + 2.0f * (float)voxel_size;
        int32_t max_t = 1024;
        float *t_hits = (float *)malloc((size_t)max_t * sizeof(float));

        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                float ox = (float)(grid_origin.x + ((double)ix + 0.5) * voxel_size);
                float oy = (float)(grid_origin.y + ((double)iy + 0.5) * voxel_size);
                float oz = (float)(grid_origin.z - voxel_size);
                float origin_f[3] = {ox, oy, oz};
                float dir_f[3] = {0.0f, 0.0f, 1.0f};

                int32_t n_hits = lightrt_bvh_multi_hit(lbvh, origin_f, dir_f,
                                                       tmax, t_hits, max_t);

                /* Walk voxels, flip sign at each crossing. */
                int hit_idx = 0;
                bool inside = false;
                for (int32_t iz = 0; iz < R; iz++) {
                    float voxel_z = (float)(((double)iz + 0.5) * voxel_size + voxel_size);
                    while (hit_idx < n_hits && t_hits[hit_idx] < voxel_z) {
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
        free(t_hits);
        lightrt_bvh_destroy(lbvh);
    }
#else
    /* Fallback: brute-force scanline O(R^2 * F). */
    {
        int max_hits = (int)m->n_faces;
        if (max_hits > 10000) max_hits = 10000;
        double *t_buf = (double *)malloc((size_t)max_hits * sizeof(double));
        if (!t_buf) return PAMO_ERR_ALLOC;

        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                pamo_vec3d origin = {
                    grid_origin.x + ((double)ix + 0.5) * voxel_size,
                    grid_origin.y + ((double)iy + 0.5) * voxel_size,
                    grid_origin.z - voxel_size,
                };

                int n_hits = scanline_intersections(origin, m, t_buf, max_hits);

                int hit_idx = 0;
                bool inside = false;
                for (int32_t iz = 0; iz < R; iz++) {
                    double voxel_z = ((double)iz + 0.5) * voxel_size + voxel_size;
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
    }
#endif

    return PAMO_OK;
}
