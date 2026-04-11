/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle mesh -> signed distance field on a 3D grid.
 *
 * Phase 1: Unsigned distance (BVH nearest-point, narrow-band optimized)
 * Phase 2: Sign determination (Z-axis scanline parity ray casting)
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

/* ── Grid indexing ───────────────────────────────────────────────── */

#define GRID_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

/* ── Lightrt BVH helper ──────────────────────────────────────────── */

#ifdef PAMO_USE_LIGHTRT
/* Build a lightrt BVH from a pamo mesh. Caller must free returned BVH
 * with lightrt_bvh_destroy(). Returns NULL on failure. */
static lightrt_bvh *build_lightrt_from_mesh(const pamo_mesh *m) {
    /* Overflow check. */
    if (m->n_verts > SIZE_MAX / (3 * sizeof(float)) ||
        m->n_faces > SIZE_MAX / (3 * sizeof(int32_t))) {
        return NULL;
    }

    float *fverts = (float *)malloc(m->n_verts * 3 * sizeof(float));
    int32_t *ifaces = (int32_t *)malloc(m->n_faces * 3 * sizeof(int32_t));
    int32_t *remap = (int32_t *)malloc(m->n_verts * sizeof(int32_t));
    if (!fverts || !ifaces || !remap) {
        free(fverts); free(ifaces); free(remap);
        return NULL;
    }

    int32_t av = 0, af = 0;
    for (size_t i = 0; i < m->n_verts; i++) {
        if (m->vert_alive[i]) {
            fverts[av*3+0] = (float)m->verts[i].x;
            fverts[av*3+1] = (float)m->verts[i].y;
            fverts[av*3+2] = (float)m->verts[i].z;
            remap[i] = av++;
        } else {
            remap[i] = -1;
        }
    }
    for (size_t i = 0; i < m->n_faces; i++) {
        if (!m->face_alive[i]) continue;
        ifaces[af*3+0] = remap[m->faces[i].v[0]];
        ifaces[af*3+1] = remap[m->faces[i].v[1]];
        ifaces[af*3+2] = remap[m->faces[i].v[2]];
        af++;
    }
    free(remap);

    lightrt_bvh *lbvh = lightrt_bvh_build(fverts, av, ifaces, af);
    free(fverts);
    free(ifaces);
    return lbvh;
}
#endif

/* ── Parallel Phase 1 worker ─────────────────────────────────────── */

#if defined(PAMO_USE_PTHREADS) && defined(PAMO_USE_LIGHTRT)
typedef struct {
    const void *bvh;
    const float *coarse;
    double *grid;
    int32_t R, CR, iz_start, iz_end;
    pamo_vec3d grid_origin;
    double voxel_size;
    float coarse_diag;
} sdf_thread_arg;

static void *sdf_phase1_worker(void *arg) {
    sdf_thread_arg *a = (sdf_thread_arg *)arg;
    float band_thresh = a->coarse_diag * 2.0f;

    for (int32_t iz = a->iz_start; iz < a->iz_end; iz++) {
        int32_t cz = iz / 4;
        if (cz >= a->CR) cz = a->CR - 1;
        for (int32_t iy = 0; iy < a->R; iy++) {
            int32_t cy = iy / 4;
            if (cy >= a->CR) cy = a->CR - 1;
            for (int32_t ix = 0; ix < a->R; ix++) {
                int32_t cx = ix / 4;
                if (cx >= a->CR) cx = a->CR - 1;

                float cd = 1e30f;
                if (a->coarse)
                    cd = a->coarse[(size_t)cz*a->CR*a->CR + (size_t)cy*a->CR + cx];

                if (cd > band_thresh) {
                    a->grid[GRID_IDX(ix, iy, iz, a->R)] = (double)cd;
                } else {
                    float p[3] = {
                        (float)(a->grid_origin.x + ((double)ix + 0.5) * a->voxel_size),
                        (float)(a->grid_origin.y + ((double)iy + 0.5) * a->voxel_size),
                        (float)(a->grid_origin.z + ((double)iz + 0.5) * a->voxel_size),
                    };
                    float bound = (cd + a->coarse_diag);
                    bound = bound * bound;
                    float dsq = lightrt_bvh_nearest_bounded(
                        (const lightrt_bvh *)a->bvh, p, bound);
                    a->grid[GRID_IDX(ix, iy, iz, a->R)] = (double)sqrtf(dsq);
                }
            }
        }
    }
    return NULL;
}
#endif

/* ── Ray-triangle intersection (for fallback scanline) ───────────── */

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

/* Scanline: collect sorted intersection t-values along +Z. */
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
        if (t >= 0.0 && n < max_t) t_buf[n++] = t;
    }
    /* Insertion sort (few hits typically). */
    for (int i = 1; i < n; i++) {
        double key = t_buf[i];
        int j = i - 1;
        while (j >= 0 && t_buf[j] > key) { t_buf[j+1] = t_buf[j]; j--; }
        t_buf[j+1] = key;
    }
    return n;
}

/* ── Main SDF computation ────────────────────────────────────────── */

pamo_error pamo_compute_sdf(double *grid, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh) {
    if (!grid || R < 2 || !m) return PAMO_ERR_INVALID_ARG;

    /* ── Phase 1: Unsigned distances ─────────────────────────────── */
#ifdef PAMO_USE_LIGHTRT
    {
        lightrt_bvh *lbvh = build_lightrt_from_mesh(m);
        if (!lbvh) return PAMO_ERR_ALLOC;

        /* Coarse grid for narrow-band optimization. */
        int32_t CR = R / 4;
        if (CR < 2) CR = 2;
        double cvoxel = voxel_size * 4.0;
        float coarse_diag = (float)(cvoxel * 1.733);
        size_t csize = (size_t)CR * (size_t)CR * (size_t)CR;
        float *coarse = (float *)malloc(csize * sizeof(float));
        if (coarse) {
            for (int32_t cz = 0; cz < CR; cz++)
                for (int32_t cy = 0; cy < CR; cy++)
                    for (int32_t cx = 0; cx < CR; cx++) {
                        float p[3] = {
                            (float)(grid_origin.x + ((double)cx + 0.5) * cvoxel),
                            (float)(grid_origin.y + ((double)cy + 0.5) * cvoxel),
                            (float)(grid_origin.z + ((double)cz + 0.5) * cvoxel),
                        };
                        coarse[(size_t)cz*CR*CR + (size_t)cy*CR + cx] =
                            sqrtf(lightrt_bvh_nearest_dist_sq(lbvh, p));
                    }
        }

#ifdef PAMO_USE_PTHREADS
        int n_threads = 8;
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        sdf_thread_arg *args = (sdf_thread_arg *)malloc((size_t)n_threads * sizeof(sdf_thread_arg));
        if (threads && args) {
            int32_t per = (R + n_threads - 1) / n_threads;
            for (int t = 0; t < n_threads; t++) {
                args[t] = (sdf_thread_arg){
                    .bvh = lbvh, .coarse = coarse, .grid = grid,
                    .R = R, .CR = CR,
                    .iz_start = t * per,
                    .iz_end = (t+1)*per > R ? R : (t+1)*per,
                    .grid_origin = grid_origin,
                    .voxel_size = voxel_size,
                    .coarse_diag = coarse_diag,
                };
                pthread_create(&threads[t], NULL, sdf_phase1_worker, &args[t]);
            }
            for (int t = 0; t < n_threads; t++)
                pthread_join(threads[t], NULL);
        }
        free(threads);
        free(args);
#else
        float band_thresh = coarse_diag * 2.0f;
        for (int32_t iz = 0; iz < R; iz++) {
            int32_t cz = iz/4; if (cz >= CR) cz = CR-1;
            for (int32_t iy = 0; iy < R; iy++) {
                int32_t cy = iy/4; if (cy >= CR) cy = CR-1;
                for (int32_t ix = 0; ix < R; ix++) {
                    int32_t cx = ix/4; if (cx >= CR) cx = CR-1;
                    float cd = coarse ? coarse[(size_t)cz*CR*CR + (size_t)cy*CR + cx] : 1e30f;
                    if (cd > band_thresh) {
                        grid[GRID_IDX(ix, iy, iz, R)] = (double)cd;
                    } else {
                        float p[3] = {
                            (float)(grid_origin.x + ((double)ix+0.5)*voxel_size),
                            (float)(grid_origin.y + ((double)iy+0.5)*voxel_size),
                            (float)(grid_origin.z + ((double)iz+0.5)*voxel_size),
                        };
                        float bound = (cd + coarse_diag) * (cd + coarse_diag);
                        grid[GRID_IDX(ix, iy, iz, R)] =
                            (double)sqrtf(lightrt_bvh_nearest_bounded(lbvh, p, bound));
                    }
                }
            }
        }
#endif
        free(coarse);
        lightrt_bvh_destroy(lbvh);
    }
#else
    /* Fallback: pamo BVH nearest-point queries. */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
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
#endif

    /* ── Phase 2: Sign via Z-scanline parity ─────────────────────── */
#ifdef PAMO_USE_LIGHTRT
    {
        lightrt_bvh *lbvh = build_lightrt_from_mesh(m);
        if (!lbvh) return PAMO_ERR_ALLOC;

        float tmax = (float)(voxel_size * (double)R) + 2.0f * (float)voxel_size;
        int32_t max_t = 1024;
        float *t_hits = (float *)malloc((size_t)max_t * sizeof(float));
        if (!t_hits) { lightrt_bvh_destroy(lbvh); return PAMO_ERR_ALLOC; }

        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                float o[3] = {
                    (float)(grid_origin.x + ((double)ix + 0.5) * voxel_size),
                    (float)(grid_origin.y + ((double)iy + 0.5) * voxel_size),
                    (float)(grid_origin.z - voxel_size),
                };
                float dir[3] = {0.0f, 0.0f, 1.0f};
                int32_t n = lightrt_bvh_multi_hit(lbvh, o, dir, tmax,
                                                   t_hits, max_t);
                int hit_idx = 0;
                bool inside = false;
                for (int32_t iz = 0; iz < R; iz++) {
                    float vz = (float)(((double)iz + 0.5) * voxel_size + voxel_size);
                    while (hit_idx < n && t_hits[hit_idx] < vz) {
                        inside = !inside;
                        hit_idx++;
                    }
                    if (inside) grid[GRID_IDX(ix, iy, iz, R)] = -grid[GRID_IDX(ix, iy, iz, R)];
                }
            }
        }
        free(t_hits);
        lightrt_bvh_destroy(lbvh);
    }
#else
    /* Fallback: brute-force scanline. */
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
                int n = scanline_intersections(origin, m, t_buf, max_hits);
                int hit_idx = 0;
                bool inside = false;
                for (int32_t iz = 0; iz < R; iz++) {
                    double vz = ((double)iz + 0.5) * voxel_size + voxel_size;
                    while (hit_idx < n && t_buf[hit_idx] < vz) {
                        inside = !inside;
                        hit_idx++;
                    }
                    if (inside) grid[GRID_IDX(ix, iy, iz, R)] = -grid[GRID_IDX(ix, iy, iz, R)];
                }
            }
        }
        free(t_buf);
    }
#endif

    return PAMO_OK;
}
