/* POSIX feature macro: required for clock_gettime / CLOCK_MONOTONIC on glibc. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Triangle mesh -> signed distance field on a 3D grid.
 *
 * Algorithm ported from cumesh2sdf (Apache 2.0):
 *   Phase 1 (Rasterization): Compute unsigned distance for near-surface
 *     voxels only. Far voxels get large default (1e9). Uses BVH queries
 *     within a narrow band.
 *   Phase 2 (Sign): 3-axis ray collision flags + union-find flood fill.
 *     Voxels connected to grid boundary are outside (positive).
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_alloc.h"
#ifdef PAMO_USE_LIGHTRT
#include "pamo/pamo_lightrt.h"
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#ifdef PAMO_USE_PTHREADS
#include <unistd.h>
#endif
#ifdef PAMO_USE_PTHREADS
#include <pthread.h>
#endif

#define GRID_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

/* Thresholds matching cumesh2sdf. */
#define SDF_PRESCAN_COEFF 0.87f
#define SDF_RAY_EPS       1e-6f
#define SDF_MAX_R         1024

/* ── Union-Find ──────────────────────────────────────────────────── */

static int32_t uf_find(int32_t *parent, int32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static void uf_union(int32_t *parent, int32_t a, int32_t b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a != b) parent[a > b ? a : b] = (a > b ? b : a);
}

/* ── Lightrt helper ──────────────────────────────────────────────── */

#ifdef PAMO_USE_LIGHTRT
static lightrt_bvh *build_lightrt_from_mesh(const pamo_mesh *m) {
    if (m->n_verts > SIZE_MAX / (3 * sizeof(float)) ||
        m->n_faces > SIZE_MAX / (3 * sizeof(int32_t)))
        return NULL;
    float *fv = (float *)malloc(m->n_verts * 3 * sizeof(float));
    int32_t *fi = (int32_t *)malloc(m->n_faces * 3 * sizeof(int32_t));
    int32_t *rm = (int32_t *)malloc(m->n_verts * sizeof(int32_t));
    if (!fv || !fi || !rm) { free(fv); free(fi); free(rm); return NULL; }
    int32_t av = 0, af = 0;
    for (size_t i = 0; i < m->n_verts; i++) {
        if (m->vert_alive[i]) {
            fv[av*3]=   (float)m->verts[i].x;
            fv[av*3+1]= (float)m->verts[i].y;
            fv[av*3+2]= (float)m->verts[i].z;
            rm[i] = av++;
        } else rm[i] = -1;
    }
    for (size_t i = 0; i < m->n_faces; i++) {
        if (!m->face_alive[i]) continue;
        if (!pamo_mesh_face_is_valid(m, i)) {
            free(rm); free(fv); free(fi);
            return NULL;
        }
        fi[af*3]   = rm[m->faces[i].v[0]];
        fi[af*3+1] = rm[m->faces[i].v[1]];
        fi[af*3+2] = rm[m->faces[i].v[2]];
        if (fi[af*3] < 0 || fi[af*3+1] < 0 || fi[af*3+2] < 0) {
            free(rm); free(fv); free(fi);
            return NULL;
        }
        af++;
    }
    free(rm);
    lightrt_bvh *b = lightrt_bvh_build(fv, av, fi, af);
    free(fv); free(fi);
    return b;
}
#endif

/* ── Parallel Phase 1 worker ─────────────────────────────────────── */

#ifdef PAMO_USE_PTHREADS
static int pamo_sdf_thread_count(int32_t max_tasks) {
    int nt = 0;
    const char *ev = getenv("PAMO_NUM_THREADS");
    if (ev && ev[0]) nt = atoi(ev);
    if (nt <= 0) {
        long on = sysconf(_SC_NPROCESSORS_ONLN);
        nt = (on > 0) ? (int)on : 8;
    }
    if (max_tasks > 0 && nt > max_tasks) nt = max_tasks;
    if (nt < 1) nt = 1;
    return nt;
}
#endif

#if defined(PAMO_USE_PTHREADS) && defined(PAMO_USE_LIGHTRT)
typedef struct {
    const void *bvh;
    float *grid;      /* float32 grid for narrow-band */
    const uint8_t *active; /* optional; if non-NULL, skip inactive voxels */
    int32_t R, iz_start, iz_end;
    float grid_ox, grid_oy, grid_oz;
    float voxel_size, band;
} sdf_worker_arg;

static void *sdf_worker(void *arg) {
    sdf_worker_arg *a = (sdf_worker_arg *)arg;
    float band_sq = a->band * a->band;
    for (int32_t iz = a->iz_start; iz < a->iz_end; iz++)
        for (int32_t iy = 0; iy < a->R; iy++)
            for (int32_t ix = 0; ix < a->R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, a->R);
                if (a->active && !a->active[idx]) continue;
                float p[3] = {
                    a->grid_ox + ((float)ix + 0.5f) * a->voxel_size,
                    a->grid_oy + ((float)iy + 0.5f) * a->voxel_size,
                    a->grid_oz + ((float)iz + 0.5f) * a->voxel_size,
                };
                float dsq = lightrt_bvh_nearest_bounded(
                    (const lightrt_bvh *)a->bvh, p, band_sq);
                a->grid[idx] = sqrtf(dsq);
            }
    return NULL;
}
#endif

#ifdef PAMO_USE_PTHREADS
typedef struct {
    const pamo_mesh *mesh;
    const pamo_bvh *bvh;
    float *grid;
    const uint8_t *active;
    int32_t R, iz_start, iz_end;
    double grid_ox, grid_oy, grid_oz;
    double voxel_size;
    float band;
} sdf_fallback_nearest_arg;

static void *sdf_fallback_nearest_worker(void *arg) {
    sdf_fallback_nearest_arg *a = (sdf_fallback_nearest_arg *)arg;
    const double band = (double)a->band;
    for (int32_t iz = a->iz_start; iz < a->iz_end; iz++)
        for (int32_t iy = 0; iy < a->R; iy++)
            for (int32_t ix = 0; ix < a->R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, a->R);
                if (a->active && !a->active[idx]) continue;
                pamo_vec3d p = {
                    a->grid_ox + ((double)ix + 0.5) * a->voxel_size,
                    a->grid_oy + ((double)iy + 0.5) * a->voxel_size,
                    a->grid_oz + ((double)iz + 0.5) * a->voxel_size,
                };
                pamo_nearest_result res;
                if (pamo_bvh_nearest(a->bvh, a->mesh, p, &res) != PAMO_OK ||
                    res.prim_id < 0) {
                    continue;
                }
                double d = sqrt(res.dist_sq);
                if (d < band) a->grid[idx] = (float)d;
            }
    return NULL;
}
#endif

typedef struct {
    int32_t ixlo, ixhi;
    int32_t iylo, iyhi;
    int32_t izlo, izhi;
} sdf_grid_bounds;

static int sdf_face_grid_bounds(const pamo_mesh *m, size_t fi,
                                pamo_vec3d grid_origin, double voxel_size,
                                int32_t R, double band,
                                int32_t iz_start, int32_t iz_end,
                                sdf_grid_bounds *out) {
    if (!m || !out || fi >= m->n_faces || !m->face_alive[fi])
        return 0;
    if (!pamo_mesh_face_is_valid(m, fi))
        return 0;
    const int32_t *fv = m->faces[fi].v;
    double v0x=m->verts[fv[0]].x, v0y=m->verts[fv[0]].y, v0z=m->verts[fv[0]].z;
    double v1x=m->verts[fv[1]].x, v1y=m->verts[fv[1]].y, v1z=m->verts[fv[1]].z;
    double v2x=m->verts[fv[2]].x, v2y=m->verts[fv[2]].y, v2z=m->verts[fv[2]].z;
    double xlo = fmin(fmin(v0x,v1x),v2x) - band;
    double xhi = fmax(fmax(v0x,v1x),v2x) + band;
    double ylo = fmin(fmin(v0y,v1y),v2y) - band;
    double yhi = fmax(fmax(v0y,v1y),v2y) + band;
    double zlo = fmin(fmin(v0z,v1z),v2z) - band;
    double zhi = fmax(fmax(v0z,v1z),v2z) + band;
    int32_t ixlo = (int32_t)floor((xlo - grid_origin.x) / voxel_size);
    int32_t ixhi = (int32_t)ceil ((xhi - grid_origin.x) / voxel_size);
    int32_t iylo = (int32_t)floor((ylo - grid_origin.y) / voxel_size);
    int32_t iyhi = (int32_t)ceil ((yhi - grid_origin.y) / voxel_size);
    int32_t izlo = (int32_t)floor((zlo - grid_origin.z) / voxel_size);
    int32_t izhi = (int32_t)ceil ((zhi - grid_origin.z) / voxel_size);
    if (ixlo < 0) ixlo = 0;
    if (ixhi > R - 1) ixhi = R - 1;
    if (iylo < 0) iylo = 0;
    if (iyhi > R - 1) iyhi = R - 1;
    if (izlo < 0) izlo = 0;
    if (izhi > R - 1) izhi = R - 1;
    if (izlo < iz_start) izlo = iz_start;
    if (izhi >= iz_end) izhi = iz_end - 1;
    if (ixlo > ixhi || iylo > iyhi || izlo > izhi)
        return 0;
    out->ixlo = ixlo; out->ixhi = ixhi;
    out->iylo = iylo; out->iyhi = iyhi;
    out->izlo = izlo; out->izhi = izhi;
    return 1;
}

typedef struct {
    const pamo_mesh *mesh;
    float *grid;
    int32_t R, iz_start, iz_end;
    pamo_vec3d grid_origin;
    double voxel_size;
    double band;
} sdf_raster_distance_arg;

static void sdf_rasterize_distance_range(const pamo_mesh *m, float *grid,
                                         int32_t R,
                                         pamo_vec3d grid_origin,
                                         double voxel_size, double band,
                                         int32_t iz_start, int32_t iz_end) {
    const double band_sq = band * band;
    for (size_t fi = 0; fi < m->n_faces; fi++) {
        sdf_grid_bounds gb;
        if (!sdf_face_grid_bounds(m, fi, grid_origin, voxel_size, R, band,
                                  iz_start, iz_end, &gb)) {
            continue;
        }
        const int32_t *fv = m->faces[fi].v;
        pamo_vec3d v0 = m->verts[fv[0]];
        pamo_vec3d v1 = m->verts[fv[1]];
        pamo_vec3d v2 = m->verts[fv[2]];
        for (int32_t iz = gb.izlo; iz <= gb.izhi; iz++) {
            double pz = grid_origin.z + ((double)iz + 0.5) * voxel_size;
            for (int32_t iy = gb.iylo; iy <= gb.iyhi; iy++) {
                double py = grid_origin.y + ((double)iy + 0.5) * voxel_size;
                for (int32_t ix = gb.ixlo; ix <= gb.ixhi; ix++) {
                    size_t idx = GRID_IDX(ix, iy, iz, R);
                    float old = grid[idx];
                    double old_sq = (double)old * (double)old;
                    if (old_sq <= 0.0 || old_sq <= band_sq * 1.0e-12)
                        continue;
                    pamo_vec3d p = {
                        grid_origin.x + ((double)ix + 0.5) * voxel_size,
                        py,
                        pz,
                    };
                    double dsq;
                    pamo_closest_point_on_tri(p, v0, v1, v2, &dsq);
                    if (dsq < band_sq && dsq < old_sq) {
                        grid[idx] = (float)sqrt(dsq);
                    }
                }
            }
        }
    }
}

#ifdef PAMO_USE_PTHREADS
static void *sdf_raster_distance_worker(void *arg) {
    sdf_raster_distance_arg *a = (sdf_raster_distance_arg *)arg;
    sdf_rasterize_distance_range(a->mesh, a->grid, a->R, a->grid_origin,
                                 a->voxel_size, a->band,
                                 a->iz_start, a->iz_end);
    return NULL;
}
#endif

/* ── Ray-triangle hit (for fallback) ─────────────────────────────── */

static float ray_tri_hitf(float ox, float oy, float oz,
                          float dx, float dy, float dz,
                          float v0x, float v0y, float v0z,
                          float v1x, float v1y, float v1z,
                          float v2x, float v2y, float v2z) {
    float e1x=v1x-v0x, e1y=v1y-v0y, e1z=v1z-v0z;
    float e2x=v2x-v0x, e2y=v2y-v0y, e2z=v2z-v0z;
    float hx=dy*e2z-dz*e2y, hy=dz*e2x-dx*e2z, hz=dx*e2y-dy*e2x;
    float a = e1x*hx + e1y*hy + e1z*hz;
    if (fabsf(a) < 1e-12f) return -1.0f;
    float f = 1.0f/a;
    float sx=ox-v0x, sy=oy-v0y, sz=oz-v0z;
    float u = f*(sx*hx+sy*hy+sz*hz);
    if (u < -1e-6f || u > 1.0f+1e-6f) return -1.0f;
    float qx=sy*e1z-sz*e1y, qy=sz*e1x-sx*e1z, qz=sx*e1y-sy*e1x;
    float v = f*(dx*qx+dy*qy+dz*qz);
    if (v < -1e-6f || u+v > 1.0f+1e-6f) return -1.0f;
    float t = f*(e2x*qx+e2y*qy+e2z*qz);
    return t > 1e-6f ? t : -1.0f;
}

/* ── Parallel Phase 2 ray workers ────────────────────────────────── */

#if defined(PAMO_USE_PTHREADS) && defined(PAMO_USE_LIGHTRT)
typedef struct {
    const void *bvh;
    uint8_t *collide;
    const float *dist;
    int32_t R, iz_start, iz_end;
    float grid_ox, grid_oy, grid_oz;
    float voxel_size, rayth;
} sdf_lightrt_rays_arg;

static void *sdf_lightrt_rays_worker(void *arg) {
    sdf_lightrt_rays_arg *a = (sdf_lightrt_rays_arg *)arg;
    const lightrt_bvh *bvh = (const lightrt_bvh *)a->bvh;
    for (int32_t iz = a->iz_start; iz < a->iz_end; iz++)
        for (int32_t iy = 0; iy < a->R; iy++)
            for (int32_t ix = 0; ix < a->R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, a->R);
                if (a->dist[idx] > a->rayth) continue;
                float px = a->grid_ox + ((float)ix + 0.5f) * a->voxel_size;
                float py = a->grid_oy + ((float)iy + 0.5f) * a->voxel_size;
                float pz = a->grid_oz + ((float)iz + 0.5f) * a->voxel_size;
                float o[3] = {px, py, pz};
                float d[3] = {1.0f, 0.0f, 0.0f};
                if (lightrt_bvh_any_hit(bvh, o, d, a->rayth)) a->collide[idx*3] = 1;
                d[0]=0.0f; d[1]=1.0f; d[2]=0.0f;
                if (lightrt_bvh_any_hit(bvh, o, d, a->rayth)) a->collide[idx*3+1] = 1;
                d[0]=0.0f; d[1]=0.0f; d[2]=1.0f;
                if (lightrt_bvh_any_hit(bvh, o, d, a->rayth)) a->collide[idx*3+2] = 1;
            }
    return NULL;
}
#endif

#ifdef PAMO_USE_PTHREADS
typedef struct {
    const pamo_mesh *mesh;
    const pamo_bvh *bvh;
    uint8_t *collide;
    const float *dist;
    int32_t R, iz_start, iz_end;
    float grid_ox, grid_oy, grid_oz;
    float voxel_size, rayth;
} sdf_fallback_rays_arg;

static void pamo_sdf_test_candidate_tri(const pamo_mesh *m, int32_t fi,
                                        float px, float py, float pz,
                                        float rayth, uint8_t *flags) {
    if (fi < 0 || (size_t)fi >= m->n_faces) return;
    if (!m->face_alive[fi]) return;
    const int32_t *fv = m->faces[fi].v;
    float v0x=(float)m->verts[fv[0]].x;
    float v0y=(float)m->verts[fv[0]].y;
    float v0z=(float)m->verts[fv[0]].z;
    float v1x=(float)m->verts[fv[1]].x;
    float v1y=(float)m->verts[fv[1]].y;
    float v1z=(float)m->verts[fv[1]].z;
    float v2x=(float)m->verts[fv[2]].x;
    float v2y=(float)m->verts[fv[2]].y;
    float v2z=(float)m->verts[fv[2]].z;
    float t;
    if (!flags[0]) {
        t = ray_tri_hitf(px,py,pz, 1,0,0, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z);
        if (t >= 0.0f && t <= rayth) flags[0] = 1;
    }
    if (!flags[1]) {
        t = ray_tri_hitf(px,py,pz, 0,1,0, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z);
        if (t >= 0.0f && t <= rayth) flags[1] = 1;
    }
    if (!flags[2]) {
        t = ray_tri_hitf(px,py,pz, 0,0,1, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z);
        if (t >= 0.0f && t <= rayth) flags[2] = 1;
    }
}

static void *sdf_fallback_rays_worker(void *arg) {
    sdf_fallback_rays_arg *a = (sdf_fallback_rays_arg *)arg;
    pamo_overlap_result ov;
    pamo_overlap_result_init(&ov, &a->mesh->alloc);
    for (int32_t iz = a->iz_start; iz < a->iz_end; iz++)
        for (int32_t iy = 0; iy < a->R; iy++)
            for (int32_t ix = 0; ix < a->R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, a->R);
                if (a->dist[idx] > a->rayth) continue;
                float px = a->grid_ox + ((float)ix + 0.5f) * a->voxel_size;
                float py = a->grid_oy + ((float)iy + 0.5f) * a->voxel_size;
                float pz = a->grid_oz + ((float)iz + 0.5f) * a->voxel_size;

                pamo_aabb box;
                double eps = (double)a->rayth + 1e-6;
                box.lo.x = (double)px - eps;
                box.lo.y = (double)py - eps;
                box.lo.z = (double)pz - eps;
                box.hi.x = (double)px + eps;
                box.hi.y = (double)py + eps;
                box.hi.z = (double)pz + eps;
                if (pamo_bvh_overlap(a->bvh, box, &ov) != PAMO_OK) continue;

                uint8_t *flags = &a->collide[idx*3];
                for (size_t hi = 0; hi < ov.n_hits; hi++) {
                    pamo_sdf_test_candidate_tri(a->mesh, ov.hits[hi],
                                                px, py, pz, a->rayth, flags);
                    if (flags[0] && flags[1] && flags[2]) break;
                }
            }
    pamo_overlap_result_destroy(&ov);
    return NULL;
}
#endif

/* ── Main SDF computation ────────────────────────────────────────── */

/* Phase profiling: set PAMO_SDF_PROFILE=1 to print per-phase wall time. */
static int pamo_sdf_profile_on(void) {
    static int checked = 0, on = 0;
    if (!checked) { const char *e = getenv("PAMO_SDF_PROFILE");
                    on = (e && e[0] == '1'); checked = 1; }
    return on;
}
static double pamo_sdf_now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int pamo_sdf_raster_phase1_on(void) {
    const char *e = getenv("PAMO_SDF_RASTER_PHASE1");
    return !(e && e[0] == '0');
}

pamo_error pamo_compute_sdf(double *grid_out, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh) {
    if (!grid_out || R < 2 || !m) return PAMO_ERR_INVALID_ARG;
    int prof = pamo_sdf_profile_on();
    double t0 = prof ? pamo_sdf_now_s() : 0.0;
    double tprev = t0;
    #define PAMO_SDF_TICK(label) do { \
        if (prof) { double _t = pamo_sdf_now_s(); \
            fprintf(stderr, "[sdf-prof] %-22s %.3f s\n", (label), _t - tprev); \
            tprev = _t; } \
    } while (0)

    /* Guard against integer overflow: R^3 * sizeof(float) must fit in size_t.
     * R=1290 gives 1290^3*4 = ~8.6 GB. Cap at 1024 for safety. */
    if (R > SDF_MAX_R) return PAMO_ERR_INVALID_ARG;

    size_t grid_size = (size_t)R * (size_t)R * (size_t)R;
    float vs = (float)voxel_size;
    float gox = (float)grid_origin.x, goy = (float)grid_origin.y, goz = (float)grid_origin.z;

    /* Band: 0.87/N + 3/N (matching cumesh2sdf threshold). */
    float band = (0.87f + 3.0f) / (float)R * vs * (float)R;

    /* Allocate float32 working grid (distance + collision flags). */
    float *dist = (float *)malloc(grid_size * sizeof(float));
    uint8_t *collide = (uint8_t *)calloc(grid_size, 3); /* 3 flags per voxel */
    if (!dist || !collide) { free(dist); free(collide); return PAMO_ERR_ALLOC; }

    /* Initialize distance to large value. */
    for (size_t i = 0; i < grid_size; i++) dist[i] = 1e9f;
    PAMO_SDF_TICK("alloc+init");

    /* ── Active-voxel mask (hierarchical band clip) ───────────────
     * Only voxels whose grid cell intersects a triangle's AABB expanded
     * by `band` can ever have dist < band. All others are guaranteed
     * "far-field" at 1e9f, which is safely ≥ prescan_thresh in Phase 3
     * and >rayth in Phase 2.
     *
     * At R=512 on a thin mechanical shell this reduces the Phase 1
     * BVH-nearest call count from R^3 (134M) to roughly R^2 × band/vs
     * (a few M), a large-factor reduction in the dominant Stage 1 cost.
     * The mask itself costs O(sum_tri voxels_in_tri_AABB). */
    uint8_t *active = (uint8_t *)calloc(grid_size, 1);
    if (!active) { free(dist); free(collide); return PAMO_ERR_ALLOC; }
    {
        double band_d = (double)band;
        for (size_t fi = 0; fi < m->n_faces; fi++) {
            if (!m->face_alive[fi]) continue;
            if (!pamo_mesh_face_is_valid(m, fi)) {
                free(dist); free(collide); free(active);
                return PAMO_ERR_INVALID_ARG;
            }
            const int32_t *fv = m->faces[fi].v;
            double v0x=m->verts[fv[0]].x, v0y=m->verts[fv[0]].y, v0z=m->verts[fv[0]].z;
            double v1x=m->verts[fv[1]].x, v1y=m->verts[fv[1]].y, v1z=m->verts[fv[1]].z;
            double v2x=m->verts[fv[2]].x, v2y=m->verts[fv[2]].y, v2z=m->verts[fv[2]].z;
            double xlo = fmin(fmin(v0x,v1x),v2x) - band_d;
            double xhi = fmax(fmax(v0x,v1x),v2x) + band_d;
            double ylo = fmin(fmin(v0y,v1y),v2y) - band_d;
            double yhi = fmax(fmax(v0y,v1y),v2y) + band_d;
            double zlo = fmin(fmin(v0z,v1z),v2z) - band_d;
            double zhi = fmax(fmax(v0z,v1z),v2z) + band_d;
            int32_t ixlo = (int32_t)floor((xlo - grid_origin.x) / voxel_size);
            int32_t ixhi = (int32_t)ceil ((xhi - grid_origin.x) / voxel_size);
            int32_t iylo = (int32_t)floor((ylo - grid_origin.y) / voxel_size);
            int32_t iyhi = (int32_t)ceil ((yhi - grid_origin.y) / voxel_size);
            int32_t izlo = (int32_t)floor((zlo - grid_origin.z) / voxel_size);
            int32_t izhi = (int32_t)ceil ((zhi - grid_origin.z) / voxel_size);
            if (ixlo < 0) ixlo = 0;
            if (ixhi > R - 1) ixhi = R - 1;
            if (iylo < 0) iylo = 0;
            if (iyhi > R - 1) iyhi = R - 1;
            if (izlo < 0) izlo = 0;
            if (izhi > R - 1) izhi = R - 1;
            for (int32_t iz = izlo; iz <= izhi; iz++)
                for (int32_t iy = iylo; iy <= iyhi; iy++)
                    for (int32_t ix = ixlo; ix <= ixhi; ix++)
                        active[GRID_IDX(ix, iy, iz, R)] = 1;
        }
    }
    PAMO_SDF_TICK("active-mask");

    /* ── BVH backend selection ───────────────────────────────────────
     * Runtime toggle between the vendored lightrt BVH (faster, tighter
     * fit) and pamo's fallback BVH. Callers can force the fallback by
     * setting PAMO_USE_FALLBACK_BVH=1 — useful when the lightrt path's
     * smoother narrow-band distances over-simplify concave features
     * that the looser fallback distances preserve. Requires pamo to be
     * compiled with PAMO_USE_LIGHTRT; otherwise the fallback is the
     * only option. */
    int use_lightrt = 0;
#ifdef PAMO_USE_LIGHTRT
    lightrt_bvh *lbvh = NULL;
    {
        const char *e = getenv("PAMO_USE_FALLBACK_BVH");
        use_lightrt = !(e && e[0] == '1');
    }
    if (use_lightrt) {
        lbvh = build_lightrt_from_mesh(m);
        PAMO_SDF_TICK("lightrt-bvh-build");
        if (!lbvh) use_lightrt = 0;  /* graceful fallback on alloc failure */
    }
#endif
    if (prof) {
        fprintf(stderr, "[sdf-prof] %-22s %s\n",
                "sdf-backend", use_lightrt ? "lightrt" : "pamo-fallback");
    }

    /* ── Phase 1: Narrow-band distance computation ───────────────── */
    if (use_lightrt) {
#ifdef PAMO_USE_LIGHTRT
#ifdef PAMO_USE_PTHREADS
        int nt = pamo_sdf_thread_count(R);
        if (prof) fprintf(stderr, "[sdf-prof] %-22s %d\n", "pamo-threads", nt);
        pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        sdf_worker_arg *args = (sdf_worker_arg *)malloc((size_t)nt * sizeof(sdf_worker_arg));
        if (thr && args) {
            int32_t per = (R + nt - 1) / nt;
            int created = 0;
            for (int t = 0; t < nt; t++) {
                args[t] = (sdf_worker_arg){
                    .bvh = lbvh, .grid = dist, .active = active, .R = R,
                    .iz_start = t*per, .iz_end = (t+1)*per > R ? R : (t+1)*per,
                    .grid_ox = gox, .grid_oy = goy, .grid_oz = goz,
                    .voxel_size = vs, .band = band,
                };
                if (pthread_create(&thr[t], NULL, sdf_worker, &args[t]) == 0)
                    created++;
                else
                    break;
            }
            for (int t = 0; t < created; t++) pthread_join(thr[t], NULL);
            if (created != nt) {
                float band_sq = band * band;
                for (int32_t iz = 0; iz < R; iz++)
                    for (int32_t iy = 0; iy < R; iy++)
                        for (int32_t ix = 0; ix < R; ix++) {
                            size_t idx = GRID_IDX(ix,iy,iz,R);
                            if (!active[idx]) continue;
                            float p[3] = {gox+((float)ix+0.5f)*vs, goy+((float)iy+0.5f)*vs, goz+((float)iz+0.5f)*vs};
                            dist[idx] = sqrtf(lightrt_bvh_nearest_bounded(lbvh, p, band_sq));
                        }
            }
        } else {
            float band_sq = band * band;
            for (int32_t iz = 0; iz < R; iz++)
                for (int32_t iy = 0; iy < R; iy++)
                    for (int32_t ix = 0; ix < R; ix++) {
                        size_t idx = GRID_IDX(ix,iy,iz,R);
                        if (!active[idx]) continue;
                        float p[3] = {gox+((float)ix+0.5f)*vs, goy+((float)iy+0.5f)*vs, goz+((float)iz+0.5f)*vs};
                        dist[idx] = sqrtf(lightrt_bvh_nearest_bounded(lbvh, p, band_sq));
                    }
        }
        free(thr); free(args);
#else
        float band_sq = band * band;
        for (int32_t iz = 0; iz < R; iz++)
            for (int32_t iy = 0; iy < R; iy++)
                for (int32_t ix = 0; ix < R; ix++) {
                    size_t idx = GRID_IDX(ix,iy,iz,R);
                    if (!active[idx]) continue;
                    float p[3] = {gox+((float)ix+0.5f)*vs, goy+((float)iy+0.5f)*vs, goz+((float)iz+0.5f)*vs};
                    dist[idx] = sqrtf(lightrt_bvh_nearest_bounded(lbvh, p, band_sq));
                }
#endif /* PAMO_USE_PTHREADS */
#endif /* PAMO_USE_LIGHTRT */
    } else {
        /* Fallback: pamo BVH. */
        if (pamo_sdf_raster_phase1_on()) {
            if (prof) fprintf(stderr, "[sdf-prof] %-22s %s\n",
                              "phase1-mode", "triangle-raster");
#ifdef PAMO_USE_PTHREADS
            int nt = pamo_sdf_thread_count(R);
            if (prof) fprintf(stderr, "[sdf-prof] %-22s %d\n", "pamo-threads", nt);
            pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
            sdf_raster_distance_arg *args =
                (sdf_raster_distance_arg *)malloc((size_t)nt * sizeof(sdf_raster_distance_arg));
            int threaded = 0;
            if (thr && args) {
                int32_t per = (R + nt - 1) / nt;
                int created = 0;
                for (int t = 0; t < nt; t++) {
                    args[t] = (sdf_raster_distance_arg){
                        .mesh = m, .grid = dist, .R = R,
                        .iz_start = t*per,
                        .iz_end = (t+1)*per > R ? R : (t+1)*per,
                        .grid_origin = grid_origin,
                        .voxel_size = voxel_size,
                        .band = (double)band,
                    };
                    if (pthread_create(&thr[t], NULL, sdf_raster_distance_worker, &args[t]) == 0)
                        created++;
                    else
                        break;
                }
                for (int t = 0; t < created; t++) pthread_join(thr[t], NULL);
                threaded = (created == nt);
            }
            free(thr); free(args);
            if (threaded) {
                goto pamo_sdf_phase1_done;
            }
#endif
            sdf_rasterize_distance_range(m, dist, R, grid_origin, voxel_size,
                                         (double)band, 0, R);
            goto pamo_sdf_phase1_done;
        }

#ifdef PAMO_USE_PTHREADS
        {
            int nt = pamo_sdf_thread_count(R);
            if (prof) fprintf(stderr, "[sdf-prof] %-22s %d\n", "pamo-threads", nt);
            pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
            sdf_fallback_nearest_arg *args =
                (sdf_fallback_nearest_arg *)malloc((size_t)nt * sizeof(sdf_fallback_nearest_arg));
            if (thr && args) {
                int32_t per = (R + nt - 1) / nt;
                int created = 0;
                for (int t = 0; t < nt; t++) {
                    args[t] = (sdf_fallback_nearest_arg){
                        .mesh = m, .bvh = bvh, .grid = dist, .active = active, .R = R,
                        .iz_start = t*per, .iz_end = (t+1)*per > R ? R : (t+1)*per,
                        .grid_ox = grid_origin.x, .grid_oy = grid_origin.y,
                        .grid_oz = grid_origin.z, .voxel_size = voxel_size,
                        .band = band,
                    };
                    if (pthread_create(&thr[t], NULL, sdf_fallback_nearest_worker, &args[t]) == 0)
                        created++;
                    else
                        break;
                }
                for (int t = 0; t < created; t++) pthread_join(thr[t], NULL);
                if (created == nt) {
                    free(thr); free(args);
                    goto pamo_sdf_phase1_done;
                }
            }
            free(thr); free(args);
        }
#endif
        for (int32_t iz = 0; iz < R; iz++)
            for (int32_t iy = 0; iy < R; iy++)
                for (int32_t ix = 0; ix < R; ix++) {
                    size_t idx = GRID_IDX(ix,iy,iz,R);
                    if (!active[idx]) continue;
                    pamo_vec3d p = {grid_origin.x+((double)ix+0.5)*voxel_size,
                                    grid_origin.y+((double)iy+0.5)*voxel_size,
                                    grid_origin.z+((double)iz+0.5)*voxel_size};
                    pamo_nearest_result res;
                    if (pamo_bvh_nearest(bvh, m, p, &res) != PAMO_OK ||
                        res.prim_id < 0) {
                        continue;
                    }
                    float d = (float)sqrt(res.dist_sq);
                    if (d < band) dist[idx] = d;
                }
    }
pamo_sdf_phase1_done:
    PAMO_SDF_TICK("phase1-distance");

    /* ── Phase 2: Collision flags (3-axis ray tests for near-surface voxels) */
    float rayth = vs + 1e-6f;
    if (use_lightrt) {
#ifdef PAMO_USE_LIGHTRT
        /* Reuse lbvh from Phase 1. */
        if (lbvh) {
#ifdef PAMO_USE_PTHREADS
            int nt = pamo_sdf_thread_count(R);
            pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
            sdf_lightrt_rays_arg *args =
                (sdf_lightrt_rays_arg *)malloc((size_t)nt * sizeof(sdf_lightrt_rays_arg));
            int threaded = 0;
            if (thr && args) {
                int32_t per = (R + nt - 1) / nt;
                int created = 0;
                for (int t = 0; t < nt; t++) {
                    args[t] = (sdf_lightrt_rays_arg){
                        .bvh = lbvh, .collide = collide, .dist = dist, .R = R,
                        .iz_start = t*per, .iz_end = (t+1)*per > R ? R : (t+1)*per,
                        .grid_ox = gox, .grid_oy = goy, .grid_oz = goz,
                        .voxel_size = vs, .rayth = rayth,
                    };
                    if (pthread_create(&thr[t], NULL, sdf_lightrt_rays_worker, &args[t]) == 0)
                        created++;
                    else
                        break;
                }
                for (int t = 0; t < created; t++) pthread_join(thr[t], NULL);
                threaded = (created == nt);
            }
            free(thr); free(args);
            if (!threaded) {
#endif
            for (int32_t iz = 0; iz < R; iz++)
                for (int32_t iy = 0; iy < R; iy++)
                    for (int32_t ix = 0; ix < R; ix++) {
                        size_t idx = GRID_IDX(ix, iy, iz, R);
                        if (dist[idx] > rayth) continue;
                        float px = gox + ((float)ix + 0.5f) * vs;
                        float py = goy + ((float)iy + 0.5f) * vs;
                        float pz = goz + ((float)iz + 0.5f) * vs;
                        float o[3], d[3];
                        /* +X */
                        o[0]=px; o[1]=py; o[2]=pz; d[0]=1; d[1]=0; d[2]=0;
                        if (lightrt_bvh_any_hit(lbvh, o, d, rayth)) collide[idx*3] = 1;
                        /* +Y */
                        d[0]=0; d[1]=1; d[2]=0;
                        if (lightrt_bvh_any_hit(lbvh, o, d, rayth)) collide[idx*3+1] = 1;
                        /* +Z */
                        d[0]=0; d[1]=0; d[2]=1;
                        if (lightrt_bvh_any_hit(lbvh, o, d, rayth)) collide[idx*3+2] = 1;
                    }
#ifdef PAMO_USE_PTHREADS
            }
#endif
            lightrt_bvh_destroy(lbvh);
        }
#endif
    } else {
    /* Fallback: for each near-surface voxel, query the BVH for triangles
     * whose AABB overlaps the voxel (+ rayth margin along +X/+Y/+Z so all
     * three short axis-aligned rays get a candidate list), then ray-test
     * only those triangles instead of the full face list.
     *
     * The old brute-force version was O(near_voxels × n_faces) per SDF,
     * and dominated Stage 1 runtime on medium-sized meshes without the
     * optional lightrt BVH. The pruned version is typically 100–1000×
     * faster: each near-surface voxel touches only a handful of nearby
     * triangles. */
    {
#ifdef PAMO_USE_PTHREADS
        int nt = pamo_sdf_thread_count(R);
        pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        sdf_fallback_rays_arg *args =
            (sdf_fallback_rays_arg *)malloc((size_t)nt * sizeof(sdf_fallback_rays_arg));
        int threaded = 0;
        if (thr && args) {
            int32_t per = (R + nt - 1) / nt;
            int created = 0;
            for (int t = 0; t < nt; t++) {
                args[t] = (sdf_fallback_rays_arg){
                    .mesh = m, .bvh = bvh, .collide = collide, .dist = dist,
                    .R = R, .iz_start = t*per,
                    .iz_end = (t+1)*per > R ? R : (t+1)*per,
                    .grid_ox = gox, .grid_oy = goy, .grid_oz = goz,
                    .voxel_size = vs, .rayth = rayth,
                };
                if (pthread_create(&thr[t], NULL, sdf_fallback_rays_worker, &args[t]) == 0)
                    created++;
                else
                    break;
            }
            for (int t = 0; t < created; t++) pthread_join(thr[t], NULL);
            threaded = (created == nt);
        }
        free(thr); free(args);
        if (!threaded) {
#endif
        pamo_overlap_result ov;
        pamo_overlap_result_init(&ov, &m->alloc);
        for (int32_t iz = 0; iz < R; iz++)
            for (int32_t iy = 0; iy < R; iy++)
                for (int32_t ix = 0; ix < R; ix++) {
                    size_t idx = GRID_IDX(ix, iy, iz, R);
                    if (dist[idx] > rayth) continue;
                    float px = gox + ((float)ix + 0.5f) * vs;
                    float py = goy + ((float)iy + 0.5f) * vs;
                    float pz = goz + ((float)iz + 0.5f) * vs;

                    /* Voxel center expanded by rayth in each axis so the
                     * box covers the three axis-aligned rays we will cast. */
                    pamo_aabb box;
                    double eps = (double)rayth + 1e-6;
                    box.lo.x = (double)px - eps;
                    box.lo.y = (double)py - eps;
                    box.lo.z = (double)pz - eps;
                    box.hi.x = (double)px + eps;
                    box.hi.y = (double)py + eps;
                    box.hi.z = (double)pz + eps;
                    if (pamo_bvh_overlap(bvh, box, &ov) != PAMO_OK) continue;

                    for (size_t hi = 0; hi < ov.n_hits; hi++) {
                        int32_t fi = ov.hits[hi];
                        if (fi < 0 || (size_t)fi >= m->n_faces) continue;
                        if (!m->face_alive[fi]) continue;
                        const int32_t *fv = m->faces[fi].v;
                        float v0x=(float)m->verts[fv[0]].x;
                        float v0y=(float)m->verts[fv[0]].y;
                        float v0z=(float)m->verts[fv[0]].z;
                        float v1x=(float)m->verts[fv[1]].x;
                        float v1y=(float)m->verts[fv[1]].y;
                        float v1z=(float)m->verts[fv[1]].z;
                        float v2x=(float)m->verts[fv[2]].x;
                        float v2y=(float)m->verts[fv[2]].y;
                        float v2z=(float)m->verts[fv[2]].z;
                        float t;
                        if (!(collide[idx*3])) {
                            t = ray_tri_hitf(px,py,pz, 1,0,0, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z);
                            if (t >= 0.0f && t <= rayth) collide[idx*3] = 1;
                        }
                        if (!(collide[idx*3+1])) {
                            t = ray_tri_hitf(px,py,pz, 0,1,0, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z);
                            if (t >= 0.0f && t <= rayth) collide[idx*3+1] = 1;
                        }
                        if (!(collide[idx*3+2])) {
                            t = ray_tri_hitf(px,py,pz, 0,0,1, v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z);
                            if (t >= 0.0f && t <= rayth) collide[idx*3+2] = 1;
                        }
                    }
                }
        pamo_overlap_result_destroy(&ov);
#ifdef PAMO_USE_PTHREADS
        }
#endif
    }
    }
    PAMO_SDF_TICK("phase2-rays");

    /* ── Phase 3: Union-Find sign determination ──────────────────── */
    /* Node 0 = boundary sentinel. Nodes 1..N^3 = voxels. */
    size_t uf_size = grid_size + 1;
    int32_t *parent = (int32_t *)malloc(uf_size * sizeof(int32_t));
    if (!parent) { free(dist); free(collide); free(active); return PAMO_ERR_ALLOC; }
    for (size_t i = 0; i < uf_size; i++) parent[i] = (int32_t)i;

    /* Prescan: walk X-columns, matching cumesh2sdf's
     * volume_sign_prescan_kernel. The `skip` state is important on open
     * meshes: it prevents far-field spans from being chained through
     * narrow ambiguous bands and reduces false inside pockets. */
    float prescan_thresh = SDF_PRESCAN_COEFF / (float)R * vs * (float)R;
    float prescan_skip_thresh = 2.0f * vs;
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++) {
            bool skip = false;
            bool flag = false;
            int32_t chain_start = 0; /* boundary sentinel */
            for (int32_t ix = 0; ix < R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, R);
                int32_t node = (int32_t)(idx + 1);
                if (skip) {
                    skip = false;
                    if (flag) {
                        flag = false;
                        chain_start = node;
                    }
                } else {
                    if (dist[idx] < prescan_thresh) {
                        flag = true;
                        chain_start = node;
                    } else if (flag) {
                        flag = false;
                        chain_start = node;
                    }
                    skip = dist[idx] > prescan_skip_thresh;
                }
                parent[node] = chain_start;
            }
        }

    /* 3D union: link +X/+Y/+Z neighbors if no collision flag separates them. */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, R);
                int32_t node = (int32_t)(idx + 1);

                /* Connect boundary voxels to sentinel (node 0). */
                if (ix == 0 || ix == R-1 || iy == 0 || iy == R-1 ||
                    iz == 0 || iz == R-1) {
                    if (dist[idx] >= prescan_thresh)
                        uf_union(parent, node, 0);
                }

                /* +X neighbor. */
                if (ix < R-1 && !collide[idx*3]) {
                    int32_t nb = (int32_t)(GRID_IDX(ix+1, iy, iz, R) + 1);
                    uf_union(parent, node, nb);
                }
                /* +Y neighbor. */
                if (iy < R-1 && !collide[idx*3+1]) {
                    int32_t nb = (int32_t)(GRID_IDX(ix, iy+1, iz, R) + 1);
                    uf_union(parent, node, nb);
                }
                /* +Z neighbor. */
                if (iz < R-1 && !collide[idx*3+2]) {
                    int32_t nb = (int32_t)(GRID_IDX(ix, iy, iz+1, R) + 1);
                    uf_union(parent, node, nb);
                }
            }

    /* Apply sign: voxels not connected to boundary sentinel are inside. */
    int32_t boundary_root = uf_find(parent, 0);
    for (size_t i = 0; i < grid_size; i++) {
        int32_t root = uf_find(parent, (int32_t)(i + 1));
        float d = dist[i];
        grid_out[i] = (root != boundary_root) ? -(double)d : (double)d;
    }

    PAMO_SDF_TICK("phase3-sign");
    if (prof) fprintf(stderr, "[sdf-prof] %-22s %.3f s (R=%d)\n",
                      "TOTAL", pamo_sdf_now_s() - t0, R);

    free(parent);
    free(collide);
    free(dist);
    free(active);
    return PAMO_OK;
}
#undef PAMO_SDF_TICK
