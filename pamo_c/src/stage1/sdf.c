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
        fi[af*3]   = rm[m->faces[i].v[0]];
        fi[af*3+1] = rm[m->faces[i].v[1]];
        fi[af*3+2] = rm[m->faces[i].v[2]];
        af++;
    }
    free(rm);
    lightrt_bvh *b = lightrt_bvh_build(fv, av, fi, af);
    free(fv); free(fi);
    return b;
}
#endif

/* ── Parallel Phase 1 worker ─────────────────────────────────────── */

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
            if (ixlo < 0) ixlo = 0; if (ixhi > R-1) ixhi = R-1;
            if (iylo < 0) iylo = 0; if (iyhi > R-1) iyhi = R-1;
            if (izlo < 0) izlo = 0; if (izhi > R-1) izhi = R-1;
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
        if (!lbvh) use_lightrt = 0;  /* graceful fallback on alloc failure */
    }
#endif

    /* ── Phase 1: Narrow-band distance computation ───────────────── */
    if (use_lightrt) {
#ifdef PAMO_USE_LIGHTRT
        float band_sq = band * band;
#ifdef PAMO_USE_PTHREADS
        /* Thread count: env override PAMO_NUM_THREADS, else online CPUs
         * (clamped to [1, R]; per-z-slab partition gives at most R tasks). */
        int nt = 0;
        const char *ev = getenv("PAMO_NUM_THREADS");
        if (ev && ev[0]) nt = atoi(ev);
        if (nt <= 0) {
            long on = sysconf(_SC_NPROCESSORS_ONLN);
            nt = (on > 0) ? (int)on : 8;
        }
        if (nt > R) nt = R;
        if (nt < 1) nt = 1;
        pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        sdf_worker_arg *args = (sdf_worker_arg *)malloc((size_t)nt * sizeof(sdf_worker_arg));
        int32_t per = (R + nt - 1) / nt;
        for (int t = 0; t < nt; t++) {
            args[t] = (sdf_worker_arg){
                .bvh = lbvh, .grid = dist, .active = active, .R = R,
                .iz_start = t*per, .iz_end = (t+1)*per > R ? R : (t+1)*per,
                .grid_ox = gox, .grid_oy = goy, .grid_oz = goz,
                .voxel_size = vs, .band = band,
            };
            pthread_create(&thr[t], NULL, sdf_worker, &args[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(thr[t], NULL);
        free(thr); free(args);
#else
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
        for (int32_t iz = 0; iz < R; iz++)
            for (int32_t iy = 0; iy < R; iy++)
                for (int32_t ix = 0; ix < R; ix++) {
                    size_t idx = GRID_IDX(ix,iy,iz,R);
                    if (!active[idx]) continue;
                    pamo_vec3d p = {grid_origin.x+((double)ix+0.5)*voxel_size,
                                    grid_origin.y+((double)iy+0.5)*voxel_size,
                                    grid_origin.z+((double)iz+0.5)*voxel_size};
                    pamo_nearest_result res;
                    pamo_bvh_nearest(bvh, m, p, &res);
                    float d = (float)sqrt(res.dist_sq);
                    if (d < band) dist[idx] = d;
                }
    }
    PAMO_SDF_TICK("phase1-nearest");

    /* ── Phase 2: Collision flags (3-axis ray tests for near-surface voxels) */
    float rayth = vs + 1e-6f;
    if (use_lightrt) {
#ifdef PAMO_USE_LIGHTRT
        /* Reuse lbvh from Phase 1. */
        if (lbvh) {
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
    }
    }
    PAMO_SDF_TICK("phase2-rays");

    /* ── Phase 3: Union-Find sign determination ──────────────────── */
    /* Node 0 = boundary sentinel. Nodes 1..N^3 = voxels. */
    size_t uf_size = grid_size + 1;
    int32_t *parent = (int32_t *)malloc(uf_size * sizeof(int32_t));
    if (!parent) { free(dist); free(collide); free(active); return PAMO_ERR_ALLOC; }
    for (size_t i = 0; i < uf_size; i++) parent[i] = (int32_t)i;

    /* Prescan: walk X-columns, link consecutive voxels that aren't
     * near the surface (dist*N >= 0.87, matching cumesh2sdf). */
    float prescan_thresh = SDF_PRESCAN_COEFF / (float)R * vs * (float)R;
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++) {
            int32_t chain_start = 0; /* boundary sentinel */
            for (int32_t ix = 0; ix < R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, R);
                int32_t node = (int32_t)(idx + 1);
                if (dist[idx] < prescan_thresh) {
                    /* Near surface: start new chain. */
                    chain_start = node;
                } else {
                    /* Far from surface: link to current chain. */
                    parent[node] = chain_start;
                }
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
