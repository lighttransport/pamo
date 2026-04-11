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
#include <string.h>
#include <stdlib.h>
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
                float p[3] = {
                    a->grid_ox + ((float)ix + 0.5f) * a->voxel_size,
                    a->grid_oy + ((float)iy + 0.5f) * a->voxel_size,
                    a->grid_oz + ((float)iz + 0.5f) * a->voxel_size,
                };
                float dsq = lightrt_bvh_nearest_bounded(
                    (const lightrt_bvh *)a->bvh, p, band_sq);
                a->grid[GRID_IDX(ix, iy, iz, a->R)] = sqrtf(dsq);
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

pamo_error pamo_compute_sdf(double *grid_out, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh) {
    if (!grid_out || R < 2 || !m) return PAMO_ERR_INVALID_ARG;

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

    /* ── Phase 1: Narrow-band distance computation ───────────────── */
#ifdef PAMO_USE_LIGHTRT
    lightrt_bvh *lbvh = build_lightrt_from_mesh(m);
    if (!lbvh) { free(dist); free(collide); return PAMO_ERR_ALLOC; }
    {
        float band_sq = band * band;
#ifdef PAMO_USE_PTHREADS
        int nt = 8;
        pthread_t *thr = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
        sdf_worker_arg *args = (sdf_worker_arg *)malloc((size_t)nt * sizeof(sdf_worker_arg));
        int32_t per = (R + nt - 1) / nt;
        for (int t = 0; t < nt; t++) {
            args[t] = (sdf_worker_arg){
                .bvh = lbvh, .grid = dist, .R = R,
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
                    float p[3] = {gox+((float)ix+0.5f)*vs, goy+((float)iy+0.5f)*vs, goz+((float)iz+0.5f)*vs};
                    dist[GRID_IDX(ix,iy,iz,R)] = sqrtf(lightrt_bvh_nearest_bounded(lbvh, p, band_sq));
                }
#endif
    }
    /* lbvh reused for Phase 2 below, destroyed after. */
#else
    /* Fallback: pamo BVH. */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < R; ix++) {
                pamo_vec3d p = {grid_origin.x+((double)ix+0.5)*voxel_size,
                                grid_origin.y+((double)iy+0.5)*voxel_size,
                                grid_origin.z+((double)iz+0.5)*voxel_size};
                pamo_nearest_result res;
                pamo_bvh_nearest(bvh, m, p, &res);
                float d = (float)sqrt(res.dist_sq);
                if (d < band) dist[GRID_IDX(ix,iy,iz,R)] = d;
            }
#endif

    /* ── Phase 2: Collision flags (3-axis ray tests for near-surface voxels) */
    float rayth = vs + 1e-6f;
#ifdef PAMO_USE_LIGHTRT
    /* Reuse lbvh from Phase 1. */
    {
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
    }
#else
    /* Fallback: brute-force ray-face test. */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < R; ix++) {
                size_t idx = GRID_IDX(ix, iy, iz, R);
                if (dist[idx] > rayth) continue;
                float px = gox + ((float)ix + 0.5f) * vs;
                float py = goy + ((float)iy + 0.5f) * vs;
                float pz = goz + ((float)iz + 0.5f) * vs;
                for (size_t fi = 0; fi < m->n_faces; fi++) {
                    if (!m->face_alive[fi]) continue;
                    float v0x=(float)m->verts[m->faces[fi].v[0]].x;
                    float v0y=(float)m->verts[m->faces[fi].v[0]].y;
                    float v0z=(float)m->verts[m->faces[fi].v[0]].z;
                    float v1x=(float)m->verts[m->faces[fi].v[1]].x;
                    float v1y=(float)m->verts[m->faces[fi].v[1]].y;
                    float v1z=(float)m->verts[m->faces[fi].v[1]].z;
                    float v2x=(float)m->verts[m->faces[fi].v[2]].x;
                    float v2y=(float)m->verts[m->faces[fi].v[2]].y;
                    float v2z=(float)m->verts[m->faces[fi].v[2]].z;
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
#endif

    /* ── Phase 3: Union-Find sign determination ──────────────────── */
    /* Node 0 = boundary sentinel. Nodes 1..N^3 = voxels. */
    size_t uf_size = grid_size + 1;
    int32_t *parent = (int32_t *)malloc(uf_size * sizeof(int32_t));
    if (!parent) { free(dist); free(collide); return PAMO_ERR_ALLOC; }
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

    free(parent);
    free(collide);
    free(dist);
    return PAMO_OK;
}
