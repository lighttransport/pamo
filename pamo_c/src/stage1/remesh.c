/* POSIX feature macro: required for clock_gettime / CLOCK_MONOTONIC on glibc. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

/* On macOS, _POSIX_C_SOURCE alone hides BSD extensions and narrows libc
 * (e.g. snprintf). _DARWIN_C_SOURCE re-exposes them. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 1: Volumetric remeshing driver.
 */
#include "pamo/pamo_stage1.h"
#include "pamo/pamo_bvh.h"
#include "pamo/pamo_progress.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Forward declarations from sdf.c and dual_mc.c */
pamo_error pamo_compute_sdf(double *grid, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh);

pamo_error pamo_dual_mc(pamo_mesh *out, const double *sdf, int32_t R,
                        pamo_vec3d origin, double voxel_size, double iso,
                        const pamo_allocator *alloc);

static int pamo_stage1_profile_on(void) {
    static int checked = 0, on = 0;
    if (!checked) {
        const char *e = getenv("PAMO_SDF_PROFILE");
        on = (e && e[0] == '1');
        checked = 1;
    }
    return on;
}

static double pamo_stage1_now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

pamo_remesh_opts pamo_remesh_opts_default(void) {
    return (pamo_remesh_opts){
        .resolution              = 128,
        .band                    = 0.0,
        .topology_closing_voxels = 0,
    };
}

static size_t pamo_sdf_idx3(int32_t ix, int32_t iy, int32_t iz, int32_t R) {
    return (size_t)ix + (size_t)R * ((size_t)iy + (size_t)R * (size_t)iz);
}

static void pamo_binary_dilate_axis(const uint8_t *src, uint8_t *dst,
                                    int32_t R, int radius, int axis) {
    for (int32_t iz = 0; iz < R; iz++) {
        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                uint8_t v = 0;
                for (int d = -radius; d <= radius; d++) {
                    int32_t x = ix, y = iy, z = iz;
                    if (axis == 0) x += d;
                    else if (axis == 1) y += d;
                    else z += d;
                    if (x < 0 || y < 0 || z < 0 ||
                        x >= R || y >= R || z >= R) {
                        continue;
                    }
                    if (src[pamo_sdf_idx3(x, y, z, R)]) {
                        v = 1;
                        break;
                    }
                }
                dst[pamo_sdf_idx3(ix, iy, iz, R)] = v;
            }
        }
    }
}

static void pamo_binary_erode_axis(const uint8_t *src, uint8_t *dst,
                                   int32_t R, int radius, int axis) {
    for (int32_t iz = 0; iz < R; iz++) {
        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                uint8_t v = 1;
                for (int d = -radius; d <= radius; d++) {
                    int32_t x = ix, y = iy, z = iz;
                    if (axis == 0) x += d;
                    else if (axis == 1) y += d;
                    else z += d;
                    if (x < 0 || y < 0 || z < 0 ||
                        x >= R || y >= R || z >= R ||
                        !src[pamo_sdf_idx3(x, y, z, R)]) {
                        v = 0;
                        break;
                    }
                }
                dst[pamo_sdf_idx3(ix, iy, iz, R)] = v;
            }
        }
    }
}

static pamo_error pamo_close_sdf_topology(double *sdf, int32_t R,
                                          int radius, double min_abs,
                                          const pamo_allocator *alloc) {
    if (!sdf || R < 2 || radius <= 0 || !alloc) return PAMO_OK;
    if (radius > R / 4) radius = R / 4;
    if (radius <= 0) return PAMO_OK;

    size_t n = (size_t)R * (size_t)R * (size_t)R;
    uint8_t *a = (uint8_t *)pamo_alloc(alloc, n);
    uint8_t *b = (uint8_t *)pamo_alloc(alloc, n);
    uint8_t *c = (uint8_t *)pamo_alloc(alloc, n);
    if (!a || !b || !c) {
        if (a) pamo_free(alloc, a, n);
        if (b) pamo_free(alloc, b, n);
        if (c) pamo_free(alloc, c, n);
        return PAMO_ERR_ALLOC;
    }

    for (size_t i = 0; i < n; i++) a[i] = sdf[i] < 0.0 ? 1u : 0u;

    pamo_binary_dilate_axis(a, b, R, radius, 0);
    pamo_binary_dilate_axis(b, c, R, radius, 1);
    pamo_binary_dilate_axis(c, b, R, radius, 2);
    pamo_binary_erode_axis(b, c, R, radius, 0);
    pamo_binary_erode_axis(c, b, R, radius, 1);
    pamo_binary_erode_axis(b, c, R, radius, 2);

    size_t changed = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] == c[i]) continue;
        double mag = fabs(sdf[i]);
        if (mag < min_abs) mag = min_abs;
        sdf[i] = c[i] ? -mag : mag;
        changed++;
    }

    fprintf(stderr,
            "Stage 1: topology closing radius=%d changed %zu voxels\n",
            radius, changed);

    pamo_free(alloc, c, n);
    pamo_free(alloc, b, n);
    pamo_free(alloc, a, n);
    return PAMO_OK;
}

pamo_error pamo_remesh(pamo_mesh *out, const pamo_mesh *in,
                       const pamo_remesh_opts *opts,
                       const pamo_allocator *alloc) {
    if (!out || !in || !opts || !alloc) return PAMO_ERR_INVALID_ARG;
    if (in->n_verts == 0 || in->n_faces == 0 ||
        pamo_mesh_count_alive_verts(in) == 0 ||
        pamo_mesh_count_alive_faces(in) == 0) {
        return PAMO_ERR_INVALID_ARG;
    }
    int prof = pamo_stage1_profile_on();
    double stage1_t0 = prof ? pamo_stage1_now_s() : 0.0;

    int32_t R = opts->resolution;
    if (R < 4) R = 4;
    if (R > 1024) R = 1024;
    double band = opts->band;
    if (band <= 0.0) band = 3.0 / (double)R;

    /* Compute bounding box of input mesh. */
    pamo_aabb bb = pamo_mesh_bounds(in);
    pamo_vec3d center = pamo_v3_scale(pamo_v3_add(bb.lo, bb.hi), 0.5);
    pamo_vec3d extent = pamo_v3_sub(bb.hi, bb.lo);
    double max_ext = extent.x;
    if (extent.y > max_ext) max_ext = extent.y;
    if (extent.z > max_ext) max_ext = extent.z;
    if (!(max_ext > 0.0) || !isfinite(max_ext)) return PAMO_ERR_INVALID_ARG;

    /* Expand by margin. */
    double half_size = (max_ext * 0.5) * (1.0 + band * 2.0);
    pamo_vec3d grid_origin = pamo_v3_sub(center,
        (pamo_vec3d){half_size, half_size, half_size});
    double voxel_size = 2.0 * half_size / (double)R;

    fprintf(stderr, "Stage 1: R=%d, voxel=%.6f, grid_size=%.4f\n",
            R, voxel_size, 2.0 * half_size);
    pamo_emit_progress("remesh_start", (int64_t)R, (int64_t)R);

    /* Build BVH for input mesh. */
    pamo_bvh bvh;
    double bvh_t0 = prof ? pamo_stage1_now_s() : 0.0;
    pamo_error err = pamo_bvh_build_triangles(&bvh, in, alloc);
    if (prof) {
        fprintf(stderr, "[sdf-prof] %-22s %.3f s\n",
                "pamo-bvh-build", pamo_stage1_now_s() - bvh_t0);
    }
    if (err != PAMO_OK) return err;

    /* Allocate SDF grid. */
    size_t grid_size = (size_t)R * (size_t)R * (size_t)R;
    double *sdf = (double *)pamo_alloc(alloc, grid_size * sizeof(double));
    if (!sdf) {
        pamo_bvh_destroy(&bvh);
        return PAMO_ERR_ALLOC;
    }

    /* Compute SDF. */
    pamo_emit_progress("remesh_sdf", 0, (int64_t)R);
    err = pamo_compute_sdf(sdf, R, grid_origin, voxel_size, in, &bvh);
    pamo_bvh_destroy(&bvh);
    if (err != PAMO_OK) {
        pamo_free(alloc, sdf, grid_size * sizeof(double));
        return err;
    }

    /* Offset SDF inward: sdf = sdf - 0.9/R
     * This shifts the isosurface slightly inside the original mesh. */
    double offset = 0.9 / (double)R * (2.0 * half_size);
    for (size_t i = 0; i < grid_size; i++) {
        sdf[i] -= offset;
    }

    int topology_closing = opts->topology_closing_voxels;
    const char *closing_env = getenv("PAMO_TOPOLOGY_CLOSE_VOXELS");
    if (closing_env && closing_env[0]) topology_closing = atoi(closing_env);
    if (topology_closing > 0) {
        double close_t0 = prof ? pamo_stage1_now_s() : 0.0;
        err = pamo_close_sdf_topology(sdf, R, topology_closing,
                                      voxel_size * 0.25, alloc);
        if (prof) {
            fprintf(stderr, "[sdf-prof] %-22s %.3f s\n",
                    "topology-closing", pamo_stage1_now_s() - close_t0);
        }
        if (err != PAMO_OK) {
            pamo_free(alloc, sdf, grid_size * sizeof(double));
            return err;
        }
    }

    /* Extract mesh via dual marching cubes at iso=0. */
    pamo_emit_progress("remesh_dual_mc", 0, (int64_t)R);
    double dmc_t0 = prof ? pamo_stage1_now_s() : 0.0;
    err = pamo_dual_mc(out, sdf, R, grid_origin, voxel_size, 0.0, alloc);
    if (prof) {
        fprintf(stderr, "[sdf-prof] %-22s %.3f s\n",
                "dual-mc", pamo_stage1_now_s() - dmc_t0);
    }
    pamo_free(alloc, sdf, grid_size * sizeof(double));

    if (err != PAMO_OK) return err;

    fprintf(stderr, "Stage 1: extracted %zu verts, %zu faces\n",
            out->n_verts, out->n_faces);
    if (prof) {
        fprintf(stderr, "[sdf-prof] %-22s %.3f s\n",
                "stage1-remesh-total", pamo_stage1_now_s() - stage1_t0);
    }

    /* Post-process: resolve non-manifold edges by iterative edge collapse.
     * For each edge shared by >2 faces, collapse it (merge v into u).
     * Degenerate faces are removed. Iterate until clean. */
    for (int nm_pass = 0; nm_pass < 5; nm_pass++) {
        err = pamo_mesh_build_adjacency(out);
        if (err != PAMO_OK) return err;
        if (!out->edge_face_offset) break;

        int32_t nm_count = 0;
        for (size_t ei = 0; ei < out->n_edges; ei++) {
            if (out->edge_face_offset[ei+1] - out->edge_face_offset[ei] > 2)
                nm_count++;
        }

        if (nm_count == 0) {
            pamo_mesh_free_adjacency(out);
            break;
        }

        fprintf(stderr, "Stage 1: resolving %d non-manifold edges (pass %d)\n",
                nm_count, nm_pass + 1);

        for (size_t ei = 0; ei < out->n_edges; ei++) {
            if (out->edge_face_offset[ei+1] - out->edge_face_offset[ei] <= 2)
                continue;

            int32_t u = out->edges[ei].u;
            int32_t v = out->edges[ei].v;
            if (u < 0 || v < 0 || (size_t)u >= out->n_verts ||
                (size_t)v >= out->n_verts) {
                pamo_mesh_free_adjacency(out);
                return PAMO_ERR_INVALID_ARG;
            }
            if (!out->vert_alive[u] || !out->vert_alive[v]) continue;

            out->verts[u] = pamo_v3_scale(
                pamo_v3_add(out->verts[u], out->verts[v]), 0.5);
            out->vert_alive[v] = false;

            for (size_t fi = 0; fi < out->n_faces; fi++) {
                if (!out->face_alive[fi]) continue;
                int32_t *fv = out->faces[fi].v;
                bool valid = true;
                for (int k = 0; k < 3; k++) {
                    if (fv[k] < 0 || (size_t)fv[k] >= out->n_verts) {
                        valid = false;
                        break;
                    }
                    if (fv[k] == v) fv[k] = u;
                }
                if (!valid) {
                    pamo_mesh_free_adjacency(out);
                    return PAMO_ERR_INVALID_ARG;
                }
                if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0])
                    out->face_alive[fi] = false;
            }
        }

        pamo_mesh_free_adjacency(out);
        err = pamo_mesh_compact(out);
        if (err != PAMO_OK) return err;
    }

    return PAMO_OK;
}
