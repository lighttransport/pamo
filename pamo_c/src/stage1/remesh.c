/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stage 1: Volumetric remeshing driver.
 */
#include "pamo/pamo_stage1.h"
#include "pamo/pamo_bvh.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* Forward declarations from sdf.c and dual_mc.c */
pamo_error pamo_compute_sdf(double *grid, int32_t R,
                            pamo_vec3d grid_origin, double voxel_size,
                            const pamo_mesh *m, const pamo_bvh *bvh);

pamo_error pamo_dual_mc(pamo_mesh *out, const double *sdf, int32_t R,
                        pamo_vec3d origin, double voxel_size, double iso,
                        const pamo_allocator *alloc);

pamo_remesh_opts pamo_remesh_opts_default(void) {
    return (pamo_remesh_opts){
        .resolution = 128,
        .band       = 0.0,
    };
}

pamo_error pamo_remesh(pamo_mesh *out, const pamo_mesh *in,
                       const pamo_remesh_opts *opts,
                       const pamo_allocator *alloc) {
    if (!out || !in || !opts || !alloc) return PAMO_ERR_INVALID_ARG;

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

    /* Expand by margin. */
    double margin = band * 2.0 + 1.0 / (double)R;
    double half_size = (max_ext * 0.5) * (1.0 + margin * 2.0);
    pamo_vec3d grid_origin = pamo_v3_sub(center,
        (pamo_vec3d){half_size, half_size, half_size});
    double voxel_size = 2.0 * half_size / (double)R;

    fprintf(stderr, "Stage 1: R=%d, voxel=%.6f, grid_size=%.4f\n",
            R, voxel_size, 2.0 * half_size);

    /* Build BVH for input mesh. */
    pamo_bvh bvh;
    pamo_error err = pamo_bvh_build_triangles(&bvh, in, alloc);
    if (err != PAMO_OK) return err;

    /* Allocate SDF grid. */
    size_t grid_size = (size_t)R * (size_t)R * (size_t)R;
    double *sdf = (double *)pamo_alloc(alloc, grid_size * sizeof(double));
    if (!sdf) {
        pamo_bvh_destroy(&bvh);
        return PAMO_ERR_ALLOC;
    }

    /* Compute SDF. */
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

    /* Extract mesh via dual marching cubes at iso=0. */
    err = pamo_dual_mc(out, sdf, R, grid_origin, voxel_size, 0.0, alloc);
    pamo_free(alloc, sdf, grid_size * sizeof(double));

    if (err != PAMO_OK) return err;

    fprintf(stderr, "Stage 1: extracted %zu verts, %zu faces\n",
            out->n_verts, out->n_faces);

    /* Post-process: resolve non-manifold edges by iterative edge collapse.
     * For each edge shared by >2 faces, collapse it (merge v into u).
     * Degenerate faces are removed. Iterate until clean. */
    for (int nm_pass = 0; nm_pass < 5; nm_pass++) {
        err = pamo_mesh_build_adjacency(out);
        if (err != PAMO_OK) break;
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
            if (!out->vert_alive[u] || !out->vert_alive[v]) continue;

            out->verts[u] = pamo_v3_scale(
                pamo_v3_add(out->verts[u], out->verts[v]), 0.5);
            out->vert_alive[v] = false;

            for (size_t fi = 0; fi < out->n_faces; fi++) {
                if (!out->face_alive[fi]) continue;
                int32_t *fv = out->faces[fi].v;
                for (int k = 0; k < 3; k++) {
                    if (fv[k] == v) fv[k] = u;
                }
                if (fv[0] == fv[1] || fv[1] == fv[2] || fv[2] == fv[0])
                    out->face_alive[fi] = false;
            }
        }

        pamo_mesh_free_adjacency(out);
        pamo_mesh_compact(out);
    }

    return PAMO_OK;
}
