/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dual Marching Cubes: extract a triangle mesh from an SDF grid.
 *
 * This is a simplified implementation that places a vertex at each
 * sign-change cell (the dual vertex), then connects adjacent cells
 * that share a sign-change edge.
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_alloc.h"

#include <string.h>

/* Grid indexing: sdf[iz * R * R + iy * R + ix] */
#define SDF_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

/* Cell indexing (R-1 cells per axis): cell[iz*(R-1)*(R-1) + iy*(R-1) + ix] */
#define CELL_IDX(ix, iy, iz, C) \
    ((size_t)(iz) * (size_t)(C) * (size_t)(C) + \
     (size_t)(iy) * (size_t)(C) + (size_t)(ix))

pamo_error pamo_dual_mc(pamo_mesh *out, const double *sdf, int32_t R,
                        pamo_vec3d origin, double voxel_size, double iso,
                        const pamo_allocator *alloc) {
    if (!out || !sdf || R < 2) return PAMO_ERR_INVALID_ARG;

    int32_t C = R - 1; /* cells per axis */
    size_t n_cells = (size_t)C * (size_t)C * (size_t)C;

    /* Phase 1: Find cells that contain a sign change (surface cells).
     * A cell (ix,iy,iz) has 8 corners; if they don't all have the
     * same sign relative to iso, it's a surface cell. */
    int32_t *cell_vert = (int32_t *)pamo_alloc(alloc,
        n_cells * sizeof(int32_t));
    if (!cell_vert) return PAMO_ERR_ALLOC;
    memset(cell_vert, -1, n_cells * sizeof(int32_t));

    /* Count surface cells. */
    size_t n_surf = 0;
    for (int32_t iz = 0; iz < C; iz++) {
        for (int32_t iy = 0; iy < C; iy++) {
            for (int32_t ix = 0; ix < C; ix++) {
                /* 8 corners of cell. */
                int pos = 0, neg = 0;
                for (int dz = 0; dz < 2; dz++)
                    for (int dy = 0; dy < 2; dy++)
                        for (int dx = 0; dx < 2; dx++) {
                            double v = sdf[SDF_IDX(ix+dx, iy+dy, iz+dz, R)];
                            if (v - iso >= 0) pos++;
                            else neg++;
                        }
                if (pos > 0 && neg > 0) {
                    cell_vert[CELL_IDX(ix, iy, iz, C)] = (int32_t)n_surf;
                    n_surf++;
                }
            }
        }
    }

    if (n_surf == 0) {
        pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));
        return pamo_mesh_create(out, 0, 0, alloc);
    }

    /* Phase 2: Create a vertex for each surface cell.
     * Position: average of edge-crossing interpolations, or simply
     * cell center weighted by SDF values. */
    size_t max_faces = n_surf * 6; /* upper bound */
    pamo_error err = pamo_mesh_create(out, n_surf, max_faces, alloc);
    if (err != PAMO_OK) {
        pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));
        return err;
    }

    /* Place vertices at cell centers (simple version). */
    size_t vi = 0;
    for (int32_t iz = 0; iz < C; iz++) {
        for (int32_t iy = 0; iy < C; iy++) {
            for (int32_t ix = 0; ix < C; ix++) {
                if (cell_vert[CELL_IDX(ix, iy, iz, C)] < 0) continue;
                out->verts[vi] = (pamo_vec3d){
                    origin.x + ((double)ix + 0.5) * voxel_size,
                    origin.y + ((double)iy + 0.5) * voxel_size,
                    origin.z + ((double)iz + 0.5) * voxel_size,
                };
                vi++;
            }
        }
    }

    /* Phase 3: Connect adjacent surface cells that share a sign-change
     * edge.  For each internal edge of the grid that has a sign change,
     * the 4 cells sharing that edge contribute a quad (2 triangles). */
    size_t fi = 0;

    /* X-edges: between (ix, iy, iz) and (ix+1, iy, iz).
     * 4 adjacent cells share this edge:
     * (ix, iy-1, iz-1), (ix, iy, iz-1), (ix, iy-1, iz), (ix, iy, iz) */
    for (int32_t iz = 0; iz < R; iz++) {
        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < C; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix+1, iy, iz, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;

                /* 4 adjacent cells. */
                int32_t c[4];
                int nc = 0;
                for (int dy = -1; dy <= 0; dy++) {
                    for (int dz = -1; dz <= 0; dz++) {
                        int32_t cy = iy + dy, cz = iz + dz;
                        if (cy < 0 || cy >= C || cz < 0 || cz >= C) continue;
                        int32_t cv = cell_vert[CELL_IDX(ix, cy, cz, C)];
                        if (cv >= 0) c[nc++] = cv;
                    }
                }
                if (nc >= 3 && fi + 2 <= max_faces) {
                    out->faces[fi++] = (pamo_tri){{c[0], c[1], c[2]}};
                    if (nc == 4)
                        out->faces[fi++] = (pamo_tri){{c[0], c[2], c[3]}};
                }
            }
        }
    }

    /* Y-edges */
    for (int32_t iz = 0; iz < R; iz++) {
        for (int32_t iy = 0; iy < C; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix, iy+1, iz, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;

                int32_t c[4];
                int nc = 0;
                for (int dx = -1; dx <= 0; dx++) {
                    for (int dz = -1; dz <= 0; dz++) {
                        int32_t cx = ix + dx, cz = iz + dz;
                        if (cx < 0 || cx >= C || cz < 0 || cz >= C) continue;
                        int32_t cv = cell_vert[CELL_IDX(cx, iy, cz, C)];
                        if (cv >= 0) c[nc++] = cv;
                    }
                }
                if (nc >= 3 && fi + 2 <= max_faces) {
                    out->faces[fi++] = (pamo_tri){{c[0], c[1], c[2]}};
                    if (nc == 4)
                        out->faces[fi++] = (pamo_tri){{c[0], c[2], c[3]}};
                }
            }
        }
    }

    /* Z-edges */
    for (int32_t iz = 0; iz < C; iz++) {
        for (int32_t iy = 0; iy < R; iy++) {
            for (int32_t ix = 0; ix < R; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix, iy, iz+1, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;

                int32_t c[4];
                int nc = 0;
                for (int dx = -1; dx <= 0; dx++) {
                    for (int dy = -1; dy <= 0; dy++) {
                        int32_t cx = ix + dx, cy = iy + dy;
                        if (cx < 0 || cx >= C || cy < 0 || cy >= C) continue;
                        int32_t cv = cell_vert[CELL_IDX(cx, cy, iz, C)];
                        if (cv >= 0) c[nc++] = cv;
                    }
                }
                if (nc >= 3 && fi + 2 <= max_faces) {
                    out->faces[fi++] = (pamo_tri){{c[0], c[1], c[2]}};
                    if (nc == 4)
                        out->faces[fi++] = (pamo_tri){{c[0], c[2], c[3]}};
                }
            }
        }
    }

    out->n_faces = fi;
    /* Mark unused face slots as dead. */
    for (size_t i = fi; i < max_faces; i++) {
        out->face_alive[i] = false;
    }

    pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));

    /* Compact to remove unused capacity. */
    return pamo_mesh_compact(out);
}
