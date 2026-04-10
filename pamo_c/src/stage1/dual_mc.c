/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dual Marching Cubes: extract a triangle mesh from an SDF grid.
 *
 * For each grid edge with a sign change, create a quad (2 triangles)
 * from the 4 adjacent cells. Winding order is determined by the
 * SDF gradient direction to ensure consistent normals.
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_alloc.h"

#include <string.h>

#define SDF_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

#define CELL_IDX(ix, iy, iz, C) \
    ((size_t)(iz) * (size_t)(C) * (size_t)(C) + \
     (size_t)(iy) * (size_t)(C) + (size_t)(ix))

/* Emit a quad as 2 triangles. Flip winding if needed so normal
 * points in the direction of positive SDF (outward). */
static size_t emit_quad(pamo_tri *faces, size_t fi, size_t max_fi,
                        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                        bool flip) {
    if (fi + 2 > max_fi) return fi;
    if (flip) {
        faces[fi++] = (pamo_tri){{c0, c2, c1}};
        faces[fi++] = (pamo_tri){{c0, c3, c2}};
    } else {
        faces[fi++] = (pamo_tri){{c0, c1, c2}};
        faces[fi++] = (pamo_tri){{c0, c2, c3}};
    }
    return fi;
}

pamo_error pamo_dual_mc(pamo_mesh *out, const double *sdf, int32_t R,
                        pamo_vec3d origin, double voxel_size, double iso,
                        const pamo_allocator *alloc) {
    if (!out || !sdf || R < 2) return PAMO_ERR_INVALID_ARG;

    int32_t C = R - 1;
    size_t n_cells = (size_t)C * (size_t)C * (size_t)C;

    /* Phase 1: Find surface cells (contain a sign change). */
    int32_t *cell_vert = (int32_t *)pamo_alloc(alloc,
        n_cells * sizeof(int32_t));
    if (!cell_vert) return PAMO_ERR_ALLOC;
    memset(cell_vert, -1, n_cells * sizeof(int32_t));

    size_t n_surf = 0;
    for (int32_t iz = 0; iz < C; iz++)
        for (int32_t iy = 0; iy < C; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                int pos = 0, neg = 0;
                for (int dz = 0; dz < 2; dz++)
                    for (int dy = 0; dy < 2; dy++)
                        for (int dx = 0; dx < 2; dx++) {
                            double v = sdf[SDF_IDX(ix+dx, iy+dy, iz+dz, R)];
                            if (v - iso >= 0) pos++; else neg++;
                        }
                if (pos > 0 && neg > 0) {
                    cell_vert[CELL_IDX(ix, iy, iz, C)] = (int32_t)n_surf;
                    n_surf++;
                }
            }

    if (n_surf == 0) {
        pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));
        return pamo_mesh_create(out, 0, 0, alloc);
    }

    /* Phase 2: Place a vertex at each surface cell center. */
    size_t max_faces = n_surf * 6;
    pamo_error err = pamo_mesh_create(out, n_surf, max_faces, alloc);
    if (err != PAMO_OK) {
        pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));
        return err;
    }

    size_t vi = 0;
    for (int32_t iz = 0; iz < C; iz++)
        for (int32_t iy = 0; iy < C; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                if (cell_vert[CELL_IDX(ix, iy, iz, C)] < 0) continue;
                out->verts[vi++] = (pamo_vec3d){
                    origin.x + ((double)ix + 0.5) * voxel_size,
                    origin.y + ((double)iy + 0.5) * voxel_size,
                    origin.z + ((double)iz + 0.5) * voxel_size,
                };
            }

    /* Phase 3: For each grid edge with a sign change, emit a quad
     * from the (up to 4) adjacent surface cells.
     *
     * For an X-edge at grid point (ix,iy,iz) connecting
     * (ix,iy,iz)-(ix+1,iy,iz), the 4 adjacent cells are:
     *   (ix, iy-1, iz-1), (ix, iy,   iz-1)
     *   (ix, iy-1, iz  ), (ix, iy,   iz  )
     * arranged in a ring around the edge. We use a fixed order
     * that gives consistent winding. */
    size_t fi = 0;

    /* X-edges */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix+1, iy, iz, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;

                /* 4 cells in ring order around the X-edge. */
                int32_t cy[4] = {iy-1, iy,   iy,   iy-1};
                int32_t cz[4] = {iz-1, iz-1, iz,   iz  };
                int32_t c[4] = {-1, -1, -1, -1};
                int nc = 0;
                int32_t valid[4];
                for (int k = 0; k < 4; k++) {
                    if (cy[k] >= 0 && cy[k] < C && cz[k] >= 0 && cz[k] < C) {
                        int32_t cv = cell_vert[CELL_IDX(ix, cy[k], cz[k], C)];
                        if (cv >= 0) { valid[nc] = k; c[nc] = cv; nc++; }
                    }
                }
                if (nc == 4) {
                    bool flip = v0 < v1;
                    fi = emit_quad(out->faces, fi, max_faces,
                                   c[0], c[1], c[2], c[3], flip);
                } else if (nc == 3 && fi < max_faces) {
                    if (v0 < v1)
                        out->faces[fi++] = (pamo_tri){{c[0], c[2], c[1]}};
                    else
                        out->faces[fi++] = (pamo_tri){{c[0], c[1], c[2]}};
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
                int32_t c[4] = {-1,-1,-1,-1};
                int nc = 0;
                for (int k = 0; k < 4; k++) {
                    if (cx[k] >= 0 && cx[k] < C && cz[k] >= 0 && cz[k] < C) {
                        int32_t cv = cell_vert[CELL_IDX(cx[k], iy, cz[k], C)];
                        if (cv >= 0) { c[nc] = cv; nc++; }
                    }
                }
                if (nc >= 3) {
                    bool flip = v0 < v1;
                    fi = emit_quad(out->faces, fi, max_faces,
                                   c[0], c[1], c[nc>3?2:2],
                                   c[nc>3?3:2], flip);
                    if (nc == 3) fi--;
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
                int32_t c[4] = {-1,-1,-1,-1};
                int nc = 0;
                for (int k = 0; k < 4; k++) {
                    if (cx[k] >= 0 && cx[k] < C && cy[k] >= 0 && cy[k] < C) {
                        int32_t cv = cell_vert[CELL_IDX(cx[k], cy[k], iz, C)];
                        if (cv >= 0) { c[nc] = cv; nc++; }
                    }
                }
                if (nc >= 3) {
                    bool flip = v0 < v1;
                    fi = emit_quad(out->faces, fi, max_faces,
                                   c[0], c[1], c[nc>3?2:2],
                                   c[nc>3?3:2], flip);
                    if (nc == 3) fi--;
                }
            }

    out->n_faces = fi;
    for (size_t i = fi; i < max_faces; i++)
        out->face_alive[i] = false;

    pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));
    return pamo_mesh_compact(out);
}
