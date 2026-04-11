/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dual Marching Cubes: extract a triangle mesh from an SDF grid.
 */
#include "pamo/pamo_mesh.h"
#include "pamo/pamo_alloc.h"

#include <string.h>
#include <stdlib.h>

#define SDF_IDX(ix, iy, iz, R) \
    ((size_t)(iz) * (size_t)(R) * (size_t)(R) + \
     (size_t)(iy) * (size_t)(R) + (size_t)(ix))

#define CELL_IDX(ix, iy, iz, C) \
    ((size_t)(iz) * (size_t)(C) * (size_t)(C) + \
     (size_t)(iy) * (size_t)(C) + (size_t)(ix))

/* ── Edge usage tracking to prevent non-manifold edges ───────────── */

typedef struct {
    int32_t u, v;
    int16_t count;
} dmc_edge_slot;

typedef struct {
    dmc_edge_slot *slots;
    size_t capacity;
} dmc_edge_map;

static void emap_init(dmc_edge_map *em, size_t cap) {
    em->capacity = cap;
    em->slots = (dmc_edge_slot *)malloc(cap * sizeof(dmc_edge_slot));
    if (em->slots) {
        for (size_t i = 0; i < cap; i++) em->slots[i].u = -1;
    }
}

static void emap_destroy(dmc_edge_map *em) {
    free(em->slots);
    em->slots = NULL;
}

static size_t emap_hash(int32_t u, int32_t v, size_t cap) {
    uint32_t h = (uint32_t)u * 2654435761u ^ (uint32_t)v * 40503u;
    return h % cap;
}

/* Returns current count for edge (u,v), or 0 if not found. */
static int emap_get(const dmc_edge_map *em, int32_t u, int32_t v) {
    if (u > v) { int32_t t = u; u = v; v = t; }
    size_t h = emap_hash(u, v, em->capacity);
    for (size_t probe = 0; probe < 64; probe++) {
        size_t idx = (h + probe) % em->capacity;
        if (em->slots[idx].u < 0) return 0;
        if (em->slots[idx].u == u && em->slots[idx].v == v)
            return em->slots[idx].count;
    }
    return 0;
}

/* Increment count for edge (u,v). */
static void emap_add(dmc_edge_map *em, int32_t u, int32_t v) {
    if (u > v) { int32_t t = u; u = v; v = t; }
    size_t h = emap_hash(u, v, em->capacity);
    for (size_t probe = 0; probe < 64; probe++) {
        size_t idx = (h + probe) % em->capacity;
        if (em->slots[idx].u < 0) {
            em->slots[idx].u = u;
            em->slots[idx].v = v;
            em->slots[idx].count = 1;
            return;
        }
        if (em->slots[idx].u == u && em->slots[idx].v == v) {
            em->slots[idx].count++;
            return;
        }
    }
}

/* Check if a triangle (a,b,c) can be added without creating >2-face edges. */
static int tri_can_add(const dmc_edge_map *em, int32_t a, int32_t b, int32_t c) {
    return emap_get(em, a, b) < 2 &&
           emap_get(em, b, c) < 2 &&
           emap_get(em, a, c) < 2;
}

/* Record a triangle's edges. */
static void tri_record(dmc_edge_map *em, int32_t a, int32_t b, int32_t c) {
    emap_add(em, a, b);
    emap_add(em, b, c);
    emap_add(em, a, c);
}

/* ── DMC face emission ───────────────────────────────────────────── */

static size_t emit_quad(pamo_tri *faces, size_t fi, size_t max_fi,
                        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                        bool flip, dmc_edge_map *em) {
    if (fi + 2 > max_fi) return fi;

    /* Two possible triangulations of the quad:
     * Split A: (c0,c1,c2) + (c0,c2,c3)
     * Split B: (c0,c1,c3) + (c1,c2,c3)
     * Try both; pick the one that doesn't create non-manifold edges.
     * If both fail, emit anyway (prefer completeness over manifoldness). */
    int32_t ta[2][3], tb[2][3];
    if (flip) {
        ta[0][0]=c0; ta[0][1]=c2; ta[0][2]=c1;
        ta[1][0]=c0; ta[1][1]=c3; ta[1][2]=c2;
        tb[0][0]=c0; tb[0][1]=c3; tb[0][2]=c1;
        tb[1][0]=c1; tb[1][1]=c3; tb[1][2]=c2;
    } else {
        ta[0][0]=c0; ta[0][1]=c1; ta[0][2]=c2;
        ta[1][0]=c0; ta[1][1]=c2; ta[1][2]=c3;
        tb[0][0]=c0; tb[0][1]=c1; tb[0][2]=c3;
        tb[1][0]=c1; tb[1][1]=c2; tb[1][2]=c3;
    }

    /* Try split A. */
    bool a_ok = tri_can_add(em, ta[0][0], ta[0][1], ta[0][2]) &&
                tri_can_add(em, ta[1][0], ta[1][1], ta[1][2]);
    if (a_ok) {
        faces[fi] = (pamo_tri){{ta[0][0], ta[0][1], ta[0][2]}}; fi++;
        tri_record(em, ta[0][0], ta[0][1], ta[0][2]);
        faces[fi] = (pamo_tri){{ta[1][0], ta[1][1], ta[1][2]}}; fi++;
        tri_record(em, ta[1][0], ta[1][1], ta[1][2]);
        return fi;
    }

    /* Try split B. */
    bool b_ok = tri_can_add(em, tb[0][0], tb[0][1], tb[0][2]) &&
                tri_can_add(em, tb[1][0], tb[1][1], tb[1][2]);
    if (b_ok) {
        faces[fi] = (pamo_tri){{tb[0][0], tb[0][1], tb[0][2]}}; fi++;
        tri_record(em, tb[0][0], tb[0][1], tb[0][2]);
        faces[fi] = (pamo_tri){{tb[1][0], tb[1][1], tb[1][2]}}; fi++;
        tri_record(em, tb[1][0], tb[1][1], tb[1][2]);
        return fi;
    }

    /* Both splits create nm edges; emit split A anyway to avoid holes. */
    faces[fi] = (pamo_tri){{ta[0][0], ta[0][1], ta[0][2]}}; fi++;
    tri_record(em, ta[0][0], ta[0][1], ta[0][2]);
    faces[fi] = (pamo_tri){{ta[1][0], ta[1][1], ta[1][2]}}; fi++;
    tri_record(em, ta[1][0], ta[1][1], ta[1][2]);
    return fi;
}

/* ── Main DMC ────────────────────────────────────────────────────── */

pamo_error pamo_dual_mc(pamo_mesh *out, const double *sdf, int32_t R,
                        pamo_vec3d origin, double voxel_size, double iso,
                        const pamo_allocator *alloc) {
    if (!out || !sdf || R < 2) return PAMO_ERR_INVALID_ARG;

    int32_t C = R - 1;
    size_t n_cells = (size_t)C * (size_t)C * (size_t)C;

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

    /* Edge map to track usage. Size ~4x expected edges for low collision. */
    dmc_edge_map em;
    emap_init(&em, n_surf * 8);

    size_t fi = 0;

    /* X-edges */
    for (int32_t iz = 0; iz < R; iz++)
        for (int32_t iy = 0; iy < R; iy++)
            for (int32_t ix = 0; ix < C; ix++) {
                double v0 = sdf[SDF_IDX(ix, iy, iz, R)] - iso;
                double v1 = sdf[SDF_IDX(ix+1, iy, iz, R)] - iso;
                if ((v0 >= 0) == (v1 >= 0)) continue;
                int32_t cy[4] = {iy-1, iy,   iy,   iy-1};
                int32_t cz[4] = {iz-1, iz-1, iz,   iz  };
                int32_t c[4]; int nc = 0;
                for (int k = 0; k < 4; k++) {
                    if (cy[k] >= 0 && cy[k] < C && cz[k] >= 0 && cz[k] < C) {
                        int32_t cv = cell_vert[CELL_IDX(ix, cy[k], cz[k], C)];
                        if (cv >= 0) c[nc++] = cv;
                    }
                }
                if (nc == 4) {
                    fi = emit_quad(out->faces, fi, max_faces,
                                   c[0], c[1], c[2], c[3], v0 < v1, &em);
                } else if (nc == 3 && fi < max_faces) {
                    int32_t a = c[0], b = c[1], cc = c[2];
                    if (v0 < v1) { int32_t t = b; b = cc; cc = t; }
                    if (tri_can_add(&em, a, b, cc)) {
                        out->faces[fi++] = (pamo_tri){{a, b, cc}};
                        tri_record(&em, a, b, cc);
                    }
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
                int32_t c[4]; int nc = 0;
                for (int k = 0; k < 4; k++) {
                    if (cx[k] >= 0 && cx[k] < C && cz[k] >= 0 && cz[k] < C) {
                        int32_t cv = cell_vert[CELL_IDX(cx[k], iy, cz[k], C)];
                        if (cv >= 0) c[nc++] = cv;
                    }
                }
                if (nc == 4) {
                    fi = emit_quad(out->faces, fi, max_faces,
                                   c[0], c[1], c[2], c[3], v0 < v1, &em);
                } else if (nc == 3 && fi < max_faces) {
                    int32_t a = c[0], b = c[1], cc = c[2];
                    if (v0 < v1) { int32_t t = b; b = cc; cc = t; }
                    if (tri_can_add(&em, a, b, cc)) {
                        out->faces[fi++] = (pamo_tri){{a, b, cc}};
                        tri_record(&em, a, b, cc);
                    }
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
                int32_t c[4]; int nc = 0;
                for (int k = 0; k < 4; k++) {
                    if (cx[k] >= 0 && cx[k] < C && cy[k] >= 0 && cy[k] < C) {
                        int32_t cv = cell_vert[CELL_IDX(cx[k], cy[k], iz, C)];
                        if (cv >= 0) c[nc++] = cv;
                    }
                }
                if (nc == 4) {
                    fi = emit_quad(out->faces, fi, max_faces,
                                   c[0], c[1], c[2], c[3], v0 < v1, &em);
                } else if (nc == 3 && fi < max_faces) {
                    int32_t a = c[0], b = c[1], cc = c[2];
                    if (v0 < v1) { int32_t t = b; b = cc; cc = t; }
                    if (tri_can_add(&em, a, b, cc)) {
                        out->faces[fi++] = (pamo_tri){{a, b, cc}};
                        tri_record(&em, a, b, cc);
                    }
                }
            }

    emap_destroy(&em);

    out->n_faces = fi;
    for (size_t i = fi; i < max_faces; i++)
        out->face_alive[i] = false;

    pamo_free(alloc, cell_vert, n_cells * sizeof(int32_t));
    return pamo_mesh_compact(out);
}
