/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * JavaScript-friendly wrapper around the pamo simplification pipeline.
 *
 * Workflow from JS:
 *   1. Module._malloc input verts (Float32, nv*3) and faces (Int32, nf*3),
 *      copy data into HEAPF32 / HEAP32, then free those after the call.
 *   2. Call _pamo_wasm_run(verts_ptr, nv, faces_ptr, nf, ratio, st1, st3).
 *   3. Read back via _pamo_wasm_n_verts / _pamo_wasm_n_faces and the
 *      corresponding *_ptr getters into Module.HEAPF32 / Module.HEAP32.
 *   4. _pamo_wasm_reset before the next run (otherwise the static result
 *      buffers from the previous call remain allocated).
 */
#include "pamo/pamo.h"

#include <emscripten.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float   *verts;       /* nv * 3 (x, y, z) — JS reads from HEAPF32   */
    int32_t *faces;       /* nf * 3 (a, b, c) — JS reads from HEAP32    */
    int32_t  n_verts;
    int32_t  n_faces;
} wasm_result;

static wasm_result g_result = {0};

EMSCRIPTEN_KEEPALIVE
void pamo_wasm_reset(void) {
    free(g_result.verts);
    free(g_result.faces);
    g_result.verts = NULL;
    g_result.faces = NULL;
    g_result.n_verts = 0;
    g_result.n_faces = 0;
}

EMSCRIPTEN_KEEPALIVE int  pamo_wasm_n_verts(void)    { return g_result.n_verts; }
EMSCRIPTEN_KEEPALIVE int  pamo_wasm_n_faces(void)    { return g_result.n_faces; }
EMSCRIPTEN_KEEPALIVE float* pamo_wasm_verts_ptr(void){ return g_result.verts;   }
EMSCRIPTEN_KEEPALIVE int*   pamo_wasm_faces_ptr(void){ return g_result.faces;   }

/* Build a pamo_mesh from JS-allocated f32 verts + i32 faces. */
static pamo_error build_mesh(pamo_mesh *m, pamo_allocator *alloc,
                             const float *verts, int32_t nv,
                             const int32_t *faces, int32_t nf) {
    pamo_error e = pamo_mesh_create(m, (size_t)nv, (size_t)nf, alloc);
    if (e != PAMO_OK) return e;
    for (int32_t i = 0; i < nv; i++) {
        m->verts[i] = (pamo_vec3d){
            (double)verts[i*3+0],
            (double)verts[i*3+1],
            (double)verts[i*3+2]};
    }
    for (int32_t i = 0; i < nf; i++) {
        m->faces[i] = (pamo_tri){{faces[i*3+0], faces[i*3+1], faces[i*3+2]}};
    }
    return PAMO_OK;
}

/* Compact + flatten an alive-mesh into f32 verts + i32 faces.
 * Returns 0 on success and stores result in g_result. */
static int flatten_to_result(pamo_mesh *m) {
    pamo_error e = pamo_mesh_compact(m);
    if (e != PAMO_OK) return -1;

    int32_t nv = (int32_t)m->n_verts;
    int32_t nf = (int32_t)m->n_faces;

    pamo_wasm_reset();
    g_result.verts = (float *)malloc((size_t)nv * 3 * sizeof(float));
    g_result.faces = (int32_t *)malloc((size_t)nf * 3 * sizeof(int32_t));
    if (!g_result.verts || !g_result.faces) {
        pamo_wasm_reset();
        return -1;
    }
    g_result.n_verts = nv;
    g_result.n_faces = nf;

    for (int32_t i = 0; i < nv; i++) {
        g_result.verts[i*3+0] = (float)m->verts[i].x;
        g_result.verts[i*3+1] = (float)m->verts[i].y;
        g_result.verts[i*3+2] = (float)m->verts[i].z;
    }
    for (int32_t i = 0; i < nf; i++) {
        g_result.faces[i*3+0] = m->faces[i].v[0];
        g_result.faces[i*3+1] = m->faces[i].v[1];
        g_result.faces[i*3+2] = m->faces[i].v[2];
    }
    return 0;
}

/* Run the pamo pipeline. Returns 0 on success, negative pamo_error on
 * failure. The caller MUST keep the input verts/faces buffers alive for
 * the duration of this call (they're read once and copied internally).
 *
 * sdf_resolution: 0 = auto-pick (matches Python: 256/128/64 by target).
 *                 Positive values force that grid resolution for Stage 1.
 * preserve_boundary: lock vertices on one-face edges. The C library defaults
 *                    this to true (good for watertight inputs with intentional
 *                    cracks/seams), but most demo inputs are non-watertight
 *                    consumer scans where every internal seam is a "boundary"
 *                    — locking starves the simplifier and stalls reduction
 *                    well before the target. WASM ABI exposes it explicitly
 *                    so the caller can pick. */
EMSCRIPTEN_KEEPALIVE
int pamo_wasm_run(const float *verts, int32_t n_verts,
                  const int32_t *faces, int32_t n_faces,
                  float ratio,
                  int use_stage1, int use_stage3,
                  int sdf_resolution,
                  int preserve_boundary) {
    if (!verts || !faces || n_verts <= 0 || n_faces <= 0 || ratio <= 0.0f) {
        return -1;
    }

    pamo_allocator alloc = pamo_tracking_allocator_create();
    pamo_mesh mesh;
    pamo_error e = build_mesh(&mesh, &alloc, verts, n_verts, faces, n_faces);
    if (e != PAMO_OK) {
        pamo_tracking_allocator_destroy(&alloc);
        return -(int)e;
    }

    /* Match the original Python pamo (PaMO.run): target_faces is computed
     * from the *input* mesh, not the post-Stage-1 mesh. */
    int32_t target_faces = (int32_t)((double)n_faces * (double)ratio);
    if (target_faces < 10) target_faces = 10;

    /* Stage 1: optional remesh. R selection mirrors Python (PaMO.run):
     *   default R=256, R=128 if target<=1000, R=64 if target<=50.
     * Caller can override with sdf_resolution > 0. */
    if (use_stage1) {
        pamo_remesh_opts ropts = pamo_remesh_opts_default();
        if (sdf_resolution > 0) {
            ropts.resolution = sdf_resolution;
        } else if (target_faces <=   50) ropts.resolution = 64;
        else if  (target_faces <= 1000) ropts.resolution = 128;
        else                            ropts.resolution = 256;
        pamo_mesh remeshed;
        pamo_error r = pamo_remesh(&remeshed, &mesh, &ropts, &alloc);
        if (r == PAMO_OK && remeshed.n_faces > 0) {
            pamo_mesh_destroy(&mesh);
            mesh = remeshed;
        } else if (remeshed.n_verts > 0) {
            pamo_mesh_destroy(&remeshed);
        }
        /* If remesh failed, silently fall through with the original mesh. */
    }

    /* Stage 2: simplify (always). target_faces is the ratio of the *input*
     * mesh, regardless of how big the Stage 1 output ended up. */
    {
        pamo_simplify_opts opts = pamo_simplify_opts_default();
        opts.target_faces = target_faces;
        opts.preserve_boundary = (preserve_boundary != 0);
        e = pamo_simplify(&mesh, &opts);
        if (e != PAMO_OK) {
            pamo_mesh_destroy(&mesh);
            pamo_tracking_allocator_destroy(&alloc);
            return -(int)e;
        }
    }

    /* Stage 3: optional SAFE projection against the original geometry. */
    if (use_stage3) {
        pamo_mesh gt;
        pamo_error g = build_mesh(&gt, &alloc, verts, n_verts, faces, n_faces);
        if (g == PAMO_OK) {
            pamo_safe_opts sopts = pamo_safe_opts_default();
            (void)pamo_safe_project(&mesh, &gt, &sopts, &alloc);
            pamo_mesh_destroy(&gt);
        }
    }

    int rc = flatten_to_result(&mesh);
    pamo_mesh_destroy(&mesh);
    pamo_tracking_allocator_destroy(&alloc);
    return rc;
}
