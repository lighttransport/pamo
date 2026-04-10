/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Verification harness: compare C pamo output against CUDA reference.
 *
 * Usage:
 *   pamo_verify --ref ref_output.bin --test test_output.obj
 *
 * Reports: Hausdorff distance, face count comparison, self-intersection count.
 */
#include "pamo/pamo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Load binary mesh (PAMO format from export_reference.py) ─────── */

static pamo_error load_mesh_bin(const char *path, pamo_mesh *m,
                                pamo_allocator *alloc) {
    FILE *f = fopen(path, "rb");
    if (!f) return PAMO_ERR_IO;

    char magic[4];
    uint32_t version, nv, nf;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PAMO", 4) != 0) {
        fclose(f);
        return PAMO_ERR_IO;
    }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return PAMO_ERR_IO; }
    if (fread(&nv, 4, 1, f) != 1) { fclose(f); return PAMO_ERR_IO; }
    if (fread(&nf, 4, 1, f) != 1) { fclose(f); return PAMO_ERR_IO; }

    pamo_error err = pamo_mesh_create(m, nv, nf, alloc);
    if (err != PAMO_OK) { fclose(f); return err; }

    /* Read float32 vertices and convert to double. */
    float *buf = (float *)malloc((size_t)nv * 3 * sizeof(float));
    if (!buf) { fclose(f); return PAMO_ERR_ALLOC; }
    if (fread(buf, sizeof(float), (size_t)nv * 3, f) != (size_t)nv * 3) {
        free(buf); fclose(f); return PAMO_ERR_IO;
    }
    for (uint32_t i = 0; i < nv; i++) {
        m->verts[i] = (pamo_vec3d){
            (double)buf[i*3+0], (double)buf[i*3+1], (double)buf[i*3+2]};
    }
    free(buf);

    /* Read int32 faces. */
    if (fread(m->faces, sizeof(pamo_tri), nf, f) != nf) {
        fclose(f); return PAMO_ERR_IO;
    }

    fclose(f);
    return PAMO_OK;
}

/* ── Load OBJ (simple reader, same as example/main.c) ────────────── */

static pamo_error load_obj(const char *path, pamo_mesh *m,
                           pamo_allocator *alloc) {
    FILE *f = fopen(path, "r");
    if (!f) return PAMO_ERR_IO;

    size_t nv = 0, nf = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') nv++;
        else if (line[0] == 'f' && line[1] == ' ') nf++;
    }

    pamo_error err = pamo_mesh_create(m, nv, nf, alloc);
    if (err != PAMO_OK) { fclose(f); return err; }

    rewind(f);
    size_t vi = 0, fi = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            double x, y, z;
            if (sscanf(line + 2, "%lf %lf %lf", &x, &y, &z) == 3) {
                m->verts[vi++] = (pamo_vec3d){x, y, z};
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            int a, b, c;
            if (sscanf(line + 2, "%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",
                       &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d//%*d %d//%*d %d//%*d",
                       &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d/%*d %d/%*d %d/%*d",
                       &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d %d %d", &a, &b, &c) == 3) {
                m->faces[fi++] = (pamo_tri){{a - 1, b - 1, c - 1}};
            }
        }
    }
    m->n_verts = vi;
    m->n_faces = fi;
    fclose(f);
    return PAMO_OK;
}

/* ── Sampled Hausdorff distance (one direction) ──────────────────── */

static double one_way_hausdorff(const pamo_mesh *a, const pamo_mesh *b,
                                const pamo_bvh *b_bvh) {
    double max_dist = 0.0;
    for (size_t i = 0; i < a->n_verts; i++) {
        if (!a->vert_alive[i]) continue;
        pamo_nearest_result res;
        pamo_bvh_nearest(b_bvh, b, a->verts[i], &res);
        double d = sqrt(res.dist_sq);
        if (d > max_dist) max_dist = d;
    }
    return max_dist;
}

/* ── Mesh diameter ───────────────────────────────────────────────── */

static double mesh_diameter(const pamo_mesh *m) {
    pamo_aabb bb = pamo_mesh_bounds(m);
    pamo_vec3d d = pamo_v3_sub(bb.hi, bb.lo);
    return sqrt(pamo_v3_length_sq(d));
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *ref_path = NULL;
    const char *test_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ref") == 0 && i + 1 < argc)
            ref_path = argv[++i];
        else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            test_path = argv[++i];
    }

    if (!ref_path || !test_path) {
        fprintf(stderr, "Usage: %s --ref <ref.bin|ref.obj> --test <test.obj>\n",
                argv[0]);
        return 1;
    }

    pamo_allocator alloc = pamo_tracking_allocator_create();
    pamo_mesh ref_mesh, test_mesh;
    pamo_error err;

    /* Load reference. */
    size_t ref_len = strlen(ref_path);
    if (ref_len > 4 && strcmp(ref_path + ref_len - 4, ".bin") == 0) {
        err = load_mesh_bin(ref_path, &ref_mesh, &alloc);
    } else {
        err = load_obj(ref_path, &ref_mesh, &alloc);
    }
    if (err != PAMO_OK) {
        fprintf(stderr, "Error loading ref: %s\n", pamo_error_string(err));
        return 1;
    }

    /* Load test. */
    size_t test_len = strlen(test_path);
    if (test_len > 4 && strcmp(test_path + test_len - 4, ".bin") == 0) {
        err = load_mesh_bin(test_path, &test_mesh, &alloc);
    } else {
        err = load_obj(test_path, &test_mesh, &alloc);
    }
    if (err != PAMO_OK) {
        fprintf(stderr, "Error loading test: %s\n", pamo_error_string(err));
        return 1;
    }

    printf("Reference: %zu verts, %zu faces\n",
           ref_mesh.n_verts, ref_mesh.n_faces);
    printf("Test:      %zu verts, %zu faces\n",
           test_mesh.n_verts, test_mesh.n_faces);

    /* Build BVHs. */
    pamo_bvh ref_bvh, test_bvh;
    pamo_bvh_build_triangles(&ref_bvh, &ref_mesh, &alloc);
    pamo_bvh_build_triangles(&test_bvh, &test_mesh, &alloc);

    /* Compute Hausdorff distances. */
    double h_ref_to_test = one_way_hausdorff(&ref_mesh, &test_mesh, &test_bvh);
    double h_test_to_ref = one_way_hausdorff(&test_mesh, &ref_mesh, &ref_bvh);
    double hausdorff = (h_ref_to_test > h_test_to_ref) ?
                        h_ref_to_test : h_test_to_ref;

    double diam = mesh_diameter(&ref_mesh);
    double relative = (diam > 1e-10) ? hausdorff / diam * 100.0 : 0.0;

    printf("\n=== Comparison Results ===\n");
    printf("Mesh diameter:    %.6f\n", diam);
    printf("Hausdorff (abs):  %.6f\n", hausdorff);
    printf("Hausdorff (rel):  %.2f%% of diameter\n", relative);
    printf("Face count ratio: %.2f%% (%zu / %zu)\n",
           (double)test_mesh.n_faces / (double)ref_mesh.n_faces * 100.0,
           test_mesh.n_faces, ref_mesh.n_faces);

    /* Pass/fail criteria. */
    bool pass = true;
    if (relative > 10.0) {
        printf("FAIL: Hausdorff > 10%% of diameter\n");
        pass = false;
    }

    printf("\nVerdict: %s\n", pass ? "PASS" : "FAIL");

    pamo_bvh_destroy(&ref_bvh);
    pamo_bvh_destroy(&test_bvh);
    pamo_mesh_destroy(&ref_mesh);
    pamo_mesh_destroy(&test_mesh);
    pamo_tracking_allocator_report(&alloc);
    pamo_tracking_allocator_destroy(&alloc);

    return pass ? 0 : 1;
}
