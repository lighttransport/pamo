/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PaMO example: Load OBJ -> simplify -> save OBJ
 */
#include "pamo/pamo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* TODO: Integrate tinyobj_loader_c.h for full OBJ support.
 * For now, a minimal OBJ reader/writer. */

static pamo_error load_obj(const char *path, pamo_mesh *m,
                           pamo_allocator *alloc) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return PAMO_ERR_IO;
    }

    /* First pass: count vertices and faces. */
    size_t nv = 0, nf = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') nv++;
        else if (line[0] == 'f' && line[1] == ' ') nf++;
    }

    pamo_error err = pamo_mesh_create(m, nv, nf, alloc);
    if (err != PAMO_OK) { fclose(f); return err; }

    /* Second pass: read data. */
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
            /* Handle f v1 v2 v3 and f v1/vt1/vn1 v2/... v3/... */
            if (sscanf(line + 2, "%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",
                       &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d//%*d %d//%*d %d//%*d",
                       &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d/%*d %d/%*d %d/%*d",
                       &a, &b, &c) == 3 ||
                sscanf(line + 2, "%d %d %d", &a, &b, &c) == 3) {
                /* OBJ indices are 1-based. */
                m->faces[fi++] = (pamo_tri){{a - 1, b - 1, c - 1}};
            }
        }
    }
    m->n_verts = vi;
    m->n_faces = fi;

    fclose(f);
    printf("Loaded %s: %zu verts, %zu faces\n", path, vi, fi);
    return PAMO_OK;
}

static pamo_error save_obj(const char *path, const pamo_mesh *m) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Error: cannot write %s\n", path);
        return PAMO_ERR_IO;
    }

    for (size_t i = 0; i < m->n_verts; i++) {
        if (!m->vert_alive[i]) continue;
        fprintf(f, "v %f %f %f\n", m->verts[i].x, m->verts[i].y,
                m->verts[i].z);
    }

    /* OBJ indices are 1-based. */
    for (size_t i = 0; i < m->n_faces; i++) {
        if (!m->face_alive[i]) continue;
        fprintf(f, "f %d %d %d\n",
                m->faces[i].v[0] + 1,
                m->faces[i].v[1] + 1,
                m->faces[i].v[2] + 1);
    }

    fclose(f);
    printf("Saved %s: %zu verts, %zu faces\n", path, m->n_verts, m->n_faces);
    return PAMO_OK;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --input <in.obj> --output <out.obj> "
            "--ratio <0.001> [--disable_stage1] [--disable_stage3]\n", prog);
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *output_path = NULL;
    double ratio = 0.1;
    bool stage1 = true;
    bool stage3 = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--ratio") == 0 && i + 1 < argc) {
            ratio = atof(argv[++i]);
        } else if (strcmp(argv[i], "--disable_stage1") == 0) {
            stage1 = false;
        } else if (strcmp(argv[i], "--disable_stage3") == 0) {
            stage3 = false;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!input_path || !output_path) {
        usage(argv[0]);
        return 1;
    }

    pamo_allocator alloc = pamo_tracking_allocator_create();
    pamo_mesh mesh;
    pamo_error err;

    err = load_obj(input_path, &mesh, &alloc);
    if (err != PAMO_OK) {
        fprintf(stderr, "Error loading: %s\n", pamo_error_string(err));
        return 1;
    }

    printf("# input verts : %zu\n", mesh.n_verts);
    printf("# input faces : %zu\n", mesh.n_faces);
    printf("Stage1 : %s\n", stage1 ? "True" : "False");
    printf("Stage3 : %s\n", stage3 ? "True" : "False");

    /* Stage 1: Remeshing (TODO) */
    if (stage1) {
        fprintf(stderr, "Stage 1 not yet implemented, skipping.\n");
    }

    /* Stage 2: Simplification */
    {
        pamo_simplify_opts opts = pamo_simplify_opts_default();
        opts.target_faces = (int32_t)((double)mesh.n_faces * ratio);
        if (opts.target_faces < 10) opts.target_faces = 10;

        printf("Target faces : %d\n", opts.target_faces);

        err = pamo_simplify(&mesh, &opts);
        if (err != PAMO_OK) {
            fprintf(stderr, "Simplification error: %s\n",
                    pamo_error_string(err));
        }
    }

    /* Stage 3: SAFE Projection */
    if (stage3) {
        /* Use the original mesh as ground truth. */
        pamo_mesh gt_mesh;
        err = load_obj(input_path, &gt_mesh, &alloc);
        if (err == PAMO_OK) {
            pamo_safe_opts sopts = pamo_safe_opts_default();
            fprintf(stderr, "Running Stage 3: SAFE projection...\n");
            err = pamo_safe_project(&mesh, &gt_mesh, &sopts, &alloc);
            if (err != PAMO_OK) {
                fprintf(stderr, "Stage 3 error: %s\n",
                        pamo_error_string(err));
            }
            pamo_mesh_destroy(&gt_mesh);
        }
    }

    err = save_obj(output_path, &mesh);
    if (err != PAMO_OK) {
        fprintf(stderr, "Error saving: %s\n", pamo_error_string(err));
    }

    printf("# output verts : %zu\n", pamo_mesh_count_alive_verts(&mesh));
    printf("# output faces : %zu\n", pamo_mesh_count_alive_faces(&mesh));

    pamo_mesh_destroy(&mesh);
    pamo_tracking_allocator_report(&alloc);
    if (pamo_tracking_allocator_has_leaks(&alloc)) {
        fprintf(stderr, "WARNING: memory leaks detected!\n");
    }
    pamo_tracking_allocator_destroy(&alloc);

    return 0;
}
