// SPDX-License-Identifier: Apache-2.0
// pamo_c embind bindings for browser/Node WASM.
//
// JS-facing API (all mesh I/O is typed arrays):
//
//   const mod = await (await import('./pamo.js')).default();
//
//   const out = mod.remesh({
//     vertices: Float64Array,   // flat xyz
//     indices:  Int32Array,     // flat triangle indices
//     options:  { resolution: 64, band: 0.05 }   // optional
//   });
//   // -> { vertices: Float64Array, indices: Int32Array, error: "OK" }
//
//   mod.simplify({ vertices, indices, options: { target_faces: 500 } });
//   mod.safe_project({ vertices, indices, gt_vertices, gt_indices, options: {...} });
//
// Allocator is hidden: the default malloc-backed allocator is used
// internally.

#include <emscripten/bind.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "pamo/pamo.h"
}

using namespace emscripten;

namespace {

template <typename T>
static std::vector<T> readTypedArray(const val &arr) {
    if (arr.isUndefined() || arr.isNull()) return {};
    const size_t n = arr["length"].as<size_t>();
    std::vector<T> out(n);
    val dst = val(typed_memory_view(n, out.data()));
    dst.call<void>("set", arr);
    return out;
}

// Build a pamo_mesh from flat typed arrays.  Returns an initialized
// mesh (caller must call pamo_mesh_destroy).
static pamo_error buildMesh(pamo_mesh *m, const val &verts_v,
                            const val &idx_v,
                            const pamo_allocator *alloc) {
    auto verts = readTypedArray<double>(verts_v);
    auto idx = readTypedArray<int32_t>(idx_v);
    if (verts.size() % 3u != 0 || idx.size() % 3u != 0) {
        return PAMO_ERR_INVALID_ARG;
    }
    const size_t n_verts = verts.size() / 3u;
    const size_t n_faces = idx.size() / 3u;

    pamo_error err = pamo_mesh_create(m, n_verts, n_faces, alloc);
    if (err != PAMO_OK) return err;

    for (size_t i = 0; i < n_verts; ++i) {
        m->verts[i].x = verts[3 * i + 0];
        m->verts[i].y = verts[3 * i + 1];
        m->verts[i].z = verts[3 * i + 2];
    }
    for (size_t i = 0; i < n_faces; ++i) {
        m->faces[i].v[0] = idx[3 * i + 0];
        m->faces[i].v[1] = idx[3 * i + 1];
        m->faces[i].v[2] = idx[3 * i + 2];
    }
    return PAMO_OK;
}

// Serialize an (alive-subset of a) pamo_mesh to {vertices, indices}.
static val meshToJS(const pamo_mesh *m) {
    // Compact to dense arrays while respecting vert_alive / face_alive.
    std::vector<int32_t> vert_map(m->n_verts_cap, -1);
    std::vector<double> verts;
    verts.reserve(m->n_verts * 3);
    for (size_t i = 0; i < m->n_verts_cap; ++i) {
        if (!m->vert_alive || m->vert_alive[i]) {
            vert_map[i] = static_cast<int32_t>(verts.size() / 3);
            verts.push_back(m->verts[i].x);
            verts.push_back(m->verts[i].y);
            verts.push_back(m->verts[i].z);
        }
    }
    std::vector<int32_t> idx;
    idx.reserve(m->n_faces * 3);
    for (size_t i = 0; i < m->n_faces_cap; ++i) {
        if (m->face_alive && !m->face_alive[i]) continue;
        int32_t a = vert_map[m->faces[i].v[0]];
        int32_t b = vert_map[m->faces[i].v[1]];
        int32_t c = vert_map[m->faces[i].v[2]];
        if (a < 0 || b < 0 || c < 0) continue;
        idx.push_back(a);
        idx.push_back(b);
        idx.push_back(c);
    }

    val Float64Array = val::global("Float64Array");
    val Int32Array = val::global("Int32Array");
    val v_out = Float64Array.new_(verts.size());
    if (!verts.empty()) {
        v_out.call<void>("set",
            val(typed_memory_view(verts.size(), verts.data())));
    }
    val i_out = Int32Array.new_(idx.size());
    if (!idx.empty()) {
        i_out.call<void>("set",
            val(typed_memory_view(idx.size(), idx.data())));
    }

    val obj = val::object();
    obj.set("vertices", v_out);
    obj.set("indices", i_out);
    return obj;
}

static val resultOk(val mesh_obj) {
    mesh_obj.set("error", std::string("OK"));
    return mesh_obj;
}

static val resultErr(pamo_error e) {
    val obj = val::object();
    val F64 = val::global("Float64Array");
    val I32 = val::global("Int32Array");
    obj.set("vertices", F64.new_(0));
    obj.set("indices", I32.new_(0));
    obj.set("error", std::string(pamo_error_string(e)));
    return obj;
}

template <typename T>
static void pick(const val &js, const char *k, T &dst) {
    if (js.isUndefined() || js.isNull()) return;
    val v = js[k];
    if (!v.isUndefined() && !v.isNull()) dst = v.as<T>();
}

// ── Stage 1: remesh ────────────────────────────────────────────────
static val remesh(val input) {
    pamo_allocator alloc = pamo_default_allocator();
    pamo_mesh in_mesh;
    pamo_error err = buildMesh(&in_mesh, input["vertices"],
                               input["indices"], &alloc);
    if (err != PAMO_OK) return resultErr(err);

    pamo_remesh_opts opts = pamo_remesh_opts_default();
    val o = input["options"];
    pick(o, "resolution", opts.resolution);
    pick(o, "band", opts.band);

    pamo_mesh out_mesh;
    err = pamo_remesh(&out_mesh, &in_mesh, &opts, &alloc);
    pamo_mesh_destroy(&in_mesh);
    if (err != PAMO_OK) return resultErr(err);

    val js = meshToJS(&out_mesh);
    pamo_mesh_destroy(&out_mesh);
    return resultOk(js);
}

// ── Stage 2: simplify (in-place) ───────────────────────────────────
static val simplify(val input) {
    pamo_allocator alloc = pamo_default_allocator();
    pamo_mesh mesh;
    pamo_error err = buildMesh(&mesh, input["vertices"],
                               input["indices"], &alloc);
    if (err != PAMO_OK) return resultErr(err);

    pamo_simplify_opts opts = pamo_simplify_opts_default();
    val o = input["options"];
    pick(o, "target_faces", opts.target_faces);
    pick(o, "min_faces", opts.min_faces);
    pick(o, "max_iters", opts.max_iters);
    pick(o, "tolerance", opts.tolerance);
    pick(o, "max_undo_retries", opts.max_undo_retries);
    pick(o, "skinny_area_threshold", opts.skinny_area_threshold);
    pick(o, "skinny_penalty_weight", opts.skinny_penalty_weight);
    pick(o, "cost_range", opts.cost_range);
    pick(o, "check_self_intersection", opts.check_self_intersection);

    err = pamo_simplify(&mesh, &opts);
    if (err != PAMO_OK) {
        pamo_mesh_destroy(&mesh);
        return resultErr(err);
    }
    pamo_mesh_compact(&mesh);
    val js = meshToJS(&mesh);
    pamo_mesh_destroy(&mesh);
    return resultOk(js);
}

// ── Stage 3: SAFE projection ───────────────────────────────────────
static val safe_project(val input) {
    pamo_allocator alloc = pamo_default_allocator();
    pamo_mesh mesh, gt;
    pamo_error err = buildMesh(&mesh, input["vertices"],
                               input["indices"], &alloc);
    if (err != PAMO_OK) return resultErr(err);
    err = buildMesh(&gt, input["gt_vertices"], input["gt_indices"], &alloc);
    if (err != PAMO_OK) {
        pamo_mesh_destroy(&mesh);
        return resultErr(err);
    }

    pamo_safe_opts opts = pamo_safe_opts_default();
    val o = input["options"];
    pick(o, "n_outer_iters", opts.n_outer_iters);
    pick(o, "n_newton_iters", opts.n_newton_iters);
    pick(o, "n_cg_iters", opts.n_cg_iters);
    pick(o, "n_line_search_iters", opts.n_line_search_iters);
    pick(o, "d_hat", opts.d_hat);
    pick(o, "coll_stiffness", opts.coll_stiffness);
    pick(o, "elas_young_modulus", opts.elas_young_modulus);
    pick(o, "elas_poisson_ratio", opts.elas_poisson_ratio);
    pick(o, "hinge_stiffness", opts.hinge_stiffness);
    pick(o, "dist_stiffness", opts.dist_stiffness);
    pick(o, "ccd_slackness", opts.ccd_slackness);
    pick(o, "ccd_thickness", opts.ccd_thickness);
    pick(o, "ccd_max_iters", opts.ccd_max_iters);
    pick(o, "spd_max_iters", opts.spd_max_iters);
    pick(o, "enable_collision", opts.enable_collision);
    pick(o, "enable_hinge", opts.enable_hinge);

    err = pamo_safe_project(&mesh, &gt, &opts, &alloc);
    pamo_mesh_destroy(&gt);
    if (err != PAMO_OK) {
        pamo_mesh_destroy(&mesh);
        return resultErr(err);
    }
    val js = meshToJS(&mesh);
    pamo_mesh_destroy(&mesh);
    return resultOk(js);
}

}  // namespace

EMSCRIPTEN_BINDINGS(pamo_module) {
    function("remesh", &remesh);
    function("simplify", &simplify);
    function("safe_project", &safe_project);
}
