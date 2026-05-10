/*
 * Copyright 2024 Light Transport Entertainment Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C bridge to lightrt's TriangleBVH for high-performance BVH ray
 * traversal and nearest-point queries.
 */
#include "lightrt.hh"

extern "C" {

/* Opaque handle to lightrt TriangleBVH. */
typedef struct lightrt_bvh lightrt_bvh;

/* Build a TriangleBVH from triangle data.
 * verts: float[n_verts * 3], faces: int32_t[n_faces * 3]
 * Returns NULL on failure. */
lightrt_bvh *lightrt_bvh_build(const float *verts, int32_t n_verts,
                                const int32_t *faces, int32_t n_faces) {
    if (!verts || !faces || n_verts < 0 || n_faces < 0) return nullptr;
    std::vector<lightrt::Triangle> tris((size_t)n_faces);
    for (int32_t i = 0; i < n_faces; i++) {
        int32_t i0 = faces[i * 3 + 0];
        int32_t i1 = faces[i * 3 + 1];
        int32_t i2 = faces[i * 3 + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            i0 >= n_verts || i1 >= n_verts || i2 >= n_verts) {
            return nullptr;
        }
        tris[(size_t)i].v0 = lightrt::Vec3(verts[i0*3], verts[i0*3+1], verts[i0*3+2]);
        tris[(size_t)i].v1 = lightrt::Vec3(verts[i1*3], verts[i1*3+1], verts[i1*3+2]);
        tris[(size_t)i].v2 = lightrt::Vec3(verts[i2*3], verts[i2*3+1], verts[i2*3+2]);
    }

    auto *bvh = new (std::nothrow) lightrt::TriangleBVH();
    if (!bvh) return nullptr;

    lightrt::BVHBuildConfig config;
    if (!bvh->build(tris, config)) {
        delete bvh;
        return nullptr;
    }
    return reinterpret_cast<lightrt_bvh *>(bvh);
}

/* Destroy a TriangleBVH. */
void lightrt_bvh_destroy(lightrt_bvh *bvh) {
    if (!bvh) return;
    delete reinterpret_cast<lightrt::TriangleBVH *>(bvh);
}

/* Ray-triangle traversal.  Returns triangle index or -1 if no hit.
 * ray_origin/ray_dir: float[3], hit_t: output distance. */
int32_t lightrt_bvh_traverse(const lightrt_bvh *bvh,
                              const float *ray_origin,
                              const float *ray_dir,
                              float tmax,
                              float *hit_t) {
    if (!bvh || !ray_origin || !ray_dir || !hit_t) return -1;
    const auto *b = reinterpret_cast<const lightrt::TriangleBVH *>(bvh);
    lightrt::Ray ray(
        lightrt::Vec3(ray_origin[0], ray_origin[1], ray_origin[2]),
        lightrt::Vec3(ray_dir[0], ray_dir[1], ray_dir[2]),
        1e-6f, tmax);

    float u, v;
    uint32_t tri_id = b->traverse(ray, *hit_t, u, v);
    if (tri_id == lightrt::kInvalidIndex) return -1;
    return (int32_t)tri_id;
}

/* Any-hit traversal.  Returns 1 if any hit found, 0 otherwise. */
int lightrt_bvh_any_hit(const lightrt_bvh *bvh,
                         const float *ray_origin,
                         const float *ray_dir,
                         float tmax) {
    if (!bvh || !ray_origin || !ray_dir) return 0;
    const auto *b = reinterpret_cast<const lightrt::TriangleBVH *>(bvh);
    lightrt::Ray ray(
        lightrt::Vec3(ray_origin[0], ray_origin[1], ray_origin[2]),
        lightrt::Vec3(ray_dir[0], ray_dir[1], ray_dir[2]),
        1e-6f, tmax);
    return b->traverseAnyHit(ray) ? 1 : 0;
}

/* Count ALL intersections along a ray (for parity-based inside/outside).
 * Returns the number of intersections. */
int32_t lightrt_bvh_count_hits(const lightrt_bvh *bvh,
                                const float *ray_origin,
                                const float *ray_dir,
                                float tmax) {
    if (!bvh || !ray_origin || !ray_dir) return 0;
    const auto *b = reinterpret_cast<const lightrt::TriangleBVH *>(bvh);
    lightrt::Ray ray(
        lightrt::Vec3(ray_origin[0], ray_origin[1], ray_origin[2]),
        lightrt::Vec3(ray_dir[0], ray_dir[1], ray_dir[2]),
        1e-6f, tmax);

    lightrt::MultiHitResult result;
    uint32_t n = b->traverseMultiHit(ray, result, 1024);
    return (int32_t)n;
}

/* Get sorted hit t-values along a ray.
 * Writes up to max_hits t-values into out_t (sorted ascending).
 * Returns the number of hits written. */
int32_t lightrt_bvh_multi_hit(const lightrt_bvh *bvh,
                               const float *ray_origin,
                               const float *ray_dir,
                               float tmax,
                               float *out_t, int32_t max_hits) {
    if (!bvh || !ray_origin || !ray_dir || !out_t || max_hits <= 0) return 0;
    const auto *b = reinterpret_cast<const lightrt::TriangleBVH *>(bvh);
    lightrt::Ray ray(
        lightrt::Vec3(ray_origin[0], ray_origin[1], ray_origin[2]),
        lightrt::Vec3(ray_dir[0], ray_dir[1], ray_dir[2]),
        1e-6f, tmax);

    lightrt::MultiHitResult result;
    uint32_t n = b->traverseMultiHit(ray, result, (uint32_t)max_hits);

    /* Copy and sort t-values. */
    int32_t count = (int32_t)n;
    if (count > max_hits) count = max_hits;
    for (int32_t i = 0; i < count; i++) {
        out_t[i] = result.hits[i].t;
    }
    /* Insertion sort (usually few hits). */
    for (int32_t i = 1; i < count; i++) {
        float key = out_t[i];
        int32_t j = i - 1;
        while (j >= 0 && out_t[j] > key) {
            out_t[j + 1] = out_t[j];
            j--;
        }
        out_t[j + 1] = key;
    }
    return count;
}

/* Nearest distance query: find distance to closest triangle.
 * Returns squared distance. point: float[3]. */
float lightrt_bvh_nearest_dist_sq(const lightrt_bvh *bvh,
                                   const float *point) {
    if (!bvh || !point) return std::numeric_limits<float>::infinity();
    const auto *b = reinterpret_cast<const lightrt::TriangleBVH *>(bvh);
    float dist_sq = std::numeric_limits<float>::infinity();
    b->queryNearest(lightrt::Vec3(point[0], point[1], point[2]), dist_sq);
    return dist_sq;
}

/* Bounded nearest distance query.  Initialize with max_dist_sq to
 * enable early termination in the BVH traversal. */
float lightrt_bvh_nearest_bounded(const lightrt_bvh *bvh,
                                   const float *point,
                                   float max_dist_sq) {
    if (!bvh || !point) return max_dist_sq;
    const auto *b = reinterpret_cast<const lightrt::TriangleBVH *>(bvh);
    float dist_sq = max_dist_sq;
    b->queryNearest(lightrt::Vec3(point[0], point[1], point[2]), dist_sq);
    return dist_sq;
}

} /* extern "C" */
