// Copyright 2024 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Static 3-D KD-tree on a Float32Array of xyz triplets. Bulk-built once,
// then queried for nearest-neighbour distances. Used by the diff-colour
// pipeline (browser + node).

export class KDTree3 {
    // points: Float32Array, length = 3 * N (xyz triplets)
    constructor(points) {
        this.points = points;
        this.n = (points.length / 3) | 0;
        this.idx = new Int32Array(this.n);
        for (let i = 0; i < this.n; i++) this.idx[i] = i;
        if (this.n > 0) {
            this.#build(0, this.n, 0);
        }
    }

    // In-place quickselect on this.idx[lo..hi) over axis, then recurse.
    #build(lo, hi, axis) {
        if (hi - lo <= 1) return;
        const mid = (lo + hi) >> 1;
        this.#nthElement(lo, hi, mid, axis);
        const next = (axis + 1) % 3;
        this.#build(lo, mid, next);
        this.#build(mid + 1, hi, next);
    }

    // Quickselect: arrange this.idx[lo..hi) so that this.idx[k] is the
    // (k - lo)-th smallest by axis component.
    #nthElement(lo, hi, k, axis) {
        while (hi - lo > 1) {
            const pivot = this.#partition(lo, hi, axis);
            if (pivot === k) return;
            if (k < pivot) hi = pivot;
            else lo = pivot + 1;
        }
    }

    #partition(lo, hi, axis) {
        const points = this.points, idx = this.idx;
        // Median-of-three for the pivot to avoid worst-case on sorted input.
        const mid = (lo + hi) >> 1;
        const a = idx[lo], b = idx[mid], c = idx[hi - 1];
        const va = points[a*3 + axis], vb = points[b*3 + axis], vc = points[c*3 + axis];
        let pivotIdx;
        if (va < vb) {
            if (vb < vc) pivotIdx = mid;
            else if (va < vc) pivotIdx = hi - 1;
            else pivotIdx = lo;
        } else {
            if (va < vc) pivotIdx = lo;
            else if (vb < vc) pivotIdx = hi - 1;
            else pivotIdx = mid;
        }
        // Move pivot to end.
        const tmp = idx[pivotIdx]; idx[pivotIdx] = idx[hi - 1]; idx[hi - 1] = tmp;
        const pivotVal = points[idx[hi - 1] * 3 + axis];
        let store = lo;
        for (let i = lo; i < hi - 1; i++) {
            if (points[idx[i] * 3 + axis] < pivotVal) {
                const t = idx[store]; idx[store] = idx[i]; idx[i] = t;
                store++;
            }
        }
        const t2 = idx[store]; idx[store] = idx[hi - 1]; idx[hi - 1] = t2;
        return store;
    }

    // Nearest-vertex query. Returns { index, distSq, dist }.
    nearest(qx, qy, qz) {
        const result = { bestIdx: -1, bestDistSq: Infinity };
        if (this.n > 0) this.#nearest(0, this.n, 0, qx, qy, qz, result);
        return {
            index: result.bestIdx,
            distSq: result.bestDistSq,
            dist: Math.sqrt(result.bestDistSq),
        };
    }

    #nearest(lo, hi, axis, qx, qy, qz, result) {
        if (hi - lo <= 0) return;
        const mid = (lo + hi) >> 1;
        const i = this.idx[mid];
        const px = this.points[i*3], py = this.points[i*3+1], pz = this.points[i*3+2];
        const dx = px - qx, dy = py - qy, dz = pz - qz;
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < result.bestDistSq) {
            result.bestDistSq = d2;
            result.bestIdx = i;
        }
        if (hi - lo === 1) return;
        const q = axis === 0 ? qx : axis === 1 ? qy : qz;
        const p = axis === 0 ? px : axis === 1 ? py : pz;
        const diff = q - p;
        const next = (axis + 1) % 3;
        if (diff < 0) {
            this.#nearest(lo, mid, next, qx, qy, qz, result);
            if (diff * diff < result.bestDistSq) {
                this.#nearest(mid + 1, hi, next, qx, qy, qz, result);
            }
        } else {
            this.#nearest(mid + 1, hi, next, qx, qy, qz, result);
            if (diff * diff < result.bestDistSq) {
                this.#nearest(lo, mid, next, qx, qy, qz, result);
            }
        }
    }
}

// For each vertex of `from`, compute the nearest-vertex distance into `to`.
// Both are Float32Array xyz triplets. Returns Float32Array of distances.
export function vertexDistances(from, to) {
    const tree = new KDTree3(to);
    const n = from.length / 3;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const r = tree.nearest(from[i*3], from[i*3+1], from[i*3+2]);
        out[i] = r.dist;
    }
    return out;
}
