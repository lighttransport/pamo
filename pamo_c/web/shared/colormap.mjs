// Copyright 2024 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Diff-magnitude → RGB colour ramp. Default ramp is blue → white → red,
// matching the user's request ("large = red, small = blue").

// Map a normalised t in [0, 1] to an RGB triplet [r, g, b] in [0, 1].
//   t = 0   → blue   (0, 0, 1)
//   t = 0.5 → white  (1, 1, 1)
//   t = 1   → red    (1, 0, 0)
function rampBlueWhiteRed(t) {
    if (t <= 0) return [0, 0, 1];
    if (t >= 1) return [1, 0, 0];
    if (t < 0.5) {
        const k = t * 2;            // 0 → 1 across the lower half
        return [k, k, 1];           // blue → white
    }
    const k = (t - 0.5) * 2;        // 0 → 1 across the upper half
    return [1, 1 - k, 1 - k];       // white → red
}

// Build a per-vertex RGB Float32Array from a per-vertex distance array.
// `scale` saturates the colour at this distance value (caller picks e.g.
// 5 % of the input mesh diameter so visualisations are comparable).
export function colourise(distances, scale) {
    const n = distances.length;
    const out = new Float32Array(n * 3);
    const denom = scale > 0 ? scale : 1;
    for (let i = 0; i < n; i++) {
        const t = distances[i] / denom;
        const [r, g, b] = rampBlueWhiteRed(t);
        out[i*3]     = r;
        out[i*3 + 1] = g;
        out[i*3 + 2] = b;
    }
    return out;
}

// Summary statistics for a distance array (used in CLI report + browser
// metric panel).
export function distanceStats(distances) {
    const n = distances.length;
    if (n === 0) return { n: 0, max: 0, mean: 0, rms: 0, p95: 0 };
    let max = 0, sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) {
        const d = distances[i];
        if (d > max) max = d;
        sum += d;
        sumSq += d * d;
    }
    const sorted = distances.slice().sort();
    return {
        n,
        max,
        mean: sum / n,
        rms: Math.sqrt(sumSq / n),
        p95: sorted[Math.min(n - 1, Math.floor(n * 0.95))],
    };
}
