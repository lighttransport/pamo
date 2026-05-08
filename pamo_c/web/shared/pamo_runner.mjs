// Copyright 2024 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Loads the Emscripten-built pamo.mjs WASM module and exposes a small
// `simplify` helper that handles HEAP marshalling. Works in both Node and
// browser (the WASM module is built with -s ENVIRONMENT=web,node).

let modulePromise = null;

// Load the WASM module exactly once. `loaderUrl` lets callers point at
// non-default locations; pass an absolute or page-relative URL.
export async function loadPamo(loaderUrl = './pamo.mjs') {
    if (!modulePromise) {
        const m = await import(loaderUrl);
        modulePromise = m.default();
    }
    const M = await modulePromise;
    return new PamoRunner(M);
}

class PamoRunner {
    constructor(M) { this.M = M; }

    // Run simplification on the given input mesh.
    //   verts: Float32Array (xyz triplets)
    //   faces: Int32Array (triangle indices)
    //   opts:  { ratio, useStage1, useStage3, sdfResolution }
    //          sdfResolution = 0 (default) → auto (Python rules: 256/128/64)
    //                        = positive int → force that R for Stage 1
    // Returns { verts: Float32Array, faces: Int32Array, ms: number }.
    simplify(verts, faces, {
        ratio = 0.1,
        useStage1 = false,
        useStage3 = false,
        sdfResolution = 0,
    } = {}) {
        const M = this.M;
        const nv = (verts.length / 3) | 0;
        const nf = (faces.length / 3) | 0;
        if (nv <= 0 || nf <= 0) {
            throw new Error('pamo.simplify: empty input mesh');
        }

        const vBytes = verts.length * 4;
        const fBytes = faces.length * 4;
        const vPtr = M._malloc(vBytes);
        const fPtr = M._malloc(fBytes);
        try {
            M.HEAPF32.set(verts, vPtr >> 2);
            M.HEAP32.set(faces, fPtr >> 2);

            const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
            const rc = M._pamo_wasm_run(
                vPtr, nv,
                fPtr, nf,
                ratio,
                useStage1 ? 1 : 0,
                useStage3 ? 1 : 0,
                sdfResolution | 0,
            );
            const ms = ((typeof performance !== 'undefined' ? performance : Date).now() - t0);
            if (rc !== 0) {
                throw new Error(`pamo_wasm_run failed (rc=${rc})`);
            }

            const outNv = M._pamo_wasm_n_verts();
            const outNf = M._pamo_wasm_n_faces();
            const outVerts = new Float32Array(
                M.HEAPF32.buffer,
                M._pamo_wasm_verts_ptr(),
                outNv * 3,
            ).slice();   // copy out of the heap before reset
            const outFaces = new Int32Array(
                M.HEAP32.buffer,
                M._pamo_wasm_faces_ptr(),
                outNf * 3,
            ).slice();

            return { verts: outVerts, faces: outFaces, ms };
        } finally {
            M._free(vPtr);
            M._free(fPtr);
            M._pamo_wasm_reset();
        }
    }
}
