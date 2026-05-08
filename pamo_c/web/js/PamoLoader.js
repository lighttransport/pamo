// SPDX-License-Identifier: Apache-2.0
// Thin ES6 wrapper around the pamo_c WASM module.
//
//   import { loadPamo } from './PamoLoader.js';
//   const pamo = await loadPamo();
//   const r1 = pamo.remesh(verts, idx, { resolution: 64 });
//   const r2 = pamo.simplify(r1.vertices, r1.indices, { target_faces: 500 });
//   const r3 = pamo.safeProject(r2.vertices, r2.indices, gtV, gtI, opts);

let _modulePromise = null;

export async function loadPamo(opts = {}) {
    if (!_modulePromise) {
        const url = opts.scriptURL ?? new URL('./src/pamo.js', import.meta.url).href;
        const mod = await import(/* @vite-ignore */ url);
        _modulePromise = mod.default(opts.moduleArg ?? {});
    }
    const Module = await _modulePromise;

    const toF64 = (a) => a instanceof Float64Array ? a : new Float64Array(a);
    const toI32 = (a) => a instanceof Int32Array ? a : new Int32Array(a);

    return {
        remesh(vertices, indices, options = {}) {
            return Module.remesh({
                vertices: toF64(vertices),
                indices: toI32(indices),
                options,
            });
        },
        simplify(vertices, indices, options = {}) {
            return Module.simplify({
                vertices: toF64(vertices),
                indices: toI32(indices),
                options,
            });
        },
        safeProject(vertices, indices, gtVertices, gtIndices, options = {}) {
            return Module.safe_project({
                vertices: toF64(vertices),
                indices: toI32(indices),
                gt_vertices: toF64(gtVertices),
                gt_indices: toI32(gtIndices),
                options,
            });
        },
        _raw: Module,
    };
}
