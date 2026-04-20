# pamo_c WASM/Emscripten build

Single-threaded browser/Node WebAssembly build of pamo_c with embind bindings.

## Build

```bash
# requires emcmake on PATH
./bootstrap-linux.sh
# or:  BUILD_TYPE=Debug ./bootstrap-linux.sh
```

Artifacts land in `js/src/pamo.{js,wasm}`.

## Usage

```js
import { loadPamo } from './js/PamoLoader.js';
const pamo = await loadPamo();

// Stage 1 — volumetric remeshing
const r = pamo.remesh(vertices, indices, { resolution: 64 });
// r: { vertices: Float64Array, indices: Int32Array, error: "OK" | ... }

// Stage 2 — iterative simplification
const s = pamo.simplify(r.vertices, r.indices, { target_faces: 500 });

// Stage 3 — SAFE projection back toward a ground-truth mesh
const f = pamo.safeProject(s.vertices, s.indices, gtV, gtI, {
  n_outer_iters: 3,
});
```

## Notes

- Single-threaded: `PAMO_USE_PTHREADS=OFF`, `PAMO_USE_LIGHTRT=OFF`.
- The native `pamo_allocator` is **not** exposed to JS; the bindings use
  `pamo_default_allocator()` internally.
- Input / output meshes are flat typed arrays
  (`Float64Array` verts, `Int32Array` indices). The binding compacts
  internally before returning, so dead verts/faces are already gone.
