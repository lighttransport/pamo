# pamo_c web

This directory hosts two independent WebAssembly entry points:

1. **Embind library build** (`binding.cc`, `embind-utils.hpp`,
   `bootstrap-linux.sh`, `js/`, `demo/`) — single-threaded WASM with
   embind bindings exposing `remesh` / `simplify` / `safeProject`
   directly to JS. Suitable for embedding pamo_c as a library.

2. **Python-vs-WASM comparison demo** (`wasm/`, `shared/`, `cli/`,
   `server/`, `web/`) — three.js viewports that run the original GPU
   `pamo` (via a local Python HTTP server) and `pamo_c` (WASM) on the
   same input, with colour-coded per-vertex distance diffs. Includes
   a Node.js CLI variant.

The two builds are independent: each has its own bootstrap script and
artifact directory, so you can build either or both.

---

## 1. Embind library build

Single-threaded browser/Node WebAssembly build of pamo_c with embind
bindings.

### Build

```bash
# requires emcmake on PATH
./bootstrap-linux.sh
# or:  BUILD_TYPE=Debug ./bootstrap-linux.sh
```

Artifacts land in `js/src/pamo.{js,wasm}`.

### Usage

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

### Notes

- Single-threaded: `PAMO_USE_PTHREADS=OFF`, `PAMO_USE_LIGHTRT=OFF`.
- The native `pamo_allocator` is **not** exposed to JS; the bindings use
  `pamo_default_allocator()` internally.
- Input / output meshes are flat typed arrays
  (`Float64Array` verts, `Int32Array` indices). The binding compacts
  internally before returning, so dead verts/faces are already gone.

---

## 2. Python-vs-WASM comparison demo

Two zero-install ways to compare the original Python `pamo` against the
C-port `pamo_c`:

- **Node.js CLI** (`cli/compare.mjs`) — runs both pipelines on a single OBJ
  and writes per-vertex-coloured OBJs visualising the geometric divergence.
- **Browser demo** (`web/`) — three side-by-side three.js viewports, sliders,
  and live colour-coded diff. Backed by a tiny Python HTTP server
  (`server/server.py`) that runs the original pamo and serves the static
  files.

`pamo_c` is built once as a WebAssembly module (`wasm/pamo.mjs` +
`pamo.wasm`) and shared between Node and browser.

```
pamo_c/web/
├── wasm/      pamo_wasm.c + build.sh                  → pamo.mjs / pamo.wasm
├── shared/    obj.mjs, kdtree.mjs, colormap.mjs, …    pure ESM, used by both
├── cli/       package.json + compare.mjs              Node CLI
├── server/    server.py                               static + /process
└── web/       index.html + styles.css + app.mjs      three.js demo
```

### Prerequisites

- **Emscripten** for the WASM build. Activate it with e.g.
  `source <emsdk>/emsdk_env.sh` (set `EMSDK_DIR` for `run-web-demo.sh`
  to find a non-default location).
- **Node.js ≥ 20** (Emscripten ships one at `<emsdk>/node/.../bin/node`).
- The repo's CUDA Python venv (`.venv-cuda12` from `setup_uv_cuda12.sh`)
  for anything that calls the original `pamo` package.

### 2.1 Build the WASM module

```sh
cd pamo_c/web/wasm
./build.sh
```

Outputs `pamo.mjs` (~9 KB) and `pamo.wasm` (~60 KB).

### 2.2 Node.js CLI

```sh
cd pamo_c/web/cli
node compare.mjs \
    -i ../../../mesh/Dino_B015CZP872_SmallOrangeTRex.obj \
    -o ./out -r 0.1 \
    --disable_stage1 --disable_stage3
```

Writes five OBJs into `out/`:

| file | meaning |
|---|---|
| `pamo.obj` | Python pamo output (uncoloured) |
| `pamo_c.obj` | pamo_c WASM output (uncoloured) |
| `pamo_vs_input.obj` | pamo, vertex-coloured by distance to input |
| `pamo_c_vs_input.obj` | pamo_c, vertex-coloured by distance to input |
| `pamo_vs_pamo_c.obj` | pamo_c, vertex-coloured by distance to pamo |

Vertex colours use the OBJ extension `v x y z r g b`; ramp is
**blue (close) → white → red (far)**, saturating at 5 % of the input
diameter by default (`--colour-scale-pct`).

Useful flags:

- `-r / --ratio` — target face ratio (default `0.1`).
- `--disable_stage1`, `--disable_stage3` — skip heavy stages.
- `--skip-python`, `--skip-wasm` — reuse cached outputs in `out/`.
- `--python-bin` — override the Python interpreter.

### 2.3 Browser demo

Start the server with the CUDA venv (so the `/process` endpoint can import
`pamo`):

```sh
# from the repo root
.venv-cuda12/bin/python pamo_c/web/server/server.py
```

Open <http://127.0.0.1:5050/>. Choose a sample (or upload an `.obj`),
adjust the **ratio** slider and **stage 1 / stage 3** toggles, hit **Apply**
to re-run both pipelines. The **View** dropdown switches between:

- *Plain* — both outputs in solid colour.
- *Diff vs input* — outputs coloured by per-vertex distance to the input.
- *Diff vs other port* — pamo coloured by distance to pamo_c (and vice-versa).
- *Wireframe* — overlay mesh edges.

The three OrbitControls cameras are linked, so dragging in any viewport
rotates all three.

### Notes

- "Geometric diff" is **per-vertex distance to the nearest vertex of the
  reference mesh** (KD-tree). Same metric as `pamo_c/verify/verify_main.c`.
- The WASM build is single-threaded for portability; pthreads would need
  `-pthread` plus COOP/COEP headers on the server.
- Stage 1 (Dual MC at R=128 in WASM, R=256 in Python) can be memory-heavy.
  If the Python side OOMs on the GPU, untick **Stage 1** in the demo or
  pass `--disable_stage1` to the CLI.
- The demo server is HTTP only (no auth), bound to `127.0.0.1`. Don't
  expose it.

### Re-running the verification suite

The earlier `pamo_c/verify/verify_compare.py` script is independent of
this directory and still works:

```sh
.venv-cuda12/bin/python pamo_c/verify/verify_compare.py \
    -i mesh/Dino_*.obj -o pamo_c/verify/out -r 0.1 \
    --disable_stage1 --disable_stage3
```
