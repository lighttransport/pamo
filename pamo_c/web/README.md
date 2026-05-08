# pamo_c web/CLI demo

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

## Prerequisites

- **Emscripten** for the WASM build. Activate it with e.g.
  `source /mnt/nvme02/work/emsdk/emsdk_env.sh`.
- **Node.js ≥ 20** (Emscripten ships one at `<emsdk>/node/.../bin/node`).
- The repo's CUDA Python venv (`.venv-cuda12` from `setup_uv_cuda12.sh`)
  for anything that calls the original `pamo` package.

## 1. Build the WASM module

```sh
cd pamo_c/web/wasm
./build.sh
```

Outputs `pamo.mjs` (~9 KB) and `pamo.wasm` (~60 KB).

## 2. Node.js CLI

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

## 3. Browser demo

Start the server with the CUDA venv (so the `/process` endpoint can import
`pamo`):

```sh
/mnt/nvme02/work/pamo/.venv-cuda12/bin/python pamo_c/web/server/server.py
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

## Notes

- "Geometric diff" is **per-vertex distance to the nearest vertex of the
  reference mesh** (KD-tree). Same metric as `pamo_c/verify/verify_main.c`.
- The WASM build is single-threaded for portability; pthreads would need
  `-pthread` plus COOP/COEP headers on the server.
- Stage 1 (Dual MC at R=128 in WASM, R=256 in Python) can be memory-heavy.
  If the Python side OOMs on the GPU, untick **Stage 1** in the demo or
  pass `--disable_stage1` to the CLI.
- The demo server is HTTP only (no auth), bound to `127.0.0.1`. Don't
  expose it.

## Re-running the verification suite

The earlier `pamo_c/verify/verify_compare.py` script is independent of
this directory and still works:

```sh
.venv-cuda12/bin/python pamo_c/verify/verify_compare.py \
    -i mesh/Dino_*.obj -o pamo_c/verify/out -r 0.1 \
    --disable_stage1 --disable_stage3
```
