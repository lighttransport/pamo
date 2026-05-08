#!/usr/bin/env bash
# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Build pamo as a WebAssembly module usable from both Node and browsers.
#
# Outputs (next to this script):
#   pamo.mjs   — ES module loader emitted by Emscripten
#   pamo.wasm  — the compiled WebAssembly binary
#
# Prereqs: emcc on PATH (e.g. `source /path/to/emsdk/emsdk_env.sh`).

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v emcc >/dev/null 2>&1; then
  echo "error: emcc not found on PATH. Run e.g.:" >&2
  echo "  source <emsdk>/emsdk_env.sh" >&2
  exit 1
fi

PAMO_C_ROOT="$(cd ../.. && pwd)"
BUILD_DIR="${PWD}/build"

echo "==> Configuring libpamo for WASM (build dir: ${BUILD_DIR})"
emcmake cmake -S "${PAMO_C_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPAMO_BUILD_EXAMPLES=OFF \
    -DPAMO_BUILD_TESTS=OFF \
    -DPAMO_BUILD_VERIFY=OFF \
    >/dev/null

echo "==> Building libpamo.a"
emmake make -C "${BUILD_DIR}" pamo -j

echo "==> Linking pamo.mjs / pamo.wasm"
emcc pamo_wasm.c "${BUILD_DIR}/src/libpamo.a" \
    -I"${PAMO_C_ROOT}/include" \
    -O3 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s ENVIRONMENT=web,node \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=64MB \
    -s STACK_SIZE=4MB \
    -s EXPORTED_FUNCTIONS=_pamo_wasm_run,_pamo_wasm_n_verts,_pamo_wasm_n_faces,_pamo_wasm_verts_ptr,_pamo_wasm_faces_ptr,_pamo_wasm_reset,_malloc,_free \
    -s EXPORTED_RUNTIME_METHODS=HEAPF32,HEAP32,HEAPU8 \
    -o pamo.mjs

echo "==> Done. Outputs:"
ls -lh pamo.mjs pamo.wasm
