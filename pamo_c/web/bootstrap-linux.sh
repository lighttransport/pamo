#!/usr/bin/env bash
# Build the pamo_c WASM module. Requires `emcmake` on PATH.
set -eu

BUILD_TYPE="${BUILD_TYPE:-MinSizeRel}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf build
mkdir build
emcmake cmake \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -Bbuild
cmake --build build -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "Output:"
ls -la js/src/pamo.js js/src/pamo.wasm 2>/dev/null || true
