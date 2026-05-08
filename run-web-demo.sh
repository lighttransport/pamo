#!/usr/bin/env bash
# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Launch the pamo vs pamo_c web demo on http://127.0.0.1:5050/
#
# - Sources Emscripten and builds pamo.mjs / pamo.wasm if missing.
# - Picks the CUDA-12 venv if it exists (so the /process endpoint can
#   import the pamo package), else falls back to the plain venv.
# - Forwards extra args (e.g. --port 8000) to server.py.
#
# Env overrides:
#   EMSDK_DIR    — emsdk root (default: /mnt/nvme02/work/emsdk)
#   PAMO_VENV    — path to a venv to use (overrides auto-pick)
#   PAMO_PORT    — server port (default: 5050)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_ROOT="${REPO_ROOT}/pamo_c/web"

EMSDK_DIR="${EMSDK_DIR:-/mnt/nvme02/work/emsdk}"
PAMO_PORT="${PAMO_PORT:-5050}"

# ── 1. Build the WASM module if missing ────────────────────────────────
if [[ ! -f "${WEB_ROOT}/wasm/pamo.mjs" || ! -f "${WEB_ROOT}/wasm/pamo.wasm" ]]; then
    echo "==> WASM artefacts missing, building..."
    if [[ ! -f "${EMSDK_DIR}/emsdk_env.sh" ]]; then
        echo "error: cannot find ${EMSDK_DIR}/emsdk_env.sh." >&2
        echo "       Set EMSDK_DIR to your emsdk install path." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${EMSDK_DIR}/emsdk_env.sh" >/dev/null
    (cd "${WEB_ROOT}/wasm" && ./build.sh)
fi

# ── 2. Pick the Python interpreter (must have pamo importable) ─────────
if [[ -n "${PAMO_VENV:-}" ]]; then
    PYTHON_BIN="${PAMO_VENV}/bin/python"
elif [[ -x "${REPO_ROOT}/.venv-cuda12/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv-cuda12/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    echo "error: no Python venv found. Run setup_uv_cuda12.sh first," >&2
    echo "       or set PAMO_VENV=<path-to-venv>." >&2
    exit 1
fi

# ── 3. Launch ──────────────────────────────────────────────────────────
echo "==> Python : ${PYTHON_BIN}"
echo "==> URL    : http://127.0.0.1:${PAMO_PORT}/"
echo "==> Ctrl-C to stop"
exec "${PYTHON_BIN}" "${WEB_ROOT}/server/server.py" --port "${PAMO_PORT}" "$@"
