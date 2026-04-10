#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv-cuda12}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.9}"

# ── Check uv is installed ──────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# ── Init git submodules ────────────────────────────────────────────
echo "==> Initializing git submodules..."
git submodule update --init --recursive

# ── Install Python via uv and create venv ──────────────────────────
echo "==> Installing Python ${PYTHON_VERSION} via uv..."
uv python install "$PYTHON_VERSION"

echo "==> Creating virtual environment in ${VENV_DIR}..."
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

# Activate (for subprocesses in this script)
source "${VENV_DIR}/bin/activate"

# ── Install PyTorch (CUDA 12.9) ───────────────────────────────────
echo "==> Installing PyTorch (CUDA 12.9)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# ── Install base dependencies ──────────────────────────────────────
echo "==> Installing base dependencies..."
uv pip install \
  "numpy==1.26.4" \
  "libigl==2.5.1" \
  "trimesh==4.4.0" \
  "scipy>=1.12.0" \
  "networkx"

# ── Stage 1: Remeshing ────────────────────────────────────────────
echo "==> Installing Stage 1: remeshing..."
uv pip install --no-build-isolation "git+https://github.com/eliphatfs/cumesh2sdf.git"
uv pip install --no-build-isolation pdmc

# ── Stage 2: Simplification (CUDA extension) ──────────────────────
echo "==> Installing Stage 2: simplification..."
cd simp_cuda
CUDA_HOME="$CUDA_PATH" uv pip install --no-build-isolation .
cd "$SCRIPT_DIR"

# ── Stage 3: Safe Projection (warp fork + CUDA extension) ─────────
echo "==> Installing Stage 3: safe projection (warp fork)..."
cd simp_cuda/safe_project/warp_
chmod +x ./tools/packman/packman
python build_lib.py --cuda_path "$CUDA_PATH"
uv pip install .
cd "$SCRIPT_DIR"

cd simp_cuda/safe_project
uv pip install --no-build-isolation .
cd "$SCRIPT_DIR"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete! (CUDA 12.9)"
echo "  Activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo "============================================"
