# Building PaMO with uv

Two build scripts are provided depending on your CUDA version:

| Script | CUDA | Venv | Stage 3 |
|--------|------|------|---------|
| `setup_uv.sh` | 13.x (`cu130`) | `.venv` | Stages 1+2 only |
| `setup_uv_cuda12.sh` | 12.9 (`cu129`) | `.venv-cuda12` | All 3 stages |

## Prerequisites

- Linux x86_64
- NVIDIA GPU with CUDA toolkit installed
- [uv](https://docs.astral.sh/uv/) package manager
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### For CUDA 12.9 (Stage 3 support)

- CUDA 12.9 toolkit (e.g. `/usr/local/cuda-12.9`)
- `cuda-nvrtc-dev` package for your CUDA version:
  ```
  sudo apt install cuda-nvrtc-dev-12-9
  ```

## Quick Start

### CUDA 13.x (Stages 1+2 only)

```bash
./setup_uv.sh
source .venv/bin/activate
```

### CUDA 12.9 (All 3 stages)

```bash
./setup_uv_cuda12.sh
source .venv-cuda12/bin/activate
```

## What the Scripts Do

1. Initialize git submodules (CUDA 12 script only, for the warp fork)
2. Install Python 3.12 via `uv python install`
3. Create a virtual environment
4. Install PyTorch with appropriate CUDA wheels
5. Install base dependencies (numpy, libigl, trimesh, scipy, networkx)
6. **Stage 1 (Remeshing)**: Install `cumesh2sdf` and `pdmc`
7. **Stage 2 (Simplification)**: Build the CUDA extension in `simp_cuda/`
8. **Stage 3 (Safe Projection)**: Install `pamo_safe_project`
   - CUDA 12: builds the custom warp fork (`simp_cuda/safe_project/warp_`) from source
   - CUDA 13: installs upstream `warp-lang` from PyPI (Stage 3 does not fully work)

## Environment Variables

| Variable | Default (CUDA 13) | Default (CUDA 12) | Description |
|----------|-------------------|--------------------|-------------|
| `PYTHON_VERSION` | `3.12` | `3.12` | Python version to install |
| `VENV_DIR` | `.venv` | `.venv-cuda12` | Virtual environment directory |
| `CUDA_PATH` | `/usr/local/cuda` | `/usr/local/cuda-12.9` | Path to CUDA toolkit |

## Running the Demo

```bash
source .venv-cuda12/bin/activate   # or .venv for CUDA 13
mkdir -p examples
python example.py --input ./mesh/BirdHouse_B019SXLRJ2_MetalLeafRoofGreenWalls_TU.obj \
                  --output ./examples/BirdHouse.obj --ratio 0.001
```

Or run all demos:

```bash
bash demo.sh
```

Use `--disable_stage3` to skip Stage 3 (Safe Projection):

```bash
python example.py --input ./mesh/BirdHouse_B019SXLRJ2_MetalLeafRoofGreenWalls_TU.obj \
                  --output ./examples/BirdHouse.obj --ratio 0.001 --disable_stage3
```

### Stage 3 memory note

Stage 3 pre-allocates large GPU buffers (`max_blocks = 2^25`). On GPUs with less than
16 GB VRAM, this may cause OOM errors. The default settings target 16+ GB GPUs.

## CUDA 13 Compatibility Notes

### What works on CUDA 13

- **Stage 1 (Remeshing)**: Fully working
- **Stage 2 (Simplification)**: Fully working (with CCCL include path fix in
  `simp_cuda/setup.py`)

### What does not work on CUDA 13

**Stage 3 (Safe Projection)** requires the custom warp fork
([Rabbit-Hu/warp](https://github.com/Rabbit-Hu/warp), pinned at warp 1.0.0-beta.2)
which adds `wp.spd_project_blocks` -- a custom built-in function for SPD matrix
projection. This function does not exist in any official `warp-lang` release.

The custom fork cannot compile against CUDA 13 due to:

- **CCCL header reorganization**: `cub/` and `thrust/` headers moved under
  `include/cccl/` (partially fixable by adding include paths)
- **Removed CUB APIs**: `cub::CountingInputIterator` removed entirely in CCCL 2.x
  shipped with CUDA 13 (requires code rewrite)
- **Removed CUDA typedefs**: `PFN_cuGetProcAddress` unversioned typedef removed
- **Dropped GPU architectures**: `compute_52` through `compute_70` no longer supported

### Source-level compatibility shims

The following changes in `pamo_safe_project` allow the same source to work with both
the warp fork (1.0.0-beta.2) and upstream warp-lang (1.12+):

- **`wp.select()` compat**: A shim in `__init__.py` aliases `wp.where` to `wp.select`
  when running on newer warp versions that removed `wp.select`
- **`owner` parameter**: `wp_slice()` in `utils.py` conditionally passes the `owner`
  parameter only when the warp version supports it
- **CCCL include path**: `simp_cuda/setup.py` auto-detects and adds the
  `include/cccl/` path for CUDA 13+ thrust/cub headers
