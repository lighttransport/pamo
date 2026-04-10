# Building PaMO with uv (setup_uv.sh)

## Prerequisites

- Linux x86_64
- NVIDIA GPU with CUDA toolkit installed (`/usr/local/cuda`)
- [uv](https://docs.astral.sh/uv/) package manager
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Quick Start

```bash
./setup_uv.sh
source .venv/bin/activate
```

## What the Script Does

1. Installs Python 3.12 via `uv python install`
2. Creates a virtual environment in `.venv`
3. Installs PyTorch with CUDA 13.0 wheels (`cu130`)
4. Installs base dependencies (numpy, libigl, trimesh, scipy, networkx, warp-lang)
5. **Stage 1 (Remeshing)**: Installs `cumesh2sdf` and `pdmc`
6. **Stage 2 (Simplification)**: Builds the CUDA extension in `simp_cuda/`
7. **Stage 3 (Safe Projection)**: Installs `pamo_safe_project`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_VERSION` | `3.12` | Python version to install |
| `VENV_DIR` | `.venv` | Virtual environment directory |
| `CUDA_PATH` | `/usr/local/cuda` | Path to CUDA toolkit |

Example with custom settings:

```bash
PYTHON_VERSION=3.12 CUDA_PATH=/usr/local/cuda-13.1 ./setup_uv.sh
```

## Running the Demo

```bash
source .venv/bin/activate
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

## CUDA 13 Compatibility Fixes

The following source-level fixes were applied for CUDA 13.x support:

### simp_cuda/setup.py

CUDA 13 moved thrust/cub/libcudacxx headers under `include/cccl/`. The CCCL include
path is automatically added when detected.

### simp_cuda/safe_project (pamo_safe_project)

- **`wp.select()` replaced with `wp.where()`**: The `wp.select()` API was removed in
  warp-lang 1.8+.
- **`owner` parameter removed from `wp.array()`**: The `owner=False` keyword argument
  was removed in newer warp-lang versions.

## Stage 3 Limitations on CUDA 13

Stage 3 (Safe Projection) **does not fully work** with CUDA 13 + upstream `warp-lang`.

The original project uses a custom warp fork
([Rabbit-Hu/warp](https://github.com/Rabbit-Hu/warp), pinned at warp 1.0.0-beta.2)
which adds `wp.spd_project_blocks` -- a custom built-in function for SPD matrix
projection used by the collision energy solver. This function does not exist in any
official `warp-lang` release.

The custom fork itself cannot compile against CUDA 13 due to:

- **CCCL header reorganization**: `cub/` and `thrust/` headers moved under
  `include/cccl/` (partially fixable by adding include paths)
- **Removed CUB APIs**: `cub::CountingInputIterator` was removed entirely in CCCL 2.x
  shipped with CUDA 13 (requires code rewrite)
- **Removed CUDA typedefs**: `PFN_cuGetProcAddress` unversioned typedef removed
  (fixable with `decltype`)
- **Dropped GPU architectures**: `compute_52` through `compute_70` are no longer
  supported (fixable by updating gencode list)

### Workarounds

- **Use `--disable_stage3`** to run Stages 1+2 only (remeshing + simplification).
  These work correctly on CUDA 13.
- **Use CUDA 12.x** if Stage 3 is required. The original `setup.sh` with conda
  targets CUDA 12.x where the custom warp fork compiles successfully.
- **Port the fork**: To enable Stage 3 on CUDA 13, the custom warp fork needs to be
  updated for CCCL 2.x compatibility (mainly replacing removed CUB iterators).
