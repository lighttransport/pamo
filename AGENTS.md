# Repository Guidelines

## Project Structure & Module Organization
`README.md` is the primary entry point for installation and usage. The core package lives in `simp_cuda/`: `simp_cuda/pamo/` contains the Python API, `simp_cuda/src/` contains the CUDA/C++ extension sources, and `simp_cuda/safe_project/src/pamo_safe_project/` contains the stage-3 safe-projection code. The Blender add-on is under `pamo_blender/`. Demo meshes live in `mesh/`, container tooling in `docker/`, and top-level scripts such as `example.py`, `demo.sh`, and `setup.sh` are the main developer entry points.

## Build, Test, and Development Commands
Create the base environment with `conda env create -f env.yaml && conda activate pamo`. Install all three pipeline stages with `bash setup.sh`. For iterative work on the CUDA extension, use `cd simp_cuda && pip install .`. Build and install stage 3 directly with `cd simp_cuda/safe_project/warp_ && python build_lib.py --cuda_path /usr/local/cuda && pip install .`, then `cd .. && pip install .`. Run the sample pipeline with `bash demo.sh` or a single mesh with `python example.py --input ./mesh/Dino_B015CZP872_SmallOrangeTRex.obj --output ./examples/Dino.obj --ratio 0.0001`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions and variables, and CamelCase for classes such as `PaMO` and `PaSP`. Keep Python wrappers thin and push performance-critical logic into `simp_cuda/src/`. Match current filename patterns: CUDA headers use `.cuh`, CUDA sources use `.cu`, and Python packages expose minimal `__init__.py` surfaces. No formatter configuration is checked in, so keep changes PEP 8-aligned and avoid large unrelated rewrites.

## Testing Guidelines
Automated coverage is currently minimal and centered in `simp_cuda/safe_project/tests/`. Run targeted checks with `python -m pytest simp_cuda/safe_project/tests`. Because this project depends on CUDA, Warp, and PyTorch GPU execution, also include a manual reproduction command in your change notes when you touch kernels, build scripts, or projection logic.

## Commit & Pull Request Guidelines
The local checkout exposes very little commit history, so keep commits conservative: short imperative subjects, one logical change per commit, and enough body text to explain CUDA, dependency, or mesh-processing impacts. Pull requests should include the input mesh used, the exact command run, expected GPU/CUDA prerequisites, and before/after output notes or screenshots when behavior changes affect geometry or the Blender plugin.

## Environment & Configuration Tips
Assume a Linux CUDA environment unless you are explicitly working on the Windows Blender plugin path (`pamo_blender/cusimp.dll`). Do not hardcode local CUDA paths beyond build instructions; prefer documented flags such as `--cuda_path /usr/local/cuda`.
