# PaMO Benchmark Results

GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM)

Timings are with warm warp kernel cache. First run adds ~40s of JIT compilation.

Stage 3 memory reduced (`max_blocks = 2^20`) to fit 8 GB VRAM. Default settings
(`max_blocks = 2^25`) target 16+ GB GPUs.

## CUDA 12.9 -- All 3 Stages (warp fork 1.0.0-beta.2, PyTorch 2.11.0+cu129)

| Mesh | Ratio | In V | In F | Out V | Out F | Init | Stage 1 | Stage 2 | Stage 3 | Total |
|------|------:|-----:|-----:|------:|------:|-----:|--------:|--------:|--------:|------:|
| BirdHouse | 0.1% | 145,491 | 249,680 | 153 | 310 | 0.18s | 0.095s | 0.393s | 1.15s | **1.82s** |
| Dino | 0.01% | 52,433 | 97,112 | 136 | 268 | 0.02s | 0.007s | 0.130s | 0.29s | **0.45s** |
| Dumbbell | 0.01% | 132,671 | 249,644 | 109 | 214 | 0.05s | 0.015s | 0.158s | 0.40s | **0.62s** |

## CUDA 13.0 -- Stages 1+2 only (upstream warp-lang 1.12.1, PyTorch 2.11.0+cu130)

| Mesh | Ratio | In V | In F | Out V | Out F | Init | Stage 1 | Stage 2 | Total |
|------|------:|-----:|-----:|------:|------:|-----:|--------:|--------:|------:|
| BirdHouse | 0.1% | 145,491 | 249,680 | 161 | 326 | 0.05s | 0.092s | 0.417s | **0.56s** |
| Dino | 0.01% | 52,433 | 97,112 | 132 | 260 | 0.02s | 0.007s | 0.128s | **0.15s** |
| Dumbbell | 0.01% | 132,671 | 249,644 | 112 | 220 | 0.04s | 0.015s | 0.136s | **0.20s** |

## Per-stage summary

- **Stage 1 (Remeshing)**: ~7--95 ms. Scales with mesh size. Uses cumesh2sdf + PDMC.
- **Stage 2 (Simplification)**: ~130--420 ms. Iterative CUDA edge collapse. Dominant cost when Stage 3 is disabled.
- **Stage 3 (Safe Projection)**: ~290 ms--1.15 s. Runs 5 Newton iterations with collision detection, CG solver, and CCD line search. Most expensive stage. Only available with CUDA 12.x + warp fork.
- **Init**: Includes `PaMO` constructor, warp system allocation, and energy calculator setup. Higher on first mesh due to warp device initialization.
