#!/usr/bin/env python3
# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the original Python pamo and the C-port pamo_example on the same input
OBJ, then compare the two output OBJs with a sampled symmetric Hausdorff
metric.

Run with the project venv (CUDA build of pamo is required for the Python half):

    ../../.venv-cuda12/bin/python verify_compare.py \\
        --input ../../mesh/Dino_B015CZP872_SmallOrangeTRex.obj \\
        --outdir ./out --ratio 0.1

The C binary defaults to `pamo_example` on PATH or under
`pamo_c/build/example/`; override with `--c-binary`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunTime:
    """Wall-clock + child CPU time captured around a subprocess."""
    wall: float = 0.0
    user_cpu: float = 0.0
    sys_cpu: float = 0.0
    rss_peak_kb: int = 0  # max resident set of children, in kilobytes

    @property
    def cpu(self) -> float:
        return self.user_cpu + self.sys_cpu


def _run_timed(cmd: list[str], cwd: str | None = None) -> RunTime:
    """Run ``cmd`` and return a RunTime; raises if the subprocess fails.
    Uses os.wait4 to harvest per-process rusage (ru_maxrss is per-child, not
    cumulative, unlike resource.getrusage(RUSAGE_CHILDREN))."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=cwd)
    _, status, ru = os.wait4(proc.pid, 0)
    wall = time.monotonic() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise SystemExit(f"subprocess failed (exit {proc.returncode}): {cmd[0]}")
    return RunTime(
        wall=wall,
        user_cpu=ru.ru_utime,
        sys_cpu=ru.ru_stime,
        rss_peak_kb=int(ru.ru_maxrss),
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
PAMO_C_ROOT = Path(__file__).resolve().parents[1]


def find_python_interpreter() -> str:
    """Return a Python interpreter that has the `pamo` package importable."""
    if _interpreter_has_pamo(sys.executable):
        return sys.executable
    for candidate in (
        REPO_ROOT / ".venv-cuda12" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.exists() and _interpreter_has_pamo(str(candidate)):
            return str(candidate)
    raise SystemExit(
        "Could not find a Python interpreter with the `pamo` package "
        "installed. Run setup_uv.sh or setup_uv_cuda12.sh first, then "
        "re-run this script with that venv's python."
    )


def _interpreter_has_pamo(python_path: str) -> bool:
    try:
        res = subprocess.run(
            [python_path, "-c", "import pamo"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0


def find_c_binary(override: str | None) -> str:
    if override:
        if not Path(override).exists():
            raise SystemExit(f"--c-binary not found: {override}")
        return override
    on_path = shutil.which("pamo_example")
    if on_path:
        return on_path
    for candidate in (
        PAMO_C_ROOT / "build" / "example" / "pamo_example",
        REPO_ROOT / "build" / "example" / "pamo_example",
    ):
        if candidate.exists():
            return str(candidate)
    raise SystemExit(
        "pamo_example binary not found. Build it with:\n"
        "  cmake -S pamo_c -B pamo_c/build -DPAMO_BUILD_EXAMPLES=ON && "
        "cmake --build pamo_c/build -j\n"
        "or pass --c-binary <path>."
    )


def run_python_pipeline(
    python_bin: str,
    example_py: Path,
    input_obj: Path,
    output_obj: Path,
    ratio: float,
    disable_stage1: bool,
    disable_stage3: bool,
) -> RunTime:
    cmd = [
        python_bin,
        str(example_py),
        "-i", str(input_obj),
        "-o", str(output_obj),
        "-r", str(ratio),
    ]
    if disable_stage1:
        cmd.append("--disable_stage1")
    if disable_stage3:
        cmd.append("--disable_stage3")
    print(f"[python/GPU] $ {' '.join(cmd)}", flush=True)
    rt = _run_timed(cmd, cwd=str(example_py.parent))
    if not output_obj.exists():
        raise SystemExit(f"Python pipeline produced no output at {output_obj}.")
    return rt


def run_c_pipeline(
    c_binary: str,
    input_obj: Path,
    output_obj: Path,
    ratio: float,
    disable_stage1: bool,
    disable_stage3: bool,
) -> RunTime:
    cmd = [
        c_binary,
        "--input", str(input_obj),
        "--output", str(output_obj),
        "--ratio", str(ratio),
    ]
    if disable_stage1:
        cmd.append("--disable_stage1")
    if disable_stage3:
        cmd.append("--disable_stage3")
    print(f"[c/CPU]      $ {' '.join(cmd)}", flush=True)
    rt = _run_timed(cmd)
    if not output_obj.exists():
        raise SystemExit(f"C pipeline produced no output at {output_obj}.")
    return rt


def distance_metrics(mesh_a, mesh_b, n_samples: int) -> dict:
    """Sampled symmetric distance metrics between two surfaces.

    Both meshes are dense-sampled, then nearest-neighbour distances are
    computed via KD-trees against the other side's point cloud (avoids the
    `rtree` dependency that ``trimesh.proximity`` needs). With ~20k samples
    per side this approaches the true mesh-to-mesh Hausdorff/Chamfer."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples)
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)

    both = np.concatenate([d_ab, d_ba])
    return {
        "hausdorff": float(max(np.max(d_ab), np.max(d_ba))),
        "chamfer_mean": float((np.mean(d_ab) + np.mean(d_ba)) / 2.0),
        "rms": float(np.sqrt(np.mean(both * both))),
        "p95": float(np.quantile(both, 0.95)),
    }


def aspect_ratios(mesh) -> "np.ndarray":
    """Per-triangle quality in [0, 1]: ``4*sqrt(3)*area / sum(edge_len^2)``.
    1.0 = equilateral, 0 = degenerate sliver."""
    import numpy as np

    tris = mesh.triangles
    e0 = tris[:, 1] - tris[:, 0]
    e1 = tris[:, 2] - tris[:, 1]
    e2 = tris[:, 0] - tris[:, 2]
    edge_sq = (np.sum(e0 * e0, axis=1)
               + np.sum(e1 * e1, axis=1)
               + np.sum(e2 * e2, axis=1))
    area = mesh.area_faces
    out = np.zeros(len(tris), dtype=np.float64)
    nz = edge_sq > 1e-30
    out[nz] = 4.0 * np.sqrt(3.0) * area[nz] / edge_sq[nz]
    return out


def mesh_quality(mesh) -> dict:
    """Per-mesh shape quality: topology flags, area/volume, triangle and
    edge distributions, degenerate count."""
    import numpy as np

    areas = mesh.area_faces
    edges = mesh.edges_unique_length
    aq = aspect_ratios(mesh)
    return {
        "n_verts": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "area_min": float(areas.min()) if len(areas) else 0.0,
        "area_med": float(np.median(areas)) if len(areas) else 0.0,
        "area_max": float(areas.max()) if len(areas) else 0.0,
        "edge_min": float(edges.min()) if len(edges) else 0.0,
        "edge_med": float(np.median(edges)) if len(edges) else 0.0,
        "edge_max": float(edges.max()) if len(edges) else 0.0,
        "aspect_min": float(aq.min()) if len(aq) else 0.0,
        "aspect_med": float(np.median(aq)) if len(aq) else 0.0,
        "aspect_mean": float(aq.mean()) if len(aq) else 0.0,
        "n_degenerate": int(np.sum(areas < 1e-20)),
        "n_sliver": int(np.sum(aq < 0.1)),
    }


def print_quality_table(rows: dict[str, dict]) -> None:
    """Render a side-by-side quality table. ``rows`` maps label -> metrics."""
    labels = list(rows.keys())
    fmt_row = lambda name, fn: (
        f"  {name:<20}"
        + "".join(f"{fn(rows[lbl]):>16}" for lbl in labels)
    )

    def fnum(v, prec=6):
        if v is None:
            return "n/a"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, int):
            return f"{v:d}"
        return f"{v:.{prec}g}"

    header = f"  {'metric':<20}" + "".join(f"{lbl:>16}" for lbl in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(fmt_row("verts",            lambda r: fnum(r["n_verts"])))
    print(fmt_row("faces",            lambda r: fnum(r["n_faces"])))
    print(fmt_row("watertight",       lambda r: fnum(r["watertight"])))
    print(fmt_row("winding-consistent", lambda r: fnum(r["winding_consistent"])))
    print(fmt_row("surface area",     lambda r: fnum(r["surface_area"])))
    print(fmt_row("volume",           lambda r: fnum(r["volume"])))
    print(fmt_row("tri area min",     lambda r: fnum(r["area_min"])))
    print(fmt_row("tri area median",  lambda r: fnum(r["area_med"])))
    print(fmt_row("tri area max",     lambda r: fnum(r["area_max"])))
    print(fmt_row("edge len min",     lambda r: fnum(r["edge_min"])))
    print(fmt_row("edge len median",  lambda r: fnum(r["edge_med"])))
    print(fmt_row("edge len max",     lambda r: fnum(r["edge_max"])))
    print(fmt_row("aspect min",       lambda r: fnum(r["aspect_min"], 4)))
    print(fmt_row("aspect mean",      lambda r: fnum(r["aspect_mean"], 4)))
    print(fmt_row("aspect median",    lambda r: fnum(r["aspect_med"], 4)))
    print(fmt_row("# degenerate",     lambda r: fnum(r["n_degenerate"])))
    print(fmt_row("# sliver(<0.1)",   lambda r: fnum(r["n_sliver"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Input .obj")
    p.add_argument("-o", "--outdir", default="./verify_out",
                   help="Directory to receive out_python.obj and out_c.obj")
    p.add_argument("-r", "--ratio", type=float, default=0.1)
    p.add_argument("--disable_stage1", action="store_true")
    p.add_argument("--disable_stage3", action="store_true")
    p.add_argument("--c-binary", default=None,
                   help="Path to pamo_example (default: search PATH and "
                        "pamo_c/build/example/)")
    p.add_argument("--python-example", default=str(REPO_ROOT / "example.py"),
                   help="Path to the Python pamo example.py")
    p.add_argument("--python-bin", default=None,
                   help="Override the Python interpreter used to run "
                        "example.py (must have pamo importable)")
    p.add_argument("--samples", type=int, default=20000,
                   help="Hausdorff surface samples per side (default 20000)")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="Fail if Hausdorff exceeds this %% of input diameter "
                        "(default 10)")
    p.add_argument("--skip-python", action="store_true",
                   help="Reuse an existing out_python.obj from --outdir")
    p.add_argument("--skip-c", action="store_true",
                   help="Reuse an existing out_c.obj from --outdir")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_obj = Path(args.input).resolve()
    if not input_obj.exists():
        raise SystemExit(f"Input not found: {input_obj}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_py = outdir / "out_python.obj"
    out_c = outdir / "out_c.obj"

    py_rt = RunTime()
    c_rt = RunTime()
    if not args.skip_python:
        python_bin = args.python_bin or find_python_interpreter()
        py_rt = run_python_pipeline(
            python_bin,
            Path(args.python_example).resolve(),
            input_obj, out_py, args.ratio,
            args.disable_stage1, args.disable_stage3,
        )
    elif not out_py.exists():
        raise SystemExit(f"--skip-python set but {out_py} missing.")

    if not args.skip_c:
        c_binary = find_c_binary(args.c_binary)
        c_rt = run_c_pipeline(
            c_binary, input_obj, out_c, args.ratio,
            args.disable_stage1, args.disable_stage3,
        )
    elif not out_c.exists():
        raise SystemExit(f"--skip-c set but {out_c} missing.")

    import numpy as np
    import trimesh

    in_mesh = trimesh.load(str(input_obj), force="mesh", process=False)
    py_mesh = trimesh.load(str(out_py), force="mesh", process=False)
    c_mesh = trimesh.load(str(out_c), force="mesh", process=False)

    bb = in_mesh.bounds
    diameter = float(np.linalg.norm(bb[1] - bb[0]))

    def _fmt_rt(rt: RunTime) -> str:
        if rt.wall == 0.0:
            return "(skipped)"
        rss_mb = rt.rss_peak_kb / 1024.0
        return (f"wall {rt.wall:7.2f}s  cpu(u+s) {rt.cpu:7.2f}s  "
                f"par {rt.cpu / rt.wall:4.2f}x  rss-peak {rss_mb:7.1f} MB")

    print("\n=== Processing time ===")
    print(f"  python (GPU+host): {_fmt_rt(py_rt)}")
    print(f"  c      (CPU)     : {_fmt_rt(c_rt)}")
    if py_rt.wall > 0 and c_rt.wall > 0:
        speedup = c_rt.wall / py_rt.wall
        verdict = (f"GPU is {speedup:.2f}x faster" if speedup > 1.0
                   else f"CPU is {1.0 / speedup:.2f}x faster")
        print(f"  wall speedup     : {speedup:.2f}x   ({verdict})")
        if c_rt.cpu > 0:
            print(f"  c CPU efficiency : {c_rt.cpu / c_rt.wall:.2f} cores busy "
                  f"on average (1.00 = single-threaded)")

    print("\n=== Mesh quality ===")
    quality = {
        "input":  mesh_quality(in_mesh),
        "python": mesh_quality(py_mesh),
        "c":      mesh_quality(c_mesh),
    }
    print_quality_table(quality)

    print("\n=== Fidelity to input ===")
    py_in = distance_metrics(py_mesh, in_mesh, args.samples)
    c_in = distance_metrics(c_mesh, in_mesh, args.samples)
    print(f"  {'metric':<22}{'python':>16}{'c':>16}")
    print("  " + "-" * 52)
    for k in ("hausdorff", "chamfer_mean", "rms", "p95"):
        rel_py = py_in[k] / diameter * 100.0 if diameter > 1e-10 else 0.0
        rel_c = c_in[k] / diameter * 100.0 if diameter > 1e-10 else 0.0
        print(f"  {k:<22}{py_in[k]:>10.6f} ({rel_py:4.2f}%) "
              f"{c_in[k]:>10.6f} ({rel_c:4.2f}%)")

    print("\n=== Python vs C output ===")
    py_vs_c = distance_metrics(py_mesh, c_mesh, args.samples)
    h_rel = py_vs_c["hausdorff"] / diameter * 100.0 if diameter > 1e-10 else 0.0
    cm_rel = py_vs_c["chamfer_mean"] / diameter * 100.0 if diameter > 1e-10 else 0.0
    print(f"  input diameter   : {diameter:.6f}")
    print(f"  Hausdorff abs/rel: {py_vs_c['hausdorff']:.6f}  ({h_rel:.3f}%)")
    print(f"  Chamfer  abs/rel : {py_vs_c['chamfer_mean']:.6f}  ({cm_rel:.3f}%)")
    print(f"  RMS distance     : {py_vs_c['rms']:.6f}")
    print(f"  P95 distance     : {py_vs_c['p95']:.6f}")
    if quality["python"]["surface_area"] > 0:
        area_pct = (quality["c"]["surface_area"]
                    / quality["python"]["surface_area"]) * 100.0
        print(f"  area ratio c/py  : {area_pct:.2f}%")
    print(f"  face ratio c/py  : "
          f"{(len(c_mesh.faces) / max(1, len(py_mesh.faces))) * 100.0:.2f}%")

    passed = h_rel <= args.threshold
    print(f"\nVerdict: {'PASS' if passed else 'FAIL'} "
          f"(Hausdorff threshold {args.threshold:.2f}% of diameter)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
