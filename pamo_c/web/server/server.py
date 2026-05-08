#!/usr/bin/env python3
# Copyright 2024 Light Transport Entertainment Inc.
# SPDX-License-Identifier: Apache-2.0
"""HTTP server backing the pamo_c web demo.

Serves the static web/, shared/, and wasm/ directories, plus:
  GET  /samples              — JSON list of mesh/*.obj filenames
  GET  /samples/<name>       — stream that OBJ
  POST /process              — run Python pamo on the posted mesh,
                               return {verts, faces, ms}

The /process endpoint imports the `pamo` package in-process, so this script
must be launched with the venv that has pamo installed (e.g.
`<repo>/.venv-cuda12/bin/python`).

Run (from the repo root):
  .venv-cuda12/bin/python pamo_c/web/server/server.py
Open:
  http://127.0.0.1:5050/
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent              # pamo_c/web
PAMO_C_ROOT = WEB_ROOT.parent       # pamo_c
REPO_ROOT = PAMO_C_ROOT.parent      # repo root
MESH_DIR = REPO_ROOT / "mesh"

# Make sure .mjs files are served as JavaScript so browsers ESM-import them.
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


# ── HMR (Vite-style live reload) ─────────────────────────────────────
#
# Opt-in via `--watch`. Polls the served static dirs for mtime changes and
# fans the latest tick out to all SSE clients on /__hmr; the injected
# client script does a full window.location.reload() on each tick.
#
# Polling rather than inotify keeps it stdlib-only and portable. Cost is
# negligible (a few hundred stat() calls every 500 ms).

_HMR_CLIENT_JS = b"""
(() => {
  if (window.__pamoHmrAttached) return;
  window.__pamoHmrAttached = true;
  let connected = false;
  function connect() {
    const es = new EventSource('/__hmr');
    es.addEventListener('open', () => { connected = true; });
    es.addEventListener('reload', () => {
      console.log('[hmr] reload');
      es.close();
      window.location.reload();
    });
    es.addEventListener('error', () => {
      es.close();
      // Server gone -- retry every second so a restarted server picks
      // up the same tab.
      setTimeout(connect, 1000);
    });
  }
  connect();
})();
"""

_HMR_INJECT = (b"<script src=\"/__hmr_client.js\"></script>\n</body>")


class HmrHub:
    """Multi-subscriber pub/sub keyed on monotonic 'tick' counter."""
    def __init__(self):
        self._cv = threading.Condition()
        self._tick = 0

    def bump(self):
        with self._cv:
            self._tick += 1
            self._cv.notify_all()

    def wait_for(self, last_seen, timeout):
        """Block until tick > last_seen or timeout elapses. Returns the
        current tick (which may equal last_seen on timeout)."""
        with self._cv:
            self._cv.wait_for(lambda: self._tick > last_seen, timeout=timeout)
            return self._tick


def _start_watcher(roots, hub: HmrHub, exts, interval=0.5):
    """Background thread: scan `roots` for any file with a suffix in `exts`
    and bump `hub` whenever the max mtime advances."""
    def scan():
        latest = 0.0
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.suffix.lower() not in exts:
                    continue
                try:
                    m = p.stat().st_mtime
                except OSError:
                    continue
                if m > latest:
                    latest = m
        return latest

    last = scan()

    def loop():
        nonlocal last
        while True:
            time.sleep(interval)
            try:
                cur = scan()
            except Exception:
                continue
            if cur > last:
                last = cur
                hub.bump()
                print("[hmr] file change detected → notifying clients",
                      flush=True)

    t = threading.Thread(target=loop, name="pamo-hmr-watch", daemon=True)
    t.start()


# Globals populated by main() when --watch is set.
HMR_ENABLED = False
HMR_HUB: HmrHub | None = None


# ── Lazy-loaded pamo runner (only when /process is hit) ──────────────

_pamo = None

def _get_pamo():
    """Import the heavy pamo + torch stack once on first /process call."""
    global _pamo
    if _pamo is not None:
        return _pamo
    print("[server] loading pamo (torch + cumesh2sdf + pamo_safe_project)...",
          flush=True)
    import numpy as np
    import torch
    import trimesh
    import pamo as pamo_module
    import pamo_safe_project.config as pamo_cfg
    _pamo = {
        "np": np, "torch": torch, "trimesh": trimesh,
        "PaMO": pamo_module.PaMO, "Stage3Config": pamo_cfg.Stage3Config,
    }
    return _pamo


def _scaled_stage3_sizes(n_verts_in, n_faces_in, use_stage1, torch=None):
    """Pick Stage3Config sizes based on the actual input mesh + free GPU
    memory.

    The stock Stage3Config pre-allocates for million-vertex inputs
    (max_particles=1<<20, max_gt_particles=1<<24, max_blocks=1<<25), which
    OOMs on small GPUs even for tiny meshes (shelf.obj 1.5K verts → 1.6 GB
    just for one collision-energy buffer).  Scale particles/gt counts to
    the input mesh, and pick max_blocks to fit the free GPU memory budget.

    max_blocks is the dominant memory consumer (the collision/BVH buffers
    are arrays of vec3 indexed by block, e.g. dd_dx[(MB, 4), vec3] alone
    uses 48 * MB bytes). We cap it at a fraction of the currently free
    GPU memory and floor it at a heuristic minimum derived from
    max_particles, so very small meshes still get a usable contact buffer."""
    n_in = max(int(n_verts_in), int(n_faces_in))
    if use_stage1:
        # Dual-MC at R=256 can produce ~5x the input vert count in the
        # worst case for complex geometry; bump the floor.
        n_in = max(n_in * 8, 1 << 17)  # min 128k
    max_particles    = max(n_in * 4, 1 << 14)              # min 16k
    max_gt_particles = max(int(n_verts_in) * 2, 1 << 13)

    # GPU memory budget for max_blocks-sized arrays.
    # Stage 3 has multiple buffers indexed by block (dd_dx, hess blocks,
    # contact pair indices, etc.) — empirically ~64 bytes/block in total
    # is a safe upper-bound estimate for the static allocations.
    BYTES_PER_BLOCK = 64
    BUDGET_FRACTION = 0.30                 # use ≤ 30% of free for these
    FLOOR_BLOCKS    = max_particles * 8    # heuristic minimum
    CEIL_BLOCKS     = 1 << 25              # stock library default

    free_mb = budget_mb = 0
    cap = FLOOR_BLOCKS
    if torch is not None:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_mb = free_bytes // (1024 * 1024)
            budget_bytes = int(free_bytes * BUDGET_FRACTION)
            budget_mb = budget_bytes // (1024 * 1024)
            cap = max(FLOOR_BLOCKS, budget_bytes // BYTES_PER_BLOCK)
        except Exception:
            pass
    max_blocks = min(cap, CEIL_BLOCKS)

    return {
        "max_particles":    max_particles,
        "max_gt_particles": max_gt_particles,
        "max_gt_samples":   1 << 14,                # stock default
        "max_blocks":       int(max_blocks),
        "_free_mb":         int(free_mb),           # for logging only
        "_budget_mb":       int(budget_mb),
    }


def run_python_pamo(verts, faces, ratio, use_stage1, use_stage3):
    """Run the original GPU pamo and return (verts, faces, ms)."""
    p = _get_pamo()
    np, torch, trimesh, PaMO = p["np"], p["torch"], p["trimesh"], p["PaMO"]
    Stage3Config = p["Stage3Config"]

    verts_np = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    faces_np = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    input_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

    # Monkey-patch Stage3Config to size against the actual input mesh.
    # PaMO.__init__ calls Stage3Config() with no args, so we override the
    # defaults for the duration of this PaMO construction.
    runner = None
    orig_init = Stage3Config.__init__
    if use_stage3:
        sizes = _scaled_stage3_sizes(len(verts_np), len(faces_np),
                                     use_stage1, torch=torch)
        # Strip telemetry-only keys before applying.
        log_meta = {k: v for k, v in sizes.items() if k.startswith("_")}
        sizes = {k: v for k, v in sizes.items() if not k.startswith("_")}
        def sized_init(self):
            orig_init(self)
            for k, v in sizes.items():
                setattr(self, k, v)
        Stage3Config.__init__ = sized_init
        print(f"[server] Stage3Config sizes for {len(verts_np)}v/"
              f"{len(faces_np)}f: {sizes}  "
              f"(GPU free {log_meta.get('_free_mb','?')} MB, "
              f"budget {log_meta.get('_budget_mb','?')} MB)",
              flush=True)
    try:
        runner = PaMO(input_mesh, use_stage1=use_stage1, use_stage3=use_stage3)
        t0 = time.time()
        out_v, out_f = runner.run(
            torch.from_numpy(verts_np).float().cuda(),
            torch.from_numpy(faces_np).int().cuda(),
            ratio=ratio,
        )
        ms = (time.time() - t0) * 1000.0
    finally:
        Stage3Config.__init__ = orig_init
        # Free CUDA / warp arrays so the next /process call starts clean.
        # Without this, repeated requests pile up Stage3System allocations.
        del runner
        torch.cuda.empty_cache()

    if hasattr(out_v, "cpu"):
        out_v = out_v.cpu().numpy()
        out_f = out_f.cpu().numpy()
    return (np.asarray(out_v, dtype=np.float32),
            np.asarray(out_f, dtype=np.int32),
            ms)


# ── Static file serving ──────────────────────────────────────────────

# Prefix → on-disk dir. Order matters: longest first.
ROUTES = [
    ("/shared/", WEB_ROOT / "shared"),
    ("/wasm/",   WEB_ROOT / "wasm"),
    ("/web/",    WEB_ROOT / "web"),
]


def resolve_static(path: str) -> Path | None:
    if path == "/" or path == "":
        return WEB_ROOT / "web" / "index.html"
    for prefix, root in ROUTES:
        if path.startswith(prefix):
            rel = path[len(prefix):]
            target = (root / rel).resolve()
            if target.is_file() and root.resolve() in target.parents:
                return target
    # Allow naked /index.html, /styles.css, /app.mjs from the web/ root.
    direct = (WEB_ROOT / "web" / path.lstrip("/")).resolve()
    if direct.is_file() and (WEB_ROOT / "web").resolve() in direct.parents:
        return direct
    return None


# ── Request handler ──────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] "
                         f"{self.address_string()} {fmt % args}\n")

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        ctype, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        # In --watch mode, inject the HMR client into served HTML so a
        # tab opened on / picks up future file changes automatically.
        if HMR_ENABLED and ctype == "text/html" and b"</body>" in data:
            data = data.replace(b"</body>", _HMR_INJECT, 1)
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        # COOP/COEP not strictly needed (single-threaded WASM), keep CORS open
        # for `fetch` from same origin to be explicit.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_hmr_stream(self):
        """SSE endpoint: hold the connection open and emit a `reload`
        event whenever the watcher's tick advances."""
        if not HMR_ENABLED or HMR_HUB is None:
            return self._send_json(404, {"error": "hmr disabled"})
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            # Initial comment line flushes headers in some browsers.
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            last = HMR_HUB._tick
            while True:
                cur = HMR_HUB.wait_for(last, timeout=15.0)
                if cur > last:
                    last = cur
                    self.wfile.write(
                        f"event: reload\ndata: {cur}\n\n".encode())
                else:
                    # Keep-alive ping so proxies / browsers don't time out.
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client disconnected, normal

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/__hmr":
            return self._serve_hmr_stream()
        if path == "/__hmr_client.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(_HMR_CLIENT_JS)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(_HMR_CLIENT_JS)
            return
        if path == "/samples":
            samples = sorted(p.name for p in MESH_DIR.glob("*.obj"))
            return self._send_json(200, {"samples": samples})
        if path.startswith("/samples/"):
            name = path[len("/samples/"):]
            target = (MESH_DIR / name).resolve()
            if target.is_file() and MESH_DIR.resolve() in target.parents:
                return self._send_file(target)
            return self._send_json(404, {"error": "not found"})
        target = resolve_static(path)
        if target is None:
            return self._send_json(404, {"error": "not found", "path": path})
        return self._send_file(target)

    def do_POST(self):
        if self.path != "/process":
            return self._send_json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 256 * 1024 * 1024:
            return self._send_json(413, {"error": "bad payload size"})
        try:
            body = self.rfile.read(length)
            req = json.loads(body)
            verts = req["verts"]
            faces = req["faces"]
            ratio = float(req.get("ratio", 0.1))
            use_stage1 = bool(req.get("stage1", False))
            use_stage3 = bool(req.get("stage3", False))
            out_v, out_f, ms = run_python_pamo(
                verts, faces, ratio, use_stage1, use_stage3,
            )
            return self._send_json(200, {
                "verts": out_v.flatten().tolist(),
                "faces": out_f.flatten().tolist(),
                "ms": ms,
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._send_json(500, {"error": str(e)})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--watch", action="store_true",
                   help="Vite-style live reload: poll JS/HTML/CSS in "
                        "shared/ + web/ and auto-reload connected tabs.")
    args = p.parse_args()

    if not (WEB_ROOT / "wasm" / "pamo.mjs").exists():
        print("warning: wasm/pamo.mjs not found. Build it first:\n"
              "  cd pamo_c/web/wasm && ./build.sh", file=sys.stderr)

    if args.watch:
        global HMR_ENABLED, HMR_HUB
        HMR_ENABLED = True
        HMR_HUB = HmrHub()
        _start_watcher(
            roots=[WEB_ROOT / "shared", WEB_ROOT / "web"],
            hub=HMR_HUB,
            exts={".js", ".mjs", ".html", ".css"},
        )
        print("[hmr] watching shared/ + web/ for .js/.mjs/.html/.css "
              "changes (full page reload on change)", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] http://{args.host}:{args.port}/   "
          f"(serving {WEB_ROOT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
