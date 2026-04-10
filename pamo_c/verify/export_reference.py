#!/usr/bin/env python3
"""
Export reference data from the CUDA PaMO pipeline for verification
against the C implementation.

Usage:
    source ../.venv/bin/activate  # or .venv-cuda12
    python export_reference.py --input ../mesh/Dino_*.obj --output ref_dino/

Produces binary dumps at each stage boundary.
"""
import argparse
import os
import struct
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def write_mesh_bin(path, verts, faces):
    """Write mesh as binary: magic(4) version(4) nv(4) nf(4) verts(nv*3*f32) faces(nf*3*i32)"""
    nv, nf = len(verts), len(faces)
    with open(path, 'wb') as f:
        f.write(b'PAMO')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<II', nv, nf))
        f.write(verts.astype(np.float32).tobytes())
        f.write(faces.astype(np.int32).tobytes())
    print(f"  Wrote {path}: {nv} verts, {nf} faces")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--ratio', type=float, default=0.001)
    parser.add_argument('--disable_stage3', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    import trimesh
    import torch

    input_mesh = trimesh.load(args.input, force='mesh', process=False)
    verts_np = np.array(input_mesh.vertices, dtype=np.float32)
    faces_np = np.array(input_mesh.faces, dtype=np.int32)

    write_mesh_bin(os.path.join(args.output, 'ref_input.bin'),
                   verts_np, faces_np)

    # Run CUDA pipeline
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import pamo as pamo_module

    points = torch.from_numpy(verts_np).float().cuda()
    triangles = torch.from_numpy(faces_np).int().cuda()

    pamo_runner = pamo_module.PaMO(
        input_mesh=input_mesh,
        use_stage1=True,
        use_stage3=not args.disable_stage3,
    )

    result_v, result_f = pamo_runner.run(
        points, triangles, args.ratio,
    )

    if hasattr(result_v, 'cpu'):
        out_verts = result_v.cpu().numpy()
        out_faces = result_f.cpu().numpy()
    else:
        out_verts = np.asarray(result_v, dtype=np.float32)
        out_faces = np.asarray(result_f, dtype=np.int32)

    write_mesh_bin(os.path.join(args.output, 'ref_output.bin'),
                   out_verts, out_faces)

    print(f"Reference export complete: {args.output}")
    print(f"  Input:  {len(verts_np)} verts, {len(faces_np)} faces")
    print(f"  Output: {len(out_verts)} verts, {len(out_faces)} faces")


if __name__ == '__main__':
    main()
