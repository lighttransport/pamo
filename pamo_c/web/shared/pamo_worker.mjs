// Copyright 2024 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Web Worker entry point: runs pamo_c (WASM) off the main thread so the
// page can paint live progress updates and stay responsive during long
// simplifications. Communication protocol (over postMessage):
//
//   main → worker:
//     { type: 'run', id, verts, faces, opts }
//
//   worker → main:
//     { type: 'progress', id, stage, pct, alive, target }
//     { type: 'done',     id, verts, faces, ms }
//     { type: 'error',    id, message }
//
// `verts` is a Float32Array, `faces` an Int32Array. ArrayBuffers are
// transferred (zero-copy) where possible.

import { loadPamo } from './pamo_runner.mjs';

let runner = null;

self.addEventListener('message', async (e) => {
    const msg = e.data;
    if (!msg || msg.type !== 'run') return;
    const { id, verts, faces, opts } = msg;

    try {
        if (!runner) {
            runner = await loadPamo('/wasm/pamo.mjs');
        }
        // Forward C-side progress to the main thread.
        const post = (stage, pct, alive, target) => {
            self.postMessage({
                type: 'progress', id, stage, pct, alive, target,
            });
            return 0;
        };
        const result = runner.simplify(verts, faces, {
            ...opts,
            onProgress: post,
        });
        // Transfer the result buffers (zero-copy).
        self.postMessage({
            type: 'done',
            id,
            verts: result.verts,
            faces: result.faces,
            ms: result.ms,
        }, [result.verts.buffer, result.faces.buffer]);
    } catch (err) {
        self.postMessage({
            type: 'error',
            id,
            message: err && err.message ? err.message : String(err),
        });
    }
});
