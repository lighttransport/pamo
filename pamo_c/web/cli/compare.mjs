#!/usr/bin/env node
// Copyright 2024 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Run the original Python pamo (subprocess) and pamo_c (WASM) on the same
// input OBJ, then write per-vertex-coloured OBJs visualising the geometric
// divergence between the three meshes.

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { parse as parseObj, serialise as writeObj, bounds, weldVertices } from '../shared/obj.mjs';
import { vertexDistances } from '../shared/kdtree.mjs';
import { colourise, distanceStats } from '../shared/colormap.mjs';
import { loadPamo } from '../shared/pamo_runner.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, '..');
const PAMO_C_ROOT = resolve(WEB_ROOT, '..');
const REPO_ROOT = resolve(PAMO_C_ROOT, '..');
const WASM_LOADER = pathToFileURL(join(WEB_ROOT, 'wasm', 'pamo.mjs')).href;

function parseArgs(argv) {
    const args = {
        input: null,
        outdir: './out',
        ratio: 0.1,
        disableStage1: false,
        disableStage3: false,
        sdfResolution: 0,         // 0 = auto (mirror Python pamo)
        pythonBin: null,
        pythonExample: join(REPO_ROOT, 'example.py'),
        colourScale: null,    // absolute distance for red saturation
        colourScalePct: 5.0,  // % of input diameter (used if colourScale unset)
        skipPython: false,
        skipWasm: false,
    };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        const next = () => argv[++i];
        switch (a) {
            case '-i': case '--input':         args.input = next(); break;
            case '-o': case '--outdir':        args.outdir = next(); break;
            case '-r': case '--ratio':         args.ratio = +next(); break;
            case '--disable_stage1':           args.disableStage1 = true; break;
            case '--disable_stage3':           args.disableStage3 = true; break;
            case '--sdf-r':                    args.sdfResolution = parseInt(next(), 10) || 0; break;
            case '--python-bin':               args.pythonBin = next(); break;
            case '--python-example':           args.pythonExample = next(); break;
            case '--colour-scale':             args.colourScale = +next(); break;
            case '--colour-scale-pct':         args.colourScalePct = +next(); break;
            case '--skip-python':              args.skipPython = true; break;
            case '--skip-wasm':                args.skipWasm = true; break;
            case '-h': case '--help':          usage(); process.exit(0);
            default:
                console.error(`Unknown arg: ${a}`); usage(); process.exit(2);
        }
    }
    if (!args.input) { usage(); process.exit(2); }
    return args;
}

function usage() {
    console.log(`Usage: compare.mjs -i <input.obj> [-o outdir] [-r ratio] \\
                    [--disable_stage1] [--disable_stage3] [--sdf-r <64|128|256|0>] \\
                    [--python-bin <path>] [--python-example <path>] \\
                    [--colour-scale <abs>] [--colour-scale-pct <%>] \\
                    [--skip-python] [--skip-wasm]

Outputs (in <outdir>):
  pamo.obj                — original Python pamo (uncoloured)
  pamo_c.obj              — pamo_c WASM (uncoloured)
  pamo_vs_input.obj       — pamo coloured by per-vertex distance to input
  pamo_c_vs_input.obj     — pamo_c coloured by per-vertex distance to input
  pamo_vs_pamo_c.obj      — pamo_c coloured by per-vertex distance to pamo
`);
}

function findPythonBin(override) {
    if (override) return override;
    const candidates = [
        join(REPO_ROOT, '.venv-cuda12', 'bin', 'python'),
        join(REPO_ROOT, '.venv', 'bin', 'python'),
    ];
    for (const c of candidates) if (existsSync(c)) return c;
    throw new Error(
        'Could not find a Python interpreter. Tried .venv-cuda12 and .venv. ' +
        'Pass --python-bin <path> explicitly.'
    );
}

function runPython(args, outPath) {
    const pythonBin = findPythonBin(args.pythonBin);
    const argv = [
        args.pythonExample,
        '-i', args.input,
        '-o', outPath,
        '-r', String(args.ratio),
    ];
    if (args.disableStage1) argv.push('--disable_stage1');
    if (args.disableStage3) argv.push('--disable_stage3');
    console.log(`[python] ${pythonBin} ${argv.join(' ')}`);
    const t0 = Date.now();
    const res = spawnSync(pythonBin, argv, {
        cwd: dirname(args.pythonExample),
        stdio: 'inherit',
    });
    const ms = Date.now() - t0;
    if (res.status !== 0) {
        throw new Error(`Python pamo exited ${res.status}`);
    }
    return ms;
}

async function runWasm(input, args) {
    console.log('[wasm]   loading pamo.mjs');
    if (!existsSync(join(WEB_ROOT, 'wasm', 'pamo.mjs'))) {
        throw new Error(
            'pamo.mjs not found. Build the WASM module first:\n' +
            '  cd pamo_c/web/wasm && ./build.sh'
        );
    }
    const runner = await loadPamo(WASM_LOADER);
    console.log(`[wasm]   simplify(ratio=${args.ratio}, stage1=${!args.disableStage1}, stage3=${!args.disableStage3})`);
    return runner.simplify(input.verts, input.faces, {
        ratio: args.ratio,
        useStage1: !args.disableStage1,
        useStage3: !args.disableStage3,
        sdfResolution: args.sdfResolution,
    });
}

function writeColouredObj(path, mesh, refMesh, scale, label) {
    const dists = vertexDistances(mesh.verts, refMesh.verts);
    const colors = colourise(dists, scale);
    writeFileSync(path, writeObj({
        verts: mesh.verts,
        faces: mesh.faces,
        colors,
    }));
    const s = distanceStats(dists);
    console.log(`  ${label.padEnd(22)} max ${s.max.toExponential(3)} `
              + `mean ${s.mean.toExponential(3)} `
              + `rms ${s.rms.toExponential(3)} `
              + `p95 ${s.p95.toExponential(3)}`);
    return s;
}

async function main() {
    const args = parseArgs(process.argv);
    args.input = resolve(args.input);  // python subprocess runs with cwd=repo_root
    if (!existsSync(args.input)) {
        console.error(`Input not found: ${args.input}`);
        process.exit(1);
    }
    mkdirSync(args.outdir, { recursive: true });
    const outdir = resolve(args.outdir);

    const pamoPath   = join(outdir, 'pamo.obj');
    const pamoCPath  = join(outdir, 'pamo_c.obj');

    // ── Python pamo ────────────────────────────────────────────────
    let pyMs = 0;
    if (!args.skipPython) {
        pyMs = runPython(args, pamoPath);
    } else if (!existsSync(pamoPath)) {
        console.error(`--skip-python set but ${pamoPath} missing.`);
        process.exit(1);
    }

    // ── Load input ─────────────────────────────────────────────────
    console.log(`[load]   ${args.input}`);
    const raw = parseObj(readFileSync(args.input, 'utf8'));
    const welded = weldVertices(raw.verts, raw.faces);
    const input = { verts: welded.verts, faces: welded.faces };
    if (welded.merged > 0) {
        console.log(`[weld]   merged ${welded.merged} duplicate verts `
                  + `(${raw.verts.length/3} → ${input.verts.length/3})`);
    }
    const inputBounds = bounds(input.verts);
    console.log(`[input]  ${input.verts.length / 3} verts, `
              + `${input.faces.length / 3} faces, `
              + `diameter ${inputBounds.diameter.toExponential(3)}`);

    // ── pamo_c WASM ────────────────────────────────────────────────
    let wasmResult, wasmMs = 0;
    if (!args.skipWasm) {
        wasmResult = await runWasm(input, args);
        writeFileSync(pamoCPath, writeObj({
            verts: wasmResult.verts,
            faces: wasmResult.faces,
        }));
        wasmMs = wasmResult.ms;
    } else if (!existsSync(pamoCPath)) {
        console.error(`--skip-wasm set but ${pamoCPath} missing.`);
        process.exit(1);
    } else {
        wasmResult = parseObj(readFileSync(pamoCPath, 'utf8'));
    }

    // Re-load Python output (may have been written by subprocess just now).
    const pyMesh = parseObj(readFileSync(pamoPath, 'utf8'));
    console.log(`[pamo]   ${pyMesh.verts.length / 3} verts, `
              + `${pyMesh.faces.length / 3} faces  (${pyMs} ms)`);
    console.log(`[pamo_c] ${wasmResult.verts.length / 3} verts, `
              + `${wasmResult.faces.length / 3} faces  (${wasmMs.toFixed(0)} ms)`);

    // ── Colour scale ────────────────────────────────────────────────
    const scale = args.colourScale ?? (inputBounds.diameter * args.colourScalePct / 100);
    console.log(`\n[colour] saturate at ${scale.toExponential(3)} `
              + `(${(scale / inputBounds.diameter * 100).toFixed(2)}% of diameter)`);

    // ── 3 diff combos ──────────────────────────────────────────────
    console.log('\n=== Per-vertex distance summaries ===');
    writeColouredObj(
        join(outdir, 'pamo_vs_input.obj'),
        pyMesh, input, scale, 'pamo vs input',
    );
    writeColouredObj(
        join(outdir, 'pamo_c_vs_input.obj'),
        wasmResult, input, scale, 'pamo_c vs input',
    );
    writeColouredObj(
        join(outdir, 'pamo_vs_pamo_c.obj'),
        wasmResult, pyMesh, scale, 'pamo_c vs pamo',
    );

    console.log(`\nWrote 5 OBJs to ${outdir}`);
}

main().catch(e => { console.error(e); process.exit(1); });
