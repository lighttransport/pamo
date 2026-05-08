// Copyright 2024 Light Transport Entertainment Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pamo vs pamo_c interactive web demo. Renders three viewports, runs
// pamo_c in WASM and the original Python pamo via the local server, and
// colour-codes per-vertex distance.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { parse as parseObj, bounds, weldVertices } from '/shared/obj.mjs';
import { vertexDistances } from '/shared/kdtree.mjs';
import { colourise, distanceStats } from '/shared/colormap.mjs';
import { loadPamo } from '/shared/pamo_runner.mjs';

// ── Scene helpers ────────────────────────────────────────────────────

function createViewport(host) {
    const wrap = host.querySelector('.canvas-wrap');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    wrap.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1116);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
    camera.position.set(0, 0, 2);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.9));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(2, 3, 1);
    scene.add(dir);

    let mesh = null;
    let wire = null;

    function setMesh(geometry, opts = {}) {
        if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); mesh = null; }
        if (wire) { scene.remove(wire); wire.geometry.dispose(); wire.material.dispose(); wire = null; }
        if (!geometry) return;
        const useColor = !!opts.vertexColors;
        const wireframe = !!opts.wireframe;
        const mat = new THREE.MeshStandardMaterial({
            color: useColor ? 0xffffff : 0xb6c2d2,
            vertexColors: useColor,
            metalness: 0.05, roughness: 0.85,
            flatShading: false,
            wireframe: false,
            side: THREE.DoubleSide,
        });
        mesh = new THREE.Mesh(geometry, mat);
        scene.add(mesh);
        if (wireframe) {
            const wmat = new THREE.LineBasicMaterial({ color: 0x222a36, transparent: true, opacity: 0.7 });
            wire = new THREE.LineSegments(new THREE.WireframeGeometry(geometry), wmat);
            scene.add(wire);
        }
    }

    function frame(box) {
        if (!box) return;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        controls.target.copy(center);
        camera.position.set(center.x, center.y, center.z + maxDim * 2.2);
        camera.near = maxDim / 1000;
        camera.far  = maxDim * 100;
        camera.updateProjectionMatrix();
        controls.update();
    }

    function resize() {
        const w = wrap.clientWidth, h = wrap.clientHeight;
        if (w === 0 || h === 0) return;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(wrap);
    resize();

    function render() {
        controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }
    render();

    return { camera, controls, setMesh, frame };
}

function buildGeometry(verts, faces, colors = null) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    // three.js needs an UNSIGNED index buffer for drawElements; signed
    // Int32Array gets mapped to gl.INT and triggers GL_INVALID_ENUM.
    const idx = (faces instanceof Uint32Array || faces instanceof Uint16Array)
        ? faces
        : new Uint32Array(faces);
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    if (colors) g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    g.computeVertexNormals();
    g.computeBoundingBox();
    return g;
}

function linkControls(...vps) {
    // Shared OrbitControls: when one viewport's camera moves, sync the others.
    let updating = false;
    function sync(source) {
        if (updating) return;
        updating = true;
        for (const v of vps) {
            if (v === source) continue;
            v.camera.position.copy(source.camera.position);
            v.controls.target.copy(source.controls.target);
            v.camera.near = source.camera.near;
            v.camera.far = source.camera.far;
            v.camera.updateProjectionMatrix();
        }
        updating = false;
    }
    for (const v of vps) {
        v.controls.addEventListener('change', () => sync(v));
    }
}

// ── State + UI ───────────────────────────────────────────────────────

const state = {
    inputMesh: null,    // {verts, faces}
    inputBounds: null,
    pamo: null,         // {verts, faces, ms}
    pamoC: null,        // {verts, faces, ms}
    runner: null,
};

const vps = {
    input:  createViewport(document.querySelector('.viewport[data-name=input]')),
    pamo:   createViewport(document.querySelector('.viewport[data-name=pamo]')),
    pamo_c: createViewport(document.querySelector('.viewport[data-name=pamo_c]')),
};
linkControls(vps.input, vps.pamo, vps.pamo_c);

const ui = {
    select:    document.getElementById('mesh-select'),
    upload:    document.getElementById('mesh-upload'),
    ratio:     document.getElementById('ratio'),
    ratioVal:  document.getElementById('ratio-val'),
    stage1:    document.getElementById('stage1'),
    stage3:    document.getElementById('stage3'),
    preserveBoundary: document.getElementById('preserve-boundary'),
    sdfR:      document.getElementById('sdf-r'),
    cscale:    document.getElementById('cscale'),
    cscaleVal: document.getElementById('cscale-val'),
    view:      document.getElementById('view-mode'),
    run:       document.getElementById('run'),
    status:    document.getElementById('status'),
    metrics:   document.getElementById('metrics'),
    legendMid: document.getElementById('legend-mid'),
    legendMax: document.getElementById('legend-max'),
};

function ratioFromSlider() { return Math.pow(10, +ui.ratio.value); }

function fmtTime(ms) {
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 10_000) return `${(ms / 1000).toFixed(2)} s`;
    return `${(ms / 1000).toFixed(1)} s`;
}

function setStatus(text, cls = '') {
    ui.status.className = `status ${cls}`;
    ui.status.textContent = text;
}

function setInfo(name, text) {
    document.querySelector(`.viewport[data-name=${name}] .info`).textContent = text;
}

ui.ratio.addEventListener('input', () => {
    ui.ratioVal.textContent = ratioFromSlider().toFixed(3);
});
ui.ratio.dispatchEvent(new Event('input'));

// Stage 3 in pamo_safe_project requires a closed manifold input. Without
// Stage 1 (which remeshes via SDF + dual MC, producing a watertight mesh),
// non-watertight meshes assert with "Number of hinges != number of edges".
function updateStage3Hint() {
    setStatus(
        ui.stage3.checked && !ui.stage1.checked
            ? 'hint: Stage 3 needs Stage 1 (or a closed manifold input)'
            : 'idle',
        ui.stage3.checked && !ui.stage1.checked ? 'busy' : '',
    );
}
ui.stage3.addEventListener('change', updateStage3Hint);
ui.stage1.addEventListener('change', updateStage3Hint);
updateStage3Hint();   // apply on load

ui.cscale.addEventListener('input', () => {
    const pct = +ui.cscale.value;
    ui.cscaleVal.textContent = `${pct.toFixed(1)}%`;
    ui.legendMid.textContent = `${(pct/2).toFixed(1)}%`;
    ui.legendMax.textContent = `${pct.toFixed(1)}%`;
    if (state.pamo && state.pamoC) recolour();
});

ui.view.addEventListener('change', () => {
    if (state.pamo && state.pamoC) recolour();
});

ui.run.addEventListener('click', run);

ui.upload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    loadInput(text, file.name);
});

// ── Input mesh handling ──────────────────────────────────────────────

async function fetchSamples() {
    const r = await fetch('/samples');
    const { samples } = await r.json();
    ui.select.innerHTML = '';
    for (const name of samples) {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        ui.select.appendChild(opt);
    }
    ui.select.addEventListener('change', () => loadSample(ui.select.value));
    if (samples.length) loadSample(samples[0]);
}

async function loadSample(name) {
    setStatus(`loading ${name}…`, 'busy');
    const r = await fetch(`/samples/${encodeURIComponent(name)}`);
    const text = await r.text();
    loadInput(text, name);
}

function loadInput(text, name) {
    const raw = parseObj(text);
    // Weld co-located vertices before sending to either pipeline. Many
    // OBJs from modellers/scanners ship with duplicate verts on internal
    // seams, which simplification would otherwise pull apart into
    // visible cracks (e.g. BirdHouse's bottom panel).
    const welded = weldVertices(raw.verts, raw.faces);
    const mesh = { verts: welded.verts, faces: welded.faces, colors: null };
    if (welded.merged > 0) {
        console.log(`[obj] welded ${welded.merged} duplicate verts `
                  + `(${raw.verts.length/3} → ${mesh.verts.length/3})`);
    }
    state.inputMesh = mesh;
    state.inputBounds = bounds(mesh.verts);
    state.pamo = state.pamoC = null;
    setInfo('input', `${mesh.verts.length/3}v ${mesh.faces.length/3}f`
                   + (welded.merged > 0 ? ` (welded ${welded.merged})` : ''));
    setInfo('pamo', '—');
    setInfo('pamo_c', '—');
    const geo = buildGeometry(mesh.verts.slice(), mesh.faces.slice());
    vps.input.setMesh(geo);
    vps.input.frame(geo.boundingBox);
    vps.pamo.setMesh(null);
    vps.pamo_c.setMesh(null);
    ui.metrics.textContent = '';
    setStatus(`loaded ${name}`, 'ok');
}

// ── Run pipelines ────────────────────────────────────────────────────

async function ensureRunner() {
    if (!state.runner) {
        setStatus('loading WASM…', 'busy');
        state.runner = await loadPamo('/wasm/pamo.mjs');
    }
    return state.runner;
}

async function run() {
    if (!state.inputMesh) { setStatus('no input mesh', 'error'); return; }
    ui.run.disabled = true;
    setStatus('running…', 'busy');

    const opts = {
        ratio: ratioFromSlider(),
        useStage1: ui.stage1.checked,
        useStage3: ui.stage3.checked,
        sdfResolution: parseInt(ui.sdfR.value, 10) || 0,  // 0 = auto
        preserveBoundary: ui.preserveBoundary.checked,
    };

    try {
        const [pamoRes, wasmRes] = await Promise.all([
            runPython(opts),
            runWasm(opts),
        ]);
        state.pamo  = pamoRes;
        state.pamoC = wasmRes;
        setInfo('pamo',   `${pamoRes.verts.length/3}v ${pamoRes.faces.length/3}f  ${fmtTime(pamoRes.ms)}`);
        setInfo('pamo_c', `${wasmRes.verts.length/3}v ${wasmRes.faces.length/3}f  ${fmtTime(wasmRes.ms)}`);
        recolour();
        setStatus('ok', 'ok');
    } catch (e) {
        console.error(e);
        setStatus(`error: ${e.message}`, 'error');
    } finally {
        ui.run.disabled = false;
    }
}

async function runPython(opts) {
    const t0 = performance.now();
    const r = await fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            verts: Array.from(state.inputMesh.verts),
            faces: Array.from(state.inputMesh.faces),
            ratio: opts.ratio,
            stage1: opts.useStage1,
            stage3: opts.useStage3,
        }),
    });
    if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(`python: ${err.error || r.statusText}`);
    }
    const j = await r.json();
    return {
        verts: new Float32Array(j.verts),
        faces: new Int32Array(j.faces),
        ms: j.ms ?? (performance.now() - t0),
    };
}

async function runWasm(opts) {
    const runner = await ensureRunner();
    // Surface C-side progress in the status bar. RAF-throttled so the
    // per-iteration callbacks don't pin the layout thread.
    let pendingMsg = null, scheduled = false;
    const flush = () => {
        scheduled = false;
        if (pendingMsg !== null) setStatus(pendingMsg, 'busy');
    };
    return runner.simplify(state.inputMesh.verts, state.inputMesh.faces, {
        ...opts,
        onProgress: (stage, pct, alive, target) => {
            const pctStr = `${(pct * 100).toFixed(0)}%`;
            if (stage === 'stage2' && alive >= 0 && target >= 0) {
                pendingMsg = `pamo_c stage2: ${pctStr} (${alive}f → target ${target}f)`;
            } else {
                pendingMsg = `pamo_c ${stage}: ${pctStr}`;
            }
            if (!scheduled) {
                scheduled = true;
                requestAnimationFrame(flush);
            }
            return 0;
        },
    });
}

// ── Diff colouring ───────────────────────────────────────────────────

function recolour() {
    const mode = ui.view.value;
    const scaleAbs = state.inputBounds.diameter * (+ui.cscale.value) / 100;

    const wireframe = (mode === 'wireframe');
    const useColors = (mode === 'diff-input' || mode === 'diff-other');

    // Plain / wireframe view
    if (!useColors) {
        const pg = buildGeometry(state.pamo.verts.slice(), state.pamo.faces.slice());
        const cg = buildGeometry(state.pamoC.verts.slice(), state.pamoC.faces.slice());
        vps.pamo.setMesh(pg, { wireframe });
        vps.pamo_c.setMesh(cg, { wireframe });
        vps.input.setMesh(buildGeometry(state.inputMesh.verts.slice(),
                                        state.inputMesh.faces.slice()),
                          { wireframe });
        ui.metrics.textContent = '';
        return;
    }

    // Diff view — colour each output by distance to the chosen reference.
    let dPamo, dPamoC;
    if (mode === 'diff-input') {
        dPamo  = vertexDistances(state.pamo.verts,  state.inputMesh.verts);
        dPamoC = vertexDistances(state.pamoC.verts, state.inputMesh.verts);
    } else {  // diff-other
        dPamo  = vertexDistances(state.pamo.verts,  state.pamoC.verts);
        dPamoC = vertexDistances(state.pamoC.verts, state.pamo.verts);
    }
    const cPamo  = colourise(dPamo,  scaleAbs);
    const cPamoC = colourise(dPamoC, scaleAbs);

    vps.pamo.setMesh(buildGeometry(state.pamo.verts.slice(),
                                   state.pamo.faces.slice(), cPamo),
                     { vertexColors: true });
    vps.pamo_c.setMesh(buildGeometry(state.pamoC.verts.slice(),
                                     state.pamoC.faces.slice(), cPamoC),
                       { vertexColors: true });

    // Input stays plain in diff view.
    vps.input.setMesh(buildGeometry(state.inputMesh.verts.slice(),
                                    state.inputMesh.faces.slice()));

    const sP = distanceStats(dPamo);
    const sC = distanceStats(dPamoC);
    const D = state.inputBounds.diameter;
    const pct = (x) => `${(x / D * 100).toFixed(2)}%`;
    ui.metrics.textContent =
        `pamo:  max ${pct(sP.max)} mean ${pct(sP.mean)} p95 ${pct(sP.p95)}   |   ` +
        `pamo_c: max ${pct(sC.max)} mean ${pct(sC.mean)} p95 ${pct(sC.p95)}`;
}

// ── Bootstrap ────────────────────────────────────────────────────────

fetchSamples();
