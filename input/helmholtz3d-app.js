/*
 * helmholtz3d-app.js — 3D far-field scattering visualisation
 *
 * Cube-map sphere, CIF-parsed crystal, intensity-dependent transparency.
 * Camera defaults to inside the sphere ("night-sky" view).
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

'use strict';

/* ══════════════════════════════════════════
   CIF Parser
   ══════════════════════════════════════════ */

function parseCIF(text) {
    const lines = text.split(/\r?\n/);
    const kv = {};
    const loops = [];

    let i = 0;
    while (i < lines.length) {
        const line = lines[i].trim();
        if (!line || line.startsWith('#')) { i++; continue; }
        if (line.startsWith('data_')) { i++; continue; }

        if (line === 'loop_') {
            i++;
            const columns = [];
            while (i < lines.length && lines[i].trim().startsWith('_')) {
                columns.push(lines[i].trim());
                i++;
            }
            const rows = [];
            while (i < lines.length) {
                const row = lines[i].trim();
                if (!row || row.startsWith('_') || row === 'loop_' || row.startsWith('data_')) break;
                if (row.startsWith('#')) { i++; continue; }
                rows.push(row);
                i++;
            }
            loops.push({ columns, rows });
            continue;
        }

        if (line.startsWith('_')) {
            const spaceIdx = line.indexOf(' ');
            if (spaceIdx > 0) {
                const key = line.substring(0, spaceIdx).trim();
                const val = line.substring(spaceIdx).trim().replace(/^'|'$/g, '');
                kv[key] = val;
            } else {
                const key = line;
                i++;
                if (i < lines.length) {
                    let val = lines[i].trim();
                    if (val === ';') {
                        i++;
                        const parts = [];
                        while (i < lines.length && lines[i].trim() !== ';') {
                            parts.push(lines[i]);
                            i++;
                        }
                        val = parts.join('\n');
                    }
                    kv[key] = val.replace(/^'|'$/g, '');
                }
            }
            i++;
            continue;
        }
        i++;
    }

    const a     = parseFloat(kv['_cell_length_a']) || 1;
    const b     = parseFloat(kv['_cell_length_b']) || a;
    const c     = parseFloat(kv['_cell_length_c']) || a;
    const alpha = (parseFloat(kv['_cell_angle_alpha']) || 90) * Math.PI / 180;
    const beta  = (parseFloat(kv['_cell_angle_beta'])  || 90) * Math.PI / 180;
    const gamma = (parseFloat(kv['_cell_angle_gamma']) || 90) * Math.PI / 180;

    const avec = [a, 0, 0];
    const bvec = [b * Math.cos(gamma), b * Math.sin(gamma), 0];
    const cx = c * Math.cos(beta);
    const cy = c * (Math.cos(alpha) - Math.cos(beta) * Math.cos(gamma)) / Math.sin(gamma);
    const cz = Math.sqrt(Math.max(0, c * c - cx * cx - cy * cy));
    const cvec = [cx, cy, cz];

    /* Symmetry operations */
    let symOps = [];
    for (const loop of loops) {
        const idx = loop.columns.indexOf('_space_group_symop_operation_xyz');
        if (idx < 0) continue;
        for (const row of loop.rows) {
            if (loop.columns.length === 1) {
                symOps.push(row.replace(/'/g, '').trim());
            } else {
                const parts = splitCIFRow(row, loop.columns.length);
                if (parts[idx]) symOps.push(parts[idx].replace(/'/g, ''));
            }
        }
    }
    if (symOps.length === 0) symOps = ['x,y,z'];

    /* Atom sites */
    const asymAtoms = [];
    for (const loop of loops) {
        const iLabel = loop.columns.indexOf('_atom_site_label');
        const iFx = loop.columns.indexOf('_atom_site_fract_x');
        const iFy = loop.columns.indexOf('_atom_site_fract_y');
        const iFz = loop.columns.indexOf('_atom_site_fract_z');
        if (iLabel < 0 || iFx < 0) continue;
        for (const row of loop.rows) {
            const parts = splitCIFRow(row, loop.columns.length);
            const label = parts[iLabel] || 'X';
            const sym = label.match(/^[A-Z][a-z]?/)?.[0] || 'X';
            asymAtoms.push({
                sym,
                fx: parseFloat(parts[iFx]) || 0,
                fy: parseFloat(parts[iFy]) || 0,
                fz: parseFloat(parts[iFz]) || 0,
            });
        }
    }

    /* Apply symmetry → full basis */
    const parsedOps = symOps.map(parseSymOp);
    const basis = [];
    const symbols = [];

    for (const atom of asymAtoms) {
        for (const op of parsedOps) {
            const pos = applySymOp(op, atom.fx, atom.fy, atom.fz);
            const mx = mod1(pos[0]);
            const my = mod1(pos[1]);
            const mz = mod1(pos[2]);
            const isDup = basis.some(b =>
                closeModulo(b[0], mx) && closeModulo(b[1], my) && closeModulo(b[2], mz)
            );
            if (!isDup) {
                basis.push([mx, my, mz]);
                symbols.push(atom.sym);
            }
        }
    }

    const name = kv['_chemical_name_mineral'] || kv['_chemical_formula_sum'] || 'Unknown';
    return { name, a, b, c,
        alpha: alpha * 180 / Math.PI, beta: beta * 180 / Math.PI, gamma: gamma * 180 / Math.PI,
        avec, bvec, cvec, basis, symbols };
}

function splitCIFRow(row, nCols) {
    const parts = [];
    let s = row.trim();
    while (s.length > 0 && parts.length < nCols) {
        if (s[0] === "'") {
            const end = s.indexOf("'", 1);
            if (end > 0) { parts.push(s.substring(1, end)); s = s.substring(end + 1).trim(); }
            else { parts.push(s.substring(1)); s = ''; }
        } else {
            const sp = s.indexOf(' ');
            if (sp > 0) { parts.push(s.substring(0, sp)); s = s.substring(sp).trim(); }
            else { parts.push(s); s = ''; }
        }
    }
    return parts;
}

function parseSymOp(opStr) {
    const parts = opStr.replace(/'/g, '').split(',').map(s => s.trim());
    const matrix = [];
    for (const part of parts) {
        let cx = 0, cy = 0, cz = 0, offset = 0;
        let s = part.replace(/\s/g, '');
        let i = 0, sign = 1;
        while (i < s.length) {
            if (s[i] === '+') { sign = 1; i++; continue; }
            if (s[i] === '-') { sign = -1; i++; continue; }
            if (s[i] === 'x') { cx = sign; sign = 1; i++; continue; }
            if (s[i] === 'y') { cy = sign; sign = 1; i++; continue; }
            if (s[i] === 'z') { cz = sign; sign = 1; i++; continue; }
            let numStr = '';
            while (i < s.length && /[0-9./]/.test(s[i])) { numStr += s[i]; i++; }
            if (numStr) {
                let val;
                if (numStr.includes('/')) {
                    const [num, den] = numStr.split('/');
                    val = parseFloat(num) / parseFloat(den);
                } else { val = parseFloat(numStr); }
                if (i < s.length && /[xyz]/.test(s[i])) {
                    const v = s[i]; i++;
                    if (v === 'x') cx = sign * val;
                    else if (v === 'y') cy = sign * val;
                    else cz = sign * val;
                    sign = 1;
                } else { offset += sign * val; sign = 1; }
            }
        }
        matrix.push([cx, cy, cz, offset]);
    }
    return matrix;
}

function applySymOp(matrix, x, y, z) {
    return [
        matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z + matrix[0][3],
        matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z + matrix[1][3],
        matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z + matrix[2][3],
    ];
}

function mod1(v) { return ((v % 1) + 1) % 1; }

function closeModulo(a, b, tol = 0.01) {
    const d = Math.abs(a - b);
    return d < tol || d > 1 - tol;
}

/* ══════════════════════════════════════════
   Element colours (CPK)
   ══════════════════════════════════════════ */
const ELEM_COLOR = {
    H:0xffffff, He:0xd9ffff, Li:0xcc80ff, Be:0xc2ff00, B:0xffb5b5,
    C:0x909090, N:0x3050f8, O:0xff0d0d, F:0x90e050, Na:0xab5cf2,
    Mg:0x8aff00, Al:0xbfa6a6, Si:0xf0c8a0, P:0xff8000, S:0xffff30,
    Cl:0x1ff01f, K:0x8f40d4, Ca:0x3dff00, Ti:0xbfc2c7, Cr:0x8a99c7,
    Mn:0x9c7ac7, Fe:0xe06633, Co:0xf090a0, Ni:0x50d050, Cu:0xc88033,
    Zn:0x7d80b0, Ga:0xc28f8f, Ge:0x668f8f, Mo:0x54b5b5, Ag:0xc0c0c0,
    W:0x2194d6, Au:0xffd123, Pt:0xd0d0e0,
};
function elemColor(sym) { return ELEM_COLOR[sym] ?? 0xcccccc; }

/* ══════════════════════════════════════════
   Wavelength mapping (logarithmic, like 2D module)
   ══════════════════════════════════════════ */
const LAMBDA_MIN = 0.1;
const LAMBDA_MAX = 5.0;
const LOG_LAMBDA_RATIO = Math.log(LAMBDA_MAX / LAMBDA_MIN);

function sliderToLambda(t) { return LAMBDA_MIN * Math.exp(t * LOG_LAMBDA_RATIO); }
function lambdaToSlider(lam) { return Math.log(lam / LAMBDA_MIN) / LOG_LAMBDA_RATIO; }

/* ══════════════════════════════════════════
   State
   ══════════════════════════════════════════ */
let wasm = null;
let fieldPtr = null;
let faceRes = 1024;

let crystal = null;
let nCells = 5;
let wavelengthA = 0.35;         // 0.35 Å (synchrotron / SAXS-compatible)
let beamRotDeg = 0;
let beamTiltDeg = 0;
let beamSigmaLog = 3.0;        // log10(sigma/A)  => 1000 A = 100 nm
let brightness = 1.5;
let sphereAlpha = 0.9;
let sphereRadiusLog = 6.0;     // log10(R) => 1e6
let sphereRadius = 1000000;
let showAtoms = true;
let showBeam = true;
let useLogScale = true;
let logEpsilon = 4;            // log10(eps) => 1e4
let transparencyPower = 1.5;
let beamMode = 'plane';        // 'plane' or 'gaussian'
let showBraggRings = true;     // overlay Bragg circles
let linearCmaxLog = 6;         // log10(cmax) for linear colormap, default 1e6
let linearCmax = 1e6;

/* Bunge Euler angles (degrees) for crystal rotation */
let bungePhi1 = 0;
let bungePHI  = 0;
let bungePhi2 = 0;

let fineTimer = null;

/* ══════════════════════════════════════════
   DOM refs
   ══════════════════════════════════════════ */
const container       = document.getElementById('threeContainer');
const statusBar       = document.getElementById('statusBar');
const slWavelength    = document.getElementById('slWavelength');
const valWavelength   = document.getElementById('valWavelength');
const slRotation      = document.getElementById('slRotation');
const valRotation     = document.getElementById('valRotation');
const slTilt          = document.getElementById('slTilt');
const valTilt         = document.getElementById('valTilt');
const slBeamSigma     = document.getElementById('slBeamSigma');
const valBeamSigma    = document.getElementById('valBeamSigma');
const cifFileInput    = document.getElementById('cifFileInput');
const cifNameEl       = document.getElementById('cifName');
const slNCells        = document.getElementById('slNCells');
const valNCells       = document.getElementById('valNCells');
const chkAtoms        = document.getElementById('chkAtoms');
const chkBeam         = document.getElementById('chkBeam');
const chkLogScale     = document.getElementById('chkLogScale');
const slLogEps        = document.getElementById('slLogEps');
const valLogEps       = document.getElementById('valLogEps');
const slTransparency  = document.getElementById('slTransparency');
const valTransparency = document.getElementById('valTransparency');
const slSphereRadius  = document.getElementById('slSphereRadius');
const valSphereRadius = document.getElementById('valSphereRadius');
const slSphereAlpha   = document.getElementById('slSphereAlpha');
const valSphereAlpha  = document.getElementById('valSphereAlpha');
const slBrightness    = document.getElementById('slBrightness');
const valBrightness   = document.getElementById('valBrightness');
const selResolution   = document.getElementById('selResolution');
const slBungePhi1     = document.getElementById('slBungePhi1');
const valBungePhi1    = document.getElementById('valBungePhi1');
const slBungePHI      = document.getElementById('slBungePHI');
const valBungePHI     = document.getElementById('valBungePHI');
const slBungePhi2     = document.getElementById('slBungePhi2');
const valBungePhi2    = document.getElementById('valBungePhi2');
const selBeamMode     = document.getElementById('selBeamMode');
const chkBraggRings   = document.getElementById('chkBraggRings');
const slLinearCmax    = document.getElementById('slLinearCmax');
const valLinearCmax   = document.getElementById('valLinearCmax');
const cmaxRow         = document.getElementById('cmaxRow');

/* ══════════════════════════════════════════
   Three.js setup
   ══════════════════════════════════════════ */
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);

const camera = new THREE.PerspectiveCamera(75, 1, 0.01, sphereRadius * 10);
camera.position.set(sphereRadius * 1.0, sphereRadius * 0.35, sphereRadius * 0.7);   // outside, to the side

const renderer = new THREE.WebGLRenderer({ antialias: true });
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.minDistance = 0.01;
controls.maxDistance = 5e8;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(10, 15, 10);
scene.add(dirLight);

/* ── Cube-map sphere ── */
const cubeCanvases = [];
for (let i = 0; i < 6; i++) {
    const c = document.createElement('canvas');
    c.width = faceRes; c.height = faceRes;
    cubeCanvases.push(c);
}

let cubeTex = new THREE.CubeTexture(cubeCanvases);
cubeTex.needsUpdate = true;
let lastCanvasSize = faceRes;  // track for dispose/recreate

const sphereVert = `
varying vec3 vWorldNormal;
void main() {
    vWorldNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;
const sphereFrag = `
uniform samplerCube uCubeMap;
uniform float uOpacity;
varying vec3 vWorldNormal;
void main() {
    vec4 c = textureCube(uCubeMap, vWorldNormal);
    gl_FragColor = vec4(c.rgb, c.a * uOpacity);
}
`;

const sphereGeo = new THREE.SphereGeometry(sphereRadius, 128, 64);
const sphereMat = new THREE.ShaderMaterial({
    uniforms: {
        uCubeMap: { value: cubeTex },
        uOpacity: { value: sphereAlpha },
    },
    vertexShader: sphereVert,
    fragmentShader: sphereFrag,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
});
const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
scene.add(sphereMesh);

const atomGroup = new THREE.Group();
scene.add(atomGroup);

const beamGroup = new THREE.Group();
scene.add(beamGroup);

const braggGroup = new THREE.Group();
scene.add(braggGroup);

const axesHelper = new THREE.AxesHelper(3);
scene.add(axesHelper);

/* ══════════════════════════════════════════
   Resize
   ══════════════════════════════════════════ */
function onResize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
new ResizeObserver(onResize).observe(container);
onResize();

/* ══════════════════════════════════════════
   Rebuild sphere
   ══════════════════════════════════════════ */
function rebuildSphere() {
    sphereMesh.geometry.dispose();
    sphereMesh.geometry = new THREE.SphereGeometry(sphereRadius, 128, 64);
    /* Keep far clip large enough to see the sphere */
    camera.far = Math.max(100000, sphereRadius * 10);
    camera.updateProjectionMatrix();
}

/* ══════════════════════════════════════════
   Build atoms — 1:1 A coordinates, InstancedMesh
   ══════════════════════════════════════════ */
function rebuildAtoms() {
    while (atomGroup.children.length > 0) {
        const m = atomGroup.children[0];
        atomGroup.remove(m);
        m.geometry?.dispose();
        m.material?.dispose();
    }
    if (!showAtoms || !crystal) return;

    const cr = crystal;
    /* Apply Bunge rotation to lattice vectors for atom display */
    const R = bungeToMatrix(bungePhi1, bungePHI, bungePhi2);
    const a = rotateVec(R, cr.avec);
    const b = rotateVec(R, cr.bvec);
    const c = rotateVec(R, cr.cvec);
    /* Cap atom display at 8³ grid to avoid GPU overload */
    const dispN = Math.min(nCells, 8);
    const half = Math.floor(dispN / 2);
    const geo = new THREE.SphereGeometry(0.5, 8, 6);

    const symSet = [...new Set(cr.symbols)];
    for (const sym of symSet) {
        const indices = cr.symbols.reduce((acc, s, i) => s === sym ? [...acc, i] : acc, []);
        const count = (2 * half + 1) ** 3 * indices.length;
        if (count === 0) continue;

        const mat = new THREE.MeshStandardMaterial({
            color: elemColor(sym), metalness: 0.3, roughness: 0.6,
        });
        const mesh = new THREE.InstancedMesh(geo, mat, count);
        const mtx = new THREE.Matrix4();
        let idx = 0;

        for (let n1 = -half; n1 <= half; n1++) {
            for (let n2 = -half; n2 <= half; n2++) {
                for (let n3 = -half; n3 <= half; n3++) {
                    for (const j of indices) {
                        const fx = cr.basis[j][0] + n1;
                        const fy = cr.basis[j][1] + n2;
                        const fz = cr.basis[j][2] + n3;
                        const x = fx * a[0] + fy * b[0] + fz * c[0];
                        const y = fx * a[1] + fy * b[1] + fz * c[1];
                        const z = fx * a[2] + fy * b[2] + fz * c[2];
                        mtx.makeTranslation(x, y, z);
                        mesh.setMatrixAt(idx++, mtx);
                    }
                }
            }
        }
        mesh.instanceMatrix.needsUpdate = true;
        atomGroup.add(mesh);
    }
}

/* ══════════════════════════════════════════
   Beam cylinder with Gaussian transparency / plane wave thin beam
   ══════════════════════════════════════════ */
function rebuildBeam() {
    while (beamGroup.children.length > 0) {
        const m = beamGroup.children[0];
        beamGroup.remove(m);
        m.geometry?.dispose();
        m.material?.dispose();
    }
    if (!showBeam) return;

    const dir = getBeamDirection();
    const beamDir = new THREE.Vector3(dir.x, dir.y, dir.z);
    const halfLen = sphereRadius * 1.5;

    if (beamMode === 'plane') {
        /* Plane wave: thin blue beam, 10 nm = 100 Å radius */
        const beamR = 100.0;  /* 10 nm in Å */
        const coreGeo = new THREE.CylinderGeometry(beamR, beamR, halfLen * 2, 16);
        const coreMat = new THREE.MeshBasicMaterial({
            color: 0x4488ff, transparent: true, opacity: 0.5,
            side: THREE.DoubleSide, depthWrite: false,
        });
        const core = new THREE.Mesh(coreGeo, coreMat);
        core.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), beamDir);
        beamGroup.add(core);

        /* Arrow tip */
        const arrowR = beamR * 3;
        const arrowGeo = new THREE.ConeGeometry(arrowR, arrowR * 3, 12);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0x4488ff, transparent: true, opacity: 0.7 });
        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
        arrow.position.copy(beamDir.clone().multiplyScalar(halfLen * 0.4));
        arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), beamDir);
        beamGroup.add(arrow);
    } else {
        /* Gaussian beam */
        const sigma = Math.pow(10, beamSigmaLog);
        const visSigma = Math.min(sigma, sphereRadius * 0.3);
        const nRings = 8;

        for (let i = 0; i < nRings; i++) {
            const t = (i + 0.5) / nRings;
            const r = visSigma * (0.1 + t * 2.0);
            const alpha = Math.exp(-0.5 * (0.1 + t * 2.0) ** 2) * 0.4;

            const geo = new THREE.CylinderGeometry(r, r, halfLen * 2, 16, 1, true);
            const mat = new THREE.MeshBasicMaterial({
                color: 0x6c63ff, transparent: true, opacity: alpha,
                side: THREE.DoubleSide, depthWrite: false,
            });
            const cyl = new THREE.Mesh(geo, mat);
            cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), beamDir);
            beamGroup.add(cyl);
        }

        const coreR = Math.max(0.1, visSigma * 0.02);
        const coreGeo = new THREE.CylinderGeometry(coreR, coreR, halfLen * 2, 8);
        const coreMat = new THREE.MeshBasicMaterial({ color: 0x6c63ff, transparent: true, opacity: 0.7 });
        const core = new THREE.Mesh(coreGeo, coreMat);
        core.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), beamDir);
        beamGroup.add(core);

        const arrowR = Math.max(1, visSigma * 0.15);
        const arrowGeo = new THREE.ConeGeometry(arrowR, arrowR * 3, 12);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0x6c63ff, transparent: true, opacity: 0.8 });
        const arrow = new THREE.Mesh(arrowGeo, arrowMat);
        arrow.position.copy(beamDir.clone().multiplyScalar(halfLen * 0.4));
        arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), beamDir);
        beamGroup.add(arrow);
    }
}

/* ══════════════════════════════════════════
   Bragg / Debye-Scherrer rings
   ══════════════════════════════════════════ */
function computeReciprocalVectors() {
    if (!crystal) return null;
    const R = bungeToMatrix(bungePhi1, bungePHI, bungePhi2);
    const a = rotateVec(R, crystal.avec);
    const b = rotateVec(R, crystal.bvec);
    const c = rotateVec(R, crystal.cvec);

    /* V = a . (b x c) */
    const bxc = [
        b[1]*c[2] - b[2]*c[1],
        b[2]*c[0] - b[0]*c[2],
        b[0]*c[1] - b[1]*c[0],
    ];
    const V = a[0]*bxc[0] + a[1]*bxc[1] + a[2]*bxc[2];
    if (Math.abs(V) < 1e-30) return null;
    const inv = 2 * Math.PI / V;

    /* a* = 2π (b×c) / V, etc. */
    const cxa = [
        c[1]*a[2] - c[2]*a[1],
        c[2]*a[0] - c[0]*a[2],
        c[0]*a[1] - c[1]*a[0],
    ];
    const axb = [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ];
    return {
        astar: [bxc[0]*inv, bxc[1]*inv, bxc[2]*inv],
        bstar: [cxa[0]*inv, cxa[1]*inv, cxa[2]*inv],
        cstar: [axb[0]*inv, axb[1]*inv, axb[2]*inv],
    };
}

function rebuildBraggRings() {
    /* Clear old rings */
    while (braggGroup.children.length > 0) {
        const m = braggGroup.children[0];
        braggGroup.remove(m);
        m.geometry?.dispose();
        m.material?.dispose();
    }
    if (!showBraggRings || !crystal) return;

    const recip = computeReciprocalVectors();
    if (!recip) return;

    const k = 2 * Math.PI / wavelengthA;   /* wavenumber in Å⁻¹ */
    const twoK = 2 * k;
    const HMAX = 5;

    /* Helper: compute atomic form factor f(q) for element sym */
    function atomFF(sym, q) {
        const atom = (typeof ATOM_BY_SYM !== 'undefined') && ATOM_BY_SYM[sym];
        if (!atom) return 1;   /* fallback */
        const s2 = (q / (4 * Math.PI)) ** 2;   /* (sin θ / λ)² */
        let f = atom.c;
        for (let i = 0; i < 4; i++) f += atom.a[i] * Math.exp(-atom.b[i] * s2);
        return f;
    }

    /* Compute structure factor |S(hkl)|² for each unique (h,k,l) group */
    /* Group reflections by rounded |G| to merge equivalent d-spacings */
    const ringMap = new Map();  /* key → { twoTheta, sfSq } */

    for (let h = -HMAX; h <= HMAX; h++) {
        for (let kk = -HMAX; kk <= HMAX; kk++) {
            for (let l = -HMAX; l <= HMAX; l++) {
                if (h === 0 && kk === 0 && l === 0) continue;
                const gx = h*recip.astar[0] + kk*recip.bstar[0] + l*recip.cstar[0];
                const gy = h*recip.astar[1] + kk*recip.bstar[1] + l*recip.cstar[1];
                const gz = h*recip.astar[2] + kk*recip.bstar[2] + l*recip.cstar[2];
                const gMag = Math.sqrt(gx*gx + gy*gy + gz*gz);
                if (gMag > twoK) continue;  /* not accessible */

                /* Structure factor: S = Σ_j f_j(q) * exp(2πi (h x_j + k y_j + l z_j)) */
                let reS = 0, imS = 0;
                for (let j = 0; j < crystal.basis.length; j++) {
                    const [fx, fy, fz] = crystal.basis[j];
                    const phase = 2 * Math.PI * (h * fx + kk * fy + l * fz);
                    const fj = atomFF(crystal.symbols[j], gMag);
                    reS += fj * Math.cos(phase);
                    imS += fj * Math.sin(phase);
                }
                const sfSq = reS * reS + imS * imS;
                if (sfSq < 1e-6) continue;  /* systematically absent */

                const key = (gMag * 1000 | 0);
                const existing = ringMap.get(key);
                if (!existing || sfSq > existing.sfSq) {
                    const sinTheta = gMag / twoK;
                    const twoTheta = 2 * Math.asin(sinTheta);
                    ringMap.set(key, { twoTheta, sfSq });
                }
            }
        }
    }

    /* Keep only top 6 by |S|² */
    const sorted = [...ringMap.values()].sort((a, b) => b.sfSq - a.sfSq);
    const top = sorted.slice(0, 10);

    if (top.length === 0) return;

    /* Beam direction as THREE.Vector3 */
    const dir = getBeamDirection();
    const beam = new THREE.Vector3(dir.x, dir.y, dir.z);

    /* Build an orthonormal basis perpendicular to beam */
    const up = Math.abs(beam.y) < 0.99
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(beam, up).normalize();
    const v = new THREE.Vector3().crossVectors(beam, u).normalize();

    const nSegs = 128;
    const maxSF = top[0].sfSq;

    /* Draw one circle per ring, brighter = stronger */
    for (const { twoTheta, sfSq } of top) {
        const cosA = Math.cos(twoTheta);
        const sinA = Math.sin(twoTheta);
        const alpha = 0.1 + 0.2 * (sfSq / maxSF);

        const ringMat = new THREE.LineBasicMaterial({
            color: 0xff3333, transparent: true, opacity: alpha, depthTest: false
        });

        const pts = [];
        for (let i = 0; i <= nSegs; i++) {
            const phi = (i / nSegs) * 2 * Math.PI;
            const cp = Math.cos(phi), sp = Math.sin(phi);
            pts.push(new THREE.Vector3(
                sphereRadius * (cosA * beam.x + sinA * (cp * u.x + sp * v.x)),
                sphereRadius * (cosA * beam.y + sinA * (cp * u.y + sp * v.y)),
                sphereRadius * (cosA * beam.z + sinA * (cp * u.z + sp * v.z)),
            ));
        }
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const ring = new THREE.Line(geo, ringMat);
        braggGroup.add(ring);
    }
}

/* ══════════════════════════════════════════
   Beam direction
   ══════════════════════════════════════════ */
function getBeamDirection() {
    const rotRad = beamRotDeg * Math.PI / 180;
    const tiltRad = beamTiltDeg * Math.PI / 180;
    return {
        x: Math.sin(rotRad) * Math.cos(tiltRad),
        y: -Math.sin(tiltRad),
        z: Math.cos(rotRad) * Math.cos(tiltRad),
    };
}

/* ══════════════════════════════════════════
   Cube-map face directions (OpenGL)
   ══════════════════════════════════════════ */
const CUBE_FACES = [
    { dir: [ 1, 0, 0], up: [0,-1, 0], right: [0, 0,-1] },
    { dir: [-1, 0, 0], up: [0,-1, 0], right: [0, 0, 1] },
    { dir: [ 0, 1, 0], up: [0, 0, 1], right: [1, 0, 0] },
    { dir: [ 0,-1, 0], up: [0, 0,-1], right: [1, 0, 0] },
    { dir: [ 0, 0, 1], up: [0,-1, 0], right: [1, 0, 0] },
    { dir: [ 0, 0,-1], up: [0,-1, 0], right: [-1,0, 0] },
];

/* ══════════════════════════════════════════
   Bunge rotation matrix (ZXZ convention)
   ══════════════════════════════════════════ */
function bungeToMatrix(phi1D, PHID, phi2D) {
    const DEG = Math.PI / 180;
    const p1 = phi1D * DEG, P = PHID * DEG, p2 = phi2D * DEG;
    const c1 = Math.cos(p1), s1 = Math.sin(p1);
    const cP = Math.cos(P),  sP = Math.sin(P);
    const c2 = Math.cos(p2), s2 = Math.sin(p2);
    return [
        [ c1*c2 - s1*s2*cP, -c1*s2 - s1*c2*cP,  s1*sP ],
        [ s1*c2 + c1*s2*cP, -s1*s2 + c1*c2*cP, -c1*sP ],
        [ s2*sP,              c2*sP,              cP    ]
    ];
}

function rotateVec(R, v) {
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ];
}

/* ══════════════════════════════════════════
   WebGL2 GPU compute pipeline
   ══════════════════════════════════════════ */
let glCompute = null;  /* { gl, program, fb, tex, ... } or null if unavailable */

const GPU_VERT = `#version 300 es
layout(location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

/* Maximum atoms we can pass through a uniform float array */
const GPU_MAX_BASIS = 64;

const GPU_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
out vec4 fragColor;

/* Uniforms */
uniform float uK;                /* wavenumber 2π/λ */
uniform vec3  uKinDir;           /* incident beam direction (unit) */
uniform vec3  uAvec, uBvec, uCvec; /* lattice vectors */
uniform ivec3 uN;                /* number of unit cells each axis */
uniform float uBeamSigma;        /* beam σ in Å (0 = plane wave) */
uniform int   uNBasis;           /* number of basis atoms */

/* Basis atom positions (Cartesian, Å) and form factor at |q| via texture */
uniform vec3  uBasisPos[${GPU_MAX_BASIS}];
uniform int   uBasisType[${GPU_MAX_BASIS}];

/* Form factor lookup: 1D float texture, one row per atom type */
uniform sampler2D uFFTex;
uniform float uFFQMax;
uniform int   uFFSize;
uniform int   uNTypes;

/* Log-scale rendering */
uniform float uLogEps;
uniform float uBrightness;
uniform int   uUseLog;
uniform float uAlphaPower;

/* Viridis colormap texture (256×1 RGB) */
uniform sampler2D uViridis;

/* Output range for normalization — two-pass or computed on CPU */
uniform float uVMin;
uniform float uVMax;

#define PI 3.14159265358979

float ffLookup(int typeIdx, float q) {
    float u = clamp(q / uFFQMax, 0.0, 1.0);
    float v = (float(typeIdx) + 0.5) / float(uNTypes);
    return texture(uFFTex, vec2(u, v)).r;
}

/* Dirichlet kernel D_N(x) = sin(N*x/2) / sin(x/2) */
float dirichlet(int N, float x) {
    float half_x = x * 0.5;
    float s = sin(half_x);
    if (abs(s) < 1e-7) return float(N);
    return sin(float(N) * half_x) / s;
}

/* Analytical Gaussian lattice sum via Poisson summation.
 * L(qa) = (σ√(2π)/|a|) Σ_m exp(-σ²(qa - 2πm)²/(2|a|²))
 * Falls back to Dirichlet when beam wider than crystal. */
float gaussianLatticeSum1D(float qa, float a_len, float sigma, int N) {
    float ratio = sigma / a_len;
    if (ratio > float(N) * 0.4) return dirichlet(N, qa);

    float prefactor = ratio * sqrt(2.0 * PI);
    float half_inv_w2 = ratio * ratio * 0.5;
    float cutoff = 5.0 * a_len / sigma;
    float two_pi = 2.0 * PI;

    int m_min = int(floor((qa - cutoff) / two_pi));
    int m_max = int(ceil((qa + cutoff) / two_pi));

    float sum = 0.0;
    for (int m = m_min; m <= m_max; m++) {
        float dq = qa - two_pi * float(m);
        sum += exp(-dq * dq * half_inv_w2);
    }
    return prefactor * sum;
}

void main() {
    /* Equirectangular mapping: vUV.x → phi [0,2π], vUV.y → theta [0,π] */
    float phi   = vUV.x * 2.0 * PI;
    float theta = vUV.y * PI;

    float sin_t = sin(theta), cos_t = cos(theta);
    float sin_p = sin(phi),   cos_p = cos(phi);

    /* k_out direction */
    vec3 kout = uK * vec3(sin_t * cos_p, sin_t * sin_p, cos_t);
    vec3 kin  = uK * uKinDir;
    vec3 q    = kout - kin;
    float qmag = length(q);

    /* Structure factor S_cell(q) */
    float S_re = 0.0, S_im = 0.0;
    for (int j = 0; j < ${GPU_MAX_BASIS}; j++) {
        if (j >= uNBasis) break;
        float fj = ffLookup(uBasisType[j], qmag);
        float phase = dot(q, uBasisPos[j]);
        S_re += fj * cos(phase);
        S_im -= fj * sin(phase);
    }

    /* Lattice sum L(q) */
    float qa = dot(q, uAvec);
    float qb = dot(q, uBvec);
    float qc = dot(q, uCvec);
    float L_re, L_im;

    if (uBeamSigma <= 0.0) {
        /* Plane wave: analytical Dirichlet kernel */
        L_re = dirichlet(uN.x, qa) * dirichlet(uN.y, qb) * dirichlet(uN.z, qc);
    } else {
        /* Gaussian beam: analytical Poisson-summed Gaussians */
        float a_len = length(uAvec);
        float b_len = length(uBvec);
        float c_len = length(uCvec);
        L_re = gaussianLatticeSum1D(qa, a_len, uBeamSigma, uN.x)
             * gaussianLatticeSum1D(qb, b_len, uBeamSigma, uN.y)
             * gaussianLatticeSum1D(qc, c_len, uBeamSigma, uN.z);
    }
    L_im = 0.0;

    /* F = S_cell * L */
    float F_re = S_re * L_re - S_im * L_im;
    float F_im = S_re * L_im + S_im * L_re;
    float I = F_re * F_re + F_im * F_im;

    /* Store raw intensity in R channel for readback */
    fragColor = vec4(I, 0.0, 0.0, 1.0);
}
`;

/* Second pass: colormap + alpha */
const GPU_COLORMAP_FRAG = `#version 300 es
precision highp float;

in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uIntensity; /* raw intensity texture */
uniform sampler2D uViridis;   /* 256×1 colormap */
uniform float uLogEps;
uniform float uBrightness;
uniform int   uUseLog;
uniform float uAlphaPower;
uniform float uVMin;
uniform float uVMax;

void main() {
    float I = texture(uIntensity, vUV).r;
    float v = uUseLog == 1 ? log(I + uLogEps) : I;
    float range = uVMax - uVMin;
    if (range < 1e-10) range = 1.0;
    float t = clamp((v - uVMin) * uBrightness / range, 0.0, 1.0);

    vec3 color = texture(uViridis, vec2(t, 0.5)).rgb;
    float a = uAlphaPower > 0.001 ? pow(t, uAlphaPower) : 1.0;
    fragColor = vec4(color, a);
}
`;

function initGLCompute() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', {
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true,
    });
    if (!gl) { console.warn('WebGL2 not available, using WASM fallback'); return null; }

    /* Check float texture support */
    const extFloat = gl.getExtension('EXT_color_buffer_float');
    if (!extFloat) { console.warn('EXT_color_buffer_float not available'); return null; }
    gl.getExtension('OES_texture_float_linear');  /* enable if available */

    function compileShader(src, type) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', gl.getShaderInfoLog(sh));
            return null;
        }
        return sh;
    }

    function linkProgram(vs, fs) {
        const vsh = compileShader(vs, gl.VERTEX_SHADER);
        const fsh = compileShader(fs, gl.FRAGMENT_SHADER);
        if (!vsh || !fsh) return null;
        const prog = gl.createProgram();
        gl.attachShader(prog, vsh);
        gl.attachShader(prog, fsh);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            console.error('Program link error:', gl.getProgramInfoLog(prog));
            return null;
        }
        return prog;
    }

    const computeProg = linkProgram(GPU_VERT, GPU_FRAG);
    const colormapProg = linkProgram(GPU_VERT, GPU_COLORMAP_FRAG);
    if (!computeProg || !colormapProg) return null;

    /* Fullscreen quad VAO */
    const quadVerts = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
    /* Use location 0 — matches layout(location=0) in vertex shader */
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    /* Create viridis 256×1 texture */
    const viridisData = new Uint8Array(256 * 3);
    /* We need to copy viridis LUT from C side — use JS-side copy */
    const viridisRGB = [
        [68,1,84],[68,2,85],[68,3,87],[69,5,88],[69,6,90],[69,8,91],[70,9,92],[70,11,94],
        [70,12,95],[70,14,97],[71,15,98],[71,17,99],[71,18,101],[71,20,102],[71,21,103],[71,22,105],
        [71,24,106],[72,25,107],[72,26,108],[72,28,110],[72,29,111],[72,30,112],[72,32,113],[72,33,114],
        [72,34,115],[72,35,116],[71,37,117],[71,38,118],[71,39,119],[71,40,120],[71,42,121],[71,43,122],
        [71,44,123],[70,45,124],[70,47,124],[70,48,125],[70,49,126],[69,50,127],[69,52,127],[69,53,128],
        [69,54,129],[68,55,129],[68,57,130],[67,58,131],[67,59,131],[67,60,132],[66,61,132],[66,62,133],
        [66,64,133],[65,65,134],[65,66,134],[64,67,135],[64,68,135],[63,69,135],[63,71,136],[62,72,136],
        [62,73,137],[61,74,137],[61,75,137],[61,76,137],[60,77,138],[60,78,138],[59,80,138],[59,81,138],
        [58,82,139],[58,83,139],[57,84,139],[57,85,139],[56,86,139],[56,87,140],[55,88,140],[55,89,140],
        [54,90,140],[54,91,140],[53,92,140],[53,93,140],[52,94,141],[52,95,141],[51,96,141],[51,97,141],
        [50,98,141],[50,99,141],[49,100,141],[49,101,141],[49,102,141],[48,103,141],[48,104,141],[47,105,141],
        [47,106,141],[46,107,142],[46,108,142],[46,109,142],[45,110,142],[45,111,142],[44,112,142],[44,113,142],
        [44,114,142],[43,115,142],[43,116,142],[42,117,142],[42,118,142],[42,119,142],[41,120,142],[41,121,142],
        [40,122,142],[40,122,142],[40,123,142],[39,124,142],[39,125,142],[39,126,142],[38,127,142],[38,128,142],
        [38,129,142],[37,130,142],[37,131,141],[36,132,141],[36,133,141],[36,134,141],[35,135,141],[35,136,141],
        [35,137,141],[34,137,141],[34,138,141],[34,139,141],[33,140,141],[33,141,140],[33,142,140],[32,143,140],
        [32,144,140],[32,145,140],[31,146,140],[31,147,139],[31,148,139],[31,149,139],[31,150,139],[30,151,138],
        [30,152,138],[30,153,138],[30,153,138],[30,154,137],[30,155,137],[30,156,137],[30,157,136],[30,158,136],
        [30,159,136],[30,160,135],[31,161,135],[31,162,134],[31,163,134],[32,164,133],[32,165,133],[33,166,133],
        [33,167,132],[34,167,132],[35,168,131],[35,169,130],[36,170,130],[37,171,129],[38,172,129],[39,173,128],
        [40,174,127],[41,175,127],[42,176,126],[43,177,125],[44,177,125],[46,178,124],[47,179,123],[48,180,122],
        [50,181,122],[51,182,121],[53,183,120],[54,184,119],[56,185,118],[57,185,118],[59,186,117],[61,187,116],
        [62,188,115],[64,189,114],[66,190,113],[68,190,112],[69,191,111],[71,192,110],[73,193,109],[75,194,108],
        [77,194,107],[79,195,105],[81,196,104],[83,197,103],[85,198,102],[87,198,101],[89,199,100],[91,200,98],
        [94,201,97],[96,201,96],[98,202,95],[100,203,93],[103,204,92],[105,204,91],[107,205,89],[109,206,88],
        [112,206,86],[114,207,85],[116,208,84],[119,208,82],[121,209,81],[124,210,79],[126,210,78],[129,211,76],
        [131,211,75],[134,212,73],[136,213,71],[139,213,70],[141,214,68],[144,214,67],[146,215,65],[149,215,63],
        [151,216,62],[154,216,60],[157,217,58],[159,217,56],[162,218,55],[165,218,53],[167,219,51],[170,219,50],
        [173,220,48],[175,220,46],[178,221,44],[181,221,43],[183,221,41],[186,222,39],[189,222,38],[191,223,36],
        [194,223,34],[197,223,33],[199,224,31],[202,224,30],[205,224,29],[207,225,28],[210,225,27],[212,225,26],
        [215,226,25],[218,226,24],[220,226,24],[223,227,24],[225,227,24],[228,227,24],[231,228,25],[233,228,25],
        [236,228,26],[238,229,27],[241,229,28],[243,229,30],[246,230,31],[248,230,33],[250,230,34],[253,231,36],
    ];
    for (let i = 0; i < 256; i++) {
        viridisData[i*3+0] = viridisRGB[i][0];
        viridisData[i*3+1] = viridisRGB[i][1];
        viridisData[i*3+2] = viridisRGB[i][2];
    }
    const viridisTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, viridisTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, viridisData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    console.log('WebGL2 GPU compute pipeline initialized');

    return {
        gl, canvas, vao, computeProg, colormapProg, viridisTex,
        /* Textures & FBOs created per-resolution */
        intensityTex: null, intensityFB: null,
        colormapTex: null, colormapFB: null,
        ffTex: null,
        curW: 0, curH: 0,
    };
}

function ensureGLTextures(gc, w, h) {
    const gl = gc.gl;
    if (gc.curW === w && gc.curH === h) return;

    /* Intensity float texture + FBO */
    if (gc.intensityTex) gl.deleteTexture(gc.intensityTex);
    if (gc.intensityFB) gl.deleteFramebuffer(gc.intensityFB);
    gc.intensityTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, gc.intensityTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gc.intensityFB = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, gc.intensityFB);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gc.intensityTex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        console.warn('Intensity FBO incomplete, disabling GPU compute');
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gc.curW = 0; gc.curH = 0;
        return;
    }

    /* Colormap RGBA texture + FBO */
    if (gc.colormapTex) gl.deleteTexture(gc.colormapTex);
    if (gc.colormapFB) gl.deleteFramebuffer(gc.colormapFB);
    gc.colormapTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, gc.colormapTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gc.colormapFB = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, gc.colormapFB);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gc.colormapTex, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gc.curW = w;
    gc.curH = h;
    gc.canvas.width = w;
    gc.canvas.height = h;
}

function uploadFFTexture(gc, tables, nTypes, ffSize) {
    const gl = gc.gl;
    if (gc.ffTex) gl.deleteTexture(gc.ffTex);
    gc.ffTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, gc.ffTex);
    /* Store form factors as a ffSize × nTypes float texture */
    const data = new Float32Array(ffSize * nTypes);
    for (let t = 0; t < nTypes; t++) data.set(tables[t], t * ffSize);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, ffSize, nTypes, 0, gl.RED, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function gpuComputePattern(gc, res) {
    if (!crystal) return null;
    const cr = crystal;
    const gl = gc.gl;
    const nBasis = cr.basis.length;

    const R = bungeToMatrix(bungePhi1, bungePHI, bungePhi2);
    const rAvec = rotateVec(R, cr.avec);
    const rBvec = rotateVec(R, cr.bvec);
    const rCvec = rotateVec(R, cr.cvec);

    const uniqueSyms = [...new Set(cr.symbols)];
    const nTypes = uniqueSyms.length;
    const typeIdxArr = cr.symbols.map(s => uniqueSyms.indexOf(s));

    const tables = uniqueSyms.map(sym => {
        const atom = ATOM_BY_SYM[sym];
        if (!atom) return computeFormFactorTable(ATOM_BY_SYM['C']);
        return computeFormFactorTable(atom);
    });

    /* Precompute Cartesian basis positions */
    const basisPos = [];
    for (let j = 0; j < nBasis; j++) {
        const fx = cr.basis[j][0], fy = cr.basis[j][1], fz = cr.basis[j][2];
        basisPos.push([
            fx * rAvec[0] + fy * rBvec[0] + fz * rCvec[0],
            fx * rAvec[1] + fy * rBvec[1] + fz * rCvec[1],
            fx * rAvec[2] + fy * rBvec[2] + fz * rCvec[2],
        ]);
    }

    const eqW = res * 2, eqH = res;
    ensureGLTextures(gc, eqW, eqH);
    if (gc.curW !== eqW || gc.curH !== eqH) return null;  /* FBO creation failed */
    uploadFFTexture(gc, tables, nTypes, FF_TABLE_SIZE);

    const kVal = 2.0 * Math.PI / wavelengthA;
    const dir = getBeamDirection();
    const rawSigma = beamMode === 'plane' ? 0.0 : Math.pow(10, beamSigmaLog);
    /* Clamp sigma so Gaussian peaks span ≥ 2 pixels (instrument resolution) */
    const sigmaMax = rawSigma > 0 ? Math.min(eqW, eqH) / (2.0 * Math.PI * kVal) : 0;
    const sigma = rawSigma > 0 ? Math.min(rawSigma, sigmaMax) : 0;

    /* ── Pass 1: compute intensity ── */
    gl.bindFramebuffer(gl.FRAMEBUFFER, gc.intensityFB);
    gl.viewport(0, 0, eqW, eqH);
    gl.useProgram(gc.computeProg);
    gl.bindVertexArray(gc.vao);

    const u = (name) => gl.getUniformLocation(gc.computeProg, name);
    gl.uniform1f(u('uK'), kVal);
    gl.uniform3f(u('uKinDir'), dir.x, dir.y, dir.z);
    gl.uniform3f(u('uAvec'), rAvec[0], rAvec[1], rAvec[2]);
    gl.uniform3f(u('uBvec'), rBvec[0], rBvec[1], rBvec[2]);
    gl.uniform3f(u('uCvec'), rCvec[0], rCvec[1], rCvec[2]);
    gl.uniform3i(u('uN'), nCells, nCells, nCells);
    gl.uniform1f(u('uBeamSigma'), sigma);
    gl.uniform1i(u('uNBasis'), Math.min(nBasis, GPU_MAX_BASIS));
    gl.uniform1f(u('uFFQMax'), FF_Q_MAX);
    gl.uniform1i(u('uFFSize'), FF_TABLE_SIZE);
    gl.uniform1i(u('uNTypes'), nTypes);

    for (let j = 0; j < Math.min(nBasis, GPU_MAX_BASIS); j++) {
        gl.uniform3f(u(`uBasisPos[${j}]`), basisPos[j][0], basisPos[j][1], basisPos[j][2]);
        gl.uniform1i(u(`uBasisType[${j}]`), typeIdxArr[j]);
    }

    /* Bind form factor texture to unit 0 */
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, gc.ffTex);
    gl.uniform1i(u('uFFTex'), 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    /* Read back intensity — must use RGBA/FLOAT (guaranteed for RGBA32F FBOs) */
    const floatBuf4 = new Float32Array(eqW * eqH * 4);
    gl.readPixels(0, 0, eqW, eqH, gl.RGBA, gl.FLOAT, floatBuf4);
    const floatBuf = new Float32Array(eqW * eqH);
    for (let i = 0; i < eqW * eqH; i++) floatBuf[i] = floatBuf4[i * 4];

    /* Validate — if all zeros, GPU failed silently */
    let maxVal = 0;
    for (let i = 0; i < floatBuf.length; i++) { if (floatBuf[i] > maxVal) maxVal = floatBuf[i]; }
    if (maxVal === 0) {
        console.warn('GPU compute produced all zeros — falling back to WASM');
        return null;
    }

    return { floatBuf, eqW, eqH };
}

function gpuColormapAndReadback(gc, floatBuf, eqW, eqH) {
    const gl = gc.gl;
    const eps = Math.pow(10, logEpsilon);

    /* Compute min/max for normalization */
    let vmin = 1e30, vmax = -1e30;
    const N = eqW * eqH;
    if (useLogScale) {
        for (let i = 0; i < N; i++) {
            const v = Math.log(floatBuf[i] + eps);
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    } else {
        for (let i = 0; i < N; i++) {
            if (floatBuf[i] < vmin) vmin = floatBuf[i];
            if (floatBuf[i] > vmax) vmax = floatBuf[i];
        }
        /* Override with user cmax for linear mode */
        if (linearCmax > 0) {
            vmax = linearCmax;
            vmin = 0;
        }
    }

    /* ── Pass 2: colormap ── */
    gl.bindFramebuffer(gl.FRAMEBUFFER, gc.colormapFB);
    gl.viewport(0, 0, eqW, eqH);
    gl.useProgram(gc.colormapProg);
    gl.bindVertexArray(gc.vao);

    const u2 = (name) => gl.getUniformLocation(gc.colormapProg, name);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, gc.intensityTex);
    gl.uniform1i(u2('uIntensity'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, gc.viridisTex);
    gl.uniform1i(u2('uViridis'), 1);
    gl.uniform1f(u2('uLogEps'), eps);
    gl.uniform1f(u2('uBrightness'), brightness);
    gl.uniform1i(u2('uUseLog'), useLogScale ? 1 : 0);
    gl.uniform1f(u2('uAlphaPower'), transparencyPower);
    gl.uniform1f(u2('uVMin'), vmin);
    gl.uniform1f(u2('uVMax'), vmax);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    /* Read back RGBA */
    const rgba = new Uint8Array(eqW * eqH * 4);
    gl.readPixels(0, 0, eqW, eqH, gl.RGBA, gl.UNSIGNED_BYTE, rgba);

    /* WebGL renders bottom-up, flip vertically */
    const flipped = new Uint8Array(eqW * eqH * 4);
    const rowBytes = eqW * 4;
    for (let y = 0; y < eqH; y++) {
        flipped.set(
            rgba.subarray((eqH - 1 - y) * rowBytes, (eqH - y) * rowBytes),
            y * rowBytes
        );
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return flipped;
}

/* Cached GPU intensity for render-only updates */
let gpuCachedIntensity = null;
let gpuCachedW = 0, gpuCachedH = 0;

/* ══════════════════════════════════════════
   WASM compute
   ══════════════════════════════════════════ */
function computePattern(res) {
    if (!crystal) return;

    if (glCompute) {
        /* GPU path */
        const t0 = performance.now();
        const result = gpuComputePattern(glCompute, res);
        if (result) {
            gpuCachedIntensity = result.floatBuf;
            gpuCachedW = result.eqW;
            gpuCachedH = result.eqH;

            const eqCopy = gpuColormapAndReadback(glCompute, result.floatBuf, result.eqW, result.eqH);
            const dt = performance.now() - t0;

            remapEquirectToCubefaces(eqCopy, result.eqW, result.eqH);

            statusBar.textContent =
                `${result.eqW}\u00d7${result.eqH} GPU in ${dt.toFixed(0)} ms | \u03bb = ${wavelengthA.toFixed(3)} \u00c5` +
                ` | ${beamMode === 'plane' ? 'plane wave' : '\u03c3 = ' + formatSigma(Math.pow(10, beamSigmaLog))}`;
            return;
        }
    }

    /* WASM fallback */
    if (!wasm) return;

    const cr = crystal;
    const nBasis = cr.basis.length;

    /* Apply Bunge rotation to lattice vectors */
    const R = bungeToMatrix(bungePhi1, bungePHI, bungePhi2);
    const rAvec = rotateVec(R, cr.avec);
    const rBvec = rotateVec(R, cr.bvec);
    const rCvec = rotateVec(R, cr.cvec);

    const uniqueSyms = [...new Set(cr.symbols)];
    const nTypes = uniqueSyms.length;
    const typeIdxArr = cr.symbols.map(s => uniqueSyms.indexOf(s));

    const tables = uniqueSyms.map(sym => {
        const atom = ATOM_BY_SYM[sym];
        if (!atom) {
            console.warn(`No form factor for '${sym}', using C`);
            return computeFormFactorTable(ATOM_BY_SYM['C']);
        }
        return computeFormFactorTable(atom);
    });

    const basisFlat = new Float32Array(nBasis * 3);
    for (let j = 0; j < nBasis; j++) {
        basisFlat[j * 3 + 0] = cr.basis[j][0];
        basisFlat[j * 3 + 1] = cr.basis[j][1];
        basisFlat[j * 3 + 2] = cr.basis[j][2];
    }

    const basisPtr = wasm._malloc(basisFlat.byteLength);
    wasm.HEAPF32.set(basisFlat, basisPtr >> 2);

    const typeIdxFlat = new Int32Array(typeIdxArr);
    const typeIdxPtr = wasm._malloc(typeIdxFlat.byteLength);
    wasm.HEAP32.set(typeIdxFlat, typeIdxPtr >> 2);

    const ffConcat = new Float32Array(nTypes * FF_TABLE_SIZE);
    for (let t = 0; t < nTypes; t++) ffConcat.set(tables[t], t * FF_TABLE_SIZE);
    const ffPtr = wasm._malloc(ffConcat.byteLength);
    wasm.HEAPF32.set(ffConcat, ffPtr >> 2);

    const k = 2.0 * Math.PI / wavelengthA;
    const dir = getBeamDirection();
    const sigma = beamMode === 'plane' ? 0.0 : Math.pow(10, beamSigmaLog);
    const eps = Math.pow(10, logEpsilon);

    const t0 = performance.now();

    const eqW = res * 2;
    const eqH = res;

    if (fieldPtr) wasm._helmholtz3d_wasm_free(fieldPtr);
    fieldPtr = wasm._helmholtz3d_wasm_alloc(eqW, eqH);

    wasm._helmholtz3d_wasm_compute(
        fieldPtr,
        basisPtr, typeIdxPtr, nBasis,
        ffPtr, nTypes, FF_TABLE_SIZE, FF_Q_MAX,
        rAvec[0], rAvec[1], rAvec[2],
        rBvec[0], rBvec[1], rBvec[2],
        rCvec[0], rCvec[1], rCvec[2],
        nCells, nCells, nCells,
        k,
        dir.x, dir.y, dir.z,
        sigma
    );

    const cmaxVal = useLogScale ? -1.0 : linearCmax;
    const rgbaPtr = wasm._helmholtz3d_wasm_render(
        fieldPtr, eps, brightness,
        useLogScale ? 1 : 0,
        transparencyPower,
        cmaxVal
    );

    const dt = performance.now() - t0;

    const eqRGBA = new Uint8Array(wasm.HEAPU8.buffer, rgbaPtr, eqW * eqH * 4);
    const eqCopy = new Uint8Array(eqW * eqH * 4);
    eqCopy.set(eqRGBA);

    wasm._free(basisPtr);
    wasm._free(typeIdxPtr);
    wasm._free(ffPtr);

    remapEquirectToCubefaces(eqCopy, eqW, eqH);

    statusBar.textContent =
        `${eqW}\u00d7${eqH} cubemap in ${dt.toFixed(0)} ms | \u03bb = ${wavelengthA.toFixed(3)} \u00c5` +
        ` | ${beamMode === 'plane' ? 'plane wave' : '\u03c3 = ' + formatSigma(Math.pow(10, beamSigmaLog))}`;
}

/* ══════════════════════════════════════════
   Remap equirectangular RGBA → 6 cube-face canvases
   Always writes to faceRes-sized canvases.
   ══════════════════════════════════════════ */
function remapEquirectToCubefaces(eqCopy, eqW, eqH) {
    const outRes = faceRes;
    for (let face = 0; face < 6; face++) {
        const fd = CUBE_FACES[face];
        const canvas = cubeCanvases[face];
        if (canvas.width !== outRes || canvas.height !== outRes) {
            canvas.width = outRes; canvas.height = outRes;
        }
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(outRes, outRes);
        const faceData = imgData.data;

        for (let ty = 0; ty < outRes; ty++) {
            for (let tx = 0; tx < outRes; tx++) {
                const s = (tx + 0.5) / outRes * 2.0 - 1.0;
                const t = (ty + 0.5) / outRes * 2.0 - 1.0;

                let dx = fd.dir[0] + s * fd.right[0] + t * fd.up[0];
                let dy = fd.dir[1] + s * fd.right[1] + t * fd.up[1];
                let dz = fd.dir[2] + s * fd.right[2] + t * fd.up[2];
                const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
                dx /= len; dy /= len; dz /= len;

                const theta = Math.acos(Math.max(-1, Math.min(1, dz)));
                const phi = Math.atan2(dy, dx);
                const phiPos = phi < 0 ? phi + 2 * Math.PI : phi;

                const eqCol = (phiPos / (2 * Math.PI)) * eqW;
                const eqRow = (theta / Math.PI) * eqH;

                const c0 = Math.floor(eqCol - 0.5);
                const r0 = Math.floor(eqRow - 0.5);
                const fc = eqCol - 0.5 - c0;
                const fr = eqRow - 0.5 - r0;

                const dstIdx = (ty * outRes + tx) * 4;
                for (let ch = 0; ch < 4; ch++) {
                    const v00 = eqSample(eqCopy, eqW, eqH, r0, c0, ch);
                    const v10 = eqSample(eqCopy, eqW, eqH, r0 + 1, c0, ch);
                    const v01 = eqSample(eqCopy, eqW, eqH, r0, c0 + 1, ch);
                    const v11 = eqSample(eqCopy, eqW, eqH, r0 + 1, c0 + 1, ch);
                    faceData[dstIdx + ch] = Math.round(
                        v00*(1-fc)*(1-fr) + v01*fc*(1-fr) +
                        v10*(1-fc)*fr     + v11*fc*fr
                    );
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }

    /* Update CubeTexture — recreate if canvas size changed */
    if (outRes !== lastCanvasSize) {
        cubeTex.dispose();
        cubeTex = new THREE.CubeTexture(cubeCanvases);
        sphereMat.uniforms.uCubeMap.value = cubeTex;
        lastCanvasSize = outRes;
    }
    cubeTex.needsUpdate = true;
}

function eqSample(data, W, H, row, col, ch) {
    col = ((col % W) + W) % W;
    if (row < 0) row = 0;
    if (row >= H) row = H - 1;
    return data[(row * W + col) * 4 + ch];
}

/* ══════════════════════════════════════════
   Scheduling — low-res during interaction, full-res 500ms after
   ══════════════════════════════════════════ */
function scheduleCompute() {
    if (fineTimer) clearTimeout(fineTimer);
    /* Immediate coarse render at 1/4 of selected resolution */
    const coarseRes = Math.max(64, faceRes >> 2);
    computePattern(coarseRes);
    /* Full resolution 500ms after last interaction */
    fineTimer = setTimeout(() => {
        computePattern(faceRes);
        fineTimer = null;
    }, 500);
}

/* ══════════════════════════════════════════
   Render-only recompute (reuse cached intensity, skip WASM compute)
   Used for log-eps, brightness, transparency, log-scale changes.
   ══════════════════════════════════════════ */
function scheduleRenderOnly() {
    if (fineTimer) clearTimeout(fineTimer);
    /* GPU path: re-colormap from cached intensity */
    if (glCompute && gpuCachedIntensity) {
        const eqCopy = gpuColormapAndReadback(glCompute, gpuCachedIntensity, gpuCachedW, gpuCachedH);
        remapEquirectToCubefaces(eqCopy, gpuCachedW, gpuCachedH);
        return;
    }
    /* WASM path */
    if (!wasm || !fieldPtr) { scheduleCompute(); return; }
    rerenderFromCache(faceRes);
    /* No coarse/fine cycle needed — render is fast */
}

function rerenderFromCache(res) {
    if (!wasm || !fieldPtr) return;
    const eps = Math.pow(10, logEpsilon);
    const cmaxVal = useLogScale ? -1.0 : linearCmax;
    const rgbaPtr = wasm._helmholtz3d_wasm_render(
        fieldPtr, eps, brightness,
        useLogScale ? 1 : 0,
        transparencyPower,
        cmaxVal
    );
    const eqW = wasm._helmholtz3d_wasm_width(fieldPtr);
    const eqH = wasm._helmholtz3d_wasm_height(fieldPtr);
    const eqRGBA = new Uint8Array(wasm.HEAPU8.buffer, rgbaPtr, eqW * eqH * 4);
    const eqCopy = new Uint8Array(eqW * eqH * 4);
    eqCopy.set(eqRGBA);
    remapEquirectToCubefaces(eqCopy, eqW, eqH);
}

/* ══════════════════════════════════════════
   Format helpers
   ══════════════════════════════════════════ */
function formatSigma(sigmaA) {
    if (sigmaA < 10)    return sigmaA.toFixed(2) + ' \u00c5';
    if (sigmaA < 1e4)   return (sigmaA / 10).toFixed(1) + ' nm';
    if (sigmaA < 1e7)   return (sigmaA / 1e4).toFixed(2) + ' \u00b5m';
    return (sigmaA / 1e7).toFixed(2) + ' mm';
}

/* ══════════════════════════════════════════
   UI wiring
   ══════════════════════════════════════════ */
function wireUI() {
    slWavelength.addEventListener('input', () => {
        wavelengthA = sliderToLambda(parseFloat(slWavelength.value));
        valWavelength.textContent = wavelengthA.toFixed(3) + ' \u00c5';
        rebuildBraggRings();
        scheduleCompute();
    });

    slRotation.addEventListener('input', () => {
        beamRotDeg = parseFloat(slRotation.value);
        valRotation.textContent = beamRotDeg.toFixed(1) + '\u00b0';
        rebuildBeam();
        rebuildBraggRings();
        scheduleCompute();
    });

    slTilt.addEventListener('input', () => {
        beamTiltDeg = parseFloat(slTilt.value);
        valTilt.textContent = beamTiltDeg.toFixed(1) + '\u00b0';
        rebuildBeam();
        rebuildBraggRings();
        scheduleCompute();
    });

    slBeamSigma.addEventListener('input', () => {
        beamSigmaLog = parseFloat(slBeamSigma.value);
        valBeamSigma.textContent = formatSigma(Math.pow(10, beamSigmaLog));
        rebuildBeam();
        scheduleCompute();
    });

    cifFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        loadCrystalFromCIF(await file.text(), file.name);
    });

    slNCells.addEventListener('input', () => {
        nCells = Math.max(1, Math.round(Math.pow(10, parseFloat(slNCells.value))));
        valNCells.textContent = nCells;
        rebuildAtoms();
        rebuildBeam();
        rebuildBraggRings();
        scheduleCompute();
    });

    chkAtoms.addEventListener('change', () => { showAtoms = chkAtoms.checked; rebuildAtoms(); });
    chkBeam.addEventListener('change', () => { showBeam = chkBeam.checked; rebuildBeam(); });
    if (chkBraggRings) chkBraggRings.addEventListener('change', () => { showBraggRings = chkBraggRings.checked; rebuildBraggRings(); });
    chkLogScale.addEventListener('change', () => {
        useLogScale = chkLogScale.checked;
        /* Show/hide cmax slider */
        if (cmaxRow) cmaxRow.style.display = useLogScale ? 'none' : '';
        scheduleRenderOnly();
    });
    /* Initial visibility of cmax slider */
    if (cmaxRow) cmaxRow.style.display = useLogScale ? 'none' : '';

    selBeamMode.addEventListener('change', () => {
        beamMode = selBeamMode.value;
        /* Hide sigma slider when plane wave */
        const sigmaRow = slBeamSigma.closest('.slider-group');
        if (sigmaRow) sigmaRow.style.display = beamMode === 'plane' ? 'none' : '';
        rebuildBeam();
        scheduleCompute();
    });
    /* Hide sigma slider initially for default plane wave */
    { const sigmaRow = slBeamSigma.closest('.slider-group');
      if (sigmaRow) sigmaRow.style.display = beamMode === 'plane' ? 'none' : ''; }

    slLogEps.addEventListener('input', () => {
        logEpsilon = parseFloat(slLogEps.value);
        valLogEps.innerHTML = '10<sup>' + Math.round(logEpsilon) + '</sup>';
        scheduleRenderOnly();
    });

    if (slLinearCmax) slLinearCmax.addEventListener('input', () => {
        linearCmaxLog = parseFloat(slLinearCmax.value);
        linearCmax = Math.pow(10, linearCmaxLog);
        valLinearCmax.innerHTML = '10<sup>' + linearCmaxLog.toFixed(1) + '</sup>';
        scheduleRenderOnly();
    });

    slTransparency.addEventListener('input', () => {
        transparencyPower = parseFloat(slTransparency.value);
        valTransparency.textContent = transparencyPower.toFixed(2);
        scheduleRenderOnly();
    });

    slSphereRadius.addEventListener('input', () => {
        sphereRadiusLog = parseFloat(slSphereRadius.value);
        sphereRadius = Math.round(Math.pow(10, sphereRadiusLog));
        valSphereRadius.textContent = sphereRadius >= 1e6
            ? sphereRadius.toExponential(1) : sphereRadius;
        rebuildSphere();
        rebuildBeam();
        rebuildBraggRings();
    });

    slSphereAlpha.addEventListener('input', () => {
        sphereAlpha = parseInt(slSphereAlpha.value) / 100;
        valSphereAlpha.textContent = sphereAlpha.toFixed(2);
        sphereMat.uniforms.uOpacity.value = sphereAlpha;
    });

    slBrightness.addEventListener('input', () => {
        brightness = parseFloat(slBrightness.value);
        valBrightness.textContent = brightness.toFixed(2);
        scheduleRenderOnly();
    });

    selResolution.addEventListener('change', () => {
        faceRes = parseInt(selResolution.value);
        scheduleCompute();
    });

    /* Bunge Euler angle sliders */
    slBungePhi1.addEventListener('input', () => {
        bungePhi1 = parseFloat(slBungePhi1.value);
        valBungePhi1.textContent = bungePhi1.toFixed(1) + '\u00b0';
        rebuildAtoms();
        rebuildBraggRings();
        scheduleCompute();
    });
    slBungePHI.addEventListener('input', () => {
        bungePHI = parseFloat(slBungePHI.value);
        valBungePHI.textContent = bungePHI.toFixed(1) + '\u00b0';
        rebuildAtoms();
        rebuildBraggRings();
        scheduleCompute();
    });
    slBungePhi2.addEventListener('input', () => {
        bungePhi2 = parseFloat(slBungePhi2.value);
        valBungePhi2.textContent = bungePhi2.toFixed(1) + '\u00b0';
        rebuildAtoms();
        rebuildBraggRings();
        scheduleCompute();
    });
}

/* ══════════════════════════════════════════
   Load crystal from CIF
   ══════════════════════════════════════════ */
function loadCrystalFromCIF(text, filename) {
    try {
        crystal = parseCIF(text);
        cifNameEl.textContent = crystal.name || filename || 'Loaded';
        console.log(`Loaded crystal: ${crystal.name}, ${crystal.basis.length} atoms/cell, ` +
                     `a=${crystal.a.toFixed(4)} b=${crystal.b.toFixed(4)} c=${crystal.c.toFixed(4)}`);
        rebuildAtoms();
        rebuildBeam();
        rebuildBraggRings();
        scheduleCompute();
    } catch (e) {
        console.error('CIF parse error:', e);
        cifNameEl.textContent = 'Parse error!';
    }
}

/* ══════════════════════════════════════════
   Animation loop
   ══════════════════════════════════════════ */
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

/* ══════════════════════════════════════════
   Initialise
   ══════════════════════════════════════════ */
async function init() {
    /* Try to initialize WebGL2 GPU compute pipeline */
    glCompute = initGLCompute();
    if (glCompute) {
        statusBar.textContent = 'GPU compute ready \u2014 loading crystal\u2026';
    }

    try {
        wasm = await Helmholtz3DModule();
        if (!glCompute) {
            statusBar.textContent = 'WASM loaded \u2014 loading crystal\u2026';
        }
    } catch (e) {
        if (!glCompute) {
            statusBar.textContent = 'Failed to load WASM: ' + e.message;
            console.error(e);
            return;
        }
        console.warn('WASM load failed, using GPU only:', e);
        wasm = null;
    }

    wireUI();

    try {
        const resp = await fetch('cifs/aluminum.cif');
        if (resp.ok) {
            loadCrystalFromCIF(await resp.text(), 'aluminum.cif');
        } else {
            useFallbackCrystal();
        }
    } catch (e) {
        console.warn('CIF fetch failed, using fallback:', e);
        useFallbackCrystal();
    }

    animate();
}

function useFallbackCrystal() {
    crystal = {
        name: 'Aluminium (FCC)',
        a: 4.04958, b: 4.04958, c: 4.04958,
        alpha: 90, beta: 90, gamma: 90,
        avec: [4.04958, 0, 0],
        bvec: [0, 4.04958, 0],
        cvec: [0, 0, 4.04958],
        basis: [[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]],
        symbols: ['Al','Al','Al','Al'],
    };
    cifNameEl.textContent = 'Aluminium (fallback)';
    rebuildAtoms();
    rebuildBeam();
    rebuildBraggRings();
    scheduleCompute();
}

init();
