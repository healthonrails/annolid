import {
  THREE,
  OrbitControls,
  RoomEnvironment,
} from './annolid_threejs_runtime.js';

import { AnnolidShaders } from './annolid_shaders.js';
import {
  RAT_KEYPOINTS,
  RAT_NCAP_SPEC,
  RatNCAPController,
  computeBumpyTerrainHeight,
  defaultSpeedForBehavior,
} from './rat_ncap_circuit.js';
import { loadRatV2SkinnedModel } from './rat_ncap_model.js';

const query = new URLSearchParams(window.location.search);
const DATASET_CAPTURE = query.get('dataset') === '1';
const DATASET_FRAME_INDEX = Math.max(0, Math.min(10000, Number.parseInt(query.get('frame') || '0', 10) || 0));
const DATASET_CAPTURE_FPS = Math.max(1, Math.min(30, Number.parseInt(query.get('fps') || '5', 10) || 5));
const DATASET_SEED = Number.parseInt(query.get('seed') || '20261125', 10) >>> 0;
const DATASET_SIM_HZ = 50;

const INITIAL_SCENARIO = (query.get('scenario') || query.get('behavior') || 'trot').trim().toLowerCase();
const INITIAL_COAT = (query.get('coat') || 'black_hooded').trim().toLowerCase();
const INITIAL_INTERFACE = (query.get('interface') || 'flexor_extensor').trim().toLowerCase();
const INITIAL_CAMERA_VIEW = (query.get('camera_view') || 'oblique').trim().toLowerCase();
const INITIAL_ARENA = (query.get('arena') || (INITIAL_SCENARIO === 'bumpy_terrain' ? 'bumpy_terrain' : 'treadmill')).trim().toLowerCase();
const INITIAL_LIGHTING = (query.get('lighting') || 'lab_daylight').trim().toLowerCase();
const INITIAL_ABLATION = (query.get('ablation') || 'none').trim().toLowerCase();
const INITIAL_ABLATION_SCALE = query.has('ablation_scale')
  ? Math.max(0, Math.min(1, Number.parseFloat(query.get('ablation_scale')) || 0))
  : 0;
const INITIAL_RG_NOISE = Math.max(0, Math.min(0.5, Number.parseFloat(query.get('rg_noise') || '0') || 0));
const INITIAL_BUMP_CM = query.has('bump_cm')
  ? Math.max(0, Math.min(4, Number.parseFloat(query.get('bump_cm')) || 0))
  : (INITIAL_SCENARIO === 'bumpy_terrain' || INITIAL_ARENA === 'bumpy_terrain' ? 2.2 : 0.0);
const INITIAL_TARGET_SPEED = query.has('target_speed')
  ? Math.max(0, Math.min(1, Number.parseFloat(query.get('target_speed')) || 0))
  : defaultSpeedForBehavior(INITIAL_SCENARIO);

if (DATASET_CAPTURE) {
  document.body.classList.add('dataset-capture');
}

const WORLD_SCALE = 10.0;

// ---------------------------------------------------------------------------
// Studio Renderer, Environment & Paper-Demo Lighting
// ---------------------------------------------------------------------------
const canvas = document.querySelector('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(DATASET_CAPTURE ? 1 : Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.94;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
pmrem.dispose();

const annolidShaders = new AnnolidShaders(THREE);

const camera = new THREE.PerspectiveCamera(34, window.innerWidth / Math.max(1, window.innerHeight), 0.05, 160);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 1.2;
controls.maxDistance = 24.0;

const hemiLight = new THREE.HemisphereLight(0xf6f9ff, 0x6b7788, 0.72);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xfffaf2, 1.48);
keyLight.position.set(4.5, 8.5, 5.2);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 26;
keyLight.shadow.camera.left = -5.5;
keyLight.shadow.camera.right = 5.5;
keyLight.shadow.camera.top = 5.5;
keyLight.shadow.camera.bottom = -5.5;
keyLight.shadow.bias = -0.0004;
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0xd9ebff, 0.42);
rimLight.position.set(-4.5, 4.2, -4.8);
scene.add(rimLight);

function applyLightingPreset(preset) {
  const mode = String(preset || 'lab_daylight').toLowerCase();
  if (mode === 'infrared' || mode === 'nir_night' || mode === 'ir_night') {
    scene.background = new THREE.Color(0x1f232a);
    scene.fog = new THREE.Fog(0x1f232a, 14, 34);
    hemiLight.color.setHex(0xc8cfd8);
    hemiLight.groundColor.setHex(0x2a2d33);
    hemiLight.intensity = 0.76;
    keyLight.color.setHex(0xd8dfe8);
    keyLight.intensity = 1.32;
  } else if (mode === 'dim_red' || mode === 'dark_lab' || mode === 'dim_warm') {
    scene.background = new THREE.Color(0x231d1d);
    scene.fog = new THREE.Fog(0x231d1d, 14, 34);
    hemiLight.color.setHex(0xffc3b8);
    hemiLight.groundColor.setHex(0x302122);
    hemiLight.intensity = 0.68;
    keyLight.color.setHex(0xffd2c2);
    keyLight.intensity = 1.24;
  } else if (mode === 'calms_gray') {
    scene.background = new THREE.Color(0xbec4cc);
    scene.fog = new THREE.Fog(0xbec4cc, 16, 36);
    hemiLight.color.setHex(0xf2f4f7);
    hemiLight.groundColor.setHex(0x707782);
    hemiLight.intensity = 0.78;
    keyLight.color.setHex(0xffffff);
    keyLight.intensity = 1.38;
  } else {
    // Clean, balanced Rat NCAP paper-demo studio backdrop
    scene.background = new THREE.Color(0xdbe3ee);
    scene.fog = new THREE.Fog(0xdbe3ee, 15, 36);
    hemiLight.color.setHex(0xf6f9ff);
    hemiLight.groundColor.setHex(0x6b7788);
    hemiLight.intensity = 0.72;
    keyLight.color.setHex(0xfffaf2);
    keyLight.intensity = 1.48;
  }
}
applyLightingPreset(INITIAL_LIGHTING);

// ---------------------------------------------------------------------------
// Paper-Demo Checkered Lab Floor, Treadmill & Bumpy Heightfield Arenas
// ---------------------------------------------------------------------------
function createFloorTexture(arenaId) {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext('2d');
  const isTreadmill = arenaId === 'treadmill';
  const isBumpy = arenaId === 'bumpy_terrain';

  if (isTreadmill) {
    // Paper-demo clean blue-gray checkered studio plane (classic MuJoCo/dm_control look!)
    const tile = 128;
    for (let y = 0; y < 512; y += tile) {
      for (let x = 0; x < 512; x += tile) {
        const even = ((x / tile) + (y / tile)) % 2 === 0;
        ctx.fillStyle = even ? '#90a4bc' : '#b7c7da';
        ctx.fillRect(x, y, tile, tile);
      }
    }
    ctx.strokeStyle = 'rgba(30, 41, 59, 0.16)';
    ctx.lineWidth = 2;
    for (let p = 0; p <= 512; p += 128) {
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 512); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(512, p); ctx.stroke();
    }
  } else if (isBumpy) {
    const tile = 128;
    for (let y = 0; y < 512; y += tile) {
      for (let x = 0; x < 512; x += tile) {
        const even = ((x / tile) + (y / tile)) % 2 === 0;
        ctx.fillStyle = even ? '#899f83' : '#b0c4aa';
        ctx.fillRect(x, y, tile, tile);
      }
    }
  } else {
    ctx.fillStyle = '#bcc4cf';
    ctx.fillRect(0, 0, 512, 512);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.16)';
    ctx.lineWidth = 2;
    for (let p = 0; p <= 512; p += 64) {
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, 512); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(512, p); ctx.stroke();
    }
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(9, 9);
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy?.() || 8;
  return tex;
}

const arenaGroup = new THREE.Group();
scene.add(arenaGroup);
let floorMesh = null;
let floorTex = null;
let activeArena = INITIAL_ARENA;
let treadmillScroll = 0;

function rebuildArena(arenaId, bumpCm = 0) {
  activeArena = String(arenaId || 'treadmill').toLowerCase();
  while (arenaGroup.children.length > 0) {
    const ch = arenaGroup.children.pop();
    ch.traverse?.(node => {
      node.geometry?.dispose?.();
      if (node.material) {
        (Array.isArray(node.material) ? node.material : [node.material]).forEach(m => m.dispose?.());
      }
    });
  }

  const bumpM = Math.max(0, Number(bumpCm) || 0) * 0.01;
  const size = 24;
  const segs = bumpM > 0.0005 ? 110 : 16;
  const geo = new THREE.PlaneGeometry(size, size, segs, segs);
  geo.rotateX(-Math.PI / 2);

  if (bumpM > 0.0005) {
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i) / WORLD_SCALE;
      const wz = pos.getZ(i) / WORLD_SCALE;
      const hy = computeBumpyTerrainHeight(wx, wz, bumpM) * WORLD_SCALE;
      pos.setY(i, hy);
    }
    geo.computeVertexNormals();
  }

  floorTex = createFloorTexture(activeArena);
  const floorMat = new THREE.MeshStandardMaterial({
    map: floorTex,
    roughness: 0.86,
    metalness: 0.03,
  });
  floorMesh = new THREE.Mesh(geo, floorMat);
  floorMesh.receiveShadow = true;
  arenaGroup.add(floorMesh);

  const wallMat = new THREE.MeshPhysicalMaterial({
    color: 0xdce4ee,
    roughness: 0.22,
    metalness: 0.04,
    transparent: true,
    opacity: 0.26,
  });

  if (activeArena === 'open_field') {
    const half = 6.2;
    const h = 1.6;
    const thick = 0.12;
    const specs = [
      [0, h * 0.5, -half, half * 2, h, thick],
      [0, h * 0.5, half, half * 2, h, thick],
      [-half, h * 0.5, 0, thick, h, half * 2],
      [half, h * 0.5, 0, thick, h, half * 2],
    ];
    for (const [x, y, z, w, hh, d] of specs) {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(w, hh, d), wallMat);
      wall.position.set(x, y, z);
      arenaGroup.add(wall);
    }
  } else if (activeArena === 'circular_open_field') {
    const wall = new THREE.Mesh(new THREE.CylinderGeometry(6.2, 6.2, 1.6, 48, 1, true), wallMat);
    wall.position.y = 0.8;
    wall.material.side = THREE.DoubleSide;
    arenaGroup.add(wall);
  } else if (activeArena === 'elevated_plus_maze') {
    const armMat = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.78 });
    const arm1 = new THREE.Mesh(new THREE.BoxGeometry(11.0, 0.14, 1.8), armMat);
    arm1.position.y = -0.07;
    arm1.receiveShadow = true;
    const arm2 = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.14, 11.0), armMat);
    arm2.position.y = -0.07;
    arm2.receiveShadow = true;
    arenaGroup.add(arm1, arm2);
  }
}

// ---------------------------------------------------------------------------
// Controller, Camera Presets & Live Spinal Oscilloscope
// ---------------------------------------------------------------------------
const controller = new RatNCAPController({
  behavior: INITIAL_SCENARIO,
  interface: INITIAL_INTERFACE,
  targetSpeed: INITIAL_TARGET_SPEED,
  ablation: INITIAL_ABLATION,
  ablationScale: INITIAL_ABLATION_SCALE,
  rgNoiseStd: INITIAL_RG_NOISE,
  terrainBumpCm: INITIAL_BUMP_CM,
  dt: 1 / DATASET_SIM_HZ,
  seed: DATASET_SEED,
});

rebuildArena(INITIAL_ARENA, controller.terrainBumpCm);

const uiState = {
  paused: false,
  cameraView: INITIAL_CAMERA_VIEW,
  showMuscleEffort: false,
  showKeypoints: false,
  v0Scale: INITIAL_SCENARIO === 'v0_ablation_bound' ? 0.0 : (INITIAL_ABLATION.startsWith('v0') ? INITIAL_ABLATION_SCALE : 1.0),
  v3Scale: INITIAL_SCENARIO === 'v3_ablation_trot' ? 0.0 : (INITIAL_ABLATION.startsWith('v3') ? INITIAL_ABLATION_SCALE : 1.0),
};

let rat = null;

function applyCameraPreset(viewName, immediate = true) {
  uiState.cameraView = String(viewName || 'oblique').toLowerCase();
  const basePos = rat ? rat.root.position : new THREE.Vector3(0, 0, 0);
  // Frame the full 33 cm rat (nose +1.2 to tail -2.1) cleanly in view.
  // In interactive mode, shift look target slightly right so the rat sits centered left of the HUD panel.
  const hudShiftX = DATASET_CAPTURE ? -0.35 : 0.25;
  const target = basePos.clone().add(new THREE.Vector3(hudShiftX, 0.44, 0.0));
  const offsets = {
    track_right: new THREE.Vector3(0.0, 0.72, 4.95),
    track_left: new THREE.Vector3(0.0, 0.72, -4.95),
    track_front: new THREE.Vector3(4.2, 0.85, 0.45),
    track_back: new THREE.Vector3(-4.8, 1.05, 0.35),
    track_top: new THREE.Vector3(-0.15, 5.4, 0.04),
    top_down: new THREE.Vector3(-0.15, 5.4, 0.04),
    oblique: new THREE.Vector3(2.85, 1.75, 4.15),
    ventral: new THREE.Vector3(0.1, 0.32, 4.35),
    egocentric: new THREE.Vector3(1.25, 0.72, 0.0),
  };
  const offset = offsets[uiState.cameraView] || offsets.oblique;
  if (immediate) {
    camera.position.copy(target).add(offset);
    controls.target.copy(target);
    controls.update();
  }
}

const TRACE_LEN = 120;
const traceHistory = {
  flx_FL: new Array(TRACE_LEN).fill(0),
  flx_FR: new Array(TRACE_LEN).fill(0),
  flx_HL: new Array(TRACE_LEN).fill(0),
  flx_HR: new Array(TRACE_LEN).fill(0),
  v0d: new Array(TRACE_LEN).fill(0),
  v3f: new Array(TRACE_LEN).fill(0),
  contact_FL: new Array(TRACE_LEN).fill(1),
  contact_FR: new Array(TRACE_LEN).fill(1),
  contact_HL: new Array(TRACE_LEN).fill(1),
  contact_HR: new Array(TRACE_LEN).fill(1),
};

function pushTrace(state) {
  const push = (arr, val) => {
    arr.push(val);
    if (arr.length > TRACE_LEN) arr.shift();
  };
  push(traceHistory.flx_FL, state.rg.flx.FL);
  push(traceHistory.flx_FR, state.rg.flx.FR);
  push(traceHistory.flx_HL, state.rg.flx.HL);
  push(traceHistory.flx_HR, state.rg.flx.HR);
  const v0dAvg = 0.25 * (
    state.rg.interneurons.cross_v0d_FL + state.rg.interneurons.cross_v0d_FR
    + state.rg.interneurons.cross_v0d_HL + state.rg.interneurons.cross_v0d_HR
  );
  const v3fAvg = 0.25 * (
    state.rg.interneurons.cross_v3f_FL + state.rg.interneurons.cross_v3f_FR
    + state.rg.interneurons.cross_v3f_HL + state.rg.interneurons.cross_v3f_HR
  );
  push(traceHistory.v0d, v0dAvg);
  push(traceHistory.v3f, v3fAvg);
  push(traceHistory.contact_FL, state.footContacts.FL);
  push(traceHistory.contact_FR, state.footContacts.FR);
  push(traceHistory.contact_HL, state.footContacts.HL);
  push(traceHistory.contact_HR, state.footContacts.HR);
}

function drawSpinalMonitor() {
  const scope = document.getElementById('rat-spinal-scope');
  if (!scope) return;
  const ctx = scope.getContext('2d');
  const w = scope.width;
  const h = scope.height;
  ctx.fillStyle = '#0c1220';
  ctx.fillRect(0, 0, w, h);

  const topH = Math.floor(h * 0.56);
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
  ctx.beginPath();
  ctx.moveTo(0, topH);
  ctx.lineTo(w, topH);
  ctx.stroke();

  const series = [
    ['flx_FL', '#38bdf8'],
    ['flx_FR', '#34d399'],
    ['flx_HL', '#f59e0b'],
    ['flx_HR', '#f472b6'],
    ['v0d', '#a78bfa'],
    ['v3f', '#fb7185'],
  ];
  for (const [key, color] of series) {
    const arr = traceHistory[key];
    ctx.strokeStyle = color;
    ctx.lineWidth = key.startsWith('v') ? 1.15 : 1.5;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = (i / (TRACE_LEN - 1)) * w;
      const y = topH - 4 - arr[i] * (topH - 10);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const limbs = [
    ['FL', '#38bdf8'],
    ['FR', '#34d399'],
    ['HL', '#f59e0b'],
    ['HR', '#f472b6'],
  ];
  const rowH = (h - topH - 6) / 4;
  ctx.font = '9px monospace';
  for (let r = 0; r < limbs.length; r++) {
    const [limb, color] = limbs[r];
    const y0 = topH + 4 + r * rowH;
    const arr = traceHistory[`contact_${limb}`];
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(limb, 2, y0 + rowH - 2);
    ctx.fillStyle = color;
    const xStart = 20;
    const plotW = w - xStart - 2;
    const cellW = plotW / TRACE_LEN;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > 0.5) {
        ctx.fillRect(xStart + i * cellW, y0 + 1, Math.max(1.2, cellW), Math.max(3, rowH - 3));
      }
    }
  }
}

function updateHudReadout(state) {
  const setText = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  setText('hud-gait', state.gait.toUpperCase());
  setText('hud-speed', `${state.forwardSpeedMps.toFixed(2)} m/s`);
  setText('hud-freq', `${state.stepFrequencyHz.toFixed(2)} Hz`);
  setText('hud-step-len', `${(state.stepLengthM * 100).toFixed(1)} cm`);
  setText('hud-duty', `${(state.dutyFactor * 100).toFixed(0)}%`);
  setText(
    'hud-power',
    `+${state.power.positivePowerW.toFixed(3)} / ${state.power.negativePowerW.toFixed(3)} W`
  );
  setText(
    'hud-phases',
    `FL ${state.phaseOffsetsRelFR.FL.toFixed(2)} · HL ${state.phaseOffsetsRelFR.HL.toFixed(2)} · HR ${state.phaseOffsetsRelFR.HR.toFixed(2)}`
  );
  setText('rat-gait-pill', state.gait.toUpperCase());
  setText('rat-speed-val', `${state.targetSpeedMps.toFixed(2)} m/s`);
  setText('rat-v0-val', `${uiState.v0Scale.toFixed(2)}×`);
  setText('rat-v3-val', `${uiState.v3Scale.toFixed(2)}×`);
  setText('rat-noise-val', `σ=${controller.rg.rgNoiseStd.toFixed(2)}`);
  setText('rat-bump-val', `${controller.terrainBumpCm.toFixed(1)} cm`);
}

function wireControls() {
  const bind = (id, ev, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener(ev, fn);
    return el;
  };

  const panel = document.getElementById('rat-ncap-panel');
  const collapseBtn = bind('rat-collapse-btn', 'click', () => {
    if (!panel) return;
    panel.classList.toggle('collapsed');
    collapseBtn.textContent = panel.classList.contains('collapsed') ? '+' : '−';
  });

  const speedSlider = bind('rat-speed-slider', 'input', e => {
    controller.autoRamp = false;
    controller.targetSpeed = Number.parseFloat(e.target.value) || 0;
    const rampChk = document.getElementById('rat-auto-ramp');
    if (rampChk) rampChk.checked = false;
  });
  if (speedSlider) speedSlider.value = String(controller.targetSpeed);

  const rampCheck = bind('rat-auto-ramp', 'change', e => {
    controller.autoRamp = Boolean(e.target.checked);
  });
  if (rampCheck) rampCheck.checked = controller.autoRamp;

  const behaviorSel = bind('rat-behavior-select', 'change', e => {
    const beh = e.target.value;
    controller.behavior = beh;
    controller.autoRamp = beh === 'gait_transition';
    controller.targetSpeed = defaultSpeedForBehavior(beh);
    if (speedSlider) speedSlider.value = String(controller.targetSpeed);
    if (rampCheck) rampCheck.checked = controller.autoRamp;

    if (beh === 'v0_ablation_bound') {
      uiState.v0Scale = 0.0;
      uiState.v3Scale = 1.0;
    } else if (beh === 'v3_ablation_trot') {
      uiState.v0Scale = 1.0;
      uiState.v3Scale = 0.0;
    } else {
      uiState.v0Scale = 1.0;
      uiState.v3Scale = 1.0;
    }
    controller.rg.setPopulationScales({ v0Scale: uiState.v0Scale, v3Scale: uiState.v3Scale });
    const v0s = document.getElementById('rat-v0-slider');
    const v3s = document.getElementById('rat-v3-slider');
    if (v0s) v0s.value = String(uiState.v0Scale);
    if (v3s) v3s.value = String(uiState.v3Scale);

    if (beh === 'bumpy_terrain') {
      controller.terrainBumpCm = Math.max(2.2, controller.terrainBumpCm);
      const bs = document.getElementById('rat-bump-slider');
      if (bs) bs.value = String(controller.terrainBumpCm);
      rebuildArena('bumpy_terrain', controller.terrainBumpCm);
      const as = document.getElementById('rat-arena-select');
      if (as) as.value = 'bumpy_terrain';
    }
  });
  if (behaviorSel) behaviorSel.value = controller.behavior;

  const ifaceSel = bind('rat-interface-select', 'change', e => {
    controller.interface.setInterfaceType(e.target.value);
  });
  if (ifaceSel) ifaceSel.value = controller.interface.interfaceType;

  const coatSel = bind('rat-coat-select', 'change', e => {
    rat?.setCoat(e.target.value);
  });
  if (coatSel) coatSel.value = INITIAL_COAT;

  const arenaSel = bind('rat-arena-select', 'change', e => {
    const id = e.target.value;
    if (id === 'bumpy_terrain' && controller.terrainBumpCm < 0.5) {
      controller.terrainBumpCm = 2.2;
      const bs = document.getElementById('rat-bump-slider');
      if (bs) bs.value = '2.2';
    }
    rebuildArena(id, controller.terrainBumpCm);
  });
  if (arenaSel) arenaSel.value = activeArena;

  const camSel = bind('rat-camera-select', 'change', e => {
    applyCameraPreset(e.target.value, true);
  });
  if (camSel) camSel.value = uiState.cameraView;

  const v0Slider = bind('rat-v0-slider', 'input', e => {
    uiState.v0Scale = Number.parseFloat(e.target.value);
    controller.rg.setPopulationScales({ v0Scale: uiState.v0Scale, v3Scale: uiState.v3Scale });
  });
  if (v0Slider) v0Slider.value = String(uiState.v0Scale);

  const v3Slider = bind('rat-v3-slider', 'input', e => {
    uiState.v3Scale = Number.parseFloat(e.target.value);
    controller.rg.setPopulationScales({ v0Scale: uiState.v0Scale, v3Scale: uiState.v3Scale });
  });
  if (v3Slider) v3Slider.value = String(uiState.v3Scale);

  bind('rat-ablate-v0-btn', 'click', () => {
    uiState.v0Scale = uiState.v0Scale > 0.05 ? 0.0 : 1.0;
    if (v0Slider) v0Slider.value = String(uiState.v0Scale);
    controller.rg.setPopulationScales({ v0Scale: uiState.v0Scale, v3Scale: uiState.v3Scale });
  });

  bind('rat-ablate-v3-btn', 'click', () => {
    uiState.v3Scale = uiState.v3Scale > 0.05 ? 0.0 : 1.0;
    if (v3Slider) v3Slider.value = String(uiState.v3Scale);
    controller.rg.setPopulationScales({ v0Scale: uiState.v0Scale, v3Scale: uiState.v3Scale });
  });

  const noiseSlider = bind('rat-noise-slider', 'input', e => {
    controller.rg.applyWeightNoise(Number.parseFloat(e.target.value) || 0, controller.seed);
  });
  if (noiseSlider) noiseSlider.value = String(controller.rg.rgNoiseStd);

  const bumpSlider = bind('rat-bump-slider', 'input', e => {
    controller.terrainBumpCm = Number.parseFloat(e.target.value) || 0;
    rebuildArena(activeArena, controller.terrainBumpCm);
  });
  if (bumpSlider) bumpSlider.value = String(controller.terrainBumpCm);

  bind('rat-lock-lumbar', 'change', e => {
    controller.lockLumbar = Boolean(e.target.checked);
  });
  bind('rat-muscle-vis', 'change', e => {
    uiState.showMuscleEffort = Boolean(e.target.checked);
  });
  bind('rat-kp-vis', 'change', e => {
    uiState.showKeypoints = Boolean(e.target.checked);
  });
  const pauseBtn = bind('rat-pause-btn', 'click', () => {
    uiState.paused = !uiState.paused;
    if (pauseBtn) pauseBtn.textContent = uiState.paused ? 'Resume' : 'Pause';
  });
  bind('rat-reset-btn', 'click', () => {
    controller.reset();
    uiState.v0Scale = 1.0;
    uiState.v3Scale = 1.0;
    controller.rg.setPopulationScales({ v0Scale: 1.0, v3Scale: 1.0 });
    controller.rg.applyWeightNoise(0.0, controller.seed);
    if (v0Slider) v0Slider.value = '1';
    if (v3Slider) v3Slider.value = '1';
    if (noiseSlider) noiseSlider.value = '0';
  });
}

// ---------------------------------------------------------------------------
// Deterministic Dataset Capture API (`window.annolidPoseDataset`) for Sim2Real
// ---------------------------------------------------------------------------
const SEGMENTATION_FINE_PARTS = Object.freeze({
  1: 'head', 2: 'snout', 3: 'left_ear', 4: 'right_ear', 5: 'neck',
  6: 'torso', 7: 'left_forelimb', 8: 'right_forelimb',
  9: 'left_hindlimb', 10: 'right_hindlimb', 11: 'tail',
});
const SEGMENTATION_COARSE_PARTS = Object.freeze({
  1: 'head', 2: 'torso', 3: 'limbs', 4: 'tail',
});

let datasetFrameIndex = 0;
let datasetInitialized = false;
let lastState = null;
let segRenderTarget = null;
let segPixels = null;

function advanceDatasetFrame(targetFrame) {
  const parsed = Math.max(0, Math.min(10000, Number.parseInt(targetFrame, 10) || 0));
  if (parsed < datasetFrameIndex) {
    throw new RangeError('Dataset capture frames must advance monotonically.');
  }
  const dt = 1 / DATASET_SIM_HZ;
  if (!datasetInitialized) {
    lastState = controller.step(dt);
    rat.applyControllerState(lastState, { arenaId: activeArena, bumpCm: controller.terrainBumpCm });
    datasetInitialized = true;
  }
  const targetSteps = Math.round(parsed * DATASET_SIM_HZ / DATASET_CAPTURE_FPS);
  const currentSteps = Math.round(datasetFrameIndex * DATASET_SIM_HZ / DATASET_CAPTURE_FPS);
  for (let s = 0; s < targetSteps - currentSteps; s++) {
    lastState = controller.step(dt);
    if (activeArena === 'treadmill' && floorTex) {
      treadmillScroll = (treadmillScroll + lastState.forwardSpeedMps * dt * 2.2) % 1.0;
      floorTex.offset.x = treadmillScroll;
    }
    rat.applyControllerState(lastState, { arenaId: activeArena, bumpCm: controller.terrainBumpCm });
  }
  datasetFrameIndex = parsed;
  applyCameraPreset(uiState.cameraView, true);
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
}

function projectWorldToScreen(worldPos, width, height) {
  const v = worldPos.clone().project(camera);
  const x = (v.x * 0.5 + 0.5) * width;
  const y = (-v.y * 0.5 + 0.5) * height;
  const inBounds = v.z >= -1 && v.z <= 1 && x >= 0 && x <= width && y >= 0 && y <= height;
  return { x, y, visible: inBounds ? 2 : 0 };
}

function buildDatasetFramePayload() {
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const width = Math.max(1, Math.round(canvas.getBoundingClientRect().width || window.innerWidth));
  const height = Math.max(1, Math.round(canvas.getBoundingClientRect().height || window.innerHeight));
  const wp = new THREE.Vector3();
  const keypoints = {};
  const keypoints3d = {};
  let minX = width; let minY = height; let maxX = 0; let maxY = 0;
  for (const name of RAT_KEYPOINTS) {
    const node = rat.trackingNodes[name];
    if (!node) continue;
    node.getWorldPosition(wp);
    const proj = projectWorldToScreen(wp, width, height);
    keypoints[name] = [Number(proj.x.toFixed(2)), Number(proj.y.toFixed(2)), proj.visible];
    keypoints3d[name] = [
      Number((wp.x / WORLD_SCALE).toFixed(5)),
      Number((wp.y / WORLD_SCALE).toFixed(5)),
      Number((wp.z / WORLD_SCALE).toFixed(5)),
    ];
    if (proj.visible > 0) {
      minX = Math.min(minX, proj.x);
      minY = Math.min(minY, proj.y);
      maxX = Math.max(maxX, proj.x);
      maxY = Math.max(maxY, proj.y);
    }
  }
  const pad = Math.max(10, Math.min(width, height) * 0.025);
  const bbox = maxX > minX
    ? [
      Number(Math.max(0, minX - pad).toFixed(2)),
      Number(Math.max(0, minY - pad).toFixed(2)),
      Number(Math.min(width, maxX + pad).toFixed(2)),
      Number(Math.min(height, maxY + pad).toFixed(2)),
    ]
    : [0, 0, 0, 0];

  return {
    frame_id: datasetFrameIndex,
    width,
    height,
    species: 'rat',
    arena: activeArena,
    camera_view: uiState.cameraView,
    interface: controller.interface.interfaceType,
    gait: lastState ? lastState.gait : 'stand',
    subjects: [
      {
        track_id: 1,
        label: 'rat_1',
        coat: rat.coatId,
        bbox,
        keypoints,
        keypoints_3d: keypoints3d,
        gait: lastState ? lastState.gait : 'stand',
        speed_mps: lastState ? Number(lastState.forwardSpeedMps.toFixed(4)) : 0,
      },
    ],
  };
}

function captureDatasetSegmentationMasks() {
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const width = Math.max(1, Math.round(canvas.getBoundingClientRect().width || window.innerWidth));
  const height = Math.max(1, Math.round(canvas.getBoundingClientRect().height || window.innerHeight));

  if (!segRenderTarget || segRenderTarget.width !== width || segRenderTarget.height !== height) {
    segRenderTarget?.dispose?.();
    segRenderTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      colorSpace: THREE.NoColorSpace,
      depthBuffer: true,
      stencilBuffer: false,
      samples: 0,
    });
    segPixels = new Uint8Array(width * height * 4);
  }

  const savedArenaVis = arenaGroup.visible;
  const savedShadowVis = rat.contactShadow.visible;
  const savedKpVis = rat.keypointMarkers.visible;
  const savedSkinMat = rat.skinnedMesh.material;
  const savedMuscleVis = {};
  for (const [k, sp] of Object.entries(rat.muscleMarkers)) {
    savedMuscleVis[k] = sp.visible;
    sp.visible = false;
  }

  arenaGroup.visible = false;
  rat.contactShadow.visible = false;
  rat.keypointMarkers.visible = false;
  rat.skinnedMesh.material = rat.segmentationMaterial;

  const prevTarget = renderer.getRenderTarget();
  const prevBg = scene.background;
  const prevFog = scene.fog;
  const prevTone = renderer.toneMapping;
  const prevShadow = renderer.shadowMap.enabled;
  try {
    renderer.shadowMap.enabled = false;
    renderer.toneMapping = THREE.NoToneMapping;
    scene.background = null;
    scene.fog = null;
    renderer.setRenderTarget(segRenderTarget);
    renderer.setClearColor(0x000000, 1.0);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(segRenderTarget, 0, 0, width, height, segPixels);
  } finally {
    renderer.setRenderTarget(prevTarget);
    scene.background = prevBg;
    scene.fog = prevFog;
    renderer.toneMapping = prevTone;
    renderer.shadowMap.enabled = prevShadow;
    arenaGroup.visible = savedArenaVis;
    rat.contactShadow.visible = savedShadowVis;
    rat.keypointMarkers.visible = savedKpVis;
    rat.skinnedMesh.material = savedSkinMat;
    for (const [k, vis] of Object.entries(savedMuscleVis)) {
      rat.muscleMarkers[k].visible = vis;
    }
  }

  const rle = [];
  let curR = -1; let curG = -1; let curB = -1; let runLen = 0;
  for (let y = 0; y < height; y++) {
    const glRow = (height - 1 - y) * width * 4;
    for (let x = 0; x < width; x++) {
      const idx = glRow + x * 4;
      const r = segPixels[idx];
      const g = segPixels[idx + 1];
      const b = segPixels[idx + 2];
      if (r === curR && g === curG && b === curB) {
        runLen += 1;
      } else {
        if (runLen > 0) rle.push(runLen, curR, curG, curB);
        curR = r; curG = g; curB = b; runLen = 1;
      }
    }
  }
  if (runLen > 0) rle.push(runLen, curR, curG, curB);

  return {
    frame_id: datasetFrameIndex,
    width,
    height,
    rle,
    fine_parts: SEGMENTATION_FINE_PARTS,
    coarse_parts: SEGMENTATION_COARSE_PARTS,
    subjects: [{ track_id: 1, coat: rat.coatId, label: 'rat_1' }],
  };
}

// ---------------------------------------------------------------------------
// Async Boot: Load Authentic 41-Bone Rat v2 GLB & Start Loop
// ---------------------------------------------------------------------------
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / Math.max(1, window.innerHeight);
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

wireControls();

const clock = new THREE.Clock();
let simAccum = 0;

function animate() {
  if (!DATASET_CAPTURE) {
    requestAnimationFrame(animate);
    const frameDt = Math.min(clock.getDelta(), 0.1);
    if (!uiState.paused && !document.hidden) {
      simAccum += frameDt;
      const fixedDt = 1 / DATASET_SIM_HZ;
      let steps = 0;
      while (simAccum >= fixedDt && steps < 4) {
        lastState = controller.step(fixedDt);
        pushTrace(lastState);
        if (activeArena === 'treadmill' && floorTex) {
          treadmillScroll = (treadmillScroll + lastState.forwardSpeedMps * fixedDt * 2.2) % 1.0;
          floorTex.offset.x = treadmillScroll;
        }
        simAccum -= fixedDt;
        steps += 1;
      }
    }
    if (!lastState) {
      lastState = controller.step(1 / DATASET_SIM_HZ);
      pushTrace(lastState);
    }
    rat.applyControllerState(lastState, {
      arenaId: activeArena,
      bumpCm: controller.terrainBumpCm,
      showMuscleEffort: uiState.showMuscleEffort,
      showKeypoints: uiState.showKeypoints,
    });
    if (activeArena !== 'treadmill') {
      controls.target.lerp(rat.root.position.clone().add(new THREE.Vector3(-0.22, 0.44, 0)), 0.08);
    }
    updateHudReadout(lastState);
    drawSpinalMonitor();
  }
  controls.update();
  renderer.render(scene, camera);
  if (DATASET_CAPTURE) canvas.dataset.datasetReady = 'true';
  canvas.dataset.sceneReady = 'true';
  const statusEl = document.getElementById('scene-status');
  if (statusEl) statusEl.textContent = '';
}

async function initScene() {
  const statusEl = document.getElementById('scene-status');
  if (statusEl) statusEl.textContent = 'Loading Rat v2 41-bone skinned model…';

  rat = await loadRatV2SkinnedModel(THREE, annolidShaders, {
    coatId: INITIAL_COAT,
    worldScale: WORLD_SCALE,
    maxAnisotropy: renderer.capabilities.getMaxAnisotropy?.() || 8,
  });
  scene.add(rat.sceneGroup);
  applyCameraPreset(INITIAL_CAMERA_VIEW, true);

  window.annolidRatNCAP = Object.freeze({
    spec: RAT_NCAP_SPEC,
    getState: () => lastState,
    setSpeed: v => {
      controller.autoRamp = false;
      controller.targetSpeed = Math.max(0, Math.min(1, Number(v) || 0));
    },
    setInterface: iface => controller.interface.setInterfaceType(iface),
    setCoat: coat => rat.setCoat(coat),
    setAblation: (pop, scale = 0) => controller.rg.setAblation(pop, scale),
    setRgNoise: sigma => controller.rg.applyWeightNoise(sigma, controller.seed),
    setArena: (arenaId, bumpCm = controller.terrainBumpCm) => {
      controller.terrainBumpCm = Math.max(0, Number(bumpCm) || 0);
      rebuildArena(arenaId, controller.terrainBumpCm);
    },
    setCameraView: view => applyCameraPreset(view, true),
  });

  if (DATASET_CAPTURE) {
    advanceDatasetFrame(DATASET_FRAME_INDEX);
    window.annolidPoseDataset = Object.freeze({
      getFrame: buildDatasetFramePayload,
      setFrame: targetFrame => {
        advanceDatasetFrame(targetFrame);
        renderer.render(scene, camera);
        canvas.dataset.datasetReady = 'true';
        return buildDatasetFramePayload();
      },
      captureSegmentationMasks: captureDatasetSegmentationMasks,
    });
  }

  animate();
}

initScene().catch(err => {
  console.error('Failed to initialize Rat NCAP 3D scene:', err);
  const statusEl = document.getElementById('scene-status');
  if (statusEl) {
    statusEl.textContent = `Could not load Rat v2 3D model: ${err?.message || err}`;
  }
});
