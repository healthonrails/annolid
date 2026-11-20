import {
  THREE,
  OrbitControls,
  RoomEnvironment,
  GUI
} from './annolid_threejs_runtime.js';

import { AnnolidShaders, SPINE_DEFORM_GLSL } from './annolid_shaders.js';
import { createMouseModel } from './two_mice_model.js';
import {
  createSeededRandom,
  FixedStepClock,
  wallNormalForce,
  STANCE_CONTACT_LIMITS,
  evaluateSpinePoint,
  computeOrientedWallMargin,
} from './two_mice_simulation.js';
import {
  MOUSE_JOINT_SPEC,
  MouseMotorController,
  getMouseActionSpec,
  sanitizeMotorCommand,
} from './two_mice_control.js';

const datasetQuery = new URLSearchParams(window.location.search);
const DATASET_CAPTURE = datasetQuery.get('dataset') === '1';
const DATASET_FRAME_INDEX = Math.max(0, Math.min(10000, Number.parseInt(datasetQuery.get('frame') || '0', 10) || 0));
const DATASET_CAPTURE_FPS = Math.max(1, Math.min(30, Number.parseInt(datasetQuery.get('fps') || '5', 10) || 5));
const DATASET_SEED = Number.parseInt(datasetQuery.get('seed') || '20260710', 10) >>> 0;
const DATASET_SIMULATION_HZ = 30;
const CONTACT_QA = datasetQuery.get('contact_qa');
const REARING_QA = datasetQuery.get('rearing_qa');
const GROOMING_QA = datasetQuery.get('grooming_qa');
const ANATOMY_QA = datasetQuery.get('anatomy_qa');
const DATASET_CAMERA_VIEW = datasetQuery.get('camera_view');
const DATASET_SCENARIO = datasetQuery.get('scenario');
const DATASET_COAT_PAIR = datasetQuery.get('coat_pair') || '';
const DATASET_LIGHTING = datasetQuery.get('lighting') || '';
const DATASET_OCCLUDER = datasetQuery.get('occluder') || '';
const DATASET_ARENA = datasetQuery.get('arena') || '';
const DATASET_HEADSTAGE = datasetQuery.get('headstage') || '';
const ANATOMY_QA_VIEWS = new Set([
  'ears', 'ears-front', 'torso-side', 'torso-front',
  'head-side', 'head-three-quarter', 'head-top', 'mouth-front',
]);
const IS_ANATOMY_QA = ANATOMY_QA_VIEWS.has(ANATOMY_QA);
const IS_TORSO_ANATOMY_QA = ANATOMY_QA === 'torso-side'
  || ANATOMY_QA === 'torso-front';

const simulationSeed = DATASET_CAPTURE || datasetQuery.has('seed')
  ? DATASET_SEED : (Date.now() >>> 0);
const random = createSeededRandom(simulationSeed);
const appearanceRandom = createSeededRandom(simulationSeed ^ 0xA53C91E7);
if (DATASET_CAPTURE) document.body.classList.add('dataset-capture');

// ═══════════════════════════════════════════════════════════════
// 1. CORE RENDERER SETUP
// ═══════════════════════════════════════════════════════════════
const canvas = document.querySelector('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
const gl = renderer.getContext();
const debugRendererInfo = gl?.getExtension?.('WEBGL_debug_renderer_info');
const glRendererString = String(
  (debugRendererInfo && gl?.getParameter?.(debugRendererInfo.UNMASKED_RENDERER_WEBGL))
  || gl?.getParameter?.(gl?.RENDERER)
  || ''
);
const IS_SOFTWARE_WEBGL = /SwiftShader|llvmpipe|Software|Basic Render Driver/i.test(glRendererString);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.92;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio((DATASET_CAPTURE || IS_SOFTWARE_WEBGL) ? 1 : Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd5d4cf);
scene.fog = new THREE.FogExp2(0xd5d4cf, 0.0042);

const camera = new THREE.PerspectiveCamera(34, innerWidth / innerHeight, 0.03, 500);
const cameraBasePosition = new THREE.Vector3(8.4, 5.4, 22.5);
const cameraTarget = new THREE.Vector3(0, 1.05, 0);
camera.position.copy(cameraBasePosition);

const controls = new OrbitControls(camera, canvas);
controls.enabled = !DATASET_CAPTURE;
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.copy(cameraTarget);
controls.maxPolarAngle = (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'bottom-up')
  ? Math.PI
  : (Math.PI / 2 - 0.02);
controls.minDistance = 3.5;
controls.maxDistance = 70;

// IBL environment (True HDR Studio Lighting)
const pmremGenerator = new THREE.PMREMGenerator(renderer);
pmremGenerator.compileEquirectangularShader();
const roomEnvironment = new RoomEnvironment();
scene.environment = pmremGenerator.fromScene(roomEnvironment).texture;
scene.environmentIntensity = 0.70;
roomEnvironment.dispose();
pmremGenerator.dispose();

// ═══════════════════════════════════════════════════════════════
// 2. PROPORTIONS SYSTEM (ratio-driven placement)
// ═══════════════════════════════════════════════════════════════
const annolidShaders = new AnnolidShaders(THREE);
const {
  PROP,
  tailRoughnessMap,
  tailNormalMap,
  earNormalMap,
  earRoughnessMap,
  earAlbedoMap,
  noseAlbedoMap,
  noseNormalMap,
  noseRoughnessMap,
  labFloorAlbedoMap,
  labFloorRoughnessMap,
  labFloorNormalMap,
  homeCageBeddingMaps,
  urineSpotTexture,
  shadowSmudgeTexture,
  fleshMatBase,
  eyeMat,
  corneaMat,
  createCoatMaterial,
  enhanceFurMaterial,
  deform,
  deformBody,
  deformHead,
  getHeadFeatureAnchors,
  deformCervicalCore,
  bodyGeo,
  headGeo,
  neckGeo,
  cervicalCoreGeo,
  rebuildSharedAnatomyGeometries,
  createForelimbChain,
  createHindlimbChain,
  createAnatomicalWhiskers,
  createSupraorbitalWhiskers,
  FUR_DETAIL,
  generateMultiLayerFur,
  buildMouseEar,
  createTailBaseFur,
  disposeSubject,
  disposeSharedResources,
} = createMouseModel(THREE, annolidShaders, appearanceRandom);

const ENCLOSURE_W = 36, ENCLOSURE_H = 9, ENCLOSURE_D = 26, ENCLOSURE_PAD = 1.6;
const ENCLOSURE_HALF_X = ENCLOSURE_W * 0.5 - ENCLOSURE_PAD;
const ENCLOSURE_HALF_Z = ENCLOSURE_D * 0.5 - ENCLOSURE_PAD;

// Contact shadow (elongated murine dual-zone AO footprint under torso + haunches)
function createContactShadow() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const ctx = c.getContext('2d');
  ctx.save();
  ctx.translate(64, 64);
  ctx.scale(0.64, 1.0);
  const core = ctx.createRadialGradient(0, 6, 3, 0, 2, 50);
  core.addColorStop(0.0, 'rgba(18,15,12,0.44)');
  core.addColorStop(0.38, 'rgba(20,17,14,0.26)');
  core.addColorStop(0.72, 'rgba(22,19,16,0.09)');
  core.addColorStop(1.0, 'rgba(22,19,16,0)');
  ctx.fillStyle = core;
  ctx.fillRect(-96, -64, 192, 128);
  ctx.restore();
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.92, depthWrite: false, depthTest: true });
  const geo = new THREE.PlaneGeometry(4.0, 5.6);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.018;
  mesh.renderOrder = 5;
  return mesh;
}

const POSE_CONNECTIONS = [
  ['snout', 'lear'], ['snout', 'rear'], ['lear', 'neck'], ['rear', 'neck'],
  ['neck', 'spine1'], ['spine1', 'spine2'], ['spine2', 'tailbase'],
  ['neck', 'lforepaw'], ['neck', 'rforepaw'],
  ['spine2', 'lhindpaw'], ['spine2', 'rhindpaw'],
  ['tailbase', 'tailmid'], ['tailmid', 'tailend']
];
const POSE_KEYPOINTS = Object.freeze([
  { id: 'snout', name: 'nose' },
  { id: 'lear', name: 'left_ear' },
  { id: 'rear', name: 'right_ear' },
  { id: 'neck', name: 'neck' },
  { id: 'spine1', name: 'spine_1' },
  { id: 'spine2', name: 'spine_2' },
  { id: 'tailbase', name: 'tail_base' },
  { id: 'lforepaw', name: 'left_forepaw' },
  { id: 'rforepaw', name: 'right_forepaw' },
  { id: 'lhindpaw', name: 'left_hindpaw' },
  { id: 'rhindpaw', name: 'right_hindpaw' },
  { id: 'tailmid', name: 'tail_mid' },
  { id: 'tailend', name: 'tail_tip' },
]);
const POSE_KEYPOINT_INDEX = new Map(
  POSE_KEYPOINTS.map(({ id }, index) => [id, index + 1])
);
const POSE_CATEGORY = Object.freeze({
  id: 1,
  name: 'mouse',
  supercategory: 'animal',
  keypoints: POSE_KEYPOINTS.map(({ name }) => name),
  skeleton: POSE_CONNECTIONS.map(([start, end]) => [
    POSE_KEYPOINT_INDEX.get(start),
    POSE_KEYPOINT_INDEX.get(end),
  ]),
});
const POSE_COLORS = {
  snout: 0xff0055, lear: 0xff5500, rear: 0xff5500, neck: 0x00ffaa,
  spine1: 0x00aaff, spine2: 0x0055ff, tailbase: 0xaa00ff,
  lforepaw: 0xaaff00, rforepaw: 0xaaff00, lhindpaw: 0x00ff00, rhindpaw: 0x00ff00,
  tailmid: 0xff00ff, tailend: 0xff00aa
};

// The mouse silhouette is elongated, so a single center-radius collision
// allows heads and rumps to cross. These discs follow the body heading and
// approximate the rump, abdomen, thorax, head, and snout independently.
const MOUSE_COLLISION_DISC_LAYOUT = Object.freeze([
  { offset: -1.75, radius: 1.18 },
  { offset: -0.35, radius: 1.32 },
  { offset: 1.15, radius: 1.18 },
  { offset: 2.45, radius: 0.92 },
  { offset: 3.55, radius: 0.52 },
]);

function getMouseCollisionDiscs(mouse) {
  const heading = mouse.heading.lengthSq() > 1e-8
    ? mouse.heading
    : new THREE.Vector2(Math.sin(mouse.mesh.rotation.y), Math.cos(mouse.mesh.rotation.y));
  const leftAxisX = heading.y;
  const leftAxisZ = -heading.x;
  const bodyScale = THREE.MathUtils.clamp(mouse.guiParams.bodyLength ?? 1.0, 0.65, 1.6);
  const snoutScale = THREE.MathUtils.clamp(mouse.guiParams.snoutLength ?? 1.0, 0.65, 1.6);
  const widthScale = THREE.MathUtils.clamp(mouse.guiParams.chonkiness ?? 1.0, 0.65, 1.6);
  return MOUSE_COLLISION_DISC_LAYOUT.map(({ offset, radius }) => {
    const headExtension = offset > 2.4 ? (snoutScale - 1.0) * 0.55 : 0.0;
    const sp = evaluateSpinePoint(0, 0, offset, mouse.spineFlex, PROP.bodyLength);
    const forwardDist = sp.z * bodyScale + headExtension;
    const lateralDist = sp.x * widthScale;
    return {
      x: mouse.mesh.position.x + heading.x * forwardDist + leftAxisX * lateralDist,
      z: mouse.mesh.position.z + heading.y * forwardDist + leftAxisZ * lateralDist,
      radius: radius * widthScale * sp.girthScale,
    };
  });
}

function findMouseBodyContact(mouseA, mouseB) {
  const discsA = getMouseCollisionDiscs(mouseA);
  const discsB = getMouseCollisionDiscs(mouseB);
  let best = { clearance: Infinity, nx: 1.0, nz: 0.0 };
  for (const discA of discsA) {
    for (const discB of discsB) {
      let dx = discA.x - discB.x;
      let dz = discA.z - discB.z;
      let distSq = dx * dx + dz * dz;
      if (distSq < 1e-8) {
        dx = mouseA.mesh.position.x - mouseB.mesh.position.x;
        dz = mouseA.mesh.position.z - mouseB.mesh.position.z;
        distSq = dx * dx + dz * dz;
      }
      if (distSq < 1e-8) {
        dx = -mouseA.heading.y;
        dz = mouseA.heading.x;
        distSq = 1.0;
      }
      const dist = Math.sqrt(distSq);
      const clearance = dist - discA.radius - discB.radius;
      if (clearance < best.clearance) {
        best = { clearance, nx: dx / dist, nz: dz / dist };
      }
    }
  }
  return best;
}

function resolveMouseBodyOverlap(mouseA, mouseB) {
  let touched = false;
  for (let pass = 0; pass < 4; pass++) {
    const contact = findMouseBodyContact(mouseA, mouseB);
    if (contact.clearance >= -0.001) break;
    touched = true;
    const overlap = -contact.clearance;
    const correction = Math.min((overlap + 0.004) * 0.50, 0.42);
    mouseA.mesh.position.x += contact.nx * correction;
    mouseA.mesh.position.z += contact.nz * correction;
    mouseB.mesh.position.x -= contact.nx * correction;
    mouseB.mesh.position.z -= contact.nz * correction;

    const relativeNormalSpeed = (
      (mouseA.velocity.x - mouseB.velocity.x) * contact.nx
      + (mouseA.velocity.y - mouseB.velocity.y) * contact.nz
    );
    if (relativeNormalSpeed < 0) {
      const impulse = Math.min(-relativeNormalSpeed * 0.58, 1.8);
      mouseA.velocity.x += contact.nx * impulse;
      mouseA.velocity.y += contact.nz * impulse;
      mouseB.velocity.x -= contact.nx * impulse;
      mouseB.velocity.y -= contact.nz * impulse;
    }
  }
  return { touched, contact: findMouseBodyContact(mouseA, mouseB) };
}

function resolveAllMouseBodyOverlaps(subjects) {
  for (let pass = 0; pass < 3; pass++) {
    let corrected = false;
    for (let i = 0; i < subjects.length; i++) {
      for (let j = i + 1; j < subjects.length; j++) {
        corrected = resolveMouseBodyOverlap(subjects[i], subjects[j]).touched || corrected;
      }
    }
    if (!corrected) break;
  }
}

function getMinimumMouseClearance(subjects) {
  let minimum = Infinity;
  for (let i = 0; i < subjects.length; i++) {
    for (let j = i + 1; j < subjects.length; j++) {
      minimum = Math.min(minimum, findMouseBodyContact(subjects[i], subjects[j]).clearance);
    }
  }
  return minimum;
}

function tailSurfaceRadiusAt(p, thickness = 1.0) {
  const fraction = THREE.MathUtils.clamp(p, 0, 1);
  const taper = Math.max(0.08, 1.0 - fraction * PROP.tailTaper);
  const rootBlend = 1.0 - THREE.MathUtils.smoothstep(fraction, 0.0, 0.20);
  return Math.max(0.008, 0.08 * taper * (1.0 + rootBlend * 0.45) * thickness);
}

function createDynamicTailTubeGeometry(segmentCount, radialSegments = 14) {
  const ringCount = segmentCount + 1;
  const ringStride = radialSegments + 1;
  const sideVertexCount = ringCount * ringStride;
  const capStride = radialSegments + 1;
  const startCapOffset = sideVertexCount;
  const endCapOffset = startCapOffset + capStride;
  const vertexCount = sideVertexCount + capStride * 2;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  colors.fill(1.0);
  const uvs = new Float32Array(vertexCount * 2);
  const indices = [];

  for (let ring = 0; ring < ringCount; ring++) {
    const v = ring / segmentCount;
    for (let side = 0; side <= radialSegments; side++) {
      const vertex = ring * ringStride + side;
      uvs[vertex * 2] = side / radialSegments;
      uvs[vertex * 2 + 1] = v;
    }
  }
  for (let ring = 0; ring < segmentCount; ring++) {
    const row = ring * ringStride;
    const nextRow = (ring + 1) * ringStride;
    for (let side = 0; side < radialSegments; side++) {
      const a = row + side;
      const b = row + side + 1;
      const c = nextRow + side;
      const d = nextRow + side + 1;
      indices.push(a, b, c, b, d, c);
    }
  }

  const startCenter = startCapOffset + radialSegments;
  const endCenter = endCapOffset + radialSegments;
  uvs[startCenter * 2] = uvs[startCenter * 2 + 1] = 0.5;
  uvs[endCenter * 2] = uvs[endCenter * 2 + 1] = 0.5;
  for (let side = 0; side < radialSegments; side++) {
    const nextSide = (side + 1) % radialSegments;
    const angle = side / radialSegments * Math.PI * 2;
    for (const offset of [startCapOffset, endCapOffset]) {
      const vertex = offset + side;
      uvs[vertex * 2] = 0.5 + Math.cos(angle) * 0.5;
      uvs[vertex * 2 + 1] = 0.5 + Math.sin(angle) * 0.5;
    }
    indices.push(startCenter, startCapOffset + nextSide, startCapOffset + side);
    indices.push(endCenter, endCapOffset + side, endCapOffset + nextSide);
  }

  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3);
  const normalAttribute = new THREE.BufferAttribute(normals, 3);
  const colorAttribute = new THREE.BufferAttribute(colors, 3);
  positionAttribute.setUsage(THREE.DynamicDrawUsage);
  normalAttribute.setUsage(THREE.DynamicDrawUsage);
  colorAttribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', positionAttribute);
  geometry.setAttribute('normal', normalAttribute);
  geometry.setAttribute('color', colorAttribute);
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.userData.tailTube = {
    segmentCount,
    radialSegments,
    ringStride,
    ringCount,
    startCapOffset,
    endCapOffset,
    startCenter,
    endCenter,
    tangents: new Float32Array(ringCount * 3),
    normals: new Float32Array(ringCount * 3),
    binormals: new Float32Array(ringCount * 3),
  };
  return geometry;
}

function updateDynamicTailTubeGeometry(geometry, points, thickness = 1.0) {
  const state = geometry.userData.tailTube;
  if (!state || points.length < state.ringCount) return;
  const positions = geometry.attributes.position.array;
  const vertexNormals = geometry.attributes.normal.array;
  const vertexColors = geometry.attributes.color.array;
  const {
    tangents,
    normals,
    binormals,
    radialSegments,
    ringStride,
    ringCount,
  } = state;

  for (let ring = 0; ring < ringCount; ring++) {
    const previous = points[Math.max(0, ring - 1)];
    const next = points[Math.min(ringCount - 1, ring + 1)];
    let tx = next.x - previous.x;
    let ty = next.y - previous.y;
    let tz = next.z - previous.z;
    let tangentLength = Math.hypot(tx, ty, tz);
    if (tangentLength < 1e-7 && ring > 0) {
      tx = tangents[(ring - 1) * 3];
      ty = tangents[(ring - 1) * 3 + 1];
      tz = tangents[(ring - 1) * 3 + 2];
      tangentLength = 1;
    } else if (tangentLength < 1e-7) {
      tx = 0; ty = 0; tz = -1;
      tangentLength = 1;
    }
    tx /= tangentLength; ty /= tangentLength; tz /= tangentLength;

    let nx;
    let ny;
    let nz;
    if (ring === 0) {
      const useXAxis = Math.abs(ty) > 0.90;
      nx = useXAxis ? 1 : 0;
      ny = useXAxis ? 0 : 1;
      nz = 0;
    } else {
      nx = normals[(ring - 1) * 3];
      ny = normals[(ring - 1) * 3 + 1];
      nz = normals[(ring - 1) * 3 + 2];
    }
    const normalAlongTangent = nx * tx + ny * ty + nz * tz;
    nx -= tx * normalAlongTangent;
    ny -= ty * normalAlongTangent;
    nz -= tz * normalAlongTangent;
    let normalLength = Math.hypot(nx, ny, nz);
    if (normalLength < 1e-7) {
      nx = Math.abs(ty) > 0.90 ? 1 : 0;
      ny = Math.abs(ty) > 0.90 ? 0 : 1;
      nz = 0;
      const fallbackAlongTangent = nx * tx + ny * ty + nz * tz;
      nx -= tx * fallbackAlongTangent;
      ny -= ty * fallbackAlongTangent;
      nz -= tz * fallbackAlongTangent;
      normalLength = Math.max(Math.hypot(nx, ny, nz), 1e-7);
    }
    nx /= normalLength; ny /= normalLength; nz /= normalLength;
    let bx = ty * nz - tz * ny;
    let by = tz * nx - tx * nz;
    let bz = tx * ny - ty * nx;
    const binormalLength = Math.max(Math.hypot(bx, by, bz), 1e-7);
    bx /= binormalLength; by /= binormalLength; bz /= binormalLength;
    nx = by * tz - bz * ty;
    ny = bz * tx - bx * tz;
    nz = bx * ty - by * tx;

    const frameOffset = ring * 3;
    tangents[frameOffset] = tx;
    tangents[frameOffset + 1] = ty;
    tangents[frameOffset + 2] = tz;
    normals[frameOffset] = nx;
    normals[frameOffset + 1] = ny;
    normals[frameOffset + 2] = nz;
    binormals[frameOffset] = bx;
    binormals[frameOffset + 1] = by;
    binormals[frameOffset + 2] = bz;

    const frac = ring / (ringCount - 1);
    const radius = tailSurfaceRadiusAt(frac, thickness);
    const tipWarmth = THREE.MathUtils.smoothstep(frac, 0.72, 1.0);
    for (let side = 0; side <= radialSegments; side++) {
      const angle = side / radialSegments * Math.PI * 2;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const rx = nx * cosine + bx * sine;
      const ry = ny * cosine + by * sine;
      const rz = nz * cosine + bz * sine;
      const vertexOffset = (ring * ringStride + side) * 3;
      positions[vertexOffset] = points[ring].x + rx * radius;
      positions[vertexOffset + 1] = points[ring].y + ry * radius;
      positions[vertexOffset + 2] = points[ring].z + rz * radius;
      vertexNormals[vertexOffset] = rx;
      vertexNormals[vertexOffset + 1] = ry;
      vertexNormals[vertexOffset + 2] = rz;
      // Murine dorsoventral tail countershading: darker dorsal scale band, paler ventral aspect
      const dorsalShade = THREE.MathUtils.clamp(0.96 - ry * 0.14, 0.80, 1.10);
      vertexColors[vertexOffset] = Math.min(1.14, dorsalShade + tipWarmth * 0.05);
      vertexColors[vertexOffset + 1] = dorsalShade - tipWarmth * 0.01;
      vertexColors[vertexOffset + 2] = dorsalShade - tipWarmth * 0.02;
    }
  }

  for (const cap of [
    { ring: 0, offset: state.startCapOffset, center: state.startCenter, sign: -1 },
    { ring: ringCount - 1, offset: state.endCapOffset, center: state.endCenter, sign: 1 },
  ]) {
    const frameOffset = cap.ring * 3;
    const centerOffset = cap.center * 3;
    positions[centerOffset] = points[cap.ring].x;
    positions[centerOffset + 1] = points[cap.ring].y;
    positions[centerOffset + 2] = points[cap.ring].z;
    vertexNormals[centerOffset] = tangents[frameOffset] * cap.sign;
    vertexNormals[centerOffset + 1] = tangents[frameOffset + 1] * cap.sign;
    vertexNormals[centerOffset + 2] = tangents[frameOffset + 2] * cap.sign;
    vertexColors[centerOffset] = 0.96;
    vertexColors[centerOffset + 1] = 0.95;
    vertexColors[centerOffset + 2] = 0.94;
    for (let side = 0; side < radialSegments; side++) {
      const sourceOffset = (cap.ring * ringStride + side) * 3;
      const targetOffset = (cap.offset + side) * 3;
      positions[targetOffset] = positions[sourceOffset];
      positions[targetOffset + 1] = positions[sourceOffset + 1];
      positions[targetOffset + 2] = positions[sourceOffset + 2];
      vertexNormals[targetOffset] = vertexNormals[centerOffset];
      vertexNormals[targetOffset + 1] = vertexNormals[centerOffset + 1];
      vertexNormals[targetOffset + 2] = vertexNormals[centerOffset + 2];
      vertexColors[targetOffset] = vertexColors[sourceOffset];
      vertexColors[targetOffset + 1] = vertexColors[sourceOffset + 1];
      vertexColors[targetOffset + 2] = vertexColors[sourceOffset + 2];
    }
  }

  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.normal.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
}

function getMinimumTailClearance(subjects) {
  let minimum = Infinity;
  for (const mouse of subjects) {
    if (Number.isFinite(mouse.minimumTailClearance)) {
      minimum = Math.min(minimum, mouse.minimumTailClearance);
    }
  }
  return minimum;
}

function getMaximumTailSegmentLengthError(subjects) {
  return Math.max(
    0,
    ...subjects.map(mouse => mouse.maximumTailSegmentLengthError || 0)
  );
}

function getMaximumTailBendAngle(subjects) {
  return Math.max(0, ...subjects.map(mouse => mouse.maximumTailBendAngle || 0));
}

function getMaximumFootSlip(subjects) {
  let maximum = 0;
  for (const mouse of subjects) {
    maximum = Math.max(maximum, mouse.maximumFootSlip || 0);
  }
  return maximum;
}

function getMaximumPawGroundError(subjects) {
  let maximum = 0;
  for (const mouse of subjects) {
    maximum = Math.max(maximum, mouse.maximumPawGroundError || 0);
  }
  return maximum;
}

function integrateRearAngularDynamics(state, dt, mass, effectiveLength) {
  const stepDt = THREE.MathUtils.clamp(dt, 0, 0.05);
  const poseFraction = THREE.MathUtils.clamp(
    state.angle / Math.max(state.maxAngle, 0.01),
    0,
    1
  );
  const contactQuality = state.mode === 'wall-supported'
    ? 1.0 - THREE.MathUtils.clamp(state.wallSupportError / 0.22, 0, 1)
    : 0;
  state.wallContactLoad = state.mode === 'wall-supported'
    ? THREE.MathUtils.smoothstep(poseFraction, 0.62, 0.92) * contactQuality
    : 0;
  state.balanceAngleCorrection = state.mode === 'unsupported'
    ? THREE.MathUtils.clamp(
      state.centerOfMassForwardOffset * 0.12,
      -0.10,
      0.10
    )
    : 0;
  const targetAngle = state.phase === 'settle'
    ? 0
    : THREE.MathUtils.clamp(
      state.maxAngle + state.balanceAngleCorrection,
      state.maxAngle * 0.88,
      state.maxAngle * 1.06
    );

  const centerOfMassLever = Math.max(0.9, effectiveLength * 0.46);
  const momentOfInertia = Math.max(
    0.055,
    mass * Math.pow(effectiveLength * 1.15, 2) / 3
  );
  const gravity = 9.81;
  const gravityTorque = mass * gravity * centerOfMassLever
    * Math.sin(state.angle);
  const wallReactionTorque = gravityTorque * state.wallContactLoad * 0.72;
  const muscleStiffness = state.phase === 'rise'
    ? 2.35
    : (state.phase === 'settle' ? 1.75 : 1.30);
  const angularDamping = state.phase === 'rise'
    ? 0.86
    : (state.phase === 'settle' ? 0.78 : 0.64);
  const targetGravityCompensation = mass * gravity * centerOfMassLever
    * Math.sin(targetAngle) * (1.0 - state.wallContactLoad * 0.72);
  const muscleTorque = muscleStiffness * (targetAngle - state.angle)
    + targetGravityCompensation;
  state.angularAcceleration = (
    muscleTorque + wallReactionTorque - gravityTorque
    - angularDamping * state.angularVelocity
  ) / momentOfInertia;
  state.angularVelocity = THREE.MathUtils.clamp(
    state.angularVelocity + state.angularAcceleration * stepDt,
    -3.2,
    3.2
  );
  state.angle += state.angularVelocity * stepDt;
  if (state.angle < 0) {
    state.angle = 0;
    state.angularVelocity = Math.max(0, state.angularVelocity) * 0.12;
  }
  const maximumAngle = state.maxAngle * 1.08;
  if (state.angle > maximumAngle) {
    state.angle = maximumAngle;
    state.angularVelocity = Math.min(0, state.angularVelocity) * 0.18;
  }
  state.gravityTorque = gravityTorque;
  state.muscleTorque = muscleTorque;
  state.wallReactionTorque = wallReactionTorque;
}

function resolveAllMouseContacts(subjects, dt = 0.016) {
  resolveAllMouseBodyOverlaps(subjects);
  for (const mouse of subjects) mouse.solveRearBalanceConstraints(dt);
  for (const mouse of subjects) mouse.solveFootPlantConstraints(dt);
  for (const mouse of subjects) mouse.solveRearSupportConstraints();
  for (const mouse of subjects) mouse.solveForelimbBehaviorConstraints();
  for (const mouse of subjects) mouse.updateTailGeometry(subjects, dt);
  for (const mouse of subjects) mouse.updateHeadstageTether?.();
  for (const mouse of subjects) mouse.updateTrackingGeometry();
}

// ═══════════════════════════════════════════════════════════════
// 10. EXPERT MOUSE CLASS (with new anatomy)
// ═══════════════════════════════════════════════════════════════
class ExpertMouse {
  constructor(params) {
    this.mesh = new THREE.Group();
    this.timeOffset = random() * 1000;
    this.locomotionTime = 0;
    this.guiParams = params;

    // Physics state
    this.velocity = new THREE.Vector2(0, 0);
    this.force = new THREE.Vector2(0, 0);
    this.heading = new THREE.Vector2(0, 1);
    this.mass = 0.025 + random() * 0.010;
    this.maxSpeed = params.maxSpeed * (0.90 + random() * 0.18);
    this.drag = 2.5;
    this.wanderAngle = random() * Math.PI * 2;
    this.wanderRate = 0.8 + random() * 0.6;
    this.wanderStrength = 2.5 + random() * 1.5;
    this.burstPhase = random() * Math.PI * 2;
    this.speedMult = 0.8 + random() * 0.5;
    this.dir = random() > 0.5 ? 1 : -1;
    this.wallFollowDir = random() > 0.5 ? 1 : -1;
    this.lastPos = new THREE.Vector2(0, 0);
    this.stuckTimer = 0;
    this.lowSpeedTimer = 0;
    this.tailSegLen = 0.168;
    this.rearAmount = 0;
    this.rearVel = 0;
    this.rearCooldown = 0;
    this.rearing = {
      mode: 'none',
      phase: 'idle',
      phaseTime: 0,
      elapsed: 0,
      peakAmount: 0,
      riseDuration: 0,
      holdDuration: 0,
      settleDuration: 0,
      sideLean: 0,
      maxAngle: 0,
      angle: 0,
      angularVelocity: 0,
      angularAcceleration: 0,
      gravityTorque: 0,
      muscleTorque: 0,
      wallReactionTorque: 0,
      wallContactLoad: 0,
      balanceAngleCorrection: 0,
      centerOfMassForwardOffset: 0,
      centerOfMassLateralOffset: 0,
      centerOfMassProjectionError: 0,
      hindSupportError: 0,
      hindSupportHorizontalError: 0,
      hindSupportVerticalError: 0,
      wallSupportError: 0,
      wallSupportNormalError: 0,
      wallSupportVerticalError: 0,
      baseAdvance: 0,
      baseAdvanceVelocity: 0,
      maximumBaseAdvance: 0,
      hindStanceOffset: 0,
      hindStanceVelocity: 0,
      maximumHindStanceOffset: 0,
      lastConstraintDt: 0,
      rearFeetPlanted: false,
      rearSupportReleased: false,
      hindAnchors: [new THREE.Vector3(), new THREE.Vector3()],
      supportDirection: new THREE.Vector2(0, 1),
      supportPoint: new THREE.Vector2(0, 0),
      nextSpontaneousAt: 3.2 + (this.timeOffset % 4.8),
    };
    this.groomAmount = 0;
    this.groomVel = 0;
    this.groomCooldown = 0;
    this.grooming = {
      mode: 'none',
      phase: 'idle',
      phaseTime: 0,
      elapsed: 0,
      peakAmount: 1,
      prepareDuration: 0,
      activeDuration: 0,
      settleDuration: 0,
      side: 1,
      sequenceIndex: Math.sin(this.timeOffset * 0.17) > 0 ? 0 : 1,
      nextSpontaneousAt: 5.5 + (this.timeOffset % 5.5),
    };
    this.maximumRearSupportError = 0;
    this.prevVelocity = new THREE.Vector2(0, 0);
    this.prevHeadingAngle = Math.atan2(this.heading.x, this.heading.y);
    this.yawAngle = this.prevHeadingAngle;
    this.yawVel = 0;
    this.bodyDyn = { y: 1.10, vy: 0, pitch: 0, pitchV: 0, roll: 0, rollV: 0 };
    this.tailDyn = { sway: 0, swayV: 0, lift: 0, liftV: 0 };
    this.spineFlex = {
      bend: 0, bendV: 0,
      sCurve: 0, sCurveV: 0,
      arch: 0, archV: 0,
      stretch: 0, stretchV: 0,
    };
    this.spineHeadYaw = 0;
    this.tailWorldPoints = null;
    this.tailPrevRootWorld = new THREE.Vector3(1e6, 1e6, 1e6);
    this.boutState = {
      mode: 'explore-walk',
      timer: 0.75 + (this.timeOffset % 1.35),
      scanPhase: (this.timeOffset * 0.37) % (Math.PI * 2),
    };
    this.socialState = {
      mode: 'none',
      duration: 0,
      cooldown: 0,
      targetWorld: new THREE.Vector2(0, 0),
    };
    this.earPinAmount = 0;
    this.yawErr = 0;
    this.behaviorTimer = 0.8 + random() * 2.4;
    this.pauseTimer = random() * 0.9;
    this.socialContact = false;
    this.socialDistance = Infinity;
    this.socialTarget = new THREE.Vector2(0, 1);

    const furColor = params.furColor;
    const skinColor = params.skinColor;
    const interactiveDensityScale = (!DATASET_CAPTURE && IS_SOFTWARE_WEBGL) ? 0.35 : 1.0;
    const density = FUR_DETAIL[params.furDetailLevel || 'Med']
      * (params.furDensity || 0.85)
      * interactiveDensityScale;
    const chonk = params.chonkiness;
    const earScaleByHead = THREE.MathUtils.clamp(0.9 + (params.headSize - 1.0) * 0.18, 0.82, 1.08);
    const eSize = params.earSize * earScaleByHead;
    const strain = getStrainProfile(params.presets);
    this.appearance = Object.freeze({ furColor, skinColor, density, strain: params.presets });

    // Per-instance subtle asymmetry
    this.asymmetry = {
      earTiltL: (appearanceRandom() - 0.5) * 0.06,
      earTiltR: (appearanceRandom() - 0.5) * 0.06,
      earYawDelta: (appearanceRandom() - 0.5) * 0.05,
      earHeightDiff: (appearanceRandom() - 0.5) * 0.03,
      noseYaw: (appearanceRandom() - 0.5) * 0.03,
      haunchDiff: (appearanceRandom() - 0.5) * 0.02,
      eyeOffsetL: (appearanceRandom() - 0.5) * 0.01,
      eyeOffsetR: (appearanceRandom() - 0.5) * 0.01,
    };

    this.motion = {
      gaitAmpMul: strain.gaitAmpMul ?? 1.0,
      headIdleYawMul: strain.headIdleYawMul ?? 1.0,
      headIdlePitchMul: strain.headIdlePitchMul ?? 1.0,
      sniffAmpMul: strain.sniffAmpMul ?? 1.0,
      breathIdleAmp: strain.breathIdleAmp ?? 0.013,
      breathMoveAmp: strain.breathMoveAmp ?? 0.016,
      breathIdleFreq: strain.breathIdleFreq ?? 12.0,
      breathMoveFreq: strain.breathMoveFreq ?? 20.0,
      tailWaveAmpMul: strain.tailWaveAmpMul ?? 1.0,
      tailWaveFreqMul: strain.tailWaveFreqMul ?? 1.0
    };

    // Ear micro-twitch state (independent per ear)
    this.earMicro = {
      twitchTimerL: appearanceRandom() * 3,
      twitchTimerR: appearanceRandom() * 3 + 1.5,
      twitchAngleL: 0, twitchAngleR: 0,
      nextIntervalL: 1.0 + appearanceRandom() * 3.0,
      nextIntervalR: 1.5 + appearanceRandom() * 3.5,
      baseRotL: null, baseRotR: null,
    };
    this.blink = {
      timer: appearanceRandom() * 3.0,
      interval: 2.2 + appearanceRandom() * 6.0,
      phase: 1.0,
    };

    // Flesh material (owned by this subject even though meshes use per-part clones)
    this.fleshMat = fleshMatBase.clone();
    this.mesh.userData.ownedMaterials = [this.fleshMat];
    this.fleshMat.color.set(skinColor);
    const skinHSL = {}; this.fleshMat.color.getHSL(skinHSL);
    this.fleshMat.attenuationColor.setHSL(skinHSL.h, skinHSL.s, skinHSL.l * 0.5);

    const applySkinShaderStack = (mat, aoRadiusCap = 0.52) => {
      const baseColor = mat.color.clone();
      if (params.enableSSS !== false) {
        annolidShaders.applySSSShader(mat, {
          scatterColor: [baseColor.r * 0.9, baseColor.g * 0.85, baseColor.b * 0.84],
          thickness: (params.sssThickness ?? 0.45) * 0.20,
          distortion: 0.2,
          wrap: params.sssWrap ?? 0.5,
          backlight: (params.sssBacklight ?? 0.6) * 0.25
        });
      }
      if (params.enableContactAO !== false) {
        annolidShaders.applyContactAO(mat, {
          groundY: 0.0,
          radius: Math.min(aoRadiusCap, params.aoRadius ?? aoRadiusCap),
          intensity: Math.min(0.16, (params.aoIntensity ?? 0.18) * 0.45),
          color: [0.02, 0.01, 0.0]
        });
      }
      if (params.enableMicroDetail !== false) {
        annolidShaders.applyMicroDetail(mat, {
          scale: params.microScale ?? 40.0,
          strength: (params.microStrength ?? 0.06) * 0.45
        });
      }
      mat.userData.annolidShaderScale = {
        sssThickness: 0.20,
        sssBacklight: 0.25,
        aoRadiusCap,
        aoIntensityScale: 0.45,
        aoIntensityCap: 0.16,
        microScaleFloor: 5.0,
        microStrengthScale: 0.45,
        microStrengthCap: 0.3,
      };
    };

    // ═══ APPLY GLSL SHADERS ═══
    applySkinShaderStack(this.fleshMat, 0.52);
    this.surfaceMat = createCoatMaterial(furColor, skinColor, density, params);
    this.bodyMat = createCoatMaterial(furColor, skinColor, density, params);
    this.mesh.userData.ownedMaterials.push(this.surfaceMat, this.bodyMat);
    for (const coatMat of [this.surfaceMat, this.bodyMat]) {
      if (params.enableContactAO !== false) {
        annolidShaders.applyContactAO(coatMat, {
          groundY: 0.0,
          radius: Math.min(0.72, params.aoRadius ?? 0.72),
          intensity: Math.min(0.18, (params.aoIntensity ?? 0.18) * 0.9),
          color: [0.018, 0.014, 0.010]
        });
      }
      coatMat.userData.annolidShaderScale = {
        sssThickness: 0.35,
        sssBacklight: 0.40,
        aoRadiusCap: 0.72,
        aoIntensityScale: 0.9,
        aoIntensityCap: 0.18,
        microScaleFloor: 70.0,
        microStrengthScale: 0.32,
        microStrengthCap: 0.024,
      };
    }
    annolidShaders.applySpinalFlexShader(this.bodyMat, { halfLen: PROP.bodyLength });
    this.bodyDepthMat = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    annolidShaders.applySpinalFlexShader(this.bodyDepthMat, { halfLen: PROP.bodyLength });
    this.mesh.userData.ownedMaterials.push(this.bodyDepthMat);

    // --- RIG HIERARCHY ---
    this.rig = new THREE.Group();
    this.mesh.add(this.rig);

    // BODY (with GPU spinal flexure + concertina squash-and-stretch)
    const furLenMul = THREE.MathUtils.clamp(params.furLength ?? 1.0, 0.3, 2.5);
    this.bodyMesh = new THREE.Mesh(bodyGeo, this.bodyMat);
    this.bodyMesh.customDepthMaterial = this.bodyDepthMat;
    this.bodyMesh.scale.set(chonk, chonk, params.bodyLength || 1.0);
    this.bodyMesh.castShadow = true;
    this.rig.add(this.bodyMesh);
    this.bodyFur = generateMultiLayerFur(deformBody, density, furColor, false, params);
    this.bodyFur.name = 'body-fur-coat';
    this.spinalMaterials = [this.bodyMat, this.bodyDepthMat];
    for (const furLayer of this.bodyFur.children) {
      if (furLayer?.material) {
        annolidShaders.applySpinalFlexShader(furLayer.material, { halfLen: PROP.bodyLength });
        this.spinalMaterials.push(furLayer.material);
      }
    }
    this.bodyMesh.add(this.bodyFur);

    // Tapered cervical mantle: broad at the shoulders, narrower under the
    // occiput, and long enough to remain connected while the head bends.
    const neckCoreMat = this.surfaceMat.clone();
    neckCoreMat.transparent = true;
    neckCoreMat.opacity = 0.0;
    neckCoreMat.depthWrite = false;
    neckCoreMat.colorWrite = false;
    const neckMesh = new THREE.Mesh(neckGeo, neckCoreMat);
    neckMesh.name = 'tapered-cervical-mantle';
    neckMesh.position.set(0, 0.14, PROP.bodyLength * 0.78);
    neckMesh.scale.set(chonk, chonk, 1.0);
    neckMesh.castShadow = false;
    neckMesh.receiveShadow = false;
    this.neckMesh = neckMesh;
    this.neckRestPosition = neckMesh.position.clone();
    this.neckRestRotation = neckMesh.rotation.clone();
    this.rig.add(neckMesh);
    const cervicalCore = new THREE.Mesh(cervicalCoreGeo, this.surfaceMat);
    cervicalCore.name = 'cervical-solid-core';
    cervicalCore.castShadow = false;
    cervicalCore.receiveShadow = true;
    neckMesh.add(cervicalCore);
    // Neck coat: the same two-layer pelage generator (materials, strand shading,
    // and caudal flow) as the body and head, seated on the cervical core, so the
    // head -> neck -> shoulder transition has no shading collar.
    const cervicalCoat = generateMultiLayerFur(
      deformCervicalCore,
      density * 0.11,
      furColor,
      false,
      { ...params, furLength: (params.furLength ?? 1.0) * 0.68 }
    );
    const [cervicalUndercoat = null, cervicalGuardHairs = null] = cervicalCoat.children;
    if (cervicalUndercoat) cervicalUndercoat.name = 'cervical-undercoat';
    if (cervicalGuardHairs) cervicalGuardHairs.name = 'cervical-guard-hairs';
    cervicalCoat.name = 'cervical-coat';
    this.cervicalUndercoat = cervicalUndercoat;
    this.cervicalGuardHairs = cervicalGuardHairs;
    neckMesh.add(cervicalCoat);

    // BELLY FUR ANCHOR (no visible geometry; avoids hard side cut-lines from above)
    const bellyFurAnchor = new THREE.Group();
    bellyFurAnchor.position.set(0, -0.43, PROP.bodyLength * 0.56);
    bellyFurAnchor.rotation.x = Math.PI / 2;
    bellyFurAnchor.scale.set(0.82, 0.76, 0.58);
    this.bellyFurAnchor = bellyFurAnchor;
    this.bellyRestZ = bellyFurAnchor.position.z;
    this.rig.add(bellyFurAnchor);
    // Add short ventral + flank fur to soften the dorsal/ventral transition.
    const bellyFurCount = Math.floor(7000 * density);
    if (bellyFurCount > 0) {
      const bellyFurLen = 0.092 * furLenMul;
      const bfGeo = new THREE.ConeGeometry(0.0065, bellyFurLen, 3);
      bfGeo.translate(0, bellyFurLen * 0.5, 0);
      const bellyTint = new THREE.Color(furColor).lerp(new THREE.Color(0xf4efe8), 0.14);
      const bfMat = enhanceFurMaterial(
        new THREE.MeshStandardMaterial({ color: bellyTint, roughness: 0.86, metalness: 0.0 }),
        furColor,
        true
      );
      const bfInst = new THREE.InstancedMesh(bfGeo, bfMat, bellyFurCount);
      const bDummy = new THREE.Object3D();
      const bNrm = new THREE.Vector3();
      for (let i = 0; i < bellyFurCount; i++) {
        const flank = appearanceRandom() < 0.42;
        const theta = flank
          ? ((appearanceRandom() < 0.5 ? 0 : Math.PI) + (appearanceRandom() - 0.5) * 0.90)
          : (appearanceRandom() * Math.PI * 2);
        const r = Math.pow(appearanceRandom(), 0.70);
        const x = Math.cos(theta) * r * (flank ? 0.30 : 0.24);
        const z = Math.sin(theta) * r * 0.23;
        const y = flank ? (-0.06 + (appearanceRandom() - 0.5) * 0.30) : (-0.18 + (appearanceRandom() - 0.5) * 0.22);
        bDummy.position.set(x, y, z);
        bNrm.set(x, y * 1.35 - 0.06, z).normalize();
        bDummy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), bNrm);
        const s = 0.65 + appearanceRandom() * 0.45;
        bDummy.scale.set(s, s, s);
        bDummy.updateMatrix();
        bfInst.setMatrixAt(i, bDummy.matrix);
      }
      bellyFurAnchor.add(bfInst);
    }

    // HEAD: rotate from the atlanto-occipital joint rather than the center
    // of the skull. The child offset preserves the established rest pose.
    this.headGroup = new THREE.Group();
    this.headGroup.name = 'atlanto-occipital-head-pivot';
    this.headGroup.position.set(0, 0.18 * chonk, PROP.bodyLength * 0.73);
    this.headRestPosition = this.headGroup.position.clone();
    this.rig.add(this.headGroup);

    this.head = new THREE.Mesh(headGeo, this.surfaceMat);
    this.head.name = 'cranium-and-muzzle';
    this.head.position.set(0, 0.14 * chonk, PROP.bodyLength * 0.20);
    this.headCenterOffset = this.head.position.clone();
    // The head overlaps the cervical volume; casting into that overlap
    // creates an artificial dark collar around the posterior skull.
    this.head.castShadow = false;
    this.headGroup.add(this.head);
    this.headFur = generateMultiLayerFur(deformHead, density, furColor, true, params);
    this.headFur.name = 'head-fur-coat';
    this.head.add(this.headFur);

    // SNOUT — Compact heart-shaped murine rhinarium with pavimentous dermal texture
    const noseMat = this.fleshMat.clone();
    noseMat.color.set(strain.noseColor ?? skinColor);
    applySkinShaderStack(noseMat, 0.45);
    noseMat.map = noseAlbedoMap;
    noseMat.normalMap = noseNormalMap;
    noseMat.normalScale = new THREE.Vector2(0.30, 0.30);
    noseMat.roughnessMap = noseRoughnessMap;
    noseMat.roughness = 0.40;
    noseMat.transmission = 0.10;
    noseMat.thickness = 0.12;
    noseMat.clearcoat = 0.32;
    noseMat.clearcoatRoughness = 0.22;
    // Heart-shaped rhinarium with median philtrum cleft and comma-shaped alar wings
    const rhinariumGeo = new THREE.SphereGeometry(0.135, 28, 28);
    deform(rhinariumGeo, (v) => {
      const yN = THREE.MathUtils.clamp((v.y / 0.135 + 1) * 0.5, 0, 1);
      v.x *= 0.76 + yN * 0.34;   // wider dorsolateral narial wings, narrower philtrum tip
      v.y *= 0.68;
      v.z *= 0.66;
      // Gentle median philtrum sulcus (shallow enough that head mesh never splits nose)
      if (Math.abs(v.x) < 0.028 && v.z > 0) v.z -= (0.028 - Math.abs(v.x)) * 0.36;
      // Dorsolateral alar folds framing comma-shaped nostrils
      const nDist = Math.sqrt(Math.pow(Math.abs(v.x) - 0.038, 2) + Math.pow(v.y - 0.008, 2));
      if (nDist < 0.052 && v.z > 0.008) v.z += (0.052 - nDist) * 0.30;
    });
    this.snoutTip = new THREE.Mesh(rhinariumGeo, noseMat);
    this.noseBaseScale = new THREE.Vector3(0.64, 0.52, 0.48);
    this.snoutTip.scale.set(
      this.noseBaseScale.x,
      this.noseBaseScale.y,
      this.noseBaseScale.z * (params.snoutLength || 1.0)
    );
    this.head.add(this.snoutTip);
    // Subtle satin moist sheen on the rhinarium apex (not a glossy blister)
    const moistMat = noseMat.clone();
    moistMat.roughness = 0.24; moistMat.clearcoat = 0.44;
    moistMat.clearcoatRoughness = 0.16; moistMat.transmission = 0.04;
    const moistTip = new THREE.Mesh(new THREE.SphereGeometry(0.044, 14, 14), moistMat);
    moistTip.scale.set(0.92, 0.60, 0.40);
    moistTip.position.set(0, 0.010, 0.046);
    this.snoutTip.add(moistTip);
    // Comma-shaped narial slits nestled flush inside the alar folds
    const nostrilMat = new THREE.MeshPhysicalMaterial({
      color: skinHSL.l > 0.5 ? 0x3a1c1d : 0x1e1112,
      roughness: 0.84, clearcoat: 0.14, clearcoatRoughness: 0.38
    });
    const nSlitGeo = new THREE.CapsuleGeometry(0.0072, 0.014, 8, 8);
    const nL = new THREE.Mesh(nSlitGeo, nostrilMat);
    nL.scale.set(0.72, 0.96, 0.48); nL.rotation.z = 0.46;
    nL.position.set(0.032, 0.000, 0.064); this.snoutTip.add(nL);
    const nR = new THREE.Mesh(nSlitGeo, nostrilMat);
    nR.scale.set(0.72, 0.96, 0.48); nR.rotation.z = -0.46;
    nR.position.set(-0.032, 0.000, 0.064); this.snoutTip.add(nR);
    // Delicate flush narial rims
    const rimMoistMat = noseMat.clone();
    rimMoistMat.roughness = 0.26; rimMoistMat.clearcoat = 0.46;
    for (const s of [1, -1]) {
      const rim = new THREE.Mesh(new THREE.TorusGeometry(0.011, 0.0022, 8, 12), rimMoistMat);
      rim.position.set(s * 0.032, 0.001, 0.061);
      rim.rotation.y = s * 0.40; rim.rotation.z = s * 0.42;
      this.snoutTip.add(rim);
    }
    // Median philtrum cleft
    const philtrumMat = new THREE.MeshPhysicalMaterial({
      color: skinHSL.l > 0.5 ? 0x5c2e2f : 0x261516,
      roughness: 0.68, transparent: true, opacity: 0.44
    });
    const philtrum = new THREE.Mesh(new THREE.CapsuleGeometry(0.0036, 0.034, 4, 8), philtrumMat);
    philtrum.position.set(0, -0.022, 0.060); this.snoutTip.add(philtrum);
    // Alar wing accents
    for (const s of [1, -1]) {
      const alar = new THREE.Mesh(new THREE.SphereGeometry(0.016, 12, 12), noseMat);
      alar.scale.set(0.84, 0.54, 0.56);
      alar.position.set(s * 0.036, 0.003, 0.052);
      this.snoutTip.add(alar);
    }

    // SPLIT RODENT UPPER LIP + ORAL CLEFT + CURVED MURINE INCISORS + LOWER LIP
    // Assembled in mouth-local coordinates (+Z outward, +Y toward nose) and
    // seated inside the recessed sub-philtral oral fossa by placeHeadFeatures().
    this.mouthGroup = new THREE.Group();
    this.mouthGroup.name = 'murine-mouth-assembly';
    this.head.add(this.mouthGroup);

    const oralCavityMat = new THREE.MeshStandardMaterial({
      color: skinHSL.l > 0.5 ? 0x642b33 : 0x261216,
      roughness: 0.88,
      metalness: 0.0,
    });
    const oralCavityGeo = new THREE.SphereGeometry(0.032, 12, 10);
    this.mouth = new THREE.Mesh(oralCavityGeo, oralCavityMat);
    this.mouth.name = 'oral-cleft-recess';
    this.mouth.scale.set(1.06, 0.46, 0.34);
    this.mouth.position.set(0, -0.004, -0.007);
    this.mouth.castShadow = false;
    this.mouthGroup.add(this.mouth);

    // Subtle median philtrum shadow crease linking rhinarium to the split upper lip
    const philtrumCreaseMat = new THREE.MeshStandardMaterial({
      color: skinHSL.l > 0.5 ? 0xcab0b0 : 0x20181a,
      roughness: 0.88,
      metalness: 0.0,
      transparent: true,
      opacity: 0.34,
    });
    const philtrumCrease = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.0024, 0.038, 4, 6),
      philtrumCreaseMat
    );
    philtrumCrease.name = 'philtrum-sulcus';
    philtrumCrease.position.set(0, 0.032, 0.001);
    this.mouthGroup.add(philtrumCrease);

    // Soft mucosal lip tint matched closely to strain nose/skin tone (standard
    // material avoids spawning extra transmission passes for 2-mm lip rims).
    const labialSkinMat = new THREE.MeshStandardMaterial({
      color: noseMat.color.clone().lerp(new THREE.Color(skinColor), 0.30),
      roughness: 0.52,
      metalness: 0.0,
      envMapIntensity: 0.45,
    });

    // Split rodent upper lip: smooth furred bilateral pads tucked seamlessly
    // into the muzzle with soft inverted-Y mucosal edges hugging the incisors.
    const upperLipGeo = new THREE.SphereGeometry(0.028, 10, 10);
    const upperLipRimGeo = new THREE.CapsuleGeometry(0.0045, 0.032, 4, 6);
    this.upperLips = [1, -1].map(s => {
      const upperLip = new THREE.Mesh(upperLipGeo, this.surfaceMat);
      upperLip.name = s > 0 ? 'left-upper-lip-lobe' : 'right-upper-lip-lobe';
      upperLip.scale.set(1.36, 0.82, 0.52);
      upperLip.position.set(s * 0.027, 0.016, -0.009);
      upperLip.rotation.z = -s * 0.42;
      upperLip.userData.side = s;
      upperLip.castShadow = false;
      this.mouthGroup.add(upperLip);

      const lipRim = new THREE.Mesh(upperLipRimGeo, labialSkinMat);
      lipRim.name = s > 0 ? 'left-upper-lip-margin' : 'right-upper-lip-margin';
      lipRim.position.set(s * 0.020, 0.008, 0.001);
      lipRim.rotation.z = s * (Math.PI * 0.5 - 0.48);
      lipRim.scale.set(0.86, 1.0, 0.62);
      lipRim.castShadow = false;
      this.mouthGroup.add(lipRim);
      return upperLip;
    });

    // Murine incisors: warm ivory-cream labial enamel, curved chisel profile.
    const enamelMat = new THREE.MeshStandardMaterial({
      color: skinHSL.l > 0.5 ? 0xf2e3be : 0xecddb6,
      roughness: 0.16,
      metalness: 0.02,
      envMapIntensity: 1.30,
    });
    const buildIncisorGeo = (w, h, d, curveSign = 1) => {
      const geo = new THREE.BoxGeometry(w, h, d, 2, 6, 1);
      deform(geo, (tv) => {
        // yN = 0 at gum root, 1 at occlusal chisel tip
        const yN = curveSign > 0
          ? THREE.MathUtils.clamp((h * 0.5 - tv.y) / Math.max(1e-4, h), 0, 1)
          : THREE.MathUtils.clamp((tv.y + h * 0.5) / Math.max(1e-4, h), 0, 1);
        // Gentle labial convexity & natural backward chisel curve toward tip
        const xN = tv.x / Math.max(1e-4, w * 0.5);
        tv.z += (1.0 - xN * xN) * d * 0.24;
        tv.z += Math.sin(yN * Math.PI) * 0.0042 - yN * yN * 0.0052;
        // Narrow slightly into a crisp chisel bevel at the biting tip
        tv.x *= 1.0 - yN * 0.16;
        tv.z *= 1.0 - yN * 0.52;
      });
      return geo;
    };
    const upperIncisorGeo = buildIncisorGeo(0.0112, 0.027, 0.010, 1);
    this.upperIncisors = [1, -1].map(s => {
      const tooth = new THREE.Mesh(upperIncisorGeo, enamelMat);
      tooth.name = s > 0 ? 'left-upper-incisor' : 'right-upper-incisor';
      tooth.position.set(s * 0.0068, 0.004, 0.0005);
      tooth.rotation.z = -s * 0.04;
      tooth.rotation.x = 0.10;
      tooth.castShadow = false;
      this.mouthGroup.add(tooth);
      return tooth;
    });
    const lowerIncisorGeo = buildIncisorGeo(0.0086, 0.019, 0.0075, -1);
    this.lowerIncisors = [1, -1].map(s => {
      const tooth = new THREE.Mesh(lowerIncisorGeo, enamelMat);
      tooth.name = s > 0 ? 'left-lower-incisor' : 'right-lower-incisor';
      tooth.position.set(s * 0.0054, -0.010, -0.0032);
      tooth.rotation.z = s * 0.04;
      tooth.rotation.x = -0.14;
      tooth.castShadow = false;
      this.mouthGroup.add(tooth);
      return tooth;
    });

    // Soft rose lower lip hugging cleanly beneath the incisor tips, backed by
    // a small rounded furred chin cushion.
    this.lowerLip = new THREE.Group();
    this.lowerLip.name = 'lower-lip-and-chin';
    this.lowerLip.position.set(0, -0.016, -0.003);
    const chinCushion = new THREE.Mesh(new THREE.SphereGeometry(0.026, 10, 10), this.surfaceMat);
    chinCushion.name = 'furred-chin-cushion';
    chinCushion.scale.set(1.22, 0.56, 0.58);
    chinCushion.position.set(0, -0.010, -0.008);
    chinCushion.castShadow = false;
    this.lowerLip.add(chinCushion);
    const lowerLipEdge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.0078, 0.024, 4, 8),
      labialSkinMat
    );
    lowerLipEdge.name = 'mucosal-lower-lip';
    lowerLipEdge.rotation.z = Math.PI / 2;
    lowerLipEdge.scale.set(0.88, 1.0, 0.64);
    lowerLipEdge.position.set(0, 0.003, 0.004);
    lowerLipEdge.castShadow = false;
    this.lowerLip.add(lowerLipEdge);
    this.mouthGroup.add(this.lowerLip);
    this.lowerLipRestY = this.lowerLip.position.y;

    // WHISKERS — Mystacial pad array + supraorbital & genal tactile vibrissae
    const wColor = strain.whiskerColor ?? (skinHSL.l > 0.5 ? 0xe6e0d8 : 0xb5aba2);
    const whiskerOpacity = strain.whiskerOpacity ?? (skinHSL.l > 0.5 ? 0.54 : 0.62);
    this.padL = new THREE.Group(); this.head.add(this.padL);
    this.padR = new THREE.Group(); this.head.add(this.padR);
    this.padL.add(createAnatomicalWhiskers(wColor, true, whiskerOpacity));
    this.padR.add(createAnatomicalWhiskers(wColor, false, whiskerOpacity));
    this.supraorbitalL = createSupraorbitalWhiskers(wColor, true, whiskerOpacity * 0.86);
    this.supraorbitalR = createSupraorbitalWhiskers(wColor, false, whiskerOpacity * 0.86);
    this.head.add(this.supraorbitalL);
    this.head.add(this.supraorbitalR);
    this.placeHeadFeatures();

    // EARS (with asymmetry) — rounded mouse pinnae integrated into the skull fur.
    const earOuterMat = this.fleshMat.clone();
    earOuterMat.color.set(strain.earOuterColor ?? skinColor);
    earOuterMat.color.lerp(new THREE.Color(skinColor), strain.earSkinBlend ?? 0.48);
    earOuterMat.transmission = strain.earOuterTransmission ?? 0.38;
    earOuterMat.ior = 1.38;
    earOuterMat.thickness = 0.035;
    earOuterMat.roughness = strain.earOuterRoughness ?? 0.56;
    earOuterMat.map = earAlbedoMap;
    earOuterMat.roughnessMap = earRoughnessMap;
    earOuterMat.normalMap = earNormalMap;
    earOuterMat.normalScale = new THREE.Vector2(0.18, 0.16);
    earOuterMat.clearcoat = 0.025;
    earOuterMat.clearcoatRoughness = 0.42;
    earOuterMat.sheen = 0.16;
    earOuterMat.sheenColor = new THREE.Color(strain.earSheenColor ?? 0xd0b5a8);
    earOuterMat.side = THREE.DoubleSide;
    earOuterMat.attenuationColor = new THREE.Color(strain.earAttenuationColor ?? 0xd89a8a);
    earOuterMat.attenuationDistance = strain.earAttenuationDistance ?? 0.18;
    const earInnerMat = this.fleshMat.clone();
    earInnerMat.color.set(strain.earInnerColor ?? strain.earOuterColor ?? skinColor);
    earInnerMat.color.lerp(new THREE.Color(skinColor), strain.earInnerSkinBlend ?? 0.30);
    earInnerMat.color.offsetHSL(0.0, strain.earInnerSatOffset ?? 0.035, strain.earInnerLightOffset ?? 0.015);
    earInnerMat.transmission = strain.earInnerTransmission ?? 0.42;
    earInnerMat.ior = 1.38;
    earInnerMat.thickness = 0.015;
    earInnerMat.roughness = 0.58;
    earInnerMat.map = earAlbedoMap;
    earInnerMat.roughnessMap = earRoughnessMap;
    earInnerMat.normalMap = earNormalMap;
    earInnerMat.normalScale = new THREE.Vector2(0.23, 0.20);
    earInnerMat.clearcoat = 0.025;
    earInnerMat.clearcoatRoughness = 0.40;
    earInnerMat.sheen = 0.22;
    earInnerMat.sheenColor = new THREE.Color(strain.earSheenColor ?? 0xffdfc8);
    earInnerMat.side = THREE.DoubleSide;
    earInnerMat.attenuationColor = new THREE.Color(strain.earAttenuationColor ?? 0xd89a8a);
    earInnerMat.attenuationDistance = strain.earAttenuationDistance ?? 0.16;

    const earL = buildMouseEar({
      sideSign: 1,
      eSize,
      strain,
      skinColor,
      furColor,
      density,
      outerMat: earOuterMat,
      innerMat: earInnerMat,
      asymmetryTilt: this.asymmetry.earTiltL,
      asymmetryYaw: this.asymmetry.earYawDelta,
      heightOffset: (strain.earHeightOffset ?? 0.0) + this.asymmetry.earHeightDiff,
    });
    this.head.add(earL);

    const earR = buildMouseEar({
      sideSign: -1,
      eSize,
      strain,
      skinColor,
      furColor,
      density,
      outerMat: earOuterMat,
      innerMat: earInnerMat,
      asymmetryTilt: this.asymmetry.earTiltR,
      asymmetryYaw: this.asymmetry.earYawDelta,
      heightOffset: (strain.earHeightOffset ?? 0.0) - this.asymmetry.earHeightDiff,
    });
    this.head.add(earR);

    // Store ear refs for micro-motion and live GUI scaling
    this.earL = earL; this.earR = earR;
    this.earMicro.baseRotL = earL.rotation.clone();
    this.earMicro.baseRotR = earR.rotation.clone();
    this.spawnEarSize = params.earSize || 0.69;
    this.baseEarScaleL = earL.scale.clone();
    this.baseEarScaleR = earR.scale.clone();

    // EYES — Lateral almond palpebral rims + wet tear meniscus + dual-layer globe/cornea
    const subjectEyeMat = eyeMat.clone();
    subjectEyeMat.color.set(strain.eyeColor ?? '#060505');
    subjectEyeMat.attenuationColor = subjectEyeMat.color.clone();
    const eyeHsl = {};
    subjectEyeMat.color.getHSL(eyeHsl);
    if (eyeHsl.l > 0.14) {
      // Albino ruby-garnet eye: soft internal retinal vascular luminosity
      subjectEyeMat.emissive = subjectEyeMat.color.clone().multiplyScalar(0.36);
      subjectEyeMat.emissiveIntensity = 0.42;
      subjectEyeMat.transmission = 0.14;
      subjectEyeMat.thickness = 0.18;
    }
    const subjectCorneaMat = corneaMat.clone();
    subjectCorneaMat.attenuationColor = subjectEyeMat.color.clone().offsetHSL(0, 0, 0.08);
    const lidMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(strain.eyeRimColor ?? 0x221717).lerp(new THREE.Color(skinColor), 0.28),
      roughness: 0.58,
      metalness: 0.0,
      clearcoat: 0.14,
      clearcoatRoughness: 0.40,
    });
    const tearMeniscusMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(strain.eyeRimColor ?? 0x2a1b1b).lerp(new THREE.Color(0xf6e6e4), 0.10),
      roughness: 0.06,
      metalness: 0.0,
      clearcoat: 0.90,
      clearcoatRoughness: 0.06,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
    });
    // Almond palpebral rim seated cleanly around the orbital aperture so the
    // glossy globe emerges from a living eyelid margin without crossing the pupil.
    const lidArcGeo = new THREE.TorusGeometry(PROP.eyeSize * 0.96, 0.0105, 8, 22, Math.PI * 0.92);
    const tearArcGeo = new THREE.TorusGeometry(PROP.eyeSize * 0.92, 0.0030, 6, 18, Math.PI * 0.54);
    const localForwardZ = new THREE.Vector3(0, 0, 1);
    // SphereGeometry centers the iris texture on local +X; rotate it onto +Z so
    // the pupil looks along the eye's outward axis and the UV seam stays hidden.
    const eyeGeo = new THREE.SphereGeometry(PROP.eyeSize, 28, 28).rotateY(-Math.PI / 2);
    const corneaGeo = new THREE.SphereGeometry(PROP.eyeSize + 0.0045, 28, 28).rotateY(-Math.PI / 2);
    const headAnchors = getHeadFeatureAnchors();

    const buildEyeAssembly = (sideSign, offsetAsym) => {
      // Murine eyes are lateral (~60 deg from the midline) and sit half-buried
      // in the orbital fossa, bulging out of an almond palpebral aperture.
      const anchor = sideSign > 0 ? headAnchors.eyeL : headAnchors.eyeR;
      const designDir = new THREE.Vector3(sideSign * 0.86, 0.14, 0.48).normalize();
      const outDir = anchor.normal.clone().lerp(designDir, 0.6).normalize();
      const eyeCenter = anchor.position.clone()
        .addScaledVector(outDir, -PROP.eyeSize * 0.26);
      eyeCenter.x += offsetAsym;
      const eyeQuat = new THREE.Quaternion().setFromUnitVectors(localForwardZ, outDir);

      const eye = new THREE.Mesh(eyeGeo, subjectEyeMat);
      eye.scale.set(0.98, 0.94, PROP.eyeProtrusion);
      eye.position.copy(eyeCenter);
      eye.quaternion.copy(eyeQuat);
      eye.castShadow = false;
      this.head.add(eye);

      const cornea = new THREE.Mesh(corneaGeo, subjectCorneaMat);
      cornea.scale.set(0.98, 0.94, PROP.eyeProtrusion + 0.04);
      cornea.position.copy(eyeCenter);
      cornea.quaternion.copy(eyeQuat);
      cornea.castShadow = false;
      this.head.add(cornea);

      // Almond palpebral rim seated right where the globe emerges from the orbit
      const lidFrame = new THREE.Group();
      lidFrame.position.copy(eyeCenter).addScaledVector(outDir, PROP.eyeSize * 0.21);
      lidFrame.quaternion.copy(eyeQuat);
      this.head.add(lidFrame);

      const upperLid = new THREE.Mesh(lidArcGeo, lidMat);
      upperLid.rotation.z = Math.PI * 0.04;
      upperLid.scale.set(1.06, 0.90, 0.72);
      lidFrame.add(upperLid);

      const lowerLid = new THREE.Mesh(lidArcGeo, lidMat);
      lowerLid.rotation.z = Math.PI * 1.04;
      lowerLid.scale.set(1.04, 0.86, 0.70);
      lidFrame.add(lowerLid);

      const tear = new THREE.Mesh(tearArcGeo, tearMeniscusMat);
      tear.rotation.z = Math.PI * 1.22;
      tear.position.set(0, -0.002, 0.003);
      lidFrame.add(tear);

      return { eye, cornea, upperLid, lowerLid };
    };

    const leftEyeSet = buildEyeAssembly(1, this.asymmetry.eyeOffsetL);
    const rightEyeSet = buildEyeAssembly(-1, this.asymmetry.eyeOffsetR);
    this.eyeL = leftEyeSet.eye;
    this.eyeR = rightEyeSet.eye;
    this.corneaL = leftEyeSet.cornea;
    this.corneaR = rightEyeSet.cornea;
    this.upperLidL = leftEyeSet.upperLid;
    this.lowerLidL = leftEyeSet.lowerLid;
    this.upperLidR = rightEyeSet.upperLid;
    this.lowerLidR = rightEyeSet.lowerLid;

    // ═══ LIMBS (anatomically correct chains) ═══
    const legLengthScale = params.legLength ?? 1.0;
    const legThicknessScale = params.legThickness ?? 1.0;
    // Hindlimbs
    this.hindL = createHindlimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'L', legLengthScale, legThicknessScale);
    this.rig.add(this.hindL);
    this.hindR = createHindlimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'R', legLengthScale, legThicknessScale);
    this.rig.add(this.hindR);
    this.hindRestZ = this.hindL.position.z;
    this.hindRestPosL = this.hindL.position.clone();
    this.hindRestPosR = this.hindR.position.clone();

    // Forelimbs
    this.armL = createForelimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'L', legLengthScale, legThicknessScale);
    this.rig.add(this.armL);
    this.armR = createForelimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'R', legLengthScale, legThicknessScale);
    this.rig.add(this.armR);
    this.armRestZ = this.armL.position.z;
    this.armRestPosL = this.armL.position.clone();
    this.armRestPosR = this.armR.position.clone();

    // References for animation
    this.thighL = this.hindL._thigh;
    this.thighR = this.hindR._thigh;
    this.shankL = this.hindL._shank;
    this.shankR = this.hindR._shank;
    this.upperArmL = this.armL._upperArm;
    this.upperArmR = this.armR._upperArm;

    const makeFootPlant = (
      paw,
      joints,
      limb,
      isFore,
      side,
      maximumReach = null
    ) => ({
      paw,
      contact: paw._groundContact,
      joints,
      limb,
      isFore,
      side,
      maximumReach,
      planted: false,
      exhausted: false,
      anchor: new THREE.Vector3(),
      slip: 0,
      groundError: 0,
      reachRatio: 0,
      rejectionReason: 'none',
      candidateSlip: 0,
      candidateGroundError: 0,
      constraintError: new THREE.Vector3(),
      constraintTargetLocal: new THREE.Vector3(),
      constraintContactLocal: new THREE.Vector3(),
    });
    this.footPlants = [
      makeFootPlant(this.hindL._paw, [this.hindL._ankle, this.shankL, this.hindL], 'LH', false, 1),
      makeFootPlant(this.hindR._paw, [this.hindR._ankle, this.shankR, this.hindR], 'RH', false, -1),
      makeFootPlant(
        this.armL._paw,
        [this.armL._wrist, this.armL._forearm, this.upperArmL],
        'LF',
        true,
        1,
        this.armL._maximumReach
      ),
      makeFootPlant(
        this.armR._paw,
        [this.armR._wrist, this.armR._forearm, this.upperArmR],
        'RF',
        true,
        -1,
        this.armR._maximumReach
      ),
    ];
    // FlyBody-style layered motor control: descending drive -> limb CPG ->
    // named joint targets (see two_mice_control.js). The jitter is derived from
    // timeOffset so it never consumes the behavioral random stream.
    this.motor = new MouseMotorController({
      gait: 'trot',
      jitter: (((this.timeOffset * 0.37) % 1) - 0.5) * 0.08,
    });
    this.externalCommand = null;
    this.jointBindings = this.createJointBindings();
    this.contactFrame = {
      limbStance: { LF: true, RF: true, LH: true, RH: true },
      isMoving: false,
      rear: 0,
      groom: 0,
      lateralAccel: 0,
      turnRate: 0,
      forwardAccel: 0,
    };
    this.maximumFootSlip = 0;
    this.maximumPawGroundError = 0;
    this.maximumForelimbExtensionRatio = 0;
    this.groomingPawContactError = 0;
    this.groomingSupportError = 0;
    this.groomingReachDeficit = 0;
    this.groomingTargetReachRatio = 0;
    this.minimumTailClearance = Infinity;
    this.minimumTailClearanceIndex = -1;
    this.maximumTailSegmentLengthError = 0;
    this.maximumTailBendAngle = 0;

    // CONTACT SHADOW
    this.contactShadow = createContactShadow();
    this.mesh.add(this.contactShadow);

    // TAIL — Countershaded annular epidermal scale rings + normal relief
    this.tailCount = 45;
    const tailMat = this.fleshMat.clone();
    tailMat.color.set(strain.tailColor ?? skinColor);
    applySkinShaderStack(tailMat, 0.52);
    const tailHSL = {}; tailMat.color.getHSL(tailHSL);
    tailMat.attenuationColor.setHSL(tailHSL.h, Math.min(1, tailHSL.s * 1.15), tailHSL.l * 0.55);
    tailMat.roughness = 0.56;
    tailMat.transmission = strain.tailTransmission ?? 0.20;
    tailMat.thickness = 0.36;
    tailMat.clearcoat = 0.12;
    tailMat.clearcoatRoughness = 0.36;
    tailMat.roughnessMap = tailRoughnessMap;
    tailMat.normalMap = tailNormalMap;
    tailMat.normalScale = new THREE.Vector2(0.30, 0.38);
    tailMat.vertexColors = true;
    const tGeo = createDynamicTailTubeGeometry(this.tailCount, 14);
    this.tailMesh = new THREE.Mesh(tGeo, tailMat);
    this.tailMesh.name = 'continuous-tail-tube';
    this.tailMesh.castShadow = true;
    this.tailMesh.receiveShadow = true;
    this.tailMesh.frustumCulled = false;
    this.mesh.add(this.tailMesh);
    // Fur-to-skin sacral root bridge blending tail emergence into the rump.
    const tailRootMat = tailMat.clone();
    tailRootMat.vertexColors = false;
    applySkinShaderStack(tailRootMat, 0.52);
    const tailRootGeo = new THREE.CylinderGeometry(0.105, 0.132, 0.28, 14);
    tailRootGeo.translate(0, 0.14, 0);
    tailRootGeo.rotateX(Math.PI / 2);
    this.tailRoot = new THREE.Mesh(tailRootGeo, tailRootMat);
    this.tailRoot.castShadow = true;
    this.tailRoot.scale.set(1.0, 0.74, 1.16);
    const tailBaseFur = createTailBaseFur(furColor, density);
    this.tailBaseFur = tailBaseFur;
    this.tailRoot.add(tailBaseFur);
    this.mesh.add(this.tailRoot);

    // Optional cranial implant headstage + flexible commutator tether (inspired by CalMS21 / miniscope rigs)
    this.hasHeadstage = false;
    this.headstageGroup = new THREE.Group();
    this.headstageGroup.name = 'cranial-implant-headstage';
    this.headstageGroup.position.set(0, 0.45, 0.04);
    this.headstageGroup.visible = false;
    const cementMat = new THREE.MeshStandardMaterial({
      color: 0xe9e6de,
      roughness: 0.64,
      metalness: 0.0,
      emissive: 0xffffff,
      emissiveIntensity: 0.03,
    });
    const headbarMat = new THREE.MeshStandardMaterial({
      color: 0xf7f6f2,
      roughness: 0.40,
      metalness: 0.06,
      emissive: 0xffffff,
      emissiveIntensity: 0.06,
    });
    const holeMat = new THREE.MeshStandardMaterial({ color: 0x18191d, roughness: 0.65, metalness: 0.25 });
    const scopeMat = new THREE.MeshStandardMaterial({ color: 0x1b1c20, roughness: 0.36, metalness: 0.35 });
    this.mesh.userData.ownedMaterials.push(cementMat, headbarMat, holeMat, scopeMat);

    const cementCap = new THREE.Mesh(new THREE.SphereGeometry(0.32, 18, 12), cementMat);
    cementCap.scale.set(1.12, 0.42, 1.24);
    this.headstageGroup.add(cementCap);

    const wingBar = new THREE.Mesh(new THREE.BoxGeometry(1.36, 0.042, 0.29), headbarMat);
    wingBar.position.set(0, 0.072, -0.02);
    wingBar.castShadow = true;
    this.headstageGroup.add(wingBar);
    for (const sx of [-0.49, 0.49]) {
      const screwHole = new THREE.Mesh(new THREE.CylinderGeometry(0.066, 0.066, 0.050, 14), holeMat);
      screwHole.position.set(sx, 0.074, -0.02);
      this.headstageGroup.add(screwHole);
    }
    const scopeBody = new THREE.Mesh(new THREE.CylinderGeometry(0.088, 0.105, 0.20, 12), scopeMat);
    scopeBody.position.set(0, 0.16, -0.03);
    scopeBody.castShadow = true;
    this.headstageGroup.add(scopeBody);
    const strainCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.074, 0.12, 10), scopeMat);
    strainCollar.position.set(0, 0.28, -0.03);
    this.headstageGroup.add(strainCollar);
    this.headstageConnector = new THREE.Object3D();
    this.headstageConnector.position.set(0, 0.34, -0.03);
    this.headstageGroup.add(this.headstageConnector);
    this.head.add(this.headstageGroup);

    this.tetherSegments = 26;
    const tetherMat = new THREE.MeshStandardMaterial({ color: 0x141518, roughness: 0.44, metalness: 0.10 });
    this.mesh.userData.ownedMaterials.push(tetherMat);
    const tetherGeo = createDynamicTailTubeGeometry(this.tetherSegments, 8);
    this.tetherMesh = new THREE.Mesh(tetherGeo, tetherMat);
    this.tetherMesh.name = 'commutator-tether-cable';
    this.tetherMesh.castShadow = true;
    this.tetherMesh.frustumCulled = false;
    this.tetherMesh.visible = false;
    this.tetherPoints = Array.from({ length: this.tetherSegments + 1 }, () => new THREE.Vector3());
    this.mesh.add(this.tetherMesh);

    // POSE TRACKING SKELETON
    this.poseNodes = {};
    this.trackingGroup = new THREE.Group();
    this.trackingGroup.visible = Boolean(params.showTrackingKeypoints);
    this.mesh.add(this.trackingGroup);

    const createTrackingNode = (id, parent, localPos) => {
      const mat = new THREE.MeshBasicMaterial({
        color: POSE_COLORS[id] || 0xffffff, depthTest: false, transparent: true, opacity: 0.9
      });
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 12), mat);
      sphere.renderOrder = 999;
      sphere.visible = Boolean(params.showTrackingKeypoints);
      if (parent) {
        sphere.position.copy(localPos); parent.add(sphere);
        this.poseNodes[id] = sphere;
      } else {
        this.trackingGroup.add(sphere);
        this.poseNodes[id] = sphere;
      }
    };

    // Tracking nodes at anatomical landmarks: the nose keypoint sits on the
    // rhinarium tip and ear keypoints at each pinna's center.
    const noseAnchor = getHeadFeatureAnchors().nose;
    createTrackingNode(
      'snout',
      this.head,
      noseAnchor.position.clone().addScaledVector(noseAnchor.normal, 0.02)
    );
    const pinnaCenterInHead = ear => {
      ear.updateMatrix();
      return ear._pinna.position.clone().applyMatrix4(ear.matrix);
    };
    createTrackingNode('lear', this.head, pinnaCenterInHead(this.earL));
    createTrackingNode('rear', this.head, pinnaCenterInHead(this.earR));
    createTrackingNode(
      'neck',
      this.neckMesh,
      new THREE.Vector3(0, 0.46, -PROP.neckLength * 0.26)
    );
    createTrackingNode('spine1', this.rig, new THREE.Vector3(0, 0.9, 0.2));
    createTrackingNode('spine2', this.rig, new THREE.Vector3(0, 0.8, -PROP.bodyLength * 0.41));
    createTrackingNode('tailbase', this.rig, new THREE.Vector3(0, 0.3, -PROP.bodyLength * 0.93));
    // Paws attached to limb chains
    createTrackingNode('lforepaw', this.armL._paw, new THREE.Vector3(0, -0.05, 0.12));
    createTrackingNode('rforepaw', this.armR._paw, new THREE.Vector3(0, -0.05, 0.12));
    createTrackingNode('lhindpaw', this.hindL._paw, new THREE.Vector3(0, -0.08, 0.18));
    createTrackingNode('rhindpaw', this.hindR._paw, new THREE.Vector3(0, -0.08, 0.18));

    this.mesh.updateMatrixWorld(true);
    this.mesh.traverse(child => {
      if (child.isMesh || child.isInstancedMesh) child.frustumCulled = false;
    });

    createTrackingNode('tailmid', null, new THREE.Vector3());
    createTrackingNode('tailend', null, new THREE.Vector3());

    // Skeleton edges
    const lineMat = new THREE.LineBasicMaterial({ color: 0x00ffcc, depthTest: false, transparent: true, opacity: 0.7, linewidth: 2 });
    this.skeletonLines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMat);
    this.skeletonLines.renderOrder = 998;
    this.trackingGroup.add(this.skeletonLines);
    const linePositions = new Float32Array(POSE_CONNECTIONS.length * 6);
    this.skeletonLines.geometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  }

  syncTrackingVisibility(visible = Boolean(this.guiParams.showTrackingKeypoints)) {
    const show = Boolean(visible);
    this.trackingGroup.visible = show;
    const isTailVisible = this.tailMesh ? Boolean(this.tailMesh.visible) : true;
    for (const [key, node] of Object.entries(this.poseNodes)) {
      if (!node) continue;
      node.visible = (key === 'tailmid' || key === 'tailend')
        ? (show && isTailVisible)
        : show;
    }
    if (show) this.updateTrackingGeometry();
  }

  setLiveEarSize(nextEarSize) {
    if (!this.earL || !this.earR || !this.baseEarScaleL || !this.baseEarScaleR) return;
    const ratio = THREE.MathUtils.clamp(
      Number(nextEarSize || this.spawnEarSize) / Math.max(0.1, this.spawnEarSize),
      0.4,
      3.0
    );
    this.earL.scale.copy(this.baseEarScaleL).multiplyScalar(ratio);
    this.earR.scale.copy(this.baseEarScaleR).multiplyScalar(ratio);
  }

  setHeadstageEnabled(enabled) {
    const active = Boolean(enabled);
    this.hasHeadstage = active;
    if (this.headstageGroup) this.headstageGroup.visible = active;
    if (this.tetherMesh) this.tetherMesh.visible = active;
    if (active) this.updateHeadstageTether();
  }

  // ── Actuators: bind MOUSE_JOINT_SPEC names to rig transforms ──
  createJointBindings() {
    const chains = {
      LF: { chain: this.armL, upper: this.upperArmL },
      RF: { chain: this.armR, upper: this.upperArmR },
      LH: { chain: this.hindL, shank: this.shankL },
      RH: { chain: this.hindR, shank: this.shankR },
    };
    const segmentObject = (limb, segment) => {
      const entry = chains[limb];
      switch (segment) {
        case 'hip': return entry.chain;
        case 'knee': return entry.shank;
        case 'ankle': return entry.chain._ankle;
        case 'toe': return entry.chain._paw;
        case 'shoulder': return entry.upper;
        case 'elbow': return entry.chain._forearm;
        case 'wrist': return entry.chain._wrist;
        case 'carpal': return entry.chain._paw;
        default: return null;
      }
    };
    return MOUSE_JOINT_SPEC.map(joint => {
      const object = segmentObject(joint.limb, joint.segment);
      if (!object) throw new Error(`Mouse rig has no segment for joint ${joint.name}.`);
      return {
        name: joint.name,
        target: joint.type === 'slide' ? object.position : object.rotation,
        axis: joint.axis,
      };
    });
  }

  applyJointTargets(targets) {
    for (const binding of this.jointBindings) {
      const value = targets[binding.name];
      if (Number.isFinite(value)) binding.target[binding.axis] = value;
    }
  }

  readJointPositions() {
    const positions = {};
    for (const binding of this.jointBindings) {
      positions[binding.name] = binding.target[binding.axis];
    }
    return positions;
  }

  // Proprioceptive feedback for the CPG lift-off reflex, in MOUSE_LIMBS order.
  getLiftOffRequests() {
    const byLimb = {};
    for (const foot of this.footPlants) {
      byLimb[foot.limb] = foot.rejectionReason === 'reach'
        || (foot.planted && foot.reachRatio > 0.96)
        || (foot.planted && foot.slip > STANCE_CONTACT_LIMITS.slip * 0.8);
    }
    return [byLimb.LF, byLimb.RF, byLimb.LH, byLimb.RH].map(Boolean);
  }

  // ── External control (analogous to a FlyBody task command) ──
  setMotorCommand(command, time = 0) {
    const sanitized = sanitizeMotorCommand(command, { maxSpeed: this.guiParams.maxSpeed });
    this.externalCommand = {
      ...sanitized,
      heading: this.yawAngle,
      startedAt: time,
      expiresAt: time + sanitized.durationSec,
    };
    // A command replaces spontaneous pauses but lets active postures finish.
    this.pauseTimer = 0;
    this.lowSpeedTimer = 0;
    this.stuckTimer = 0;
    return this.describeMotorCommand();
  }

  clearMotorCommand() {
    this.externalCommand = null;
  }

  describeMotorCommand() {
    const command = this.externalCommand;
    if (!command) return null;
    return {
      forwardSpeed: command.forwardSpeed,
      turnRate: command.turnRate,
      gait: command.gait,
      durationSec: Number.isFinite(command.durationSec) ? command.durationSec : null,
      startedAt: command.startedAt,
      expiresAt: Number.isFinite(command.expiresAt) ? command.expiresAt : null,
      commandedHeading: command.heading,
    };
  }

  /** Egocentric observation, analogous to FlyBody's walker/* observables. */
  getObservation() {
    const forward = this.velocity.x * this.heading.x + this.velocity.y * this.heading.y;
    const lateral = this.velocity.x * this.heading.y - this.velocity.y * this.heading.x;
    const footContacts = {};
    const footSlip = {};
    for (const foot of this.footPlants) {
      footContacts[foot.limb] = Boolean(foot.planted);
      footSlip[foot.limb] = foot.planted ? foot.slip : 0;
    }
    const flex = this.effectiveSpineFlex || this.spineFlex;
    return {
      trackId: this.trackId,
      strain: this.appearance.strain,
      position: [this.mesh.position.x, this.mesh.position.z],
      heading: this.yawAngle,
      yawRate: this.yawVel,
      egocentricVelocity: { forward, lateral },
      speed: this.velocity.length(),
      body: {
        height: this.rig.position.y,
        pitch: this.rig.rotation.x,
        roll: this.rig.rotation.z,
      },
      spine: {
        bend: flex.bend, sCurve: flex.sCurve, arch: flex.arch, stretch: flex.stretch,
      },
      jointPositions: this.readJointPositions(),
      jointTargets: { ...this.motor.targets },
      footContacts,
      footSlip,
      motor: this.motor.snapshot(),
      behavior: {
        bout: this.boutState.mode,
        social: this.socialState.mode,
        rearing: this.rearing.mode,
        rearingPhase: this.rearing.phase,
        grooming: this.grooming.mode,
        groomingPhase: this.grooming.phase,
      },
      command: this.describeMotorCommand(),
    };
  }

  resetNoseScale() {
    this.snoutTip.scale.set(
      this.noseBaseScale.x,
      this.noseBaseScale.y,
      this.noseBaseScale.z * (this.guiParams.snoutLength || 1.0)
    );
  }

  // ── Face: place nose, mouth, lips and whisker pads on the head surface ──
  placeHeadFeatures() {
    const anchors = getHeadFeatureAnchors();
    const Z = new THREE.Vector3(0, 0, 1);
    // Rhinarium: seated cleanly into the blunt muzzle dome, facing along its normal.
    const nose = anchors.nose;
    this.noseRestPosition = nose.position.clone()
      .addScaledVector(nose.normal, -0.024);
    this.noseRestPosition.x += this.asymmetry.noseYaw * 0.55;
    this.snoutTip.position.copy(this.noseRestPosition);
    this.snoutTip.quaternion.setFromUnitVectors(
      Z,
      nose.normal.clone().lerp(Z, 0.55).normalize()
    );
    // Seat the mouth assembly (philtrum, split upper lip, curved incisors,
    // oral cleft, and lower lip/chin) into the recessed sub-philtral fossa.
    const mouth = anchors.mouth;
    const mouthForward = mouth.normal.clone()
      .lerp(new THREE.Vector3(0, -0.24, 0.97).normalize(), 0.45)
      .normalize();
    this.mouthGroup.position.copy(mouth.position)
      .addScaledVector(mouth.normal, 0.002);
    this.mouthGroup.quaternion.setFromUnitVectors(Z, mouthForward);
    // Whisker follicle pads on the mystacial cushions.
    this.padL.position.copy(anchors.padL.position).addScaledVector(anchors.padL.normal, -0.012);
    this.padR.position.copy(anchors.padR.position).addScaledVector(anchors.padR.normal, -0.012);
    // Supraorbital/genal vibrissae were authored for eyes at x = +/-0.415.
    if (this.supraorbitalL) this.supraorbitalL.position.x = anchors.eyeL.position.x - 0.415;
    if (this.supraorbitalR) this.supraorbitalR.position.x = anchors.eyeR.position.x + 0.415;
  }

  updateHeadstageTether() {
    if (!this.hasHeadstage || !this.tetherMesh || !this.headstageConnector) return;
    this.mesh.updateMatrixWorld(true);
    const startWorld = new THREE.Vector3();
    this.headstageConnector.getWorldPosition(startWorld);
    // Overhead commutator swivel offset slightly toward the front-left of the cage
    // so the cable sweeps naturally across the frame just like mouse083_task1_annotator1.mp4.
    const endWorld = new THREE.Vector3(-15.4, 23.5, 11.6);
    const dx = endWorld.x - startWorld.x;
    const dz = endWorld.z - startWorld.z;
    const horizLen = Math.hypot(dx, dz) || 1.0;
    const perpX = -dz / horizLen;
    const perpZ = dx / horizLen;
    // Natural slack arc visible from overhead (avoids dead-straight ruler line)
    const headCarryX = -Math.sin(this.yawAngle) * 1.35 + perpX * 1.85;
    const headCarryZ = -Math.cos(this.yawAngle) * 1.35 + perpZ * 1.85;
    const ctrl1 = startWorld.clone().add(new THREE.Vector3(
      dx * 0.22 + headCarryX,
      2.4,
      dz * 0.22 + headCarryZ
    ));
    const ctrl2 = startWorld.clone().lerp(endWorld, 0.62);
    ctrl2.x += perpX * 1.15;
    ctrl2.z += perpZ * 1.15;
    ctrl2.y -= 2.0;
    const curve = new THREE.CubicBezierCurve3(startWorld, ctrl1, ctrl2, endWorld);
    for (let i = 0; i <= this.tetherSegments; i++) {
      curve.getPoint(i / this.tetherSegments, this.tetherPoints[i]);
      this.mesh.worldToLocal(this.tetherPoints[i]);
    }
    updateDynamicTailTubeGeometry(this.tetherMesh.geometry, this.tetherPoints, 0.42);
    // Keep tether thickness uniform rather than tapered like a biological tail
    const state = this.tetherMesh.geometry.userData.tailTube;
    if (state) {
      const posArr = this.tetherMesh.geometry.attributes.position.array;
      const nrmArr = this.tetherMesh.geometry.attributes.normal.array;
      const colArr = this.tetherMesh.geometry.attributes.color.array;
      const cableR = 0.082;
      for (let ring = 0; ring < state.ringCount; ring++) {
        const pt = this.tetherPoints[ring];
        for (let side = 0; side <= state.radialSegments; side++) {
          const vOff = (ring * state.ringStride + side) * 3;
          const rx = nrmArr[vOff];
          const ry = nrmArr[vOff + 1];
          const rz = nrmArr[vOff + 2];
          posArr[vOff] = pt.x + rx * cableR;
          posArr[vOff + 1] = pt.y + ry * cableR;
          posArr[vOff + 2] = pt.z + rz * cableR;
          colArr[vOff] = 1.0;
          colArr[vOff + 1] = 1.0;
          colArr[vOff + 2] = 1.0;
        }
      }
      this.tetherMesh.geometry.attributes.position.needsUpdate = true;
      this.tetherMesh.geometry.attributes.color.needsUpdate = true;
    }
  }

  applySpinalRigTransforms(chonk, bodyLenScale) {
    const poseDamp = Math.max(
      0.0,
      (1.0 - (this.groomAmount || 0) * 0.94) * (1.0 - (this.rearAmount || 0) * 0.92)
    );
    const flex = {
      bend: this.spineFlex.bend * poseDamp,
      sCurve: this.spineFlex.sCurve * poseDamp,
      arch: this.spineFlex.arch * poseDamp,
      stretch: this.spineFlex.stretch * poseDamp,
    };
    this.effectiveSpineFlex = flex;
    const uniformsPayload = {
      uSpineBend: flex.bend,
      uSpineSCurve: flex.sCurve,
      uSpineArch: flex.arch,
      uAxialStretch: flex.stretch,
      uSpineHalfLen: PROP.bodyLength,
    };
    for (const mat of this.spinalMaterials || []) {
      annolidShaders.updateUniforms(mat, uniformsPayload);
    }

    const spNeck = evaluateSpinePoint(0, this.neckRestPosition.y / Math.max(0.4, chonk), this.neckRestPosition.z, flex, PROP.bodyLength);
    this.neckMesh.position.set(spNeck.x * chonk, spNeck.y * chonk, spNeck.z * bodyLenScale);
    this.neckMesh.rotation.set(
      this.neckRestRotation.x,
      this.neckRestRotation.y + spNeck.yaw,
      this.neckRestRotation.z
    );

    const spHead = evaluateSpinePoint(0, this.headRestPosition.y / Math.max(0.4, chonk), this.headRestPosition.z, flex, PROP.bodyLength);
    this.headGroup.position.set(spHead.x * chonk, spHead.y * chonk, spHead.z * bodyLenScale);
    this.spineHeadYaw = spHead.yaw;

    const spArmL = evaluateSpinePoint(this.armRestPosL.x, this.armRestPosL.y, this.armRestZ, flex, PROP.bodyLength);
    const spArmR = evaluateSpinePoint(this.armRestPosR.x, this.armRestPosR.y, this.armRestZ, flex, PROP.bodyLength);
    this.armL.position.set(spArmL.x * chonk, spArmL.y, spArmL.z * bodyLenScale);
    this.armR.position.set(spArmR.x * chonk, spArmR.y, spArmR.z * bodyLenScale);
    this.armL.rotation.y = spArmL.yaw;
    this.armR.rotation.y = spArmR.yaw;

    const spHindL = evaluateSpinePoint(this.hindRestPosL.x, this.hindRestPosL.y, this.hindRestZ, flex, PROP.bodyLength);
    const spHindR = evaluateSpinePoint(this.hindRestPosR.x, this.hindRestPosR.y, this.hindRestZ, flex, PROP.bodyLength);
    this.hindL.position.set(spHindL.x * chonk, spHindL.y, spHindL.z * bodyLenScale);
    this.hindR.position.set(spHindR.x * chonk, spHindR.y, spHindR.z * bodyLenScale);
    this.hindL.rotation.y = spHindL.yaw * 0.55;
    this.hindR.rotation.y = spHindR.yaw * 0.55;

    if (this.bellyFurAnchor) {
      const spBelly = evaluateSpinePoint(0, -0.43, this.bellyRestZ, flex, PROP.bodyLength);
      this.bellyFurAnchor.position.set(spBelly.x * chonk, spBelly.y, spBelly.z * bodyLenScale);
      this.bellyFurAnchor.rotation.z = -spBelly.yaw;
    }
    if (this.poseNodes.spine1) {
      const sp1 = evaluateSpinePoint(0, 0.9, 0.2, flex, PROP.bodyLength);
      this.poseNodes.spine1.position.set(sp1.x * chonk, sp1.y * chonk, sp1.z * bodyLenScale);
    }
    if (this.poseNodes.spine2) {
      const sp2 = evaluateSpinePoint(0, 0.8, -PROP.bodyLength * 0.41, flex, PROP.bodyLength);
      this.poseNodes.spine2.position.set(sp2.x * chonk, sp2.y * chonk, sp2.z * bodyLenScale);
    }
    if (this.poseNodes.tailbase) {
      const spTb = evaluateSpinePoint(0, 0.3, -PROP.bodyLength * 0.93, flex, PROP.bodyLength);
      this.poseNodes.tailbase.position.set(spTb.x * chonk, spTb.y * chonk, spTb.z * bodyLenScale);
    }
  }

  syncLiveProportions(otherMice = []) {
    const chonk = this.guiParams.chonkiness || 1.0;
    const bodyLenScale = this.guiParams.bodyLength || 1.0;
    this.bodyMesh.scale.set(chonk, chonk, bodyLenScale);
    this.neckMesh.scale.set(chonk, chonk, 1.0);
    this.neckRestPosition.z = PROP.bodyLength * 0.78;
    this.headRestPosition.y = 0.18 * chonk;
    this.headRestPosition.z = PROP.bodyLength * 0.73;
    this.head.position.set(0, 0.14 * chonk, PROP.bodyLength * 0.20);
    this.head.scale.set(
      this.guiParams.headSize * 0.92,
      this.guiParams.headSize * 0.94,
      this.guiParams.headSize * 1.03
    );
    this.placeHeadFeatures();
    this.resetNoseScale();
    this.armRestZ = PROP.bodyLength * 0.70;
    this.hindRestZ = -PROP.bodyLength * 0.36;
    this.bellyRestZ = PROP.bodyLength * 0.56;
    this.armRestPosL.set(0.38 * PROP.shoulderWidth / 0.90, -0.11, this.armRestZ);
    this.armRestPosR.set(-0.38 * PROP.shoulderWidth / 0.90, -0.11, this.armRestZ);
    this.hindRestPosL.set(0.62 * PROP.haunchWidth / 1.04, -0.04, this.hindRestZ);
    this.hindRestPosR.set(-0.62 * PROP.haunchWidth / 1.04, -0.04, this.hindRestZ);
    this.applySpinalRigTransforms(chonk, bodyLenScale);
    this.updateTailGeometry(otherMice, 0);
    this.updateHeadstageTether();
    if (this.trackingGroup.visible) this.updateTrackingGeometry();
  }

  rotateIkJointToward(joint, axisName, contact, target, baseRotation, maxOffset, scratch) {
    joint.updateWorldMatrix(true, false);
    contact.updateWorldMatrix(true, false);
    joint.getWorldPosition(scratch.jointWorld);
    contact.getWorldPosition(scratch.effectorWorld);
    scratch.toEffector.copy(scratch.effectorWorld).sub(scratch.jointWorld);
    scratch.toTarget.copy(target).sub(scratch.jointWorld);
    scratch.axisWorld
      .set(axisName === 'x' ? 1 : 0, 0, axisName === 'z' ? 1 : 0)
      .transformDirection(joint.matrixWorld);
    scratch.toEffector.addScaledVector(
      scratch.axisWorld,
      -scratch.toEffector.dot(scratch.axisWorld)
    );
    scratch.toTarget.addScaledVector(
      scratch.axisWorld,
      -scratch.toTarget.dot(scratch.axisWorld)
    );
    if (scratch.toEffector.lengthSq() < 1e-8 || scratch.toTarget.lengthSq() < 1e-8) return;
    scratch.toEffector.normalize();
    scratch.toTarget.normalize();
    scratch.cross.crossVectors(scratch.toEffector, scratch.toTarget);
    const angle = Math.atan2(
      scratch.axisWorld.dot(scratch.cross),
      THREE.MathUtils.clamp(scratch.toEffector.dot(scratch.toTarget), -1, 1)
    );
    const nextRotation = joint.rotation[axisName]
      + THREE.MathUtils.clamp(angle * 0.72, -0.16, 0.16);
    joint.rotation[axisName] = THREE.MathUtils.clamp(
      nextRotation,
      baseRotation - maxOffset,
      baseRotation + maxOffset
    );
  }

  blendForelimbPose(chain, pose, amount = 1) {
    const shoulder = chain._upperArm;
    shoulder.position.z = THREE.MathUtils.lerp(
      shoulder.position.z,
      pose.shoulderProtraction ?? 0,
      amount
    );
    shoulder.rotation.x = THREE.MathUtils.lerp(
      shoulder.rotation.x,
      pose.shoulderX,
      amount
    );
    shoulder.rotation.z = THREE.MathUtils.lerp(
      shoulder.rotation.z,
      pose.shoulderZ,
      amount
    );
    chain._forearm.rotation.x = THREE.MathUtils.lerp(
      chain._forearm.rotation.x,
      pose.elbowX,
      amount
    );
    chain._wrist.rotation.x = THREE.MathUtils.lerp(
      chain._wrist.rotation.x,
      pose.wristX,
      amount
    );
    chain._paw.rotation.x = THREE.MathUtils.lerp(
      chain._paw.rotation.x,
      pose.pawX,
      amount
    );
  }

  getGroomingMotion() {
    const washCycle = this.grooming.elapsed * 8.6;
    const leftWashStroke = 0.5 + 0.5 * Math.sin(washCycle);
    const rightWashStroke = 0.5 + 0.5 * Math.sin(washCycle + 0.72);
    const washStroke = (leftWashStroke + rightWashStroke) * 0.5;
    const pawLead = leftWashStroke - rightWashStroke;
    const pawLick = Math.pow(
      Math.max(0, 1.0 - Math.max(leftWashStroke, rightWashStroke)),
      1.35
    );
    const flankLick = 0.5 + 0.5 * Math.sin(this.grooming.elapsed * 7.4);
    return {
      leftWashStroke,
      rightWashStroke,
      washStroke,
      pawLead,
      pawLick,
      flankLick,
      flankPulse: flankLick * 2.0 - 1.0,
    };
  }

  solveIkConstraint(
    plant,
    contact,
    target,
    {
      iterations = 8,
      jointXOffsets = [0.48, 0.70, 0.72],
      jointXLimits = null,
      rootZOffset = 0.28,
    } = {}
  ) {
    const scratch = {
      jointWorld: new THREE.Vector3(),
      effectorWorld: new THREE.Vector3(),
      toEffector: new THREE.Vector3(),
      toTarget: new THREE.Vector3(),
      axisWorld: new THREE.Vector3(),
      cross: new THREE.Vector3(),
    };
    const baselines = plant.joints.map(joint => ({
      x: joint.rotation.x,
      z: joint.rotation.z,
    }));
    for (let iteration = 0; iteration < iterations; iteration++) {
      for (let index = 0; index < plant.joints.length; index++) {
        this.rotateIkJointToward(
          plant.joints[index],
          'x',
          contact,
          target,
          baselines[index].x,
          jointXOffsets[index],
          scratch
        );
        if (jointXLimits?.[index]) {
          plant.joints[index].rotation.x = THREE.MathUtils.clamp(
            plant.joints[index].rotation.x,
            jointXLimits[index][0],
            jointXLimits[index][1]
          );
        }
      }
      const rootIndex = plant.joints.length - 1;
      this.rotateIkJointToward(
        plant.joints[rootIndex],
        'z',
        contact,
        target,
        baselines[rootIndex].z,
        rootZOffset,
        scratch
      );
    }
    contact.updateWorldMatrix(true, false);
    contact.getWorldPosition(scratch.effectorWorld);
    return {
      error: scratch.effectorWorld.distanceTo(target),
    };
  }

  updateForelimbExtensionMetrics() {
    this.maximumForelimbExtensionRatio = 0;
    const shoulderPosition = new THREE.Vector3();
    const wristPosition = new THREE.Vector3();
    const contactPosition = new THREE.Vector3();
    for (const plant of this.footPlants.slice(2)) {
      const shoulder = plant.joints[plant.joints.length - 1];
      const wrist = plant.joints[0];
      const chain = plant.side > 0 ? this.armL : this.armR;
      shoulder.updateWorldMatrix(true, false);
      wrist.updateWorldMatrix(true, false);
      plant.contact.updateWorldMatrix(true, false);
      shoulder.getWorldPosition(shoulderPosition);
      wrist.getWorldPosition(wristPosition);
      plant.contact.getWorldPosition(contactPosition);
      const boneLength = chain._segmentLengths.humerus
        + chain._segmentLengths.radiusUlna;
      plant.extensionRatio = shoulderPosition.distanceTo(wristPosition)
        / boneLength;
      plant.pawReachRatio = shoulderPosition.distanceTo(contactPosition)
        / plant.maximumReach;
      this.maximumForelimbExtensionRatio = Math.max(
        this.maximumForelimbExtensionRatio,
        plant.extensionRatio
      );
    }
  }

  solveForelimbBehaviorConstraints() {
    this.groomingPawContactError = 0;
    this.groomingSupportError = 0;
    this.groomingReachDeficit = 0;
    this.groomingTargetReachRatio = 0;
    const groomAmount = THREE.MathUtils.clamp(
      this.groomAmount / Math.max(this.grooming.peakAmount, 0.01),
      0,
      1
    );
    if (this.grooming.mode === 'face-wash' && groomAmount > 0.08) {
      const motion = this.getGroomingMotion();
      const targets = [
        { plant: this.footPlants[2], stroke: motion.leftWashStroke, side: 1 },
        { plant: this.footPlants[3], stroke: motion.rightWashStroke, side: -1 },
      ];
      const desiredTarget = new THREE.Vector3();
      const constrainedTarget = new THREE.Vector3();
      const currentContact = new THREE.Vector3();
      const shoulderPosition = new THREE.Vector3();
      const constraintAmount = THREE.MathUtils.smoothstep(
        groomAmount,
        0.35,
        0.95
      );
      this.mesh.updateMatrixWorld(true);
      for (const { plant, stroke, side } of targets) {
        const contact = plant.paw._groomingContact;
        desiredTarget.set(
          side * THREE.MathUtils.lerp(0.15, 0.29, stroke),
          THREE.MathUtils.lerp(-0.27, 0.07, stroke) - motion.pawLick * 0.035,
          THREE.MathUtils.lerp(0.48, 0.44, stroke) + motion.pawLick * 0.025
        );
        plant.constraintTargetLocal.copy(desiredTarget);
        this.head.localToWorld(desiredTarget);
        contact.updateWorldMatrix(true, false);
        contact.getWorldPosition(currentContact);
        constrainedTarget.copy(currentContact).lerp(desiredTarget, constraintAmount);
        const shoulder = plant.joints[plant.joints.length - 1];
        shoulder.updateWorldMatrix(true, false);
        shoulder.getWorldPosition(shoulderPosition);
        this.groomingTargetReachRatio = Math.max(
          this.groomingTargetReachRatio,
          shoulderPosition.distanceTo(desiredTarget) / plant.maximumReach
        );
        this.groomingReachDeficit = Math.max(
          this.groomingReachDeficit,
          shoulderPosition.distanceTo(desiredTarget) - plant.maximumReach * 0.97
        );
        const elbowPole = THREE.MathUtils.lerp(
          plant.joints[1].rotation.x,
          -1.0,
          constraintAmount
        );
        this.solveIkConstraint(plant, contact, constrainedTarget, {
          iterations: 14,
          jointXOffsets: [0.72, 0.78, 0.92],
          jointXLimits: [
            [-0.20, 1.10],
            [elbowPole, elbowPole],
            [-1.65, -0.20],
          ],
          rootZOffset: 0.42,
        });
        contact.updateWorldMatrix(true, false);
        contact.getWorldPosition(currentContact);
        plant.constraintError.copy(currentContact).sub(desiredTarget);
        plant.constraintContactLocal.copy(currentContact);
        this.head.worldToLocal(plant.constraintContactLocal);
        this.groomingPawContactError = Math.max(
          this.groomingPawContactError,
          currentContact.distanceTo(desiredTarget) * constraintAmount
        );
        if (constraintAmount > 0.02) {
          plant.planted = false;
          plant.exhausted = false;
        }
      }
      this.groomingReachDeficit = Math.max(0, this.groomingReachDeficit);
    } else if (this.grooming.mode === 'flank-groom' && groomAmount > 0.08) {
      const loadedSupportSide = -this.grooming.side;
      const supportPlant = this.footPlants.slice(2).find(
        plant => plant.side === loadedSupportSide
      );
      this.groomingSupportError = supportPlant?.planted
        ? Math.hypot(supportPlant.slip, supportPlant.groundError)
        : 1.0;
    }
    this.updateForelimbExtensionMetrics();
  }

  captureRearFootAnchors() {
    const state = this.rearing;
    for (let index = 0; index < 2; index++) {
      const plant = this.footPlants[index];
      plant.contact.updateWorldMatrix(true, false);
      plant.contact.getWorldPosition(state.hindAnchors[index]);
      state.hindAnchors[index].y = 0.055;
      plant.anchor.copy(state.hindAnchors[index]);
      plant.planted = true;
      plant.exhausted = false;
    }
    state.rearFeetPlanted = true;
  }

  solveRearBalanceConstraints(dt = 0.016) {
    const state = this.rearing;
    state.lastConstraintDt = dt;
    this.maximumRearSupportError = 0;
    if (state.mode === 'none' || state.angle < 0.012) {
      state.hindSupportError = 0;
      state.hindSupportHorizontalError = 0;
      state.hindSupportVerticalError = 0;
      state.centerOfMassProjectionError = 0;
      return;
    }
    const poseAmount = THREE.MathUtils.clamp(
      state.angle / Math.max(state.maxAngle, 0.01),
      0,
      1
    );
    if (state.phase === 'settle' && poseAmount < 0.82) {
      state.rearSupportReleased = true;
      state.rearFeetPlanted = false;
      state.hindSupportError = 0;
      state.hindSupportHorizontalError = 0;
      state.hindSupportVerticalError = 0;
      for (const plant of this.footPlants.slice(0, 2)) {
        plant.planted = false;
        plant.exhausted = false;
      }
      return;
    }
    if (state.rearSupportReleased) return;
    if (!state.rearFeetPlanted) this.captureRearFootAnchors();

    const placementAmount = state.phase === 'settle' ? 1.0 : poseAmount;
    if (state.mode === 'wall-supported' && state.maximumBaseAdvance > 0) {
      const targetAdvance = state.maximumBaseAdvance
        * THREE.MathUtils.smoothstep(placementAmount, 0.25, 0.95);
      let nextAdvance;
      if (guiParams.isPaused) {
        nextAdvance = targetAdvance;
        state.baseAdvanceVelocity = 0;
      } else {
        const advanceAcceleration = 34.0 * (targetAdvance - state.baseAdvance)
          - 10.5 * state.baseAdvanceVelocity;
        state.baseAdvanceVelocity += advanceAcceleration * Math.min(dt, 0.05);
        nextAdvance = THREE.MathUtils.clamp(
          state.baseAdvance + state.baseAdvanceVelocity * Math.min(dt, 0.05),
          0,
          state.maximumBaseAdvance * 1.02
        );
      }
      const advanceDelta = nextAdvance - state.baseAdvance;
      state.baseAdvance = nextAdvance;
      for (const anchor of state.hindAnchors) {
        anchor.x += state.supportDirection.x * advanceDelta;
        anchor.z += state.supportDirection.y * advanceDelta;
      }
    }
    if (state.maximumHindStanceOffset > 0) {
      const targetStanceOffset = state.maximumHindStanceOffset
        * THREE.MathUtils.smoothstep(placementAmount, 0.15, 0.65);
      let nextStanceOffset;
      if (guiParams.isPaused) {
        nextStanceOffset = targetStanceOffset;
        state.hindStanceVelocity = 0;
      } else {
        const stanceAcceleration = 30.0 * (
          targetStanceOffset - state.hindStanceOffset
        ) - 10.0 * state.hindStanceVelocity;
        state.hindStanceVelocity += stanceAcceleration * Math.min(dt, 0.05);
        nextStanceOffset = THREE.MathUtils.clamp(
          state.hindStanceOffset + state.hindStanceVelocity * Math.min(dt, 0.05),
          0,
          state.maximumHindStanceOffset * 1.02
        );
      }
      const stanceDelta = nextStanceOffset - state.hindStanceOffset;
      state.hindStanceOffset = nextStanceOffset;
      const lateralX = this.heading.y;
      const lateralZ = -this.heading.x;
      // Draw the rear contacts under the COM as the haunches adduct.
      state.hindAnchors[0].x -= lateralX * stanceDelta;
      state.hindAnchors[0].z -= lateralZ * stanceDelta;
      state.hindAnchors[1].x += lateralX * stanceDelta;
      state.hindAnchors[1].z += lateralZ * stanceDelta;
    }

    const leftContact = new THREE.Vector3();
    const rightContact = new THREE.Vector3();
    this.footPlants[0].contact.updateWorldMatrix(true, false);
    this.footPlants[1].contact.updateWorldMatrix(true, false);
    this.footPlants[0].contact.getWorldPosition(leftContact);
    this.footPlants[1].contact.getWorldPosition(rightContact);
    const currentCenter = leftContact.add(rightContact).multiplyScalar(0.5);
    const targetCenter = state.hindAnchors[0].clone()
      .add(state.hindAnchors[1])
      .multiplyScalar(0.5);
    const correction = targetCenter.sub(currentCenter);
    // Ground contact is a positional constraint. Resolve it as a bounded
    // impulse so the body rotates around the rear support polygon instead
    // of being translated upward by the animation pose.
    const horizontalLength = Math.hypot(correction.x, correction.z);
    const maximumHorizontalCorrection = guiParams.isPaused
      ? Infinity
      : 0.18 + poseAmount * 0.16;
    if (horizontalLength > maximumHorizontalCorrection) {
      const scale = maximumHorizontalCorrection / horizontalLength;
      correction.x *= scale;
      correction.z *= scale;
    }
    this.mesh.position.x += correction.x;
    this.mesh.position.z += correction.z;
    this.rig.position.y += THREE.MathUtils.clamp(
      correction.y,
      guiParams.isPaused ? -2.0 : -0.42,
      guiParams.isPaused ? 2.0 : 0.42
    );

    for (let index = 0; index < 2; index++) {
      const plant = this.footPlants[index];
      plant.anchor.copy(state.hindAnchors[index]);
      plant.planted = true;
      plant.exhausted = false;
    }
  }

  solveFootPlantConstraints(dt = 0.016) {
    const scratch = {
      jointWorld: new THREE.Vector3(),
      effectorWorld: new THREE.Vector3(),
      toEffector: new THREE.Vector3(),
      toTarget: new THREE.Vector3(),
      axisWorld: new THREE.Vector3(),
      cross: new THREE.Vector3(),
    };
    this.maximumFootSlip = 0;
    this.maximumPawGroundError = 0;

    for (const foot of this.footPlants) {
      const flankGroomSupport = foot.isFore
        && this.grooming.mode === 'flank-groom'
        && this.contactFrame.rear < 0.08;
      const faceWashRecovery = foot.isFore
        && this.grooming.mode === 'face-wash'
        && this.grooming.phase === 'settle'
        && this.contactFrame.groom < 0.12;
      const mayTouchGround = !foot.isFore || (
        this.contactFrame.rear < 0.08
          && (
            this.contactFrame.groom < 0.08
              || flankGroomSupport
              || faceWashRecovery
          )
      );
      // Stance timing comes from the limb CPG; posture controllers (rearing,
      // grooming) own the supporting limbs while they are engaged.
      const postureHold = this.contactFrame.rear >= 0.08
        || this.contactFrame.groom >= 0.08;
      const inStance = mayTouchGround && (
        postureHold || this.contactFrame.limbStance[foot.limb] !== false
      );
      if (!inStance) {
        foot.planted = false;
        foot.exhausted = false;
        foot.slip = 0;
        foot.groundError = 0;
        foot.rejectionReason = 'not-stance';
        continue;
      }
      if (foot.exhausted) {
        const mayReplantDuringBodyPose = flankGroomSupport
          || faceWashRecovery
          || (
            !foot.isFore && (
              this.contactFrame.rear >= 0.08 || this.contactFrame.groom >= 0.08
            )
          );
        const mayRecoverAtRest = !this.contactFrame.isMoving
          && this.contactFrame.rear < 0.08
          && this.contactFrame.groom < 0.08;
        if (!mayReplantDuringBodyPose && !mayRecoverAtRest) continue;
        foot.exhausted = false;
        foot.planted = false;
      }

      foot.contact.updateWorldMatrix(true, false);
      foot.contact.getWorldPosition(scratch.effectorWorld);
      const justPlanted = !foot.planted;
      if (justPlanted) {
        foot.anchor.copy(scratch.effectorWorld);
        foot.anchor.y = 0.055;
        foot.planted = true;
      }

      const rootJoint = foot.joints[foot.joints.length - 1];
      rootJoint.updateWorldMatrix(true, false);
      rootJoint.getWorldPosition(scratch.jointWorld);
      const rearSupportFoot = !foot.isFore && this.contactFrame.rear >= 0.08;
      const groomingHindSupport = !foot.isFore
        && this.contactFrame.groom >= 0.08;
      const maximumReach = foot.isFore
        ? foot.maximumReach * 0.985
        : (rearSupportFoot ? 1.78 : (groomingHindSupport ? 1.68 : 1.52));
      if (justPlanted) {
        // Place a new stance inside the limb's reachable ground circle.
        // Simply projecting a swinging paw vertically onto the floor can
        // ask the articulated leg to stretch beyond its length.
        const height = scratch.jointWorld.y - foot.anchor.y;
        const groundReach = Math.sqrt(Math.max(0, maximumReach * maximumReach - height * height)) * 0.97;
        const dx = foot.anchor.x - scratch.jointWorld.x;
        const dz = foot.anchor.z - scratch.jointWorld.z;
        const horizontal = Math.hypot(dx, dz);
        if (horizontal > groundReach && horizontal > 1e-8) {
          foot.anchor.x = scratch.jointWorld.x + dx * groundReach / horizontal;
          foot.anchor.z = scratch.jointWorld.z + dz * groundReach / horizontal;
        }
      }
      const targetDistance = scratch.jointWorld.distanceTo(foot.anchor);
      foot.reachRatio = targetDistance / maximumReach;
      if (targetDistance > maximumReach) {
        foot.planted = false;
        foot.exhausted = true;
        foot.slip = 0;
        foot.groundError = 0;
        foot.rejectionReason = 'reach';
        continue;
      }

      const baselines = foot.joints.map(joint => ({
        x: joint.rotation.x,
        z: joint.rotation.z,
      }));
      // Allow difficult stance targets to converge, but stop inexpensive
      // near-rest solves as soon as the contact is within 0.005 scene units.
      const iterationCount = flankGroomSupport ? 20 : 18;
      for (let iteration = 0; iteration < iterationCount; iteration++) {
        for (let index = 0; index < foot.joints.length; index++) {
          const joint = foot.joints[index];
          const maxOffset = rearSupportFoot || groomingHindSupport
            ? (index === 0 ? 0.62 : (index === 1 ? 0.78 : 0.68))
            : (flankGroomSupport
              ? (index === 0 ? 0.58 : (index === 1 ? 0.78 : 0.72))
              : (index === 0 ? 0.52 : (index === 1 ? 0.66 : 0.58)));
          this.rotateIkJointToward(
            joint,
            'x',
            foot.contact,
            foot.anchor,
            baselines[index].x,
            maxOffset,
            scratch
          );
          if (foot.isFore && index === 1) {
            const maximumElbowExtension = flankGroomSupport ? -0.46 : -0.50;
            joint.rotation.x = Math.min(
              joint.rotation.x,
              maximumElbowExtension
            );
          }
        }
        const rootIndex = foot.joints.length - 1;
        this.rotateIkJointToward(
          foot.joints[rootIndex],
          'z',
          foot.contact,
          foot.anchor,
          baselines[rootIndex].z,
          foot.isFore
            ? 0.34
            : (rearSupportFoot ? 0.48 : (groomingHindSupport ? 0.38 : 0.26)),
          scratch
        );
        foot.contact.updateWorldMatrix(true, false);
        foot.contact.getWorldPosition(scratch.effectorWorld);
        if (scratch.effectorWorld.distanceToSquared(foot.anchor) < 0.000025) break;
      }

      foot.contact.updateWorldMatrix(true, false);
      foot.contact.getWorldPosition(scratch.effectorWorld);
      if (justPlanted) {
        foot.anchor.x = scratch.effectorWorld.x;
        foot.anchor.z = scratch.effectorWorld.z;
      }
      const flankRepositioningPaw = flankGroomSupport
        && foot.side === this.grooming.side;
      if (flankRepositioningPaw && !justPlanted) {
        const regripAmount = 1.0 - Math.exp(-36.0 * Math.min(dt, 0.05));
        foot.anchor.x = THREE.MathUtils.lerp(
          foot.anchor.x,
          scratch.effectorWorld.x,
          regripAmount
        );
        foot.anchor.z = THREE.MathUtils.lerp(
          foot.anchor.z,
          scratch.effectorWorld.z,
          regripAmount
        );
      }
      foot.slip = Math.hypot(
        scratch.effectorWorld.x - foot.anchor.x,
        scratch.effectorWorld.z - foot.anchor.z
      );
      foot.groundError = Math.abs(scratch.effectorWorld.y - foot.anchor.y);
      foot.candidateSlip = foot.slip;
      foot.candidateGroundError = foot.groundError;
      const poseSupportFoot = rearSupportFoot
        || groomingHindSupport
        || flankGroomSupport;
      // Release an unsupportable stance instead of keeping a visibly sliding
      // paw planted. Candidate errors remain in telemetry for diagnosis.
      const maximumSlip = poseSupportFoot ? 0.11 : STANCE_CONTACT_LIMITS.slip;
      const maximumGroundError = poseSupportFoot ? 0.12 : STANCE_CONTACT_LIMITS.height;
      if (foot.slip > maximumSlip || foot.groundError > maximumGroundError) {
        foot.planted = false;
        foot.exhausted = true;
        foot.slip = 0;
        foot.groundError = 0;
        foot.rejectionReason = 'contact';
        continue;
      }
      foot.rejectionReason = 'none';
      this.maximumFootSlip = Math.max(this.maximumFootSlip, foot.slip);
      this.maximumPawGroundError = Math.max(
        this.maximumPawGroundError,
        foot.groundError
      );
    }
  }

  updateTailGeometry(otherMice = [], dt = 0) {
    const { lateralAccel, turnRate, forwardAccel } = this.contactFrame;
    const clampedLatAccel = THREE.MathUtils.clamp(lateralAccel || 0, -18.0, 18.0);
    const clampedTurnRate = THREE.MathUtils.clamp(turnRate || 0, -12.0, 12.0);
    const clampedFwdAccel = THREE.MathUtils.clamp(forwardAccel || 0, -18.0, 18.0);
    const speedNorm = Math.min(this.velocity.length() / Math.max(this.maxSpeed, 0.1), 1.0);
    const restDrape = (1.0 - speedNorm * 0.55) * 0.34 + this.rearAmount * 0.12;
    const swayPolarity = (this.trackId || 1) % 2 === 0 ? -1 : 1;
    const chonk = this.guiParams.chonkiness || 1.0;
    const bodyLenScale = this.guiParams.bodyLength || 1.0;
    const spSacral = evaluateSpinePoint(
      0,
      -0.16 / Math.max(0.4, chonk),
      -PROP.bodyLength * 0.90,
      this.spineFlex,
      PROP.bodyLength
    );
    const tailBase = new THREE.Vector3(
      spSacral.x * chonk,
      spSacral.y * chonk,
      spSacral.z * bodyLenScale
    );
    this.rig.updateMatrix();
    tailBase.applyMatrix4(this.rig.matrix);
    this.mesh.updateMatrixWorld(true);

    const tailBaseWorld = tailBase.clone();
    this.mesh.localToWorld(tailBaseWorld);
    const tailLenMul = this.guiParams.tailLength;
    const segLen = this.tailSegLen * tailLenMul;

    // Build ideal caudal rest/muscle curve in mesh-local space aligned with sacral spine yaw
    const sacralYaw = spSacral.yaw;
    const cosSac = Math.cos(sacralYaw);
    const sinSac = Math.sin(sacralYaw);
    const idealLocal = [];
    for (let i = 0; i <= this.tailCount; i++) {
      const p = i / this.tailCount;
      const lag = p * p;
      const transmit = Math.pow(p, 1.35);
      const midCurve = p * (1.0 - p);
      const travelWave = Math.sin(
        this.locomotionTime * 3.4 * this.motion.tailWaveFreqMul - p * 3.8
      ) * 0.10 * speedNorm * this.motion.tailWaveAmpMul * transmit;
      const bodyForceX = (
        this.bodyDyn.roll * 0.34 + clampedLatAccel * 0.012 - clampedTurnRate * 0.024
      ) * transmit;
      const bodyForceY = (
        this.bodyDyn.pitch * 0.32 - clampedFwdAccel * 0.008
      ) * transmit;
      const naturalSway = (
        Math.sin(p * Math.PI * 1.35) * (0.14 + this.rearAmount * 0.24)
        + Math.sin(p * Math.PI * 0.65) * (0.06 + this.rearAmount * 0.16)
      ) * swayPolarity * (1.0 - speedNorm * 0.45);
      const localX = this.tailDyn.sway * lag + bodyForceX + naturalSway + travelWave
        + midCurve * 0.16 * Math.sign(this.tailDyn.sway || bodyForceX || swayPolarity);
      const distalCurl = Math.pow(Math.max(0, p - 0.72) / 0.28, 2.0) * 0.030;
      let waveY = -i * PROP.tailSag * p - restDrape * Math.pow(p, 1.15)
        + this.tailDyn.lift * lag + bodyForceY + midCurve * 0.05 + distalCurl;
      waveY = Math.max(waveY, -1.35);
      const localZ = -i * segLen;
      idealLocal.push(new THREE.Vector3(
        tailBase.x + localX * cosSac + localZ * sinSac,
        tailBase.y + waveY,
        tailBase.z - localX * sinSac + localZ * cosSac
      ));
    }

    // World-space viscoelastic path-trailing chain: preserves S/J curves along the mouse's trajectory
    const rootJump = tailBaseWorld.distanceTo(this.tailPrevRootWorld);
    const shouldReinitWorld = !this.tailWorldPoints
      || this.tailWorldPoints.length !== this.tailCount + 1
      || rootJump > 1.35
      || CONTACT_QA === 'tail';
    const tailCurve = [];
    if (shouldReinitWorld) {
      this.tailWorldPoints = idealLocal.map(pt => {
        const wp = pt.clone();
        this.mesh.localToWorld(wp);
        return wp;
      });
      for (let i = 0; i <= this.tailCount; i++) {
        tailCurve.push(idealLocal[i].clone());
      }
    } else {
      const stepDt = THREE.MathUtils.clamp(dt, 0, 0.05);
      this.tailWorldPoints[0].copy(tailBaseWorld);
      const idealWorld = new THREE.Vector3();
      const segVec = new THREE.Vector3();
      if (!guiParams.isPaused && stepDt > 0) {
        for (let i = 1; i <= this.tailCount; i++) {
          const p = i / this.tailCount;
          idealWorld.copy(idealLocal[i]);
          this.mesh.localToWorld(idealWorld);

          // 1. Follow-the-leader advection along previous world trajectory
          segVec.copy(this.tailWorldPoints[i]).sub(this.tailWorldPoints[i - 1]);
          if (segVec.lengthSq() < 1e-8) {
            segVec.copy(idealWorld).sub(this.tailWorldPoints[i - 1]);
            if (segVec.lengthSq() < 1e-8) segVec.set(0, 0, -1);
          }
          segVec.normalize().multiplyScalar(segLen);
          this.tailWorldPoints[i].copy(this.tailWorldPoints[i - 1]).add(segVec);

          // 2. Proximal-to-distal caudal muscular tone (root aligns with sacrum; tip trails fluidly)
          const rate = THREE.MathUtils.lerp(13.5, 2.6, Math.pow(p, 0.68));
          const muscleBlend = (1.0 - Math.exp(-rate * stepDt))
            * THREE.MathUtils.lerp(0.82, 0.22, p);
          this.tailWorldPoints[i].lerp(idealWorld, muscleBlend);
        }
      }
      for (let i = 0; i <= this.tailCount; i++) {
        const lp = this.tailWorldPoints[i].clone();
        this.mesh.worldToLocal(lp);
        tailCurve.push(lp);
      }
    }
    this.tailPrevRootWorld.copy(tailBaseWorld);

    const avoidanceAxisX = this.heading.y;
    const avoidanceAxisZ = -this.heading.x;
    const obstacleBodies = otherMice
      .filter(mouse => mouse !== this)
      .map(mouse => {
        const ownerSide = (
          (this.mesh.position.x - mouse.mesh.position.x) * avoidanceAxisX
          + (this.mesh.position.z - mouse.mesh.position.z) * avoidanceAxisZ
        );
        const sideSign = Math.abs(ownerSide) > 0.20
          ? Math.sign(ownerSide)
          : ((this.trackId || 0) > (mouse.trackId || 0) ? -1 : 1);
        return {
          mouse,
          discs: getMouseCollisionDiscs(mouse),
          directionX: avoidanceAxisX * sideSign,
          directionZ: avoidanceAxisZ * sideSign,
        };
      });
    const obstacles = obstacleBodies.flatMap(body => body.discs);
    const worldPoint = new THREE.Vector3();
    const localPoint = new THREE.Vector3();
    const worldStart = new THREE.Vector3();
    const worldEnd = new THREE.Vector3();
    const delta = new THREE.Vector3();

    const tailRadiusAt = index => {
      const p = index / this.tailCount;
      return tailSurfaceRadiusAt(p, this.guiParams.tailThickness) + 0.035;
    };
    const clampPointToArena = (point, index) => {
      worldPoint.copy(point);
      this.mesh.localToWorld(worldPoint);
      const surfaceRadius = tailSurfaceRadiusAt(
        index / this.tailCount,
        this.guiParams.tailThickness
      );
      const horizontalMargin = surfaceRadius + 0.018;
      const minimumY = surfaceRadius + 0.012;
      let changed = false;
      if (worldPoint.y < minimumY) {
        worldPoint.y = minimumY;
        changed = true;
      }
      if (index > 0) {
        const clampedX = THREE.MathUtils.clamp(
          worldPoint.x,
          -ENCLOSURE_HALF_X + horizontalMargin,
          ENCLOSURE_HALF_X - horizontalMargin
        );
        const clampedZ = THREE.MathUtils.clamp(
          worldPoint.z,
          -ENCLOSURE_HALF_Z + horizontalMargin,
          ENCLOSURE_HALF_Z - horizontalMargin
        );
        changed = changed || clampedX !== worldPoint.x || clampedZ !== worldPoint.z;
        worldPoint.x = clampedX;
        worldPoint.z = clampedZ;
      }
      if (!changed) return false;
      localPoint.copy(worldPoint);
      this.mesh.worldToLocal(localPoint);
      point.copy(localPoint);
      return true;
    };
    const projectPointOutsideBodies = (point, index) => {
      worldPoint.copy(point);
      this.mesh.localToWorld(worldPoint);
      const tailRadius = tailRadiusAt(index);
      let changed = false;
      for (const body of obstacleBodies) {
        const penetratesBody = body.discs.some(disc => (
          Math.hypot(worldPoint.x - disc.x, worldPoint.z - disc.z)
            < disc.radius + tailRadius
        ));
        if (!penetratesBody) continue;

        // All constrained vertices choose one side of a subject. Projecting
        // along the tail's lateral axis creates a continuous route around
        // the silhouette instead of putting vertices on opposite sides.
        let requiredShift = 0;
        for (const disc of body.discs) {
          const relativeX = worldPoint.x - disc.x;
          const relativeZ = worldPoint.z - disc.z;
          const alongRay = relativeX * body.directionX
            + relativeZ * body.directionZ;
          const perpendicularSq = Math.max(
            0,
            relativeX * relativeX + relativeZ * relativeZ - alongRay * alongRay
          );
          const requiredDistance = disc.radius + tailRadius;
          if (perpendicularSq >= requiredDistance * requiredDistance) continue;
          const farIntersection = -alongRay
            + Math.sqrt(requiredDistance * requiredDistance - perpendicularSq);
          if (farIntersection > requiredShift) requiredShift = farIntersection;
        }
        worldPoint.x += body.directionX * (requiredShift + 0.006);
        worldPoint.z += body.directionZ * (requiredShift + 0.006);
        changed = true;
      }
      if (!changed) return false;
      localPoint.copy(worldPoint);
      this.mesh.worldToLocal(localPoint);
      point.x = localPoint.x;
      point.z = localPoint.z;
      return true;
    };

    const getSegmentContact = (start, end, index) => {
      worldStart.copy(start);
      worldEnd.copy(end);
      this.mesh.localToWorld(worldStart);
      this.mesh.localToWorld(worldEnd);
      const segmentX = worldEnd.x - worldStart.x;
      const segmentZ = worldEnd.z - worldStart.z;
      const segmentLengthSq = segmentX * segmentX + segmentZ * segmentZ;
      const tailRadius = Math.max(tailRadiusAt(index), tailRadiusAt(index + 1));
      let best = { clearance: Infinity, body: null, disc: null, closestX: 0, closestZ: 0, requiredDistance: 0 };
      for (const body of obstacleBodies) {
        for (const disc of body.discs) {
          const toCenterX = disc.x - worldStart.x;
          const toCenterZ = disc.z - worldStart.z;
          const along = segmentLengthSq > 1e-8
            ? THREE.MathUtils.clamp(
              (toCenterX * segmentX + toCenterZ * segmentZ) / segmentLengthSq,
              0,
              1
            )
            : 0;
          const closestX = worldStart.x + segmentX * along;
          const closestZ = worldStart.z + segmentZ * along;
          const clearance = Math.hypot(closestX - disc.x, closestZ - disc.z)
            - disc.radius - tailRadius;
          if (clearance < best.clearance) {
            best = {
              clearance,
              body,
              disc,
              closestX,
              closestZ,
              requiredDistance: disc.radius + tailRadius,
            };
          }
        }
      }
      return best;
    };

    const projectSegmentOutsideBodies = (start, end, index) => {
      worldStart.copy(start);
      worldEnd.copy(end);
      this.mesh.localToWorld(worldStart);
      this.mesh.localToWorld(worldEnd);
      const tailRadius = Math.max(tailRadiusAt(index), tailRadiusAt(index + 1));
      let changed = false;

      for (const body of obstacleBodies) {
        const segmentX = worldEnd.x - worldStart.x;
        const segmentZ = worldEnd.z - worldStart.z;
        const segmentLengthSq = segmentX * segmentX + segmentZ * segmentZ;
        let requiredShift = 0;
        for (const disc of body.discs) {
          const toCenterX = disc.x - worldStart.x;
          const toCenterZ = disc.z - worldStart.z;
          const alongSegment = segmentLengthSq > 1e-8
            ? THREE.MathUtils.clamp(
              (toCenterX * segmentX + toCenterZ * segmentZ) / segmentLengthSq,
              0,
              1
            )
            : 0;
          const closestX = worldStart.x + segmentX * alongSegment;
          const closestZ = worldStart.z + segmentZ * alongSegment;
          const relativeX = closestX - disc.x;
          const relativeZ = closestZ - disc.z;
          const requiredDistance = disc.radius + tailRadius;
          if (Math.hypot(relativeX, relativeZ) >= requiredDistance) continue;
          const alongDirection = relativeX * body.directionX
            + relativeZ * body.directionZ;
          const perpendicularSq = Math.max(
            0,
            relativeX * relativeX + relativeZ * relativeZ
              - alongDirection * alongDirection
          );
          if (perpendicularSq >= requiredDistance * requiredDistance) continue;
          const axisExit = -alongDirection + Math.sqrt(
            requiredDistance * requiredDistance - perpendicularSq
          );
          requiredShift = Math.max(requiredShift, axisExit + 0.016);
        }
        if (requiredShift <= 0) continue;
        if (index > 0) {
          worldStart.x += body.directionX * requiredShift;
          worldStart.z += body.directionZ * requiredShift;
        }
        worldEnd.x += body.directionX * requiredShift;
        worldEnd.z += body.directionZ * requiredShift;
        changed = true;
      }

      if (!changed) return false;
      if (index > 0) {
        localPoint.copy(worldStart);
        this.mesh.worldToLocal(localPoint);
        start.x = localPoint.x;
        start.z = localPoint.z;
      }
      localPoint.copy(worldEnd);
      this.mesh.worldToLocal(localPoint);
      end.x = localPoint.x;
      end.z = localPoint.z;
      return true;
    };

    for (let i = 0; i <= this.tailCount; i++) {
      clampPointToArena(tailCurve[i], i);
    }
    const fixedRoot = tailCurve[0].clone();
    const bendTarget = new THREE.Vector3();
    for (let pass = 0; pass < 28; pass++) {
      let corrected = false;
      if (pass > 0) {
        // Backward FABRIK tension pass: propagate distal obstacle/wall
        // deflections smoothly toward the root instead of snapping one chord.
        for (let i = this.tailCount - 1; i >= 1; i--) {
          delta.copy(tailCurve[i]).sub(tailCurve[i + 1]);
          if (delta.lengthSq() < 1e-8) {
            delta.copy(fixedRoot).sub(tailCurve[i + 1]);
            if (delta.lengthSq() < 1e-8) delta.set(0, 0, 1);
          }
          tailCurve[i].copy(tailCurve[i + 1]).addScaledVector(delta.normalize(), segLen);
        }
      }
      tailCurve[0].copy(fixedRoot);

      // A light bend constraint removes contact-induced kinks before the
      // inextensible centerline and collision constraints are projected.
      for (let parity = 0; parity < 2; parity++) {
        for (let i = 1 + parity; i < this.tailCount; i += 2) {
          bendTarget.copy(tailCurve[i - 1]).add(tailCurve[i + 1]).multiplyScalar(0.5);
          tailCurve[i].lerp(bendTarget, pass < 6 ? 0.18 : 0.11);
        }
      }
      for (let i = 1; i <= this.tailCount; i++) {
        delta.copy(tailCurve[i]).sub(tailCurve[i - 1]);
        if (delta.lengthSq() < 1e-8) delta.set(0, 0, -1);
        tailCurve[i].copy(tailCurve[i - 1]).addScaledVector(delta.normalize(), segLen);
        corrected = clampPointToArena(tailCurve[i], i) || corrected;
      }
      for (let i = 1; i <= this.tailCount; i++) {
        corrected = projectPointOutsideBodies(tailCurve[i], i) || corrected;
        corrected = clampPointToArena(tailCurve[i], i) || corrected;
      }
      for (let i = 0; i < this.tailCount; i++) {
        corrected = projectSegmentOutsideBodies(tailCurve[i], tailCurve[i + 1], i)
          || corrected;
      }
      for (let i = 1; i <= this.tailCount; i++) {
        corrected = clampPointToArena(tailCurve[i], i) || corrected;
      }
      if (!corrected) break;
    }

    // Finish on contact constraints so every chord of the continuous tube
    // remains outside nearby mouse bodies and the enclosure surfaces.
    for (let pass = 0; pass < this.tailCount * 2; pass++) {
      let corrected = false;
      for (let i = 1; i <= this.tailCount; i++) {
        corrected = projectPointOutsideBodies(tailCurve[i], i) || corrected;
        corrected = clampPointToArena(tailCurve[i], i) || corrected;
      }
      for (let i = 0; i < this.tailCount; i++) {
        corrected = projectSegmentOutsideBodies(tailCurve[i], tailCurve[i + 1], i)
          || corrected;
      }
      for (let i = 1; i <= this.tailCount; i++) {
        corrected = clampPointToArena(tailCurve[i], i) || corrected;
      }
      if (!corrected) break;
    }

    this.minimumTailClearance = Infinity;
    this.minimumTailClearanceIndex = -1;
    for (let i = 0; i <= this.tailCount; i++) {
      clampPointToArena(tailCurve[i], i);
      worldPoint.copy(tailCurve[i]);
      this.mesh.localToWorld(worldPoint);
      const tailRadius = tailRadiusAt(i);
      for (const disc of obstacles) {
        const clearance = Math.hypot(worldPoint.x - disc.x, worldPoint.z - disc.z)
          - disc.radius - tailRadius;
        if (clearance < this.minimumTailClearance) {
          this.minimumTailClearance = clearance;
          this.minimumTailClearanceIndex = i;
        }
      }
    }
    for (let i = 0; i < this.tailCount; i++) {
      const contact = getSegmentContact(tailCurve[i], tailCurve[i + 1], i);
      if (contact.clearance < this.minimumTailClearance) {
        this.minimumTailClearance = contact.clearance;
        this.minimumTailClearanceIndex = i + 0.5;
      }
    }

    this.maximumTailSegmentLengthError = 0;
    this.maximumTailBendAngle = 0;
    let previousDx = 0;
    let previousDy = 0;
    let previousDz = 0;
    let previousLength = 0;
    for (let i = 0; i < this.tailCount; i++) {
      const dx = tailCurve[i + 1].x - tailCurve[i].x;
      const dy = tailCurve[i + 1].y - tailCurve[i].y;
      const dz = tailCurve[i + 1].z - tailCurve[i].z;
      const length = Math.max(Math.hypot(dx, dy, dz), 1e-7);
      this.maximumTailSegmentLengthError = Math.max(
        this.maximumTailSegmentLengthError,
        Math.abs(length / segLen - 1.0)
      );
      if (previousLength > 0) {
        const cosine = THREE.MathUtils.clamp(
          (dx * previousDx + dy * previousDy + dz * previousDz)
            / (length * previousLength),
          -1,
          1
        );
        this.maximumTailBendAngle = Math.max(
          this.maximumTailBendAngle,
          THREE.MathUtils.radToDeg(Math.acos(cosine))
        );
      }
      previousDx = dx;
      previousDy = dy;
      previousDz = dz;
      previousLength = length;
    }

    updateDynamicTailTubeGeometry(
      this.tailMesh.geometry,
      tailCurve,
      this.guiParams.tailThickness
    );
    if (this.tailWorldPoints && this.tailWorldPoints.length === this.tailCount + 1) {
      for (let i = 0; i <= this.tailCount; i++) {
        this.tailWorldPoints[i].copy(tailCurve[i]);
        this.mesh.localToWorld(this.tailWorldPoints[i]);
      }
    }
    const isTailVisible = this.tailMesh.visible;
    if (this.tailRoot) this.tailRoot.visible = isTailVisible;
    const tailMidIndex = Math.max(1, Math.floor(this.tailCount * 0.4));
    this.poseNodes.tailmid.position.copy(tailCurve[tailMidIndex]);
    this.poseNodes.tailmid.visible = guiParams.showTrackingKeypoints && isTailVisible;
    this.poseNodes.tailend.position.copy(tailCurve[this.tailCount]);
    this.poseNodes.tailend.visible = guiParams.showTrackingKeypoints && isTailVisible;
    if (this.tailRoot && tailCurve.length > 4) {
      const thickMul = this.guiParams.tailThickness;
      const dir = tailCurve[3].clone().sub(tailCurve[0]);
      if (dir.lengthSq() < 1e-8) dir.set(0, 0, -1);
      else dir.normalize();
      this.tailRoot.position.copy(tailCurve[0]).addScaledVector(dir, segLen * 0.25);
      worldPoint.copy(tailCurve[0]).addScaledVector(dir, segLen * 3.0);
      this.mesh.localToWorld(worldPoint);
      this.tailRoot.lookAt(worldPoint);
      const scale = 1.0 + (thickMul - 1.0) * 0.7;
      this.tailRoot.scale.set(0.95 * scale, 0.68 * scale, 1.15 * scale);
    }
  }

  updateTrackingGeometry() {
    this.mesh.updateMatrixWorld(true);
    if (!this.trackingGroup.visible) return;
    const positions = this.skeletonLines.geometry.attributes.position.array;
    let index = 0;
    const inverseTrackingMatrix = this.trackingGroup.matrixWorld.clone().invert();
    const pointA = new THREE.Vector3();
    const pointB = new THREE.Vector3();
    for (const [idA, idB] of POSE_CONNECTIONS) {
      const nodeA = this.poseNodes[idA];
      const nodeB = this.poseNodes[idB];
      if (!nodeA || !nodeB) continue;
      nodeA.getWorldPosition(pointA);
      nodeB.getWorldPosition(pointB);
      pointA.applyMatrix4(inverseTrackingMatrix);
      pointB.applyMatrix4(inverseTrackingMatrix);
      positions[index++] = pointA.x;
      positions[index++] = pointA.y;
      positions[index++] = pointA.z;
      positions[index++] = pointB.x;
      positions[index++] = pointB.y;
      positions[index++] = pointB.z;
    }
    this.skeletonLines.geometry.attributes.position.needsUpdate = true;
  }

  startGrooming(mode, activeDuration = null) {
    if (!['face-wash', 'flank-groom'].includes(mode)) return false;
    if (this.grooming.mode !== 'none' || this.rearing.mode !== 'none') return false;

    const state = this.grooming;
    state.mode = mode;
    state.phase = 'prepare';
    state.phaseTime = 0;
    state.elapsed = 0;
    state.peakAmount = 1;
    state.prepareDuration = mode === 'face-wash' ? 0.30 : 0.38;
    state.activeDuration = activeDuration ?? (
      mode === 'face-wash'
        ? 1.75 + (this.timeOffset * 0.013) % 0.65
        : 1.95 + (this.timeOffset * 0.017) % 0.75
    );
    state.settleDuration = mode === 'face-wash' ? 0.48 : 0.56;
    state.side = Math.sin(
      this.timeOffset * 0.31 + state.sequenceIndex * 1.7
    ) >= 0 ? 1 : -1;
    state.sequenceIndex += 1;

    this.pauseTimer = Math.max(
      this.pauseTimer,
      state.prepareDuration + state.activeDuration + state.settleDuration + 0.25
    );
    this.stuckTimer = 0;
    this.velocity.multiplyScalar(0.08);
    return true;
  }

  updateGroomingBehavior(time, dt, context) {
    const state = this.grooming;
    this.groomCooldown = Math.max(0, this.groomCooldown - dt);

    if (
      state.mode === 'none'
      && this.rearing.mode === 'none'
      && guiParams.enableGrooming
      && this.groomCooldown <= 0
      && !this.externalCommand
    ) {
      const quietOpportunity = time >= state.nextSpontaneousAt
        && !context.hitWall
        && context.behaviorPause
        && context.speed < 1.25
        && !this.socialContact
        && context.wallDist > context.bodyMargin + 0.35;
      if (quietOpportunity) {
        const nextMode = state.sequenceIndex % 2 === 0
          ? 'face-wash'
          : 'flank-groom';
        this.startGrooming(nextMode);
      }
    }

    if (state.mode === 'none') {
      this.groomAmount = 0;
      this.groomVel = 0;
      return;
    }

    if (this.socialContact && state.phase !== 'settle') {
      state.phase = 'settle';
      state.phaseTime = 0;
    }

    state.phaseTime += dt;
    state.elapsed += dt;
    if (state.phase === 'prepare' && state.phaseTime >= state.prepareDuration) {
      state.phase = 'active';
      state.phaseTime = 0;
    } else if (state.phase === 'active' && state.phaseTime >= state.activeDuration) {
      state.phase = 'settle';
      state.phaseTime = 0;
    }

    const target = state.phase === 'settle' ? 0 : state.peakAmount;
    const stiffness = state.phase === 'settle' ? 24.0 : 34.0;
    const damping = state.phase === 'settle' ? 9.6 : 8.4;
    this.groomVel += (
      stiffness * (target - this.groomAmount) - damping * this.groomVel
    ) * dt;
    this.groomAmount = THREE.MathUtils.clamp(
      this.groomAmount + this.groomVel * dt,
      0,
      1
    );
    this.velocity.multiplyScalar(Math.exp(-10.0 * dt));

    if (
      state.phase === 'settle'
      && state.phaseTime >= state.settleDuration
      && this.groomAmount < 0.018
      && Math.abs(this.groomVel) < 0.08
    ) {
      const frequency = THREE.MathUtils.clamp(
        guiParams.groomingFrequency ?? 1,
        0.25,
        2
      );
      state.mode = 'none';
      state.phase = 'idle';
      state.phaseTime = 0;
      state.elapsed = 0;
      state.nextSpontaneousAt = time
        + (8.5 + (this.timeOffset * 0.023) % 5.0) / frequency;
      this.groomAmount = 0;
      this.groomVel = 0;
      this.groomCooldown = (3.2 + (this.timeOffset * 0.015) % 2.2) / frequency;
    }
  }

  startRearing(mode, supportDirection = null, holdDuration = null) {
    if (!['unsupported', 'wall-supported'].includes(mode)) return false;
    if (this.rearing.mode !== 'none' || this.grooming.mode !== 'none') return false;

    const state = this.rearing;
    state.mode = mode;
    state.phase = 'rise';
    state.phaseTime = 0;
    state.elapsed = 0;
    state.peakAmount = mode === 'wall-supported' ? 0.94 : 0.80;
    state.maxAngle = mode === 'wall-supported' ? 1.06 : 1.02;
    state.angle = 0;
    state.angularVelocity = 0;
    state.angularAcceleration = 0;
    state.gravityTorque = 0;
    state.muscleTorque = 0;
    state.wallReactionTorque = 0;
    state.wallContactLoad = 0;
    state.balanceAngleCorrection = 0;
    state.centerOfMassForwardOffset = 0;
    state.centerOfMassLateralOffset = 0;
    state.centerOfMassProjectionError = 0;
    state.hindSupportError = 0;
    state.hindSupportHorizontalError = 0;
    state.hindSupportVerticalError = 0;
    state.wallSupportError = mode === 'wall-supported' ? 1.0 : 0;
    state.wallSupportNormalError = 0;
    state.wallSupportVerticalError = 0;
    state.baseAdvance = 0;
    state.baseAdvanceVelocity = 0;
    state.maximumBaseAdvance = 0;
    state.hindStanceOffset = 0;
    state.hindStanceVelocity = 0;
    state.maximumHindStanceOffset = mode === 'unsupported' ? 0.52 : 0;
    state.rearFeetPlanted = false;
    state.rearSupportReleased = false;
    state.riseDuration = mode === 'wall-supported' ? 0.66 : 0.58;
    state.holdDuration = holdDuration ?? (
      mode === 'wall-supported'
        ? 1.05 + (this.timeOffset * 0.017) % 0.50
        : 0.85 + (this.timeOffset * 0.013) % 0.55
    );
    state.settleDuration = mode === 'wall-supported' ? 0.58 : 0.50;
    state.sideLean = Math.sin(this.timeOffset * 0.37) * (
      mode === 'wall-supported' ? 0.018 : 0.052
    );

    if (mode === 'wall-supported') {
      if (supportDirection && supportDirection.lengthSq() > 1e-8) {
        state.supportDirection.copy(supportDirection).normalize();
      } else {
        state.supportDirection.copy(this.heading).normalize();
      }
      if (Math.abs(state.supportDirection.x) >= Math.abs(state.supportDirection.y)) {
        state.supportPoint.set(
          Math.sign(state.supportDirection.x || 1) * (ENCLOSURE_HALF_X - 0.08),
          this.mesh.position.z
        );
      } else {
        state.supportPoint.set(
          this.mesh.position.x,
          Math.sign(state.supportDirection.y || 1) * (ENCLOSURE_HALF_Z - 0.08)
        );
      }
      const wallDistance = (
        (state.supportPoint.x - this.mesh.position.x) * state.supportDirection.x
        + (state.supportPoint.y - this.mesh.position.z) * state.supportDirection.y
      );
      state.maximumBaseAdvance = THREE.MathUtils.clamp(
        wallDistance - 0.95,
        0,
        2.82
      );
    } else {
      state.supportDirection.copy(this.heading).normalize();
      state.supportPoint.set(this.mesh.position.x, this.mesh.position.z);
    }

    for (const plant of this.footPlants.slice(0, 2)) {
      plant.planted = false;
      plant.exhausted = false;
    }

    this.pauseTimer = Math.max(
      this.pauseTimer,
      state.riseDuration + state.holdDuration + state.settleDuration + 0.2
    );
    this.velocity.multiplyScalar(0.14);
    return true;
  }

  updateRearingBehavior(time, dt, context) {
    const state = this.rearing;
    this.rearCooldown = Math.max(0, this.rearCooldown - dt);

    if (
      state.mode === 'none'
      && this.grooming.mode === 'none'
      && guiParams.enableRearing
      && this.rearCooldown <= 0
      && !this.externalCommand
    ) {
      const supportDirection = new THREE.Vector2(
        -context.nearestWall.normalX,
        -context.nearestWall.normalZ
      );
      const impactRear = context.hitWall && context.impactNorm > 0.14;
      const nearWall = context.wallDist < context.bodyMargin + 0.55;
      const wallScanOpportunity = Math.sin(
        time * 0.86 + this.timeOffset * 0.031
      ) > 0.992;
      const supportedRear = nearWall
        && context.behaviorPause
        && context.speed < 1.35
        && wallScanOpportunity;
      const openZone = context.wallDist > context.bodyMargin + 1.35;
      const unsupportedRear = time >= state.nextSpontaneousAt
        && context.behaviorPause
        && context.speed < 1.15
        && !this.socialContact
        && openZone;

      if (impactRear || supportedRear) {
        this.startRearing('wall-supported', supportDirection);
      } else if (unsupportedRear) {
        this.startRearing('unsupported');
      }
    }

    if (state.mode === 'none') {
      this.rearAmount = 0;
      this.rearVel = 0;
      state.angle = 0;
      state.angularVelocity = 0;
      state.angularAcceleration = 0;
      state.gravityTorque = 0;
      state.muscleTorque = 0;
      state.wallReactionTorque = 0;
      state.wallContactLoad = 0;
      state.balanceAngleCorrection = 0;
      state.centerOfMassForwardOffset = 0;
      state.centerOfMassLateralOffset = 0;
      state.centerOfMassProjectionError = 0;
      return;
    }

    state.phaseTime += dt;
    state.elapsed += dt;
    const angleError = state.maxAngle - state.angle;
    if (
      state.phase === 'rise'
      && state.phaseTime >= state.riseDuration
      && (Math.abs(angleError) < 0.08 || state.phaseTime >= state.riseDuration * 1.75)
    ) {
      state.phase = 'hold';
      state.phaseTime = 0;
    } else if (state.phase === 'hold' && state.phaseTime >= state.holdDuration) {
      state.phase = 'settle';
      state.phaseTime = 0;
    }

    const effectiveLength = PROP.bodyLength * this.guiParams.bodyLength;
    integrateRearAngularDynamics(
      state,
      dt,
      this.mass,
      effectiveLength
    );
    this.rearAmount = state.peakAmount * THREE.MathUtils.clamp(
      state.angle / Math.max(state.maxAngle, 0.01),
      0,
      1
    );
    this.rearVel = state.angularVelocity
      / Math.max(state.maxAngle, 0.01) * state.peakAmount;
    this.velocity.multiplyScalar(Math.exp(-8.0 * dt));

    if (
      state.phase === 'settle'
      && state.phaseTime >= state.settleDuration
      && state.angle < 0.018
      && Math.abs(state.angularVelocity) < 0.08
    ) {
      const frequency = THREE.MathUtils.clamp(guiParams.rearingFrequency ?? 1, 0.25, 2);
      state.mode = 'none';
      state.phase = 'idle';
      state.phaseTime = 0;
      state.elapsed = 0;
      state.nextSpontaneousAt = time
        + (7.5 + (this.timeOffset * 0.019) % 4.5) / frequency;
      this.rearAmount = 0;
      this.rearVel = 0;
      state.angle = 0;
      state.angularVelocity = 0;
      state.angularAcceleration = 0;
      state.baseAdvance = 0;
      state.baseAdvanceVelocity = 0;
      state.maximumBaseAdvance = 0;
      state.hindStanceOffset = 0;
      state.hindStanceVelocity = 0;
      state.maximumHindStanceOffset = 0;
      state.rearFeetPlanted = false;
      state.rearSupportReleased = false;
      for (const plant of this.footPlants.slice(0, 2)) {
        plant.planted = false;
        plant.exhausted = false;
      }
      this.rearCooldown = (2.4 + (this.timeOffset * 0.011) % 1.8) / frequency;
    }
  }

  solveRearSupportConstraints() {
    const state = this.rearing;
    this.maximumRearSupportError = 0;
    if (state.mode === 'none' || state.angle < 0.012) {
      state.hindSupportError = 0;
      state.wallSupportError = 0;
      state.wallSupportNormalError = 0;
      state.wallSupportVerticalError = 0;
      state.centerOfMassProjectionError = 0;
      return;
    }
    if (state.rearSupportReleased) {
      state.hindSupportError = 0;
      state.hindSupportHorizontalError = 0;
      state.hindSupportVerticalError = 0;
      state.wallSupportError = 0;
      state.wallSupportNormalError = 0;
      state.wallSupportVerticalError = 0;
      state.wallContactLoad = 0;
      return;
    }
    if (!state.rearFeetPlanted) this.captureRearFootAnchors();

    const contactPosition = new THREE.Vector3();
    let hindSupportError = 0;
    let hindSupportHorizontalError = 0;
    let hindSupportVerticalError = 0;
    for (let index = 0; index < 2; index++) {
      this.footPlants[index].contact.updateWorldMatrix(true, false);
      this.footPlants[index].contact.getWorldPosition(contactPosition);
      const anchor = state.hindAnchors[index];
      hindSupportHorizontalError = Math.max(
        hindSupportHorizontalError,
        Math.hypot(contactPosition.x - anchor.x, contactPosition.z - anchor.z)
      );
      hindSupportVerticalError = Math.max(
        hindSupportVerticalError,
        Math.abs(contactPosition.y - anchor.y)
      );
      hindSupportError = Math.max(
        hindSupportError,
        contactPosition.distanceTo(anchor)
      );
    }
    state.hindSupportError = hindSupportError;
    state.hindSupportHorizontalError = hindSupportHorizontalError;
    state.hindSupportVerticalError = hindSupportVerticalError;
    this.maximumRearSupportError = hindSupportError;

    const bodyCenter = new THREE.Vector3();
    const headCenter = new THREE.Vector3();
    this.bodyMesh.updateWorldMatrix(true, false);
    this.head.updateWorldMatrix(true, false);
    this.bodyMesh.getWorldPosition(bodyCenter);
    this.head.getWorldPosition(headCenter);
    const centerOfMass = bodyCenter.multiplyScalar(0.76)
      .addScaledVector(headCenter, 0.24);
    const supportCenter = state.hindAnchors[0].clone()
      .add(state.hindAnchors[1])
      .multiplyScalar(0.5);
    const toCenterOfMass = centerOfMass.sub(supportCenter);
    const forwardOffset = toCenterOfMass.x * this.heading.x
      + toCenterOfMass.z * this.heading.y;
    const lateralX = this.heading.y;
    const lateralZ = -this.heading.x;
    const lateralOffset = toCenterOfMass.x * lateralX
      + toCenterOfMass.z * lateralZ;
    const hindSpan = state.hindAnchors[0].distanceTo(state.hindAnchors[1]);
    const lateralLimit = hindSpan * 0.5 + 0.16;
    const forwardLimit = state.mode === 'wall-supported'
      ? 3.8
      : (forwardOffset >= 0 ? 0.68 : 1.10);
    const forwardError = Math.max(0, Math.abs(forwardOffset) - forwardLimit);
    const lateralError = Math.max(0, Math.abs(lateralOffset) - lateralLimit);
    state.centerOfMassForwardOffset = forwardOffset;
    state.centerOfMassLateralOffset = lateralOffset;
    state.centerOfMassProjectionError = Math.hypot(forwardError, lateralError);

    if (state.mode !== 'wall-supported') {
      state.wallSupportError = 0;
      state.wallSupportNormalError = 0;
      state.wallSupportVerticalError = 0;
      state.wallContactLoad = 0;
      return;
    }

    const support = state.supportDirection;
    const poseAmount = THREE.MathUtils.clamp(
      state.angle / Math.max(state.maxAngle, 0.01),
      0,
      1
    );
    if (poseAmount < 0.58) {
      state.wallSupportError = 1.0;
      state.wallSupportNormalError = 1.0;
      state.wallSupportVerticalError = 0;
      state.wallContactLoad = 0;
      for (const plant of this.footPlants.slice(2)) plant.planted = false;
      return;
    }
    const wallLateralX = support.y;
    const wallLateralZ = -support.x;
    const targets = [
      { plant: this.footPlants[2], side: 1 },
      { plant: this.footPlants[3], side: -1 },
    ];
    const shoulderPosition = new THREE.Vector3();
    const wallContactPosition = new THREE.Vector3();
    let wallSupportError = 0;
    let wallSupportNormalError = 0;
    let wallSupportVerticalError = 0;

    for (const { plant, side } of targets) {
      const target = new THREE.Vector3(
        state.supportPoint.x + wallLateralX * side * 0.25,
        0,
        state.supportPoint.y + wallLateralZ * side * 0.25
      );
      const shoulder = plant.joints[plant.joints.length - 1];
      shoulder.updateWorldMatrix(true, false);
      shoulder.getWorldPosition(shoulderPosition);
      const horizontalReachSquared = Math.pow(target.x - shoulderPosition.x, 2)
        + Math.pow(target.z - shoulderPosition.z, 2);
      const usableReach = plant.maximumReach * 0.96;
      const availableVerticalReach = Math.sqrt(Math.max(
        0,
        usableReach * usableReach - horizontalReachSquared
      ));
      target.y = Math.min(
        1.30 + poseAmount * 1.82,
        shoulderPosition.y + availableVerticalReach
      );
      const result = this.solveIkConstraint(plant, plant.contact, target, {
        iterations: 14,
        jointXOffsets: [0.62, 0.82, 0.86],
        jointXLimits: [[-0.35, 0.82], [-1.18, -0.58], [-1.20, 0.62]],
        rootZOffset: 0.36,
      });
      plant.contact.updateWorldMatrix(true, false);
      plant.contact.getWorldPosition(wallContactPosition);
      const error = result.error;
      wallSupportError = Math.max(wallSupportError, error);
      wallSupportNormalError = Math.max(
        wallSupportNormalError,
        Math.abs(
          (target.x - wallContactPosition.x) * support.x
            + (target.z - wallContactPosition.z) * support.y
        )
      );
      wallSupportVerticalError = Math.max(
        wallSupportVerticalError,
        Math.abs(target.y - wallContactPosition.y)
      );
      plant.anchor.copy(target);
      plant.planted = error < 0.16;
      plant.exhausted = false;
      plant.slip = error;
      plant.groundError = 0;
      plant.reachRatio = shoulderPosition.distanceTo(target)
        / plant.maximumReach;
      plant.candidateSlip = error;
      plant.candidateGroundError = Math.abs(
        target.y - wallContactPosition.y
      );
      plant.rejectionReason = 'wall-support';
    }
    state.wallSupportError = wallSupportError;
    state.wallSupportNormalError = wallSupportNormalError;
    state.wallSupportVerticalError = wallSupportVerticalError;
    state.wallContactLoad = THREE.MathUtils.smoothstep(poseAmount, 0.62, 0.92)
      * (1.0 - THREE.MathUtils.clamp(wallSupportError / 0.22, 0, 1));
    this.maximumRearSupportError = Math.max(
      this.maximumRearSupportError,
      wallSupportError * state.wallContactLoad
    );
  }


  update(time, otherMice = [], hudRef = null, dt = 0.016) {
    dt = Math.min(dt, 0.05);
    let t = time * 2.0 * this.speedMult + this.timeOffset;
    if (!guiParams.isPaused) {
      this.behaviorTimer -= dt;
      this.pauseTimer = Math.max(0, this.pauseTimer - dt);
      this.socialState.cooldown = Math.max(0, this.socialState.cooldown - dt);
      this.boutState.timer -= dt;
      this.boutState.scanPhase += dt * 3.4;

      if (this.behaviorTimer <= 0) {
        if (random() < 0.44) {
          this.pauseTimer = 0.30 + random() * 0.95;
        }
        this.behaviorTimer = 1.3 + random() * 3.4;
        if (random() < 0.22) this.wallFollowDir *= -1;
      }

      // Saltatory bout-and-pause rhythm (stop-sniff -> stretch-attend -> explore-walk -> dart-trot)
      if (this.boutState.timer <= 0 && this.rearing.mode === 'none' && this.grooming.mode === 'none') {
        const roll = random();
        if (this.pauseTimer > 0.12) {
          this.boutState.mode = roll < 0.38 ? 'stretch-attend' : 'pause-sniff';
          this.boutState.timer = 0.32 + random() * 0.55;
        } else if (roll < 0.20) {
          this.boutState.mode = 'pause-sniff';
          this.boutState.timer = 0.32 + random() * 0.65;
          this.pauseTimer = Math.max(this.pauseTimer, this.boutState.timer * 0.85);
        } else if (roll < 0.36) {
          this.boutState.mode = 'stretch-attend';
          this.boutState.timer = 0.28 + random() * 0.48;
        } else if (roll < 0.68) {
          this.boutState.mode = 'explore-walk';
          this.boutState.timer = 0.85 + random() * 1.65;
        } else {
          this.boutState.mode = 'dart-trot';
          this.boutState.timer = 0.48 + random() * 0.92;
        }
      }
    }
    if (this.externalCommand && time >= this.externalCommand.expiresAt) {
      this.externalCommand = null;
    }
    const commandActive = Boolean(this.externalCommand);
    const behaviorPause = !commandActive
      && (this.pauseTimer > 0 || this.boutState.mode === 'pause-sniff');
    const rearingActive = this.rearing.mode !== 'none';
    const groomingActive = this.grooming.mode !== 'none';
    const commandTracking = commandActive && !rearingActive && !groomingActive;
    const isPausing = behaviorPause || rearingActive || groomingActive;
    const boutSpeedFactor = this.boutState.mode === 'dart-trot'
      ? 1.18
      : (this.boutState.mode === 'stretch-attend'
        ? 0.30
        : (this.boutState.mode === 'pause-sniff' ? 0.08 : 0.72));

    // --- FORCE ACCUMULATION ---
    this.force.set(0, 0);
    let burst = Math.sin(t * 1.5 + this.burstPhase);
    burst = 0.7 + 0.3 * Math.max(0, burst);
    const driveStrength = burst * this.guiParams.wanderForce * (
      rearingActive || groomingActive ? 0.04 : (isPausing ? 0.14 : boutSpeedFactor)
    );

    // Thigmotaxis & anticipatory corner steering
    const px = this.mesh.position.x, pz = this.mesh.position.z;
    const wallDists = [
      { dist: ENCLOSURE_HALF_X - px, normalX: -1, normalZ: 0, tangentX: 0, tangentZ: 1 },
      { dist: ENCLOSURE_HALF_X + px, normalX: 1, normalZ: 0, tangentX: 0, tangentZ: -1 },
      { dist: ENCLOSURE_HALF_Z - pz, normalX: 0, normalZ: -1, tangentX: -1, tangentZ: 0 },
      { dist: ENCLOSURE_HALF_Z + pz, normalX: 0, normalZ: 1, tangentX: 1, tangentZ: 0 },
    ];
    wallDists.sort((a, b) => a.dist - b.dist);
    const nearest = wallDists[0];
    const wallDist = nearest.dist;
    const wallAttr = this.guiParams.wallAttraction;
    for (const w of wallDists) {
      const wallForce = wallNormalForce(w.dist, driveStrength, wallAttr, w === nearest);
      this.force.x += w.normalX * wallForce;
      this.force.y += w.normalZ * wallForce;
      // Smooth anticipatory turn along wall tangent when approaching head-on
      const facingIntoWall = -(this.heading.x * w.normalX + this.heading.y * w.normalZ);
      if (w.dist < 5.2 && facingIntoWall > 0.25 && !rearingActive && !groomingActive) {
        const turnUrgency = THREE.MathUtils.clamp((5.2 - w.dist) / 3.4, 0, 1) * facingIntoWall;
        this.force.x += (w.tangentX * this.wallFollowDir * 1.65 + w.normalX * 0.55)
          * driveStrength * turnUrgency * 1.8;
        this.force.y += (w.tangentZ * this.wallFollowDir * 1.65 + w.normalZ * 0.55)
          * driveStrength * turnUrgency * 1.8;
      }
    }
    const tX = nearest.tangentX * this.wallFollowDir;
    const tZ = nearest.tangentZ * this.wallFollowDir;
    const nwF = Math.max(0, 1.0 - wallDist / 4.8);
    this.force.x += tX * driveStrength * (0.52 + nwF * 1.65);
    this.force.y += tZ * driveStrength * (0.52 + nwF * 1.65);

    if (!guiParams.isPaused && !groomingActive) {
      this.force.x += (Math.sin(t * 2.3) * 0.34 + (random() - 0.5) * 0.24) * driveStrength * 0.34;
      this.force.y += (Math.cos(t * 1.7) * 0.34 + (random() - 0.5) * 0.24) * driveStrength * 0.34;
    }

    // Anti-stuck
    if (!guiParams.isPaused && !isPausing) {
      const dxS = px - this.lastPos.x, dzS = pz - this.lastPos.y;
      this.stuckTimer += dt;
      if (this.stuckTimer > 1.0) {
        if (Math.sqrt(dxS * dxS + dzS * dzS) < 0.35) {
          const escape = new THREE.Vector2(
            nearest.tangentX * this.wallFollowDir + nearest.normalX * 0.7,
            nearest.tangentZ * this.wallFollowDir + nearest.normalZ * 0.7
          );
          if (escape.lengthSq() < 1e-6) escape.set(Math.cos(t), Math.sin(t));
          escape.normalize();
          const escapeForce = 30.0 + (wallDist < 1.5 ? 20.0 : 0.0);
          this.force.x += escape.x * escapeForce;
          this.force.y += escape.y * escapeForce;
          this.yawVel *= 0.5;
          this.wallFollowDir *= -1;
        }
        this.lastPos.set(px, pz); this.stuckTimer = 0;
      }
    }

    // Directed resident-intruder social interactions (nose-to-nose, anogenital follow, flank huddle, breakaway)
    const SOCIAL_RADIUS = 11.2;
    let isColliding = false;
    let nearestSocialDist = Infinity;
    this.socialContact = false;
    let activeSocialMode = 'none';
    const myBodyScale = this.guiParams.bodyLength || 1.0;
    const mySnoutX = this.mesh.position.x + this.heading.x * 3.35 * myBodyScale;
    const mySnoutZ = this.mesh.position.z + this.heading.y * 3.35 * myBodyScale;

    for (const other of otherMice) {
      if (other === this) continue;
      const dx = this.mesh.position.x - other.mesh.position.x;
      const dz = this.mesh.position.z - other.mesh.position.z;
      const distSq = dx * dx + dz * dz;
      const bodyResult = resolveMouseBodyOverlap(this, other);
      const bodyClearance = bodyResult.contact.clearance;
      isColliding = bodyResult.touched || bodyClearance < 0.04 || isColliding;
      if (distSq > 0.001) {
        const dist = Math.sqrt(distSq);
        const otherScale = other.guiParams.bodyLength || 1.0;
        const otherSnoutX = other.mesh.position.x + other.heading.x * 3.35 * otherScale;
        const otherSnoutZ = other.mesh.position.z + other.heading.y * 3.35 * otherScale;
        const otherRumpX = other.mesh.position.x - other.heading.x * 2.25 * otherScale;
        const otherRumpZ = other.mesh.position.z - other.heading.y * otherScale * 2.25;

        const dNoseToSnout = Math.hypot(mySnoutX - otherSnoutX, mySnoutZ - otherSnoutZ);
        const dNoseToRump = Math.hypot(mySnoutX - otherRumpX, mySnoutZ - otherRumpZ);
        const isResident = (this.trackId || 1) === 1;
        const preferAnogenital = (dNoseToRump < dNoseToSnout * 0.92)
          || (isResident && dNoseToRump < 6.4 && this.socialState.cooldown <= 0);

        const aimX = preferAnogenital ? otherRumpX : (bodyClearance < 0.95 ? other.mesh.position.x : otherSnoutX);
        const aimZ = preferAnogenital ? otherRumpZ : (bodyClearance < 0.95 ? other.mesh.position.z : otherSnoutZ);
        const toAimX = aimX - mySnoutX;
        const toAimZ = aimZ - mySnoutZ;
        const aimDist = Math.max(1e-4, Math.hypot(toAimX, toAimZ));
        const dirAimX = toAimX / aimDist;
        const dirAimZ = toAimZ / aimDist;

        if (dist < nearestSocialDist) {
          nearestSocialDist = dist;
          this.socialTarget.set(dirAimX, dirAimZ);
          this.socialState.targetWorld.set(aimX, aimZ);
        }

        if (dist < SOCIAL_RADIUS && !rearingActive && !groomingActive && !commandActive) {
          const socialBand = THREE.MathUtils.clamp((SOCIAL_RADIUS - dist) / 6.2, 0, 1);
          const headDot = this.heading.x * other.heading.x + this.heading.y * other.heading.y;

          if (this.socialState.cooldown > 0 && !isResident) {
            // Energetic post-contact evasive breakaway along enclosure perimeter & nestlet
            activeSocialMode = 'evasive-dart';
            this.force.x += (tX * 1.55 - dirAimX * 0.65) * this.guiParams.wanderForce * 2.10;
            this.force.y += (tZ * 1.55 - dirAimZ * 0.65) * this.guiParams.wanderForce * 2.10;
          } else if (preferAnogenital && bodyClearance > 0.08) {
            // Resident/initiator briskly trails partner tail-base in active anogenital pursuit
            activeSocialMode = 'anogenital-follow';
            const followGain = (isResident ? 1.85 : 1.05) * (0.45 + 0.55 * socialBand);
            this.force.x += dirAimX * this.guiParams.wanderForce * followGain;
            this.force.y += dirAimZ * this.guiParams.wanderForce * followGain;
            if (dNoseToRump < 2.2) {
              this.socialContact = true;
              if (other.velocity.lengthSq() > 0.6) {
                this.velocity.lerp(other.velocity, 0.08);
              }
            }
          } else if (headDot < -0.25 && dNoseToSnout < 5.5) {
            // Mutual nose-to-nose stretch-attend sniff
            activeSocialMode = 'nose-to-nose';
            if (bodyClearance > 0.32) {
              this.force.x += dirAimX * driveStrength * 0.85 * socialBand;
              this.force.y += dirAimZ * driveStrength * 0.85 * socialBand;
            } else if (DATASET_SCENARIO === 'social_huddle') {
              this.velocity.multiplyScalar(0.92);
            }
            if (dNoseToSnout < 2.4) this.socialContact = true;
          } else if (bodyClearance > 0.65) {
            const curiosity = (0.34 + 0.42 * Math.max(0, Math.sin(time * 0.84 + this.timeOffset * 0.13))) * socialBand;
            this.force.x += dirAimX * driveStrength * curiosity;
            this.force.y += dirAimZ * driveStrength * curiosity;
          }

          if (bodyClearance < 1.35) {
            if (activeSocialMode === 'none') activeSocialMode = 'flank-huddle';
            const side = this.dir;
            this.force.x += -dirAimZ * side * socialBand * driveStrength * 0.22;
            this.force.y += dirAimX * side * socialBand * driveStrength * 0.22;
            if (DATASET_SCENARIO === 'social_huddle') {
              this.velocity.multiplyScalar(bodyClearance < 0.40 ? 0.955 : 0.985);
            }
            this.socialContact = true;
          }
        }
      }
    }
    this.socialState.mode = activeSocialMode;
    if (!guiParams.isPaused) {
      if (this.socialContact) {
        this.socialState.duration += dt;
        const breakThreshold = DATASET_SCENARIO === 'calms_resident_intruder' ? 0.65 : 1.45;
        if (this.socialState.duration > breakThreshold && (this.trackId || 1) > 1 && DATASET_SCENARIO !== 'social_huddle') {
          this.socialState.cooldown = 1.65 + random() * 1.1;
          this.socialState.duration = 0;
          this.boutState.mode = 'dart-trot';
          this.boutState.timer = 1.15;
        }
      } else {
        this.socialState.duration = Math.max(0, this.socialState.duration - dt * 1.5);
      }
    }
    this.socialDistance = nearestSocialDist;

    const preSpeed = this.velocity.length();
    const isCloseCropDataset = DATASET_CAPTURE && (
      DATASET_CAMERA_VIEW === 'top-close'
      || DATASET_CAMERA_VIEW === 'oblique-close'
      || DATASET_CAMERA_VIEW === 'bottom-up'
    );
    const speedCapScale = isCloseCropDataset ? 0.56 : 1.0;

    // Saltatory target-speed assist: brisk darts, moderate walks, genuine stationary sniff pauses
    if (!guiParams.isPaused && !isPausing && !commandActive) {
      const modeCruise = this.socialState.mode === 'evasive-dart'
        ? 0.96
        : (this.socialState.mode === 'anogenital-follow'
          ? 0.78
          : (this.socialContact && DATASET_SCENARIO === 'social_huddle'
            ? 0.16
            : (this.boutState.mode === 'dart-trot'
              ? 0.92
              : (this.boutState.mode === 'stretch-attend' ? 0.24 : 0.54))));
      const cruiseTarget = Math.max(
        this.guiParams.maxSpeed * speedCapScale * modeCruise,
        this.socialContact ? 0.85 : 1.65
      );
      if (preSpeed < cruiseTarget) {
        const deficit = (cruiseTarget - preSpeed) / Math.max(cruiseTarget, 1e-3);
        const cruiseDir = this.heading.lengthSq() > 1e-6
          ? this.heading.clone().normalize()
          : new THREE.Vector2(Math.cos(t), Math.sin(t)).normalize();
        const cruiseBoost = (this.guiParams.wanderForce * 5.2 + 12.0) * deficit;
        this.force.x += cruiseDir.x * cruiseBoost;
        this.force.y += cruiseDir.y * cruiseBoost;
      }
    }

    // Drag (firmer braking during investigative pauses so feet plant crisply without skating).
    const dragC = this.guiParams.dragCoeff * (
      isPausing ? 4.2 : (this.boutState.mode === 'dart-trot' || this.socialState.mode === 'evasive-dart' ? 0.72 : (preSpeed < 0.9 ? 0.55 : 0.92))
    );
    this.force.x -= this.velocity.x * dragC;
    this.force.y -= this.velocity.y * dragC;

    // Low-speed rescue: inject motion only if stalled unintentionally outside a pause/social bout.
    if (!guiParams.isPaused && !isPausing && !this.socialContact && !commandActive) {
      if (preSpeed < 0.85) this.lowSpeedTimer += dt;
      else this.lowSpeedTimer = Math.max(0, this.lowSpeedTimer - dt * 0.5);
      if (this.lowSpeedTimer > 1.05) {
        const rescueDir = new THREE.Vector2(
          nearest.tangentX * this.wallFollowDir + nearest.normalX * 0.55,
          nearest.tangentZ * this.wallFollowDir + nearest.normalZ * 0.55
        );
        if (rescueDir.lengthSq() < 1e-6) {
          if (this.heading.lengthSq() > 1e-6) rescueDir.copy(this.heading);
          else rescueDir.set(Math.cos(t), Math.sin(t));
        }
        rescueDir.normalize();
        const kickMag = 36.0;
        this.force.x += rescueDir.x * kickMag;
        this.force.y += rescueDir.y * kickMag;
        this.lowSpeedTimer = 0;
      }
    }

    // Integration (responsive murine mass scaling so darting bursts accelerate crisply)
    if (!guiParams.isPaused) {
      if (behaviorPause && !rearingActive && !groomingActive) {
        this.velocity.multiplyScalar(Math.exp(-9.0 * dt));
      }
      let speedCap;
      if (commandTracking) {
        // External command: first-order velocity tracking along a commanded
        // heading that integrates the commanded turn rate (FlyBody-style task
        // target). Wall and body contacts are still resolved below.
        const command = this.externalCommand;
        command.heading = Math.atan2(
          Math.sin(command.heading + command.turnRate * dt),
          Math.cos(command.heading + command.turnRate * dt)
        );
        const tracking = 1.0 - Math.exp(-6.0 * dt);
        this.velocity.x += (Math.sin(command.heading) * command.forwardSpeed - this.velocity.x) * tracking;
        this.velocity.y += (Math.cos(command.heading) * command.forwardSpeed - this.velocity.y) * tracking;
        speedCap = this.guiParams.maxSpeed * 1.24;
      } else {
        const effectiveInertia = this.mass * (isCloseCropDataset ? 1000 : 360);
        const ax = this.force.x / effectiveInertia;
        const az = this.force.y / effectiveInertia;
        this.velocity.x += ax * dt; this.velocity.y += az * dt;
        const dartMul = (this.boutState.mode === 'dart-trot' || this.socialState.mode === 'evasive-dart') ? 1.24 : 1.0;
        speedCap = this.guiParams.maxSpeed * speedCapScale * dartMul;
      }
      const cS = this.velocity.length();
      if (cS > speedCap) this.velocity.multiplyScalar(speedCap / cS);
      this.mesh.position.x += this.velocity.x * dt;
      this.mesh.position.z += this.velocity.y * dt;
      this.locomotionTime += this.velocity.length() * dt;
    }

    // Orientation-aware wall clearance: allows realistic close wall thigmotaxis (~1.55u flank clearance)
    // and corner exploration while keeping snout and rump strictly inside enclosure walls.
    let hitWall = false;
    let hitXWall = false;
    let hitZWall = false;
    const chonk = this.guiParams.chonkiness || 1.0;
    const bodyLenScale = this.guiParams.bodyLength || 1.0;
    const snoutLenScale = this.guiParams.snoutLength || 1.0;
    const uprightReduction = this.rearing.mode === 'wall-supported'
      ? this.rearAmount * 1.45
      : 0;
    const marginPosX = Math.max(1.25, computeOrientedWallMargin(
      this.heading.x, this.heading.y, -1, 0, bodyLenScale, snoutLenScale, chonk, this.spineFlex.stretch
    ) - uprightReduction);
    const marginNegX = Math.max(1.25, computeOrientedWallMargin(
      this.heading.x, this.heading.y, 1, 0, bodyLenScale, snoutLenScale, chonk, this.spineFlex.stretch
    ) - uprightReduction);
    const marginPosZ = Math.max(1.25, computeOrientedWallMargin(
      this.heading.x, this.heading.y, 0, -1, bodyLenScale, snoutLenScale, chonk, this.spineFlex.stretch
    ) - uprightReduction);
    const marginNegZ = Math.max(1.25, computeOrientedWallMargin(
      this.heading.x, this.heading.y, 0, 1, bodyLenScale, snoutLenScale, chonk, this.spineFlex.stretch
    ) - uprightReduction);
    const baseBodyMargin = computeOrientedWallMargin(
      this.heading.x, this.heading.y, nearest.normalX, nearest.normalZ,
      bodyLenScale, snoutLenScale, chonk, this.spineFlex.stretch
    );

    if (this.mesh.position.x > ENCLOSURE_HALF_X - marginPosX) { this.mesh.position.x = ENCLOSURE_HALF_X - marginPosX; this.velocity.x = -Math.abs(this.velocity.x) * 0.22; hitWall = true; hitXWall = true; }
    if (this.mesh.position.x < -ENCLOSURE_HALF_X + marginNegX) { this.mesh.position.x = -ENCLOSURE_HALF_X + marginNegX; this.velocity.x = Math.abs(this.velocity.x) * 0.22; hitWall = true; hitXWall = true; }
    if (this.mesh.position.z > ENCLOSURE_HALF_Z - marginPosZ) { this.mesh.position.z = ENCLOSURE_HALF_Z - marginPosZ; this.velocity.y = -Math.abs(this.velocity.y) * 0.22; hitWall = true; hitZWall = true; }
    if (this.mesh.position.z < -ENCLOSURE_HALF_Z + marginNegZ) { this.mesh.position.z = -ENCLOSURE_HALF_Z + marginNegZ; this.velocity.y = Math.abs(this.velocity.y) * 0.22; hitWall = true; hitZWall = true; }
    if (hitWall) {
      const minTurnSpeed = Math.max(this.guiParams.maxSpeed * 0.42, 2.0);
      const clearPosX = ENCLOSURE_HALF_X - this.mesh.position.x;
      const clearNegX = ENCLOSURE_HALF_X + this.mesh.position.x;
      const clearPosZ = ENCLOSURE_HALF_Z - this.mesh.position.z;
      const clearNegZ = ENCLOSURE_HALF_Z + this.mesh.position.z;
      if (hitXWall && !hitZWall) {
        const turnSignZ = clearPosZ >= clearNegZ ? 1 : -1;
        this.velocity.y = turnSignZ * Math.max(Math.abs(this.velocity.y), minTurnSpeed);
        this.wallFollowDir = turnSignZ;
      } else if (hitZWall && !hitXWall) {
        const turnSignX = clearPosX >= clearNegX ? 1 : -1;
        this.velocity.x = turnSignX * Math.max(Math.abs(this.velocity.x), minTurnSpeed);
        this.wallFollowDir = turnSignX;
      } else {
        // Corner: choose axis with more free space.
        if (Math.max(clearPosX, clearNegX) >= Math.max(clearPosZ, clearNegZ)) {
          const turnSignX = clearPosX >= clearNegX ? 1 : -1;
          this.velocity.x = turnSignX * minTurnSpeed;
        } else {
          const turnSignZ = clearPosZ >= clearNegZ ? 1 : -1;
          this.velocity.y = turnSignZ * minTurnSpeed;
        }
        this.wallFollowDir *= -1;
      }
      // Heading will reorient through angular inertia below.
    }
    const impactNorm = Math.min(
      this.velocity.length() / Math.max(this.guiParams.maxSpeed, 1e-3),
      1.0
    );
    if (!guiParams.isPaused) {
      const behaviorContext = {
        hitWall,
        impactNorm,
        nearestWall: nearest,
        wallDist,
        bodyMargin: baseBodyMargin,
        behaviorPause,
        speed: this.velocity.length(),
      };
      this.updateGroomingBehavior(time, dt, behaviorContext);
      this.updateRearingBehavior(time, dt, behaviorContext);
    }

    // Heading with responsive murine pivot agility
    const speed = this.velocity.length();
    let isMoving = speed > 0.26
      && this.rearing.mode === 'none'
      && this.grooming.mode === 'none';
    if (!guiParams.isPaused) {
      let targetYaw = this.yawAngle;
      if (this.rearing.mode === 'wall-supported') {
        targetYaw = Math.atan2(
          this.rearing.supportDirection.x,
          this.rearing.supportDirection.y
        );
      } else if (commandTracking) {
        // Commanded heading also covers in-place turns and backward walking.
        targetYaw = this.externalCommand.heading;
      } else if (isMoving) {
        targetYaw = Math.atan2(this.velocity.x, this.velocity.y);
      } else if (this.socialContact && this.socialTarget.lengthSq() > 1e-6) {
        // Slowly orient forebody toward social partner while investigating
        const socialYaw = Math.atan2(this.socialTarget.x, this.socialTarget.y);
        const diff = Math.atan2(Math.sin(socialYaw - this.yawAngle), Math.cos(socialYaw - this.yawAngle));
        targetYaw = this.yawAngle + THREE.MathUtils.clamp(diff, -0.35, 0.35);
      } else if (this.force.lengthSq() > 1e-6 && !behaviorPause) {
        targetYaw = Math.atan2(this.force.x, this.force.y);
      }
      const yawErr = Math.atan2(Math.sin(targetYaw - this.yawAngle), Math.cos(targetYaw - this.yawAngle));
      this.yawErr = yawErr;
      const yawK = this.rearing.mode === 'wall-supported'
        ? 24.0
        : (isColliding ? 20.0 : 15.5);
      const yawD = 7.4;
      this.yawVel += (yawK * yawErr - yawD * this.yawVel) * dt;
      const maxYawRate = this.rearing.mode === 'wall-supported'
        ? 3.2
        : (isColliding ? 3.4 : 3.0);
      this.yawVel = THREE.MathUtils.clamp(this.yawVel, -maxYawRate, maxYawRate);
      this.yawAngle += this.yawVel * dt;
      this.yawAngle = Math.atan2(Math.sin(this.yawAngle), Math.cos(this.yawAngle));
    } else {
      this.yawErr = 0;
    }
    this.heading.set(Math.sin(this.yawAngle), Math.cos(this.yawAngle));
    this.mesh.rotation.y = this.yawAngle;

    if (hudRef) {
      let stateText = "EXPLORING";
      let stateColor = "#2f3c30";
      if (this.rearAmount > 0.10 && this.rearing.mode === 'wall-supported') {
        stateText = this.rearing.phase === 'settle'
          ? "SETTLING // WALL REAR"
          : "SUPPORTED REARING";
        stateColor = "#6e4d2f";
      } else if (this.rearAmount > 0.10 && this.rearing.mode === 'unsupported') {
        stateText = this.rearing.phase === 'settle'
          ? "SETTLING // REARING"
          : "EXPLORATORY REARING";
        stateColor = "#6e4d2f";
      } else if (this.groomAmount > 0.08 && this.grooming.mode !== 'none') {
        if (this.grooming.phase === 'settle') {
          stateText = "SETTLING // GROOMING";
        } else {
          stateText = this.grooming.mode === 'face-wash'
            ? "FACE GROOMING"
            : "FLANK GROOMING";
        }
        stateColor = "#6e4d2f";
      } else if (this.socialState.mode === 'anogenital-follow') {
        stateText = "ANOGENITAL PURSUIT";
        stateColor = "#7a3e34";
      } else if (this.socialState.mode === 'nose-to-nose' || this.socialContact) {
        stateText = "SOCIAL SNIFFING";
        stateColor = "#7a3e34";
      } else if (this.boutState.mode === 'stretch-attend' && !isMoving) {
        stateText = "STRETCH-ATTEND";
        stateColor = "#6e4d2f";
      } else if (isPausing || !isMoving) {
        stateText = "PAUSE // SNIFFING";
        stateColor = "#6e4d2f";
      } else if (this.boutState.mode === 'dart-trot') {
        stateText = "DARTING BOUT";
        stateColor = "#35543d";
      } else if (wallDist < 3.2) {
        stateText = "WALL THIGMOTAXIS";
        stateColor = "#526247";
      }
      hudRef.state.innerText = stateText;
      hudRef.state.style.color = stateColor;
      hudRef.hr.innerText = isMoving && !isPausing ? "560 BPM" : "415 BPM";
      // Display estimates must not consume the behavioral random stream.
      hudRef.resp.innerText = isMoving && !isPausing ? "215 BPM" : (162 + Math.round(Math.sin(time * 0.7) * 7)) + " BPM";
      hudRef.coord.innerText = `${this.mesh.position.x.toFixed(2)}, ${this.mesh.position.z.toFixed(2)}`;
      if (hudRef.gait) {
        hudRef.gait.innerText = this.motor.isStepping()
          ? `${this.motor.cpg.gait.toUpperCase()} · ${this.motor.cpg.frequencyHz.toFixed(1)} HZ`
          : 'STANDING';
      }
    }

    // Body + spine + tail dynamics (gravity + inertia + vertebral flexion + concertina elongation)
    const speedNorm = Math.min(speed / this.maxSpeed, 1.15);
    const invDt = dt > 1e-5 ? 1.0 / dt : 0.0;
    const accelX = (this.velocity.x - this.prevVelocity.x) * invDt;
    const accelZ = (this.velocity.y - this.prevVelocity.y) * invDt;
    const forwardAccel = accelX * this.heading.x + accelZ * this.heading.y;
    const lateralAccel = accelX * (-this.heading.y) + accelZ * this.heading.x;
    const headingAngle = Math.atan2(this.heading.x, this.heading.y);
    let headingDelta = headingAngle - this.prevHeadingAngle;
    if (headingDelta > Math.PI) headingDelta -= Math.PI * 2;
    if (headingDelta < -Math.PI) headingDelta += Math.PI * 2;
    const turnRate = headingDelta * invDt;
    this.prevHeadingAngle = headingAngle;

    const rearPoseSuppress = 1.0 - THREE.MathUtils.clamp(
      this.rearAmount / Math.max(this.rearing.peakAmount || 1.0, 0.01),
      0,
      1
    );
    const groomPoseNorm = THREE.MathUtils.clamp(
      this.groomAmount / Math.max(this.grooming.peakAmount || 1.0, 0.01),
      0,
      1
    );
    const isFaceWash = this.grooming.mode === 'face-wash';
    const isFlankGroom = this.grooming.mode === 'flank-groom';
    const qaRigidLock = IS_ANATOMY_QA || Boolean(REARING_QA) || CONTACT_QA === 'framing';

    if (qaRigidLock) {
      this.spineFlex.bend = 0;
      this.spineFlex.bendV = 0;
      this.spineFlex.sCurve = 0;
      this.spineFlex.sCurveV = 0;
      this.spineFlex.arch = 0;
      this.spineFlex.archV = 0;
      this.spineFlex.stretch = 0;
      this.spineFlex.stretchV = 0;
    } else if (GROOMING_QA === 'flank-groom' && guiParams.isPaused) {
      this.spineFlex.bend = this.grooming.side * 0.42;
      this.spineFlex.stretch = -0.10;
      this.spineFlex.arch = 0.08;
      this.spineFlex.sCurve = 0;
    } else if (!guiParams.isPaused) {
      // Vertical body suspension with gravity (slightly sleeker crouch when darting).
      const crouchOffset = this.boutState.mode === 'dart-trot' ? -0.025 : 0.0;
      const targetBodyY = 1.16 + Math.min(speedNorm, 1.0) * 0.035 + crouchOffset;
      const bodyK = 24.0, bodyD = 8.0, gravity = 3.4;
      this.bodyDyn.vy += (bodyK * (targetBodyY - this.bodyDyn.y) - bodyD * this.bodyDyn.vy - gravity) * dt;
      this.bodyDyn.y += this.bodyDyn.vy * dt;
      if (this.bodyDyn.y < 1.03) {
        this.bodyDyn.y = 1.03;
        this.bodyDyn.vy = Math.max(0, this.bodyDyn.vy) * 0.2;
      }

      // Pitch/roll inertia from accel and turning.
      const targetPitch = THREE.MathUtils.clamp(-forwardAccel * 0.015 + speedNorm * 0.02, -0.16, 0.16);
      const targetRoll = THREE.MathUtils.clamp(-lateralAccel * 0.02 - turnRate * 0.01, -0.20, 0.20);
      const angK = 18.0, angD = 6.5;
      this.bodyDyn.pitchV += (angK * (targetPitch - this.bodyDyn.pitch) - angD * this.bodyDyn.pitchV) * dt;
      this.bodyDyn.rollV += (angK * (targetRoll - this.bodyDyn.roll) - angD * this.bodyDyn.rollV) * dt;
      this.bodyDyn.pitch += this.bodyDyn.pitchV * dt;
      this.bodyDyn.roll += this.bodyDyn.rollV * dt;

      // 1. Lateral spinal flexion (C-bend into turns, investigative scans, social arcs, and flank grooming)
      let socialSpineLead = 0;
      if (this.socialContact && this.socialTarget.lengthSq() > 1e-6) {
        const sYaw = Math.atan2(this.socialTarget.x, this.socialTarget.y);
        const sRel = Math.atan2(Math.sin(sYaw - this.yawAngle), Math.cos(sYaw - this.yawAngle));
        socialSpineLead = THREE.MathUtils.clamp(sRel * 0.42, -0.28, 0.28);
      }
      const idleScanLead = (!isMoving && !this.socialContact && !groomingActive && !rearingActive)
        ? Math.sin(this.boutState.scanPhase) * 0.12
        : 0;
      const flankCurl = isFlankGroom ? this.grooming.side * groomPoseNorm * 0.44 : 0;
      const faceWashSuppress = isFaceWash ? (1.0 - groomPoseNorm * 0.92) : 1.0;
      const rawTargetBend = (
        (turnRate * 0.145 + this.yawErr * 0.34 - lateralAccel * 0.012 + socialSpineLead + idleScanLead)
        * rearPoseSuppress * faceWashSuppress
      ) + flankCurl;
      const targetBend = THREE.MathUtils.clamp(rawTargetBend, -0.56, 0.56);

      // 2. Subtle S-curve counter-undulation phase-locked to the hindlimb CPG
      const gaitWave = isMoving ? this.motor.spinalWave() * 0.045 : 0;
      const targetSCurve = THREE.MathUtils.clamp(
        (gaitWave - this.yawErr * 0.08) * rearPoseSuppress * faceWashSuppress,
        -0.18,
        0.18
      );

      // 3. Concertina axial squash-and-stretch (matches CalMS21 mouse083 148px..232px axial range)
      let boutStretch = 0;
      if (isFlankGroom) {
        boutStretch = -0.11 * groomPoseNorm;
      } else if (isFaceWash) {
        boutStretch = -0.04 * groomPoseNorm;
      } else if (this.socialState.mode === 'nose-to-nose' || this.boutState.mode === 'stretch-attend') {
        boutStretch = 0.14;
      } else if (this.boutState.mode === 'dart-trot' && isMoving) {
        boutStretch = 0.08 + Math.min(speedNorm, 1.1) * 0.10;
      } else if (isMoving) {
        boutStretch = -0.02 + Math.min(speedNorm, 1.0) * 0.11 + THREE.MathUtils.clamp(forwardAccel * 0.008, -0.05, 0.06);
      } else {
        boutStretch = -0.085;
      }
      const targetStretch = THREE.MathUtils.clamp(
        boutStretch * rearPoseSuppress,
        -0.16,
        0.21
      );

      // 4. Coupled dorsal kyphotic arch (hunches when bunched/grooming, flattens sleekly when darting)
      const targetArch = THREE.MathUtils.clamp(
        (-targetStretch * 0.42 + (isFlankGroom ? groomPoseNorm * 0.075 : 0)) * rearPoseSuppress,
        -0.08,
        0.14
      );

      const spK = 28.0, spD = 9.2;
      this.spineFlex.bendV += (spK * (targetBend - this.spineFlex.bend) - spD * this.spineFlex.bendV) * dt;
      this.spineFlex.sCurveV += (spK * (targetSCurve - this.spineFlex.sCurve) - spD * this.spineFlex.sCurveV) * dt;
      this.spineFlex.stretchV += (22.0 * (targetStretch - this.spineFlex.stretch) - 8.2 * this.spineFlex.stretchV) * dt;
      this.spineFlex.archV += (22.0 * (targetArch - this.spineFlex.arch) - 8.2 * this.spineFlex.archV) * dt;
      this.spineFlex.bend += this.spineFlex.bendV * dt;
      this.spineFlex.sCurve += this.spineFlex.sCurveV * dt;
      this.spineFlex.stretch += this.spineFlex.stretchV * dt;
      this.spineFlex.arch += this.spineFlex.archV * dt;

      // Ear pin-back during brisk darts, grooming, or close evasive turns
      const targetEarPin = (this.boutState.mode === 'dart-trot' && speedNorm > 0.65)
        ? 0.16
        : (this.socialContact ? 0.08 : 0.0);
      this.earPinAmount += (targetEarPin - this.earPinAmount) * Math.min(1.0, dt * 8.0);

      // Tail inertial follow-through.
      const groomingTailSway = this.grooming.mode === 'flank-groom'
        ? this.grooming.side * this.groomAmount * 0.18
        : 0;
      const targetTailSway = THREE.MathUtils.clamp(
        -turnRate * 0.065 - lateralAccel * 0.012 + groomingTailSway,
        -0.9,
        0.9
      );
      const targetTailLift = THREE.MathUtils.clamp(
        -0.05 - forwardAccel * 0.004 + speedNorm * 0.035
          + this.rearAmount * 0.10 + this.groomAmount * 0.025,
        -0.18,
        0.15
      );
      const tailK = 14.0, tailD = 4.8;
      this.tailDyn.swayV += (tailK * (targetTailSway - this.tailDyn.sway) - tailD * this.tailDyn.swayV) * dt;
      this.tailDyn.liftV += (tailK * (targetTailLift - this.tailDyn.lift) - tailD * this.tailDyn.liftV) * dt;
      this.tailDyn.sway += this.tailDyn.swayV * dt;
      this.tailDyn.lift += this.tailDyn.liftV * dt;
      if (speed < 0.45 && this.lowSpeedTimer > 0.6) {
        const calm = Math.exp(-8.0 * dt);
        this.tailDyn.sway *= calm;
        this.tailDyn.lift *= calm;
        this.tailDyn.swayV *= calm;
        this.tailDyn.liftV *= calm;
      }
    }

    this.rig.position.y = this.bodyDyn.y;
    this.rig.position.z = 0;
    this.rig.rotation.x = this.bodyDyn.pitch;
    this.rig.rotation.z = this.bodyDyn.roll;
    this.applySpinalRigTransforms(chonk, bodyLenScale);
    const spineHeadBaseX = this.headGroup.position.x;
    const spineHeadBaseY = this.headGroup.position.y;
    const spineHeadBaseZ = this.headGroup.position.z;

    // Contact shadow sits on the arena floor (this.mesh y=0, NOT this.rig)
    // and follows the curved & stretched murine footprint with subtle height attenuation.
    this.contactShadow.position.set(
      this.spineFlex.bend * 0.18 * chonk,
      0.018,
      0
    );
    this.contactShadow.scale.set(
      THREE.MathUtils.clamp(1.0 - this.spineFlex.stretch * 0.35, 0.82, 1.15) * chonk,
      THREE.MathUtils.clamp(1.0 + this.spineFlex.stretch * 0.85, 0.80, 1.24) * bodyLenScale,
      1.0
    );
    if (this.contactShadow.material) {
      this.contactShadow.material.opacity = THREE.MathUtils.clamp(
        0.92 - Math.max(0, this.rig.position.y - 1.16) * 0.28,
        0.38,
        0.95
      );
    }

    // ═══ LOCOMOTOR CONTROL (descending drive -> limb CPG -> joint targets) ═══
    // Mirrors the FlyBody brain/VNC pipeline: body state sets a descending
    // drive, coupled limb oscillators produce gait-specific phases, and the
    // decoder writes named joint targets clipped to MOUSE_JOINT_SPEC ranges.
    // Rearing/grooming poses below blend on top of these targets.
    const forwardSpeed = this.velocity.x * this.heading.x + this.velocity.y * this.heading.y;
    const motorTargets = this.motor.step(guiParams.isPaused ? 0 : dt, {
      speed,
      forwardSpeed,
      maxSpeed: this.maxSpeed,
      yawRate: this.yawVel,
      posture: rearingActive ? 'rearing' : (groomingActive ? 'grooming' : 'none'),
      dart: this.boutState.mode === 'dart-trot' || this.socialState.mode === 'evasive-dart',
      gaitOverride: this.externalCommand?.gait ?? guiParams.gaitMode ?? 'auto',
      gaitAmpMul: this.motion.gaitAmpMul,
      idleSway: isMoving ? 0 : Math.sin(time * 1.3 + this.timeOffset) * 0.008,
      liftOff: this.getLiftOffRequests(),
    });
    this.applyJointTargets(motorTargets);
    this.rig.position.y += this.motor.bodyOscillation() * 0.018;

    const rear = this.rearAmount;
    if (rear > 0 && !this.rearing.rearFeetPlanted) {
      this.captureRearFootAnchors();
    }
    if (rear > 0) {
      const supported = this.rearing.mode === 'wall-supported';
      const poseAmount = THREE.MathUtils.clamp(
        rear / Math.max(this.rearing.peakAmount, 0.01),
        0,
        1
      );
      const balanceResponse = THREE.MathUtils.clamp(
        this.rearing.angularVelocity * 0.035
          - this.rearing.centerOfMassLateralOffset * 0.025,
        -0.065,
        0.065
      );
      this.rig.rotation.x -= this.rearing.angle;
      this.rig.rotation.z += poseAmount * (
        this.rearing.sideLean + (supported ? 0 : balanceResponse * 0.35)
      );

      const shoulderTargetL = supported ? -0.14 : -0.72 + balanceResponse;
      const shoulderTargetR = supported ? -0.14 : -0.72 - balanceResponse;
      const forearmTargetL = supported ? -0.58 : -1.20 + balanceResponse * 0.65;
      const forearmTargetR = supported ? -0.58 : -1.20 - balanceResponse * 0.65;
      const wristTarget = supported ? 0.22 : 0.36;
      const pawTarget = supported ? 0.04 : 0.18;
      this.blendForelimbPose(this.armL, {
        shoulderProtraction: supported ? 0.08 : 0.04,
        shoulderX: shoulderTargetL,
        shoulderZ: supported ? -0.08 : -0.18,
        elbowX: forearmTargetL,
        wristX: wristTarget,
        pawX: pawTarget,
      }, poseAmount);
      this.blendForelimbPose(this.armR, {
        shoulderProtraction: supported ? 0.08 : 0.04,
        shoulderX: shoulderTargetR,
        shoulderZ: supported ? 0.08 : 0.18,
        elbowX: forearmTargetR,
        wristX: wristTarget,
        pawX: pawTarget,
      }, poseAmount);

      this.hindL.rotation.x = THREE.MathUtils.lerp(this.hindL.rotation.x, 0.46, poseAmount);
      this.hindR.rotation.x = THREE.MathUtils.lerp(this.hindR.rotation.x, 0.46, poseAmount);
      this.shankL.rotation.x = THREE.MathUtils.lerp(this.shankL.rotation.x, 0.76, poseAmount);
      this.shankR.rotation.x = THREE.MathUtils.lerp(this.shankR.rotation.x, 0.76, poseAmount);
      if (this.hindL._ankle) this.hindL._ankle.rotation.x = THREE.MathUtils.lerp(this.hindL._ankle.rotation.x, -0.58, poseAmount);
      if (this.hindR._ankle) this.hindR._ankle.rotation.x = THREE.MathUtils.lerp(this.hindR._ankle.rotation.x, -0.58, poseAmount);
      if (this.hindL._paw) this.hindL._paw.rotation.x = THREE.MathUtils.lerp(this.hindL._paw.rotation.x, -0.04, poseAmount);
      if (this.hindR._paw) this.hindR._paw.rotation.x = THREE.MathUtils.lerp(this.hindR._paw.rotation.x, -0.04, poseAmount);
      this.contactShadow.position.set(0, 0.018, -poseAmount * 0.65);
      this.contactShadow.scale.set(
        chonk * THREE.MathUtils.lerp(1.0, 0.84, poseAmount),
        bodyLenScale * THREE.MathUtils.lerp(1.0, 0.62, poseAmount),
        1.0
      );
      if (this.contactShadow.material) {
        this.contactShadow.material.opacity = THREE.MathUtils.lerp(0.92, 0.72, poseAmount);
      }
    }
    const groom = this.groomAmount;
    const {
      leftWashStroke,
      rightWashStroke,
      washStroke,
      pawLead,
      pawLick,
      flankLick,
      flankPulse,
    } = this.getGroomingMotion();
    if (groom > 0) {
      const groomPose = THREE.MathUtils.clamp(
        groom / Math.max(this.grooming.peakAmount, 0.01),
        0,
        1
      );
      const faceWash = this.grooming.mode === 'face-wash';

      if (faceWash) {
        this.rig.position.y += groomPose * (0.68 + pawLick * 0.04);
        this.rig.position.z -= groomPose * 0.10;
        this.rig.rotation.x -= groomPose * (0.56 + pawLick * 0.04);
        this.rig.rotation.z += pawLead * groomPose * 0.040;

        this.blendForelimbPose(this.armL, {
          shoulderProtraction: 0.04 + leftWashStroke * 0.02,
          shoulderX: -0.82 - leftWashStroke * 0.38 - pawLick * 0.04,
          shoulderZ: -0.20 - leftWashStroke * 0.14,
          elbowX: -0.66 - leftWashStroke * 0.31 - pawLick * 0.05,
          wristX: 0.32 + leftWashStroke * 0.32,
          pawX: 0.14 + leftWashStroke * 0.27,
        }, groomPose);
        this.blendForelimbPose(this.armR, {
          shoulderProtraction: 0.04 + rightWashStroke * 0.02,
          shoulderX: -0.82 - rightWashStroke * 0.38 - pawLick * 0.04,
          shoulderZ: 0.20 + rightWashStroke * 0.14,
          elbowX: -0.66 - rightWashStroke * 0.31 - pawLick * 0.05,
          wristX: 0.32 + rightWashStroke * 0.32,
          pawX: 0.14 + rightWashStroke * 0.27,
        }, groomPose);
      } else {
        const side = this.grooming.side;
        this.rig.position.y -= groomPose * 0.04;
        this.rig.position.z -= groomPose * 0.08;
        this.rig.rotation.x += groomPose * 0.07;
        this.rig.rotation.z += side * groomPose * (0.10 + flankPulse * 0.020);

        const sameSideShoulder = 0.02 - flankLick * 0.04;
        const supportShoulder = 0.23 + flankLick * 0.015;
        this.blendForelimbPose(this.armL, {
          shoulderProtraction: 0.02,
          shoulderX: side > 0 ? sameSideShoulder : supportShoulder,
          shoulderZ: side > 0 ? -0.10 : -0.16,
          elbowX: side > 0 ? -0.30 + flankLick * 0.05 : -0.58,
          wristX: side > 0 ? 0.16 : 0.22,
          pawX: side > 0 ? 0.10 : 0.12,
        }, groomPose);
        this.blendForelimbPose(this.armR, {
          shoulderProtraction: 0.02,
          shoulderX: side < 0 ? sameSideShoulder : supportShoulder,
          shoulderZ: side < 0 ? 0.10 : 0.16,
          elbowX: side < 0 ? -0.30 + flankLick * 0.05 : -0.58,
          wristX: side < 0 ? 0.16 : 0.22,
          pawX: side < 0 ? 0.10 : 0.12,
        }, groomPose);
      }

      const hipTarget = faceWash ? 0.28 : 0.14;
      const shankTarget = faceWash ? 0.74 : 0.62;
      this.hindL.rotation.x = THREE.MathUtils.lerp(this.hindL.rotation.x, hipTarget, groomPose);
      this.hindR.rotation.x = THREE.MathUtils.lerp(this.hindR.rotation.x, hipTarget, groomPose);
      this.shankL.rotation.x = THREE.MathUtils.lerp(this.shankL.rotation.x, shankTarget, groomPose);
      this.shankR.rotation.x = THREE.MathUtils.lerp(this.shankR.rotation.x, shankTarget, groomPose);
      if (this.hindL._ankle) this.hindL._ankle.rotation.x = THREE.MathUtils.lerp(this.hindL._ankle.rotation.x, -0.48, groomPose);
      if (this.hindR._ankle) this.hindR._ankle.rotation.x = THREE.MathUtils.lerp(this.hindR._ankle.rotation.x, -0.48, groomPose);
      this.contactShadow.position.set(0, 0.018, -groomPose * 0.35);
      if (this.contactShadow.material) {
        this.contactShadow.material.opacity = THREE.MathUtils.lerp(0.92, 0.78, groomPose);
      }
    }
    for (const limb of Object.keys(this.contactFrame.limbStance)) {
      this.contactFrame.limbStance[limb] = this.motor.isLimbInStance(limb);
    }
    this.contactFrame.isMoving = isMoving;
    this.contactFrame.rear = rear;
    this.contactFrame.groom = groom;
    this.contactFrame.lateralAccel = lateralAccel;
    this.contactFrame.turnRate = turnRate;
    this.contactFrame.forwardAccel = forwardAccel;

    // Head animation with irregular sniff pulses + active 8 Hz exploratory whisking
    const calmIdle = !isMoving && speed < 0.45 && this.lowSpeedTimer > 0.6;
    // Irregular sniff: combine multiple frequencies for biological plausibility
    const sniffBase = Math.sin(time * 40.0 + this.timeOffset);
    const sniffMod = Math.sin(time * 27.3 + this.timeOffset * 1.7) * 0.4;
    const sniffBurst = Math.max(0, Math.sin(time * 3.2 + this.timeOffset)) * 0.6;
    const sniffPulse = (sniffBase + sniffMod) * (0.4 + sniffBurst) * this.motion.sniffAmpMul;
    const whiskSweep = (
      Math.sin(time * 50.2 + this.timeOffset * 0.9) * 0.085
      + Math.sin(time * 31.4 + this.timeOffset * 1.4) * 0.032
    ) * (calmIdle ? 0.22 : 1.0);
    if (this.socialContact) {
      const targetYaw = Math.atan2(this.socialTarget.x, this.socialTarget.y);
      let relYaw = targetYaw - (this.yawAngle + this.spineHeadYaw);
      if (relYaw > Math.PI) relYaw -= Math.PI * 2;
      if (relYaw < -Math.PI) relYaw += Math.PI * 2;
      this.headGroup.rotation.y = this.spineHeadYaw
        + THREE.MathUtils.clamp(relYaw, -0.52, 0.52) + sniffPulse * 0.032;
      this.headGroup.rotation.x = -0.075 + sniffPulse * 0.030;
      this.snoutTip.position.y = this.noseRestPosition.y + sniffPulse * 0.024;
      const noseScalePulse = 1.0 + sniffPulse * 0.018;
      this.snoutTip.scale.z = this.noseBaseScale.z * this.guiParams.snoutLength * noseScalePulse;
      this.padL.rotation.y = 0.14 + sniffPulse * 0.060 + whiskSweep * 1.20;
      this.padR.rotation.y = -0.14 - sniffPulse * 0.060 - whiskSweep * 1.20;
      this.padL.rotation.x = -0.015 + sniffPulse * 0.008;
      this.padR.rotation.x = -0.015 + sniffPulse * 0.008;
      this.padL.rotation.z = sniffPulse * 0.009;
      this.padR.rotation.z = -sniffPulse * 0.009;
    } else if (!isMoving) {
      if (calmIdle) {
        this.headGroup.rotation.y = THREE.MathUtils.lerp(this.headGroup.rotation.y, this.spineHeadYaw, 0.15);
        this.headGroup.rotation.x *= 0.85;
        this.snoutTip.position.y = this.noseRestPosition.y + sniffPulse * 0.006;
        this.padL.rotation.y = 0.045 + sniffPulse * 0.012 + whiskSweep;
        this.padR.rotation.y = -0.045 - sniffPulse * 0.012 - whiskSweep;
        this.padL.rotation.x = -0.008;
        this.padR.rotation.x = -0.008;
        this.padL.rotation.z = 0;
        this.padR.rotation.z = 0;
      } else {
        const activeScan = !guiParams.isPaused
          ? Math.sin(this.boutState.scanPhase) * 0.24
          : 0;
        this.headGroup.rotation.y = this.spineHeadYaw + (
          activeScan
          + Math.sin(time * 2.1 + this.timeOffset * 0.03) * 0.12
          + Math.sin(time * 0.73 + this.timeOffset) * 0.045
        ) * this.motion.headIdleYawMul;
        this.headGroup.rotation.x = (
          Math.sin(time * 3.2 + this.timeOffset * 0.05) * 0.048
          + sniffPulse * 0.018
        ) * this.motion.headIdlePitchMul;
        this.snoutTip.position.y = this.noseRestPosition.y + sniffPulse * 0.022;
        const noseScalePulse = 1.0 + sniffPulse * 0.016;
        this.snoutTip.scale.z = this.noseBaseScale.z * this.guiParams.snoutLength * noseScalePulse;
        this.padL.rotation.y = sniffPulse * 0.045 + 0.085 + whiskSweep;
        this.padR.rotation.y = -sniffPulse * 0.045 - 0.085 - whiskSweep;
        this.padL.rotation.x = -0.010 + sniffPulse * 0.006;
        this.padR.rotation.x = -0.010 + sniffPulse * 0.006;
        this.padL.rotation.z = sniffPulse * 0.007;
        this.padR.rotation.z = -sniffPulse * 0.007;
      }
    } else {
      const turnAmount = this.velocity.x * this.heading.y - this.velocity.y * this.heading.x;
      // Expressive rodent head-lead into turns + vestibulo-colic pitch stabilization
      const headLeadYaw = THREE.MathUtils.clamp(
        this.yawErr * 0.42 + turnRate * 0.045 + turnAmount * 0.045,
        -0.44,
        0.44
      );
      const headStab = Math.sin(this.locomotionTime * 3.8) * 0.018;
      this.headGroup.rotation.set(
        Math.sin(time * 1.8) * 0.026 + headStab - Math.min(speedNorm, 1.0) * 0.022,
        this.spineHeadYaw + headLeadYaw,
        -headLeadYaw * 0.08
      );
      this.snoutTip.position.y = this.noseRestPosition.y + sniffPulse * 0.010;
      this.padL.rotation.y = -0.06 + whiskSweep * 1.25;
      this.padR.rotation.y = 0.06 - whiskSweep * 1.25;
      this.padL.rotation.x = 0.010; this.padR.rotation.x = 0.010;
      this.padL.rotation.z = 0; this.padR.rotation.z = 0;
    }
    if (rear > 0) {
      const supported = this.rearing.mode === 'wall-supported';
      const poseAmount = THREE.MathUtils.clamp(
        rear / Math.max(this.rearing.peakAmount, 0.01),
        0,
        1
      );
      this.headGroup.rotation.x += poseAmount * (supported ? 0.72 : 0.66);
      if (!supported) {
        this.headGroup.rotation.y += Math.sin(
          this.rearing.elapsed * 2.0 + this.timeOffset * 0.015
        ) * poseAmount * 0.07;
      }
    }
    if (this.lowerLip && this.lowerLipRestY !== undefined) {
      const lickOpen = groom > 0
        ? THREE.MathUtils.clamp(groom / Math.max(this.grooming.peakAmount, 0.01), 0, 1)
          * (this.grooming.mode === 'face-wash' ? pawLick : flankLick) * 0.006
        : 0;
      const sniffTwitch = !guiParams.isPaused && !calmIdle
        ? Math.max(0, sniffPulse) * 0.0018
        : 0;
      this.lowerLip.position.y = this.lowerLipRestY - lickOpen - sniffTwitch;
    }
    if (groom > 0) {
      const groomPose = THREE.MathUtils.clamp(
        groom / Math.max(this.grooming.peakAmount, 0.01),
        0,
        1
      );
      if (this.grooming.mode === 'face-wash') {
        this.headGroup.position.y -= groomPose * (
          0.05 + washStroke * 0.025 + pawLick * 0.015
        );
        this.headGroup.position.z -= groomPose * 0.04;
        this.headGroup.rotation.x = THREE.MathUtils.lerp(
          this.headGroup.rotation.x,
          0.50 + washStroke * 0.13 + pawLick * 0.05,
          groomPose
        );
        this.headGroup.rotation.y = THREE.MathUtils.lerp(
          this.headGroup.rotation.y,
          -pawLead * 0.13,
          groomPose
        );
        this.headGroup.rotation.z = THREE.MathUtils.lerp(
          this.headGroup.rotation.z,
          pawLead * 0.055,
          groomPose
        );
        this.snoutTip.position.y -= groomPose * (
          washStroke * 0.017 + pawLick * 0.014
        );
      } else {
        const side = this.grooming.side;
        this.headGroup.position.x += side * groomPose * (0.08 + flankLick * 0.025);
        this.headGroup.position.y -= groomPose * (0.02 + flankLick * 0.035);
        this.headGroup.position.z -= groomPose * (0.12 + flankLick * 0.04);
        this.headGroup.rotation.x = THREE.MathUtils.lerp(
          this.headGroup.rotation.x,
          0.15 + flankLick * 0.09,
          groomPose
        );
        this.headGroup.rotation.y = THREE.MathUtils.lerp(
          this.headGroup.rotation.y,
          side * (1.30 + flankLick * 0.12),
          groomPose
        );
        this.headGroup.rotation.z = THREE.MathUtils.lerp(
          this.headGroup.rotation.z,
          -side * (0.18 + flankLick * 0.035),
          groomPose
        );
        this.snoutTip.position.y -= groomPose * (0.014 + flankLick * 0.026);
      }
    }

    // The cervical mantle follows a fraction of local head translation and
    // extra atlanto-occipital rotation relative to the curved cervical spine.
    const headOffsetX = this.headGroup.position.x - spineHeadBaseX;
    const headOffsetY = this.headGroup.position.y - spineHeadBaseY;
    const headOffsetZ = this.headGroup.position.z - spineHeadBaseZ;
    this.neckMesh.position.x += headOffsetX * 0.48;
    this.neckMesh.position.y += headOffsetY * 0.36;
    this.neckMesh.position.z += headOffsetZ * 0.32;
    this.neckMesh.rotation.x += this.headGroup.rotation.x * 0.24;
    this.neckMesh.rotation.y += (this.headGroup.rotation.y - this.spineHeadYaw) * 0.32;
    this.neckMesh.rotation.z += this.headGroup.rotation.z * 0.26;

    // ═══ EAR MICRO-TWITCHES & DYNAMIC PIN-BACK ═══
    if (this.earL && this.earR && this.earMicro.baseRotL) {
      if (!guiParams.isPaused) {
        this.earMicro.twitchTimerL += dt;
        this.earMicro.twitchTimerR += dt;
        if (this.earMicro.twitchTimerL > this.earMicro.nextIntervalL) {
          this.earMicro.twitchAngleL = (random() - 0.5) * 0.08;
          this.earMicro.twitchTimerL = 0;
          this.earMicro.nextIntervalL = 0.5 + random() * 4.0;
        }
        if (this.earMicro.twitchTimerR > this.earMicro.nextIntervalR) {
          this.earMicro.twitchAngleR = (random() - 0.5) * 0.08;
          this.earMicro.twitchTimerR = 0;
          this.earMicro.nextIntervalR = 0.5 + random() * 4.0;
        }
        // Smooth exponential decay
        this.earMicro.twitchAngleL *= Math.exp(-4.0 * dt);
        this.earMicro.twitchAngleR *= Math.exp(-4.0 * dt);
      }
      // Locomotion secondary bounce + aerodynamic/social ear pin-back
      const earBounce = isMoving ? Math.sin(this.locomotionTime * 5.0) * 0.022 : 0;
      const earScan = Math.sin(time * 0.72 + this.timeOffset * 0.013) * 0.026;
      const pin = this.earPinAmount || 0;
      this.earL.rotation.z = this.earMicro.baseRotL.z + this.earMicro.twitchAngleL + earBounce - pin * 0.28;
      this.earR.rotation.z = this.earMicro.baseRotR.z + this.earMicro.twitchAngleR - earBounce + pin * 0.28;
      this.earL.rotation.x = this.earMicro.baseRotL.x - pin * 0.42 + this.earMicro.twitchAngleL * 0.5;
      this.earR.rotation.x = this.earMicro.baseRotR.x - pin * 0.42 + this.earMicro.twitchAngleR * 0.5;
      this.earL.rotation.y = this.earMicro.baseRotL.y + earScan + pin * 0.16 + this.earMicro.twitchAngleL * 0.25;
      this.earR.rotation.y = this.earMicro.baseRotR.y - earScan - pin * 0.16 + this.earMicro.twitchAngleR * 0.25;
      if (groom > 0) {
        const groomPose = THREE.MathUtils.clamp(groom, 0, 1);
        if (this.grooming.mode === 'face-wash') {
          this.earL.rotation.x += groomPose * leftWashStroke * 0.075;
          this.earR.rotation.x += groomPose * rightWashStroke * 0.075;
          this.earL.rotation.z -= groomPose * leftWashStroke * 0.050;
          this.earR.rotation.z += groomPose * rightWashStroke * 0.050;
        } else {
          const side = this.grooming.side;
          this.earL.rotation.y += side > 0
            ? groomPose * (0.07 + flankLick * 0.025)
            : -groomPose * 0.03;
          this.earR.rotation.y += side < 0
            ? -groomPose * (0.07 + flankLick * 0.025)
            : groomPose * 0.03;
        }
      }
    }

    // Subtle rodent blink/wince cycle. It briefly compresses the glossy eye
    // layer rather than adding human-like eyelids.
    if (this.eyeL && this.eyeR && this.blink) {
      if (!guiParams.isPaused) {
        this.blink.timer += dt;
        if (this.blink.timer > this.blink.interval) {
          this.blink.timer = 0;
          this.blink.interval = 2.6 + random() * 7.2;
          this.blink.phase = 0.0;
        }
        if (this.blink.phase < 1.0) {
          this.blink.phase = Math.min(1.0, this.blink.phase + dt * 6.5);
        }
      }
      const blinkAmount = this.blink.phase < 1.0
        ? Math.sin(this.blink.phase * Math.PI)
        : 0.0;
      const groomSquint = (groom > 0 && this.grooming.mode === 'face-wash')
        ? THREE.MathUtils.clamp(groom / Math.max(this.grooming.peakAmount, 0.01), 0, 1) * 0.34
        : 0.0;
      const closeAmount = Math.min(1.0, blinkAmount + groomSquint);
      const eyeY = 0.92 * (1.0 - closeAmount * 0.60);
      const corneaY = 0.92 * (1.0 - closeAmount * 0.62);
      this.eyeL.scale.y = eyeY;
      this.eyeR.scale.y = eyeY;
      this.corneaL.scale.y = corneaY;
      this.corneaR.scale.y = corneaY;
      if (this.upperLidL && this.lowerLidL && this.upperLidR && this.lowerLidR) {
        const lidScaleY = 0.88 * (1.0 - closeAmount * 0.52);
        const lowScaleY = 0.84 * (1.0 - closeAmount * 0.48);
        this.upperLidL.scale.y = lidScaleY;
        this.upperLidR.scale.y = lidScaleY;
        this.lowerLidL.scale.y = lowScaleY;
        this.lowerLidR.scale.y = lowScaleY;
      }
    }

    // Tail geometry is refreshed after final body contact resolution.
    // Tracking visibility
    this.syncTrackingVisibility(guiParams.showTrackingKeypoints);

    // Live adjustments — thorax/abdomen breathing differentiation
    const breathFreq = isMoving ? this.motion.breathMoveFreq : this.motion.breathIdleFreq;
    const breathAmp = isMoving ? this.motion.breathMoveAmp : this.motion.breathIdleAmp;
    const breathPhase = time * breathFreq;
    const thoraxBreath = Math.sin(breathPhase) * breathAmp;
    const abdomenBreath = Math.sin(breathPhase - 0.4) * breathAmp * 1.3; // abdomen lags & is larger
    this.bodyMesh.scale.x = this.guiParams.chonkiness + thoraxBreath * 0.35;
    this.bodyMesh.scale.y = this.guiParams.chonkiness + abdomenBreath * 0.3;
    this.bodyMesh.scale.z = this.guiParams.bodyLength + thoraxBreath;
    this.neckMesh.scale.set(
      this.guiParams.chonkiness + thoraxBreath * 0.18,
      this.guiParams.chonkiness + thoraxBreath * 0.12,
      1.0 + thoraxBreath * 0.10
    );
    this.head.scale.set(
      this.guiParams.headSize * 0.92,
      this.guiParams.headSize * 0.94,
      this.guiParams.headSize * 1.03
    );
    if (calmIdle || (isMoving && !this.socialContact)) {
      this.resetNoseScale();
    } else {
      this.snoutTip.scale.x = this.noseBaseScale.x;
      this.snoutTip.scale.y = this.noseBaseScale.y;
    }
    this.prevVelocity.copy(this.velocity);

    this.updateTrackingGeometry();
  }
}


function createBeddingLayer(count = 1100) {
  const group = new THREE.Group();
  // Curled, tapered aspen wood-shaving geometry (deterministic, zero RNG calls)
  const chipGeo = new THREE.PlaneGeometry(1, 1, 3, 5);
  chipGeo.rotateX(-Math.PI / 2);
  const chipPos = chipGeo.attributes.position;
  for (let idx = 0; idx < chipPos.count; idx++) {
    let vx = chipPos.getX(idx);
    const vz = chipPos.getZ(idx);
    const endTaper = 1.0 - Math.pow(Math.abs(vz * 2.0), 2.2) * 0.42;
    vx *= endTaper;
    const vy = Math.cos(vz * Math.PI) * 0.38 + vx * vx * 0.90 + Math.sin(vz * 4.0) * 0.12;
    chipPos.setXYZ(idx, vx, vy, vz);
  }
  chipGeo.computeVertexNormals();
  const chipMat = new THREE.MeshStandardMaterial({
    color: 0xd8c69e,
    roughness: 0.90,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const chips = new THREE.InstancedMesh(chipGeo, chipMat, count);
  chips.castShadow = false;
  chips.receiveShadow = true;
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  for (let i = 0; i < count; i++) {
    const nearWall = random() < 0.38;
    let x = (random() - 0.5) * (ENCLOSURE_W - 2.6);
    let z = (random() - 0.5) * (ENCLOSURE_D - 2.6);
    if (nearWall) {
      if (random() < 0.5) {
        x = (random() < 0.5 ? -1 : 1) * (ENCLOSURE_HALF_X - 1.0 - random() * 2.4);
      } else {
        z = (random() < 0.5 ? -1 : 1) * (ENCLOSURE_HALF_Z - 1.0 - random() * 2.4);
      }
    }
    dummy.position.set(x, 0.022 + random() * 0.028, z);
    dummy.rotation.set(
      (random() - 0.5) * 0.22,
      random() * Math.PI,
      (random() - 0.5) * 0.22
    );
    const len = 0.11 + random() * 0.36;
    const thin = 0.014 + random() * 0.032;
    dummy.scale.set(0.040 + random() * 0.072, thin, len);
    dummy.updateMatrix();
    chips.setMatrixAt(i, dummy.matrix);
    color.setHSL(
      0.102 + (random() - 0.5) * 0.028,
      0.24 + random() * 0.18,
      0.58 + random() * 0.20
    );
    chips.setColorAt(i, color);
  }
  group.add(chips);
  return group;
}

function createFloorMarks() {
  const group = new THREE.Group();
  const markGeo = new THREE.PlaneGeometry(1, 1);
  const addMark = (texture, x, z, sx, sz, opacity, rotation = 0) => {
    const mat = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity,
      depthWrite: false,
      color: 0xffffff,
    });
    const mesh = new THREE.Mesh(markGeo, mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.rotation.z = rotation;
    mesh.position.set(x, 0.034, z);
    mesh.scale.set(sx, sz, 1);
    mesh.renderOrder = 3;
    group.add(mesh);
  };
  for (let i = 0; i < 22; i++) {
    addMark(
      urineSpotTexture,
      (random() - 0.5) * (ENCLOSURE_W - 5.2),
      (random() - 0.5) * (ENCLOSURE_D - 4.8),
      0.7 + random() * 1.8,
      0.25 + random() * 1.1,
      0.30 + random() * 0.32,
      random() * Math.PI
    );
  }
  for (let i = 0; i < 34; i++) {
    addMark(
      shadowSmudgeTexture,
      (random() - 0.5) * (ENCLOSURE_W - 4.6),
      (random() - 0.5) * (ENCLOSURE_D - 4.4),
      0.35 + random() * 1.3,
      0.10 + random() * 0.55,
      0.15 + random() * 0.26,
      random() * Math.PI
    );
  }
  return group;
}

function createArenaHardware() {
  const group = new THREE.Group();
  const postMat = new THREE.MeshPhysicalMaterial({
    color: 0xc7c0b2,
    roughness: 0.34,
    metalness: 0.18,
    clearcoat: 0.18,
    clearcoatRoughness: 0.32,
  });
  const rimMat = new THREE.MeshPhysicalMaterial({
    color: 0xd8d2c4,
    roughness: 0.48,
    metalness: 0.06,
  });
  const postGeo = new THREE.CylinderGeometry(0.075, 0.075, ENCLOSURE_H, 16);
  for (const x of [-ENCLOSURE_W * 0.5, ENCLOSURE_W * 0.5]) {
    for (const z of [-ENCLOSURE_D * 0.5, ENCLOSURE_D * 0.5]) {
      const post = new THREE.Mesh(postGeo, postMat);
      post.position.set(x, ENCLOSURE_H * 0.5, z);
      post.castShadow = true;
      group.add(post);
    }
  }
  const railH = 0.16;
  const railY = ENCLOSURE_H + 0.02;
  const railZGeo = new THREE.BoxGeometry(0.18, railH, ENCLOSURE_D + 0.24);
  const railXGeo = new THREE.BoxGeometry(ENCLOSURE_W + 0.24, railH, 0.18);
  for (const x of [-ENCLOSURE_W * 0.5, ENCLOSURE_W * 0.5]) {
    const rail = new THREE.Mesh(railZGeo, rimMat);
    rail.position.set(x, railY, 0);
    rail.castShadow = true;
    group.add(rail);
  }
  for (const z of [-ENCLOSURE_D * 0.5, ENCLOSURE_D * 0.5]) {
    const rail = new THREE.Mesh(railXGeo, rimMat);
    rail.position.set(0, railY, z);
    rail.castShadow = true;
    group.add(rail);
  }
  // Subtle wall-base silicone seal & contact AO grounding the acrylic walls
  const baseTrimMat = new THREE.MeshStandardMaterial({
    color: 0x938e84,
    roughness: 0.72,
    metalness: 0.05,
  });
  const baseZGeo = new THREE.BoxGeometry(0.10, 0.07, ENCLOSURE_D + 0.08);
  const baseXGeo = new THREE.BoxGeometry(ENCLOSURE_W + 0.08, 0.07, 0.10);
  for (const x of [-ENCLOSURE_W * 0.5, ENCLOSURE_W * 0.5]) {
    const trim = new THREE.Mesh(baseZGeo, baseTrimMat);
    trim.position.set(x, 0.035, 0);
    group.add(trim);
  }
  for (const z of [-ENCLOSURE_D * 0.5, ENCLOSURE_D * 0.5]) {
    const trim = new THREE.Mesh(baseXGeo, baseTrimMat);
    trim.position.set(0, 0.035, z);
    group.add(trim);
  }
  return group;
}

function createHomeCageEnvironment() {
  // Isolated PRNG so enabling/constructing the shoebox home cage never perturbs subject RNG.
  const cageRng = createSeededRandom((simulationSeed ^ 0x5F3759DF) >>> 0);
  const group = new THREE.Group();
  group.name = 'home-cage-shoebox-environment';

  // Dark matte laboratory bench / optical breadboard surrounding the shoebox cage
  const benchMat = new THREE.MeshStandardMaterial({
    color: 0x16171a,
    roughness: 0.88,
    metalness: 0.06,
  });
  const benchPlane = new THREE.Mesh(new THREE.PlaneGeometry(96, 72), benchMat);
  benchPlane.rotation.x = -Math.PI / 2;
  benchPlane.position.y = -0.028;
  benchPlane.receiveShadow = true;
  group.add(benchPlane);

  // Molded sloped translucent polycarbonate rodent shoebox cage with rounded corners + rolled upper flange
  const innerHX = ENCLOSURE_HALF_X + 0.55; // 16.95 -> mice can press flanks right against inner wall
  const innerHZ = ENCLOSURE_HALF_Z + 0.40; // 11.80
  const wallH = 6.8;
  const segAround = 72;
  const ringProfiles = [
    { hx: innerHX, hz: innerHZ, y: 0.0, r: 1.35 },
    { hx: innerHX + 1.55, hz: innerHZ + 0.95, y: wallH * 0.48, r: 1.65 },
    { hx: innerHX + 3.15, hz: innerHZ + 1.92, y: wallH, r: 1.95 },
    { hx: innerHX + 4.10, hz: innerHZ + 2.65, y: wallH + 0.14, r: 2.15 },
    { hx: innerHX + 4.18, hz: innerHZ + 2.72, y: wallH - 0.18, r: 2.18 },
  ];
  const sampleRoundedRect = (theta, hx, hz, cornerR) => {
    const cr = Math.min(cornerR, hx * 0.45, hz * 0.45);
    const cx = Math.max(0.01, hx - cr);
    const cz = Math.max(0.01, hz - cr);
    const cosT = Math.cos(theta);
    const sinT = Math.sin(theta);
    // Superellipse-rounded rectangle perimeter with exact corner arc blend
    const absC = Math.abs(cosT);
    const absS = Math.abs(sinT);
    const scale = 1.0 / Math.max(absC / hx, absS / hz, 1e-4);
    let px = cosT * scale;
    let pz = sinT * scale;
    if (Math.abs(px) > cx && Math.abs(pz) > cz) {
      const dx = Math.abs(px) - cx;
      const dz = Math.abs(pz) - cz;
      const dLen = Math.hypot(dx, dz) || 1;
      px = Math.sign(px) * (cx + (dx / dLen) * cr);
      pz = Math.sign(pz) * (cz + (dz / dLen) * cr);
    }
    return [px, pz];
  };

  const wallPositions = [];
  const wallUVs = [];
  const wallIndices = [];
  const ringStride = segAround + 1;
  for (let rIdx = 0; rIdx < ringProfiles.length; rIdx++) {
    const prof = ringProfiles[rIdx];
    const vCoord = rIdx / (ringProfiles.length - 1);
    for (let sIdx = 0; sIdx <= segAround; sIdx++) {
      const uCoord = sIdx / segAround;
      const theta = uCoord * Math.PI * 2;
      const [px, pz] = sampleRoundedRect(theta, prof.hx, prof.hz, prof.r);
      wallPositions.push(px, prof.y, pz);
      wallUVs.push(uCoord * 4.0, vCoord);
    }
  }
  for (let rIdx = 0; rIdx < ringProfiles.length - 1; rIdx++) {
    for (let sIdx = 0; sIdx < segAround; sIdx++) {
      const a = rIdx * ringStride + sIdx;
      const b = a + 1;
      const c = (rIdx + 1) * ringStride + sIdx;
      const d = c + 1;
      wallIndices.push(a, c, b, b, c, d);
    }
  }
  const cageWallGeo = new THREE.BufferGeometry();
  cageWallGeo.setAttribute('position', new THREE.Float32BufferAttribute(wallPositions, 3));
  cageWallGeo.setAttribute('uv', new THREE.Float32BufferAttribute(wallUVs, 2));
  cageWallGeo.setIndex(wallIndices);
  cageWallGeo.computeVertexNormals();

  const polycarbonateMat = new THREE.MeshPhysicalMaterial({
    color: 0x6c6862,
    roughness: 0.20,
    metalness: 0.06,
    transmission: 0.34,
    thickness: 0.24,
    ior: 1.58,
    transparent: true,
    opacity: 0.86,
    clearcoat: 0.72,
    clearcoatRoughness: 0.16,
    side: THREE.DoubleSide,
  });
  const cageWalls = new THREE.Mesh(cageWallGeo, polycarbonateMat);
  cageWalls.receiveShadow = true;
  cageWalls.castShadow = true;
  cageWalls.renderOrder = 12;
  group.add(cageWalls);

  // Molded polycarbonate upper rim specular highlight loop (signature top/bottom cage lip in mouse083)
  const rimLoopPts = [];
  for (let sIdx = 0; sIdx <= segAround; sIdx++) {
    const theta = (sIdx / segAround) * Math.PI * 2;
    const [rx, rz] = sampleRoundedRect(theta, innerHX + 3.55, innerHZ + 2.24, 2.05);
    rimLoopPts.push(new THREE.Vector3(rx, wallH + 0.16, rz));
  }
  const rimCurve = new THREE.CatmullRomCurve3(rimLoopPts, true);
  const rimTubeGeo = new THREE.TubeGeometry(rimCurve, 96, 0.16, 8, true);
  const rimHighlightMat = new THREE.MeshPhysicalMaterial({
    color: 0x9b9892,
    roughness: 0.18,
    metalness: 0.12,
    transparent: true,
    opacity: 0.62,
    clearcoat: 0.85,
  });
  const rimTube = new THREE.Mesh(rimTubeGeo, rimHighlightMat);
  rimTube.renderOrder = 14;
  group.add(rimTube);

  // Subtle inner floor-wall meniscus cove darkening (bedding shadow in shoebox corner radius)
  const coveRingPositions = [];
  const coveIndices = [];
  for (let rIdx = 0; rIdx < 2; rIdx++) {
    const shrink = rIdx === 0 ? 0.02 : 0.72;
    const yVal = rIdx === 0 ? 0.08 : 0.012;
    for (let sIdx = 0; sIdx <= segAround; sIdx++) {
      const theta = (sIdx / segAround) * Math.PI * 2;
      const [px, pz] = sampleRoundedRect(theta, innerHX - shrink, innerHZ - shrink, Math.max(0.8, 1.55 - shrink));
      coveRingPositions.push(px, yVal, pz);
    }
  }
  for (let sIdx = 0; sIdx < segAround; sIdx++) {
    const a = sIdx, b = sIdx + 1, c = ringStride + sIdx, d = c + 1;
    coveIndices.push(a, b, c, b, d, c);
  }
  const coveGeo = new THREE.BufferGeometry();
  coveGeo.setAttribute('position', new THREE.Float32BufferAttribute(coveRingPositions, 3));
  coveGeo.setIndex(coveIndices);
  coveGeo.computeVertexNormals();
  const coveMat = new THREE.MeshBasicMaterial({
    color: 0x26211b,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const coveMesh = new THREE.Mesh(coveGeo, coveMat);
  coveMesh.renderOrder = 4;
  group.add(coveMesh);

  // 3D instanced corncob / wood-chip granules + darker husk/bolus flecks across the cage floor
  const granuleGeo = new THREE.IcosahedronGeometry(1.0, 0);
  const gPos = granuleGeo.attributes.position;
  for (let i = 0; i < gPos.count; i++) {
    gPos.setXYZ(
      i,
      gPos.getX(i) * 0.92,
      gPos.getY(i) * 0.46,
      gPos.getZ(i) * 1.18
    );
  }
  granuleGeo.computeVertexNormals();
  const granuleMat = new THREE.MeshStandardMaterial({
    color: 0xc8ba9e,
    roughness: 0.94,
    metalness: 0.0,
  });
  const granuleCount = 2400;
  const granules = new THREE.InstancedMesh(granuleGeo, granuleMat, granuleCount);
  granules.receiveShadow = true;
  granules.castShadow = false;
  const dummy = new THREE.Object3D();
  const gColor = new THREE.Color();
  for (let i = 0; i < granuleCount; i++) {
    let gx = (cageRng() - 0.5) * (innerHX * 2.0 - 0.9);
    let gz = (cageRng() - 0.5) * (innerHZ * 2.0 - 0.9);
    // Keep granules inside rounded corners
    const [boundX, boundZ] = sampleRoundedRect(Math.atan2(gz, gx), innerHX - 0.35, innerHZ - 0.35, 1.35);
    if (Math.abs(gx) > Math.abs(boundX)) gx = Math.sign(gx) * Math.abs(boundX) * (0.90 + cageRng() * 0.09);
    if (Math.abs(gz) > Math.abs(boundZ)) gz = Math.sign(gz) * Math.abs(boundZ) * (0.90 + cageRng() * 0.09);

    const isDarkFleck = cageRng() < 0.15;
    const sBase = isDarkFleck ? (0.065 + cageRng() * 0.095) : (0.085 + cageRng() * 0.145);
    dummy.position.set(gx, 0.020 + cageRng() * 0.032, gz);
    dummy.rotation.set(
      (cageRng() - 0.5) * 0.45,
      cageRng() * Math.PI * 2,
      (cageRng() - 0.5) * 0.45
    );
    dummy.scale.set(
      sBase * (0.78 + cageRng() * 0.52),
      sBase * (0.50 + cageRng() * 0.45),
      sBase * (0.85 + cageRng() * 0.65)
    );
    dummy.updateMatrix();
    granules.setMatrixAt(i, dummy.matrix);
    if (isDarkFleck) {
      gColor.setHSL(0.075 + (cageRng() - 0.5) * 0.02, 0.24, 0.18 + cageRng() * 0.14);
    } else {
      gColor.setHSL(0.098 + (cageRng() - 0.5) * 0.024, 0.18 + cageRng() * 0.14, 0.48 + cageRng() * 0.28);
    }
    granules.setColorAt(i, gColor);
  }
  group.add(granules);

  // Soft crumpled tissue/nestlet petals & curls in the upper-left zone + scattered wisps
  const nestGroup = new THREE.Group();
  const nestPaperMat = new THREE.MeshStandardMaterial({
    color: 0xe6e3da,
    roughness: 0.93,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const stripGeo = new THREE.PlaneGeometry(1, 1, 4, 7);
  stripGeo.rotateX(-Math.PI / 2);
  const sPos = stripGeo.attributes.position;
  for (let i = 0; i < sPos.count; i++) {
    const sx = sPos.getX(i), sz = sPos.getZ(i);
    // Taper ends &scallop edges so fragments read as torn crumpled tissue rather than rectangles
    const endFade = Math.max(0.18, 1.0 - Math.pow(Math.abs(sz * 1.95), 1.6) * 0.82);
    const waveX = Math.sin(sz * 5.2) * 0.22 + Math.cos(sx * 4.1) * 0.08;
    const curlY = Math.sin(sz * Math.PI * 2.2) * 0.24 + Math.cos(sx * Math.PI * 1.8) * 0.16;
    sPos.setXYZ(i, sx * endFade + waveX, Math.max(0.0, curlY + 0.15), sz);
  }
  stripGeo.computeVertexNormals();
  const stripCount = 260;
  const strips = new THREE.InstancedMesh(stripGeo, nestPaperMat, stripCount);
  strips.receiveShadow = true;
  for (let i = 0; i < stripCount; i++) {
    const zoneRoll = cageRng();
    let sx, sz, liftY;
    const isBroadPetal = cageRng() < 0.48;
    if (zoneRoll < 0.58) {
      sx = -innerHX + 1.2 + Math.pow(cageRng(), 0.82) * 6.4;
      sz = -innerHZ + 1.1 + Math.pow(cageRng(), 0.85) * 9.6;
      liftY = 0.026 + cageRng() * 0.055;
    } else if (zoneRoll < 0.84) {
      sx = -10.8 + cageRng() * 14.0;
      sz = -4.5 + cageRng() * 9.5;
      liftY = 0.020 + cageRng() * 0.032;
    } else {
      sx = 2.0 + cageRng() * 9.2;
      sz = 1.2 + cageRng() * 7.5;
      liftY = 0.020 + cageRng() * 0.030;
    }
    sx = THREE.MathUtils.clamp(sx, -innerHX + 0.9, innerHX - 1.8);
    sz = THREE.MathUtils.clamp(sz, -innerHZ + 0.9, innerHZ - 0.9);
    dummy.position.set(sx, liftY, sz);
    dummy.rotation.set(
      (cageRng() - 0.5) * 0.30,
      cageRng() * Math.PI * 2,
      (cageRng() - 0.5) * 0.30
    );
    const wScale = isBroadPetal ? (0.42 + cageRng() * 0.52) : (0.18 + cageRng() * 0.26);
    const lScale = isBroadPetal ? (0.55 + cageRng() * 0.85) : (0.65 + cageRng() * 1.25);
    dummy.scale.set(wScale, 0.09 + cageRng() * 0.14, lScale);
    dummy.updateMatrix();
    strips.setMatrixAt(i, dummy.matrix);
    gColor.setHSL(0.102, 0.06 + cageRng() * 0.08, 0.68 + cageRng() * 0.20);
    strips.setColorAt(i, gColor);
  }
  nestGroup.add(strips);
  group.add(nestGroup);

  // Right-wall brushed stainless-steel circular water-port dome & grommet (matching mouse083)
  const waterPort = new THREE.Group();
  waterPort.position.set(innerHX - 0.55, 1.85, 0.35);
  const steelMat = new THREE.MeshStandardMaterial({
    color: 0xc8ccd2,
    roughness: 0.30,
    metalness: 0.82,
  });
  const darkBoreMat = new THREE.MeshStandardMaterial({
    color: 0x18191d,
    roughness: 0.68,
    metalness: 0.25,
  });
  const shroudDome = new THREE.Mesh(new THREE.SphereGeometry(1.58, 24, 18), steelMat);
  shroudDome.scale.set(0.72, 0.88, 1.05);
  shroudDome.castShadow = true;
  waterPort.add(shroudDome);
  const grommetCollar = new THREE.Mesh(new THREE.TorusGeometry(1.48, 0.16, 12, 28), steelMat);
  grommetCollar.rotation.y = Math.PI / 2;
  grommetCollar.position.set(0.42, 0, 0);
  waterPort.add(grommetCollar);
  for (const [dy, dz] of [[0.35, -0.52], [0.35, 0.52], [-0.35, 0.0]]) {
    const portHole = new THREE.Mesh(new THREE.SphereGeometry(0.16, 10, 10), darkBoreMat);
    portHole.position.set(-0.92, dy, dz);
    waterPort.add(portHole);
  }
  group.add(waterPort);

  return group;
}


// ═══════════════════════════════════════════════════════════════
// 11. SCENE ASSEMBLY
// ═══════════════════════════════════════════════════════════════
const floorGeo = new THREE.PlaneGeometry(46, 34);
const floorMat = new THREE.MeshPhysicalMaterial({
  color: 0xf5f6f4,
  map: labFloorAlbedoMap,
  roughnessMap: labFloorRoughnessMap,
  normalMap: labFloorNormalMap,
  normalScale: new THREE.Vector2(0.075, 0.075),
  roughness: 0.82,
  metalness: 0.0,
  emissive: 0xc8cbc8,
  emissiveIntensity: 0.010,
  clearcoat: 0.05,
  clearcoatRoughness: 0.78,
  polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1
});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
scene.add(floor);
const floorMarksLayer = createFloorMarks();
const beddingLayer = createBeddingLayer();
scene.add(floorMarksLayer);
scene.add(beddingLayer);

const enclosureMat = new THREE.MeshPhysicalMaterial({
  color: 0xd7eef8,
  transparent: true,
  opacity: 0.18,
  roughness: 0.06,
  metalness: 0.0,
  transmission: 0.18,
  thickness: 0.08,
  ior: 1.45,
  clearcoat: 0.30,
  clearcoatRoughness: 0.18,
  side: THREE.BackSide,
  depthWrite: false
});
const enclosureBox = new THREE.Mesh(new THREE.BoxGeometry(ENCLOSURE_W, ENCLOSURE_H, ENCLOSURE_D), enclosureMat);
enclosureBox.position.set(0, ENCLOSURE_H * 0.5, 0); enclosureBox.receiveShadow = true;
enclosureBox.renderOrder = 10;
scene.add(enclosureBox);

const enclosureEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(ENCLOSURE_W, ENCLOSURE_H, ENCLOSURE_D)),
  new THREE.LineBasicMaterial({ color: 0x8b8f8a, transparent: true, opacity: 0.28 })
);
enclosureEdges.material.depthTest = true;
enclosureEdges.renderOrder = 25;
enclosureEdges.position.copy(enclosureBox.position); scene.add(enclosureEdges);
const arenaHardwareGroup = createArenaHardware();
scene.add(arenaHardwareGroup);

for (const tex of [homeCageBeddingMaps.albedo, homeCageBeddingMaps.normal, homeCageBeddingMaps.roughness]) {
  tex.repeat.set(1.45, 1.10);
}
const homeCageGroup = createHomeCageEnvironment();
homeCageGroup.visible = false;
scene.add(homeCageGroup);

const grid = new THREE.GridHelper(38, 19, 0xb8b1a4, 0xd1c7b5);
grid.position.y = 0.05;
const gridMats = Array.isArray(grid.material) ? grid.material : [grid.material];
for (const gm of gridMats) {
  gm.transparent = true;
  gm.opacity = 0.010;
  gm.depthWrite = false;
  gm.depthTest = true;
}
grid.renderOrder = 20;
scene.add(grid);

// ═══════════════════════════════════════════════════════════════
// 12. LIGHTING (3-point photographic behavioral-arena studio setup)
// ═══════════════════════════════════════════════════════════════
const ambientLight = new THREE.AmbientLight(0xf7f2ea, 0.26);
scene.add(ambientLight);

const hemiLight = new THREE.HemisphereLight(0xf6f8fc, 0xb2a696, 0.44);
hemiLight.position.set(0, 20, 0); scene.add(hemiLight);

// Key light (overhead diffused lab panel)
const defaultShadowMapSize = (!DATASET_CAPTURE && IS_SOFTWARE_WEBGL) ? 1024 : 4096;
const keyLight = new THREE.DirectionalLight(0xfff5ea, 0.98);
keyLight.position.set(-5, 17, 6); keyLight.castShadow = true;
keyLight.shadow.mapSize.width = defaultShadowMapSize;
keyLight.shadow.mapSize.height = defaultShadowMapSize;
keyLight.shadow.camera.near = 0.5; keyLight.shadow.camera.far = 30;
keyLight.shadow.camera.left = -20; keyLight.shadow.camera.right = 20;
keyLight.shadow.camera.top = 15; keyLight.shadow.camera.bottom = -15;
keyLight.shadow.bias = -0.0008; keyLight.shadow.normalBias = 0.018;
scene.add(keyLight);

// Fill light (soft cool bounce from acrylic enclosure)
const fillLight = new THREE.DirectionalLight(0xeaf2ff, 0.24);
fillLight.position.set(8, 7, 8); fillLight.castShadow = false;
scene.add(fillLight);

// Rim light (crisp dorsal guard-hair specular separation for dark & white coats)
const rimLight = new THREE.DirectionalLight(0xf9fcff, 0.34);
rimLight.position.set(6, 8.5, -9); rimLight.castShadow = false;
scene.add(rimLight);

const softbox = new THREE.RectAreaLight(0xfff2e4, 0.88, 16, 10);
softbox.position.set(-2, 9, 5);
softbox.lookAt(0, 1.0, 0);
scene.add(softbox);

// ═══════════════════════════════════════════════════════════════
// 13. GUI & LOGIC
// ═══════════════════════════════════════════════════════════════
const mice = [];
let nextTrackId = 1;
const hudElements = {
  count: document.getElementById('hud-count'),
  state: document.getElementById('hud-state'),
  hr: document.getElementById('hud-hr'),
  resp: document.getElementById('hud-resp'),
  coord: document.getElementById('hud-coord'),
  gait: document.getElementById('hud-gait'),
};

// Shared shader engine is initialized with the model factory above.
const postFX = annolidShaders.createPostProcessing(renderer, {
  vigIntensity: 0.16, vigSoftness: 0.54, grainStrength: 0.004,
  saturation: 0.98, brightness: 1.0
});

let occluderGroup = null;
if (DATASET_OCCLUDER === 'wire_bars') {
  occluderGroup = new THREE.Group();
  occluderGroup.name = 'wire-cage-lid-occluder';
  const barMat = new THREE.MeshStandardMaterial({
    color: 0xaeb4ba,
    roughness: 0.34,
    metalness: 0.82,
  });
  const barGeo = new THREE.CylinderGeometry(0.085, 0.085, ENCLOSURE_D - 1.2, 12);
  barGeo.rotateX(Math.PI / 2);
  for (let bx = -14.0; bx <= 14.001; bx += 1.85) {
    const barMesh = new THREE.Mesh(barGeo, barMat);
    barMesh.position.set(bx, 3.35, 0);
    barMesh.castShadow = true;
    barMesh.receiveShadow = false;
    occluderGroup.add(barMesh);
  }
  scene.add(occluderGroup);
}

if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'bottom-up') {
  // Ventral glass-floor imaging: hide opaque bedding/floor from below and add ventral illumination.
  floor.visible = false;
  floorMarksLayer.visible = false;
  beddingLayer.visible = false;
  grid.visible = false;
  scene.background = new THREE.Color(0x1b1f24);
  scene.fog = null;
  const ventralLight = new THREE.DirectionalLight(0xf6f8ff, 0.92);
  ventralLight.position.set(2.0, -14.0, 3.0);
  ventralLight.castShadow = false;
  scene.add(ventralLight);
}

function applyArenaStyle(styleName) {
  const isHomeCage = styleName === 'Home Cage (Bedding)';
  const isVentral = DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'bottom-up';
  homeCageGroup.visible = isHomeCage && !isVentral;
  enclosureBox.visible = !isHomeCage;
  enclosureEdges.visible = !isHomeCage;
  arenaHardwareGroup.visible = !isHomeCage;
  floorMarksLayer.visible = !isHomeCage && !isVentral;
  beddingLayer.visible = !isVentral;
  grid.visible = !isHomeCage && !isVentral;
  if (isHomeCage) {
    // Fit the textured bedding plane cleanly inside the rounded polycarbonate shoebox base
    floor.scale.set(0.732, 0.688, 1.0);
    floorMat.color.setHex(0xd4c7af);
    floorMat.map = homeCageBeddingMaps.albedo;
    floorMat.roughnessMap = homeCageBeddingMaps.roughness;
    floorMat.normalMap = homeCageBeddingMaps.normal;
    floorMat.normalScale.set(0.52, 0.52);
    floorMat.roughness = 0.94;
    floorMat.emissiveIntensity = 0.0;
    floorMat.clearcoat = 0.0;
  } else {
    floor.scale.set(1.0, 1.0, 1.0);
    floorMat.color.setHex(0xf5f6f4);
    floorMat.map = labFloorAlbedoMap;
    floorMat.roughnessMap = labFloorRoughnessMap;
    floorMat.normalMap = labFloorNormalMap;
    floorMat.normalScale.set(0.075, 0.075);
    floorMat.roughness = 0.82;
    floorMat.emissiveIntensity = 0.010;
    floorMat.clearcoat = 0.05;
  }
  floorMat.needsUpdate = true;
}

if (DATASET_LIGHTING === 'ir_night') {
  scene.background = new THREE.Color(0x20191b);
  scene.fog = new THREE.FogExp2(0x20191b, 0.006);
  scene.environmentIntensity = 0.16;
  ambientLight.color.setHex(0xd8b4ba);
  ambientLight.intensity = 0.22;
  hemiLight.color.setHex(0xd4b6bc);
  hemiLight.groundColor.setHex(0x3a2c30);
  hemiLight.intensity = 0.28;
  keyLight.color.setHex(0xe8c6cc);
  keyLight.intensity = 0.72;
  fillLight.color.setHex(0xc8aeb4);
  fillLight.intensity = 0.18;
  rimLight.color.setHex(0xf0d4da);
  rimLight.intensity = 0.38;
  softbox.color.setHex(0xdcb8be);
  softbox.intensity = 0.42;
} else if (DATASET_LIGHTING === 'dim_warm') {
  scene.environmentIntensity = 0.34;
  ambientLight.intensity = 0.14;
  keyLight.color.setHex(0xffdfbe);
  keyLight.intensity = 0.48;
  softbox.intensity = 0.38;
} else if (DATASET_LIGHTING === 'calms_gray') {
  // Overhead monochrome / near-IR behavioral cage-rack camera (like mouse083_task1_annotator1.mp4)
  scene.background = new THREE.Color(0x161719);
  scene.fog = null;
  scene.environmentIntensity = 0.44;
  ambientLight.color.setHex(0xf4f4f4);
  ambientLight.intensity = 0.34;
  hemiLight.color.setHex(0xf8f8fa);
  hemiLight.groundColor.setHex(0x4c4a46);
  hemiLight.intensity = 0.54;
  keyLight.color.setHex(0xffffff);
  keyLight.intensity = 0.94;
  keyLight.position.set(-2.2, 22.0, 2.5);
  fillLight.color.setHex(0xeeeeff);
  fillLight.intensity = 0.28;
  rimLight.color.setHex(0xffffff);
  rimLight.intensity = 0.36;
  softbox.color.setHex(0xffffff);
  softbox.intensity = 0.68;
}

const isHomeCageDefault = DATASET_ARENA === 'home_cage';
const guiParams = {
  furColor: '#ffffff', skinColor: '#f0c8b8',
  chonkiness: isHomeCageDefault ? 1.20 : 0.96,
  earSize: isHomeCageDefault ? 0.94 : 0.82,
  furDensity: 1.0,
  showTrackingKeypoints: false, wireframeMode: false, isPaused: false,
  implantHeadstage: false,
  arenaStyle: isHomeCageDefault ? 'Home Cage (Bedding)' : 'Open Field',
  playbackSpeed: 1.0, cameraView: 'Both mice',
  addSubject: () => spawnMouse(), clearSubjects: () => clearMice(),
  presets: 'BALB/c (White)',
  // Body part controls (standard shoebox home cage is ~35x21 cm vs ~50x36 cm open field,
  // so adult mice span ~23% of home-cage width just like mouse083_task1_annotator1.mp4)
  tailLength: isHomeCageDefault ? 1.14 : 0.94,
  tailThickness: isHomeCageDefault ? 1.32 : 1.15,
  headSize: isHomeCageDefault ? PROP.headSize * 1.18 : PROP.headSize,
  snoutLength: isHomeCageDefault ? 1.08 : 1.0,
  bodyLength: isHomeCageDefault ? 1.22 : 1.0,
  legLength: isHomeCageDefault ? 1.16 : 1.06,
  legThickness: isHomeCageDefault ? 1.18 : 1.10,
  // Physics
  maxSpeed: 9.2, dragCoeff: 2.35, wanderForce: 9.5, wallAttraction: 2.1,
  enableRearing: true, rearingFrequency: 1.0,
  triggerRearing: () => {
    const subject = mice.find(mouse => (
      mouse.rearing.mode === 'none' && mouse.grooming.mode === 'none'
    ));
    if (subject) subject.startRearing('unsupported');
  },
  enableGrooming: true, groomingFrequency: 1.0, groomingMode: 'Face wash',
  // Motor control: 'auto' selects walk/trot/gallop from speed with hysteresis.
  gaitMode: 'auto',
  triggerGrooming: () => {
    const subject = mice.find(mouse => (
      mouse.rearing.mode === 'none' && mouse.grooming.mode === 'none'
    ));
    if (!subject) return;
    subject.startGrooming(
      guiParams.groomingMode === 'Flank groom' ? 'flank-groom' : 'face-wash'
    );
  },
  // Quality
  qualityMode: 'Photo', furDetailLevel: 'Med',
  enableVignette: false, enableGrain: false, enableAO: true,
  // Lighting
  keyIntensity: 0.96, fillIntensity: 0.20, rimIntensity: 0.30, softboxIntensity: 0.92,
  keyColorTemp: 5600, ambientIntensity: 0.24,
  // Proportions (exposed)
  propBodyLength: PROP.bodyLength, propBodyGirthX: PROP.bodyGirthX,
  propWaistTaper: PROP.waistTaper, propHaunchWidth: PROP.haunchWidth,
  propShoulderWidth: PROP.shoulderWidth,
  propNeckLength: PROP.neckLength, propNeckGirth: PROP.neckGirth,
  propSnoutLength: PROP.snoutLength, propSnoutWidth: PROP.snoutWidth,
  propCheekFullness: PROP.cheekFullness,
  propEarYaw: PROP.earYaw, propEarThickness: PROP.earThickness,
  propEyeSize: PROP.eyeSize, propEyeSpacing: PROP.eyeSpacing,
  propForeLimbLength: PROP.foreLimbLength, propHindLimbLength: PROP.hindLimbLength,
  propForePawSize: PROP.forePawSize, propHindPawSize: PROP.hindPawSize,
  propTailSag: PROP.tailSag,
  // Fur
  furGuardRatio: 0.5, furLength: 1.0, furClumpiness: 0.3,
  // Shader controls
  enableSSS: true, sssThickness: 0.45, sssWrap: 0.5, sssBacklight: 0.6,
  enableContactAO: true, aoRadius: 1.5, aoIntensity: 0.35,
  enableMicroDetail: true, microScale: 40.0, microStrength: 0.06,
  enablePostFX: DATASET_LIGHTING === 'calms_gray',
  postVignette: DATASET_LIGHTING === 'calms_gray' ? 0.18 : 0.16,
  postGrain: DATASET_LIGHTING === 'calms_gray' ? 0.010 : 0.004,
  postSaturation: DATASET_LIGHTING === 'calms_gray' ? 0.03 : 0.98,
};
postFX.enabled = guiParams.enablePostFX;
postFX.uniforms.uVigIntensity.value = guiParams.postVignette;
postFX.uniforms.uGrainStrength.value = guiParams.postGrain;
postFX.uniforms.uSaturation.value = guiParams.postSaturation;
if (DATASET_ARENA === 'home_cage') {
  applyArenaStyle('Home Cage (Bedding)');
}

function getStrainProfile(presetName) {
  if (presetName === 'C57BL/6 (Black)') {
    return {
      eyeColor: '#060505', eyeRimColor: '#241a1b',
      earOuterTransmission: 0.16, earInnerTransmission: 0.26,
      earInnerSatOffset: 0.030, earInnerLightOffset: 0.02, earOuterRoughness: 0.62,
      earOuterColor: '#625456', earInnerColor: '#967c7a',
      earSkinBlend: 0.22, earInnerSkinBlend: 0.16, earScaleMul: 0.96,
      earSheenColor: 0xb29a90, earAttenuationColor: 0x8a5148, earAttenuationDistance: 0.12,
      earYaw: 0.72, earHeightOffset: -0.02, padDarken: 0.92, clawColor: 0xd0c6be,
      pawColor: '#8c6d6b', pawPadColor: '#6e5250',
      noseColor: '#8e6e6e', tailColor: '#4c3e3e', tailTransmission: 0.10,
      whiskerColor: 0x3a3533, whiskerOpacity: 0.80, pawSpreadMul: 1.0, toeLenMul: 0.98,
      gaitAmpMul: 0.92, headIdleYawMul: 0.82, headIdlePitchMul: 0.88,
      sniffAmpMul: 0.75, breathIdleAmp: 0.010, breathMoveAmp: 0.012,
      breathIdleFreq: 10.0, breathMoveFreq: 18.0, tailWaveAmpMul: 0.85, tailWaveFreqMul: 0.90
    };
  }
  if (presetName === 'CBA (Brown)') {
    return {
      eyeColor: '#100a08', eyeRimColor: '#351e18',
      earOuterTransmission: 0.20, earInnerTransmission: 0.30,
      earInnerSatOffset: 0.035, earInnerLightOffset: 0.015, earOuterRoughness: 0.58,
      earOuterColor: '#765b4e', earInnerColor: '#b78378',
      earSkinBlend: 0.24, earInnerSkinBlend: 0.16, earScaleMul: 0.96,
      earSheenColor: 0xc6a08e, earAttenuationColor: 0xb77668, earAttenuationDistance: 0.14,
      earYaw: 0.58, earHeightOffset: -0.01, padDarken: 0.93, clawColor: 0xd6c8bd,
      pawColor: '#c4928a', pawPadColor: '#a67670',
      noseColor: '#be8880', tailColor: '#7a5d50', tailTransmission: 0.16,
      whiskerColor: 0xc2b7ae, pawSpreadMul: 1.0, toeLenMul: 1.0,
      gaitAmpMul: 1.0, headIdleYawMul: 1.0, headIdlePitchMul: 1.0,
      sniffAmpMul: 1.0, breathIdleAmp: 0.013, breathMoveAmp: 0.016,
      breathIdleFreq: 12.0, breathMoveFreq: 20.0, tailWaveAmpMul: 1.0, tailWaveFreqMul: 1.0
    };
  }
  if (presetName === 'Nude/Hairless') {
    return {
      eyeColor: '#7a1922', eyeRimColor: '#be7c80',
      earOuterTransmission: 0.36, earInnerTransmission: 0.46,
      earInnerSatOffset: 0.10, earInnerLightOffset: 0.08, earOuterRoughness: 0.44,
      earOuterColor: '#d9b2ad', earInnerColor: '#e8b9b5',
      earSkinBlend: 0.28, earInnerSkinBlend: 0.18, earScaleMul: 1.02,
      earSheenColor: 0xffd6c8, earAttenuationColor: 0xea9d8f, earAttenuationDistance: 0.18,
      earYaw: 0.48, earHeightOffset: 0.00, padDarken: 0.94, clawColor: 0xe2d6cd,
      pawColor: '#e8bab4', pawPadColor: '#d29692',
      noseColor: '#e4ada8', tailColor: '#e2aca6', tailTransmission: 0.28,
      whiskerColor: 0xe6dcd5, pawSpreadMul: 1.04, toeLenMul: 1.05,
      gaitAmpMul: 1.08, headIdleYawMul: 1.15, headIdlePitchMul: 1.12,
      sniffAmpMul: 1.25, breathIdleAmp: 0.018, breathMoveAmp: 0.024,
      breathIdleFreq: 13.0, breathMoveFreq: 24.0, tailWaveAmpMul: 1.18, tailWaveFreqMul: 1.12
    };
  }
  return { // BALB/c default — luminous ruby-garnet albino iris + pale rose pinnae
    eyeColor: '#a82234', eyeRimColor: '#d9a4a4',
    earOuterTransmission: 0.36, earInnerTransmission: 0.48,
    earInnerSatOffset: 0.040, earInnerLightOffset: 0.030, earOuterRoughness: 0.52,
    earOuterColor: '#dcd0cd', earInnerColor: '#e8c6c4',
    earSkinBlend: 0.24, earInnerSkinBlend: 0.20, earScaleMul: 0.96,
    earSheenColor: 0xf4dada, earAttenuationColor: 0xde9ca0, earAttenuationDistance: 0.17,
    earYaw: 0.52, earHeightOffset: -0.01, padDarken: 0.94, clawColor: 0xe6dcd5,
    pawColor: '#e8c4c0', pawPadColor: '#d6a6a4',
    noseColor: '#efc2c0', tailColor: '#e4b8b5', tailTransmission: 0.28,
    whiskerColor: 0xddd6ce, pawSpreadMul: 1.00, toeLenMul: 1.02,
    gaitAmpMul: 1.0, headIdleYawMul: 1.0, headIdlePitchMul: 1.0,
    sniffAmpMul: 1.0, breathIdleAmp: 0.013, breathMoveAmp: 0.016,
    breathIdleFreq: 12.0, breathMoveFreq: 20.0, tailWaveAmpMul: 1.0, tailWaveFreqMul: 1.0
  };
}

function applyStrainPreset(name) {
  guiParams.presets = name;
  const earScale = isHomeCageDefault ? 1.18 : 1.0;
  if (name === 'BALB/c (White)') { guiParams.furColor = '#efebe2'; guiParams.skinColor = '#ebb4b0'; guiParams.furDensity = 0.90; guiParams.earSize = 0.70 * earScale; }
  if (name === 'C57BL/6 (Black)') { guiParams.furColor = '#171618'; guiParams.skinColor = '#483b3d'; guiParams.furDensity = 0.95; guiParams.earSize = 0.68 * earScale; }
  if (name === 'CBA (Brown)') { guiParams.furColor = '#6d4d38'; guiParams.skinColor = '#cfa090'; guiParams.furDensity = 0.88; guiParams.earSize = 0.68 * earScale; }
  if (name === 'Nude/Hairless') { guiParams.furColor = '#ffd0c8'; guiParams.skinColor = '#ffc0b8'; guiParams.furDensity = 0.0; guiParams.earSize = 0.75 * earScale; }
}

const gui = new GUI({ title: 'OPEN FIELD CONTROLS' });
const respawnNoticeState = new Map();

function notifyRespawnRequired(key, value, message) {
  const token = `${key}:${JSON.stringify(value)}`;
  if (respawnNoticeState.get(key) === token) return;
  respawnNoticeState.set(key, token);
  alert(message);
}

gui.add(guiParams, 'presets', ['BALB/c (White)', 'C57BL/6 (Black)', 'CBA (Brown)', 'Nude/Hairless']).name('Strains').onChange(v => {
  applyStrainPreset(v);
  gui.controllersRecursive().forEach(c => c.updateDisplay());
});

function applySubjectWireframe(mouse, enabled) {
  const flag = Boolean(enabled);
  const trackedNodes = mouse.poseNodes ? new Set(Object.values(mouse.poseNodes)) : new Set();
  for (const root of [mouse.rig, mouse.tailMesh, mouse.tailRoot]) {
    if (!root) continue;
    root.traverse(child => {
      if (!child.isMesh || child.isInstancedMesh || trackedNodes.has(child)) return;
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const mat of mats) {
        if (mat && mat.colorWrite !== false) mat.wireframe = flag;
      }
    });
  }
}

function syncSubjectShaderUniforms() {
  for (const mouse of mice) {
    const materials = new Set(mouse.mesh.userData?.ownedMaterials || []);
    mouse.mesh.traverse(node => {
      if (!node.material) return;
      const list = Array.isArray(node.material) ? node.material : [node.material];
      for (const mat of list) {
        if (mat) materials.add(mat);
      }
    });
    for (const mat of materials) {
      const scale = mat.userData?.annolidShaderScale || {};
      annolidShaders.updateUniforms(mat, {
        uSSSThickness: guiParams.sssThickness * (scale.sssThickness ?? 1.0),
        uSSSWrap: guiParams.sssWrap,
        uSSSBacklight: guiParams.sssBacklight * (scale.sssBacklight ?? 1.0),
        uAORadius: Math.min(scale.aoRadiusCap ?? 5.0, guiParams.aoRadius),
        uAOIntensity: Math.min(
          scale.aoIntensityCap ?? 1.0,
          guiParams.aoIntensity * (scale.aoIntensityScale ?? 1.0)
        ),
        uMicroScale: Math.max(scale.microScaleFloor ?? 5.0, guiParams.microScale),
        uMicroStrength: Math.min(
          scale.microStrengthCap ?? 0.3,
          guiParams.microStrength * (scale.microStrengthScale ?? 1.0)
        ),
      });
    }
  }
}

// Neuroscience
const neuroscienceFolder = gui.addFolder('Neuroscience & Tracking');
neuroscienceFolder.add(guiParams, 'showTrackingKeypoints').name('Show Pose Skeleton').onChange(v => {
  mice.forEach(m => m.syncTrackingVisibility(v));
});
neuroscienceFolder.add(guiParams, 'wireframeMode').name('Wireframe Bodies').onChange(v => {
  mice.forEach(m => applySubjectWireframe(m, v));
});
neuroscienceFolder.add(guiParams, 'implantHeadstage').name('Neural Headstage').onChange(v => {
  if (mice[0]) mice[0].setHeadstageEnabled(v);
});

const syncPausedAnatomy = () => {
  if (guiParams.isPaused) {
    mice.forEach(m => m.syncLiveProportions(mice));
  }
};
const syncSharedAnatomy = () => {
  rebuildSharedAnatomyGeometries();
  mice.forEach(m => m.syncLiveProportions(mice));
};
const notifyAnatomyRespawn = (key, value) => {
  notifyRespawnRequired(
    key,
    Number(value).toFixed(3),
    'Detailed anatomy and coat alignment apply fully to newly spawned subjects.'
  );
};

// Biological Traits
const traitFolder = gui.addFolder('Biological Traits');
traitFolder.addColor(guiParams, 'furColor').name('Fur Pigment').onFinishChange(v => {
  notifyRespawnRequired('furColor', v, 'Fur pigment applies to newly spawned subjects.');
});
traitFolder.addColor(guiParams, 'skinColor').name('Skin Pigment').onFinishChange(v => {
  notifyRespawnRequired('skinColor', v, 'Skin pigment applies to newly spawned subjects.');
});
traitFolder.add(guiParams, 'chonkiness', 0.5, 2.0).name('Fatness / Mass').onChange(syncPausedAnatomy);
traitFolder.add(guiParams, 'earSize', 0.5, 2.0).name('Ear Magnitude').onChange(v => {
  mice.forEach(m => m.setLiveEarSize(v));
});
traitFolder.add(guiParams, 'furDensity', 0.0, 1.0).name('Fur Density').onFinishChange(v => {
  notifyRespawnRequired(
    'furDensity',
    Number(v).toFixed(3),
    "Fur density changes require respawning subjects to take effect."
  );
});

// Simulation Controls
const controlFolder = gui.addFolder('Simulation Controls');
controlFolder.add(guiParams, 'arenaStyle', ['Open Field', 'Home Cage (Bedding)'])
  .name('Arena').onChange(applyArenaStyle);
controlFolder.add(guiParams, 'isPaused').name('Pause Simulation');
controlFolder.add(guiParams, 'playbackSpeed', 0.25, 1.5, 0.25).name('Playback Speed');
controlFolder.add(guiParams, 'cameraView', ['Both mice', 'Follow black', 'Follow white', 'Top down', 'Home cage top'])
  .name('Camera').onChange(frameCameraView);
controlFolder.add(guiParams, 'addSubject').name('(+) Spawn Subject');
controlFolder.add(guiParams, 'clearSubjects').name('(!) Clear All');

// Body Part Controls
const bodyFolder = gui.addFolder('Body Part Controls');
bodyFolder.add(guiParams, 'tailLength', 0.2, 2.5, 0.05).name('Tail Length').onChange(syncPausedAnatomy);
bodyFolder.add(guiParams, 'tailThickness', 0.3, 2.0, 0.05).name('Tail Thickness').onChange(syncPausedAnatomy);
bodyFolder.add(guiParams, 'headSize', 0.5, 2.0, 0.05).name('Head Size').onChange(syncPausedAnatomy);
bodyFolder.add(guiParams, 'snoutLength', 0.5, 2.0, 0.05).name('Snout Length').onChange(syncPausedAnatomy);
bodyFolder.add(guiParams, 'bodyLength', 0.5, 2.0, 0.05).name('Body Length').onChange(syncPausedAnatomy);
bodyFolder.add(guiParams, 'legLength', 0.5, 2.0, 0.05).name('Leg Length').onFinishChange(v => {
  notifyRespawnRequired(
    'legLength',
    Number(v).toFixed(3),
    'Limb proportion changes apply to newly spawned subjects.'
  );
});
bodyFolder.add(guiParams, 'legThickness', 0.5, 2.0, 0.05).name('Leg Thickness').onFinishChange(v => {
  notifyRespawnRequired(
    'legThickness',
    Number(v).toFixed(3),
    'Limb proportion changes apply to newly spawned subjects.'
  );
});

// Proportions (detailed)
const propFolder = gui.addFolder('Proportions');
propFolder.add(guiParams, 'propBodyLength', 1.5, 4.0, 0.1).name('Body Length')
  .onChange(v => { PROP.bodyLength = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propBodyLength', v));
propFolder.add(guiParams, 'propBodyGirthX', 1.0, 2.5, 0.05).name('Body Girth X')
  .onChange(v => { PROP.bodyGirthX = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propBodyGirthX', v));
propFolder.add(guiParams, 'propWaistTaper', 0.3, 1.0, 0.05).name('Waist Taper')
  .onChange(v => { PROP.waistTaper = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propWaistTaper', v));
propFolder.add(guiParams, 'propHaunchWidth', 0.8, 2.0, 0.05).name('Haunch Width')
  .onChange(v => { PROP.haunchWidth = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propHaunchWidth', v));
propFolder.add(guiParams, 'propShoulderWidth', 0.5, 1.5, 0.05).name('Shoulder Width')
  .onChange(v => { PROP.shoulderWidth = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propShoulderWidth', v));
propFolder.add(guiParams, 'propNeckLength', 0.3, 1.5, 0.05).name('Neck Length')
  .onChange(v => { PROP.neckLength = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propNeckLength', v));
propFolder.add(guiParams, 'propNeckGirth', 0.3, 1.0, 0.05).name('Neck Girth')
  .onChange(v => { PROP.neckGirth = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propNeckGirth', v));
propFolder.add(guiParams, 'propSnoutLength', 0.8, 2.0, 0.05).name('Snout Length')
  .onChange(v => { PROP.snoutLength = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propSnoutLength', v));
propFolder.add(guiParams, 'propSnoutWidth', 0.8, 1.8, 0.05).name('Snout Width')
  .onChange(v => { PROP.snoutWidth = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propSnoutWidth', v));
propFolder.add(guiParams, 'propCheekFullness', 0.1, 0.8, 0.05).name('Cheek Fullness')
  .onChange(v => { PROP.cheekFullness = v; syncSharedAnatomy(); })
  .onFinishChange(v => notifyAnatomyRespawn('propCheekFullness', v));
propFolder.add(guiParams, 'propEarYaw', 0.2, 1.2, 0.05).name('Ear Yaw')
  .onChange(v => { PROP.earYaw = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propEarYaw', v));
propFolder.add(guiParams, 'propEyeSize', 0.08, 0.25, 0.01).name('Eye Size')
  .onChange(v => { PROP.eyeSize = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propEyeSize', v));
propFolder.add(guiParams, 'propEyeSpacing', 0.3, 0.7, 0.02).name('Eye Spacing')
  .onChange(v => { PROP.eyeSpacing = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propEyeSpacing', v));
propFolder.add(guiParams, 'propForeLimbLength', 0.5, 1.5, 0.05).name('Forelimb Length')
  .onChange(v => { PROP.foreLimbLength = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propForeLimbLength', v));
propFolder.add(guiParams, 'propHindLimbLength', 0.5, 1.5, 0.05).name('Hindlimb Length')
  .onChange(v => { PROP.hindLimbLength = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propHindLimbLength', v));
propFolder.add(guiParams, 'propForePawSize', 0.5, 2.0, 0.05).name('Forepaw Size')
  .onChange(v => { PROP.forePawSize = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propForePawSize', v));
propFolder.add(guiParams, 'propHindPawSize', 0.5, 2.0, 0.05).name('Hindpaw Size')
  .onChange(v => { PROP.hindPawSize = v; })
  .onFinishChange(v => notifyAnatomyRespawn('propHindPawSize', v));
propFolder.add(guiParams, 'propTailSag', 0.0, 0.1, 0.005).name('Tail Sag')
  .onChange(v => { PROP.tailSag = v; syncPausedAnatomy(); });
propFolder.close();

// Fur controls
const furFolder = gui.addFolder('Fur');
furFolder.add(guiParams, 'furDetailLevel', ['High', 'Med', 'Low', 'Off']).name('Fur Detail').onFinishChange(v => {
  notifyRespawnRequired(
    'furDetailLevel',
    String(v || ''),
    "Fur detail changes require respawning subjects to take effect."
  );
});
furFolder.add(guiParams, 'furGuardRatio', 0.0, 1.0, 0.05).name('Guard Hair Ratio').onFinishChange(v => {
  notifyRespawnRequired(
    'furGuardRatio',
    Number(v).toFixed(3),
    'Fur coat parameter changes apply to newly spawned subjects.'
  );
});
furFolder.add(guiParams, 'furLength', 0.3, 2.0, 0.05).name('Fur Length').onFinishChange(v => {
  notifyRespawnRequired(
    'furLength',
    Number(v).toFixed(3),
    'Fur coat parameter changes apply to newly spawned subjects.'
  );
});
furFolder.add(guiParams, 'furClumpiness', 0.0, 1.0, 0.05).name('Clumpiness').onFinishChange(v => {
  notifyRespawnRequired(
    'furClumpiness',
    Number(v).toFixed(3),
    'Fur coat parameter changes apply to newly spawned subjects.'
  );
});
furFolder.close();

// Lighting
const lightFolder = gui.addFolder('Lighting');
lightFolder.add(guiParams, 'keyIntensity', 0.0, 2.0, 0.05).name('Key Intensity').onChange(v => { keyLight.intensity = v; });
lightFolder.add(guiParams, 'fillIntensity', 0.0, 2.0, 0.05).name('Fill Intensity').onChange(v => { fillLight.intensity = v; });
lightFolder.add(guiParams, 'rimIntensity', 0.0, 2.0, 0.05).name('Rim Intensity').onChange(v => { rimLight.intensity = v; });
lightFolder.add(guiParams, 'softboxIntensity', 0.0, 5.0, 0.05).name('Softbox').onChange(v => { softbox.intensity = v; });
lightFolder.add(guiParams, 'ambientIntensity', 0.0, 2.0, 0.05).name('Ambient Intensity').onChange(v => { ambientLight.intensity = v; });
lightFolder.close();

// Quality
const qualityFolder = gui.addFolder('Quality');
qualityFolder.add(guiParams, 'qualityMode', ['Photo', 'Realtime', 'Low']).name('Quality Mode').onChange(v => {
  // Quality changes resolution, not exposure or the captured strain pigment.
  renderer.toneMappingExposure = 0.92;
  if (v === 'Photo') {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    keyLight.shadow.mapSize.width = 4096; keyLight.shadow.mapSize.height = 4096;
    keyLight.shadow.map?.dispose(); keyLight.shadow.map = null;
  } else if (v === 'Low') {
    renderer.setPixelRatio(1.0);
    keyLight.shadow.mapSize.width = 1024; keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.map?.dispose(); keyLight.shadow.map = null;
  } else {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    keyLight.shadow.mapSize.width = 2048; keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.map?.dispose(); keyLight.shadow.map = null;
  }
  postFX.resize();
  gui.controllersRecursive().forEach(c => c.updateDisplay());
});
qualityFolder.add(guiParams, 'enableAO').name('Contact Shadows').onChange(v => {
  mice.forEach(m => { if (m.contactShadow) m.contactShadow.visible = v; });
});
qualityFolder.close();

// Shaders
const shaderFolder = gui.addFolder('Shaders');
shaderFolder.add(guiParams, 'enableSSS').name('Subsurface Scatter').onFinishChange(v => {
  notifyRespawnRequired(
    'enableSSS',
    Boolean(v),
    'Shader changes require respawning subjects.'
  );
});
shaderFolder.add(guiParams, 'sssThickness', 0.0, 1.0, 0.05).name('SSS Thickness').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'sssWrap', 0.0, 1.0, 0.05).name('SSS Wrap').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'sssBacklight', 0.0, 1.5, 0.05).name('SSS Backlight').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'enableContactAO').name('Contact AO').onFinishChange(v => {
  notifyRespawnRequired(
    'enableContactAO',
    Boolean(v),
    'Shader changes require respawning subjects.'
  );
});
shaderFolder.add(guiParams, 'aoRadius', 0.5, 5.0, 0.1).name('AO Radius').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'aoIntensity', 0.0, 1.0, 0.05).name('AO Intensity').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'enableMicroDetail').name('Micro Detail').onFinishChange(v => {
  notifyRespawnRequired(
    'enableMicroDetail',
    Boolean(v),
    'Shader changes require respawning subjects.'
  );
});
shaderFolder.add(guiParams, 'microScale', 5.0, 200.0, 5.0).name('Detail Scale').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'microStrength', 0.0, 0.3, 0.01).name('Detail Strength').onChange(syncSubjectShaderUniforms);
shaderFolder.add(guiParams, 'enablePostFX').name('Post-Processing').onChange(v => {
  postFX.enabled = v;
});
shaderFolder.add(guiParams, 'postVignette', 0.0, 1.0, 0.05).name('Vignette').onChange(v => {
  postFX.uniforms.uVigIntensity.value = v;
});
shaderFolder.add(guiParams, 'postGrain', 0.0, 0.1, 0.005).name('Film Grain').onChange(v => {
  postFX.uniforms.uGrainStrength.value = v;
});
shaderFolder.add(guiParams, 'postSaturation', 0.5, 1.5, 0.05).name('Saturation').onChange(v => {
  postFX.uniforms.uSaturation.value = v;
});
shaderFolder.close();

// Physics
const physicsFolder = gui.addFolder('Physics Controls');
physicsFolder.add(guiParams, 'maxSpeed', 1.0, 20.0, 0.5).name('Max Speed');
physicsFolder.add(guiParams, 'dragCoeff', 0.5, 10.0, 0.5).name('Drag / Friction');
physicsFolder.add(guiParams, 'wanderForce', 0.5, 10.0, 0.5).name('Wander Force');
physicsFolder.add(guiParams, 'wallAttraction', 0.0, 5.0, 0.25).name('Wall Attraction');

const behaviorFolder = gui.addFolder('Behavior Controls');
behaviorFolder.add(guiParams, 'enableRearing').name('Rearing');
behaviorFolder.add(guiParams, 'rearingFrequency', 0.25, 2.0, 0.05).name('Rear Frequency');
behaviorFolder.add(guiParams, 'triggerRearing').name('Rear Now');
behaviorFolder.add(guiParams, 'enableGrooming').name('Grooming');
behaviorFolder.add(guiParams, 'groomingFrequency', 0.25, 2.0, 0.05).name('Groom Frequency');
behaviorFolder.add(guiParams, 'groomingMode', ['Face wash', 'Flank groom']).name('Groom Type');
behaviorFolder.add(guiParams, 'triggerGrooming').name('Groom Now');
behaviorFolder.add(guiParams, 'gaitMode', ['auto', ...getMouseActionSpec().gaits]).name('Gait');
gui.close();

// ═══════════════════════════════════════════════════════════════
// 14. SPAWN / CLEAR
// ═══════════════════════════════════════════════════════════════
function spawnMouse(options = {}) {
  const m = new ExpertMouse(guiParams);
  m.trackId = nextTrackId++;
  m.strainName = guiParams.presets;
  let inX = 0, inZ = 0;
  if (options.position && Array.isArray(options.position)) {
    m.mesh.position.set(
      Number(options.position[0] || 0),
      0,
      Number(options.position[1] || 0)
    );
    const heading = Array.isArray(options.heading) ? options.heading : [0, 1];
    inX = Number(heading[0] || 0);
    inZ = Number(heading[1] || 0);
  } else {
    const wall = Math.floor(random() * 4);
    const wo = 4.6;
    switch (wall) {
      case 0: m.mesh.position.x = ENCLOSURE_HALF_X - wo; m.mesh.position.z = (random() - 0.5) * ENCLOSURE_HALF_Z * 1.5; inX = -1; break;
      case 1: m.mesh.position.x = -ENCLOSURE_HALF_X + wo; m.mesh.position.z = (random() - 0.5) * ENCLOSURE_HALF_Z * 1.5; inX = 1; break;
      case 2: m.mesh.position.x = (random() - 0.5) * ENCLOSURE_HALF_X * 1.5; m.mesh.position.z = ENCLOSURE_HALF_Z - wo; inZ = -1; break;
      case 3: m.mesh.position.x = (random() - 0.5) * ENCLOSURE_HALF_X * 1.5; m.mesh.position.z = -ENCLOSURE_HALF_Z + wo; inZ = 1; break;
    }
  }
  const dir = new THREE.Vector2(inX, inZ);
  if (!options.position) {
    const tangentialKick = (random() - 0.5) * 0.9;
    if (Math.abs(inX) > 0) dir.y += tangentialKick;
    else dir.x += tangentialKick;
  }
  if (dir.lengthSq() < 1e-6) dir.set((random() - 0.5), (random() - 0.5));
  dir.normalize();
  const launchSpeed = options.speed != null
    ? Number(options.speed)
    : guiParams.maxSpeed * (0.50 + random() * 0.18);
  m.velocity.set(dir.x * launchSpeed, dir.y * launchSpeed);
  m.heading.copy(dir);
  m.yawAngle = Math.atan2(dir.x, dir.y);
  m.prevHeadingAngle = m.yawAngle;
  m.mesh.rotation.y = m.yawAngle;
  m.prevVelocity.copy(m.velocity);
  if (m.tailDyn) {
    m.tailDyn.sway = (random() - 0.5) * 0.25;
    m.tailDyn.lift = -0.04 + random() * 0.03;
  }
  scene.add(m.mesh);
  if (guiParams.wireframeMode) applySubjectWireframe(m, true);
  m.syncTrackingVisibility(guiParams.showTrackingKeypoints);
  mice.push(m);
  hudElements.count.innerText = mice.length;
}

function clearMice() {
  mice.forEach(m => {
    scene.remove(m.mesh);
    disposeSubject(m.mesh);
  });
  mice.length = 0;
  hudElements.count.innerText = 0;
  hudElements.state.innerText = "NO SUBJECTS";
  hudElements.hr.innerText = "-- BPM";
  hudElements.resp.innerText = "-- BPM";
  hudElements.coord.innerText = "0.00, 0.00";
  if (hudElements.gait) hudElements.gait.innerText = "--";
}

// Spawn initial subjects
if (DATASET_COAT_PAIR === 'c57_pair' || DATASET_COAT_PAIR === 'black_pair') {
  applyStrainPreset('C57BL/6 (Black)');
  spawnMouse({ position: [-3.6, -2.6], heading: [1.0, 0.12], speed: 0.9 });
  applyStrainPreset('C57BL/6 (Black)');
  gui.controllersRecursive().forEach(c => c.updateDisplay());
  spawnMouse({ position: [3.3, 2.6], heading: [-1.0, -0.12], speed: 0.85 });
} else if (DATASET_COAT_PAIR === 'balbc_pair' || DATASET_COAT_PAIR === 'white_pair') {
  applyStrainPreset('BALB/c (White)');
  spawnMouse({ position: [-3.6, -2.6], heading: [1.0, 0.12], speed: 0.9 });
  applyStrainPreset('BALB/c (White)');
  gui.controllersRecursive().forEach(c => c.updateDisplay());
  spawnMouse({ position: [3.3, 2.6], heading: [-1.0, -0.12], speed: 0.85 });
} else if (DATASET_COAT_PAIR === 'cba_pair' || DATASET_COAT_PAIR === 'brown_pair') {
  applyStrainPreset('CBA (Brown)');
  spawnMouse({ position: [-3.6, -2.6], heading: [1.0, 0.12], speed: 0.9 });
  applyStrainPreset('CBA (Brown)');
  gui.controllersRecursive().forEach(c => c.updateDisplay());
  spawnMouse({ position: [3.3, 2.6], heading: [-1.0, -0.12], speed: 0.85 });
} else {
  applyStrainPreset('C57BL/6 (Black)');
  spawnMouse({ position: [-3.6, -2.6], heading: [1.0, 0.12], speed: 0.9 });
  applyStrainPreset('BALB/c (White)');
  gui.controllersRecursive().forEach(c => c.updateDisplay());
  spawnMouse({ position: [3.3, 2.6], heading: [-1.0, -0.12], speed: 0.85 });
}
if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'bottom-up') {
  mice.forEach(m => {
    if (m.contactShadow) m.contactShadow.visible = false;
  });
}
if (DATASET_HEADSTAGE) {
  if ((DATASET_HEADSTAGE === 'resident' || DATASET_HEADSTAGE === '1' || DATASET_HEADSTAGE === 'both') && mice[0]) {
    mice[0].setHeadstageEnabled(true);
    guiParams.implantHeadstage = true;
  }
  if ((DATASET_HEADSTAGE === 'intruder' || DATASET_HEADSTAGE === 'both') && mice[1]) {
    mice[1].setHeadstageEnabled(true);
  }
}

if (IS_ANATOMY_QA) {
  const anatomySubjectIndex = datasetQuery.get('anatomy_subject') === 'white' ? 1 : 0;
  const subject = mice[anatomySubjectIndex];
  const observer = mice[1 - anatomySubjectIndex];
  subject.mesh.position.set(0, 0, 0);
  subject.velocity.set(0, 0);
  subject.prevVelocity.set(0, 0);
  subject.heading.set(0, -1);
  subject.yawAngle = Math.PI;
  subject.prevHeadingAngle = Math.PI;
  subject.pauseTimer = 999;
  subject.groomCooldown = 999;
  subject.rearCooldown = 999;
  subject.tailMesh.visible = false;
  subject.tailRoot.visible = false;
  observer.mesh.visible = false;
  observer.mesh.position.set(8.0, 0, 5.0);
  observer.velocity.set(0, 0);
  observer.prevVelocity.set(0, 0);
  observer.pauseTimer = 999;
  guiParams.isPaused = true;
}

// Deterministic, hidden QA pose: one tail starts through the other subject
// while the body envelopes remain separated. The post-contact solver must
// route it around the obstacle without changing the normal scene defaults.
if (CONTACT_QA === 'tail') {
  const tailOwner = mice[1];
  const bodyObstacle = mice[0];
  tailOwner.mesh.position.set(0, 0, 3.5);
  tailOwner.velocity.set(0, 0);
  tailOwner.prevVelocity.set(0, 0);
  tailOwner.heading.set(0, 1);
  tailOwner.yawAngle = 0;
  tailOwner.prevHeadingAngle = 0;
  tailOwner.tailDyn.sway = 0;
  tailOwner.tailDyn.lift = -0.04;
  bodyObstacle.mesh.position.set(0, 0, -3.0);
  bodyObstacle.velocity.set(0, 0);
  bodyObstacle.prevVelocity.set(0, 0);
  bodyObstacle.heading.set(1, 0);
  bodyObstacle.yawAngle = Math.PI / 2;
  bodyObstacle.prevHeadingAngle = Math.PI / 2;
  bodyObstacle.tailDyn.sway = 0;
  bodyObstacle.tailDyn.lift = -0.04;
  guiParams.isPaused = true;
  camera.position.set(0, 18, 0.1);
  camera.up.set(0, 0, -1);
  controls.target.set(0, 0, 0);
  controls.update();
}
if (CONTACT_QA === 'framing') {
  const leftMouse = mice[0];
  const rightMouse = mice[1];
  leftMouse.mesh.position.set(-8.0, 0, 0);
  leftMouse.velocity.set(0, 0);
  leftMouse.prevVelocity.set(0, 0);
  leftMouse.heading.set(-1, 0);
  leftMouse.yawAngle = -Math.PI / 2;
  leftMouse.prevHeadingAngle = -Math.PI / 2;
  rightMouse.mesh.position.set(8.0, 0, 0);
  rightMouse.velocity.set(0, 0);
  rightMouse.prevVelocity.set(0, 0);
  rightMouse.heading.set(1, 0);
  rightMouse.yawAngle = Math.PI / 2;
  rightMouse.prevHeadingAngle = Math.PI / 2;
  guiParams.isPaused = true;
}
if (
  REARING_QA === 'unsupported'
  || REARING_QA === 'wall-supported'
  || REARING_QA === 'wall-transition'
) {
  const subject = mice[0];
  const observer = mice[1];
  observer.velocity.set(0, 0);
  observer.prevVelocity.set(0, 0);
  observer.pauseTimer = 5;
  if (REARING_QA === 'wall-supported' || REARING_QA === 'wall-transition') {
    observer.mesh.position.set(-8.0, 0, 4.5);
    const supportDirection = new THREE.Vector2(0, -1);
    // Begin with the nose at the boundary; rear-foot advance then carries
    // the support base toward the wall as the body rotates upright.
    subject.mesh.position.set(0, 0, -ENCLOSURE_HALF_Z + 4.10);
    subject.heading.copy(supportDirection);
    subject.yawAngle = Math.PI;
    subject.prevHeadingAngle = Math.PI;
    subject.startRearing('wall-supported', supportDirection, 999);
  } else {
    observer.mesh.position.set(6.0, 0, -5.5);
    subject.mesh.position.set(0, 0, 0.5);
    subject.heading.set(-0.60, 0.80).normalize();
    subject.yawAngle = Math.atan2(subject.heading.x, subject.heading.y);
    subject.prevHeadingAngle = subject.yawAngle;
    subject.startRearing('unsupported', null, 999);
  }
  subject.velocity.set(0, 0);
  subject.prevVelocity.set(0, 0);
  if (REARING_QA === 'wall-transition') {
    subject.rearing.holdDuration = 0.70;
  } else {
    subject.rearing.phase = 'hold';
    subject.rearing.phaseTime = 0;
    subject.rearing.angle = subject.rearing.maxAngle;
    subject.rearing.angularVelocity = 0;
    subject.rearAmount = subject.rearing.peakAmount;
    subject.rearVel = 0;
    guiParams.isPaused = true;
  }
}
if (
  GROOMING_QA === 'face-wash'
  || GROOMING_QA === 'flank-groom'
  || GROOMING_QA === 'transition'
) {
  const subject = mice[0];
  const observer = mice[1];
  const groomingMode = GROOMING_QA === 'flank-groom'
    ? 'flank-groom'
    : 'face-wash';
  subject.mesh.position.set(0, 0, 0.5);
  subject.velocity.set(0, 0);
  subject.prevVelocity.set(0, 0);
  subject.heading.set(-0.58, 0.82).normalize();
  subject.yawAngle = Math.atan2(subject.heading.x, subject.heading.y);
  subject.prevHeadingAngle = subject.yawAngle;
  subject.startGrooming(groomingMode, 999);
  if (groomingMode === 'flank-groom') subject.grooming.side = 1;
  observer.mesh.position.set(7.0, 0, -5.5);
  observer.velocity.set(0, 0);
  observer.prevVelocity.set(0, 0);
  observer.pauseTimer = 5;
  observer.groomCooldown = 999;
  observer.rearCooldown = 999;
  if (GROOMING_QA === 'transition') {
    subject.grooming.activeDuration = 0.75;
  } else {
    subject.grooming.phase = 'active';
    subject.grooming.phaseTime = 0;
    subject.grooming.elapsed = groomingMode === 'face-wash' ? 0.17 : 0.23;
    subject.groomAmount = subject.grooming.peakAmount;
    subject.groomVel = 0;
    guiParams.isPaused = true;
  }
}
if (DATASET_CAPTURE && DATASET_SCENARIO && mice.length >= 2) {
  const mouseA = mice[0];
  const mouseB = mice[1];
  const placeMouse = (mouse, px, pz, hx, hz, speed) => {
    const dir = new THREE.Vector2(hx, hz).normalize();
    mouse.mesh.position.set(px, 0, pz);
    mouse.heading.copy(dir);
    mouse.yawAngle = Math.atan2(dir.x, dir.y);
    mouse.prevHeadingAngle = mouse.yawAngle;
    mouse.mesh.rotation.y = mouse.yawAngle;
    mouse.velocity.set(dir.x * speed, dir.y * speed);
    mouse.prevVelocity.copy(mouse.velocity);
  };
  if (DATASET_SCENARIO === 'social_huddle') {
    placeMouse(mouseA, -3.1, -0.68, 1.0, 0.12, 1.85);
    placeMouse(mouseB, 3.1, 0.68, -1.0, -0.12, 1.85);
    mouseA.groomCooldown = 999;
    mouseB.groomCooldown = 999;
    mouseA.rearCooldown = 999;
    mouseB.rearCooldown = 999;
  } else if (DATASET_SCENARIO === 'curled_groom') {
    placeMouse(mouseA, -1.65, 0.35, -0.45, 0.89, 0.0);
    mouseA.startGrooming('flank-groom', 999);
    mouseA.grooming.side = 1;
    mouseA.grooming.phase = 'active';
    mouseA.grooming.phaseTime = 0;
    mouseA.grooming.elapsed = 0.22;
    mouseA.grooming.activeDuration = 6.0;
    mouseA.groomAmount = mouseA.grooming.peakAmount;
    mouseA.groomVel = 0;
    placeMouse(mouseB, 2.35, -1.35, -0.78, 0.62, 1.35);
    mouseB.groomCooldown = 999;
    mouseB.rearCooldown = 999;
  } else if (DATASET_SCENARIO === 'rearing_social') {
    placeMouse(mouseA, -1.35, 0.30, 0.68, 0.73, 0.0);
    mouseA.startRearing('unsupported', null, 999);
    mouseA.rearing.holdDuration = 1.65;
    placeMouse(mouseB, 2.45, -0.75, -0.92, 0.38, 1.20);
    mouseB.groomCooldown = 999;
    mouseB.rearCooldown = 999;
  } else if (DATASET_SCENARIO === 'arena_explore') {
    placeMouse(mouseA, -2.55, -1.35, 0.94, 0.34, 1.65);
    placeMouse(mouseB, 2.55, 1.35, -0.94, -0.34, 1.55);
  } else if (DATASET_SCENARIO === 'ventral_gait') {
    placeMouse(mouseA, -2.60, -0.95, 0.96, 0.28, 2.10);
    placeMouse(mouseB, 2.60, 0.95, -0.96, -0.28, 2.00);
    mouseA.groomCooldown = 999;
    mouseB.groomCooldown = 999;
    mouseA.rearCooldown = 999;
    mouseB.rearCooldown = 999;
  } else if (DATASET_SCENARIO === 'calms_resident_intruder') {
    // Resident-intruder home-cage investigation inspired by mouse083_task1_annotator1.mp4:
    // resident (black, cranial headstage) greets intruder (white) near the left nestlet, then
    // actively trails/pursues intruder as they dart and circle along the cage walls.
    placeMouse(mouseA, -8.40, -2.60, 0.88, 0.47, 7.60);
    placeMouse(mouseB, -3.20, 1.40, 0.72, -0.69, 6.80);
    mouseA.boutState.mode = 'dart-trot';
    mouseA.boutState.timer = 1.35;
    mouseB.boutState.mode = 'dart-trot';
    mouseB.boutState.timer = 1.15;
    mouseA.groomCooldown = 999;
    mouseB.groomCooldown = 999;
    mouseA.rearCooldown = 999;
    mouseB.rearCooldown = 999;
  }
}

function projectPoseNode(node, width, height) {
  const worldPosition = new THREE.Vector3();
  node.getWorldPosition(worldPosition);
  const projected = worldPosition.clone().project(camera);
  const x = (projected.x * 0.5 + 0.5) * width;
  const y = (-projected.y * 0.5 + 0.5) * height;
  const inFrame = Number.isFinite(x) && Number.isFinite(y)
    && projected.z >= -1 && projected.z <= 1
    && x >= 0 && x < width && y >= 0 && y < height;
  return inFrame
    ? { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)), visibility: 2 }
    : { x: 0, y: 0, visibility: 0 };
}

function buildPoseAnnotation(mouse, imageId, width, height) {
  const projectedPoints = POSE_KEYPOINTS.map(({ id }) =>
    projectPoseNode(mouse.poseNodes[id], width, height)
  );
  const visiblePoints = projectedPoints.filter(point => point.visibility > 0);
  const keypoints = projectedPoints.flatMap(point => [point.x, point.y, point.visibility]);
  let bbox = [0, 0, 0, 0];
  if (visiblePoints.length > 0) {
    const padding = 10;
    const minX = Math.max(0, Math.min(...visiblePoints.map(point => point.x)) - padding);
    const minY = Math.max(0, Math.min(...visiblePoints.map(point => point.y)) - padding);
    const maxX = Math.min(width, Math.max(...visiblePoints.map(point => point.x)) + padding);
    const maxY = Math.min(height, Math.max(...visiblePoints.map(point => point.y)) + padding);
    bbox = [minX, minY, Math.max(0, maxX - minX), Math.max(0, maxY - minY)]
      .map(value => Number(value.toFixed(3)));
  }
  return {
    id: (imageId - 1) * mice.length + mouse.trackId,
    image_id: imageId,
    category_id: POSE_CATEGORY.id,
    track_id: mouse.trackId,
    subject_name: mouse.strainName,
    behavior: mouse.rearing.mode !== 'none'
      ? 'rearing'
      : (mouse.grooming.mode !== 'none' ? 'grooming' : 'ground'),
    rearing_mode: mouse.rearing.mode,
    rearing_phase: mouse.rearing.phase,
    rearing_amount: Number(mouse.rearAmount.toFixed(4)),
    rearing_angle_degrees: Number(
      THREE.MathUtils.radToDeg(mouse.rearing.angle).toFixed(3)
    ),
    rearing_angular_velocity: Number(mouse.rearing.angularVelocity.toFixed(5)),
    rearing_com_projection_error: Number(
      mouse.rearing.centerOfMassProjectionError.toFixed(5)
    ),
    rearing_hind_support_error: Number(mouse.rearing.hindSupportError.toFixed(5)),
    rearing_wall_support_error: Number(mouse.rearing.wallSupportError.toFixed(5)),
    rearing_wall_contact_load: Number(mouse.rearing.wallContactLoad.toFixed(4)),
    rearing_base_advance: Number(mouse.rearing.baseAdvance.toFixed(5)),
    grooming_mode: mouse.grooming.mode,
    grooming_phase: mouse.grooming.phase,
    grooming_amount: Number(mouse.groomAmount.toFixed(4)),
    grooming_paw_contact_error: Number(
      mouse.groomingPawContactError.toFixed(5)
    ),
    grooming_support_error: Number(mouse.groomingSupportError.toFixed(5)),
    forelimb_extension_ratio: Number(
      mouse.maximumForelimbExtensionRatio.toFixed(5)
    ),
    keypoints,
    num_keypoints: visiblePoints.length,
    bbox,
    area: Number((bbox[2] * bbox[3]).toFixed(3)),
    iscrowd: 0,
  };
}

function buildDatasetFramePayload() {
  controls.update();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const width = Math.round(canvas.getBoundingClientRect().width || innerWidth);
  const height = Math.round(canvas.getBoundingClientRect().height || innerHeight);
  const imageId = datasetCurrentFrameIndex + 1;
  const clearance = getMinimumMouseClearance(mice);
  const tailClearance = getMinimumTailClearance(mice);
  const maximumTailSegmentLengthError = getMaximumTailSegmentLengthError(mice);
  const maximumTailBendAngle = getMaximumTailBendAngle(mice);
  const maximumFootSlip = getMaximumFootSlip(mice);
  const maximumPawGroundError = getMaximumPawGroundError(mice);
  return {
    schema_version: 1,
    seed: DATASET_SEED,
    fps: DATASET_CAPTURE_FPS,
    image: {
      id: imageId,
      file_name: `images/frame_${String(datasetCurrentFrameIndex).padStart(6, '0')}.jpg`,
      width,
      height,
      video_id: 1,
      frame_id: datasetCurrentFrameIndex,
      timestamp: Number((datasetCurrentFrameIndex / DATASET_CAPTURE_FPS).toFixed(6)),
      minimum_subject_clearance: Number.isFinite(clearance)
        ? Number(clearance.toFixed(6))
        : null,
      subject_overlap: Number.isFinite(clearance) && clearance < -0.01,
      minimum_tail_clearance: Number.isFinite(tailClearance)
        ? Number(tailClearance.toFixed(6))
        : null,
      tail_overlap: Number.isFinite(tailClearance) && tailClearance < -0.001,
      maximum_tail_segment_length_error: Number(
        maximumTailSegmentLengthError.toFixed(6)
      ),
      maximum_tail_bend_angle_degrees: Number(maximumTailBendAngle.toFixed(4)),
      maximum_foot_slip: Number(maximumFootSlip.toFixed(6)),
      maximum_paw_ground_error: Number(maximumPawGroundError.toFixed(6)),
    },
    annotations: mice.map(mouse => buildPoseAnnotation(mouse, imageId, width, height)),
    category: POSE_CATEGORY,
  };
}

// ═══════════════════════════════════════════════════════════════
// 15. RENDER LOOP
// ═══════════════════════════════════════════════════════════════
function frameCameraView() {
  if (DATASET_CAPTURE) return;
  const subject = mice.find(mouse => mouse.appearance.strain === (
    guiParams.cameraView === 'Follow black' ? 'C57BL/6 (Black)' : 'BALB/c (White)'
  ));
  camera.up.set(0, 1, 0);
  if (guiParams.cameraView.startsWith('Follow') && subject) {
    controls.target.copy(subject.mesh.position).y = 1.0;
    camera.position.copy(controls.target).add(new THREE.Vector3(8.5, 3.8, 8.5));
  } else if (guiParams.cameraView === 'Top down') {
    controls.target.set(0, 0, 0);
    camera.position.set(0, 44, 0.01);
  } else if (guiParams.cameraView === 'Home cage top') {
    camera.up.set(0, 0, -1);
    camera.fov = 25.5;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0.25, 0);
    camera.position.set(0, 56.5, 0.02);
  } else {
    controls.target.copy(cameraTarget);
    camera.position.copy(cameraBasePosition);
  }
  controls.update();
}

const cameraFollowDelta = new THREE.Vector3();
function followCamera(dt) {
  if (DATASET_CAPTURE || !guiParams.cameraView.startsWith('Follow')) return;
  const strain = guiParams.cameraView === 'Follow black' ? 'C57BL/6 (Black)' : 'BALB/c (White)';
  const subject = mice.find(mouse => mouse.appearance.strain === strain);
  if (!subject) return;
  cameraFollowDelta.copy(subject.mesh.position);
  cameraFollowDelta.y = 1.0;
  cameraFollowDelta.sub(controls.target).multiplyScalar(1 - Math.exp(-4 * dt));
  controls.target.add(cameraFollowDelta);
  camera.position.add(cameraFollowDelta);
}

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  const portraitFit = THREE.MathUtils.clamp((0.90 - camera.aspect) / 0.45, 0, 1);
  const cameraDistanceScale = 1.0 + portraitFit * 0.42;
  camera.fov = THREE.MathUtils.lerp(34, 50, portraitFit);
  if (ANATOMY_QA === 'torso-side') {
    const anatomyTarget = new THREE.Vector3(0, 0.62, -0.35);
    const anatomyCamera = new THREE.Vector3(9.20, 2.20, -0.35);
    camera.fov = THREE.MathUtils.lerp(31, 50, portraitFit);
    camera.up.set(0, 1, 0);
    camera.position.copy(anatomyCamera).sub(anatomyTarget)
      .multiplyScalar(1.0 + portraitFit * 0.55)
      .add(anatomyTarget);
    controls.target.copy(anatomyTarget);
  } else if (ANATOMY_QA === 'torso-front') {
    const anatomyTarget = new THREE.Vector3(0, 0.64, -1.45);
    const anatomyCamera = new THREE.Vector3(0, 1.42, -10.60);
    camera.fov = THREE.MathUtils.lerp(31, 48, portraitFit);
    camera.up.set(0, 1, 0);
    camera.position.copy(anatomyCamera).sub(anatomyTarget)
      .multiplyScalar(1.0 + portraitFit * 0.48)
      .add(anatomyTarget);
    controls.target.copy(anatomyTarget);
  } else if (ANATOMY_QA === 'ears' || ANATOMY_QA === 'ears-front') {
    const anatomyTarget = new THREE.Vector3(0, 0.64, -2.60);
    const anatomyCamera = ANATOMY_QA === 'ears-front'
      ? new THREE.Vector3(0, 1.28, -9.10)
      : new THREE.Vector3(5.00, 2.32, -7.85);
    camera.fov = THREE.MathUtils.lerp(31, 46, portraitFit);
    camera.up.set(0, 1, 0);
    camera.position.copy(anatomyCamera).sub(anatomyTarget)
      .multiplyScalar(1.0 + portraitFit * 0.42)
      .add(anatomyTarget);
    controls.target.copy(anatomyTarget);
  } else if (ANATOMY_QA === 'head-side' || ANATOMY_QA === 'head-three-quarter' || ANATOMY_QA === 'head-top' || ANATOMY_QA === 'mouth-front') {
    // Close head/neck/mouth inspection views (subject faces -Z at the origin).
    const isMouthFront = ANATOMY_QA === 'mouth-front';
    const anatomyTarget = isMouthFront
      ? new THREE.Vector3(0, 1.24, -3.22)
      : new THREE.Vector3(0, 1.36, -2.75);
    const anatomyCamera = isMouthFront
      ? new THREE.Vector3(0.0, 0.98, -5.05)
      : (ANATOMY_QA === 'head-side'
        ? new THREE.Vector3(4.60, 1.62, -2.75)
        : (ANATOMY_QA === 'head-top'
          ? new THREE.Vector3(0.0, 6.20, -2.70)
          : new THREE.Vector3(2.85, 2.25, -6.10)));
    camera.fov = THREE.MathUtils.lerp(isMouthFront ? 22 : 30, isMouthFront ? 32 : 44, portraitFit);
    camera.up.set(0, ANATOMY_QA === 'head-top' ? 0 : 1, ANATOMY_QA === 'head-top' ? -1 : 0);
    camera.position.copy(anatomyCamera).sub(anatomyTarget)
      .multiplyScalar(1.0 + portraitFit * 0.42)
      .add(anatomyTarget);
    controls.target.copy(anatomyTarget);
  } else if (REARING_QA === 'wall-supported' || REARING_QA === 'wall-transition') {
    camera.position.set(15.0, 6.2, 2.0);
    camera.up.set(0, 1, 0);
    controls.target.set(0, 2.4, -ENCLOSURE_HALF_Z + 2.15);
  } else if (CONTACT_QA === 'tail') {
    camera.position.set(0, 18, 0.1);
    camera.up.set(0, 0, -1);
    controls.target.set(0, 0, 0);
  } else if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'top-close') {
    camera.position.set(0, 19.5, 0.02);
    camera.up.set(0, 0, -1);
    controls.target.set(0, 0.35, 0);
  } else if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'top') {
    camera.position.set(0, 29.0, 0.02);
    camera.up.set(0, 0, -1);
    controls.target.set(0, 0.35, 0);
  } else if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'home-cage-top') {
    camera.fov = 25.5;
    camera.position.set(0, 56.5, 0.02);
    camera.up.set(0, 0, -1);
    controls.target.set(0, 0.25, 0);
  } else if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'oblique-close') {
    camera.position.set(5.8, 4.4, 12.6);
    camera.up.set(0, 1, 0);
    controls.target.set(0, 0.85, 0);
  } else if (DATASET_CAPTURE && DATASET_CAMERA_VIEW === 'bottom-up') {
    camera.position.set(0, -18.5, 0.02);
    camera.up.set(0, 0, 1);
    controls.target.set(0, 0.28, 0);
  } else {
    camera.up.set(0, 1, 0);
    camera.position.copy(cameraBasePosition).sub(cameraTarget).multiplyScalar(cameraDistanceScale).add(cameraTarget);
    controls.target.copy(cameraTarget);
  }
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  postFX.resize();
  if (!DATASET_CAPTURE && guiParams.cameraView !== 'Both mice') frameCameraView();
});
window.dispatchEvent(new Event('resize'));

const clock = new THREE.Clock();
// Keep controls responsive on software WebGL: avoid a six-step catch-up
// spiral when a dense photographic frame takes longer than a physics tick.
const simulationClock = new FixedStepClock(DATASET_SIMULATION_HZ, 2);
let lastTelemetryTime = -1;
let simulationDurationMs = 0;
let renderDurationMs = 0;
let accumulatedTime = 0;
let minimumObservedSubjectClearance = Infinity;
let minimumObservedTailClearance = Infinity;
let minimumObservedTailDetails = null;
let maximumObservedFootSlip = 0;
let maximumObservedPawGroundError = 0;
let maximumObservedRearSupportError = 0;
let maximumObservedRearSupportDetails = null;
const observedRearingModes = new Set();
const observedGroomingModes = new Set();
let datasetCurrentFrameIndex = 0;
let datasetInitialized = false;

function findSubjectByTrackId(trackId) {
  const subject = mice.find(mouse => mouse.trackId === Number(trackId));
  if (!subject) {
    throw new RangeError(
      `No subject with trackId ${JSON.stringify(trackId)}; available: ${mice.map(m => m.trackId).join(', ') || 'none'}.`
    );
  }
  return subject;
}

// Diagnostics and observations are read-only snapshots; setCommand/clearCommand
// are the only supported control entry points (FlyBody-style task commands).
// Seeded dataset captures stay reproducible as long as no command is issued.
window.annolidTwoMice = Object.freeze({
  getDiagnostics: () => ({
    time: accumulatedTime,
    paused: guiParams.isPaused,
    simulationHz: DATASET_SIMULATION_HZ,
    subjects: mice.map(mouse => ({
      trackId: mouse.trackId,
      strain: mouse.appearance.strain,
      position: mouse.mesh.position.toArray(),
      heading: mouse.yawAngle,
      gait: mouse.motor.cpg.gait,
      stepping: mouse.motor.isStepping(),
      commanded: Boolean(mouse.externalCommand),
    })),
    renderer: {
      simulationMs: simulationDurationMs,
      renderMs: renderDurationMs,
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      geometries: renderer.info.memory.geometries,
      textures: renderer.info.memory.textures,
    },
  }),
  getActionSpec: () => getMouseActionSpec(),
  getObservation: trackId => findSubjectByTrackId(trackId).getObservation(),
  getObservations: () => mice.map(mouse => mouse.getObservation()),
  setCommand: (trackId, command) => findSubjectByTrackId(trackId)
    .setMotorCommand(command, accumulatedTime),
  clearCommand: trackId => {
    findSubjectByTrackId(trackId).clearMotorCommand();
    return null;
  },
});

function updateCollisionTelemetry() {
  const minimumSubjectClearance = getMinimumMouseClearance(mice);
  const minimumTailClearance = getMinimumTailClearance(mice);
  const maximumTailSegmentLengthError = getMaximumTailSegmentLengthError(mice);
  const maximumTailBendAngle = getMaximumTailBendAngle(mice);
  const maximumFootSlip = getMaximumFootSlip(mice);
  const maximumPawGroundError = getMaximumPawGroundError(mice);
  const maximumRearSupportError = Math.max(
    0,
    ...mice.map(mouse => mouse.maximumRearSupportError || 0)
  );
  for (const mouse of mice) {
    if (mouse.rearing.mode !== 'none') observedRearingModes.add(mouse.rearing.mode);
    if (mouse.grooming.mode !== 'none') observedGroomingModes.add(mouse.grooming.mode);
  }
  if (Number.isFinite(minimumSubjectClearance)) {
    minimumObservedSubjectClearance = Math.min(
      minimumObservedSubjectClearance,
      minimumSubjectClearance
    );
  }
  if (Number.isFinite(minimumTailClearance)) {
    if (minimumTailClearance < minimumObservedTailClearance) {
      const closestTail = mice.reduce((closest, mouse) => (
        mouse.minimumTailClearance < (closest?.minimumTailClearance ?? Infinity)
          ? mouse
          : closest
      ), null);
      minimumObservedTailDetails = closestTail ? {
        trackId: closestTail.trackId,
        clearance: Number(closestTail.minimumTailClearance.toFixed(6)),
        pointIndex: closestTail.minimumTailClearanceIndex,
      } : null;
    }
    minimumObservedTailClearance = Math.min(
      minimumObservedTailClearance,
      minimumTailClearance
    );
  }
  maximumObservedFootSlip = Math.max(maximumObservedFootSlip, maximumFootSlip);
  maximumObservedPawGroundError = Math.max(
    maximumObservedPawGroundError,
    maximumPawGroundError
  );
  if (maximumRearSupportError > maximumObservedRearSupportError) {
    const mostLoadedSupport = mice.reduce((current, mouse) => (
      mouse.maximumRearSupportError > (current?.maximumRearSupportError ?? -1)
        ? mouse
        : current
    ), null);
    maximumObservedRearSupportDetails = mostLoadedSupport ? {
      trackId: mostLoadedSupport.trackId,
      mode: mostLoadedSupport.rearing.mode,
      phase: mostLoadedSupport.rearing.phase,
      angleDegrees: Number(
        THREE.MathUtils.radToDeg(mostLoadedSupport.rearing.angle).toFixed(3)
      ),
      hindSupportError: Number(
        mostLoadedSupport.rearing.hindSupportError.toFixed(5)
      ),
      wallSupportError: Number(
        mostLoadedSupport.rearing.wallSupportError.toFixed(5)
      ),
      wallContactLoad: Number(
        mostLoadedSupport.rearing.wallContactLoad.toFixed(4)
      ),
    } : null;
    maximumObservedRearSupportError = maximumRearSupportError;
  }
  canvas.dataset.minimumSubjectClearance = Number.isFinite(minimumSubjectClearance)
    ? minimumSubjectClearance.toFixed(4)
    : 'na';
  canvas.dataset.minimumObservedSubjectClearance = Number.isFinite(minimumObservedSubjectClearance)
    ? minimumObservedSubjectClearance.toFixed(4)
    : 'na';
  canvas.dataset.subjectOverlap = minimumSubjectClearance < -0.01 ? 'true' : 'false';
  canvas.dataset.minimumTailClearance = Number.isFinite(minimumTailClearance)
    ? minimumTailClearance.toFixed(4)
    : 'na';
  canvas.dataset.minimumObservedTailClearance = Number.isFinite(minimumObservedTailClearance)
    ? minimumObservedTailClearance.toFixed(4)
    : 'na';
  canvas.dataset.tailOverlap = minimumTailClearance < -0.001 ? 'true' : 'false';
  canvas.dataset.observedTailOverlap = minimumObservedTailClearance < -0.001
    ? 'true'
    : 'false';
  canvas.dataset.minimumObservedTailDetails = JSON.stringify(minimumObservedTailDetails);
  canvas.dataset.maximumTailSegmentLengthError = maximumTailSegmentLengthError.toFixed(4);
  canvas.dataset.maximumTailBendAngle = maximumTailBendAngle.toFixed(2);
  canvas.dataset.maximumFootSlip = maximumFootSlip.toFixed(4);
  canvas.dataset.maximumObservedFootSlip = maximumObservedFootSlip.toFixed(4);
  canvas.dataset.maximumPawGroundError = maximumPawGroundError.toFixed(4);
  canvas.dataset.maximumObservedPawGroundError = maximumObservedPawGroundError.toFixed(4);
  canvas.dataset.maximumRearSupportError = maximumRearSupportError.toFixed(4);
  canvas.dataset.maximumObservedRearSupportError = maximumObservedRearSupportError.toFixed(4);
  canvas.dataset.maximumObservedRearSupportDetails = JSON.stringify(
    maximumObservedRearSupportDetails
  );
  canvas.dataset.rearingActive = mice.some(mouse => mouse.rearing.mode !== 'none')
    ? 'true'
    : 'false';
  canvas.dataset.observedRearingModes = JSON.stringify([...observedRearingModes]);
  canvas.dataset.rearingBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    mode: mouse.rearing.mode,
    phase: mouse.rearing.phase,
    amount: Number(mouse.rearAmount.toFixed(4)),
  })));
  canvas.dataset.rearingPhysicsBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    angleDegrees: Number(THREE.MathUtils.radToDeg(mouse.rearing.angle).toFixed(3)),
    angularVelocity: Number(mouse.rearing.angularVelocity.toFixed(4)),
    angularAcceleration: Number(mouse.rearing.angularAcceleration.toFixed(4)),
    centerOfMassForwardOffset: Number(
      mouse.rearing.centerOfMassForwardOffset.toFixed(4)
    ),
    centerOfMassLateralOffset: Number(
      mouse.rearing.centerOfMassLateralOffset.toFixed(4)
    ),
    centerOfMassProjectionError: Number(
      mouse.rearing.centerOfMassProjectionError.toFixed(4)
    ),
    hindSupportError: Number(mouse.rearing.hindSupportError.toFixed(4)),
    hindSupportHorizontalError: Number(
      mouse.rearing.hindSupportHorizontalError.toFixed(4)
    ),
    hindSupportVerticalError: Number(
      mouse.rearing.hindSupportVerticalError.toFixed(4)
    ),
    wallSupportError: Number(mouse.rearing.wallSupportError.toFixed(4)),
    wallSupportNormalError: Number(
      mouse.rearing.wallSupportNormalError.toFixed(4)
    ),
    wallSupportVerticalError: Number(
      mouse.rearing.wallSupportVerticalError.toFixed(4)
    ),
    wallContactLoad: Number(mouse.rearing.wallContactLoad.toFixed(4)),
    baseAdvance: Number(mouse.rearing.baseAdvance.toFixed(4)),
    baseAdvanceVelocity: Number(mouse.rearing.baseAdvanceVelocity.toFixed(4)),
    maximumBaseAdvance: Number(mouse.rearing.maximumBaseAdvance.toFixed(4)),
    hindStanceOffset: Number(mouse.rearing.hindStanceOffset.toFixed(4)),
    hindStanceVelocity: Number(mouse.rearing.hindStanceVelocity.toFixed(4)),
    supportReleased: mouse.rearing.rearSupportReleased,
    constraintDt: Number(mouse.rearing.lastConstraintDt.toFixed(5)),
    positionX: Number(mouse.mesh.position.x.toFixed(4)),
    positionZ: Number(mouse.mesh.position.z.toFixed(4)),
    rigHeight: Number(mouse.rig.position.y.toFixed(4)),
    gravityTorque: Number(mouse.rearing.gravityTorque.toFixed(5)),
    muscleTorque: Number(mouse.rearing.muscleTorque.toFixed(5)),
    wallReactionTorque: Number(mouse.rearing.wallReactionTorque.toFixed(5)),
    forelimbExtensionRatio: Number(
      mouse.maximumForelimbExtensionRatio.toFixed(4)
    ),
  })));
  canvas.dataset.groomingActive = mice.some(mouse => mouse.grooming.mode !== 'none')
    ? 'true'
    : 'false';
  canvas.dataset.observedGroomingModes = JSON.stringify([...observedGroomingModes]);
  canvas.dataset.groomingBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    mode: mouse.grooming.mode,
    phase: mouse.grooming.phase,
    amount: Number(mouse.groomAmount.toFixed(4)),
    pawContactError: Number(mouse.groomingPawContactError.toFixed(4)),
    supportError: Number(mouse.groomingSupportError.toFixed(4)),
    reachDeficit: Number(mouse.groomingReachDeficit.toFixed(4)),
    targetReachRatio: Number(mouse.groomingTargetReachRatio.toFixed(4)),
  })));
  canvas.dataset.forelimbKinematicsBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    maximumExtensionRatio: Number(
      mouse.maximumForelimbExtensionRatio.toFixed(4)
    ),
    leftReach: Number(mouse.armL._maximumReach.toFixed(4)),
    rightReach: Number(mouse.armR._maximumReach.toFixed(4)),
    humerusLength: Number(mouse.armL._segmentLengths.humerus.toFixed(4)),
    radiusUlnaLength: Number(
      mouse.armL._segmentLengths.radiusUlna.toFixed(4)
    ),
    forepaws: mouse.footPlants.slice(2).map(plant => ({
      side: plant.side,
      planted: plant.planted,
      exhausted: plant.exhausted,
      rejectionReason: plant.rejectionReason,
      reachRatio: Number(plant.reachRatio.toFixed(4)),
      candidateSlip: Number(plant.candidateSlip.toFixed(4)),
      candidateGroundError: Number(plant.candidateGroundError.toFixed(4)),
      extensionRatio: Number((plant.extensionRatio || 0).toFixed(4)),
      pawReachRatio: Number((plant.pawReachRatio || 0).toFixed(4)),
      constraintError: {
        x: Number(plant.constraintError.x.toFixed(4)),
        y: Number(plant.constraintError.y.toFixed(4)),
        z: Number(plant.constraintError.z.toFixed(4)),
      },
      constraintTargetLocal: plant.constraintTargetLocal.toArray().map(
        value => Number(value.toFixed(4))
      ),
      constraintContactLocal: plant.constraintContactLocal.toArray().map(
        value => Number(value.toFixed(4))
      ),
    })),
  })));
  canvas.dataset.tailClearanceBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    clearance: Number.isFinite(mouse.minimumTailClearance)
      ? Number(mouse.minimumTailClearance.toFixed(4))
      : null,
    pointIndex: mouse.minimumTailClearanceIndex,
  })));
  canvas.dataset.tailContinuityBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    segmentLengthError: Number(
      (mouse.maximumTailSegmentLengthError || 0).toFixed(4)
    ),
    bendAngleDegrees: Number((mouse.maximumTailBendAngle || 0).toFixed(2)),
  })));
  canvas.dataset.footContactBySubject = JSON.stringify(mice.map(mouse => ({
    trackId: mouse.trackId,
    slip: Number((mouse.maximumFootSlip || 0).toFixed(4)),
    groundError: Number((mouse.maximumPawGroundError || 0).toFixed(4)),
    plantedFeet: mouse.footPlants.filter(foot => foot.planted).length,
    contacts: mouse.footPlants.map(foot => ({
      side: foot.side,
      fore: foot.isFore,
      planted: foot.planted,
      exhausted: foot.exhausted,
      rejectionReason: foot.rejectionReason,
      reachRatio: Number(foot.reachRatio.toFixed(4)),
      candidateSlip: Number(foot.candidateSlip.toFixed(4)),
      candidateGroundError: Number(foot.candidateGroundError.toFixed(4)),
    })),
  })));
  if (ANATOMY_QA) {
    canvas.dataset.anatomyQa = ANATOMY_QA;
    canvas.dataset.anatomySubject = datasetQuery.get('anatomy_subject') === 'white'
      ? 'white'
      : 'black';
    canvas.dataset.torsoQa = IS_TORSO_ANATOMY_QA ? 'true' : 'false';
    canvas.dataset.torsoProportions = JSON.stringify({
      bodyLength: PROP.bodyLength,
      bodyGirthX: PROP.bodyGirthX,
      bodyGirthY: PROP.bodyGirthY,
      neckLength: PROP.neckLength,
      neckGirth: PROP.neckGirth,
      headSize: PROP.headSize,
      headScale: [0.92, 0.94, 1.03],
      lengthToHeightRatio: Number(
        (PROP.bodyLength / PROP.bodyGirthY).toFixed(4)
      ),
    });
    canvas.dataset.headNeckBodyBySubject = JSON.stringify(mice.map(mouse => {
      const bodyCenter = new THREE.Vector3();
      const neckCenter = new THREE.Vector3();
      const headPivot = new THREE.Vector3();
      const headCenter = new THREE.Vector3();
      mouse.bodyMesh.getWorldPosition(bodyCenter);
      mouse.neckMesh.getWorldPosition(neckCenter);
      mouse.headGroup.getWorldPosition(headPivot);
      mouse.head.getWorldPosition(headCenter);
      return {
        trackId: mouse.trackId,
        bodyToNeck: Number(bodyCenter.distanceTo(neckCenter).toFixed(4)),
        neckToHeadPivot: Number(neckCenter.distanceTo(headPivot).toFixed(4)),
        headPivotToCenter: Number(headPivot.distanceTo(headCenter).toFixed(4)),
        headScale: mouse.head.scale.toArray().map(
          value => Number(value.toFixed(4))
        ),
        neckRotation: [
          mouse.neckMesh.rotation.x,
          mouse.neckMesh.rotation.y,
          mouse.neckMesh.rotation.z,
        ].map(value => Number(value.toFixed(4))),
      };
    }));
    canvas.dataset.earAttachmentBySubject = JSON.stringify(mice.map(mouse => {
      const headCenter = new THREE.Vector3();
      const leftRoot = new THREE.Vector3();
      const rightRoot = new THREE.Vector3();
      const leftPinna = new THREE.Vector3();
      const rightPinna = new THREE.Vector3();
      mouse.head.getWorldPosition(headCenter);
      mouse.earL.getWorldPosition(leftRoot);
      mouse.earR.getWorldPosition(rightRoot);
      mouse.earL._pinna.getWorldPosition(leftPinna);
      mouse.earR._pinna.getWorldPosition(rightPinna);
      return {
        trackId: mouse.trackId,
        leftRootToHead: Number(leftRoot.distanceTo(headCenter).toFixed(4)),
        rightRootToHead: Number(rightRoot.distanceTo(headCenter).toFixed(4)),
        leftPinnaRise: Number(leftPinna.distanceTo(leftRoot).toFixed(4)),
        rightPinnaRise: Number(rightPinna.distanceTo(rightRoot).toFixed(4)),
      };
    }));
  }
}

function advanceDatasetFrame(targetFrame) {
  const parsedTarget = Number.parseInt(targetFrame, 10);
  const clampedTarget = Math.max(0, Math.min(10000, parsedTarget || 0));
  if (clampedTarget < datasetCurrentFrameIndex) {
    throw new RangeError('Dataset capture frames must advance monotonically.');
  }
  const dt = 1 / DATASET_SIMULATION_HZ;
  if (!datasetInitialized) {
    mice.forEach(m => m.update(0, mice, null, 0));
    resolveAllMouseContacts(mice, dt);
    datasetInitialized = true;
  }
  const targetSimulationSteps = Math.round(
    clampedTarget * DATASET_SIMULATION_HZ / DATASET_CAPTURE_FPS
  );
  const currentSimulationSteps = Math.round(
    datasetCurrentFrameIndex * DATASET_SIMULATION_HZ / DATASET_CAPTURE_FPS
  );
  const simulationSteps = targetSimulationSteps - currentSimulationSteps;
  for (let step = 0; step < simulationSteps; step++) {
    accumulatedTime += dt;
    guiParams.isPaused = false;
    mice.forEach(m => m.update(accumulatedTime, mice, null, dt));
    resolveAllMouseContacts(mice, dt);
    updateCollisionTelemetry();
  }
  datasetCurrentFrameIndex = clampedTarget;
  guiParams.isPaused = true;
  updateCollisionTelemetry();
  controls.update();
  scene.updateMatrixWorld(true);
}

const SEGMENTATION_FINE_PARTS = Object.freeze({
  1: 'head',
  2: 'snout',
  3: 'left_ear',
  4: 'right_ear',
  5: 'neck',
  6: 'torso',
  7: 'left_forelimb',
  8: 'right_forelimb',
  9: 'left_hindlimb',
  10: 'right_hindlimb',
  11: 'tail',
});
const SEGMENTATION_COARSE_PARTS = Object.freeze({
  1: 'head',
  2: 'torso',
  3: 'limbs',
  4: 'tail',
});

let segmentationRenderTarget = null;
let segmentationPixelBuffer = null;
let segmentationDiscardMaterial = null;
const segmentationIdMaterials = new Map();
const spinalSegmentationIdMaterials = new Map();

function getSegmentationDiscardMaterial() {
  if (!segmentationDiscardMaterial) {
    segmentationDiscardMaterial = new THREE.ShaderMaterial({
      vertexShader: 'void main() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }',
      fragmentShader: 'void main() { discard; }',
      depthTest: false,
      depthWrite: false,
      colorWrite: false,
      toneMapped: false,
    });
  }
  return segmentationDiscardMaterial;
}

function getSegmentationIdMaterial(r, g, b, side = THREE.FrontSide) {
  const key = `${r}:${g}:${b}:${side}`;
  let material = segmentationIdMaterials.get(key);
  if (!material) {
    material = new THREE.ShaderMaterial({
      uniforms: {
        uIdColor: { value: new THREE.Vector4(r / 255, g / 255, b / 255, 1.0) },
      },
      vertexShader: `
        void main() {
          vec4 mvPosition = vec4(position, 1.0);
          #ifdef USE_INSTANCING
            mvPosition = instanceMatrix * mvPosition;
          #endif
          mvPosition = modelViewMatrix * mvPosition;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec4 uIdColor;
        void main() {
          gl_FragColor = uIdColor;
        }
      `,
      side,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      toneMapped: false,
    });
    segmentationIdMaterials.set(key, material);
  }
  return material;
}

function getSpinalSegmentationIdMaterial(r, g, b, side = THREE.FrontSide, spineFlex = null, halfLen = 2.68) {
  const key = `${r}:${g}:${b}:${side}`;
  let material = spinalSegmentationIdMaterials.get(key);
  if (!material) {
    material = new THREE.ShaderMaterial({
      uniforms: {
        uIdColor: { value: new THREE.Vector4(r / 255, g / 255, b / 255, 1.0) },
        uSpineBend: { value: 0.0 },
        uSpineSCurve: { value: 0.0 },
        uSpineArch: { value: 0.0 },
        uAxialStretch: { value: 0.0 },
        uSpineHalfLen: { value: halfLen },
      },
      vertexShader: `
        ${SPINE_DEFORM_GLSL}
        void main() {
          vec4 mvPosition = vec4(position, 1.0);
          #ifdef USE_INSTANCING
            mvPosition = instanceMatrix * mvPosition;
          #endif
          float spinePsi = 0.0;
          mvPosition.xyz = annolidDeformSpinePos(mvPosition.xyz, spinePsi);
          mvPosition = modelViewMatrix * mvPosition;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec4 uIdColor;
        void main() {
          gl_FragColor = uIdColor;
        }
      `,
      side,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      toneMapped: false,
    });
    spinalSegmentationIdMaterials.set(key, material);
  }
  if (spineFlex) {
    material.uniforms.uSpineBend.value = spineFlex.bend || 0.0;
    material.uniforms.uSpineSCurve.value = spineFlex.sCurve || 0.0;
    material.uniforms.uSpineArch.value = spineFlex.arch || 0.0;
    material.uniforms.uAxialStretch.value = spineFlex.stretch || 0.0;
  }
  material.uniforms.uSpineHalfLen.value = halfLen;
  return material;
}

function classifySubjectSegmentationNodes(mouse) {
  const partByNode = new Map();
  const markSubtree = (root, fineId, coarseId) => {
    if (!root) return;
    root.traverse(node => {
      partByNode.set(node, { fineId, coarseId });
    });
  };
  markSubtree(mouse.rig, 6, 2);
  markSubtree(mouse.neckMesh, 5, 2);
  markSubtree(mouse.headGroup, 1, 1);
  markSubtree(mouse.snoutTip, 2, 1);
  markSubtree(mouse.earL, 3, 1);
  markSubtree(mouse.earR, 4, 1);
  markSubtree(mouse.armL?._forearm, 7, 3);
  markSubtree(mouse.armR?._forearm, 8, 3);
  markSubtree(mouse.hindL?._shank, 9, 3);
  markSubtree(mouse.hindR?._shank, 10, 3);
  markSubtree(mouse.tailMesh, 11, 4);
  markSubtree(mouse.tailRoot, 11, 4);
  // Sacral base pelage belongs to the furred torso contour rather than the scaly caudal tail.
  markSubtree(mouse.tailBaseFur, 6, 2);

  const excludeSubtree = root => {
    if (!root) return;
    root.traverse(node => {
      partByNode.delete(node);
    });
  };
  excludeSubtree(mouse.contactShadow);
  excludeSubtree(mouse.trackingGroup);
  excludeSubtree(mouse.padL);
  excludeSubtree(mouse.padR);
  excludeSubtree(mouse.headstageGroup);
  excludeSubtree(mouse.tetherMesh);
  return partByNode;
}

function captureDatasetSegmentationMasks() {
  controls.update();
  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const width = Math.max(1, Math.round(canvas.getBoundingClientRect().width || innerWidth));
  const height = Math.max(1, Math.round(canvas.getBoundingClientRect().height || innerHeight));

  if (
    !segmentationRenderTarget
    || segmentationRenderTarget.width !== width
    || segmentationRenderTarget.height !== height
  ) {
    segmentationRenderTarget?.dispose?.();
    segmentationRenderTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      colorSpace: THREE.NoColorSpace,
      depthBuffer: true,
      stencilBuffer: false,
      samples: 0,
    });
    segmentationPixelBuffer = new Uint8Array(width * height * 4);
  }

  const mouseRoots = new Set(mice.map(m => m.mesh));
  const savedChildVisibility = [];
  for (const child of scene.children) {
    savedChildVisibility.push({ child, visible: child.visible });
    if (!mouseRoots.has(child) && child !== occluderGroup) {
      child.visible = false;
    }
  }

  const savedMeshMaterials = [];
  const bgOccluderMat = getSegmentationIdMaterial(0, 0, 0, THREE.FrontSide);
  if (occluderGroup && occluderGroup.visible) {
    occluderGroup.traverse(node => {
      if (!node.isMesh) return;
      savedMeshMaterials.push({ node, material: node.material });
      node.material = bgOccluderMat;
    });
  }
  const discardMat = getSegmentationDiscardMaterial();
  for (const mouse of mice) {
    if (!mouse.mesh.visible) continue;
    const trackByte = Math.max(1, Math.min(254, Number(mouse.trackId) || 1));
    const partByNode = classifySubjectSegmentationNodes(mouse);
    const spinalNodes = new Set([mouse.bodyMesh]);
    if (mouse.bodyFur) {
      mouse.bodyFur.traverse(n => {
        if (n.isMesh) spinalNodes.add(n);
      });
    }
    const implantNodes = new Set();
    if (mouse.hasHeadstage) {
      if (mouse.headstageGroup) {
        mouse.headstageGroup.traverse(n => { if (n.isMesh) implantNodes.add(n); });
      }
      if (mouse.tetherMesh) implantNodes.add(mouse.tetherMesh);
    }
    mouse.mesh.traverse(node => {
      if (!node.isMesh) return;
      savedMeshMaterials.push({ node, material: node.material });
      if (implantNodes.has(node) && node.visible) {
        node.material = bgOccluderMat;
        return;
      }
      const primaryMat = Array.isArray(node.material) ? node.material[0] : node.material;
      const partInfo = partByNode.get(node);
      const isWhisker = Boolean(
        node.name && (node.name.includes('vibrissa') || node.name.includes('whisker'))
      );
      const isSolidPart = Boolean(
        partInfo
        && !isWhisker
        && node.visible
        && primaryMat
        && primaryMat.colorWrite !== false
        && primaryMat.depthWrite !== false
        && !(primaryMat.transparent && (primaryMat.opacity ?? 1.0) < 0.65)
      );
      if (isSolidPart) {
        const sideMode = primaryMat.side ?? THREE.FrontSide;
        node.material = spinalNodes.has(node)
          ? getSpinalSegmentationIdMaterial(
            trackByte,
            partInfo.fineId,
            partInfo.coarseId,
            sideMode,
            mouse.effectiveSpineFlex || mouse.spineFlex,
            PROP.bodyLength
          )
          : getSegmentationIdMaterial(
            trackByte,
            partInfo.fineId,
            partInfo.coarseId,
            sideMode
          );
      } else {
        node.material = discardMat;
      }
    });
  }

  const prevRenderTarget = renderer.getRenderTarget();
  const prevClearColor = renderer.getClearColor(new THREE.Color());
  const prevClearAlpha = renderer.getClearAlpha();
  const prevToneMapping = renderer.toneMapping;
  const prevAutoClear = renderer.autoClear;
  const prevShadowEnabled = renderer.shadowMap.enabled;
  const prevBackground = scene.background;
  const prevFog = scene.fog;

  try {
    renderer.shadowMap.enabled = false;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.autoClear = true;
    scene.background = null;
    scene.fog = null;
    renderer.setRenderTarget(segmentationRenderTarget);
    renderer.setClearColor(0x000000, 1.0);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(
      segmentationRenderTarget,
      0,
      0,
      width,
      height,
      segmentationPixelBuffer
    );
  } finally {
    renderer.setRenderTarget(prevRenderTarget);
    renderer.setClearColor(prevClearColor, prevClearAlpha);
    renderer.toneMapping = prevToneMapping;
    renderer.autoClear = prevAutoClear;
    renderer.shadowMap.enabled = prevShadowEnabled;
    scene.background = prevBackground;
    scene.fog = prevFog;
    for (const { node, material } of savedMeshMaterials) {
      node.material = material;
    }
    for (const { child, visible } of savedChildVisibility) {
      child.visible = visible;
    }
  }

  const rle = [];
  let curR = -1;
  let curG = -1;
  let curB = -1;
  let runLen = 0;
  for (let y = 0; y < height; y++) {
    const glRow = (height - 1 - y) * width * 4;
    for (let x = 0; x < width; x++) {
      const idx = glRow + x * 4;
      const r = segmentationPixelBuffer[idx];
      const g = segmentationPixelBuffer[idx + 1];
      const b = segmentationPixelBuffer[idx + 2];
      if (r === curR && g === curG && b === curB) {
        runLen += 1;
      } else {
        if (runLen > 0) {
          rle.push(runLen, curR, curG, curB);
        }
        curR = r;
        curG = g;
        curB = b;
        runLen = 1;
      }
    }
  }
  if (runLen > 0) {
    rle.push(runLen, curR, curG, curB);
  }

  return {
    frame_id: datasetCurrentFrameIndex,
    width,
    height,
    rle,
    fine_parts: SEGMENTATION_FINE_PARTS,
    coarse_parts: SEGMENTATION_COARSE_PARTS,
    subjects: mice.map(m => ({
      track_id: m.trackId,
      strain: m.strainName,
      label: `mouse_${m.trackId}`,
    })),
  };
}

if (DATASET_CAPTURE) {
  advanceDatasetFrame(DATASET_FRAME_INDEX);
  window.annolidPoseDataset = Object.freeze({
    getFrame: buildDatasetFramePayload,
    setFrame: targetFrame => {
      advanceDatasetFrame(targetFrame);
      postFX.render(scene, camera, accumulatedTime, controls.target);
      canvas.dataset.datasetReady = 'true';
      return buildDatasetFramePayload();
    },
    captureSegmentationMasks: captureDatasetSegmentationMasks,
  });
}

const visualQaPixel = new Uint8Array(4);
function updateVisualQaPixelTelemetry() {
  if (!CONTACT_QA && !REARING_QA && !GROOMING_QA && !ANATOMY_QA) return;
  const gl = renderer.getContext();
  const width = gl.drawingBufferWidth;
  const height = gl.drawingBufferHeight;
  if (width < 2 || height < 2) return;
  const luminances = [];
  for (let row = 1; row <= 5; row++) {
    for (let column = 1; column <= 5; column++) {
      const x = Math.min(width - 1, Math.floor(width * column / 6));
      const y = Math.min(height - 1, Math.floor(height * row / 6));
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, visualQaPixel);
      luminances.push(
        visualQaPixel[0] * 0.2126
          + visualQaPixel[1] * 0.7152
          + visualQaPixel[2] * 0.0722
      );
    }
  }
  const mean = luminances.reduce((total, value) => total + value, 0)
    / luminances.length;
  const variance = luminances.reduce(
    (total, value) => total + Math.pow(value - mean, 2),
    0
  ) / luminances.length;
  canvas.dataset.pixelLuminanceMean = mean.toFixed(2);
  canvas.dataset.pixelLuminanceStd = Math.sqrt(variance).toFixed(2);
  canvas.dataset.pixelLuminanceRange = (
    Math.max(...luminances) - Math.min(...luminances)
  ).toFixed(2);
}

let animationFrameId = 0;
let sceneDisposed = false;

function disposeSceneResources() {
  if (sceneDisposed) return;
  sceneDisposed = true;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = 0;
  }
  clearMice();
  postFX.dispose();
  segmentationRenderTarget?.dispose?.();
  segmentationRenderTarget = null;
  segmentationPixelBuffer = null;
  segmentationDiscardMaterial?.dispose?.();
  segmentationDiscardMaterial = null;
  for (const material of segmentationIdMaterials.values()) {
    material?.dispose?.();
  }
  segmentationIdMaterials.clear();
  for (const material of spinalSegmentationIdMaterials.values()) {
    material?.dispose?.();
  }
  spinalSegmentationIdMaterials.clear();
  scene.environment?.dispose?.();
  scene.environment = null;
  keyLight.shadow.map?.dispose?.();
  keyLight.shadow.map = null;
  const remainingResources = new Set();
  scene.traverse(node => {
    if (node.isInstancedMesh) node.dispose();
    if (node.geometry) remainingResources.add(node.geometry);
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) {
      if (!material) continue;
      remainingResources.add(material);
      for (const value of Object.values(material)) {
        if (value?.isTexture) remainingResources.add(value);
      }
    }
  });
  for (const resource of remainingResources) {
    resource?.dispose?.();
  }
  disposeSharedResources();
  controls.dispose();
  gui.destroy();
}
window.addEventListener('pagehide', disposeSceneResources, { once: true });

function animate() {
  if (sceneDisposed) return;
  if (!DATASET_CAPTURE) animationFrameId = requestAnimationFrame(animate);
  if (!DATASET_CAPTURE) {
    const frameDt = Math.min(clock.getDelta(), 0.2);
    const simulationStart = performance.now();
    simulationClock.advance(frameDt * guiParams.playbackSpeed, (time, dt) => {
      accumulatedTime = time;
      mice.forEach((m, i) => {
        m.update(time, mice, i === 0 ? hudElements : null, dt);
      });
      resolveAllMouseContacts(mice, dt);
    }, guiParams.isPaused || document.hidden);
    simulationDurationMs = performance.now() - simulationStart;
    followCamera(frameDt);
    // Contact solving runs every physics step; expensive DOM diagnostics run at 5 Hz.
    if (accumulatedTime - lastTelemetryTime >= 0.2) {
      updateCollisionTelemetry();
      lastTelemetryTime = accumulatedTime;
    }
  }
  controls.update();
  // Render via post-processing pipeline (passes through if disabled)
  const renderStart = performance.now();
  postFX.render(scene, camera, accumulatedTime, controls.target);
  renderDurationMs = performance.now() - renderStart;
  updateVisualQaPixelTelemetry();
  if (DATASET_CAPTURE) canvas.dataset.datasetReady = 'true';
  canvas.dataset.sceneReady = 'true';
  document.getElementById('scene-status').textContent = '';
}
animate();
