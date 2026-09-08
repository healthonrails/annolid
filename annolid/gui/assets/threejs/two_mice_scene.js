import {
  THREE,
  OrbitControls,
  RoomEnvironment,
  GUI
} from './annolid_threejs_runtime.js';

import { AnnolidShaders } from './annolid_shaders.js';
import { createMouseModel } from './two_mice_model.js';
import { createSeededRandom, FixedStepClock, wallNormalForce, STANCE_CONTACT_LIMITS } from './two_mice_simulation.js';

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
const ANATOMY_QA_VIEWS = new Set(['ears', 'ears-front', 'torso-side', 'torso-front']);
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
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.94;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(DATASET_CAPTURE ? 1 : Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd8dad9);
scene.fog = new THREE.FogExp2(0xd8dad9, 0.0045);

const camera = new THREE.PerspectiveCamera(34, innerWidth / innerHeight, 0.03, 500);
const cameraBasePosition = new THREE.Vector3(8.4, 5.4, 22.5);
const cameraTarget = new THREE.Vector3(0, 1.05, 0);
camera.position.copy(cameraBasePosition);

const controls = new OrbitControls(camera, canvas);
controls.enabled = !DATASET_CAPTURE;
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.copy(cameraTarget);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
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
  FORELIMB_NEUTRAL,
  tailRoughnessMap,
  earNormalMap,
  earRoughnessMap,
  earAlbedoMap,
  noseAlbedoMap,
  noseNormalMap,
  noseRoughnessMap,
  labFloorAlbedoMap,
  labFloorRoughnessMap,
  labFloorNormalMap,
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
  getCervicalMantleProfile,
  bodyGeo,
  headGeo,
  neckGeo,
  createForelimbChain,
  createHindlimbChain,
  createAnatomicalWhiskers,
  FUR_DETAIL,
  generateMultiLayerFur,
  buildMouseEar,
  disposeSubject,
} = createMouseModel(THREE, annolidShaders, appearanceRandom);

const ENCLOSURE_W = 36, ENCLOSURE_H = 9, ENCLOSURE_D = 26, ENCLOSURE_PAD = 1.6;
const ENCLOSURE_HALF_X = ENCLOSURE_W * 0.5 - ENCLOSURE_PAD;
const ENCLOSURE_HALF_Z = ENCLOSURE_D * 0.5 - ENCLOSURE_PAD;

// Contact shadow (soft AO disc under each mouse)
function createContactShadow() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const ctx = c.getContext('2d');
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(0,0,0,0.35)');
  grad.addColorStop(0.5, 'rgba(0,0,0,0.15)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad; ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: true });
  const geo = new THREE.PlaneGeometry(5, 4);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.02;
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
  const bodyScale = THREE.MathUtils.clamp(mouse.guiParams.bodyLength ?? 1.0, 0.65, 1.6);
  const snoutScale = THREE.MathUtils.clamp(mouse.guiParams.snoutLength ?? 1.0, 0.65, 1.6);
  const widthScale = THREE.MathUtils.clamp(mouse.guiParams.chonkiness ?? 1.0, 0.65, 1.6);
  return MOUSE_COLLISION_DISC_LAYOUT.map(({ offset, radius }) => {
    const headExtension = offset > 2.4 ? (snoutScale - 1.0) * 0.55 : 0.0;
    const scaledOffset = offset * bodyScale + headExtension;
    return {
      x: mouse.mesh.position.x + heading.x * scaledOffset,
      z: mouse.mesh.position.z + heading.y * scaledOffset,
      radius: radius * widthScale,
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

function createDynamicTailTubeGeometry(segmentCount, radialSegments = 12) {
  const ringCount = segmentCount + 1;
  const ringStride = radialSegments + 1;
  const sideVertexCount = ringCount * ringStride;
  const capStride = radialSegments + 1;
  const startCapOffset = sideVertexCount;
  const endCapOffset = startCapOffset + capStride;
  const vertexCount = sideVertexCount + capStride * 2;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
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
  positionAttribute.setUsage(THREE.DynamicDrawUsage);
  normalAttribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', positionAttribute);
  geometry.setAttribute('normal', normalAttribute);
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

    const radius = tailSurfaceRadiusAt(ring / (ringCount - 1), thickness);
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
    for (let side = 0; side < radialSegments; side++) {
      const sourceOffset = (cap.ring * ringStride + side) * 3;
      const targetOffset = (cap.offset + side) * 3;
      positions[targetOffset] = positions[sourceOffset];
      positions[targetOffset + 1] = positions[sourceOffset + 1];
      positions[targetOffset + 2] = positions[sourceOffset + 2];
      vertexNormals[targetOffset] = vertexNormals[centerOffset];
      vertexNormals[targetOffset + 1] = vertexNormals[centerOffset + 1];
      vertexNormals[targetOffset + 2] = vertexNormals[centerOffset + 2];
    }
  }

  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.normal.needsUpdate = true;
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
  for (const mouse of subjects) mouse.updateTailGeometry(subjects);
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
    this.tailSegLen = 0.22;
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
    this.behaviorTimer = 0.8 + random() * 2.4;
    this.pauseTimer = random() * 0.9;
    this.socialContact = false;
    this.socialDistance = Infinity;
    this.socialTarget = new THREE.Vector2(0, 1);

    const furColor = params.furColor;
    const skinColor = params.skinColor;
    const density = FUR_DETAIL[params.furDetailLevel || 'Med'] * (params.furDensity || 0.85);
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

    // Flesh material
    this.fleshMat = fleshMatBase.clone();
    this.fleshMat.color.set(skinColor);
    const skinHSL = {}; this.fleshMat.color.getHSL(skinHSL);
    this.fleshMat.attenuationColor.setHSL(skinHSL.h, skinHSL.s, skinHSL.l * 0.5);

    // ═══ APPLY GLSL SHADERS ═══
    if (params.enableSSS !== false) {
      annolidShaders.applySSSShader(this.fleshMat, {
        scatterColor: [this.fleshMat.color.r * 0.9, this.fleshMat.color.g * 0.9, this.fleshMat.color.b * 0.9],
        thickness: params.sssThickness ?? 0.45,
        distortion: 0.2,
        wrap: params.sssWrap ?? 0.5,
        backlight: params.sssBacklight ?? 0.6
      });
    }
    if (params.enableContactAO !== false) {
      annolidShaders.applyContactAO(this.fleshMat, {
        groundY: 0.0,
        radius: Math.min(0.52, params.aoRadius ?? 0.52),
        intensity: Math.min(0.18, params.aoIntensity ?? 0.18),
        color: [0.02, 0.01, 0.0]
      });
    }
    if (params.enableMicroDetail !== false) {
      annolidShaders.applyMicroDetail(this.fleshMat, {
        scale: params.microScale ?? 40.0,
        strength: params.microStrength ?? 0.06
      });
    }
    this.surfaceMat = createCoatMaterial(furColor, skinColor, density, params);
    if (params.enableContactAO !== false) {
      annolidShaders.applyContactAO(this.surfaceMat, {
        groundY: 0.0,
        radius: Math.min(0.72, params.aoRadius ?? 0.72),
        intensity: Math.min(0.18, (params.aoIntensity ?? 0.18) * 0.9),
        color: [0.018, 0.014, 0.010]
      });
    }

    // --- RIG HIERARCHY ---
    this.rig = new THREE.Group();
    this.mesh.add(this.rig);

    // BODY
    this.bodyMesh = new THREE.Mesh(bodyGeo, this.surfaceMat);
    this.bodyMesh.scale.set(chonk, chonk, params.bodyLength || 1.0);
    this.bodyMesh.castShadow = true;
    this.rig.add(this.bodyMesh);
    this.bodyFur = generateMultiLayerFur(deformBody, density, furColor, false);
    this.bodyFur.name = 'body-fur-coat';
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
    const buildCervicalFurLayer = ({
      baseCount,
      length,
      radius,
      normalWeight,
      caudalFlow,
      roughness,
      opacity,
      name,
    }) => {
      const count = Math.floor(baseCount * density);
      if (count <= 0) return null;
      const geometry = new THREE.ConeGeometry(radius, length, 3);
      geometry.translate(0, length * 0.5, 0);
      const material = enhanceFurMaterial(
        new THREE.MeshPhysicalMaterial({
          color: new THREE.Color(furColor),
          roughness,
          transparent: true,
          opacity,
        }),
        furColor,
        normalWeight > 0.5
      );
      const layer = new THREE.InstancedMesh(geometry, material, count);
      layer.name = name;
      const nkDummy = new THREE.Object3D();
      const nkNrm = new THREE.Vector3();
      const nkFlow = new THREE.Vector3(0, -0.045, -caudalFlow);
      const localUp = new THREE.Vector3(0, 1, 0);
      for (let i = 0; i < count; i++) {
        const t = appearanceRandom();
        const angle = appearanceRandom() * Math.PI * 2;
        const profile = getCervicalMantleProfile(t);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const anteriorFade = 1.0 - THREE.MathUtils.smoothstep(t, 0.35, 0.82);
        const anteriorInset = 0.68 + anteriorFade * 0.32;
        nkDummy.position.set(
          cos * profile.radiusX * anteriorInset,
          profile.centerY + sin * profile.radiusY * anteriorInset,
          profile.z
        );
        nkNrm.set(cos / profile.radiusX, sin / profile.radiusY, 0).normalize();
        nkNrm.multiplyScalar(normalWeight).add(nkFlow).normalize();
        nkDummy.quaternion.setFromUnitVectors(localUp, nkNrm);
        const baseScale = 0.72 + appearanceRandom() * (normalWeight > 0.5 ? 0.26 : 0.36);
        const s = baseScale * (0.18 + anteriorFade * 0.82);
        nkDummy.scale.set(s, s, s); nkDummy.updateMatrix();
        layer.setMatrixAt(i, nkDummy.matrix);
      }
      return layer;
    };
    const cervicalUndercoat = buildCervicalFurLayer({
      baseCount: 11000,
      length: 0.036,
      radius: 0.0044,
      normalWeight: 0.82,
      caudalFlow: 0.12,
      roughness: 0.94,
      opacity: 0.80,
      name: 'cervical-undercoat',
    });
    const cervicalGuardHairs = buildCervicalFurLayer({
      baseCount: 7000,
      length: 0.078,
      radius: 0.0038,
      normalWeight: 0.24,
      caudalFlow: 0.88,
      roughness: 0.80,
      opacity: 0.84,
      name: 'cervical-guard-hairs',
    });
    this.cervicalUndercoat = cervicalUndercoat;
    this.cervicalGuardHairs = cervicalGuardHairs;
    if (cervicalUndercoat) neckMesh.add(cervicalUndercoat);
    if (cervicalGuardHairs) neckMesh.add(cervicalGuardHairs);

    // BELLY FUR ANCHOR (no visible geometry; avoids hard side cut-lines from above)
    const bellyFurAnchor = new THREE.Group();
    bellyFurAnchor.position.set(0, -0.43, PROP.bodyLength * 0.56);
    bellyFurAnchor.rotation.x = Math.PI / 2;
    bellyFurAnchor.scale.set(0.82, 0.76, 0.58);
    this.rig.add(bellyFurAnchor);
    // Add short ventral + flank fur to soften the dorsal/ventral transition.
    const bellyFurCount = Math.floor(7000 * density);
    if (bellyFurCount > 0) {
      const bfGeo = new THREE.ConeGeometry(0.006, 0.09, 3);
      bfGeo.translate(0, 0.045, 0);
      const bfMat = enhanceFurMaterial(
        new THREE.MeshPhysicalMaterial({ color: new THREE.Color(furColor), roughness: 0.80, transparent: true, opacity: 0.78 }),
        furColor
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
    this.headFur = generateMultiLayerFur(deformHead, density, furColor, true);
    this.headFur.name = 'head-fur-coat';
    this.head.add(this.headFur);

    // SNOUT — Anatomically correct rhinarium
    const noseMat = this.fleshMat.clone();
    noseMat.color.set(strain.noseColor ?? skinColor);
    noseMat.map = noseAlbedoMap;
    noseMat.normalMap = noseNormalMap;
    noseMat.normalScale = new THREE.Vector2(0.18, 0.18);
    noseMat.roughnessMap = noseRoughnessMap;
    noseMat.roughness = 0.32;
    noseMat.transmission = 0.14;
    noseMat.thickness = 0.12;
    noseMat.clearcoat = 0.48;
    noseMat.clearcoatRoughness = 0.18;
    // Triangular rhinarium with philtrum groove and alar bulges
    const rhinariumGeo = new THREE.SphereGeometry(0.14, 28, 28);
    deform(rhinariumGeo, (v) => {
      const yN = (v.y + 1) / 2;
      v.x *= 1.10 - yN * 0.24;   // wider bottom = inverted triangle
      v.y *= 0.66;                // flatten vertically
      v.z *= 0.68;
      // Philtrum groove at center front
      if (Math.abs(v.x) < 0.035 && v.z > 0) v.z -= (0.035 - Math.abs(v.x)) * 0.7;
      // Alar bulge around nostril positions
      const nDist = Math.sqrt(Math.pow(Math.abs(v.x) - 0.04, 2) + Math.pow(v.y + 0.02, 2));
      if (nDist < 0.055 && v.z > 0.01) v.z += (0.055 - nDist) * 0.32;
    });
    this.snoutTip = new THREE.Mesh(rhinariumGeo, noseMat);
    this.snoutTip.scale.set(1.06, 0.76, 0.72);
    this.snoutTip.position.set(this.asymmetry.noseYaw * 0.65, -0.158, PROP.snoutLength * 0.99);
    this.head.add(this.snoutTip);
    // Moist tip overlay — wet specular on nose center
    const moistMat = noseMat.clone();
    moistMat.roughness = 0.10; moistMat.clearcoat = 0.88;
    moistMat.clearcoatRoughness = 0.04; moistMat.transmission = 0.05;
    const moistTip = new THREE.Mesh(new THREE.SphereGeometry(0.06, 14, 14), moistMat);
    moistTip.scale.set(1.0, 0.7, 0.55);
    moistTip.position.set(0, 0.012, 0.048);
    this.snoutTip.add(moistTip);
    // Nostrils — forward-facing elongated slits
    const nostrilMat = new THREE.MeshPhysicalMaterial({
      color: 0x2a1210, roughness: 0.82, clearcoat: 0.15, clearcoatRoughness: 0.4
    });
    const nSlitGeo = new THREE.CapsuleGeometry(0.009, 0.016, 8, 8);
    const nL = new THREE.Mesh(nSlitGeo, nostrilMat);
    nL.scale.set(0.80, 1.0, 0.60); nL.rotation.z = 0.35;
    nL.position.set(0.033, -0.008, 0.074); this.snoutTip.add(nL);
    const nR = new THREE.Mesh(nSlitGeo, nostrilMat);
    nR.scale.set(0.80, 1.0, 0.60); nR.rotation.z = -0.35;
    nR.position.set(-0.033, -0.008, 0.074); this.snoutTip.add(nR);
    // Nostril rim moisture rings
    const rimMoistMat = noseMat.clone();
    rimMoistMat.roughness = 0.18; rimMoistMat.clearcoat = 0.65;
    for (const s of [1, -1]) {
      const rim = new THREE.Mesh(new THREE.TorusGeometry(0.013, 0.0035, 8, 12), rimMoistMat);
      rim.position.set(s * 0.033, -0.006, 0.070);
      rim.rotation.y = Math.PI / 2; rim.rotation.z = s * 0.35;
      this.snoutTip.add(rim);
    }
    // Philtrum groove
    const philtrumMat = new THREE.MeshPhysicalMaterial({
      color: 0x3a1d1c, roughness: 0.65, transparent: true, opacity: 0.55
    });
    const philtrum = new THREE.Mesh(new THREE.CapsuleGeometry(0.005, 0.038, 4, 8), philtrumMat);
    philtrum.position.set(0, -0.020, 0.068); this.snoutTip.add(philtrum);
    // Alar bulge accents
    for (const s of [1, -1]) {
      const alar = new THREE.Mesh(new THREE.SphereGeometry(0.020, 12, 12), noseMat);
      alar.scale.set(0.9, 0.60, 0.65);
      alar.position.set(s * 0.040, -0.003, 0.058);
      this.snoutTip.add(alar);
    }

    // MOUTH + INCISORS
    const mouthMat = new THREE.MeshPhysicalMaterial({
      color: 0x80504d, roughness: 0.72, metalness: 0.0, clearcoat: 0.05, transparent: true, opacity: 0.72
    });
    this.mouth = new THREE.Mesh(new THREE.CapsuleGeometry(0.012, 0.075, 8, 10), mouthMat);
    this.mouth.rotation.z = Math.PI / 2;
    this.mouth.position.set(0, -0.235, PROP.snoutLength * 0.988);
    this.head.add(this.mouth);
    const lowerLip = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 12), this.fleshMat);
    lowerLip.scale.set(1.25, 0.46, 0.66);
    lowerLip.position.set(0, -0.248, PROP.snoutLength * 0.968);
    lowerLip.castShadow = true;
    this.head.add(lowerLip);

    // WHISKERS
    const wColor = skinHSL.l > 0.5 ? 0x958f8c : 0xd0c7c1;
    const whiskerOpacity = skinHSL.l > 0.5 ? 0.46 : 0.56;
    this.padL = new THREE.Group(); this.padL.position.set(0.22, -0.07, 0.74); this.head.add(this.padL);
    this.padR = new THREE.Group(); this.padR.position.set(-0.22, -0.07, 0.74); this.head.add(this.padR);
    this.padL.add(createAnatomicalWhiskers(wColor, true, whiskerOpacity));
    this.padR.add(createAnatomicalWhiskers(wColor, false, whiskerOpacity));

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

    // Store ear refs for micro-motion
    this.earL = earL; this.earR = earR;
    this.earMicro.baseRotL = earL.rotation.clone();
    this.earMicro.baseRotR = earR.rotation.clone();

    // EYES (lateral placement with asymmetry, two-layer)
    const subjectEyeMat = eyeMat.clone();
    subjectEyeMat.color.set(strain.eyeColor ?? '#070606');
    subjectEyeMat.attenuationColor = subjectEyeMat.color.clone();
    const subjectCorneaMat = corneaMat.clone();
    subjectCorneaMat.attenuationColor = subjectEyeMat.color.clone().offsetHSL(0, 0, 0.08);
    const eyeRimMat = new THREE.MeshBasicMaterial({
      color: strain.eyeRimColor ?? 0x2a1b1b,
      transparent: true,
      opacity: 0.34,
    });
    const eyeL = new THREE.Mesh(new THREE.SphereGeometry(PROP.eyeSize, 20, 20), subjectEyeMat);
    eyeL.scale.set(1.0, 0.95, PROP.eyeProtrusion);
    eyeL.position.set(PROP.eyeSpacing + this.asymmetry.eyeOffsetL, 0.15, PROP.eyeYaw);
    eyeL.castShadow = true; this.head.add(eyeL);
    const cL = new THREE.Mesh(new THREE.SphereGeometry(PROP.eyeSize + 0.005, 20, 20), subjectCorneaMat);
    cL.scale.set(1.0, 0.95, PROP.eyeProtrusion + 0.05);
    cL.position.copy(eyeL.position); this.head.add(cL);
    // Tear line highlight
    const tearL = new THREE.Mesh(
      new THREE.TorusGeometry(PROP.eyeSize * 0.85, 0.008, 6, 16, Math.PI * 0.6),
      eyeRimMat
    );
    tearL.position.copy(eyeL.position); tearL.position.y -= 0.02;
    tearL.rotation.y = Math.PI / 2; this.head.add(tearL);

    const eyeR = new THREE.Mesh(new THREE.SphereGeometry(PROP.eyeSize, 20, 20), subjectEyeMat);
    eyeR.scale.set(1.0, 0.95, PROP.eyeProtrusion);
    eyeR.position.set(-PROP.eyeSpacing + this.asymmetry.eyeOffsetR, 0.15, PROP.eyeYaw);
    eyeR.castShadow = true; this.head.add(eyeR);
    const cR = new THREE.Mesh(new THREE.SphereGeometry(PROP.eyeSize + 0.005, 20, 20), subjectCorneaMat);
    cR.scale.set(1.0, 0.95, PROP.eyeProtrusion + 0.05);
    cR.position.copy(eyeR.position); this.head.add(cR);
    const tearR = new THREE.Mesh(
      new THREE.TorusGeometry(PROP.eyeSize * 0.85, 0.008, 6, 16, Math.PI * 0.6),
      eyeRimMat
    );
    tearR.position.copy(eyeR.position); tearR.position.y -= 0.02;
    tearR.rotation.y = -Math.PI / 2; this.head.add(tearR);
    this.eyeL = eyeL;
    this.eyeR = eyeR;
    this.corneaL = cL;
    this.corneaR = cR;

    // ═══ LIMBS (anatomically correct chains) ═══
    const legLengthScale = params.legLength ?? 1.0;
    const legThicknessScale = params.legThickness ?? 1.0;
    // Hindlimbs
    this.hindL = createHindlimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'L', legLengthScale, legThicknessScale);
    this.rig.add(this.hindL);
    this.hindR = createHindlimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'R', legLengthScale, legThicknessScale);
    this.rig.add(this.hindR);

    // Forelimbs
    this.armL = createForelimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'L', legLengthScale, legThicknessScale);
    this.rig.add(this.armL);
    this.armR = createForelimbChain(this.surfaceMat, this.fleshMat, strain, density, furColor, 'R', legLengthScale, legThicknessScale);
    this.rig.add(this.armR);

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
      phaseOffset,
      isFore,
      side,
      maximumReach = null
    ) => ({
      paw,
      contact: paw._groundContact,
      joints,
      phaseOffset,
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
      makeFootPlant(this.hindL._paw, [this.hindL._ankle, this.shankL, this.hindL], 0, false, 1),
      makeFootPlant(this.hindR._paw, [this.hindR._ankle, this.shankR, this.hindR], Math.PI, false, -1),
      makeFootPlant(
        this.armL._paw,
        [this.armL._wrist, this.armL._forearm, this.upperArmL],
        Math.PI,
        true,
        1,
        this.armL._maximumReach
      ),
      makeFootPlant(
        this.armR._paw,
        [this.armR._wrist, this.armR._forearm, this.upperArmR],
        0,
        true,
        -1,
        this.armR._maximumReach
      ),
    ];
    this.contactFrame = {
      legCycle: 0,
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

    // TAIL
    this.tailCount = 45;
    const tailMat = this.fleshMat.clone();
    tailMat.roughness = 0.6; tailMat.transmission = 0.25;
    tailMat.thickness = 0.4; tailMat.clearcoat = 0.15;
    tailMat.roughnessMap = tailRoughnessMap;
    const tGeo = createDynamicTailTubeGeometry(this.tailCount, 12);
    this.tailMesh = new THREE.Mesh(tGeo, tailMat);
    this.tailMesh.name = 'continuous-tail-tube';
    this.tailMesh.castShadow = true;
    this.tailMesh.receiveShadow = true;
    this.tailMesh.frustumCulled = false;
    this.mesh.add(this.tailMesh);
    // Root bridge to blend tail emergence into the rump.
    const tailRootGeo = new THREE.CylinderGeometry(0.11, 0.13, 0.30, 12);
    tailRootGeo.translate(0, 0.15, 0);
    tailRootGeo.rotateX(Math.PI / 2);
    this.tailRoot = new THREE.Mesh(tailRootGeo, tailMat);
    this.tailRoot.castShadow = true;
    this.tailRoot.scale.set(1.0, 0.72, 1.18);
    this.mesh.add(this.tailRoot);

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
      if (parent) {
        sphere.position.copy(localPos); parent.add(sphere);
        this.poseNodes[id] = sphere;
      } else {
        this.trackingGroup.add(sphere);
        this.poseNodes[id] = sphere;
      }
    };

    // Tracking nodes at anatomical landmarks
    createTrackingNode('snout', this.head, new THREE.Vector3(0, -0.15, PROP.snoutLength * 1.04));
    createTrackingNode('lear', this.head, new THREE.Vector3(0.55, 0.45, -0.15));
    createTrackingNode('rear', this.head, new THREE.Vector3(-0.55, 0.45, -0.15));
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
          side * THREE.MathUtils.lerp(0.11, 0.27, stroke),
          THREE.MathUtils.lerp(-0.28, 0.07, stroke) - motion.pawLick * 0.035,
          THREE.MathUtils.lerp(0.47, 0.43, stroke) + motion.pawLick * 0.025
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
      const stancePhase = Math.sin(this.contactFrame.legCycle + foot.phaseOffset);
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
      const inStance = mayTouchGround && (
        !this.contactFrame.isMoving || stancePhase < 0.08
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

  updateTailGeometry(otherMice = []) {
    const { lateralAccel, turnRate, forwardAccel } = this.contactFrame;
    const tailBase = new THREE.Vector3(0, -0.16, -PROP.bodyLength * 0.90);
    this.rig.updateMatrix();
    tailBase.applyMatrix4(this.rig.matrix);
    const tailCurve = [];
    const tailLenMul = this.guiParams.tailLength;
    const segLen = this.tailSegLen * tailLenMul;
    for (let i = 0; i <= this.tailCount; i++) {
      const p = i / this.tailCount;
      const lag = p * p;
      const transmit = Math.pow(p, 1.35);
      const midCurve = p * (1.0 - p);
      const bodyForceX = (
        this.bodyDyn.roll * 0.34 + lateralAccel * 0.010 - turnRate * 0.018
      ) * transmit;
      const bodyForceY = (
        this.bodyDyn.pitch * 0.38 - forwardAccel * 0.008
      ) * transmit;
      const waveX = this.tailDyn.sway * lag + bodyForceX
        + midCurve * 0.18 * Math.sign(this.tailDyn.sway || bodyForceX || 1);
      let waveY = -i * PROP.tailSag * p + this.tailDyn.lift * lag
        + bodyForceY + midCurve * 0.08;
      waveY = Math.max(waveY, -1.3);
      tailCurve.push(new THREE.Vector3(waveX, waveY, -i * segLen));
    }

    const offset = tailBase.clone().sub(tailCurve[0]);
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
    this.mesh.updateMatrixWorld(true);

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
      tailCurve[i].add(offset);
      clampPointToArena(tailCurve[i], i);
    }
    const fixedRoot = tailCurve[0].clone();
    const bendTarget = new THREE.Vector3();
    for (let pass = 0; pass < 10; pass++) {
      let corrected = false;
      tailCurve[0].copy(fixedRoot);

      // A light bend constraint removes contact-induced kinks before the
      // inextensible centerline and collision constraints are projected.
      for (let parity = 0; parity < 2; parity++) {
        for (let i = 1 + parity; i < this.tailCount; i += 2) {
          bendTarget.copy(tailCurve[i - 1]).add(tailCurve[i + 1]).multiplyScalar(0.5);
          tailCurve[i].lerp(bendTarget, pass < 4 ? 0.16 : 0.10);
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
    const isTailVisible = this.tailMesh.visible;
    const tailMidIndex = Math.max(1, Math.floor(this.tailCount * 0.4));
    this.poseNodes.tailmid.position.copy(tailCurve[tailMidIndex]);
    this.poseNodes.tailmid.visible = guiParams.showTrackingKeypoints && isTailVisible;
    this.poseNodes.tailend.position.copy(tailCurve[this.tailCount]);
    this.poseNodes.tailend.visible = guiParams.showTrackingKeypoints && isTailVisible;
    if (this.tailRoot && tailCurve.length > 4) {
      const thickMul = this.guiParams.tailThickness;
      const dir = tailCurve[3].clone().sub(tailCurve[0]).normalize();
      this.tailRoot.position.copy(tailCurve[0]).addScaledVector(dir, segLen * 0.25);
      this.tailRoot.lookAt(tailCurve[3]);
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
      if (this.behaviorTimer <= 0) {
        if (random() < 0.52) {
          this.pauseTimer = 0.35 + random() * 1.35;
        }
        this.behaviorTimer = 1.4 + random() * 4.2;
        if (random() < 0.22) this.wallFollowDir *= -1;
      }
    }
    const behaviorPause = this.pauseTimer > 0;
    const rearingActive = this.rearing.mode !== 'none';
    const groomingActive = this.grooming.mode !== 'none';
    const isPausing = behaviorPause || rearingActive || groomingActive;

    // --- FORCE ACCUMULATION ---
    this.force.set(0, 0);
    let burst = Math.sin(t * 1.5 + this.burstPhase);
    burst = 0.7 + 0.3 * Math.max(0, burst);
    const driveStrength = burst * this.guiParams.wanderForce * (
      rearingActive || groomingActive ? 0.04 : (isPausing ? 0.18 : 1.0)
    );

    // Thigmotaxis
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
    }
    const tX = nearest.tangentX * this.wallFollowDir;
    const tZ = nearest.tangentZ * this.wallFollowDir;
    const nwF = Math.max(0, 1.0 - wallDist / 5.0);
    this.force.x += tX * driveStrength * (0.5 + nwF * 1.5);
    this.force.y += tZ * driveStrength * (0.5 + nwF * 1.5);

    if (!guiParams.isPaused && !groomingActive) {
      this.force.x += (Math.sin(t * 2.3) * 0.3 + (random() - 0.5) * 0.2) * driveStrength * 0.3;
      this.force.y += (Math.cos(t * 1.7) * 0.3 + (random() - 0.5) * 0.2) * driveStrength * 0.3;
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

    // Mouse-to-mouse interaction uses oriented body clearance so social
    // investigation stops before either elongated silhouette penetrates.
    const SOCIAL_RADIUS = 8.2;
    let isColliding = false;
    let nearestSocialDist = Infinity;
    this.socialContact = false;
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
        const toOtherX = -dx / dist;
        const toOtherZ = -dz / dist;
        if (dist < nearestSocialDist) {
          nearestSocialDist = dist;
          this.socialTarget.set(toOtherX, toOtherZ);
        }
        if (dist < SOCIAL_RADIUS) {
          const socialBand = THREE.MathUtils.clamp((SOCIAL_RADIUS - dist) / 4.6, 0, 1);
          const facing = this.heading.x * toOtherX + this.heading.y * toOtherZ;
          if (bodyClearance > 0.75) {
            const curiosity = (0.22 + 0.32 * Math.max(0, Math.sin(time * 0.78 + this.timeOffset * 0.13))) * socialBand;
            const faceGate = facing > -0.55 ? 1.0 : 0.35;
            this.force.x += toOtherX * driveStrength * curiosity * faceGate;
            this.force.y += toOtherZ * driveStrength * curiosity * faceGate;
          }
          if (bodyClearance < 1.35) {
            const side = this.dir;
            this.force.x += -toOtherZ * side * socialBand * driveStrength * 0.14;
            this.force.y += toOtherX * side * socialBand * driveStrength * 0.14;
            this.velocity.multiplyScalar(bodyClearance < 0.45 ? 0.965 : 0.988);
            this.socialContact = true;
          }
        }
      }
    }
    this.socialDistance = nearestSocialDist;

    const preSpeed = this.velocity.length();

    // Cruise assist: keep mice from decaying to full stop after startup.
    if (!guiParams.isPaused && !isPausing) {
      const cruiseBase = this.socialContact ? 0.16 : 0.30;
      const cruiseTarget = Math.max(this.guiParams.maxSpeed * cruiseBase, this.socialContact ? 0.75 : 1.8);
      if (preSpeed < cruiseTarget) {
        const deficit = (cruiseTarget - preSpeed) / Math.max(cruiseTarget, 1e-3);
        const cruiseDir = this.heading.lengthSq() > 1e-6
          ? this.heading.clone().normalize()
          : new THREE.Vector2(Math.cos(t), Math.sin(t)).normalize();
        const cruiseBoost = (this.guiParams.wanderForce * 5.0 + 10.0) * deficit;
        this.force.x += cruiseDir.x * cruiseBoost;
        this.force.y += cruiseDir.y * cruiseBoost;
      }
    }

    // Drag (lighter at very low speed so movement does not stall).
    const dragC = this.guiParams.dragCoeff * (isPausing ? 3.0 : (preSpeed < 1.0 ? 0.45 : 1.0));
    this.force.x -= this.velocity.x * dragC;
    this.force.y -= this.velocity.y * dragC;

    // Low-speed rescue: inject motion if the mouse lingers too long.
    if (!guiParams.isPaused && !isPausing && !this.socialContact) {
      if (preSpeed < 0.8) this.lowSpeedTimer += dt;
      else this.lowSpeedTimer = Math.max(0, this.lowSpeedTimer - dt * 0.5);
      if (this.lowSpeedTimer > 1.0) {
        const rescueDir = new THREE.Vector2(
          nearest.tangentX * this.wallFollowDir + nearest.normalX * 0.6,
          nearest.tangentZ * this.wallFollowDir + nearest.normalZ * 0.6
        );
        if (rescueDir.lengthSq() < 1e-6) {
          if (this.heading.lengthSq() > 1e-6) rescueDir.copy(this.heading);
          else rescueDir.set(Math.cos(t), Math.sin(t));
        }
        rescueDir.normalize();
        const kickMag = 32.0;
        this.force.x += rescueDir.x * kickMag;
        this.force.y += rescueDir.y * kickMag;
        this.lowSpeedTimer = 0;
      }
    }

    // Integration
    if (!guiParams.isPaused) {
      if (behaviorPause && !rearingActive && !groomingActive) {
        this.velocity.multiplyScalar(Math.exp(-7.0 * dt));
      }
      const ax = this.force.x / (this.mass * 1000);
      const az = this.force.y / (this.mass * 1000);
      this.velocity.x += ax * dt; this.velocity.y += az * dt;
      const uMS = this.guiParams.maxSpeed;
      const cS = this.velocity.length();
      if (cS > uMS) this.velocity.multiplyScalar(uMS / cS);
      this.mesh.position.x += this.velocity.x * dt;
      this.mesh.position.z += this.velocity.y * dt;
      this.locomotionTime += this.velocity.length() * dt;
    }

    // Hard boundary clamp with tangent turn selection (avoid wall-sticking).
    let hitWall = false;
    let hitXWall = false;
    let hitZWall = false;
    // Keep full body+head away from walls (not just the body center point).
    const baseBodyMargin = Math.max(
      3.8,
      3.2 * this.guiParams.bodyLength + 0.9 * this.guiParams.snoutLength
    );
    const uprightMarginReduction = this.rearing.mode === 'wall-supported'
      ? this.rearAmount * 2.10
      : 0;
    const HM = baseBodyMargin - uprightMarginReduction;
    if (this.mesh.position.x > ENCLOSURE_HALF_X - HM) { this.mesh.position.x = ENCLOSURE_HALF_X - HM; this.velocity.x = -Math.abs(this.velocity.x) * 0.25; hitWall = true; hitXWall = true; }
    if (this.mesh.position.x < -ENCLOSURE_HALF_X + HM) { this.mesh.position.x = -ENCLOSURE_HALF_X + HM; this.velocity.x = Math.abs(this.velocity.x) * 0.25; hitWall = true; hitXWall = true; }
    if (this.mesh.position.z > ENCLOSURE_HALF_Z - HM) { this.mesh.position.z = ENCLOSURE_HALF_Z - HM; this.velocity.y = -Math.abs(this.velocity.y) * 0.25; hitWall = true; hitZWall = true; }
    if (this.mesh.position.z < -ENCLOSURE_HALF_Z + HM) { this.mesh.position.z = -ENCLOSURE_HALF_Z + HM; this.velocity.y = Math.abs(this.velocity.y) * 0.25; hitWall = true; hitZWall = true; }
    if (hitWall) {
      const minTurnSpeed = Math.max(this.guiParams.maxSpeed * 0.45, 2.2);
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

    // Heading
    const speed = this.velocity.length();
    let isMoving = speed > 0.3
      && this.rearing.mode === 'none'
      && this.grooming.mode === 'none';
    if (!guiParams.isPaused) {
      let targetYaw = this.yawAngle;
      if (this.rearing.mode === 'wall-supported') {
        targetYaw = Math.atan2(
          this.rearing.supportDirection.x,
          this.rearing.supportDirection.y
        );
      } else if (isMoving) {
        targetYaw = Math.atan2(this.velocity.x, this.velocity.y);
      } else if (this.force.lengthSq() > 1e-6) {
        targetYaw = Math.atan2(this.force.x, this.force.y);
      }
      const yawErr = Math.atan2(Math.sin(targetYaw - this.yawAngle), Math.cos(targetYaw - this.yawAngle));
      const yawK = this.rearing.mode === 'wall-supported'
        ? 24.0
        : (isColliding ? 18.0 : 12.0);
      const yawD = 7.0;
      this.yawVel += (yawK * yawErr - yawD * this.yawVel) * dt;
      const maxYawRate = this.rearing.mode === 'wall-supported'
        ? 3.0
        : (isColliding ? 2.8 : 2.2);
      this.yawVel = THREE.MathUtils.clamp(this.yawVel, -maxYawRate, maxYawRate);
      this.yawAngle += this.yawVel * dt;
      this.yawAngle = Math.atan2(Math.sin(this.yawAngle), Math.cos(this.yawAngle));
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
      } else if (this.socialContact) {
        stateText = "SOCIAL SNIFFING";
        stateColor = "#7a3e34";
      } else if (isPausing || !isMoving) {
        stateText = "PAUSE // SNIFFING";
        stateColor = "#6e4d2f";
      } else if (wallDist < 3.2) {
        stateText = "WALL EXPLORATION";
        stateColor = "#526247";
      }
      hudRef.state.innerText = stateText;
      hudRef.state.style.color = stateColor;
      hudRef.hr.innerText = isMoving && !isPausing ? "540 BPM" : "410 BPM";
      // Display estimates must not consume the behavioral random stream.
      hudRef.resp.innerText = isMoving && !isPausing ? "210 BPM" : (162 + Math.round(Math.sin(time * 0.7) * 7)) + " BPM";
      hudRef.coord.innerText = `${this.mesh.position.x.toFixed(2)}, ${this.mesh.position.z.toFixed(2)}`;
    }

    // Body + tail dynamics (gravity + inertia)
    const speedNorm = Math.min(speed / this.maxSpeed, 1.0);
    const invDt = 1.0 / Math.max(dt, 1e-4);
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

    if (!guiParams.isPaused) {
      // Vertical body suspension with gravity.
      const targetBodyY = 1.17 + speedNorm * 0.045;
      const bodyK = 24.0, bodyD = 8.0, gravity = 3.4;
      this.bodyDyn.vy += (bodyK * (targetBodyY - this.bodyDyn.y) - bodyD * this.bodyDyn.vy - gravity) * dt;
      this.bodyDyn.y += this.bodyDyn.vy * dt;
      if (this.bodyDyn.y < 1.04) {
        this.bodyDyn.y = 1.04;
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

      // Tail inertial follow-through.
      const groomingTailSway = this.grooming.mode === 'flank-groom'
        ? this.grooming.side * this.groomAmount * 0.18
        : 0;
      const targetTailSway = THREE.MathUtils.clamp(
        -turnRate * 0.06 - lateralAccel * 0.01 + groomingTailSway,
        -0.9,
        0.9
      );
      const targetTailLift = THREE.MathUtils.clamp(
        -0.05 - forwardAccel * 0.004 + speedNorm * 0.03
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
        // Calm tiny oscillations while nearly stationary.
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
    this.neckMesh.position.copy(this.neckRestPosition);
    this.neckMesh.rotation.copy(this.neckRestRotation);
    this.headGroup.position.copy(this.headRestPosition);

    // Contact shadow follows mouse on ground
    this.contactShadow.position.set(0, -this.rig.position.y + 0.02, 0);

    // ═══ GAIT ANIMATION (anatomically correct) ═══
    const legCycle = this.locomotionTime * 4.1;
    const gaitAmp = Math.min(speedNorm * 1.35, 1.0) * 0.44 * this.motion.gaitAmpMul;

    if (isMoving) {
      // Diagonal rodent walk: hind hip drives while knee and ankle fold in swing.
      this.hindL.rotation.x = -0.24 + Math.sin(legCycle) * gaitAmp * 0.48;
      this.hindR.rotation.x = -0.24 + Math.sin(legCycle + Math.PI) * gaitAmp * 0.48;
      this.hindL.rotation.z = 0;
      this.hindR.rotation.z = 0;
      this.shankL.rotation.x = 0.56 + Math.max(0, Math.sin(legCycle + 0.5)) * gaitAmp * 0.34;
      this.shankR.rotation.x = 0.56 + Math.max(0, Math.sin(legCycle + Math.PI + 0.5)) * gaitAmp * 0.34;
      // Ankle follows shank, pushes off
      if (this.hindL._ankle) this.hindL._ankle.rotation.x = -0.36 + Math.sin(legCycle - 0.3) * gaitAmp * 0.22;
      if (this.hindR._ankle) this.hindR._ankle.rotation.x = -0.36 + Math.sin(legCycle + Math.PI - 0.3) * gaitAmp * 0.22;
      if (this.hindL._paw) this.hindL._paw.rotation.x = -0.10 - Math.sin(legCycle - 0.3) * gaitAmp * 0.12;
      if (this.hindR._paw) this.hindR._paw.rotation.x = -0.10 - Math.sin(legCycle + Math.PI - 0.3) * gaitAmp * 0.12;

      // Forelimb: shoulder drives, elbow/wrist follow
      this.upperArmL.position.z = 0;
      this.upperArmR.position.z = 0;
      this.upperArmL.rotation.x = FORELIMB_NEUTRAL.shoulderX
        + Math.sin(legCycle + Math.PI) * gaitAmp * 0.46;
      this.upperArmR.rotation.x = FORELIMB_NEUTRAL.shoulderX
        + Math.sin(legCycle) * gaitAmp * 0.46;
      this.upperArmL.rotation.z = -0.04;
      this.upperArmR.rotation.z = 0.04;
      // Elbow flex
      if (this.armL._forearm) this.armL._forearm.rotation.x = -0.62 + Math.max(0, Math.sin(legCycle + Math.PI + 0.4)) * gaitAmp * 0.28;
      if (this.armR._forearm) this.armR._forearm.rotation.x = -0.62 + Math.max(0, Math.sin(legCycle + 0.4)) * gaitAmp * 0.28;
      // Wrist flex
      if (this.armL._wrist) this.armL._wrist.rotation.x = -0.12 + Math.sin(legCycle + Math.PI + 0.6) * gaitAmp * 0.16;
      if (this.armR._wrist) this.armR._wrist.rotation.x = -0.12 + Math.sin(legCycle + 0.6) * gaitAmp * 0.16;
      if (this.armL._paw) this.armL._paw.rotation.x = -0.12 - Math.sin(legCycle + Math.PI + 0.6) * gaitAmp * 0.10;
      if (this.armR._paw) this.armR._paw.rotation.x = -0.12 - Math.sin(legCycle + 0.6) * gaitAmp * 0.10;
      this.rig.position.y += Math.sin(legCycle * 2.0) * speedNorm * 0.018;
    } else {
      // Idle pose
      this.hindL.rotation.x = -0.24; this.hindR.rotation.x = -0.24;
      this.hindL.rotation.z = 0; this.hindR.rotation.z = 0;
      this.shankL.rotation.x = 0.56; this.shankR.rotation.x = 0.56;
      if (this.hindL._ankle) this.hindL._ankle.rotation.x = -0.36;
      if (this.hindR._ankle) this.hindR._ankle.rotation.x = -0.36;
      if (this.hindL._paw) this.hindL._paw.rotation.x = -0.10;
      if (this.hindR._paw) this.hindR._paw.rotation.x = -0.10;

      // Neutral planted stance with subtle weight-shift micro-motion
      const pawShift = Math.sin(time * 1.3 + this.timeOffset) * 0.008;
      this.upperArmL.position.z = 0;
      this.upperArmR.position.z = 0;
      this.upperArmL.rotation.x = FORELIMB_NEUTRAL.shoulderX;
      this.upperArmR.rotation.x = FORELIMB_NEUTRAL.shoulderX;
      this.upperArmL.rotation.z = -0.04 + pawShift;
      this.upperArmR.rotation.z = 0.04 - pawShift;
      if (this.armL._forearm) this.armL._forearm.rotation.x = -0.62;
      if (this.armR._forearm) this.armR._forearm.rotation.x = -0.62;
      if (this.armL._wrist) this.armL._wrist.rotation.x = -0.12;
      if (this.armR._wrist) this.armR._wrist.rotation.x = -0.12;
      if (this.armL._paw) this.armL._paw.rotation.x = -0.12;
      if (this.armR._paw) this.armR._paw.rotation.x = -0.12;
    }

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
      this.contactShadow.position.set(0, -this.rig.position.y + 0.02, 0);
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
          wristX: side > 0 ? -0.06 : -0.13,
          pawX: side > 0 ? -0.08 : -0.12,
        }, groomPose);
        this.blendForelimbPose(this.armR, {
          shoulderProtraction: 0.02,
          shoulderX: side < 0 ? sameSideShoulder : supportShoulder,
          shoulderZ: side < 0 ? 0.10 : 0.16,
          elbowX: side < 0 ? -0.30 + flankLick * 0.05 : -0.58,
          wristX: side < 0 ? -0.06 : -0.13,
          pawX: side < 0 ? -0.08 : -0.12,
        }, groomPose);
      }

      const hipTarget = faceWash ? 0.46 : 0.14;
      const shankTarget = faceWash ? 0.82 : 0.62;
      this.hindL.rotation.x = THREE.MathUtils.lerp(this.hindL.rotation.x, hipTarget, groomPose);
      this.hindR.rotation.x = THREE.MathUtils.lerp(this.hindR.rotation.x, hipTarget, groomPose);
      this.shankL.rotation.x = THREE.MathUtils.lerp(this.shankL.rotation.x, shankTarget, groomPose);
      this.shankR.rotation.x = THREE.MathUtils.lerp(this.shankR.rotation.x, shankTarget, groomPose);
      if (this.hindL._ankle) this.hindL._ankle.rotation.x = THREE.MathUtils.lerp(this.hindL._ankle.rotation.x, -0.48, groomPose);
      if (this.hindR._ankle) this.hindR._ankle.rotation.x = THREE.MathUtils.lerp(this.hindR._ankle.rotation.x, -0.48, groomPose);
      this.contactShadow.position.set(0, -this.rig.position.y + 0.02, 0);
    }
    this.contactFrame.legCycle = legCycle;
    this.contactFrame.isMoving = isMoving;
    this.contactFrame.rear = rear;
    this.contactFrame.groom = groom;
    this.contactFrame.lateralAccel = lateralAccel;
    this.contactFrame.turnRate = turnRate;
    this.contactFrame.forwardAccel = forwardAccel;

    // Head animation with irregular sniff pulses
    const calmIdle = !isMoving && speed < 0.45 && this.lowSpeedTimer > 0.6;
    // Irregular sniff: combine multiple frequencies for biological plausibility
    const sniffBase = Math.sin(time * 40.0 + this.timeOffset);
    const sniffMod = Math.sin(time * 27.3 + this.timeOffset * 1.7) * 0.4;
    const sniffBurst = Math.max(0, Math.sin(time * 3.2 + this.timeOffset)) * 0.6;
    const sniffPulse = (sniffBase + sniffMod) * (0.4 + sniffBurst) * this.motion.sniffAmpMul;
    if (this.socialContact) {
      const targetYaw = Math.atan2(this.socialTarget.x, this.socialTarget.y);
      let relYaw = targetYaw - this.yawAngle;
      if (relYaw > Math.PI) relYaw -= Math.PI * 2;
      if (relYaw < -Math.PI) relYaw += Math.PI * 2;
      this.headGroup.rotation.y = THREE.MathUtils.clamp(relYaw, -0.46, 0.46) + sniffPulse * 0.028;
      this.headGroup.rotation.x = -0.075 + sniffPulse * 0.030;
      this.snoutTip.position.y = -0.15 + sniffPulse * 0.024;
      const noseScalePulse = 1.0 + sniffPulse * 0.018;
      this.snoutTip.scale.z = 0.72 * this.guiParams.snoutLength * noseScalePulse;
      this.padL.rotation.y = 0.11 + sniffPulse * 0.055;
      this.padR.rotation.y = -0.11 - sniffPulse * 0.055;
      this.padL.rotation.x = -0.015 + sniffPulse * 0.008;
      this.padR.rotation.x = -0.015 + sniffPulse * 0.008;
      this.padL.rotation.z = sniffPulse * 0.009;
      this.padR.rotation.z = -sniffPulse * 0.009;
    } else if (!isMoving) {
      if (calmIdle) {
        this.headGroup.rotation.y *= 0.85;
        this.headGroup.rotation.x *= 0.85;
        this.snoutTip.position.y = -0.15 + sniffPulse * 0.006;
        this.padL.rotation.y = 0.045 + sniffPulse * 0.012;
        this.padR.rotation.y = -0.045 - sniffPulse * 0.012;
        this.padL.rotation.x = -0.008;
        this.padR.rotation.x = -0.008;
        this.padL.rotation.z = 0;
        this.padR.rotation.z = 0;
      } else {
        this.headGroup.rotation.y = (
          Math.sin(time * 2.1 + this.timeOffset * 0.03) * 0.11
          + Math.sin(time * 0.73 + this.timeOffset) * 0.035
        ) * this.motion.headIdleYawMul;
        this.headGroup.rotation.x = (
          Math.sin(time * 3.2 + this.timeOffset * 0.05) * 0.045
          + sniffPulse * 0.014
        ) * this.motion.headIdlePitchMul;
        this.snoutTip.position.y = -0.15 + sniffPulse * 0.02;
        // Nose scale pulse during sniffing (compression)
        const noseScalePulse = 1.0 + sniffPulse * 0.015;
        this.snoutTip.scale.z = 0.72 * this.guiParams.snoutLength * noseScalePulse;
        this.padL.rotation.y = sniffPulse * 0.040 + 0.070;
        this.padR.rotation.y = -sniffPulse * 0.040 - 0.070;
        this.padL.rotation.x = -0.010 + sniffPulse * 0.006;
        this.padR.rotation.x = -0.010 + sniffPulse * 0.006;
        this.padL.rotation.z = sniffPulse * 0.007;
        this.padR.rotation.z = -sniffPulse * 0.007;
      }
    } else {
      const turnAmount = this.velocity.x * this.heading.y - this.velocity.y * this.heading.x;
      // Head stabilization: slight counter-rotation during locomotion
      const headStab = Math.sin(this.locomotionTime * 3.8) * 0.018;
      this.headGroup.rotation.set(Math.sin(time * 1.8) * 0.030 + headStab, turnAmount * 0.06, 0);
      this.snoutTip.position.y = -0.15;
      this.padL.rotation.y = -0.14; this.padR.rotation.y = 0.14;
      this.padL.rotation.x = 0.012; this.padR.rotation.x = 0.012;
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

    // The cervical mantle follows a fraction of head translation and
    // rotation, preserving a continuous neck during sniffing and large
    // rearing/grooming arcs without making the shoulders rotate as a unit.
    const headOffsetX = this.headGroup.position.x - this.headRestPosition.x;
    const headOffsetY = this.headGroup.position.y - this.headRestPosition.y;
    const headOffsetZ = this.headGroup.position.z - this.headRestPosition.z;
    this.neckMesh.position.x += headOffsetX * 0.48;
    this.neckMesh.position.y += headOffsetY * 0.36;
    this.neckMesh.position.z += headOffsetZ * 0.32;
    this.neckMesh.rotation.x += this.headGroup.rotation.x * 0.24;
    this.neckMesh.rotation.y += this.headGroup.rotation.y * 0.30;
    this.neckMesh.rotation.z += this.headGroup.rotation.z * 0.26;

    // ═══ EAR MICRO-TWITCHES ═══
    if (this.earL && this.earR && this.earMicro.baseRotL) {
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
      // Locomotion secondary bounce
      const earBounce = isMoving ? Math.sin(this.locomotionTime * 5.0) * 0.02 : 0;
      const earScan = Math.sin(time * 0.72 + this.timeOffset * 0.013) * 0.022;
      this.earL.rotation.z = this.earMicro.baseRotL.z + this.earMicro.twitchAngleL + earBounce;
      this.earR.rotation.z = this.earMicro.baseRotR.z + this.earMicro.twitchAngleR - earBounce;
      this.earL.rotation.x = this.earMicro.baseRotL.x + this.earMicro.twitchAngleL * 0.5;
      this.earR.rotation.x = this.earMicro.baseRotR.x + this.earMicro.twitchAngleR * 0.5;
      this.earL.rotation.y = this.earMicro.baseRotL.y + earScan + this.earMicro.twitchAngleL * 0.25;
      this.earR.rotation.y = this.earMicro.baseRotR.y - earScan + this.earMicro.twitchAngleR * 0.25;
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
      const eyeY = 0.95 * (1.0 - blinkAmount * 0.62);
      const corneaY = 0.95 * (1.0 - blinkAmount * 0.64);
      this.eyeL.scale.y = eyeY;
      this.eyeR.scale.y = eyeY;
      this.corneaL.scale.y = corneaY;
      this.corneaR.scale.y = corneaY;
    }

    // Tail geometry is refreshed after final body contact resolution.
    // Tracking visibility
    this.trackingGroup.visible = guiParams.showTrackingKeypoints;
    for (const key in this.poseNodes) this.poseNodes[key].visible = guiParams.showTrackingKeypoints;

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
    if (!isMoving || calmIdle) {
      this.snoutTip.scale.set(1.06, 0.76, this.guiParams.snoutLength * 0.72);
    }
    this.prevVelocity.copy(this.velocity);

    this.updateTrackingGeometry();
  }
}


function createBeddingLayer(count = 1100) {
  const group = new THREE.Group();
  const chipGeo = new THREE.BoxGeometry(1, 1, 1);
  const chipMat = new THREE.MeshStandardMaterial({
    color: 0xd0bd93,
    roughness: 0.94,
    metalness: 0.0,
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
    dummy.position.set(x, 0.035 + random() * 0.035, z);
    dummy.rotation.set(
      (random() - 0.5) * 0.18,
      random() * Math.PI,
      (random() - 0.5) * 0.18
    );
    const len = 0.09 + random() * 0.34;
    const thin = 0.008 + random() * 0.022;
    dummy.scale.set(0.018 + random() * 0.045, thin, len);
    dummy.updateMatrix();
    chips.setMatrixAt(i, dummy.matrix);
    color.setHSL(
      0.10 + (random() - 0.5) * 0.030,
      0.23 + random() * 0.18,
      0.56 + random() * 0.20
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
scene.add(createFloorMarks());
scene.add(createBeddingLayer());

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
scene.add(createArenaHardware());

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
// 12. LIGHTING (3-point photographic lab setup)
// ═══════════════════════════════════════════════════════════════
const ambientLight = new THREE.AmbientLight(0xf7f1e7, 0.24);
scene.add(ambientLight);

const hemiLight = new THREE.HemisphereLight(0xf5f7fb, 0xb7aa98, 0.38);
hemiLight.position.set(0, 20, 0); scene.add(hemiLight);

// Key light (overhead panel)
const keyLight = new THREE.DirectionalLight(0xfff4e8, 0.96);
keyLight.position.set(-5, 17, 6); keyLight.castShadow = true;
keyLight.shadow.mapSize.width = 4096; keyLight.shadow.mapSize.height = 4096;
keyLight.shadow.camera.near = 0.5; keyLight.shadow.camera.far = 30;
keyLight.shadow.camera.left = -20; keyLight.shadow.camera.right = 20;
keyLight.shadow.camera.top = 15; keyLight.shadow.camera.bottom = -15;
keyLight.shadow.bias = -0.001; keyLight.shadow.normalBias = 0.02;
scene.add(keyLight);

// Fill light
const fillLight = new THREE.DirectionalLight(0xe9f1ff, 0.20);
fillLight.position.set(8, 7, 8); fillLight.castShadow = false;
scene.add(fillLight);

// Rim light
const rimLight = new THREE.DirectionalLight(0xf8fbff, 0.30);
rimLight.position.set(6, 7, -8); rimLight.castShadow = false;
scene.add(rimLight);

const softbox = new THREE.RectAreaLight(0xfff1e2, 0.92, 16, 10);
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
  coord: document.getElementById('hud-coord')
};

// Shared shader engine is initialized with the model factory above.
const postFX = annolidShaders.createPostProcessing(renderer, {
  vigIntensity: 0.24, vigSoftness: 0.54, grainStrength: 0.012,
  saturation: 1.02, brightness: 1.0
});

const guiParams = {
  furColor: '#ffffff', skinColor: '#f0c8b8',
  chonkiness: 0.96, earSize: 0.82, furDensity: 1.0,
  showTrackingKeypoints: false, wireframeMode: false, isPaused: false,
  playbackSpeed: 1.0, cameraView: 'Both mice',
  addSubject: () => spawnMouse(), clearSubjects: () => clearMice(),
  presets: 'BALB/c (White)',
  // Body part controls
  tailLength: 0.85, tailThickness: 0.92,
  headSize: PROP.headSize, snoutLength: 1.0,
  bodyLength: 1.0, legLength: 1.06, legThickness: 1.10,
  // Physics
  maxSpeed: 5.4, dragCoeff: 2.55, wanderForce: 5.8, wallAttraction: 1.8,
  enableRearing: true, rearingFrequency: 1.0,
  triggerRearing: () => {
    const subject = mice.find(mouse => (
      mouse.rearing.mode === 'none' && mouse.grooming.mode === 'none'
    ));
    if (subject) subject.startRearing('unsupported');
  },
  enableGrooming: true, groomingFrequency: 1.0, groomingMode: 'Face wash',
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
  enablePostFX: false, postVignette: 0.16, postGrain: 0.004, postSaturation: 0.98,
};
postFX.enabled = guiParams.enablePostFX;

function getStrainProfile(presetName) {
  if (presetName === 'C57BL/6 (Black)') {
    return {
      eyeColor: '#070505', eyeRimColor: '#241818',
      earOuterTransmission: 0.12, earInnerTransmission: 0.22,
      earInnerSatOffset: 0.035, earInnerLightOffset: 0.00, earOuterRoughness: 0.66,
      earOuterColor: '#312e2f', earInnerColor: '#765a59',
      earSkinBlend: 0.18, earInnerSkinBlend: 0.12, earScaleMul: 0.94,
      earSheenColor: 0x9b8278, earAttenuationColor: 0x8a5148, earAttenuationDistance: 0.10,
      earYaw: 0.72, earHeightOffset: -0.02, padDarken: 0.94, clawColor: 0xc8beb6,
      pawColor: '#b2827f', pawPadColor: '#9a6c69',
      noseColor: '#b97972', pawSpreadMul: 1.0, toeLenMul: 0.98,
      gaitAmpMul: 0.92, headIdleYawMul: 0.82, headIdlePitchMul: 0.88,
      sniffAmpMul: 0.75, breathIdleAmp: 0.010, breathMoveAmp: 0.012,
      breathIdleFreq: 10.0, breathMoveFreq: 18.0, tailWaveAmpMul: 0.85, tailWaveFreqMul: 0.90
    };
  }
  if (presetName === 'CBA (Brown)') {
    return {
      eyeColor: '#120b09', eyeRimColor: '#3a201a',
      earOuterTransmission: 0.20, earInnerTransmission: 0.30,
      earInnerSatOffset: 0.035, earInnerLightOffset: 0.015, earOuterRoughness: 0.58,
      earOuterColor: '#765b4e', earInnerColor: '#b78378',
      earSkinBlend: 0.24, earInnerSkinBlend: 0.16, earScaleMul: 0.96,
      earSheenColor: 0xc6a08e, earAttenuationColor: 0xb77668, earAttenuationDistance: 0.14,
      earYaw: 0.58, earHeightOffset: -0.01, padDarken: 0.93, clawColor: 0xd1c2b7,
      pawColor: '#c8948b', pawPadColor: '#aa7771',
      noseColor: '#b8786f', pawSpreadMul: 1.0, toeLenMul: 1.0,
      gaitAmpMul: 1.0, headIdleYawMul: 1.0, headIdlePitchMul: 1.0,
      sniffAmpMul: 1.0, breathIdleAmp: 0.013, breathMoveAmp: 0.016,
      breathIdleFreq: 12.0, breathMoveFreq: 20.0, tailWaveAmpMul: 1.0, tailWaveFreqMul: 1.0
    };
  }
  if (presetName === 'Nude/Hairless') {
    return {
      eyeColor: '#5a171c', eyeRimColor: '#8c4d50',
      earOuterTransmission: 0.36, earInnerTransmission: 0.46,
      earInnerSatOffset: 0.10, earInnerLightOffset: 0.08, earOuterRoughness: 0.44,
      earOuterColor: '#d9b2ad', earInnerColor: '#e8b9b5',
      earSkinBlend: 0.28, earInnerSkinBlend: 0.18, earScaleMul: 1.02,
      earSheenColor: 0xffd6c8, earAttenuationColor: 0xea9d8f, earAttenuationDistance: 0.18,
      earYaw: 0.48, earHeightOffset: 0.00, padDarken: 0.94, clawColor: 0xddd1c8,
      pawColor: '#e9b8b2', pawPadColor: '#d39390',
      noseColor: '#d29089', pawSpreadMul: 1.04, toeLenMul: 1.05,
      gaitAmpMul: 1.08, headIdleYawMul: 1.15, headIdlePitchMul: 1.12,
      sniffAmpMul: 1.25, breathIdleAmp: 0.018, breathMoveAmp: 0.024,
      breathIdleFreq: 13.0, breathMoveFreq: 24.0, tailWaveAmpMul: 1.18, tailWaveFreqMul: 1.12
    };
  }
  return { // BALB/c default
    eyeColor: '#65171c', eyeRimColor: '#9a5b5e',
    earOuterTransmission: 0.28, earInnerTransmission: 0.40,
    earInnerSatOffset: 0.055, earInnerLightOffset: 0.035, earOuterRoughness: 0.60,
    earOuterColor: '#d6c9c7', earInnerColor: '#e5c0c2',
    earSkinBlend: 0.16, earInnerSkinBlend: 0.12, earScaleMul: 0.95,
    earSheenColor: 0xecd0d0, earAttenuationColor: 0xd7a3a6, earAttenuationDistance: 0.16,
    earYaw: 0.50, earHeightOffset: -0.01, padDarken: 0.94, clawColor: 0xddd2cb,
    pawColor: '#e3b3b1', pawPadColor: '#d3999b',
    noseColor: '#e39b9e', pawSpreadMul: 1.04, toeLenMul: 1.03,
    gaitAmpMul: 1.0, headIdleYawMul: 1.0, headIdlePitchMul: 1.0,
    sniffAmpMul: 1.0, breathIdleAmp: 0.013, breathMoveAmp: 0.016,
    breathIdleFreq: 12.0, breathMoveFreq: 20.0, tailWaveAmpMul: 1.0, tailWaveFreqMul: 1.0
  };
}

function applyStrainPreset(name) {
  guiParams.presets = name;
  if (name === 'BALB/c (White)') { guiParams.furColor = '#cec9bf'; guiParams.skinColor = '#e2aaa4'; guiParams.furDensity = 0.82; guiParams.earSize = 0.68; }
  if (name === 'C57BL/6 (Black)') { guiParams.furColor = '#171515'; guiParams.skinColor = '#40383a'; guiParams.furDensity = 0.92; guiParams.earSize = 0.66; }
  if (name === 'CBA (Brown)') { guiParams.furColor = '#6a4b3a'; guiParams.skinColor = '#d0a090'; guiParams.furDensity = 0.80; guiParams.earSize = 0.68; }
  if (name === 'Nude/Hairless') { guiParams.furColor = '#ffd0c8'; guiParams.skinColor = '#ffc0b8'; guiParams.furDensity = 0.0; guiParams.earSize = 0.75; }
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

// Neuroscience
const neuroscienceFolder = gui.addFolder('Neuroscience & Tracking');
neuroscienceFolder.add(guiParams, 'showTrackingKeypoints').name('Show Pose Skeleton').onChange(v => {
  mice.forEach(m => m.trackingGroup.visible = v);
});
neuroscienceFolder.add(guiParams, 'wireframeMode').name('Wireframe Bodies').onChange(v => {
  scene.traverse(child => {
    if (child.isMesh && child.material && child.material !== floorMat && child !== enclosureBox) {
      if (child.geometry && child.geometry.type !== 'ConeGeometry' && child.geometry.type !== 'CylinderGeometry') {
        child.material.wireframe = v;
      }
    }
  });
});

// Biological Traits
const traitFolder = gui.addFolder('Biological Traits');
traitFolder.addColor(guiParams, 'furColor').name('Fur Pigment').onFinishChange(v => {
  notifyRespawnRequired('furColor', v, 'Fur pigment applies to newly spawned subjects.');
});
traitFolder.addColor(guiParams, 'skinColor').name('Skin Pigment').onFinishChange(v => {
  notifyRespawnRequired('skinColor', v, 'Skin pigment applies to newly spawned subjects.');
});
traitFolder.add(guiParams, 'chonkiness', 0.5, 2.0).name('Fatness / Mass');
traitFolder.add(guiParams, 'earSize', 0.5, 2.0).name('Ear Magnitude');
traitFolder.add(guiParams, 'furDensity', 0.0, 1.0).name('Fur Density').onFinishChange(v => {
  notifyRespawnRequired(
    'furDensity',
    Number(v).toFixed(3),
    "Fur density changes require respawning subjects to take effect."
  );
});

// Simulation Controls
const controlFolder = gui.addFolder('Simulation Controls');
controlFolder.add(guiParams, 'isPaused').name('Pause Simulation');
controlFolder.add(guiParams, 'playbackSpeed', 0.25, 1.5, 0.25).name('Playback Speed');
controlFolder.add(guiParams, 'cameraView', ['Both mice', 'Follow black', 'Follow white', 'Top down'])
  .name('Camera').onChange(frameCameraView);
controlFolder.add(guiParams, 'addSubject').name('(+) Spawn Subject');
controlFolder.add(guiParams, 'clearSubjects').name('(!) Clear All');

// Body Part Controls
const bodyFolder = gui.addFolder('Body Part Controls');
bodyFolder.add(guiParams, 'tailLength', 0.2, 2.5, 0.05).name('Tail Length');
bodyFolder.add(guiParams, 'tailThickness', 0.3, 2.0, 0.05).name('Tail Thickness');
bodyFolder.add(guiParams, 'headSize', 0.5, 2.0, 0.05).name('Head Size');
bodyFolder.add(guiParams, 'snoutLength', 0.5, 2.0, 0.05).name('Snout Length');
bodyFolder.add(guiParams, 'bodyLength', 0.5, 2.0, 0.05).name('Body Length');
bodyFolder.add(guiParams, 'legLength', 0.5, 2.0, 0.05).name('Leg Length');
bodyFolder.add(guiParams, 'legThickness', 0.5, 2.0, 0.05).name('Leg Thickness');

// Proportions (detailed)
const propFolder = gui.addFolder('Proportions');
propFolder.add(guiParams, 'propBodyLength', 1.5, 4.0, 0.1).name('Body Length').onChange(v => { PROP.bodyLength = v; });
propFolder.add(guiParams, 'propBodyGirthX', 1.0, 2.5, 0.05).name('Body Girth X').onChange(v => { PROP.bodyGirthX = v; });
propFolder.add(guiParams, 'propWaistTaper', 0.3, 1.0, 0.05).name('Waist Taper').onChange(v => { PROP.waistTaper = v; });
propFolder.add(guiParams, 'propHaunchWidth', 0.8, 2.0, 0.05).name('Haunch Width').onChange(v => { PROP.haunchWidth = v; });
propFolder.add(guiParams, 'propShoulderWidth', 0.5, 1.5, 0.05).name('Shoulder Width').onChange(v => { PROP.shoulderWidth = v; });
propFolder.add(guiParams, 'propNeckLength', 0.3, 1.5, 0.05).name('Neck Length').onChange(v => { PROP.neckLength = v; });
propFolder.add(guiParams, 'propNeckGirth', 0.3, 1.0, 0.05).name('Neck Girth').onChange(v => { PROP.neckGirth = v; });
propFolder.add(guiParams, 'propSnoutLength', 0.8, 2.0, 0.05).name('Snout Length').onChange(v => { PROP.snoutLength = v; });
propFolder.add(guiParams, 'propSnoutWidth', 0.8, 1.8, 0.05).name('Snout Width').onChange(v => { PROP.snoutWidth = v; });
propFolder.add(guiParams, 'propCheekFullness', 0.1, 0.8, 0.05).name('Cheek Fullness').onChange(v => { PROP.cheekFullness = v; });
propFolder.add(guiParams, 'propEarYaw', 0.2, 1.2, 0.05).name('Ear Yaw').onChange(v => { PROP.earYaw = v; });
propFolder.add(guiParams, 'propEyeSize', 0.08, 0.25, 0.01).name('Eye Size').onChange(v => { PROP.eyeSize = v; });
propFolder.add(guiParams, 'propEyeSpacing', 0.3, 0.7, 0.02).name('Eye Spacing').onChange(v => { PROP.eyeSpacing = v; });
propFolder.add(guiParams, 'propForeLimbLength', 0.5, 1.5, 0.05).name('Forelimb Length').onChange(v => { PROP.foreLimbLength = v; });
propFolder.add(guiParams, 'propHindLimbLength', 0.5, 1.5, 0.05).name('Hindlimb Length').onChange(v => { PROP.hindLimbLength = v; });
propFolder.add(guiParams, 'propForePawSize', 0.5, 2.0, 0.05).name('Forepaw Size').onChange(v => { PROP.forePawSize = v; });
propFolder.add(guiParams, 'propHindPawSize', 0.5, 2.0, 0.05).name('Hindpaw Size').onChange(v => { PROP.hindPawSize = v; });
propFolder.add(guiParams, 'propTailSag', 0.0, 0.1, 0.005).name('Tail Sag').onChange(v => { PROP.tailSag = v; });
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
furFolder.add(guiParams, 'furGuardRatio', 0.0, 1.0, 0.05).name('Guard Hair Ratio');
furFolder.add(guiParams, 'furLength', 0.3, 2.0, 0.05).name('Fur Length');
furFolder.add(guiParams, 'furClumpiness', 0.0, 1.0, 0.05).name('Clumpiness');
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
  renderer.toneMappingExposure = 0.94;
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
shaderFolder.add(guiParams, 'sssThickness', 0.0, 1.0, 0.05).name('SSS Thickness');
shaderFolder.add(guiParams, 'sssWrap', 0.0, 1.0, 0.05).name('SSS Wrap');
shaderFolder.add(guiParams, 'sssBacklight', 0.0, 1.5, 0.05).name('SSS Backlight');
shaderFolder.add(guiParams, 'enableContactAO').name('Contact AO').onFinishChange(v => {
  notifyRespawnRequired(
    'enableContactAO',
    Boolean(v),
    'Shader changes require respawning subjects.'
  );
});
shaderFolder.add(guiParams, 'aoRadius', 0.5, 5.0, 0.1).name('AO Radius');
shaderFolder.add(guiParams, 'aoIntensity', 0.0, 1.0, 0.05).name('AO Intensity');
shaderFolder.add(guiParams, 'enableMicroDetail').name('Micro Detail').onFinishChange(v => {
  notifyRespawnRequired(
    'enableMicroDetail',
    Boolean(v),
    'Shader changes require respawning subjects.'
  );
});
shaderFolder.add(guiParams, 'microScale', 5.0, 200.0, 5.0).name('Detail Scale');
shaderFolder.add(guiParams, 'microStrength', 0.0, 0.3, 0.01).name('Detail Strength');
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
}

// Spawn initial subjects
applyStrainPreset('C57BL/6 (Black)');
spawnMouse({ position: [-3.6, -2.6], heading: [1.0, 0.12], speed: 0.9 });
applyStrainPreset('BALB/c (White)');
gui.controllersRecursive().forEach(c => c.updateDisplay());
spawnMouse({ position: [3.3, 2.6], heading: [-1.0, -0.12], speed: 0.85 });

if (IS_ANATOMY_QA) {
  const anatomySubjectIndex = datasetQuery.get('anatomy_subject') === 'white' ? 1 : 0;
  const subject = mice[anatomySubjectIndex];
  const observer = mice[1 - anatomySubjectIndex];
  subject.mesh.position.set(0, 0, 0);
  subject.velocity.set(0, 0);
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
  tailOwner.heading.set(0, 1);
  tailOwner.yawAngle = 0;
  tailOwner.prevHeadingAngle = 0;
  tailOwner.tailDyn.sway = 0;
  tailOwner.tailDyn.lift = -0.04;
  bodyObstacle.mesh.position.set(0, 0, -3.0);
  bodyObstacle.velocity.set(0, 0);
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
  leftMouse.heading.set(-1, 0);
  leftMouse.yawAngle = -Math.PI / 2;
  leftMouse.prevHeadingAngle = -Math.PI / 2;
  rightMouse.mesh.position.set(8.0, 0, 0);
  rightMouse.velocity.set(0, 0);
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
  subject.heading.set(-0.58, 0.82).normalize();
  subject.yawAngle = Math.atan2(subject.heading.x, subject.heading.y);
  subject.prevHeadingAngle = subject.yawAngle;
  subject.startGrooming(groomingMode, 999);
  if (groomingMode === 'flank-groom') subject.grooming.side = 1;
  observer.mesh.position.set(7.0, 0, -5.5);
  observer.velocity.set(0, 0);
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
  } else if (REARING_QA === 'wall-supported' || REARING_QA === 'wall-transition') {
    camera.position.set(15.0, 6.2, 2.0);
    camera.up.set(0, 1, 0);
    controls.target.set(0, 2.4, -ENCLOSURE_HALF_Z + 2.15);
  } else if (CONTACT_QA === 'tail') {
    camera.position.set(0, 18, 0.1);
    camera.up.set(0, 0, -1);
    controls.target.set(0, 0, 0);
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

// Read-only state for browser QA; scene objects and simulation controls stay private.
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

if (DATASET_CAPTURE) {
  advanceDatasetFrame(DATASET_FRAME_INDEX);
  window.annolidPoseDataset = Object.freeze({
    getFrame: buildDatasetFramePayload,
    setFrame: targetFrame => {
      advanceDatasetFrame(targetFrame);
      postFX.render(scene, camera, accumulatedTime);
      canvas.dataset.datasetReady = 'true';
      return buildDatasetFramePayload();
    },
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

function animate() {
  if (!DATASET_CAPTURE) requestAnimationFrame(animate);
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
  postFX.render(scene, camera, accumulatedTime);
  renderDurationMs = performance.now() - renderStart;
  updateVisualQaPixelTelemetry();
  if (DATASET_CAPTURE) canvas.dataset.datasetReady = 'true';
  canvas.dataset.sceneReady = 'true';
  document.getElementById('scene-status').textContent = '';
}
animate();
