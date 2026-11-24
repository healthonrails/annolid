/**
 * Neuroscience behavioral arenas for the 3D mouse simulation.
 *
 * Pure geometry, wall/obstacle collision, tail containment, zone classification,
 * and spawn/navigation helpers are dependency-free (testable in Node without
 * WebGL or DOM). Three.js 3D apparatus builders are instantiated lazily via
 * `createArenaVisual(THREE, arenaSpec, context)` so default Open Field startup
 * adds zero GPU or RNG overhead.
 */

import {
  createSeededRandom,
  computeOrientedWallMargin,
} from './two_mice_simulation.js';

export const DEFAULT_ARENA_ID = 'open_field';

const OPEN_FIELD_HALF_X = 36 * 0.5 - 1.6; // 16.4
const OPEN_FIELD_HALF_Z = 26 * 0.5 - 1.6; // 11.4

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function wrapAngleDelta(delta) {
  let a = delta;
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

/**
 * Build a validated closed polygon edge + reflex-corner specification.
 * Vertices must have positive signed area in XZ (`sum(x_i*z_{i+1} - x_{i+1}*z_i) > 0`),
 * which guarantees `normal = (-dz/L, dx/L)` points strictly inward.
 */
function buildPolygonGeometry(vertices, edgeWalledFlags, obstacles = []) {
  const n = vertices.length;
  let signedArea2 = 0;
  for (let i = 0; i < n; i++) {
    const [x1, z1] = vertices[i];
    const [x2, z2] = vertices[(i + 1) % n];
    signedArea2 += x1 * z2 - x2 * z1;
  }
  if (signedArea2 <= 1e-6) {
    throw new RangeError(
      `Arena polygon must have positive signed area in XZ (got ${signedArea2}).`
    );
  }
  const edges = [];
  for (let i = 0; i < n; i++) {
    const [x1, z1] = vertices[i];
    const [x2, z2] = vertices[(i + 1) % n];
    const dx = x2 - x1;
    const dz = z2 - z1;
    const length = Math.hypot(dx, dz);
    const tx = dx / length;
    const tz = dz / length;
    const nx = -tz;
    const nz = tx;
    edges.push(Object.freeze({
      x1,
      z1,
      x2,
      z2,
      dx,
      dz,
      length,
      tx,
      tz,
      nx,
      nz,
      walled: edgeWalledFlags ? Boolean(edgeWalledFlags[i]) : true,
    }));
  }
  const reflexVertices = [];
  for (let i = 0; i < n; i++) {
    const prev = edges[(i - 1 + n) % n];
    const next = edges[i];
    // Positive cross turns inward (convex); negative cross turns outward (reflex).
    const cross = prev.tx * next.tz - prev.tz * next.tx;
    if (cross < -1e-4) {
      let bisX = prev.nx + next.nx;
      let bisZ = prev.nz + next.nz;
      const bLen = Math.hypot(bisX, bisZ) || 1;
      bisX /= bLen;
      bisZ /= bLen;
      reflexVertices.push(Object.freeze({
        x: vertices[i][0],
        z: vertices[i][1],
        bisectorX: bisX,
        bisectorZ: bisZ,
        walled: Boolean(prev.walled || next.walled),
      }));
    }
  }
  return Object.freeze({
    type: 'polygon',
    vertices: Object.freeze(vertices.map(v => Object.freeze([v[0], v[1]]))),
    edges: Object.freeze(edges),
    reflexVertices: Object.freeze(reflexVertices),
    obstacles: Object.freeze(obstacles.map(o => Object.freeze({ ...o }))),
  });
}

function buildYMazePolygon(halfWidth = 4.4, armLength = 18.2) {
  const armAngles = [-Math.PI / 2, Math.PI / 6, (5 * Math.PI) / 6];
  const crotchRadius = halfWidth / Math.sin(Math.PI / 3);
  const vertices = [];
  for (let k = 0; k < 3; k++) {
    const a = armAngles[k];
    const ux = Math.cos(a);
    const uz = Math.sin(a);
    // Right-hand perpendicular in XZ (+X right, +Z down): (-uz, ux)
    const px = -uz;
    const pz = ux;
    vertices.push([ux * armLength - px * halfWidth, uz * armLength - pz * halfWidth]);
    vertices.push([ux * armLength + px * halfWidth, uz * armLength + pz * halfWidth]);
    const mid = a + Math.PI / 3;
    vertices.push([Math.cos(mid) * crotchRadius, Math.sin(mid) * crotchRadius]);
  }
  return buildPolygonGeometry(vertices, null);
}

const EPM_HW = 4.4;
const EPM_REACH = 18.6;
const EPM_VERTICES = [
  [-EPM_HW, -EPM_REACH],
  [EPM_HW, -EPM_REACH],
  [EPM_HW, -EPM_HW],
  [EPM_REACH, -EPM_HW],
  [EPM_REACH, EPM_HW],
  [EPM_HW, EPM_HW],
  [EPM_HW, EPM_REACH],
  [-EPM_HW, EPM_REACH],
  [-EPM_HW, EPM_HW],
  [-EPM_REACH, EPM_HW],
  [-EPM_REACH, -EPM_HW],
  [-EPM_HW, -EPM_HW],
];
// Open arms along -Z (edges 0, 1, 11) and +Z (edges 5, 6, 7);
// Closed arms along +X (edges 2, 3, 4) and -X (edges 8, 9, 10).
const EPM_EDGE_WALLED = [
  false, false,
  true, true, true,
  false, false, false,
  true, true, true,
  false,
];

const T_MAZE_VERTICES = [
  [-18.0, -13.0],
  [18.0, -13.0],
  [18.0, -4.2],
  [4.4, -4.2],
  [4.4, 13.6],
  [-4.4, 13.6],
  [-4.4, -4.2],
  [-18.0, -4.2],
];

const BARNES_HOLE_COUNT = 18;
const BARNES_HOLE_RING_RADIUS = 13.6;
const BARNES_HOLE_RADIUS = 1.08;
export const BARNES_MAZE_HOLES = Object.freeze(
  Array.from({ length: BARNES_HOLE_COUNT }, (_, idx) => {
    const angle = -Math.PI / 2 + (idx / BARNES_HOLE_COUNT) * Math.PI * 2;
    return Object.freeze({
      index: idx,
      angle,
      x: Math.cos(angle) * BARNES_HOLE_RING_RADIUS,
      z: Math.sin(angle) * BARNES_HOLE_RING_RADIUS,
      radius: BARNES_HOLE_RADIUS,
      isEscape: idx === 0,
    });
  })
);

export const ARENA_CATALOG = Object.freeze([
  Object.freeze({
    id: 'open_field',
    label: 'Open Field',
    aliases: Object.freeze(['open', 'oft', 'square_open_field', 'open field']),
    category: 'Locomotion & Anxiety',
    description: 'Standard rectangular acrylic open-field arena with matte anti-glare floor and aspen wood shavings.',
    hudTitle: 'Open Field · Two Mice',
    legacyRect: true,
    geometry: Object.freeze({
      type: 'rect',
      halfX: OPEN_FIELD_HALF_X,
      halfZ: OPEN_FIELD_HALF_Z,
      walled: true,
      obstacles: Object.freeze([]),
    }),
    bounds: Object.freeze({ minX: -18, maxX: 18, minZ: -13, maxZ: 13 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([8.4, 5.4, 22.5]),
      target: Object.freeze([0, 1.05, 0]),
      topDownHeight: 44,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 6.8, z: -4.8, radius: 2.75, color: 0xe6af42 }),
      hazard: Object.freeze({ x: -6.8, z: 4.8, radius: 2.75, color: 0xd95656 }),
    }),
    zones: Object.freeze(['center', 'periphery']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-3.6, -2.6]), heading: Object.freeze([1.0, 0.12]) }),
      Object.freeze({ position: Object.freeze([3.3, 2.6]), heading: Object.freeze([-1.0, -0.12]) }),
      Object.freeze({ position: Object.freeze([-3.4, 2.5]), heading: Object.freeze([0.85, -0.52]) }),
      Object.freeze({ position: Object.freeze([3.5, -2.5]), heading: Object.freeze([-0.85, 0.52]) }),
      Object.freeze({ position: Object.freeze([0.0, -3.2]), heading: Object.freeze([0.15, 0.98]) }),
      Object.freeze({ position: Object.freeze([0.0, 3.2]), heading: Object.freeze([-0.15, -0.98]) }),
    ]),
    navigationWaypoints: Object.freeze([]),
    navNodes: Object.freeze([]),
  }),
  Object.freeze({
    id: 'home_cage',
    label: 'Home Cage (Bedding)',
    aliases: Object.freeze(['home', 'shoebox', 'calms', 'home cage', 'home cage (bedding)']),
    category: 'Social & Ethology',
    description: 'Molded polycarbonate shoebox home cage with corncob bedding, shredded nestlet, and stainless water port.',
    hudTitle: 'Home Cage · Two Mice',
    legacyRect: true,
    geometry: Object.freeze({
      type: 'rect',
      halfX: OPEN_FIELD_HALF_X,
      halfZ: OPEN_FIELD_HALF_Z,
      walled: true,
      obstacles: Object.freeze([]),
    }),
    bounds: Object.freeze({ minX: -18, maxX: 18, minZ: -13, maxZ: 13 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([8.4, 8.8, 24.5]),
      target: Object.freeze([0, 0.95, 0]),
      topDownHeight: 54,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 6.8, z: -4.8, radius: 2.75, color: 0xe6af42 }),
      hazard: Object.freeze({ x: -6.8, z: 4.8, radius: 2.75, color: 0xd95656 }),
    }),
    zones: Object.freeze(['nest_zone', 'water_port_zone', 'open_bedding']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-3.6, -2.6]), heading: Object.freeze([1.0, 0.12]) }),
      Object.freeze({ position: Object.freeze([3.3, 2.6]), heading: Object.freeze([-1.0, -0.12]) }),
      Object.freeze({ position: Object.freeze([-3.4, 2.5]), heading: Object.freeze([0.85, -0.52]) }),
      Object.freeze({ position: Object.freeze([3.5, -2.5]), heading: Object.freeze([-0.85, 0.52]) }),
      Object.freeze({ position: Object.freeze([0.0, -3.2]), heading: Object.freeze([0.15, 0.98]) }),
      Object.freeze({ position: Object.freeze([0.0, 3.2]), heading: Object.freeze([-0.15, -0.98]) }),
    ]),
    navigationWaypoints: Object.freeze([]),
    navNodes: Object.freeze([]),
  }),
  Object.freeze({
    id: 'circular_open_field',
    label: 'Circular Open Field',
    aliases: Object.freeze(['circular', 'circle', 'circular_oft', 'circular open field']),
    category: 'Locomotion & Anxiety',
    description: 'Cylindrical open-field drum eliminating corner bias, with concentric center, intermediate, and periphery scoring rings.',
    hudTitle: 'Circular Open Field · Two Mice',
    legacyRect: false,
    geometry: Object.freeze({
      type: 'circle',
      radius: 15.2,
      walled: true,
      obstacles: Object.freeze([]),
    }),
    bounds: Object.freeze({ minX: -15.6, maxX: 15.6, minZ: -15.6, maxZ: 15.6 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([11.0, 19.8, 28.5]),
      target: Object.freeze([0, 0.8, 0]),
      topDownHeight: 52,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 6.4, z: -4.4, radius: 2.6, color: 0xe6af42 }),
      hazard: Object.freeze({ x: -6.4, z: 4.4, radius: 2.6, color: 0xd95656 }),
    }),
    zones: Object.freeze(['center', 'middle', 'periphery']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-4.4, -2.6]), heading: Object.freeze([0.92, 0.39]) }),
      Object.freeze({ position: Object.freeze([4.4, 2.6]), heading: Object.freeze([-0.92, -0.39]) }),
      Object.freeze({ position: Object.freeze([-4.2, 4.2]), heading: Object.freeze([0.72, -0.69]) }),
      Object.freeze({ position: Object.freeze([4.2, -4.2]), heading: Object.freeze([-0.72, 0.69]) }),
      Object.freeze({ position: Object.freeze([0.0, -6.4]), heading: Object.freeze([0.2, 0.98]) }),
      Object.freeze({ position: Object.freeze([0.0, 6.4]), heading: Object.freeze([-0.2, -0.98]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: 0, z: 0, weight: 0.7 }),
      Object.freeze({ x: 9.2, z: 0, weight: 1.0 }),
      Object.freeze({ x: -9.2, z: 0, weight: 1.0 }),
      Object.freeze({ x: 0, z: 9.2, weight: 1.0 }),
      Object.freeze({ x: 0, z: -9.2, weight: 1.0 }),
    ]),
    navNodes: Object.freeze([]),
  }),
  Object.freeze({
    id: 'elevated_plus_maze',
    label: 'Elevated Plus Maze',
    aliases: Object.freeze(['epm', 'plus_maze', 'elevated plus maze', 'elevated_plus']),
    category: 'Anxiety',
    description: 'Classic elevated plus maze with two tall-walled closed arms (±X), two exposed open arms (±Z) with low safety lips, and a central hub.',
    hudTitle: 'Elevated Plus Maze · Two Mice',
    legacyRect: false,
    geometry: buildPolygonGeometry(EPM_VERTICES, EPM_EDGE_WALLED),
    bounds: Object.freeze({ minX: -19.0, maxX: 19.0, minZ: -19.0, maxZ: 19.0 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([23.0, 25.0, 32.0]),
      target: Object.freeze([0, 0.4, 0]),
      topDownHeight: 62,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 11.8, z: 0.0, radius: 2.35, color: 0xe6af42 }),
      hazard: Object.freeze({ x: 0.0, z: -11.8, radius: 2.35, color: 0xd95656 }),
    }),
    zones: Object.freeze(['center', 'closed_arm_west', 'closed_arm_east', 'open_arm_north', 'open_arm_south']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-9.8, 0.0]), heading: Object.freeze([1.0, 0.0]) }),
      Object.freeze({ position: Object.freeze([9.8, 0.0]), heading: Object.freeze([-1.0, 0.0]) }),
      Object.freeze({ position: Object.freeze([0.0, -9.8]), heading: Object.freeze([0.0, 1.0]) }),
      Object.freeze({ position: Object.freeze([0.0, 9.8]), heading: Object.freeze([0.0, -1.0]) }),
      Object.freeze({ position: Object.freeze([-13.8, 0.0]), heading: Object.freeze([1.0, 0.0]) }),
      Object.freeze({ position: Object.freeze([13.8, 0.0]), heading: Object.freeze([-1.0, 0.0]) }),
    ]),
    // Naturalistic EPM preference: higher closed-arm weight, periodic open-arm exploration.
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: -13.2, z: 0.0, weight: 1.5 }),
      Object.freeze({ x: 13.2, z: 0.0, weight: 1.5 }),
      Object.freeze({ x: 0.0, z: 0.0, weight: 1.2 }),
      Object.freeze({ x: 0.0, z: -12.6, weight: 0.75 }),
      Object.freeze({ x: 0.0, z: 12.6, weight: 0.75 }),
    ]),
    navNodes: Object.freeze([
      Object.freeze({ x: 0.0, z: 0.0 }),
    ]),
  }),
  Object.freeze({
    id: 'y_maze',
    label: 'Y-Maze',
    aliases: Object.freeze(['ymaze', 'y-maze', 'y maze', 'spontaneous_alternation']),
    category: 'Spatial Working Memory',
    description: 'Symmetrical 120° three-arm Y-maze for spontaneous alternation with distinct intra-maze visual cues on arms A, B, and C.',
    hudTitle: 'Y-Maze · Two Mice',
    legacyRect: false,
    geometry: buildYMazePolygon(4.4, 18.2),
    bounds: Object.freeze({ minX: -18.5, maxX: 18.5, minZ: -18.6, maxZ: 13.5 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([15.5, 25.5, 32.0]),
      target: Object.freeze([0, 0.6, -2.0]),
      topDownHeight: 60,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 0.0, z: -12.2, radius: 2.35, color: 0xe6af42 }),
      hazard: Object.freeze({ x: -10.6, z: 6.1, radius: 2.35, color: 0xd95656 }),
    }),
    zones: Object.freeze(['center_hub', 'arm_a', 'arm_b', 'arm_c']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-9.2, 5.3]), heading: Object.freeze([0.866, -0.5]) }),
      Object.freeze({ position: Object.freeze([9.2, 5.3]), heading: Object.freeze([-0.866, -0.5]) }),
      Object.freeze({ position: Object.freeze([0.0, -10.8]), heading: Object.freeze([0.0, 1.0]) }),
      Object.freeze({ position: Object.freeze([-13.0, 7.5]), heading: Object.freeze([0.866, -0.5]) }),
      Object.freeze({ position: Object.freeze([13.0, 7.5]), heading: Object.freeze([-0.866, -0.5]) }),
      Object.freeze({ position: Object.freeze([0.0, -6.2]), heading: Object.freeze([0.0, 1.0]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: 0.0, z: -13.0, weight: 1.2 }),
      Object.freeze({ x: 11.2, z: 6.5, weight: 1.2 }),
      Object.freeze({ x: -11.2, z: 6.5, weight: 1.2 }),
      Object.freeze({ x: 0.0, z: 0.0, weight: 1.0 }),
    ]),
    navNodes: Object.freeze([
      Object.freeze({ x: 0.0, z: 0.0 }),
    ]),
  }),
  Object.freeze({
    id: 't_maze',
    label: 'T-Maze',
    aliases: Object.freeze(['tmaze', 't-maze', 't maze', 'forced_alternation']),
    category: 'Working Memory & Decision',
    description: 'T-shaped alternation maze with start-box guillotine gate frame, central choice point, and left/right goal-arm reward wells.',
    hudTitle: 'T-Maze · Two Mice',
    legacyRect: false,
    geometry: buildPolygonGeometry(T_MAZE_VERTICES, null),
    bounds: Object.freeze({ minX: -18.5, maxX: 18.5, minZ: -13.5, maxZ: 14.0 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([14.0, 23.5, 33.5]),
      target: Object.freeze([0, 0.6, -1.5]),
      topDownHeight: 59,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 12.6, z: -8.6, radius: 2.35, color: 0xe6af42 }),
      hazard: Object.freeze({ x: -12.6, z: -8.6, radius: 2.35, color: 0xd95656 }),
    }),
    zones: Object.freeze(['start_box', 'start_stem', 'choice_point', 'left_arm', 'right_arm']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([0.0, 8.2]), heading: Object.freeze([0.0, -1.0]) }),
      Object.freeze({ position: Object.freeze([0.0, 0.6]), heading: Object.freeze([0.0, -1.0]) }),
      Object.freeze({ position: Object.freeze([-10.8, -8.6]), heading: Object.freeze([1.0, 0.0]) }),
      Object.freeze({ position: Object.freeze([10.8, -8.6]), heading: Object.freeze([-1.0, 0.0]) }),
      Object.freeze({ position: Object.freeze([-5.6, -8.6]), heading: Object.freeze([-1.0, 0.0]) }),
      Object.freeze({ position: Object.freeze([5.6, -8.6]), heading: Object.freeze([1.0, 0.0]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: 0.0, z: 8.4, weight: 1.0 }),
      Object.freeze({ x: 0.0, z: -8.6, weight: 1.2 }),
      Object.freeze({ x: -12.8, z: -8.6, weight: 1.25 }),
      Object.freeze({ x: 12.8, z: -8.6, weight: 1.25 }),
    ]),
    navNodes: Object.freeze([
      Object.freeze({ x: 0.0, z: -8.6 }),
    ]),
  }),
  Object.freeze({
    id: 'three_chamber',
    label: 'Three-Chamber Social',
    aliases: Object.freeze(['3_chamber', '3-chamber', 'three chamber', 'crawley', 'sociability', 'three-chamber social']),
    category: 'Social Preference',
    description: 'Crawley three-chamber sociability apparatus with clear dividing partitions, central doorways, and two cylindrical wire holding cups.',
    hudTitle: 'Three-Chamber Social · Two Mice',
    legacyRect: false,
    geometry: Object.freeze({
      type: 'rect',
      halfX: 21.0,
      halfZ: 11.2,
      walled: true,
      obstacles: Object.freeze([
        Object.freeze({ type: 'box', minX: -7.35, maxX: -6.65, minZ: -11.2, maxZ: -3.7, walled: true, label: 'left_divider_north' }),
        Object.freeze({ type: 'box', minX: -7.35, maxX: -6.65, minZ: 3.7, maxZ: 11.2, walled: true, label: 'left_divider_south' }),
        Object.freeze({ type: 'box', minX: 6.65, maxX: 7.35, minZ: -11.2, maxZ: -3.7, walled: true, label: 'right_divider_north' }),
        Object.freeze({ type: 'box', minX: 6.65, maxX: 7.35, minZ: 3.7, maxZ: 11.2, walled: true, label: 'right_divider_south' }),
        Object.freeze({ type: 'circle', x: -14.2, z: 0.0, radius: 2.30, walled: true, label: 'social_cup_left' }),
        Object.freeze({ type: 'circle', x: 14.2, z: 0.0, radius: 2.30, walled: true, label: 'object_cup_right' }),
      ]),
    }),
    bounds: Object.freeze({ minX: -21.4, maxX: 21.4, minZ: -11.6, maxZ: 11.6 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([10.5, 22.0, 35.5]),
      target: Object.freeze([0, 0.9, 0]),
      topDownHeight: 60,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: -14.2, z: 5.8, radius: 2.45, color: 0xe6af42 }),
      hazard: Object.freeze({ x: 14.2, z: -5.8, radius: 2.45, color: 0xd95656 }),
    }),
    zones: Object.freeze(['social_cup_zone', 'left_chamber', 'center_chamber', 'right_chamber', 'object_cup_zone']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-2.2, -2.0]), heading: Object.freeze([-0.96, 0.28]) }),
      Object.freeze({ position: Object.freeze([2.2, 2.0]), heading: Object.freeze([0.96, -0.28]) }),
      Object.freeze({ position: Object.freeze([-14.2, 6.2]), heading: Object.freeze([0.0, -1.0]) }),
      Object.freeze({ position: Object.freeze([14.2, -6.2]), heading: Object.freeze([0.0, 1.0]) }),
      Object.freeze({ position: Object.freeze([-14.2, -6.2]), heading: Object.freeze([0.0, 1.0]) }),
      Object.freeze({ position: Object.freeze([14.2, 6.2]), heading: Object.freeze([0.0, -1.0]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: 0.0, z: 0.0, weight: 1.0 }),
      Object.freeze({ x: -7.0, z: 0.0, weight: 0.9 }),
      Object.freeze({ x: 7.0, z: 0.0, weight: 0.9 }),
      Object.freeze({ x: -14.2, z: 5.8, weight: 1.4 }),
      Object.freeze({ x: -14.2, z: -5.8, weight: 1.4 }),
      Object.freeze({ x: 14.2, z: 5.8, weight: 1.0 }),
      Object.freeze({ x: 14.2, z: -5.8, weight: 1.0 }),
    ]),
    navNodes: Object.freeze([
      Object.freeze({ x: 0.0, z: 0.0 }),
      Object.freeze({ x: -7.0, z: 0.0 }),
      Object.freeze({ x: 7.0, z: 0.0 }),
      Object.freeze({ x: -10.2, z: 0.0 }),
      Object.freeze({ x: 10.2, z: 0.0 }),
      Object.freeze({ x: -14.2, z: -6.2 }),
      Object.freeze({ x: -14.2, z: 6.2 }),
      Object.freeze({ x: 14.2, z: -6.2 }),
      Object.freeze({ x: 14.2, z: 6.2 }),
      Object.freeze({ x: -18.0, z: 0.0 }),
      Object.freeze({ x: 18.0, z: 0.0 }),
    ]),
  }),
  Object.freeze({
    id: 'light_dark_box',
    label: 'Light-Dark Box',
    aliases: Object.freeze(['light_dark', 'ld_box', 'light-dark', 'light dark box', 'light-dark box']),
    category: 'Anxiety & Photophobia',
    description: 'Two-compartment light-dark transition box pairing a bright white chamber (2/3) with a sheltered charcoal chamber (1/3) linked by a doorway.',
    hudTitle: 'Light-Dark Box · Two Mice',
    legacyRect: false,
    geometry: Object.freeze({
      type: 'rect',
      halfX: 17.6,
      halfZ: 11.2,
      walled: true,
      obstacles: Object.freeze([
        Object.freeze({ type: 'box', minX: -5.75, maxX: -5.05, minZ: -11.2, maxZ: -3.7, walled: true, label: 'ld_divider_north' }),
        Object.freeze({ type: 'box', minX: -5.75, maxX: -5.05, minZ: 3.7, maxZ: 11.2, walled: true, label: 'ld_divider_south' }),
      ]),
    }),
    bounds: Object.freeze({ minX: -18.0, maxX: 18.0, minZ: -11.6, maxZ: 11.6 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([9.0, 22.0, 30.5]),
      target: Object.freeze([0, 0.9, 0]),
      topDownHeight: 54,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: -11.6, z: -4.2, radius: 2.45, color: 0xe6af42 }),
      hazard: Object.freeze({ x: 10.2, z: 3.8, radius: 2.55, color: 0xd95656 }),
    }),
    zones: Object.freeze(['dark_compartment', 'doorway', 'light_compartment']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([4.6, -2.4]), heading: Object.freeze([-0.96, 0.28]) }),
      Object.freeze({ position: Object.freeze([-11.4, 2.2]), heading: Object.freeze([0.96, -0.28]) }),
      Object.freeze({ position: Object.freeze([10.8, 4.0]), heading: Object.freeze([-0.82, -0.57]) }),
      Object.freeze({ position: Object.freeze([-11.4, -4.6]), heading: Object.freeze([0.85, 0.52]) }),
      Object.freeze({ position: Object.freeze([5.2, 5.2]), heading: Object.freeze([-0.7, -0.71]) }),
      Object.freeze({ position: Object.freeze([11.2, -4.8]), heading: Object.freeze([-0.9, 0.44]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: -11.8, z: 0.0, weight: 1.5 }),
      Object.freeze({ x: -11.8, z: -5.2, weight: 1.2 }),
      Object.freeze({ x: -5.4, z: 0.0, weight: 1.0 }),
      Object.freeze({ x: 5.2, z: 0.0, weight: 1.0 }),
      Object.freeze({ x: 11.4, z: 4.8, weight: 0.8 }),
      Object.freeze({ x: 11.4, z: -4.8, weight: 0.8 }),
    ]),
    navNodes: Object.freeze([
      Object.freeze({ x: -5.4, z: 0.0 }),
      Object.freeze({ x: -11.4, z: 0.0 }),
      Object.freeze({ x: 2.0, z: 0.0 }),
    ]),
  }),
  Object.freeze({
    id: 'barnes_maze',
    label: 'Barnes Maze',
    aliases: Object.freeze(['barnes', 'barnes maze', 'spatial_memory']),
    category: 'Spatial Reference Memory',
    description: 'Elevated circular spatial memory table with 18 perimeter escape holes, a recessed under-table target escape box, and four distal spatial cues.',
    hudTitle: 'Barnes Maze · Two Mice',
    legacyRect: false,
    geometry: Object.freeze({
      type: 'circle',
      radius: 15.6,
      walled: false,
      obstacles: Object.freeze([]),
    }),
    bounds: Object.freeze({ minX: -16.5, maxX: 16.5, minZ: -16.5, maxZ: 16.5 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([16.0, 22.8, 34.5]),
      target: Object.freeze([0, 0.3, 0]),
      topDownHeight: 60,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 0.0, z: -11.0, radius: 2.45, color: 0xe6af42 }),
      hazard: Object.freeze({ x: 0.0, z: 10.6, radius: 2.45, color: 0xd95656 }),
    }),
    zones: Object.freeze(['center', 'escape_quadrant', 'perimeter_ring', 'inner_table']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-3.6, 1.8]), heading: Object.freeze([0.85, -0.52]) }),
      Object.freeze({ position: Object.freeze([3.6, -1.8]), heading: Object.freeze([-0.85, 0.52]) }),
      Object.freeze({ position: Object.freeze([-4.8, -4.2]), heading: Object.freeze([0.6, 0.8]) }),
      Object.freeze({ position: Object.freeze([4.8, 4.2]), heading: Object.freeze([-0.6, -0.8]) }),
      Object.freeze({ position: Object.freeze([0.0, 5.6]), heading: Object.freeze([0.1, -0.99]) }),
      Object.freeze({ position: Object.freeze([0.0, -5.6]), heading: Object.freeze([-0.1, 0.99]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: 0.0, z: -11.4, weight: 1.5 }),
      Object.freeze({ x: 10.6, z: -4.2, weight: 1.0 }),
      Object.freeze({ x: 10.6, z: 4.2, weight: 1.0 }),
      Object.freeze({ x: 0.0, z: 11.4, weight: 1.0 }),
      Object.freeze({ x: -10.6, z: 4.2, weight: 1.0 }),
      Object.freeze({ x: -10.6, z: -4.2, weight: 1.0 }),
      Object.freeze({ x: 0.0, z: 0.0, weight: 0.7 }),
    ]),
    navNodes: Object.freeze([]),
  }),
  Object.freeze({
    id: 'novel_object',
    label: 'Novel Object (NOR)',
    aliases: Object.freeze(['nor', 'novel_object_recognition', 'novel object', 'novel object (nor)']),
    category: 'Recognition Memory',
    description: 'Square recognition-memory chamber containing a familiar cylindrical lab flask and a novel stepped geometric block tower.',
    hudTitle: 'Novel Object (NOR) · Two Mice',
    legacyRect: false,
    geometry: Object.freeze({
      type: 'rect',
      halfX: 15.6,
      halfZ: 15.6,
      walled: true,
      obstacles: Object.freeze([
        Object.freeze({ type: 'circle', x: -6.4, z: -2.4, radius: 2.05, walled: true, label: 'familiar_object' }),
        Object.freeze({ type: 'box', minX: 4.4, maxX: 8.4, minZ: -4.4, maxZ: -0.4, walled: true, label: 'novel_object' }),
      ]),
    }),
    bounds: Object.freeze({ minX: -16.0, maxX: 16.0, minZ: -16.0, maxZ: 16.0 }),
    defaultCamera: Object.freeze({
      position: Object.freeze([11.5, 20.5, 32.0]),
      target: Object.freeze([0, 0.9, -0.5]),
      topDownHeight: 54,
    }),
    flyZones: Object.freeze({
      reward: Object.freeze({ x: 6.4, z: 3.4, radius: 2.4, color: 0xe6af42 }),
      hazard: Object.freeze({ x: -6.4, z: 3.4, radius: 2.4, color: 0xd95656 }),
    }),
    zones: Object.freeze(['familiar_object_zone', 'novel_object_zone', 'center', 'periphery']),
    spawnSlots: Object.freeze([
      Object.freeze({ position: Object.freeze([-3.2, 6.4]), heading: Object.freeze([-0.32, -0.95]) }),
      Object.freeze({ position: Object.freeze([3.2, 6.4]), heading: Object.freeze([0.32, -0.95]) }),
      Object.freeze({ position: Object.freeze([-9.4, 7.6]), heading: Object.freeze([0.38, -0.92]) }),
      Object.freeze({ position: Object.freeze([9.4, 7.6]), heading: Object.freeze([-0.38, -0.92]) }),
      Object.freeze({ position: Object.freeze([0.0, -9.8]), heading: Object.freeze([0.0, 1.0]) }),
      Object.freeze({ position: Object.freeze([0.0, 10.6]), heading: Object.freeze([0.0, -1.0]) }),
    ]),
    navigationWaypoints: Object.freeze([
      Object.freeze({ x: -6.4, z: 2.2, weight: 1.15 }),
      Object.freeze({ x: -6.4, z: -7.0, weight: 1.0 }),
      Object.freeze({ x: 6.4, z: 2.2, weight: 1.55 }),
      Object.freeze({ x: 6.4, z: -7.0, weight: 1.35 }),
      Object.freeze({ x: 0.0, z: 7.4, weight: 0.9 }),
    ]),
    navNodes: Object.freeze([
      Object.freeze({ x: 0.0, z: -2.4 }),
      Object.freeze({ x: 0.0, z: 4.6 }),
      Object.freeze({ x: 0.0, z: -9.4 }),
      Object.freeze({ x: -6.4, z: 3.6 }),
      Object.freeze({ x: 6.4, z: 3.6 }),
      Object.freeze({ x: -6.4, z: -8.6 }),
      Object.freeze({ x: 6.4, z: -8.6 }),
      Object.freeze({ x: -12.0, z: -2.4 }),
      Object.freeze({ x: 12.0, z: -2.4 }),
    ]),
  }),
]);

export const ARENA_IDS = Object.freeze(ARENA_CATALOG.map(a => a.id));
export const ARENA_LABELS = Object.freeze(ARENA_CATALOG.map(a => a.label));

const ARENA_BY_KEY = new Map();
for (const spec of ARENA_CATALOG) {
  ARENA_BY_KEY.set(spec.id.toLowerCase(), spec);
  ARENA_BY_KEY.set(spec.label.toLowerCase(), spec);
  for (const alias of spec.aliases) {
    ARENA_BY_KEY.set(alias.toLowerCase(), spec);
  }
}

export function resolveArenaId(value, { allowDefault = false } = {}) {
  const raw = String(value ?? '').trim();
  if (!raw) {
    if (allowDefault) return DEFAULT_ARENA_ID;
    throw new RangeError(
      `Arena identifier must be a non-empty string; available: ${ARENA_IDS.join(', ')}.`
    );
  }
  const normalized = raw.toLowerCase();
  const matched = ARENA_BY_KEY.get(normalized)
    || ARENA_BY_KEY.get(normalized.replace(/[\s-]+/g, '_'));
  if (matched) return matched.id;
  if (allowDefault) return DEFAULT_ARENA_ID;
  throw new RangeError(
    `Unknown arena ${JSON.stringify(value)}; available: ${ARENA_IDS.join(', ')}.`
  );
}

export function getArenaSpec(idOrLabel, { allowDefault = false } = {}) {
  const id = resolveArenaId(idOrLabel, { allowDefault });
  return ARENA_BY_KEY.get(id);
}

export function summarizeArenaSpec(spec) {
  return {
    id: spec.id,
    label: spec.label,
    category: spec.category,
    description: spec.description,
    hudTitle: spec.hudTitle,
    bounds: { ...spec.bounds },
    zones: [...spec.zones],
  };
}

// ═══════════════════════════════════════════════════════════════
// PURE 2D GEOMETRY & SIGNED-DISTANCE ENGINE
// ═══════════════════════════════════════════════════════════════

export function isPointInsidePolygon(vertices, x, z) {
  let inside = false;
  for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
    const xi = vertices[i][0];
    const zi = vertices[i][1];
    const xj = vertices[j][0];
    const zj = vertices[j][1];
    const intersects = ((zi > z) !== (zj > z))
      && (x < ((xj - xi) * (z - zi)) / ((zj - zi) || 1e-12) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

function closestPointOnSegment(x, z, x1, z1, dx, dz, length) {
  const lenSq = length * length;
  const rawT = lenSq > 1e-12 ? ((x - x1) * dx + (z - z1) * dz) / lenSq : 0;
  const t = clamp(rawT, 0, 1);
  const cx = x1 + dx * t;
  const cz = z1 + dz * t;
  return {
    cx,
    cz,
    t,
    dist: Math.hypot(x - cx, z - cz),
  };
}

function queryObstacleConstraint(obs, x, z) {
  if (obs.type === 'circle') {
    const dx = x - obs.x;
    const dz = z - obs.z;
    const d = Math.hypot(dx, dz);
    const nx = d > 1e-6 ? dx / d : 1;
    const nz = d > 1e-6 ? dz / d : 0;
    return {
      dist: d - obs.radius,
      normalX: nx,
      normalZ: nz,
      tangentX: -nz,
      tangentZ: nx,
      walled: Boolean(obs.walled),
      contactX: obs.x + nx * obs.radius,
      contactZ: obs.z + nz * obs.radius,
      isObstacle: true,
      obstacle: obs,
    };
  }
  // Axis-aligned box obstacle
  const qx = clamp(x, obs.minX, obs.maxX);
  const qz = clamp(z, obs.minZ, obs.maxZ);
  const dx = x - qx;
  const dz = z - qz;
  const outsideDist = Math.hypot(dx, dz);
  if (outsideDist > 1e-6) {
    const nx = dx / outsideDist;
    const nz = dz / outsideDist;
    return {
      dist: outsideDist,
      normalX: nx,
      normalZ: nz,
      tangentX: -nz,
      tangentZ: nx,
      walled: Boolean(obs.walled),
      contactX: qx,
      contactZ: qz,
      isObstacle: true,
      obstacle: obs,
    };
  }
  // Strictly inside box: push out through nearest face
  const dLeft = x - obs.minX;
  const dRight = obs.maxX - x;
  const dTop = z - obs.minZ;
  const dBottom = obs.maxZ - z;
  const minPen = Math.min(dLeft, dRight, dTop, dBottom);
  let nx = -1;
  let nz = 0;
  let cx = obs.minX;
  let cz = z;
  if (minPen === dRight) {
    nx = 1; nz = 0; cx = obs.maxX; cz = z;
  } else if (minPen === dTop) {
    nx = 0; nz = -1; cx = x; cz = obs.minZ;
  } else if (minPen === dBottom) {
    nx = 0; nz = 1; cx = x; cz = obs.maxZ;
  }
  return {
    dist: -minPen,
    normalX: nx,
    normalZ: nz,
    tangentX: -nz,
    tangentZ: nx,
    walled: Boolean(obs.walled),
    contactX: cx,
    contactZ: cz,
    isObstacle: true,
    obstacle: obs,
  };
}

/**
 * Signed distance from (x, z) to the walkable interior of `arena`.
 * Positive strictly inside walkable space, 0 on boundary, negative outside or inside an obstacle.
 */
export function arenaSignedDistance(arena, x, z) {
  const geom = arena.geometry;
  let minDist = Infinity;

  if (geom.type === 'rect') {
    const dx = geom.halfX - Math.abs(x);
    const dz = geom.halfZ - Math.abs(z);
    if (dx >= 0 && dz >= 0) {
      minDist = Math.min(dx, dz);
    } else {
      minDist = -Math.hypot(Math.max(0, -dx), Math.max(0, -dz));
    }
  } else if (geom.type === 'circle') {
    minDist = geom.radius - Math.hypot(x, z);
  } else if (geom.type === 'polygon') {
    let bestEdgeDist = Infinity;
    for (let i = 0; i < geom.edges.length; i++) {
      const e = geom.edges[i];
      const cp = closestPointOnSegment(x, z, e.x1, e.z1, e.dx, e.dz, e.length);
      if (cp.dist < bestEdgeDist) bestEdgeDist = cp.dist;
    }
    const inside = isPointInsidePolygon(geom.vertices, x, z);
    minDist = inside ? bestEdgeDist : -bestEdgeDist;
  }

  for (let i = 0; i < geom.obstacles.length; i++) {
    const c = queryObstacleConstraint(geom.obstacles[i], x, z);
    if (c.dist < minDist) minDist = c.dist;
  }
  return minDist;
}

export function isPointInArena(arena, x, z, margin = 0) {
  return arenaSignedDistance(arena, x, z) >= margin - 1e-6;
}

/**
 * Query active wall / cliff / obstacle constraints around (x, z), sorted by distance.
 * Each entry has `{ dist, normalX, normalZ, tangentX, tangentZ, walled, contactX, contactZ }`.
 */
export function queryArenaWalls(arena, x, z) {
  const geom = arena.geometry;
  const walls = [];

  if (geom.type === 'rect') {
    walls.push(
      { dist: geom.halfX - x, normalX: -1, normalZ: 0, tangentX: 0, tangentZ: 1, walled: geom.walled, contactX: geom.halfX, contactZ: z },
      { dist: geom.halfX + x, normalX: 1, normalZ: 0, tangentX: 0, tangentZ: -1, walled: geom.walled, contactX: -geom.halfX, contactZ: z },
      { dist: geom.halfZ - z, normalX: 0, normalZ: -1, tangentX: -1, tangentZ: 0, walled: geom.walled, contactX: x, contactZ: geom.halfZ },
      { dist: geom.halfZ + z, normalX: 0, normalZ: 1, tangentX: 1, tangentZ: 0, walled: geom.walled, contactX: x, contactZ: -geom.halfZ },
    );
  } else if (geom.type === 'circle') {
    const r = Math.hypot(x, z);
    const nx = r > 1e-6 ? -x / r : -1;
    const nz = r > 1e-6 ? -z / r : 0;
    walls.push({
      dist: geom.radius - r,
      normalX: nx,
      normalZ: nz,
      tangentX: -nz,
      tangentZ: nx,
      walled: geom.walled,
      contactX: -nx * geom.radius,
      contactZ: -nz * geom.radius,
    });
  } else if (geom.type === 'polygon') {
    const inside = isPointInsidePolygon(geom.vertices, x, z);
    for (let i = 0; i < geom.edges.length; i++) {
      const e = geom.edges[i];
      const cp = closestPointOnSegment(x, z, e.x1, e.z1, e.dx, e.dz, e.length);
      const signedToLine = (x - e.x1) * e.nx + (z - e.z1) * e.nz;
      // Only include interior segment hits that actually face the point from inside.
      if (cp.t > 1e-5 && cp.t < 1 - 1e-5) {
        if (inside && signedToLine < -0.25) continue;
        walls.push({
          dist: inside ? Math.max(0, signedToLine) : -cp.dist,
          normalX: e.nx,
          normalZ: e.nz,
          tangentX: e.tx,
          tangentZ: e.tz,
          walled: e.walled,
          contactX: cp.cx,
          contactZ: cp.cz,
        });
      }
    }
    // Reflex vertices act as smooth rounded corner posts.
    for (let i = 0; i < geom.reflexVertices.length; i++) {
      const rv = geom.reflexVertices[i];
      const dx = x - rv.x;
      const dz = z - rv.z;
      const d = Math.hypot(dx, dz);
      // Only active when on the walkable side of the reflex corner.
      const nx = d > 1e-6 ? dx / d : rv.bisectorX;
      const nz = d > 1e-6 ? dz / d : rv.bisectorZ;
      if (d < 12.5 && (nx * rv.bisectorX + nz * rv.bisectorZ) > -0.2) {
        walls.push({
          dist: inside ? d : -d,
          normalX: nx,
          normalZ: nz,
          tangentX: -nz,
          tangentZ: nx,
          walled: rv.walled,
          contactX: rv.x,
          contactZ: rv.z,
          isReflexCorner: true,
        });
      }
    }
    // Fallback if point lies directly on a convex corner ray and skipped above
    if (walls.length === 0) {
      let best = null;
      let bestDist = Infinity;
      for (let i = 0; i < geom.edges.length; i++) {
        const e = geom.edges[i];
        const cp = closestPointOnSegment(x, z, e.x1, e.z1, e.dx, e.dz, e.length);
        if (cp.dist < bestDist) {
          bestDist = cp.dist;
          best = {
            dist: inside ? cp.dist : -cp.dist,
            normalX: e.nx,
            normalZ: e.nz,
            tangentX: e.tx,
            tangentZ: e.tz,
            walled: e.walled,
            contactX: cp.cx,
            contactZ: cp.cz,
          };
        }
      }
      if (best) walls.push(best);
    }
  }

  for (let i = 0; i < geom.obstacles.length; i++) {
    const c = queryObstacleConstraint(geom.obstacles[i], x, z);
    walls.push(c);
  }

  walls.sort((a, b) => a.dist - b.dist);
  return walls;
}

/**
 * Project any escaped point back strictly inside the outer polygon/circle/rect.
 */
function projectPointInsideOuterBoundary(arena, x, z, inset = 0.05) {
  const geom = arena.geometry;
  if (geom.type === 'rect') {
    return {
      x: clamp(x, -geom.halfX + inset, geom.halfX - inset),
      z: clamp(z, -geom.halfZ + inset, geom.halfZ - inset),
    };
  }
  if (geom.type === 'circle') {
    const r = Math.hypot(x, z);
    const maxR = Math.max(0.1, geom.radius - inset);
    if (r <= maxR) return { x, z };
    return { x: (x / r) * maxR, z: (z / r) * maxR };
  }
  if (geom.type === 'polygon') {
    if (isPointInsidePolygon(geom.vertices, x, z)) return { x, z };
    let bestEdge = null;
    let bestCp = null;
    let bestDist = Infinity;
    for (let i = 0; i < geom.edges.length; i++) {
      const e = geom.edges[i];
      const cp = closestPointOnSegment(x, z, e.x1, e.z1, e.dx, e.dz, e.length);
      if (cp.dist < bestDist) {
        bestDist = cp.dist;
        bestEdge = e;
        bestCp = cp;
      }
    }
    if (bestEdge && bestCp) {
      return {
        x: bestCp.cx + bestEdge.nx * inset,
        z: bestCp.cz + bestEdge.nz * inset,
      };
    }
  }
  return { x, z };
}

/**
 * Enforce orientation-aware wall & obstacle clearance for a mouse in any arena.
 * Returns `{ x, z, vx, vz, hitWall, wallFollowDir }`.
 */
export function resolveArenaPosition(
  arena,
  x,
  z,
  vx,
  vz,
  headingX,
  headingZ,
  bodyLenScale = 1.0,
  snoutLenScale = 1.0,
  chonk = 1.0,
  stretch = 0.0,
  uprightReduction = 0.0,
  maxSpeed = 9.2,
  wallFollowDir = 1
) {
  let px = x;
  let pz = z;
  let nvx = vx;
  let nvz = vz;
  let hitWall = false;
  let nextWallFollowDir = wallFollowDir;
  let lastHitNormalX = 0;
  let lastHitNormalZ = 0;
  let lastHitTangentX = 0;
  let lastHitTangentZ = 0;

  const geom = arena.geometry;
  const computeMarginForNormal = (nx, nz) => Math.max(
    1.25,
    computeOrientedWallMargin(
      headingX,
      headingZ,
      nx,
      nz,
      bodyLenScale,
      snoutLenScale,
      chonk,
      stretch
    ) - uprightReduction
  );

  // First ensure point is inside outer boundary
  const projected = projectPointInsideOuterBoundary(arena, px, pz, 0.08);
  if (projected.x !== px || projected.z !== pz) {
    px = projected.x;
    pz = projected.z;
    hitWall = true;
  }

  // Iterative constraint projection handles corridors, doorways, corners, and obstacles cleanly.
  for (let iter = 0; iter < 4; iter++) {
    let iterMoved = false;

    if (geom.type === 'rect') {
      const mPosX = computeMarginForNormal(-1, 0);
      const mNegX = computeMarginForNormal(1, 0);
      const mPosZ = computeMarginForNormal(0, -1);
      const mNegZ = computeMarginForNormal(0, 1);
      if (px > geom.halfX - mPosX) {
        px = geom.halfX - mPosX;
        hitWall = true; iterMoved = true;
        lastHitNormalX = -1; lastHitNormalZ = 0; lastHitTangentX = 0; lastHitTangentZ = 1;
      }
      if (px < -geom.halfX + mNegX) {
        px = -geom.halfX + mNegX;
        hitWall = true; iterMoved = true;
        lastHitNormalX = 1; lastHitNormalZ = 0; lastHitTangentX = 0; lastHitTangentZ = -1;
      }
      if (pz > geom.halfZ - mPosZ) {
        pz = geom.halfZ - mPosZ;
        hitWall = true; iterMoved = true;
        lastHitNormalX = 0; lastHitNormalZ = -1; lastHitTangentX = -1; lastHitTangentZ = 0;
      }
      if (pz < -geom.halfZ + mNegZ) {
        pz = -geom.halfZ + mNegZ;
        hitWall = true; iterMoved = true;
        lastHitNormalX = 0; lastHitNormalZ = 1; lastHitTangentX = 1; lastHitTangentZ = 0;
      }
    } else if (geom.type === 'circle') {
      const r = Math.hypot(px, pz);
      if (r > 1e-6) {
        const nx = -px / r;
        const nz = -pz / r;
        const margin = computeMarginForNormal(nx, nz);
        const maxR = Math.max(0.5, geom.radius - margin);
        if (r > maxR) {
          px = (px / r) * maxR;
          pz = (pz / r) * maxR;
          hitWall = true; iterMoved = true;
          lastHitNormalX = nx; lastHitNormalZ = nz;
          lastHitTangentX = -nz; lastHitTangentZ = nx;
        }
      }
    } else if (geom.type === 'polygon') {
      if (!isPointInsidePolygon(geom.vertices, px, pz)) {
        const insidePt = projectPointInsideOuterBoundary(arena, px, pz, 0.2);
        px = insidePt.x;
        pz = insidePt.z;
        hitWall = true; iterMoved = true;
      }
      for (let i = 0; i < geom.edges.length; i++) {
        const e = geom.edges[i];
        const cp = closestPointOnSegment(px, pz, e.x1, e.z1, e.dx, e.dz, e.length);
        if (cp.t <= 1e-5 || cp.t >= 1 - 1e-5) continue;
        const signed = (px - e.x1) * e.nx + (pz - e.z1) * e.nz;
        if (signed < -0.25) continue;
        const margin = computeMarginForNormal(e.nx, e.nz);
        if (signed < margin) {
          const push = margin - signed;
          px += e.nx * push;
          pz += e.nz * push;
          hitWall = true; iterMoved = true;
          lastHitNormalX = e.nx; lastHitNormalZ = e.nz;
          lastHitTangentX = e.tx; lastHitTangentZ = e.tz;
        }
      }
      for (let i = 0; i < geom.reflexVertices.length; i++) {
        const rv = geom.reflexVertices[i];
        const dx = px - rv.x;
        const dz = pz - rv.z;
        const d = Math.hypot(dx, dz);
        const nx = d > 1e-6 ? dx / d : rv.bisectorX;
        const nz = d > 1e-6 ? dz / d : rv.bisectorZ;
        if ((nx * rv.bisectorX + nz * rv.bisectorZ) <= -0.25) continue;
        const margin = computeMarginForNormal(nx, nz) * 0.92;
        if (d < margin) {
          const push = margin - d;
          px += nx * push;
          pz += nz * push;
          hitWall = true; iterMoved = true;
          lastHitNormalX = nx; lastHitNormalZ = nz;
          lastHitTangentX = -nz; lastHitTangentZ = nx;
        }
      }
    }

    for (let i = 0; i < geom.obstacles.length; i++) {
      const obs = geom.obstacles[i];
      const c = queryObstacleConstraint(obs, px, pz);
      const margin = computeMarginForNormal(c.normalX, c.normalZ) * (obs.type === 'box' ? 0.90 : 0.95);
      if (c.dist < margin) {
        const push = margin - c.dist;
        px += c.normalX * push;
        pz += c.normalZ * push;
        hitWall = true; iterMoved = true;
        lastHitNormalX = c.normalX; lastHitNormalZ = c.normalZ;
        lastHitTangentX = c.tangentX; lastHitTangentZ = c.tangentZ;
      }
    }

    if (!iterMoved) break;
  }

  // Final guarantee that center stayed inside outer polygon after obstacle pushes
  if (!isPointInArena(arena, px, pz, 0.05)) {
    const safe = projectPointInsideOuterBoundary(arena, px, pz, 0.8);
    px = safe.x;
    pz = safe.z;
    hitWall = true;
  }

  if (hitWall && (lastHitNormalX !== 0 || lastHitNormalZ !== 0)) {
    const vn = nvx * lastHitNormalX + nvz * lastHitNormalZ;
    if (vn < 0) {
      // Gently bounce normal component and deflect along the roomier tangent direction
      nvx -= lastHitNormalX * vn * 1.22;
      nvz -= lastHitNormalZ * vn * 1.22;
    }
    const minTurnSpeed = Math.max(maxSpeed * 0.36, 1.8);
    const probeDist = 2.6;
    const clearPos = arenaSignedDistance(
      arena,
      px + lastHitTangentX * probeDist,
      pz + lastHitTangentZ * probeDist
    );
    const clearNeg = arenaSignedDistance(
      arena,
      px - lastHitTangentX * probeDist,
      pz - lastHitTangentZ * probeDist
    );
    const tangentSign = clearPos >= clearNeg ? 1 : -1;
    const vt = nvx * lastHitTangentX + nvz * lastHitTangentZ;
    const desiredVt = tangentSign * Math.max(Math.abs(vt), minTurnSpeed);
    nvx += lastHitTangentX * (desiredVt - vt) * 0.72;
    nvz += lastHitTangentZ * (desiredVt - vt) * 0.72;
    nextWallFollowDir = tangentSign;
  }

  return {
    x: px,
    z: pz,
    vx: nvx,
    vz: nvz,
    hitWall,
    wallFollowDir: nextWallFollowDir,
  };
}

/**
 * Clamp a single 2D tail point inside the arena with surface margin `margin`.
 */
export function clampTailPointToArena(arena, x, z, margin) {
  let px = x;
  let pz = z;
  const geom = arena.geometry;
  const proj = projectPointInsideOuterBoundary(arena, px, pz, margin);
  px = proj.x;
  pz = proj.z;

  for (let iter = 0; iter < 3; iter++) {
    let moved = false;
    if (geom.type === 'rect') {
      const cx = clamp(px, -geom.halfX + margin, geom.halfX - margin);
      const cz = clamp(pz, -geom.halfZ + margin, geom.halfZ - margin);
      if (cx !== px || cz !== pz) { px = cx; pz = cz; moved = true; }
    } else if (geom.type === 'circle') {
      const r = Math.hypot(px, pz);
      const maxR = Math.max(0.05, geom.radius - margin);
      if (r > maxR) { px = (px / r) * maxR; pz = (pz / r) * maxR; moved = true; }
    } else if (geom.type === 'polygon') {
      if (!isPointInsidePolygon(geom.vertices, px, pz)) {
        const insidePt = projectPointInsideOuterBoundary(arena, px, pz, margin);
        px = insidePt.x; pz = insidePt.z; moved = true;
      }
      for (let i = 0; i < geom.edges.length; i++) {
        const e = geom.edges[i];
        const cp = closestPointOnSegment(px, pz, e.x1, e.z1, e.dx, e.dz, e.length);
        if (cp.t <= 1e-5 || cp.t >= 1 - 1e-5) continue;
        const signed = (px - e.x1) * e.nx + (pz - e.z1) * e.nz;
        if (signed >= -0.15 && signed < margin) {
          const push = margin - signed;
          px += e.nx * push;
          pz += e.nz * push;
          moved = true;
        }
      }
      for (let i = 0; i < geom.reflexVertices.length; i++) {
        const rv = geom.reflexVertices[i];
        const dx = px - rv.x;
        const dz = pz - rv.z;
        const d = Math.hypot(dx, dz);
        const nx = d > 1e-6 ? dx / d : rv.bisectorX;
        const nz = d > 1e-6 ? dz / d : rv.bisectorZ;
        if ((nx * rv.bisectorX + nz * rv.bisectorZ) > -0.25 && d < margin) {
          const push = margin - d;
          px += nx * push;
          pz += nz * push;
          moved = true;
        }
      }
    }
    for (let i = 0; i < geom.obstacles.length; i++) {
      const c = queryObstacleConstraint(geom.obstacles[i], px, pz);
      if (c.dist < margin) {
        const push = margin - c.dist;
        px += c.normalX * push;
        pz += c.normalZ * push;
        moved = true;
      }
    }
    if (!moved) break;
  }

  return {
    x: px,
    z: pz,
    shiftX: px - x,
    shiftZ: pz - z,
    hitWall: Math.abs(px - x) > 1e-9 || Math.abs(pz - z) > 1e-9,
  };
}

/**
 * Tail curvature-safe lookahead along curved/polygon arena boundaries & obstacles.
 */
export function computeTailArenaLookahead(
  arena,
  x,
  z,
  hdx,
  hdz,
  margin,
  turnRadius,
  prevHx,
  prevHz,
  naturalSide = 1
) {
  const walls = queryArenaWalls(arena, x, z);
  let best = null;
  let worstDeficit = 0;

  for (let i = 0; i < walls.length; i++) {
    const w = walls[i];
    const clear = w.dist - margin;
    if (clear > turnRadius + 1.1) continue;
    const intoWall = -(hdx * w.normalX + hdz * w.normalZ);
    if (intoWall <= 0.005) continue;

    // Choose tangent direction matching current flow or roomier side
    const dotPrev = prevHx * w.tangentX + prevHz * w.tangentZ;
    const probePos = arenaSignedDistance(
      arena,
      x + w.tangentX * (turnRadius + 0.8),
      z + w.tangentZ * (turnRadius + 0.8)
    );
    const probeNeg = arenaSignedDistance(
      arena,
      x - w.tangentX * (turnRadius + 0.8),
      z - w.tangentZ * (turnRadius + 0.8)
    );
    let sign;
    if (Math.min(probePos, probeNeg) < 0.8 && Math.abs(probePos - probeNeg) > 0.4) {
      sign = probePos >= probeNeg ? 1 : -1;
    } else if (Math.abs(dotPrev) > 0.14) {
      sign = Math.sign(dotPrev);
    } else {
      sign = probePos >= probeNeg ? 1 : (naturalSide >= 0 ? 1 : -1);
    }
    const tx = w.tangentX * sign;
    const tz = w.tangentZ * sign;
    const cosAlign = hdx * tx + hdz * tz;
    const req = turnRadius * (1.0 - cosAlign) + 0.28;
    const deficit = req - clear;
    if (deficit > worstDeficit) {
      worstDeficit = deficit;
      best = {
        triggered: true,
        safeX: tx + w.normalX * 0.08,
        safeZ: tz + w.normalZ * 0.08,
        minClearance: clear,
      };
    }
  }

  return best || { triggered: false, safeX: hdx, safeZ: hdz, minClearance: Infinity };
}

/**
 * Verify a short tail segment (x0,z0)->(x1,z1) stays strictly inside the arena & clear of obstacles.
 */
export function isTailSegmentClearOfArena(arena, x0, z0, x1, z1, margin) {
  const xm = 0.5 * (x0 + x1);
  const zm = 0.5 * (z0 + z1);
  return arenaSignedDistance(arena, x1, z1) >= margin - 1e-5
    && arenaSignedDistance(arena, xm, zm) >= margin - 1e-5;
}

/**
 * Collect candidate heading angles for exact-length tail segment containment fallback.
 */
export function collectArenaTailCandidateAngles(arena, px, pz, horizR, margin, theta0, horizMaxRad) {
  const angles = [];
  const walls = queryArenaWalls(arena, px, pz);
  if (walls.length > 0) {
    const n = walls[0];
    angles.push(Math.atan2(n.normalZ, n.normalX));
    angles.push(Math.atan2(n.tangentZ, n.tangentX));
    angles.push(Math.atan2(-n.tangentZ, -n.tangentX));
  }
  for (const mult of [1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 9, -9, 12, -12, 16, -16]) {
    angles.push(theta0 + mult * horizMaxRad);
  }
  for (let k = 0; k < 16; k++) {
    angles.push((k / 16) * Math.PI * 2 - Math.PI);
  }
  return angles;
}

/**
 * Return cylindrical obstacle discs (e.g., wire holding cups, lab flask) so tail body-wrapping
 * can treat them seamlessly alongside partner mice.
 */
export function getArenaObstacleDiscs(arena) {
  const discs = [];
  for (const obs of arena.geometry.obstacles) {
    if (obs.type === 'circle') {
      discs.push({ x: obs.x, z: obs.z, radius: obs.radius, label: obs.label });
    }
  }
  return discs;
}

/**
 * Find exact wall contact point along `supportDir` for wall-supported rearing.
 * Returns `null` if the approached boundary is a cliff / open arm (`walled === false`).
 */
export function getArenaWallSupportPoint(arena, px, pz, dirX, dirZ) {
  const len = Math.hypot(dirX, dirZ) || 1;
  const ux = dirX / len;
  const uz = dirZ / len;
  const walls = queryArenaWalls(arena, px, pz);
  let best = null;
  let bestScore = Infinity;

  for (const w of walls) {
    if (!w.walled) continue;
    // Facing into this wall means `u · (-n) > 0.35`
    const facing = -(ux * w.normalX + uz * w.normalZ);
    if (facing < 0.35) continue;
    const score = w.dist / facing;
    if (score < bestScore) {
      bestScore = score;
      best = {
        x: w.contactX + w.normalX * 0.08,
        z: w.contactZ + w.normalZ * 0.08,
        normalX: w.normalX,
        normalZ: w.normalZ,
        distance: w.dist,
      };
    }
  }
  return best;
}

/**
 * Anterior whisker wall clearance for Fly-Brain connectome sensory encoding.
 */
export function computeArenaWhiskerClearance(arena, probeX, probeZ) {
  return arenaSignedDistance(arena, probeX, probeZ);
}

/**
 * Check whether the straight corridor segment (x0,z0)->(x1,z1) stays strictly
 * inside walkable space with clearance `margin` away from walls and obstacles.
 */
export function isArenaSegmentClear(arena, x0, z0, x1, z1, margin = 0.9) {
  const dist = Math.hypot(x1 - x0, z1 - z0);
  if (dist < 1e-4) return arenaSignedDistance(arena, x1, z1) >= margin - 1e-4;
  const startClear = arenaSignedDistance(arena, x0, z0);
  const endClear = arenaSignedDistance(arena, x1, z1);
  const steps = Math.max(2, Math.ceil(dist / 0.65));
  for (let i = 1; i < steps; i++) {
    const t = i / steps;
    const sx = x0 + (x1 - x0) * t;
    const sz = z0 + (z1 - z0) * t;
    let req = margin;
    if (t < 0.22) {
      req = Math.min(margin, Math.max(0.28, startClear - 0.12));
    } else if (t > 0.78) {
      req = Math.min(margin, Math.max(0.28, endClear - 0.12));
    }
    if (arenaSignedDistance(arena, sx, sz) < req) {
      return false;
    }
  }
  return true;
}

/**
 * Route an olfactory plume or exploratory target through corridor junctions,
 * doorways, and around solid obstacles using exact shortest-path Dijkstra over
 * the arena's `navNodes`. Direct unobstructed targets pass straight through.
 */
export function computeArenaNavigationRoute(arena, fromX, fromZ, targetX, targetZ, margin = 0.9) {
  const directDist = Math.hypot(targetX - fromX, targetZ - fromZ);
  const navNodes = arena.navNodes || [];
  if (
    arena.legacyRect
    || navNodes.length === 0
    || directDist < 1.2
    || isArenaSegmentClear(arena, fromX, fromZ, targetX, targetZ, margin)
  ) {
    return {
      x: targetX,
      z: targetZ,
      effectiveDist: directDist,
      routed: false,
      path: [{ x: fromX, z: fromZ }, { x: targetX, z: targetZ }],
    };
  }

  const nodes = [
    { x: fromX, z: fromZ },
    ...navNodes,
    { x: targetX, z: targetZ },
  ];
  const n = nodes.length;
  const startIdx = 0;
  const goalIdx = n - 1;

  // Build adjacency weights using line-of-sight clearance
  const adj = Array.from({ length: n }, () => []);
  const connect = (u, v, m) => {
    const d = Math.hypot(nodes[v].x - nodes[u].x, nodes[v].z - nodes[u].z);
    if (d < 1e-5 || isArenaSegmentClear(arena, nodes[u].x, nodes[u].z, nodes[v].x, nodes[v].z, m)) {
      adj[u].push({ to: v, w: d });
      adj[v].push({ to: u, w: d });
      return true;
    }
    return false;
  };

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (i === startIdx && j === goalIdx) continue;
      connect(i, j, margin);
    }
  }

  // Fallback connections if start or goal is tightly wedged near a wall/corner
  const ensureConnected = (idx) => {
    if (adj[idx].length > 0) return;
    let bestNode = -1;
    let bestD = Infinity;
    for (let k = 1; k < goalIdx; k++) {
      const d = Math.hypot(nodes[k].x - nodes[idx].x, nodes[k].z - nodes[idx].z);
      if (isArenaSegmentClear(arena, nodes[idx].x, nodes[idx].z, nodes[k].x, nodes[k].z, 0.25) && d < bestD) {
        bestD = d;
        bestNode = k;
      }
    }
    if (bestNode === -1) {
      for (let k = 1; k < goalIdx; k++) {
        const d = Math.hypot(nodes[k].x - nodes[idx].x, nodes[k].z - nodes[idx].z);
        if (d < bestD) {
          bestD = d;
          bestNode = k;
        }
      }
    }
    if (bestNode !== -1) {
      adj[idx].push({ to: bestNode, w: bestD });
      adj[bestNode].push({ to: idx, w: bestD });
    }
  };
  ensureConnected(startIdx);
  ensureConnected(goalIdx);

  // Exact Dijkstra over small graph (n <= 13)
  const dist = new Float64Array(n).fill(Infinity);
  const prev = new Int32Array(n).fill(-1);
  const visited = new Uint8Array(n);
  dist[startIdx] = 0;

  for (let step = 0; step < n; step++) {
    let u = -1;
    let best = Infinity;
    for (let i = 0; i < n; i++) {
      if (!visited[i] && dist[i] < best) {
        best = dist[i];
        u = i;
      }
    }
    if (u === -1 || u === goalIdx) break;
    visited[u] = 1;
    for (const edge of adj[u]) {
      const alt = dist[u] + edge.w;
      if (alt < dist[edge.to]) {
        dist[edge.to] = alt;
        prev[edge.to] = u;
      }
    }
  }

  if (!Number.isFinite(dist[goalIdx]) || prev[goalIdx] === -1) {
    return {
      x: targetX,
      z: targetZ,
      effectiveDist: directDist,
      routed: false,
      path: [{ x: fromX, z: fromZ }, { x: targetX, z: targetZ }],
    };
  }

  const indices = [];
  for (let cur = goalIdx; cur !== -1; cur = prev[cur]) {
    indices.push(cur);
  }
  indices.reverse();
  const path = indices.map(idx => ({ x: nodes[idx].x, z: nodes[idx].z }));

  // Advance past immediate hub if already practically on top of it and next hop is clear
  let nextIdx = indices.length > 1 ? indices[1] : goalIdx;
  if (indices.length > 2) {
    const hopDist = Math.hypot(nodes[nextIdx].x - fromX, nodes[nextIdx].z - fromZ);
    const afterIdx = indices[2];
    if (
      hopDist < 2.2
      && isArenaSegmentClear(arena, fromX, fromZ, nodes[afterIdx].x, nodes[afterIdx].z, margin * 0.75)
    ) {
      nextIdx = afterIdx;
    }
  }

  return {
    x: nodes[nextIdx].x,
    z: nodes[nextIdx].z,
    effectiveDist: dist[goalIdx],
    routed: nextIdx !== goalIdx,
    path,
  };
}

/**
 * Compute a safe open corridor hub / doorway escape direction when anterior whiskers
 * detect a wall ahead in Fly-Brain mode.
 */
export function computeArenaWallEscapeTarget(arena, x, z) {
  const navNodes = arena.navNodes || [];
  if (arena.legacyRect || navNodes.length === 0) {
    return { x: 0, z: 0 };
  }
  let best = null;
  let bestDist = Infinity;
  let fallback = navNodes[0];
  let fallbackDist = Infinity;
  for (const node of navNodes) {
    const d = Math.hypot(node.x - x, node.z - z);
    if (d < fallbackDist) {
      fallbackDist = d;
      fallback = node;
    }
    if (d < 1.35) continue;
    if (isArenaSegmentClear(arena, x, z, node.x, node.z, 0.55) && d < bestDist) {
      bestDist = d;
      best = node;
    }
  }
  return best ? { x: best.x, z: best.z } : { x: fallback.x, z: fallback.z };
}

/**
 * Classify a 2D coordinate into a semantic neuroscience zone name for HUD/telemetry.
 */
export function classifyArenaZone(arena, x, z) {
  switch (arena.id) {
    case 'open_field':
      return (Math.abs(x) <= 9.0 && Math.abs(z) <= 6.5) ? 'center' : 'periphery';
    case 'home_cage':
      if (x < -7.5 && z < 1.5) return 'nest_zone';
      if (x > 10.5 && Math.abs(z - 0.35) < 4.8) return 'water_port_zone';
      return 'open_bedding';
    case 'circular_open_field': {
      const r = Math.hypot(x, z);
      if (r <= 7.6) return 'center';
      if (r <= 11.4) return 'middle';
      return 'periphery';
    }
    case 'elevated_plus_maze':
      if (Math.abs(x) <= EPM_HW && Math.abs(z) <= EPM_HW) return 'center';
      if (x < -EPM_HW) return 'closed_arm_west';
      if (x > EPM_HW) return 'closed_arm_east';
      if (z < -EPM_HW) return 'open_arm_north';
      return 'open_arm_south';
    case 'y_maze': {
      const r = Math.hypot(x, z);
      if (r <= 5.2) return 'center_hub';
      const angle = Math.atan2(z, x);
      const dA = Math.abs(wrapAngleDelta(angle - (-Math.PI / 2)));
      const dB = Math.abs(wrapAngleDelta(angle - Math.PI / 6));
      const dC = Math.abs(wrapAngleDelta(angle - (5 * Math.PI) / 6));
      if (dA <= dB && dA <= dC) return 'arm_a';
      if (dB <= dC) return 'arm_b';
      return 'arm_c';
    }
    case 't_maze':
      if (z > 6.2) return 'start_box';
      if (z > -4.2) return 'start_stem';
      if (Math.abs(x) <= 4.4) return 'choice_point';
      return x < 0 ? 'left_arm' : 'right_arm';
    case 'three_chamber':
      if (Math.hypot(x + 14.2, z) <= 5.2) return 'social_cup_zone';
      if (Math.hypot(x - 14.2, z) <= 5.2) return 'object_cup_zone';
      if (x < -7.0) return 'left_chamber';
      if (x > 7.0) return 'right_chamber';
      return 'center_chamber';
    case 'light_dark_box':
      if (x < -5.75) return 'dark_compartment';
      if (x <= -5.05) return 'doorway';
      return 'light_compartment';
    case 'barnes_maze': {
      const r = Math.hypot(x, z);
      if (r <= 5.5) return 'center';
      if (z < -4.0 && Math.abs(x) <= -z) return 'escape_quadrant';
      if (r >= 10.5) return 'perimeter_ring';
      return 'inner_table';
    }
    case 'novel_object':
      if (Math.hypot(x + 6.4, z + 2.4) <= 5.2) return 'familiar_object_zone';
      if (Math.hypot(x - 6.4, z + 2.4) <= 5.2) return 'novel_object_zone';
      if (Math.abs(x) <= 8.0 && Math.abs(z) <= 8.0) return 'center';
      return 'periphery';
    default:
      return 'arena';
  }
}

export function formatArenaZoneLabel(zoneId) {
  return String(zoneId || '--').replace(/_/g, ' ').toUpperCase();
}

export function getArenaFlyZones(arena) {
  return arena.flyZones;
}

/**
 * Select a weighted exploratory waypoint inside multi-arm / multi-chamber arenas
 * so autonomous mice visit all arms, doorways, and objects naturally.
 */
export function pickArenaNavigationTarget(arena, currentX, currentZ, rng) {
  const waypoints = arena.navigationWaypoints;
  if (!waypoints || waypoints.length === 0) return null;
  const candidates = waypoints.filter(wp => Math.hypot(wp.x - currentX, wp.z - currentZ) > 4.5);
  const pool = candidates.length > 0 ? candidates : waypoints;
  const totalWeight = pool.reduce((sum, wp) => sum + (wp.weight || 1.0), 0);
  let pick = rng() * totalWeight;
  for (const wp of pool) {
    pick -= wp.weight || 1.0;
    if (pick <= 0) return { x: wp.x, z: wp.z };
  }
  const last = pool[pool.length - 1];
  return { x: last.x, z: last.z };
}

/**
 * Sample a valid, unobstructed spawn pose strictly inside `arena` and away from `existingPositions`.
 */
export function sampleArenaSpawnPose(arena, rng, existingPositions = [], slotIndex = null) {
  const slots = arena.spawnSlots;
  if (Number.isInteger(slotIndex) && slotIndex >= 0 && slotIndex < slots.length) {
    const s = slots[slotIndex];
    return {
      position: [s.position[0], s.position[1]],
      heading: [s.heading[0], s.heading[1]],
    };
  }

  let bestSlot = null;
  let bestClearance = -Infinity;
  for (const s of slots) {
    const [sx, sz] = s.position;
    if (!isPointInArena(arena, sx, sz, 2.2)) continue;
    let minOther = Infinity;
    for (const pos of existingPositions) {
      minOther = Math.min(minOther, Math.hypot(sx - pos[0], sz - pos[1]));
    }
    const jitter = rng() * 0.6;
    const score = (Number.isFinite(minOther) ? minOther : 12.0) + jitter;
    if (score > bestClearance) {
      bestClearance = score;
      bestSlot = s;
    }
  }
  if (bestSlot) {
    return {
      position: [bestSlot.position[0], bestSlot.position[1]],
      heading: [bestSlot.heading[0], bestSlot.heading[1]],
    };
  }
  // Fallback rejection sampling inside arena bounds
  const { minX, maxX, minZ, maxZ } = arena.bounds;
  for (let attempt = 0; attempt < 80; attempt++) {
    const sx = minX + rng() * (maxX - minX);
    const sz = minZ + rng() * (maxZ - minZ);
    if (!isPointInArena(arena, sx, sz, 2.5)) continue;
    const ang = rng() * Math.PI * 2;
    return { position: [sx, sz], heading: [Math.cos(ang), Math.sin(ang)] };
  }
  return { position: [0, 0], heading: [1, 0] };
}

// ═══════════════════════════════════════════════════════════════
// THREE.JS PHOTOREAL 3D APPARATUS BUILDERS (LAZY PER ARENA)
// ═══════════════════════════════════════════════════════════════

function createCanvasCueTexture(THREE, patternType, labelText = '') {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.fillStyle = '#f3f2ee';
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = '#24272b';
  ctx.fillStyle = '#202428';
  ctx.lineWidth = 8;
  ctx.strokeRect(8, 8, 240, 240);

  if (patternType === 'stripes') {
    for (let x = 24; x < 232; x += 38) {
      ctx.fillRect(x, 24, 19, 208);
    }
  } else if (patternType === 'bullseye') {
    ctx.beginPath();
    ctx.arc(128, 128, 86, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#f3f2ee';
    ctx.beginPath();
    ctx.arc(128, 128, 54, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#202428';
    ctx.beginPath();
    ctx.arc(128, 128, 26, 0, Math.PI * 2);
    ctx.fill();
  } else if (patternType === 'checker') {
    const step = 48;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if ((r + c) % 2 === 0) {
          ctx.fillRect(32 + c * step, 32 + r * step, step, step);
        }
      }
    }
  } else if (patternType === 'cross') {
    ctx.fillRect(102, 34, 52, 188);
    ctx.fillRect(34, 102, 188, 52);
  } else if (patternType === 'triangle') {
    ctx.beginPath();
    ctx.moveTo(128, 34);
    ctx.lineTo(220, 214);
    ctx.lineTo(36, 214);
    ctx.closePath();
    ctx.fill();
  } else if (patternType === 'letter' && labelText) {
    ctx.fillStyle = '#e7e8e5';
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = '#6c727a';
    ctx.font = 'bold 148px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, 128, 134);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function createLabSubFloor(THREE, y = -0.06, size = 120, color = 0x1b1e23) {
  const mat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.90,
    metalness: 0.04,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y;
  mesh.receiveShadow = true;
  return mesh;
}

function createExtrudedDeckMesh(THREE, vertices, thickness, material) {
  // Reverse vertices so 2D Shape is CCW in (x, y); rotateX(+PI/2) maps (x, y, z) -> (x, -z, y)
  // so the top deck lies at exact y = 0.0 and thickness extends downward to y = -thickness.
  const shape = new THREE.Shape();
  const rev = [...vertices].reverse();
  shape.moveTo(rev[0][0], rev[0][1]);
  for (let i = 1; i < rev.length; i++) {
    shape.lineTo(rev[i][0], rev[i][1]);
  }
  shape.closePath();
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: thickness,
    bevelEnabled: true,
    bevelThickness: 0.06,
    bevelSize: 0.06,
    bevelSegments: 2,
    curveSegments: 1,
  });
  geo.rotateX(Math.PI / 2);
  // Shift slightly so the top bevel sits flush at y = 0.0
  geo.translate(0, -0.06, 0);
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, material);
  mesh.receiveShadow = true;
  mesh.castShadow = true;
  return mesh;
}

function addWallSegment(THREE, group, x1, z1, x2, z2, nx, nz, height, thickness, wallMat, railMat = null, endPad = null) {
  const len = Math.hypot(x2 - x1, z2 - z1);
  if (len < 1e-4) return;
  const pad = endPad == null ? thickness : endPad;
  const mx = 0.5 * (x1 + x2) - nx * (thickness * 0.5);
  const mz = 0.5 * (z1 + z2) - nz * (thickness * 0.5);
  const angle = Math.atan2(-(z2 - z1), x2 - x1);

  const wallGeo = new THREE.BoxGeometry(len + pad, height, thickness);
  const wall = new THREE.Mesh(wallGeo, wallMat);
  wall.position.set(mx, height * 0.5, mz);
  wall.rotation.y = angle;
  wall.castShadow = true;
  wall.receiveShadow = true;
  group.add(wall);

  if (railMat) {
    const railGeo = new THREE.BoxGeometry(len + pad + 0.04, 0.14, thickness + 0.06);
    const rail = new THREE.Mesh(railGeo, railMat);
    rail.position.set(mx, height + 0.05, mz);
    rail.rotation.y = angle;
    rail.castShadow = true;
    group.add(rail);
  }
}

function addFloorRing(THREE, group, x, z, innerR, outerR, color, opacity = 0.28, y = 0.014) {
  const geo = new THREE.RingGeometry(innerR, outerR, 64);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, y, z);
  mesh.renderOrder = 3;
  group.add(mesh);
  return mesh;
}

function addFloorLine(THREE, group, x1, z1, x2, z2, width, color, opacity = 0.30, y = 0.014) {
  const len = Math.hypot(x2 - x1, z2 - z1);
  const geo = new THREE.PlaneGeometry(len, width);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0.5 * (x1 + x2), y, 0.5 * (z1 + z2));
  mesh.rotation.y = Math.atan2(-(z2 - z1), x2 - x1);
  mesh.renderOrder = 3;
  group.add(mesh);
}

function buildCircularOpenFieldVisual(THREE, spec, shared) {
  const group = new THREE.Group();
  group.name = 'arena-circular-open-field';
  const R = 15.6;
  const wallH = 8.4;

  group.add(createLabSubFloor(THREE, -0.35, 110, 0x202328));

  // Matte clinical cylindrical base plinth (top flush at y = 0.0)
  const floorMat = new THREE.MeshPhysicalMaterial({
    color: 0xf0f2f0,
    map: shared?.labFloorAlbedoMap || null,
    roughnessMap: shared?.labFloorRoughnessMap || null,
    normalMap: shared?.labFloorNormalMap || null,
    normalScale: new THREE.Vector2(0.06, 0.06),
    roughness: 0.84,
    metalness: 0.0,
    clearcoat: 0.04,
  });
  const floorDisc = new THREE.Mesh(new THREE.CylinderGeometry(R + 0.35, R + 0.55, 0.34, 72), floorMat);
  floorDisc.position.y = -0.17;
  floorDisc.receiveShadow = true;
  floorDisc.castShadow = true;
  group.add(floorDisc);

  // Subtle concentric scoring zones (50% center, 75% intermediate) + crosshair ticks
  addFloorRing(THREE, group, 0, 0, 7.52, 7.66, 0x62707a, 0.34, 0.014);
  addFloorRing(THREE, group, 0, 0, 11.34, 11.46, 0x7c868e, 0.24, 0.014);
  addFloorLine(THREE, group, -1.6, 0, 1.6, 0, 0.08, 0x6c7780, 0.30, 0.014);
  addFloorLine(THREE, group, 0, -1.6, 0, 1.6, 0.08, 0x6c7780, 0.30, 0.014);

  // Translucent frosted acrylic drum wall (open-ended cylinder)
  const wallMat = new THREE.MeshPhysicalMaterial({
    color: 0xe4edf2,
    roughness: 0.16,
    metalness: 0.02,
    transparent: true,
    opacity: 0.30,
    clearcoat: 0.45,
    clearcoatRoughness: 0.14,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const drum = new THREE.Mesh(new THREE.CylinderGeometry(R, R, wallH, 72, 1, true), wallMat);
  drum.position.y = wallH * 0.5;
  drum.renderOrder = 10;
  group.add(drum);

  // Anodized aluminum top collar ring + dark silicone base gasket + 4 vertical stainless tie rods
  const metalMat = new THREE.MeshStandardMaterial({ color: 0xc5c9ce, roughness: 0.32, metalness: 0.72 });
  const gasketMat = new THREE.MeshStandardMaterial({ color: 0x3b3f45, roughness: 0.78, metalness: 0.08 });
  const topRim = new THREE.Mesh(new THREE.TorusGeometry(R, 0.16, 12, 72), metalMat);
  topRim.rotation.x = Math.PI / 2;
  topRim.position.y = wallH;
  topRim.castShadow = true;
  group.add(topRim);

  const baseGasket = new THREE.Mesh(new THREE.TorusGeometry(R + 0.02, 0.10, 10, 72), gasketMat);
  baseGasket.rotation.x = Math.PI / 2;
  baseGasket.position.y = 0.05;
  group.add(baseGasket);

  const rodGeo = new THREE.CylinderGeometry(0.11, 0.11, wallH + 0.25, 14);
  for (let i = 0; i < 4; i++) {
    const ang = (i / 4) * Math.PI * 2 + Math.PI / 4;
    const rod = new THREE.Mesh(rodGeo, metalMat);
    rod.position.set(Math.cos(ang) * (R + 0.16), wallH * 0.5, Math.sin(ang) * (R + 0.16));
    rod.castShadow = true;
    group.add(rod);
  }
  return group;
}

function buildElevatedPlusMazeVisual(THREE, spec) {
  const group = new THREE.Group();
  group.name = 'arena-elevated-plus-maze';
  const dropY = -8.5;
  const deckThick = 0.58;
  const closedWallH = 6.2;
  const openLipH = 0.42;

  // Recessed laboratory room floor beneath the elevated maze
  group.add(createLabSubFloor(THREE, dropY, 130, 0x25282e));
  const roomGrid = new THREE.GridHelper(96, 24, 0x383d46, 0x2e333b);
  roomGrid.position.y = dropY + 0.02;
  group.add(roomGrid);

  // Matte slate polypropylene maze deck (top flush at y = 0.0)
  const deckMat = new THREE.MeshStandardMaterial({
    color: 0xced3d8,
    roughness: 0.86,
    metalness: 0.02,
  });
  group.add(createExtrudedDeckMesh(THREE, spec.geometry.vertices, deckThick, deckMat));

  // Center square zone demarcation lines
  const hw = EPM_HW;
  addFloorLine(THREE, group, -hw, -hw, hw, -hw, 0.10, 0x4b5563, 0.42, 0.015);
  addFloorLine(THREE, group, -hw, hw, hw, hw, 0.10, 0x4b5563, 0.42, 0.015);
  addFloorLine(THREE, group, -hw, -hw, -hw, hw, 0.10, 0x4b5563, 0.42, 0.015);
  addFloorLine(THREE, group, hw, -hw, hw, hw, 0.10, 0x4b5563, 0.42, 0.015);

  // Tall matte charcoal closed-arm walls (±X) vs low translucent open-arm safety ledges (±Z)
  const closedWallMat = new THREE.MeshStandardMaterial({
    color: 0x3b4048,
    roughness: 0.62,
    metalness: 0.08,
  });
  const closedRailMat = new THREE.MeshStandardMaterial({
    color: 0x5c636d,
    roughness: 0.45,
    metalness: 0.25,
  });
  const openLipMat = new THREE.MeshPhysicalMaterial({
    color: 0xd9e6ef,
    roughness: 0.22,
    metalness: 0.04,
    transparent: true,
    opacity: 0.65,
    clearcoat: 0.4,
  });

  for (const edge of spec.geometry.edges) {
    if (edge.walled) {
      addWallSegment(
        THREE, group,
        edge.x1, edge.z1, edge.x2, edge.z2,
        edge.nx, edge.nz,
        closedWallH, 0.36, closedWallMat, closedRailMat
      );
    } else {
      addWallSegment(
        THREE, group,
        edge.x1, edge.z1, edge.x2, edge.z2,
        edge.nx, edge.nz,
        openLipH, 0.18, openLipMat, null
      );
    }
  }

  // Brushed stainless-steel tubular support pedestal legs & cross-braces beneath deck
  const steelMat = new THREE.MeshStandardMaterial({ color: 0xb6bcc4, roughness: 0.30, metalness: 0.80 });
  const footMat = new THREE.MeshStandardMaterial({ color: 0x1c1e22, roughness: 0.85, metalness: 0.05 });
  const legH = Math.abs(dropY) - deckThick;
  const legGeo = new THREE.CylinderGeometry(0.32, 0.34, legH, 16);
  const footGeo = new THREE.CylinderGeometry(0.65, 0.72, 0.30, 16);
  const legPositions = [[-13.5, 0], [13.5, 0], [0, -13.5], [0, 13.5]];
  for (const [lx, lz] of legPositions) {
    const leg = new THREE.Mesh(legGeo, steelMat);
    leg.position.set(lx, -deckThick - legH * 0.5, lz);
    leg.castShadow = true;
    group.add(leg);
    const foot = new THREE.Mesh(footGeo, footMat);
    foot.position.set(lx, dropY + 0.15, lz);
    group.add(foot);
  }
  const hubCol = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.65, legH, 20), steelMat);
  hubCol.position.set(0, -deckThick - legH * 0.5, 0);
  hubCol.castShadow = true;
  group.add(hubCol);

  // Horizontal cross-bracing struts
  const strutX = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 27.0, 12), steelMat);
  strutX.rotation.z = Math.PI / 2;
  strutX.position.set(0, dropY + 2.4, 0);
  group.add(strutX);
  const strutZ = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 27.0, 12), steelMat);
  strutZ.rotation.x = Math.PI / 2;
  strutZ.position.set(0, dropY + 2.4, 0);
  group.add(strutZ);

  return group;
}

function buildYMazeVisual(THREE, spec) {
  const group = new THREE.Group();
  group.name = 'arena-y-maze';
  const deckThick = 0.45;
  const wallH = 6.2;

  group.add(createLabSubFloor(THREE, -deckThick, 120, 0x1d2026));

  const deckMat = new THREE.MeshStandardMaterial({
    color: 0xe6e7e4,
    roughness: 0.85,
    metalness: 0.0,
  });
  group.add(createExtrudedDeckMesh(THREE, spec.geometry.vertices, deckThick, deckMat));

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xd2d6dc,
    roughness: 0.54,
    metalness: 0.04,
  });
  const railMat = new THREE.MeshStandardMaterial({
    color: 0x9aa2ab,
    roughness: 0.36,
    metalness: 0.42,
  });

  for (const edge of spec.geometry.edges) {
    addWallSegment(
      THREE, group,
      edge.x1, edge.z1, edge.x2, edge.z2,
      edge.nx, edge.nz,
      wallH, 0.36, wallMat, railMat, 0
    );
  }

  // Brushed corner joint posts at all 9 vertices
  const postMat = new THREE.MeshStandardMaterial({ color: 0xaeb5bd, roughness: 0.34, metalness: 0.65 });
  const postGeo = new THREE.CylinderGeometry(0.24, 0.24, wallH + 0.14, 14);
  for (const [vx, vz] of spec.geometry.vertices) {
    const post = new THREE.Mesh(postGeo, postMat);
    post.position.set(vx, wallH * 0.5, vz);
    group.add(post);
  }

  // Intra-maze spatial cue cards mounted on inner face of each arm's end wall + floor arm markers A/B/C
  const armSpecs = [
    { angle: -Math.PI / 2, cue: 'stripes', label: 'A' },
    { angle: Math.PI / 6, cue: 'bullseye', label: 'B' },
    { angle: (5 * Math.PI) / 6, cue: 'checker', label: 'C' },
  ];
  for (const arm of armSpecs) {
    const ux = Math.cos(arm.angle);
    const uz = Math.sin(arm.angle);
    const cueTex = createCanvasCueTexture(THREE, arm.cue);
    if (cueTex) {
      const cueMat = new THREE.MeshStandardMaterial({ map: cueTex, roughness: 0.72 });
      const cueCard = new THREE.Mesh(new THREE.PlaneGeometry(4.8, 3.6), cueMat);
      cueCard.position.set(ux * 17.95, 3.3, uz * 17.95);
      cueCard.lookAt(0, 3.3, 0);
      group.add(cueCard);
    }
    const floorTex = createCanvasCueTexture(THREE, 'letter', arm.label);
    if (floorTex) {
      const decalMat = new THREE.MeshBasicMaterial({
        map: floorTex,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      });
      const decal = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 2.6), decalMat);
      decal.rotation.x = -Math.PI / 2;
      decal.position.set(ux * 7.6, 0.016, uz * 7.6);
      decal.renderOrder = 3;
      group.add(decal);
    }
  }
  return group;
}

function buildTMazeVisual(THREE, spec) {
  const group = new THREE.Group();
  group.name = 'arena-t-maze';
  const deckThick = 0.45;
  const wallH = 6.2;

  group.add(createLabSubFloor(THREE, -deckThick, 120, 0x1d2026));

  const deckMat = new THREE.MeshStandardMaterial({
    color: 0xe7e8e5,
    roughness: 0.85,
    metalness: 0.0,
  });
  group.add(createExtrudedDeckMesh(THREE, spec.geometry.vertices, deckThick, deckMat));

  // Floor zone demarcation lines (Start Box line + Choice Point boundaries)
  addFloorLine(THREE, group, -4.3, 6.2, 4.3, 6.2, 0.12, 0x62707a, 0.40, 0.015);
  addFloorLine(THREE, group, -4.3, -4.2, 4.3, -4.2, 0.10, 0x62707a, 0.34, 0.015);
  addFloorLine(THREE, group, -4.4, -12.9, -4.4, -4.2, 0.10, 0x62707a, 0.34, 0.015);
  addFloorLine(THREE, group, 4.4, -12.9, 4.4, -4.2, 0.10, 0x62707a, 0.34, 0.015);

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xcfd4da,
    roughness: 0.55,
    metalness: 0.05,
  });
  const railMat = new THREE.MeshStandardMaterial({
    color: 0x9199a3,
    roughness: 0.38,
    metalness: 0.45,
  });

  for (const edge of spec.geometry.edges) {
    addWallSegment(
      THREE, group,
      edge.x1, edge.z1, edge.x2, edge.z2,
      edge.nx, edge.nz,
      wallH, 0.36, wallMat, railMat, 0
    );
  }

  // Brushed corner joint posts at all 8 T-maze vertices for crisp non-overlapping corners
  const tPostMat = new THREE.MeshStandardMaterial({ color: 0xaeb5bd, roughness: 0.34, metalness: 0.65 });
  const tPostGeo = new THREE.CylinderGeometry(0.24, 0.24, wallH + 0.14, 14);
  for (const [vx, vz] of spec.geometry.vertices) {
    const post = new THREE.Mesh(tPostGeo, tPostMat);
    post.position.set(vx, wallH * 0.5, vz);
    group.add(post);
  }

  // Open/raised guillotine start-box gate frame at z = +6.2
  const trackMat = new THREE.MeshStandardMaterial({ color: 0x6c747e, roughness: 0.35, metalness: 0.72 });
  for (const sx of [-4.45, 4.45]) {
    const channel = new THREE.Mesh(new THREE.BoxGeometry(0.28, 10.6, 0.42), trackMat);
    channel.position.set(sx, 5.3, 6.2);
    channel.castShadow = true;
    group.add(channel);
  }
  const topBar = new THREE.Mesh(new THREE.BoxGeometry(9.2, 0.28, 0.44), trackMat);
  topBar.position.set(0, 10.6, 6.2);
  group.add(topBar);
  const raisedGateMat = new THREE.MeshPhysicalMaterial({
    color: 0xd8eef9,
    roughness: 0.12,
    transparent: true,
    opacity: 0.35,
    clearcoat: 0.5,
    depthWrite: false,
  });
  const raisedGate = new THREE.Mesh(new THREE.BoxGeometry(8.6, 4.8, 0.14), raisedGateMat);
  raisedGate.position.set(0, 7.9, 6.2);
  raisedGate.renderOrder = 11;
  group.add(raisedGate);

  // Recessed stainless-steel reward food wells + tiny sucrose pellets at Left & Right goal ends
  const wellMat = new THREE.MeshStandardMaterial({ color: 0xc2c7ce, roughness: 0.28, metalness: 0.82 });
  const pelletMat = new THREE.MeshStandardMaterial({ color: 0xe6bf68, roughness: 0.72, metalness: 0.0 });
  for (const wx of [-16.2, 16.2]) {
    const wellOuter = new THREE.Mesh(new THREE.CylinderGeometry(1.05, 1.15, 0.26, 24), wellMat);
    wellOuter.position.set(wx, 0.13, -8.6);
    wellOuter.castShadow = true;
    wellOuter.receiveShadow = true;
    group.add(wellOuter);
    const wellInner = new THREE.Mesh(
      new THREE.CylinderGeometry(0.78, 0.78, 0.10, 20),
      new THREE.MeshStandardMaterial({ color: 0x454a52, roughness: 0.65, metalness: 0.4 })
    );
    wellInner.position.set(wx, 0.23, -8.6);
    group.add(wellInner);
    for (const [dx, dz] of [[-0.22, -0.15], [0.24, -0.05], [-0.04, 0.24]]) {
      const pellet = new THREE.Mesh(new THREE.SphereGeometry(0.14, 10, 8), pelletMat);
      pellet.scale.set(1.0, 0.72, 1.0);
      pellet.position.set(wx + dx, 0.32, -8.6 + dz);
      group.add(pellet);
    }
  }

  return group;
}

function buildWireHoldingCup(THREE, x, z, accentColor, isSocial) {
  const cup = new THREE.Group();
  cup.position.set(x, 0, z);
  const R = 2.22;
  const H = 5.6;

  const baseMat = new THREE.MeshStandardMaterial({ color: 0x6f757e, roughness: 0.46, metalness: 0.55 });
  const wireMat = new THREE.MeshStandardMaterial({ color: 0xd0d5dc, roughness: 0.26, metalness: 0.84 });
  const accentMat = new THREE.MeshStandardMaterial({ color: accentColor, roughness: 0.52, metalness: 0.10 });

  // Weighted base plate & top cap
  const base = new THREE.Mesh(new THREE.CylinderGeometry(R, R + 0.08, 0.32, 28), baseMat);
  base.position.y = 0.16;
  base.castShadow = true;
  base.receiveShadow = true;
  cup.add(base);

  const lid = new THREE.Mesh(new THREE.CylinderGeometry(R + 0.08, R, 0.36, 28), baseMat);
  lid.position.y = H + 0.18;
  lid.castShadow = true;
  cup.add(lid);

  const tagDot = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.75, 0.08, 20), accentMat);
  tagDot.position.y = H + 0.38;
  cup.add(tagDot);

  // 18 vertical steel wire bars + mid-height reinforcing ring
  const barGeo = new THREE.CylinderGeometry(0.055, 0.055, H, 8);
  const barCount = 18;
  for (let i = 0; i < barCount; i++) {
    const a = (i / barCount) * Math.PI * 2;
    const bar = new THREE.Mesh(barGeo, wireMat);
    bar.position.set(Math.cos(a) * (R - 0.10), H * 0.5, Math.sin(a) * (R - 0.10));
    bar.castShadow = true;
    cup.add(bar);
  }
  for (const ry of [H * 0.35, H * 0.68]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(R - 0.10, 0.055, 8, 32), wireMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = ry;
    cup.add(ring);
  }

  // Distinct interior stimulus cue so Left (Social stimulus) vs Right (Novel Object) read immediately
  if (isSocial) {
    // Soft bedding nestlet pad + subtle warm stimulus silhouette inside the social cup
    const pad = new THREE.Mesh(
      new THREE.CylinderGeometry(1.55, 1.65, 0.24, 18),
      new THREE.MeshStandardMaterial({ color: 0xe3d6be, roughness: 0.92 })
    );
    pad.position.y = 0.42;
    cup.add(pad);
    const plushBody = new THREE.Mesh(
      new THREE.SphereGeometry(0.95, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0xb98a62, roughness: 0.85 })
    );
    plushBody.scale.set(0.85, 0.72, 1.22);
    plushBody.position.set(0, 1.05, 0.1);
    cup.add(plushBody);
  } else {
    // Clean geometric acrylic prism inside the non-social object cup
    const objMesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(1.05, 0),
      new THREE.MeshStandardMaterial({ color: 0x4b7d8e, roughness: 0.36, metalness: 0.14 })
    );
    objMesh.position.y = 1.45;
    cup.add(objMesh);
  }
  return cup;
}

function buildThreeChamberVisual(THREE, spec, shared) {
  const group = new THREE.Group();
  group.name = 'arena-three-chamber';
  const hx = spec.geometry.halfX; // 21.0
  const hz = spec.geometry.halfZ; // 11.2
  const wallH = 7.8;

  group.add(createLabSubFloor(THREE, -0.35, 130, 0x1e2126));

  // Crisp non-glare rectangular floor deck (top flush at y = 0.0)
  const floorMat = new THREE.MeshPhysicalMaterial({
    color: 0xf1f3f2,
    map: shared?.labFloorAlbedoMap || null,
    roughnessMap: shared?.labFloorRoughnessMap || null,
    normalMap: shared?.labFloorNormalMap || null,
    normalScale: new THREE.Vector2(0.05, 0.05),
    roughness: 0.84,
    metalness: 0.0,
  });
  const deck = new THREE.Mesh(new THREE.BoxGeometry(hx * 2 + 0.8, 0.34, hz * 2 + 0.8), floorMat);
  deck.position.y = -0.17;
  deck.receiveShadow = true;
  group.add(deck);

  // Subtle floor interaction rings around Left (Social) and Right (Novel Object) cups
  addFloorRing(THREE, group, -14.2, 0, 4.95, 5.15, 0xd89c3c, 0.42, 0.015);
  addFloorRing(THREE, group, 14.2, 0, 4.95, 5.15, 0x4f8498, 0.42, 0.015);
  addFloorLine(THREE, group, -7.0, -3.7, -7.0, 3.7, 0.10, 0x7b858e, 0.32, 0.014);
  addFloorLine(THREE, group, 7.0, -3.7, 7.0, 3.7, 0.10, 0x7b858e, 0.32, 0.014);

  // Outer clear polycarbonate walls + anodized frame rails
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0xdcf0f9,
    roughness: 0.09,
    metalness: 0.0,
    transparent: true,
    opacity: 0.24,
    clearcoat: 0.42,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const railMat = new THREE.MeshStandardMaterial({ color: 0xbac0c8, roughness: 0.34, metalness: 0.60 });

  const outerEdges = [
    [-hx, -hz, hx, -hz, 0, 1],
    [hx, -hz, hx, hz, -1, 0],
    [hx, hz, -hx, hz, 0, -1],
    [-hx, hz, -hx, -hz, 1, 0],
  ];
  for (const [x1, z1, x2, z2, nx, nz] of outerEdges) {
    addWallSegment(THREE, group, x1, z1, x2, z2, nx, nz, wallH, 0.24, glassMat, railMat);
  }

  // Two clear polycarbonate internal dividers at x = ±7.0 with framed central doorways (z ∈ [-3.7, +3.7])
  const dividerMat = new THREE.MeshPhysicalMaterial({
    color: 0xcbe4f2,
    roughness: 0.12,
    metalness: 0.02,
    transparent: true,
    opacity: 0.42,
    clearcoat: 0.50,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const jambMat = new THREE.MeshStandardMaterial({ color: 0x606872, roughness: 0.36, metalness: 0.68 });
  for (const divX of [-7.0, 7.0]) {
    // North & South solid partition panels
    for (const [zMin, zMax] of [[-hz, -3.7], [3.7, hz]]) {
      const segLen = zMax - zMin;
      const panel = new THREE.Mesh(new THREE.BoxGeometry(0.46, wallH, segLen), dividerMat);
      panel.position.set(divX, wallH * 0.5, 0.5 * (zMin + zMax));
      panel.renderOrder = 11;
      group.add(panel);
      const topRail = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.16, segLen + 0.06), railMat);
      topRail.position.set(divX, wallH + 0.06, 0.5 * (zMin + zMax));
      group.add(topRail);
    }
    // Doorway vertical jambs + header lintel + raised clear sliding door above opening
    for (const jz of [-3.7, 3.7]) {
      const jamb = new THREE.Mesh(new THREE.BoxGeometry(0.62, wallH + 1.8, 0.24), jambMat);
      jamb.position.set(divX, (wallH + 1.8) * 0.5, jz);
      jamb.castShadow = true;
      group.add(jamb);
    }
    const header = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.30, 7.64), jambMat);
    header.position.set(divX, 4.45, 0);
    group.add(header);
    const raisedDoor = new THREE.Mesh(new THREE.BoxGeometry(0.18, 4.4, 7.2), glassMat);
    raisedDoor.position.set(divX, 6.8, 0);
    raisedDoor.renderOrder = 12;
    group.add(raisedDoor);
  }

  // Two cylindrical wire holding cups (Left = Social stimulus, Right = Novel object)
  group.add(buildWireHoldingCup(THREE, -14.2, 0.0, 0xe0a23b, true));
  group.add(buildWireHoldingCup(THREE, 14.2, 0.0, 0x4b879e, false));

  return group;
}

function buildLightDarkBoxVisual(THREE, spec) {
  const group = new THREE.Group();
  group.name = 'arena-light-dark-box';
  const hx = spec.geometry.halfX; // 17.6
  const hz = spec.geometry.halfZ; // 11.2
  const divX = -5.4;
  const wallH = 7.4;

  group.add(createLabSubFloor(THREE, -0.35, 120, 0x1c1f24));

  // Light compartment floor (x ∈ [-5.4, +17.6], crisp bright matte white)
  const lightW = hx - divX; // 23.0
  const lightFloorMat = new THREE.MeshStandardMaterial({ color: 0xf7f8f6, roughness: 0.80, metalness: 0.0 });
  const lightDeck = new THREE.Mesh(new THREE.BoxGeometry(lightW + 0.4, 0.34, hz * 2 + 0.6), lightFloorMat);
  lightDeck.position.set(divX + lightW * 0.5 + 0.2, -0.17, 0);
  lightDeck.receiveShadow = true;
  group.add(lightDeck);

  // Dark compartment floor (x ∈ [-17.6, -5.4], matte charcoal black)
  const darkW = divX - (-hx); // 12.2
  const darkFloorMat = new THREE.MeshStandardMaterial({ color: 0x24262b, roughness: 0.88, metalness: 0.04 });
  const darkDeck = new THREE.Mesh(new THREE.BoxGeometry(darkW + 0.4, 0.34, hz * 2 + 0.6), darkFloorMat);
  darkDeck.position.set(-hx + darkW * 0.5 - 0.2, -0.17, 0);
  darkDeck.receiveShadow = true;
  group.add(darkDeck);

  // Materials for Light walls vs Dark walls (front +Z light wall is translucent viewing acrylic so mice remain fully visible)
  const lightWallMat = new THREE.MeshStandardMaterial({ color: 0xeeeff0, roughness: 0.46, metalness: 0.02 });
  const lightFrontGlassMat = new THREE.MeshPhysicalMaterial({
    color: 0xe6f0f5,
    roughness: 0.14,
    metalness: 0.0,
    transparent: true,
    opacity: 0.24,
    clearcoat: 0.42,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const lightRailMat = new THREE.MeshStandardMaterial({ color: 0xd0d4da, roughness: 0.40, metalness: 0.25 });
  const darkWallMat = new THREE.MeshStandardMaterial({ color: 0x1e2025, roughness: 0.56, metalness: 0.08 });
  const darkRailMat = new THREE.MeshStandardMaterial({ color: 0x3b3f47, roughness: 0.42, metalness: 0.35 });

  // Light compartment outer walls (North opaque, East opaque, South clear viewing acrylic — zero overlap at divX)
  addWallSegment(THREE, group, divX, -hz, hx, -hz, 0, 1, wallH, 0.36, lightWallMat, lightRailMat, 0);
  addWallSegment(THREE, group, hx, -hz, hx, hz, -1, 0, wallH, 0.36, lightWallMat, lightRailMat, 0.72);
  addWallSegment(THREE, group, hx, hz, divX, hz, 0, -1, wallH, 0.22, lightFrontGlassMat, lightRailMat, 0);

  // Dark compartment outer walls (North, West, South — clean butt joint at divX with no z-fighting)
  addWallSegment(THREE, group, -hx, -hz, divX, -hz, 0, 1, wallH, 0.36, darkWallMat, darkRailMat, 0);
  addWallSegment(THREE, group, -hx, hz, -hx, -hz, 1, 0, wallH, 0.36, darkWallMat, darkRailMat, 0.72);
  addWallSegment(THREE, group, divX, hz, -hx, hz, 0, -1, wallH, 0.36, darkWallMat, darkRailMat, 0);

  // Dual-faced transition wall at x = -5.4 with framed doorway (z ∈ [-3.7, +3.7], height 4.1)
  for (const [zMin, zMax] of [[-hz, -3.7], [3.7, hz]]) {
    const segLen = zMax - zMin;
    const zMid = 0.5 * (zMin + zMax);
    const darkSide = new THREE.Mesh(new THREE.BoxGeometry(0.32, wallH, segLen), darkWallMat);
    darkSide.position.set(divX - 0.16, wallH * 0.5, zMid);
    darkSide.castShadow = true;
    darkSide.receiveShadow = true;
    group.add(darkSide);
    const lightSide = new THREE.Mesh(new THREE.BoxGeometry(0.32, wallH, segLen), lightWallMat);
    lightSide.position.set(divX + 0.16, wallH * 0.5, zMid);
    lightSide.castShadow = true;
    lightSide.receiveShadow = true;
    group.add(lightSide);
  }
  // Lintel above the mouse transition opening
  const lintelH = wallH - 4.1;
  const lintelDark = new THREE.Mesh(new THREE.BoxGeometry(0.32, lintelH, 7.4), darkWallMat);
  lintelDark.position.set(divX - 0.16, 4.1 + lintelH * 0.5, 0);
  lintelDark.castShadow = true;
  group.add(lintelDark);
  const lintelLight = new THREE.Mesh(new THREE.BoxGeometry(0.32, lintelH, 7.4), lightWallMat);
  lintelLight.position.set(divX + 0.16, 4.1 + lintelH * 0.5, 0);
  lintelLight.castShadow = true;
  group.add(lintelLight);

  // Brushed doorway frame trim
  const frameMat = new THREE.MeshStandardMaterial({ color: 0x8e96a0, roughness: 0.34, metalness: 0.72 });
  for (const jz of [-3.7, 3.7]) {
    const post = new THREE.Mesh(new THREE.BoxGeometry(0.76, 4.2, 0.16), frameMat);
    post.position.set(divX, 2.1, jz);
    group.add(post);
  }
  const sillTop = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.16, 7.56), frameMat);
  sillTop.position.set(divX, 4.15, 0);
  group.add(sillTop);

  // Translucent smoked cutaway acrylic canopy over the Dark compartment
  // (conveys darkened shelter while keeping top-down/oblique mouse visibility clear!)
  const canopyMat = new THREE.MeshPhysicalMaterial({
    color: 0x16181d,
    roughness: 0.18,
    metalness: 0.06,
    transparent: true,
    opacity: 0.32,
    clearcoat: 0.65,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const canopy = new THREE.Mesh(new THREE.BoxGeometry(darkW + 0.3, 0.16, hz * 2 + 0.3), canopyMat);
  canopy.position.set(-hx + darkW * 0.5, wallH + 0.10, 0);
  canopy.renderOrder = 14;
  group.add(canopy);

  return group;
}

function buildBarnesMazeVisual(THREE) {
  const group = new THREE.Group();
  group.name = 'arena-barnes-maze';
  const tableR = 16.0;
  const deckThick = 0.62;
  const dropY = -8.5;

  // Recessed laboratory room floor below the Barnes table
  group.add(createLabSubFloor(THREE, dropY, 140, 0x24272d));
  const roomGrid = new THREE.GridHelper(104, 26, 0x373c45, 0x2d323a);
  roomGrid.position.y = dropY + 0.02;
  group.add(roomGrid);

  // Elevated circular white tabletop (top flush at y = 0.0)
  const tableMat = new THREE.MeshStandardMaterial({
    color: 0xf0f1ee,
    roughness: 0.80,
    metalness: 0.0,
  });
  const table = new THREE.Mesh(new THREE.CylinderGeometry(tableR, tableR, deckThick, 72), tableMat);
  table.position.y = -deckThick * 0.5;
  table.receiveShadow = true;
  table.castShadow = true;
  group.add(table);

  // Subtle center start-ring & perimeter ring markings
  addFloorRing(THREE, group, 0, 0, 5.42, 5.54, 0x808992, 0.30, 0.012);
  addFloorRing(THREE, group, 0, 0, 11.45, 11.55, 0x9aa2aa, 0.22, 0.012);

  // 18 evenly spaced circular perimeter escape holes
  const holeDarkMat = new THREE.MeshStandardMaterial({ color: 0x101216, roughness: 0.96, metalness: 0.0 });
  const holeRimMat = new THREE.MeshStandardMaterial({ color: 0xd0d4d8, roughness: 0.55, metalness: 0.1 });
  const escapeRimMat = new THREE.MeshStandardMaterial({ color: 0x5c8f73, roughness: 0.50, metalness: 0.15 });
  const holeDiscGeo = new THREE.CircleGeometry(BARNES_HOLE_RADIUS, 28);
  holeDiscGeo.rotateX(-Math.PI / 2);
  const holeRimGeo = new THREE.TorusGeometry(BARNES_HOLE_RADIUS + 0.04, 0.075, 10, 28);
  holeRimGeo.rotateX(Math.PI / 2);
  const holeWellGeo = new THREE.CylinderGeometry(
    BARNES_HOLE_RADIUS + 0.06,
    BARNES_HOLE_RADIUS + 0.06,
    0.55,
    24,
    1,
    true
  );

  for (const hole of BARNES_MAZE_HOLES) {
    const disc = new THREE.Mesh(holeDiscGeo, holeDarkMat);
    disc.position.set(hole.x, 0.010, hole.z);
    disc.renderOrder = 3;
    group.add(disc);

    const rim = new THREE.Mesh(holeRimGeo, hole.isEscape ? escapeRimMat : holeRimMat);
    rim.position.set(hole.x, 0.015, hole.z);
    group.add(rim);

    const well = new THREE.Mesh(holeWellGeo, holeDarkMat);
    well.position.set(hole.x, -deckThick - 0.15, hole.z);
    group.add(well);
  }

  // Visible dark under-table escape box slung under Hole #0 (North, x=0, z=-13.6)
  const escapeBoxMat = new THREE.MeshStandardMaterial({ color: 0x262b33, roughness: 0.65, metalness: 0.18 });
  const escapeBox = new THREE.Mesh(new THREE.BoxGeometry(4.6, 3.0, 4.2), escapeBoxMat);
  escapeBox.position.set(0, -deckThick - 1.5, -BARNES_HOLE_RING_RADIUS);
  escapeBox.castShadow = true;
  group.add(escapeBox);

  // Central pedestal column + 4 splayed tripod legs on lab floor
  const steelMat = new THREE.MeshStandardMaterial({ color: 0xb6bcc4, roughness: 0.30, metalness: 0.80 });
  const colH = Math.abs(dropY) - deckThick;
  const column = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.35, colH, 24), steelMat);
  column.position.set(0, -deckThick - colH * 0.5, 0);
  column.castShadow = true;
  group.add(column);
  const baseHub = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.5, 0.5, 24), steelMat);
  baseHub.position.set(0, dropY + 0.25, 0);
  group.add(baseHub);
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
    const leg = new THREE.Mesh(new THREE.BoxGeometry(9.0, 0.42, 0.58), steelMat);
    leg.position.set(Math.cos(a) * 4.5, dropY + 0.28, Math.sin(a) * 4.5);
    leg.rotation.y = -a;
    leg.castShadow = true;
    group.add(leg);
  }

  // 4 Distal extra-maze spatial cue cards mounted on tripod stands around the perimeter
  const cueDefs = [
    { angle: -Math.PI / 2, pattern: 'triangle' },
    { angle: 0, pattern: 'stripes' },
    { angle: Math.PI / 2, pattern: 'cross' },
    { angle: Math.PI, pattern: 'bullseye' },
  ];
  const standMat = new THREE.MeshStandardMaterial({ color: 0x3c414a, roughness: 0.50, metalness: 0.60 });
  const cueR = 22.4;
  for (const def of cueDefs) {
    const ux = Math.cos(def.angle);
    const uz = Math.sin(def.angle);
    const cx = ux * cueR;
    const cz = uz * cueR;
    // Place support pole cleanly behind the placard backing so it never intersects the printed cue face
    const poleH = Math.abs(dropY) + 1.0;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.18, poleH, 10), standMat);
    pole.position.set(cx + ux * 0.24, dropY + poleH * 0.5, cz + uz * 0.24);
    group.add(pole);

    const cardBack = new THREE.Mesh(new THREE.BoxGeometry(5.6, 5.6, 0.16), standMat);
    cardBack.position.set(cx, 3.2, cz);
    cardBack.lookAt(0, 3.2, 0);
    group.add(cardBack);

    const tex = createCanvasCueTexture(THREE, def.pattern);
    if (tex) {
      const cardFace = new THREE.Mesh(
        new THREE.PlaneGeometry(5.2, 5.2),
        new THREE.MeshStandardMaterial({ map: tex, roughness: 0.72 })
      );
      cardFace.position.set(cx - ux * 0.11, 3.2, cz - uz * 0.11);
      cardFace.lookAt(0, 3.2, 0);
      group.add(cardFace);
    }
  }
  return group;
}

function buildNovelObjectVisual(THREE, spec, shared) {
  const group = new THREE.Group();
  group.name = 'arena-novel-object';
  const hx = spec.geometry.halfX; // 15.6
  const hz = spec.geometry.halfZ; // 15.6
  const wallH = 8.0;

  group.add(createLabSubFloor(THREE, -0.35, 120, 0x1e2126));

  // Matte low-glare square floor deck (top flush at y = 0.0)
  const floorMat = new THREE.MeshPhysicalMaterial({
    color: 0xf1f2ef,
    map: shared?.labFloorAlbedoMap || null,
    roughnessMap: shared?.labFloorRoughnessMap || null,
    normalMap: shared?.labFloorNormalMap || null,
    normalScale: new THREE.Vector2(0.055, 0.055),
    roughness: 0.84,
    metalness: 0.0,
  });
  const deck = new THREE.Mesh(new THREE.BoxGeometry(hx * 2 + 0.8, 0.34, hz * 2 + 0.8), floorMat);
  deck.position.y = -0.17;
  deck.receiveShadow = true;
  group.add(deck);

  // Subtle floor investigation-zone halos around Familiar & Novel objects + center square
  addFloorRing(THREE, group, -6.4, -2.4, 4.95, 5.12, 0x458096, 0.36, 0.015);
  addFloorRing(THREE, group, 6.4, -2.4, 4.95, 5.12, 0xc76d44, 0.36, 0.015);
  addFloorLine(THREE, group, -8.0, -8.0, 8.0, -8.0, 0.08, 0x7c848c, 0.22, 0.014);
  addFloorLine(THREE, group, -8.0, 8.0, 8.0, 8.0, 0.08, 0x7c848c, 0.22, 0.014);
  addFloorLine(THREE, group, -8.0, -8.0, -8.0, 8.0, 0.08, 0x7c848c, 0.22, 0.014);
  addFloorLine(THREE, group, 8.0, -8.0, 8.0, 8.0, 0.08, 0x7c848c, 0.22, 0.014);

  // Matte light-gray acrylic outer walls (front wall slightly translucent so oblique view stays unobstructed)
  const backWallMat = new THREE.MeshStandardMaterial({ color: 0xd9ddd9, roughness: 0.62, metalness: 0.02 });
  const frontGlassMat = new THREE.MeshPhysicalMaterial({
    color: 0xe4eef4,
    roughness: 0.14,
    transparent: true,
    opacity: 0.24,
    clearcoat: 0.4,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const railMat = new THREE.MeshStandardMaterial({ color: 0xadb4bc, roughness: 0.36, metalness: 0.52 });

  addWallSegment(THREE, group, -hx, -hz, hx, -hz, 0, 1, wallH, 0.34, backWallMat, railMat);
  addWallSegment(THREE, group, -hx, hz, -hx, -hz, 1, 0, wallH, 0.34, backWallMat, railMat);
  addWallSegment(THREE, group, hx, -hz, hx, hz, -1, 0, wallH, 0.34, backWallMat, railMat);
  addWallSegment(THREE, group, hx, hz, -hx, hz, 0, -1, wallH, 0.22, frontGlassMat, railMat);

  // Familiar Object (-6.4, -2.4): Weighted laboratory reagent bottle / cylindrical glass flask
  const familiar = new THREE.Group();
  familiar.position.set(-6.4, 0, -2.4);
  const bottleBodyMat = new THREE.MeshPhysicalMaterial({
    color: 0x5c9db3,
    roughness: 0.18,
    metalness: 0.05,
    clearcoat: 0.85,
    clearcoatRoughness: 0.12,
  });
  const bottleCapMat = new THREE.MeshStandardMaterial({ color: 0x24566a, roughness: 0.42, metalness: 0.15 });
  const bottleBase = new THREE.Mesh(new THREE.CylinderGeometry(1.98, 2.02, 3.6, 28), bottleBodyMat);
  bottleBase.position.y = 1.8;
  bottleBase.castShadow = true;
  bottleBase.receiveShadow = true;
  familiar.add(bottleBase);
  const bottleShoulder = new THREE.Mesh(new THREE.CylinderGeometry(1.12, 1.98, 0.85, 28), bottleBodyMat);
  bottleShoulder.position.y = 4.02;
  bottleShoulder.castShadow = true;
  familiar.add(bottleShoulder);
  const bottleCap = new THREE.Mesh(new THREE.CylinderGeometry(1.22, 1.22, 0.92, 24), bottleCapMat);
  bottleCap.position.y = 4.88;
  bottleCap.castShadow = true;
  familiar.add(bottleCap);
  // Subtle white measurement graduation bands
  const bandMat = new THREE.MeshStandardMaterial({ color: 0xf0f4f6, roughness: 0.65 });
  for (const by of [1.2, 2.0, 2.8]) {
    const band = new THREE.Mesh(new THREE.TorusGeometry(2.01, 0.035, 6, 28), bandMat);
    band.rotation.x = Math.PI / 2;
    band.position.y = by;
    familiar.add(band);
  }
  group.add(familiar);

  // Novel Object (+6.4, -2.4): Contrasting terracotta & cream stepped geometric block tower
  const novel = new THREE.Group();
  novel.position.set(6.4, 0, -2.4);
  const tier1Mat = new THREE.MeshStandardMaterial({ color: 0xc9663d, roughness: 0.48, metalness: 0.04 });
  const tier2Mat = new THREE.MeshStandardMaterial({ color: 0xe6aa4f, roughness: 0.46, metalness: 0.04 });
  const tier3Mat = new THREE.MeshStandardMaterial({ color: 0xa6462b, roughness: 0.48, metalness: 0.06 });

  const tier1 = new THREE.Mesh(new THREE.BoxGeometry(3.85, 1.75, 3.85), tier1Mat);
  tier1.position.y = 0.875;
  tier1.castShadow = true;
  tier1.receiveShadow = true;
  novel.add(tier1);

  const tier2 = new THREE.Mesh(new THREE.BoxGeometry(2.75, 1.65, 2.75), tier2Mat);
  tier2.position.y = 2.575;
  tier2.rotation.y = Math.PI / 4;
  tier2.castShadow = true;
  tier2.receiveShadow = true;
  novel.add(tier2);

  const tier3 = new THREE.Mesh(new THREE.BoxGeometry(1.65, 1.45, 1.65), tier3Mat);
  tier3.position.y = 4.125;
  tier3.castShadow = true;
  novel.add(tier3);

  // Small tactile Lego-style studs on top
  const studGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.22, 12);
  for (const [sx, sz] of [[-0.42, -0.42], [0.42, -0.42], [-0.42, 0.42], [0.42, 0.42]]) {
    const stud = new THREE.Mesh(studGeo, tier2Mat);
    stud.position.set(sx, 4.94, sz);
    stud.castShadow = true;
    novel.add(stud);
  }
  group.add(novel);

  return group;
}

/**
 * Lazily construct the photoreal Three.js apparatus group for any non-legacy arena.
 * Uses an isolated deterministic PRNG so subject simulation streams remain untouched.
 */
export function createArenaVisual(THREE, arenaSpec, context = {}) {
  const seed = ((context.simulationSeed ?? 20260710) ^ 0x3C6EF372) >>> 0;
  const _arenaRng = createSeededRandom(seed);
  void _arenaRng;

  switch (arenaSpec.id) {
    case 'circular_open_field':
      return buildCircularOpenFieldVisual(THREE, arenaSpec, context);
    case 'elevated_plus_maze':
      return buildElevatedPlusMazeVisual(THREE, arenaSpec);
    case 'y_maze':
      return buildYMazeVisual(THREE, arenaSpec);
    case 't_maze':
      return buildTMazeVisual(THREE, arenaSpec);
    case 'three_chamber':
      return buildThreeChamberVisual(THREE, arenaSpec, context);
    case 'light_dark_box':
      return buildLightDarkBoxVisual(THREE, arenaSpec);
    case 'barnes_maze':
      return buildBarnesMazeVisual(THREE);
    case 'novel_object':
      return buildNovelObjectVisual(THREE, arenaSpec, context);
    default:
      return null;
  }
}
