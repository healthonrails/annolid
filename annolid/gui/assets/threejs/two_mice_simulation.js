/** Deterministic simulation primitives, independent of Three.js and the DOM. */
export const STANCE_CONTACT_LIMITS = Object.freeze({ slip: 0.07, height: 0.095 });

export function createSeededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Bounded fixed steps avoid display-dependent forces and tab-resume jumps. */
export class FixedStepClock {
  constructor(hz = 30, maxSubsteps = 6) {
    if (!(Number.isFinite(hz) && hz > 0 && Number.isInteger(maxSubsteps) && maxSubsteps > 0)) {
      throw new RangeError('A positive frequency and substep count are required.');
    }
    this.dt = 1 / hz;
    this.maxSubsteps = maxSubsteps;
    this.remainder = 0;
    this.steps = 0;
  }

  advance(elapsed, step, paused = false) {
    if (paused) {
      this.remainder = 0;
      return 0;
    }
    if (!Number.isFinite(elapsed) || elapsed <= 0) return 0;
    this.remainder += Math.min(elapsed, this.dt * this.maxSubsteps);
    const count = Math.min(this.maxSubsteps, Math.floor((this.remainder + 1e-10) / this.dt));
    for (let i = 0; i < count; i++) {
      this.steps++;
      step(this.steps * this.dt, this.dt);
    }
    this.remainder = Math.max(0, this.remainder - count * this.dt);
    return count;
  }
}

/** Signed force along an INWARD wall normal: repel nearby, approach farther away. */
export function wallNormalForce(distance, driveStrength, attraction, nearest = true) {
  if (distance < 2) {
    const penetration = Math.min(2, (2 - distance) / 2);
    return penetration * penetration * 50;
  }
  if (nearest && distance < 12) return -(distance - 2) / 12 * attraction * driveStrength;
  return 0;
}

export const SPINE_PIVOT_S = -0.15;

function clampScalar(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/**
 * Analytical inextensible vertebral arc + concertina squash-and-stretch + dorsal arch.
 * Matches the GLSL vertex deformation in AnnolidShaders.applySpinalFlexShader.
 */
export function evaluateSpinePoint(x, y, z, flex = {}, halfLen = 2.68) {
  const stretch = clampScalar(Number(flex?.stretch) || 0, -0.22, 0.26);
  const bend = clampScalar(Number(flex?.bend) || 0, -0.68, 0.68);
  const sCurve = clampScalar(Number(flex?.sCurve) || 0, -0.28, 0.28);
  const arch = clampScalar(Number(flex?.arch) || 0, -0.12, 0.20);
  const L = Math.max(0.5, Number(halfLen) || 2.68);
  const s = clampScalar(z / L, -1.25, 1.25);

  const waistBell = Math.exp(-Math.pow((s - 0.10) / 0.65, 2.0));
  const girthScale = 1.0 / Math.sqrt(
    Math.max(0.55, 1.0 + stretch * (0.55 + 0.45 * waistBell))
  );

  const zStretched = z * (1.0 + stretch);
  const z0 = SPINE_PIVOT_S * L * (1.0 + stretch);
  const dz = zStretched - z0;
  const xi = s - SPINE_PIVOT_S;
  const xiClamped = clampScalar(xi, -1.1, 1.1);
  const phi = bend * xi * (0.62 + 0.38 * Math.min(1.2, Math.abs(xi)))
    + sCurve * Math.sin(Math.PI * clampScalar(xiClamped, -1.0, 1.0));

  let sinc;
  let oneMinusCosOverPhi;
  if (Math.abs(phi) > 1e-4) {
    sinc = Math.sin(phi) / phi;
    oneMinusCosOverPhi = (1.0 - Math.cos(phi)) / phi;
  } else {
    sinc = 1.0 - (phi * phi) / 6.0;
    oneMinusCosOverPhi = 0.5 * phi;
  }

  const xc = dz * oneMinusCosOverPhi;
  const zc = z0 + dz * sinc;
  const psi = phi * 0.88;
  const cosPsi = Math.cos(psi);
  const sinPsi = Math.sin(psi);
  const xScaled = x * girthScale;

  const archBell = Math.exp(-Math.pow((s + 0.05) / 0.55, 2.0));
  return {
    x: xc + xScaled * cosPsi,
    y: y * (1.0 - 0.24 * stretch) + arch * archBell,
    z: zc - xScaled * sinPsi,
    yaw: psi,
    girthScale,
  };
}

/**
 * Orientation-aware wall clearance so mice can run flush alongside enclosure walls
 * (thigmotaxis) and enter corners while still keeping snout and rump inside bounds.
 */
export function computeOrientedWallMargin(
  headingX,
  headingZ,
  inwardNormalX,
  inwardNormalZ,
  bodyLengthScale = 1.0,
  snoutLengthScale = 1.0,
  widthScale = 1.0,
  axialStretch = 0.0
) {
  const hLen = Math.hypot(headingX, headingZ) || 1.0;
  const hx = headingX / hLen;
  const hz = headingZ / hLen;
  // Positive when facing outward into this wall (opposite to inward normal).
  const towardWall = -(hx * inwardNormalX + hz * inwardNormalZ);
  const stretchMul = 1.0 + clampScalar(axialStretch, -0.22, 0.26) * 0.85;
  const snoutReach = (2.85 * bodyLengthScale + 0.82 * snoutLengthScale) * stretchMul;
  // Cover full posterior rump + sacral tail emergence (bodyLength 2.68) so backing
  // rump-first into a wall never pushes the tail root outside the enclosure.
  const rumpReach = 2.68 * bodyLengthScale * stretchMul;
  const flankRadius = 1.42 * widthScale;

  const axialReach = towardWall >= 0
    ? towardWall * snoutReach
    : (-towardWall) * rumpReach;
  const lateralComponent = Math.sqrt(Math.max(0, 1.0 - towardWall * towardWall));
  return Math.max(
    flankRadius,
    Math.hypot(axialReach, lateralComponent * flankRadius) + 0.14
  );
}
