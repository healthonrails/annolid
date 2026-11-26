/**
 * Rat NCAP Spinal Circuit Controller, Neuromechanical Interfaces, and Rat v2 Body Spec.
 *
 * Pure dependency-free ES module mirroring `annolid/simulation/rat_ncap.py` and faithful to:
 *   Bhattasali, Cui, Pinto, and Lindsay (2026),
 *   "Neural Circuit Architectural Priors for Quadruped Locomotion"
 *   (arXiv:2611.23364v1; https://github.com/rat-ncap/rat-ncap).
 */

const DEG2RAD = Math.PI / 180;

export const RAT_LIMB_NAMES = Object.freeze(['FL', 'FR', 'HL', 'HR']);

export const RAT_V2_SEGMENTS = Object.freeze({
  head: Object.freeze(['mandible', 'atlas']),
  cervicalE: Object.freeze(['cervical_extend']),
  cervicalBT: Object.freeze(['cervical_bend', 'cervical_twist']),
  limbFL: Object.freeze([
    'scapula_L_supinate', 'scapula_L_abduct', 'scapula_L_extend',
    'shoulder_L_supinate', 'shoulder_L_extend', 'elbow_L', 'wrist_L', 'finger_L',
  ]),
  limbFR: Object.freeze([
    'scapula_R_supinate', 'scapula_R_abduct', 'scapula_R_extend',
    'shoulder_R_supinate', 'shoulder_R_extend', 'elbow_R', 'wrist_R', 'finger_R',
  ]),
  lumbarE: Object.freeze(['lumbar_extend']),
  lumbarBT: Object.freeze(['lumbar_bend', 'lumbar_twist']),
  limbHL: Object.freeze([
    'hip_L_supinate', 'hip_L_abduct', 'hip_L_extend', 'knee_L', 'ankle_L', 'toe_L',
  ]),
  limbHR: Object.freeze([
    'hip_R_supinate', 'hip_R_abduct', 'hip_R_extend', 'knee_R', 'ankle_R', 'toe_R',
  ]),
  tailE: Object.freeze(['tail_extend']),
  tailB: Object.freeze(['tail_bend']),
});

export const RAT_V2_LOCOMOTION_SEGMENTS = Object.freeze([
  'limbFL', 'limbFR', 'lumbarE', 'lumbarBT', 'limbHL', 'limbHR',
]);

export const RAT_V2_ACTUATOR_NAMES = Object.freeze(
  Object.values(RAT_V2_SEGMENTS).flat()
);

export const RAT_V2_LOCOMOTION_ACTUATOR_NAMES = Object.freeze(
  RAT_V2_LOCOMOTION_SEGMENTS.flatMap(seg => RAT_V2_SEGMENTS[seg])
);

export const RAT_V2_POS_MID = Object.freeze({
  head: [-36 * DEG2RAD, 20 * DEG2RAD],
  cervicalE: [-0.33],
  cervicalBT: [0, 0],
  limbFL: [4.5, 0, 0, -16.25, 20, -2, 61.2, -65].map(v => v * DEG2RAD),
  limbFR: [4.5, 0, 0, -16.25, 20, -2, 61.2, -65].map(v => v * DEG2RAD),
  lumbarE: [0.1],
  lumbarBT: [0, 0],
  limbHL: [-1.5, 12.75, -15, -35.2, 14, 0].map(v => v * DEG2RAD),
  limbHR: [-1.5, 12.75, -15, -35.2, 14, 0].map(v => v * DEG2RAD),
  tailE: [0],
  tailB: [0],
});

export const RAT_V2_JOINT_RANGES = Object.freeze({
  head: [[-17 * DEG2RAD, 15 * DEG2RAD], [-5 * DEG2RAD, 45 * DEG2RAD]],
  cervicalE: [[-1, 1]],
  cervicalBT: [[-1, 1], [-1, 1]],
  limbFL: [
    [-1, 10], [-15, 15], [-2, 2], [-65, 3], [-75, 50], [-75, 40], [-20, 90], [-180, 0],
  ].map(([a, b]) => [a * DEG2RAD, b * DEG2RAD]),
  limbFR: [
    [-1, 10], [-15, 15], [-2, 2], [-65, 3], [-75, 50], [-75, 40], [-20, 90], [-180, 0],
  ].map(([a, b]) => [a * DEG2RAD, b * DEG2RAD]),
  lumbarE: [[-1, 1]],
  lumbarBT: [[-1, 1], [-1, 1]],
  limbHL: [
    [-40, 30], [-30, 60], [-90, 60], [-70, 88], [0, 80], [-100, 50],
  ].map(([a, b]) => [a * DEG2RAD, b * DEG2RAD]),
  limbHR: [
    [-40, 30], [-30, 60], [-90, 60], [-70, 88], [0, 80], [-100, 50],
  ].map(([a, b]) => [a * DEG2RAD, b * DEG2RAD]),
  tailE: [[-1, 1]],
  tailB: [[-1, 1]],
});

export const RAT_V2_KEQ = Object.freeze({
  head: [0.1, 0.05],
  cervicalE: [0.2],
  cervicalBT: [0.2, 0.2],
  limbFL: [0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.15, 0.03],
  limbFR: [0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.15, 0.03],
  lumbarE: [0.1],
  lumbarBT: [0.1, 0.1],
  limbHL: [0.4, 0.4, 0.6, 0.4, 0.2, 0.03],
  limbHR: [0.4, 0.4, 0.6, 0.4, 0.2, 0.03],
  tailE: [0.1],
  tailB: [0.1],
});

export const RAT_V2_TRQ_MAX = Object.freeze({
  head: [0.1, 0.05],
  cervicalE: [1.0],
  cervicalBT: [0.5, 0.5],
  limbFL: [0.4, 0.4, 0.4, 1.0, 1.5, 1.0, 0.29, 0.05],
  limbFR: [0.4, 0.4, 0.4, 1.0, 1.5, 1.0, 0.29, 0.05],
  lumbarE: [2.0],
  lumbarBT: [0.2, 0.2],
  limbHL: [0.49, 0.63, 1.58, 1.11, 0.28, 0.08],
  limbHR: [0.49, 0.63, 1.58, 1.11, 0.28, 0.08],
  tailE: [0.5],
  tailB: [0.5],
});

export const RAT_V2_KD = Object.freeze({
  head: [0.002, 0.002],
  cervicalE: [0.05],
  cervicalBT: [0.05, 0.05],
  limbFL: [0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.002, 0.002],
  limbFR: [0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.002, 0.002],
  lumbarE: [0.03],
  lumbarBT: [0.03, 0.03],
  limbHL: [0.02, 0.02, 0.02, 0.005, 0.003, 0.001],
  limbHR: [0.02, 0.02, 0.02, 0.005, 0.003, 0.001],
  tailE: [0.002],
  tailB: [0.002],
});

export const RAT_COAT_PROFILES = Object.freeze({
  black_hooded: Object.freeze({
    id: 'black_hooded',
    label: 'Long-Evans Black-Hooded',
    hoodColor: '#1f1f22',
    bodyColor: '#eeeae2',
    bellyColor: '#f4f0e8',
    skinColor: '#dfb1a8',
    eyeColor: '#120d0e',
    tailColor: '#d9aba1',
  }),
  cinnamon: Object.freeze({
    id: 'cinnamon',
    label: 'Long-Evans Cinnamon',
    hoodColor: '#7b4c2d',
    bodyColor: '#ede4d8',
    bellyColor: '#f4ede2',
    skinColor: '#e2b5aa',
    eyeColor: '#231312',
    tailColor: '#d7a89d',
  }),
  gray: Object.freeze({
    id: 'gray',
    label: 'Long-Evans Gray',
    hoodColor: '#676b74',
    bodyColor: '#eaebee',
    bellyColor: '#f3f4f6',
    skinColor: '#deb4ae',
    eyeColor: '#141418',
    tailColor: '#d5aea7',
  }),
});

export const RAT_ABLATION_POPULATIONS = Object.freeze({
  none: Object.freeze([]),
  v0v_v0d: Object.freeze([
    'cross_v0d_FL', 'cross_v0d_FR', 'cross_v0d_HL', 'cross_v0d_HR',
    'diagd_v0d_FL', 'diagd_v0d_FR', 'diagd_v0v_FL', 'diagd_v0v_FR',
  ]),
  v0d: Object.freeze([
    'cross_v0d_FL', 'cross_v0d_FR', 'cross_v0d_HL', 'cross_v0d_HR',
    'diagd_v0d_FL', 'diagd_v0d_FR',
  ]),
  v0v: Object.freeze(['diagd_v0v_FL', 'diagd_v0v_FR']),
  v3f_v3a: Object.freeze([
    'cross_v3f_FL', 'cross_v3f_FR', 'cross_v3f_HL', 'cross_v3f_HR',
    'diaga_v3a_HL', 'diaga_v3a_HR',
  ]),
  v3f: Object.freeze(['cross_v3f_FL', 'cross_v3f_FR', 'cross_v3f_HL', 'cross_v3f_HR']),
  v3a: Object.freeze(['diaga_v3a_HL', 'diaga_v3a_HR']),
});

export const RAT_KEYPOINTS = Object.freeze([
  'nose', 'head', 'left_ear', 'right_ear', 'neck', 'thorax', 'torso_center',
  'lumbar', 'pelvis', 'tail_base', 'tail_mid', 'tail_tip',
  'left_forepaw', 'right_forepaw', 'left_hindpaw', 'right_hindpaw',
]);

export const RAT_NCAP_SPEC = Object.freeze({
  id: 'rat-ncap-v2',
  paper: 'arXiv:2611.23364v1',
  totalDofs: RAT_V2_ACTUATOR_NAMES.length,
  locomotionDofs: RAT_V2_LOCOMOTION_ACTUATOR_NAMES.length,
  interfaces: Object.freeze(['flexor_extensor', 'position_stiffness', 'position', 'torque']),
  coats: Object.freeze(['black_hooded', 'cinnamon', 'gray']),
  behaviors: Object.freeze([
    'stand', 'walk', 'trot', 'bound', 'gait_transition',
    'bumpy_terrain', 'v0_ablation_bound', 'v3_ablation_trot', 'open_field_explore',
  ]),
});

function clamp(v, lo, hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

function unitRelu(x) {
  return x <= 0 ? 0 : (x >= 1 ? 1 : x);
}

export function createSeededRng(seed = 7) {
  let a = (seed >>> 0) || 1;
  return () => {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleStandardNormal(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function normalizeRatInterface(name) {
  const raw = String(name || 'flexor_extensor').trim().toLowerCase().replace(/-/g, '_');
  const map = {
    flxext: 'flexor_extensor',
    flexor_extensor: 'flexor_extensor',
    spring_muscle: 'flexor_extensor',
    poskp: 'position_stiffness',
    position_stiffness: 'position_stiffness',
    pos: 'position',
    position: 'position',
    trq: 'torque',
    torque: 'torque',
  };
  return map[raw] || 'flexor_extensor';
}

export function defaultSpeedForBehavior(behavior) {
  const key = String(behavior || 'trot').trim().toLowerCase();
  const map = {
    stand: 0.0,
    walk: 0.22,
    trot: 0.50,
    bound: 0.88,
    gait_transition: 0.50,
    bumpy_terrain: 0.48,
    v0_ablation_bound: 0.45,
    v3_ablation_trot: 0.45,
    open_field_explore: 0.36,
  };
  return key in map ? map[key] : 0.50;
}

/**
 * Exact spring-muscle parameter matching from `composer.transforms.SpringMuscleController`.
 */
export function computeSpringMuscleParams(low, high, poseq, keq, trqMaxFlx, trqMaxExt) {
  const n = low.length;
  const kf = new Float64Array(n);
  const ke = new Float64Array(n);
  const cf = new Float64Array(n);
  const ce = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const span = Math.max(high[i] - low[i], 1e-6);
    const kfi = (keq[i] * (poseq[i] - low[i]) + (trqMaxFlx[i] - trqMaxExt[i])) / span;
    const kei = (keq[i] * (high[i] - poseq[i]) + (trqMaxExt[i] - trqMaxFlx[i])) / span;
    const kfSafe = Math.abs(kfi) < 1e-9 ? (kfi >= 0 ? 1e-9 : -1e-9) : kfi;
    const keSafe = Math.abs(kei) < 1e-9 ? (kei >= 0 ? 1e-9 : -1e-9) : kei;
    kf[i] = kfi;
    ke[i] = kei;
    cf[i] = trqMaxFlx[i] / kfSafe - span;
    ce[i] = trqMaxExt[i] / keSafe - span;
  }
  return { kf, ke, cf, ce };
}

export class NeuromechanicalInterface {
  constructor(interfaceType = 'flexor_extensor', posMargin = 0.2) {
    this.interfaceType = normalizeRatInterface(interfaceType);
    this.posMargin = posMargin;
    this.segments = [...RAT_V2_LOCOMOTION_SEGMENTS];
    const low = [];
    const high = [];
    const poseq = [];
    const keq = [];
    const trqMax = [];
    const kd = [];
    for (const seg of this.segments) {
      for (const [lo, hi] of RAT_V2_JOINT_RANGES[seg]) {
        low.push(lo);
        high.push(hi);
      }
      poseq.push(...RAT_V2_POS_MID[seg]);
      keq.push(...RAT_V2_KEQ[seg]);
      trqMax.push(...RAT_V2_TRQ_MAX[seg]);
      kd.push(...RAT_V2_KD[seg]);
    }
    this.dof = low.length;
    this.low = Float64Array.from(low);
    this.high = Float64Array.from(high);
    this.poseq = Float64Array.from(poseq);
    this.keq = Float64Array.from(keq);
    this.trqMax = Float64Array.from(trqMax);
    this.kd = Float64Array.from(kd);
    this.spring = computeSpringMuscleParams(
      this.low, this.high, this.poseq, this.keq, this.trqMax, this.trqMax
    );
  }

  setInterfaceType(interfaceType) {
    this.interfaceType = normalizeRatInterface(interfaceType);
  }

  computeTorque({ q, dq, flx, ext, qTarget, kpScale, trqCmd }) {
    const out = new Float64Array(this.dof);
    for (let i = 0; i < this.dof; i++) {
      let tau = 0;
      if (this.interfaceType === 'flexor_extensor') {
        const fa = clamp(flx ? flx[i] : 0, 0, 1);
        const ea = clamp(ext ? ext[i] : 0, 0, 1);
        const xf = q[i] - this.low[i] + this.spring.cf[i];
        const xe = this.high[i] - q[i] + this.spring.ce[i];
        tau = ea * this.spring.ke[i] * xe - fa * this.spring.kf[i] * xf - this.kd[i] * dq[i];
      } else if (this.interfaceType === 'position') {
        const span = this.high[i] - this.low[i];
        const qd = clamp(
          qTarget ? qTarget[i] : this.poseq[i],
          this.low[i] - this.posMargin * span,
          this.high[i] + this.posMargin * span
        );
        tau = this.keq[i] * (qd - q[i]) - this.kd[i] * dq[i];
      } else if (this.interfaceType === 'position_stiffness') {
        const span = this.high[i] - this.low[i];
        const qd = clamp(
          qTarget ? qTarget[i] : this.poseq[i],
          this.low[i] - this.posMargin * span,
          this.high[i] + this.posMargin * span
        );
        const ks = clamp(kpScale ? kpScale[i] : 1.0, 0, 1);
        tau = (this.keq[i] * ks) * (qd - q[i]) - this.kd[i] * dq[i];
      } else {
        tau = trqCmd ? trqCmd[i] : 0;
      }
      out[i] = clamp(tau, -this.trqMax[i], this.trqMax[i]);
    }
    return out;
  }

  effectiveStiffness({ flx, ext, kpScale }) {
    const out = new Float64Array(this.dof);
    for (let i = 0; i < this.dof; i++) {
      if (this.interfaceType === 'flexor_extensor') {
        const fa = clamp(flx ? flx[i] : 0, 0, 1);
        const ea = clamp(ext ? ext[i] : 0, 0, 1);
        out[i] = ea * this.spring.ke[i] + fa * this.spring.kf[i];
      } else if (this.interfaceType === 'position_stiffness') {
        out[i] = this.keq[i] * clamp(kpScale ? kpScale[i] : 1.0, 0, 1);
      } else if (this.interfaceType === 'position') {
        out[i] = this.keq[i];
      } else {
        out[i] = 0;
      }
    }
    return out;
  }
}

export function computeMechanicalPower(torque, velocity) {
  let pos = 0;
  let neg = 0;
  const n = Math.min(torque.length, velocity.length);
  for (let i = 0; i < n; i++) {
    const p = torque[i] * velocity[i];
    if (p >= 0) pos += p;
    else neg += p;
  }
  return {
    positivePowerW: pos,
    negativePowerW: neg,
    netPowerW: pos + neg,
    absPowerW: pos - neg,
  };
}

export function computeBumpyTerrainHeight(x, z, bumpHeightM = 0.02, wavelengthM = 0.05) {
  const amp = Math.max(0, Number(bumpHeightM) || 0);
  if (amp <= 1e-6) return 0;
  const k = (2 * Math.PI) / Math.max(Number(wavelengthM) || 0.05, 1e-3);
  const h = 0.5 * (
    Math.sin(k * x) * Math.cos(k * z)
    + 0.45 * Math.sin(0.67 * k * x + 0.53 * k * z + 0.9)
  );
  return amp * 0.5 * (h + 1.0);
}

export function classifyQuadrupedGait(speedMps, phaseOffsetsRelFR, stanceFraction = 0.6) {
  if (speedMps < 0.05) return 'stand';
  const circDist = (a, b) => {
    const d = Math.abs((((a - b) % 1) + 1) % 1);
    return Math.min(d, 1 - d);
  };
  const pFL = (((phaseOffsetsRelFR.FL ?? 0.5) % 1) + 1) % 1;
  const pHL = (((phaseOffsetsRelFR.HL ?? 0.0) % 1) + 1) % 1;
  const pHR = (((phaseOffsetsRelFR.HR ?? 0.5) % 1) + 1) % 1;

  const lrForeSync = circDist(pFL, 0.0);
  const lrHindSync = circDist(pHL, pHR);
  if (lrForeSync < 0.23 && lrHindSync < 0.25) return 'bound';

  const diagSync = 0.5 * (circDist(pHL, 0.0) + circDist(pHR, pFL));
  if (lrForeSync >= 0.26 && diagSync < 0.24 && stanceFraction < 0.72) return 'trot';
  return 'walk';
}

class BasicUnit {
  constructor(name, voltageTime, bias = 0.0) {
    this.name = name;
    this.voltageTime = voltageTime;
    this.bias = bias;
    this.v = 0.0;
    this.value = 0.0;
    this.activityScale = 1.0;
  }

  reset(v = 0.0) {
    this.v = clamp(v, 0, 1);
    this.value = unitRelu(this.v) * this.activityScale;
  }

  step(current, dt) {
    const tau = Math.max(this.voltageTime, 1e-4);
    this.v = clamp(this.v + (dt / tau) * (-this.v + this.bias + current), 0, 1);
    this.value = unitRelu(this.v) * this.activityScale;
    return this.value;
  }
}

class BurstingOscillatorUnit {
  constructor(name) {
    this.name = name;
    this.activeTime = 0.16;
    this.quietTime = 0.48;
    this.adaptationTime = 0.72;
    this.voltageTime = 0.01;
    this.activeBoundZero = 1.0 - Math.exp(-this.activeTime / this.adaptationTime);
    this.quietBoundZero = Math.exp(-this.quietTime / this.adaptationTime);
    if (this.activeBoundZero <= this.quietBoundZero) {
      this.activeBoundZero = Math.min(0.85, this.quietBoundZero + 0.18);
    }
    this.v = -1.0;
    this.a = 0.5;
    this.value = 0.0;
    this.activityScale = 1.0;
  }

  reset(aFraction = 0.8) {
    this.v = -1.0;
    this.a = clamp(aFraction * this.quietBoundZero, 0, 1);
    this.value = unitRelu((this.v - 0.1) / 0.9) * this.activityScale;
  }

  step(drive, synCurrent, dt) {
    const d = clamp(drive, 0, 1);
    const quietThresh = this.quietBoundZero * (1.0 - 0.62 * d);
    const activeThresh = this.activeBoundZero * (1.0 + 0.12 * d);

    if (d < 0.03) {
      this.v = clamp(this.v + (dt / this.voltageTime) * (-1.0 - this.v), -1, 1);
      this.a = clamp(this.a + (dt / this.adaptationTime) * (this.quietBoundZero - this.a), 0, 1);
      this.value = unitRelu((this.v - 0.1) / 0.9) * this.activityScale;
      return this.value;
    }

    const effectiveA = this.a - 0.28 * synCurrent;
    let targetV = -1.0;
    if (this.v > 0.0) {
      this.a = clamp(this.a + (dt / this.activeTime) * (1.0 - this.a), 0, 1);
      targetV = effectiveA >= activeThresh ? -1.0 : 1.0;
    } else {
      const rate = (1.0 + 2.6 * d) / this.quietTime;
      this.a = clamp(this.a - dt * rate * this.a, 0, 1);
      targetV = effectiveA <= quietThresh ? 1.0 : -1.0;
    }
    this.v = clamp(this.v + (dt / this.voltageTime) * (targetV - this.v), -1, 1);
    this.value = unitRelu((this.v - 0.1) / 0.9) * this.activityScale;
    return this.value;
  }
}

export class RatRhythmGenerationCircuit {
  constructor({ dt = 0.02, rgNoiseStd = 0.0, seed = 7 } = {}) {
    this.dt = dt;
    this.seed = seed >>> 0;
    this.rgNoiseStd = Math.max(0, Number(rgNoiseStd) || 0);
    this.flx = {};
    this.ext = {};
    this.flxHist = {};
    this.flx2 = {};
    this.ext2 = {};
    for (const limb of RAT_LIMB_NAMES) {
      this.flx[limb] = new BurstingOscillatorUnit(`flx_${limb}`);
      this.ext[limb] = new BasicUnit(`ext_${limb}`, 0.03, 0.85);
      this.flxHist[limb] = [0, 0];
      this.flx2[limb] = 0;
      this.ext2[limb] = new BasicUnit(`ext2_${limb}`, 0.08, -0.3);
    }

    this.interneurons = {};
    for (const limb of RAT_LIMB_NAMES) {
      this.interneurons[`cross_v0d_${limb}`] = new BasicUnit(`cross_v0d_${limb}`, 0.01);
      this.interneurons[`cross_v3f_${limb}`] = new BasicUnit(`cross_v3f_${limb}`, 0.01);
      this.interneurons[`cross_v3e_${limb}`] = new BasicUnit(`cross_v3e_${limb}`, 0.01);
    }
    for (const fore of ['FL', 'FR']) {
      this.interneurons[`diagd_v0d_${fore}`] = new BasicUnit(`diagd_v0d_${fore}`, 0.01);
      this.interneurons[`diagd_v0v_${fore}`] = new BasicUnit(`diagd_v0v_${fore}`, 0.01);
      this.interneurons[`diagd_in2_${fore}`] = new BasicUnit(`diagd_in2_${fore}`, 0.01);
    }
    for (const hind of ['HL', 'HR']) {
      this.interneurons[`diaga_v3a_${hind}`] = new BasicUnit(`diaga_v3a_${hind}`, 0.01);
    }

    this.baseWeights = Object.freeze({
      osc_flxext_i: -2.0,
      osc_extflx_i: -0.1,
      cross_v0d_i: -1.35,
      cross_v3f_e: 0.95,
      cross_v3e_e: 0.45,
      diagd_v0d_i: -0.75,
      diagd_v0v_e: 0.85,
      diagd_in2_i: -0.35,
      diaga_v3a_e: 0.65,
      sided_flxflx_i: -1.2,
      sided_extflx_e: 0.05,
      sidea_extflx_e: 0.05,
    });
    this.weights = { ...this.baseWeights };
    this.applyWeightNoise(this.rgNoiseStd, this.seed);
    this.reset();
  }

  applyWeightNoise(sigma = 0.0, seed = this.seed) {
    this.rgNoiseStd = Math.max(0, Number(sigma) || 0);
    this.seed = seed >>> 0;
    const rng = createSeededRng(this.seed);
    this.weights = {};
    for (const [k, base] of Object.entries(this.baseWeights)) {
      if (this.rgNoiseStd <= 0) {
        this.weights[k] = base;
      } else {
        const scale = Math.max(0.05, 1.0 + this.rgNoiseStd * sampleStandardNormal(rng));
        this.weights[k] = base * scale;
      }
    }
  }

  setPopulationScales({ v0Scale = 1.0, v3Scale = 1.0 } = {}) {
    for (const unit of Object.values(this.interneurons)) {
      unit.activityScale = 1.0;
    }
    const v0k = clamp(v0Scale, 0, 1);
    const v3k = clamp(v3Scale, 0, 1);
    for (const u of RAT_ABLATION_POPULATIONS.v0v_v0d) {
      if (this.interneurons[u]) this.interneurons[u].activityScale = v0k;
    }
    for (const u of RAT_ABLATION_POPULATIONS.v3f_v3a) {
      if (this.interneurons[u]) this.interneurons[u].activityScale = v3k;
    }
  }

  setAblation(population = 'none', scale = 0.0) {
    for (const unit of Object.values(this.interneurons)) {
      unit.activityScale = 1.0;
    }
    const pop = String(population || 'none').trim().toLowerCase();
    const list = RAT_ABLATION_POPULATIONS[pop] || [];
    const k = clamp(scale, 0, 1);
    for (const u of list) {
      if (this.interneurons[u]) this.interneurons[u].activityScale = k;
    }
  }

  getMeanPopulationScale(population) {
    const list = RAT_ABLATION_POPULATIONS[population] || [];
    if (!list.length) return 1.0;
    let sum = 0;
    for (const u of list) sum += this.interneurons[u]?.activityScale ?? 1.0;
    return sum / list.length;
  }

  reset() {
    const stagger = { FL: 0.7, FR: 0.9, HL: 0.8, HR: 1.0 };
    for (const limb of RAT_LIMB_NAMES) {
      this.flx[limb].reset(stagger[limb]);
      this.ext[limb].reset(0.85);
      this.ext2[limb].reset(0.5);
      this.flxHist[limb] = [0, 0];
      this.flx2[limb] = 0;
    }
    for (const unit of Object.values(this.interneurons)) {
      unit.reset(0.0);
    }
  }

  step(command, { afferentFlx = {}, afferentExt = {}, dt = this.dt } = {}) {
    const c = clamp(command, 0, 1);
    const w = this.weights;
    const v0dGate = clamp(1.15 - 1.15 * c, 0.05, 1.0);
    const v0vGate = clamp(1.0 - Math.abs(c - 0.50) * 2.1, 0.05, 1.0);
    const v3Gate = clamp((c - 0.55) / 0.35, 0.0, 1.0);
    const v0MeanScale = this.getMeanPopulationScale('v0v_v0d');
    const v3EffGate = Math.max(v3Gate, 0.45 * (1.0 - v0MeanScale) * (c > 0.15 ? 1 : 0));

    const prevFlx = {};
    const prevExt = {};
    for (const limb of RAT_LIMB_NAMES) {
      prevFlx[limb] = this.flx[limb].value;
      prevExt[limb] = this.ext[limb].value;
    }

    for (const limb of RAT_LIMB_NAMES) {
      this.interneurons[`cross_v0d_${limb}`].step(prevFlx[limb] * v0dGate, dt);
      this.interneurons[`cross_v3f_${limb}`].step(prevFlx[limb] * v3EffGate, dt);
      this.interneurons[`cross_v3e_${limb}`].step(prevExt[limb] * v3EffGate, dt);
    }
    for (const fore of ['FL', 'FR']) {
      this.interneurons[`diagd_v0d_${fore}`].step(prevFlx[fore] * v0dGate, dt);
      this.interneurons[`diagd_v0v_${fore}`].step(prevFlx[fore] * v0vGate, dt);
      this.interneurons[`diagd_in2_${fore}`].step(prevExt[fore] * 0.6, dt);
    }
    for (const hind of ['HL', 'HR']) {
      this.interneurons[`diaga_v3a_${hind}`].step(prevFlx[hind] * v3EffGate, dt);
    }

    const contra = { FL: 'FR', FR: 'FL', HL: 'HR', HR: 'HL' };
    const homolat = { FL: 'HL', FR: 'HR', HL: 'FL', HR: 'FR' };
    const diag = { FL: 'HR', FR: 'HL', HL: 'FR', HR: 'FL' };

    const flxOut = {};
    const extOut = {};
    for (const limb of RAT_LIMB_NAMES) {
      const cLimb = contra[limb];
      const hLimb = homolat[limb];
      const dLimb = diag[limb];
      let synFlx = 0;
      synFlx += w.osc_extflx_i * prevExt[limb];
      synFlx += w.cross_v0d_i * this.interneurons[`cross_v0d_${cLimb}`].value;
      synFlx += w.cross_v3f_e * this.interneurons[`cross_v3f_${cLimb}`].value;
      if (limb === 'HL' || limb === 'HR') {
        synFlx += w.sided_flxflx_i * prevFlx[hLimb] * (1.0 - 0.55 * v3EffGate);
        synFlx += w.sided_extflx_e * prevExt[hLimb];
        synFlx += w.diagd_v0v_e * this.interneurons[`diagd_v0v_${dLimb}`].value;
        synFlx += w.diagd_v0d_i * this.interneurons[`diagd_v0d_${dLimb}`].value * (1.0 - v0vGate);
        synFlx += w.diagd_in2_i * this.interneurons[`diagd_in2_${dLimb}`].value;
      } else {
        synFlx += w.sidea_extflx_e * prevExt[hLimb];
        synFlx += w.sided_flxflx_i * 0.45 * prevFlx[hLimb];
        synFlx += w.diaga_v3a_e * this.interneurons[`diaga_v3a_${dLimb}`].value;
      }
      synFlx += Number(afferentFlx[limb] || 0);
      const flxVal = this.flx[limb].step(c, synFlx, dt);

      const synExt = w.osc_flxext_i * flxVal
        + w.cross_v3e_e * this.interneurons[`cross_v3e_${cLimb}`].value
        + Number(afferentExt[limb] || 0);
      const extVal = this.ext[limb].step(synExt, dt);

      const hist = this.flxHist[limb];
      this.flx2[limb] = hist[0];
      hist.push(flxVal);
      if (hist.length > 2) hist.shift();
      this.ext2[limb].step(extVal - flxVal, dt);

      flxOut[limb] = flxVal;
      extOut[limb] = extVal;
    }

    return {
      command: c,
      flx: flxOut,
      flx2: { ...this.flx2 },
      ext: extOut,
      ext2: {
        FL: this.ext2.FL.value,
        FR: this.ext2.FR.value,
        HL: this.ext2.HL.value,
        HR: this.ext2.HR.value,
      },
      interneurons: Object.fromEntries(
        Object.entries(this.interneurons).map(([k, u]) => [k, u.value])
      ),
    };
  }
}

export class RatNCAPController {
  constructor(options = {}) {
    this.behavior = String(options.behavior || 'trot').trim().toLowerCase();
    this.interface = new NeuromechanicalInterface(options.interface || 'flexor_extensor');
    this.dt = Number(options.dt || 0.02);
    this.seed = (options.seed ?? 7) >>> 0;
    this.targetSpeed = options.targetSpeed != null
      ? clamp(Number(options.targetSpeed), 0, 1)
      : defaultSpeedForBehavior(this.behavior);
    this.autoRamp = this.behavior === 'gait_transition' || Boolean(options.autoRamp);
    this.terrainBumpCm = Math.max(
      0,
      options.terrainBumpCm != null
        ? Number(options.terrainBumpCm)
        : (this.behavior === 'bumpy_terrain' ? 2.0 : 0.0)
    );
    this.lockLumbar = Boolean(options.lockLumbar);
    this.rg = new RatRhythmGenerationCircuit({
      dt: this.dt,
      rgNoiseStd: Number(options.rgNoiseStd || 0),
      seed: this.seed,
    });

    const ablation = String(options.ablation || 'none').trim().toLowerCase();
    if (ablation !== 'none') {
      this.rg.setAblation(ablation, Number(options.ablationScale ?? 0));
    } else if (this.behavior === 'v0_ablation_bound') {
      this.rg.setAblation('v0v_v0d', 0.0);
    } else if (this.behavior === 'v3_ablation_trot') {
      this.rg.setAblation('v3f_v3a', 0.0);
    }

    this.segSlices = {};
    let off = 0;
    for (const seg of this.interface.segments) {
      const n = RAT_V2_SEGMENTS[seg].length;
      this.segSlices[seg] = [off, off + n];
      off += n;
    }
    this.reset();
  }

  reset() {
    this.stepCount = 0;
    this.timeSec = 0;
    this.forwardSpeed = 0;
    this.x = 0;
    this.z = 0;
    this.yaw = 0;
    this.pitch = 0;
    this.roll = 0;
    this.q = Float64Array.from(this.interface.poseq);
    this.dq = new Float64Array(this.interface.dof);
    this.tau = new Float64Array(this.interface.dof);
    this.flexorAct = new Float64Array(this.interface.dof).fill(0.5);
    this.extensorAct = new Float64Array(this.interface.dof).fill(0.5);
    this.limbPhase = { FL: 0.5, FR: 0.0, HL: 0.0, HR: 0.5 };
    this.footContact = { FL: 1, FR: 1, HL: 1, HR: 1 };
    this.rg.reset();
  }

  _effectivePhaseOffsets(command) {
    const v0Scale = this.rg.getMeanPopulationScale('v0v_v0d');
    const v3Scale = this.rg.getMeanPopulationScale('v3f_v3a');
    const speedBound = clamp((command - 0.66) / 0.20, 0, 1) * (0.35 + 0.65 * v3Scale);
    const ablationBound = clamp((0.35 - v0Scale) / 0.25, 0, 1) * (command >= 0.18 ? 1 : 0);
    const boundBlend = Math.max(speedBound, ablationBound);
    const trotBlend = clamp((command - 0.26) / 0.16, 0, 1);

    const pFL = 0.50 * (1.0 - boundBlend);
    let pHL = (0.75 + 0.25 * trotBlend) % 1.0;
    let pHR = (0.25 + 0.25 * trotBlend) % 1.0;
    if (boundBlend > 0) {
      const dHL = (((0.50 - pHL + 0.5) % 1.0) + 1.0) % 1.0 - 0.5;
      pHL = (((pHL + boundBlend * dHL) % 1.0) + 1.0) % 1.0;
      const dHR = (((0.50 - pHR + 0.5) % 1.0) + 1.0) % 1.0 - 0.5;
      pHR = (((pHR + boundBlend * dHR) % 1.0) + 1.0) % 1.0;
    }
    return { FR: 0.0, FL: pFL, HL: pHL, HR: pHR };
  }

  step(dt = this.dt, overrides = {}) {
    const stepDt = Math.max(1e-4, Number(dt) || this.dt);
    let vTarget = overrides.targetSpeed != null ? clamp(overrides.targetSpeed, 0, 1) : this.targetSpeed;
    if (this.autoRamp && overrides.targetSpeed == null) {
      const cycle = this.timeSec % 8.0;
      if (cycle < 1.2) vTarget = 0.0;
      else if (cycle < 3.2) vTarget = 0.22;
      else if (cycle < 5.6) vTarget = 0.52;
      else vTarget = 0.90;
    } else if (this.behavior === 'open_field_explore' && overrides.targetSpeed == null) {
      vTarget = clamp(this.targetSpeed + 0.14 * Math.sin(0.8 * this.timeSec), 0.12, 0.72);
    }

    const command = clamp(vTarget, 0, 1);
    const afferentFlx = {};
    const afferentExt = {};
    for (const limb of RAT_LIMB_NAMES) {
      afferentFlx[limb] = 0.12 * (1.0 - this.footContact[limb]);
      afferentExt[limb] = 0.18 * this.footContact[limb];
    }
    const rgState = this.rg.step(command, { afferentFlx, afferentExt, dt: stepDt });

    const stepFreqHz = command < 0.03 ? 0.0 : (1.5 + 3.9 * command);
    const targetOffsets = this._effectivePhaseOffsets(command);

    if (stepFreqHz > 0) {
      this.limbPhase.FR = (this.limbPhase.FR + stepFreqHz * stepDt) % 1.0;
      const entrain = Math.min(1.0, 7.5 * stepDt);
      for (const limb of ['FL', 'HL', 'HR']) {
        const desired = (this.limbPhase.FR + targetOffsets[limb]) % 1.0;
        const diff = ((((desired - this.limbPhase[limb]) + 0.5) % 1.0) + 1.0) % 1.0 - 0.5;
        this.limbPhase[limb] = (((this.limbPhase[limb] + stepFreqHz * stepDt + entrain * diff) % 1.0) + 1.0) % 1.0;
      }
    }

    const dutyFactor = command < 0.03 ? 1.0 : clamp(0.80 - 0.36 * command, 0.42, 0.85);
    const swingFrac = 1.0 - dutyFactor;

    const qTarget = Float64Array.from(this.interface.poseq);
    const kpScale = new Float64Array(this.interface.dof).fill(0.72);
    const trqCmd = new Float64Array(this.interface.dof);
    const flexorAct = new Float64Array(this.interface.dof).fill(0.5);
    const extensorAct = new Float64Array(this.interface.dof).fill(0.5);
    const limbSwing = {};
    const limbStance = {};

    for (const limb of RAT_LIMB_NAMES) {
      const phase = this.limbPhase[limb];
      const inSwing = command >= 0.03 && phase < swingFrac;
      let liftoff = 0; let advance = 0; let touchdown = 0; let support = 0; let propulsion = 0;
      let contact = 1;
      if (inSwing) {
        const su = phase / Math.max(swingFrac, 1e-4);
        liftoff = su < 0.5 ? Math.sin(Math.PI * Math.min(1.0, su * 2.0)) : 0;
        advance = su > 0.35 ? Math.sin(Math.PI * Math.max(0.0, (su - 0.35) / 0.65)) : 0;
        contact = 0;
      } else {
        const stu = command < 0.03 ? 0 : (phase - swingFrac) / Math.max(dutyFactor, 1e-4);
        touchdown = command >= 0.03 ? Math.exp(-Math.pow((stu - 0.12) / 0.12, 2)) : 0;
        support = command < 0.03 ? 1.0 : Math.sin(Math.PI * stu);
        propulsion = command >= 0.03 ? Math.exp(-Math.pow((stu - 0.82) / 0.18, 2)) : 0;
        contact = 1;
      }

      const swingDrive = clamp(0.75 * (liftoff + advance) + 0.25 * (rgState.flx[limb] + 0.5 * rgState.flx2[limb]), 0, 1);
      const stanceDrive = clamp(0.75 * (support + 0.6 * propulsion + 0.4 * touchdown) + 0.25 * rgState.ext[limb], 0, 1);
      limbSwing[limb] = swingDrive;
      limbStance[limb] = stanceDrive;
      this.footContact[limb] = contact;

      const [start, end] = this.segSlices[`limb${limb}`];
      const amp = 0.25 + 0.75 * command;
      const deltaQ = (limb === 'FL' || limb === 'FR')
        ? [
          0.04 * (advance - propulsion) * amp,
          0.05 * (liftoff - support) * amp,
          0.02 * (advance - propulsion) * amp,
          0.16 * (liftoff - touchdown) * amp,
          0.48 * (advance - propulsion) * amp,
          0.52 * (liftoff - 0.7 * support) * amp,
          0.38 * (touchdown - liftoff) * amp,
          0.30 * (propulsion - liftoff) * amp,
        ]
        : [
          0.12 * (liftoff - support) * amp,
          0.14 * (liftoff - touchdown) * amp,
          0.62 * (advance - propulsion) * amp,
          0.58 * (liftoff - 0.65 * propulsion) * amp,
          0.44 * (propulsion - liftoff) * amp,
          0.35 * (propulsion - liftoff) * amp,
        ];
      const baseCo = 0.32 + 0.42 * stanceDrive;
      for (let j = 0; j < end - start; j++) {
        const idx = start + j;
        const qt = clamp(this.interface.poseq[idx] + deltaQ[j], this.interface.low[idx], this.interface.high[idx]);
        qTarget[idx] = qt;
        const span = Math.max(this.interface.high[idx] - this.interface.low[idx], 1e-4);
        const normShift = clamp(deltaQ[j] / span, -0.6, 0.6);
        extensorAct[idx] = clamp(baseCo + 0.75 * normShift, 0, 1);
        flexorAct[idx] = clamp(baseCo - 0.75 * normShift, 0, 1);
        kpScale[idx] = clamp(0.35 + 0.60 * stanceDrive, 0.15, 1.0);
        trqCmd[idx] = this.interface.keq[idx] * (qt - this.q[idx]);
      }
    }

    // Trunk lumbar flexion/extension & lateral bending.
    const [le0] = this.segSlices.lumbarE;
    const [lbt0] = this.segSlices.lumbarBT;
    const foreSync = 0.5 * (limbSwing.FL + limbSwing.FR) - 0.5 * (limbSwing.HL + limbSwing.HR);
    const latDiff = (limbSwing.FL - limbSwing.FR) - (limbSwing.HL - limbSwing.HR);
    if (!this.lockLumbar && command >= 0.03) {
      qTarget[le0] = clamp(this.interface.poseq[le0] + 0.42 * command * foreSync, this.interface.low[le0], this.interface.high[le0]);
      qTarget[lbt0] = clamp(this.interface.poseq[lbt0] + 0.28 * command * latDiff, this.interface.low[lbt0], this.interface.high[lbt0]);
      qTarget[lbt0 + 1] = clamp(this.interface.poseq[lbt0 + 1] + 0.18 * command * latDiff, this.interface.low[lbt0 + 1], this.interface.high[lbt0 + 1]);
      extensorAct[le0] = clamp(0.5 + 0.4 * foreSync * command, 0, 1);
      flexorAct[le0] = clamp(0.5 - 0.4 * foreSync * command, 0, 1);
      extensorAct[lbt0] = clamp(0.5 + 0.35 * latDiff * command, 0, 1);
      flexorAct[lbt0] = clamp(0.5 - 0.35 * latDiff * command, 0, 1);
    }

    const tau = this.interface.computeTorque({
      q: this.q,
      dq: this.dq,
      flx: flexorAct,
      ext: extensorAct,
      qTarget,
      kpScale,
      trqCmd,
    });
    for (let i = 0; i < this.interface.dof; i++) {
      const inertia = Math.max(0.008 * this.interface.keq[i], 0.0012);
      const damping = Math.max(0.18 * this.interface.keq[i], 0.025);
      const ddq = clamp((tau[i] - damping * this.dq[i]) / inertia, -140, 140);
      this.dq[i] = clamp(this.dq[i] + ddq * stepDt, -20, 20);
      this.q[i] = clamp(this.q[i] + this.dq[i] * stepDt, this.interface.low[i], this.interface.high[i]);
    }
    this.tau = tau;
    this.flexorAct = flexorAct;
    this.extensorAct = extensorAct;

    const bumpM = this.terrainBumpCm * 0.01;
    const terrainDrag = 1.0 - Math.min(0.18, 2.5 * bumpM);
    const noiseDrag = 1.0 - Math.min(0.15, 0.22 * this.rg.rgNoiseStd);
    const desiredSpeed = vTarget * terrainDrag * noiseDrag;
    this.forwardSpeed += (desiredSpeed - this.forwardSpeed) * Math.min(1.0, 6.0 * stepDt);

    const turnRate = overrides.turnRate != null
      ? Number(overrides.turnRate)
      : (this.behavior === 'open_field_explore' ? 0.55 * Math.sin(0.55 * this.timeSec) : 0.0);
    this.yaw += turnRate * stepDt;
    this.x += this.forwardSpeed * Math.cos(this.yaw) * stepDt;
    this.z += this.forwardSpeed * Math.sin(this.yaw) * stepDt;

    this.pitch = -0.05 + 0.16 * command * foreSync;
    this.roll = 0.06 * command * ((limbSwing.FL + limbSwing.HL) - (limbSwing.FR + limbSwing.HR));
    this.stepCount += 1;
    this.timeSec += stepDt;

    const phaseOffsetsRelFR = {};
    for (const limb of RAT_LIMB_NAMES) {
      phaseOffsetsRelFR[limb] = (((this.limbPhase[limb] - this.limbPhase.FR) % 1.0) + 1.0) % 1.0;
    }
    const gait = classifyQuadrupedGait(this.forwardSpeed, phaseOffsetsRelFR, dutyFactor);
    const power = computeMechanicalPower(this.tau, this.dq);
    const stepLengthM = stepFreqHz > 0.1 ? (this.forwardSpeed / stepFreqHz) : 0.0;

    return {
      step: this.stepCount,
      timeSec: this.timeSec,
      command,
      targetSpeedMps: vTarget,
      forwardSpeedMps: this.forwardSpeed,
      gait,
      stepFrequencyHz: stepFreqHz,
      stepLengthM,
      dutyFactor,
      phaseOffsetsRelFR,
      footContacts: { ...this.footContact },
      limbSwing,
      limbStance,
      lumbarExtend: this.q[le0],
      lumbarBend: this.q[lbt0],
      lumbarTwist: this.q[lbt0 + 1],
      q: this.q,
      dq: this.dq,
      tau: this.tau,
      flexorAct: this.flexorAct,
      extensorAct: this.extensorAct,
      power,
      rg: rgState,
      pose: {
        x: this.x,
        z: this.z,
        yaw: this.yaw,
        pitch: this.pitch,
        roll: this.roll,
        bounce: (0.003 + 0.009 * command) * Math.sin(2.0 * Math.PI * this.limbPhase.FR),
      },
    };
  }
}
