/**
 * Mouse motor control, modelled on Annolid's FlyBody brain/VNC controller
 * (`annolid/simulation/fly_brain.py`):
 *
 *   observation + behavior state
 *     -> descending drive   (forward/turn/arrest/direction, gait selection)
 *     -> limb CPG           (4 coupled phase oscillators, gait-specific offsets)
 *     -> motor decoder      (stroke/lift -> named joint targets, clipped to spec)
 *     -> proprioceptive feedback (paw reach/slip -> lift-off reflex)
 *
 * Dependency-free (no Three.js, no DOM) so it can be validated in Node.
 * This is a procedural visualization controller, not a validated biomechanical
 * model of the mouse.
 */

export const MOUSE_LIMBS = Object.freeze(['LF', 'RF', 'LH', 'RH']);

/**
 * Gait library. Offsets are the phase (fraction of a stride) at which each limb
 * starts its swing; `duty` is the stance fraction of the stride.
 * - walk: lateral-sequence walk (LH -> LF -> RH -> RF), high duty factor.
 * - trot: diagonal pairs (LF+RH, RF+LH) in anti-phase; the dominant mouse gait.
 * - gallop: half-bound, hindlimbs near-synchronous and forelimbs leading as a pair.
 */
export const MOUSE_GAITS = Object.freeze({
  walk: Object.freeze({
    offsets: Object.freeze({ LH: 0.00, LF: 0.25, RH: 0.50, RF: 0.75 }),
    duty: 0.70,
    strideScale: 0.85,
  }),
  trot: Object.freeze({
    offsets: Object.freeze({ LH: 0.00, RF: 0.00, RH: 0.50, LF: 0.50 }),
    duty: 0.56,
    strideScale: 1.00,
  }),
  gallop: Object.freeze({
    offsets: Object.freeze({ LH: 0.00, RH: 0.10, LF: 0.55, RF: 0.65 }),
    duty: 0.42,
    strideScale: 1.18,
  }),
});

export const MOUSE_GAIT_NAMES = Object.freeze(Object.keys(MOUSE_GAITS));

// Gait transition thresholds on normalized speed (speed / maxSpeed). The gaps
// between up- and down-thresholds are hysteresis bands that prevent flicker.
export const GAIT_THRESHOLDS = Object.freeze({
  walkToTrot: 0.40,
  trotToWalk: 0.30,
  trotToGallop: 0.92,
  dartTrotToGallop: 0.82,
  gallopToTrot: 0.74,
});

const MIN_LOCOMOTOR_SPEED = 0.26; // scene units/s, matches the scene's isMoving gate
const TURN_STEP_THRESHOLD = 0.45; // rad/s of yaw before in-place stepping engages
const SETTLE_HZ = 2.5; // swing-finish rate once the drive stops
const MAX_STEP_HZ = 11.0;
const DEFAULT_STRIDE_LENGTH = 1.53; // scene units; 2*pi/4.1 from the previous gait

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function finiteOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function wrapPhase(phase) {
  const wrapped = phase - Math.floor(phase);
  return wrapped >= 1 ? 0 : wrapped;
}

/** Signed phase difference wrapped to [-0.5, 0.5). */
export function wrapPhaseSigned(delta) {
  return wrapPhase(delta + 0.5) - 0.5;
}

// ───────────────────────────────────────────────────────────────
// Body / actuator specification (MJCF-like, declarative)
// ───────────────────────────────────────────────────────────────

const LIMB_JOINT_TEMPLATES = Object.freeze({
  hind: Object.freeze([
    { dof: 'hip_flex', segment: 'hip', type: 'hinge', axis: 'x', neutral: -0.24, range: [-0.95, 0.70] },
    { dof: 'hip_abduct', segment: 'hip', type: 'hinge', axis: 'z', neutral: 0.0, range: [-0.50, 0.50] },
    { dof: 'knee_flex', segment: 'knee', type: 'hinge', axis: 'x', neutral: 0.56, range: [0.05, 1.35] },
    { dof: 'ankle_flex', segment: 'ankle', type: 'hinge', axis: 'x', neutral: -0.34, range: [-1.10, 0.30] },
    { dof: 'toe_flex', segment: 'toe', type: 'hinge', axis: 'x', neutral: 0.02, range: [-0.60, 0.50] },
  ]),
  fore: Object.freeze([
    { dof: 'shoulder_protract', segment: 'shoulder', type: 'slide', axis: 'z', neutral: 0.0, range: [-0.05, 0.12] },
    { dof: 'shoulder_flex', segment: 'shoulder', type: 'hinge', axis: 'x', neutral: 0.38, range: [-1.80, 0.90] },
    // Mirrored: left abducts with negative z, right with positive z.
    { dof: 'shoulder_abduct', segment: 'shoulder', type: 'hinge', axis: 'z', neutral: -0.04, range: [-0.60, 0.30], mirror: true },
    { dof: 'elbow_flex', segment: 'elbow', type: 'hinge', axis: 'x', neutral: -0.74, range: [-1.60, 0.10] },
    { dof: 'wrist_flex', segment: 'wrist', type: 'hinge', axis: 'x', neutral: 0.24, range: [-0.80, 1.00] },
    { dof: 'carpal_flex', segment: 'carpal', type: 'hinge', axis: 'x', neutral: 0.12, range: [-0.70, 0.80] },
  ]),
});

function buildJointSpec() {
  const joints = [];
  for (const limb of MOUSE_LIMBS) {
    const side = limb[0];
    const isFore = limb[1] === 'F';
    for (const template of LIMB_JOINT_TEMPLATES[isFore ? 'fore' : 'hind']) {
      const mirrorSign = template.mirror && side === 'R' ? -1 : 1;
      const range = mirrorSign > 0
        ? [template.range[0], template.range[1]]
        : [-template.range[1], -template.range[0]];
      joints.push(Object.freeze({
        name: `${template.dof}_${side}${isFore ? 'F' : 'H'}`,
        limb,
        dof: template.dof,
        segment: template.segment,
        type: template.type,
        axis: template.axis,
        neutral: template.neutral * mirrorSign,
        range: Object.freeze(range),
      }));
    }
  }
  return Object.freeze(joints);
}

/** Every actuated limb DOF, in stable order (the action vector order). */
export const MOUSE_JOINT_SPEC = buildJointSpec();
const JOINT_BY_NAME = new Map(MOUSE_JOINT_SPEC.map(joint => [joint.name, joint]));

/** High-level command interface, analogous to a FlyBody walking task target. */
export const MOUSE_COMMAND_SPEC = Object.freeze({
  forwardSpeed: Object.freeze({ unit: 'scene units/s', minimumScale: -0.25, maximumScale: 1.24 }),
  turnRate: Object.freeze({ unit: 'rad/s', minimum: -3.0, maximum: 3.0 }),
  durationSec: Object.freeze({ unit: 's', minimum: 0, optional: true }),
  gait: Object.freeze({ values: Object.freeze(['auto', ...MOUSE_GAIT_NAMES]), optional: true }),
});

export function getMouseActionSpec() {
  return {
    names: MOUSE_JOINT_SPEC.map(joint => joint.name),
    minimum: MOUSE_JOINT_SPEC.map(joint => joint.range[0]),
    maximum: MOUSE_JOINT_SPEC.map(joint => joint.range[1]),
    neutral: MOUSE_JOINT_SPEC.map(joint => joint.neutral),
    joints: MOUSE_JOINT_SPEC.map(joint => ({
      name: joint.name,
      limb: joint.limb,
      segment: joint.segment,
      type: joint.type,
      axis: joint.axis,
      range: [...joint.range],
      neutral: joint.neutral,
    })),
    gaits: [...MOUSE_GAIT_NAMES],
    command: MOUSE_COMMAND_SPEC,
  };
}

export function clipJointTarget(name, value) {
  const joint = JOINT_BY_NAME.get(name);
  if (!joint) throw new RangeError(`Unknown mouse joint ${JSON.stringify(name)}.`);
  return clamp(finiteOr(value, joint.neutral), joint.range[0], joint.range[1]);
}

// ───────────────────────────────────────────────────────────────
// Descending drive (brain -> spinal cord)
// ───────────────────────────────────────────────────────────────

/** Speed-based gait selection with hysteresis. */
export function selectGait(current, speedNorm, { dart = false } = {}) {
  const t = GAIT_THRESHOLDS;
  const s = finiteOr(speedNorm, 0);
  const gallopEntry = dart ? t.dartTrotToGallop : t.trotToGallop;
  if (current === 'gallop') {
    if (s >= t.gallopToTrot) return 'gallop';
    return s < t.trotToWalk ? 'walk' : 'trot';
  }
  if (current === 'walk') {
    if (s <= t.walkToTrot) return 'walk';
    return s > gallopEntry ? 'gallop' : 'trot';
  }
  if (s < t.trotToWalk) return 'walk';
  return s > gallopEntry ? 'gallop' : 'trot';
}

/**
 * Convert body-level state into a locomotor drive. `posture` other than 'none'
 * (rearing, grooming) arrests stepping, as Pair1/aDN1 do for the fly.
 */
export function computeDescendingDrive({
  speed = 0,
  forwardSpeed = null,
  maxSpeed = 1,
  yawRate = 0,
  posture = 'none',
  dart = false,
  gaitOverride = 'auto',
  previousGait = 'trot',
  strideLength = DEFAULT_STRIDE_LENGTH,
} = {}) {
  const absSpeed = Math.abs(finiteOr(speed, 0));
  const speedNorm = clamp(absSpeed / Math.max(finiteOr(maxSpeed, 1), 1e-3), 0, 1.4);
  const arrest = posture && posture !== 'none' ? 1 : 0;
  const override = String(gaitOverride || 'auto').toLowerCase();
  const gait = MOUSE_GAITS[override]
    ? override
    : selectGait(MOUSE_GAITS[previousGait] ? previousGait : 'trot', speedNorm, { dart });
  const profile = MOUSE_GAITS[gait];
  const stride = Math.max(0.2, finiteOr(strideLength, DEFAULT_STRIDE_LENGTH))
    * profile.strideScale * (0.80 + 0.35 * Math.min(speedNorm, 1));
  const absYawRate = Math.abs(finiteOr(yawRate, 0));
  const locomotorHz = absSpeed > MIN_LOCOMOTOR_SPEED ? absSpeed / stride : 0;
  const turnHz = absYawRate > TURN_STEP_THRESHOLD ? Math.min(3.2, absYawRate * 0.9) : 0;
  const frequencyHz = arrest ? 0 : clamp(Math.max(locomotorHz, turnHz), 0, MAX_STEP_HZ);
  const walkDrive = arrest || frequencyHz <= 0
    ? 0
    : clamp(Math.max(speedNorm * 1.15, turnHz > 0 ? 0.35 : 0), 0, 1.2);
  const signedForward = forwardSpeed == null ? absSpeed : finiteOr(forwardSpeed, absSpeed);
  return {
    gait,
    speedNorm,
    frequencyHz,
    duty: profile.duty,
    walkDrive,
    turn: clamp(finiteOr(yawRate, 0) / 3.0, -1, 1),
    arrest,
    direction: signedForward < -1e-3 ? -1 : 1,
    strideLength: stride,
  };
}

// ───────────────────────────────────────────────────────────────
// Limb central pattern generator (spinal cord)
// ───────────────────────────────────────────────────────────────

export class MouseLimbCPG {
  constructor({ gait = 'trot', coupling = 1.6, jitter = 0 } = {}) {
    this.gait = MOUSE_GAITS[gait] ? gait : 'trot';
    this.coupling = Math.max(0, finiteOr(coupling, 1.6));
    this.duty = MOUSE_GAITS[this.gait].duty;
    this.frequencyHz = 0;
    this.reflexCount = 0;
    const offsets = MOUSE_GAITS[this.gait].offsets;
    const j = finiteOr(jitter, 0);
    // Start with all four feet in stance (staggered by gait offsets) so a
    // stationary or paused animal stands naturally on all four paws, while
    // locomotion immediately breaks into the gait's interlimb coordination.
    this.phases = MOUSE_LIMBS.map((limb, index) => wrapPhase(
      1 - this.duty + clamp(0.24 + offsets[limb] * 0.56 + j * (index % 2 ? 1 : -1), 0.04, 0.94) * this.duty
    ));
  }

  setGait(name) {
    if (MOUSE_GAITS[name]) this.gait = name;
  }

  limbIndex(limb) {
    return MOUSE_LIMBS.indexOf(limb);
  }

  isStance(index) {
    return this.phases[index] >= 1 - this.duty;
  }

  hasSwingingLimb() {
    return this.phases.some((_, index) => !this.isStance(index));
  }

  /** Phase relationship of `index` to the left hindlimb, in [0, 1). */
  relativePhase(index) {
    return wrapPhase(this.phases[index] - this.phases[this.limbIndex('LH')]);
  }

  /**
   * Advance limb phases.
   * @param {number} dt seconds
   * @param {{frequencyHz:number, duty?:number, liftOff?:boolean[]}} drive
   */
  step(dt, { frequencyHz = 0, duty = null, liftOff = null } = {}) {
    const stepDt = finiteOr(dt, 0);
    if (!(stepDt > 0)) return;
    const f = clamp(finiteOr(frequencyHz, 0), 0, MAX_STEP_HZ);
    this.frequencyHz = f;
    const targetDuty = clamp(finiteOr(duty, MOUSE_GAITS[this.gait].duty), 0.3, 0.85);
    this.duty += (targetDuty - this.duty) * (1 - Math.exp(-4.0 * stepDt));

    const offsets = MOUSE_GAITS[this.gait].offsets;
    const next = this.phases.slice();
    if (f > 0) {
      // Kuramoto coupling toward the gait's phase offsets. Coupling is gated by
      // activity so feet planted on a stationary animal never drift.
      const gate = clamp(f, 0, 1);
      const neighborScale = 1 / (MOUSE_LIMBS.length - 1);
      for (let i = 0; i < MOUSE_LIMBS.length; i++) {
        let dphi = f;
        for (let k = 0; k < MOUSE_LIMBS.length; k++) {
          if (k === i) continue;
          const desired = offsets[MOUSE_LIMBS[k]] - offsets[MOUSE_LIMBS[i]];
          const error = wrapPhaseSigned(this.phases[k] - this.phases[i] - desired);
          dphi += this.coupling * gate * neighborScale * Math.sin(2 * Math.PI * error);
        }
        // Phases never run backwards; reversing would drag planted feet.
        next[i] = wrapPhase(this.phases[i] + Math.max(0.25 * f, dphi) * stepDt);
      }
    } else {
      // Drive withdrawn: finish any swing in progress, then hold every limb in
      // stance so the animal stops with all four paws down.
      for (let i = 0; i < MOUSE_LIMBS.length; i++) {
        if (this.isStance(i)) continue;
        const swingEnd = 1 - this.duty;
        next[i] = Math.min(swingEnd, this.phases[i] + SETTLE_HZ * stepDt);
      }
    }
    this.phases = next;

    // Proprioceptive lift-off reflex (extension/slip-triggered swing onset):
    // a late-stance limb that has reached its reach limit starts its swing.
    if (f > 0 && Array.isArray(liftOff)) {
      for (let i = 0; i < MOUSE_LIMBS.length; i++) {
        if (!liftOff[i] || !this.isStance(i)) continue;
        const stanceProgress = (this.phases[i] - (1 - this.duty)) / this.duty;
        if (stanceProgress > 0.55) {
          this.phases[i] = 0;
          this.reflexCount += 1;
        }
      }
    }
  }

  /**
   * Stroke in [-1, 1] (-1 retracted/posterior, +1 protracted/anterior) and lift
   * in [0, 1]. Swing protracts with a smooth cosine; stance retracts linearly,
   * matching a paw fixed on the ground under a body moving at constant speed.
   */
  limbState(index) {
    const phase = this.phases[index];
    const swingEnd = 1 - this.duty;
    if (phase < swingEnd) {
      const s = phase / swingEnd;
      return {
        phase,
        stance: false,
        progress: s,
        stroke: -Math.cos(Math.PI * s),
        lift: Math.sin(Math.PI * Math.pow(s, 0.85)),
      };
    }
    const u = (phase - swingEnd) / this.duty;
    return { phase, stance: true, progress: u, stroke: 1 - 2 * u, lift: 0 };
  }

  snapshot() {
    const phases = {};
    const stance = {};
    MOUSE_LIMBS.forEach((limb, index) => {
      phases[limb] = this.phases[index];
      stance[limb] = this.isStance(index);
    });
    return {
      gait: this.gait,
      frequencyHz: this.frequencyHz,
      duty: this.duty,
      phases,
      stance,
      reflexCount: this.reflexCount,
    };
  }
}

// ───────────────────────────────────────────────────────────────
// Motor decoder (CPG phase -> named joint targets)
// ───────────────────────────────────────────────────────────────

/**
 * Map CPG state into joint targets for every DOF in MOUSE_JOINT_SPEC.
 * Sign conventions follow the scene rig: positive flexion about +x moves a
 * downward-pointing segment backwards, so protraction is a negative hip/shoulder
 * angle.
 */
export function decodeLocomotorTargets(cpg, drive, {
  amplitude = 0,
  gaitAmpMul = 1,
  idleSway = 0,
} = {}, out = {}) {
  const amp = clamp(finiteOr(amplitude, 0), 0, 1) * clamp(finiteOr(gaitAmpMul, 1), 0.5, 1.5);
  const turn = clamp(finiteOr(drive?.turn, 0), -1, 1);
  const direction = drive?.direction === -1 ? -1 : 1;
  const sway = finiteOr(idleSway, 0);
  for (let index = 0; index < MOUSE_LIMBS.length; index++) {
    const limb = MOUSE_LIMBS[index];
    const isLeft = limb[0] === 'L';
    const sideSign = isLeft ? 1 : -1;
    const state = cpg.limbState(index);
    // Turning: inner-side limbs take shorter strides (left turn = positive yaw).
    const turnScale = 1 + (isLeft ? -1 : 1) * 0.32 * turn;
    const x = state.stroke * amp * turnScale * direction;
    // Settling swings still clear the floor even when the drive is gone.
    const h = state.lift * clamp(0.40 + 0.60 * amp, 0, 1);
    if (limb[1] === 'H') {
      out[`hip_flex_${limb}`] = -0.24 - 0.24 * x;
      out[`hip_abduct_${limb}`] = sideSign * 0.03 * h;
      out[`knee_flex_${limb}`] = 0.56 + 0.26 * h;
      out[`ankle_flex_${limb}`] = -0.34 - 0.14 * h + 0.08 * x;
      out[`toe_flex_${limb}`] = 0.02 + 0.22 * h - 0.05 * x;
    } else {
      out[`shoulder_protract_${limb}`] = 0.0;
      out[`shoulder_flex_${limb}`] = 0.38 - 0.22 * x;
      out[`shoulder_abduct_${limb}`] = -sideSign * (0.04 + 0.03 * h) + sideSign * sway;
      out[`elbow_flex_${limb}`] = -0.74 - 0.20 * h;
      out[`wrist_flex_${limb}`] = 0.24 + 0.34 * h + 0.05 * x;
      out[`carpal_flex_${limb}`] = 0.12 + 0.18 * h - 0.04 * x;
    }
  }
  for (const joint of MOUSE_JOINT_SPEC) {
    out[joint.name] = clamp(finiteOr(out[joint.name], joint.neutral), joint.range[0], joint.range[1]);
  }
  return out;
}

// ───────────────────────────────────────────────────────────────
// Controller facade
// ───────────────────────────────────────────────────────────────

export class MouseMotorController {
  constructor({ gait = 'trot', jitter = 0, coupling = 1.6 } = {}) {
    this.cpg = new MouseLimbCPG({ gait, jitter, coupling });
    this.amplitude = 0;
    this.drive = computeDescendingDrive({ previousGait: this.cpg.gait });
    this.targets = decodeLocomotorTargets(this.cpg, this.drive, { amplitude: 0 });
  }

  /**
   * One control step. Returns joint targets keyed by MOUSE_JOINT_SPEC names.
   * When dt <= 0 (paused, initial pose) state is held and targets re-decoded.
   */
  step(dt, input = {}) {
    const stepDt = finiteOr(dt, 0);
    const drive = computeDescendingDrive({ ...input, previousGait: this.cpg.gait });
    this.drive = drive;
    if (stepDt > 0) {
      if (drive.gait !== this.cpg.gait) this.cpg.setGait(drive.gait);
      this.cpg.step(stepDt, {
        frequencyHz: drive.frequencyHz,
        duty: drive.duty,
        liftOff: input.liftOff,
      });
      const response = drive.walkDrive > this.amplitude ? 7.0 : 3.5;
      this.amplitude += (drive.walkDrive - this.amplitude) * (1 - Math.exp(-response * stepDt));
      if (this.amplitude < 1e-4) this.amplitude = 0;
    }
    this.targets = decodeLocomotorTargets(this.cpg, drive, {
      amplitude: this.amplitude,
      gaitAmpMul: input.gaitAmpMul,
      idleSway: input.idleSway,
    }, this.targets);
    return this.targets;
  }

  /** True while any limb is still cycling (stepping or finishing a swing). */
  isStepping() {
    return this.cpg.frequencyHz > 0 || this.cpg.hasSwingingLimb();
  }

  isLimbInStance(limb) {
    const index = this.cpg.limbIndex(limb);
    return index < 0 ? true : this.cpg.isStance(index);
  }

  /** Vertical body oscillation, two bumps per stride (mid-stance highs). */
  bodyOscillation() {
    const lh = this.cpg.phases[this.cpg.limbIndex('LH')];
    return Math.sin(4 * Math.PI * lh) * this.amplitude;
  }

  /** Lateral spinal undulation phase-locked to the hindlimb cycle. */
  spinalWave() {
    const lh = this.cpg.phases[this.cpg.limbIndex('LH')];
    return Math.sin(2 * Math.PI * lh) * this.amplitude;
  }

  snapshot() {
    return {
      ...this.cpg.snapshot(),
      amplitude: this.amplitude,
      descending: {
        walkDrive: this.drive.walkDrive,
        turn: this.drive.turn,
        arrest: this.drive.arrest,
        direction: this.drive.direction,
        speedNorm: this.drive.speedNorm,
        strideLength: this.drive.strideLength,
      },
    };
  }
}

/** Validate and normalize an external motor command. Throws on bad input. */
export function sanitizeMotorCommand(command, { maxSpeed = 1 } = {}) {
  if (!command || typeof command !== 'object') {
    throw new TypeError('Motor command must be an object with forwardSpeed and/or turnRate.');
  }
  const speedCap = Math.max(1e-3, finiteOr(maxSpeed, 1));
  const forward = Number(command.forwardSpeed ?? 0);
  const turn = Number(command.turnRate ?? 0);
  if (!Number.isFinite(forward) || !Number.isFinite(turn)) {
    throw new TypeError('Motor command forwardSpeed and turnRate must be finite numbers.');
  }
  let durationSec = Infinity;
  if (command.durationSec != null) {
    durationSec = Number(command.durationSec);
    if (!Number.isFinite(durationSec) || durationSec <= 0) {
      throw new RangeError('Motor command durationSec must be a positive finite number.');
    }
  }
  const gait = String(command.gait ?? 'auto').toLowerCase();
  if (gait !== 'auto' && !MOUSE_GAITS[gait]) {
    throw new RangeError(`Unknown gait ${JSON.stringify(command.gait)}; expected auto or ${MOUSE_GAIT_NAMES.join(', ')}.`);
  }
  const spec = MOUSE_COMMAND_SPEC;
  return {
    forwardSpeed: clamp(forward, spec.forwardSpeed.minimumScale * speedCap, spec.forwardSpeed.maximumScale * speedCap),
    turnRate: clamp(turn, spec.turnRate.minimum, spec.turnRate.maximum),
    durationSec,
    gait,
  };
}
