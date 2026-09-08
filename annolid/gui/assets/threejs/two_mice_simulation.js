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
