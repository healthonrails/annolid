/**
 * Biophysical Drosophila connectome controller driving the Three.js mouse rig.
 *
 * Inspired by:
 *   - Annolid's `annolid/simulation/fly_brain.py` (canonical Drosophila Brain + VNC
 *     connectome and LIF spiking parameters from Shiu et al. 2024 / FlyWire / MaleCNS)
 *   - `nftechie/stonkfly` (PAM reward & PPL1 aversive dopamine plasticity modulating
 *     Kenyon-cell -> MBON synaptic weights in closed sensorimotor loops)
 *   - `FlyHead` / `fly-brain-drone` (zero-training LC4 / ORN / PFL3 / DNa02 / MDN steering)
 *
 * Dependency-free (no Three.js, no DOM) so it can be validated deterministically in Node.
 * This is an illustrative cross-species NeuroAI visualization, not a validated mouse brain model.
 */

export const FLY_BRAIN_METADATA = Object.freeze({
  id: 'drosophila-lif-mouse-bridge-v1',
  title: 'Drosophila LIF Connectome -> Mouse Motor Bridge',
  referenceDatasets: Object.freeze(['male-cns:v1.0', 'flywire-fafb:v783', 'manc:v1.2.1']),
  dynamicsModes: Object.freeze(['spiking', 'rate']),
});

export const FLY_NEURON_GROUPS = Object.freeze([
  // Sensory neuropils (Mouse world -> Drosophila sensory populations)
  { id: 'ORN_L', region: 'sensory', nt: 'ach', side: 'L', tau: 0.018, bias: 0.22, count: 10, label: 'Olfactory L' },
  { id: 'ORN_R', region: 'sensory', nt: 'ach', side: 'R', tau: 0.018, bias: 0.22, count: 10, label: 'Olfactory R' },
  { id: 'Visual_Target_L', region: 'sensory', nt: 'ach', side: 'L', tau: 0.014, bias: 0.22, count: 10, label: 'LC10a Vis L' },
  { id: 'Visual_Target_R', region: 'sensory', nt: 'ach', side: 'R', tau: 0.014, bias: 0.22, count: 10, label: 'LC10a Vis R' },
  { id: 'LC4_Looming', region: 'sensory', nt: 'ach', side: 'B', tau: 0.007, bias: -0.05, count: 12, label: 'LC4 Looming' },
  { id: 'Tactile_Ant', region: 'sensory', nt: 'ach', side: 'B', tau: 0.010, bias: 0.0, count: 8, label: 'Whisker Wall' },
  { id: 'JO_Mech', region: 'sensory', nt: 'ach', side: 'B', tau: 0.012, bias: 0.0, count: 8, label: 'JO Groom Mech' },
  { id: 'FeCO_Load_L', region: 'sensory', nt: 'ach', side: 'L', tau: 0.010, bias: 0.0, count: 6, label: 'FeCO Load L' },
  { id: 'FeCO_Load_R', region: 'sensory', nt: 'ach', side: 'R', tau: 0.010, bias: 0.0, count: 6, label: 'FeCO Load R' },

  // Mushroom Body + Dopaminergic Reinforcement (stonkfly-style PAM / PPL1 plasticity)
  { id: 'KC_Reward_Zone', region: 'mushroom_body', nt: 'ach', side: 'B', tau: 0.018, bias: 0.0, count: 14, label: 'KC Reward Cue' },
  { id: 'KC_Hazard_Zone', region: 'mushroom_body', nt: 'ach', side: 'B', tau: 0.018, bias: 0.0, count: 14, label: 'KC Hazard Cue' },
  { id: 'KC_Partner', region: 'mushroom_body', nt: 'ach', side: 'B', tau: 0.018, bias: 0.0, count: 12, label: 'KC Social Cue' },
  { id: 'MBON_Approach', region: 'mushroom_body', nt: 'ach', side: 'B', tau: 0.016, bias: 0.0, count: 8, label: 'MBON Approach' },
  { id: 'MBON_Avoid', region: 'mushroom_body', nt: 'ach', side: 'B', tau: 0.016, bias: 0.0, count: 8, label: 'MBON Avoid' },
  { id: 'PAM', region: 'dopamine', nt: 'da', side: 'B', tau: 0.020, bias: 0.0, count: 8, label: 'PAM Reward DA' },
  { id: 'PPL1', region: 'dopamine', nt: 'da', side: 'B', tau: 0.020, bias: 0.0, count: 8, label: 'PPL1 Aversive DA' },

  // Central Complex (CX), Lateral Accessory Lobe (LAL), SEZ, and State Hub
  { id: 'CX_Forward_Goal', region: 'central_complex', nt: 'ach', side: 'B', tau: 0.022, bias: 0.22, count: 10, label: 'CX Forward Goal' },
  { id: 'PFL3_L', region: 'central_complex', nt: 'ach', side: 'L', tau: 0.016, bias: 0.20, count: 8, label: 'PFL3 Steer L' },
  { id: 'PFL3_R', region: 'central_complex', nt: 'ach', side: 'R', tau: 0.016, bias: 0.20, count: 8, label: 'PFL3 Steer R' },
  { id: 'PFL2_Brake', region: 'central_complex', nt: 'ach', side: 'B', tau: 0.018, bias: 0.0, count: 6, label: 'PFL2 Brake' },
  { id: 'LAL_Inh_L', region: 'lal', nt: 'gaba', side: 'L', tau: 0.014, bias: 0.0, count: 8, label: 'LAL GABA L' },
  { id: 'LAL_Inh_R', region: 'lal', nt: 'gaba', side: 'R', tau: 0.014, bias: 0.0, count: 8, label: 'LAL GABA R' },
  { id: 'SEZ_Groom', region: 'sez', nt: 'ach', side: 'B', tau: 0.020, bias: 0.0, count: 8, label: 'SEZ Groom Hub' },
  { id: 'P1_Explore', region: 'central_complex', nt: 'oa', side: 'B', tau: 0.022, bias: 0.0, count: 8, label: 'P1 Explore/Rear' },

  // Neck-Connective Descending Neurons (Brain -> Mouse Spinal CPG & Behaviors)
  { id: 'DNp09_L', region: 'descending', nt: 'ach', side: 'L', tau: 0.015, bias: 0.0, count: 4, label: 'DNp09 Walk L' },
  { id: 'DNp09_R', region: 'descending', nt: 'ach', side: 'R', tau: 0.015, bias: 0.0, count: 4, label: 'DNp09 Walk R' },
  { id: 'DNa02_L', region: 'descending', nt: 'ach', side: 'L', tau: 0.011, bias: 0.16, count: 4, label: 'DNa02 Turn L' },
  { id: 'DNa02_R', region: 'descending', nt: 'ach', side: 'R', tau: 0.011, bias: 0.16, count: 4, label: 'DNa02 Turn R' },
  { id: 'MDN', region: 'descending', nt: 'ach', side: 'B', tau: 0.015, bias: -0.08, count: 6, label: 'MDN Moonwalk' },
  { id: 'Pair1', region: 'descending', nt: 'gaba', side: 'B', tau: 0.014, bias: -0.05, count: 4, label: 'Pair1 Arrest' },
  { id: 'aDN1', region: 'descending', nt: 'ach', side: 'B', tau: 0.016, bias: -0.06, count: 6, label: 'aDN1 Groom' },
  { id: 'DNp01', region: 'descending', nt: 'ach', side: 'B', tau: 0.006, bias: -0.18, count: 4, label: 'DNp01 Giant Fiber' },
  { id: 'DNg02', region: 'descending', nt: 'ach', side: 'B', tau: 0.014, bias: -0.06, count: 6, label: 'DNg02 Elevate/Rear' },
  { id: 'VNC_MDN_Inh', region: 'vnc', nt: 'gaba', side: 'B', tau: 0.012, bias: 0.0, count: 6, label: 'VNC MDN GABA' },
]);

/**
 * Canonical Drosophila connectome pathways.
 * `synapses` is scaled relative to 200 synapses (~1.0 unit synaptic weight).
 * `plastic` marks KC -> MBON synapses modulated online by PAM / PPL1 dopamine.
 */
export const FLY_CONNECTOME_EDGES = Object.freeze([
  // Sensory -> Central Complex & Steering
  { pre: 'ORN_L', post: 'PFL3_L', synapses: 170 },
  { pre: 'ORN_R', post: 'PFL3_R', synapses: 170 },
  { pre: 'ORN_L', post: 'CX_Forward_Goal', synapses: 90 },
  { pre: 'ORN_R', post: 'CX_Forward_Goal', synapses: 90 },
  { pre: 'Visual_Target_L', post: 'PFL3_L', synapses: 180 },
  { pre: 'Visual_Target_R', post: 'PFL3_R', synapses: 180 },
  { pre: 'Visual_Target_L', post: 'DNa02_L', synapses: 120 },
  { pre: 'Visual_Target_R', post: 'DNa02_R', synapses: 120 },
  { pre: 'Visual_Target_L', post: 'CX_Forward_Goal', synapses: 70 },
  { pre: 'Visual_Target_R', post: 'CX_Forward_Goal', synapses: 70 },

  // Plastic Mushroom Body KC -> MBON associative pathways (stonkfly-style)
  { pre: 'KC_Reward_Zone', post: 'MBON_Approach', synapses: 180, plastic: true, key: 'reward_approach' },
  { pre: 'KC_Reward_Zone', post: 'MBON_Avoid', synapses: 90, plastic: true, key: 'reward_avoid' },
  { pre: 'KC_Hazard_Zone', post: 'MBON_Approach', synapses: 90, plastic: true, key: 'hazard_approach' },
  { pre: 'KC_Hazard_Zone', post: 'MBON_Avoid', synapses: 180, plastic: true, key: 'hazard_avoid' },
  { pre: 'KC_Partner', post: 'MBON_Approach', synapses: 150, plastic: true, key: 'social_approach' },
  { pre: 'KC_Partner', post: 'MBON_Avoid', synapses: 85, plastic: true, key: 'social_avoid' },

  // MBON Valence -> Locomotion, Exploration, and Retreat
  { pre: 'MBON_Approach', post: 'CX_Forward_Goal', synapses: 190 },
  { pre: 'MBON_Approach', post: 'P1_Explore', synapses: 130 },
  { pre: 'MBON_Avoid', post: 'MDN', synapses: 185 },
  { pre: 'MBON_Avoid', post: 'PFL2_Brake', synapses: 110 },

  // Central Complex -> Forward Locomotion & Bilateral Steering with LAL Flip-Flop
  { pre: 'CX_Forward_Goal', post: 'DNp09_L', synapses: 220 },
  { pre: 'CX_Forward_Goal', post: 'DNp09_R', synapses: 220 },
  { pre: 'DNp09_L', post: 'DNp09_R', synapses: 45 },
  { pre: 'DNp09_R', post: 'DNp09_L', synapses: 45 },
  { pre: 'PFL3_L', post: 'DNa02_L', synapses: 220 },
  { pre: 'PFL3_R', post: 'DNa02_R', synapses: 220 },
  { pre: 'PFL3_L', post: 'LAL_Inh_L', synapses: 160 },
  { pre: 'PFL3_R', post: 'LAL_Inh_R', synapses: 160 },
  { pre: 'LAL_Inh_L', post: 'DNa02_R', synapses: 210, nt: 'gaba' },
  { pre: 'LAL_Inh_R', post: 'DNa02_L', synapses: 210, nt: 'gaba' },
  { pre: 'PFL2_Brake', post: 'Pair1', synapses: 160 },

  // Grooming circuit: JO_Mech -> SEZ_Groom -> aDN1 + Pair1 arrest
  { pre: 'JO_Mech', post: 'SEZ_Groom', synapses: 220 },
  { pre: 'SEZ_Groom', post: 'aDN1', synapses: 250 },
  { pre: 'SEZ_Groom', post: 'Pair1', synapses: 210 },
  { pre: 'aDN1', post: 'Pair1', synapses: 170 },

  // Pair1 GABAergic Arrest suppresses forward walking DNs
  { pre: 'Pair1', post: 'DNp09_L', synapses: 290, nt: 'gaba' },
  { pre: 'Pair1', post: 'DNp09_R', synapses: 290, nt: 'gaba' },

  // Anterior tactile collision -> Moonwalker (MDN) -> backward walk & DNp09 suppression
  { pre: 'Tactile_Ant', post: 'MDN', synapses: 260 },
  { pre: 'MDN', post: 'VNC_MDN_Inh', synapses: 220 },
  { pre: 'VNC_MDN_Inh', post: 'DNp09_L', synapses: 260, nt: 'gaba' },
  { pre: 'VNC_MDN_Inh', post: 'DNp09_R', synapses: 260, nt: 'gaba' },

  // Looming visual threat -> LC4 -> Giant Fiber (DNp01) Escape Burst
  { pre: 'LC4_Looming', post: 'DNp01', synapses: 340 },
  { pre: 'DNp01', post: 'DNp09_L', synapses: 180 },
  { pre: 'DNp01', post: 'DNp09_R', synapses: 180 },
  { pre: 'DNp01', post: 'Pair1', synapses: 240, nt: 'gaba' },

  // Exploration / Elevation arousal -> DNg02 (drives vertical rearing in quadruped)
  { pre: 'P1_Explore', post: 'DNg02', synapses: 240 },
  { pre: 'DNg02', post: 'Pair1', synapses: 140 },

  // Proprioceptive load stabilization
  { pre: 'FeCO_Load_L', post: 'DNp09_L', synapses: 45 },
  { pre: 'FeCO_Load_R', post: 'DNp09_R', synapses: 45 },
]);

export const FLY_OPTOGENETIC_PRESETS = Object.freeze([
  { id: 'walk', label: 'Walk (DNp09)', targets: [{ cellType: 'DNp09_L', current: 1.5 }, { cellType: 'DNp09_R', current: 1.5 }], durationSec: 2.2 },
  { id: 'turn_left', label: 'Turn L (DNa02_L)', targets: [{ cellType: 'DNa02_L', current: 1.8 }, { cellType: 'DNa02_R', current: -0.9 }], durationSec: 1.8 },
  { id: 'turn_right', label: 'Turn R (DNa02_R)', targets: [{ cellType: 'DNa02_R', current: 1.8 }, { cellType: 'DNa02_L', current: -0.9 }], durationSec: 1.8 },
  { id: 'moonwalk', label: 'Moonwalk (MDN)', targets: [{ cellType: 'MDN', current: 1.9 }], durationSec: 2.0 },
  { id: 'freeze', label: 'Freeze (Pair1)', targets: [{ cellType: 'Pair1', current: 1.9 }], durationSec: 2.2 },
  { id: 'groom', label: 'Groom (aDN1)', targets: [{ cellType: 'aDN1', current: 1.9 }], durationSec: 2.4 },
  { id: 'rear', label: 'Rear (DNg02)', targets: [{ cellType: 'DNg02', current: 1.9 }], durationSec: 2.2 },
  { id: 'escape', label: 'Escape (LC4→DNp01)', targets: [{ cellType: 'LC4_Looming', current: 1.9 }, { cellType: 'DNp01', current: 1.2 }], durationSec: 1.6 },
  { id: 'reward_da', label: 'Reward (PAM DA)', targets: [{ cellType: 'PAM', current: 1.8 }], durationSec: 1.5, dopamine: 'PAM' },
  { id: 'punish_da', label: 'Punish (PPL1 DA)', targets: [{ cellType: 'PPL1', current: 1.8 }], durationSec: 1.5, dopamine: 'PPL1' },
]);

const LIF_V_REST_MV = -52.0;
const LIF_V_RESET_MV = -52.0;
const LIF_V_THRESH_MV = -45.0;
const LIF_REFRACTORY_SEC = 0.0022;
const LIF_TAU_SYN_SEC = 0.005;
const LIF_TAU_RATE_SEC = 0.045;
const LIF_DRIVE_SCALE_MV = 16.0;
const LIF_SYN_PULSE_MV = 31.0;
const LIF_SUB_DT_SEC = 0.0005;

const PLASTIC_MIN_SCALE = 0.15;
const PLASTIC_MAX_SCALE = 2.4;
const ELIGIBILITY_TAU_SEC = 2.0;
const PLASTICITY_LR = 0.72;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function makeRng(seed) {
  let state = (seed >>> 0) || 0x12345678;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function neurotransmitterPolarity(nt) {
  switch (String(nt || 'ach').toLowerCase()) {
    case 'gaba': return -1.0;
    case 'glu': return -0.6;
    case 'oa': return 0.85;
    case 'da': return 0.5;
    case '5ht': return -0.3;
    default: return 1.0;
  }
}

/**
 * Biophysical spiking/rate Drosophila connectome wired to mouse sensory inputs
 * and motor/behavioral readouts.
 */
export class FlyBrainMouseController {
  constructor({
    seed = 7,
    dynamics = 'spiking',
    plasticityEnabled = true,
    connectionProbability = 0.55,
  } = {}) {
    this.seed = seed >>> 0;
    this.dynamics = dynamics === 'rate' ? 'rate' : 'spiking';
    this.plasticityEnabled = Boolean(plasticityEnabled);
    this.connectionProbability = clamp(Number(connectionProbability) || 0.55, 0.15, 1.0);
    this._rng = makeRng(this.seed);

    this.groupMeta = FLY_NEURON_GROUPS;
    this.groupById = new Map();
    this.groupRanges = new Map();

    // Instantiate individual heterogeneous neurons per canonical population.
    const taus = [];
    const biases = [];
    const neuronGroupIndex = [];
    let offset = 0;
    this.groupMeta.forEach((group, gIdx) => {
      this.groupById.set(group.id, { ...group, index: gIdx });
      this.groupRanges.set(group.id, { start: offset, count: group.count });
      for (let k = 0; k < group.count; k++) {
        const tauJitter = group.tau * (0.90 + this._rng() * 0.20);
        taus.push(Math.max(0.003, tauJitter));
        biases.push(group.bias);
        neuronGroupIndex.push(gIdx);
      }
      offset += group.count;
    });
    this.numNeurons = offset;
    this.tau = new Float64Array(taus);
    this.bias = new Float64Array(biases);
    this.neuronGroupIndex = new Int32Array(neuronGroupIndex);

    // Build sparse pre -> post adjacency lists (CSR-like by presynaptic neuron)
    // so spiking propagation is strictly O(active spikes * avg out-degree).
    const outBuckets = Array.from({ length: this.numNeurons }, () => []);
    this.plasticPathways = new Map();

    for (const edge of FLY_CONNECTOME_EDGES) {
      const preRange = this.groupRanges.get(edge.pre);
      const postRange = this.groupRanges.get(edge.post);
      const preMeta = this.groupById.get(edge.pre);
      if (!preRange || !postRange || !preMeta) continue;
      const polarity = neurotransmitterPolarity(edge.nt ?? preMeta.nt);
      const baseStrength = (edge.synapses / 200.0) * polarity;
      const smallBottleneck = preRange.count * postRange.count <= 32;
      const plasticKey = edge.plastic ? edge.key : null;

      if (plasticKey) {
        this.plasticPathways.set(plasticKey, {
          key: plasticKey,
          pre: edge.pre,
          post: edge.post,
          scale: 1.0,
          eligibility: 0.0,
        });
      }

      for (let p = 0; p < postRange.count; p++) {
        const postIdx = postRange.start + p;
        const selectedPre = [];
        for (let s = 0; s < preRange.count; s++) {
          if (smallBottleneck || this._rng() < this.connectionProbability) {
            selectedPre.push(preRange.start + s);
          }
        }
        if (selectedPre.length === 0) {
          selectedPre.push(preRange.start + Math.floor(this._rng() * preRange.count));
        }
        const fanIn = selectedPre.length;
        for (const preIdx of selectedPre) {
          const jitter = 0.88 + this._rng() * 0.24;
          outBuckets[preIdx].push({
            post: postIdx,
            weight: (baseStrength * jitter) / fanIn,
            plasticKey,
          });
        }
      }
    }

    // Flatten into compact typed arrays for fast inner-loop stepping.
    const outStart = new Int32Array(this.numNeurons + 1);
    let totalEdges = 0;
    for (let i = 0; i < this.numNeurons; i++) {
      outStart[i] = totalEdges;
      totalEdges += outBuckets[i].length;
    }
    outStart[this.numNeurons] = totalEdges;
    this.outStart = outStart;
    this.edgePost = new Int32Array(totalEdges);
    this.edgeBaseWeight = new Float64Array(totalEdges);
    this.edgePlasticSlot = new Int32Array(totalEdges);
    this.plasticKeys = Array.from(this.plasticPathways.keys());
    const plasticKeyToSlot = new Map(this.plasticKeys.map((k, idx) => [k, idx]));
    this.plasticScales = new Float64Array(this.plasticKeys.length).fill(1.0);
    this.plasticEligibility = new Float64Array(this.plasticKeys.length).fill(0.0);

    let cursor = 0;
    for (let i = 0; i < this.numNeurons; i++) {
      for (const e of outBuckets[i]) {
        this.edgePost[cursor] = e.post;
        this.edgeBaseWeight[cursor] = e.weight;
        this.edgePlasticSlot[cursor] = e.plasticKey ? plasticKeyToSlot.get(e.plasticKey) : -1;
        cursor++;
      }
    }
    this.numSynapses = totalEdges;

    // Neural state vectors
    this.voltage = new Float64Array(this.numNeurons);
    this.rates = new Float64Array(this.numNeurons);
    this.synConductance = new Float64Array(this.numNeurons);
    this.refractoryRem = new Float64Array(this.numNeurons);
    this.extDrive = new Float64Array(this.numNeurons);
    this.groupRates = new Float64Array(this.groupMeta.length);

    this.stimulations = [];
    this.simTime = 0;
    this.lastStepSpikes = 0;
    this.totalSpikes = 0;
    this.recentSpikes = []; // ring buffer of { t, neuron, groupIndex, region } for raster HUD
    this.lastSensory = {};
    this.lastMotorOutput = {
      forwardSpeed: 0,
      turnRate: 0,
      gait: 'auto',
      behaviorTrigger: 'none',
      arrest: false,
      dominantCircuit: 'foraging_exploration',
    };

    this.reset();
  }

  reset() {
    const rng = makeRng(this.seed ^ 0x9E3779B9);
    for (let i = 0; i < this.numNeurons; i++) {
      this.voltage[i] = this.dynamics === 'spiking'
        ? LIF_V_REST_MV + (rng() - 0.5) * 3.0
        : 0.0;
    }
    this.rates.fill(0);
    this.synConductance.fill(0);
    this.refractoryRem.fill(0);
    this.groupRates.fill(0);
    this.stimulations = [];
    this.simTime = 0;
    this.lastStepSpikes = 0;
    this.totalSpikes = 0;
    this.recentSpikes = [];
    this.resetPlasticity();
  }

  resetPlasticity() {
    this.plasticScales.fill(1.0);
    this.plasticEligibility.fill(0.0);
    for (const p of this.plasticPathways.values()) {
      p.scale = 1.0;
      p.eligibility = 0.0;
    }
  }

  setDynamics(mode) {
    const next = mode === 'rate' ? 'rate' : 'spiking';
    if (next === this.dynamics) return;
    this.dynamics = next;
    for (let i = 0; i < this.numNeurons; i++) {
      this.voltage[i] = next === 'spiking' ? LIF_V_REST_MV : this.rates[i];
    }
    this.synConductance.fill(0);
    this.refractoryRem.fill(0);
  }

  /**
   * Apply a Chrimson (current > 0) or GtACR1 (current < 0) optogenetic pulse
   * to a named Drosophila neuron population.
   */
  stimulate(cellType, current = 1.6, durationSec = 1.8) {
    if (!this.groupRanges.has(cellType)) {
      throw new RangeError(`Unknown Drosophila neuron group ${JSON.stringify(cellType)}.`);
    }
    const clampedCurrent = clamp(Number(current) || 0, -2.5, 2.5);
    const clampedDuration = clamp(Number(durationSec) || 0, 0.05, 15.0);
    this.stimulations.push({
      cellType,
      current: clampedCurrent,
      remainingSec: clampedDuration,
    });
    return { cellType, current: clampedCurrent, durationSec: clampedDuration };
  }

  applyPreset(presetId) {
    const preset = FLY_OPTOGENETIC_PRESETS.find(p => p.id === presetId);
    if (!preset) {
      throw new RangeError(`Unknown fly optogenetic preset ${JSON.stringify(presetId)}.`);
    }
    for (const target of preset.targets) {
      this.stimulate(target.cellType, target.current, preset.durationSec);
    }
    return preset;
  }

  clearStimulations() {
    this.stimulations = [];
  }

  rate(cellType) {
    const meta = this.groupById.get(cellType);
    return meta ? this.groupRates[meta.index] : 0;
  }

  _addDrive(cellType, value) {
    if (!value) return;
    const range = this.groupRanges.get(cellType);
    if (!range) return;
    const end = range.start + range.count;
    for (let i = range.start; i < end; i++) {
      this.extDrive[i] += value;
    }
  }

  /**
   * Encode mouse egocentric arena observations into Drosophila sensory populations.
   */
  _encodeSensory(sensory = {}) {
    this.extDrive.fill(0);
    const odorLeft = clamp(Number(sensory.odorLeft) || 0, 0, 2.0);
    const odorRight = clamp(Number(sensory.odorRight) || 0, 0, 2.0);
    const visLeft = clamp(Number(sensory.visualLeft) || 0, 0, 2.0);
    const visRight = clamp(Number(sensory.visualRight) || 0, 0, 2.0);
    const looming = clamp(Number(sensory.looming) || 0, 0, 2.0);
    const wallAhead = clamp(Number(sensory.wallAhead) || 0, 0, 2.0);
    const groomUrge = clamp(Number(sensory.groomUrge) || 0, 0, 2.0);
    const loadLeft = clamp(Number(sensory.pawLoadLeft) || 0, 0, 1.5);
    const loadRight = clamp(Number(sensory.pawLoadRight) || 0, 0, 1.5);
    const rewardCue = clamp(Number(sensory.rewardCue) || 0, 0, 1.5);
    const hazardCue = clamp(Number(sensory.hazardCue) || 0, 0, 1.5);
    const partnerCue = clamp(Number(sensory.partnerCue) || 0, 0, 1.5);
    const rewardDA = clamp(Number(sensory.rewardDopamine) || 0, 0, 2.0);
    const punishDA = clamp(Number(sensory.punishDopamine) || 0, 0, 2.0);
    const exploreDrive = clamp(Number(sensory.exploreDrive ?? 0.55), 0, 1.6);

    this._addDrive('ORN_L', odorLeft);
    this._addDrive('ORN_R', odorRight);
    this._addDrive('Visual_Target_L', visLeft);
    this._addDrive('Visual_Target_R', visRight);
    this._addDrive('LC4_Looming', looming);
    this._addDrive('Tactile_Ant', wallAhead);
    this._addDrive('JO_Mech', groomUrge);
    this._addDrive('FeCO_Load_L', loadLeft);
    this._addDrive('FeCO_Load_R', loadRight);
    this._addDrive('KC_Reward_Zone', rewardCue);
    this._addDrive('KC_Hazard_Zone', hazardCue);
    this._addDrive('KC_Partner', partnerCue);
    this._addDrive('PAM', rewardDA);
    this._addDrive('PPL1', punishDA);
    this._addDrive('CX_Forward_Goal', exploreDrive);

    // Subtle oscillatory intrinsic search bias so an unstimulated fly brain
    // naturally weaves and investigates rather than running dead straight.
    const searchWave = Math.sin(this.simTime * 1.15 + (this.seed % 17) * 0.4);
    this._addDrive('PFL3_L', Math.max(0, searchWave) * 0.34);
    this._addDrive('PFL3_R', Math.max(0, -searchWave) * 0.34);

    // Active optogenetic pulses override/modulate target populations directly.
    for (const stim of this.stimulations) {
      this._addDrive(stim.cellType, stim.current);
    }

    this.lastSensory = {
      odorLeft,
      odorRight,
      visualLeft: visLeft,
      visualRight: visRight,
      looming,
      wallAhead,
      groomUrge,
      rewardCue,
      hazardCue,
      partnerCue,
      rewardDopamine: rewardDA,
      punishDopamine: punishDA,
    };
  }

  _updateGroupRates() {
    for (let g = 0; g < this.groupMeta.length; g++) {
      const range = this.groupRanges.get(this.groupMeta[g].id);
      let sum = 0;
      const end = range.start + range.count;
      for (let i = range.start; i < end; i++) sum += this.rates[i];
      this.groupRates[g] = sum / range.count;
    }
  }

  _integrate(totalDt) {
    const dtClamped = clamp(Number(totalDt) || 0, 0, 0.1);
    if (dtClamped <= 0) return 0;

    if (this.dynamics === 'spiking') {
      const substeps = Math.max(1, Math.round(dtClamped / LIF_SUB_DT_SEC));
      const dt = dtClamped / substeps;
      const synDecay = Math.exp(-dt / LIF_TAU_SYN_SEC);
      const rateDecay = Math.exp(-dt / LIF_TAU_RATE_SEC);
      const spikeRateJump = 1.0 / (90.0 * LIF_TAU_RATE_SEC);
      let stepSpikes = 0;

      for (let step = 0; step < substeps; step++) {
        const subTime = this.simTime + (step + 1) * dt;
        for (let i = 0; i < this.numNeurons; i++) {
          this.synConductance[i] *= synDecay;
          this.rates[i] *= rateDecay;
          if (this.refractoryRem[i] > 0) {
            this.refractoryRem[i] = Math.max(0, this.refractoryRem[i] - dt);
            this.voltage[i] = LIF_V_RESET_MV;
            continue;
          }
          const driveMv = (this.bias[i] + this.extDrive[i]) * LIF_DRIVE_SCALE_MV;
          const dv = ((LIF_V_REST_MV - this.voltage[i]) + this.synConductance[i] + driveMv) / this.tau[i];
          const nextV = clamp(this.voltage[i] + dt * dv, -75.0, -35.0);
          if (nextV >= LIF_V_THRESH_MV) {
            stepSpikes++;
            this.voltage[i] = LIF_V_RESET_MV;
            this.refractoryRem[i] = LIF_REFRACTORY_SEC;
            this.rates[i] = Math.min(1.0, this.rates[i] + spikeRateJump);
            // Event-driven sparse postsynaptic propagation
            const eStart = this.outStart[i];
            const eEnd = this.outStart[i + 1];
            for (let e = eStart; e < eEnd; e++) {
              const pSlot = this.edgePlasticSlot[e];
              const pScale = pSlot >= 0 ? this.plasticScales[pSlot] : 1.0;
              const post = this.edgePost[e];
              this.synConductance[post] = clamp(
                this.synConductance[post] + this.edgeBaseWeight[e] * pScale * LIF_SYN_PULSE_MV,
                -38.0,
                48.0
              );
            }
            const gIdx = this.neuronGroupIndex[i];
            this.recentSpikes.push({
              t: subTime,
              neuron: i,
              groupIndex: gIdx,
              region: this.groupMeta[gIdx].region,
            });
          } else {
            this.voltage[i] = nextV;
          }
        }
      }
      this.lastStepSpikes = stepSpikes;
      this.totalSpikes += stepSpikes;
      return stepSpikes;
    }

    // Continuous leaky firing-rate dynamics fallback
    const substeps = Math.max(1, Math.ceil(dtClamped / 0.002));
    const dt = dtClamped / substeps;
    const synInput = new Float64Array(this.numNeurons);
    for (let step = 0; step < substeps; step++) {
      synInput.fill(0);
      for (let i = 0; i < this.numNeurons; i++) {
        const r = this.rates[i];
        if (r <= 1e-6) continue;
        const eStart = this.outStart[i];
        const eEnd = this.outStart[i + 1];
        for (let e = eStart; e < eEnd; e++) {
          const pSlot = this.edgePlasticSlot[e];
          const pScale = pSlot >= 0 ? this.plasticScales[pSlot] : 1.0;
          synInput[this.edgePost[e]] += this.edgeBaseWeight[e] * pScale * r;
        }
      }
      for (let i = 0; i < this.numNeurons; i++) {
        const dv = (-this.voltage[i] + synInput[i] + this.bias[i] + this.extDrive[i]) / this.tau[i];
        this.voltage[i] = clamp(this.voltage[i] + dt * dv, -4.0, 6.0);
        this.rates[i] = Math.tanh(Math.max(0, this.voltage[i]));
      }
    }
    this.lastStepSpikes = 0;
    return 0;
  }

  /**
   * Three-factor Drosophila Mushroom Body dopamine learning (KC -> MBON):
   *   - Eligibility trace tracks active Kenyon-cell context.
   *   - PAM reward dopamine strengthens KC -> MBON_Approach & depresses KC -> MBON_Avoid.
   *   - PPL1 aversive dopamine depresses KC -> MBON_Approach & strengthens KC -> MBON_Avoid.
   */
  _updatePlasticity(dt) {
    if (!this.plasticityEnabled || dt <= 0) return;
    const eligDecay = Math.exp(-dt / ELIGIBILITY_TAU_SEC);
    const pam = this.rate('PAM');
    const ppl1 = this.rate('PPL1');
    const daNet = pam - ppl1;

    for (let slot = 0; slot < this.plasticKeys.length; slot++) {
      const key = this.plasticKeys[slot];
      const info = this.plasticPathways.get(key);
      const preRate = this.rate(info.pre);
      // Eligibility integrates presynaptic KC activity; decays over ~2 s.
      let elig = this.plasticEligibility[slot] * eligDecay + preRate * (1 - eligDecay);
      elig = clamp(elig, 0, 1.5);
      this.plasticEligibility[slot] = elig;
      info.eligibility = elig;

      if (Math.abs(daNet) > 0.08 && elig > 0.05) {
        const isApproach = info.post === 'MBON_Approach';
        const direction = isApproach ? 1.0 : -1.0;
        const delta = PLASTICITY_LR * daNet * elig * direction * dt;
        const nextScale = clamp(this.plasticScales[slot] + delta, PLASTIC_MIN_SCALE, PLASTIC_MAX_SCALE);
        this.plasticScales[slot] = nextScale;
        info.scale = nextScale;
      }
    }
  }

  dominantCircuit() {
    const scores = [
      ['escape_giant_fiber', this.rate('DNp01') * 1.45 + this.rate('LC4_Looming') * 0.55],
      ['grooming_aDN1', this.rate('aDN1') * 1.35 + this.rate('SEZ_Groom') * 0.45],
      ['moonwalk_MDN', this.rate('MDN') * 1.35],
      ['rear_DNg02', this.rate('DNg02') * 1.25],
      ['freeze_Pair1', this.rate('Pair1') * 1.10],
      ['steering_DNa02', Math.abs(this.rate('DNa02_L') - this.rate('DNa02_R')) * 1.25],
      ['forward_DNp09', (this.rate('DNp09_L') + this.rate('DNp09_R')) * 0.55],
    ];
    scores.sort((a, b) => b[1] - a[1]);
    return scores[0][1] > 0.14 ? scores[0][0] : 'resting_baseline';
  }

  /**
   * Decode Drosophila descending-neuron firing rates into high-level mouse
   * locomotor commands (consumed by `two_mice_control.js`) and behavioral triggers.
   */
  _decodeMouseMotor({ maxSpeed = 8.6 } = {}) {
    const walkL = this.rate('DNp09_L');
    const walkR = this.rate('DNp09_R');
    const walkMean = 0.5 * (walkL + walkR);
    const turnL = this.rate('DNa02_L');
    const turnR = this.rate('DNa02_R');
    const mdn = this.rate('MDN');
    const pair1 = this.rate('Pair1');
    const adn1 = this.rate('aDN1');
    const dnp01 = this.rate('DNp01');
    const dng02 = this.rate('DNg02');
    const mbonAvoid = this.rate('MBON_Avoid');

    let behaviorTrigger = 'none';
    if (dnp01 > 0.42) behaviorTrigger = 'escape';
    else if (adn1 > 0.45) behaviorTrigger = 'groom';
    else if (dng02 > 0.48 && mdn < 0.35) behaviorTrigger = 'rear';

    const arrest = (pair1 > 0.52 && dnp01 < 0.35 && mdn < 0.40)
      || behaviorTrigger === 'groom'
      || behaviorTrigger === 'rear';

    // Turning rate (rad/s): positive turns left, negative turns right.
    let turnRate = (turnL - turnR) * 2.55;
    if (mbonAvoid > 0.45 && Math.abs(turnRate) < 0.4) {
      // Learned avoidance bias steers away from hazard when centered.
      turnRate += ((this.seed % 2 === 0) ? 1 : -1) * mbonAvoid * 1.1;
    }
    turnRate = clamp(turnRate, -2.8, 2.8);

    // Forward / backward velocity (scene units/s)
    let forwardSpeed = 0;
    let gait = 'auto';
    if (behaviorTrigger === 'escape') {
      forwardSpeed = maxSpeed * clamp(0.88 + dnp01 * 0.30, 0.85, 1.18);
      gait = 'gallop';
    } else if (arrest) {
      forwardSpeed = 0;
      turnRate *= 0.15;
    } else if (mdn > 0.38 && mdn > walkMean * 0.72) {
      // Moonwalker Descending Neuron -> mouse walks backward!
      forwardSpeed = -maxSpeed * clamp(mdn * 0.24, 0.10, 0.24);
      gait = 'walk';
    } else {
      const effectiveWalk = Math.max(0, walkMean - pair1 * 0.55);
      forwardSpeed = maxSpeed * clamp(effectiveWalk * 0.68, 0, 0.92);
    }

    return {
      forwardSpeed: Number(forwardSpeed.toFixed(4)),
      turnRate: Number(turnRate.toFixed(4)),
      gait,
      behaviorTrigger,
      arrest,
      dominantCircuit: this.dominantCircuit(),
    };
  }

  /**
   * Advance the Drosophila connectome by `dt` seconds and return mouse commands.
   */
  step(dt, sensory = {}, { maxSpeed = 8.6 } = {}) {
    const stepDt = clamp(Number(dt) || 0, 0, 0.1);
    // Keep only the last 1.5 s (and bounded length) of spike events for the live HUD raster.
    const cutoff = this.simTime - 1.5;
    if (this.recentSpikes.length > 0 && this.recentSpikes[0].t < cutoff) {
      this.recentSpikes = this.recentSpikes.filter(sp => sp.t >= cutoff);
    }
    if (this.recentSpikes.length > 1400) {
      this.recentSpikes = this.recentSpikes.slice(-1100);
    }

    this._encodeSensory(sensory);
    this._integrate(stepDt);
    this._updateGroupRates();
    this._updatePlasticity(stepDt);

    // Advance optogenetic timers
    if (stepDt > 0 && this.stimulations.length > 0) {
      for (const stim of this.stimulations) stim.remainingSec -= stepDt;
      this.stimulations = this.stimulations.filter(stim => stim.remainingSec > 0);
    }
    this.simTime += stepDt;

    this.lastMotorOutput = this._decodeMouseMotor({ maxSpeed });
    return this.lastMotorOutput;
  }

  snapshot() {
    const rates = {};
    for (let g = 0; g < this.groupMeta.length; g++) {
      rates[this.groupMeta[g].id] = Number(this.groupRates[g].toFixed(4));
    }
    const plasticity = {};
    for (const [key, info] of this.plasticPathways.entries()) {
      plasticity[key] = {
        scale: Number(info.scale.toFixed(4)),
        eligibility: Number(info.eligibility.toFixed(4)),
      };
    }
    return {
      dynamics: this.dynamics,
      plasticityEnabled: this.plasticityEnabled,
      numNeurons: this.numNeurons,
      numSynapses: this.numSynapses,
      simTime: Number(this.simTime.toFixed(4)),
      lastStepSpikes: this.lastStepSpikes,
      totalSpikes: this.totalSpikes,
      dominantCircuit: this.lastMotorOutput.dominantCircuit,
      rates,
      plasticity,
      motor: { ...this.lastMotorOutput },
      activeStimulations: this.stimulations.map(s => ({
        cellType: s.cellType,
        current: s.current,
        remainingSec: Number(s.remainingSec.toFixed(3)),
      })),
    };
  }
}
