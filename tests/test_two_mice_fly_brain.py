"""Deterministic Node tests for the Drosophila connectome mouse controller."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_two_mice_simulation import _find_node_binary

REPO_ROOT = Path(__file__).resolve().parents[1]
FLY_BRAIN_IMPORT = (
    "import * as FB from './annolid/gui/assets/threejs/two_mice_fly_brain.js';\n"
)


def run_fly_brain(body: str):
    node = _find_node_binary()
    if node is None:
        pytest.skip(
            "Node.js is required to validate the JavaScript fly-brain controller"
        )
    result = subprocess.run(
        [node, "--input-type=module", "-e", FLY_BRAIN_IMPORT + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def test_fly_brain_is_deterministic_and_fires_bounded_lif_spikes():
    result = run_fly_brain(
        """
        const a = new FB.FlyBrainMouseController({ seed: 42, dynamics: 'spiking' });
        const b = new FB.FlyBrainMouseController({ seed: 42, dynamics: 'spiking' });
        for (let i = 0; i < 40; i++) {
          const sensory = { odorLeft: 0.7, odorRight: 0.1, exploreDrive: 0.6 };
          a.step(1 / 30, sensory);
          b.step(1 / 30, sensory);
        }
        const sa = a.snapshot();
        const sb = b.snapshot();
        const ratesInBounds = Object.values(sa.rates).every(r => Number.isFinite(r) && r >= 0 && r <= 1);
        console.log(JSON.stringify({
          numNeurons: sa.numNeurons,
          numSynapses: sa.numSynapses,
          totalSpikes: sa.totalSpikes,
          identical: JSON.stringify(sa) === JSON.stringify(sb),
          ratesInBounds,
          recentSpikesCount: a.recentSpikes.length,
        }));
        """
    )
    assert result["numNeurons"] >= 200
    assert result["numSynapses"] > 500
    assert result["totalSpikes"] > 200
    assert result["identical"] is True
    assert result["ratesInBounds"] is True
    assert result["recentSpikesCount"] > 0


@pytest.mark.parametrize("dynamics", ["spiking", "rate"])
def test_lateralized_odor_and_vision_steer_via_pfl3_lal_dna02(dynamics):
    result = run_fly_brain(
        f"""
        const leftBrain = new FB.FlyBrainMouseController({{ seed: 7, dynamics: {json.dumps(dynamics)} }});
        const rightBrain = new FB.FlyBrainMouseController({{ seed: 7, dynamics: {json.dumps(dynamics)} }});
        let leftOut = null, rightOut = null;
        for (let i = 0; i < 36; i++) {{
          leftOut = leftBrain.step(1 / 30, {{ odorLeft: 1.2, odorRight: 0.0, visualLeft: 0.9, visualRight: 0.0, exploreDrive: 0.5 }});
          rightOut = rightBrain.step(1 / 30, {{ odorLeft: 0.0, odorRight: 1.2, visualLeft: 0.0, visualRight: 0.9, exploreDrive: 0.5 }});
        }}
        console.log(JSON.stringify({{
          leftTurn: leftOut.turnRate,
          rightTurn: rightOut.turnRate,
          leftDNa02L: leftBrain.rate('DNa02_L'),
          leftDNa02R: leftBrain.rate('DNa02_R'),
          rightDNa02L: rightBrain.rate('DNa02_L'),
          rightDNa02R: rightBrain.rate('DNa02_R'),
        }}));
        """
    )
    assert result["leftDNa02L"] > result["leftDNa02R"] + 0.15
    assert result["rightDNa02R"] > result["rightDNa02L"] + 0.15
    assert result["leftTurn"] > 0.45
    assert result["rightTurn"] < -0.45


@pytest.mark.parametrize("dynamics", ["spiking", "rate"])
def test_optogenetic_circuits_drive_walk_moonwalk_freeze_groom_rear_escape(dynamics):
    result = run_fly_brain(
        f"""
        function runPreset(presetId, sensory = {{ exploreDrive: 0.25 }}) {{
          const brain = new FB.FlyBrainMouseController({{ seed: 11, dynamics: {json.dumps(dynamics)} }});
          brain.applyPreset(presetId);
          let out = null;
          for (let i = 0; i < 30; i++) {{
            out = brain.step(1 / 30, sensory);
          }}
          return {{
            out,
            rates: {{
              dnp09: 0.5 * (brain.rate('DNp09_L') + brain.rate('DNp09_R')),
              mdn: brain.rate('MDN'),
              pair1: brain.rate('Pair1'),
              adn1: brain.rate('aDN1'),
              dng02: brain.rate('DNg02'),
              dnp01: brain.rate('DNp01'),
            }},
          }};
        }}
        const tactileBrain = new FB.FlyBrainMouseController({{ seed: 11, dynamics: {json.dumps(dynamics)} }});
        let tactileOut = null;
        for (let i = 0; i < 36; i++) {{
          tactileOut = tactileBrain.step(1 / 30, {{ wallAhead: 1.5, exploreDrive: 0.45 }});
        }}
        console.log(JSON.stringify({{
          walk: runPreset('walk'),
          moonwalk: runPreset('moonwalk'),
          freeze: runPreset('freeze'),
          groom: runPreset('groom'),
          rear: runPreset('rear'),
          escape: runPreset('escape'),
          tactileRetreat: {{
            out: tactileOut,
            mdn: tactileBrain.rate('MDN'),
            vncInh: tactileBrain.rate('VNC_MDN_Inh'),
          }},
        }}));
        """
    )
    assert result["walk"]["out"]["forwardSpeed"] > 2.5
    assert result["walk"]["out"]["arrest"] is False

    assert result["moonwalk"]["rates"]["mdn"] > 0.5
    assert result["moonwalk"]["out"]["forwardSpeed"] < -0.8
    assert result["moonwalk"]["out"]["dominantCircuit"] == "moonwalk_MDN"

    assert result["freeze"]["rates"]["pair1"] > 0.55
    assert result["freeze"]["out"]["arrest"] is True
    assert result["freeze"]["out"]["forwardSpeed"] == 0

    assert result["groom"]["rates"]["adn1"] > 0.5
    assert result["groom"]["out"]["behaviorTrigger"] == "groom"
    assert result["groom"]["out"]["arrest"] is True

    assert result["rear"]["rates"]["dng02"] > 0.5
    assert result["rear"]["out"]["behaviorTrigger"] == "rear"
    assert result["rear"]["out"]["arrest"] is True

    assert result["escape"]["rates"]["dnp01"] > 0.45
    assert result["escape"]["out"]["behaviorTrigger"] == "escape"
    assert result["escape"]["out"]["gait"] == "gallop"
    assert result["escape"]["out"]["forwardSpeed"] > 7.0

    assert result["tactileRetreat"]["mdn"] > 0.45
    assert result["tactileRetreat"]["out"]["forwardSpeed"] < -0.5


@pytest.mark.parametrize("dynamics", ["spiking", "rate"])
def test_dopamine_plasticity_modulates_kc_to_mbon_weights_and_resets(dynamics):
    result = run_fly_brain(
        f"""
        const brain = new FB.FlyBrainMouseController({{ seed: 19, dynamics: {json.dumps(dynamics)}, plasticityEnabled: true }});
        // Pair Reward KC with PAM reward dopamine (stonkfly-style positive reinforcement)
        for (let i = 0; i < 60; i++) {{
          brain.step(1 / 30, {{ rewardCue: 1.3, rewardDopamine: 1.5, exploreDrive: 0.3 }});
        }}
        const afterReward = brain.snapshot().plasticity;

        // Pair Hazard KC with PPL1 aversive dopamine (negative reinforcement)
        for (let i = 0; i < 60; i++) {{
          brain.step(1 / 30, {{ hazardCue: 1.3, punishDopamine: 1.5, exploreDrive: 0.3 }});
        }}
        const afterPunish = brain.snapshot().plasticity;

        // Reset restores neutral 1.0 scales
        brain.resetPlasticity();
        const afterReset = brain.snapshot().plasticity;

        // Disabled plasticity keeps scales frozen
        const frozen = new FB.FlyBrainMouseController({{ seed: 19, dynamics: {json.dumps(dynamics)}, plasticityEnabled: false }});
        for (let i = 0; i < 40; i++) {{
          frozen.step(1 / 30, {{ rewardCue: 1.3, rewardDopamine: 1.6 }});
        }}
        console.log(JSON.stringify({{
          afterReward,
          afterPunish,
          afterReset,
          frozen: frozen.snapshot().plasticity,
        }}));
        """
    )
    assert result["afterReward"]["reward_approach"]["scale"] > 1.25
    assert result["afterReward"]["reward_avoid"]["scale"] < 0.75
    assert result["afterPunish"]["hazard_approach"]["scale"] < 0.75
    assert result["afterPunish"]["hazard_avoid"]["scale"] > 1.25
    for entry in result["afterReset"].values():
        assert entry["scale"] == pytest.approx(1.0)
        assert entry["eligibility"] == pytest.approx(0.0)
    for entry in result["frozen"].values():
        assert entry["scale"] == pytest.approx(1.0)


def test_invalid_neuron_or_preset_raises_range_error():
    result = run_fly_brain(
        """
        const brain = new FB.FlyBrainMouseController({ seed: 3 });
        let badNeuron = false, badPreset = false;
        try { brain.stimulate('NonExistentNeuron', 1.0, 1.0); } catch (e) { badNeuron = e instanceof RangeError; }
        try { brain.applyPreset('teleport'); } catch (e) { badPreset = e instanceof RangeError; }
        console.log(JSON.stringify({ badNeuron, badPreset }));
        """
    )
    assert result["badNeuron"] is True
    assert result["badPreset"] is True
