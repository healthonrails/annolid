"""Deterministic Node checks for the FlyBody-inspired mouse motor controller."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_two_mice_simulation import _find_node_binary

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_IMPORT = (
    "import * as C from './annolid/gui/assets/threejs/two_mice_control.js';\n"
)


def run_control(body: str):
    node = _find_node_binary()
    if node is None:
        pytest.skip("Node.js is required to validate the JavaScript controller")
    result = subprocess.run(
        [node, "--input-type=module", "-e", CONTROL_IMPORT + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


RELATIVE_PHASE_HELPER = """
const relativeErrors = (controller) => {
  const offsets = C.MOUSE_GAITS[controller.cpg.gait].offsets;
  return C.MOUSE_LIMBS.map((limb, index) => Math.abs(C.wrapPhaseSigned(
    controller.cpg.relativePhase(index) - (offsets[limb] - offsets.LH)
  )));
};
"""


@pytest.mark.parametrize(
    ("speed", "dart", "expected_gait"),
    [(2.0, False, "walk"), (5.5, False, "trot"), (8.8, True, "gallop")],
)
def test_cpg_selects_gait_by_speed_and_locks_its_interlimb_phases(
    speed, dart, expected_gait
):
    result = run_control(
        RELATIVE_PHASE_HELPER
        + f"""
        const controller = new C.MouseMotorController({{ gait: 'trot', jitter: 0.07 }});
        for (let i = 0; i < 300; i++) {{
          controller.step(1 / 30, {{ speed: {speed}, maxSpeed: 9.2, dart: {str(dart).lower()} }});
        }}
        const stanceFraction = (() => {{
          let stance = 0, total = 0;
          for (let i = 0; i < 300; i++) {{
            controller.step(1 / 300, {{ speed: {speed}, maxSpeed: 9.2, dart: {str(dart).lower()} }});
            stance += controller.isLimbInStance('LH') ? 1 : 0;
            total += 1;
          }}
          return stance / total;
        }})();
        console.log(JSON.stringify({{
          gait: controller.cpg.gait,
          frequencyHz: controller.cpg.frequencyHz,
          duty: controller.cpg.duty,
          maxPhaseError: Math.max(...relativeErrors(controller)),
          stanceFraction,
        }}));
        """
    )
    assert result["gait"] == expected_gait
    assert result["maxPhaseError"] < 0.03
    assert 0.5 < result["frequencyHz"] < 11.0
    # Duty factor falls as the gait speeds up, and the stance fraction measured
    # from the oscillator matches the gait's target duty factor.
    expected_duty = {"walk": 0.70, "trot": 0.56, "gallop": 0.42}[expected_gait]
    assert result["duty"] == pytest.approx(expected_duty, abs=0.01)
    assert result["stanceFraction"] == pytest.approx(expected_duty, abs=0.05)


def test_trot_pairs_diagonals_and_alternates_left_right():
    result = run_control(
        """
        const controller = new C.MouseMotorController({ gait: 'walk', jitter: 0.1 });
        for (let i = 0; i < 360; i++) controller.step(1 / 30, { speed: 5.0, maxSpeed: 9.2 });
        const rel = Object.fromEntries(C.MOUSE_LIMBS.map((limb, i) => [
          limb, controller.cpg.relativePhase(i),
        ]));
        console.log(JSON.stringify({ gait: controller.cpg.gait, rel }));
        """
    )
    rel = result["rel"]
    assert result["gait"] == "trot"
    # Diagonal LH+RF in phase; ipsilateral LF and contralateral RH in anti-phase.
    assert min(rel["RF"], 1 - rel["RF"]) < 0.03
    assert abs(rel["RH"] - 0.5) < 0.03
    assert abs(rel["LF"] - 0.5) < 0.03


def test_gait_selection_has_hysteresis_and_explicit_override():
    result = run_control(
        """
        const up = [];
        const down = [];
        let gait = 'walk';
        for (let s = 0; s <= 1.2; s += 0.02) { gait = C.selectGait(gait, s); up.push([+s.toFixed(2), gait]); }
        for (let s = 1.2; s >= 0; s -= 0.02) { gait = C.selectGait(gait, s); down.push([+s.toFixed(2), gait]); }
        const firstAt = (rows, name) => rows.find(([, g]) => g === name)?.[0] ?? null;
        const forced = C.computeDescendingDrive({ speed: 1.0, maxSpeed: 9.2, gaitOverride: 'gallop' });
        console.log(JSON.stringify({
          walkToTrot: firstAt(up, 'trot'),
          trotToGallop: firstAt(up, 'gallop'),
          gallopToTrot: firstAt(down, 'trot'),
          trotToWalk: firstAt(down, 'walk'),
          forcedGait: forced.gait,
          dartEntry: C.selectGait('trot', 0.86, { dart: true }),
          normalEntry: C.selectGait('trot', 0.86),
        }));
        """
    )
    assert result["walkToTrot"] > result["trotToWalk"]
    assert result["trotToGallop"] > result["gallopToTrot"]
    assert result["forcedGait"] == "gallop"
    assert result["dartEntry"] == "gallop"
    assert result["normalEntry"] == "trot"


def test_stopping_finishes_swings_and_plants_all_four_paws():
    result = run_control(
        """
        const controller = new C.MouseMotorController({ gait: 'trot' });
        for (let i = 0; i < 90; i++) controller.step(1 / 30, { speed: 6.0, maxSpeed: 9.2 });
        const swingingBefore = controller.cpg.hasSwingingLimb();
        for (let i = 0; i < 30; i++) controller.step(1 / 30, { speed: 0, maxSpeed: 9.2 });
        const settled = controller.cpg.snapshot();
        const phasesAfterSettle = controller.cpg.phases.slice();
        for (let i = 0; i < 60; i++) controller.step(1 / 30, { speed: 0, maxSpeed: 9.2 });
        const posture = C.computeDescendingDrive({ speed: 6.0, maxSpeed: 9.2, posture: 'rearing' });
        console.log(JSON.stringify({
          swingingBefore,
          stance: settled.stance,
          stepping: controller.isStepping(),
          drift: Math.max(...controller.cpg.phases.map((p, i) => Math.abs(p - phasesAfterSettle[i]))),
          amplitude: controller.amplitude,
          postureFrequency: posture.frequencyHz,
          postureArrest: posture.arrest,
        }));
        """
    )
    assert result["swingingBefore"] is True
    assert all(result["stance"].values())
    assert result["stepping"] is False
    assert result["drift"] == 0
    assert result["amplitude"] < 0.01
    assert result["postureFrequency"] == 0
    assert result["postureArrest"] == 1


def test_turning_in_place_steps_and_shortens_inner_strides():
    result = run_control(
        """
        const controller = new C.MouseMotorController({ gait: 'trot' });
        for (let i = 0; i < 60; i++) controller.step(1 / 30, { speed: 0, yawRate: 2.4, maxSpeed: 9.2 });
        const turning = controller.cpg.frequencyHz;
        let leftRange = [Infinity, -Infinity], rightRange = [Infinity, -Infinity];
        const run = new C.MouseMotorController({ gait: 'trot' });
        for (let i = 0; i < 600; i++) {
          const t = run.step(1 / 120, { speed: 5.0, yawRate: 2.4, maxSpeed: 9.2 });
          if (i < 240) continue;
          leftRange = [Math.min(leftRange[0], t.hip_flex_LH), Math.max(leftRange[1], t.hip_flex_LH)];
          rightRange = [Math.min(rightRange[0], t.hip_flex_RH), Math.max(rightRange[1], t.hip_flex_RH)];
        }
        console.log(JSON.stringify({
          turning,
          leftSweep: leftRange[1] - leftRange[0],
          rightSweep: rightRange[1] - rightRange[0],
        }));
        """
    )
    assert result["turning"] > 0.5
    # Positive yaw is a left turn: the left (inner) hindlimb sweeps less.
    assert result["leftSweep"] < result["rightSweep"] * 0.8


def test_stance_retracts_monotonically_so_planted_paws_move_backward():
    result = run_control(
        """
        const cpg = new C.MouseLimbCPG({ gait: 'trot' });
        const samples = [];
        for (let i = 0; i <= 400; i++) {
          cpg.phases[0] = i / 400;
          samples.push(cpg.limbState(0));
        }
        let stanceMonotonic = true, swingMonotonic = true;
        for (let i = 1; i < samples.length; i++) {
          const [a, b] = [samples[i - 1], samples[i]];
          if (a.stance && b.stance && b.stroke > a.stroke + 1e-9) stanceMonotonic = false;
          if (!a.stance && !b.stance && b.stroke < a.stroke - 1e-9) swingMonotonic = false;
          if (b.stance && b.lift !== 0) stanceMonotonic = false;
        }
        console.log(JSON.stringify({
          stanceMonotonic,
          swingMonotonic,
          maxLift: Math.max(...samples.map(s => s.lift)),
        }));
        """
    )
    assert result["stanceMonotonic"] is True
    assert result["swingMonotonic"] is True
    assert result["maxLift"] == pytest.approx(1.0, abs=0.02)


def test_lift_off_reflex_only_triggers_in_late_stance():
    result = run_control(
        """
        const cpg = new C.MouseLimbCPG({ gait: 'trot' });
        cpg.phases = [0.30, 0.30, 0.90, 0.90];
        cpg.step(1 / 120, { frequencyHz: 4, liftOff: [true, false, true, false] });
        console.log(JSON.stringify({ phases: cpg.phases, reflexCount: cpg.reflexCount }));
        """
    )
    phases = result["phases"]
    assert result["reflexCount"] == 1
    assert phases[2] == 0  # LH in late stance: forced swing onset
    assert phases[0] > 0.3  # LF in swing: unaffected
    assert phases[3] > 0.9  # RH: no reflex requested


def test_action_spec_decoder_and_commands_are_bounded_and_deterministic():
    result = run_control(
        """
        const spec = C.getMouseActionSpec();
        const a = new C.MouseMotorController({ gait: 'trot', jitter: 0.05 });
        const b = new C.MouseMotorController({ gait: 'trot', jitter: 0.05 });
        let inRange = true, identical = true;
        for (let i = 0; i < 900; i++) {
          const input = {
            speed: 9.2 * Math.abs(Math.sin(i * 0.013)) * 1.3,
            yawRate: 3 * Math.sin(i * 0.021),
            maxSpeed: 9.2,
            dart: i % 200 > 120,
            liftOff: [i % 7 === 0, false, i % 11 === 0, false],
          };
          const ta = a.step(1 / 30, input);
          const tb = b.step(1 / 30, input);
          spec.names.forEach((name, k) => {
            if (!(ta[name] >= spec.minimum[k] && ta[name] <= spec.maximum[k])) inRange = false;
            if (ta[name] !== tb[name]) identical = false;
          });
        }
        const errors = [];
        for (const bad of [null, { forwardSpeed: NaN }, { durationSec: -1 }, { gait: 'hop' }]) {
          try { C.sanitizeMotorCommand(bad, { maxSpeed: 9.2 }); errors.push(false); }
          catch (error) { errors.push(true); }
        }
        const clipped = C.sanitizeMotorCommand({ forwardSpeed: 99, turnRate: -9 }, { maxSpeed: 9.2 });
        console.log(JSON.stringify({
          joints: spec.names.length,
          uniqueNames: new Set(spec.names).size,
          inRange,
          identical,
          errors,
          clipped,
          neutralInRange: spec.neutral.every((v, k) => v >= spec.minimum[k] && v <= spec.maximum[k]),
          mirroredAbduction: [
            spec.neutral[spec.names.indexOf('shoulder_abduct_LF')],
            spec.neutral[spec.names.indexOf('shoulder_abduct_RF')],
          ],
          clipUnknownThrows: (() => { try { C.clipJointTarget('tail', 0); return false; } catch { return true; } })(),
        }));
        """
    )
    assert result["joints"] == 22
    assert result["uniqueNames"] == 22
    assert result["inRange"] is True
    assert result["identical"] is True
    assert result["neutralInRange"] is True
    assert result["errors"] == [True, True, True, True]
    assert result["clipped"]["forwardSpeed"] == pytest.approx(9.2 * 1.24)
    assert result["clipped"]["turnRate"] == -3.0
    assert result["mirroredAbduction"][0] == pytest.approx(
        -result["mirroredAbduction"][1]
    )
    assert result["clipUnknownThrows"] is True
