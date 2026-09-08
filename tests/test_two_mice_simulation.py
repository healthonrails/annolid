"""Run the dependency-free simulation core in Node; no browser or CDN needed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


SIMULATION = (
    Path(__file__).resolve().parents[1]
    / "annolid/gui/assets/threejs/two_mice_simulation.js"
)


def run_javascript(body):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to validate the JavaScript simulation")
    source = SIMULATION.read_text(encoding="utf-8")
    result = subprocess.run(
        [node, "--input-type=module", "-e", source + "\n" + body],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("fps", [30, 60, 144])
def test_fixed_physics_matches_across_display_rates(fps):
    result = run_javascript(
        f"""
        const clock = new FixedStepClock();
        const random = createSeededRandom(123);
        let position = 0, velocity = 1;
        for (let i = 0; i < {fps} * 10; i++) {{
          clock.advance(1 / {fps}, (time, dt) => {{
            velocity += (random() - 0.5 - velocity * 0.3) * dt;
            position += velocity * dt;
          }});
        }}
        console.log(JSON.stringify({{steps: clock.steps, position, velocity}}));
        """
    )
    assert result["steps"] == 300
    reference = run_javascript(
        """
        const random = createSeededRandom(123);
        let position = 0, velocity = 1;
        for (let i = 0; i < 300; i++) {
          velocity += (random() - 0.5 - velocity * 0.3) / 30;
          position += velocity / 30;
        }
        console.log(JSON.stringify({position, velocity}));
        """
    )
    assert result["position"] == pytest.approx(reference["position"], abs=1e-12)
    assert result["velocity"] == pytest.approx(reference["velocity"], abs=1e-12)


def test_pause_invalid_deltas_and_long_resume_are_bounded():
    result = run_javascript(
        """
        const clock = new FixedStepClock();
        let calls = 0;
        const step = () => calls++;
        clock.advance(1 / 60, step);
        clock.advance(90, step, true);
        const paused = [clock.steps, calls, clock.remainder];
        for (const dt of [NaN, Infinity, -1, 0]) clock.advance(dt, step);
        clock.advance(1 / 60, step);
        const beforeResume = calls;
        clock.advance(90, step);
        console.log(JSON.stringify({paused, beforeResume, calls, remainder: clock.remainder}));
        """
    )
    assert result["paused"] == [0, 0, 0]
    assert result["beforeResume"] == 0
    assert result["calls"] == 6
    assert 0 <= result["remainder"] < 1 / 30


def test_wall_normals_repel_and_attract_on_all_four_sides():
    result = run_javascript(
        """
        const normals = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        const forces = normals.map(([x, z]) => {
          const near = wallNormalForce(0.4, 5.8, 1.8);
          const far = wallNormalForce(5, 5.8, 1.8);
          return {nearDot: x*x*near + z*z*near, farDot: x*x*far + z*z*far};
        });
        console.log(JSON.stringify({forces,
          boundary: wallNormalForce(2, 5.8, 1.8),
          otherWall: wallNormalForce(5, 5.8, 1.8, false),
          disabled: wallNormalForce(5, 5.8, 0)}));
        """
    )
    assert all(force["nearDot"] > 0 for force in result["forces"])
    assert all(force["farDot"] < 0 for force in result["forces"])
    assert result["boundary"] == result["otherWall"] == result["disabled"] == 0


def test_seeded_streams_repeat_without_modifying_global_random():
    result = run_javascript(
        """
        const original = Math.random;
        const a = createSeededRandom(123), b = createSeededRandom(123);
        const appearance = createSeededRandom(999);
        let equal = true, bounded = true;
        for (let i = 0; i < 10000; i++) {
          for (let j = 0; j < i % 11; j++) appearance();
          const value = a();
          equal &&= value === b();
          bounded &&= value >= 0 && value < 1;
        }
        console.log(JSON.stringify({equal, bounded, unchanged: original === Math.random}));
        """
    )
    assert result == {"equal": True, "bounded": True, "unchanged": True}
