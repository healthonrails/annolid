from __future__ import annotations

import os
from pathlib import Path

import pytest

from annolid.simulation import build_simulation_view_payload, write_simulation_ndjson
from annolid.simulation.adapters.flybody import FlyBodyAdapter
from annolid.simulation.flybody_live import generate_live_rollout_payload
from annolid.simulation.types import Pose3DFrame


def _require_flybody_runtime() -> None:
    if os.environ.get("ANNOLID_RUN_FLYBODY_RUNTIME") != "1":
        pytest.skip(
            "set ANNOLID_RUN_FLYBODY_RUNTIME=1 to run FlyBody runtime smoke tests"
        )
    pytest.importorskip("flybody")
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")


@pytest.mark.simulation
def test_flybody_runtime_optional_ik_fit_smoke() -> None:
    """Optional integration smoke test for real FlyBody IK fitting and pose extraction."""
    _require_flybody_runtime()

    with FlyBodyAdapter() as adapter:
        adapter.configure(
            {
                "keypoint_to_site": {"nose": "head_site"},
                "ik_kwargs": {"max_steps": 50},
                "dry_run": False,
            }
        )
        result = adapter.fit_3d(
            [
                Pose3DFrame(
                    frame_index=0,
                    video_name="smoke.mp4",
                    points={"nose": (0.09, 0.0, 0.14)},
                )
            ]
        )

    assert len(result.frames) == 1
    frame = result.frames[0]
    assert frame.diagnostics.get("mode") == "ik_fit"
    assert "ik_steps" in frame.diagnostics
    assert frame.state.get("backend") == "flybody"
    assert isinstance(frame.state.get("qpos"), list)
    assert "thorax" in frame.state.get("body_poses", {})
    assert frame.state.get("model_pose", {}).get("scale") == 7.5


@pytest.mark.simulation
def test_flybody_runtime_optional_connectome_spiking_rollout(tmp_path: Path) -> None:
    """Verify real MuJoCo closed-loop spiking Brain+VNC rollout and NDJSON round-trip."""
    _require_flybody_runtime()

    with FlyBodyAdapter() as adapter:
        adapter.configure(
            {
                "behavior": "walk_imitation",
                "dynamics": "spiking",
                "dry_run": False,
            }
        )
        result = adapter.rollout({"seed": 7}, steps=15)

    assert len(result.frames) == 15
    last = result.frames[-1]
    assert last.diagnostics["mode"] == "connectome_rollout"
    assert last.diagnostics["brain"]["dynamics"] == "spiking"
    assert last.diagnostics["brain"]["spiking"]["total_spikes"] > 0
    assert "thorax" in last.state["body_poses"]
    assert "head" in last.state["site_targets"]

    out_ndjson = tmp_path / "walk_spiking.ndjson"
    write_simulation_ndjson(
        out_ndjson,
        pose_frames=(),
        result=result,
        adapter_name="flybody",
    )
    view_payload = build_simulation_view_payload(out_ndjson)
    assert len(view_payload["frames"]) == 15
    assert "thorax" in view_payload["frames"][-1]["body_poses"]


@pytest.mark.simulation
def test_flybody_runtime_optional_calibrated_flight_stability() -> None:
    """Verify calibrated wingbeat + Haltere/Ocellar connectome sustains flight altitude."""
    _require_flybody_runtime()

    payload = generate_live_rollout_payload(
        steps=20,
        seed=7,
        behavior="flight_imitation",
        controller="connectome",
        dynamics="spiking",
    )
    frames = payload["frames"]
    assert len(frames) == 20
    z_start = frames[0]["model_pose"]["position"][2]
    z_end = frames[-1]["model_pose"]["position"][2]
    assert z_end > z_start - 0.05
    assert frames[-1]["diagnostics"]["brain"]["spiking"]["total_spikes"] > 0
