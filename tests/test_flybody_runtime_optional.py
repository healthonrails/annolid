from __future__ import annotations

import os

import pytest

from annolid.simulation.adapters.flybody import FlyBodyAdapter
from annolid.simulation.types import Pose3DFrame


@pytest.mark.simulation
def test_flybody_runtime_optional_ik_fit_smoke() -> None:
    """
    Optional integration smoke test for real FlyBody runtime.

    This test is intentionally gated to avoid adding heavy simulator deps to
    default CI. Enable with:
      ANNOLID_RUN_FLYBODY_RUNTIME=1 pytest -m simulation
    """
    if os.environ.get("ANNOLID_RUN_FLYBODY_RUNTIME") != "1":
        pytest.skip(
            "set ANNOLID_RUN_FLYBODY_RUNTIME=1 to run FlyBody runtime smoke tests"
        )

    pytest.importorskip("flybody")
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")

    adapter = FlyBodyAdapter()
    adapter.configure(
        {
            "keypoint_to_site": {"nose": "head"},
            "ik_kwargs": {"max_steps": 50},
            "dry_run": False,
        }
    )

    result = adapter.fit_3d(
        [
            Pose3DFrame(
                frame_index=0,
                video_name="smoke.mp4",
                points={"nose": (1.0, 2.0, 0.0)},
            )
        ]
    )

    assert len(result.frames) == 1
    frame = result.frames[0]
    assert frame.diagnostics.get("mode") == "ik_fit"
    assert frame.state.get("backend") == "flybody"
    assert "qpos" in frame.state
