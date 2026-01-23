from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from annolid.core.agent.runner import AgentRunner
from annolid.core.behavior.spec import default_behavior_spec, save_behavior_spec
from annolid.engine.cli import main as annolid_run


def _write_tiny_video(path: Path, *, frames: int = 3, fps: int = 5) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:
        pytest.skip("cv2 is required for this test.")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (16, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(frames):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :, 1] = idx * 50
            writer.write(img)
    finally:
        writer.release()


def test_agent_runner_output_validates_via_cli(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path)

    schema = default_behavior_spec()
    schema.behaviors[0].code = "digging"
    save_behavior_spec(schema, tmp_path / "project.annolid.json")

    out_path = tmp_path / "out.ndjson"
    AgentRunner().run(video_path=video_path, out_ndjson=out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines
    first = json.loads(lines[0])
    assert first["video_name"] == "tiny.avi"

    rc = annolid_run(["validate-agent-output", "--ndjson", str(out_path)])
    assert rc == 0
