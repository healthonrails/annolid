from __future__ import annotations

from pathlib import Path

import pytest

from annolid.core.agent.orchestrator import AnnolidAgent
from annolid.core.agent.tools.base import (
    FrameBatch,
    Instance,
    Instances,
    Tool,
    ToolContext,
)
from annolid.core.agent.tools.detection import DetectionResult
from annolid.core.behavior.spec import default_behavior_spec
from annolid.core.output.validate import validate_agent_record
from annolid.core.types import BBoxGeometry

pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _write_tiny_video(path: Path, *, frames: int = 3, fps: int = 5) -> None:
    import cv2  # type: ignore
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (16, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(frames):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :, 0] = idx * 40
            writer.write(img)
    finally:
        writer.release()


class DummyDetectionTool(Tool[FrameBatch, DetectionResult]):
    name = "dummy_detection"

    def run(self, ctx: ToolContext, payload: FrameBatch) -> DetectionResult:
        outputs = []
        for frame in payload:
            inst = Instance(
                frame=frame.ref,
                geometry=BBoxGeometry("bbox", (1.0, 1.0, 10.0, 10.0)),
                label="mouse",
                score=0.9,
            )
            outputs.append(Instances(frame=frame.ref, instances=[inst]))
        return DetectionResult(frames=outputs)


def test_orchestrator_streams_valid_records(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path)

    agent = AnnolidAgent(tools=[DummyDetectionTool()])
    schema = default_behavior_spec()
    records = agent.iter_records(
        video_path=video_path,
        schema=schema,
        agent_meta={"test": True},
    )
    record = next(records)
    assert record["video_name"] == "tiny.avi"
    assert "shapes" in record
    assert "otherData" in record
    validate_agent_record(record)
