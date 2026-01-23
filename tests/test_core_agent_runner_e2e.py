from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from annolid.core.agent.runner import AgentRunner
from annolid.core.behavior.spec import default_behavior_spec, save_behavior_spec
from annolid.core.models.adapters.maskrcnn_torchvision import TorchvisionMaskRCNNAdapter
from annolid.core.output.validate import validate_agent_record


def _write_tiny_video(path: Path, *, frames: int = 3, fps: int = 5) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:
        pytest.skip("cv2 is required for this test.")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (16, 16))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[:, :, 0] = idx * 50
            writer.write(img)
    finally:
        writer.release()


def test_agent_runner_emits_validated_ndjson(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path)

    schema = default_behavior_spec()
    schema.behaviors[0].code = "digging"
    save_behavior_spec(schema, tmp_path / "project.annolid.json")

    out_path = tmp_path / "out.ndjson"
    runner = AgentRunner()
    runner.run(video_path=video_path, out_ndjson=out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        validate_agent_record(record)
        assert record["otherData"]["agent"]["behavior_codes"] == ["digging"]


def test_agent_runner_can_call_vision_adapter(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path)

    class FakeModel:
        def to(self, device):  # noqa: ANN001
            return self

        def eval(self):
            return self

        def __call__(self, batch):  # noqa: ANN001
            return [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
            ]

    adapter = TorchvisionMaskRCNNAdapter(
        pretrained=False,
        score_threshold=0.1,
        model_factory=FakeModel,
    )

    out_path = tmp_path / "out.ndjson"
    runner = AgentRunner(vision_model=adapter)
    runner.run(video_path=video_path, out_ndjson=out_path)

    record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    validate_agent_record(record)
    assert record["shapes"]
    assert record["shapes"][0]["shape_type"] == "rectangle"
