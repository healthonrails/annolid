from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from annolid.core.agent.runner import AgentRunner
from annolid.core.behavior.spec import default_behavior_spec, save_behavior_spec
from annolid.core.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RuntimeModel,
)
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


def test_agent_runner_merges_seed_keypoints(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_tiny_video(video_path, frames=1)

    schema = default_behavior_spec()
    agent_meta = AgentRunner()._build_agent_meta(schema, None, video_path)  # type: ignore[attr-defined]

    class FakeKeypointModel(RuntimeModel):
        @property
        def model_id(self) -> str:
            return "fake:keypoints"

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(
                tasks=("detect",),
                input_modalities=("image",),
                output_modalities=("detections", "keypoints"),
            )

        def load(self) -> None:
            return

        def predict(self, request: ModelRequest) -> ModelResponse:
            _ = request
            return ModelResponse(
                task="detect",
                output={
                    "detections": [
                        {
                            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                            "label_id": 0,
                            "score": 0.9,
                            "keypoints_xy": [[2.0, 2.0], [4.0, 4.0]],
                            "keypoint_scores": [0.9, 0.8],
                            "keypoint_visible": [True, True],
                            "keypoint_names": ["nose", "tail"],
                        }
                    ]
                },
            )

        def close(self) -> None:
            return

    def seed_provider(_: int):
        return {
            "shapes": [
                {
                    "label": "manual_poly",
                    "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                },
                {
                    "label": "nose",
                    "points": [[10.0, 11.0]],
                    "shape_type": "point",
                    "flags": {},
                    "visible": True,
                },
            ]
        }

    runner = AgentRunner(vision_model=FakeKeypointModel())
    record = next(
        runner.iter_records(
            video_path=video_path,
            schema=schema,
            agent_meta=agent_meta,
            seed_record_provider=seed_provider,
        )
    )

    validate_agent_record(record)
    points = {
        shape["label"]: shape["points"][0]
        for shape in record["shapes"]
        if shape.get("shape_type") == "point"
    }
    assert points["nose"] == [10.0, 11.0]
    assert points["tail"] == [4.0, 4.0]
