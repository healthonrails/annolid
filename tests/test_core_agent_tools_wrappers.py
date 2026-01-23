from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from annolid.core.agent.tools.base import FrameBatch, FrameData, ToolContext
from annolid.core.agent.tools.detection import DetectionTool
from annolid.core.agent.tools.llm import CaptionTool
from annolid.core.agent.tools.tracking import SimpleTrackTool
from annolid.core.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RuntimeModel,
)
from annolid.core.types import BBoxGeometry, FrameRef


@dataclass
class _FakeModel(RuntimeModel):
    detections: list[dict]
    text: Optional[str] = None

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tasks=("detect", "caption"))

    def load(self) -> None:
        return None

    def predict(self, request: ModelRequest) -> ModelResponse:
        if request.task == "detect":
            return ModelResponse(task="detect", output={"detections": self.detections})
        return ModelResponse(task="caption", output={"text": self.text}, text=self.text)

    def close(self) -> None:
        return None


def test_detection_tool_emits_instances(tmp_path) -> None:
    model = _FakeModel(
        detections=[{"bbox_xyxy": [0, 0, 10, 10], "label_id": 1, "score": 0.9}]
    )
    tool = DetectionTool(model=model)
    frame = FrameData(ref=FrameRef(frame_index=0))
    ctx = ToolContext(video_path=tmp_path / "v.mp4", results_dir=tmp_path, run_id="r1")
    result = tool.run(ctx, FrameBatch(frames=[frame]))
    assert result.frames
    inst = result.frames[0].instances[0]
    assert isinstance(inst.geometry, BBoxGeometry)
    assert inst.label == "label_1"


def test_caption_tool_emits_text(tmp_path) -> None:
    model = _FakeModel(detections=[], text="hello")
    tool = CaptionTool(model=model)
    frame = FrameData(ref=FrameRef(frame_index=1))
    ctx = ToolContext(video_path=tmp_path / "v.mp4", results_dir=tmp_path, run_id="r1")
    result = tool.run(ctx, FrameBatch(frames=[frame]))
    assert result.frames[1] == "hello"


def test_simple_track_tool_groups_instances(tmp_path) -> None:
    model = _FakeModel(
        detections=[{"bbox_xyxy": [1, 1, 2, 2], "label_id": 0, "score": 0.5}]
    )
    det_tool = DetectionTool(model=model)
    frame = FrameData(ref=FrameRef(frame_index=0))
    ctx = ToolContext(video_path=tmp_path / "v.mp4", results_dir=tmp_path, run_id="r1")
    instances = det_tool.run(ctx, FrameBatch(frames=[frame])).frames
    track_tool = SimpleTrackTool()
    tracks = track_tool.run(ctx, instances).tracks
    assert tracks
