from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from annolid.core.models.base import ModelRequest, RuntimeModel
from annolid.core.types import BBoxGeometry, FrameRef

from .base import FrameBatch, Instance, Instances, Tool, ToolContext


@dataclass(frozen=True)
class DetectionResult:
    frames: Sequence[Instances]


class DetectionTool(Tool[FrameBatch, DetectionResult]):
    """Run a RuntimeModel detector over a batch of frames."""

    name = "detection"

    def __init__(
        self,
        *,
        model: RuntimeModel,
        label_names: Optional[Sequence[str]] = None,
        config: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__(config=config)
        self._model = model
        self._label_names = list(label_names) if label_names else None

    def run(self, ctx: ToolContext, payload: FrameBatch) -> DetectionResult:
        outputs: List[Instances] = []
        with self._model:
            for frame in payload:
                response = self._model.predict(
                    ModelRequest(
                        task="detect",
                        image=frame.image_rgb,
                        image_path=str(frame.image_path) if frame.image_path else None,
                    )
                )
                detections = (response.output or {}).get("detections") or []
                instances: List[Instance] = []
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    bbox = det.get("bbox_xyxy")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    label_id = det.get("label_id")
                    label = None
                    if label_id is not None:
                        label = self._label_name(int(label_id))
                    instances.append(
                        Instance(
                            frame=frame.ref
                            if isinstance(frame.ref, FrameRef)
                            else FrameRef(frame_index=int(frame.ref.frame_index)),
                            geometry=BBoxGeometry(
                                "bbox", (float(x1), float(y1), float(x2), float(y2))
                            ),
                            label=label,
                            score=float(det.get("score"))
                            if det.get("score") is not None
                            else None,
                            meta={"label_id": label_id} if label_id is not None else {},
                        )
                    )
                outputs.append(Instances(frame=frame.ref, instances=instances))
        return DetectionResult(frames=outputs)

    def _label_name(self, label_id: int) -> str:
        if self._label_names and 0 <= label_id < len(self._label_names):
            name = str(self._label_names[label_id]).strip()
            if name:
                return name
        return f"label_{label_id}"
