from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from annolid.core.behavior.spec import BehaviorSpec, load_behavior_spec
from annolid.core.media.video import CV2Video
from annolid.core.models.base import ModelRequest, RuntimeModel
from annolid.core.output.validate import validate_agent_record


@dataclass(frozen=True)
class AgentRunConfig:
    max_frames: Optional[int] = None
    stride: int = 1
    include_llm_summary: bool = False
    llm_summary_prompt: str = "Summarize the behaviors defined in this behavior spec."


class AgentRunner:
    """Initial headless agent orchestrator producing validated NDJSON records."""

    def __init__(
        self,
        *,
        vision_model: Optional[RuntimeModel] = None,
        llm_model: Optional[RuntimeModel] = None,
        config: Optional[AgentRunConfig] = None,
    ) -> None:
        self._vision_model = vision_model
        self._llm_model = llm_model
        self._config = config or AgentRunConfig()

    def run(
        self,
        *,
        video_path: str | Path,
        out_ndjson: str | Path,
        behavior_spec_path: Optional[str | Path] = None,
    ) -> Path:
        video_path = Path(video_path).expanduser().resolve()
        out_ndjson = Path(out_ndjson).expanduser().resolve()
        out_ndjson.parent.mkdir(parents=True, exist_ok=True)

        schema, schema_path = load_behavior_spec(
            path=behavior_spec_path,
            video_path=video_path,
        )
        agent_meta = self._build_agent_meta(schema, schema_path, video_path)

        if self._config.include_llm_summary and self._llm_model is not None:
            agent_meta["llm_summary"] = self._compute_llm_summary(schema)

        writer = out_ndjson.open("w", encoding="utf-8")
        try:
            for record in self.iter_records(
                video_path=video_path,
                schema=schema,
                agent_meta=agent_meta,
            ):
                validate_agent_record(record)
                writer.write(json.dumps(record, ensure_ascii=False))
                writer.write("\n")
        finally:
            writer.close()
        return out_ndjson

    def iter_records(
        self,
        *,
        video_path: Path,
        schema: BehaviorSpec,
        agent_meta: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        _ = schema  # reserved for future behavior-driven flagging logic
        video = CV2Video(video_path)
        try:
            with ExitStack() as stack:
                if self._vision_model is not None:
                    stack.enter_context(self._vision_model)
                if self._llm_model is not None:
                    stack.enter_context(self._llm_model)

            total = video.total_frames()
            stride = max(1, int(self._config.stride))
            max_frames = self._config.max_frames
            emitted = 0

            frame_indices = range(0, total, stride)
            for frame_index in frame_indices:
                if max_frames is not None and emitted >= int(max_frames):
                    break

                frame = video.load_frame(frame_index)
                height, width = int(frame.shape[0]), int(frame.shape[1])
                timestamp_sec = video.last_timestamp_sec()

                shapes: List[Dict[str, Any]] = []
                if self._vision_model is not None:
                    shapes = self._shapes_from_vision_model(frame)

                record: Dict[str, Any] = {
                    "version": "AnnolidAgentRunner.1",
                    "video_name": video_path.name,
                    "frame_index": int(frame_index),
                    "imagePath": "",
                    "imageHeight": int(height),
                    "imageWidth": int(width),
                    "flags": {},
                    "shapes": shapes,
                    "otherData": {"agent": dict(agent_meta)},
                }
                if timestamp_sec is not None:
                    record["timestamp_sec"] = float(timestamp_sec)

                emitted += 1
                yield record
        finally:
            video.release()

    def _shapes_from_vision_model(self, frame_rgb) -> List[Dict[str, Any]]:
        model = self._vision_model
        if model is None:
            return []

        caps = model.capabilities
        task = "detect" if "detect" in caps.tasks else "caption"
        response = model.predict(ModelRequest(task=task, image=frame_rgb))

        if task != "detect":
            text = response.text or (response.output or {}).get("text")
            if text:
                return [
                    {
                        "label": "caption",
                        "points": [],
                        "shape_type": "text",
                        "description": str(text),
                    }
                ]
            return []

        detections = (response.output or {}).get("detections", [])
        if not isinstance(detections, list):
            return []

        shapes: List[Dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            label_id = det.get("label_id")
            score = det.get("score")
            shapes.append(
                {
                    "label": f"label_{int(label_id) if label_id is not None else 0}",
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "shape_type": "rectangle",
                    "flags": {},
                    "score": float(score) if score is not None else None,
                }
            )
        return shapes

    def _build_agent_meta(
        self,
        schema: BehaviorSpec,
        schema_path: Optional[Path],
        video_path: Path,
    ) -> Dict[str, Any]:
        return {
            "video": str(video_path),
            "behavior_spec_path": str(schema_path) if schema_path is not None else None,
            "behavior_codes": [b.code for b in schema.behaviors],
            "subjects": [s.id for s in schema.subjects],
            "vision_model_id": getattr(self._vision_model, "model_id", None),
            "llm_model_id": getattr(self._llm_model, "model_id", None),
        }

    def _compute_llm_summary(self, schema: BehaviorSpec) -> str:
        model = self._llm_model
        if model is None:
            return ""
        caps = model.capabilities
        if "chat" not in caps.tasks:
            return ""

        behaviors = ", ".join(f"{b.code}={b.name}" for b in schema.behaviors[:30])
        prompt = f"{self._config.llm_summary_prompt}\n\nBehaviors:\n{behaviors}"
        with model:
            resp = model.predict(ModelRequest(task="chat", text=prompt))
        return str(resp.text or (resp.output or {}).get("text") or "").strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Annolid core agent runner and emit validated NDJSON records."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--out", required=True, help="Output NDJSON path.")
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional behavior spec path (project.annolid.json/yaml).",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    runner = AgentRunner(
        config=AgentRunConfig(max_frames=args.max_frames, stride=int(args.stride)),
    )
    runner.run(
        video_path=args.video, out_ndjson=args.out, behavior_spec_path=args.schema
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
