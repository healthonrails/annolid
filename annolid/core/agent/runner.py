from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

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
        seed_record_provider: Optional[
            Callable[[int], Optional[Dict[str, Any]]]
        ] = None,
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

                    seed_shapes: List[Dict[str, Any]] = []
                    seed_keypoints = None
                    if seed_record_provider is not None:
                        try:
                            seed_record = seed_record_provider(int(frame_index))
                        except Exception:
                            seed_record = None
                        if isinstance(seed_record, dict):
                            seed_shapes = list(seed_record.get("shapes") or [])
                            seed_keypoints = self._seed_keypoints_from_shapes(
                                seed_shapes
                            )

                    shapes: List[Dict[str, Any]] = []
                    if self._vision_model is not None:
                        shapes = self._shapes_from_vision_model(
                            frame,
                            frame_index=int(frame_index),
                            seed_keypoints=seed_keypoints,
                        )
                    if seed_shapes:
                        shapes = self._merge_seed_shapes(seed_shapes, shapes)

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

    def _shapes_from_vision_model(
        self,
        frame_rgb,
        *,
        frame_index: int,
        seed_keypoints: Optional[Dict[str, Dict[str, object]]],
    ) -> List[Dict[str, Any]]:
        model = self._vision_model
        if model is None:
            return []

        caps = model.capabilities
        task = "detect" if "detect" in caps.tasks else "caption"
        params: Dict[str, Any] = {"frame_index": int(frame_index)}
        if seed_keypoints:
            params["seed_keypoints"] = seed_keypoints
        response = model.predict(
            ModelRequest(task=task, image=frame_rgb, params=params)
        )

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
            shapes.extend(self._keypoint_shapes_from_detection(det))
        return shapes

    @staticmethod
    def _seed_keypoints_from_shapes(
        shapes: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Dict[str, object]]]:
        seeds: Dict[str, Dict[str, object]] = {}
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type not in {"point", "circle"}:
                continue
            label = str(shape.get("label") or "").strip()
            if not label:
                continue
            points = shape.get("points")
            if not isinstance(points, list) or not points:
                continue
            first = points[0]
            if not isinstance(first, list) or len(first) < 2:
                continue
            x, y = float(first[0]), float(first[1])
            visible = shape.get("visible")
            if visible is None:
                visible = True
            seeds[label] = {"xy": [x, y], "visible": bool(visible)}
        return seeds or None

    @staticmethod
    def _sanitize_shape(shape: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(shape, dict):
            return None

        label_raw = shape.get("label")
        label = str(label_raw).strip() if label_raw is not None else ""
        if not label:
            label = "unknown"
        shape["label"] = label

        st_raw = shape.get("shape_type")
        shape_type = str(st_raw).strip() if st_raw is not None else ""
        if not shape_type:
            return None
        shape["shape_type"] = shape_type
        shape_type_norm = shape_type.lower()

        points = shape.get("points")
        if not isinstance(points, list):
            points = []
        clean_points: list[list[float]] = []
        for pt in points:
            if not isinstance(pt, list) or len(pt) < 2:
                continue
            try:
                x = float(pt[0])
                y = float(pt[1])
            except Exception:
                continue
            clean_points.append([x, y])
        shape["points"] = clean_points

        if "group_id" in shape:
            gid = shape.get("group_id")
            if gid is None or isinstance(gid, int):
                pass
            elif isinstance(gid, str) and gid.strip().isdigit():
                shape["group_id"] = int(gid.strip())
            else:
                shape["group_id"] = None

        if "flags" in shape:
            flags = shape.get("flags")
            if isinstance(flags, dict):
                pass
            elif flags is None:
                shape.pop("flags", None)
            else:
                shape["flags"] = {}

        if "visible" in shape:
            vis = shape.get("visible")
            if isinstance(vis, bool):
                pass
            elif isinstance(vis, int) and vis in {0, 1}:
                shape["visible"] = bool(vis)
            elif isinstance(vis, str):
                val = vis.strip().lower()
                if val in {"true", "1", "yes"}:
                    shape["visible"] = True
                elif val in {"false", "0", "no"}:
                    shape["visible"] = False
                else:
                    shape.pop("visible", None)
            else:
                shape.pop("visible", None)

        if "description" in shape:
            desc = shape.get("description")
            if desc is None:
                shape.pop("description", None)
            elif isinstance(desc, str):
                pass
            else:
                shape["description"] = str(desc)

        if "mask" in shape:
            mask_val = shape.get("mask")
            if isinstance(mask_val, str) and mask_val:
                pass
            else:
                shape.pop("mask", None)
                if shape_type_norm == "mask":
                    return None

        return shape

    @staticmethod
    def _merge_seed_shapes(
        seed_shapes: Sequence[Dict[str, Any]],
        predicted_shapes: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for shape in seed_shapes:
            if not isinstance(shape, dict):
                continue
            sanitized = AgentRunner._sanitize_shape(dict(shape))
            if sanitized is not None:
                merged.append(sanitized)
        seed_point_labels: set[str] = set()
        for shape in merged:
            if not isinstance(shape, dict):
                continue
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type not in {"point", "circle"}:
                continue
            label = str(shape.get("label") or "").strip()
            if label:
                seed_point_labels.add(label)

        for shape in predicted_shapes:
            if not isinstance(shape, dict):
                continue
            sanitized = AgentRunner._sanitize_shape(dict(shape))
            if sanitized is None:
                continue
            shape_type = str(sanitized.get("shape_type") or "").strip().lower()
            if shape_type in {"point", "circle"}:
                label = str(sanitized.get("label") or "").strip()
                if label and label in seed_point_labels:
                    continue
            merged.append(sanitized)
        return merged

    @staticmethod
    def _keypoint_shapes_from_detection(det: Dict[str, Any]) -> List[Dict[str, Any]]:
        keypoints = det.get("keypoints_xy")
        if not isinstance(keypoints, list) or not keypoints:
            return []
        names = det.get("keypoint_names")
        if not isinstance(names, list):
            names = None
        scores = det.get("keypoint_scores")
        if not isinstance(scores, list):
            scores = None
        visible = det.get("keypoint_visible")
        if not isinstance(visible, list):
            visible = None

        shapes: List[Dict[str, Any]] = []
        for idx, kp in enumerate(keypoints):
            if not isinstance(kp, list) or len(kp) < 2:
                continue
            x, y = float(kp[0]), float(kp[1])
            label = None
            if names is not None and idx < len(names):
                label = str(names[idx]).strip() or None
            if label is None:
                label = f"kp_{idx}"
            kp_visible = True
            if visible is not None and idx < len(visible):
                kp_visible = bool(visible[idx])
            kp_score = None
            if scores is not None and idx < len(scores):
                try:
                    kp_score = float(scores[idx])
                except Exception:
                    kp_score = None
            shapes.append(
                {
                    "label": label,
                    "points": [[x, y]],
                    "shape_type": "point",
                    "flags": {},
                    "visible": kp_visible,
                    "score": kp_score,
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
