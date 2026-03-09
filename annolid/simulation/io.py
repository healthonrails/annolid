from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from annolid.annotation.pose_schema import PoseSchema
from annolid.core.output.validate import validate_agent_record
from annolid.simulation.types import Pose2DFrame, SimulationRunResult


def load_pose_schema(path: str | Path | None) -> Optional[PoseSchema]:
    if path is None:
        return None
    return PoseSchema.load(path)


def read_pose_frames(
    path: str | Path,
    *,
    pose_schema: PoseSchema | str | Path | None = None,
    video_name: str | None = None,
) -> List[Pose2DFrame]:
    schema = (
        load_pose_schema(pose_schema)
        if not isinstance(pose_schema, PoseSchema)
        else pose_schema
    )
    input_path = Path(path).expanduser()
    if input_path.suffix.lower() == ".ndjson":
        return _read_pose_frames_ndjson(input_path, pose_schema=schema)
    return [
        _read_pose_frame_labelme(input_path, pose_schema=schema, video_name=video_name)
    ]


def write_simulation_ndjson(
    path: str | Path,
    *,
    pose_frames: Sequence[Pose2DFrame],
    result: SimulationRunResult,
    adapter_name: str,
    extra_metadata: Dict[str, Any] | None = None,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    by_frame = {item.frame_index: item for item in result.frames}
    metadata = dict(extra_metadata or {})

    lines: list[str] = []
    for pose_frame in pose_frames:
        sim_frame = by_frame.get(pose_frame.frame_index)
        record = _build_agent_output_record(
            pose_frame,
            sim_frame=sim_frame,
            adapter_name=adapter_name,
            run_metadata=result.metadata,
            extra_metadata=metadata,
        )
        validate_agent_record(record)
        lines.append(json.dumps(record, separators=(",", ":")))

    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def _read_pose_frames_ndjson(
    path: Path, *, pose_schema: PoseSchema | None
) -> List[Pose2DFrame]:
    frames: list[Pose2DFrame] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        frames.append(_pose_frame_from_record(payload, pose_schema=pose_schema))
    return frames


def _read_pose_frame_labelme(
    path: Path,
    *,
    pose_schema: PoseSchema | None,
    video_name: str | None,
) -> Pose2DFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid LabelMe annotation payload in {path}")
    payload = dict(payload)
    payload.setdefault("frame_index", 0)
    payload.setdefault("video_name", str(video_name or path.stem))
    payload.setdefault("flags", {})
    payload.setdefault("otherData", {})
    return _pose_frame_from_record(payload, pose_schema=pose_schema)


def _pose_frame_from_record(
    record: Dict[str, Any], *, pose_schema: PoseSchema | None
) -> Pose2DFrame:
    shapes = record.get("shapes") or []
    if not isinstance(shapes, list):
        raise ValueError("Pose record shapes must be a list")
    points: Dict[str, Tuple[float, float]] = {}
    scores: Dict[str, float] = {}
    instances: Dict[str, str] = {}

    for shape in _iter_point_shapes(shapes):
        label, instance_name = _normalize_point_label(shape, pose_schema=pose_schema)
        point = _extract_point(shape)
        if not label or point is None:
            continue
        points[label] = point
        if instance_name:
            instances[label] = instance_name
        score = _extract_score(shape)
        if score is not None:
            scores[label] = score

    return Pose2DFrame(
        frame_index=int(record.get("frame_index") or record.get("frame") or 0),
        timestamp_sec=_as_optional_float(record.get("timestamp_sec")),
        image_height=int(record.get("imageHeight") or 1),
        image_width=int(record.get("imageWidth") or 1),
        video_name=str(record.get("video_name") or record.get("videoName") or ""),
        image_path=str(record.get("imagePath") or ""),
        points=points,
        scores=scores,
        instances=instances,
        source_record=dict(record),
    )


def _iter_point_shapes(shapes: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if str(shape.get("shape_type") or "").lower() != "point":
            continue
        yield shape


def _normalize_point_label(
    shape: Dict[str, Any], *, pose_schema: PoseSchema | None
) -> Tuple[str, str | None]:
    raw_label = str(shape.get("label") or "").strip()
    instance_name = (
        str(
            shape.get("instance_label")
            or (shape.get("flags") or {}).get("instance_label")
            or ""
        ).strip()
        or None
    )
    display_label = str(
        shape.get("display_label")
        or (shape.get("flags") or {}).get("display_label")
        or ""
    ).strip()

    if display_label:
        return display_label, instance_name
    if pose_schema is not None:
        inferred_instance, base_keypoint = pose_schema.strip_instance_prefix(raw_label)
        if base_keypoint:
            return base_keypoint, instance_name or inferred_instance
    return raw_label, instance_name


def _extract_point(shape: Dict[str, Any]) -> Tuple[float, float] | None:
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None
    first = points[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return None
    try:
        return float(first[0]), float(first[1])
    except (TypeError, ValueError):
        return None


def _extract_score(shape: Dict[str, Any]) -> float | None:
    flags = shape.get("flags") or {}
    candidate = None
    if isinstance(flags, dict):
        candidate = flags.get("score")
    if candidate is None:
        candidate = shape.get("score")
    if candidate is None:
        return None
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return None


def _build_agent_output_record(
    pose_frame: Pose2DFrame,
    *,
    sim_frame: Any,
    adapter_name: str,
    run_metadata: Dict[str, Any],
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    source = dict(pose_frame.source_record)
    record = {
        "version": str(source.get("version") or "Annolid"),
        "video_name": pose_frame.video_name
        or str(source.get("video_name") or "simulation"),
        "frame_index": pose_frame.frame_index,
        "imagePath": pose_frame.image_path,
        "imageHeight": pose_frame.image_height,
        "imageWidth": pose_frame.image_width,
        "flags": dict(source.get("flags") or {}),
        "otherData": dict(source.get("otherData") or {}),
        "shapes": list(source.get("shapes") or []),
    }
    if pose_frame.timestamp_sec is not None:
        record["timestamp_sec"] = pose_frame.timestamp_sec

    simulation_payload: Dict[str, Any] = {
        "adapter": adapter_name,
        "run_metadata": dict(run_metadata or {}),
        "mapping_metadata": dict(extra_metadata or {}),
    }
    if sim_frame is not None:
        simulation_payload["state"] = dict(getattr(sim_frame, "state", {}) or {})
        simulation_payload["diagnostics"] = dict(
            getattr(sim_frame, "diagnostics", {}) or {}
        )
        if (
            getattr(sim_frame, "timestamp_sec", None) is not None
            and "timestamp_sec" not in record
        ):
            record["timestamp_sec"] = sim_frame.timestamp_sec
    record["otherData"]["simulation"] = simulation_payload
    return record


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
