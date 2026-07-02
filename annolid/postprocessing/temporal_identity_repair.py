from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from annolid.postprocessing.identity_governor import (
    IdentityCorrection,
    IdentityGovernorResult,
)


_TRACK_ID_KEYS = ("track_id", "tracking_id", "instance_id", "group_id")
_DISTANCE_WEIGHT = 0.35
_VELOCITY_WEIGHT = 0.25
_OVERLAP_WEIGHT = 0.20
_AREA_WEIGHT = 0.12
_ORIENTATION_WEIGHT = 0.08
_MATCH_COST_CAP = 3.0
_GLOBAL_ASSIGNMENT_DP_SHAPE_LIMIT = 16
_CUTIE_RECOVERY_NOTE_TOKENS = (
    "recovered",
    "filled_from_previous",
    "fallback_previous_mask",
    "repaired_bad_shape",
    "frame_sized_artifact",
    "recovery_backoff",
)


@dataclass(frozen=True)
class _ShapeFeatures:
    centroid: tuple[float, float] | None
    area: float
    bbox: tuple[float, float, float, float] | None
    orientation: float | None


@dataclass(frozen=True)
class _TrackState:
    track_id: str
    previous: _ShapeFeatures
    predicted_centroid: tuple[float, float] | None
    motion_bbox: tuple[float, float, float, float] | None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _frame_number_from_name(path: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.json$)", path.name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _point_xy(point: Any) -> tuple[float | None, float | None]:
    if isinstance(point, (str, bytes)):
        return None, None
    if not isinstance(point, Sequence) or len(point) < 2:
        return None, None
    return _safe_float(point[0]), _safe_float(point[1])


def _shape_points(shape: Mapping[str, Any]) -> list[tuple[float, float]]:
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return []

    shape_type = _normalize_text(shape.get("shape_type")).lower()
    if shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = _point_xy(points[0])
        x2, y2 = _point_xy(points[1])
        if None in (x1, y1, x2, y2):
            return []
        left = min(float(x1), float(x2))
        right = max(float(x1), float(x2))
        top = min(float(y1), float(y2))
        bottom = max(float(y1), float(y2))
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    coords: list[tuple[float, float]] = []
    for point in points:
        x, y = _point_xy(point)
        if x is not None and y is not None:
            coords.append((float(x), float(y)))
    return coords


def _shape_centroid(shape: Mapping[str, Any]) -> tuple[float, float] | None:
    shape_type = _normalize_text(shape.get("shape_type")).lower()
    points = shape.get("points")
    if isinstance(points, list) and shape_type == "rectangle" and len(points) >= 2:
        x1, y1 = _point_xy(points[0])
        x2, y2 = _point_xy(points[1])
        if None in (x1, y1, x2, y2):
            return None
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    coords = _shape_points(shape)
    if not coords:
        return None
    return (
        float(sum(x for x, _ in coords) / len(coords)),
        float(sum(y for _, y in coords) / len(coords)),
    )


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    double_area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        double_area += (float(x1) * float(y2)) - (float(x2) * float(y1))
    return abs(double_area) / 2.0


def _shape_bbox(shape: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    coords = _shape_points(shape)
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _shape_orientation(shape: Mapping[str, Any]) -> float | None:
    coords = _shape_points(shape)
    if len(coords) < 2:
        return None
    centroid = _shape_centroid(shape)
    if centroid is None:
        return None
    centered = [(x - centroid[0], y - centroid[1]) for x, y in coords]
    xx = sum(x * x for x, _y in centered) / len(centered)
    yy = sum(y * y for _x, y in centered) / len(centered)
    xy = sum(x * y for x, y in centered) / len(centered)
    if math.isclose(xx, yy, abs_tol=1e-9) and math.isclose(xy, 0.0, abs_tol=1e-9):
        return None
    return float(0.5 * math.atan2(2.0 * xy, xx - yy) % math.pi)


def _shape_features(shape: Mapping[str, Any]) -> _ShapeFeatures:
    points = _shape_points(shape)
    return _ShapeFeatures(
        centroid=_shape_centroid(shape),
        area=_polygon_area(points),
        bbox=_shape_bbox(shape),
        orientation=_shape_orientation(shape),
    )


def _distance(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> float | None:
    if a is None or b is None:
        return None
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _bbox_iou(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if a_area <= 0.0 or b_area <= 0.0:
        return None
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = a_area + b_area - intersection
    if union <= 0.0:
        return None
    return float(intersection / union)


def _shift_bbox(
    bbox: tuple[float, float, float, float] | None,
    *,
    dx: float,
    dy: float,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + float(dx), y1 + float(dy), x2 + float(dx), y2 + float(dy))


def _angle_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    delta = abs((float(a) - float(b)) % math.pi)
    if delta > math.pi / 2.0:
        delta = math.pi - delta
    return float(delta)


def _shape_instance_label(shape: Mapping[str, Any]) -> str:
    for key in ("instance_label", "instance_name"):
        value = _normalize_text(shape.get(key))
        if value:
            return value
    flags = shape.get("flags")
    if isinstance(flags, Mapping):
        for key in ("instance_label", "instance_name"):
            value = _normalize_text(flags.get(key))
            if value:
                return value
    label = _normalize_text(shape.get("label"))
    if ":" in label:
        return label.split(":", 1)[0].strip()
    return label


def _shape_track_id(shape: Mapping[str, Any]) -> str:
    for key in _TRACK_ID_KEYS:
        value = _normalize_text(shape.get(key))
        if value:
            return value
    flags = shape.get("flags")
    if isinstance(flags, Mapping):
        for key in _TRACK_ID_KEYS:
            value = _normalize_text(flags.get(key))
            if value:
                return value
    return _shape_instance_label(shape)


def _shape_source(shape: Mapping[str, Any]) -> str:
    source = shape.get("annotation_source")
    if source is not None:
        return _normalize_text(source).lower()
    other = shape.get("other_data")
    if isinstance(other, Mapping) and other.get("annotation_source") is not None:
        return _normalize_text(other.get("annotation_source")).lower()
    description = shape.get("description")
    if isinstance(description, str) and description.strip():
        token = description.split(";", 1)[0].split(":", 1)[0].strip().lower()
        if token in {
            "grounding_sam",
            "cutie",
            "cutie_vos",
            "cutie_vos_segment",
            "dinokpseg",
            "sam3",
        }:
            return token
    flags = shape.get("flags")
    if isinstance(flags, Mapping) and any(
        flags.get(key) not in (None, "") for key in _TRACK_ID_KEYS
    ):
        return "propagated_instance"
    return ""


def _shape_note(shape: Mapping[str, Any]) -> str:
    notes: list[str] = []
    description = shape.get("description")
    if isinstance(description, str) and description.strip():
        notes.append(description.strip())
    other = shape.get("other_data")
    if isinstance(other, Mapping):
        note = other.get("note")
        if isinstance(note, str) and note.strip():
            notes.append(note.strip())
    flags = shape.get("flags")
    if isinstance(flags, Mapping):
        note = flags.get("note")
        if isinstance(note, str) and note.strip():
            notes.append(note.strip())
    return "; ".join(notes)


def _has_cutie_recovery_note(shape: Mapping[str, Any]) -> bool:
    note = _shape_note(shape).lower()
    return any(token in note for token in _CUTIE_RECOVERY_NOTE_TOKENS)


def _is_temporal_track_shape(shape: Mapping[str, Any]) -> bool:
    if _shape_centroid(shape) is None:
        return False
    if not _shape_track_id(shape):
        return False
    if _shape_source(shape):
        return True
    shape_type = _normalize_text(shape.get("shape_type")).lower()
    return shape_type in {"polygon", "rectangle", "mask"}


@dataclass(frozen=True)
class _TrackTemplate:
    track_id: str
    label: str
    instance_label: str
    group_id: Any
    top_level_ids: dict[str, Any] = field(default_factory=dict)
    flag_ids: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)


def _track_template_from_shape(shape: Mapping[str, Any]) -> _TrackTemplate:
    flags = (
        dict(shape.get("flags") or {})
        if isinstance(shape.get("flags"), Mapping)
        else {}
    )
    return _TrackTemplate(
        track_id=_shape_track_id(shape),
        label=_normalize_text(shape.get("label")),
        instance_label=_shape_instance_label(shape),
        group_id=shape.get("group_id"),
        top_level_ids={
            key: shape.get(key)
            for key in _TRACK_ID_KEYS
            if shape.get(key) not in (None, "")
        },
        flag_ids={
            key: flags.get(key)
            for key in _TRACK_ID_KEYS
            if flags.get(key) not in (None, "")
        },
        flags=flags,
    )


def _track_template_with_id(shape: Mapping[str, Any], track_id: str) -> _TrackTemplate:
    template = _track_template_from_shape(shape)
    flags = dict(template.flags)
    flags.setdefault("track_id", track_id)
    return _TrackTemplate(
        track_id=track_id,
        label=template.label,
        instance_label=template.instance_label or track_id,
        group_id=template.group_id,
        top_level_ids=dict(template.top_level_ids),
        flag_ids={**template.flag_ids, "track_id": track_id},
        flags=flags,
    )


def _apply_track_template(
    shape: Mapping[str, Any],
    template: _TrackTemplate,
    *,
    reason: str,
) -> dict[str, Any]:
    updated = dict(shape)
    if template.label:
        updated["label"] = template.label
    if template.instance_label:
        if "instance_label" in updated or "instance_name" not in updated:
            updated["instance_label"] = template.instance_label
        if _normalize_text(updated.get("instance_name")):
            updated["instance_name"] = template.instance_label
    if template.group_id is not None:
        updated["group_id"] = template.group_id
    for key, value in template.top_level_ids.items():
        updated[key] = value

    flags = (
        dict(updated.get("flags") or {})
        if isinstance(updated.get("flags"), Mapping)
        else {}
    )
    for key, value in template.flag_ids.items():
        flags[key] = value
    if template.track_id:
        flags.setdefault("track_id", template.track_id)
    if template.instance_label:
        flags["instance_label"] = template.instance_label
    flags["annolid_correction"] = reason
    updated["flags"] = flags
    return updated


def _candidate_shapes(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    shapes = record.get("shapes")
    if not isinstance(shapes, list):
        return []
    return [dict(shape) for shape in shapes if isinstance(shape, dict)]


def _temporal_track_shapes(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        shape for shape in _candidate_shapes(record) if _is_temporal_track_shape(shape)
    ]


def _choose_reference_templates(
    records: list[tuple[int, Path, dict[str, Any]]],
    *,
    start_frame: int,
    expected_instance_count: int | None,
) -> dict[str, _TrackTemplate]:
    best: list[dict[str, Any]] = []
    for frame, _path, record in sorted(records, key=lambda item: item[0]):
        if frame < start_frame:
            continue
        shapes = _temporal_track_shapes(record)
        if not shapes:
            continue
        if expected_instance_count is None or len(shapes) >= expected_instance_count:
            best = (
                shapes[:expected_instance_count] if expected_instance_count else shapes
            )
            break
        if len(shapes) > len(best):
            best = shapes

    templates: dict[str, _TrackTemplate] = {}
    raw_ids = [_shape_track_id(shape) for shape in best]
    duplicate_ids = {track_id for track_id in raw_ids if raw_ids.count(track_id) > 1}
    for idx, shape in enumerate(best):
        raw_id = _shape_track_id(shape)
        if not raw_id or raw_id in duplicate_ids:
            template = _track_template_with_id(shape, f"track_{idx + 1}")
        else:
            template = _track_template_from_shape(shape)
        if template.track_id:
            templates[template.track_id] = template
    return templates


def _match_shapes_to_tracks(
    shapes: list[dict[str, Any]],
    track_history: Mapping[str, Sequence[tuple[int, dict[str, Any]]]],
    *,
    frame_number: int,
    max_match_distance: float,
) -> dict[int, str]:
    track_states = [
        state
        for track_id, history in track_history.items()
        if history and history[-1][0] < frame_number
        for state in (
            _build_track_state(
                track_id,
                history,
                frame_number=frame_number,
            ),
        )
        if state is not None
    ]
    if not shapes or not track_states:
        return {}

    shape_features = [_shape_features(shape) for shape in shapes]
    cost_matrix: list[list[float | None]] = []
    for state in track_states:
        cost_matrix.append(
            [
                _match_cost(
                    shape_features[shape_idx],
                    state,
                    max_match_distance=max_match_distance,
                )
                for shape_idx in range(len(shapes))
            ]
        )

    selected = _solve_global_assignment(cost_matrix)
    return {
        shape_idx: track_states[track_idx].track_id for track_idx, shape_idx in selected
    }


def _predict_centroid(
    history: Sequence[tuple[int, dict[str, Any]]],
    *,
    frame_number: int,
) -> tuple[float, float] | None:
    if not history:
        return None
    last_frame, last_shape = history[-1]
    last_centroid = _shape_centroid(last_shape)
    if last_centroid is None:
        return None

    for prev_frame, prev_shape in reversed(history[:-1]):
        prev_centroid = _shape_centroid(prev_shape)
        frame_delta = int(last_frame) - int(prev_frame)
        if prev_centroid is None or frame_delta <= 0:
            continue
        step_x = (last_centroid[0] - prev_centroid[0]) / float(frame_delta)
        step_y = (last_centroid[1] - prev_centroid[1]) / float(frame_delta)
        current_delta = int(frame_number) - int(last_frame)
        return (
            last_centroid[0] + step_x * float(current_delta),
            last_centroid[1] + step_y * float(current_delta),
        )
    return last_centroid


def _build_track_state(
    track_id: str,
    history: Sequence[tuple[int, dict[str, Any]]],
    *,
    frame_number: int,
) -> _TrackState | None:
    if not history:
        return None
    _last_frame, last_shape = history[-1]
    previous = _shape_features(last_shape)
    predicted_centroid = _predict_centroid(history, frame_number=frame_number)
    motion_bbox = previous.bbox
    if previous.bbox is not None and previous.centroid is not None:
        if predicted_centroid is not None:
            motion_bbox = _shift_bbox(
                previous.bbox,
                dx=predicted_centroid[0] - previous.centroid[0],
                dy=predicted_centroid[1] - previous.centroid[1],
            )
    return _TrackState(
        track_id=str(track_id),
        previous=previous,
        predicted_centroid=predicted_centroid,
        motion_bbox=motion_bbox,
    )


def _normalized_distance(distance: float | None, max_match_distance: float) -> float:
    if distance is None:
        return _MATCH_COST_CAP
    if max_match_distance <= 0.0:
        return _MATCH_COST_CAP
    return min(float(distance) / float(max_match_distance), _MATCH_COST_CAP)


def _area_penalty(current_area: float, previous_area: float) -> float:
    if current_area <= 0.0 or previous_area <= 0.0:
        return 1.0
    return min(abs(math.log(current_area / previous_area)), _MATCH_COST_CAP)


def _match_cost(
    current: _ShapeFeatures,
    state: _TrackState,
    *,
    max_match_distance: float,
) -> float | None:
    distance_to_last = _distance(current.centroid, state.previous.centroid)
    distance_to_prediction = _distance(current.centroid, state.predicted_centroid)

    distances = [
        value
        for value in (distance_to_last, distance_to_prediction)
        if value is not None
    ]
    if not distances:
        return None
    if min(distances) > max_match_distance:
        return None

    overlap = _bbox_iou(current.bbox, state.motion_bbox)
    overlap_penalty = 1.0 if overlap is None else 1.0 - overlap
    angle_penalty = _angle_delta(current.orientation, state.previous.orientation)
    orientation_penalty = (
        1.0 if angle_penalty is None else min(angle_penalty / (math.pi / 2.0), 1.0)
    )

    return float(
        _DISTANCE_WEIGHT * _normalized_distance(distance_to_last, max_match_distance)
        + _VELOCITY_WEIGHT
        * _normalized_distance(distance_to_prediction, max_match_distance)
        + _OVERLAP_WEIGHT * min(max(overlap_penalty, 0.0), 1.0)
        + _AREA_WEIGHT * _area_penalty(current.area, state.previous.area)
        + _ORIENTATION_WEIGHT * orientation_penalty
    )


def _solve_global_assignment(
    cost_matrix: Sequence[Sequence[float | None]],
) -> list[tuple[int, int]]:
    """Return track/shape pairs that maximize matched tracks, then minimize cost."""

    if not cost_matrix:
        return []
    shape_count = max((len(row) for row in cost_matrix), default=0)
    if shape_count <= 0:
        return []
    if shape_count > _GLOBAL_ASSIGNMENT_DP_SHAPE_LIMIT:
        return _solve_greedy_assignment(cost_matrix)

    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for track_idx, row in enumerate(cost_matrix):
        next_states = dict(states)
        for mask, (cost, path) in states.items():
            for shape_idx, candidate_cost in enumerate(row):
                if candidate_cost is None or mask & (1 << shape_idx):
                    continue
                next_mask = mask | (1 << shape_idx)
                next_cost = cost + float(candidate_cost)
                existing = next_states.get(next_mask)
                next_path = [*path, (track_idx, shape_idx)]
                if existing is None or (
                    len(next_path),
                    -next_cost,
                ) > (
                    len(existing[1]),
                    -existing[0],
                ):
                    next_states[next_mask] = (next_cost, next_path)
        states = next_states

    _best_mask, (_best_cost, best_path) = max(
        states.items(),
        key=lambda item: (len(item[1][1]), -item[1][0]),
    )
    return best_path


def _solve_greedy_assignment(
    cost_matrix: Sequence[Sequence[float | None]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for track_idx, row in enumerate(cost_matrix):
        for shape_idx, cost in enumerate(row):
            if cost is not None:
                candidates.append((float(cost), track_idx, shape_idx))
    candidates.sort(key=lambda item: item[0])

    selected: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_shapes: set[int] = set()
    for _cost, track_idx, shape_idx in candidates:
        if track_idx in used_tracks or shape_idx in used_shapes:
            continue
        selected.append((track_idx, shape_idx))
        used_tracks.add(track_idx)
        used_shapes.add(shape_idx)
    return selected


def _translate_shape(
    shape: Mapping[str, Any],
    *,
    dx: float,
    dy: float,
    reason: str,
) -> dict[str, Any]:
    filled = dict(shape)
    points: list[list[float]] = []
    for point in shape.get("points") or []:
        x, y = _point_xy(point)
        if x is None or y is None:
            continue
        points.append([float(x) + float(dx), float(y) + float(dy)])
    if points:
        filled["points"] = points
    flags = (
        dict(filled.get("flags") or {})
        if isinstance(filled.get("flags"), Mapping)
        else {}
    )
    flags["annolid_correction"] = reason
    flags["occlusion_fill"] = True
    filled["flags"] = flags
    return filled


def _interpolate_missing_shape(
    *,
    track_id: str,
    frame: int,
    prev_seen: tuple[int, dict[str, Any]],
    next_seen: tuple[int, dict[str, Any]] | None,
    predicted_centroid: tuple[float, float] | None = None,
    template: _TrackTemplate,
) -> dict[str, Any]:
    prev_frame, prev_shape = prev_seen
    base = dict(prev_shape)
    reason = (
        "occlusion_gap_interpolated"
        if next_seen is not None
        else "occlusion_gap_carried"
    )
    if next_seen is not None and next_seen[0] != prev_frame:
        next_frame, next_shape = next_seen
        prev_centroid = _shape_centroid(prev_shape)
        next_centroid = _shape_centroid(next_shape)
        if prev_centroid is not None and next_centroid is not None:
            ratio = float(frame - prev_frame) / float(next_frame - prev_frame)
            target_x = prev_centroid[0] + (next_centroid[0] - prev_centroid[0]) * ratio
            target_y = prev_centroid[1] + (next_centroid[1] - prev_centroid[1]) * ratio
            base_centroid = _shape_centroid(base)
            if base_centroid is not None:
                base = _translate_shape(
                    base,
                    dx=target_x - base_centroid[0],
                    dy=target_y - base_centroid[1],
                    reason=reason,
                )
    else:
        base_centroid = _shape_centroid(base)
        if base_centroid is not None and predicted_centroid is not None:
            base = _translate_shape(
                base,
                dx=predicted_centroid[0] - base_centroid[0],
                dy=predicted_centroid[1] - base_centroid[1],
                reason=reason,
            )
        else:
            base = _translate_shape(base, dx=0.0, dy=0.0, reason=reason)
    base = _apply_track_template(base, template, reason=reason)
    flags = dict(base.get("flags") or {})
    flags["filled_missing_track_id"] = track_id
    base["flags"] = flags
    return base


def _find_next_seen(
    records_by_frame: Mapping[int, dict[str, Any]],
    *,
    current_frame: int,
    track_id: str,
    max_gap_frames: int,
) -> tuple[int, dict[str, Any]] | None:
    for frame in range(current_frame + 1, current_frame + max_gap_frames + 1):
        record = records_by_frame.get(frame)
        if not record:
            continue
        for shape in _candidate_shapes(record):
            if _shape_track_id(shape) == track_id:
                return frame, shape
    return None


def _load_records(
    annotation_dir: Path,
) -> tuple[list[tuple[int, Path, dict[str, Any]]], int]:
    records: list[tuple[int, Path, dict[str, Any]]] = []
    scanned_observations = 0
    for json_path in sorted(annotation_dir.glob("*.json")):
        frame_number = _frame_number_from_name(json_path)
        if frame_number is None:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        scanned_observations += len(_temporal_track_shapes(payload))
        records.append((int(frame_number), json_path, payload))
    return records, scanned_observations


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _quality_event(
    *,
    frame: int,
    issue: str,
    **details: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "frame": int(frame),
        "issue": str(issue),
    }
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, set):
            event[key] = sorted(str(item) for item in value)
        elif isinstance(value, tuple):
            event[key] = list(value)
        else:
            event[key] = value
    return event


def _quality_event_counts(
    quality_events: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    frames_by_issue: dict[str, set[int]] = {}
    for event in quality_events:
        issue = _normalize_text(event.get("issue"))
        frame_value = event.get("frame")
        if not issue:
            continue
        try:
            frame = int(frame_value)
        except (TypeError, ValueError):
            continue
        frames_by_issue.setdefault(issue, set()).add(frame)
    return {
        str(issue): int(len(frames))
        for issue, frames in sorted(frames_by_issue.items())
    }


def _suspicious_same_id_motion(
    *,
    shape: Mapping[str, Any],
    history: Sequence[tuple[int, dict[str, Any]]],
    frame_number: int,
    max_match_distance: float,
) -> dict[str, float] | None:
    state = _build_track_state(
        _shape_track_id(shape),
        history,
        frame_number=frame_number,
    )
    if state is None:
        return None
    current = _shape_features(shape)
    distance_to_last = _distance(current.centroid, state.previous.centroid)
    distance_to_prediction = _distance(current.centroid, state.predicted_centroid)
    distances = [
        value
        for value in (distance_to_last, distance_to_prediction)
        if value is not None
    ]
    if not distances or min(distances) <= max_match_distance:
        return None
    details: dict[str, float] = {}
    if distance_to_last is not None:
        details["distance_to_last"] = round(float(distance_to_last), 3)
    if distance_to_prediction is not None:
        details["distance_to_prediction"] = round(float(distance_to_prediction), 3)
    return details


def _write_report(
    *,
    report_path: Path,
    annotation_dir: Path,
    dry_run: bool,
    scanned_files: int,
    scanned_observations: int,
    corrections: Sequence[IdentityCorrection],
    updated_files: int,
    updated_shapes: int,
    stats: Mapping[str, int],
    quality_events: Sequence[Mapping[str, Any]] = (),
) -> None:
    quality_counts = _quality_event_counts(quality_events)
    payload = {
        "annotation_dir": str(annotation_dir),
        "mode": "temporal_identity_repair",
        "dry_run": bool(dry_run),
        "scanned_files": int(scanned_files),
        "scanned_observations": int(scanned_observations),
        "updated_files": int(updated_files),
        "updated_shapes": int(updated_shapes),
        **{key: int(value) for key, value in stats.items()},
        "quality_event_counts": quality_counts,
        "quality_events": [dict(event) for event in quality_events],
        "corrections": [
            {
                "track_id": correction.track_id,
                "frame_start": correction.frame_start,
                "frame_end": correction.frame_end,
                "observed_label": correction.observed_label,
                "corrected_label": correction.corrected_label,
                "rule_name": correction.rule_name,
                "rule_frame_start": correction.rule_frame_start,
                "rule_frame_end": correction.rule_frame_end,
                "observation_count": correction.observation_count,
            }
            for correction in corrections
        ],
    }
    _write_json_atomic(report_path, payload)


def run_temporal_identity_repair(
    annotation_dir: str | Path,
    *,
    start_frame: int = 0,
    expected_instance_count: int | None = None,
    max_gap_frames: int = 5,
    max_match_distance: float = 80.0,
    apply_changes: bool = False,
    report_path: str | Path | None = None,
) -> IdentityGovernorResult:
    """Repair LabelMe frame JSON identities by temporal multi-cue continuity.

    This is intended for CUTIE/SAM-style frame annotations where the label itself
    is the tracked identity. It never rewrites files in dry-run mode.
    """

    resolved_dir = Path(annotation_dir).expanduser().resolve()
    expected_count = (
        None
        if expected_instance_count is None or int(expected_instance_count) <= 0
        else int(expected_instance_count)
    )
    records, scanned_observations = _load_records(resolved_dir)
    templates = _choose_reference_templates(
        records,
        start_frame=max(0, int(start_frame)),
        expected_instance_count=expected_count,
    )

    records_by_frame = {frame: dict(record) for frame, _path, record in records}
    track_history: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    updated_payloads: dict[Path, dict[str, Any]] = {}
    corrections: list[IdentityCorrection] = []
    quality_events: list[dict[str, Any]] = []
    template_ids = list(templates)
    id_switches = 0
    missing = 0
    max_distance = max(0.0, float(max_match_distance))

    for frame, json_path, record in sorted(records, key=lambda item: item[0]):
        if frame < max(0, int(start_frame)):
            continue
        all_shapes = _candidate_shapes(record)
        temporal_pairs = [
            (idx, shape)
            for idx, shape in enumerate(all_shapes)
            if _is_temporal_track_shape(shape)
        ]
        shapes = [shape for _idx, shape in temporal_pairs]
        raw_current_ids = [_shape_track_id(shape) for shape in shapes]
        raw_id_counts = Counter(track_id for track_id in raw_current_ids if track_id)
        duplicate_current_ids = {
            track_id for track_id, count in raw_id_counts.items() if count > 1
        }
        raw_id_set = set(raw_current_ids)
        known_id_set = set(templates)
        unexpected_ids = sorted(
            track_id for track_id in raw_id_set - known_id_set if track_id
        )
        raw_missing_ids = sorted(track_id for track_id in known_id_set - raw_id_set)
        if expected_count is not None and len(shapes) != expected_count:
            quality_events.append(
                _quality_event(
                    frame=int(frame),
                    issue="count_mismatch_before_repair",
                    observed_count=int(len(shapes)),
                    expected_count=int(expected_count),
                )
            )
        if duplicate_current_ids:
            quality_events.append(
                _quality_event(
                    frame=int(frame),
                    issue="duplicate_id_before_repair",
                    track_ids=duplicate_current_ids,
                )
            )
        if unexpected_ids:
            quality_events.append(
                _quality_event(
                    frame=int(frame),
                    issue="unexpected_id_before_repair",
                    track_ids=unexpected_ids,
                )
            )
        if raw_missing_ids and track_history:
            quality_events.append(
                _quality_event(
                    frame=int(frame),
                    issue="missing_id_before_repair",
                    track_ids=raw_missing_ids,
                )
            )
        for shape in shapes:
            current_id = _shape_track_id(shape)
            if _has_cutie_recovery_note(shape):
                quality_events.append(
                    _quality_event(
                        frame=int(frame),
                        issue="cutie_recovery_note",
                        track_id=current_id,
                        note=_shape_note(shape),
                    )
                )
            if current_id in templates and current_id not in duplicate_current_ids:
                motion_details = _suspicious_same_id_motion(
                    shape=shape,
                    history=track_history.get(current_id) or (),
                    frame_number=int(frame),
                    max_match_distance=max_distance,
                )
                if motion_details:
                    quality_events.append(
                        _quality_event(
                            frame=int(frame),
                            issue="implausible_same_id_motion",
                            track_id=current_id,
                            **motion_details,
                        )
                    )
        assignments = _match_shapes_to_tracks(
            shapes,
            track_history,
            frame_number=int(frame),
            max_match_distance=max_distance,
        )
        initializing_frame = not track_history
        updated_shapes_for_frame: list[dict[str, Any]] = []
        present_tracks: set[str] = set()

        for idx, shape in enumerate(shapes):
            current_id = _shape_track_id(shape)
            assigned_id = assignments.get(idx)
            initializing_track_id = False
            if (
                initializing_frame
                and current_id not in templates
                and idx < len(template_ids)
            ):
                assigned_id = template_ids[idx]
                initializing_track_id = True
            if assigned_id and assigned_id in templates and assigned_id != current_id:
                reason = (
                    "track_id_initialized"
                    if initializing_track_id
                    else "temporal_identity_switch_corrected"
                )
                shape = _apply_track_template(
                    shape,
                    templates[assigned_id],
                    reason=reason,
                )
                if not initializing_track_id:
                    id_switches += 1
                    corrections.append(
                        IdentityCorrection(
                            track_id=str(assigned_id),
                            frame_start=int(frame),
                            frame_end=int(frame),
                            observed_label=str(current_id),
                            corrected_label=str(assigned_id),
                            rule_name="temporal_continuity",
                            rule_frame_start=int(frame),
                            rule_frame_end=int(frame),
                            observation_count=1,
                        )
                    )
                    quality_events.append(
                        _quality_event(
                            frame=int(frame),
                            issue="id_switch_corrected",
                            observed_track_id=str(current_id),
                            corrected_track_id=str(assigned_id),
                        )
                    )
                current_id = assigned_id
            should_update_history = bool(
                current_id in templates
                and current_id not in present_tracks
                and (assigned_id is not None or current_id not in duplicate_current_ids)
            )
            if should_update_history:
                present_tracks.add(current_id)
                track_history.setdefault(current_id, []).append(
                    (int(frame), dict(shape))
                )
            elif current_id in templates and current_id in duplicate_current_ids:
                quality_events.append(
                    _quality_event(
                        frame=int(frame),
                        issue="ambiguous_duplicate_not_used_for_history",
                        track_id=current_id,
                    )
                )
            updated_shapes_for_frame.append(shape)

        repaired_all_shapes = list(all_shapes)
        for (original_idx, _shape), updated_shape in zip(
            temporal_pairs, updated_shapes_for_frame
        ):
            repaired_all_shapes[original_idx] = updated_shape

        for track_id, template in templates.items():
            if track_id in present_tracks:
                continue
            history = track_history.get(track_id) or []
            prev_seen = history[-1] if history else None
            if prev_seen is None:
                continue
            if int(frame) - int(prev_seen[0]) > max(0, int(max_gap_frames)):
                continue
            next_seen = _find_next_seen(
                records_by_frame,
                current_frame=int(frame),
                track_id=track_id,
                max_gap_frames=max(0, int(max_gap_frames)),
            )
            filled = _interpolate_missing_shape(
                track_id=track_id,
                frame=int(frame),
                prev_seen=prev_seen,
                next_seen=next_seen,
                predicted_centroid=_predict_centroid(
                    history,
                    frame_number=int(frame),
                ),
                template=template,
            )
            repaired_all_shapes.append(filled)
            track_history.setdefault(track_id, []).append((int(frame), dict(filled)))
            missing += 1
            corrections.append(
                IdentityCorrection(
                    track_id=str(track_id),
                    frame_start=int(frame),
                    frame_end=int(frame),
                    observed_label="",
                    corrected_label=str(track_id),
                    rule_name="temporal_occlusion_gap",
                    rule_frame_start=int(frame),
                    rule_frame_end=int(frame),
                    observation_count=1,
                )
            )
            quality_events.append(
                _quality_event(
                    frame=int(frame),
                    issue="missing_id_filled",
                    track_id=str(track_id),
                    fill_mode=(
                        "interpolated_to_next_seen"
                        if next_seen is not None
                        else "predicted_or_carried"
                    ),
                )
            )

        if repaired_all_shapes != all_shapes:
            updated_record = dict(record)
            updated_record["shapes"] = repaired_all_shapes
            updated_payloads[json_path] = updated_record

    if report_path is None:
        report_path = resolved_dir / "temporal_identity_repair_report.json"
    resolved_report = Path(report_path).expanduser().resolve()
    stats = {
        "temporal_reference_instances": int(len(templates)),
        "id_switches_corrected": int(id_switches),
        "missing_shapes_filled": int(missing),
        "problematic_prediction_frames": int(
            len(
                {
                    int(event["frame"])
                    for event in quality_events
                    if event.get("issue")
                    not in {"id_switch_corrected", "missing_id_filled"}
                }
            )
        ),
    }
    updated_files = 0
    updated_shapes = 0
    if apply_changes:
        for path, payload in sorted(updated_payloads.items()):
            _write_json_atomic(path, payload)
        updated_files = len(updated_payloads)
        updated_shapes = len(corrections)

    _write_report(
        report_path=resolved_report,
        annotation_dir=resolved_dir,
        dry_run=not apply_changes,
        scanned_files=len(records),
        scanned_observations=scanned_observations,
        corrections=corrections,
        updated_files=updated_files,
        updated_shapes=updated_shapes,
        stats=stats,
        quality_events=quality_events,
    )
    return IdentityGovernorResult(
        annotation_dir=resolved_dir,
        dry_run=not apply_changes,
        scanned_files=len(records),
        scanned_observations=int(scanned_observations),
        proposed_corrections=tuple(corrections),
        updated_files=updated_files,
        updated_shapes=updated_shapes,
        report_path=resolved_report,
    )
