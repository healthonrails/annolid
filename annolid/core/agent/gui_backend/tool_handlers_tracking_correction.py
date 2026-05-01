from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from annolid.core.agent.tools.common import _resolve_read_path, _resolve_write_path
from annolid.core.agent.tools.function_sam3 import Sam3AgentVideoTrackTool
from annolid.core.agent.gui_backend.direct_commands import run_awaitable_sync


@dataclass(frozen=True)
class NdjsonLine:
    raw: str
    record: Optional[dict[str, Any]]


@dataclass(frozen=True)
class TrackTemplate:
    track_id: str
    label: str
    group_id: Any
    flags: dict[str, Any]


def _read_ndjson_lines(path: Path) -> tuple[list[NdjsonLine], int]:
    lines: list[NdjsonLine] = []
    invalid_lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            text = raw.rstrip("\n")
            if not text.strip():
                lines.append(NdjsonLine(raw=text, record=None))
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                invalid_lines += 1
                lines.append(NdjsonLine(raw=text, record=None))
                continue
            if isinstance(payload, dict):
                lines.append(NdjsonLine(raw=text, record=payload))
            else:
                invalid_lines += 1
                lines.append(NdjsonLine(raw=text, record=None))
    return lines, invalid_lines


def _records_from_lines(lines: Iterable[NdjsonLine]) -> list[dict[str, Any]]:
    return [line.record for line in lines if line.record is not None]


def _frame_key(record: Mapping[str, Any]) -> str:
    if "frame" in record:
        return "frame"
    if "frame_number" in record:
        return "frame_number"
    return "frame"


def _frame_of(record: Mapping[str, Any]) -> Optional[int]:
    for key in ("frame", "frame_number"):
        try:
            return int(record.get(key))
        except Exception:
            continue
    return None


def _has_shapes(record: Mapping[str, Any]) -> bool:
    shapes = record.get("shapes")
    return isinstance(shapes, list) and len(shapes) > 0


def _shape_track_id(shape: Mapping[str, Any]) -> str:
    for key in ("track_id", "tracking_id", "instance_id", "group_id"):
        value = shape.get(key)
        if value not in (None, ""):
            return str(value).strip()
    flags = shape.get("flags")
    if isinstance(flags, Mapping):
        for key in ("track_id", "tracking_id", "instance_id", "group_id"):
            value = flags.get(key)
            if value not in (None, ""):
                return str(value).strip()
    label = str(shape.get("label") or "").strip()
    return label


def _shape_centroid(shape: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None
    coords: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            coords.append((x, y))
    if not coords:
        return None
    return (
        float(sum(x for x, _ in coords) / len(coords)),
        float(sum(y for _, y in coords) / len(coords)),
    )


def _distance(
    a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]
) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _shape_source(shape: Mapping[str, Any]) -> str:
    source = shape.get("annotation_source")
    if source is not None:
        return str(source).strip().lower()
    description = shape.get("description")
    if isinstance(description, str) and description.strip():
        token = description.split(";", 1)[0].split(":", 1)[0].strip().lower()
        if token in {
            "grounding_sam",
            "cutie",
            "cutie_vos",
            "cutie_vos_segment",
            "sam3",
        }:
            return token
    flags = shape.get("flags")
    if isinstance(flags, Mapping):
        if flags.get("instance_id") is not None:
            return "propagated_instance"
        if flags.get("instance_label") is not None:
            return "propagated_instance"
    return ""


def _shape_identity(shape: Mapping[str, Any]) -> tuple[str, Any, str, str]:
    return (
        str(shape.get("label", "")),
        shape.get("group_id"),
        str(shape.get("shape_type", "")),
        _shape_source(shape),
    )


def _track_template_from_shape(shape: Mapping[str, Any]) -> TrackTemplate:
    return TrackTemplate(
        track_id=_shape_track_id(shape),
        label=str(shape.get("label") or ""),
        group_id=shape.get("group_id"),
        flags=dict(shape.get("flags") or {})
        if isinstance(shape.get("flags"), Mapping)
        else {},
    )


def _track_template_with_id(
    shape: Mapping[str, Any],
    track_id: str,
) -> TrackTemplate:
    template = _track_template_from_shape(shape)
    flags = dict(template.flags)
    flags.setdefault("track_id", track_id)
    return TrackTemplate(
        track_id=track_id,
        label=template.label,
        group_id=template.group_id,
        flags=flags,
    )


def _apply_track_template(
    shape: Mapping[str, Any],
    template: TrackTemplate,
    *,
    reason: str,
) -> dict[str, Any]:
    updated = dict(shape)
    if template.label:
        updated["label"] = template.label
    if template.group_id is not None:
        updated["group_id"] = template.group_id
    flags = (
        dict(updated.get("flags") or {})
        if isinstance(updated.get("flags"), Mapping)
        else {}
    )
    for key in ("track_id", "tracking_id", "instance_id", "group_id"):
        if key in template.flags:
            flags[key] = template.flags[key]
    if "track_id" not in flags and template.track_id:
        flags["track_id"] = template.track_id
    if "instance_id" not in flags and template.group_id is not None:
        flags["instance_id"] = template.group_id
    flags.setdefault("annolid_correction", reason)
    updated["flags"] = flags
    return updated


def _is_propagated_shape(shape: Mapping[str, Any]) -> bool:
    if _shape_source(shape):
        return True
    description = str(shape.get("description") or "").strip().lower()
    return description in {"grounding_sam", "cutie", "dinokpseg", "sam3"}


def _merge_shapes(
    target_shapes: Iterable[Any],
    source_shapes: Iterable[Any],
    *,
    replace_all_shapes: bool,
) -> list[Any]:
    source = [dict(shape) for shape in source_shapes if isinstance(shape, dict)]
    if replace_all_shapes:
        return source

    source_keys = {_shape_identity(shape) for shape in source}
    merged: list[Any] = []
    for shape in target_shapes:
        if not isinstance(shape, dict):
            merged.append(shape)
            continue
        if _shape_identity(shape) in source_keys:
            continue
        if _is_propagated_shape(shape) and source:
            continue
        merged.append(dict(shape))
    merged.extend(source)
    return merged


def _merge_frame_record(
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    replace_all_shapes: bool,
) -> dict[str, Any]:
    frame = _frame_of(target)
    target_frame_key = _frame_key(target)
    corrected = dict(target)
    source_shapes = (
        source.get("shapes") if isinstance(source.get("shapes"), list) else []
    )
    target_shapes = (
        target.get("shapes") if isinstance(target.get("shapes"), list) else []
    )
    corrected["shapes"] = _merge_shapes(
        target_shapes,
        source_shapes,
        replace_all_shapes=replace_all_shapes,
    )
    if frame is not None:
        corrected[target_frame_key] = int(frame)
    for key in ("imageHeight", "imageWidth", "imagePath"):
        if corrected.get(key) in (None, "") and source.get(key) not in (None, ""):
            corrected[key] = source.get(key)
    return corrected


def _source_record_for_new_frame(source: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(source)
    frame = _frame_of(record)
    if frame is not None and "frame" not in record:
        record["frame"] = int(frame)
    return record


def _serialize_record(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))


def _valid_frame_records(
    lines: Iterable[NdjsonLine],
) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line in lines:
        if line.record is None:
            continue
        frame = _frame_of(line.record)
        if frame is None:
            continue
        records.append((int(frame), dict(line.record)))
    return records


def _candidate_shapes(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    shapes = record.get("shapes")
    if not isinstance(shapes, list):
        return []
    return [dict(shape) for shape in shapes if isinstance(shape, dict)]


def _is_temporal_track_shape(shape: Mapping[str, Any]) -> bool:
    if _shape_centroid(shape) is None:
        return False
    shape_type = str(shape.get("shape_type") or "").strip().lower()
    if shape_type in {"polygon", "rectangle", "mask"}:
        return True
    if shape.get("group_id") not in (None, ""):
        return True
    flags = shape.get("flags")
    return isinstance(flags, Mapping) and any(
        flags.get(key) not in (None, "")
        for key in ("track_id", "tracking_id", "instance_id", "group_id")
    )


def _temporal_track_shapes(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        shape for shape in _candidate_shapes(record) if _is_temporal_track_shape(shape)
    ]


def _choose_reference_templates(
    records: list[tuple[int, dict[str, Any]]],
    *,
    start_frame: int,
    expected_instance_count: Optional[int],
) -> dict[str, TrackTemplate]:
    best: list[dict[str, Any]] = []
    for frame, record in sorted(records, key=lambda item: item[0]):
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
    templates: dict[str, TrackTemplate] = {}
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
    last_seen: Mapping[str, tuple[int, dict[str, Any]]],
    *,
    max_match_distance: float,
) -> dict[int, str]:
    available_tracks = list(last_seen.keys())
    costs: list[tuple[float, int, str]] = []
    for shape_idx, shape in enumerate(shapes):
        centroid = _shape_centroid(shape)
        for track_id in available_tracks:
            _, prev_shape = last_seen[track_id]
            dist = _distance(centroid, _shape_centroid(prev_shape))
            if dist is None or dist > max_match_distance:
                continue
            costs.append((float(dist), int(shape_idx), track_id))
    costs.sort(key=lambda item: item[0])
    matched_shapes: set[int] = set()
    matched_tracks: set[str] = set()
    assignments: dict[int, str] = {}
    for _, shape_idx, track_id in costs:
        if shape_idx in matched_shapes or track_id in matched_tracks:
            continue
        assignments[shape_idx] = track_id
        matched_shapes.add(shape_idx)
        matched_tracks.add(track_id)
    return assignments


def _translate_shape(
    shape: Mapping[str, Any],
    *,
    dx: float,
    dy: float,
    reason: str,
) -> dict[str, Any]:
    filled = dict(shape)
    points = []
    for point in shape.get("points") or []:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        try:
            points.append([float(point[0]) + float(dx), float(point[1]) + float(dy)])
        except Exception:
            continue
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
    next_seen: Optional[tuple[int, dict[str, Any]]],
    template: TrackTemplate,
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
) -> Optional[tuple[int, dict[str, Any]]]:
    for frame in range(current_frame + 1, current_frame + max_gap_frames + 1):
        record = records_by_frame.get(frame)
        if not record:
            continue
        for shape in _candidate_shapes(record):
            if _shape_track_id(shape) == track_id:
                return frame, shape
    return None


def _repair_temporal_records(
    records: list[tuple[int, dict[str, Any]]],
    *,
    start_frame: int,
    expected_instance_count: Optional[int],
    max_gap_frames: int,
    max_match_distance: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    templates = _choose_reference_templates(
        records,
        start_frame=start_frame,
        expected_instance_count=expected_instance_count,
    )
    if not templates:
        return {}, {
            "temporal_reference_instances": 0,
            "id_switches_corrected": 0,
            "missing_shapes_filled": 0,
        }
    records_by_frame = {frame: dict(record) for frame, record in records}
    last_seen: dict[str, tuple[int, dict[str, Any]]] = {}
    repaired: dict[int, dict[str, Any]] = {}
    id_switches = 0
    missing = 0
    template_ids = list(templates)

    for frame, record in sorted(records, key=lambda item: item[0]):
        if frame < start_frame:
            continue
        all_shapes = _candidate_shapes(record)
        temporal_pairs = [
            (idx, shape)
            for idx, shape in enumerate(all_shapes)
            if _is_temporal_track_shape(shape)
        ]
        shapes = [shape for _, shape in temporal_pairs]
        assignments = _match_shapes_to_tracks(
            shapes,
            last_seen,
            max_match_distance=max_match_distance,
        )
        initializing_frame = not last_seen
        updated_shapes: list[dict[str, Any]] = []
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
                shape = _apply_track_template(
                    shape,
                    templates[assigned_id],
                    reason="track_id_initialized"
                    if initializing_track_id
                    else "id_switch_corrected",
                )
                if not initializing_track_id:
                    id_switches += 1
                current_id = assigned_id
            if current_id in templates:
                present_tracks.add(current_id)
                last_seen[current_id] = (frame, dict(shape))
            updated_shapes.append(shape)

        repaired_all_shapes = list(all_shapes)
        for (original_idx, _), updated_shape in zip(temporal_pairs, updated_shapes):
            repaired_all_shapes[original_idx] = updated_shape

        for track_id, template in templates.items():
            if track_id in present_tracks:
                continue
            prev_seen = last_seen.get(track_id)
            if prev_seen is None:
                continue
            if frame - prev_seen[0] > max_gap_frames:
                continue
            next_seen = _find_next_seen(
                records_by_frame,
                current_frame=frame,
                track_id=track_id,
                max_gap_frames=max_gap_frames,
            )
            filled = _interpolate_missing_shape(
                track_id=track_id,
                frame=frame,
                prev_seen=prev_seen,
                next_seen=next_seen,
                template=template,
            )
            repaired_all_shapes.append(filled)
            last_seen[track_id] = (frame, dict(filled))
            missing += 1

        if repaired_all_shapes != all_shapes:
            updated_record = dict(record)
            updated_record["shapes"] = repaired_all_shapes
            repaired[frame] = updated_record

    return repaired, {
        "temporal_reference_instances": int(len(templates)),
        "id_switches_corrected": int(id_switches),
        "missing_shapes_filled": int(missing),
    }


def _lines_to_ndjson_lines(lines: Iterable[str]) -> list[NdjsonLine]:
    out: list[NdjsonLine] = []
    for line in lines:
        text = str(line)
        try:
            payload = json.loads(text) if text.strip() else None
        except json.JSONDecodeError:
            payload = None
        out.append(
            NdjsonLine(raw=text, record=payload if isinstance(payload, dict) else None)
        )
    return out


def _apply_temporal_repair(
    lines: list[str],
    *,
    start_frame: int,
    expected_instance_count: Optional[int],
    max_gap_frames: int,
    max_match_distance: float,
) -> tuple[list[str], dict[str, int]]:
    parsed_lines = _lines_to_ndjson_lines(lines)
    records = _valid_frame_records(parsed_lines)
    repaired, stats = _repair_temporal_records(
        records,
        start_frame=max(0, int(start_frame)),
        expected_instance_count=expected_instance_count,
        max_gap_frames=max(0, int(max_gap_frames)),
        max_match_distance=max(0.0, float(max_match_distance)),
    )
    if not repaired:
        return lines, stats
    out: list[str] = []
    for line in parsed_lines:
        frame = _frame_of(line.record) if line.record is not None else None
        if frame is not None and int(frame) in repaired:
            out.append(_serialize_record(repaired[int(frame)]))
        else:
            out.append(line.raw)
    return out, stats


def _discover_sam3_output_ndjson(output_dir: Path) -> Optional[Path]:
    candidates: list[Path] = []
    for path in output_dir.rglob("*.ndjson"):
        if path.name.endswith("_annotations.ndjson"):
            candidates.append(path)
    if not candidates:
        for path in output_dir.rglob("*.ndjson"):
            if "sam3" in path.name.lower():
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def _merge_records(
    *,
    target_lines: list[NdjsonLine],
    source_records: list[dict[str, Any]],
    replace_only_empty_shapes: bool,
    allow_append_new_frames: bool,
    replace_all_shapes: bool,
) -> tuple[list[str], dict[str, int]]:
    source_by_frame: dict[int, dict[str, Any]] = {}
    ignored_source_records = 0
    for row in source_records:
        frame = _frame_of(row)
        if frame is None:
            ignored_source_records += 1
            continue
        source_by_frame[frame] = dict(row)

    replaced = 0
    skipped_non_empty = 0
    out: list[str] = []
    seen_frames: set[int] = set()
    for line in target_lines:
        row = line.record
        if row is None:
            out.append(line.raw)
            continue
        frame = _frame_of(row)
        if frame is None:
            out.append(line.raw)
            continue
        seen_frames.add(frame)
        src = source_by_frame.get(frame)
        if src is None:
            out.append(line.raw)
            continue
        if replace_only_empty_shapes and _has_shapes(row):
            skipped_non_empty += 1
            out.append(line.raw)
            continue
        replaced += 1
        corrected = _merge_frame_record(
            row,
            src,
            replace_all_shapes=replace_all_shapes,
        )
        out.append(json.dumps(corrected, ensure_ascii=False, separators=(",", ":")))

    appended = 0
    skipped_new = 0
    if allow_append_new_frames:
        for frame in sorted(source_by_frame):
            if frame in seen_frames:
                continue
            out.append(
                json.dumps(
                    _source_record_for_new_frame(source_by_frame[frame]),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            appended += 1
    else:
        skipped_new = len(set(source_by_frame) - seen_frames)

    return out, {
        "candidate_source_frames": int(len(source_by_frame)),
        "ignored_source_records": int(ignored_source_records),
        "replaced_frames": int(replaced),
        "appended_frames": int(appended),
        "skipped_new_frames": int(skipped_new),
        "skipped_non_empty_frames": int(skipped_non_empty),
    }


def _write_ndjson_lines(path: Path, lines: Iterable[str]) -> None:
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            for line in lines:
                fh.write(str(line))
                fh.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _parse_tool_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or ""))
    except Exception:
        return {"ok": False, "error": "Invalid SAM3 payload"}
    if isinstance(parsed, dict):
        return parsed
    return {"ok": False, "error": "Invalid SAM3 payload"}


def _run_sam3_agent_tracking(
    *,
    video_path: str,
    agent_prompt: str,
    window_size: int,
    stride: Optional[int],
    allowed_dir: Path,
    allowed_read_roots: Optional[Sequence[str | Path]],
) -> Dict[str, Any]:
    tool = Sam3AgentVideoTrackTool(
        allowed_dir=allowed_dir,
        allowed_read_roots=allowed_read_roots,
    )
    raw = run_awaitable_sync(
        tool.execute(
            video_path=video_path,
            agent_prompt=agent_prompt,
            window_size=window_size,
            stride=stride,
        )
    )
    return _parse_tool_payload(raw)


def _resolve_source_ndjson_path(
    *,
    source_ndjson_path: str,
    sam3_payload: Mapping[str, Any],
    allowed_dir: Path,
    allowed_read_roots: Optional[Sequence[str | Path]],
) -> Optional[Path]:
    output_dir_text = str(sam3_payload.get("output_dir") or "").strip()
    if output_dir_text:
        discovered = _discover_sam3_output_ndjson(Path(output_dir_text))
        if discovered is not None:
            return _resolve_read_path(
                str(discovered),
                allowed_dir=allowed_dir,
                allowed_read_roots=allowed_read_roots,
            )

    if str(source_ndjson_path or "").strip():
        return _resolve_read_path(
            source_ndjson_path,
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
    return None


def correct_tracking_ndjson_tool(
    *,
    ndjson_path: str,
    source_ndjson_path: str = "",
    output_ndjson_path: str = "",
    video_path: str = "",
    agent_prompt: str = "",
    run_sam3_agent: bool = False,
    window_size: int = 5,
    stride: Optional[int] = None,
    replace_only_empty_shapes: bool = True,
    allow_append_new_frames: bool = False,
    replace_all_shapes: bool = False,
    temporal_repair: bool = False,
    start_frame: int = 0,
    expected_instance_count: Optional[int] = None,
    max_gap_frames: int = 5,
    max_match_distance: float = 80.0,
    allowed_dir: Path,
    allowed_read_roots: Optional[Sequence[str | Path]] = None,
) -> Dict[str, Any]:
    try:
        target_path = _resolve_read_path(
            ndjson_path,
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
        if not target_path.exists() or not target_path.is_file():
            return {"ok": False, "error": f"NDJSON file not found: {target_path}"}

        write_path = (
            _resolve_write_path(output_ndjson_path, allowed_dir=allowed_dir)
            if str(output_ndjson_path or "").strip()
            else _resolve_write_path(str(target_path), allowed_dir=allowed_dir)
        )

        source_path: Optional[Path] = None
        sam3_payload: Dict[str, Any] = {}
        if run_sam3_agent:
            if not str(video_path or "").strip():
                return {
                    "ok": False,
                    "error": "video_path is required when run_sam3_agent=true.",
                }
            if not str(agent_prompt or "").strip():
                return {
                    "ok": False,
                    "error": "agent_prompt is required when run_sam3_agent=true.",
                }
            sam3_payload = _run_sam3_agent_tracking(
                video_path=video_path,
                agent_prompt=agent_prompt,
                window_size=window_size,
                stride=stride,
                allowed_dir=allowed_dir,
                allowed_read_roots=allowed_read_roots,
            )
            if not bool(sam3_payload.get("ok", False)):
                return {
                    "ok": False,
                    "error": str(sam3_payload.get("error") or "SAM3 agent failed."),
                    "sam3": sam3_payload,
                }

        source_path = _resolve_source_ndjson_path(
            source_ndjson_path=source_ndjson_path,
            sam3_payload=sam3_payload,
            allowed_dir=allowed_dir,
            allowed_read_roots=allowed_read_roots,
        )
        if source_path is None:
            if bool(temporal_repair):
                source_path = target_path
            else:
                return {
                    "ok": False,
                    "error": (
                        "No source NDJSON was resolved. Provide source_ndjson_path or "
                        "run_sam3_agent=true with SAM3 output artifacts."
                    ),
                    "sam3": sam3_payload or None,
                }
        if not source_path.exists() or not source_path.is_file():
            return {"ok": False, "error": f"Source NDJSON not found: {source_path}"}

        target_lines, target_invalid = _read_ndjson_lines(target_path)
        source_lines, source_invalid = _read_ndjson_lines(source_path)
        target_records = _records_from_lines(target_lines)
        source_records = _records_from_lines(source_lines)
        merged_lines, stats = _merge_records(
            target_lines=target_lines,
            source_records=source_records,
            replace_only_empty_shapes=bool(replace_only_empty_shapes),
            allow_append_new_frames=bool(allow_append_new_frames),
            replace_all_shapes=bool(replace_all_shapes),
        )
        if int(stats["candidate_source_frames"]) == 0:
            return {
                "ok": False,
                "error": f"Source NDJSON contains no records with frame metadata: {source_path}",
                "target_ndjson_path": str(target_path),
                "source_ndjson_path": str(source_path),
                "source_records": int(len(source_records)),
                "source_invalid_lines": int(source_invalid),
            }
        temporal_stats = {
            "temporal_reference_instances": 0,
            "id_switches_corrected": 0,
            "missing_shapes_filled": 0,
        }
        if bool(temporal_repair):
            merged_lines, temporal_stats = _apply_temporal_repair(
                merged_lines,
                start_frame=int(start_frame),
                expected_instance_count=expected_instance_count,
                max_gap_frames=int(max_gap_frames),
                max_match_distance=float(max_match_distance),
            )

        write_path.parent.mkdir(parents=True, exist_ok=True)
        _write_ndjson_lines(write_path, merged_lines)

        return {
            "ok": True,
            "target_ndjson_path": str(target_path),
            "source_ndjson_path": str(source_path),
            "output_ndjson_path": str(write_path),
            "target_records": int(len(target_records)),
            "source_records": int(len(source_records)),
            "target_invalid_lines": int(target_invalid),
            "source_invalid_lines": int(source_invalid),
            **stats,
            **temporal_stats,
            "sam3": sam3_payload or None,
        }
    except PermissionError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
