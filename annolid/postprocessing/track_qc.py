"""Multi-animal track quality control and active-learning review queue scoring.

Evaluates LabelMe frame JSON directories (and AnnotationStore sidecars) or
Annolid tracking CSV exports (`*_tracking.csv` / `*_tracked.csv`) for:

- duplicate instance assignments in the same frame,
- missing-instance gaps and frame-level instance count mismatches,
- centroid velocity / teleportation spikes,
- abrupt mask or bounding-box area jumps (merges / fragmentations),
- close-contact spatial overlaps and trajectory-crossing identity swaps,
- anatomical keypoint bone-length outliers against an optional ``PoseSchema``,
- low prediction confidence or Cutie recovery markers.

Contiguous frame anomalies are aggregated into ranked ``ReviewBout`` items so
researchers and downstream GUI/CLI workflows can jump straight to the highest-risk
interaction frames first.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import io
from itertools import combinations
import math
import os
from pathlib import Path
import re
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except Exception:  # pragma: no cover - optional geometry acceleration
    _ShapelyPolygon = None  # type: ignore[assignment]

from annolid.annotation.pose_schema import PoseSchema
from annolid.core.behavior.spec import ProjectSchema
from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text
from annolid.utils.zone_shapes import is_zone_shape_payload

__all__ = [
    "TrackQCComputationError",
    "TrackQCIssue",
    "InstanceTrackSummary",
    "ReviewBout",
    "TrackQCReport",
    "evaluate_track_qc",
    "resolve_pose_schema",
    "format_track_qc_report_table",
    "write_track_qc_report_json",
    "write_track_qc_issues_csv",
    "write_review_queue_csv",
]

_SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1}
_RECOVERY_NOTE_TOKENS = (
    "recovered",
    "filled_from_previous",
    "fallback_previous_mask",
    "repaired_bad_shape",
    "frame_sized_artifact",
    "recovery_backoff",
)
_RECOMMENDED_ACTION_BY_CODE = {
    "identity_swap_suspect": "verify_identity_or_swap",
    "duplicate_instance": "deduplicate_or_relabel_instance",
    "spatial_overlap": "inspect_close_contact_masks",
    "anatomical_edge_outlier": "check_keypoint_outlier",
    "velocity_spike": "verify_jump_or_id_switch",
    "area_jump": "inspect_merged_or_fragmented_mask",
    "missing_instance_gap": "fill_or_repropagate_occlusion_gap",
    "count_mismatch": "verify_missing_or_extra_animal",
    "low_confidence": "confirm_low_confidence_prediction",
}


class TrackQCComputationError(Exception):
    """Raised when tracking QC input cannot be parsed or validated."""


@dataclass(frozen=True)
class TrackQCIssue:
    """Single tracking anomaly or quality-control finding."""

    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    frame_start: int
    frame_end: int
    anchor_frame: int
    instances: Tuple[str, ...] = ()
    score: float = 1.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "anchor_frame": int(self.anchor_frame),
            "instances": list(self.instances),
            "score": round(float(self.score), 6),
            "metrics": {
                str(k): round(float(v), 6)
                for k, v in sorted(self.metrics.items())
                if math.isfinite(float(v))
            },
        }


@dataclass(frozen=True)
class InstanceTrackSummary:
    """Per-instance coverage, motion, and confidence summary across the recording."""

    instance: str
    observed_frames: int
    missing_frames: int
    coverage_ratio: float
    duplicate_frames: int
    median_speed_px_per_frame: float
    max_speed_px_per_frame: float
    median_area_px: float
    mean_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "observed_frames": int(self.observed_frames),
            "missing_frames": int(self.missing_frames),
            "coverage_ratio": round(float(self.coverage_ratio), 6),
            "duplicate_frames": int(self.duplicate_frames),
            "median_speed_px_per_frame": round(
                float(self.median_speed_px_per_frame), 6
            ),
            "max_speed_px_per_frame": round(float(self.max_speed_px_per_frame), 6),
            "median_area_px": round(float(self.median_area_px), 6),
            "mean_score": (
                round(float(self.mean_score), 6)
                if self.mean_score is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ReviewBout:
    """Ranked contiguous window of anomalous frames for active-learning review."""

    rank: int
    frame_start: int
    frame_end: int
    anchor_frame: int
    risk_score: float
    max_severity: str
    instances: Tuple[str, ...]
    issue_codes: Tuple[str, ...]
    issue_count: int
    recommended_action: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "anchor_frame": int(self.anchor_frame),
            "risk_score": round(float(self.risk_score), 6),
            "max_severity": self.max_severity,
            "instances": list(self.instances),
            "issue_codes": list(self.issue_codes),
            "issue_count": int(self.issue_count),
            "recommended_action": self.recommended_action,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class TrackQCReport:
    """Complete tracking quality-control report and prioritized review queue."""

    source_path: str
    source_kind: str  # "labelme_dir" | "tracking_csv"
    frame_min: Optional[int]
    frame_max: Optional[int]
    total_frame_span: int
    observed_frame_count: int
    expected_instances: int
    fps: Optional[float]
    instances: Tuple[InstanceTrackSummary, ...]
    issues: Tuple[TrackQCIssue, ...]
    review_queue: Tuple[ReviewBout, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == "info")

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "frame_min": self.frame_min,
            "frame_max": self.frame_max,
            "total_frame_span": int(self.total_frame_span),
            "observed_frame_count": int(self.observed_frame_count),
            "expected_instances": int(self.expected_instances),
            "fps": round(float(self.fps), 6) if self.fps is not None else None,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "has_errors": self.has_errors,
            "instances": [item.to_dict() for item in self.instances],
            "issues": [item.to_dict() for item in self.issues],
            "review_queue": [item.to_dict() for item in self.review_queue],
        }


@dataclass(frozen=True)
class _InstanceObservation:
    frame: int
    instance: str
    centroid: Tuple[float, float]
    area: float
    bbox: Optional[Tuple[float, float, float, float]]
    polygon: Tuple[Tuple[float, float], ...]
    score: Optional[float]
    recovery_note: str = ""


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _frame_number_from_filename(name: str) -> Optional[int]:
    match = re.search(r"(\d+)(?=\.json$)", name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def resolve_pose_schema(
    *,
    pose_schema: Optional[PoseSchema] = None,
    pose_schema_path: Optional[Path | str] = None,
    project_schema: Optional[ProjectSchema] = None,
    project_schema_dir: Optional[Path] = None,
) -> Optional[PoseSchema]:
    """Resolve an optional ``PoseSchema`` from explicit path or ``ProjectSchema``."""
    if pose_schema is not None:
        return pose_schema
    if pose_schema_path is not None:
        resolved = Path(pose_schema_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Pose schema file not found: {resolved}")
        return PoseSchema.load(resolved)
    if project_schema is not None:
        if (
            isinstance(project_schema.pose_schema, Mapping)
            and project_schema.pose_schema
        ):
            return PoseSchema.from_dict(dict(project_schema.pose_schema))
        if project_schema.pose_schema_path:
            candidate = Path(project_schema.pose_schema_path).expanduser()
            if not candidate.is_absolute() and project_schema_dir is not None:
                candidate = project_schema_dir / candidate
            if candidate.exists():
                return PoseSchema.load(candidate)
    return None


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    n_pts = len(points)
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % n_pts]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _extract_shape_geometry(
    shape: Mapping[str, Any],
) -> Optional[
    Tuple[
        Tuple[float, float],
        float,
        Optional[Tuple[float, float, float, float]],
        Tuple[Tuple[float, float], ...],
    ]
]:
    raw_points = shape.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        return None
    coords: List[Tuple[float, float]] = []
    for pt in raw_points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            px = _safe_float(pt[0])
            py = _safe_float(pt[1])
            if px is not None and py is not None:
                coords.append((px, py))
    if not coords:
        return None

    shape_type = str(shape.get("shape_type") or "").strip().lower()
    if shape_type == "rectangle" and len(coords) >= 2:
        (x1, y1), (x2, y2) = coords[0], coords[1]
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        poly = ((left, top), (right, top), (right, bottom), (left, bottom))
        area = max(0.0, (right - left) * (bottom - top))
        cx = 0.5 * (left + right)
        cy = 0.5 * (top + bottom)
        return (cx, cy), area, (left, top, right, bottom), poly

    if shape_type == "circle" and len(coords) >= 2:
        (cx, cy), (rx, ry) = coords[0], coords[1]
        radius = math.hypot(rx - cx, ry - cy)
        area = math.pi * radius * radius
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        return (cx, cy), area, bbox, ()

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    if len(coords) >= 3:
        area = _polygon_area(coords)
        bbox: Optional[Tuple[float, float, float, float]] = (left, top, right, bottom)
        return (cx, cy), area, bbox, tuple(coords)

    bbox = (left, top, right, bottom) if (right > left or bottom > top) else None
    area = max(0.0, (right - left) * (bottom - top))
    return (cx, cy), area, bbox, ()


def _shape_score(shape: Mapping[str, Any]) -> Optional[float]:
    for key in ("score", "confidence", "class_score"):
        val = _safe_float(shape.get(key))
        if val is not None:
            return val
    for container_key in ("other_data", "otherData", "flags"):
        container = shape.get(container_key)
        if isinstance(container, Mapping):
            for key in ("score", "confidence", "class_score"):
                val = _safe_float(container.get(key))
                if val is not None:
                    return val
    desc = str(shape.get("description") or "")
    match = re.search(
        r"(?:score|conf(?:idence)?)\s*[:=]\s*([0-9]*\.?[0-9]+)", desc, re.I
    )
    if match:
        return _safe_float(match.group(1))
    return None


def _shape_recovery_note(shape: Mapping[str, Any]) -> str:
    texts: List[str] = [str(shape.get("description") or "")]
    for key in ("flags", "other_data", "otherData"):
        payload = shape.get(key)
        if isinstance(payload, Mapping):
            for sub_val in payload.values():
                if isinstance(sub_val, str):
                    texts.append(sub_val)
            for flag_key, flag_val in payload.items():
                if flag_val is True:
                    texts.append(str(flag_key))
    joined = " ".join(t for t in texts if t).lower()
    for token in _RECOVERY_NOTE_TOKENS:
        if token in joined:
            return token
    return ""


def _resolve_instance_label(shape: Mapping[str, Any]) -> str:
    for key in ("instance_label", "instance_name"):
        val = str(shape.get(key) or "").strip()
        if val:
            return val
    for container_key in ("flags", "other_data", "otherData"):
        container = shape.get(container_key)
        if isinstance(container, Mapping):
            for key in ("instance_label", "instance_name"):
                val = str(container.get(key) or "").strip()
                if val:
                    return val
    raw_label = str(shape.get("label") or "").strip()
    if ":" in raw_label:
        prefix = raw_label.split(":", 1)[0].strip()
        if prefix:
            return prefix
    group_id = (
        shape.get("track_id") or shape.get("tracking_id") or shape.get("group_id")
    )
    if raw_label and group_id is not None and str(group_id).strip():
        gid = str(group_id).strip()
        if not raw_label.endswith(f"_{gid}") and raw_label != gid:
            return f"{raw_label}_{gid}"
    if raw_label:
        return raw_label
    if group_id is not None and str(group_id).strip():
        return str(group_id).strip()
    return ""


def _parse_keypoint_identity(
    raw_label: str,
    *,
    explicit_instance: str = "",
    explicit_keypoint: str = "",
    pose_schema: Optional[PoseSchema] = None,
) -> Optional[Tuple[str, str]]:
    """Map a point label onto ``(instance, keypoint_name)``."""
    inst = explicit_instance.strip()
    kpt = explicit_keypoint.strip()
    label = raw_label.strip()
    if inst and kpt:
        return inst, kpt
    if ":" in label:
        left, right = label.split(":", 1)
        if left.strip() and right.strip():
            return left.strip(), right.strip()

    if pose_schema is not None and pose_schema.keypoints:
        sep = pose_schema.instance_separator or "_"
        # Prefer longest keypoint match first (e.g. "tail_base" before "base").
        sorted_kpts = sorted(pose_schema.keypoints, key=len, reverse=True)
        if pose_schema.instances:
            for candidate_inst in sorted(pose_schema.instances, key=len, reverse=True):
                prefix = pose_schema.instance_prefix(candidate_inst)
                if prefix and label.startswith(prefix):
                    remainder = label[len(prefix) :]
                    if remainder in pose_schema.keypoints:
                        return candidate_inst, remainder
        for candidate_kpt in sorted_kpts:
            if label == candidate_kpt:
                return (inst or "__single__"), candidate_kpt
            token = f"{sep}{candidate_kpt}"
            if label.endswith(token) and len(label) > len(token):
                inferred_inst = label[: -len(token)].strip()
                if inferred_inst:
                    return inferred_inst, candidate_kpt
        return None

    if inst and label and label != inst:
        return inst, label
    return None


def _load_labelme_dir_observations(
    annotation_dir: Path,
    *,
    pose_schema: Optional[PoseSchema],
) -> Tuple[
    List[_InstanceObservation],
    Dict[int, Dict[str, Dict[str, Tuple[float, float]]]],
    List[int],
]:
    json_names = {
        name for name in os.listdir(annotation_dir) if name.lower().endswith(".json")
    }
    store = AnnotationStore.for_frame_path(
        annotation_dir / f"{annotation_dir.name}_000000000.json"
    )
    if store.store_path.exists():
        for frame_idx in store.iter_frames():
            json_names.add(f"{annotation_dir.name}_{int(frame_idx):09d}.json")

    observations: List[_InstanceObservation] = []
    keypoints_by_frame: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    seen_frames: List[int] = []

    for name in sorted(json_names):
        frame_num = _frame_number_from_filename(name)
        if frame_num is None:
            continue
        frame_path = annotation_dir / name
        try:
            payload = load_labelme_json(frame_path)
        except (FileNotFoundError, AnnotationStoreError, ValueError):
            continue
        if not isinstance(payload, Mapping):
            continue
        shapes = payload.get("shapes")
        if not isinstance(shapes, list):
            continue
        seen_frames.append(frame_num)

        for shape in shapes:
            if not isinstance(shape, Mapping) or is_zone_shape_payload(shape):
                continue
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type == "point":
                pts = shape.get("points")
                if not isinstance(pts, list) or not pts:
                    continue
                first = pts[0]
                if not isinstance(first, (list, tuple)) or len(first) < 2:
                    continue
                px, py = _safe_float(first[0]), _safe_float(first[1])
                if px is None or py is None:
                    continue
                explicit_inst = ""
                explicit_kpt = str(shape.get("display_label") or "").strip()
                for key in ("instance_label", "instance_name"):
                    if str(shape.get(key) or "").strip():
                        explicit_inst = str(shape.get(key)).strip()
                        break
                for container_key in ("flags", "other_data", "otherData"):
                    container = shape.get(container_key)
                    if isinstance(container, Mapping):
                        if not explicit_inst:
                            explicit_inst = str(
                                container.get("instance_label")
                                or container.get("instance_name")
                                or ""
                            ).strip()
                        if not explicit_kpt:
                            explicit_kpt = str(
                                container.get("display_label")
                                or container.get("keypoint_label")
                                or ""
                            ).strip()
                parsed_kp = _parse_keypoint_identity(
                    str(shape.get("label") or ""),
                    explicit_instance=explicit_inst,
                    explicit_keypoint=explicit_kpt,
                    pose_schema=pose_schema,
                )
                if parsed_kp is not None:
                    kp_inst, kp_name = parsed_kp
                    keypoints_by_frame[frame_num][kp_inst][kp_name] = (px, py)
                continue

            geom = _extract_shape_geometry(shape)
            if geom is None:
                continue
            instance = _resolve_instance_label(shape)
            if not instance:
                continue
            centroid, area, bbox, poly = geom
            observations.append(
                _InstanceObservation(
                    frame=frame_num,
                    instance=instance,
                    centroid=centroid,
                    area=area,
                    bbox=bbox,
                    polygon=poly,
                    score=_shape_score(shape),
                    recovery_note=_shape_recovery_note(shape),
                )
            )

    return observations, keypoints_by_frame, sorted(set(seen_frames))


def _resolve_csv_column(
    fieldnames: Sequence[str], candidates: Sequence[str]
) -> Optional[str]:
    lower_map = {col.strip().lower(): col for col in fieldnames if col}
    for cand in candidates:
        if cand in fieldnames:
            return cand
        hit = lower_map.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _load_tracking_csv_observations(
    csv_path: Path,
    *,
    pose_schema: Optional[PoseSchema],
) -> Tuple[
    List[_InstanceObservation],
    Dict[int, Dict[str, Dict[str, Tuple[float, float]]]],
    List[int],
]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise TrackQCComputationError(
                f"Tracking CSV '{csv_path}' is empty or missing a header row."
            )
        fields = list(reader.fieldnames)
        frame_col = _resolve_csv_column(
            fields, ("frame_number", "frame", "frame_idx", "frame_index")
        )
        inst_col = _resolve_csv_column(
            fields, ("instance_name", "instance_label", "instance", "track_id", "label")
        )
        cx_col = _resolve_csv_column(fields, ("cx", "cx_tracked", "cx_tracking", "x"))
        cy_col = _resolve_csv_column(fields, ("cy", "cy_tracked", "cy_tracking", "y"))
        if frame_col is None or inst_col is None:
            raise TrackQCComputationError(
                f"Tracking CSV '{csv_path}' must contain frame_number and instance_name columns."
            )

        x1_col = _resolve_csv_column(fields, ("x1", "xmin", "left"))
        y1_col = _resolve_csv_column(fields, ("y1", "ymin", "top"))
        x2_col = _resolve_csv_column(fields, ("x2", "xmax", "right"))
        y2_col = _resolve_csv_column(fields, ("y2", "ymax", "bottom"))
        score_col = _resolve_csv_column(fields, ("class_score", "score", "confidence"))
        area_col = _resolve_csv_column(fields, ("area", "mask_area"))

        observations: List[_InstanceObservation] = []
        keypoints_by_frame: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        seen_frames: List[int] = []

        for row in reader:
            frame_num = _safe_int(row.get(frame_col))
            if frame_num is None:
                continue
            raw_inst = str(row.get(inst_col) or "").strip()
            if not raw_inst:
                continue

            x1 = _safe_float(row.get(x1_col)) if x1_col else None
            y1 = _safe_float(row.get(y1_col)) if y1_col else None
            x2 = _safe_float(row.get(x2_col)) if x2_col else None
            y2 = _safe_float(row.get(y2_col)) if y2_col else None
            cx = _safe_float(row.get(cx_col)) if cx_col else None
            cy = _safe_float(row.get(cy_col)) if cy_col else None
            if cx is None and x1 is not None and x2 is not None:
                cx = 0.5 * (x1 + x2)
            if cy is None and y1 is not None and y2 is not None:
                cy = 0.5 * (y1 + y2)
            if cx is None or cy is None:
                continue

            seen_frames.append(frame_num)

            is_point_like_bbox = (
                x1 is not None
                and x2 is not None
                and y1 is not None
                and y2 is not None
                and abs(x2 - x1) <= 1e-6
                and abs(y2 - y1) <= 1e-6
            )
            if pose_schema is not None and (x1_col is None or is_point_like_bbox):
                parsed_kp = _parse_keypoint_identity(raw_inst, pose_schema=pose_schema)
                if parsed_kp is not None:
                    kp_inst, kp_name = parsed_kp
                    keypoints_by_frame[frame_num][kp_inst][kp_name] = (cx, cy)
                    continue

            bbox: Optional[Tuple[float, float, float, float]] = None
            area = _safe_float(row.get(area_col)) if area_col else None
            if (
                x1 is not None
                and y1 is not None
                and x2 is not None
                and y2 is not None
                and (abs(x2 - x1) > 1e-6 or abs(y2 - y1) > 1e-6)
            ):
                left, right = min(x1, x2), max(x1, x2)
                top, bottom = min(y1, y2), max(y1, y2)
                bbox = (left, top, right, bottom)
                if area is None:
                    area = max(0.0, (right - left) * (bottom - top))
            if area is None:
                area = 0.0

            score = _safe_float(row.get(score_col)) if score_col else None
            observations.append(
                _InstanceObservation(
                    frame=frame_num,
                    instance=raw_inst,
                    centroid=(cx, cy),
                    area=float(area),
                    bbox=bbox,
                    polygon=(),
                    score=score,
                    recovery_note="",
                )
            )

    return observations, keypoints_by_frame, sorted(set(seen_frames))


def _bbox_iou(
    b1: Optional[Tuple[float, float, float, float]],
    b2: Optional[Tuple[float, float, float, float]],
) -> float:
    if b1 is None or b2 is None:
        return 0.0
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def _observation_iou(a: _InstanceObservation, b: _InstanceObservation) -> float:
    box_iou = _bbox_iou(a.bbox, b.bbox)
    if box_iou <= 0.0:
        return 0.0
    if _ShapelyPolygon is not None and len(a.polygon) >= 3 and len(b.polygon) >= 3:
        try:
            p1 = _ShapelyPolygon(a.polygon)
            p2 = _ShapelyPolygon(b.polygon)
            if not p1.is_valid:
                p1 = p1.buffer(0)
            if not p2.is_valid:
                p2 = p2.buffer(0)
            union_area = float(p1.union(p2).area)
            if union_area > 0.0:
                return float(p1.intersection(p2).area) / union_area
        except Exception:
            return box_iou
    return box_iou


def _robust_speed_threshold(
    speeds: Sequence[float],
    *,
    explicit_max_speed: Optional[float],
    mad_z_threshold: float,
    min_jump_px: float,
) -> float:
    if explicit_max_speed is not None and explicit_max_speed > 0:
        return float(explicit_max_speed)
    if not speeds:
        return float(min_jump_px)
    med = float(median(speeds))
    abs_devs = [abs(val - med) for val in speeds]
    mad = float(median(abs_devs)) if abs_devs else 0.0
    robust_sigma = 1.4826 * mad
    adaptive = med + float(mad_z_threshold) * robust_sigma
    # Guard stationary tracks where MAD == 0 with a physical floor and 4x median floor.
    return max(float(min_jump_px), adaptive, med * 3.5)


def _group_contiguous_frames(frames: Sequence[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    ordered = sorted(set(int(f) for f in frames))
    spans: List[Tuple[int, int]] = []
    start = prev = ordered[0]
    for f in ordered[1:]:
        if f == prev + 1:
            prev = f
        else:
            spans.append((start, prev))
            start = prev = f
    spans.append((start, prev))
    return spans


def _build_review_queue(
    issues: Sequence[TrackQCIssue],
    *,
    merge_gap_frames: int,
    top_k: Optional[int],
) -> Tuple[ReviewBout, ...]:
    if not issues:
        return ()

    sorted_issues = sorted(
        issues,
        key=lambda item: (item.frame_start, item.frame_end, -item.score),
    )
    clusters: List[List[TrackQCIssue]] = []
    current_cluster: List[TrackQCIssue] = [sorted_issues[0]]
    current_end = sorted_issues[0].frame_end

    for issue in sorted_issues[1:]:
        if issue.frame_start <= current_end + max(0, int(merge_gap_frames)):
            current_cluster.append(issue)
            current_end = max(current_end, issue.frame_end)
        else:
            clusters.append(current_cluster)
            current_cluster = [issue]
            current_end = issue.frame_end
    clusters.append(current_cluster)

    unranked: List[Tuple[float, int, Dict[str, Any]]] = []
    for cluster in clusters:
        f_start = min(item.frame_start for item in cluster)
        f_end = max(item.frame_end for item in cluster)

        # Pick the exact anchor frame inside the bout with highest concentrated risk.
        anchor_weights: Dict[int, float] = defaultdict(float)
        for item in cluster:
            anchor_weights[item.anchor_frame] += float(item.score) * (
                1.0 + 0.5 * _SEVERITY_RANK.get(item.severity, 1)
            )
        anchor_frame = max(
            sorted(anchor_weights),
            key=lambda frame_idx: anchor_weights[frame_idx],
        )

        max_sev = max(
            (item.severity for item in cluster),
            key=lambda sev: _SEVERITY_RANK.get(sev, 0),
        )
        # Rank bouts by peak issue severity + log-damped cumulative evidence so a
        # 200-frame missing-gap warning doesn't drown out a sharp 2-frame ID swap.
        peak_score = max(float(item.score) for item in cluster)
        extra_score = sum(float(item.score) for item in cluster) - peak_score
        risk_score = peak_score + 0.35 * math.log1p(max(0.0, extra_score))

        # Order issue codes by their highest contribution in the bout.
        code_peak: Dict[str, float] = defaultdict(float)
        for item in cluster:
            code_peak[item.code] = max(code_peak[item.code], float(item.score))
        ordered_codes = tuple(
            sorted(
                code_peak,
                key=lambda c: (-code_peak[c], c),
            )
        )
        primary_code = ordered_codes[0]
        recommended_action = _RECOMMENDED_ACTION_BY_CODE.get(
            primary_code, "inspect_frame_window"
        )
        instances = tuple(
            sorted({inst for item in cluster for inst in item.instances if inst})
        )
        primary_issue = max(cluster, key=lambda item: (item.score, -item.anchor_frame))
        unranked.append(
            (
                risk_score,
                f_start,
                {
                    "frame_start": f_start,
                    "frame_end": f_end,
                    "anchor_frame": anchor_frame,
                    "risk_score": risk_score,
                    "max_severity": max_sev,
                    "instances": instances,
                    "issue_codes": ordered_codes,
                    "issue_count": len(cluster),
                    "recommended_action": recommended_action,
                    "summary": primary_issue.message,
                },
            )
        )

    unranked.sort(key=lambda entry: (-entry[0], entry[1]))
    if top_k is not None and int(top_k) > 0:
        unranked = unranked[: int(top_k)]

    bouts: List[ReviewBout] = []
    for idx, (_score, _start, data) in enumerate(unranked, start=1):
        bouts.append(
            ReviewBout(
                rank=idx,
                frame_start=int(data["frame_start"]),
                frame_end=int(data["frame_end"]),
                anchor_frame=int(data["anchor_frame"]),
                risk_score=float(data["risk_score"]),
                max_severity=str(data["max_severity"]),
                instances=data["instances"],
                issue_codes=data["issue_codes"],
                issue_count=int(data["issue_count"]),
                recommended_action=str(data["recommended_action"]),
                summary=str(data["summary"]),
            )
        )
    return tuple(bouts)


def evaluate_track_qc(
    input_path: Path | str,
    *,
    pose_schema: Optional[PoseSchema] = None,
    project_schema: Optional[ProjectSchema] = None,
    expected_instances: Optional[int] = None,
    fps: Optional[float] = None,
    max_speed_px_per_frame: Optional[float] = None,
    speed_mad_z: float = 6.0,
    min_jump_px: float = 25.0,
    max_area_ratio: float = 2.5,
    min_area_px: float = 16.0,
    overlap_iou_threshold: float = 0.25,
    close_contact_px: float = 80.0,
    swap_margin_px: float = 10.0,
    max_swap_lookahead_frames: int = 10,
    max_edge_stretch_ratio: float = 2.5,
    min_edge_baseline_px: float = 5.0,
    min_score: float = 0.35,
    bout_merge_gap_frames: int = 5,
    top_k_review: Optional[int] = 25,
) -> TrackQCReport:
    """Evaluate tracking quality and build a prioritized active-learning review queue."""
    resolved = Path(input_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Tracking input not found: {resolved}")

    effective_pose_schema = resolve_pose_schema(
        pose_schema=pose_schema,
        project_schema=project_schema,
    )

    if resolved.is_dir():
        source_kind = "labelme_dir"
        observations, keypoints_by_frame, seen_frames = _load_labelme_dir_observations(
            resolved,
            pose_schema=effective_pose_schema,
        )
    elif resolved.is_file():
        source_kind = "tracking_csv"
        observations, keypoints_by_frame, seen_frames = _load_tracking_csv_observations(
            resolved,
            pose_schema=effective_pose_schema,
        )
    else:
        raise TrackQCComputationError(f"Unsupported tracking input path: {resolved}")

    all_frames = sorted(
        set(seen_frames)
        | {obs.frame for obs in observations}
        | set(keypoints_by_frame.keys())
    )
    if not all_frames:
        return TrackQCReport(
            source_path=str(resolved),
            source_kind=source_kind,
            frame_min=None,
            frame_max=None,
            total_frame_span=0,
            observed_frame_count=0,
            expected_instances=int(expected_instances or 0),
            fps=fps,
            instances=(),
            issues=(),
            review_queue=(),
        )

    frame_min = int(all_frames[0])
    frame_max = int(all_frames[-1])
    total_span = frame_max - frame_min + 1

    obs_by_frame: Dict[int, List[_InstanceObservation]] = defaultdict(list)
    for obs in observations:
        obs_by_frame[obs.frame].append(obs)

    known_instances = sorted({obs.instance for obs in observations})
    inferred_expected = (
        int(expected_instances)
        if expected_instances is not None and int(expected_instances) > 0
        else len(known_instances)
    )

    issues: List[TrackQCIssue] = []
    duplicate_frames_by_inst: Counter[str] = Counter()
    canonical_by_inst: Dict[str, List[_InstanceObservation]] = defaultdict(list)

    # 1. Per-frame duplicate instances, low confidence, recovery flags, and spatial overlaps.
    for frame in sorted(obs_by_frame):
        frame_obs = obs_by_frame[frame]
        by_inst: Dict[str, List[_InstanceObservation]] = defaultdict(list)
        for obs in frame_obs:
            by_inst[obs.instance].append(obs)

        canonical_for_frame: List[_InstanceObservation] = []
        for inst_name, items in sorted(by_inst.items()):
            if len(items) > 1:
                duplicate_frames_by_inst[inst_name] += 1
                issues.append(
                    TrackQCIssue(
                        severity="error",
                        code="duplicate_instance",
                        message=(
                            f"Instance '{inst_name}' appears {len(items)} times on frame {frame}."
                        ),
                        frame_start=frame,
                        frame_end=frame,
                        anchor_frame=frame,
                        instances=(inst_name,),
                        score=8.0 + float(len(items) - 2),
                        metrics={"duplicate_count": float(len(items))},
                    )
                )
            # Pick highest-area / highest-score representative for trajectory continuity checks.
            chosen = max(
                items,
                key=lambda o: (
                    o.score if o.score is not None else 1.0,
                    o.area,
                ),
            )
            canonical_by_inst[inst_name].append(chosen)
            canonical_for_frame.append(chosen)

            for obs in items:
                if obs.score is not None and obs.score < float(min_score):
                    issues.append(
                        TrackQCIssue(
                            severity="warning",
                            code="low_confidence",
                            message=(
                                f"Low confidence ({obs.score:.3f} < {float(min_score):.3f}) "
                                f"for '{inst_name}' on frame {frame}."
                            ),
                            frame_start=frame,
                            frame_end=frame,
                            anchor_frame=frame,
                            instances=(inst_name,),
                            score=2.0 + 3.0 * max(0.0, float(min_score) - obs.score),
                            metrics={
                                "score": float(obs.score),
                                "min_score": float(min_score),
                            },
                        )
                    )
                elif obs.recovery_note:
                    issues.append(
                        TrackQCIssue(
                            severity="info",
                            code="low_confidence",
                            message=(
                                f"Tracker recovery flag '{obs.recovery_note}' "
                                f"for '{inst_name}' on frame {frame}."
                            ),
                            frame_start=frame,
                            frame_end=frame,
                            anchor_frame=frame,
                            instances=(inst_name,),
                            score=1.5,
                            metrics={},
                        )
                    )

        # Pairwise spatial IoU overlap in the same frame.
        if len(canonical_for_frame) >= 2 and float(overlap_iou_threshold) > 0:
            for left_obs, right_obs in combinations(canonical_for_frame, 2):
                iou = _observation_iou(left_obs, right_obs)
                if iou >= float(overlap_iou_threshold):
                    pair = tuple(sorted((left_obs.instance, right_obs.instance)))
                    issues.append(
                        TrackQCIssue(
                            severity="warning",
                            code="spatial_overlap",
                            message=(
                                f"Spatial overlap (IoU={iou:.2f}) between "
                                f"'{pair[0]}' and '{pair[1]}' on frame {frame}."
                            ),
                            frame_start=frame,
                            frame_end=frame,
                            anchor_frame=frame,
                            instances=pair,
                            score=3.5 + 4.0 * iou,
                            metrics={"iou": float(iou)},
                        )
                    )

    # Extra-animal count mismatch (when expected_instances is explicitly configured).
    if expected_instances is not None and int(expected_instances) > 0:
        exp = int(expected_instances)
        extra_frames = [
            frame
            for frame, items in sorted(obs_by_frame.items())
            if len({o.instance for o in items}) > exp
        ]
        for seg_start, seg_end in _group_contiguous_frames(extra_frames):
            peak_count = max(
                len({o.instance for o in obs_by_frame[f]})
                for f in range(seg_start, seg_end + 1)
                if f in obs_by_frame
            )
            involved = tuple(
                sorted(
                    {
                        o.instance
                        for f in range(seg_start, seg_end + 1)
                        for o in obs_by_frame.get(f, ())
                    }
                )
            )
            issues.append(
                TrackQCIssue(
                    severity="warning",
                    code="count_mismatch",
                    message=(
                        f"Observed up to {peak_count} distinct instances vs expected {exp} "
                        f"across frames {seg_start}..{seg_end}."
                    ),
                    frame_start=seg_start,
                    frame_end=seg_end,
                    anchor_frame=seg_start,
                    instances=involved,
                    score=4.5,
                    metrics={
                        "observed_instances": float(peak_count),
                        "expected_instances": float(exp),
                    },
                )
            )

    # 2. Per-instance trajectory checks: missing gaps, velocity spikes, area jumps, summaries.
    instance_summaries: List[InstanceTrackSummary] = []
    full_frame_set = set(range(frame_min, frame_max + 1))

    for inst_name in known_instances:
        traj = sorted(canonical_by_inst[inst_name], key=lambda o: o.frame)
        present_frames = {o.frame for o in traj}
        missing_frames_sorted = sorted(full_frame_set - present_frames)
        for gap_start, gap_end in _group_contiguous_frames(missing_frames_sorted):
            gap_len = gap_end - gap_start + 1
            issues.append(
                TrackQCIssue(
                    severity="warning",
                    code="missing_instance_gap",
                    message=(
                        f"Instance '{inst_name}' missing for {gap_len} frame(s) "
                        f"({gap_start}..{gap_end})."
                    ),
                    frame_start=gap_start,
                    frame_end=gap_end,
                    anchor_frame=gap_start,
                    instances=(inst_name,),
                    score=2.5 + min(4.0, math.log1p(gap_len)),
                    metrics={"gap_frames": float(gap_len)},
                )
            )

        speeds: List[float] = []
        step_records: List[
            Tuple[_InstanceObservation, _InstanceObservation, float, int]
        ] = []
        for prev_obs, curr_obs in zip(traj, traj[1:]):
            dt_frames = max(1, curr_obs.frame - prev_obs.frame)
            dist = math.hypot(
                curr_obs.centroid[0] - prev_obs.centroid[0],
                curr_obs.centroid[1] - prev_obs.centroid[1],
            )
            speed = dist / float(dt_frames)
            speeds.append(speed)
            step_records.append((prev_obs, curr_obs, speed, dt_frames))

        speed_threshold = _robust_speed_threshold(
            speeds,
            explicit_max_speed=max_speed_px_per_frame,
            mad_z_threshold=speed_mad_z,
            min_jump_px=min_jump_px,
        )
        med_speed = float(median(speeds)) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0

        for prev_obs, curr_obs, speed, dt_frames in step_records:
            if speed > speed_threshold:
                ratio = speed / max(1e-6, speed_threshold)
                sev = "error" if ratio >= 2.0 else "warning"
                issues.append(
                    TrackQCIssue(
                        severity=sev,
                        code="velocity_spike",
                        message=(
                            f"Velocity spike for '{inst_name}' at frame {curr_obs.frame}: "
                            f"{speed:.1f} px/frame (threshold {speed_threshold:.1f} px/frame)."
                        ),
                        frame_start=prev_obs.frame,
                        frame_end=curr_obs.frame,
                        anchor_frame=curr_obs.frame,
                        instances=(inst_name,),
                        score=(6.5 if sev == "error" else 4.0) + min(5.0, ratio),
                        metrics={
                            "speed_px_per_frame": float(speed),
                            "threshold_px_per_frame": float(speed_threshold),
                            "median_speed_px_per_frame": med_speed,
                            "delta_frames": float(dt_frames),
                        },
                    )
                )

        valid_areas = [o.area for o in traj if o.area >= float(min_area_px)]
        med_area = float(median(valid_areas)) if valid_areas else 0.0
        if valid_areas and float(max_area_ratio) > 1.0:
            ratio_limit = float(max_area_ratio)
            for prev_obs, curr_obs in zip(traj, traj[1:]):
                if prev_obs.area < float(min_area_px) or curr_obs.area < float(
                    min_area_px
                ):
                    continue
                step_ratio = max(
                    curr_obs.area / prev_obs.area,
                    prev_obs.area / curr_obs.area,
                )
                baseline_ratio = (
                    max(curr_obs.area / med_area, med_area / curr_obs.area)
                    if med_area > 0
                    else 1.0
                )
                if step_ratio >= ratio_limit and baseline_ratio >= 1.5:
                    issues.append(
                        TrackQCIssue(
                            severity="warning",
                            code="area_jump",
                            message=(
                                f"Abrupt mask/bbox area change ({step_ratio:.2f}x) for "
                                f"'{inst_name}' between frames {prev_obs.frame} and {curr_obs.frame}."
                            ),
                            frame_start=prev_obs.frame,
                            frame_end=curr_obs.frame,
                            anchor_frame=curr_obs.frame,
                            instances=(inst_name,),
                            score=3.5 + min(4.5, step_ratio - ratio_limit),
                            metrics={
                                "area_px": float(curr_obs.area),
                                "prev_area_px": float(prev_obs.area),
                                "median_area_px": med_area,
                                "area_ratio": float(step_ratio),
                            },
                        )
                    )

        scores = [o.score for o in traj if o.score is not None]
        mean_score = (sum(scores) / len(scores)) if scores else None
        coverage = len(present_frames) / float(total_span) if total_span > 0 else 0.0
        instance_summaries.append(
            InstanceTrackSummary(
                instance=inst_name,
                observed_frames=len(present_frames),
                missing_frames=len(missing_frames_sorted),
                coverage_ratio=coverage,
                duplicate_frames=int(duplicate_frames_by_inst[inst_name]),
                median_speed_px_per_frame=med_speed,
                max_speed_px_per_frame=max_speed,
                median_area_px=med_area,
                mean_score=mean_score,
            )
        )

    # 3. Pairwise trajectory-crossing / close-contact identity-swap detector.
    if len(known_instances) >= 2:
        obs_lookup: Dict[str, Dict[int, _InstanceObservation]] = {
            inst: {o.frame: o for o in canonical_by_inst[inst]}
            for inst in known_instances
        }
        for inst_a, inst_b in combinations(known_instances, 2):
            map_a = obs_lookup[inst_a]
            map_b = obs_lookup[inst_b]
            shared_frames = sorted(set(map_a) & set(map_b))
            if len(shared_frames) < 2:
                continue

            for idx in range(1, len(shared_frames)):
                t1 = shared_frames[idx - 1]
                t2 = shared_frames[idx]
                dt = t2 - t1
                if dt < 1 or dt > max(1, int(max_swap_lookahead_frames)):
                    continue

                a1, a2 = map_a[t1], map_a[t2]
                b1, b2 = map_b[t1], map_b[t2]

                # Estimate expected positions at t2 using prior velocity when available.
                pred_a = a1.centroid
                pred_b = b1.centroid
                if idx >= 2:
                    t0 = shared_frames[idx - 2]
                    dt_prev = t1 - t0
                    if 1 <= dt_prev <= max(1, int(max_swap_lookahead_frames)):
                        a0 = map_a[t0]
                        b0 = map_b[t0]
                        va_x = (a1.centroid[0] - a0.centroid[0]) / float(dt_prev)
                        va_y = (a1.centroid[1] - a0.centroid[1]) / float(dt_prev)
                        vb_x = (b1.centroid[0] - b0.centroid[0]) / float(dt_prev)
                        vb_y = (b1.centroid[1] - b0.centroid[1]) / float(dt_prev)
                        # Damped extrapolation avoids overshooting on sharp turns.
                        damp = 0.5
                        pred_a = (
                            a1.centroid[0] + damp * va_x * dt,
                            a1.centroid[1] + damp * va_y * dt,
                        )
                        pred_b = (
                            b1.centroid[0] + damp * vb_x * dt,
                            b1.centroid[1] + damp * vb_y * dt,
                        )

                sep_prev = math.hypot(
                    a1.centroid[0] - b1.centroid[0],
                    a1.centroid[1] - b1.centroid[1],
                )
                sep_curr = math.hypot(
                    a2.centroid[0] - b2.centroid[0],
                    a2.centroid[1] - b2.centroid[1],
                )
                had_close_contact = (
                    min(sep_prev, sep_curr) <= float(close_contact_px)
                    or _observation_iou(a1, b1) > 0.0
                    or _observation_iou(a2, b2) > 0.0
                )

                keep_cost = math.hypot(
                    a2.centroid[0] - pred_a[0], a2.centroid[1] - pred_a[1]
                ) + math.hypot(b2.centroid[0] - pred_b[0], b2.centroid[1] - pred_b[1])
                swap_cost = math.hypot(
                    a2.centroid[0] - pred_b[0], a2.centroid[1] - pred_b[1]
                ) + math.hypot(b2.centroid[0] - pred_a[0], b2.centroid[1] - pred_a[1])

                step_a = math.hypot(
                    a2.centroid[0] - a1.centroid[0],
                    a2.centroid[1] - a1.centroid[1],
                )
                step_b = math.hypot(
                    b2.centroid[0] - b1.centroid[0],
                    b2.centroid[1] - b1.centroid[1],
                )
                min_required_jump = (
                    max(8.0, 0.5 * float(min_jump_px))
                    if had_close_contact
                    else float(min_jump_px)
                )

                if (
                    max(step_a, step_b) >= min_required_jump
                    and swap_cost + float(swap_margin_px) < keep_cost
                    and swap_cost <= 0.65 * keep_cost
                ):
                    advantage = keep_cost - swap_cost
                    issues.append(
                        TrackQCIssue(
                            severity="error",
                            code="identity_swap_suspect",
                            message=(
                                f"Suspected identity swap between '{inst_a}' and '{inst_b}' "
                                f"across frames {t1}->{t2} (swap cost {swap_cost:.1f}px vs "
                                f"keep cost {keep_cost:.1f}px)."
                            ),
                            frame_start=t1,
                            frame_end=t2,
                            anchor_frame=t2,
                            instances=(inst_a, inst_b),
                            score=9.5 + min(6.0, advantage / 20.0),
                            metrics={
                                "keep_cost_px": float(keep_cost),
                                "swap_cost_px": float(swap_cost),
                                "cost_advantage_px": float(advantage),
                                "min_separation_px": float(min(sep_prev, sep_curr)),
                            },
                        )
                    )

    # 4. Anatomical PoseSchema skeleton edge-length outlier check.
    if (
        effective_pose_schema is not None
        and effective_pose_schema.edges
        and keypoints_by_frame
    ):
        edge_samples: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = defaultdict(
            list
        )
        for frame in sorted(keypoints_by_frame):
            for inst_name, kp_map in sorted(keypoints_by_frame[frame].items()):
                for src_kp, dst_kp in effective_pose_schema.edges:
                    if src_kp in kp_map and dst_kp in kp_map:
                        p1 = kp_map[src_kp]
                        p2 = kp_map[dst_kp]
                        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        edge_samples[(inst_name, src_kp, dst_kp)].append(
                            (frame, length)
                        )

        stretch_limit = max(1.2, float(max_edge_stretch_ratio))
        for (inst_name, src_kp, dst_kp), samples in sorted(edge_samples.items()):
            if len(samples) < 3:
                continue
            med_len = float(median(length for _f, length in samples))
            if med_len < float(min_edge_baseline_px):
                continue
            display_inst = "" if inst_name == "__single__" else inst_name
            for frame, length in samples:
                if length <= 0.0:
                    continue
                stretch = length / med_len
                if stretch >= stretch_limit:
                    sev = "error" if stretch >= stretch_limit * 1.5 else "warning"
                    inst_tuple = (display_inst,) if display_inst else ()
                    prefix = f"'{display_inst}' " if display_inst else ""
                    issues.append(
                        TrackQCIssue(
                            severity=sev,
                            code="anatomical_edge_outlier",
                            message=(
                                f"Anatomical edge {prefix}{src_kp}->{dst_kp} stretched "
                                f"{stretch:.2f}x median ({length:.1f}px vs {med_len:.1f}px) "
                                f"on frame {frame}."
                            ),
                            frame_start=frame,
                            frame_end=frame,
                            anchor_frame=frame,
                            instances=inst_tuple,
                            score=(7.5 if sev == "error" else 4.5)
                            + min(5.0, stretch - stretch_limit),
                            metrics={
                                "edge_length_px": float(length),
                                "median_edge_length_px": med_len,
                                "stretch_ratio": float(stretch),
                            },
                        )
                    )

    issues.sort(
        key=lambda item: (
            -_SEVERITY_RANK.get(item.severity, 0),
            -item.score,
            item.frame_start,
            item.code,
        )
    )
    review_queue = _build_review_queue(
        issues,
        merge_gap_frames=bout_merge_gap_frames,
        top_k=top_k_review,
    )

    return TrackQCReport(
        source_path=str(resolved),
        source_kind=source_kind,
        frame_min=frame_min,
        frame_max=frame_max,
        total_frame_span=total_span,
        observed_frame_count=len(all_frames),
        expected_instances=inferred_expected,
        fps=fps,
        instances=tuple(instance_summaries),
        issues=tuple(issues),
        review_queue=review_queue,
    )


def format_track_qc_report_table(report: TrackQCReport) -> str:
    """Format a human-readable summary of track QC metrics and the review queue."""
    lines: List[str] = [
        f"Source: {report.source_path} ({report.source_kind})",
        (
            f"Frames: {report.frame_min}..{report.frame_max} "
            f"(span={report.total_frame_span}, observed={report.observed_frame_count}) | "
            f"Instances: {len(report.instances)} (expected={report.expected_instances})"
        ),
        (
            f"Issues: {report.error_count} error(s), "
            f"{report.warning_count} warning(s), {report.info_count} info"
        ),
    ]

    if report.instances:
        lines.append("")
        lines.append(
            f"{'Instance':<18} {'Observed':>8} {'Missing':>8} {'Coverage':>9} "
            f"{'MedSpeed':>9} {'MaxSpeed':>9} {'MedArea':>10}"
        )
        lines.append("-" * 78)
        for inst in report.instances:
            lines.append(
                f"{inst.instance:<18} {inst.observed_frames:>8d} {inst.missing_frames:>8d} "
                f"{inst.coverage_ratio * 100.0:>8.1f}% "
                f"{inst.median_speed_px_per_frame:>9.2f} "
                f"{inst.max_speed_px_per_frame:>9.2f} "
                f"{inst.median_area_px:>10.1f}"
            )

    if report.review_queue:
        lines.append("")
        lines.append("Prioritized Active-Learning Review Queue:")
        lines.append(
            f"{'Rank':>4} {'Anchor':>7} {'Window':<15} {'Severity':<8} "
            f"{'Risk':>6} {'Action':<28} {'Codes'}"
        )
        lines.append("-" * 92)
        for bout in report.review_queue:
            window = f"{bout.frame_start}..{bout.frame_end}"
            codes = ",".join(bout.issue_codes)
            lines.append(
                f"{bout.rank:>4d} {bout.anchor_frame:>7d} {window:<15} "
                f"{bout.max_severity:<8} {bout.risk_score:>6.2f} "
                f"{bout.recommended_action:<28} {codes}"
            )
    else:
        lines.append("")
        lines.append("No tracking anomalies detected.")

    return "\n".join(lines)


def write_track_qc_report_json(
    report: TrackQCReport,
    output_path: Path | str,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically write ``report`` (and optional provenance block) to JSON."""
    dst = Path(output_path)
    payload = report.to_dict()
    if provenance:
        payload["provenance"] = dict(provenance)
    atomic_write_json(dst, payload, indent=2, ensure_ascii=False)
    return dst


def write_track_qc_issues_csv(
    issues: Iterable[TrackQCIssue], output_path: Path | str
) -> Path:
    """Atomically write detected ``TrackQCIssue`` rows to CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Severity",
            "Code",
            "AnchorFrame",
            "FrameStart",
            "FrameEnd",
            "Instances",
            "Score",
            "Message",
        ]
    )
    for item in issues:
        writer.writerow(
            [
                item.severity,
                item.code,
                item.anchor_frame,
                item.frame_start,
                item.frame_end,
                ";".join(item.instances),
                f"{item.score:.6f}",
                item.message,
            ]
        )
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def write_review_queue_csv(
    review_queue: Iterable[ReviewBout], output_path: Path | str
) -> Path:
    """Atomically write prioritized ``ReviewBout`` rows to CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "Rank",
            "AnchorFrame",
            "FrameStart",
            "FrameEnd",
            "MaxSeverity",
            "RiskScore",
            "RecommendedAction",
            "Instances",
            "IssueCodes",
            "IssueCount",
            "Summary",
        ]
    )
    for bout in review_queue:
        writer.writerow(
            [
                bout.rank,
                bout.anchor_frame,
                bout.frame_start,
                bout.frame_end,
                bout.max_severity,
                f"{bout.risk_score:.6f}",
                bout.recommended_action,
                ";".join(bout.instances),
                ";".join(bout.issue_codes),
                bout.issue_count,
                bout.summary,
            ]
        )
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst
