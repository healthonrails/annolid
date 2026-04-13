from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from annolid.postprocessing.zone_schema import (
    ZoneShapeSpec,
    load_zone_shapes,
    zone_shape_covers_point,
    zone_shape_distance_to_point,
)


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


def _normalize_id(value: Any) -> str:
    token = _normalize_text(value)
    if token:
        return token
    return ""


def _shape_track_id(shape: Mapping[str, Any]) -> str:
    keys = ("track_id", "tracking_id", "instance_id", "group_id")
    for key in keys:
        value = _normalize_id(shape.get(key))
        if value:
            return value
    flags = shape.get("flags")
    if isinstance(flags, Mapping):
        for key in keys:
            value = _normalize_id(flags.get(key))
            if value:
                return value
    return ""


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


def _polygon_area(points: Sequence[Sequence[Any]]) -> float:
    if len(points) < 3:
        return 0.0
    coords: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        x = _safe_float(point[0])
        y = _safe_float(point[1])
        if x is None or y is None:
            continue
        coords.append((x, y))
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(coords)):
        x1, y1 = coords[idx]
        x2, y2 = coords[(idx + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _shape_centroid_and_area(
    shape: Mapping[str, Any],
) -> tuple[tuple[float, float] | None, float]:
    points = shape.get("points") or []
    if not isinstance(points, list) or not points:
        return None, 0.0
    shape_type = _normalize_text(shape.get("shape_type")).lower()
    if shape_type == "rectangle" and len(points) >= 2:
        x1 = _safe_float(points[0][0])
        y1 = _safe_float(points[0][1])
        x2 = _safe_float(points[1][0])
        y2 = _safe_float(points[1][1])
        if None in (x1, y1, x2, y2):
            return None, 0.0
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        area = abs(float(x2) - float(x1)) * abs(float(y2) - float(y1))
        return (cx, cy), float(area)

    coords: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        x = _safe_float(point[0])
        y = _safe_float(point[1])
        if x is None or y is None:
            continue
        coords.append((x, y))
    if not coords:
        return None, 0.0
    cx = sum(x for x, _ in coords) / len(coords)
    cy = sum(y for _, y in coords) / len(coords)
    return (float(cx), float(cy)), float(_polygon_area(points))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _distance(
    p1: tuple[float, float] | None, p2: tuple[float, float] | None
) -> float | None:
    if p1 is None or p2 is None:
        return None
    return float(math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))


@dataclass(frozen=True)
class MetricCondition:
    metric: str
    op: str
    value: Any

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricCondition":
        return cls(
            metric=str(payload.get("metric") or "").strip(),
            op=str(payload.get("op") or "").strip().lower(),
            value=payload.get("value"),
        )


@dataclass(frozen=True)
class EvidenceRule:
    name: str
    assign_label: str
    conditions: tuple[MetricCondition, ...]
    min_streak_frames: int = 1
    priority: int = 0
    apply_to_labels: tuple[str, ...] = ()
    apply_to_track_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRule":
        conditions_payload = payload.get("conditions") or []
        conditions: list[MetricCondition] = []
        for item in conditions_payload:
            if isinstance(item, Mapping):
                conditions.append(MetricCondition.from_dict(item))
        return cls(
            name=str(payload.get("name") or payload.get("assign_label") or "rule"),
            assign_label=str(payload.get("assign_label") or "").strip(),
            conditions=tuple(conditions),
            min_streak_frames=max(1, int(payload.get("min_streak_frames", 1))),
            priority=int(payload.get("priority", 0)),
            apply_to_labels=tuple(
                str(item).strip()
                for item in (payload.get("apply_to_labels") or [])
                if str(item).strip()
            ),
            apply_to_track_ids=tuple(
                _normalize_id(item)
                for item in (payload.get("apply_to_track_ids") or [])
                if _normalize_id(item)
            ),
        )


@dataclass(frozen=True)
class GovernorPolicy:
    rules: tuple[EvidenceRule, ...]
    metric_aliases: dict[str, str] = field(default_factory=dict)
    ambiguity_conditions: tuple[MetricCondition, ...] = ()
    interesting_labels: tuple[str, ...] = ()
    interesting_track_ids: tuple[str, ...] = ()
    max_backtrack_frames: int = 500
    max_forward_gap_frames: int = 1
    min_correction_span_frames: int = 1
    canonical_track_ids: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GovernorPolicy":
        rules_payload = payload.get("rules") or []
        rules: list[EvidenceRule] = []
        for item in rules_payload:
            if isinstance(item, Mapping):
                rule = EvidenceRule.from_dict(item)
                if rule.assign_label and rule.conditions:
                    rules.append(rule)
        aliases = {
            str(key).strip(): str(value).strip()
            for key, value in dict(payload.get("metric_aliases") or {}).items()
            if str(key).strip() and str(value).strip()
        }
        ambiguity_conditions: list[MetricCondition] = []
        for item in payload.get("ambiguity_conditions") or []:
            if isinstance(item, Mapping):
                ambiguity_conditions.append(MetricCondition.from_dict(item))
        canonical_track_ids = {
            str(label).strip(): _normalize_id(track_id)
            for label, track_id in dict(
                payload.get("canonical_track_ids") or {}
            ).items()
            if str(label).strip() and _normalize_id(track_id)
        }
        return cls(
            rules=tuple(rules),
            metric_aliases=aliases,
            ambiguity_conditions=tuple(ambiguity_conditions),
            interesting_labels=tuple(
                str(item).strip()
                for item in (payload.get("interesting_labels") or [])
                if str(item).strip()
            ),
            interesting_track_ids=tuple(
                _normalize_id(item)
                for item in (payload.get("interesting_track_ids") or [])
                if _normalize_id(item)
            ),
            max_backtrack_frames=max(1, int(payload.get("max_backtrack_frames", 500))),
            max_forward_gap_frames=max(
                0, int(payload.get("max_forward_gap_frames", 1))
            ),
            min_correction_span_frames=max(
                1, int(payload.get("min_correction_span_frames", 1))
            ),
            canonical_track_ids=canonical_track_ids,
        )


@dataclass
class _Observation:
    frame_number: int
    json_path: Path
    shape_index: int
    track_id: str
    observed_label: str
    centroid: tuple[float, float] | None
    area: float
    features: dict[str, Any] = field(default_factory=dict)
    evidence_label: str = ""
    evidence_rule: str = ""


@dataclass(frozen=True)
class IdentityCorrection:
    track_id: str
    frame_start: int
    frame_end: int
    observed_label: str
    corrected_label: str
    rule_name: str
    rule_frame_start: int
    rule_frame_end: int
    observation_count: int


@dataclass(frozen=True)
class IdentityGovernorResult:
    annotation_dir: Path
    dry_run: bool
    scanned_files: int
    scanned_observations: int
    proposed_corrections: tuple[IdentityCorrection, ...]
    updated_files: int
    updated_shapes: int
    report_path: Path


class IdentityGovernor:
    """Policy-driven identity correction over LabelMe JSON frame annotations."""

    def __init__(
        self,
        annotation_dir: str | Path,
        policy: GovernorPolicy | Mapping[str, Any],
        *,
        zone_file: str | Path | None = None,
    ) -> None:
        self.annotation_dir = Path(annotation_dir).expanduser().resolve()
        self.policy = (
            policy
            if isinstance(policy, GovernorPolicy)
            else GovernorPolicy.from_dict(policy)
        )
        if not self.policy.rules:
            raise ValueError("Governor policy has no valid rules.")
        self.zone_specs = self._load_zone_specs(zone_file)

    def _load_zone_specs(self, zone_file: str | Path | None) -> list[ZoneShapeSpec]:
        if zone_file is None:
            return []
        zone_path = Path(zone_file).expanduser().resolve()
        if not zone_path.exists():
            return []
        try:
            payload = json.loads(zone_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, Mapping):
            return []
        return load_zone_shapes(payload)

    def _iter_frame_files(self) -> list[Path]:
        files: list[Path] = []
        for path in sorted(self.annotation_dir.glob("*.json")):
            frame_number = _frame_number_from_name(path)
            if frame_number is None:
                continue
            files.append(path)
        return files

    def _load_observations(
        self,
    ) -> tuple[dict[int, list[_Observation]], dict[Path, dict[str, Any]], int]:
        by_frame: dict[int, list[_Observation]] = {}
        payloads: dict[Path, dict[str, Any]] = {}
        count = 0
        for json_path in self._iter_frame_files():
            frame_number = _frame_number_from_name(json_path)
            if frame_number is None:
                continue
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            shapes = payload.get("shapes") or []
            if not isinstance(shapes, list):
                continue
            payloads[json_path] = payload
            frame_rows: list[_Observation] = []
            for shape_index, shape in enumerate(shapes):
                if not isinstance(shape, Mapping):
                    continue
                track_id = _shape_track_id(shape)
                if not track_id:
                    continue
                label = _shape_instance_label(shape)
                if not label:
                    continue
                centroid, area = _shape_centroid_and_area(shape)
                frame_rows.append(
                    _Observation(
                        frame_number=int(frame_number),
                        json_path=json_path,
                        shape_index=int(shape_index),
                        track_id=track_id,
                        observed_label=label,
                        centroid=centroid,
                        area=float(area),
                    )
                )
                count += 1
            if frame_rows:
                by_frame[int(frame_number)] = frame_rows
        return by_frame, payloads, count

    def _populate_features(self, by_frame: dict[int, list[_Observation]]) -> None:
        for frame_rows in by_frame.values():
            for row in frame_rows:
                row.features["area"] = float(row.area)
                row.features["frame_number"] = int(row.frame_number)
                row.features["track_id"] = row.track_id
                row.features["label"] = row.observed_label
                if row.centroid is not None:
                    row.features["x"] = float(row.centroid[0])
                    row.features["y"] = float(row.centroid[1])
                for zone in self.zone_specs:
                    name = zone.display_label
                    inside = False
                    distance = None
                    if row.centroid is not None:
                        inside = bool(zone_shape_covers_point(zone, row.centroid))
                        distance = zone_shape_distance_to_point(zone, row.centroid)
                    row.features[f"zone.inside.{name}"] = inside
                    if distance is not None:
                        row.features[f"zone.distance.{name}"] = float(distance)

            for row in frame_rows:
                nearest: float | None = None
                by_label: dict[str, float] = {}
                for other in frame_rows:
                    if other is row:
                        continue
                    d = _distance(row.centroid, other.centroid)
                    if d is None:
                        continue
                    if nearest is None or d < nearest:
                        nearest = d
                    current = by_label.get(other.observed_label)
                    if current is None or d < current:
                        by_label[other.observed_label] = float(d)
                    row.features[f"distance.to_track.{other.track_id}"] = float(d)
                if nearest is not None:
                    row.features["distance.nearest"] = float(nearest)
                for label, d in by_label.items():
                    row.features[f"distance.to_label.{label}"] = float(d)

    def _resolve_metric(self, metric: str) -> str:
        token = str(metric or "").strip()
        if not token:
            return token
        return self.policy.metric_aliases.get(token, token)

    def _compare(self, left: Any, op: str, right: Any) -> bool:
        if op in {"eq", "=="}:
            return left == right
        if op in {"ne", "!="}:
            return left != right
        if op in {"gt", ">"}:
            return bool(left > right)
        if op in {"gte", ">="}:
            return bool(left >= right)
        if op in {"lt", "<"}:
            return bool(left < right)
        if op in {"lte", "<="}:
            return bool(left <= right)
        if op == "in":
            return left in right
        if op == "not_in":
            return left not in right
        return False

    def _condition_matches(
        self, features: Mapping[str, Any], condition: MetricCondition
    ) -> bool:
        metric = self._resolve_metric(condition.metric)
        if metric not in features:
            return False
        left = features.get(metric)
        right = condition.value
        if isinstance(left, bool) and isinstance(right, str):
            right = str(right).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(left, (int, float)) and isinstance(right, str):
            parsed = _safe_float(right)
            if parsed is not None:
                right = parsed
        try:
            return self._compare(left, condition.op, right)
        except Exception:
            return False

    def _targeted(self, row: _Observation) -> bool:
        if self.policy.interesting_labels and row.observed_label not in set(
            self.policy.interesting_labels
        ):
            return False
        if self.policy.interesting_track_ids and row.track_id not in set(
            self.policy.interesting_track_ids
        ):
            return False
        return True

    def _rule_applies_to_row(self, rule: EvidenceRule, row: _Observation) -> bool:
        if rule.apply_to_labels and row.observed_label not in set(rule.apply_to_labels):
            return False
        if rule.apply_to_track_ids and row.track_id not in set(rule.apply_to_track_ids):
            return False
        return True

    def _assign_evidence(self, by_frame: dict[int, list[_Observation]]) -> None:
        for frame_number in sorted(by_frame):
            for row in by_frame[frame_number]:
                if not self._targeted(row):
                    continue
                matched: list[EvidenceRule] = []
                for rule in self.policy.rules:
                    if not self._rule_applies_to_row(rule, row):
                        continue
                    if all(
                        self._condition_matches(row.features, condition)
                        for condition in rule.conditions
                    ):
                        matched.append(rule)
                if not matched:
                    continue
                best = sorted(
                    matched,
                    key=lambda item: (item.priority, len(item.conditions)),
                    reverse=True,
                )[0]
                row.evidence_label = best.assign_label
                row.evidence_rule = best.name

    def _collect_track_rows(
        self, by_frame: dict[int, list[_Observation]]
    ) -> dict[str, list[_Observation]]:
        out: dict[str, list[_Observation]] = {}
        for frame in sorted(by_frame):
            for row in by_frame[frame]:
                out.setdefault(row.track_id, []).append(row)
        for track_rows in out.values():
            track_rows.sort(key=lambda item: item.frame_number)
        return out

    def _ambiguity_true(self, row: _Observation) -> bool:
        if not self.policy.ambiguity_conditions:
            return False
        return all(
            self._condition_matches(row.features, condition)
            for condition in self.policy.ambiguity_conditions
        )

    def _min_rule_streak(self, label: str) -> int:
        streaks = [
            int(rule.min_streak_frames)
            for rule in self.policy.rules
            if rule.assign_label == label
        ]
        if not streaks:
            return 1
        return max(1, max(streaks))

    def _build_track_corrections(
        self, rows: list[_Observation]
    ) -> list[IdentityCorrection]:
        mismatches = [
            row
            for row in rows
            if row.evidence_label and row.evidence_label != row.observed_label
        ]
        if not mismatches:
            return []
        segments: list[list[_Observation]] = []
        current: list[_Observation] = []
        for row in mismatches:
            if not current:
                current = [row]
                continue
            prev = current[-1]
            contiguous = row.frame_number <= prev.frame_number + 1
            same_label = row.evidence_label == prev.evidence_label
            if contiguous and same_label:
                current.append(row)
            else:
                segments.append(current)
                current = [row]
        if current:
            segments.append(current)

        rows_by_frame = {row.frame_number: row for row in rows}
        corrections: list[IdentityCorrection] = []
        for segment in segments:
            first = segment[0]
            last = segment[-1]
            evidence_label = first.evidence_label
            if len(segment) < self._min_rule_streak(evidence_label):
                continue
            frame_start = first.frame_number
            frame_end = last.frame_number
            original_start = frame_start
            original_end = frame_end

            steps = 0
            while steps < self.policy.max_backtrack_frames:
                prev_frame = frame_start - 1
                prev_row = rows_by_frame.get(prev_frame)
                if prev_row is None:
                    break
                if not self._ambiguity_true(prev_row):
                    break
                frame_start = prev_frame
                steps += 1

            probe_frame = frame_end
            gap_count = 0
            while gap_count <= self.policy.max_forward_gap_frames:
                probe_frame += 1
                next_row = rows_by_frame.get(probe_frame)
                if next_row is None:
                    gap_count += 1
                    continue
                if (
                    next_row.evidence_label
                    and next_row.evidence_label != evidence_label
                ):
                    break
                frame_end = probe_frame
                gap_count = 0

            affected = [
                row
                for row in rows
                if frame_start <= row.frame_number <= frame_end
                and row.observed_label != evidence_label
            ]
            if len(affected) < self.policy.min_correction_span_frames:
                continue
            if not affected:
                continue
            corrections.append(
                IdentityCorrection(
                    track_id=first.track_id,
                    frame_start=int(frame_start),
                    frame_end=int(frame_end),
                    observed_label=str(affected[0].observed_label),
                    corrected_label=str(evidence_label),
                    rule_name=str(first.evidence_rule),
                    rule_frame_start=int(original_start),
                    rule_frame_end=int(original_end),
                    observation_count=len(affected),
                )
            )
        return corrections

    def _relabel_text(self, current_label: str, corrected_label: str) -> str:
        text = str(current_label or "")
        if ":" in text:
            _, suffix = text.split(":", 1)
            suffix = suffix.strip()
            if suffix:
                return f"{corrected_label}:{suffix}"
        return corrected_label

    def _apply_shape_identity(
        self,
        shape: dict[str, Any],
        corrected_label: str,
    ) -> bool:
        changed = False
        label = _normalize_text(shape.get("label"))
        new_label_text = self._relabel_text(label, corrected_label)
        if label and new_label_text != label:
            shape["label"] = new_label_text
            changed = True
        if _normalize_text(shape.get("instance_label")) != corrected_label:
            if "instance_label" in shape or "instance_name" not in shape:
                shape["instance_label"] = corrected_label
                changed = True
        if (
            _normalize_text(shape.get("instance_name"))
            and _normalize_text(shape.get("instance_name")) != corrected_label
        ):
            shape["instance_name"] = corrected_label
            changed = True
        flags = shape.get("flags")
        if isinstance(flags, dict):
            if _normalize_text(flags.get("instance_label")) != corrected_label:
                flags["instance_label"] = corrected_label
                changed = True
            if _normalize_text(flags.get("instance_name")):
                if _normalize_text(flags.get("instance_name")) != corrected_label:
                    flags["instance_name"] = corrected_label
                    changed = True

        canonical = self.policy.canonical_track_ids.get(corrected_label)
        if canonical:
            for key in ("track_id", "tracking_id", "instance_id", "group_id"):
                current = _normalize_id(shape.get(key))
                if current and current != canonical:
                    shape[key] = canonical
                    changed = True
            if isinstance(flags, dict):
                for key in ("track_id", "tracking_id", "instance_id", "group_id"):
                    current = _normalize_id(flags.get(key))
                    if current and current != canonical:
                        flags[key] = canonical
                        changed = True
        return changed

    def _serialize_report(
        self,
        *,
        report_path: Path,
        corrections: Sequence[IdentityCorrection],
        dry_run: bool,
        scanned_files: int,
        scanned_observations: int,
        updated_files: int,
        updated_shapes: int,
    ) -> None:
        payload = {
            "annotation_dir": str(self.annotation_dir),
            "dry_run": bool(dry_run),
            "scanned_files": int(scanned_files),
            "scanned_observations": int(scanned_observations),
            "updated_files": int(updated_files),
            "updated_shapes": int(updated_shapes),
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

    def run(
        self,
        *,
        apply_changes: bool = False,
        report_path: str | Path | None = None,
    ) -> IdentityGovernorResult:
        by_frame, payloads, scanned_observations = self._load_observations()
        self._populate_features(by_frame)
        self._assign_evidence(by_frame)

        corrections: list[IdentityCorrection] = []
        by_track = self._collect_track_rows(by_frame)
        for rows in by_track.values():
            corrections.extend(self._build_track_corrections(rows))

        updated_files: set[Path] = set()
        updated_shapes = 0
        if corrections and apply_changes:
            corrections_by_track: dict[str, list[IdentityCorrection]] = {}
            for correction in corrections:
                corrections_by_track.setdefault(correction.track_id, []).append(
                    correction
                )
            for rows in by_track.values():
                if not rows:
                    continue
                track_id = rows[0].track_id
                candidate_corrections = corrections_by_track.get(track_id, [])
                if not candidate_corrections:
                    continue
                for row in rows:
                    for correction in candidate_corrections:
                        if not (
                            correction.frame_start
                            <= row.frame_number
                            <= correction.frame_end
                        ):
                            continue
                        if row.observed_label == correction.corrected_label:
                            continue
                        payload = payloads.get(row.json_path)
                        if payload is None:
                            continue
                        shapes = payload.get("shapes")
                        if not isinstance(shapes, list):
                            continue
                        if row.shape_index < 0 or row.shape_index >= len(shapes):
                            continue
                        shape = shapes[row.shape_index]
                        if not isinstance(shape, dict):
                            continue
                        if self._apply_shape_identity(
                            shape, correction.corrected_label
                        ):
                            updated_shapes += 1
                            updated_files.add(row.json_path)

            for path in sorted(updated_files):
                payload = payloads.get(path)
                if isinstance(payload, Mapping):
                    _write_json_atomic(path, payload)

        if report_path is None:
            report_path = self.annotation_dir / "identity_governor_report.json"
        resolved_report = Path(report_path).expanduser().resolve()
        self._serialize_report(
            report_path=resolved_report,
            corrections=corrections,
            dry_run=not apply_changes,
            scanned_files=len(payloads),
            scanned_observations=scanned_observations,
            updated_files=len(updated_files),
            updated_shapes=updated_shapes,
        )
        return IdentityGovernorResult(
            annotation_dir=self.annotation_dir,
            dry_run=not apply_changes,
            scanned_files=len(payloads),
            scanned_observations=int(scanned_observations),
            proposed_corrections=tuple(corrections),
            updated_files=len(updated_files),
            updated_shapes=int(updated_shapes),
            report_path=resolved_report,
        )


def run_identity_governor(
    annotation_dir: str | Path,
    policy: GovernorPolicy | Mapping[str, Any],
    *,
    zone_file: str | Path | None = None,
    apply_changes: bool = False,
    report_path: str | Path | None = None,
) -> IdentityGovernorResult:
    governor = IdentityGovernor(
        annotation_dir=annotation_dir,
        policy=policy,
        zone_file=zone_file,
    )
    return governor.run(apply_changes=apply_changes, report_path=report_path)
