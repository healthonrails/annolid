from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd
from shapely.geometry import Polygon

from annolid.postprocessing.zone_schema import (
    ZoneShapeSpec,
    zone_shape_bounds,
    zone_shape_covers_point,
)


OUTSIDE_ZONE_LABEL = "__outside__"


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _zone_area(spec: ZoneShapeSpec) -> float:
    try:
        bounds = zone_shape_bounds(spec)
        if bounds is None:
            return float("inf")
        points = spec.analysis_points
        if len(points) < 3:
            return float("inf")
        return float(Polygon(points).area)
    except Exception:
        return float("inf")


def _is_barrier_adjacent_zone(spec: ZoneShapeSpec) -> bool:
    kind = str(spec.zone_kind or "").strip().lower()
    flags = dict(spec.flags or {})
    if _is_truthy(flags.get("barrier_adjacent")):
        return True
    if _is_truthy(flags.get("adjacent_to_barrier")):
        return True
    if _is_truthy(flags.get("mesh_adjacent")):
        return True
    return kind in {"barrier_edge", "barrier", "doorway", "passage"}


def _frame_column(dataframe: pd.DataFrame) -> str | None:
    for candidate in ("frame_number", "frame", "frame_idx"):
        if candidate in dataframe.columns:
            return candidate
    return None


def _shape_point_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("cx_tracked", "cy_tracked"),
        ("cx_tracking", "cy_tracking"),
        ("cx", "cy"),
        ("x", "y"),
    ]
    for x_col, y_col in candidates:
        if x_col in dataframe.columns and y_col in dataframe.columns:
            return x_col, y_col
    raise KeyError(
        "Could not resolve centroid columns. Expected one of "
        "cx_tracked/cy_tracked, cx_tracking/cy_tracking, cx/cy, or x/y."
    )


@dataclass(frozen=True)
class ZoneVisit:
    zone_label: str
    start_frame: int
    end_frame: int
    frame_count: int


@dataclass
class ZoneAnalysisResult:
    instance_name: str
    frame_count: int
    assay_profile: str = "generic"
    occupancy_frames: dict[str, int] = field(default_factory=dict)
    dwell_frames: dict[str, int] = field(default_factory=dict)
    entry_counts: dict[str, int] = field(default_factory=dict)
    first_entry_frames: dict[str, int] = field(default_factory=dict)
    barrier_adjacent_frames: int = 0
    transition_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    segments: list[ZoneVisit] = field(default_factory=list)
    outside_frames: int = 0

    def _seconds(self, fps: float | None, frames: int) -> float:
        if fps and fps > 0:
            return float(frames) / float(fps)
        return float(frames)

    def occupancy_seconds(self, fps: float | None) -> dict[str, float]:
        return {
            zone_label: self._seconds(fps, frames)
            for zone_label, frames in self.occupancy_frames.items()
        }

    def dwell_seconds(self, fps: float | None) -> dict[str, float]:
        return {
            zone_label: self._seconds(fps, frames)
            for zone_label, frames in self.dwell_frames.items()
        }

    def first_entry_seconds(self, fps: float | None) -> dict[str, float | None]:
        return {
            zone_label: (
                self._seconds(fps, frame) if frame is not None and frame >= 0 else None
            )
            for zone_label, frame in self.first_entry_frames.items()
        }

    def barrier_adjacent_seconds(self, fps: float | None) -> float:
        return self._seconds(fps, self.barrier_adjacent_frames)

    def total_transitions(self) -> int:
        return sum(sum(targets.values()) for targets in self.transition_counts.values())

    def to_summary_row(
        self,
        *,
        fps: float | None = None,
        include_transition_columns: bool = False,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "instance_name": self.instance_name,
            "assay_profile": self.assay_profile,
            "frame_count": self.frame_count,
            "outside_frames": self.outside_frames,
            "outside_seconds": self._seconds(fps, self.outside_frames),
            "barrier_adjacent_frames": self.barrier_adjacent_frames,
            "barrier_adjacent_seconds": self.barrier_adjacent_seconds(fps),
            "total_transitions": self.total_transitions(),
        }
        for zone_label in sorted(set(self.occupancy_frames) | set(self.entry_counts)):
            row[f"occupancy_frames__{zone_label}"] = self.occupancy_frames.get(
                zone_label, 0
            )
            row[f"occupancy_seconds__{zone_label}"] = self._seconds(
                fps, self.occupancy_frames.get(zone_label, 0)
            )
            row[f"dwell_frames__{zone_label}"] = self.dwell_frames.get(zone_label, 0)
            row[f"dwell_seconds__{zone_label}"] = self._seconds(
                fps, self.dwell_frames.get(zone_label, 0)
            )
            row[f"entry_count__{zone_label}"] = self.entry_counts.get(zone_label, 0)
            first_entry_frame = self.first_entry_frames.get(zone_label)
            row[f"first_entry_frame__{zone_label}"] = (
                int(first_entry_frame) if first_entry_frame is not None else None
            )
            row[f"first_entry_seconds__{zone_label}"] = (
                self._seconds(fps, first_entry_frame)
                if first_entry_frame is not None and first_entry_frame >= 0
                else None
            )

        if include_transition_columns:
            for source_label in sorted(self.transition_counts.keys()):
                targets = self.transition_counts[source_label]
                for target_label in sorted(targets.keys()):
                    row[f"transition_count__{source_label}__{target_label}"] = targets[
                        target_label
                    ]
        return row


class GenericZoneEngine:
    """Generic zone analysis engine for arbitrary saved zone shapes."""

    def __init__(
        self,
        zone_specs: Sequence[ZoneShapeSpec],
        *,
        fps: float | None = None,
        point_columns: tuple[str, str] | None = None,
    ) -> None:
        self.zone_specs = [spec for spec in zone_specs if spec.is_zone]
        self.fps = fps
        self.point_columns = point_columns
        self._ordered_zone_specs = sorted(
            enumerate(self.zone_specs),
            key=lambda item: (_zone_area(item[1]), item[0]),
        )
        self._barrier_zone_labels = {
            spec.display_label
            for spec in self.zone_specs
            if _is_barrier_adjacent_zone(spec)
        }

    def _resolve_zone_label(self, x: Any, y: Any) -> str | None:
        point = [_safe_float(x), _safe_float(y)]
        matches: list[tuple[float, int, ZoneShapeSpec]] = []
        for order, spec in self._ordered_zone_specs:
            try:
                if zone_shape_covers_point(spec, point):
                    matches.append((_zone_area(spec), order, spec))
            except Exception:
                continue
        if not matches:
            return None
        _, _, spec = min(matches, key=lambda item: (item[0], item[1]))
        return spec.display_label

    def _prepare_instance_dataframe(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> pd.DataFrame:
        if dataframe.empty or "instance_name" not in dataframe.columns:
            return dataframe.iloc[0:0].copy()
        instance_df = dataframe[dataframe["instance_name"] == instance_label].copy()
        if instance_df.empty:
            return instance_df

        frame_col = _frame_column(instance_df)
        if frame_col is not None:
            instance_df = instance_df.sort_values(frame_col)
            instance_df = instance_df.drop_duplicates(subset=[frame_col], keep="last")
        else:
            instance_df = instance_df.sort_index()
        return instance_df

    def _iter_observations(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> Iterable[tuple[int, str | None]]:
        instance_df = self._prepare_instance_dataframe(dataframe, instance_label)
        if instance_df.empty:
            return []
        frame_col = _frame_column(instance_df)
        if self.point_columns is not None:
            x_col, y_col = self.point_columns
            if x_col not in instance_df.columns or y_col not in instance_df.columns:
                x_col, y_col = _shape_point_columns(instance_df)
        else:
            x_col, y_col = _shape_point_columns(instance_df)
        observations: list[tuple[int, str | None]] = []
        for index, row in instance_df.iterrows():
            frame = _safe_int(row[frame_col]) if frame_col else _safe_int(index)
            zone_label = self._resolve_zone_label(row[x_col], row[y_col])
            observations.append((frame, zone_label))
        return observations

    def analyze_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> ZoneAnalysisResult:
        observations = list(self._iter_observations(dataframe, instance_label))
        occupancy_frames: Counter[str] = Counter()
        dwell_frames: Counter[str] = Counter()
        entry_counts: Counter[str] = Counter()
        first_entry_frames: dict[str, int] = {}
        transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        segments: list[ZoneVisit] = []
        barrier_adjacent_frames = 0
        outside_frames = 0

        current_zone: str | None = None
        current_start_frame: int | None = None
        current_end_frame: int | None = None
        current_frame_count = 0
        prev_frame: int | None = None

        def finalize_current_segment() -> None:
            nonlocal \
                current_zone, \
                current_start_frame, \
                current_end_frame, \
                current_frame_count
            if (
                current_zone is None
                or current_start_frame is None
                or current_end_frame is None
            ):
                return
            segments.append(
                ZoneVisit(
                    zone_label=current_zone,
                    start_frame=current_start_frame,
                    end_frame=current_end_frame,
                    frame_count=current_frame_count,
                )
            )
            dwell_frames[current_zone] += current_frame_count
            current_zone = None
            current_start_frame = None
            current_end_frame = None
            current_frame_count = 0

        for frame, zone_label in observations:
            if zone_label is None:
                outside_frames += 1
            else:
                occupancy_frames[zone_label] += 1
                if zone_label in self._barrier_zone_labels:
                    barrier_adjacent_frames += 1

            gap_break = prev_frame is not None and frame - prev_frame > 1

            if current_zone is None:
                if zone_label is not None:
                    current_zone = zone_label
                    current_start_frame = frame
                    current_end_frame = frame
                    current_frame_count = 1
                    entry_counts[zone_label] += 1
                    if zone_label not in first_entry_frames:
                        first_entry_frames[zone_label] = frame
                prev_frame = frame
                continue

            if gap_break or zone_label != current_zone:
                previous_zone = current_zone
                finalize_current_segment()
                if zone_label is not None:
                    if not gap_break and previous_zone != zone_label:
                        transition_counts[previous_zone][zone_label] += 1
                    current_zone = zone_label
                    current_start_frame = frame
                    current_end_frame = frame
                    current_frame_count = 1
                    entry_counts[zone_label] += 1
                    if zone_label not in first_entry_frames:
                        first_entry_frames[zone_label] = frame
            else:
                current_end_frame = frame
                current_frame_count += 1

            prev_frame = frame

        finalize_current_segment()

        return ZoneAnalysisResult(
            instance_name=instance_label,
            frame_count=len(observations),
            assay_profile="generic",
            occupancy_frames=dict(occupancy_frames),
            dwell_frames=dict(dwell_frames),
            entry_counts=dict(entry_counts),
            first_entry_frames=first_entry_frames,
            barrier_adjacent_frames=barrier_adjacent_frames,
            transition_counts={
                src: dict(targets) for src, targets in transition_counts.items()
            },
            segments=segments,
            outside_frames=outside_frames,
        )

    def analyze_dataframe(
        self, dataframe: pd.DataFrame
    ) -> dict[str, ZoneAnalysisResult]:
        if dataframe.empty or "instance_name" not in dataframe.columns:
            return {}
        results: dict[str, ZoneAnalysisResult] = {}
        for instance_label in dataframe["instance_name"].dropna().astype(str).unique():
            results[instance_label] = self.analyze_instance(dataframe, instance_label)
        return results

    def occupancy_seconds_for_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> dict[str, float]:
        return self.analyze_instance(dataframe, instance_label).occupancy_seconds(
            self.fps
        )

    def occupancy_frames_for_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> dict[str, int]:
        return self.analyze_instance(dataframe, instance_label).occupancy_frames

    def transition_counts_for_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> dict[str, dict[str, int]]:
        return self.analyze_instance(dataframe, instance_label).transition_counts

    def entry_counts_for_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> dict[str, int]:
        return self.analyze_instance(dataframe, instance_label).entry_counts

    def dwell_frames_for_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> dict[str, int]:
        return self.analyze_instance(dataframe, instance_label).dwell_frames

    def barrier_adjacent_frames_for_instance(
        self, dataframe: pd.DataFrame, instance_label: str
    ) -> int:
        return self.analyze_instance(dataframe, instance_label).barrier_adjacent_frames

    def summary_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        results = self.analyze_dataframe(dataframe)
        rows = [
            result.to_summary_row(fps=self.fps, include_transition_columns=True)
            for result in results.values()
        ]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("instance_name")
