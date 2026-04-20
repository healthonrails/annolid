"""Aggression bout scoring from counted sub-events.

This module provides deterministic aggregation for sub-events such as:
- slap in the face
- run away
- initiation of bigger fights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from annolid.domain.behavior_agent import BehaviorSegment, BehaviorSubEvent

AGGRESSION_SUBEVENT_TYPES = (
    "slap_face",
    "run_away",
    "fight_initiation",
)

_SUBEVENT_ALIASES = {
    "slap_face": "slap_face",
    "slap in the face": "slap_face",
    "slap-in-the-face": "slap_face",
    "run_away": "run_away",
    "run away": "run_away",
    "fight_initiation": "fight_initiation",
    "fight initiation": "fight_initiation",
    "initiation_of_bigger_fights": "fight_initiation",
    "initiation of bigger fights": "fight_initiation",
}


@dataclass(frozen=True)
class AggressionBoutCount:
    bout_id: str
    segment_id: str
    start_frame: int
    end_frame: int
    duration_frames: int
    total_sub_events: int
    slap_face_count: int
    run_away_count: int
    fight_initiation_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "bout_id": self.bout_id,
            "segment_id": self.segment_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "total_sub_events": self.total_sub_events,
            "slap_face_count": self.slap_face_count,
            "run_away_count": self.run_away_count,
            "fight_initiation_count": self.fight_initiation_count,
        }


def normalize_aggression_sub_event_type(event_type: str) -> str | None:
    key = str(event_type or "").strip().lower()
    if not key:
        return None
    return _SUBEVENT_ALIASES.get(key)


def aggregate_aggression_bout_counts(
    segments: Iterable[BehaviorSegment],
) -> list[AggressionBoutCount]:
    """Aggregate deterministic bout counts from aggression segment sub-events.

    Determinism guarantee:
    - segments are processed in a stable sorted order by start/end/segment_id
    - unrecognized sub-event labels are ignored
    - negative/non-integer sub-event counts are treated as invalid and ignored
    """
    ordered_segments = sorted(
        segments,
        key=lambda seg: (int(seg.start_frame), int(seg.end_frame), str(seg.segment_id)),
    )
    rows: list[AggressionBoutCount] = []

    for idx, segment in enumerate(ordered_segments, start=1):
        counts = {key: 0 for key in AGGRESSION_SUBEVENT_TYPES}
        for event in _ordered_sub_events(segment.sub_events):
            normalized = normalize_aggression_sub_event_type(event.event_type)
            if normalized is None:
                continue
            if not isinstance(event.count, int) or event.count < 0:
                continue
            counts[normalized] += int(event.count)

        total = int(sum(counts.values()))
        duration = max(0, int(segment.end_frame) - int(segment.start_frame) + 1)
        rows.append(
            AggressionBoutCount(
                bout_id=f"bout_{idx:04d}",
                segment_id=str(segment.segment_id),
                start_frame=int(segment.start_frame),
                end_frame=int(segment.end_frame),
                duration_frames=duration,
                total_sub_events=total,
                slap_face_count=int(counts["slap_face"]),
                run_away_count=int(counts["run_away"]),
                fight_initiation_count=int(counts["fight_initiation"]),
            )
        )
    return rows


def validate_aggression_bout_counts(rows: Iterable[AggressionBoutCount]) -> list[str]:
    """Focused validation for bout-count outputs."""
    errors: list[str] = []
    seen_bout_ids: set[str] = set()

    for row in rows:
        if row.bout_id in seen_bout_ids:
            errors.append(f"Duplicate bout_id: {row.bout_id}")
        seen_bout_ids.add(row.bout_id)

        if row.end_frame < row.start_frame:
            errors.append(
                f"Invalid frame range for {row.bout_id}: "
                f"{row.start_frame}>{row.end_frame}"
            )

        subtotal = row.slap_face_count + row.run_away_count + row.fight_initiation_count
        if subtotal != row.total_sub_events:
            errors.append(
                f"Inconsistent total_sub_events for {row.bout_id}: "
                f"total={row.total_sub_events} subtotal={subtotal}"
            )

        for label, value in (
            ("slap_face_count", row.slap_face_count),
            ("run_away_count", row.run_away_count),
            ("fight_initiation_count", row.fight_initiation_count),
            ("total_sub_events", row.total_sub_events),
        ):
            if value < 0:
                errors.append(f"Negative {label} for {row.bout_id}: {value}")

    return errors


def _ordered_sub_events(events: Iterable[BehaviorSubEvent]) -> list[BehaviorSubEvent]:
    return sorted(
        events,
        key=lambda event: (
            int(event.frame_index),
            str(event.event_type),
            str(event.actor_id or ""),
            str(event.target_id or ""),
        ),
    )


__all__ = [
    "AGGRESSION_SUBEVENT_TYPES",
    "AggressionBoutCount",
    "aggregate_aggression_bout_counts",
    "normalize_aggression_sub_event_type",
    "validate_aggression_bout_counts",
]
