from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from annolid.behavior.event_utils import normalize_event_label


def compute_timeline_points(
    *,
    start_seconds: float,
    end_seconds: float,
    step_seconds: int,
    fps: float,
    total_frames: int,
) -> List[Tuple[int, float]]:
    """Return inclusive (frame_index, t_seconds) points sampled every ``step_seconds``.

    This helper is intentionally GUI/Qt-free so it can be reused in headless contexts.
    """

    fps = float(fps or 0.0)
    total_frames = int(total_frames or 0)
    step_seconds = int(step_seconds or 0)
    start_seconds = float(start_seconds or 0.0)
    end_seconds = float(end_seconds or 0.0)

    if fps <= 0 or total_frames <= 0 or step_seconds <= 0:
        return []
    if end_seconds < start_seconds:
        return []

    points: List[Tuple[int, float]] = []
    t = start_seconds
    # Guard against floating point drift for long ranges by stepping in integers.
    steps = int((end_seconds - start_seconds) // step_seconds) + 1
    for i in range(steps):
        t = start_seconds + (i * step_seconds)
        frame_idx = int(round(t * fps))
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        points.append((frame_idx, t))
    return points


def format_hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS, clamped to a 24h clock."""

    total = int(round(float(seconds or 0.0)))
    total = max(0, min(total, (24 * 60 * 60) - 1))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def timeline_intervals_to_timestamp_rows(
    intervals: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    subject: str = "Subject 1",
) -> List[Tuple[float, float, str, str, str]]:
    """Convert behavior timeline intervals into Annolid timestamp CSV rows."""

    fps_value = float(fps or 0.0)
    if fps_value <= 0:
        return []

    cleaned: List[Dict[str, Any]] = []
    for interval in intervals:
        try:
            start_frame = int(interval.get("start_frame"))  # type: ignore[arg-type]
            # type: ignore[arg-type]
            end_frame = int(interval.get("end_frame"))
        except Exception:
            continue
        if start_frame < 0 or end_frame < start_frame:
            continue
        description = str(interval.get("description", "")).strip()
        if not description:
            continue
        cleaned.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "description": description,
            }
        )

    if not cleaned:
        return []

    cleaned.sort(key=lambda item: int(item["start_frame"]))
    merged: List[Dict[str, Any]] = []
    for current in cleaned:
        if not merged:
            merged.append(dict(current))
            continue
        prev = merged[-1]
        if (
            str(prev["description"]) == str(current["description"])
            and int(current["start_frame"]) <= int(prev["end_frame"]) + 1
        ):
            prev["end_frame"] = max(int(prev["end_frame"]), int(current["end_frame"]))
            continue
        merged.append(dict(current))

    rows: List[Tuple[float, float, str, str, str]] = []
    for interval in merged:
        start_time = float(interval["start_frame"]) / fps_value
        end_time = float(interval["end_frame"]) / fps_value
        behavior = str(interval["description"])
        rows.append((start_time, start_time, subject, behavior, "state start"))
        rows.append((end_time, end_time, subject, behavior, "state stop"))

    rows.sort(key=lambda row: (float(row[1]), str(row[3]), str(row[4])))
    return rows


def timestamp_rows_to_timeline_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    fps: float,
) -> List[Dict[str, Any]]:
    """Convert Annolid timestamp CSV rows into timeline intervals."""

    fps_value = float(fps or 0.0)
    if fps_value <= 0:
        return []

    per_behavior: Dict[str, List[Tuple[float, str]]] = {}
    for row in rows:
        behavior = str(row.get("Behavior", "")).strip()
        if not behavior:
            continue
        raw_time = row.get("Recording time")
        try:
            recording_time = float(raw_time)
        except Exception:
            continue
        canonical = normalize_event_label(str(row.get("Event", "")).strip())
        if canonical is None:
            continue
        per_behavior.setdefault(behavior, []).append((recording_time, canonical))

    intervals: List[Dict[str, Any]] = []
    for behavior, events in per_behavior.items():
        events.sort(key=lambda item: (float(item[0]), 0 if item[1] == "start" else 1))
        open_starts: List[float] = []
        for recording_time, event_label in events:
            if event_label == "start":
                open_starts.append(recording_time)
                continue
            if open_starts:
                start_time = open_starts.pop()
            else:
                start_time = recording_time
            end_time = max(start_time, recording_time)
            start_frame = max(0, int(round(start_time * fps_value)))
            end_frame = max(start_frame, int(round(end_time * fps_value)))
            intervals.append(
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": format_hhmmss(start_time),
                    "end_time": format_hhmmss(end_time),
                    "description": behavior,
                }
            )

        for start_time in open_starts:
            start_frame = max(0, int(round(start_time * fps_value)))
            intervals.append(
                {
                    "start_frame": start_frame,
                    "end_frame": start_frame,
                    "start_time": format_hhmmss(start_time),
                    "end_time": format_hhmmss(start_time),
                    "description": behavior,
                }
            )

    intervals.sort(
        key=lambda item: (
            int(item.get("start_frame", 0)),
            int(item.get("end_frame", 0)),
            str(item.get("description", "")),
        )
    )
    return intervals
