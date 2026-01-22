from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, List, Optional, Sequence, Union

from annolid.core.types.behavior import BehaviorEvent
from annolid.core.types.frame import FrameRef

BEHAVIOR_CSV_HEADER: Sequence[str] = (
    "Trial time",
    "Recording time",
    "Subject",
    "Behavior",
    "Event",
)


def behavior_events_from_csv(
    path_or_file: Union[str, Path, IO[str]],
    *,
    fps: Optional[float] = None,
    video_name: Optional[str] = None,
) -> List[BehaviorEvent]:
    """Load behavior events from Annolid's behavior timestamp CSV format."""

    fps_value = float(fps) if fps is not None and fps > 0 else 29.97
    with _open_text(path_or_file, "r") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []

        required = {"Recording time", "Behavior", "Event"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing!r}")

        events: List[BehaviorEvent] = []
        for row in reader:
            recording_raw = row.get("Recording time")
            if recording_raw is None or str(recording_raw).strip() == "":
                continue
            timestamp_sec = float(recording_raw)
            frame_index = int(round(timestamp_sec * fps_value))

            behavior = str(row.get("Behavior") or "").strip()
            event_label = str(row.get("Event") or "").strip()
            subject_value = row.get("Subject")
            subject = (
                str(subject_value).strip()
                if subject_value is not None and str(subject_value).strip()
                else None
            )

            meta: dict[str, object] = {}
            trial_raw = row.get("Trial time")
            if trial_raw is not None and str(trial_raw).strip():
                try:
                    meta["trial_time_sec"] = float(trial_raw)
                except ValueError:
                    pass

            events.append(
                BehaviorEvent(
                    frame=FrameRef(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        video_name=video_name,
                    ),
                    behavior=behavior,
                    event=event_label,
                    subject=subject,
                    meta=meta,
                )
            )

        return sorted(
            events, key=lambda evt: (evt.frame.frame_index, evt.behavior, evt.event)
        )


def behavior_events_to_csv(
    events: Sequence[BehaviorEvent],
    path_or_file: Union[str, Path, IO[str]],
    *,
    fps: Optional[float] = None,
) -> None:
    """Write behavior events to Annolid's behavior timestamp CSV format."""

    fps_value = float(fps) if fps is not None and fps > 0 else 29.97
    with _open_text(path_or_file, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BEHAVIOR_CSV_HEADER))
        writer.writeheader()

        for event in sorted(
            events, key=lambda evt: (evt.frame.frame_index, evt.behavior, evt.event)
        ):
            timestamp_sec = event.frame.timestamp_sec
            if timestamp_sec is None:
                timestamp_sec = event.frame.frame_index / fps_value

            trial_time = event.meta.get("trial_time_sec") if event.meta else None
            writer.writerow(
                {
                    "Trial time": "" if trial_time is None else float(trial_time),
                    "Recording time": float(timestamp_sec),
                    "Subject": event.subject or "",
                    "Behavior": event.behavior,
                    "Event": event.event,
                }
            )


@contextmanager
def _open_text(path_or_file: Union[str, Path, IO[str]], mode: str) -> Iterator[IO[str]]:
    if isinstance(path_or_file, (str, Path)):
        with open(path_or_file, mode, newline="", encoding="utf-8") as handle:
            yield handle
        return
    yield path_or_file
