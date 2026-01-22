from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    IO,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from annolid.core.types.frame import FrameRef
from annolid.core.types.geometry import BBoxGeometry, RLEGeometry
from annolid.core.types.tracking import Track, TrackObservation

TRACKING_CSV_HEADER: Tuple[str, ...] = (
    "frame_number",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "instance_name",
    "class_score",
    "segmentation",
    "tracking_id",
)


def tracks_from_labelme_csv(
    path_or_file: Union[str, Path, IO[str]],
    *,
    video_name: Optional[str] = None,
) -> List[Track]:
    """Load Tracks from Annolid's legacy Labelme tracking CSV format."""

    with _open_text(path_or_file, "r") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        _validate_required_columns(
            reader.fieldnames, required=("frame_number", "x1", "y1", "x2", "y2")
        )

        observations: Dict[str, List[TrackObservation]] = defaultdict(list)
        track_labels: Dict[str, str] = {}
        track_meta: Dict[str, Dict[str, object]] = {}

        for row in reader:
            frame_index = _parse_int(row.get("frame_number"))
            x1 = _parse_float(row.get("x1"))
            y1 = _parse_float(row.get("y1"))
            x2 = _parse_float(row.get("x2"))
            y2 = _parse_float(row.get("y2"))

            instance_name = (row.get("instance_name") or "").strip()
            score = _parse_optional_float(row.get("class_score"))
            tracking_id_int = _parse_int(row.get("tracking_id"), default=0)
            track_id = _canonical_track_id(tracking_id_int, instance_name)

            mask = _parse_optional_rle(row.get("segmentation"))
            obs = TrackObservation(
                frame=FrameRef(
                    frame_index=frame_index, timestamp_sec=None, video_name=video_name
                ),
                geometry=BBoxGeometry("bbox", (x1, y1, x2, y2)),
                score=score,
                label=instance_name or None,
                mask=mask,
            )
            observations[track_id].append(obs)
            track_labels.setdefault(track_id, instance_name or f"track_{track_id}")
            track_meta.setdefault(track_id, {"tracking_id": tracking_id_int})

        tracks: List[Track] = []
        for track_id, obs_list in observations.items():
            sorted_obs = sorted(obs_list, key=lambda obs: obs.frame.frame_index)
            tracks.append(
                Track(
                    track_id=track_id,
                    label=track_labels.get(track_id, f"track_{track_id}"),
                    observations=sorted_obs,
                    meta=track_meta.get(track_id, {}),
                )
            )
        return sorted(tracks, key=lambda t: t.track_id)


def tracks_to_labelme_csv(
    tracks: Sequence[Track],
    path_or_file: Union[str, Path, IO[str]],
) -> None:
    """Write Tracks to Annolid's legacy Labelme tracking CSV format."""

    with _open_text(path_or_file, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACKING_CSV_HEADER))
        writer.writeheader()

        rows: List[Dict[str, object]] = []
        for track in tracks:
            tracking_id_int = _tracking_id_for_track(track)
            for obs in track.observations:
                if obs.geometry.type != "bbox":
                    raise ValueError(
                        "Legacy tracking CSV writer requires bbox geometry."
                    )
                x1, y1, x2, y2 = obs.geometry.xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                segmentation_value = _serialize_optional_rle(obs.mask)
                rows.append(
                    {
                        "frame_number": int(obs.frame.frame_index),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "cx": float(cx),
                        "cy": float(cy),
                        "instance_name": obs.label or track.label,
                        "class_score": float(obs.score)
                        if obs.score is not None
                        else 1.0,
                        "segmentation": segmentation_value,
                        "tracking_id": int(tracking_id_int),
                    }
                )

        for row in sorted(
            rows,
            key=lambda r: (
                int(r["frame_number"]),
                str(r["tracking_id"]),
                str(r["instance_name"]),
            ),
        ):
            writer.writerow(row)


@contextmanager
def _open_text(path_or_file: Union[str, Path, IO[str]], mode: str) -> Iterator[IO[str]]:
    if isinstance(path_or_file, (str, Path)):
        with open(path_or_file, mode, newline="", encoding="utf-8") as handle:
            yield handle
        return
    yield path_or_file


def _validate_required_columns(
    fieldnames: Iterable[str], *, required: Sequence[str]
) -> None:
    fields = {name.strip() for name in fieldnames}
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing!r}")


def _parse_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _parse_float(value: object) -> float:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        raise ValueError("Expected a float value.")
    return float(raw)


def _parse_optional_float(value: object) -> Optional[float]:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _canonical_track_id(tracking_id: int, instance_name: str) -> str:
    if tracking_id > 0:
        return str(tracking_id)
    suffix = instance_name.strip() or "unknown"
    return f"0:{suffix}"


def _tracking_id_for_track(track: Track) -> int:
    raw = track.meta.get("tracking_id")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    if track.track_id.isdigit():
        return int(track.track_id)
    return 0


def _parse_optional_rle(value: object) -> Optional[RLEGeometry]:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        return None

    payload: object
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(payload, Mapping):
        return None
    counts = payload.get("counts")
    size = payload.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    h, w = size
    if counts is None:
        return None
    return RLEGeometry("rle", (int(h), int(w)), str(counts))


def _serialize_optional_rle(mask: Optional[RLEGeometry]) -> str:
    if mask is None:
        return ""
    h, w = mask.size
    payload = {"counts": mask.counts, "size": [int(h), int(w)]}
    return json.dumps(payload, sort_keys=True)
