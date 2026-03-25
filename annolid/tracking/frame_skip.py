"""Helpers for selecting/skipping frames during seeded Cutie tracking."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Dict, Iterable, List

from annolid.utils.files import has_frame_annotation


def build_seeded_frame_index(
    initial_frame: int,
    seeded_frame_candidates: Iterable[int],
) -> List[int]:
    """Return sorted unique seeded frame indices."""
    return sorted({int(initial_frame), *(int(f) for f in seeded_frame_candidates)})


def remove_seeded_frame(seeded_frames: List[int], frame_number: int) -> None:
    """Remove a seeded frame if present (in-place)."""
    frame_number = int(frame_number)
    idx = bisect.bisect_left(seeded_frames, frame_number)
    if idx < len(seeded_frames) and int(seeded_frames[idx]) == frame_number:
        seeded_frames.pop(idx)


def should_skip_finished_frame_between_adjacent_seeded_frames(
    *,
    frame_number: int,
    seeded_frames: List[int],
    video_result_folder: Path,
    finished_frame_cache: Dict[int, bool],
) -> bool:
    """Return True if frame is finished and lies strictly between adjacent seeds."""
    frame_number = int(frame_number)
    index = bisect.bisect_right(seeded_frames, frame_number)
    if index <= 0 or index >= len(seeded_frames):
        return False
    previous_seed = int(seeded_frames[index - 1])
    next_seed = int(seeded_frames[index])
    if not (previous_seed < frame_number < next_seed):
        return False

    cached = finished_frame_cache.get(frame_number)
    if cached is None:
        cached = has_frame_annotation(video_result_folder, frame_number)
        finished_frame_cache[frame_number] = bool(cached)
    return bool(cached)
