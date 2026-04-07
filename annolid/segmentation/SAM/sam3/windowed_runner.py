from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch


def normalize_window_schedule(
    *,
    window_size: int,
    stride: Optional[int],
) -> Tuple[int, int]:
    window_size = max(1, int(window_size or 1))
    try:
        stride_val = int(stride) if stride is not None else max(1, window_size - 1)
    except Exception:
        stride_val = max(1, window_size - 1)
    if stride_val <= 0:
        stride_val = max(1, window_size - 1)
    if window_size > 1 and stride_val >= window_size:
        stride_val = window_size - 1
    return window_size, stride_val


def resolve_window_schedule(
    *,
    resolved_device: torch.device,
    total_frames: Optional[int],
    user_window_size: Optional[int],
    user_stride: Optional[int],
) -> Tuple[int, int]:
    """
    Choose a device-aware window schedule for long-video prompts.
    """
    if user_window_size is not None or user_stride is not None:
        return normalize_window_schedule(
            window_size=user_window_size or 5,
            stride=user_stride,
        )

    total = max(0, int(total_frames or 0))
    if resolved_device.type == "cuda":
        window_size = 48 if total >= 200 else 24
        overlap = max(4, window_size // 6)
    elif resolved_device.type == "cpu":
        window_size = 32 if total >= 2000 else 24 if total >= 400 else 12
        overlap = max(2, window_size // 5)
    else:
        window_size = 8
        overlap = 1

    if total > 0:
        window_size = min(window_size, total)
    stride = max(1, window_size - overlap)
    return normalize_window_schedule(window_size=window_size, stride=stride)


def compute_window_reuse_shift(
    *,
    previous_window_end_idx: Optional[int],
    window_start_idx: int,
    frame_count: int,
    previous_window_frame_count: int,
) -> int:
    """
    Compute how many leading frames from the previous window can be reused.

    The reuse window is the actual overlap between consecutive windows, not the
    raw start index delta. This keeps non-overlapping windows from reusing stale
    frames while still allowing overlapping windows to amortize frame writes
    and private session-state carry-forward.
    """
    if previous_window_end_idx is None:
        return 0
    if frame_count <= 1 or previous_window_frame_count <= 1:
        return 0
    if frame_count != previous_window_frame_count:
        return 0
    overlap = int(previous_window_end_idx) - int(window_start_idx)
    if overlap <= 0:
        return 0
    return min(
        int(overlap),
        max(0, int(frame_count) - 1),
        max(0, int(previous_window_frame_count) - 1),
    )


def shift_annotations_to_window(
    annotations: Iterable[dict],
    start_idx: int,
    end_idx: int,
) -> Dict[int, List[dict]]:
    """
    Group annotations by local frame index within [start_idx, end_idx).
    Returned annotations have ann_frame_idx rewritten to local window coords.
    """
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        try:
            frame_idx = int(ann.get("ann_frame_idx", 0))
        except (TypeError, ValueError):
            continue
        if frame_idx < start_idx or frame_idx >= end_idx:
            continue
        local_idx = frame_idx - start_idx
        shifted = dict(ann)
        shifted["ann_frame_idx"] = local_idx
        grouped.setdefault(local_idx, []).append(shifted)
    return grouped


def first_manual_seed_frame(annotations: Iterable[dict]) -> Optional[int]:
    """
    Return the earliest manual seed frame in a label set, if present.
    """
    first_frame: Optional[int] = None
    for ann in annotations or []:
        if not isinstance(ann, dict):
            continue
        try:
            frame_idx = int(ann.get("ann_frame_idx", -1))
        except Exception:
            continue
        if frame_idx < 0:
            continue
        if first_frame is None or frame_idx < first_frame:
            first_frame = frame_idx
    return first_frame


def build_window_seed_segments(
    seed_frame_indices: Iterable[int],
    window_length: int,
    *,
    has_text_prompt: bool,
) -> List[Tuple[int, int]]:
    """
    Normalize seeded frames into ordered local segments for one window.
    """
    normalized = sorted(
        {
            int(idx)
            for idx in seed_frame_indices or []
            if idx is not None and 0 <= int(idx) < int(window_length)
        }
    )
    if not normalized and has_text_prompt:
        normalized = [0]

    segments: List[Tuple[int, int]] = []
    for idx, start_local in enumerate(normalized):
        next_local = normalized[idx + 1] if idx + 1 < len(normalized) else int(window_length)
        if int(next_local) <= int(start_local):
            continue
        segments.append((int(start_local), int(next_local)))
    return segments
