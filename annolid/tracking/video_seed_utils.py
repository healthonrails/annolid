"""Shared video seed resolution, mask lookup, and progress helpers for keypoint trackers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.cutie_mask_manager import MaskResult
from annolid.tracking.domain import InstanceRegistry
from annolid.utils.annotation_store import AnnotationStore
from annolid.utils.cancellation import is_stop_requested
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger

__all__ = [
    "apply_mask_results_to_registry",
    "collect_manual_annotation_frames",
    "find_latest_tracked_frame_before",
    "is_processor_stop_requested",
    "mask_lookup_from_registry",
    "normalize_tracking_frame_range",
    "report_processor_progress",
]


def normalize_tracking_frame_range(
    *,
    initial_frame: int,
    start_frame: Optional[int],
    end_frame: Optional[int],
    step: int,
    total_frames: int,
) -> Tuple[int, int, int]:
    """Normalize ``(start_frame, end_frame, step)`` against video bounds and seed frame."""
    resolved_start = (
        int(initial_frame)
        if start_frame is None
        else max(int(initial_frame), int(start_frame))
    )
    max_frame = max(0, int(total_frames) - 1)
    resolved_end = (
        max_frame if end_frame is None or int(end_frame) < 0 else int(end_frame)
    )
    resolved_end = min(resolved_end, max_frame)
    if resolved_start >= resolved_end:
        resolved_end = min(max_frame, resolved_start + 1)
    resolved_step = max(1, abs(int(step)))
    return resolved_start, resolved_end, resolved_step


def collect_manual_annotation_frames(
    video_result_folder: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> Dict[int, Path]:
    """Return manual PNG+JSON seed paths within ``[start_frame, end_frame]``."""
    manual_files = find_manual_labeled_json_files(str(video_result_folder))
    mapping: Dict[int, Path] = {}
    for filename in manual_files:
        try:
            frame_idx = int(get_frame_number_from_json(filename))
        except (ValueError, IndexError):
            logger.warning(
                "Skipping manual annotation with unexpected name: %s",
                filename,
            )
            continue
        if frame_idx < int(start_frame) or frame_idx > int(end_frame):
            continue
        mapping[frame_idx] = video_result_folder / filename
    return mapping


def find_latest_tracked_frame_before(
    video_result_folder: Path,
    *,
    target_frame: int,
    min_frame: int,
) -> Optional[Tuple[int, Path]]:
    """Find the most recent persisted frame strictly between ``min_frame`` and ``target_frame``.

    Checks both the SQLite ``AnnotationStore`` sidecar (when ``persist_json=False``)
    and per-frame LabelMe JSON files on disk so resuming tracking at ``target_frame``
    maintains frame-to-frame optical flow, Cutie memory, and motion continuity instead
    of jumping directly from an old manual seed at ``min_frame``.
    """
    upper = int(target_frame)
    lower = int(min_frame)
    if upper <= lower + 1:
        return None

    best_frame: Optional[int] = None
    probe_json = video_result_folder / f"{video_result_folder.name}_000000000.json"
    try:
        store = AnnotationStore.for_frame_path(probe_json)
        if store.store_path.exists():
            for frame_idx in store.iter_frames():
                idx = int(frame_idx)
                if lower < idx < upper and (best_frame is None or idx > best_frame):
                    best_frame = idx
    except Exception:
        logger.debug(
            "Unable to inspect AnnotationStore while resolving tracked resume frame.",
            exc_info=True,
        )

    try:
        for entry in os.scandir(video_result_folder):
            if not entry.is_file() or not entry.name.lower().endswith(".json"):
                continue
            try:
                idx = int(get_frame_number_from_json(entry.name))
            except (ValueError, IndexError):
                continue
            if lower < idx < upper and (best_frame is None or idx > best_frame):
                best_frame = idx
    except FileNotFoundError:
        return None

    if best_frame is None:
        return None
    candidate_path = (
        video_result_folder / f"{video_result_folder.name}_{best_frame:09d}.json"
    )
    return best_frame, candidate_path


def apply_mask_results_to_registry(
    registry: InstanceRegistry,
    mask_results: Dict[str, MaskResult],
) -> None:
    """Update ``InstanceState`` masks and polygons from Cutie ``MaskResult`` outputs."""
    for instance in registry:
        result = mask_results.get(instance.label)
        if not result:
            continue
        instance.set_mask(
            bitmap=result.mask_bitmap,
            polygon=result.polygon,
        )


def mask_lookup_from_registry(
    registry: InstanceRegistry,
    adapter: AnnotationAdapter,
) -> Dict[str, np.ndarray]:
    """Build boolean ``{instance_label: HxW mask}`` lookup from an ``InstanceRegistry``."""
    lookup: Dict[str, np.ndarray] = {}
    for instance in registry:
        mask = instance.mask_bitmap
        if mask is None and instance.polygon is not None:
            mask = adapter.mask_bitmap_from_polygon(instance.polygon)
        if mask is not None:
            lookup[instance.label] = mask.astype(bool)
    return lookup


def report_processor_progress(
    *,
    processed: int,
    total: int,
    pred_worker: Any = None,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Emit bounded percentage progress to GUI worker and optional analytics hook."""
    if total <= 0:
        return
    progress = int(min(100, max(0, (processed / total) * 100)))
    if pred_worker is not None and hasattr(pred_worker, "report_progress"):
        pred_worker.report_progress(progress)
    if progress_hook is not None:
        progress_hook(processed, total)


def is_processor_stop_requested(
    *,
    stop_requested: bool,
    pred_worker: Any = None,
    stop_event: Any = None,
) -> bool:
    """Check cooperative stop flag, worker stop signal, and threading event."""
    if stop_requested:
        return True
    if is_stop_requested(stop_event):
        return True
    if pred_worker is not None:
        worker_event = getattr(pred_worker, "stop_event", None)
        if is_stop_requested(worker_event):
            return True
        is_stopped = getattr(pred_worker, "is_stopped", None)
        if callable(is_stopped) and bool(is_stopped()):
            return True
    return False
