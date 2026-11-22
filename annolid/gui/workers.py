from __future__ import annotations

import asyncio
import functools
import gc
import json
import logging
import os
import sys
import threading
import inspect
from collections import deque
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from qtpy import QtCore, QtGui
from qtpy.QtCore import QThread, Signal, QObject

try:
    import zmq
except ImportError:  # Optional dependency (annolid[realtime])
    zmq = None  # type: ignore[assignment]

from annolid.data.videos import extract_frames_from_video
from annolid.gui.label_file import LabelFile
from annolid.jobs.tracking_jobs import TrackingSegment
from annolid.services.tracking import (
    build_tracking_video_processor,
    create_cutie_engine,
    create_segmented_cutie_executor,
    has_tracking_completion_artifacts,
    run_tracking_video_frames,
)
from annolid.infrastructure.filesystem import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.cancellation import ObservableStopEvent
from annolid.utils.logger import logger
from annolid.utils.devices import clear_device_cache
from annolid.utils.qt2cv import numpy_to_qimage

try:
    import qimage2ndarray
except ImportError:  # Optional dependency
    qimage2ndarray = None

try:
    import torch
except ImportError:  # PyTorch is optional for lighter desktop bundles
    torch = None


def _array_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """Convert a NumPy frame to an owned, detached QImage."""
    return numpy_to_qimage(frame, fallback_qimage2ndarray=qimage2ndarray)


class PredictionWorker(QObject):
    stop_signal = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stopped = False

    def is_stopped(self):
        return self._stopped

    def stop(self):
        self._stopped = True
        self.stop_signal.emit()


class TrackAllWorker(QThread):
    """Worker thread to process videos with Cutie tracking, ensuring one processor per video."""

    progress = Signal(int, str)  # Progress percentage and message
    finished = Signal(str)  # Completion message
    error = Signal(str)  # Error message

    # video_path, output_folder_path
    video_processing_started = Signal(str, str)
    # video_path (or output_folder_path)
    video_processing_finished = Signal(str)

    _SUPPORTED_VIDEO_SUFFIXES = frozenset(
        {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"}
    )

    def __init__(self, video_paths, config=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._active_pred_worker: Optional[PredictionWorker] = None
        self.video_paths = self._validate_video_paths(video_paths)
        default_config = {
            "mem_every": 5,
            "epsilon_for_polygon": 2.0,
            "t_max_value": 5,
            "use_cpu_only": False,
            "auto_recovery_missing_instances": True,
            "save_video_with_color_mask": False,
            "compute_optical_flow": True,
            "optical_flow_backend": "farneback",  # farneback | farneback_torch | raft
            "has_occlusion": True,
        }
        self.config = {**default_config, **(config or {})}
        self.is_running = True

    def _validate_video_paths(self, video_paths):
        """Validate video paths and return a list of valid paths."""
        if not video_paths:
            raise ValueError("No video paths provided.")
        valid_paths = []
        for path in video_paths:
            path = Path(path)
            if path.is_file() and path.suffix.lower() in self._SUPPORTED_VIDEO_SUFFIXES:
                valid_paths.append(str(path))
            else:
                self.logger.warning(f"Invalid or non-existent video path: {path}")
        if not valid_paths:
            raise ValueError("No valid video files provided.")
        return valid_paths

    def _get_video_fps(self, video_path: Path) -> float:
        """Best-effort FPS lookup for a video file."""
        fps = 0.0
        cap = cv2.VideoCapture(str(video_path))
        try:
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        finally:
            cap.release()
        if fps <= 0.0:
            self.logger.warning(
                f"Unable to determine FPS for {video_path.name}; defaulting to 0.0."
            )
        return fps

    def _load_saved_segments(self, video_path: Path) -> List[TrackingSegment]:
        """Load persisted segments for a video if available."""
        sidecar_path = video_path.with_suffix(video_path.suffix + ".segments.json")
        if not sidecar_path.exists():
            return []

        try:
            with open(sidecar_path, "r") as f:
                raw_segments = json.load(f)
        except Exception as exc:
            self.logger.error(
                f"Failed to read segments from {sidecar_path.name}: {exc}"
            )
            return []

        segments: List[TrackingSegment] = []
        fps_cache = None
        for entry in raw_segments or []:
            if not isinstance(entry, dict):
                continue
            entry = dict(entry)
            entry["video_path"] = str(video_path)
            if not entry.get("fps") or entry["fps"] <= 0:
                if fps_cache is None:
                    fps_cache = self._get_video_fps(video_path)
                entry["fps"] = fps_cache
            try:
                segments.append(TrackingSegment.from_dict(entry))
            except Exception as exc:
                self.logger.error(
                    f"Invalid segment entry in {sidecar_path.name}: {exc}"
                )
        return segments

    def select_device(self):
        """Select the appropriate device (CUDA, MPS, or CPU)."""
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for tracking. Install torch to enable this feature."
            )
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _process_segments_for_video(
        self,
        video_path: Path,
        segments: List[TrackingSegment],
        idx: int,
        total_videos: int,
        device: torch.device,
    ) -> bool:
        """Run tracking for pre-defined segments of a single video."""
        if not segments:
            return False

        video_name = video_path.stem
        output_folder = video_path.with_suffix("")
        output_folder.mkdir(exist_ok=True, parents=True)

        total_segments = len(segments)
        successful_segments = 0

        shared_cutie_engine = None
        cutie_engine_config_overrides = {
            "mem_every": self.config.get("mem_every", 5),
            "max_mem_frames": self.config.get("t_max_value", 5),
        }
        selected_weights = self.config.get("model_weights_path")
        if not selected_weights:
            try:
                from annolid.utils.body_part_shapes import (
                    folder_has_body_part_annotations,
                )

                if folder_has_body_part_annotations(output_folder):
                    selected_weights = "cutie-sam3-bodyparts.pth"
            except Exception:
                pass
        try:
            shared_cutie_engine = create_cutie_engine(
                cutie_config_overrides=cutie_engine_config_overrides,
                device=device,
                model_weights_path=selected_weights,
            )
        except Exception as exc:
            error_msg = f"Failed to initialize CutieEngine for {video_name}: {exc}"
            self.error.emit(error_msg)
            self.logger.error(error_msg, exc_info=True)
            return False

        for seg_idx, segment in enumerate(segments, start=1):
            if not self.is_running:
                self.logger.info(f"Segment processing interrupted for {video_name}.")
                break

            if not segment.is_annotation_valid():
                warning_msg = (
                    f"Skipping segment annotated at frame {segment.annotated_frame} "
                    f"for {video_name}: annotation JSON not found."
                )
                self.logger.warning(warning_msg)
                self.error.emit(warning_msg)
                continue

            progress_fraction = ((idx - 1) + (seg_idx / total_segments)) / max(
                total_videos, 1
            )
            progress_value = max(0, min(100, int(progress_fraction * 100)))
            status_msg = (
                f"Processing {video_name}: segment {seg_idx}/{total_segments} "
                f"({segment.segment_start_frame}-{segment.segment_end_frame})"
            )
            self.progress.emit(progress_value, status_msg)
            self.log_gpu_memory(video_name, f"Segment {seg_idx} - Before")

            segment_executor = None
            try:
                segment_executor = create_segmented_cutie_executor(
                    video_path_str=str(video_path),
                    segment_annotated_frame=segment.annotated_frame,
                    segment_start_frame=segment.segment_start_frame,
                    segment_end_frame=segment.segment_end_frame,
                    processing_config=self.config,
                    pred_worker=self,
                    device=device,
                    cutie_engine=shared_cutie_engine,
                )
                result_message = segment_executor.process_segment()
                if any(
                    keyword in result_message
                    for keyword in ["Error", "not found", "Failed"]
                ):
                    error_msg = (
                        f"Segment {seg_idx} for {video_name} failed: {result_message}"
                    )
                    self.error.emit(error_msg)
                    self.logger.error(error_msg)
                else:
                    successful_segments += 1
                    self.logger.info(
                        f"Segment {seg_idx} for {video_name} completed: {result_message}"
                    )
            except Exception as exc:
                error_msg = f"Unexpected exception while processing segment {seg_idx} for {video_name}: {exc}"
                self.error.emit(error_msg)
                self.logger.error(error_msg, exc_info=True)
            finally:
                if segment_executor is not None:
                    segment_executor.cleanup()
                self.log_gpu_memory(video_name, f"Segment {seg_idx} - After")

        if shared_cutie_engine is not None:
            try:
                shared_cutie_engine.cleanup()
            except Exception:
                pass

        if successful_segments > 0:
            if not self.is_running:
                self.progress.emit(
                    int(((idx - 1) / max(total_videos, 1)) * 100),
                    f"Stopped {video_name}.",
                )
                return False

            try:
                from annolid.annotation.labelme2csv import convert_json_to_csv

                convert_json_to_csv(str(output_folder))
            except Exception as exc:
                csv_error = f"CSV conversion failed for {video_name}: {exc}"
                self.error.emit(csv_error)
                self.logger.error(csv_error, exc_info=True)

        if total_segments > 0:
            completion_progress = min(100, int((idx / max(total_videos, 1)) * 100))
            if successful_segments == total_segments:
                completion_msg = (
                    f"Completed all {total_segments} segments for {video_name}."
                )
            else:
                completion_msg = f"Processed {successful_segments}/{total_segments} segments for {video_name}."
            self.progress.emit(completion_progress, completion_msg)

        return successful_segments == total_segments and successful_segments > 0

    def cleanup_processor(self, processor, device):
        """Clean up the processor and free device memory."""
        if processor is not None:
            if hasattr(processor, "cleanup"):
                try:
                    processor.cleanup()
                except Exception:
                    pass
            if processor.cutie_processor is not None:
                processor.cutie_processor = None
            del processor
        gc.collect()
        if torch is None:
            return
        clear_device_cache(torch_module=torch, device=device)

    def log_gpu_memory(self, video_name, stage):
        """Log device memory usage."""
        if torch is None:
            self.logger.info(
                f"No PyTorch runtime available ({stage} - {video_name}): Running on CPU"
            )
            return
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(
                f"GPU Memory (CUDA - {stage} - {video_name}): "
                f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / 1024**2
            self.logger.info(
                f"GPU Memory (MPS - {stage} - {video_name}): "
                f"Allocated: {allocated:.2f} MB"
            )
        else:
            self.logger.info(
                f"No GPU available ({stage} - {video_name}): Running on CPU"
            )

    def is_video_finished(self, video_path, total_frames):
        """Check if a video is already processed."""
        output_folder = Path(video_path).with_suffix("")
        return has_tracking_completion_artifacts(
            video_path=str(video_path),
            output_folder=str(output_folder),
            total_frames=int(total_frames),
        )

    def process_single_video(self, video_path, idx, total_videos):
        """Process a single video either via saved segments or whole-video tracking."""
        video_path = Path(video_path)
        video_path_str = str(video_path)
        video_name = video_path.stem
        output_folder = video_path.with_suffix("")
        output_folder.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Processing {video_name}: Output folder = {output_folder}")

        self.video_processing_started.emit(video_path_str, str(output_folder))

        candidate_devices = [self.select_device()]
        if candidate_devices[0].type == "cuda":
            candidate_devices.append(torch.device("cpu"))

        result = False
        final_error = None

        for attempt_index, device in enumerate(candidate_devices):
            if attempt_index > 0:
                self.logger.info(
                    f"Retrying {video_name} on {device.type.upper()} after GPU failure."
                )
            try:
                result = self._process_single_video_attempt(
                    video_path=video_path,
                    idx=idx,
                    total_videos=total_videos,
                    device=device,
                    output_folder=output_folder,
                )
                final_error = None
                break
            except RuntimeError as exc:
                if (
                    device.type == "cuda"
                    and self._should_retry_on_cpu(exc)
                    and attempt_index + 1 < len(candidate_devices)
                ):
                    final_error = str(exc)
                    self.logger.error(
                        f"CUDA failure while processing {video_name}: {exc}",
                        exc_info=True,
                    )
                    progress_value = max(
                        0, int(((idx - 1) / max(total_videos, 1)) * 100)
                    )
                    self.progress.emit(
                        progress_value,
                        f"{video_name}: CUDA failure detected, retrying on CPU.",
                    )
                    continue

                final_error = str(exc)
                self.logger.error(f"Processing error: {exc}", exc_info=True)
                break
            except Exception as exc:
                final_error = str(exc)
                self.logger.error(f"Processing error: {exc}", exc_info=True)
                break

        if final_error and not result:
            self.error.emit(f"Failed to process {video_name}: {final_error}")

        self.video_processing_finished.emit(video_path_str)
        return result

    def _process_single_video_attempt(
        self,
        video_path: Path,
        idx: int,
        total_videos: int,
        device: torch.device,
        output_folder: Path,
    ) -> bool:
        """Run the single-video pipeline on a specific device."""
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for tracking. Install torch to enable this feature."
            )

        video_name = video_path.stem
        self.logger.info(f"Using device: {device} for {video_name}")
        self.log_gpu_memory(video_name, f"Before ({device.type})")

        segments = self._load_saved_segments(video_path)
        processor = None

        try:
            if segments:
                return self._process_segments_for_video(
                    video_path,
                    segments,
                    idx,
                    total_videos,
                    device,
                )

            json_files = find_manual_labeled_json_files(str(output_folder))
            if not json_files:
                self.progress.emit(
                    int((idx / total_videos) * 100),
                    f"Skipping {video_name}: No JSON label file found.",
                )
                return False

            json_file = output_folder / sorted(json_files)[-1]
            try:
                labeled_frame_number = get_frame_number_from_json(str(json_file))
                label_file = LabelFile(str(json_file), is_video_frame=True)
                valid_shapes = [
                    shape
                    for shape in label_file.shapes
                    if shape.get("shape_type") == "polygon"
                    and len(shape.get("points", [])) >= 3
                ]
                if not valid_shapes:
                    self.progress.emit(
                        int((idx / total_videos) * 100),
                        f"Skipping {video_name}: JSON file {json_file.name} has no valid polygons (≥3 points).",
                    )
                    return False
            except Exception as exc:
                self.error.emit(f"Invalid JSON file for {video_name}: {str(exc)}")
                self.logger.error(
                    f"JSON error for {video_name}: {str(exc)}", exc_info=True
                )
                return False

            try:
                processor = build_tracking_video_processor(
                    video_path=str(video_path),
                    model_name="Cutie",
                    save_image_to_disk=False,
                    device=device,
                    results_folder=str(output_folder),
                    **self.config,
                )
                self.logger.info(
                    f"Initialized Cutie tracking processor for {video_name} with video_path: {video_path}"
                )
            except Exception as exc:
                self.error.emit(
                    f"Failed to initialize Cutie tracking processor for {video_name}: {exc}"
                )
                self.logger.error(
                    f"Initialization error for {video_name}: {exc}", exc_info=True
                )
                return False

            total_frames = processor.get_total_frames()
            if self.is_video_finished(str(video_path), total_frames):
                self.progress.emit(
                    int((idx / total_videos) * 100),
                    f"Skipping {video_name}: Video already processed.",
                )
                return False

            pred_worker = PredictionWorker()
            self._active_pred_worker = pred_worker
            if not self.is_running:
                pred_worker.stop()
            self.logger.info(
                f"Created new pred_worker for {video_name}: {id(pred_worker)}"
            )
            try:
                processor.video_path = str(video_path)
                processor.set_pred_worker(pred_worker)
            except Exception as exc:
                self.error.emit(
                    f"Failed to set pred_worker for {video_name}: {str(exc)}"
                )
                self.logger.error(f"pred_worker error: {str(exc)}", exc_info=True)
                return False

            try:
                processor.reset_cutie_processor(mem_every=self.config["mem_every"])
                self.logger.info(f"Reset Cutie processor for {video_name}")
            except AttributeError:
                self.logger.warning(
                    f"reset_cutie_processor not implemented for {video_name}"
                )
                processor.cutie_processor = None

            self.progress.emit(
                int((idx / total_videos) * 100),
                f"Processing {video_name}...",
            )
            start_frame = labeled_frame_number + 1
            end_frame = total_frames - 1
            self.logger.info(
                f"Processing {video_name} from frame {start_frame} to {end_frame}"
            )

            with torch.no_grad():
                message = run_tracking_video_frames(
                    processor=processor,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    config=self.config,
                )
            if "No valid polygon" in message or "No label file" in message:
                self.error.emit(f"Failed to process {video_name}: {message}")
                return False

            if not self.is_running:
                self.progress.emit(
                    int(((idx - 1) / max(total_videos, 1)) * 100),
                    f"Stopped {video_name}.",
                )
                return False

            try:
                from annolid.annotation.labelme2csv import convert_json_to_csv

                convert_json_to_csv(str(output_folder))
            except Exception as exc:
                self.error.emit(
                    f"Failed to convert JSON to CSV for {video_name}: {str(exc)}"
                )
                self.logger.error(f"CSV conversion error: {str(exc)}", exc_info=True)
                return False

            self.progress.emit(
                int((idx / total_videos) * 100),
                f"Completed tracking for {video_name}.",
            )
            return True

        finally:
            self._active_pred_worker = None
            self.cleanup_processor(processor, device)
            self.log_gpu_memory(video_name, f"After ({device.type})")

    @staticmethod
    def _should_retry_on_cpu(exc: RuntimeError) -> bool:
        """Determine if a CUDA failure warrants retrying on CPU."""
        message = str(exc).lower()
        keywords = (
            "cuda error",
            "cublas",
            "device-side assert",
            "out of memory",
        )
        return any(keyword in message for keyword in keywords)

    def run(self):
        """Process videos sequentially, ensuring one processor per video."""
        if torch is None:
            msg = "Track All requires PyTorch. Install torch to enable tracking."
            self.error.emit(msg)
            self.logger.error(msg)
            self.finished.emit("Track All aborted (missing PyTorch).")
            self.is_running = False
            return

        try:
            total_videos = len(self.video_paths)
            processed_videos = 0

            for idx, video_path in enumerate(self.video_paths, start=1):
                if not self.is_running:
                    self.finished.emit("Track All stopped by user.")
                    return

                # Process each video independently
                if self.process_single_video(video_path, idx, total_videos):
                    processed_videos += 1

            self.finished.emit(
                f"Track All completed. Processed {processed_videos}/{total_videos} videos."
            )
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        finally:
            self.is_running = False

    def stop(self):
        """Stop the worker thread."""
        self.is_running = False
        pred_worker = self._active_pred_worker
        if pred_worker is not None:
            pred_worker.stop()

    def is_stopped(self) -> bool:
        """Interface for segmented executors to query stop state."""
        return not self.is_running


class LoadFrameThread(QtCore.QObject):
    """
    Worker for loading video frames with FIFO playback ordering,
    memory-bounded LRU caching, generation-guarded video switching,
    and preemptible idle lookahead prefetching.
    """

    res_frame = QtCore.Signal(int, QtGui.QImage)
    frame_error = QtCore.Signal(int, str)
    process = QtCore.Signal()
    _prefetch_process = QtCore.Signal()

    _DEFAULT_CACHE_MAX_ENTRIES = 32
    _DEFAULT_CACHE_MAX_BYTES = 64 * 1024 * 1024
    _MIN_CACHE_ENTRIES = 1
    _DEFAULT_PREFETCH_AHEAD = 3

    @classmethod
    def _resolve_default_cache_bytes(cls) -> int:
        raw = os.environ.get("ANNOLID_FRAME_CACHE_MB", "").strip()
        if not raw:
            return cls._DEFAULT_CACHE_MAX_BYTES
        try:
            mb = float(raw)
        except ValueError:
            return cls._DEFAULT_CACHE_MAX_BYTES
        if mb <= 0:
            return cls._DEFAULT_CACHE_MAX_BYTES
        clamped_mb = max(4.0, min(mb, 1024.0))
        return int(clamped_mb * 1024 * 1024)

    def __init__(
        self,
        parent=None,
        *,
        cache_size: int = _DEFAULT_CACHE_MAX_ENTRIES,
        max_cache_bytes: Optional[int] = None,
        prefetch_enabled: bool = True,
        prefetch_ahead: int = _DEFAULT_PREFETCH_AHEAD,
    ):
        """Initialize the frame loader while maintaining original interface."""
        super().__init__(parent)

        # Cross-thread synchronization for queue/cache state.
        self.working_lock = Lock()
        self._video_loader = None

        # Bounded pending demand queue: FIFO for sequential playback,
        # cleared to latest-only for non-sequential scrubbing/jumps.
        self.frame_queue = deque(maxlen=4)
        # Memory-bounded LRU cache for recently decoded frames.
        self._frame_cache: "OrderedDict[int, QtGui.QImage]" = OrderedDict()
        self._timestamp_cache: Dict[int, Optional[float]] = {}
        self._cache_size = max(1, int(cache_size))
        resolved_bytes = (
            self._resolve_default_cache_bytes()
            if max_cache_bytes is None
            else int(max_cache_bytes)
        )
        self._max_cache_bytes = max(1, resolved_bytes)
        self._cache_bytes = 0
        self._last_frame = None  # Keep last frame for error recovery
        self._last_frame_bytes = 0
        self._shutting_down = False
        self._is_processing = False
        self._process_scheduled = False
        self._active_frame_number: Optional[int] = None
        self._generation = 0

        # Idle lookahead prefetch state (always preempted by demand requests).
        self._prefetch_enabled = bool(prefetch_enabled)
        self._prefetch_ahead = max(0, int(prefetch_ahead))
        self._prefetch_queue: deque[int] = deque()
        self._prefetch_scheduled = False
        self._last_requested_frame: Optional[int] = None
        self._last_request_step: int = 1

        # Ensure frame processing runs on the worker thread event loop.
        self.process.connect(self._optimized_load, type=QtCore.Qt.QueuedConnection)
        self._prefetch_process.connect(
            self._run_prefetch_step, type=QtCore.Qt.QueuedConnection
        )

    @property
    def video_loader(self):
        return self._video_loader

    @video_loader.setter
    def video_loader(self, loader) -> None:
        self.set_video_loader(loader, release_previous=False)

    def set_video_loader(self, loader, *, release_previous: bool = False) -> None:
        """Attach a new video loader and invalidate stale queue/cache state."""
        previous_loader = None
        with self.working_lock:
            if loader is self._video_loader:
                return
            previous_loader = self._video_loader
            self._video_loader = loader
            self._generation += 1
            self.frame_queue.clear()
            self._prefetch_queue.clear()
            self._frame_cache.clear()
            self._timestamp_cache.clear()
            self._cache_bytes = 0
            self._last_frame = None
            self._last_frame_bytes = 0
            self._last_requested_frame = None
            self._last_request_step = 1
        if (
            release_previous
            and previous_loader is not None
            and hasattr(previous_loader, "release")
        ):
            try:
                previous_loader.release()
            except Exception:
                logger.debug(
                    "Failed to release previous video loader on switch.",
                    exc_info=True,
                )

    def set_cache_limits(
        self,
        *,
        max_entries: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        """Update cache entry/byte limits and evict immediately if over budget."""
        with self.working_lock:
            if max_entries is not None:
                self._cache_size = max(1, int(max_entries))
            if max_bytes is not None:
                self._max_cache_bytes = max(1, int(max_bytes))
            self._evict_cache_locked()

    def trim_cache(self, *, keep_entries: int = 1) -> None:
        """Reclaim cached frame memory while keeping the `keep_entries` newest frames."""
        target = max(0, int(keep_entries))
        with self.working_lock:
            self._prefetch_queue.clear()
            while len(self._frame_cache) > target:
                evicted_key, evicted_img = self._frame_cache.popitem(last=False)
                self._timestamp_cache.pop(evicted_key, None)
                self._cache_bytes = max(
                    0, self._cache_bytes - self._qimage_byte_size(evicted_img)
                )
            if target == 0:
                self._last_frame = None
                self._last_frame_bytes = 0
                self._cache_bytes = 0
            elif self._frame_cache:
                newest = next(reversed(self._frame_cache.values()))
                self._last_frame = newest
                self._last_frame_bytes = self._qimage_byte_size(newest)

    def clear_cache(self) -> None:
        """Clear cached frames and cancel pending prefetch work."""
        with self.working_lock:
            self._generation += 1
            self._frame_cache.clear()
            self._timestamp_cache.clear()
            self._cache_bytes = 0
            self._prefetch_queue.clear()
            self._last_frame = None
            self._last_frame_bytes = 0

    def get_cached_frame(self, frame_number: int) -> Optional[QtGui.QImage]:
        """Return a cached QImage for `frame_number` without blocking on decode."""
        try:
            key = int(frame_number)
        except (TypeError, ValueError):
            return None
        with self.working_lock:
            cached = self._frame_cache.get(key)
            if cached is not None:
                self._frame_cache.move_to_end(key)
            return cached

    def get_cached_timestamp(self, frame_number: int) -> Optional[float]:
        """Return cached timestamp (ms) for `frame_number` if decoded."""
        try:
            key = int(frame_number)
        except (TypeError, ValueError):
            return None
        with self.working_lock:
            return self._timestamp_cache.get(key)

    @staticmethod
    def _qimage_byte_size(qimage: QtGui.QImage) -> int:
        if qimage is None:
            return 0
        if hasattr(qimage, "sizeInBytes"):
            try:
                return max(0, int(qimage.sizeInBytes()))
            except Exception:
                pass
        if hasattr(qimage, "byteCount"):
            try:
                return max(0, int(qimage.byteCount()))
            except Exception:
                pass
        try:
            return max(0, int(qimage.width()) * int(qimage.height()) * 4)
        except Exception:
            return 0

    def _evict_cache_locked(self) -> None:
        """Evict oldest cached frames until within entry-count and byte budgets."""
        max_entries = max(1, int(self._cache_size))
        min_entries = min(self._MIN_CACHE_ENTRIES, max_entries)
        while len(self._frame_cache) > max_entries or (
            self._cache_bytes > self._max_cache_bytes
            and len(self._frame_cache) > min_entries
        ):
            evicted_key, evicted_img = self._frame_cache.popitem(last=False)
            self._timestamp_cache.pop(evicted_key, None)
            self._cache_bytes = max(
                0, self._cache_bytes - self._qimage_byte_size(evicted_img)
            )
        if not self._frame_cache:
            self._cache_bytes = 0
        elif (
            self._last_frame is not None
            and self._last_frame_bytes * 2 > self._max_cache_bytes
            and self._last_frame not in self._frame_cache.values()
        ):
            # Avoid pinning an evicted oversized frame outside the byte budget.
            newest = next(reversed(self._frame_cache.values()))
            self._last_frame = newest
            self._last_frame_bytes = self._qimage_byte_size(newest)

    def _update_cache_locked(
        self,
        frame_number: int,
        qimage: QtGui.QImage,
        timestamp_ms: Optional[float] = None,
    ) -> None:
        """Update LRU frame cache while enforcing entry-count and byte budgets."""
        key = int(frame_number)
        existing = self._frame_cache.get(key)
        if existing is not None:
            self._cache_bytes = max(
                0, self._cache_bytes - self._qimage_byte_size(existing)
            )
        frame_bytes = self._qimage_byte_size(qimage)
        self._frame_cache[key] = qimage
        self._frame_cache.move_to_end(key)
        self._timestamp_cache[key] = timestamp_ms
        self._cache_bytes += frame_bytes
        if frame_bytes > 0:
            self._last_frame_bytes = frame_bytes
        self._evict_cache_locked()

    def _update_cache(
        self,
        frame_number: int,
        qimage: QtGui.QImage,
        timestamp_ms: Optional[float] = None,
    ):
        """Update frame cache with size and memory-budget management."""
        with self.working_lock:
            self._update_cache_locked(frame_number, qimage, timestamp_ms=timestamp_ms)

    def _enqueue_prefetch_locked(self, base_frame: int) -> bool:
        """Queue idle lookahead frames in the current navigation direction.

        Caller must hold `self.working_lock`. Returns True if `_prefetch_process`
        should be emitted outside the lock.
        """
        if (
            not self._prefetch_enabled
            or self._prefetch_ahead <= 0
            or self._shutting_down
            or self._video_loader is None
        ):
            return False
        total_frames_fn = getattr(self._video_loader, "total_frames", None)
        if not callable(total_frames_fn):
            return False
        try:
            total_frames = int(total_frames_fn())
        except Exception:
            return False
        if total_frames <= 1:
            return False

        max_entries = max(1, int(self._cache_size))
        affordable_prefetch_slots = max_entries - 1
        if self._last_frame_bytes > 0:
            affordable_total = self._max_cache_bytes // self._last_frame_bytes
            affordable_prefetch_slots = min(
                affordable_prefetch_slots, max(0, affordable_total - 1)
            )
        if affordable_prefetch_slots <= 0:
            # Avoid prefetching when a single frame leaves no room without
            # immediately evicting the currently displayed frame.
            self._prefetch_queue.clear()
            return False

        step = int(self._last_request_step or 1)
        if step == 0:
            step = 1
        max_steps = self._prefetch_ahead if step > 0 else min(1, self._prefetch_ahead)
        max_steps = min(max_steps, affordable_prefetch_slots)

        self._prefetch_queue.clear()
        for offset in range(1, max_steps + 1):
            candidate = int(base_frame) + (step * offset)
            if candidate < 0 or candidate >= total_frames:
                break
            if candidate in self._frame_cache:
                continue
            self._prefetch_queue.append(candidate)

        if self._prefetch_queue and not self._prefetch_scheduled:
            self._prefetch_scheduled = True
            return True
        return False

    def _decode_frame_from_loader(
        self, video_loader, frame_number: int
    ) -> Tuple[QtGui.QImage, Optional[float]]:
        frame = video_loader.load_frame(frame_number)
        timestamp_ms = None
        get_ts = getattr(video_loader, "get_time_stamp", None)
        if callable(get_ts):
            try:
                timestamp_ms = get_ts()
            except Exception:
                timestamp_ms = None
        qimage = _array_to_qimage(frame)
        return qimage, timestamp_ms

    def _optimized_load(self):
        """Decode the next pending demand frame with FIFO playback ordering."""
        with self.working_lock:
            self._process_scheduled = False
            if self._shutting_down or not self.frame_queue:
                return
            frame_number = int(self.frame_queue.popleft())
            video_loader = self._video_loader
            generation = self._generation
            self._is_processing = True
            self._active_frame_number = frame_number

        decoded_ok = False
        try:
            with self.working_lock:
                cached_frame = self._frame_cache.get(frame_number)
                if cached_frame is not None:
                    self._frame_cache.move_to_end(frame_number)

            if cached_frame is not None:
                decoded_ok = True
                self.res_frame.emit(frame_number, cached_frame)
            elif video_loader is None:
                logger.debug(
                    "Skipping frame %s load: no active video loader.", frame_number
                )
            else:
                qimage, timestamp_ms = self._decode_frame_from_loader(
                    video_loader, frame_number
                )
                should_emit = False
                with self.working_lock:
                    if not self._shutting_down and generation == self._generation:
                        self._update_cache_locked(
                            frame_number, qimage, timestamp_ms=timestamp_ms
                        )
                        self._last_frame = qimage
                        should_emit = True
                        decoded_ok = True
                if should_emit:
                    self.res_frame.emit(frame_number, qimage)

        except KeyError as e:
            if not self._shutting_down:
                logger.error(f"Error loading frame {frame_number}: {e}")
                self._handle_error(frame_number, str(e), generation=generation)
        except Exception as e:
            if not self._shutting_down:
                logger.error(f"Unexpected error loading frame {frame_number}: {e}")
                self._handle_error(frame_number, str(e), generation=generation)
        finally:
            trigger_next = False
            trigger_prefetch = False
            with self.working_lock:
                self._active_frame_number = None
                self._is_processing = False
                has_more = bool(self.frame_queue) and not self._shutting_down
                if has_more and not self._process_scheduled:
                    self._process_scheduled = True
                    trigger_next = True
                elif (
                    not has_more
                    and not self._shutting_down
                    and decoded_ok
                    and generation == self._generation
                ):
                    trigger_prefetch = self._enqueue_prefetch_locked(frame_number)

            if trigger_next:
                self.process.emit()
            elif trigger_prefetch:
                self._prefetch_process.emit()

    @QtCore.Slot()
    def _run_prefetch_step(self) -> None:
        """Decode at most one lookahead frame while idle; yield to demand requests."""
        with self.working_lock:
            self._prefetch_scheduled = False
            if (
                self._shutting_down
                or self._is_processing
                or self.frame_queue
                or not self._prefetch_queue
            ):
                return
            target_frame = None
            while self._prefetch_queue:
                candidate = int(self._prefetch_queue.popleft())
                if candidate not in self._frame_cache:
                    target_frame = candidate
                    break
            if target_frame is None:
                return
            video_loader = self._video_loader
            generation = self._generation
            if video_loader is None:
                return
            self._is_processing = True
            self._active_frame_number = target_frame

        try:
            qimage, timestamp_ms = self._decode_frame_from_loader(
                video_loader, target_frame
            )
            with self.working_lock:
                if not self._shutting_down and generation == self._generation:
                    self._update_cache_locked(
                        target_frame, qimage, timestamp_ms=timestamp_ms
                    )
        except Exception:
            logger.debug(
                "Idle prefetch skipped for frame %s.", target_frame, exc_info=True
            )
            with self.working_lock:
                self._prefetch_queue.clear()
        finally:
            trigger_demand = False
            trigger_more_prefetch = False
            with self.working_lock:
                self._active_frame_number = None
                self._is_processing = False
                if self.frame_queue and not self._shutting_down:
                    if not self._process_scheduled:
                        self._process_scheduled = True
                        trigger_demand = True
                elif (
                    self._prefetch_queue
                    and not self._shutting_down
                    and not self._prefetch_scheduled
                    and generation == self._generation
                ):
                    self._prefetch_scheduled = True
                    trigger_more_prefetch = True

            if trigger_demand:
                self.process.emit()
            elif trigger_more_prefetch:
                self._prefetch_process.emit()

    def _handle_error(
        self,
        frame_number: int,
        message: str = "",
        *,
        generation: Optional[int] = None,
    ):
        """Handle frame loading errors with fallback and diagnostic signal."""
        with self.working_lock:
            if self._shutting_down:
                return
            if generation is not None and generation != self._generation:
                return
            fallback_frame = self._last_frame
        if message:
            self.frame_error.emit(int(frame_number), str(message))
        if fallback_frame is not None:
            # Use last successful frame as fallback so canvas does not go blank.
            self.res_frame.emit(int(frame_number), fallback_frame)

    def request(self, frame_number):
        """Request a frame, serving cache hits immediately and coalescing scrubs."""
        if self._shutting_down:
            return

        frame_number = int(frame_number)
        should_trigger = False
        should_schedule_prefetch = False
        with self.working_lock:
            if self._shutting_down:
                return
            if (
                self._last_requested_frame is not None
                and frame_number != self._last_requested_frame
            ):
                delta = frame_number - self._last_requested_frame
                if 1 <= abs(delta) <= 16:
                    self._last_request_step = delta
            self._last_requested_frame = frame_number
            self._prefetch_queue.clear()

            # Serve immediately when already cached.
            cached_frame = self._frame_cache.get(frame_number)
            if cached_frame is not None:
                self._frame_cache.move_to_end(frame_number)
                # Drop any stale uncached demand requests so they cannot overwrite.
                self.frame_queue.clear()
                if not self._is_processing:
                    should_schedule_prefetch = self._enqueue_prefetch_locked(
                        frame_number
                    )
            else:
                is_processing = bool(self._is_processing)
                active_frame = (
                    int(self._active_frame_number)
                    if self._active_frame_number is not None
                    else None
                )
                is_sequential_playback = (
                    is_processing
                    and active_frame is not None
                    and frame_number > active_frame
                    and frame_number - active_frame <= 2
                )

                # During sequential playback, keep near-future requests in FIFO order
                # so decode stays smooth. For scrubbing/jumps, keep only the latest.
                if is_sequential_playback:
                    if frame_number not in self.frame_queue:
                        self.frame_queue.append(frame_number)
                else:
                    self.frame_queue.clear()
                    self.frame_queue.append(frame_number)

                if not is_processing and not self._process_scheduled:
                    self._process_scheduled = True
                    should_trigger = True

        if cached_frame is not None:
            self.res_frame.emit(frame_number, cached_frame)
            if should_schedule_prefetch:
                self._prefetch_process.emit()
            return

        if should_trigger:
            self.process.emit()

    @QtCore.Slot()
    def shutdown(self):
        """Flush pending work and mark loader for deletion."""
        with self.working_lock:
            self._shutting_down = True
            self._generation += 1
            self.frame_queue.clear()
            self._prefetch_queue.clear()
            self._frame_cache.clear()
            self._timestamp_cache.clear()
            self._cache_bytes = 0
            self._last_frame = None
            self._last_frame_bytes = 0
            self._active_frame_number = None
            self._is_processing = False
            self._process_scheduled = False
            self._prefetch_scheduled = False
            video_loader = self._video_loader
            self._video_loader = None
        if video_loader is not None and hasattr(video_loader, "release"):
            try:
                video_loader.release()
            except Exception:
                logger.debug(
                    "Failed to release video loader on shutdown.", exc_info=True
                )
        self.deleteLater()

    # Maintain original interface name for compatibility
    load = _optimized_load


class FlexibleWorker(QtCore.QObject):
    """
    A flexible worker class that runs a given function in a separate thread.
    Provides signals to indicate the start, progress, return value, and completion of the task.
    """

    start_signal = QtCore.Signal()
    finished_signal = QtCore.Signal(object)
    result_signal = QtCore.Signal(object)
    stop_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    preview_signal = QtCore.Signal(object)

    def __init__(self, task_function, *args, **kwargs):
        """
        Initialize the FlexibleWorker with the function to run and its arguments.

        :param task_function: The function to be executed.
        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        """
        super().__init__()
        self._task_function = task_function
        self._args = args
        self._kwargs = kwargs
        self._is_stopped = False
        self._running = False
        self._stop_pending_before_run = False
        self._stop_event = ObservableStopEvent()

        # Connect the stop signal to the stop method
        self.stop_signal.connect(self._stop)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def run(self):
        """
        Executes the task function with the provided arguments.
        Emits signals for result and completion when done.
        """
        result = None
        error = None
        self._running = True
        try:
            if self._stop_pending_before_run:
                self._stop_pending_before_run = False
                self._is_stopped = True
                self._stop_event.set()
                result = "Stopped"
            else:
                self._is_stopped = False
                self._stop_event.clear()
                kwargs = dict(self._kwargs)
                kwargs = self._inject_cancellation_kwargs(kwargs)
                result = self._task_function(*self._args, **kwargs)
        except Exception as exc:
            error = exc
        finally:
            self._running = False

        if error is not None:
            self._emit_if_alive(self.finished_signal, error)
            return
        self._emit_if_alive(self.result_signal, result)
        self._emit_if_alive(self.finished_signal, result)

    def _stop(self):
        """
        Stops the worker by setting the stop flag.
        """
        self._is_stopped = True
        if not self._running:
            self._stop_pending_before_run = True
        self._stop_event.set()
        self._request_stop_on_target()

    def is_stopped(self):
        """
        Check if the worker has been stopped.

        :return: True if the worker is stopped, otherwise False.
        """
        return self._is_stopped

    def request_stop(self):
        """Public helper to request a cooperative stop."""
        self._stop()

    def _iter_target_candidates(self):
        """Yield task callable, unwrapped partial functions, and bound owners."""
        seen: set[int] = set()
        curr = self._task_function
        while curr is not None and id(curr) not in seen:
            seen.add(id(curr))
            yield curr
            owner = getattr(curr, "__self__", None)
            if owner is not None and id(owner) not in seen:
                seen.add(id(owner))
                yield owner
            curr = curr.func if isinstance(curr, functools.partial) else None

    def allows_force_terminate(self) -> bool:
        """Return whether the target can be safely killed with QThread.terminate()."""
        for candidate in self._iter_target_candidates():
            value = getattr(candidate, "allow_force_thread_terminate", None)
            if value is not None:
                return bool(value)
            method = getattr(candidate, "allows_force_terminate", None)
            if callable(method):
                try:
                    return bool(method())
                except Exception:
                    logger.debug(
                        "FlexibleWorker target termination policy failed.",
                        exc_info=True,
                    )
        return True

    def _inject_cancellation_kwargs(self, kwargs: dict) -> dict:
        """Inject pred_worker/stop_event if the task function supports them."""
        try:
            sig = inspect.signature(self._task_function)
        except (TypeError, ValueError):
            return kwargs

        params = sig.parameters
        if "pred_worker" in params and "pred_worker" not in kwargs:
            kwargs["pred_worker"] = self
        if "stop_event" in params and "stop_event" not in kwargs:
            kwargs["stop_event"] = self._stop_event
        return kwargs

    def _request_stop_on_target(self) -> None:
        """If the task function exposes a stop API, invoke it."""
        for candidate in self._iter_target_candidates():
            request_stop = getattr(candidate, "request_stop", None)
            if callable(request_stop):
                try:
                    request_stop()
                except Exception:
                    logger.debug(
                        "FlexibleWorker target stop request failed.", exc_info=True
                    )
                return

    def report_progress(self, progress):
        """
        Reports the progress of the task.

        :param progress: An integer representing the progress percentage.
        """
        self._emit_if_alive(self.progress_signal, progress)

    def report_preview(self, payload):
        """Emit an intermediate preview/update payload to the GUI thread."""
        self._emit_if_alive(self.preview_signal, payload)

    def _emit_if_alive(self, signal, *args) -> bool:
        """Emit a signal if the QObject is still alive."""
        try:
            return self._safe_emit(signal, *args)
        except RuntimeError:
            logger.debug("FlexibleWorker signal emission failed (object deleted).")
            return False

    def _safe_emit(self, signal, *args) -> bool:
        """Emit a Qt signal defensively if the QObject is still alive."""
        try:
            signal.emit(*args)
            return True
        except RuntimeError:
            logger.debug("FlexibleWorker signal emission failed (object deleted).")
            return False


class FrameExtractorWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, videos, output_folder, num_frames=5):
        super().__init__()
        self.videos = list(videos or [])
        self.output_folder = output_folder
        self.num_frames = num_frames
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            total = len(self.videos)
            for idx, video in enumerate(self.videos, 1):
                if not self.is_running:
                    break
                extract_frames_from_video(
                    str(video),
                    self.output_folder,
                    num_frames=self.num_frames,
                )
                progress_value = int((idx / max(total, 1)) * 100)
                self.progress.emit(progress_value)
            self.finished.emit(self.output_folder)
        except Exception as e:
            self.error.emit(str(e))


class ProcessVideosWorker(QThread):
    progress = Signal(int)  # Signal to update progress
    finished = Signal(str)  # Signal when processing is complete
    error = Signal(str)  # Signal to handle errors

    def __init__(self, videos, agent, parent=None):
        super().__init__(parent)
        self.videos = videos
        self.agent = agent

    def run(self):
        try:
            total_videos = len(self.videos)
            for idx, video_path in enumerate(self.videos, start=1):
                try:
                    # Define user prompt
                    user_prompt = "Describe the main activities in this video."

                    # Process video with the behavior-video service facade.
                    from annolid.services.inference import run_behavior_video_agent

                    response = run_behavior_video_agent(
                        video_path, user_prompt, self.agent
                    )

                    # Save response to a text file
                    response_file = Path(video_path).with_suffix(".txt")
                    with open(response_file, "w") as f:
                        f.write(response)

                    # Emit progress
                    progress = int((idx / total_videos) * 100)
                    self.progress.emit(progress)

                except Exception as e:
                    self.error.emit(f"Error processing {video_path}: {str(e)}")

            # Notify completion
            self.finished.emit("Processing complete.")
        except Exception as e:
            self.error.emit(str(e))


class PerceptionProcessWorker(QtCore.QThread):
    """
    Background thread that runs the asynchronous PerceptionProcess inside its
    own event loop. Exposes a request_stop() helper so the GUI can trigger a
    graceful shutdown without blocking.
    """

    error = Signal(str)
    stopped = Signal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._perception = None
        self._shutdown_requested = threading.Event()

    def run(self):
        try:
            # macOS: avoid AVFoundation permission prompts from this worker thread.
            # Permission should be requested in the main app process.
            if sys.platform == "darwin":
                os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_main())
        except Exception as exc:
            logger.error("Realtime perception worker error: %s", exc, exc_info=True)
            self.error.emit(str(exc))
        finally:
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass
            if self._loop is not None:
                if not self._loop.is_closed():
                    self._loop.close()
            self._loop = None
            self._perception = None
            self.stopped.emit()

    async def _run_main(self):
        from annolid.realtime.perception import PerceptionProcess

        try:
            self._perception = PerceptionProcess(self.config)
        except Exception:
            self._perception = None
            raise

        try:
            await self._perception.run()
        finally:
            # Ensure resources are released even if run() exits unexpectedly.
            try:
                await self._perception.shutdown()
            except Exception as exc:
                logger.error("Error during perception shutdown: %s", exc, exc_info=True)

    def request_stop(self):
        """Signal the perception loop to terminate."""
        if self._shutdown_requested.is_set():
            return

        self._shutdown_requested.set()
        loop = self._loop
        perception = self._perception
        if loop and loop.is_running() and perception:
            # Promptly break the run loop from the GUI thread.
            loop.call_soon_threadsafe(perception.request_stop)


class RealtimeSubscriberWorker(QtCore.QThread):
    """
    Subscribes to perception PUB socket topics and emits Qt signals with the
    decoded payloads for consumption by the GUI thread.
    """

    frame_received = Signal(object, dict, list)
    status_received = Signal(dict)
    error = Signal(str)

    def __init__(self, address: str, parent=None):
        super().__init__(parent)
        self.address = address
        self._running = False
        self._context: Optional[Any] = None
        self._socket: Optional[Any] = None
        self._detections: Dict[Tuple[str, Optional[float]], List[dict]] = {}

    def _make_key(self, frame_index, capture_timestamp):
        if frame_index is not None:
            try:
                return ("idx", float(frame_index))
            except (TypeError, ValueError):
                pass
        if capture_timestamp is not None:
            try:
                return ("ts", round(float(capture_timestamp), 3))
            except (TypeError, ValueError):
                pass
        return ("na", None)

    def _prune_detections(self, max_entries: int = 256):
        if len(self._detections) <= max_entries:
            return
        keys = sorted(
            self._detections.keys(),
            key=lambda item: item[1]
            if isinstance(item[1], (int, float))
            else float("inf"),
        )
        for key in keys[: len(self._detections) - max_entries]:
            self._detections.pop(key, None)

    def _close(self):
        if self._socket is not None:
            try:
                if zmq is not None:
                    self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.close(0)
            except Exception:
                pass
            finally:
                self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            finally:
                self._context = None

    def stop(self):
        self._running = False

    def run(self):
        if zmq is None:
            msg = (
                "Realtime subscriber requires optional dependency 'pyzmq'. "
                'Install with: pip install "annolid[realtime]"'
            )
            logger.error(msg)
            self.error.emit(msg)
            return

        self._running = True
        self._detections = {}

        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self.address)
            for topic in ("frames", "detections", "status"):
                self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)

            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)

            while self._running:
                try:
                    events = dict(poller.poll(timeout=100))
                except zmq.error.ZMQError as exc:
                    if self._running:
                        logger.error(
                            "Realtime subscriber poll error: %s", exc, exc_info=True
                        )
                        self.error.emit(str(exc))
                    break

                if self._socket not in events:
                    continue

                try:
                    parts = self._socket.recv_multipart()
                except zmq.error.ZMQError as exc:
                    if self._running:
                        logger.error(
                            "Realtime subscriber recv error: %s", exc, exc_info=True
                        )
                        self.error.emit(str(exc))
                    break

                if not parts:
                    continue

                topic = parts[0].decode(errors="ignore")

                if topic == "detections":
                    self._handle_detection(parts)
                elif topic == "frames":
                    self._handle_frame(parts)
                elif topic == "status":
                    self._handle_status(parts)
        finally:
            self._close()

    def _handle_detection(self, parts):
        if len(parts) < 2:
            return
        try:
            payload = json.loads(parts[1].decode("utf-8"))
        except Exception as exc:
            logger.debug("Failed to decode detection payload: %s", exc)
            return

        metadata = payload.get("metadata") or {}
        key = self._make_key(
            metadata.get("frame_index"), metadata.get("capture_timestamp")
        )
        self._detections.setdefault(key, []).append(payload)
        self._prune_detections()

    def _handle_frame(self, parts):
        if len(parts) < 3:
            return
        try:
            metadata = json.loads(parts[1].decode("utf-8"))
        except Exception:
            metadata = {}

        frame_bytes = parts[2]
        qimage = None
        if frame_bytes:
            np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = frame_rgb.shape
                qimage = QtGui.QImage(
                    frame_rgb.data,
                    width,
                    height,
                    channels * width,
                    QtGui.QImage.Format_RGB888,
                ).copy()

        key = self._make_key(
            metadata.get("frame_index"), metadata.get("capture_timestamp")
        )
        detections = self._detections.pop(key, [])
        self.frame_received.emit(qimage, metadata, detections)
        self._prune_detections()

    def _handle_status(self, parts):
        if len(parts) < 2:
            return
        try:
            status = json.loads(parts[1].decode("utf-8"))
        except Exception as exc:
            logger.debug("Failed to decode status payload: %s", exc)
            return
        self.status_received.emit(status)
