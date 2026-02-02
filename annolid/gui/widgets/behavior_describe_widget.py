from __future__ import annotations

import base64
from bisect import bisect_right
import csv
import importlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import QRunnable
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QTimeEdit,
    QComboBox,
    QSpinBox,
)

from annolid.behavior import prompting as behavior_prompting
from annolid.behavior.timeline_sampling import (
    compute_timeline_points,
    format_hhmmss,
    timeline_intervals_to_timestamp_rows,
    timestamp_rows_to_timeline_intervals,
)
from annolid.core.media.video import CV2Video
from annolid.data.videos import extract_frames_from_video
from annolid.jobs.tracking_jobs import TrackingSegment

if TYPE_CHECKING:
    from annolid.gui.widgets.caption import CaptionWidget


logger = logging.getLogger(__name__)

# Default number of frames to sample per segment when not specified by UI
DEFAULT_BEHAVIOR_SAMPLE_COUNT = 5
DEFAULT_BEHAVIOR_TIMELINE_STEP_SECONDS = 1
DEFAULT_BEHAVIOR_TIMELINE_BATCH_SIZE = 12
TIMELINE_CONFIRM_THRESHOLD_WINDOWS = 1800  # ~30 minutes at 1 Hz
TIMELINE_SIDECAR_SUFFIX = ".behavior_timeline.json"
TIMESTAMP_CSV_SUFFIX = "_timestamps.csv"
TIMELINE_SIDECAR_VERSION = 1
SEGMENT_PRE_EXTRACT_LIMIT = 8
TIMELINE_IMAGE_FORMAT = ".jpg"
TIMELINE_JPEG_QUALITY = 85


@dataclass(frozen=True)
class BehaviorDescribeRequest:
    mode: str  # "summary" | "timeline"
    descriptor: str
    prompt: str
    start_time: QtCore.QTime
    end_time: QtCore.QTime
    start_frame: Optional[int]
    end_frame: Optional[int]
    sample_count: int
    timeline_step_seconds: int
    timeline_batch_size: int


class BehaviorDescribeWidget(QtWidgets.QWidget):
    """Encapsulates the describe-behavior workflow and UI."""

    def __init__(self, caption_widget: "CaptionWidget") -> None:
        super().__init__(caption_widget)
        self._caption = caption_widget
        self._behavior_buffer: str = ""
        self._behavior_segment_notes: str = ""
        self._segment_snapshot_paths: List[str] = []
        self._segment_snapshot_dirs: List[str] = []
        self._segment_snapshots: Dict[int, List[str]] = {}
        self._segment_snapshot_sample_count: Dict[int, int] = {}
        self._video_duration_ms: int = 0
        self._video_path: Optional[str] = None
        self._video_fps: float = 0.0
        self._video_num_frames: int = 0
        self._video_segments: List[TrackingSegment] = []
        self._last_segment_selection: Optional[int] = None
        self._pending_segment_extract: bool = False
        self._sample_count: int = DEFAULT_BEHAVIOR_SAMPLE_COUNT
        self._current_frame: Optional[int] = None
        self._timeline_intervals: List[Dict[str, Any]] = []
        self._timeline_interval_starts: List[int] = []
        self._timeline_points_by_timestamp: Dict[str, Dict[str, Any]] = {}
        self._timeline_loaded_video_path: Optional[str] = None

        self._button = caption_widget.create_button(
            icon_name="dialog-information",
            color="#6f42c1",
            hover_color="#4c2880",
        )
        self._button.setToolTip(
            "Describe behavior from a segment, or generate a per-step timeline over long ranges."
        )
        self._label = QLabel("Describe behavior")
        self._label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._button, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self._label, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(layout)

        self._button.clicked.connect(self._on_behavior_describe_clicked)
        self._set_idle_label()

    def set_sample_count(self, n: int) -> None:
        """Set the number of frames to sample from a segment.

        Values less than 1 will be coerced to 1, and extremely large values
        will be clamped to the total number of frames in the segment.
        """
        try:
            n = int(n)
        except Exception:
            n = DEFAULT_BEHAVIOR_SAMPLE_COUNT
        if n < 1:
            n = 1
        self._sample_count = n

    def set_current_frame(self, frame_index: Optional[int]) -> None:
        """Record the app's current frame so the dialog can use it."""
        if frame_index is None:
            self._current_frame = None
            return
        try:
            frame_index = int(frame_index)
        except Exception:
            return
        if frame_index < 0:
            return
        self._current_frame = frame_index
        self._apply_timeline_description_for_frame(frame_index)

    def set_video_context(
        self,
        video_path: Optional[str],
        fps: Optional[float],
        num_frames: Optional[int],
    ) -> None:
        self._video_path = video_path
        self._video_fps = float(fps or 0.0)
        self._video_num_frames = int(num_frames or 0)
        if self._video_fps > 0 and self._video_num_frames > 0:
            self._video_duration_ms = int(
                (self._video_num_frames / self._video_fps) * 1000
            )
        else:
            self._video_duration_ms = 0
        self._cleanup_segment_snapshots()
        logger.info(
            "BehaviorDescribeWidget context updated: path=%s fps=%s frames=%s",
            video_path,
            self._video_fps,
            self._video_num_frames,
        )
        # Reset presets when context changes
        if not video_path:
            self._clear_timeline_cache()
            self.set_video_segments([])
        else:
            self._load_timeline_from_sidecar()
            if self._video_segments and self._should_pre_extract_segments():
                self._pre_extract_segment_snapshots()
            elif self._pending_segment_extract and self._should_pre_extract_segments():
                self._pre_extract_segment_snapshots()
                self._pending_segment_extract = False
            if self._timeline_intervals:
                applied = False
                if self._current_frame is not None:
                    applied = self._apply_timeline_description_for_frame(
                        self._current_frame
                    )
                if not applied:
                    self._apply_timeline_description_for_frame(0)

    def set_video_segments(
        self, segments: Optional[Sequence["TrackingSegment"]]
    ) -> None:
        self._video_segments = list(segments or [])
        self._set_idle_label()
        self._last_segment_selection = 0 if self._video_segments else None
        logger.info(
            "BehaviorDescribeWidget loaded %s predefined segments for current video.",
            len(self._video_segments),
        )
        if (
            self._video_path
            and self._video_fps > 0
            and self._video_num_frames > 0
            and self._should_pre_extract_segments()
        ):
            self._pre_extract_segment_snapshots()
            self._pending_segment_extract = False
        else:
            self._pending_segment_extract = bool(self._video_segments)

    def _should_pre_extract_segments(self) -> bool:
        return len(self._video_segments) <= SEGMENT_PRE_EXTRACT_LIMIT

    def _pre_extract_segment_snapshots(self) -> None:
        self._cleanup_segment_snapshots()
        if (
            not self._video_path
            or not self._video_segments
            or self._video_fps <= 0
            or self._video_num_frames <= 0
        ):
            return
        self._segment_snapshots = {}
        for idx, segment in enumerate(self._video_segments):
            start_time = self._frame_to_time(segment.segment_start_frame)
            end_time = self._frame_to_time(segment.segment_end_frame + 1)
            start_frame = segment.segment_start_frame
            end_frame = segment.segment_end_frame
            frames = self._segment_frame_indices(
                start_time, end_time, start_frame, end_frame, self._sample_count
            )
            if not frames:
                continue
            snapshot_dir = self._create_snapshot_dir(min(frames), max(frames))
            try:
                pattern = f"{Path(self._video_path).stem}_{{frame:09d}}.png"
                images = extract_frames_from_video(
                    self._video_path,
                    snapshot_dir,
                    frame_indices=frames,
                    name_pattern=pattern,
                )
                if images:
                    self._segment_snapshot_dirs.append(snapshot_dir)
                    self._segment_snapshot_paths.extend(images)
                    self._segment_snapshots[idx] = images
                    self._segment_snapshot_sample_count[idx] = self._sample_count
                else:
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning("Failed to extract frames for segment %s: %s", idx, exc)
                shutil.rmtree(snapshot_dir, ignore_errors=True)

    def _create_snapshot_dir(self, seg_start_frame: int, seg_end_frame: int) -> str:
        """Create a stable output directory for a segment's snapshots.

        Pattern: <video_stem>/segment_<start_frame>_<end_frame>
        """
        if self._video_path:
            base = Path(self._video_path).with_suffix("")
            target = base / f"segment_{seg_start_frame}_{seg_end_frame}"
            target.mkdir(parents=True, exist_ok=True)
            logger.info("BehaviorDescribeWidget snapshot directory: %s", target)
            return str(target)
        # Fallback to a temp directory if no video context
        temp_dir = tempfile.mkdtemp(prefix="annolid_segment_frames_")
        logger.info("BehaviorDescribeWidget snapshot directory (temp): %s", temp_dir)
        return temp_dir

    def _cleanup_segment_snapshots(self) -> None:
        for dir_path in self._segment_snapshot_dirs:
            shutil.rmtree(dir_path, ignore_errors=True)
        self._segment_snapshot_dirs.clear()
        self._segment_snapshot_paths.clear()
        self._segment_snapshots.clear()
        self._segment_snapshot_sample_count.clear()

    def _clear_timeline_cache(self) -> None:
        self._set_timeline_intervals([])
        self._timeline_points_by_timestamp = {}
        self._timeline_loaded_video_path = None

    def _set_timeline_intervals(self, intervals: Sequence[Dict[str, Any]]) -> None:
        ordered = sorted(
            list(intervals),
            key=lambda item: int(item.get("start_frame", 0)),
        )
        self._timeline_intervals = ordered
        self._timeline_interval_starts = [
            int(item.get("start_frame", 0)) for item in ordered
        ]

    def _set_timeline_points(
        self, points: Sequence[Dict[str, Any]], end_frame: int
    ) -> None:
        mapping: Dict[str, Dict[str, Any]] = {}
        cleaned_points: List[Dict[str, Any]] = []
        for point in points:
            if not isinstance(point, dict):
                continue
            ts = str(point.get("timestamp", "")).strip()
            desc = str(point.get("description", "")).strip()
            if not ts:
                continue
            try:
                frame = int(point.get("frame"))
            except Exception:
                continue
            row = {"frame": frame, "timestamp": ts, "description": desc}
            mapping[ts] = row
            cleaned_points.append(row)
        cleaned_points.sort(
            key=lambda item: (int(item["frame"]), str(item["timestamp"]))
        )
        self._timeline_points_by_timestamp = mapping
        intervals = self._build_timeline_intervals(
            points=cleaned_points, end_frame=end_frame
        )
        self._set_timeline_intervals(intervals)

    def _timeline_sidecar_path(self) -> Optional[Path]:
        if not self._video_path:
            return None
        video = Path(self._video_path)
        return video.with_suffix(video.suffix + TIMELINE_SIDECAR_SUFFIX)

    def _timeline_timestamp_csv_path(self) -> Optional[Path]:
        if not self._video_path:
            return None
        video = Path(self._video_path)
        return video.parent / f"{video.stem}{TIMESTAMP_CSV_SUFFIX}"

    def _load_timeline_from_timestamp_csv(self) -> bool:
        csv_path = self._timeline_timestamp_csv_path()
        if csv_path is None or not csv_path.exists():
            return False
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            intervals = timestamp_rows_to_timeline_intervals(
                rows,
                fps=self._video_fps,
            )
            if not intervals:
                return False
            self._set_timeline_intervals(intervals)
            self._timeline_loaded_video_path = self._video_path
            logger.info(
                "BehaviorDescribeWidget loaded %s timeline intervals from %s",
                len(intervals),
                csv_path,
            )
            return True
        except Exception as exc:
            logger.warning(
                "BehaviorDescribeWidget failed to load timestamp CSV %s: %s",
                csv_path,
                exc,
            )
            return False

    def _load_timeline_from_sidecar(self) -> None:
        self._clear_timeline_cache()
        sidecar = self._timeline_sidecar_path()
        if sidecar is None or not sidecar.exists():
            self._load_timeline_from_timestamp_csv()
            return
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            raw_points = payload.get("points")
            if isinstance(raw_points, list):
                try:
                    end_frame = int(
                        payload.get("end_frame", max(0, self._video_num_frames - 1))
                    )
                except Exception:
                    end_frame = max(0, self._video_num_frames - 1)
                self._set_timeline_points(raw_points, end_frame=end_frame)
                if self._timeline_intervals:
                    self._timeline_loaded_video_path = self._video_path
                    logger.info(
                        "BehaviorDescribeWidget loaded %s timeline points from %s",
                        len(self._timeline_points_by_timestamp),
                        sidecar,
                    )
                    return
            entries = payload.get("entries")
            if not isinstance(entries, list):
                return
            intervals: List[Dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    start_frame = int(entry.get("start_frame", -1))
                    end_frame = int(entry.get("end_frame", -1))
                except Exception:
                    continue
                if start_frame < 0 or end_frame < start_frame:
                    continue
                text = str(entry.get("description", "")).strip()
                if not text:
                    continue
                intervals.append(
                    {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time": str(entry.get("start_time", "")),
                        "end_time": str(entry.get("end_time", "")),
                        "description": text,
                    }
                )
            if intervals:
                self._set_timeline_intervals(intervals)
                self._timeline_loaded_video_path = self._video_path
                logger.info(
                    "BehaviorDescribeWidget loaded %s timeline intervals from %s",
                    len(intervals),
                    sidecar,
                )
        except Exception as exc:
            logger.warning(
                "BehaviorDescribeWidget failed to load timeline sidecar %s: %s",
                sidecar,
                exc,
            )

    def _save_timeline_to_sidecar(
        self,
        *,
        intervals: Sequence[Dict[str, Any]],
        points: Sequence[Dict[str, Any]],
        end_frame: int,
        step_seconds: int,
        prompt: str,
    ) -> None:
        sidecar = self._timeline_sidecar_path()
        if sidecar is None:
            return
        payload = {
            "version": TIMELINE_SIDECAR_VERSION,
            "video_path": self._video_path,
            "fps": self._video_fps,
            "num_frames": self._video_num_frames,
            "step_seconds": int(step_seconds),
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "prompt": prompt,
            "end_frame": int(end_frame),
            "points": list(points),
            "entries": list(intervals),
        }
        try:
            sidecar.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "BehaviorDescribeWidget saved %s timeline intervals to %s",
                len(intervals),
                sidecar,
            )
        except Exception as exc:
            logger.warning(
                "BehaviorDescribeWidget failed to save timeline sidecar %s: %s",
                sidecar,
                exc,
            )
        self._save_timeline_to_timestamp_csv(intervals)

    def _save_timeline_to_timestamp_csv(
        self, intervals: Sequence[Dict[str, Any]]
    ) -> None:
        csv_path = self._timeline_timestamp_csv_path()
        if csv_path is None:
            return
        rows = timeline_intervals_to_timestamp_rows(
            intervals,
            fps=self._video_fps,
            subject="Subject 1",
        )
        if not rows:
            return
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["Trial time", "Recording time", "Subject", "Behavior", "Event"]
                )
                for row in rows:
                    writer.writerow(row)
            logger.info(
                "BehaviorDescribeWidget exported %s timestamp rows to %s",
                len(rows),
                csv_path,
            )
        except Exception as exc:
            logger.warning(
                "BehaviorDescribeWidget failed to save timestamp CSV %s: %s",
                csv_path,
                exc,
            )

    @staticmethod
    def _parse_timestamped_descriptions(text: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        pattern = re.compile(r"^\s*(\d{2}:\d{2}:\d{2})\s*:\s*(.+?)\s*$")
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                parsed[match.group(1)] = match.group(2).strip()
        return parsed

    def _build_timeline_intervals(
        self,
        *,
        points: Sequence[Dict[str, Any]],
        end_frame: int,
    ) -> List[Dict[str, Any]]:
        intervals: List[Dict[str, Any]] = []
        if not points:
            return intervals
        for idx, point in enumerate(points):
            try:
                start_frame = int(point.get("frame"))
            except Exception:
                continue
            next_frame = end_frame
            if idx + 1 < len(points):
                try:
                    next_frame = int(points[idx + 1].get("frame")) - 1
                except Exception:
                    next_frame = end_frame
            start_frame = max(0, start_frame)
            next_frame = max(start_frame, min(next_frame, end_frame))
            description = str(point.get("description", "")).strip()
            if not description:
                continue
            intervals.append(
                {
                    "start_frame": start_frame,
                    "end_frame": next_frame,
                    "start_time": str(point.get("timestamp", "")),
                    "end_time": (
                        str(
                            points[idx + 1].get("timestamp", point.get("timestamp", ""))
                        )
                        if idx + 1 < len(points)
                        else str(point.get("timestamp", ""))
                    ),
                    "description": description,
                }
            )
        return intervals

    def _timeline_interval_for_frame(
        self, frame_index: int
    ) -> Optional[Dict[str, Any]]:
        if not self._timeline_intervals or not self._timeline_interval_starts:
            return None
        pos = bisect_right(self._timeline_interval_starts, frame_index) - 1
        if pos < 0:
            return None
        interval = self._timeline_intervals[pos]
        if interval["start_frame"] <= frame_index <= interval["end_frame"]:
            return interval
        return None

    def _apply_timeline_description_for_frame(self, frame_index: int) -> bool:
        if frame_index is None or frame_index < 0:
            return False
        interval = self._timeline_interval_for_frame(frame_index)
        if not interval:
            return False
        description = str(interval.get("description", "")).strip()
        if not description:
            return False
        current_text = (self._caption.text_edit.toPlainText() or "").strip()
        if current_text == description:
            return True
        self._caption._allow_empty_caption = True
        self._caption.set_caption(description)
        self._caption._allow_empty_caption = False
        return True

    def apply_timeline_description(
        self,
        frame_index: Optional[int] = None,
        *,
        only_if_empty: bool = False,
    ) -> bool:
        """Try to apply cached timeline description to caption UI."""
        if only_if_empty:
            current_text = (self._caption.text_edit.toPlainText() or "").strip()
            if current_text:
                return False
        target = self._current_frame if frame_index is None else frame_index
        if target is None:
            target = 0
        return self._apply_timeline_description_for_frame(int(target))

    def _sorted_timeline_points(self) -> List[Dict[str, Any]]:
        points = list(self._timeline_points_by_timestamp.values())
        points.sort(
            key=lambda item: (int(item.get("frame", 0)), str(item.get("timestamp", "")))
        )
        return points

    @QtCore.Slot(str)
    def merge_timeline_result(self, payload_json: str) -> None:
        if not payload_json:
            return
        try:
            payload = json.loads(payload_json)
        except Exception:
            return
        if payload.get("video_path") != self._video_path:
            return
        points = payload.get("points") or []
        if not isinstance(points, list):
            return
        try:
            end_frame = int(
                payload.get("end_frame", max(0, self._video_num_frames - 1))
            )
        except Exception:
            end_frame = max(0, self._video_num_frames - 1)
        merged = dict(self._timeline_points_by_timestamp)
        for point in points:
            if not isinstance(point, dict):
                continue
            ts = str(point.get("timestamp", "")).strip()
            if not ts:
                continue
            desc = str(point.get("description", "")).strip()
            if not desc:
                continue
            try:
                frame = int(point.get("frame"))
            except Exception:
                continue
            merged[ts] = {"frame": frame, "timestamp": ts, "description": desc}
        self._timeline_points_by_timestamp = merged
        sorted_points = self._sorted_timeline_points()
        self._set_timeline_points(sorted_points, end_frame=end_frame)
        self._timeline_loaded_video_path = self._video_path
        if self._timeline_intervals:
            self._save_timeline_to_sidecar(
                intervals=self._timeline_intervals,
                points=sorted_points,
                end_frame=end_frame,
                step_seconds=int(payload.get("step_seconds", 1)),
                prompt=str(payload.get("prompt", "")).strip(),
            )
            if self._current_frame is not None:
                self._apply_timeline_description_for_frame(self._current_frame)

    def _set_idle_label(self) -> None:
        suffix = ""
        if self._video_segments:
            suffix = f" ({len(self._video_segments)} segments)"
        self._label.setText(f"Describe behavior{suffix}")

    def _ensure_video_context(self) -> None:
        if self._video_path and self._video_fps > 0 and self._video_num_frames > 0:
            return
        # Try caption-provided context first
        if hasattr(self._caption, "video_context"):
            video_path, fps, num_frames = self._caption.video_context()
            logger.info(
                "BehaviorDescribeWidget attempting context rehydrate from caption: path=%s fps=%s frames=%s",
                video_path,
                fps,
                num_frames,
            )
            if video_path:
                self.set_video_context(video_path, fps, num_frames)
                if (
                    self._video_path
                    and self._video_fps > 0
                    and self._video_num_frames > 0
                ):
                    return
        # Fallback: infer from current image path on disk
        try:
            img_path = getattr(self._caption, "get_image_path", lambda: None)()
        except Exception:
            img_path = None
        if not img_path:
            return
        try:
            from annolid.data import videos as videos_mod

            img_path = Path(img_path)
            base_dir = img_path.parent  # e.g., <video_stem>
            video_stem = base_dir.name
            search_dir = base_dir.parent
            candidates = []
            exts = (".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".m4v")
            for ext in exts:
                cand = search_dir / f"{video_stem}{ext}"
                if cand.exists():
                    candidates.append(cand)
            if not candidates:
                # Try within base_dir as a fallback
                for ext in exts:
                    cand = base_dir / f"{video_stem}{ext}"
                    if cand.exists():
                        candidates.append(cand)
            if not candidates:
                logger.info(
                    "BehaviorDescribeWidget could not infer video from image path: %s",
                    img_path,
                )
                return
            real_video = str(candidates[0])
            try:
                cv2v = videos_mod.CV2Video(real_video)
                fps = float(cv2v.get_fps() or 0.0)
                total = int(cv2v.total_frames() or 0)
            except Exception:
                fps = 0.0
                total = 0
            self.set_video_context(real_video, fps, total)
            logger.info(
                "BehaviorDescribeWidget inferred video context from image: path=%s fps=%.2f frames=%s",
                real_video,
                fps,
                total,
            )
        except Exception as exc:
            logger.debug("BehaviorDescribeWidget video inference failed: %s", exc)

    def _default_segment_times(self) -> Tuple[QtCore.QTime, QtCore.QTime]:
        start = QtCore.QTime(0, 0, 0)
        if self._video_duration_ms > 0:
            clamped_ms = min(self._video_duration_ms, (24 * 60 * 60 * 1000) - 1)
            end = start.addMSecs(clamped_ms)
        else:
            end = start
        if not end.isValid():
            end = QtCore.QTime(23, 59, 59)
        return start, end

    def _prompt_behavior_segment(
        self,
    ) -> Optional["BehaviorDescribeRequest"]:
        start_time, end_time = self._default_segment_times()
        default_descriptor = BehaviorSegmentDialog.compose_descriptor(
            start_time, end_time, self._behavior_segment_notes or ""
        )
        default_prompt = behavior_prompting.build_behavior_narrative_prompt(
            segment_label=default_descriptor,
        )
        segment_presets = self._segment_presets()
        initial_index = self._last_segment_selection or 0
        dialog = BehaviorSegmentDialog(
            parent=self._caption,
            start_time=start_time,
            end_time=end_time,
            notes=self._behavior_segment_notes,
            prompt_text=default_prompt,
            segment_presets=segment_presets,
            video_fps=self._video_fps,
            total_frames=self._video_num_frames,
            initial_preset_index=initial_index,
            current_frame=self._current_frame,
            sample_count_default=self._sample_count,
            timeline_step_default=DEFAULT_BEHAVIOR_TIMELINE_STEP_SECONDS,
            timeline_batch_default=DEFAULT_BEHAVIOR_TIMELINE_BATCH_SIZE,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        self._behavior_segment_notes = dialog.notes()
        if segment_presets:
            selected_preset = dialog.selected_preset_index()
            if selected_preset is not None:
                self._last_segment_selection = selected_preset
        descriptor = dialog.segment_descriptor()
        prompt_text = dialog.prompt_text().strip()
        if not prompt_text:
            prompt_text = behavior_prompting.build_behavior_narrative_prompt(
                segment_label=descriptor
            )
        return BehaviorDescribeRequest(
            mode=dialog.run_mode(),
            descriptor=descriptor,
            prompt=prompt_text,
            start_time=dialog.start_time(),
            end_time=dialog.end_time(),
            start_frame=dialog.start_frame_value(),
            end_frame=dialog.end_frame_value(),
            sample_count=dialog.sample_count(),
            timeline_step_seconds=dialog.timeline_step_seconds(),
            timeline_batch_size=dialog.timeline_batch_size(),
        )

    def _segment_presets(self) -> List[Dict[str, Any]]:
        if not self._video_segments or self._video_fps <= 0:
            return []
        presets: List[Dict[str, Any]] = []
        for idx, segment in enumerate(self._video_segments, start=1):
            start = self._frame_to_time(segment.segment_start_frame)
            # Add 1 frame to make the end time inclusive.
            end = self._frame_to_time(segment.segment_end_frame + 1)
            label = f"{idx}. {start.toString('HH:mm:ss')}–{end.toString('HH:mm:ss')}"
            note = f"Segment {idx}, frames {segment.segment_start_frame}-{segment.segment_end_frame}"
            presets.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "notes": note,
                    "start_frame": segment.segment_start_frame,
                    "end_frame": segment.segment_end_frame,
                }
            )
        return presets

    def _frame_to_time(self, frame_index: int) -> QtCore.QTime:
        if self._video_fps <= 0:
            return QtCore.QTime(0, 0, 0)
        total_msecs = int(round((frame_index / self._video_fps) * 1000))
        max_msecs = (24 * 60 * 60 * 1000) - 1
        clamped = max(0, min(total_msecs, max_msecs))
        base = QtCore.QTime(0, 0, 0)
        return base.addMSecs(clamped)

    def _time_to_frame_index(self, time_val: QtCore.QTime) -> int:
        if not time_val.isValid() or self._video_fps <= 0:
            return 0
        msecs = time_val.msecsSinceStartOfDay()
        frame = int(round((msecs / 1000.0) * self._video_fps))
        return max(0, min(frame, max(0, self._video_num_frames - 1)))

    def _segment_frame_indices(
        self,
        start_time: QtCore.QTime,
        end_time: QtCore.QTime,
        start_frame_override: Optional[int] = None,
        end_frame_override: Optional[int] = None,
        sample_count: Optional[int] = None,
    ) -> List[int]:
        """Compute uniformly sampled, inclusive frame indices for a segment.

        Prefers user-provided frame overrides; otherwise derives from times.
        Guarantees the list is non-empty (when the range is valid) and sorted.
        """
        if self._video_fps <= 0 or self._video_num_frames <= 0:
            return []
        # Determine bounds
        if (
            start_frame_override is not None
            and end_frame_override is not None
            and 0 <= start_frame_override < self._video_num_frames
            and 0 <= end_frame_override
        ):
            start_frame = start_frame_override
            end_frame = min(end_frame_override, self._video_num_frames - 1)
            logger.info(
                "BehaviorDescribeWidget using user-defined frames %s-%s",
                start_frame,
                end_frame,
            )
        else:
            start_frame = self._time_to_frame_index(start_time)
            end_frame = self._time_to_frame_index(end_time)
            logger.info(
                "BehaviorDescribeWidget derived frame range %s-%s from times %s→%s",
                start_frame,
                end_frame,
                start_time.toString("HH:mm:ss"),
                end_time.toString("HH:mm:ss"),
            )
        # Normalize
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        start_frame = max(0, min(start_frame, self._video_num_frames - 1))
        end_frame = max(0, min(end_frame, self._video_num_frames - 1))
        # Single-frame segment
        if start_frame == end_frame:
            logger.info(
                "BehaviorDescribeWidget single-frame segment at %s", start_frame
            )
            return [start_frame]
        # Compute samples
        n = (
            sample_count
            if isinstance(sample_count, int) and sample_count > 0
            else self._sample_count
        )
        n = max(2, min(n, (end_frame - start_frame + 1)))  # inclusive
        if n == (end_frame - start_frame + 1):
            frames = list(range(start_frame, end_frame + 1))
        else:
            step = (end_frame - start_frame) / (n - 1)
            frames = [int(round(start_frame + i * step)) for i in range(n)]
        frames = sorted(set(max(start_frame, min(end_frame, f)) for f in frames))
        logger.info("BehaviorDescribeWidget sampled frames (%s): %s", n, frames)
        return frames

    def _resolve_behavior_image_paths(
        self,
        start_time: QtCore.QTime,
        end_time: QtCore.QTime,
        start_frame: Optional[int],
        end_frame: Optional[int],
        segment_index: Optional[int] = None,
        *,
        sample_count: Optional[int] = None,
    ) -> List[str]:
        if segment_index is not None:
            cached_count = self._segment_snapshot_sample_count.get(segment_index)
            use_cache = not (
                sample_count is not None
                and cached_count is not None
                and cached_count != sample_count
            )
            if not use_cache:
                logger.info(
                    "Ignoring cached snapshots for segment %s due to sample-count change (%s→%s).",
                    segment_index,
                    cached_count,
                    sample_count,
                )
            if use_cache:
                pre_paths = [
                    path
                    for path in self._segment_snapshots.get(segment_index, [])
                    if os.path.exists(path)
                ]
                if pre_paths:
                    logger.info(
                        "Using %s pre-extracted snapshots for segment %s.",
                        len(pre_paths),
                        segment_index,
                    )
                    return pre_paths
                elif segment_index in self._segment_snapshots:
                    logger.info(
                        "Cached snapshots for segment %s missing on disk; regenerating.",
                        segment_index,
                    )
        if self._video_path and self._video_fps > 0 and self._video_num_frames > 0:
            frames = self._segment_frame_indices(
                start_time,
                end_time,
                start_frame,
                end_frame,
                sample_count if sample_count is not None else self._sample_count,
            )
            if frames:
                seg_dir = self._create_snapshot_dir(min(frames), max(frames))
                try:
                    # File pattern: <video_stem>_<frame_number>.png
                    pattern = (
                        f"{Path(self._video_path).stem}_{{frame:09d}}.png"
                        if self._video_path
                        else "{video_stem}_{frame:09d}.png"
                    )
                    images: List[str] = extract_frames_from_video(
                        self._video_path,
                        seg_dir,
                        frame_indices=frames,
                        name_pattern=pattern,
                    )
                except Exception as exc:
                    logger.warning(
                        "BehaviorDescribeWidget failed to extract frames %s: %s",
                        frames,
                        exc,
                    )
                    shutil.rmtree(seg_dir, ignore_errors=True)
                else:
                    if images:
                        self._segment_snapshot_dirs.append(seg_dir)
                        self._segment_snapshot_paths.extend(images)
                        if segment_index is not None:
                            self._segment_snapshots[segment_index] = list(images)
                            self._segment_snapshot_sample_count[segment_index] = (
                                sample_count
                                if sample_count is not None
                                else self._sample_count
                            )
                        logger.info(
                            "BehaviorDescribeWidget extracted %s frame snapshots for behavior prompt at %s.",
                            len(images),
                            seg_dir,
                        )
                        return images
                    shutil.rmtree(seg_dir, ignore_errors=True)
            else:
                logger.info(
                    "BehaviorDescribeWidget computed empty frame list for segment; falling back."
                )
        else:
            logger.warning(
                "BehaviorDescribeWidget missing sufficient video context (path=%s fps=%.2f frames=%s)",
                self._video_path,
                self._video_fps,
                self._video_num_frames,
            )
        logger.info(
            "BehaviorDescribeWidget falling back to single image (video_path=%s, fps=%.2f, frames=%s).",
            self._video_path,
            self._video_fps,
            self._video_num_frames,
        )
        fallback = self._caption._resolve_image_for_description()
        return [fallback] if fallback else []

    @QtCore.Slot(str)
    def set_status_label(self, text: str) -> None:
        self._label.setText(text)

    @QtCore.Slot()
    def finish_behavior_run(self) -> None:
        """Re-enable UI without overwriting the caption contents."""
        self._set_idle_label()
        self._button.setEnabled(True)
        self._caption._allow_empty_caption = False

    def _on_behavior_describe_clicked(self) -> None:
        self._ensure_video_context()
        caption = self._caption
        if caption.selected_provider != "ollama":
            self.update_behavior_status(
                "Behavior description currently supports Ollama models. Please switch provider.",
                True,
            )
            return

        image_path = caption._resolve_image_for_description()
        if not image_path:
            self.update_behavior_status(
                "No image available for behavior description. Load or draw on the canvas first.",
                True,
            )
            return

        if not caption.selected_model:
            self.update_behavior_status(
                "Select an Ollama model (e.g., qwen3-vl) before describing behavior.",
                True,
            )
            return

        request = self._prompt_behavior_segment()
        if request is None:
            self._set_idle_label()
            return
        logger.info(
            "BehaviorDescribeWidget segment request: times %s → %s, frames %s → %s",
            request.start_time.toString("HH:mm:ss"),
            request.end_time.toString("HH:mm:ss"),
            request.start_frame,
            request.end_frame,
        )

        self._label.setText("Describing…")
        self._button.setEnabled(False)
        if request.mode == "timeline":
            self.begin_behavior_stream(
                f"Describing behavior timeline (every {request.timeline_step_seconds}s)…\n"
            )
        else:
            self.begin_behavior_stream()

        if request.mode == "timeline":
            if (
                not self._video_path
                or self._video_fps <= 0
                or self._video_num_frames <= 0
            ):
                self.update_behavior_status(
                    "Timeline mode requires an active video context. Load a video first.",
                    True,
                )
                return
            task = BehaviorTimelineTask(
                video_path=self._video_path,
                fps=self._video_fps,
                total_frames=self._video_num_frames,
                start_time=request.start_time,
                end_time=request.end_time,
                step_seconds=request.timeline_step_seconds,
                batch_size=request.timeline_batch_size,
                widget=self,
                base_prompt=request.prompt,
                model=caption.selected_model,
                provider=caption.selected_provider,
                settings=caption.llm_settings,
            )
            caption.thread_pool.start(task)
            return

        image_paths = self._resolve_behavior_image_paths(
            request.start_time,
            request.end_time,
            request.start_frame,
            request.end_frame,
            segment_index=self._last_segment_selection,
            sample_count=request.sample_count,
        )
        if not image_paths:
            fallback_image = image_path or caption._resolve_image_for_description()
            if fallback_image:
                image_paths = [fallback_image]
            else:
                self.update_behavior_status(
                    "Unable to capture frames for this segment.", True
                )
                return

        self.set_sample_count(request.sample_count)
        task = BehaviorNarrativeTask(
            image_paths=image_paths,
            widget=self,
            prompt=request.prompt,
            model=caption.selected_model,
            provider=caption.selected_provider,
            settings=caption.llm_settings,
        )
        caption.thread_pool.start(task)

    def begin_behavior_stream(self, intro_text: str = "Describing behavior…") -> None:
        self._behavior_buffer = ""
        self._caption._allow_empty_caption = True
        self._caption.set_caption(intro_text)

    @QtCore.Slot(str)
    def append_behavior_stream_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._behavior_buffer += chunk
        self._caption._allow_empty_caption = False
        self._caption.set_caption(self._behavior_buffer)

    @QtCore.Slot(str, bool)
    def update_behavior_status(self, message: str, is_error: bool) -> None:
        self._caption.set_caption(message)
        self._set_idle_label()
        if is_error:
            self._behavior_buffer = ""
        else:
            self._behavior_buffer = message
        self._button.setEnabled(True)
        self._caption._allow_empty_caption = False


class BehaviorSegmentDialog(QDialog):
    """Collect start/end times, optional notes, and allow prompt editing."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        *,
        start_time: QtCore.QTime,
        end_time: QtCore.QTime,
        notes: Optional[str],
        prompt_text: str,
        segment_presets: Optional[Sequence[Dict[str, Any]]] = None,
        video_fps: Optional[float] = None,
        total_frames: Optional[int] = None,
        initial_preset_index: int = 0,
        current_frame: Optional[int] = None,
        sample_count_default: int = DEFAULT_BEHAVIOR_SAMPLE_COUNT,
        timeline_step_default: int = DEFAULT_BEHAVIOR_TIMELINE_STEP_SECONDS,
        timeline_batch_default: int = DEFAULT_BEHAVIOR_TIMELINE_BATCH_SIZE,
    ):
        super().__init__(parent)
        self.setWindowTitle("Describe behavior segment")
        self.setModal(True)
        self._segment_presets = list(segment_presets or [])
        self._segment_selector: Optional[QComboBox] = None
        self._video_fps = float(video_fps or 0.0)
        self._total_frames = max(0, int(total_frames or 0))
        self._syncing_time = False
        self._syncing_frame = False
        self.start_frame_spin: Optional[QSpinBox] = None
        self.end_frame_spin: Optional[QSpinBox] = None
        self._current_frame = current_frame
        self._current_preset_index: int = -1
        self._initial_preset_index = max(0, initial_preset_index)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Specify the video segment to describe (HH:MM:SS)."
                " Use identical start/end times to describe a single frame."
            )
        )

        if self._segment_presets:
            layout.addWidget(QtWidgets.QLabel("Quick-select a defined segment:"))
            self._segment_selector = QComboBox()
            self._segment_selector.addItem("Custom (enter times)", None)
            for preset in self._segment_presets:
                self._segment_selector.addItem(preset["label"], preset)
            self._segment_selector.currentIndexChanged.connect(
                self._on_segment_selected
            )
            layout.addWidget(self._segment_selector)
            if self._segment_presets:
                target_index = min(
                    self._initial_preset_index, len(self._segment_presets) - 1
                )
                self._segment_selector.setCurrentIndex(target_index + 1)
                self._current_preset_index = target_index

        form_layout = QFormLayout()
        self._form_layout = form_layout
        self.start_edit = QTimeEdit(
            start_time if start_time.isValid() else QtCore.QTime(0, 0, 0)
        )
        self.start_edit.setDisplayFormat("HH:mm:ss")
        self.end_edit = QTimeEdit(
            end_time if end_time.isValid() else QtCore.QTime(0, 0, 0)
        )
        self.end_edit.setDisplayFormat("HH:mm:ss")
        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItem("Segment summary (sample frames)", "summary")
        self.run_mode_combo.addItem("Timeline (describe every N seconds)", "timeline")
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(1, 200)
        self.sample_count_spin.setValue(
            max(1, int(sample_count_default or DEFAULT_BEHAVIOR_SAMPLE_COUNT))
        )
        self.timeline_step_spin = QSpinBox()
        self.timeline_step_spin.setRange(1, 60)
        self.timeline_step_spin.setValue(
            max(1, int(timeline_step_default or DEFAULT_BEHAVIOR_TIMELINE_STEP_SECONDS))
        )
        self.timeline_batch_spin = QSpinBox()
        self.timeline_batch_spin.setRange(1, 64)
        self.timeline_batch_spin.setValue(
            max(1, int(timeline_batch_default or DEFAULT_BEHAVIOR_TIMELINE_BATCH_SIZE))
        )

        form_layout.addRow("Run mode:", self.run_mode_combo)
        form_layout.addRow("Start time:", self.start_edit)
        form_layout.addRow("End time:", self.end_edit)
        form_layout.addRow("Sample frames:", self.sample_count_spin)
        form_layout.addRow("Describe every (s):", self.timeline_step_spin)
        form_layout.addRow("Max images/request:", self.timeline_batch_spin)

        if self._total_frames > 0:
            # Start frame row with quick-set button
            row_start = QtWidgets.QHBoxLayout()
            self.start_frame_spin = QSpinBox()
            self.start_frame_spin.setRange(0, self._total_frames - 1)
            self.start_frame_spin.setValue(self._time_to_frame(self.start_edit.time()))
            row_start.addWidget(self.start_frame_spin)
            set_start_btn = QPushButton("Use current")
            set_start_btn.setToolTip("Set start frame from current playback frame")
            set_start_btn.clicked.connect(
                lambda: self._apply_current_frame(to_start=True)
            )
            row_start.addWidget(set_start_btn)
            form_layout.addRow("Start frame:", row_start)

            # End frame row with quick-set button
            row_end = QtWidgets.QHBoxLayout()
            self.end_frame_spin = QSpinBox()
            self.end_frame_spin.setRange(0, self._total_frames - 1)
            end_frame_value = max(
                self._time_to_frame(self.end_edit.time()),
                self.start_frame_spin.value(),
            )
            self.end_frame_spin.setValue(end_frame_value)
            row_end.addWidget(self.end_frame_spin)
            set_end_btn = QPushButton("Use current")
            set_end_btn.setToolTip("Set end frame from current playback frame")
            set_end_btn.clicked.connect(
                lambda: self._apply_current_frame(to_start=False)
            )
            row_end.addWidget(set_end_btn)
            form_layout.addRow("End frame:", row_end)

            # Utility row: swap and snap to times
            util_row = QtWidgets.QHBoxLayout()
            swap_btn = QPushButton("Swap")
            swap_btn.setToolTip("Swap start and end frames")
            swap_btn.clicked.connect(self._swap_frames)
            util_row.addWidget(swap_btn)
            snap_btn = QPushButton("Snap to times")
            snap_btn.setToolTip("Reset frames from the HH:MM:SS values")
            snap_btn.clicked.connect(self._sync_frames_from_times)
            util_row.addWidget(snap_btn)
            util_row.addStretch(1)
            layout.addLayout(util_row)

        layout.addLayout(form_layout)

        self._timeline_estimate_label = QLabel()
        self._timeline_estimate_label.setWordWrap(True)
        layout.addWidget(self._timeline_estimate_label)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional notes or context")
        self.notes_edit.setText(notes or "")
        layout.addWidget(self.notes_edit)

        layout.addWidget(QtWidgets.QLabel("Prompt sent to the model (edit as needed):"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the behavior here…")
        self._prompt_updating = False
        self._user_modified_prompt = False
        self._set_prompt_text(prompt_text)
        layout.addWidget(self.prompt_edit)

        reset_row = QHBoxLayout()
        reset_row.addStretch()
        reset_btn = QPushButton("Use template")
        reset_btn.clicked.connect(lambda: self._maybe_autofill_prompt(force=True))
        reset_row.addWidget(reset_btn)
        layout.addLayout(reset_row)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._handle_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.start_edit.timeChanged.connect(lambda _: self._handle_time_changed(True))
        self.end_edit.timeChanged.connect(lambda _: self._handle_time_changed(False))
        self.notes_edit.textChanged.connect(self._maybe_autofill_prompt)
        self.prompt_edit.textChanged.connect(self._on_prompt_changed)
        self.run_mode_combo.currentIndexChanged.connect(
            lambda _: self._update_mode_visibility()
        )
        self.sample_count_spin.valueChanged.connect(
            lambda _: self._update_timeline_estimate()
        )
        self.timeline_step_spin.valueChanged.connect(
            lambda _: self._update_timeline_estimate()
        )
        self.timeline_batch_spin.valueChanged.connect(
            lambda _: self._update_timeline_estimate()
        )
        if self.start_frame_spin:
            self.start_frame_spin.valueChanged.connect(
                lambda val: self._handle_frame_changed(True, val)
            )
        if self.end_frame_spin:
            self.end_frame_spin.valueChanged.connect(
                lambda val: self._handle_frame_changed(False, val)
            )

        self._update_mode_visibility()
        self._update_timeline_estimate()

    def run_mode(self) -> str:
        return str(self.run_mode_combo.currentData() or "summary")

    def sample_count(self) -> int:
        return int(self.sample_count_spin.value())

    def timeline_step_seconds(self) -> int:
        return int(self.timeline_step_spin.value())

    def timeline_batch_size(self) -> int:
        return int(self.timeline_batch_spin.value())

    def _set_row_visible(self, field: QtWidgets.QWidget, visible: bool) -> None:
        label = getattr(self, "_form_layout", None)
        if isinstance(label, QFormLayout):
            row_label = label.labelForField(field)
            if row_label is not None:
                row_label.setVisible(visible)
        field.setVisible(visible)

    def _update_mode_visibility(self) -> None:
        mode = self.run_mode()
        is_summary = mode == "summary"
        is_timeline = mode == "timeline"
        self._set_row_visible(self.sample_count_spin, is_summary)
        self._set_row_visible(self.timeline_step_spin, is_timeline)
        self._set_row_visible(self.timeline_batch_spin, is_timeline)
        self._timeline_estimate_label.setVisible(is_timeline)
        self._update_timeline_estimate()

    def _timeline_window_count(self) -> int:
        if self.end_time() < self.start_time():
            return 0
        step = max(1, int(self.timeline_step_seconds()))
        duration_ms = self.start_time().msecsTo(self.end_time())
        return int(duration_ms // (step * 1000)) + 1

    def _update_timeline_estimate(self) -> None:
        if self.run_mode() != "timeline":
            self._timeline_estimate_label.setText("")
            return
        windows = self._timeline_window_count()
        batch = max(1, int(self.timeline_batch_size()))
        requests = (windows + batch - 1) // batch if windows > 0 else 0
        msg = f"Will generate ~{windows} timestamped lines (≈{requests} model calls at {batch} images/request)."
        if windows >= TIMELINE_CONFIRM_THRESHOLD_WINDOWS:
            msg += " This may take a long time for long videos."
        self._timeline_estimate_label.setText(msg)

    def _apply_current_frame(self, to_start: bool) -> None:
        if self._current_frame is None:
            return
        spin = self.start_frame_spin if to_start else self.end_frame_spin
        if spin is None:
            return
        spin.blockSignals(True)
        spin.setValue(self._current_frame)
        spin.blockSignals(False)
        # Also update time field to keep everything consistent
        self._sync_time_from_frame(to_start, self._current_frame)

    def _swap_frames(self) -> None:
        if self.start_frame_spin is None or self.end_frame_spin is None:
            return
        a = self.start_frame_spin.value()
        b = self.end_frame_spin.value()
        self.start_frame_spin.blockSignals(True)
        self.end_frame_spin.blockSignals(True)
        self.start_frame_spin.setValue(b)
        self.end_frame_spin.setValue(a)
        self.start_frame_spin.blockSignals(False)
        self.end_frame_spin.blockSignals(False)
        self._sync_time_from_frame(True, self.start_frame_spin.value())
        self._sync_time_from_frame(False, self.end_frame_spin.value())
        self._maybe_autofill_prompt(force=True)

    def _sync_frames_from_times(self) -> None:
        self._sync_frame_from_time(True)
        self._sync_frame_from_time(False)
        self._maybe_autofill_prompt(force=True)

    def _on_segment_selected(self, index: int) -> None:
        if not self._segment_selector:
            return
        preset = self._segment_selector.itemData(index)
        if not preset:
            self._current_preset_index = -1
            return
        self._current_preset_index = index - 1 if index > 0 else -1
        start = preset.get("start")
        end = preset.get("end")
        if isinstance(start, QtCore.QTime) and start.isValid():
            self.start_edit.blockSignals(True)
            self.start_edit.setTime(start)
            self.start_edit.blockSignals(False)
        if isinstance(end, QtCore.QTime) and end.isValid():
            self.end_edit.blockSignals(True)
            self.end_edit.setTime(end)
            self.end_edit.blockSignals(False)
        if self.start_frame_spin is not None:
            start_frame_val = preset.get("start_frame")
            if isinstance(start_frame_val, int):
                self.start_frame_spin.blockSignals(True)
                self.start_frame_spin.setValue(start_frame_val)
                self.start_frame_spin.blockSignals(False)
        if self.end_frame_spin is not None:
            end_frame_val = preset.get("end_frame")
            if isinstance(end_frame_val, int):
                self.end_frame_spin.blockSignals(True)
                self.end_frame_spin.setValue(end_frame_val)
                self.end_frame_spin.blockSignals(False)
        note_text = preset.get("notes") or ""
        self.notes_edit.setText(note_text)
        self._maybe_autofill_prompt(force=True)
        self._update_timeline_estimate()

    @staticmethod
    def compose_descriptor(start: QtCore.QTime, end: QtCore.QTime, note: str) -> str:
        start_text = start.toString("HH:mm:ss")
        end_text = end.toString("HH:mm:ss")
        base = start_text if start_text == end_text else f"{start_text}-{end_text}"
        note = (note or "").strip()
        if note:
            return f"{base} ({note})"
        return base

    def _render_template(self) -> str:
        return behavior_prompting.build_behavior_narrative_prompt(
            segment_label=self.segment_descriptor()
        )

    def _set_prompt_text(self, text: str) -> None:
        self._prompt_updating = True
        self.prompt_edit.setPlainText(text)
        self._prompt_updating = False
        self._user_modified_prompt = False

    def _maybe_autofill_prompt(self, *_, force: bool = False) -> None:
        if not force and self._user_modified_prompt:
            return
        self._set_prompt_text(self._render_template())

    def _on_prompt_changed(self) -> None:
        if self._prompt_updating:
            return
        self._user_modified_prompt = True

    def selected_preset_index(self) -> Optional[int]:
        return self._current_preset_index if self._current_preset_index >= 0 else None

    def _handle_time_changed(self, is_start: bool) -> None:
        if self._syncing_frame:
            return
        self._syncing_time = True
        try:
            self._sync_frame_from_time(is_start)
        finally:
            self._syncing_time = False
        self._maybe_autofill_prompt()
        self._update_timeline_estimate()

    def _handle_frame_changed(self, is_start: bool, value: int) -> None:
        if self._syncing_time:
            return
        self._syncing_frame = True
        try:
            self._sync_time_from_frame(is_start, value)
        finally:
            self._syncing_frame = False
        self._maybe_autofill_prompt()

    def _sync_frame_from_time(self, is_start: bool) -> None:
        spin = self.start_frame_spin if is_start else self.end_frame_spin
        if spin is None or self._video_fps <= 0:
            return
        target_time = self.start_edit.time() if is_start else self.end_edit.time()
        frame_value = self._time_to_frame(target_time)
        spin.blockSignals(True)
        spin.setValue(frame_value)
        spin.blockSignals(False)

    def _sync_time_from_frame(self, is_start: bool, frame_value: int) -> None:
        if self._video_fps <= 0:
            return
        target_edit = self.start_edit if is_start else self.end_edit
        time_val = self._frame_to_time(frame_value)
        target_edit.blockSignals(True)
        target_edit.setTime(time_val)
        target_edit.blockSignals(False)

    def _time_to_frame(self, time_val: QtCore.QTime) -> int:
        if not time_val.isValid() or self._video_fps <= 0:
            return 0
        msecs = time_val.msecsSinceStartOfDay()
        frame = int(round((msecs / 1000.0) * self._video_fps))
        max_frame = max(0, self._total_frames - 1)
        return max(0, min(frame, max_frame))

    def _frame_to_time(self, frame_index: int) -> QtCore.QTime:
        if self._video_fps <= 0:
            return QtCore.QTime(0, 0, 0)
        total_msecs = int(round((frame_index / self._video_fps) * 1000))
        max_msecs = (24 * 60 * 60 * 1000) - 1
        clamped = max(0, min(total_msecs, max_msecs))
        base = QtCore.QTime(0, 0, 0)
        return base.addMSecs(clamped)

    def start_time(self) -> QtCore.QTime:
        return self.start_edit.time()

    def end_time(self) -> QtCore.QTime:
        return self.end_edit.time()

    def notes(self) -> str:
        return self.notes_edit.text().strip()

    def prompt_text(self) -> str:
        return self.prompt_edit.toPlainText().strip()

    def segment_descriptor(self) -> str:
        return self.compose_descriptor(self.start_time(), self.end_time(), self.notes())

    def start_frame_value(self) -> Optional[int]:
        if self.start_frame_spin is not None:
            return self.start_frame_spin.value()
        if self._video_fps > 0:
            return self._time_to_frame(self.start_time())
        return None

    def end_frame_value(self) -> Optional[int]:
        if self.end_frame_spin is not None:
            return self.end_frame_spin.value()
        if self._video_fps > 0:
            return self._time_to_frame(self.end_time())
        return None

    def _handle_accept(self) -> None:
        if self.end_time() < self.start_time():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid segment",
                "End time must be equal to or later than the start time.",
            )
            return
        if self.run_mode() == "timeline":
            windows = self._timeline_window_count()
            if windows >= TIMELINE_CONFIRM_THRESHOLD_WINDOWS:
                resp = QtWidgets.QMessageBox.question(
                    self,
                    "Long timeline",
                    f"This will generate about {windows} per-step descriptions.\n\n"
                    "This can take a long time and produce a large caption.\n\n"
                    "Continue?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if resp != QtWidgets.QMessageBox.Yes:
                    return
        self.accept()


class BehaviorNarrativeTask(QRunnable):
    """Generate a natural-language behavior description via Ollama."""

    def __init__(
        self,
        image_paths: Sequence[str],
        widget: "BehaviorDescribeWidget",
        prompt: str,
        model: str = "qwen3-vl",
        provider: str = "ollama",
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.image_paths = [path for path in image_paths if path]
        self.widget = widget
        self.prompt = prompt
        self.model = model
        self.provider = provider
        self.settings = settings or {}
        self._prev_host_present = False
        self._prev_host_value: Optional[str] = None

    def run(self) -> None:
        try:
            if self.provider != "ollama":
                raise ValueError(
                    "Behavior description is currently available only for Ollama providers."
                )
            if not self.image_paths:
                raise ValueError("No images were provided for behavior description.")

            ollama_module = globals().get("ollama")
            if ollama_module is None:
                try:
                    ollama_module = importlib.import_module("ollama")
                except ImportError as exc:
                    raise ImportError(
                        "The python 'ollama' package is not installed."
                    ) from exc
                globals()["ollama"] = ollama_module

            self._prev_host_present = "OLLAMA_HOST" in os.environ
            self._prev_host_value = os.environ.get("OLLAMA_HOST")
            host = self.settings.get("ollama", {}).get("host")
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            messages = behavior_prompting.qwen_messages(
                self.image_paths,
                self.prompt,
            )

            response_stream = ollama_module.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )

            chunks: List[str] = []

            if isinstance(response_stream, dict):
                message = response_stream.get("message", {})
                content = message.get("content", "")
                if not content:
                    raise ValueError(
                        "Unexpected response format: missing 'message.content' field."
                    )
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "update_behavior_status",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, content),
                    QtCore.Q_ARG(bool, False),
                )
                return

            for part in response_stream:
                if "message" in part and "content" in part["message"]:
                    chunk = part["message"]["content"]
                    if chunk:
                        chunks.append(chunk)
                        QtCore.QMetaObject.invokeMethod(
                            self.widget,
                            "append_behavior_stream_chunk",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, chunk),
                        )
                elif "error" in part:
                    raise RuntimeError(part["error"])

            behavior_text = "".join(chunks).strip()
            if not behavior_text:
                raise RuntimeError("No behavior description received from Ollama.")

            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "update_behavior_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, behavior_text),
                QtCore.Q_ARG(bool, False),
            )

        except Exception as exc:
            error_message = (
                f"An error occurred while generating the behavior description: {exc}."
            )
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "update_behavior_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_message),
                QtCore.Q_ARG(bool, True),
            )
        finally:
            if self._prev_host_present and self._prev_host_value is not None:
                # type: ignore[arg-type]
                os.environ["OLLAMA_HOST"] = self._prev_host_value
            else:
                os.environ.pop("OLLAMA_HOST", None)


class BehaviorTimelineTask(QRunnable):
    """Generate a timestamped behavior timeline by sampling frames over a range."""

    def __init__(
        self,
        *,
        video_path: str,
        fps: float,
        total_frames: int,
        start_time: QtCore.QTime,
        end_time: QtCore.QTime,
        step_seconds: int,
        batch_size: int,
        widget: "BehaviorDescribeWidget",
        base_prompt: str,
        model: str = "qwen3-vl",
        provider: str = "ollama",
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.fps = float(fps or 0.0)
        self.total_frames = int(total_frames or 0)
        self.start_time = start_time
        self.end_time = end_time
        self.step_seconds = max(
            1, int(step_seconds or DEFAULT_BEHAVIOR_TIMELINE_STEP_SECONDS)
        )
        self.batch_size = max(
            1, int(batch_size or DEFAULT_BEHAVIOR_TIMELINE_BATCH_SIZE)
        )
        self.widget = widget
        self.base_prompt = (base_prompt or "").strip()
        self.model = model
        self.provider = provider
        self.settings = settings or {}
        self._prev_host_present = False
        self._prev_host_value: Optional[str] = None

    def _timeline_points(self) -> List[Tuple[int, str]]:
        if self.fps <= 0 or self.total_frames <= 0:
            return []
        if not (self.start_time.isValid() and self.end_time.isValid()):
            return []
        if self.end_time < self.start_time:
            return []

        start_seconds = self.start_time.msecsSinceStartOfDay() / 1000.0
        end_seconds = self.end_time.msecsSinceStartOfDay() / 1000.0
        raw = compute_timeline_points(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            step_seconds=self.step_seconds,
            fps=self.fps,
            total_frames=self.total_frames,
        )
        return [(frame_idx, format_hhmmss(t)) for frame_idx, t in raw]

    def _build_batch_prompt(self, timestamps: Sequence[str]) -> str:
        guidance = self.base_prompt.strip()
        if not guidance:
            guidance = behavior_prompting.build_behavior_narrative_prompt(
                segment_label=None
            )
        lines = [
            "You are an animal behavior observer.",
            "You will receive images sampled from a video at specific timestamps, in the same order as the list below.",
            "For each timestamp, write EXACTLY one line describing only observable mouse behavior at that moment.",
            "Output format: HH:MM:SS: <description>",
            "Do not add headings, numbering, or blank lines.",
            "",
            "Timestamps (in order):",
            *[f"- {ts}" for ts in timestamps],
            "",
            "Guidance to follow for each line:",
            guidance,
        ]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _encode_frame_rgb_to_base64(frame_rgb: Any) -> str:
        if frame_rgb is None:
            raise ValueError("Empty frame cannot be encoded.")
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if TIMELINE_IMAGE_FORMAT.lower() in {".jpg", ".jpeg"}:
            ok, encoded = cv2.imencode(
                TIMELINE_IMAGE_FORMAT,
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(TIMELINE_JPEG_QUALITY)],
            )
        else:
            ok, encoded = cv2.imencode(TIMELINE_IMAGE_FORMAT, frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode video frame.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _send_batch(
        self,
        *,
        ollama_module: Any,
        images: Sequence[str],
        timestamps: Sequence[str],
    ) -> str:
        prompt = self._build_batch_prompt(timestamps)
        messages = behavior_prompting.qwen_messages(images, prompt)

        response_stream = ollama_module.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )

        if isinstance(response_stream, dict):
            message = response_stream.get("message", {})
            content = str(message.get("content", "") or "")
            if content:
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "append_behavior_stream_chunk",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, content.rstrip() + "\n\n"),
                )
            return content

        chunks: List[str] = []
        for part in response_stream:
            if "message" in part and "content" in part["message"]:
                chunk = part["message"]["content"]
                if chunk:
                    chunks.append(chunk)
            elif "error" in part:
                raise RuntimeError(part["error"])
        merged = "".join(chunks)
        if merged:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "append_behavior_stream_chunk",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, merged.rstrip() + "\n\n"),
            )
        return merged

    def _timeline_sidecar_path(self) -> Path:
        video = Path(self.video_path)
        return video.with_suffix(video.suffix + TIMELINE_SIDECAR_SUFFIX)

    def _load_existing_point_map(self) -> Dict[str, Dict[str, Any]]:
        sidecar = self._timeline_sidecar_path()
        if not sidecar.exists():
            return {}
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
        points = payload.get("points")
        if not isinstance(points, list):
            return {}
        mapping: Dict[str, Dict[str, Any]] = {}
        for point in points:
            if not isinstance(point, dict):
                continue
            ts = str(point.get("timestamp", "")).strip()
            desc = str(point.get("description", "")).strip()
            if not ts or not desc:
                continue
            try:
                frame = int(point.get("frame"))
            except Exception:
                continue
            mapping[ts] = {"frame": frame, "timestamp": ts, "description": desc}
        return mapping

    def run(self) -> None:
        try:
            if self.provider != "ollama":
                raise ValueError(
                    "Behavior description is currently available only for Ollama providers."
                )

            points = self._timeline_points()
            if not points:
                raise ValueError(
                    "No timeline points could be generated for the selected time range."
                )

            ollama_module = globals().get("ollama")
            if ollama_module is None:
                try:
                    ollama_module = importlib.import_module("ollama")
                except ImportError as exc:
                    raise ImportError(
                        "The python 'ollama' package is not installed."
                    ) from exc
                globals()["ollama"] = ollama_module

            self._prev_host_present = "OLLAMA_HOST" in os.environ
            self._prev_host_value = os.environ.get("OLLAMA_HOST")
            host = self.settings.get("ollama", {}).get("host")
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            total = len(points)
            end_seconds = self.end_time.msecsSinceStartOfDay() / 1000.0
            end_frame = int(round(end_seconds * self.fps)) if self.fps > 0 else 0
            end_frame = max(0, min(end_frame, max(0, self.total_frames - 1)))

            existing_points = self._load_existing_point_map()
            pending_points: List[Tuple[int, str]] = []
            skipped_points: List[Dict[str, Any]] = []
            for frame_idx, timestamp in points:
                existing = existing_points.get(timestamp)
                if existing and str(existing.get("description", "")).strip():
                    skipped_points.append(existing)
                else:
                    pending_points.append((frame_idx, timestamp))

            processed = len(skipped_points)
            if processed > 0:
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "append_behavior_stream_chunk",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(
                        str,
                        f"Resuming timeline: skipped {processed} existing 1s results.\n\n",
                    ),
                )

            timeline_points: List[Dict[str, Any]] = list(skipped_points)
            if not pending_points:
                payload = json.dumps(
                    {
                        "video_path": self.video_path,
                        "step_seconds": self.step_seconds,
                        "prompt": self.base_prompt,
                        "end_frame": end_frame,
                        "points": timeline_points,
                    },
                    ensure_ascii=False,
                )
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "merge_timeline_result",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, payload),
                )
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "set_status_label",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "Done (all 1s points already available)"),
                )
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "finish_behavior_run",
                    QtCore.Qt.QueuedConnection,
                )
                return

            video = CV2Video(self.video_path)
            batch_images: List[str] = []
            batch_timestamps: List[str] = []
            batch_frames: List[int] = []
            last_frame_idx: Optional[int] = None
            last_frame_image: Optional[str] = None
            try:
                for frame_idx, timestamp in pending_points:
                    if frame_idx == last_frame_idx and last_frame_image is not None:
                        encoded = last_frame_image
                    else:
                        frame_rgb = video.load_frame(frame_idx)
                        encoded = self._encode_frame_rgb_to_base64(frame_rgb)
                        last_frame_idx = frame_idx
                        last_frame_image = encoded

                    batch_images.append(encoded)
                    batch_timestamps.append(timestamp)
                    batch_frames.append(frame_idx)

                    if len(batch_images) < self.batch_size:
                        continue

                    QtCore.QMetaObject.invokeMethod(
                        self.widget,
                        "set_status_label",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(
                            str, f"Describing… {processed + len(batch_images)}/{total}"
                        ),
                    )
                    batch_text = self._send_batch(
                        ollama_module=ollama_module,
                        images=batch_images,
                        timestamps=batch_timestamps,
                    )
                    parsed = self.widget._parse_timestamped_descriptions(batch_text)
                    batch_results: List[Dict[str, Any]] = []
                    for idx, ts in enumerate(batch_timestamps):
                        description = str(parsed.get(ts, "")).strip()
                        if not description:
                            continue
                        row = {
                            "frame": batch_frames[idx],
                            "timestamp": ts,
                            "description": description,
                        }
                        batch_results.append(row)
                        timeline_points.append(row)
                    if batch_results:
                        payload = json.dumps(
                            {
                                "video_path": self.video_path,
                                "step_seconds": self.step_seconds,
                                "prompt": self.base_prompt,
                                "end_frame": end_frame,
                                "points": batch_results,
                            },
                            ensure_ascii=False,
                        )
                        QtCore.QMetaObject.invokeMethod(
                            self.widget,
                            "merge_timeline_result",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, payload),
                        )
                    processed += len(batch_images)
                    batch_images.clear()
                    batch_timestamps.clear()
                    batch_frames.clear()

                if batch_images:
                    QtCore.QMetaObject.invokeMethod(
                        self.widget,
                        "set_status_label",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(
                            str, f"Describing… {processed + len(batch_images)}/{total}"
                        ),
                    )
                    batch_text = self._send_batch(
                        ollama_module=ollama_module,
                        images=batch_images,
                        timestamps=batch_timestamps,
                    )
                    parsed = self.widget._parse_timestamped_descriptions(batch_text)
                    batch_results = []
                    for idx, ts in enumerate(batch_timestamps):
                        description = str(parsed.get(ts, "")).strip()
                        if not description:
                            continue
                        row = {
                            "frame": batch_frames[idx],
                            "timestamp": ts,
                            "description": description,
                        }
                        batch_results.append(row)
                        timeline_points.append(row)
                    if batch_results:
                        payload = json.dumps(
                            {
                                "video_path": self.video_path,
                                "step_seconds": self.step_seconds,
                                "prompt": self.base_prompt,
                                "end_frame": end_frame,
                                "points": batch_results,
                            },
                            ensure_ascii=False,
                        )
                        QtCore.QMetaObject.invokeMethod(
                            self.widget,
                            "merge_timeline_result",
                            QtCore.Qt.QueuedConnection,
                            QtCore.Q_ARG(str, payload),
                        )
                    processed += len(batch_images)
            finally:
                video.release()

            payload = json.dumps(
                {
                    "video_path": self.video_path,
                    "step_seconds": self.step_seconds,
                    "prompt": self.base_prompt,
                    "end_frame": end_frame,
                    "points": timeline_points,
                },
                ensure_ascii=False,
            )
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "merge_timeline_result",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, payload),
            )

            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "set_status_label",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "Done"),
            )
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "finish_behavior_run",
                QtCore.Qt.QueuedConnection,
            )

        except Exception as exc:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "append_behavior_stream_chunk",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(
                    str, f"\n\nAn error occurred while generating the timeline: {exc}\n"
                ),
            )
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "finish_behavior_run",
                QtCore.Qt.QueuedConnection,
            )
        finally:
            if self._prev_host_present and self._prev_host_value is not None:
                # type: ignore[arg-type]
                os.environ["OLLAMA_HOST"] = self._prev_host_value
            else:
                os.environ.pop("OLLAMA_HOST", None)
