from __future__ import annotations

import importlib
import logging
import os
import shutil
import tempfile
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
from annolid.data.videos import extract_frames_from_video
from annolid.jobs.tracking_jobs import TrackingSegment

if TYPE_CHECKING:
    from annolid.gui.widgets.caption import CaptionWidget


logger = logging.getLogger(__name__)

# Default number of frames to sample per segment when not specified by UI
DEFAULT_BEHAVIOR_SAMPLE_COUNT = 5


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
        self._video_duration_ms: int = 0
        self._video_path: Optional[str] = None
        self._video_fps: float = 0.0
        self._video_num_frames: int = 0
        self._video_segments: List[TrackingSegment] = []
        self._last_segment_selection: Optional[int] = None
        self._pending_segment_extract: bool = False
        self._sample_count: int = DEFAULT_BEHAVIOR_SAMPLE_COUNT
        self._current_frame: Optional[int] = None

        self._button = caption_widget.create_button(
            icon_name="dialog-information",
            color="#6f42c1",
            hover_color="#4c2880",
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
            self.set_video_segments([])
        else:
            if self._video_segments:
                self._pre_extract_segment_snapshots()
            elif self._pending_segment_extract:
                self._pre_extract_segment_snapshots()
                self._pending_segment_extract = False

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
        if self._video_path and self._video_fps > 0 and self._video_num_frames > 0:
            self._pre_extract_segment_snapshots()
            self._pending_segment_extract = False
        else:
            self._pending_segment_extract = bool(self._video_segments)

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
                else:
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning(
                    "Failed to extract frames for segment %s: %s", idx, exc
                )
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
                if self._video_path and self._video_fps > 0 and self._video_num_frames > 0:
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
                logger.info("BehaviorDescribeWidget could not infer video from image path: %s", img_path)
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
            clamped_ms = min(
                self._video_duration_ms, (24 * 60 * 60 * 1000) - 1
            )
            end = start.addMSecs(clamped_ms)
        else:
            end = start
        if not end.isValid():
            end = QtCore.QTime(23, 59, 59)
        return start, end

    def _prompt_behavior_segment(
        self,
    ) -> Optional[Tuple[str, str, QtCore.QTime, QtCore.QTime, Optional[int], Optional[int]]]:
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
        return (
            descriptor,
            prompt_text,
            dialog.start_time(),
            dialog.end_time(),
            dialog.start_frame_value(),
            dialog.end_frame_value(),
        )

    def _segment_presets(self) -> List[Dict[str, Any]]:
        if not self._video_segments or self._video_fps <= 0:
            return []
        presets: List[Dict[str, Any]] = []
        for idx, segment in enumerate(self._video_segments, start=1):
            start = self._frame_to_time(segment.segment_start_frame)
            # Add 1 frame to make the end time inclusive.
            end = self._frame_to_time(segment.segment_end_frame + 1)
            label = (
                f"{idx}. {start.toString('HH:mm:ss')}–{end.toString('HH:mm:ss')}"
            )
            note = (
                f"Segment {idx}, frames {segment.segment_start_frame}-{segment.segment_end_frame}"
            )
            presets.append({
                "label": label,
                "start": start,
                "end": end,
                "notes": note,
                "start_frame": segment.segment_start_frame,
                "end_frame": segment.segment_end_frame,
            })
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
            logger.info("BehaviorDescribeWidget single-frame segment at %s", start_frame)
            return [start_frame]
        # Compute samples
        n = sample_count if isinstance(sample_count, int) and sample_count > 0 else self._sample_count
        n = max(2, min(n, (end_frame - start_frame + 1)))  # inclusive
        if n == (end_frame - start_frame + 1):
            frames = list(range(start_frame, end_frame + 1))
        else:
            step = (end_frame - start_frame) / (n - 1)
            frames = [int(round(start_frame + i * step)) for i in range(n)]
        frames = sorted(set(max(start_frame, min(end_frame, f)) for f in frames))
        logger.info(
            "BehaviorDescribeWidget sampled frames (%s): %s", n, frames
        )
        return frames

    def _resolve_behavior_image_paths(
        self,
        start_time: QtCore.QTime,
        end_time: QtCore.QTime,
        start_frame: Optional[int],
        end_frame: Optional[int],
        segment_index: Optional[int] = None,
    ) -> List[str]:
        if segment_index is not None:
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
        if (
            self._video_path
            and self._video_fps > 0
            and self._video_num_frames > 0
        ):
            frames = self._segment_frame_indices(
                start_time,
                end_time,
                start_frame,
                end_frame,
                self._sample_count,
            )
            if frames:
                seg_dir = self._create_snapshot_dir(min(frames), max(frames))
                try:
                    # File pattern: <video_stem>_<frame_number>.png
                    pattern = f"{Path(self._video_path).stem}_{{frame:09d}}.png" if self._video_path else "{video_stem}_{frame:09d}.png"
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

        segment_info = self._prompt_behavior_segment()
        if segment_info is None:
            self._set_idle_label()
            return
        (
            _,
            prompt,
            seg_start_time,
            seg_end_time,
            seg_start_frame,
            seg_end_frame,
        ) = segment_info
        logger.info(
            "BehaviorDescribeWidget segment request: times %s → %s, frames %s → %s",
            seg_start_time.toString("HH:mm:ss"),
            seg_end_time.toString("HH:mm:ss"),
            seg_start_frame,
            seg_end_frame,
        )

        self._label.setText("Describing…")
        self._button.setEnabled(False)
        self.begin_behavior_stream()

        image_paths = self._resolve_behavior_image_paths(
            seg_start_time,
            seg_end_time,
            seg_start_frame,
            seg_end_frame,
            segment_index=self._last_segment_selection,
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

        task = BehaviorNarrativeTask(
            image_paths=image_paths,
            widget=self,
            prompt=prompt,
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
                target_index = min(self._initial_preset_index, len(self._segment_presets) - 1)
                self._segment_selector.setCurrentIndex(target_index + 1)
                self._current_preset_index = target_index

        form_layout = QFormLayout()
        self.start_edit = QTimeEdit(start_time if start_time.isValid() else QtCore.QTime(0, 0, 0))
        self.start_edit.setDisplayFormat("HH:mm:ss")
        self.end_edit = QTimeEdit(end_time if end_time.isValid() else QtCore.QTime(0, 0, 0))
        self.end_edit.setDisplayFormat("HH:mm:ss")
        form_layout.addRow("Start time:", self.start_edit)
        form_layout.addRow("End time:", self.end_edit)

        if self._total_frames > 0:
            # Start frame row with quick-set button
            row_start = QtWidgets.QHBoxLayout()
            self.start_frame_spin = QSpinBox()
            self.start_frame_spin.setRange(0, self._total_frames - 1)
            self.start_frame_spin.setValue(self._time_to_frame(self.start_edit.time()))
            row_start.addWidget(self.start_frame_spin)
            set_start_btn = QPushButton("Use current")
            set_start_btn.setToolTip("Set start frame from current playback frame")
            set_start_btn.clicked.connect(lambda: self._apply_current_frame(to_start=True))
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
            set_end_btn.clicked.connect(lambda: self._apply_current_frame(to_start=False))
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
        if self.start_frame_spin:
            self.start_frame_spin.valueChanged.connect(
                lambda val: self._handle_frame_changed(True, val)
            )
        if self.end_frame_spin:
            self.end_frame_spin.valueChanged.connect(
                lambda val: self._handle_frame_changed(False, val)
            )

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
                os.environ["OLLAMA_HOST"] = self._prev_host_value  # type: ignore[arg-type]
            else:
                os.environ.pop("OLLAMA_HOST", None)
