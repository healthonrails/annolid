from __future__ import annotations

import importlib
import os
import tempfile
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
)

from annolid.behavior import prompting as behavior_prompting

if TYPE_CHECKING:
    from annolid.gui.widgets.caption import CaptionWidget


class BehaviorDescribeWidget(QtWidgets.QWidget):
    """Encapsulates the describe-behavior workflow and UI."""

    def __init__(self, caption_widget: "CaptionWidget") -> None:
        super().__init__(caption_widget)
        self._caption = caption_widget
        self._behavior_buffer: str = ""
        self._behavior_segment_notes: str = ""
        self._segment_snapshot_paths: List[str] = []
        self._video_duration_ms: int = 0
        self._video_path: Optional[str] = None
        self._video_fps: float = 0.0
        self._video_num_frames: int = 0

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

    def _cleanup_segment_snapshots(self) -> None:
        stale_paths = list(self._segment_snapshot_paths)
        self._segment_snapshot_paths.clear()
        for snapshot_path in stale_paths:
            try:
                if snapshot_path and os.path.exists(snapshot_path):
                    os.remove(snapshot_path)
            except OSError:
                pass

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
    ) -> Optional[Tuple[str, str, QtCore.QTime, QtCore.QTime]]:
        start_time, end_time = self._default_segment_times()
        default_descriptor = BehaviorSegmentDialog.compose_descriptor(
            start_time, end_time, self._behavior_segment_notes or ""
        )
        default_prompt = behavior_prompting.build_behavior_narrative_prompt(
            segment_label=default_descriptor,
        )
        dialog = BehaviorSegmentDialog(
            parent=self._caption,
            start_time=start_time,
            end_time=end_time,
            notes=self._behavior_segment_notes,
            prompt_text=default_prompt,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        self._behavior_segment_notes = dialog.notes()
        descriptor = dialog.segment_descriptor()
        prompt_text = dialog.prompt_text().strip()
        if not prompt_text:
            prompt_text = behavior_prompting.build_behavior_narrative_prompt(
                segment_label=descriptor
            )
        return descriptor, prompt_text, dialog.start_time(), dialog.end_time()

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
        max_samples: int = 5,
    ) -> List[int]:
        if self._video_fps <= 0 or self._video_num_frames <= 0:
            return []
        start_frame = self._time_to_frame_index(start_time)
        end_frame = self._time_to_frame_index(end_time)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        if start_frame == end_frame:
            return [start_frame]
        span = end_frame - start_frame
        samples = max(2, min(max_samples, span + 1))
        step = span / (samples - 1)
        frames = []
        for i in range(samples):
            idx = int(round(start_frame + i * step))
            idx = max(start_frame, min(end_frame, idx))
            frames.append(idx)
        return sorted(set(frames))

    def _extract_video_frames(self, frame_indices: Sequence[int]) -> List[str]:
        if not self._video_path:
            return []
        try:
            cap = cv2.VideoCapture(self._video_path)
        except Exception:
            return []
        if not cap or not cap.isOpened():
            return []
        extracted: List[str] = []
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                fd, tmp_path = tempfile.mkstemp(
                    prefix="annolid_segment_", suffix=".png"
                )
                os.close(fd)
                success = cv2.imwrite(tmp_path, frame)
                if success:
                    self._segment_snapshot_paths.append(tmp_path)
                    extracted.append(tmp_path)
                else:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        finally:
            cap.release()
        return extracted

    def _resolve_behavior_image_paths(
        self, start_time: QtCore.QTime, end_time: QtCore.QTime
    ) -> List[str]:
        if (
            self._video_path
            and self._video_fps > 0
            and self._video_num_frames > 0
        ):
            self._cleanup_segment_snapshots()
            frames = self._segment_frame_indices(start_time, end_time)
            if frames:
                images = self._extract_video_frames(frames)
                if images:
                    return images
        fallback = self._caption._resolve_image_for_description()
        return [fallback] if fallback else []

    def _on_behavior_describe_clicked(self) -> None:
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
            self._label.setText("Describe behavior")
            return
        _, prompt, seg_start, seg_end = segment_info

        self._label.setText("Describing…")
        self._button.setEnabled(False)
        self.begin_behavior_stream()

        image_paths = self._resolve_behavior_image_paths(seg_start, seg_end)
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
        self._label.setText("Describe behavior")
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
    ):
        super().__init__(parent)
        self.setWindowTitle("Describe behavior segment")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Specify the video segment to describe (HH:MM:SS)."
                " Use identical start/end times to describe a single frame."
            )
        )

        form_layout = QFormLayout()
        self.start_edit = QTimeEdit(start_time if start_time.isValid() else QtCore.QTime(0, 0, 0))
        self.start_edit.setDisplayFormat("HH:mm:ss")
        self.end_edit = QTimeEdit(end_time if end_time.isValid() else QtCore.QTime(0, 0, 0))
        self.end_edit.setDisplayFormat("HH:mm:ss")
        form_layout.addRow("Start time:", self.start_edit)
        form_layout.addRow("End time:", self.end_edit)
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

        self.start_edit.timeChanged.connect(self._maybe_autofill_prompt)
        self.end_edit.timeChanged.connect(self._maybe_autofill_prompt)
        self.notes_edit.textChanged.connect(self._maybe_autofill_prompt)
        self.prompt_edit.textChanged.connect(self._on_prompt_changed)

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
