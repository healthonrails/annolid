from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from qtpy import QtCore, QtWidgets


def _dedupe_preserve_order(values: Sequence[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _csv_to_list(text: str) -> list[str]:
    return [
        part.strip()
        for part in str(text or "").replace(";", ",").split(",")
        if part.strip()
    ]


class BehaviorSlashDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        video_path: str = "",
        labels: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
        provider_models: Mapping[str, Sequence[str]] | None = None,
        selected_provider: str = "",
        selected_model: str = "",
        segment_seconds: float = 1.0,
        frames_per_grid: int = 9,
        max_segments: int = 120,
        subject_term: str = "",
        video_description: str = "",
        behavior_definitions: str = "",
        focus_points: str = "",
        overwrite_existing: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run behavior labeling")
        self.setModal(True)
        self.setMinimumWidth(720)
        self._provider_models = {
            str(key or "").strip(): _dedupe_preserve_order(value)
            for key, value in dict(provider_models or {}).items()
        }

        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(12)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        outer.addLayout(form)

        video_row = QtWidgets.QWidget(self)
        video_layout = QtWidgets.QHBoxLayout(video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(8)
        self.video_path_edit = QtWidgets.QLineEdit(str(video_path or ""))
        self.video_path_edit.setPlaceholderText("Select a video file")
        self.video_browse_button = QtWidgets.QPushButton("Browse...", self)
        self.video_browse_button.clicked.connect(self._browse_video)
        video_layout.addWidget(self.video_path_edit, 1)
        video_layout.addWidget(self.video_browse_button, 0)
        form.addRow("Video", video_row)

        self.labels_edit = QtWidgets.QPlainTextEdit(self)
        self.labels_edit.setPlaceholderText(
            "still, walk, front groom, back groom, abdomen move"
        )
        self.labels_edit.setPlainText(", ".join(_dedupe_preserve_order(labels)))
        self.labels_edit.setFixedHeight(72)
        form.addRow("Behavior labels", self.labels_edit)

        route_row = QtWidgets.QWidget(self)
        route_layout = QtWidgets.QHBoxLayout(route_row)
        route_layout.setContentsMargins(0, 0, 0, 0)
        route_layout.setSpacing(8)
        self.provider_combo = QtWidgets.QComboBox(self)
        self.provider_combo.setEditable(True)
        provider_names = _dedupe_preserve_order(providers)
        for provider in provider_names:
            self.provider_combo.addItem(provider)
        if selected_provider and self.provider_combo.findText(selected_provider) < 0:
            self.provider_combo.addItem(selected_provider)
        if selected_provider:
            self.provider_combo.setCurrentText(selected_provider)
        self.model_combo = QtWidgets.QComboBox(self)
        self.model_combo.setEditable(True)
        route_layout.addWidget(self.provider_combo, 1)
        route_layout.addWidget(self.model_combo, 2)
        form.addRow("AI model", route_row)
        self.provider_combo.currentTextChanged.connect(self._refresh_models)
        self._refresh_models(self.provider_combo.currentText(), selected_model)

        settings_row = QtWidgets.QWidget(self)
        settings_layout = QtWidgets.QHBoxLayout(settings_row)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)
        self.seconds_spin = QtWidgets.QDoubleSpinBox(self)
        self.seconds_spin.setRange(0.05, 3600.0)
        self.seconds_spin.setDecimals(2)
        self.seconds_spin.setValue(float(segment_seconds or 1.0))
        self.seconds_spin.setSuffix(" s")
        self.frames_spin = QtWidgets.QSpinBox(self)
        self.frames_spin.setRange(1, 64)
        self.frames_spin.setValue(max(1, int(frames_per_grid or 9)))
        self.max_segments_spin = QtWidgets.QSpinBox(self)
        self.max_segments_spin.setRange(1, 1_000_000)
        self.max_segments_spin.setValue(max(1, int(max_segments or 120)))
        settings_layout.addWidget(QtWidgets.QLabel("Every", self), 0)
        settings_layout.addWidget(self.seconds_spin, 1)
        settings_layout.addWidget(QtWidgets.QLabel("Frames/grid", self), 0)
        settings_layout.addWidget(self.frames_spin, 1)
        settings_layout.addWidget(QtWidgets.QLabel("Max", self), 0)
        settings_layout.addWidget(self.max_segments_spin, 1)
        form.addRow("Sampling", settings_row)

        self.subject_term_edit = QtWidgets.QLineEdit(str(subject_term or ""))
        self.subject_term_edit.setPlaceholderText("fly, mouse, zebrafish, animal")
        form.addRow("Subject term", self.subject_term_edit)

        self.video_description_edit = QtWidgets.QPlainTextEdit(self)
        self.video_description_edit.setPlaceholderText(
            "Short video context or assay description"
        )
        self.video_description_edit.setPlainText(str(video_description or ""))
        self.video_description_edit.setFixedHeight(72)
        form.addRow("Custom prompt", self.video_description_edit)

        self.behavior_definitions_edit = QtWidgets.QPlainTextEdit(self)
        self.behavior_definitions_edit.setPlaceholderText(
            "Optional definitions for ambiguous behaviors"
        )
        self.behavior_definitions_edit.setPlainText(str(behavior_definitions or ""))
        self.behavior_definitions_edit.setFixedHeight(72)
        form.addRow("Definitions", self.behavior_definitions_edit)

        self.focus_points_edit = QtWidgets.QPlainTextEdit(self)
        self.focus_points_edit.setPlaceholderText(
            "Optional visual cues or body parts to focus on"
        )
        self.focus_points_edit.setPlainText(str(focus_points or ""))
        self.focus_points_edit.setFixedHeight(72)
        form.addRow("Focus", self.focus_points_edit)

        self.overwrite_check = QtWidgets.QCheckBox(
            "Overwrite existing behavior timeline and segment log entries", self
        )
        self.overwrite_check.setChecked(bool(overwrite_existing))
        form.addRow("Overwrite", self.overwrite_check)

        helper = QtWidgets.QLabel(
            "Existing segment-label logs are reused by default, so only missing frame ranges are sent to the model.",
            self,
        )
        helper.setWordWrap(True)
        outer.addWidget(helper)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Run labeling")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv);;All Files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    def _refresh_models(self, provider: str, selected_model: str = "") -> None:
        current = str(selected_model or self.model_combo.currentText() or "").strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        models = self._provider_models.get(str(provider or "").strip(), [])
        for model in models:
            self.model_combo.addItem(model)
        if current and self.model_combo.findText(current) < 0:
            self.model_combo.addItem(current)
        if current:
            self.model_combo.setCurrentText(current)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

    def values(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path_edit.text().strip(),
            "behavior_labels": _csv_to_list(self.labels_edit.toPlainText()),
            "llm_provider": self.provider_combo.currentText().strip(),
            "llm_model": self.model_combo.currentText().strip(),
            "segment_seconds": float(self.seconds_spin.value()),
            "sample_frames_per_segment": int(self.frames_spin.value()),
            "max_segments": int(self.max_segments_spin.value()),
            "subject_term": self.subject_term_edit.text().strip(),
            "video_description": self.video_description_edit.toPlainText().strip(),
            "behavior_definitions": self.behavior_definitions_edit.toPlainText().strip(),
            "focus_points": self.focus_points_edit.toPlainText().strip(),
            "overwrite_existing": self.overwrite_check.isChecked(),
        }

    def accept(self) -> None:  # pragma: no cover - exercised through callers
        values = self.values()
        if not values["video_path"]:
            QtWidgets.QMessageBox.warning(
                self,
                "Behavior labeling",
                "Select a video before running behavior labeling.",
            )
            return
        if not values["behavior_labels"]:
            QtWidgets.QMessageBox.warning(
                self,
                "Behavior labeling",
                "Enter at least one behavior label.",
            )
            return
        super().accept()
