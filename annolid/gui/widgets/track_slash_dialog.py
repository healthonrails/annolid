from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from qtpy import QtCore, QtWidgets


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def build_track_slash_command(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    video_path = str(
        payload.get("video_path") or payload.get("path") or payload.get("video") or ""
    ).strip()
    if video_path:
        parts.append(f"video={shlex.quote(video_path)}")
    prompt = str(
        payload.get("text_prompt") or payload.get("prompt") or payload.get("text") or ""
    ).strip()
    if prompt:
        parts.append(f"prompt={shlex.quote(prompt)}")
    model_name = str(
        payload.get("model_name")
        or payload.get("model")
        or payload.get("modelname")
        or ""
    ).strip()
    if model_name:
        parts.append(f"model={shlex.quote(model_name)}")
    mode = str(payload.get("mode") or "track").strip().lower()
    if mode in {"segment", "track"} and mode != "track":
        parts.append(f"mode={shlex.quote(mode)}")
    if bool(payload.get("use_countgd", False)):
        parts.append("use_countgd=true")
    to_frame = payload.get("to_frame")
    if to_frame not in (None, ""):
        try:
            frame = int(to_frame)
        except Exception:
            frame = -1
        if frame > 0:
            parts.append(f"to_frame={frame}")
    return "/track" + (" " + " ".join(parts) if parts else "")


def _format_sam3_model_hint(bot_provider: str = "", bot_model: str = "") -> str:
    provider_text = str(bot_provider or "").strip()
    model_text = str(bot_model or "").strip()
    if provider_text and model_text:
        return (
            "SAM3 will reuse the current bot provider/model for the vision seed: "
            f"{provider_text} / {model_text}."
        )
    if provider_text:
        return (
            "SAM3 will reuse the current bot provider/model for the vision seed: "
            f"{provider_text}."
        )
    if model_text:
        return (
            "SAM3 will reuse the current bot provider/model for the vision seed: "
            f"{model_text}."
        )
    return "SAM3 will reuse the current bot provider/model for the vision seed."


class TrackSlashDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        video_path: str = "",
        prompt: str = "",
        model_names: Sequence[str] | None = None,
        selected_model: str = "",
        mode: str = "track",
        use_countgd: bool = False,
        to_frame: int = -1,
        bot_provider: str = "",
        bot_model: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prepare /track command")
        self.setModal(True)
        self.setMinimumWidth(620)

        layout = QtWidgets.QFormLayout(self)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        video_row = QtWidgets.QWidget(self)
        video_layout = QtWidgets.QHBoxLayout(video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(8)
        self.video_path_edit = QtWidgets.QLineEdit(str(video_path or ""))
        self.video_path_edit.setPlaceholderText("Select a video file")
        self.video_browse_button = QtWidgets.QPushButton("Browse…", self)
        self.video_browse_button.clicked.connect(self._browse_video)
        video_layout.addWidget(self.video_path_edit, 1)
        video_layout.addWidget(self.video_browse_button, 0)
        layout.addRow("Video", video_row)

        self.prompt_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_edit.setPlaceholderText("Describe the object or animal to track")
        self.prompt_edit.setPlainText(str(prompt or "").strip())
        self.prompt_edit.setFixedHeight(88)
        layout.addRow("Text prompt", self.prompt_edit)

        self.model_combo = QtWidgets.QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        models = _dedupe_preserve_order(model_names or [])
        for name in models:
            self.model_combo.addItem(name)
        initial_model = str(selected_model or "").strip()
        if initial_model and self.model_combo.findText(initial_model) < 0:
            self.model_combo.addItem(initial_model)
        if initial_model:
            self.model_combo.setCurrentText(initial_model)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.setPlaceholderText("Optional")
        layout.addRow("AI model", self.model_combo)

        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItems(["track", "segment"])
        current_mode = str(mode or "track").strip().lower()
        if current_mode not in {"track", "segment"}:
            current_mode = "track"
        self.mode_combo.setCurrentText(current_mode)
        layout.addRow("Mode", self.mode_combo)

        self.use_countgd_check = QtWidgets.QCheckBox("Enable CountGD", self)
        self.use_countgd_check.setChecked(bool(use_countgd))
        layout.addRow("CountGD", self.use_countgd_check)

        self.to_frame_spin = QtWidgets.QSpinBox(self)
        self.to_frame_spin.setRange(0, 10_000_000)
        self.to_frame_spin.setSpecialValueText("Auto")
        self.to_frame_spin.setValue(int(to_frame) if int(to_frame or 0) > 0 else 0)
        self.to_frame_spin.setToolTip(
            "Use the GUI default if you do not want to target a specific frame."
        )
        layout.addRow("Target frame", self.to_frame_spin)

        helper = QtWidgets.QLabel(
            "This dialog prepares a structured /track command. You can review it before sending.",
            self,
        )
        helper.setWordWrap(True)
        layout.addRow(helper)

        self.sam3_hint_label = QtWidgets.QLabel(
            _format_sam3_model_hint(bot_provider=bot_provider, bot_model=bot_model),
            self,
        )
        self.sam3_hint_label.setWordWrap(True)
        self.sam3_hint_label.setObjectName("sam3TrackHintLabel")
        layout.addRow(self.sam3_hint_label)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        button_box.button(QtWidgets.QDialogButtonBox.Ok).setText("Insert command")
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.mode_combo.currentTextChanged.connect(self._update_mode_controls)
        self._update_mode_controls(self.mode_combo.currentText())

    def _browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.flv);;All Files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    def _update_mode_controls(self, mode: str) -> None:
        mode_norm = str(mode or "").strip().lower()
        self.to_frame_spin.setEnabled(mode_norm == "track")

    def values(self) -> Dict[str, Any]:
        to_frame = (
            int(self.to_frame_spin.value()) if self.to_frame_spin.isEnabled() else 0
        )
        return {
            "video_path": self.video_path_edit.text().strip(),
            "text_prompt": self.prompt_edit.toPlainText().strip(),
            "model_name": self.model_combo.currentText().strip(),
            "mode": self.mode_combo.currentText().strip().lower(),
            "use_countgd": self.use_countgd_check.isChecked(),
            "to_frame": to_frame,
        }

    def accept(self) -> None:  # pragma: no cover - exercised via caller flows
        values = self.values()
        if not values["video_path"]:
            QtWidgets.QMessageBox.warning(
                self,
                "Track video",
                "Select a video before inserting the command.",
            )
            return
        if not values["text_prompt"]:
            QtWidgets.QMessageBox.warning(
                self,
                "Track video",
                "Enter a text prompt before inserting the command.",
            )
            return
        super().accept()
