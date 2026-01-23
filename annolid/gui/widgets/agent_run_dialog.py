from __future__ import annotations

from typing import Any, Dict, Optional

from qtpy import QtWidgets


class AgentRunDialog(QtWidgets.QDialog):
    """Configuration dialog for running the unified agent pipeline."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        video_path: Optional[str] = None,
        results_dir: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Run Agent Analysis"))
        self.setModal(True)

        cfg = config or {}
        layout = QtWidgets.QFormLayout(self)

        self.video_path_edit = QtWidgets.QLineEdit(video_path or "")
        self.video_path_edit.setReadOnly(True)
        layout.addRow(self.tr("Video"), self.video_path_edit)

        self.results_dir_edit = QtWidgets.QLineEdit(results_dir or "")
        self.results_dir_edit.setReadOnly(True)
        layout.addRow(self.tr("Results folder"), self.results_dir_edit)

        self.schema_edit = QtWidgets.QLineEdit(str(cfg.get("schema_path") or ""))
        self.schema_edit.setPlaceholderText(self.tr("(auto-detect from project)"))
        self.schema_btn = QtWidgets.QPushButton(self.tr("Browse…"))
        self.schema_btn.clicked.connect(self._browse_schema)
        layout.addRow(
            self.tr("Behavior spec"),
            self._wrap(self.schema_edit, self.schema_btn),
        )

        self.vision_adapter_combo = QtWidgets.QComboBox()
        self.vision_adapter_combo.addItems(["none", "maskrcnn"])
        self.vision_adapter_combo.setCurrentText(
            str(cfg.get("vision_adapter") or "none")
        )
        layout.addRow(self.tr("Vision adapter"), self.vision_adapter_combo)

        self.vision_pretrained_chk = QtWidgets.QCheckBox(
            self.tr("Use pretrained weights")
        )
        self.vision_pretrained_chk.setChecked(bool(cfg.get("vision_pretrained", False)))

        self.vision_score_spin = QtWidgets.QDoubleSpinBox()
        self.vision_score_spin.setRange(0.0, 1.0)
        self.vision_score_spin.setSingleStep(0.05)
        self.vision_score_spin.setValue(float(cfg.get("vision_score_threshold", 0.5)))
        self.vision_score_spin.setDecimals(2)

        self.vision_device_edit = QtWidgets.QLineEdit(
            str(cfg.get("vision_device") or "")
        )
        self.vision_device_edit.setPlaceholderText(self.tr("auto"))

        layout.addRow(self.tr("Vision pretrained"), self.vision_pretrained_chk)
        layout.addRow(self.tr("Vision score threshold"), self.vision_score_spin)
        layout.addRow(self.tr("Vision device"), self.vision_device_edit)

        self.llm_adapter_combo = QtWidgets.QComboBox()
        self.llm_adapter_combo.addItems(["none", "llm_chat"])
        self.llm_adapter_combo.setCurrentText(str(cfg.get("llm_adapter") or "none"))
        layout.addRow(self.tr("LLM adapter"), self.llm_adapter_combo)

        self.llm_profile_edit = QtWidgets.QLineEdit(str(cfg.get("llm_profile") or ""))
        self.llm_profile_edit.setPlaceholderText(self.tr("default"))
        layout.addRow(self.tr("LLM profile"), self.llm_profile_edit)

        self.llm_provider_edit = QtWidgets.QLineEdit(str(cfg.get("llm_provider") or ""))
        self.llm_provider_edit.setPlaceholderText(self.tr("openai / ollama"))
        layout.addRow(self.tr("LLM provider"), self.llm_provider_edit)

        self.llm_model_edit = QtWidgets.QLineEdit(str(cfg.get("llm_model") or ""))
        self.llm_model_edit.setPlaceholderText(self.tr("model name"))
        layout.addRow(self.tr("LLM model"), self.llm_model_edit)

        self.llm_persist_chk = QtWidgets.QCheckBox(self.tr("Persist LLM settings"))
        self.llm_persist_chk.setChecked(bool(cfg.get("llm_persist", False)))
        layout.addRow(self.tr("LLM persist"), self.llm_persist_chk)

        self.include_summary_chk = QtWidgets.QCheckBox(self.tr("Include LLM summary"))
        self.include_summary_chk.setChecked(bool(cfg.get("include_llm_summary", False)))
        layout.addRow(self.tr("LLM summary"), self.include_summary_chk)

        self.summary_prompt_edit = QtWidgets.QLineEdit(
            str(cfg.get("llm_summary_prompt") or "")
        )
        self.summary_prompt_edit.setPlaceholderText(
            self.tr("Summarize the behaviors defined in this behavior spec.")
        )
        layout.addRow(self.tr("Summary prompt"), self.summary_prompt_edit)

        self.stride_spin = QtWidgets.QSpinBox()
        self.stride_spin.setRange(1, 10_000)
        self.stride_spin.setValue(int(cfg.get("stride", 1)))
        layout.addRow(self.tr("Frame stride"), self.stride_spin)

        self.max_frames_spin = QtWidgets.QSpinBox()
        self.max_frames_spin.setRange(-1, 1_000_000_000)
        self.max_frames_spin.setValue(int(cfg.get("max_frames", -1)))
        self.max_frames_spin.setToolTip(self.tr("-1 = unlimited"))
        layout.addRow(self.tr("Max frames (-1 = unlimited)"), self.max_frames_spin)

        self.streaming_chk = QtWidgets.QCheckBox(self.tr("Streaming mode (preview)"))
        self.streaming_chk.setChecked(bool(cfg.get("streaming", False)))
        layout.addRow(self.tr("Streaming"), self.streaming_chk)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.vision_adapter_combo.currentTextChanged.connect(
            self._update_vision_controls
        )
        self.llm_adapter_combo.currentTextChanged.connect(self._update_llm_controls)
        self.include_summary_chk.toggled.connect(self._update_llm_controls)

        self._update_vision_controls(self.vision_adapter_combo.currentText())
        self._update_llm_controls(self.llm_adapter_combo.currentText())

    def _wrap(
        self, widget: QtWidgets.QWidget, button: QtWidgets.QWidget
    ) -> QtWidgets.QWidget:
        box = QtWidgets.QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        container = QtWidgets.QWidget()
        box.addWidget(widget)
        box.addWidget(button)
        container.setLayout(box)
        return container

    def _browse_schema(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select behavior spec"),
            "",
            self.tr("Behavior spec (*.json *.yaml *.yml);;All Files (*)"),
        )
        if path:
            self.schema_edit.setText(path)

    def _update_vision_controls(self, adapter: str) -> None:
        enabled = str(adapter).strip().lower() != "none"
        for widget in (
            self.vision_pretrained_chk,
            self.vision_score_spin,
            self.vision_device_edit,
        ):
            widget.setEnabled(enabled)

    def _update_llm_controls(self, adapter: str) -> None:
        enabled = str(adapter).strip().lower() != "none"
        for widget in (
            self.llm_profile_edit,
            self.llm_provider_edit,
            self.llm_model_edit,
            self.llm_persist_chk,
            self.include_summary_chk,
        ):
            widget.setEnabled(enabled)
        self.summary_prompt_edit.setEnabled(
            enabled and self.include_summary_chk.isChecked()
        )

    def values(self) -> Dict[str, Any]:
        schema_path = self.schema_edit.text().strip()
        max_frames = self.max_frames_spin.value()
        return {
            "schema_path": schema_path or None,
            "vision_adapter": self.vision_adapter_combo.currentText(),
            "vision_pretrained": self.vision_pretrained_chk.isChecked(),
            "vision_score_threshold": self.vision_score_spin.value(),
            "vision_device": self.vision_device_edit.text().strip() or None,
            "llm_adapter": self.llm_adapter_combo.currentText(),
            "llm_profile": self.llm_profile_edit.text().strip() or None,
            "llm_provider": self.llm_provider_edit.text().strip() or None,
            "llm_model": self.llm_model_edit.text().strip() or None,
            "llm_persist": self.llm_persist_chk.isChecked(),
            "include_llm_summary": self.include_summary_chk.isChecked(),
            "llm_summary_prompt": self.summary_prompt_edit.text().strip()
            or "Summarize the behaviors defined in this behavior spec.",
            "stride": int(self.stride_spin.value()),
            "max_frames": None if max_frames < 0 else int(max_frames),
            "streaming": self.streaming_chk.isChecked(),
        }
