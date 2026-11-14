from __future__ import annotations

from collections.abc import Sequence

from qtpy import QtCore, QtGui, QtWidgets


class AiRectangleWidget(QtWidgets.QWidget):
    """Toolbar widget for open-vocabulary text-prompt segmentation."""

    runRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()

    _MAX_PROMPT_LENGTH = 1024
    _MAX_HISTORY_ITEMS = 40
    _VALID_OUTPUT_MODES = ("polygon", "mask", "rectangle")

    def __init__(self) -> None:
        super().__init__()
        self._prompt_history: list[str] = []
        self._label_suggestions: list[str] = []
        self._is_running: bool = False
        self._completer_model = QtCore.QStringListModel(self)
        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the text-prompt toolbar controls."""
        self._aiRectanglePrompt = QtWidgets.QLineEdit()
        self._aiRectanglePrompt.setMaxLength(self._MAX_PROMPT_LENGTH)
        self._aiRectanglePrompt.setClearButtonEnabled(True)
        self._aiRectanglePrompt.setMinimumWidth(150)
        self._aiRectanglePrompt.setMaximumWidth(260)
        self._aiRectanglePrompt.setPlaceholderText(self.tr("Prompt: mouse, pup=baby…"))
        self._aiRectanglePrompt.setToolTip(
            self.tr(
                "Enter one or more object prompts and press Enter or click Go.\n"
                "• Multi-class: mouse, food pellet; water spout\n"
                "• Label mapping: black mouse=mouse_1, white mouse -> mouse_2\n"
                "• Segment everything: every 32 cell\n"
                "• Frame flags: flags: grooming, rearing"
            )
        )

        self._completer = QtWidgets.QCompleter(self._completer_model, self)
        self._completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._completer.setFilterMode(QtCore.Qt.MatchContains)
        self._aiRectanglePrompt.setCompleter(self._completer)

        self._runPromptButton = QtWidgets.QToolButton()
        self._runPromptButton.setText(self.tr("Go"))
        self._runPromptButton.setToolTip(
            self.tr("Run text-prompt segmentation on the current frame (Enter)")
        )
        self._runPromptButton.setAutoRaise(False)

        prompt_row = QtWidgets.QHBoxLayout()
        prompt_row.setContentsMargins(0, 0, 0, 0)
        prompt_row.setSpacing(3)
        prompt_row.addWidget(self._aiRectanglePrompt, 1)
        prompt_row.addWidget(self._runPromptButton, 0)

        self._outputModeCombo = QtWidgets.QComboBox()
        self._outputModeCombo.addItem(self.tr("Polygon"), "polygon")
        self._outputModeCombo.addItem(self.tr("Mask"), "mask")
        self._outputModeCombo.addItem(self.tr("Box"), "rectangle")
        self._outputModeCombo.setToolTip(
            self.tr("Output annotation geometry generated from the text prompt")
        )

        self._confidenceSpinBox = QtWidgets.QDoubleSpinBox()
        self._confidenceSpinBox.setRange(0.05, 0.95)
        self._confidenceSpinBox.setSingleStep(0.05)
        self._confidenceSpinBox.setDecimals(2)
        self._confidenceSpinBox.setValue(0.30)
        self._confidenceSpinBox.setPrefix("≥")
        self._confidenceSpinBox.setToolTip(
            self.tr("Minimum detection confidence threshold (0.05 – 0.95)")
        )

        options_row = QtWidgets.QHBoxLayout()
        options_row.setContentsMargins(0, 0, 0, 0)
        options_row.setSpacing(4)
        options_row.addWidget(self._outputModeCombo, 1)
        options_row.addWidget(self._confidenceSpinBox, 0)

        # Optional CountGD toggle (off by default) — attribute name preserved for compatibility
        self._useCountGDCheckbox = QtWidgets.QCheckBox(self.tr("Use CountGD"))
        self._useCountGDCheckbox.setToolTip(
            self.tr(
                "Enable CountGD-based object counting in addition to "
                "GroundingDINO. Improves dense counting of repeated "
                "objects when CountGD is installed."
            )
        )
        self._useCountGDCheckbox.setChecked(False)

        self._replaceExistingCheckbox = QtWidgets.QCheckBox(self.tr("Replace"))
        self._replaceExistingCheckbox.setToolTip(
            self.tr(
                "Replace previous text-prompt shapes on the current frame before adding new results."
            )
        )
        self._replaceExistingCheckbox.setChecked(False)

        toggles_row = QtWidgets.QHBoxLayout()
        toggles_row.setContentsMargins(0, 0, 0, 0)
        toggles_row.setSpacing(6)
        toggles_row.addWidget(self._useCountGDCheckbox)
        toggles_row.addWidget(self._replaceExistingCheckbox)

        aiRectangleLayout = QtWidgets.QVBoxLayout()
        aiRectangleLayout.setContentsMargins(2, 2, 2, 2)
        aiRectangleLayout.setSpacing(3)
        aiRectangleLayout.addLayout(prompt_row)
        aiRectangleLayout.addLayout(options_row)
        aiRectangleLayout.addLayout(toggles_row)

        self.aiRectangleWidget = QtWidgets.QWidget()
        self.aiRectangleWidget.setLayout(aiRectangleLayout)

        self.aiRectangleAction = QtWidgets.QWidgetAction(self)
        self.aiRectangleAction.setDefaultWidget(self.aiRectangleWidget)

        self._aiRectanglePrompt.returnPressed.connect(self.runRequested.emit)
        self._runPromptButton.clicked.connect(self._on_run_button_clicked)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.aiRectangleWidget)
        self.setLayout(layout)

    def _on_run_button_clicked(self) -> None:
        if self._is_running:
            self.stopRequested.emit()
        else:
            self.runRequested.emit()

    def is_running(self) -> bool:
        return bool(self._is_running)

    def set_running(self, running: bool) -> None:
        self._is_running = bool(running)
        if self._is_running:
            self._runPromptButton.setText(self.tr("Stop"))
            self._runPromptButton.setToolTip(
                self.tr("Cancel running text-prompt segmentation")
            )
        else:
            self._runPromptButton.setText(self.tr("Go"))
            self._runPromptButton.setToolTip(
                self.tr("Run text-prompt segmentation on the current frame (Enter)")
            )

    def prompt_text(self) -> str:
        return self._aiRectanglePrompt.text().strip()

    def set_prompt_text(self, text: str) -> None:
        self._aiRectanglePrompt.setText(str(text or "").strip())

    def use_countgd(self) -> bool:
        return bool(self._useCountGDCheckbox.isChecked())

    def set_use_countgd(self, enabled: bool) -> None:
        self._useCountGDCheckbox.setChecked(bool(enabled))

    def box_threshold(self) -> float:
        return float(self._confidenceSpinBox.value())

    def set_box_threshold(self, value: float) -> None:
        clamped = min(0.95, max(0.05, float(value)))
        self._confidenceSpinBox.setValue(clamped)

    def output_mode(self) -> str:
        mode = str(self._outputModeCombo.currentData() or "polygon").strip().lower()
        return mode if mode in self._VALID_OUTPUT_MODES else "polygon"

    def set_output_mode(self, mode: str) -> None:
        normalized = str(mode or "polygon").strip().lower()
        if normalized in {"box", "boxes", "bbox"}:
            normalized = "rectangle"
        if normalized not in self._VALID_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported output_mode '{mode}'. Expected one of {self._VALID_OUTPUT_MODES}."
            )
        idx = self._outputModeCombo.findData(normalized)
        if idx >= 0:
            self._outputModeCombo.setCurrentIndex(idx)

    def replace_existing(self) -> bool:
        return bool(self._replaceExistingCheckbox.isChecked())

    def set_replace_existing(self, enabled: bool) -> None:
        self._replaceExistingCheckbox.setChecked(bool(enabled))

    def record_prompt(self, prompt: str) -> None:
        cleaned = str(prompt or "").strip()
        if not cleaned:
            return
        deduped = [
            item
            for item in self._prompt_history
            if item.casefold() != cleaned.casefold()
        ]
        self._prompt_history = [cleaned, *deduped][: self._MAX_HISTORY_ITEMS]
        self._refresh_completer_model()

    def set_completion_suggestions(self, labels: Sequence[str]) -> None:
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in labels or ():
            clean = str(raw or "").strip()
            if not clean:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(clean)
        self._label_suggestions = normalized
        self._refresh_completer_model()

    def _current_merged_suggestions(self) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in (*self._prompt_history, *self._label_suggestions):
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _refresh_completer_model(self) -> None:
        merged = self._current_merged_suggestions()
        if (
            not getattr(self, "_completer_refresh_pending", False)
            and self._completer_model.stringList() == merged
        ):
            return
        popup = self._completer.popup() if hasattr(self, "_completer") else None
        if popup is not None and popup.isVisible():
            if not getattr(self, "_completer_refresh_pending", False):
                self._completer_refresh_pending = True
                QtCore.QTimer.singleShot(0, self._flush_completer_refresh)
        else:
            self._completer_refresh_pending = False
            self._apply_completer_items(merged)

    def _flush_completer_refresh(self) -> None:
        self._completer_refresh_pending = False
        self._apply_completer_items(self._current_merged_suggestions())

    def _apply_completer_items(self, items: list[str]) -> None:
        try:
            if self._completer_model.stringList() != items:
                self._completer_model.setStringList(items)
        except RuntimeError:
            pass

    # Preserve legacy Qt font import reference if inspected by external code
    _DEFAULT_FONT_CLASS = QtGui.QFont
