from __future__ import annotations

from qtpy import QtCore, QtWidgets


class PdfReaderControlsWidget(QtWidgets.QWidget):
    """Controls for click-to-read paragraph playback."""

    reader_enabled_changed = QtCore.Signal(bool)
    pdfjs_mode_requested = QtCore.Signal(bool)
    pause_resume_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._ignore_signals = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Click-to-Read", self)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self.status_label = QtWidgets.QLabel("Idle", self)
        self.status_label.setStyleSheet("color: #5f6368;")
        layout.addWidget(self.status_label)

        self.tip_label = QtWidgets.QLabel(
            "Tip: click a paragraph to start reading from there.", self
        )
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet("color: #5f6368;")
        layout.addWidget(self.tip_label)

        self.enable_checkbox = QtWidgets.QCheckBox(
            "Enable click-to-read", self
        )
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_checkbox)

        self.pdfjs_checkbox = QtWidgets.QCheckBox(
            "Use PDF.js (required for reader)", self
        )
        self.pdfjs_checkbox.setChecked(False)
        self.pdfjs_checkbox.toggled.connect(self._on_pdfjs_toggled)
        layout.addWidget(self.pdfjs_checkbox)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(6)

        self.pause_button = QtWidgets.QToolButton(self)
        self.pause_button.setText("Pause")
        self.pause_button.clicked.connect(self.pause_resume_requested.emit)
        controls_row.addWidget(self.pause_button)

        self.stop_button = QtWidgets.QToolButton(self)
        self.stop_button.setText("Stop")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        controls_row.addWidget(self.stop_button)

        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.progress_label = QtWidgets.QLabel("Paragraph - / -", self)
        layout.addWidget(self.progress_label)

        layout.addStretch(1)
        self._set_controls_enabled(False, reason="PDF.js required")

    def _on_enabled_toggled(self, checked: bool) -> None:
        if self._ignore_signals:
            return
        self.reader_enabled_changed.emit(bool(checked))

    def _on_pdfjs_toggled(self, checked: bool) -> None:
        if self._ignore_signals:
            return
        self.pdfjs_mode_requested.emit(bool(checked))

    def set_reader_state(self, state: str, current: int, total: int) -> None:
        state_label = state.capitalize() if state else "Idle"
        self.status_label.setText(state_label)
        if total > 0:
            current_display = max(1, min(current + 1, total))
            self.progress_label.setText(
                f"Paragraph {current_display} / {total}"
            )
        else:
            self.progress_label.setText("Paragraph - / -")

        if state == "reading":
            self.pause_button.setText("Pause")
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        elif state == "paused":
            self.pause_button.setText("Resume")
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        else:
            self.pause_button.setText("Pause")
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def set_reader_enabled(self, enabled: bool) -> None:
        self._ignore_signals = True
        try:
            self.enable_checkbox.setChecked(bool(enabled))
        finally:
            self._ignore_signals = False

    def set_pdfjs_checked(self, enabled: bool) -> None:
        self._ignore_signals = True
        try:
            self.pdfjs_checkbox.setChecked(bool(enabled))
        finally:
            self._ignore_signals = False

    def set_reader_available(self, available: bool, reason: str = "") -> None:
        self._set_controls_enabled(available, reason=reason)

    def _set_controls_enabled(self, enabled: bool, reason: str = "") -> None:
        widgets = [
            self.enable_checkbox,
            self.pause_button,
            self.stop_button,
            self.tip_label,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
            widget.setToolTip("" if enabled else reason)
        if not enabled and reason:
            self.status_label.setText(reason)
        elif enabled and self.status_label.text() == reason:
            self.status_label.setText("Idle")
        # PDF.js toggle should always be available to let the user switch modes.
        self.pdfjs_checkbox.setEnabled(True)
        self.pdfjs_checkbox.setToolTip(
            "" if enabled else "Switch to PDF.js to enable click-to-read."
        )
