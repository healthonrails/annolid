from __future__ import annotations

from qtpy import QtCore, QtWidgets

from annolid.services.speech_highlighting import parse_keywords


class PdfReaderControlsWidget(QtWidgets.QWidget):
    """Controls for click-to-read playback and keyword highlighting while speaking."""

    reader_enabled_changed = QtCore.Signal(bool)
    pdfjs_mode_requested = QtCore.Signal(bool)
    pause_resume_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    previous_requested = QtCore.Signal()
    next_requested = QtCore.Signal()
    keywords_changed = QtCore.Signal(list)
    keyword_highlighting_changed = QtCore.Signal(bool)
    auto_keywords_changed = QtCore.Signal(bool)

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
            "Tip: double-click a sentence to start reading from that word.",
            self,
        )
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet("color: #5f6368;")
        layout.addWidget(self.tip_label)

        self.enable_checkbox = QtWidgets.QCheckBox("Enable click-to-read", self)
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

        self.prev_button = QtWidgets.QToolButton(self)
        self.prev_button.setText("◀ Prev")
        self.prev_button.clicked.connect(self.previous_requested.emit)
        controls_row.addWidget(self.prev_button)

        self.pause_button = QtWidgets.QToolButton(self)
        self.pause_button.setText("Pause")
        self.pause_button.clicked.connect(self.pause_resume_requested.emit)
        controls_row.addWidget(self.pause_button)

        self.stop_button = QtWidgets.QToolButton(self)
        self.stop_button.setText("Stop")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        controls_row.addWidget(self.stop_button)

        self.next_button = QtWidgets.QToolButton(self)
        self.next_button.setText("Next ▶")
        self.next_button.clicked.connect(self.next_requested.emit)
        controls_row.addWidget(self.next_button)

        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        self.progress_label = QtWidgets.QLabel("Sentence - / -", self)
        layout.addWidget(self.progress_label)

        # Keyword highlighting while speaking controls (works in both PDF.js & fallback)
        separator = QtWidgets.QFrame(self)
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(separator)

        kw_title = QtWidgets.QLabel("Keyword Highlighting", self)
        kw_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(kw_title)

        self.highlight_keywords_checkbox = QtWidgets.QCheckBox(
            "Highlight keywords while speaking", self
        )
        self.highlight_keywords_checkbox.setChecked(True)
        self.highlight_keywords_checkbox.toggled.connect(
            self._on_keyword_highlighting_toggled
        )
        layout.addWidget(self.highlight_keywords_checkbox)

        self.auto_keywords_checkbox = QtWidgets.QCheckBox(
            "Auto-highlight key terms", self
        )
        self.auto_keywords_checkbox.setChecked(False)
        self.auto_keywords_checkbox.setToolTip(
            "Automatically emphasize technical terms and acronyms in each spoken sentence."
        )
        self.auto_keywords_checkbox.toggled.connect(self._on_auto_keywords_toggled)
        layout.addWidget(self.auto_keywords_checkbox)

        self.keywords_edit = QtWidgets.QLineEdit(self)
        self.keywords_edit.setPlaceholderText("Keywords (comma-separated)…")
        self.keywords_edit.setToolTip(
            "Enter comma-separated keywords or phrases to highlight on the PDF and emphasize while speaking."
        )
        self.keywords_edit.setClearButtonEnabled(True)
        self.keywords_edit.textChanged.connect(self._on_keywords_text_changed)
        layout.addWidget(self.keywords_edit)

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

    def _on_keyword_highlighting_toggled(self, _checked: bool) -> None:
        enabled = bool(self.highlight_keywords_checkbox.isChecked())
        self.auto_keywords_checkbox.setEnabled(enabled)
        self.keywords_edit.setEnabled(enabled)
        if self._ignore_signals:
            return
        self.keyword_highlighting_changed.emit(enabled)

    def _on_auto_keywords_toggled(self, _checked: bool) -> None:
        if self._ignore_signals:
            return
        self.auto_keywords_changed.emit(bool(self.auto_keywords_checkbox.isChecked()))

    def _on_keywords_text_changed(self, text: str) -> None:
        if self._ignore_signals:
            return
        self.keywords_changed.emit(parse_keywords(text))

    def set_reader_state(
        self, state: str, current: int, total: int, *, unit_label: str = "Sentence"
    ) -> None:
        state_label = state.capitalize() if state else "Idle"
        self.status_label.setText(state_label)
        label_prefix = str(unit_label or "Sentence").strip() or "Sentence"
        if total > 0:
            current_display = max(1, min(current + 1, total))
            self.progress_label.setText(f"{label_prefix} {current_display} / {total}")
        else:
            self.progress_label.setText(f"{label_prefix} - / -")

        if state == "reading":
            self.pause_button.setText("Pause")
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
        elif state == "paused":
            self.pause_button.setText("Resume")
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
        else:
            self.pause_button.setText("Pause")
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

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

    def set_keywords(self, keywords: list[str] | str) -> None:
        terms = parse_keywords(keywords)
        joined = ", ".join(terms)
        self._ignore_signals = True
        try:
            if self.keywords_edit.text() != joined:
                self.keywords_edit.setText(joined)
        finally:
            self._ignore_signals = False

    def set_keyword_highlighting_enabled(self, enabled: bool) -> None:
        flag = bool(enabled)
        self._ignore_signals = True
        try:
            self.highlight_keywords_checkbox.setChecked(flag)
            self.auto_keywords_checkbox.setEnabled(flag)
            self.keywords_edit.setEnabled(flag)
        finally:
            self._ignore_signals = False

    def set_auto_keywords_enabled(self, enabled: bool) -> None:
        self._ignore_signals = True
        try:
            self.auto_keywords_checkbox.setChecked(bool(enabled))
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
            self.prev_button,
            self.next_button,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)
            widget.setToolTip("" if enabled else reason)
        if not enabled and reason:
            self.status_label.setText(reason)
        elif enabled and self.status_label.text() == reason:
            self.status_label.setText("Idle")
        # PDF.js toggle and keyword highlighting remain usable across modes.
        self.pdfjs_checkbox.setEnabled(True)
        self.pdfjs_checkbox.setToolTip(
            "" if enabled else "Switch to PDF.js to enable click-to-read."
        )
