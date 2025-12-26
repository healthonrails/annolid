from __future__ import annotations

from qtpy import QtCore, QtWidgets


class PdfControlsWidget(QtWidgets.QWidget):
    """Compact navigation and zoom controls for the PDF viewer."""

    previous_requested = QtCore.Signal()
    next_requested = QtCore.Signal()
    rotation_requested = QtCore.Signal()
    reset_zoom_requested = QtCore.Signal()
    zoom_changed = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._ignore_zoom_signal = False
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        nav_row = QtWidgets.QHBoxLayout()
        nav_row.setSpacing(6)
        self.prev_button = QtWidgets.QToolButton(self)
        self.prev_button.setText("◀")
        self.prev_button.clicked.connect(self.previous_requested.emit)

        self.page_label = QtWidgets.QLabel("Page - / -", self)
        self.page_label.setMinimumWidth(110)
        self.page_label.setAlignment(QtCore.Qt.AlignCenter)

        self.next_button = QtWidgets.QToolButton(self)
        self.next_button.setText("▶")
        self.next_button.clicked.connect(self.next_requested.emit)

        self.rotate_button = QtWidgets.QToolButton(self)
        self.rotate_button.setText("⟳")
        self.rotate_button.setToolTip("Rotate 90° clockwise")
        self.rotate_button.clicked.connect(self.rotation_requested.emit)

        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.page_label, 1)
        nav_row.addWidget(self.next_button)
        nav_row.addWidget(self.rotate_button)
        outer.addLayout(nav_row)

        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.setSpacing(6)
        zoom_label = QtWidgets.QLabel("Zoom", self)
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.zoom_slider.setRange(50, 300)
        self.zoom_slider.setSingleStep(10)
        self.zoom_slider.setPageStep(25)
        self.zoom_slider.setValue(150)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

        self.zoom_value = QtWidgets.QLabel("150%", self)
        self.reset_button = QtWidgets.QToolButton(self)
        self.reset_button.setText("Reset")
        self.reset_button.clicked.connect(self.reset_zoom_requested.emit)

        zoom_row.addWidget(zoom_label)
        zoom_row.addWidget(self.zoom_slider, 1)
        zoom_row.addWidget(self.zoom_value)
        zoom_row.addWidget(self.reset_button)
        outer.addLayout(zoom_row)

        outer.addStretch(1)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum,
        )

    def set_page_info(self, current_index: int, total: int) -> None:
        """Update the page label and navigation button states."""
        if total <= 0:
            self.page_label.setText("Page - / -")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        current = max(0, min(current_index, total - 1))
        self.page_label.setText(f"Page {current + 1} / {total}")
        self.prev_button.setEnabled(current > 0)
        self.next_button.setEnabled(current + 1 < total)

    def set_zoom_percent(self, percent: float) -> None:
        """Sync the slider/value label without emitting a new signal."""
        value = int(round(percent))
        value = max(self.zoom_slider.minimum(),
                    min(self.zoom_slider.maximum(), value))
        self._ignore_zoom_signal = True
        try:
            self.zoom_slider.setValue(value)
            self.zoom_value.setText(f"{value}%")
        finally:
            self._ignore_zoom_signal = False

    def set_controls_enabled(self, enabled: bool, reason: str = "") -> None:
        """Enable/disable all controls at once (e.g., in web mode)."""
        for widget in (
            self.prev_button,
            self.next_button,
            self.rotate_button,
            self.zoom_slider,
            self.zoom_value,
            self.reset_button,
        ):
            widget.setEnabled(enabled)
            if not enabled and reason:
                widget.setToolTip(reason)
            else:
                widget.setToolTip("")

    def _on_zoom_changed(self, value: int) -> None:
        if self._ignore_zoom_signal:
            return
        self.zoom_value.setText(f"{value}%")
        self.zoom_changed.emit(float(value))
