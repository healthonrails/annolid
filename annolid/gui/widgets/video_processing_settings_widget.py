from __future__ import annotations

from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from qtpy.QtCore import Qt

from annolid.gui.widgets.crop_region import CropRegion


class VideoProcessingSettingsWidget(QWidget):
    """Shared video processing controls used by the main and override dialogs."""

    def __init__(
        self, parent=None, *, crop_preview_label: str = "Preview Crop"
    ) -> None:
        super().__init__(parent)
        self._crop_section_default_style = "font-weight: 600; color: #3a6ea5;"
        self._crop_section_active_style = "font-weight: 700; color: #2f855a;"
        self._crop_preview_label = crop_preview_label
        self._crop_preview_available = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.shared_defaults_group = QGroupBox("Shared Processing Defaults")
        defaults_form = QFormLayout(self.shared_defaults_group)
        defaults_form.setSpacing(8)

        self.scale_factor_label = QLabel("Scale Factor:")
        self.scale_factor_slider = QSlider(Qt.Horizontal)
        self.scale_factor_slider.setMinimum(0)
        self.scale_factor_slider.setMaximum(100)
        self.scale_factor_slider.setValue(25)
        self.scale_factor_slider.setTickInterval(25)
        self.scale_factor_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_factor_text = QLineEdit("0.5")
        scale_row = QVBoxLayout()
        scale_row.addWidget(self.scale_factor_slider)
        scale_row.addWidget(self.scale_factor_text)
        defaults_form.addRow(self.scale_factor_label, scale_row)

        self.fps_label = QLabel("Frames Per Second (FPS):")
        self.fps_text = QLineEdit("FPS e.g. 29.97")
        self.override_fps_checkbox = QCheckBox("Use specified FPS for all videos")
        fps_row = QVBoxLayout()
        fps_row.addWidget(self.fps_text)
        fps_row.addWidget(self.override_fps_checkbox)
        defaults_form.addRow(self.fps_label, fps_row)

        self.denoise_checkbox = QCheckBox("Apply Denoise")
        defaults_form.addRow(self.denoise_checkbox)

        self.auto_contrast_checkbox = QCheckBox("Auto Contrast Enhancement")
        self.auto_contrast_strength_label = QLabel("Auto Contrast Strength:")
        self.auto_contrast_strength_slider = QSlider(Qt.Horizontal)
        self.auto_contrast_strength_slider.setMinimum(0)
        self.auto_contrast_strength_slider.setMaximum(200)
        self.auto_contrast_strength_slider.setValue(100)
        self.auto_contrast_strength_slider.setTickInterval(25)
        self.auto_contrast_strength_slider.setTickPosition(QSlider.TicksBelow)
        self.auto_contrast_strength_text = QLineEdit("1.0")
        auto_contrast_row = QVBoxLayout()
        auto_contrast_row.addWidget(self.auto_contrast_checkbox)
        auto_contrast_row.addWidget(self.auto_contrast_strength_slider)
        auto_contrast_row.addWidget(self.auto_contrast_strength_text)
        defaults_form.addRow(self.auto_contrast_strength_label, auto_contrast_row)

        layout.addWidget(self.shared_defaults_group)

        self.crop_group = QGroupBox("Crop Region")
        crop_form = QFormLayout(self.crop_group)
        crop_form.setSpacing(8)
        self.crop_section_label = QLabel("Crop Region")
        self.crop_checkbox = QCheckBox("Enable Crop Region")
        self.crop_label = QLabel("Crop Region (x, y, width, height):")
        self.crop_x_text = QLineEdit()
        self.crop_x_text.setPlaceholderText("x")
        self.crop_y_text = QLineEdit()
        self.crop_y_text.setPlaceholderText("y")
        self.crop_width_text = QLineEdit()
        self.crop_width_text.setPlaceholderText("width")
        self.crop_height_text = QLineEdit()
        self.crop_height_text.setPlaceholderText("height")
        crop_coords_row = QHBoxLayout()
        crop_coords_row.addWidget(self.crop_x_text)
        crop_coords_row.addWidget(self.crop_y_text)
        crop_coords_row.addWidget(self.crop_width_text)
        crop_coords_row.addWidget(self.crop_height_text)
        crop_form.addRow(self.crop_checkbox)
        crop_form.addRow(self.crop_label, crop_coords_row)
        self.crop_preview_button = QPushButton(self._crop_preview_label)
        self.crop_preview_button.setToolTip("Select an input video or folder first.")
        crop_form.addRow(self.crop_preview_button)
        layout.addWidget(self.crop_group)

        self._toggle_auto_contrast_controls(False)
        self._toggle_crop_controls(False)

    def connect_signals(
        self,
        *,
        scale_slider_changed=None,
        scale_text_finished=None,
        fps_text_finished=None,
        override_fps_toggled=None,
        denoise_toggled=None,
        auto_contrast_toggled=None,
        auto_contrast_strength_slider_changed=None,
        auto_contrast_strength_text_finished=None,
        crop_toggled=None,
        crop_preview_clicked=None,
    ) -> None:
        if callable(scale_slider_changed):
            self.scale_factor_slider.valueChanged.connect(scale_slider_changed)
        if callable(scale_text_finished):
            self.scale_factor_text.editingFinished.connect(scale_text_finished)
        if callable(fps_text_finished):
            self.fps_text.editingFinished.connect(fps_text_finished)
        if callable(override_fps_toggled):
            self.override_fps_checkbox.toggled.connect(override_fps_toggled)
        if callable(denoise_toggled):
            self.denoise_checkbox.toggled.connect(denoise_toggled)
        if callable(auto_contrast_toggled):
            self.auto_contrast_checkbox.toggled.connect(auto_contrast_toggled)
        if callable(auto_contrast_strength_slider_changed):
            self.auto_contrast_strength_slider.valueChanged.connect(
                auto_contrast_strength_slider_changed
            )
        if callable(auto_contrast_strength_text_finished):
            self.auto_contrast_strength_text.editingFinished.connect(
                auto_contrast_strength_text_finished
            )
        if callable(crop_toggled):
            self.crop_checkbox.toggled.connect(crop_toggled)
        if callable(crop_preview_clicked):
            self.crop_preview_button.clicked.connect(crop_preview_clicked)

    def set_scale_factor(self, scale_factor: float) -> None:
        self.scale_factor_slider.setValue(int(max(0.0, min(scale_factor, 1.0)) * 100))
        self.scale_factor_text.setText(str(scale_factor))

    def set_fps(self, fps: float | None) -> None:
        self.fps_text.setText("" if fps is None else str(fps))

    def set_auto_contrast_strength(self, strength: float) -> None:
        self.auto_contrast_strength_slider.setValue(
            int(max(0.0, min(strength, 2.0)) * 100)
        )
        self.auto_contrast_strength_text.setText(f"{strength:.2f}")

    def set_crop_region(self, crop_region: CropRegion | None) -> None:
        if crop_region is None:
            self.crop_checkbox.setChecked(False)
            self.crop_x_text.clear()
            self.crop_y_text.clear()
            self.crop_width_text.clear()
            self.crop_height_text.clear()
            self._toggle_crop_controls(False)
            return
        self.crop_checkbox.setChecked(True)
        crop_x, crop_y, crop_width, crop_height = crop_region.as_tuple()
        self.crop_x_text.setText(str(crop_x))
        self.crop_y_text.setText(str(crop_y))
        self.crop_width_text.setText(str(crop_width))
        self.crop_height_text.setText(str(crop_height))
        self._toggle_crop_controls(True)

    def apply_settings(self, settings: dict[str, object]) -> None:
        self.set_scale_factor(float(settings.get("scale_factor", 0.5)))
        self.set_fps(settings.get("fps"))
        self.denoise_checkbox.setChecked(bool(settings.get("apply_denoise", False)))
        auto_contrast = bool(settings.get("auto_contrast", False))
        self.auto_contrast_checkbox.setChecked(auto_contrast)
        self.set_auto_contrast_strength(
            float(settings.get("auto_contrast_strength", 1.0))
        )
        crop_params = settings.get("crop_params")
        has_crop = (
            isinstance(crop_params, (tuple, list))
            and len(crop_params) == 4
            and all(isinstance(value, int) for value in crop_params)
        )
        if has_crop:
            self.set_crop_region(CropRegion.from_values(*crop_params))
        else:
            self.set_crop_region(None)
        self._toggle_auto_contrast_controls(auto_contrast)
        self._toggle_crop_controls(has_crop)

    def collect_settings(self, parent=None) -> dict[str, object] | None:
        try:
            scale_factor = float(self.scale_factor_text.text())
        except ValueError:
            self.scale_factor_text.setText("Invalid Value")
            return None
        if not 0.0 <= scale_factor <= 1.0:
            self.scale_factor_text.setText("Invalid Value")
            return None

        fps_text = self.fps_text.text().strip()
        if fps_text:
            try:
                fps = float(fps_text)
            except ValueError:
                self._warn(parent, "Invalid Input", "FPS must be a number.")
                return None
            if fps <= 0:
                self._warn(parent, "Invalid Input", "FPS must be > 0.")
                return None
        else:
            fps = None

        auto_contrast = self.auto_contrast_checkbox.isChecked()
        if auto_contrast:
            try:
                auto_contrast_strength = float(
                    self.auto_contrast_strength_text.text() or 1.0
                )
            except ValueError:
                self._warn(
                    parent,
                    "Invalid Input",
                    "Auto contrast strength must be a number.",
                )
                return None
        else:
            auto_contrast_strength = 1.0

        crop_params: tuple[int, int, int, int] | None = None
        if self.crop_checkbox.isChecked():
            try:
                crop_x = int(self.crop_x_text.text())
                crop_y = int(self.crop_y_text.text())
                crop_width = int(self.crop_width_text.text())
                crop_height = int(self.crop_height_text.text())
            except ValueError:
                self._warn(
                    parent,
                    "Invalid Input",
                    "Crop region values must be integers.",
                )
                return None
            crop_region = CropRegion.from_values(
                crop_x, crop_y, crop_width, crop_height
            )
            if crop_region is None:
                self._warn(
                    parent,
                    "Invalid Input",
                    "Crop width and height must be positive integers.",
                )
                return None
            crop_params = crop_region.as_tuple()

        return {
            "scale_factor": scale_factor,
            "fps": fps,
            "apply_denoise": self.denoise_checkbox.isChecked(),
            "auto_contrast": auto_contrast,
            "auto_contrast_strength": auto_contrast_strength,
            "crop_params": crop_params,
        }

    def _warn(self, parent, title: str, message: str) -> None:
        QMessageBox.warning(parent or self, title, message)

    def _toggle_auto_contrast_controls(self, enabled: bool) -> None:
        self.auto_contrast_strength_label.setEnabled(bool(enabled))
        self.auto_contrast_strength_slider.setEnabled(bool(enabled))
        self.auto_contrast_strength_text.setEnabled(bool(enabled))

    def _toggle_crop_controls(self, enabled: bool) -> None:
        controls_enabled = bool(enabled)
        self.crop_x_text.setEnabled(controls_enabled)
        self.crop_y_text.setEnabled(controls_enabled)
        self.crop_width_text.setEnabled(controls_enabled)
        self.crop_height_text.setEnabled(controls_enabled)
        self.crop_preview_button.setEnabled(self._crop_preview_available)

    def set_crop_preview_available(self, available: bool) -> None:
        self._crop_preview_available = bool(available)
        self.crop_preview_button.setEnabled(self._crop_preview_available)
        self.crop_preview_button.setToolTip(
            ""
            if self._crop_preview_available
            else "Select an input video or folder first."
        )

    def set_crop_section_active(self, active: bool) -> None:
        self.crop_section_label.setStyleSheet(
            self._crop_section_active_style
            if active
            else self._crop_section_default_style
        )
        self.crop_preview_button.setStyleSheet(
            "font-weight: 600; color: #2f855a;" if active else ""
        )
        self.crop_checkbox.setStyleSheet(
            "font-weight: 600; color: #2f855a;" if active else ""
        )
