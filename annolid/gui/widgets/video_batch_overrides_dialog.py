from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from annolid.gui.widgets.crop_dialog import CropDialog
from annolid.gui.widgets.crop_region import CropRegion
from annolid.gui.widgets.video_frame_preview import temporary_first_frame_image
from annolid.gui.widgets.video_processing_settings_widget import (
    VideoProcessingSettingsWidget,
)


class VideoBatchReviewDialog(QDialog):
    """Sequential per-video review dialog for folder input mode."""

    def __init__(
        self,
        *,
        video_paths: list[str],
        default_settings: dict[str, object],
        existing_overrides: dict[str, dict[str, object]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Videos One by One")
        self.resize(760, 560)
        self._video_paths = [str(Path(path)) for path in video_paths]
        self._default_settings = dict(default_settings)
        self._overrides: dict[str, dict[str, object]] = {
            str(Path(path)): dict(config)
            for path, config in (existing_overrides or {}).items()
            if str(Path(path)) in self._video_paths
        }
        self._current_index = 0
        self._building_ui = False
        self._build_ui()
        self._load_video(0)

    def overrides(self) -> dict[str, dict[str, object]]:
        return {path: dict(config) for path, config in self._overrides.items()}

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.help_label = QLabel(
            "Review videos in order. Save only the videos that need custom settings, or Skip to keep the folder defaults."
        )
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-weight: 600; color: #3a6ea5;")
        layout.addWidget(self.progress_label)

        self.current_video_label = QLabel("")
        self.current_video_label.setWordWrap(True)
        layout.addWidget(self.current_video_label)

        self.current_status_label = QLabel("")
        self.current_status_label.setWordWrap(True)
        self.current_status_label.setStyleSheet("color: #6b7280;")
        layout.addWidget(self.current_status_label)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.settings_widget = VideoProcessingSettingsWidget(
            self, crop_preview_label="Preview Crop On Current Video"
        )
        self.settings_widget.connect_signals(
            auto_contrast_toggled=self._on_auto_contrast_toggled,
            crop_toggled=self._on_crop_toggled,
            crop_preview_clicked=self._preview_crop_for_current_video,
        )
        self._bind_settings_aliases()
        layout.addWidget(self.settings_widget)

        actions_row = QHBoxLayout()
        self.load_defaults_button = QPushButton("Load Folder Defaults")
        self.load_defaults_button.clicked.connect(self._load_defaults_into_fields)
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self._go_previous)
        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self._skip_current_video)
        self.save_next_button = QPushButton("Save & Next")
        self.save_next_button.clicked.connect(self._save_current_video_and_advance)
        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self._finish_review)
        actions_row.addWidget(self.load_defaults_button)
        actions_row.addWidget(self.previous_button)
        actions_row.addWidget(self.skip_button)
        actions_row.addWidget(self.save_next_button)
        actions_row.addWidget(self.finish_button)
        layout.addLayout(actions_row)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self._toggle_auto_contrast_strength(False)
        self._toggle_crop_controls(False)
        self.settings_widget.set_crop_preview_available(False)

    def _bind_settings_aliases(self) -> None:
        settings = self.settings_widget
        self.scale_factor_label = settings.scale_factor_label
        self.scale_factor_slider = settings.scale_factor_slider
        self.scale_factor_text = settings.scale_factor_text
        self.fps_label = settings.fps_label
        self.fps_text = settings.fps_text
        self.override_fps_checkbox = settings.override_fps_checkbox
        self.denoise_checkbox = settings.denoise_checkbox
        self.auto_contrast_checkbox = settings.auto_contrast_checkbox
        self.auto_contrast_strength_label = settings.auto_contrast_strength_label
        self.auto_contrast_strength_slider = settings.auto_contrast_strength_slider
        self.auto_contrast_strength_text = settings.auto_contrast_strength_text
        self.crop_section_label = settings.crop_section_label
        self.crop_checkbox = settings.crop_checkbox
        self.crop_label = settings.crop_label
        self.crop_x_text = settings.crop_x_text
        self.crop_y_text = settings.crop_y_text
        self.crop_width_text = settings.crop_width_text
        self.crop_height_text = settings.crop_height_text
        self.crop_preview_button = settings.crop_preview_button

    def _video_count(self) -> int:
        return len(self._video_paths)

    def _active_video_path(self) -> str | None:
        if 0 <= self._current_index < len(self._video_paths):
            return self._video_paths[self._current_index]
        return None

    def _load_video(self, index: int) -> None:
        if not self._video_paths:
            self._update_empty_state()
            return

        self._current_index = max(0, min(index, len(self._video_paths) - 1))
        path = self._active_video_path()
        if path is None:
            self._update_empty_state()
            return

        self._building_ui = True
        try:
            config = self._overrides.get(path)
            if config is None:
                self.settings_widget.apply_settings(self._default_settings)
                status = "using folder defaults"
            else:
                self.settings_widget.apply_settings(config)
                status = "custom settings saved"
        finally:
            self._building_ui = False

        self.settings_widget.set_crop_preview_available(True)
        self._toggle_auto_contrast_strength(
            self.settings_widget.auto_contrast_checkbox.isChecked()
        )
        self._toggle_crop_controls(self.settings_widget.crop_checkbox.isChecked())
        self._update_navigation_state()
        self.progress_label.setText(
            f"Video {self._current_index + 1} of {self._video_count()}"
        )
        self.current_video_label.setText(f"Current video: {Path(path).name}")
        self.current_status_label.setText(
            f"{status}. Save stores a custom override. Skip leaves the folder default in place."
        )
        self._update_summary()

    def _update_empty_state(self) -> None:
        self.progress_label.setText("No supported videos were found.")
        self.current_video_label.setText("Select a folder with video files.")
        self.current_status_label.setText("")
        self.summary_label.setText("Custom overrides: 0 of 0 video(s).")
        self.settings_widget.setEnabled(False)
        self.load_defaults_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.skip_button.setEnabled(False)
        self.save_next_button.setEnabled(False)
        self.finish_button.setEnabled(False)
        self.settings_widget.set_crop_preview_available(False)

    def _update_navigation_state(self) -> None:
        has_videos = bool(self._video_paths)
        if not has_videos:
            return
        self.settings_widget.setEnabled(True)
        self.load_defaults_button.setEnabled(True)
        self.previous_button.setEnabled(self._current_index > 0)
        self.skip_button.setEnabled(True)
        self.save_next_button.setEnabled(True)
        self.finish_button.setEnabled(True)
        if self._current_index >= len(self._video_paths) - 1:
            self.save_next_button.setText("Save & Finish")
            self.skip_button.setText("Skip & Finish")
        else:
            self.save_next_button.setText("Save & Next")
            self.skip_button.setText("Skip")

    def _update_summary(self) -> None:
        override_count = len(self._overrides)
        total = len(self._video_paths)
        current = self._active_video_path()
        if current is None:
            self.summary_label.setText(
                f"Custom overrides: {override_count} of {total} video(s)."
            )
            return
        status = "custom" if current in self._overrides else "folder defaults"
        self.summary_label.setText(
            f"Selected video: {Path(current).name} ({status}). "
            f"Custom overrides: {override_count} of {total} video(s)."
        )

    def _config_matches_defaults(self, config: dict[str, object]) -> bool:
        return dict(config) == self._default_settings

    def _collect_current_config(self) -> dict[str, object] | None:
        return self.settings_widget.collect_settings(parent=self)

    def _save_current_config(self) -> bool:
        path = self._active_video_path()
        if path is None:
            return False
        config = self._collect_current_config()
        if config is None:
            return False
        if self._config_matches_defaults(config):
            self._overrides.pop(path, None)
        else:
            self._overrides[path] = config
        self._refresh_video_label(path)
        self._update_summary()
        return True

    def _refresh_video_label(self, path: str) -> None:
        status = (
            "custom settings saved"
            if path in self._overrides
            else "using folder defaults"
        )
        self.current_status_label.setText(
            f"{status}. Save stores a custom override. Skip leaves the folder default in place."
        )
        self.current_video_label.setText(f"Current video: {Path(path).name}")

    def _advance_to_next(self) -> None:
        if self._current_index >= len(self._video_paths) - 1:
            self.accept()
            return
        self._load_video(self._current_index + 1)

    def _save_current_video_and_advance(self) -> None:
        if not self._save_current_config():
            return
        self._advance_to_next()

    def _skip_current_video(self) -> None:
        path = self._active_video_path()
        if path is None:
            return
        self._overrides.pop(path, None)
        self._refresh_video_label(path)
        self._update_summary()
        self._advance_to_next()

    def _go_previous(self) -> None:
        if self._current_index <= 0:
            return
        self._load_video(self._current_index - 1)

    def _finish_review(self) -> None:
        if self._active_video_path() is not None:
            if not self._save_current_config():
                return
        self.accept()

    def _on_auto_contrast_toggled(self, enabled: bool) -> None:
        self._toggle_auto_contrast_strength(enabled)

    def _on_crop_toggled(self, enabled: bool) -> None:
        self._toggle_crop_controls(enabled)

    def _toggle_auto_contrast_strength(self, enabled: bool) -> None:
        self.settings_widget.auto_contrast_strength_label.setEnabled(bool(enabled))
        self.settings_widget.auto_contrast_strength_slider.setEnabled(bool(enabled))
        self.settings_widget.auto_contrast_strength_text.setEnabled(bool(enabled))

    def _toggle_crop_controls(self, enabled: bool) -> None:
        self.settings_widget._toggle_crop_controls(bool(enabled))

    def _load_defaults_into_fields(self) -> None:
        self.settings_widget.apply_settings(self._default_settings)
        self.settings_widget.set_crop_preview_available(bool(self._active_video_path()))
        self._toggle_auto_contrast_strength(
            self.settings_widget.auto_contrast_checkbox.isChecked()
        )
        self._toggle_crop_controls(self.settings_widget.crop_checkbox.isChecked())

    def _preview_crop_for_current_video(self) -> None:
        video_path = self._active_video_path()
        if not video_path:
            return
        try:
            with temporary_first_frame_image(video_path) as temp_image_path:
                crop_dialog = CropDialog(temp_image_path, parent=self)
                if crop_dialog.exec_() != QDialog.Accepted:
                    return
                crop_region = self._extract_crop_region(crop_dialog)
                if crop_region is None:
                    return
                self.settings_widget.set_crop_region(crop_region)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Crop Error", str(exc))

    def _extract_crop_region(self, crop_dialog) -> CropRegion | None:
        if hasattr(crop_dialog, "getCropRegion"):
            crop_region = crop_dialog.getCropRegion()
            if crop_region is not None:
                return crop_region
        if hasattr(crop_dialog, "getCropCoordinates"):
            crop_coords = crop_dialog.getCropCoordinates()
            if crop_coords:
                return CropRegion.from_values(*crop_coords)
        return None


VideoBatchOverridesDialog = VideoBatchReviewDialog
