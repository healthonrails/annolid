from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QMessageBox, QDialog

from annolid.data.videos import get_video_fps
from annolid.gui.widgets.crop_dialog import CropDialog
from annolid.gui.widgets.video_frame_preview import temporary_first_frame_image
from annolid.utils.video_processing_reports import save_processing_summary
from annolid.utils.videos import VIDEO_EXTENSIONS, compress_and_rescale_video


class VideoRescaleWorkflow:
    def __init__(self, dialog):
        self.dialog = dialog

    def apply_initial_video(self, initial_video_path: str | None) -> None:
        if initial_video_path and os.path.isfile(initial_video_path):
            self.dialog.input_video_path = initial_video_path
            self.update_input_selection_label()
            self.update_fps_from_first_video(initial_video_path)

    def selected_video_paths(self):
        if self.dialog.input_video_path and os.path.isfile(
            self.dialog.input_video_path
        ):
            return [self.dialog.input_video_path]
        if self.dialog.input_folder_path and os.path.isdir(
            self.dialog.input_folder_path
        ):
            return [
                str(Path(self.dialog.input_folder_path) / f)
                for f in sorted(os.listdir(self.dialog.input_folder_path))
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]
        return []

    def update_input_selection_label(self):
        if self.dialog.input_video_path:
            self.dialog.input_selection_label.setText(
                f"Video: {self.dialog.input_video_path}"
            )
            return
        if self.dialog.input_folder_path:
            count = len(self.selected_video_paths())
            self.dialog.input_selection_label.setText(
                f"Folder: {self.dialog.input_folder_path} ({count} video files)"
            )
            return
        self.dialog.input_selection_label.setText("No input selected")

    def update_fps_from_first_video(self, video_path):
        fps = get_video_fps(video_path)
        if fps:
            self.dialog.fps_text.setText(str(fps))

    def select_input_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self.dialog,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.mpeg *.mpg *.m4v *.mts)",
        )
        if not video_path:
            return

        self.dialog.input_video_path = video_path
        self.dialog.input_folder_path = ""
        self.update_input_selection_label()
        self.update_fps_from_first_video(video_path)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Input Folder")
        if not folder:
            return

        self.dialog.input_folder_path = folder
        self.dialog.input_video_path = ""
        self.update_input_selection_label()
        video_files = self.selected_video_paths()
        if video_files:
            self.update_fps_from_first_video(video_files[0])

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.output_folder_path = folder
            self.dialog.output_folder_label.setText(f"Output Folder: {folder}")

    def update_scale_factor_from_slider(self):
        scale_factor = self.dialog.scale_factor_slider.value() / 100
        self.dialog.scale_factor_text.setText(str(scale_factor))

    def update_scale_factor_from_text(self):
        try:
            scale_factor = float(self.dialog.scale_factor_text.text())
            if 0.0 <= scale_factor <= 1.0:
                self.dialog.scale_factor_slider.setValue(int(scale_factor * 100))
            else:
                self.dialog.scale_factor_text.setText("Invalid Value")
        except ValueError:
            self.dialog.scale_factor_text.setText("Invalid Value")

    def update_auto_contrast_strength_from_slider(self):
        strength = self.dialog.auto_contrast_strength_slider.value() / 100
        self.dialog.auto_contrast_strength_text.setText(f"{strength:.2f}")

    def update_auto_contrast_strength_from_text(self):
        try:
            strength = float(self.dialog.auto_contrast_strength_text.text())
            if 0.0 <= strength <= 2.0:
                self.dialog.auto_contrast_strength_slider.setValue(int(strength * 100))
                self.dialog.auto_contrast_strength_text.setText(f"{strength:.2f}")
            else:
                self.dialog.auto_contrast_strength_text.setText("1.00")
                self.dialog.auto_contrast_strength_slider.setValue(100)
        except ValueError:
            self.dialog.auto_contrast_strength_text.setText("1.00")
            self.dialog.auto_contrast_strength_slider.setValue(100)

    def toggle_auto_contrast_controls(self, enabled):
        self.dialog.auto_contrast_strength_label.setEnabled(bool(enabled))
        self.dialog.auto_contrast_strength_slider.setEnabled(bool(enabled))
        self.dialog.auto_contrast_strength_text.setEnabled(bool(enabled))

    def preview_and_crop(self):
        video_files = self.selected_video_paths()
        if not video_files:
            QMessageBox.warning(
                self.dialog,
                "Error",
                "Select a video file or folder that contains videos.",
            )
            return
        first_video = video_files[0]

        try:
            with temporary_first_frame_image(first_video) as temp_image_path:
                crop_dialog = CropDialog(temp_image_path, parent=self.dialog)
                if crop_dialog.exec_() == QDialog.Accepted:
                    crop_coords = crop_dialog.getCropCoordinates()
                    if crop_coords:
                        crop_x, crop_y, crop_width, crop_height = crop_coords
                        self.dialog.crop_x_text.setText(str(crop_x))
                        self.dialog.crop_y_text.setText(str(crop_y))
                        self.dialog.crop_width_text.setText(str(crop_width))
                        self.dialog.crop_height_text.setText(str(crop_height))
                        QMessageBox.information(
                            self.dialog,
                            "Crop Selected",
                            f"Crop Region set to:\nx: {crop_x}, y: {crop_y}, width: {crop_width}, height: {crop_height}",
                        )
                    else:
                        QMessageBox.information(
                            self.dialog, "No Crop", "No crop region was selected."
                        )
        except RuntimeError as exc:
            QMessageBox.warning(self.dialog, "Error", str(exc))

    def _set_run_busy(self, busy: bool) -> None:
        self.dialog.run_button.setEnabled(not busy)
        self.dialog.run_button.setText("Processing..." if busy else "Run Processing")

    def run_rescaling(self):
        self._set_run_busy(True)
        try:
            selected_videos = self.selected_video_paths()
            output_folder = self.dialog.output_folder_path

            if not selected_videos:
                QMessageBox.warning(
                    self.dialog,
                    "Error",
                    "Please select a valid input video or a folder with videos.",
                )
                return

            try:
                scale_factor = float(self.dialog.scale_factor_text.text())
            except ValueError:
                QMessageBox.warning(self.dialog, "Error", "Invalid scale factor.")
                return

            if self.dialog.override_fps_checkbox.isChecked():
                try:
                    fps = float(self.dialog.fps_text.text())
                except ValueError:
                    QMessageBox.warning(self.dialog, "Error", "Invalid FPS value.")
                    return
            else:
                fps = None

            rescale = self.dialog.rescale_checkbox.isChecked()
            collect_only = self.dialog.collect_only_checkbox.isChecked()
            if not rescale and not collect_only:
                QMessageBox.warning(
                    self.dialog,
                    "Error",
                    "Select at least one action: Rescale or Collect Metadata.",
                )
                return

            apply_denoise = self.dialog.denoise_checkbox.isChecked()
            auto_contrast = self.dialog.auto_contrast_checkbox.isChecked()
            if auto_contrast:
                try:
                    auto_contrast_strength = float(
                        self.dialog.auto_contrast_strength_text.text()
                    )
                except ValueError:
                    QMessageBox.warning(
                        self.dialog, "Error", "Invalid auto contrast strength."
                    )
                    return
            else:
                auto_contrast_strength = 1.0

            crop_params = None
            if self.dialog.crop_checkbox.isChecked():
                try:
                    crop_x = int(self.dialog.crop_x_text.text())
                    crop_y = int(self.dialog.crop_y_text.text())
                    crop_width = int(self.dialog.crop_width_text.text())
                    crop_height = int(self.dialog.crop_height_text.text())
                    crop_params = (crop_x, crop_y, crop_width, crop_height)
                except ValueError:
                    QMessageBox.warning(
                        self.dialog,
                        "Error",
                        "Invalid crop parameters. Please enter integer values.",
                    )
                    return

            is_single_input = bool(self.dialog.input_video_path)
            inferred_input_folder = (
                str(Path(self.dialog.input_video_path).parent)
                if is_single_input
                else self.dialog.input_folder_path
            )

            if collect_only:
                save_processing_summary(
                    inferred_input_folder,
                    video_paths=selected_videos if is_single_input else None,
                )
                QMessageBox.information(
                    self.dialog, "Done", "Metadata collection is done."
                )

            if rescale:
                effective_output_folder = output_folder
                if not effective_output_folder and is_single_input:
                    source_video = Path(self.dialog.input_video_path)
                    effective_output_folder = str(
                        source_video.with_name(f"{source_video.stem}_downsampled")
                    )
                    self.dialog.output_folder_path = effective_output_folder
                    self.dialog.output_folder_label.setText(
                        f"Output Folder: {effective_output_folder}"
                    )
                if not effective_output_folder:
                    QMessageBox.warning(
                        self.dialog, "Error", "Please select a valid output folder."
                    )
                    return

                command_log = compress_and_rescale_video(
                    inferred_input_folder,
                    effective_output_folder,
                    scale_factor,
                    input_video_path=self.dialog.input_video_path or None,
                    fps=fps,
                    apply_denoise=apply_denoise,
                    auto_contrast=auto_contrast,
                    auto_contrast_strength=auto_contrast_strength,
                    crop_x=crop_params[0] if crop_params else None,
                    crop_y=crop_params[1] if crop_params else None,
                    crop_width=crop_params[2] if crop_params else None,
                    crop_height=crop_params[3] if crop_params else None,
                )
                save_processing_summary(
                    effective_output_folder,
                    video_paths=selected_videos if is_single_input else None,
                    scale_factor=scale_factor,
                    fps=fps,
                    apply_denoise=apply_denoise,
                    auto_contrast=auto_contrast,
                    auto_contrast_strength=auto_contrast_strength,
                    crop_params=crop_params,
                    command_log=command_log,
                )
                input_video_count = len(selected_videos)
                success_count = len(command_log)
                failed_count = max(0, input_video_count - success_count)
                summary = (
                    "Video processing complete.\n\n"
                    f"Successful: {success_count}\n"
                    f"Failed: {failed_count}\n"
                    f"Output folder: {effective_output_folder}\n\n"
                    "Tip: If failures remain, disable denoise and keep Auto Contrast on."
                )
                if failed_count > 0:
                    QMessageBox.warning(self.dialog, "Completed with Warnings", summary)
                else:
                    QMessageBox.information(self.dialog, "Done", summary)
        finally:
            self._set_run_busy(False)
