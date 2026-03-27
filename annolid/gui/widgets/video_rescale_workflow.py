from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtCore
from qtpy.QtWidgets import QFileDialog, QMessageBox, QDialog

from annolid.data.videos import get_video_fps
from annolid.gui.widgets.crop_dialog import CropDialog
from annolid.gui.widgets.crop_region import CropRegion
from annolid.gui.widgets.video_frame_preview import temporary_first_frame_image
from annolid.gui.widgets.video_rescale_worker import VideoRescaleJob, VideoRescaleWorker
from annolid.utils.videos import VIDEO_EXTENSIONS


class VideoRescaleWorkflow(QtCore.QObject):
    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self._thread: QtCore.QThread | None = None
        self._worker: VideoRescaleWorker | None = None
        self._crop_section_default_style = "font-weight: 600; color: #3a6ea5;"
        self._crop_section_active_style = "font-weight: 700; color: #2f855a;"

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
                    crop_region = self._extract_crop_region(crop_dialog)
                    if crop_region is not None:
                        crop_x, crop_y, crop_width, crop_height = crop_region.as_tuple()
                        self.dialog.crop_x_text.setText(str(crop_x))
                        self.dialog.crop_y_text.setText(str(crop_y))
                        self.dialog.crop_width_text.setText(str(crop_width))
                        self.dialog.crop_height_text.setText(str(crop_height))
                        self.dialog.crop_checkbox.setChecked(True)
                        self._set_crop_section_active(True)
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

    def _set_crop_section_active(self, active: bool) -> None:
        if hasattr(self.dialog, "crop_section_label"):
            self.dialog.crop_section_label.setStyleSheet(
                self._crop_section_active_style
                if active
                else self._crop_section_default_style
            )
        if hasattr(self.dialog, "crop_preview_button"):
            self.dialog.crop_preview_button.setStyleSheet(
                "font-weight: 600; color: #2f855a;" if active else ""
            )
        if hasattr(self.dialog, "crop_checkbox"):
            self.dialog.crop_checkbox.setStyleSheet(
                "font-weight: 600; color: #2f855a;" if active else ""
            )

    def _read_crop_region_from_inputs(self) -> CropRegion | None:
        try:
            crop_x = int(self.dialog.crop_x_text.text())
            crop_y = int(self.dialog.crop_y_text.text())
            crop_width = int(self.dialog.crop_width_text.text())
            crop_height = int(self.dialog.crop_height_text.text())
        except ValueError:
            QMessageBox.warning(
                self.dialog,
                "Error",
                "Invalid crop parameters. Please enter integer values.",
            )
            return None

        crop_region = CropRegion.from_values(crop_x, crop_y, crop_width, crop_height)
        if crop_region is None:
            QMessageBox.warning(
                self.dialog,
                "Error",
                "Crop width and height must be positive integers.",
            )
        return crop_region

    def _set_run_busy(self, busy: bool) -> None:
        self.dialog.run_button.setEnabled(not busy)
        self.dialog.run_button.setText("Processing..." if busy else "Run Processing")
        if hasattr(self.dialog, "cancel_button"):
            self.dialog.cancel_button.setVisible(bool(busy))
            self.dialog.cancel_button.setEnabled(bool(busy))
        if hasattr(self.dialog, "progress_bar"):
            self.dialog.progress_bar.setVisible(bool(busy))
            if not busy:
                self.dialog.progress_bar.setValue(0)
        if hasattr(self.dialog, "progress_label") and not busy:
            self.dialog.progress_label.setText("")

    @QtCore.Slot(int, str)
    def _handle_progress(self, value: int, message: str) -> None:
        if hasattr(self.dialog, "progress_bar"):
            self.dialog.progress_bar.setValue(max(0, min(int(value), 100)))
        if hasattr(self.dialog, "progress_label"):
            text = str(message)
            if text.startswith("Encoding "):
                video_name = text.removeprefix("Encoding ").strip()
                text = f"Encoding {int(value)}% - {video_name}"
            self.dialog.progress_label.setText(text)

    @QtCore.Slot()
    def _cleanup_thread(self) -> None:
        thread = self._thread
        worker = self._worker
        self._thread = None
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        if thread is not None:
            thread.deleteLater()

    @QtCore.Slot(dict)
    def _handle_finished(self, result: dict) -> None:
        self._set_run_busy(False)
        summary = str(result.get("summary", "Processing complete."))
        if hasattr(self.dialog, "progress_label"):
            self.dialog.progress_label.setText("Done")
        QMessageBox.information(self.dialog, "Done", summary)

    @QtCore.Slot(str)
    def _handle_failed(self, error_text: str) -> None:
        self._set_run_busy(False)
        if hasattr(self.dialog, "progress_label"):
            self.dialog.progress_label.setText("Failed")
        QMessageBox.warning(self.dialog, "Error", str(error_text))

    @QtCore.Slot()
    def _handle_canceled(self) -> None:
        self._set_run_busy(False)
        if hasattr(self.dialog, "progress_label"):
            self.dialog.progress_label.setText("Cancelled")
        QMessageBox.information(
            self.dialog, "Cancelled", "Video processing was cancelled."
        )

    def _start_worker(self, job: VideoRescaleJob) -> None:
        if self._thread is not None:
            QMessageBox.warning(
                self.dialog,
                "Busy",
                "Video processing is already running.",
            )
            return

        self._thread = QtCore.QThread(self.dialog)
        self._worker = VideoRescaleWorker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._handle_progress)
        self._worker.finished.connect(self._handle_finished)
        self._worker.failed.connect(self._handle_failed)
        self._worker.canceled.connect(self._handle_canceled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.canceled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._set_run_busy(True)
        self._thread.start()

    def cancel_running_job(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if hasattr(self.dialog, "progress_label"):
            self.dialog.progress_label.setText("Cancelling...")
        worker.cancel()

    def run_rescaling(self):
        if self._thread is not None:
            QMessageBox.warning(
                self.dialog,
                "Busy",
                "Video processing is already running.",
            )
            return

        selected_videos = self.selected_video_paths()
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
            crop_region = self._read_crop_region_from_inputs()
            if crop_region is None:
                return
            crop_params = crop_region.as_tuple()

        is_single_input = bool(self.dialog.input_video_path)
        input_mode = "single video" if is_single_input else "folder"
        input_source = (
            self.dialog.input_video_path
            if is_single_input
            else self.dialog.input_folder_path
        )
        inferred_input_folder = (
            str(Path(self.dialog.input_video_path).parent)
            if is_single_input
            else self.dialog.input_folder_path
        )
        effective_output_folder = self.dialog.output_folder_path
        if not effective_output_folder and is_single_input:
            source_video = Path(self.dialog.input_video_path)
            effective_output_folder = str(
                source_video.with_name(f"{source_video.stem}_downsampled")
            )
            self.dialog.output_folder_path = effective_output_folder
            self.dialog.output_folder_label.setText(
                f"Output Folder: {effective_output_folder}"
            )

        job = VideoRescaleJob(
            selected_videos=selected_videos,
            input_mode=input_mode,
            input_source=input_source,
            input_folder=inferred_input_folder,
            output_folder=effective_output_folder,
            scale_factor=scale_factor,
            fps=fps,
            collect_only=collect_only,
            rescale=rescale,
            apply_denoise=apply_denoise,
            auto_contrast=auto_contrast,
            auto_contrast_strength=auto_contrast_strength,
            crop_params=crop_params,
        )
        self._start_worker(job)
