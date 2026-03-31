from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtCore
from qtpy.QtWidgets import QFileDialog, QMessageBox, QDialog

from annolid.data.videos import get_video_fps
from annolid.gui.widgets.crop_dialog import CropDialog
from annolid.gui.widgets.crop_region import CropRegion
from annolid.gui.widgets.video_batch_overrides_dialog import VideoBatchReviewDialog
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
        self._per_video_overrides: dict[str, dict[str, object]] = {}

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
        self._update_crop_preview_availability()
        if self.dialog.input_video_path:
            self.dialog.input_selection_label.setText(
                f"Video: {self.dialog.input_video_path}"
            )
            self.update_per_video_review_label()
            self.update_summary_tab()
            return
        if self.dialog.input_folder_path:
            count = len(self.selected_video_paths())
            self.dialog.input_selection_label.setText(
                f"Folder: {self.dialog.input_folder_path} ({count} video files)"
            )
            self.update_per_video_review_label()
            self.update_summary_tab()
            return
        self.dialog.input_selection_label.setText("No input selected")
        self.update_per_video_review_label()
        self.update_summary_tab()

    def _update_crop_preview_availability(self) -> None:
        if hasattr(self.dialog, "settings_widget"):
            self.dialog.settings_widget.set_crop_preview_available(
                bool(self.selected_video_paths())
            )

    def _default_output_folder(self, input_folder: str | None) -> str:
        if not input_folder:
            return ""
        source_folder = Path(input_folder)
        return str(source_folder.with_name(f"{source_folder.name}_downsampled"))

    def update_per_video_review_label(self) -> None:
        label = getattr(
            self.dialog,
            "per_video_review_label",
            getattr(self.dialog, "per_video_overrides_label", None),
        )
        button = getattr(
            self.dialog,
            "per_video_review_button",
            getattr(self.dialog, "per_video_overrides_button", None),
        )
        if label is None:
            return
        if button is not None:
            button.setEnabled(bool(self.dialog.input_folder_path))
        if self.dialog.input_video_path:
            label.setText("Per-video review: not used for single-video input.")
            return
        if not self.dialog.input_folder_path:
            label.setText("Per-video review: select a folder with videos.")
            return
        video_count = len(self.selected_video_paths())
        override_count = len(self._per_video_overrides)
        if override_count <= 0:
            label.setText(f"Per-video review: none (0/{video_count})")
            return
        label.setText(
            f"Per-video review: {override_count}/{video_count} video(s) customized"
        )
        self.update_summary_tab()

    def update_per_video_overrides_label(self) -> None:
        self.update_per_video_review_label()

    def _trim_per_video_overrides_to_selection(self) -> None:
        selected = {str(Path(path)) for path in self.selected_video_paths()}
        self._per_video_overrides = {
            str(Path(path)): dict(config)
            for path, config in self._per_video_overrides.items()
            if str(Path(path)) in selected
        }

    def update_fps_from_first_video(self, video_path):
        fps = get_video_fps(video_path)
        if fps:
            self.dialog.fps_text.setText(str(fps))
        self.update_summary_tab()

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
        self._per_video_overrides = {}
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
        self._trim_per_video_overrides_to_selection()
        if video_files:
            self.update_fps_from_first_video(video_files[0])

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self.dialog, "Select Output Folder")
        if folder:
            self.dialog.output_folder_path = folder
            self.dialog.output_folder_label.setText(f"Output Folder: {folder}")
            self.update_summary_tab()

    def update_scale_factor_from_slider(self):
        scale_factor = self.dialog.scale_factor_slider.value() / 100
        self.dialog.scale_factor_text.setText(str(scale_factor))
        self.update_summary_tab()

    def update_scale_factor_from_text(self):
        try:
            scale_factor = float(self.dialog.scale_factor_text.text())
            if 0.0 <= scale_factor <= 1.0:
                self.dialog.scale_factor_slider.setValue(int(scale_factor * 100))
            else:
                self.dialog.scale_factor_text.setText("Invalid Value")
        except ValueError:
            self.dialog.scale_factor_text.setText("Invalid Value")
        self.update_summary_tab()

    def update_fps_from_text(self):
        text = self.dialog.fps_text.text().strip()
        if not text:
            self.update_summary_tab()
            return
        try:
            fps = float(text)
        except ValueError:
            self.dialog.fps_text.setText("Invalid Value")
            self.update_summary_tab()
            return
        if fps <= 0:
            self.dialog.fps_text.setText("Invalid Value")
            self.update_summary_tab()
            return
        self.dialog.fps_text.setText(str(fps))
        self.update_summary_tab()

    def toggle_override_fps_controls(self, enabled: bool) -> None:
        self.dialog.fps_text.setEnabled(bool(enabled))
        self.update_summary_tab()

    def update_auto_contrast_strength_from_slider(self):
        strength = self.dialog.auto_contrast_strength_slider.value() / 100
        self.dialog.auto_contrast_strength_text.setText(f"{strength:.2f}")
        self.update_summary_tab()

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
        self.update_summary_tab()

    def toggle_auto_contrast_controls(self, enabled):
        self.dialog.auto_contrast_strength_label.setEnabled(bool(enabled))
        self.dialog.auto_contrast_strength_slider.setEnabled(bool(enabled))
        self.dialog.auto_contrast_strength_text.setEnabled(bool(enabled))
        self.update_summary_tab()

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
        self.update_summary_tab()

    def _read_folder_default_settings(self) -> dict[str, object] | None:
        try:
            scale_factor = float(self.dialog.scale_factor_text.text())
        except ValueError:
            return None

        if not 0.0 <= scale_factor <= 1.0:
            return None

        if self.dialog.override_fps_checkbox.isChecked():
            try:
                fps = float(self.dialog.fps_text.text())
            except ValueError:
                QMessageBox.warning(self.dialog, "Error", "Invalid FPS value.")
                return None
            if fps <= 0:
                QMessageBox.warning(self.dialog, "Error", "FPS must be > 0.")
                return None
        else:
            fps = None

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
                return None
        else:
            auto_contrast_strength = 1.0

        crop_params = None
        if self.dialog.crop_checkbox.isChecked():
            crop_region = self._read_crop_region_from_inputs()
            if crop_region is None:
                return None
            crop_params = crop_region.as_tuple()

        return {
            "scale_factor": scale_factor,
            "fps": fps,
            "apply_denoise": apply_denoise,
            "auto_contrast": auto_contrast,
            "auto_contrast_strength": auto_contrast_strength,
            "crop_params": crop_params,
        }

    def configure_per_video_review(self) -> None:
        if self.dialog.input_video_path:
            QMessageBox.information(
                self.dialog,
                "Per-Video Review",
                "Per-video review is only available when the input is a folder.",
            )
            return
        if not self.dialog.input_folder_path:
            QMessageBox.warning(
                self.dialog,
                "Per-Video Review",
                "Select an input folder first.",
            )
            return
        selected_videos = self.selected_video_paths()
        if not selected_videos:
            QMessageBox.warning(
                self.dialog,
                "Per-Video Review",
                "No supported video files were found in this folder.",
            )
            return
        defaults = self._read_folder_default_settings()
        if defaults is None:
            return

        self._trim_per_video_overrides_to_selection()
        dialog = VideoBatchReviewDialog(
            video_paths=selected_videos,
            default_settings=defaults,
            existing_overrides=self._per_video_overrides,
            parent=self.dialog,
        )
        if dialog.exec_() == QDialog.Accepted:
            self._per_video_overrides = dialog.overrides()
            self.update_per_video_review_label()

    def configure_per_video_overrides(self) -> None:
        self.configure_per_video_review()

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

    def update_summary_tab(self) -> None:
        if not hasattr(self.dialog, "summary_input_label"):
            return

        input_mode = (
            "single video"
            if self.dialog.input_video_path
            else "folder"
            if self.dialog.input_folder_path
            else "none"
        )
        input_source = (
            self.dialog.input_video_path or self.dialog.input_folder_path or "None"
        )
        inferred_input_folder = (
            str(Path(self.dialog.input_video_path).parent)
            if self.dialog.input_video_path
            else self.dialog.input_folder_path
        )
        output_folder = (
            self.dialog.output_folder_path
            or self._default_output_folder(inferred_input_folder)
            or "Not selected"
        )
        processing_lines = []
        processing_lines.append(
            f"Scale factor: {self.dialog.scale_factor_text.text().strip() or '0.5'}"
        )
        fps_text = self.dialog.fps_text.text().strip()
        if self.dialog.override_fps_checkbox.isChecked() and fps_text:
            processing_lines.append(f"FPS: {fps_text}")
        else:
            processing_lines.append("FPS: original per-video FPS")
        processing_lines.append(
            f"Denoise: {'on' if self.dialog.denoise_checkbox.isChecked() else 'off'}"
        )
        processing_lines.append(
            "Auto contrast: "
            f"{'on' if self.dialog.auto_contrast_checkbox.isChecked() else 'off'}"
        )
        if self.dialog.auto_contrast_checkbox.isChecked():
            processing_lines.append(
                "Auto contrast strength: "
                f"{self.dialog.auto_contrast_strength_text.text().strip() or '1.0'}"
            )
        crop_active = self.dialog.crop_checkbox.isChecked()
        if crop_active:
            crop_fields = (
                self.dialog.crop_x_text.text().strip(),
                self.dialog.crop_y_text.text().strip(),
                self.dialog.crop_width_text.text().strip(),
                self.dialog.crop_height_text.text().strip(),
            )
            if all(crop_fields):
                processing_lines.append(
                    "Crop: x="
                    f"{crop_fields[0]}, y={crop_fields[1]}, "
                    f"w={crop_fields[2]}, h={crop_fields[3]}"
                )
            else:
                processing_lines.append("Crop: incomplete")
        else:
            processing_lines.append("Crop: off")

        summary_lines = {
            "input": f"Input: {input_mode} | Source: {input_source}",
            "output": f"Output: {output_folder}",
            "processing": "Processing defaults:\n- " + "\n- ".join(processing_lines),
            "overrides": (
                "Per-video review: not available for single-video input"
                if self.dialog.input_video_path
                else f"Per-video review: {len(self._per_video_overrides)} custom video(s)"
            ),
            "run": (
                "Run actions: "
                f"{'Rescale' if self.dialog.rescale_checkbox.isChecked() else 'No rescale'}, "
                f"{'Metadata only' if self.dialog.collect_only_checkbox.isChecked() else 'no metadata-only'}"
            ),
        }
        self.dialog.update_summary_labels(summary_lines)

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

        settings = self._read_folder_default_settings()
        if settings is None:
            return
        scale_factor = float(settings["scale_factor"])
        fps = settings["fps"]

        rescale = self.dialog.rescale_checkbox.isChecked()
        collect_only = self.dialog.collect_only_checkbox.isChecked()
        if not rescale and not collect_only:
            QMessageBox.warning(
                self.dialog,
                "Error",
                "Select at least one action: Rescale or Collect Metadata.",
            )
            return

        apply_denoise = bool(settings["apply_denoise"])
        auto_contrast = bool(settings["auto_contrast"])
        auto_contrast_strength = float(settings["auto_contrast_strength"])
        crop_params = settings["crop_params"]

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
        if not effective_output_folder:
            effective_output_folder = self._default_output_folder(inferred_input_folder)
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
            per_video_overrides=self._per_video_overrides
            if input_mode == "folder"
            else None,
        )
        self._start_worker(job)
