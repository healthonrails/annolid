from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog
from annolid.gui.widgets.segment_editor import SegmentEditorDialog
from annolid.utils.logger import logger


class CoreInteractionMixin:
    """Core canvas interaction and advanced-parameter workflows."""

    def paintCanvas(self):
        if self.image.isNull():
            return
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.updateGeometry()
        if getattr(self, "_viewer_stack", None) is not None:
            self._viewer_stack.updateGeometry()
            if not getattr(self, "isPlaying", False):
                self._viewer_stack.adjustSize()
        else:
            if not getattr(self, "isPlaying", False):
                self.canvas.adjustSize()
        self.canvas.update()

    @QtCore.Slot()
    def _open_segment_editor_dialog(self):
        if not self.video_file or self.fps is None or self.num_frames is None:
            QtWidgets.QMessageBox.information(
                self, "No Video Loaded", "Please load a video first."
            )
            return

        initial_segment_dicts = [
            s.to_dict() for s in self._current_video_defined_segments
        ]

        dialog = SegmentEditorDialog(
            active_video_path=Path(self.video_file),
            active_video_fps=self.fps,
            active_video_total_frames=self.num_frames,
            current_annolid_frame=self.frame_number,
            initial_segments_data=initial_segment_dicts,
            annolid_config=self.config,
            parent=self,
        )

        dialog.tracking_initiated.connect(self.tracking_controller.start_tracking)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._current_video_defined_segments = dialog.get_defined_segments()
            logger.info(
                f"Segment Editor OK. {len(self._current_video_defined_segments)} segments stored."
            )
            self._save_segments_for_active_video()
            if self.caption_widget is not None:
                self.caption_widget.set_video_segments(
                    self._current_video_defined_segments
                )
        else:
            logger.info("Segment Editor Cancelled/Closed.")

        dialog.deleteLater()

    def is_tracking_busy(self) -> bool:
        return self.tracking_controller.is_tracking_busy()

    def _grounding_sam(self):
        self.toggleDrawMode(False, createMode="grounding_sam")
        prompt_text = self.aiRectangle._aiRectanglePrompt.text().lower()

        if len(prompt_text) < 1:
            logger.info(f"Invalid text prompt '{prompt_text}'")
            return

        use_countgd = False
        try:
            if hasattr(self, "aiRectangle") and hasattr(
                self.aiRectangle, "_useCountGDCheckbox"
            ):
                use_countgd = self.aiRectangle._useCountGDCheckbox.isChecked()
        except Exception:
            use_countgd = False

        if prompt_text.startswith("flags:"):
            flags = {
                k.strip(): False
                for k in prompt_text.replace("flags:", "").split(",")
                if len(k.strip()) > 0
            }
            if len(flags.keys()) > 0:
                self.flags_controller.apply_prompt_flags(flags)
            else:
                self.flags_controller.clear_flags()
        else:
            self.canvas.predictAiRectangle(prompt_text, use_countgd=use_countgd)

    def _current_text_prompt(self) -> Optional[str]:
        prompt = None
        try:
            if hasattr(self, "aiRectangle") and hasattr(
                self.aiRectangle, "_aiRectanglePrompt"
            ):
                widget = self.aiRectangle._aiRectanglePrompt
                if widget:
                    prompt = widget.text().strip() or None
        except Exception:
            prompt = None
        return prompt

    def update_step_size(self, value):
        self.step_size = value
        self.stepSizeWidget.set_value(self.step_size)

    def set_advanced_params(self):
        sam3_defaults = self.sam3_manager.dialog_defaults(self._config)
        advanced_params_dialog = AdvancedParametersDialog(
            self,
            tracker_config=self.tracker_runtime_config,
            sam3_runtime=sam3_defaults,
        )
        of_manager = getattr(self, "optical_flow_manager", None)
        advanced_params_dialog.compute_optical_flow_checkbox.setChecked(
            bool(getattr(of_manager, "compute_optical_flow", True))
        )
        advanced_params_dialog.optical_flow_backend = getattr(
            of_manager, "optical_flow_backend", "farneback"
        )
        backend_val = str(advanced_params_dialog.optical_flow_backend).lower()
        if "raft" in backend_val:
            backend_idx = 2
        elif "torch" in backend_val:
            backend_idx = 1
        else:
            backend_idx = 0
        advanced_params_dialog.optical_flow_backend_combo.setCurrentIndex(backend_idx)
        if advanced_params_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        self.epsilon_for_polygon = advanced_params_dialog.get_epsilon_value()
        self.automatic_pause_enabled = (
            advanced_params_dialog.is_automatic_pause_enabled()
        )
        self.t_max_value = advanced_params_dialog.get_t_max_value()
        self.use_cpu_only = advanced_params_dialog.is_cpu_only_enabled()
        self.save_video_with_color_mask = (
            advanced_params_dialog.is_save_video_with_color_mask_enabled()
        )
        self.auto_recovery_missing_instances = (
            advanced_params_dialog.is_auto_recovery_missing_instances_enabled()
        )
        if of_manager is not None:
            of_manager.set_compute_optical_flow(
                advanced_params_dialog.is_compute_optiocal_flow_enabled()
            )
            of_manager.set_backend(advanced_params_dialog.get_optical_flow_backend())

        tracker_settings = advanced_params_dialog.get_tracker_settings()
        for key, value in tracker_settings.items():
            setattr(self.tracker_runtime_config, key, value)

        self.sam3_manager.apply_dialog_results(advanced_params_dialog, self._config)

        of_manager = getattr(self, "optical_flow_manager", None)
        logger.info(
            "Computing optical flow is %s .",
            getattr(of_manager, "compute_optical_flow", True),
        )
        logger.info("Set epsilon for polygon to : %s", self.epsilon_for_polygon)
