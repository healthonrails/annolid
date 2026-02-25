from __future__ import annotations

import time
from typing import Any, Dict

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.step_size_widget import StepSizeWidget
from annolid.gui.workers import LoadFrameThread
from annolid.realtime.config import Config as RealtimeConfig
from annolid.utils.logger import logger


class WindowLifecycleMixin:
    """File close lifecycle, realtime proxies, and window geometry persistence."""

    def _close_active_non_canvas_view(self) -> bool:
        """Close currently active PDF/Web/3D viewer and return to canvas."""
        viewer_stack = getattr(self, "_viewer_stack", None)
        if viewer_stack is None:
            return False
        try:
            current = viewer_stack.currentWidget()
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Unable to determine active viewer widget: %s", exc)
            return False

        try:
            pdf_manager = getattr(self, "pdf_manager", None)
            if pdf_manager is not None and current is pdf_manager.pdf_widget():
                pdf_manager.close_pdf()
                return True
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed closing active PDF view: %s", exc)

        try:
            web_manager = getattr(self, "web_manager", None)
            if web_manager is not None and current is web_manager.viewer_widget():
                web_manager.close_web()
                return True
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed closing active web view: %s", exc)

        try:
            threejs_manager = getattr(self, "threejs_manager", None)
            if (
                threejs_manager is not None
                and current is threejs_manager.viewer_widget()
            ):
                threejs_manager.close_threejs()
                return True
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed closing active ThreeJS view: %s", exc)
        return False

    def closeFile(self, _value=False, *, suppress_tracking_prompt=False):
        start_ts = time.perf_counter()
        logger.info(
            "Lifecycle close started (video=%s).",
            str(getattr(self, "video_file", None) or ""),
        )
        if self._close_active_non_canvas_view():
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Lifecycle close redirected to non-canvas view close in %.1fms.",
                elapsed_ms,
            )
            return
        if not self.mayContinue():
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Lifecycle close aborted by mayContinue() in %.1fms.", elapsed_ms
            )
            return
        self._closefile_reset_view_and_core_state()
        self._closefile_reset_audio_and_slider_state()
        self._closefile_reset_tracking_prediction_state()
        if not self._closefile_handle_tracking_stop_prompt(
            suppress_tracking_prompt=suppress_tracking_prompt
        ):
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Lifecycle close aborted by tracking prompt in %.1fms.", elapsed_ms
            )
            return

        super().closeFile(_value)

        self.open_segment_editor_action.setEnabled(False)
        self._current_video_defined_segments = []
        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        logger.info("File closed in AnnolidWindow (%.1fms).", elapsed_ms)

    def _closefile_reset_view_and_core_state(self) -> None:
        self._set_active_view("canvas")
        self.resetState()
        self.dino_controller.deactivate_patch_similarity()
        self.dino_controller.deactivate_pca_map()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)
        self.uniqLabelList.clear()
        self.fileListWidget.clear()
        self.video_loader = None
        self.num_frames = None
        self.video_file = None
        self._fit_window_applied_video_key = None
        try:
            if hasattr(self, "embedding_search_widget"):
                self.embedding_search_widget.set_video_path(None)
                self.embedding_search_widget.set_query_frame_index(None)
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed clearing embedding search state: %s", exc)
        if self.caption_widget is not None:
            self.caption_widget.set_video_context(None, None, None)
            self.caption_widget.set_video_segments([])

    def _closefile_reset_audio_and_slider_state(self) -> None:
        self._release_audio_loader()
        if self.audio_widget:
            self.audio_widget.set_audio_loader(None)
            self.audio_widget.close()
        self.audio_widget = None
        if self.audio_dock:
            self.audio_dock.close()
        self.audio_dock = None
        self.annotation_dir = None
        if self.seekbar is not None:
            self.statusBar().removeWidget(self.seekbar)
            if self.saveButton is not None:
                self.statusBar().removeWidget(self.saveButton)
            if self.playButton is not None:
                self.statusBar().removeWidget(self.playButton)
            self.behavior_controller.attach_slider(None)
            self.seekbar = None
        self.behavior_controller.attach_annotation_store(None)

    def _closefile_reset_tracking_prediction_state(self) -> None:
        self._df = None
        self._df_deeplabcut = None
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_columns = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        self.label_stats = {}
        self.shape_hash_ids = {}
        self.changed_json_stats = {}
        self._pred_res_folder_suffix = "_tracking_results_labelme"
        self._depth_ndjson_records = {}
        try:
            if self.canvas:
                self.canvas.setDepthPreviewOverlay(None)
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed clearing depth preview overlay: %s", exc)
        if getattr(self, "optical_flow_manager", None) is not None:
            self.optical_flow_manager.clear()
        self.frame_number = 0
        if getattr(self, "timeline_panel", None) is not None:
            self.timeline_panel.clear()
            self.timeline_panel.set_time_range(0, 0)
            self.timeline_panel.set_current_frame(0)
        self._apply_timeline_dock_visibility(video_open=False)
        self.step_size = 5
        self.video_results_folder = None
        self.behavior_controller.clear()
        self.behavior_log_widget.clear()
        self.isPlaying = False
        self._time_stamp = ""
        self.saveButton = None
        self.playButton = None
        self.timer = None
        self.filename = None
        self.canvas.pixmap = None
        self.event_type = None
        self.stepSizeWidget = StepSizeWidget()
        self.prev_shapes = None
        self.pred_worker = None
        self.stop_prediction_flag = False
        self.imageData = None
        self._behavior_modifier_state.clear()
        self._active_subject_name = None
        if hasattr(self, "behavior_controls_widget"):
            self.behavior_controls_widget.set_modifier_states(
                [],
                allowed=self._modifier_ids_from_schema(),
            )
            self.behavior_controls_widget.set_category_badge(None, None)
            self.behavior_controls_widget.show_warning(None)
        self._stop_frame_loader()
        self.frame_loader = LoadFrameThread()
        if self.video_processor is not None and hasattr(
            self.video_processor, "cutie_processor"
        ):
            self.video_processor.cutie_processor = None
        self.video_processor = None
        self.fps = None
        self.only_json_files = False
        self._stop_prediction_folder_watcher()
        if self.seekbar:
            self.seekbar.removeMarksByType("predicted")
            self.seekbar.removeMarksByType("predicted_existing")

    def _closefile_handle_tracking_stop_prompt(
        self, *, suppress_tracking_prompt: bool
    ) -> bool:
        if not self.tracking_controller.is_tracking_busy():
            return True
        if suppress_tracking_prompt or self.tracking_controller.is_track_all_running():
            logger.info(
                "Skipping tracking stop prompt while batch processing is active."
            )
            return True

        reply = QtWidgets.QMessageBox.question(
            self,
            "Tracking in Progress",
            "Stop tracking and close video?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            stop_start = time.perf_counter()
            self.tracking_controller.stop_active_worker()
            stop_elapsed_ms = (time.perf_counter() - stop_start) * 1000.0
            logger.info(
                "Requested active tracking worker stop in %.1fms.", stop_elapsed_ms
            )
            return True
        return False

    def _update_frame_display_and_emit_update(self):
        self._emit_live_frame_update()

    def _show_realtime_control_dialog(self):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager.show_control_dialog()

    def _handle_realtime_start_request(
        self, realtime_config: RealtimeConfig, extras: Dict[str, Any]
    ):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager._handle_realtime_start_request(
                realtime_config, extras
            )

    def start_realtime_inference(
        self, realtime_config: RealtimeConfig, extras: Dict[str, Any]
    ):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager.start_realtime_inference(realtime_config, extras)

    def stop_realtime_inference(self):
        if getattr(self, "realtime_manager", None) is not None:
            self.realtime_manager.stop_realtime_inference()

    def closeEvent(self, event):
        try:
            self.stop_realtime_inference()
        except Exception as exc:
            logger.error(
                "Error stopping realtime inference on exit: %s", exc, exc_info=True
            )
        try:
            self._persist_window_geometry(force=True)
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed persisting window geometry on close: %s", exc)
        try:
            if getattr(self, "sam3_manager", None):
                self.sam3_manager.close_session()
        except Exception as exc:
            logger.warning("Error closing SAM3 session on exit: %s", exc)
        super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        timer = getattr(self, "_window_state_save_timer", None)
        if timer is not None and self.windowState() == QtCore.Qt.WindowNoState:
            timer.start()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        super().moveEvent(event)
        timer = getattr(self, "_window_state_save_timer", None)
        if timer is not None and self.windowState() == QtCore.Qt.WindowNoState:
            timer.start()

    def _persist_window_geometry(self, force: bool = False) -> None:
        if not force and self.windowState() != QtCore.Qt.WindowNoState:
            return
        try:
            self.settings.setValue("window/position", self.pos())
            self.settings.setValue("window/size", self.size())
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed to persist window geometry: %s", exc)
