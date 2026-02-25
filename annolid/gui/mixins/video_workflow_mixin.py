from __future__ import annotations

from pathlib import Path
import time

from qtpy import QtCore, QtWidgets

from annolid.data import videos
from annolid.gui.widgets.youtube_dialog import YouTubeVideoDialog
from annolid.gui.widgets.video_slider import VideoSlider
from annolid.gui.window_base import QT5
from annolid.utils.logger import logger


class VideoWorkflowMixin:
    """Video loading workflow extracted from the main app window."""

    def _load_video(self, video_path):
        """Open a video for annotation frame by frame."""
        if not video_path:
            return
        self.openVideo(from_video_list=True, video_path=video_path)

    def open_youtube_video(self):
        """Launch the YouTube download dialog and open the selected video."""
        dialog = YouTubeVideoDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.downloaded_path:
            self.openVideo(from_video_list=True, video_path=str(dialog.downloaded_path))

    def handle_extracted_frames(self, dirpath):
        self.importDirImages(dirpath)

    def openVideo(
        self,
        _value=False,
        from_video_list=False,
        video_path=None,
        programmatic_call=False,
    ):
        """Open a video for annotation frame by frame."""
        start_ts = time.perf_counter()
        logger.info(
            "Lifecycle open requested (from_list=%s, programmatic=%s, input=%s).",
            bool(from_video_list),
            bool(programmatic_call),
            str(video_path or ""),
        )
        if not self._confirm_video_switch(programmatic_call=programmatic_call):
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info("Lifecycle open cancelled by user in %.1fms.", elapsed_ms)
            return

        video_filename = self._resolve_video_filename(
            from_video_list=from_video_list,
            video_path=video_path,
        )

        video_filename = str(video_filename)
        self.stepSizeWidget.setEnabled(True)

        if video_filename:
            self._cleanup_audio_ui()
            cur_video_folder = Path(video_filename).parent
            self.video_results_folder = Path(video_filename).with_suffix("")

            self.video_results_folder.mkdir(exist_ok=True, parents=True)
            self.annotation_dir = self.video_results_folder
            self.video_file = video_filename
            try:
                if hasattr(self, "embedding_search_widget"):
                    self.embedding_search_widget.set_video_path(Path(video_filename))
                    self.embedding_search_widget.set_annotation_dir(
                        self.video_results_folder
                    )
            except Exception:
                pass
            self.behavior_controller.attach_annotation_store_for_video(
                self.video_results_folder
            )
            self._refresh_embedding_file_list()
            if getattr(self, "depth_manager", None) is not None:
                self.depth_manager.load_depth_ndjson_records()
            if getattr(self, "optical_flow_manager", None) is not None:
                self.optical_flow_manager.load_records(video_filename)
            try:
                self.video_loader = self._create_video_loader(video_filename)
            except Exception as exc:
                logger.warning(
                    "Lifecycle open loader init failed for %s: %s", video_filename, exc
                )
                QtWidgets.QMessageBox.about(
                    self,
                    "Not a valid media file",
                    "Please check and open a valid video or TIFF stack file.",
                )
                self.video_file = None
                self.video_loader = None
                return
            self._configure_project_schema_for_video(video_filename)
            self.fps = self.video_loader.get_fps()
            self.num_frames = self.video_loader.total_frames()
            self.behavior_log_widget.set_fps(self.fps)
            if self.timeline_panel is not None:
                self.timeline_panel.set_time_range(0, self.num_frames - 1)
                self.timeline_panel.refresh_behavior_catalog()
            self._apply_timeline_dock_visibility(video_open=True)
            if self.caption_widget is not None:
                self.caption_widget.set_video_context(
                    video_filename,
                    self.fps,
                    self.num_frames,
                )
                if getattr(self.caption_widget, "behavior_widget", None) is not None:
                    try:
                        self.caption_widget.behavior_widget.set_current_frame(
                            self.frame_number if self.frame_number is not None else 0
                        )
                    except Exception:
                        pass
            self._configure_audio_for_video(self.video_file, self.fps)
            if self.seekbar:
                self.statusBar().removeWidget(self.seekbar)
            if self.playButton:
                self.statusBar().removeWidget(self.playButton)
            if self.saveButton:
                self.statusBar().removeWidget(self.saveButton)
            self.seekbar = VideoSlider()
            self.behavior_controller.attach_slider(self.seekbar)
            self.seekbar.input_value.returnPressed.connect(self.jump_to_frame)
            self.seekbar.keyPress.connect(self.keyPressEvent)
            self.seekbar.keyRelease.connect(self.keyReleaseEvent)
            logger.info(f"Working on video:{self.video_file}.")
            logger.info(f"FPS: {self.fps}, Total number of frames: {self.num_frames}")

            self.seekbar.valueChanged.connect(
                lambda f: self.set_frame_number(self.seekbar.value())
            )

            self.seekbar.setMinimum(0)
            self.seekbar.setMaximum(self.num_frames - 1)
            self.seekbar.setEnabled(True)
            self.seekbar.resizeEvent()
            self.seekbar.setTooltipCallable(self.tooltip_callable)
            try:
                self._refresh_manual_seed_slider_marks(self.video_results_folder)
            except Exception:
                logger.debug(
                    "Failed to refresh manual seed slider marks.", exc_info=True
                )
            self.playButton = QtWidgets.QPushButton("Play", self)
            self.playButton.setIcon(
                QtWidgets.QApplication.style().standardIcon(
                    QtWidgets.QStyle.SP_MediaPlay
                )
            )
            self.playButton.clicked.connect(self.togglePlay)
            self.saveButton = QtWidgets.QPushButton("Save Timestamps", self)
            self.saveButton.clicked.connect(self.saveTimestampList)
            self.statusBar().addPermanentWidget(self.playButton)
            self.statusBar().addPermanentWidget(self.seekbar, stretch=1)
            self.statusBar().addPermanentWidget(self.saveButton)

            self.frame_loader.video_loader = self.video_loader
            self.frame_loader.moveToThread(self.frame_worker)
            self.frame_loader.res_frame.connect(self._on_frame_loaded)
            if not self.frame_worker.isRunning():
                self.frame_worker.start(priority=QtCore.QThread.IdlePriority)

            self.set_frame_number(self.frame_number)
            self._apply_timeline_caption_if_available(
                self.frame_number, only_if_empty=True
            )

            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)
            self.load_tracking_results(cur_video_folder, video_filename)

            if self.filename:
                self.open_segment_editor_action.setEnabled(True)
                self._load_segments_for_active_video()
                if self.caption_widget is not None:
                    self.caption_widget.set_video_segments(
                        self._current_video_defined_segments
                    )
                if not programmatic_call:
                    self._emit_live_frame_update()
                logger.info(
                    f"Video '{self.filename}' loaded. Segment definition enabled."
                )
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                logger.info("Lifecycle open completed in %.1fms.", elapsed_ms)
            else:
                self.open_segment_editor_action.setEnabled(False)
                self._current_video_defined_segments = []
                if self.caption_widget is not None:
                    self.caption_widget.set_video_segments([])
        else:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Lifecycle open ended without selecting a video (%.1fms).", elapsed_ms
            )

    def _confirm_video_switch(self, *, programmatic_call: bool) -> bool:
        if programmatic_call or not (self.dirty or self.video_loader is not None):
            return True
        message_box = QtWidgets.QMessageBox()
        message_box.setWindowTitle("Unsaved Changes or Closing the Existing Video")
        message_box.setText(
            "The existing video will be closed,\n"
            "and any unsaved changes may be lost.\n"
            "Do you want to continue and open the new video?"
        )
        message_box.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        choice = message_box.exec()
        if choice == QtWidgets.QMessageBox.Ok:
            self.closeFile()
            return True
        return False

    def _resolve_video_filename(
        self,
        *,
        from_video_list: bool,
        video_path,
    ):
        if from_video_list:
            return video_path
        start_dir = Path(self.filename).parent if self.filename else "."
        formats = ["*.*"]
        filters = self.tr(f"Video files {formats[0]}")
        video_filename = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Annolid - Choose Video"),
            str(start_dir),
            filters,
        )
        if QT5:
            video_filename, _ = video_filename
        return video_filename

    @staticmethod
    def _create_video_loader(video_filename: str):
        suffix_lower = Path(video_filename).suffix.lower()
        if suffix_lower in {".tif", ".tiff"} or video_filename.lower().endswith(
            (".ome.tif", ".ome.tiff")
        ):
            return videos.TiffStackVideo(video_filename)
        return videos.CV2Video(video_filename)
