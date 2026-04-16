from __future__ import annotations

from pathlib import Path
import time
from typing import Callable
import os

from qtpy import QtCore, QtWidgets

from annolid.data import videos
from annolid.io.large_image.common import is_large_tiff_path
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
        step_ts = start_ts

        def _log_step(name: str) -> None:
            nonlocal step_ts
            now = time.perf_counter()
            logger.info(
                "Lifecycle open step '%s' took %.1fms (total %.1fms).",
                name,
                (now - step_ts) * 1000.0,
                (now - start_ts) * 1000.0,
            )
            step_ts = now

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
        _log_step("confirm_switch")

        video_filename = self._resolve_video_filename(
            from_video_list=from_video_list,
            video_path=video_path,
        )
        _log_step("resolve_filename")

        video_filename = str(video_filename)
        if video_filename and is_large_tiff_path(video_filename):
            logger.info(
                "Redirecting TIFF stack open to the large-image viewer path: %s",
                video_filename,
            )
            if hasattr(self, "loadFile"):
                self.loadFile(video_filename)
            return
        if hasattr(self, "setLargeImageDocksActive"):
            self.setLargeImageDocksActive(False)
        self.stepSizeWidget.setEnabled(True)

        if video_filename:
            self._cleanup_audio_ui()
            _log_step("cleanup_audio_ui")
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
            _log_step("setup_paths_and_embedding")
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
            _log_step("create_video_loader")
            self._configure_project_schema_for_video(video_filename)
            self.fps = self.video_loader.get_fps()
            self.num_frames = self.video_loader.total_frames()
            _log_step("read_video_metadata")
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
            self._configure_audio_for_video(self.video_file, self.fps, eager=False)
            _log_step("ui_timeline_caption_audio")
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
            _log_step("setup_seekbar_controls")

            self.frame_loader.video_loader = self.video_loader
            self.frame_loader.moveToThread(self.frame_worker)
            self.frame_loader.res_frame.connect(self._on_frame_loaded)
            if not self.frame_worker.isRunning():
                self.frame_worker.start(priority=QtCore.QThread.IdlePriority)
            _log_step("connect_frame_loader")

            # Prioritize first-frame paint; defer expensive enrichment work so the
            # canvas becomes interactive immediately after open.
            setattr(self, "_defer_first_frame_enrichment", True)
            self.set_frame_number(self.frame_number)
            self._apply_timeline_caption_if_available(
                self.frame_number, only_if_empty=True
            )
            self._schedule_video_sidecar_preload(video_filename)
            _log_step("load_first_frame_and_defer_sidecars")

            self.actions.openNextImg.setEnabled(True)
            self.actions.openPrevImg.setEnabled(True)
            self._schedule_video_open_background_tasks(
                open_started_ts=start_ts,
                programmatic_call=programmatic_call,
                cur_video_folder=cur_video_folder,
                video_filename=video_filename,
            )
            foreground_elapsed = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Lifecycle open foreground ready in %.1fms. Remaining tasks running in background.",
                foreground_elapsed,
            )
        else:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Lifecycle open ended without selecting a video (%.1fms).", elapsed_ms
            )

    def _schedule_video_open_background_tasks(
        self,
        *,
        open_started_ts: float,
        programmatic_call: bool,
        cur_video_folder: Path,
        video_filename: str,
    ) -> None:
        """Run non-critical open tasks after first-frame readiness."""
        open_token = f"{video_filename}|{time.time_ns()}"
        setattr(self, "_video_open_background_token", open_token)

        task_queue: list[tuple[str, Callable[[], None]]] = []

        def _refresh_seed_marks() -> None:
            self._refresh_manual_seed_slider_marks(self.video_results_folder)

        def _refresh_missing_marks() -> None:
            self._refresh_missing_instance_slider_marks_from_tracking_stats(
                self.video_results_folder
            )

        def _prefetch_neighbor_labels() -> None:
            prefetch_fn = getattr(self, "_prefetch_label_for_frame", None)
            if not callable(prefetch_fn):
                return
            total_frames = int(getattr(self, "num_frames", 0) or 0)
            if total_frames <= 1:
                return
            current_frame = int(getattr(self, "frame_number", 0) or 0)
            fallback_path = (
                Path(self.filename) if getattr(self, "filename", None) else None
            )
            upper = min(total_frames, current_frame + 5)
            for frame_idx in range(current_frame + 1, upper):
                try:
                    prefetch_fn(int(frame_idx), fallback_path)
                except Exception:
                    logger.debug(
                        "Failed to prefetch annotation label for frame %s.",
                        frame_idx,
                        exc_info=True,
                    )

        def _load_tracking() -> None:
            self.load_tracking_results(cur_video_folder, video_filename)

        def _finalize_segments() -> None:
            if self.filename:
                self.open_segment_editor_action.setEnabled(True)
                self._load_segments_for_active_video()
                if self.caption_widget is not None:
                    self.caption_widget.set_video_segments(
                        self._current_video_defined_segments
                    )
                self._load_data_in_caption_widget(self.video_file)
                if not programmatic_call:
                    self._emit_live_frame_update()
                logger.info(
                    "Video '%s' loaded. Segment definition enabled.", self.filename
                )
            else:
                self.open_segment_editor_action.setEnabled(False)
                self._current_video_defined_segments = []
                if self.caption_widget is not None:
                    self.caption_widget.set_video_segments([])

        task_queue.append(("refresh_manual_seed_slider_marks", _refresh_seed_marks))
        task_queue.append(
            (
                "refresh_missing_instance_slider_marks",
                _refresh_missing_marks,
            )
        )
        task_queue.append(("prefetch_neighbor_annotations", _prefetch_neighbor_labels))
        task_queue.append(("load_tracking_results", _load_tracking))
        task_queue.append(("finalize_segments", _finalize_segments))

        def _drain(index: int = 0) -> None:
            active_video = str(getattr(self, "video_file", "") or "")
            active_token = str(getattr(self, "_video_open_background_token", "") or "")
            if active_video != str(video_filename) or active_token != open_token:
                return
            if index >= len(task_queue):
                total_ms = (time.perf_counter() - open_started_ts) * 1000.0
                logger.info("Lifecycle open completed in %.1fms.", total_ms)
                return
            task_name, task_fn = task_queue[index]
            task_start = time.perf_counter()
            try:
                task_fn()
            except Exception:
                logger.debug(
                    "Background open task '%s' failed for %s.",
                    task_name,
                    active_video,
                    exc_info=True,
                )
            task_ms = (time.perf_counter() - task_start) * 1000.0
            total_ms = (time.perf_counter() - open_started_ts) * 1000.0
            logger.info(
                "Lifecycle background task '%s' finished in %.1fms (total %.1fms).",
                task_name,
                task_ms,
                total_ms,
            )
            QtCore.QTimer.singleShot(0, lambda: _drain(index + 1))

        QtCore.QTimer.singleShot(0, _drain)

    def _schedule_video_sidecar_preload(self, video_filename: str) -> None:
        """Preload optional sidecar records after initial frame/UI are ready."""
        preload_token = str(video_filename or "")
        setattr(self, "_pending_video_sidecar_preload", preload_token)

        def _run_preload() -> None:
            active_video = str(getattr(self, "video_file", "") or "")
            pending_token = str(
                getattr(self, "_pending_video_sidecar_preload", "") or ""
            )
            if (
                not active_video
                or active_video != preload_token
                or pending_token != preload_token
            ):
                return
            start_ts = time.perf_counter()
            if getattr(self, "depth_manager", None) is not None:
                try:
                    self.depth_manager.load_depth_ndjson_records()
                except Exception:
                    logger.debug(
                        "Deferred depth sidecar preload failed for %s.",
                        active_video,
                        exc_info=True,
                    )
            if getattr(self, "optical_flow_manager", None) is not None:
                try:
                    self.optical_flow_manager.load_records(active_video)
                except Exception:
                    logger.debug(
                        "Deferred optical-flow sidecar preload failed for %s.",
                        active_video,
                        exc_info=True,
                    )
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            logger.info(
                "Deferred sidecar preload completed in %.1fms for %s.",
                elapsed_ms,
                active_video,
            )

        QtCore.QTimer.singleShot(0, _run_preload)

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
            self._remember_last_video_open_dir(video_path)
            return video_path
        start_dir = self._preferred_video_open_dir()
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
        self._remember_last_video_open_dir(video_filename)
        return video_filename

    def _preferred_video_open_dir(self) -> Path:
        """Return a stable directory for the open-video dialog.

        Prefer real video directories over frame-export folders that can contain
        hundreds of thousands of images and slow down dialog initialization.
        """
        current_video = str(getattr(self, "video_file", "") or "").strip()
        if current_video:
            return self._sanitize_video_dialog_dir(
                Path(current_video).expanduser().parent
            )

        remembered = str(getattr(self, "_last_video_open_dir", "") or "").strip()
        if remembered:
            return self._sanitize_video_dialog_dir(Path(remembered).expanduser())

        results_dir = getattr(self, "video_results_folder", None)
        if isinstance(results_dir, Path):
            return self._sanitize_video_dialog_dir(results_dir.parent)

        current_filename = str(getattr(self, "filename", "") or "").strip()
        if current_filename:
            path = Path(current_filename).expanduser()
            image_like_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            if path.suffix.lower() in image_like_suffixes and path.parent.parent:
                return self._sanitize_video_dialog_dir(path.parent.parent)
            return self._sanitize_video_dialog_dir(path.parent)

        return self._sanitize_video_dialog_dir(Path("."))

    def _sanitize_video_dialog_dir(self, candidate: Path) -> Path:
        """Return a responsive directory for QFileDialog initialization.

        Native file dialogs can stall for seconds when opening very large
        frame-export/result folders. For such directories, use the parent.
        """
        path = Path(candidate).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        if not resolved.exists() or not resolved.is_dir():
            return Path.home()

        if self._looks_like_heavy_frame_directory(resolved):
            parent = resolved.parent
            if parent.exists() and parent.is_dir():
                logger.info(
                    "Using parent directory for video picker to avoid heavy frame folder startup: %s -> %s",
                    resolved,
                    parent,
                )
                return parent
        return resolved

    @staticmethod
    def _looks_like_heavy_frame_directory(directory: Path) -> bool:
        """Heuristic: detect large frame/result folders that slow file dialogs."""
        image_suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }
        max_probe = 256
        min_total_for_heavy = 120
        min_images_for_heavy = 80
        image_count = 0
        entry_count = 0
        has_store_marker = False

        try:
            with os.scandir(directory) as it:
                for entry in it:
                    entry_count += 1
                    if entry.is_file():
                        name_lower = entry.name.lower()
                        suffix = Path(name_lower).suffix
                        if suffix in image_suffixes:
                            image_count += 1
                        if name_lower.endswith("_annotations.ndjson") or (
                            name_lower.endswith(".json")
                            and "_" in name_lower
                            and name_lower.rsplit("_", 1)[-1]
                            .replace(".json", "")
                            .isdigit()
                        ):
                            has_store_marker = True
                    if entry_count >= max_probe:
                        break
        except OSError:
            return False

        if entry_count < min_total_for_heavy:
            return bool(has_store_marker and image_count >= 20)
        return bool(image_count >= min_images_for_heavy or has_store_marker)

    def _remember_last_video_open_dir(self, video_filename) -> None:
        candidate = str(video_filename or "").strip()
        if not candidate:
            return
        try:
            path = Path(candidate).expanduser()
            parent = path.parent
            if str(parent):
                setattr(self, "_last_video_open_dir", str(parent))
        except Exception:
            pass

    @staticmethod
    def _create_video_loader(video_filename: str):
        suffix_lower = Path(video_filename).suffix.lower()
        if suffix_lower in {".tif", ".tiff"} or video_filename.lower().endswith(
            (".ome.tif", ".ome.tiff")
        ):
            return videos.TiffStackVideo(video_filename)
        return videos.CV2Video(video_filename)
