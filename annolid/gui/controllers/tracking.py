from pathlib import Path
from typing import Optional, TYPE_CHECKING

from qtpy import QtCore, QtWidgets

from annolid.jobs.tracking_worker import TrackingWorker
from annolid.gui.workers import TrackAllWorker
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class TrackingController(QtCore.QObject):
    """Encapsulate tracking worker lifecycle and UI signal wiring."""

    def __init__(self, window: "AnnolidWindow") -> None:
        super().__init__(window)
        self._window = window
        self._active_worker: Optional[TrackingWorker] = None
        self._previous_connected_worker: Optional[TrackingWorker] = None
        self._track_all_worker: Optional[TrackAllWorker] = None

    @property
    def active_worker(self) -> Optional[TrackingWorker]:
        return self._active_worker

    def is_tracking_busy(self) -> bool:
        worker = self._active_worker
        if worker and worker.isRunning():
            return True
        track_all = self._track_all_worker
        if track_all and track_all.isRunning():
            return True
        return False

    def is_track_all_running(self) -> bool:
        track_all = self._track_all_worker
        return bool(track_all and track_all.isRunning())

    @QtCore.Slot(TrackingWorker, Path)
    def start_tracking(
        self,
        worker_instance: TrackingWorker,
        video_path: Path,
    ) -> None:
        if self.is_tracking_busy():
            QtWidgets.QMessageBox.warning(
                self._window,
                "Tracking Busy",
                "Another tracking job is already active. Please wait.",
            )
            worker_instance.stop()
            worker_instance.wait(500)
            worker_instance.deleteLater()
            return

        logger.info(
            "AnnolidWindow: Tracking initiated by SegmentEditorDialog for %s",
            video_path.name,
        )
        self._active_worker = worker_instance
        self._window.set_tracking_ui_state(is_tracking=True)
        self._connect_signals_to_active_worker(worker_instance)

    # Backwards-compatible slot name used by older signal hookups.
    handle_tracking_initiated_by_dialog = start_tracking

    def stop_active_worker(self) -> None:
        worker = self._active_worker
        if worker and worker.isRunning():
            worker.stop()
        track_all = self._track_all_worker
        if track_all and track_all.isRunning():
            track_all.stop()

    # ------------------------------------------------------------------ #
    # Signal handlers
    # ------------------------------------------------------------------ #
    @QtCore.Slot(int, str)
    def _update_main_status_progress(self, percentage: int, message: str) -> None:
        self._window.statusBar().showMessage(f"{message} ({percentage}%)", 4000)

    @QtCore.Slot(str)
    def _on_tracking_job_finished(self, completion_message: str) -> None:
        QtWidgets.QMessageBox.information(
            self._window, "Tracking Job Complete", completion_message
        )
        self._window.statusBar().showMessage(completion_message, 5000)
        self._window.set_tracking_ui_state(is_tracking=False)

        worker_that_finished = self.sender()
        if worker_that_finished is None:
            logger.debug("Finished signal without sender context.")
            return

        if worker_that_finished == self._active_worker:
            self._disconnect_worker_signals(worker_that_finished)
            if worker_that_finished.parent() is None:
                worker_that_finished.deleteLater()
                logger.info("Scheduled dialog-created worker for deletion.")
            self._active_worker = None
        else:
            worker_that_finished.deleteLater()
            logger.info(
                "An external worker (%s) finished and was scheduled for deletion.",
                worker_that_finished.__class__.__name__,
            )

    def register_track_all_worker(self, worker_instance: TrackAllWorker) -> None:
        if worker_instance is None:
            logger.warning("Attempted to register a null TrackAllWorker.")
            return
        if self._active_worker and self._active_worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self._window,
                "Tracking Busy",
                "Single-video tracking is currently running. Please wait before starting Track All.",
            )
            worker_instance.stop()
            worker_instance.wait(500)
            worker_instance.deleteLater()
            return
        if self._track_all_worker and self._track_all_worker is not worker_instance:
            self._disconnect_track_all_worker(self._track_all_worker)

        self._track_all_worker = worker_instance
        worker_instance.progress.connect(self._update_main_status_progress)
        worker_instance.finished.connect(self._on_track_all_finished)
        worker_instance.error.connect(self._on_track_all_error)
        worker_instance.video_processing_started.connect(
            self._handle_track_all_video_started
        )
        worker_instance.video_processing_finished.connect(
            self._handle_track_all_video_finished
        )
        self._window.set_tracking_ui_state(is_tracking=True)
        logger.info("Registered TrackAllWorker with tracking controller.")

    @QtCore.Slot(str)
    def _on_tracking_job_error(self, error_message: str) -> None:
        QtWidgets.QMessageBox.critical(
            self._window, "Tracking Job Error", error_message
        )
        self._window.statusBar().showMessage(f"Error: {error_message}", 0)
        self._window.set_tracking_ui_state(is_tracking=False)

        worker_that_errored = self.sender()
        if worker_that_errored is None:
            return

        if worker_that_errored == self._active_worker:
            if worker_that_errored.parent() is None:
                worker_that_errored.deleteLater()
            self._active_worker = None
        else:
            worker_that_errored.deleteLater()

    @QtCore.Slot(str, str)
    def _handle_tracking_video_started_ui_update(
        self, video_path_str: str, output_folder_str: str
    ) -> None:
        window = self._window
        logger.info("AnnolidWindow UI: Job started for video %s", video_path_str)
        if window.filename != video_path_str:
            logger.info(
                "Worker started on %s, but canvas shows %s. Opening programmatically.",
                video_path_str,
                window.filename,
            )
            window.openVideo(
                from_video_list=True,
                video_path=video_path_str,
                programmatic_call=True,
            )

        if window.video_file == video_path_str:
            expected_results_folder = Path(output_folder_str)
            if window.video_results_folder != expected_results_folder:
                logger.warning(
                    "Mismatch in video_results_folder. Expected: %s, Have: %s. Forcing update.",
                    expected_results_folder,
                    window.video_results_folder,
                )
                window.video_results_folder = expected_results_folder

            window._setup_prediction_folder_watcher(str(output_folder_str))
        else:
            logger.error(
                "Critical: mismatch after attempting to open video for tracking. "
                "Current: %s, Expected by worker: %s.",
                window.video_file,
                video_path_str,
            )

    @QtCore.Slot(str)
    def _handle_tracking_video_finished_ui_update(self, video_path_str: str) -> None:
        window = self._window
        logger.info("AnnolidWindow UI: Job finished for video %s", video_path_str)
        current_watched_folder = ""
        watcher = window.prediction_progress_watcher
        if watcher and watcher.directories():
            current_watched_folder = watcher.directories()[0]

        if Path(video_path_str).with_suffix("") == Path(current_watched_folder):
            window._finalize_prediction_progress(
                f"GUI finalized for {Path(video_path_str).name}."
            )
        else:
            logger.info(
                "GUI: Video %s finished, but watcher was on %s or not active.",
                video_path_str,
                current_watched_folder,
            )

    @QtCore.Slot(str, str)
    def _handle_track_all_video_started(
        self, video_path: str, output_folder_path: str
    ) -> None:
        window = self._window
        logger.info("TrackAll: Starting processing for %s", video_path)
        window.closeFile(suppress_tracking_prompt=True)
        window.openVideo(
            from_video_list=True,
            video_path=video_path,
            programmatic_call=True,
        )
        if window.video_file == video_path and window.video_results_folder == Path(
            output_folder_path
        ):
            logger.info("TrackAll: Setting up watcher for %s", output_folder_path)
            window._setup_prediction_folder_watcher(output_folder_path)
            if hasattr(window, "progress_bar"):
                window._initialize_progress_bar(owner="prediction")
        else:
            logger.warning(
                "TrackAll: Video %s not properly loaded or output folder mismatch.",
                video_path,
            )
            logger.warning(
                "Current video_file: %s, expected: %s",
                window.video_file,
                video_path,
            )
            logger.warning(
                "Current video_results_folder: %s, expected: %s",
                window.video_results_folder,
                output_folder_path,
            )

    @QtCore.Slot(str)
    def _handle_track_all_video_finished(self, video_path: str) -> None:
        window = self._window
        logger.info("TrackAll: Finished processing for %s", video_path)
        current_video_name_in_watcher = ""
        watcher = getattr(window, "prediction_progress_watcher", None)
        watched_folder = getattr(window, "prediction_progress_folder", None)

        if watched_folder:
            current_video_name_in_watcher = Path(watched_folder).name
        elif watcher is not None and hasattr(watcher, "directories"):
            try:
                dirs = watcher.directories()
                if dirs:
                    current_video_name_in_watcher = Path(dirs[0]).name
            except Exception:
                current_video_name_in_watcher = ""

        if Path(video_path).stem == current_video_name_in_watcher:
            window._finalize_prediction_progress(
                f"Automated tracking for {Path(video_path).name} complete."
            )
        else:
            logger.info(
                "TrackAll: Video %s finished, but watcher was on %s or not active.",
                video_path,
                current_video_name_in_watcher,
            )

    @QtCore.Slot(str)
    def _on_track_all_finished(self, completion_message: str) -> None:
        self._window.statusBar().showMessage(completion_message, 5000)
        self._window.set_tracking_ui_state(is_tracking=False)
        self._window._finalize_prediction_progress("Track All run finished.")
        worker = self._track_all_worker
        if worker:
            self._disconnect_track_all_worker(worker)
            if worker.parent() is None:
                worker.deleteLater()
        self._track_all_worker = None

    @QtCore.Slot(str)
    def _on_track_all_error(self, error_message: str) -> None:
        QtWidgets.QMessageBox.critical(self._window, "Track All Error", error_message)
        self._window.statusBar().showMessage(f"Error: {error_message}", 0)
        worker = self._track_all_worker
        if worker:
            self._disconnect_track_all_worker(worker)
            if worker.parent() is None:
                worker.deleteLater()
        self._track_all_worker = None
        self._window.set_tracking_ui_state(is_tracking=False)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _connect_signals_to_active_worker(
        self, worker_instance: Optional[TrackingWorker]
    ) -> None:
        if not worker_instance:
            logger.warning("Attempted to connect signals to a null worker instance.")
            return

        previous_worker = self._previous_connected_worker
        if previous_worker and previous_worker is not worker_instance:
            self._disconnect_worker_signals(previous_worker)

        self._previous_connected_worker = worker_instance

        worker_instance.progress.connect(self._update_main_status_progress)
        worker_instance.finished.connect(self._on_tracking_job_finished)
        worker_instance.error.connect(self._on_tracking_job_error)

        if hasattr(worker_instance, "video_job_started"):
            worker_instance.video_job_started.connect(
                self._handle_tracking_video_started_ui_update
            )
        if hasattr(worker_instance, "video_job_finished"):
            worker_instance.video_job_finished.connect(
                self._handle_tracking_video_finished_ui_update
            )

        logger.info(
            "AnnolidWindow: Connected UI signals for worker: %s",
            worker_instance.__class__.__name__,
        )

    def _disconnect_worker_signals(self, worker: TrackingWorker) -> None:
        try:
            worker.progress.disconnect(self._update_main_status_progress)
        except (TypeError, RuntimeError):
            logger.debug("Progress disconnect failed (already disconnected).")

        try:
            worker.finished.disconnect(self._on_tracking_job_finished)
        except (TypeError, RuntimeError):
            logger.debug("Finished disconnect failed (already disconnected).")

        try:
            worker.error.disconnect(self._on_tracking_job_error)
        except (TypeError, RuntimeError):
            logger.debug("Error disconnect failed (already disconnected).")

        if hasattr(worker, "video_job_started"):
            try:
                worker.video_job_started.disconnect(
                    self._handle_tracking_video_started_ui_update
                )
            except (TypeError, RuntimeError):
                logger.debug("video_job_started disconnect failed.")

        if hasattr(worker, "video_job_finished"):
            try:
                worker.video_job_finished.disconnect(
                    self._handle_tracking_video_finished_ui_update
                )
            except (TypeError, RuntimeError):
                logger.debug("video_job_finished disconnect failed.")

    def _disconnect_track_all_worker(self, worker: TrackAllWorker) -> None:
        try:
            worker.progress.disconnect(self._update_main_status_progress)
        except (TypeError, RuntimeError):
            logger.debug("TrackAll progress disconnect failed.")
        try:
            worker.finished.disconnect(self._on_track_all_finished)
        except (TypeError, RuntimeError):
            logger.debug("TrackAll finished disconnect failed.")
        try:
            worker.error.disconnect(self._on_track_all_error)
        except (TypeError, RuntimeError):
            logger.debug("TrackAll error disconnect failed.")
        try:
            worker.video_processing_started.disconnect(
                self._handle_track_all_video_started
            )
        except (TypeError, RuntimeError):
            logger.debug("TrackAll start disconnect failed.")
        try:
            worker.video_processing_finished.disconnect(
                self._handle_track_all_video_finished
            )
        except (TypeError, RuntimeError):
            logger.debug("TrackAll finished disconnect failed.")
