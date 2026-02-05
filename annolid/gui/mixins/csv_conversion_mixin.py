from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.annotation import labelme2csv
from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger


class CsvConversionMixin:
    """Background CSV conversion workflow and worker lifecycle."""

    def convert_json_to_tracked_csv(
        self,
        *,
        include_tracked_output=True,
        force_rewrite_tracking_csv=False,
    ):
        out_folder = self.video_results_folder
        if not out_folder or not Path(out_folder).exists():
            QtWidgets.QMessageBox.warning(
                self,
                "No Predictions Found",
                "Help Annolid achieve precise predictions by labeling a frame. Your input is valuable!",
            )
            return

        csv_output_path = out_folder.parent / f"{out_folder.name}_tracking.csv"

        if (
            getattr(self, "csv_thread", None)
            and self.csv_thread
            and self.csv_thread.isRunning()
        ):
            job = (
                out_folder,
                csv_output_path,
                bool(include_tracked_output),
                bool(force_rewrite_tracking_csv),
            )
            if job not in self._csv_conversion_queue:
                self._csv_conversion_queue.append(job)
                self.statusBar().showMessage("Queued tracking CSV conversion...", 3000)
            return

        self._start_csv_conversion(
            out_folder,
            csv_output_path,
            include_tracked_output=bool(include_tracked_output),
            force_rewrite_tracking_csv=bool(force_rewrite_tracking_csv),
        )

    def _start_csv_conversion(
        self,
        out_folder: Path,
        csv_output_path: Path,
        *,
        include_tracked_output=True,
        force_rewrite_tracking_csv=False,
    ):
        """Kick off a background CSV conversion job for the given folder."""
        self._initialize_progress_bar(owner="csv_conversion")
        self._last_tracking_csv_path = str(csv_output_path)
        self.statusBar().showMessage(
            f"Generating tracking CSV: {csv_output_path.name}", 3000
        )

        try:
            tracked_csv_path = out_folder.parent / f"{out_folder.name}_tracked.csv"
            self.csv_worker = FlexibleWorker(
                task_function=labelme2csv.convert_json_to_csv,
                json_folder=str(out_folder),
                csv_file=str(csv_output_path),
                tracked_csv_file=(
                    str(tracked_csv_path) if include_tracked_output else None
                ),
                fps=self.fps,
                force_rewrite_tracking_csv=bool(force_rewrite_tracking_csv),
                progress_callback=self._csv_worker_progress_proxy,
            )
            self.csv_thread = QtCore.QThread()

            self.csv_worker.moveToThread(self.csv_thread)
            self._connect_worker_signals()

            self.csv_thread.start()
            QtCore.QTimer.singleShot(0, lambda: self.csv_worker.start_signal.emit())

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An unexpected error occurred: {str(e)}"
            )
            try:
                if hasattr(self, "_hide_progress_bar"):
                    self._hide_progress_bar()
            except Exception:
                pass
            self.csv_worker = None
            self.csv_thread = None
            self._last_tracking_csv_path = None

            if self._csv_conversion_queue:
                next_out, next_csv, next_include_tracked, next_force_rewrite = (
                    self._csv_conversion_queue.pop(0)
                )
                self._start_csv_conversion(
                    next_out,
                    next_csv,
                    include_tracked_output=next_include_tracked,
                    force_rewrite_tracking_csv=next_force_rewrite,
                )

    def _on_csv_conversion_finished(self, result=None):
        """Handle cleanup and user feedback after CSV conversion completes."""
        try:
            if hasattr(self, "_hide_progress_bar"):
                self._hide_progress_bar()
        except Exception:
            pass
        self._progress_bar_owner = None

        if result == "Stopped":
            self._cleanup_csv_worker()
            return

        if isinstance(result, str) and result.startswith("No annotation"):
            QtWidgets.QMessageBox.information(self, "Tracking CSV", result)
            self._last_tracking_csv_path = None
            self._cleanup_csv_worker()
            return

        if isinstance(result, Exception):
            QtWidgets.QMessageBox.critical(
                self,
                "Tracking CSV Error",
                f"Failed to generate tracking CSV:\n{result}",
            )
            self._cleanup_csv_worker()
            return

        csv_path = getattr(self, "_last_tracking_csv_path", None)
        if csv_path:
            path_obj = Path(csv_path)
            if path_obj.exists():
                QtWidgets.QMessageBox.information(
                    self, "Tracking Complete", f"Review the file at:\n{csv_path}"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Tracking CSV Missing",
                    f"Expected tracking file was not found:\n{csv_path}\n"
                    "Please try saving again.",
                )
            self._last_tracking_csv_path = None
        self._cleanup_csv_worker()

    def _connect_worker_signals(self):
        """Connect worker signals to their respective slots safely."""
        worker = self.csv_worker
        thread = self.csv_thread

        worker.start_signal.connect(worker.run)

        worker.finished_signal.connect(self._on_csv_conversion_finished)
        worker.finished_signal.connect(thread.quit)
        worker.finished_signal.connect(worker.deleteLater)
        thread.finished.connect(self._cleanup_csv_worker)
        thread.finished.connect(thread.deleteLater)

        worker.progress_signal.connect(self._update_progress_bar)
        self.seekbar.removeMarksByType("predicted")
        self.seekbar.removeMarksByType("predicted_existing")

    def _cleanup_csv_worker(self):
        """Clear references once the CSV conversion thread has fully finished."""
        try:
            if getattr(self, "csv_thread", None) and isinstance(
                self.csv_thread, QtCore.QThread
            ):
                if self.csv_thread.isRunning():
                    return
        except Exception:
            pass
        self.csv_thread = None
        self.csv_worker = None

        if self._csv_conversion_queue:
            next_out, next_csv, next_include_tracked, next_force_rewrite = (
                self._csv_conversion_queue.pop(0)
            )
            self._start_csv_conversion(
                next_out,
                next_csv,
                include_tracked_output=next_include_tracked,
                force_rewrite_tracking_csv=next_force_rewrite,
            )

    def _stop_csv_worker(self):
        """Request a graceful stop of any active CSV conversion."""
        if self._csv_conversion_queue:
            self._csv_conversion_queue.clear()

        worker = getattr(self, "csv_worker", None)
        if worker is not None:
            try:
                worker.request_stop()
            except RuntimeError:
                logger.debug("CSV worker already deleted.", exc_info=True)

        thread = getattr(self, "csv_thread", None)
        if thread is not None:
            try:
                thread.quit()
                thread.wait(2000)
            except RuntimeError:
                logger.debug("CSV thread already cleaned up.", exc_info=True)

        self._cleanup_csv_worker()
