from __future__ import annotations

import json
import os
import os.path as osp
import re
import time
from pathlib import Path

from qtpy import QtCore

from annolid.gui.widgets.video_slider import VideoSliderMark
from annolid.utils.annotation_store import AnnotationStore
from annolid.utils.logger import logger


class PredictionProgressMixin:
    """Prediction stop/progress/watcher helpers."""

    def _set_prediction_session_active(self, active: bool) -> None:
        self._prediction_session_active = bool(active)

    def _prediction_session_is_active(self) -> bool:
        return bool(getattr(self, "_prediction_session_active", False))

    def _clear_prediction_progress_mark(self) -> None:
        if self.seekbar:
            self.seekbar.removeMarksByType("prediction_progress")
        self._prediction_progress_mark = None

    @staticmethod
    def _frames_to_intervals(frame_nums: list[int] | set[int]) -> list[tuple[int, int]]:
        if not frame_nums:
            return []
        ordered = sorted({int(v) for v in frame_nums})
        intervals: list[tuple[int, int]] = []
        start = ordered[0]
        prev = ordered[0]
        for frame in ordered[1:]:
            if int(frame) == int(prev) + 1:
                prev = frame
                continue
            intervals.append((int(start), int(prev)))
            start = frame
            prev = frame
        intervals.append((int(start), int(prev)))
        return intervals

    def _render_prediction_sections(
        self,
        *,
        existing_frames: list[int] | set[int],
        predicted_frames: list[int] | set[int],
    ) -> None:
        if not self.seekbar or self.num_frames is None:
            return

        existing_intervals = self._frames_to_intervals(existing_frames)
        predicted_intervals = self._frames_to_intervals(predicted_frames)
        signature = (tuple(existing_intervals), tuple(predicted_intervals))
        if signature == getattr(self, "_prediction_marks_signature", None):
            return

        self.seekbar.blockSignals(True)
        self.seekbar.removeMarksByType("predicted")
        self.seekbar.removeMarksByType("predicted_existing")

        for start, end in existing_intervals:
            if start < 0 or start >= self.num_frames:
                continue
            clipped_end = min(int(end), int(self.num_frames) - 1)
            mark = VideoSliderMark(
                mark_type="predicted_existing",
                val=int(start),
                end_val=int(clipped_end) if clipped_end > int(start) else None,
            )
            self.seekbar.addMark(mark, update=False)

        for start, end in predicted_intervals:
            if start < 0 or start >= self.num_frames:
                continue
            clipped_end = min(int(end), int(self.num_frames) - 1)
            mark = VideoSliderMark(
                mark_type="predicted",
                val=int(start),
                end_val=int(clipped_end) if clipped_end > int(start) else None,
            )
            self.seekbar.addMark(mark, update=False)

        self.seekbar.blockSignals(False)
        self.seekbar.update()
        self._prediction_marks_signature = signature

    def _discover_manual_seed_frames(self, folder_path: Path) -> set[int]:
        """Return frame indices that have manual seed image+json pairs."""
        seeds: set[int] = set()
        if not folder_path.exists() or not folder_path.is_dir():
            return seeds

        stem = folder_path.name
        stem_prefix = f"{stem}_"
        stem_prefix_lower = stem_prefix.lower()
        for png_path in folder_path.glob("*.png"):
            name_lower = png_path.stem.lower()
            if not name_lower.startswith(stem_prefix_lower):
                continue
            suffix = name_lower[len(stem_prefix_lower) :]
            if len(suffix) != 9 or not suffix.isdigit():
                continue
            frame_idx = int(suffix)
            if png_path.with_suffix(".json").exists():
                seeds.add(frame_idx)
        return seeds

    def _refresh_manual_seed_slider_marks(self, folder_path: str | Path) -> None:
        """Synchronize `manual_seed` marks on the seekbar with current seed files."""
        if not self.seekbar:
            return
        if self.num_frames is None:
            return

        folder = Path(folder_path)
        seed_frames = self._discover_manual_seed_frames(folder)
        valid_seed_frames = {
            int(frame)
            for frame in seed_frames
            if 0 <= int(frame) < int(self.num_frames)
        }
        existing_seed_marks = self.seekbar.getMarks(mark_type="manual_seed")
        existing_seed_vals = {int(mark.val) for mark in existing_seed_marks}

        to_remove = [
            mark
            for mark in existing_seed_marks
            if int(mark.val) not in valid_seed_frames
        ]
        to_add = sorted(valid_seed_frames - existing_seed_vals)

        if not to_remove and not to_add:
            return

        self.seekbar.blockSignals(True)
        for mark in to_remove:
            self.seekbar.removeMark(mark)
        for frame in to_add:
            self.seekbar.addMark(VideoSliderMark(mark_type="manual_seed", val=frame))
        self.seekbar.blockSignals(False)
        self.seekbar.update()

    def stop_prediction(self):
        worker = getattr(self, "pred_worker", None)
        thread = getattr(self, "seg_pred_thread", None)

        self._prediction_stop_requested = True
        self.stop_prediction_flag = False
        try:
            self.stepSizeWidget.predict_button.setText("Stopping...")
            self.stepSizeWidget.predict_button.setStyleSheet(
                "background-color: orange; color: white;"
            )
            self.stepSizeWidget.predict_button.setEnabled(False)
        except Exception:
            pass

        if worker is None:
            self._finalize_prediction_progress("Stop requested (no worker).")
            return

        try:
            if hasattr(worker, "request_stop"):
                worker.request_stop()
            else:
                worker.stop_signal.emit()
        except Exception:
            logger.debug("Failed to signal prediction worker stop.", exc_info=True)

        try:
            if thread is not None and hasattr(thread, "requestInterruption"):
                thread.requestInterruption()
        except Exception:
            pass

        try:
            if thread is not None:
                thread.quit()
        except Exception:
            pass

        self._force_stop_thread_ref = thread
        QtCore.QTimer.singleShot(8000, self._force_stop_prediction_thread)
        logger.info("Prediction stop requested.")

    def _force_stop_prediction_thread(self):
        """Last-resort termination for stuck prediction threads."""
        thread = getattr(self, "seg_pred_thread", None)
        if getattr(self, "_force_stop_thread_ref", None) is not thread:
            return
        worker = getattr(self, "pred_worker", None)
        if thread is None or not isinstance(thread, QtCore.QThread):
            return
        if not thread.isRunning():
            return

        logger.warning(
            "Prediction thread did not stop in time; terminating as a last resort."
        )
        try:
            thread.terminate()
            thread.wait(2000)
        except Exception:
            logger.debug("Failed to terminate prediction thread.", exc_info=True)
        try:
            if worker is not None:
                worker.deleteLater()
        except Exception:
            pass
        try:
            thread.deleteLater()
        except Exception:
            pass
        self.pred_worker = None
        self.seg_pred_thread = None
        self._force_stop_thread_ref = None
        self._finalize_prediction_progress("Prediction force-stopped.")

    def _cleanup_prediction_worker(self):
        """Clear references once the prediction thread has fully finished."""
        try:
            thread = getattr(self, "seg_pred_thread", None)
            if isinstance(thread, QtCore.QThread) and thread.isRunning():
                return
        except Exception:
            pass
        self.pred_worker = None
        self.seg_pred_thread = None
        self._force_stop_thread_ref = None

    def _initialize_progress_bar(self, *, owner: str) -> None:
        """Initialize progress state without rendering a status-bar widget."""
        self._progress_bar_owner = owner
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _hide_progress_bar(self) -> None:
        if not hasattr(self, "progress_bar"):
            return
        try:
            if self.progress_bar.parent() is self.statusBar():
                self.statusBar().removeWidget(self.progress_bar)
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
        except Exception:
            logger.debug("Failed to hide progress bar.", exc_info=True)

    def _csv_worker_progress_proxy(self, progress):
        """Route worker progress updates through thread-safe signal emission."""
        worker = getattr(self, "csv_worker", None)
        if worker is not None:
            try:
                worker.report_progress(progress)
            except RuntimeError:
                logger.debug(
                    "CSV progress update skipped (worker deleted).",
                    exc_info=True,
                )

    def _update_progress_bar(self, progress):
        """Update the progress bar's value."""
        self.progress_bar.setValue(progress)

    def _sync_prediction_progress_frame(self, frame_number: int) -> None:
        """Move playback/frame state to progress frame regardless of timeline dock visibility."""
        if self.num_frames is None:
            return
        frame = int(frame_number)
        if frame < 0 or frame >= int(self.num_frames):
            return

        try:
            if self.frame_number != frame:
                self.set_frame_number(frame)
        except Exception:
            logger.debug("Failed to set frame number from prediction progress.")

        if self.seekbar is None:
            return
        try:
            blocker = QtCore.QSignalBlocker(self.seekbar)
            self.seekbar.setValue(frame)
            del blocker
        except Exception:
            try:
                self.seekbar.setValue(frame)
            except Exception:
                logger.debug("Failed to move seekbar from prediction progress.")

    def _finalize_prediction_progress(self, message=""):
        logger.info(f"Prediction finalization: {message}")
        prediction_folder = None
        try:
            prediction_folder = getattr(self, "video_results_folder", None)
        except Exception:
            prediction_folder = None
        if getattr(self, "_progress_bar_owner", None) in (
            None,
            "prediction",
        ) and hasattr(self, "progress_bar"):
            self._hide_progress_bar()
            self._progress_bar_owner = None
        self._stop_prediction_folder_watcher()
        if self.seekbar:
            self.seekbar.removeMarksByType("predicted")
            self.seekbar.removeMarksByType("predicted_existing")
            self._clear_prediction_progress_mark()
            self._prediction_marks_signature = None
            # Rebuild completed-frame marks from existing prediction outputs.
            if prediction_folder:
                try:
                    self._scan_prediction_folder(str(prediction_folder))
                except Exception:
                    logger.debug(
                        "Failed to rebuild completed prediction marks.",
                        exc_info=True,
                    )

        self.stepSizeWidget.predict_button.setText("Pred")
        self.stepSizeWidget.predict_button.setStyleSheet(
            "background-color: green; color: white;"
        )
        self.stepSizeWidget.predict_button.setEnabled(True)
        self.stop_prediction_flag = False

    def _setup_prediction_folder_watcher(
        self, folder_path_to_watch, *, start_frame: int | None = None
    ):
        if self.prediction_progress_watcher is None:
            self.prediction_progress_watcher = QtCore.QTimer(self)
            self.prediction_progress_watcher.timeout.connect(
                self._handle_prediction_folder_change
            )

        if osp.isdir(folder_path_to_watch):
            self._set_prediction_session_active(True)
            self._prediction_marks_signature = None
            self.prediction_progress_folder = folder_path_to_watch
            self.prediction_start_timestamp = time.time()
            if start_frame is None:
                start_frame = (
                    int(self.frame_number) if self.frame_number is not None else 0
                )
            self._prediction_start_frame = max(0, int(start_frame))
            self._prediction_existing_store_frames = set()
            self._prediction_existing_json_frames = set()
            path = Path(folder_path_to_watch)
            prefixed_pattern = re.compile(r"_(\d{9,})\.json$")
            bare_pattern = re.compile(r"^(\d{9,})\.json$")
            try:
                for f_name in os.listdir(path):
                    if not f_name.endswith(".json"):
                        continue
                    match = None
                    if path.name in f_name:
                        match = prefixed_pattern.search(f_name)
                    if match is None:
                        match = bare_pattern.match(f_name)
                    if match:
                        try:
                            frame_num = int(float(match.group(1)))
                            self._prediction_existing_json_frames.add(frame_num)
                        except (ValueError, IndexError):
                            continue
            except OSError as exc:
                logger.debug(
                    "Failed to read existing prediction JSONs in %s: %s",
                    path,
                    exc,
                )
            try:
                store = AnnotationStore.for_frame_path(
                    path / f"{path.name}_000000000.json"
                )
                if store.store_path.exists():
                    self._prediction_existing_store_frames = set(store.iter_frames())
                self._prediction_store_path = store.store_path
                try:
                    self._prediction_store_baseline_size = int(
                        store.store_path.stat().st_size
                    )
                except OSError:
                    self._prediction_store_baseline_size = 0
                self._prediction_appended_frames = set()
            except Exception:
                self._prediction_existing_store_frames = set()
                self._prediction_store_path = None
                self._prediction_store_baseline_size = 0
                self._prediction_appended_frames = set()
            self.prediction_progress_watcher.start(1000)
            logger.info(
                f"Prediction progress watcher started for: {folder_path_to_watch}"
            )
            self._scan_prediction_folder(folder_path_to_watch)
        else:
            logger.warning(f"Cannot watch non-existent folder: {folder_path_to_watch}")

    def _scan_prediction_store_updates(self) -> set[int]:
        """Return frame indices appended to the annotation store since watcher start."""
        existing: set[int] = set(
            int(frame)
            for frame in getattr(self, "_prediction_appended_frames", set()) or set()
        )
        store_path = getattr(self, "_prediction_store_path", None)
        if not store_path:
            return existing
        path = Path(store_path)
        if not path.exists():
            return existing

        baseline_size = int(getattr(self, "_prediction_store_baseline_size", 0) or 0)
        try:
            current_size = int(path.stat().st_size)
        except OSError:
            return existing
        if current_size <= baseline_size:
            return existing
        if baseline_size < 0 or baseline_size > current_size:
            baseline_size = 0

        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(baseline_size)
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    frame = payload.get("frame")
                    try:
                        existing.add(int(frame))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            return existing

        self._prediction_store_baseline_size = int(current_size)
        self._prediction_appended_frames = set(existing)
        return existing

    def _scan_prediction_folder(self, folder_path):
        if not self.seekbar:
            return
        if not self.video_loader or self.num_frames is None or self.num_frames == 0:
            self._clear_prediction_progress_mark()
            return
        self._refresh_manual_seed_slider_marks(folder_path)

        try:
            path = Path(folder_path)
            prefixed_pattern = re.compile(r"_(\d{9,})\.json$")
            bare_pattern = re.compile(r"^(\d{9,})\.json$")
            prediction_active = self._prediction_session_is_active()
            if not prediction_active:
                self._clear_prediction_progress_mark()

            json_frame_nums_set: set[int] = set()
            for f_name in os.listdir(path):
                if not f_name.endswith(".json"):
                    continue
                match = None
                if path.name in f_name:
                    match = prefixed_pattern.search(f_name)
                if match is None:
                    match = bare_pattern.match(f_name)
                if match is None:
                    continue
                try:
                    frame_num = int(float(match.group(1)))
                    json_frame_nums_set.add(frame_num)
                except (ValueError, IndexError):
                    continue

            store_frame_nums_set: set[int] = set()
            if prediction_active:
                store_frame_nums_set.update(
                    int(frame)
                    for frame in getattr(
                        self, "_prediction_existing_store_frames", set()
                    )
                )
                store_frame_nums_set.update(self._scan_prediction_store_updates())
            else:
                try:
                    store = AnnotationStore.for_frame_path(
                        path / f"{path.name}_000000000.json"
                    )
                    if store.store_path.exists():
                        for frame in store.iter_frames():
                            try:
                                store_frame_nums_set.add(int(frame))
                            except (TypeError, ValueError):
                                continue
                except Exception:
                    logger.debug(
                        "Failed to read annotation store frames for slider marks.",
                        exc_info=True,
                    )

            current_frame_set = set(json_frame_nums_set)
            current_frame_set.update(store_frame_nums_set)

            baseline_frame_set: set[int] = set()
            if prediction_active:
                baseline_frame_set.update(
                    int(frame)
                    for frame in getattr(
                        self, "_prediction_existing_json_frames", set()
                    )
                )
                baseline_frame_set.update(
                    int(frame)
                    for frame in getattr(
                        self, "_prediction_existing_store_frames", set()
                    )
                )

            # Keep completed sections visible while prediction is running:
            # - existing frames: already present before this run started
            # - predicted frames: newly produced during this run
            if prediction_active:
                existing_frame_nums = sorted(baseline_frame_set)
                all_frame_nums = sorted(current_frame_set - baseline_frame_set)
            else:
                existing_frame_nums = []
                all_frame_nums = sorted(current_frame_set)

            frames_to_mark = list(all_frame_nums)
            existing_frames_to_mark = list(existing_frame_nums)

            self._render_prediction_sections(
                existing_frames=existing_frames_to_mark,
                predicted_frames=frames_to_mark,
            )

            if not current_frame_set:
                if prediction_active and self._prediction_start_frame is not None:
                    start_frame = self._prediction_start_frame
                    if 0 <= start_frame < self.num_frames:
                        self._clear_prediction_progress_mark()
                        progress_mark = VideoSliderMark(
                            mark_type="prediction_progress", val=start_frame
                        )
                        self.seekbar.addMark(progress_mark)
                        self._prediction_progress_mark = progress_mark
                        if bool(getattr(self, "_follow_prediction_progress", False)):
                            self._sync_prediction_progress_frame(start_frame)
                        self._update_progress_bar(0)
                else:
                    self._clear_prediction_progress_mark()
                return

            latest_completed_frames = sorted(current_frame_set)
            progress_frame: int | None = None
            if prediction_active:
                run_completed_frames = sorted(all_frame_nums)
                if run_completed_frames:
                    progress_frame = int(run_completed_frames[-1])
                elif self._prediction_start_frame is not None:
                    progress_frame = int(self._prediction_start_frame)
            elif latest_completed_frames:
                progress_frame = int(latest_completed_frames[-1])

            if progress_frame is None:
                self._clear_prediction_progress_mark()
                return

            if prediction_active:
                self.last_known_predicted_frame = int(progress_frame)
            else:
                self.last_known_predicted_frame = max(
                    self.last_known_predicted_frame, int(progress_frame)
                )

            if self.num_frames > 0:
                progress = int(
                    (self.last_known_predicted_frame / self.num_frames) * 100
                )
                self._update_progress_bar(progress)

            if prediction_active and 0 <= int(progress_frame) < self.num_frames:
                self._clear_prediction_progress_mark()
                progress_mark = VideoSliderMark(
                    mark_type="prediction_progress", val=int(progress_frame)
                )
                self.seekbar.addMark(progress_mark)
                self._prediction_progress_mark = progress_mark
                if bool(getattr(self, "_follow_prediction_progress", False)):
                    self._sync_prediction_progress_frame(int(progress_frame))
            else:
                self._clear_prediction_progress_mark()

        except Exception as e:
            logger.error(
                f"Error scanning prediction folder for slider marks: {e}", exc_info=True
            )

    @QtCore.Slot()
    def _handle_prediction_folder_change(self):
        path = self.video_results_folder
        if path:
            if not self._prediction_session_is_active():
                self._clear_prediction_progress_mark()
                if (
                    self.prediction_progress_watcher
                    and self.prediction_progress_watcher.isActive()
                ):
                    self.prediction_progress_watcher.stop()
                return
            logger.debug(f"Scanning prediction folder: {path}.")
            self._scan_prediction_folder(str(path))

    def _stop_prediction_folder_watcher(self):
        self._set_prediction_session_active(False)
        if self.prediction_progress_watcher:
            self.prediction_progress_watcher.stop()
            logger.info("Prediction progress watcher stopped.")
        self.prediction_progress_folder = None
        self.last_known_predicted_frame = -1
        self.prediction_start_timestamp = 0.0
        self._prediction_start_frame = None
        self._prediction_existing_store_frames = set()
        self._prediction_existing_json_frames = set()
        self._prediction_store_path = None
        self._prediction_store_baseline_size = 0
        self._prediction_appended_frames = set()
        self._prediction_marks_signature = None
        self._clear_prediction_progress_mark()
