from __future__ import annotations

from pathlib import Path
import time
from typing import TYPE_CHECKING, Optional

import pandas as pd
from qtpy import QtCore

from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class _TrackingSidecarWorker(QtCore.QObject):
    """Load behavior/labels sidecars off the UI thread."""

    finished = QtCore.Signal(object, str)
    failed = QtCore.Signal(str, str)

    def __init__(
        self,
        *,
        request_token: str,
        behavior_candidates: list[Path],
        labels_file_path: Optional[Path],
    ) -> None:
        super().__init__()
        self._request_token = str(request_token)
        self._behavior_candidates = list(behavior_candidates or [])
        self._labels_file_path = labels_file_path

    @QtCore.Slot()
    def run(self) -> None:
        try:
            payload = TrackingDataController._build_sidecar_payload(
                behavior_candidates=self._behavior_candidates,
                labels_file_path=self._labels_file_path,
            )
        except Exception as exc:
            self.failed.emit(str(exc), self._request_token)
            return
        self.finished.emit(payload, self._request_token)


class _TrackingSidecarCallbacks(QtCore.QObject):
    """Ensure sidecar worker callbacks run on the GUI thread."""

    def __init__(self, controller: "TrackingDataController", parent=None) -> None:
        super().__init__(parent)
        self._controller = controller

    @QtCore.Slot(object, str)
    def on_finished(self, payload: object, request_token: str) -> None:
        payload_dict = payload if isinstance(payload, dict) else {}
        self._controller._handle_sidecar_worker_finished(
            payload=payload_dict,
            request_token=str(request_token),
        )

    @QtCore.Slot(str, str)
    def on_failed(self, error_text: str, request_token: str) -> None:
        self._controller._handle_sidecar_worker_failed(
            error_text=str(error_text),
            request_token=str(request_token),
        )


class TrackingDataController:
    """Handle loading of tracking/behavior CSV data for the active video."""

    def __init__(self, window: "AnnolidWindow") -> None:
        self._window = window
        self._tracking_df: pd.DataFrame | None = None
        self._tracking_frame_slices: dict[int, tuple[int, int]] | None = None
        self._tracking_frame_indices: dict[int, tuple[int, ...]] | None = None
        self._tracking_csv_path: Path | None = None
        self._sidecar_thread: Optional[QtCore.QThread] = None
        self._sidecar_worker: Optional[_TrackingSidecarWorker] = None
        self._sidecar_request_token: str = ""
        self._sidecar_video_name: str = ""
        self._sidecar_behavior_candidates: list[Path] = []
        callback_parent = window if isinstance(window, QtCore.QObject) else None
        self._sidecar_callbacks = _TrackingSidecarCallbacks(
            self,
            parent=callback_parent,
        )

    @property
    def tracking_dataframe(self) -> pd.DataFrame | None:
        return self._tracking_df

    def tracking_rows_for_frame(self, frame_number: int) -> list[dict]:
        """Return tracking rows for a frame using the precomputed frame cache."""
        self._ensure_tracking_loaded_for_lookup()
        df = self._tracking_df
        if df is None or df.empty:
            return []

        try:
            frame_key = int(frame_number)
        except (TypeError, ValueError):
            return []

        frame_slices = self._tracking_frame_slices or {}
        frame_slice = frame_slices.get(frame_key)
        if frame_slice is not None:
            start, end = frame_slice
            try:
                return df.iloc[start:end].to_dict(orient="records")
            except Exception:
                # Fall back to the indexed path below if the CSV was reloaded or
                # the slice cache is stale.
                pass

        frame_indices = self._tracking_frame_indices or {}
        indices = frame_indices.get(frame_key)
        if not indices:
            return []

        try:
            frame_df = df.iloc[list(indices)]
        except Exception:
            # Best-effort fallback to the original DataFrame if the cached index
            # became stale or the CSV was reloaded unexpectedly.
            try:
                frame_df = df[df.frame_number == frame_key]
            except Exception:
                return []

        return frame_df.to_dict(orient="records")

    def _clear_tracking_cache(self) -> None:
        self._tracking_df = None
        self._tracking_frame_slices = None
        self._tracking_frame_indices = None
        self._tracking_csv_path = None

    def _cancel_sidecar_worker(self) -> None:
        worker = self._sidecar_worker
        thread = self._sidecar_thread
        self._sidecar_worker = None
        self._sidecar_thread = None
        self._sidecar_request_token = ""
        self._sidecar_video_name = ""
        self._sidecar_behavior_candidates = []
        if thread is not None:
            try:
                thread.quit()
                thread.wait(200)
            except Exception:
                pass
        if worker is not None:
            try:
                worker.deleteLater()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Stop in-flight sidecar loading and clear transient tracking state."""
        self._cancel_sidecar_worker()
        self._clear_tracking_cache()

    def _ensure_tracking_loaded_for_lookup(self) -> None:
        """Lazy-load tracking CSV only if frame lookup explicitly needs it."""
        csv_path = self._tracking_csv_path
        if self._tracking_df is not None or csv_path is None:
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Error loading tracking file %s: %s", csv_path, exc)
            self._tracking_csv_path = None
            return
        self._apply_tracking_dataframe(df)

    def _apply_tracking_dataframe(self, df: pd.DataFrame) -> None:
        if "frame_number" not in df.columns and "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "frame_number"}, inplace=True)
        if "frame_number" not in df.columns:
            logger.warning("Tracking CSV is missing required 'frame_number' column.")
            self._tracking_df = None
            self._tracking_frame_slices = None
            self._tracking_frame_indices = None
            return
        self._tracking_df = df
        self._tracking_frame_slices = self._build_tracking_frame_slices(df)
        self._tracking_frame_indices = self._build_tracking_frame_index(df)
        self._window._df = df

    @staticmethod
    def _is_likely_behavior_csv(candidate: Path, video_name: str) -> bool:
        name_lower = candidate.name.lower()
        video_name_lower = str(video_name).lower()

        excluded_suffixes = (
            f"{video_name_lower}_tracking.csv",
            f"{video_name_lower}_tracked.csv",
            f"{video_name_lower}_labels.csv",
            f"{video_name_lower}_gaps_report.csv",
            f"{video_name_lower}_tracking_gaps_report.csv",
        )
        if any(name_lower == suffix for suffix in excluded_suffixes):
            return False

        if "tracking" in name_lower and "timestamp" not in name_lower:
            return False
        if "gaps_report" in name_lower:
            return False

        behavior_keywords = ("timestamp", "behavior", "event", "bout")
        if any(keyword in name_lower for keyword in behavior_keywords):
            return True

        # Conservative fallback for small sidecar CSVs.
        try:
            return candidate.stat().st_size <= 2 * 1024 * 1024
        except Exception:
            return False

    def _reset_tracking_load_state(self) -> None:
        w = self._window
        w.behavior_controller.clear()
        w.behavior_log_widget.clear()
        w.pinned_flags = {}
        w._df = None
        self._clear_tracking_cache()

    def _discover_behavior_files(
        self, *, search_root: Path, video_name: str
    ) -> list[Path]:
        candidates: list[Path] = []
        video_name_lower = video_name.lower()
        for candidate in search_root.glob(f"{video_name}*.csv"):
            name_lower = candidate.name.lower()
            if name_lower == f"{video_name_lower}_tracking.csv":
                continue
            if name_lower == f"{video_name_lower}_labels.csv":
                continue
            if not self._is_likely_behavior_csv(candidate, video_name):
                continue
            candidates.append(candidate)
        return sorted(candidates)

    def _build_tracking_context(
        self, cur_video_folder: Path, video_filename: str
    ) -> tuple[str, list[Path], Path | None]:
        self._reset_tracking_load_state()
        video_name = Path(video_filename).stem
        main_tracking_file = cur_video_folder / f"{video_name}_tracking.csv"
        timestamps_file = cur_video_folder / f"{video_name}_timestamps.csv"
        labels_file_path = cur_video_folder / f"{video_name}_labels.csv"

        if main_tracking_file.is_file():
            self._tracking_csv_path = main_tracking_file
            logger.info(
                "Deferring tracking CSV load until first frame lookup: %s",
                main_tracking_file,
            )

        behavior_candidates: list[Path] = []
        if timestamps_file.is_file():
            behavior_candidates.append(timestamps_file)
        else:
            behavior_candidates.extend(
                self._discover_behavior_files(
                    search_root=cur_video_folder, video_name=video_name
                )
            )
            results_dir = getattr(self._window, "video_results_folder", None)
            if (
                isinstance(results_dir, Path)
                and results_dir.exists()
                and results_dir != cur_video_folder
            ):
                behavior_candidates.extend(
                    self._discover_behavior_files(
                        search_root=results_dir,
                        video_name=video_name,
                    )
                )

        seen_paths: set[Path] = set()
        deduped_behavior_candidates: list[Path] = []
        for candidate in behavior_candidates:
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            deduped_behavior_candidates.append(candidate)

        return (
            video_name,
            deduped_behavior_candidates,
            labels_file_path if labels_file_path.is_file() else None,
        )

    @staticmethod
    def _extract_behavior_rows_from_dataframe(
        df_behaviors: pd.DataFrame,
    ) -> list[tuple[Optional[float], float, Optional[str], str, str]]:
        rows: list[tuple[Optional[float], float, Optional[str], str, str]] = []
        for payload in df_behaviors.to_dict(orient="records"):
            raw_timestamp = payload.get("Recording time")
            event_label = str(payload.get("Event"))
            behavior = str(payload.get("Behavior"))
            raw_subject = payload.get("Subject")
            raw_trial_time = payload.get("Trial time")

            try:
                timestamp_value = float(raw_timestamp)
            except (TypeError, ValueError):
                continue

            trial_time_value: Optional[float]
            try:
                trial_time_value = (
                    float(raw_trial_time)
                    if raw_trial_time is not None and pd.notna(raw_trial_time)
                    else None
                )
            except (TypeError, ValueError):
                trial_time_value = None

            subject_value = None
            if raw_subject is not None and pd.notna(raw_subject):
                subject_value = str(raw_subject)

            rows.append(
                (
                    trial_time_value,
                    timestamp_value,
                    subject_value,
                    behavior,
                    event_label,
                )
            )
        return rows

    @classmethod
    def _build_sidecar_payload(
        cls, *, behavior_candidates: list[Path], labels_file_path: Optional[Path]
    ) -> dict:
        payload: dict = {
            "loaded_behavior": False,
            "behavior_source": None,
            "behavior_rows": None,
            "fallback_behavior_path": None,
            "labels_df": None,
        }
        required_columns = {"Recording time", "Event", "Behavior"}
        for candidate in list(behavior_candidates or []):
            try:
                if candidate.stat().st_size > 8 * 1024 * 1024:
                    logger.info(
                        "Skipping large CSV during behavior discovery: %s", candidate
                    )
                    continue
            except Exception:
                pass

            try:
                logger.info("Loading behavior data from: %s", candidate)
                df_behaviors = pd.read_csv(candidate)
            except Exception as exc:
                logger.error("Failed to read behavior data from %s: %s", candidate, exc)
                continue

            if required_columns.issubset(df_behaviors.columns):
                payload["behavior_rows"] = cls._extract_behavior_rows_from_dataframe(
                    df_behaviors
                )
                payload["behavior_source"] = str(candidate)
                payload["loaded_behavior"] = True
                break

            # Preserve compatibility for non-standard behavior CSVs (e.g. DLC).
            payload["fallback_behavior_path"] = str(candidate)
            break

        if labels_file_path is not None and labels_file_path.is_file():
            try:
                labels_df = pd.read_csv(labels_file_path)
                if (
                    "frame_number" not in labels_df.columns
                    and "Unnamed: 0" in labels_df.columns
                ):
                    labels_df.rename(
                        columns={"Unnamed: 0": "frame_number"}, inplace=True
                    )
                payload["labels_df"] = labels_df
            except Exception as exc:
                logger.error(
                    "Failed to load labels data from %s: %s", labels_file_path, exc
                )

        return payload

    def _build_tracking_frame_slices(
        self, df: pd.DataFrame
    ) -> dict[int, tuple[int, int]]:
        if "frame_number" not in df.columns or df.empty:
            return {}

        try:
            frame_series = df["frame_number"]
            if not frame_series.is_monotonic_increasing:
                return {}
            frame_values = frame_series.to_numpy()
        except Exception:
            return {}

        if len(frame_values) == 0:
            return {}

        change_points = [0]
        try:
            change_points.extend(
                int(idx) + 1
                for idx in (frame_values[1:] != frame_values[:-1]).nonzero()[0]
            )
        except Exception:
            return {}
        change_points.append(len(frame_values))

        frame_slices: dict[int, tuple[int, int]] = {}
        for start, end in zip(change_points[:-1], change_points[1:]):
            try:
                frame_key = int(frame_values[start])
            except (TypeError, ValueError, IndexError):
                continue
            frame_slices[frame_key] = (int(start), int(end))
        return frame_slices

    def _build_tracking_frame_index(
        self, df: pd.DataFrame
    ) -> dict[int, tuple[int, ...]]:
        if "frame_number" not in df.columns or df.empty:
            return {}

        try:
            grouped = df.groupby("frame_number", sort=False).indices
        except Exception:
            return {}

        frame_indices: dict[int, tuple[int, ...]] = {}
        for frame_number, indices in grouped.items():
            try:
                frame_key = int(frame_number)
            except (TypeError, ValueError):
                continue
            try:
                frame_indices[frame_key] = tuple(int(idx) for idx in indices)
            except Exception:
                continue
        return frame_indices

    def load_tracking_results(
        self, cur_video_folder: Path, video_filename: str
    ) -> None:
        self._cancel_sidecar_worker()
        video_name, behavior_candidates, labels_file_path = (
            self._build_tracking_context(cur_video_folder, video_filename)
        )
        payload = self._build_sidecar_payload(
            behavior_candidates=behavior_candidates,
            labels_file_path=labels_file_path,
        )
        self._apply_sidecar_payload(
            payload=payload,
            video_name=video_name,
            behavior_candidates=behavior_candidates,
        )

    def load_tracking_results_async(
        self, cur_video_folder: Path, video_filename: str
    ) -> None:
        self._cancel_sidecar_worker()
        video_name, behavior_candidates, labels_file_path = (
            self._build_tracking_context(cur_video_folder, video_filename)
        )
        request_token = (
            f"{Path(video_filename).resolve()}|{time.time_ns()}"
            if video_filename
            else str(time.time_ns())
        )
        self._sidecar_request_token = request_token
        self._sidecar_video_name = video_name
        self._sidecar_behavior_candidates = list(behavior_candidates)

        # If no sidecars exist, avoid spinning a thread and complete immediately.
        if not behavior_candidates and labels_file_path is None:
            self._apply_sidecar_payload(
                payload={},
                video_name=video_name,
                behavior_candidates=behavior_candidates,
            )
            return

        thread_parent = (
            self._window if isinstance(self._window, QtCore.QObject) else None
        )
        thread = QtCore.QThread(thread_parent)
        worker = _TrackingSidecarWorker(
            request_token=request_token,
            behavior_candidates=behavior_candidates,
            labels_file_path=labels_file_path,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        worker.finished.connect(
            self._sidecar_callbacks.on_finished,
            QtCore.Qt.QueuedConnection,
        )
        worker.failed.connect(
            self._sidecar_callbacks.on_failed,
            QtCore.Qt.QueuedConnection,
        )
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._sidecar_worker = worker
        self._sidecar_thread = thread
        thread.start(priority=QtCore.QThread.LowPriority)

    def _cleanup_sidecar_worker_handles(self) -> None:
        self._sidecar_worker = None
        self._sidecar_thread = None

    def _handle_sidecar_worker_finished(
        self,
        *,
        payload: dict,
        request_token: str,
    ) -> None:
        if str(request_token) != str(self._sidecar_request_token):
            return
        self._cleanup_sidecar_worker_handles()
        self._sidecar_request_token = ""
        video_name = str(self._sidecar_video_name or "")
        behavior_candidates = list(self._sidecar_behavior_candidates or [])
        self._sidecar_video_name = ""
        self._sidecar_behavior_candidates = []
        self._apply_sidecar_payload(
            payload=payload,
            video_name=video_name,
            behavior_candidates=behavior_candidates,
        )

    def _handle_sidecar_worker_failed(
        self,
        *,
        error_text: str,
        request_token: str,
    ) -> None:
        if str(request_token) != str(self._sidecar_request_token):
            return
        logger.error("Background tracking sidecar load failed: %s", error_text)
        self._cleanup_sidecar_worker_handles()
        self._sidecar_request_token = ""
        video_name = str(self._sidecar_video_name or "")
        behavior_candidates = list(self._sidecar_behavior_candidates or [])
        self._sidecar_video_name = ""
        self._sidecar_behavior_candidates = []
        self._apply_sidecar_payload(
            payload={},
            video_name=video_name,
            behavior_candidates=behavior_candidates,
        )

    def _apply_behavior_rows(
        self,
        rows: list[tuple[Optional[float], float, Optional[str], str, str]],
    ) -> None:
        w = self._window
        fps = w.fps if w.fps and w.fps > 0 else 29.97

        def time_to_frame(time_value: float) -> int:
            return int(round(float(time_value) * float(fps)))

        w.behavior_controller.load_events_from_rows(rows, time_to_frame=time_to_frame)
        w.behavior_controller.attach_slider(w.seekbar)
        fps_for_log = w.fps if w.fps and w.fps > 0 else 29.97
        w.behavior_log_widget.set_events(
            list(w.behavior_controller.iter_events()),
            fps=fps_for_log,
        )
        current_flags = dict(w.pinned_flags or {})
        for behavior in w.behavior_controller.behavior_names:
            current_flags[str(behavior)] = False
        w.loadFlags(current_flags)

    def _fallback_load_behavior_from_store(self) -> bool:
        w = self._window
        try:
            w.behavior_controller.load_events_from_store()
        except Exception as exc:
            logger.debug(
                "Failed to load behavior events from annotation store: %s", exc
            )
            return False

        if not w.behavior_controller.events_count:
            return False
        w.behavior_controller.attach_slider(w.seekbar)
        fps_for_log = w.fps if w.fps and w.fps > 0 else 29.97
        w.behavior_log_widget.set_events(
            list(w.behavior_controller.iter_events()),
            fps=fps_for_log,
        )
        current_flags = dict(w.pinned_flags or {})
        for behavior in w.behavior_controller.behavior_names:
            current_flags[str(behavior)] = False
        w.loadFlags(current_flags)
        return True

    def _apply_sidecar_payload(
        self,
        *,
        payload: dict,
        video_name: str,
        behavior_candidates: list[Path],
    ) -> None:
        w = self._window
        loaded_behavior = False
        behavior_rows = payload.get("behavior_rows")
        behavior_source = payload.get("behavior_source")
        fallback_behavior_path = payload.get("fallback_behavior_path")

        if isinstance(behavior_rows, list):
            if behavior_source:
                logger.info("Loaded behavior rows from: %s", behavior_source)
            self._apply_behavior_rows(behavior_rows)
            loaded_behavior = True
        elif fallback_behavior_path:
            try:
                logger.info("Loading behavior data from: %s", fallback_behavior_path)
                w._load_behavior(str(fallback_behavior_path))
                loaded_behavior = True
            except Exception as exc:
                logger.error(
                    "Failed to load behavior data from %s: %s",
                    fallback_behavior_path,
                    exc,
                )

        if not loaded_behavior and behavior_candidates:
            logger.warning(
                "Behavior CSV files were discovered for '%s' but could not be loaded.",
                video_name,
            )
        elif not behavior_candidates:
            logger.debug(
                "No behavior CSV detected for '%s'. Looking for '%s' or similarly named files.",
                video_name,
                f"{video_name}_timestamps.csv",
            )

        if not loaded_behavior:
            self._fallback_load_behavior_from_store()

        labels_df = payload.get("labels_df")
        if isinstance(labels_df, pd.DataFrame):
            w._df = labels_df
