from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from qtpy import QtCore, QtWidgets

from annolid.gui.qt_threads import configure_worker_thread_stack
from annolid.gui.widgets.optical_flow_dialog import FlowOptionsDialog
from annolid.gui.workers import FlexibleWorker
from annolid.motion.flow_runner import (
    decode_flow_component,
    decode_flow_record,
    process_video_flow,
    render_flow_overlay,
)
from annolid.motion.optical_flow import (
    OpticalFlowPreferences,
    resolve_optical_flow_backend_flags,
)
from annolid.utils.logger import logger

FlowPreferences = OpticalFlowPreferences
_MAX_DECODED_FLOW_CACHE = 16


@dataclass
class FlowRunSettings(OpticalFlowPreferences):
    video_path: str = ""
    ndjson_path: str = ""


class OpticalFlowTool(QtCore.QObject):
    """Encapsulates optical-flow UI, running, and indexed NDJSON overlay playback."""

    def __init__(self, window: QtWidgets.QWidget, manager=None) -> None:
        super().__init__(window)
        self._window = window
        self._manager = manager
        self._worker: Optional[FlexibleWorker] = None
        self._worker_thread: Optional[QtCore.QThread] = None
        self._active_ndjson_path: Optional[Path] = None
        self._live_running: bool = False
        self._stop_requested: bool = False
        # Optional in-memory records override + lightweight byte-offset index for disk files.
        self._records: Dict[int, Dict[str, object]] = {}
        self._record_offsets: Dict[int, int] = {}
        self._indexed_ndjson_path: Optional[Path] = None
        self._decoded_flow_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._global_mag_max: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if self._is_processing():
            self._show_already_running_message()
            return

        settings = self._build_run_settings()
        if not settings:
            return
        self._start_worker(settings)

    def stop(self, timeout_ms: int = 3000) -> None:
        """Cooperatively stop an active background optical-flow worker."""
        worker = self._worker
        worker_thread = self._worker_thread
        if worker is not None:
            self._stop_requested = True
            try:
                worker.request_stop()
            except Exception:
                logger.debug("Failed to request optical-flow stop.", exc_info=True)
        if worker_thread is not None and worker_thread.isRunning():
            worker_thread.quit()
            if QtCore.QThread.currentThread() is not worker_thread:
                worker_thread.wait(int(timeout_ms))

    def is_live_running(self) -> bool:
        return bool(self._live_running)

    def configure(self) -> None:
        """Open a dialog to set optical-flow defaults without starting a run."""
        prefs = self._current_preferences()
        dialog = FlowOptionsDialog(
            self._window,
            default_backend=prefs.backend,
            default_raft_model=prefs.raft_model,
            default_viz=prefs.visualization,
            default_opacity=prefs.opacity,
            default_quiver_step=prefs.quiver_step,
            default_quiver_gain=prefs.quiver_gain,
            default_stable_hsv=prefs.stable_hsv,
            default_pyr_scale=prefs.farneback_pyr_scale,
            default_levels=prefs.farneback_levels,
            default_winsize=prefs.farneback_winsize,
            default_iterations=prefs.farneback_iterations,
            default_poly_n=prefs.farneback_poly_n,
            default_poly_sigma=prefs.farneback_poly_sigma,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        updated = dialog.preferences()
        if updated is None:
            return
        self._apply_preferences_to_window(updated)
        self._persist_preferences(updated)
        self.refresh_current_frame_overlay()
        try:
            self._window.statusBar().showMessage(
                self._window.tr("Optical flow settings updated."), 2500
            )
        except Exception:
            pass

    def refresh_current_frame_overlay(self) -> None:
        """Re-render the active frame's flow overlay after settings change."""
        if self._live_running:
            return
        frame_num = getattr(self._window, "frame_number", None)
        if frame_num is None:
            return
        try:
            frame_idx = int(frame_num)
        except (TypeError, ValueError):
            return
        if self.has_overlay_for_frame(frame_idx):
            self.update_overlay_for_frame(frame_idx)

    def clear(self) -> None:
        if self._is_processing():
            self.stop()
        self._records = {}
        self._record_offsets = {}
        self._indexed_ndjson_path = None
        self._decoded_flow_cache.clear()
        self._global_mag_max = None
        self._live_running = False
        self._stop_requested = False
        w = self._window
        try:
            if getattr(w, "canvas", None):
                w.canvas.setFlowPreviewOverlay(None)
        except Exception:
            pass

    def load_records(self, video_file: Optional[str] = None) -> None:
        path = self._flow_ndjson_path(video_file)
        if not path:
            self._records = {}
            self._record_offsets = {}
            self._indexed_ndjson_path = None
            self._decoded_flow_cache.clear()
            self._global_mag_max = None
            return
        self._load_records_from_path(path)

    def _load_records_from_path(self, path: Path) -> None:
        """Index NDJSON frame byte offsets without holding base64 payloads in RAM."""
        offsets: Dict[int, int] = {}
        global_mag_max = 0.0
        bad_lines = 0
        self._records = {}
        self._decoded_flow_cache.clear()
        if not path.exists():
            self._record_offsets = offsets
            self._indexed_ndjson_path = None
            self._global_mag_max = None
            return
        try:
            with path.open("rb") as fh:
                while True:
                    offset = fh.tell()
                    raw_line = fh.readline()
                    if not raw_line:
                        break
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped.decode("utf-8"))
                    except Exception:
                        bad_lines += 1
                        continue
                    if not isinstance(data, dict):
                        bad_lines += 1
                        continue
                    try:
                        frame_index = int(data.get("frame_index", -1))
                    except (TypeError, ValueError):
                        bad_lines += 1
                        continue
                    if frame_index >= 0:
                        offsets[frame_index] = offset
                        other = data.get("otherData") or {}
                        if isinstance(other, dict):
                            mag_info = (
                                other.get("flow_mag_raw")
                                or other.get("flow_magnitude")
                                or {}
                            )
                            scale = (
                                mag_info.get("scale")
                                if isinstance(mag_info, dict)
                                else None
                            )
                            if isinstance(scale, dict):
                                try:
                                    mag_max = float(scale.get("max", 0.0))
                                    if (
                                        np.isfinite(mag_max)
                                        and mag_max > global_mag_max
                                    ):
                                        global_mag_max = mag_max
                                except (TypeError, ValueError):
                                    pass
        except OSError:
            logger.warning(
                "Failed to index optical-flow NDJSON at %s.",
                path,
                exc_info=True,
            )
            offsets = {}
            global_mag_max = 0.0
            path = None

        if bad_lines > 0:
            logger.warning(
                "Skipped %d malformed record(s) while indexing optical-flow NDJSON %s.",
                bad_lines,
                path,
            )
        self._record_offsets = offsets
        self._indexed_ndjson_path = path if offsets else None
        self._global_mag_max = global_mag_max if global_mag_max > 0 else None

    def _read_indexed_record(self, frame_number: int) -> Optional[Dict[str, object]]:
        if frame_number in self._records:
            return self._records[frame_number]
        offset = self._record_offsets.get(frame_number)
        path = self._indexed_ndjson_path
        if offset is None or path is None or not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                fh.seek(offset)
                raw_line = fh.readline()
            data = json.loads(raw_line.decode("utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            logger.debug(
                "Failed to read optical-flow record for frame %s from %s.",
                frame_number,
                path,
                exc_info=True,
            )
            return None

    def _decoded_flow_for_frame(self, frame_number: int) -> Optional[np.ndarray]:
        cached = self._decoded_flow_cache.get(frame_number)
        if cached is not None:
            self._decoded_flow_cache.move_to_end(frame_number)
            return cached
        record = self._read_indexed_record(frame_number)
        if record is None:
            return None
        flow = decode_flow_record(record)
        if flow is None:
            return None
        self._decoded_flow_cache[frame_number] = flow
        self._decoded_flow_cache.move_to_end(frame_number)
        while len(self._decoded_flow_cache) > _MAX_DECODED_FLOW_CACHE:
            self._decoded_flow_cache.popitem(last=False)
        return flow

    def update_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional[np.ndarray] = None
    ) -> None:
        _ = frame_rgb
        w = self._window
        if not getattr(w, "canvas", None):
            return
        frame_idx = int(frame_number)
        if not self.has_overlay_for_frame(frame_idx):
            # During a live run, let preview overlays persist even if NDJSON
            # hasn't been finalized/loaded yet.
            if self._live_running:
                return
            w.canvas.setFlowPreviewOverlay(None)
            return

        flow = self._decoded_flow_for_frame(frame_idx)
        if flow is None:
            w.canvas.setFlowPreviewOverlay(None)
            return
        overlay = self._overlay_from_flow(flow)
        if overlay is None:
            w.canvas.setFlowPreviewOverlay(None)
            return
        if overlay.shape[2] == 3:
            alpha_val = self._opacity_alpha()
            alpha = np.full(
                (overlay.shape[0], overlay.shape[1], 1), alpha_val, dtype=np.uint8
            )
            overlay = np.concatenate([overlay, alpha], axis=2)
        w.canvas.setFlowPreviewOverlay(np.ascontiguousarray(overlay))

    def has_overlay_for_frame(self, frame_number: int) -> bool:
        try:
            idx = int(frame_number)
            return (idx in self._record_offsets) or (idx in self._records)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _is_processing(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.isRunning())

    def _show_already_running_message(self) -> None:
        w = self._window
        QtWidgets.QMessageBox.information(
            w,
            w.tr("Already running"),
            w.tr("Optical flow is already processing a video."),
        )

    def _build_run_settings(self) -> Optional[FlowRunSettings]:
        video_path = self._resolve_video_path()
        if not video_path:
            return None

        prefs = self._current_preferences()
        self._apply_preferences_to_window(prefs)
        return FlowRunSettings(
            video_path=video_path,
            ndjson_path=str(self._default_ndjson_path(video_path)),
            backend=prefs.backend,
            raft_model=prefs.raft_model,
            visualization=prefs.visualization,
            opacity=int(prefs.opacity),
            quiver_step=int(prefs.quiver_step),
            quiver_gain=float(prefs.quiver_gain),
            stable_hsv=bool(prefs.stable_hsv),
            farneback_pyr_scale=float(prefs.farneback_pyr_scale),
            farneback_levels=int(prefs.farneback_levels),
            farneback_winsize=int(prefs.farneback_winsize),
            farneback_iterations=int(prefs.farneback_iterations),
            farneback_poly_n=int(prefs.farneback_poly_n),
            farneback_poly_sigma=float(prefs.farneback_poly_sigma),
        )

    def _resolve_video_path(self) -> Optional[str]:
        w = self._window
        video_path = getattr(w, "video_file", "") or ""
        if not video_path:
            video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                w,
                w.tr("Select video for optical flow"),
                str(Path.home()),
                w.tr("Video Files (*.mp4 *.avi *.mov *.mkv)"),
            )
        if not video_path:
            return None
        return str(Path(video_path).expanduser().resolve())

    def _apply_preferences_to_window(self, prefs: OpticalFlowPreferences) -> None:
        prefs.apply_to_target(self._window)
        manager = self._manager or getattr(self._window, "optical_flow_manager", None)
        if manager is not None:
            prefs.apply_to_target(manager)
            if hasattr(manager, "_preferences"):
                manager._preferences = prefs

    def _persist_preferences(self, prefs: OpticalFlowPreferences) -> None:
        settings = getattr(self._window, "settings", None)
        if not isinstance(settings, QtCore.QSettings):
            return
        for key, value in prefs.qsettings_items():
            settings.setValue(key, value)

    def _current_preferences(self) -> OpticalFlowPreferences:
        return OpticalFlowPreferences.from_source(self._window)

    def _start_worker(self, settings: FlowRunSettings) -> None:
        w = self._window
        self._live_running = True
        self._stop_requested = False
        self._active_ndjson_path = Path(settings.ndjson_path)
        _, use_torch_farneback = resolve_optical_flow_backend_flags(settings.backend)
        worker = FlexibleWorker(
            process_video_flow,
            settings.video_path,
            backend=settings.backend,
            save_csv=None,
            save_ndjson=settings.ndjson_path,
            sample_stride=1,
            visualization=settings.visualization,
            raft_model=settings.raft_model,
            opacity=settings.opacity,
            quiver_step=settings.quiver_step,
            quiver_gain=settings.quiver_gain,
            stable_hsv=settings.stable_hsv,
            use_torch_farneback=use_torch_farneback,
            farneback_pyr_scale=settings.farneback_pyr_scale,
            farneback_levels=settings.farneback_levels,
            farneback_winsize=settings.farneback_winsize,
            farneback_iterations=settings.farneback_iterations,
            farneback_poly_n=settings.farneback_poly_n,
            farneback_poly_sigma=settings.farneback_poly_sigma,
            progress_callback=None,
            preview_callback=None,
        )
        worker._kwargs["progress_callback"] = (
            lambda percent: worker.progress_signal.emit(percent)
        )
        worker._kwargs["preview_callback"] = lambda payload: worker.preview_signal.emit(
            payload
        )

        worker.progress_signal.connect(self._on_progress, QtCore.Qt.QueuedConnection)
        worker.preview_signal.connect(self._on_preview, QtCore.Qt.QueuedConnection)
        worker.finished_signal.connect(
            self._on_worker_finished,
            QtCore.Qt.QueuedConnection,
        )

        worker_thread = configure_worker_thread_stack(QtCore.QThread(w))
        worker.moveToThread(worker_thread)
        worker.finished_signal.connect(worker_thread.quit, QtCore.Qt.QueuedConnection)
        worker_thread.finished.connect(worker.deleteLater)
        worker_thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        worker_thread.start()

        self._worker = worker
        self._worker_thread = worker_thread
        try:
            w.statusBar().showMessage(w.tr("Optical flow running..."), 3000)
        except Exception:
            pass

    def _opacity_alpha(self) -> int:
        w = self._window
        opacity = float(getattr(w, "flow_opacity", 70))
        return int(np.clip(opacity, 0, 100) / 100.0 * 255.0)

    def _default_ndjson_path(self, video_file: Optional[str] = None) -> Path:
        path = self._flow_ndjson_path(video_file)
        if path:
            return path
        return Path.home() / "flow.ndjson"

    def _flow_ndjson_path(self, video_file: Optional[str] = None) -> Optional[Path]:
        candidate = video_file or getattr(self._window, "video_file", None)
        if not candidate:
            return None
        video_path = Path(candidate).expanduser()
        flow_dir = video_path.parent / video_path.stem
        return flow_dir / f"{video_path.stem}_flow.ndjson"

    def _decode_flow_component(
        self, comp: Dict[str, object], height: int, width: int
    ) -> Optional[np.ndarray]:
        return decode_flow_component(comp, height, width)

    def _overlay_from_flow(self, flow: np.ndarray) -> Optional[np.ndarray]:
        prefs = self._current_preferences()
        try:
            global_max = self._global_mag_max if prefs.stable_hsv else None
            return render_flow_overlay(
                flow,
                visualization=prefs.visualization,
                opacity=prefs.opacity,
                quiver_step=prefs.quiver_step,
                quiver_gain=prefs.quiver_gain,
                max_mag=global_max,
            )
        except Exception:
            return None

    def _overlay_from_record(self, record: Dict[str, object]) -> Optional[np.ndarray]:
        flow = decode_flow_record(record)
        if flow is None:
            return None
        return self._overlay_from_flow(flow)

    def _on_progress(self, percent: int) -> None:
        w = self._window
        try:
            w.statusBar().showMessage(w.tr("Optical flow %d%%") % int(percent), 1500)
        except Exception:
            pass

    def _on_preview(self, payload: object) -> None:
        w = self._window
        try:
            overlay = None
            frame_index = None
            if isinstance(payload, dict):
                overlay = payload.get("overlay")
                frame_index = payload.get("frame_index")
            if overlay is None:
                return
            overlay = np.asarray(overlay)
            if overlay.ndim != 3 or overlay.shape[2] < 3:
                return
            if overlay.shape[2] == 3:
                alpha = np.full(
                    (overlay.shape[0], overlay.shape[1], 1),
                    self._opacity_alpha(),
                    dtype=np.uint8,
                )
                overlay = np.concatenate([overlay, alpha], axis=2)
            if frame_index is not None:
                try:
                    w.set_frame_number(int(frame_index))
                except Exception:
                    pass
            if getattr(w, "canvas", None):
                w.canvas.setFlowPreviewOverlay(np.ascontiguousarray(overlay))
            if frame_index is not None:
                try:
                    w._set_depth_preview_frame(int(frame_index))
                except Exception:
                    pass
        except Exception:
            pass

    @QtCore.Slot(object)
    def _on_worker_finished(self, result: object) -> None:
        w = self._window
        self._live_running = False
        was_stopped = self._stop_requested or (
            self._worker is not None
            and bool(getattr(self._worker, "is_stopped", lambda: False)())
        )
        self._stop_requested = False
        ndjson_path = self._active_ndjson_path
        worker_thread = self._worker_thread

        if worker_thread is not None:
            # Avoid waiting on the current thread; this can happen if callbacks run
            # off-thread and would trigger "QThread::wait: Thread tried to wait on itself".
            if worker_thread.isRunning():
                worker_thread.quit()
            if QtCore.QThread.currentThread() is not worker_thread:
                worker_thread.wait(3000)
            worker_thread.deleteLater()

        self._worker = None
        self._worker_thread = None
        self._active_ndjson_path = None

        if isinstance(result, Exception):
            QtWidgets.QMessageBox.critical(w, w.tr("Optical flow error"), str(result))
            return

        if was_stopped or result == "Stopped":
            try:
                w.statusBar().showMessage(w.tr("Optical flow stopped."), 2500)
            except Exception:
                pass
            return

        if ndjson_path is None:
            return
        self._load_records_from_path(ndjson_path)
        try:
            w.statusBar().showMessage(w.tr("Optical flow complete."), 3000)
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            w,
            w.tr("Optical flow"),
            w.tr("Optical flow completed.\nNDJSON: %s") % str(ndjson_path),
        )
