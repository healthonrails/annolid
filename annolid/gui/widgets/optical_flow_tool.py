from __future__ import annotations

import base64
import json
import gzip
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.optical_flow_dialog import FlowOptionsDialog
from annolid.gui.workers import FlexibleWorker
from annolid.motion.flow_runner import process_video_flow, flow_to_color
from annolid.utils import draw


@dataclass
class FlowPreferences:
    backend: str = "farneback"
    raft_model: str = "small"
    visualization: str = "hsv"
    opacity: int = 70
    quiver_step: int = 16
    quiver_gain: float = 1.0
    stable_hsv: bool = True
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 1
    farneback_winsize: int = 1
    farneback_iterations: int = 3
    farneback_poly_n: int = 3
    farneback_poly_sigma: float = 1.1


@dataclass
class FlowRunSettings(FlowPreferences):
    video_path: str = ""
    ndjson_path: str = ""


class OpticalFlowTool(QtCore.QObject):
    """Encapsulates optical-flow UI, running, and NDJSON overlay playback."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self._window = window
        self._worker: Optional[FlexibleWorker] = None
        self._worker_thread: Optional[QtCore.QThread] = None
        self._live_running: bool = False
        self._records: Dict[int, Dict[str, object]] = {}
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
        values = dialog.values()
        if not values:
            return
        (
            backend,
            raft_model,
            viz_choice,
            opacity,
            quiver_step,
            quiver_gain,
            stable_hsv,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
        ) = values
        updated = FlowPreferences(
            backend=backend,
            raft_model=raft_model,
            visualization=viz_choice,
            opacity=opacity,
            quiver_step=quiver_step,
            quiver_gain=quiver_gain,
            stable_hsv=stable_hsv,
            farneback_pyr_scale=pyr_scale,
            farneback_levels=levels,
            farneback_winsize=winsize,
            farneback_iterations=iterations,
            farneback_poly_n=poly_n,
            farneback_poly_sigma=poly_sigma,
        )
        self._apply_preferences_to_window(updated)
        self._persist_preferences(updated)
        try:
            self._window.statusBar().showMessage(
                self._window.tr("Optical flow settings updated."), 2500
            )
        except Exception:
            pass

    def clear(self) -> None:
        self._records = {}
        self._global_mag_max = None
        self._live_running = False
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
            self._global_mag_max = None
            return
        self._load_records_from_path(path)

    def _load_records_from_path(self, path: Path) -> None:
        records: Dict[int, Dict[str, object]] = {}
        global_mag_max = 0.0
        if not path.exists():
            self._records = records
            self._global_mag_max = None
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    frame_index = int(data.get("frame_index", -1))
                    if frame_index >= 0:
                        records[frame_index] = data
                        other = data.get("otherData") or {}
                        # Support new raw-compressed format and legacy quantized format.
                        mag_info = (
                            other.get("flow_mag_raw")
                            or other.get("flow_magnitude")
                            or {}
                        )
                        scale = mag_info.get("scale") or {}
                        try:
                            mag_max = float(scale.get("max", 0.0))
                            if mag_max > global_mag_max:
                                global_mag_max = mag_max
                        except Exception:
                            pass
        except Exception:
            records = {}
            global_mag_max = 0.0
        self._records = records
        self._global_mag_max = global_mag_max if global_mag_max > 0 else None

    def update_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional[np.ndarray] = None
    ) -> None:
        w = self._window
        if not getattr(w, "canvas", None):
            return
        record = self._records.get(int(frame_number))
        if record is None:
            # During a live run, let preview overlays persist even if NDJSON
            # hasn't been appended/loaded yet.
            if self._live_running:
                return
            w.canvas.setFlowPreviewOverlay(None)
            return
        overlay = self._overlay_from_record(record)
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
            backend=prefs.backend,
            raft_model=prefs.raft_model,
            visualization=prefs.visualization,
            ndjson_path=str(self._default_ndjson_path(video_path)),
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

    def _apply_preferences_to_window(self, prefs: FlowPreferences) -> None:
        w = self._window
        setattr(w, "optical_flow_backend", prefs.backend)
        setattr(w, "optical_flow_raft_model", prefs.raft_model)
        setattr(w, "flow_visualization", prefs.visualization)
        setattr(w, "flow_opacity", prefs.opacity)
        setattr(w, "flow_quiver_step", prefs.quiver_step)
        setattr(w, "flow_quiver_gain", prefs.quiver_gain)
        setattr(w, "flow_stable_hsv", prefs.stable_hsv)
        setattr(w, "flow_farneback_pyr_scale", prefs.farneback_pyr_scale)
        setattr(w, "flow_farneback_levels", prefs.farneback_levels)
        setattr(w, "flow_farneback_winsize", prefs.farneback_winsize)
        setattr(w, "flow_farneback_iterations", prefs.farneback_iterations)
        setattr(w, "flow_farneback_poly_n", prefs.farneback_poly_n)
        setattr(w, "flow_farneback_poly_sigma", prefs.farneback_poly_sigma)

    def _persist_preferences(self, prefs: FlowPreferences) -> None:
        settings = getattr(self._window, "settings", None)
        if not isinstance(settings, QtCore.QSettings):
            return
        settings.setValue("optical_flow/backend", prefs.backend)
        settings.setValue("optical_flow/raft_model", prefs.raft_model)
        settings.setValue("optical_flow/visualization", prefs.visualization)
        settings.setValue("optical_flow/opacity", prefs.opacity)
        settings.setValue("optical_flow/quiver_step", prefs.quiver_step)
        settings.setValue("optical_flow/quiver_gain", prefs.quiver_gain)
        settings.setValue("optical_flow/stable_hsv", prefs.stable_hsv)
        settings.setValue("optical_flow/farneback_pyr_scale", prefs.farneback_pyr_scale)
        settings.setValue("optical_flow/farneback_levels", prefs.farneback_levels)
        settings.setValue("optical_flow/farneback_winsize", prefs.farneback_winsize)
        settings.setValue(
            "optical_flow/farneback_iterations", prefs.farneback_iterations
        )
        settings.setValue("optical_flow/farneback_poly_n", prefs.farneback_poly_n)
        settings.setValue(
            "optical_flow/farneback_poly_sigma", prefs.farneback_poly_sigma
        )

    def _current_preferences(self) -> FlowPreferences:
        w = self._window
        return FlowPreferences(
            backend=str(getattr(w, "optical_flow_backend", "farneback")),
            raft_model=str(getattr(w, "optical_flow_raft_model", "small")),
            visualization=str(getattr(w, "flow_visualization", "hsv")),
            opacity=int(getattr(w, "flow_opacity", 70)),
            quiver_step=int(getattr(w, "flow_quiver_step", 16)),
            quiver_gain=float(getattr(w, "flow_quiver_gain", 1.0)),
            stable_hsv=bool(getattr(w, "flow_stable_hsv", True)),
            farneback_pyr_scale=float(getattr(w, "flow_farneback_pyr_scale", 0.5)),
            farneback_levels=int(getattr(w, "flow_farneback_levels", 1)),
            farneback_winsize=int(getattr(w, "flow_farneback_winsize", 1)),
            farneback_iterations=int(getattr(w, "flow_farneback_iterations", 3)),
            farneback_poly_n=int(getattr(w, "flow_farneback_poly_n", 3)),
            farneback_poly_sigma=float(getattr(w, "flow_farneback_poly_sigma", 1.1)),
        )

    def _start_worker(self, settings: FlowRunSettings) -> None:
        w = self._window
        self._live_running = True
        backend_val = str(settings.backend).lower()
        use_torch_farneback = ("torch" in backend_val) and ("raft" not in backend_val)
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

        worker.progress_signal.connect(self._on_progress)
        worker.preview_signal.connect(self._on_preview)

        def _finished(result: object) -> None:
            self._on_finished(result, Path(settings.ndjson_path))

        worker.finished_signal.connect(_finished)

        worker_thread = QtCore.QThread(w)
        worker.moveToThread(worker_thread)
        worker.finished_signal.connect(worker_thread.quit)
        worker_thread.started.connect(worker.run)
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
        if not comp:
            return None
        values = comp.get("values")
        if values is None and comp.get("compressed"):
            b64 = comp.get("data")
            if not b64:
                return None
            try:
                raw = gzip.decompress(base64.b64decode(b64))
                dtype = np.float16 if comp.get("dtype") == "float16" else np.uint16
                arr = np.frombuffer(raw, dtype=dtype)
                shape = comp.get("shape") or []
                try:
                    arr = arr.reshape(shape)
                except Exception:
                    pass
            except Exception:
                return None
        else:
            if values is None:
                return None
            dtype = np.float16 if comp.get("dtype") == "float16" else np.uint16
            arr = np.asarray(values, dtype=dtype)
        scale = comp.get("scale", {}) if isinstance(comp, dict) else {}
        try:
            if arr.ndim != 2:
                arr = np.array(arr, dtype=arr.dtype)
            if comp.get("dtype") == "uint16":
                v_min = float(scale.get("min", 0.0))
                v_max = float(scale.get("max", 0.0))
                if abs(v_max - v_min) < 1e-6:
                    v_max = v_min + 1e-6
                arr = (arr.astype(np.float32) / 65535.0) * (v_max - v_min) + v_min
            v_min = float(scale.get("min", 0.0))
            v_max = float(scale.get("max", 0.0))
            arr = np.clip(arr, v_min, v_max)
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
            return arr.astype(np.float32)
        except Exception:
            return None

    def _overlay_from_record(self, record: Dict[str, object]) -> Optional[np.ndarray]:
        w = self._window
        try:
            height = int(record.get("imageHeight") or 0)
            width = int(record.get("imageWidth") or 0)
            if height <= 0 or width <= 0:
                return None
            other = record.get("otherData") or {}
            # New format stores raw compressed arrays under *_raw keys.
            dx_comp = other.get("flow_dx_raw") or other.get("flow_dx")
            dy_comp = other.get("flow_dy_raw") or other.get("flow_dy")
            dx = self._decode_flow_component(dx_comp, height, width)
            dy = self._decode_flow_component(dy_comp, height, width)
            if dx is None or dy is None:
                return None
            flow = np.stack([dx, dy], axis=-1).astype(np.float32)
            viz = str(getattr(w, "flow_visualization", "hsv")).lower()
            if viz == "hsv":
                global_max = (
                    self._global_mag_max
                    if bool(getattr(w, "flow_stable_hsv", True))
                    else None
                )
                return flow_to_color(flow, max_mag=global_max)

            flow_scaled = flow * float(getattr(w, "flow_quiver_gain", 1.0))
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            arrows_bgr = draw.draw_flow(
                blank,
                flow_scaled,
                step=int(getattr(w, "flow_quiver_step", 16)),
            )
            arrows_rgb = cv2.cvtColor(arrows_bgr, cv2.COLOR_BGR2RGB)
            mask = np.any(arrows_bgr != 0, axis=2)
            alpha_val = self._opacity_alpha()
            alpha = np.zeros((height, width), dtype=np.uint8)
            alpha[mask] = alpha_val
            return np.ascontiguousarray(np.dstack([arrows_rgb, alpha]))
        except Exception:
            return None

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

    def _on_finished(self, result: object, ndjson_path: Path) -> None:
        w = self._window
        self._live_running = False
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._worker = None
        self._worker_thread = None

        if isinstance(result, Exception):
            QtWidgets.QMessageBox.critical(w, w.tr("Optical flow error"), str(result))
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
