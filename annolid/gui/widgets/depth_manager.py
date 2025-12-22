from __future__ import annotations

import base64
import contextlib
import functools
import io
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import imageio
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.depth import run_video_depth_anything as depth_run
from annolid.gui.workers import FlexibleWorker
from annolid.gui.widgets.depth_settings_dialog import DepthSettingsDialog
from annolid.utils.logger import logger
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image


class DepthManager(QtCore.QObject):
    """Encapsulates depth processing (Video-Depth-Anything) and preview overlay logic."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self._depth_ndjson_records: Dict[int, Dict[str, object]] = {}
        self._video_depth_worker = None
        self._video_depth_worker_thread: Optional[QtCore.QThread] = None
        self._depth_preview_active = False

    # ------------------------------------------------------------------ public actions
    def run_video_depth_anything(self) -> None:
        """Run Video-Depth-Anything using the currently loaded video."""
        w = self.window
        if not w.video_file:
            QtWidgets.QMessageBox.warning(
                w,
                w.tr("No video loaded"),
                w.tr("Please open a video before running Video-Depth-Anything."),
            )
            return

        if (
            self._video_depth_worker_thread
            and self._video_depth_worker_thread.isRunning()
        ):
            QtWidgets.QMessageBox.information(
                w,
                w.tr("Already running"),
                w.tr("Video Depth Anything is already processing a video."),
            )
            return

        video_path = Path(w.video_file).expanduser().resolve()
        if w.canvas:
            with contextlib.suppress(Exception):
                w.canvas.setDepthPreviewOverlay(None)

        depth_cfg = w._config.get("video_depth_anything", {})
        encoder = depth_cfg.get("encoder", "vits")
        input_size = depth_cfg.get("input_size", 518)
        max_res = depth_cfg.get("max_res", 1280)
        max_len = depth_cfg.get("max_len", -1)
        target_fps = depth_cfg.get("target_fps", -1)
        metric = depth_cfg.get("metric", False)
        fp32 = depth_cfg.get("fp32", False)
        grayscale = depth_cfg.get("grayscale", False)
        save_npz = depth_cfg.get("save_npz", False)
        save_exr = depth_cfg.get("save_exr", False)
        streaming = depth_cfg.get("streaming", True)
        save_depth_video = depth_cfg.get("save_depth_video", False)
        save_depth_frames = depth_cfg.get("save_depth_frames", False)
        include_region_labels = depth_cfg.get("include_region_labels", False)
        show_preview = depth_cfg.get("show_preview", True)

        custom_output = depth_cfg.get("output_dir")
        output_dir = (
            Path(custom_output).expanduser().resolve()
            if custom_output
            else video_path.parent / f"{video_path.stem}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        worker = FlexibleWorker(
            depth_run,
            str(video_path),
            str(output_dir),
            encoder=encoder,
            input_size=input_size,
            max_res=max_res,
            max_len=max_len,
            target_fps=target_fps,
            metric=metric,
            fp32=fp32,
            grayscale=grayscale,
            save_npz=save_npz,
            save_exr=save_exr,
            streaming=streaming,
            save_point_clouds=depth_cfg.get("save_point_clouds", False),
            include_region_labels=include_region_labels,
            progress_callback=None,
            preview_callback=None,
            save_depth_video=save_depth_video,
            save_depth_frames=save_depth_frames,
        )
        worker._kwargs["progress_callback"] = lambda percent: worker.progress_signal.emit(
            percent
        )
        worker.progress_signal.connect(
            lambda percent: w.statusBar().showMessage(
                w.tr("Video Depth Anything %d%%") % percent, 2000
            )
        )
        if show_preview:
            worker._kwargs["preview_callback"] = (
                lambda payload: worker.preview_signal.emit(payload)
            )
            worker.preview_signal.connect(self._handle_depth_preview)

        worker_thread = QtCore.QThread(w)
        worker.moveToThread(worker_thread)
        worker.finished_signal.connect(
            functools.partial(
                self._handle_video_depth_finished,
                output_dir=str(output_dir),
                worker_thread=worker_thread,
            )
        )
        worker.finished_signal.connect(worker_thread.quit)
        worker_thread.started.connect(worker.run)
        worker_thread.start()

        self._video_depth_worker = worker
        self._video_depth_worker_thread = worker_thread

        w.statusBar().showMessage(
            w.tr("Video Depth Anything running..."), 5000
        )

    def configure_video_depth_settings(self) -> None:
        dialog = DepthSettingsDialog(
            parent=self.window,
            config=self.window._config.get("video_depth_anything", {}),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.values()
        self.window._config.setdefault(
            "video_depth_anything", {}).update(values)
        if isinstance(self.window.settings, QtCore.QSettings):
            self.window.settings.setValue("video_depth_anything", values)

    # ------------------------------------------------------------------ overlays and ndjson
    def load_depth_ndjson_records(self) -> None:
        path = self._depth_ndjson_path()
        records: Dict[int, Dict[str, object]] = {}
        if not path or not path.exists():
            self._depth_ndjson_records = records
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
        except Exception:
            records = {}
        self._depth_ndjson_records = records

    def update_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional[np.ndarray] = None
    ) -> None:
        if not self.window.canvas:
            return
        record = self._depth_ndjson_records.get(frame_number)
        if record is None:
            self.window.canvas.setDepthPreviewOverlay(None)
            return
        overlay = self._load_depth_overlay_from_record(record, frame_rgb)
        self.window.canvas.setDepthPreviewOverlay(overlay)

    # ------------------------------------------------------------------ internals
    def _handle_depth_preview(self, payload: object) -> None:
        try:
            overlay_image = None
            frame_index = None
            if isinstance(payload, dict):
                overlay_image = payload.get("overlay")
                frame_index = payload.get("frame_index")
            depth_stats = payload.get("depth_stats") if isinstance(
                payload, dict) else None
            if self.window.canvas and overlay_image is not None:
                qimage = QtGui.QImage(
                    overlay_image.data,
                    overlay_image.shape[1],
                    overlay_image.shape[0],
                    overlay_image.strides[0],
                    QtGui.QImage.Format_RGB888,
                )
                pixmap = QtGui.QPixmap.fromImage(qimage)
                self.window.canvas.loadPixmap(pixmap, clear_shapes=True)
            if frame_index is not None:
                self._set_depth_preview_frame(int(frame_index))
                self.window.statusBar().showMessage(
                    self.window.tr("Video Depth Anything frame %d") % int(
                        frame_index),
                    1000,
                )
            if depth_stats is not None and frame_index is not None:
                logger.debug(
                    "Depth preview frame %d stats: min=%.5f max=%.5f mean=%.5f display=[%.5f, %.5f]",
                    frame_index,
                    depth_stats.get("min", 0.0),
                    depth_stats.get("max", 0.0),
                    depth_stats.get("mean", 0.0),
                    depth_stats.get("display_min", 0.0),
                    depth_stats.get("display_max", 0.0),
                )
        except Exception:
            pass

    def _set_depth_preview_frame(self, frame_index: int) -> None:
        self.window.frame_number = frame_index
        if hasattr(self.window, "seekbar") and self.window.seekbar is not None:
            blocker = QtCore.QSignalBlocker(self.window.seekbar)
            blocker.__enter__()
            self.window.seekbar.setValue(frame_index)
            blocker.__exit__(None, None, None)

    def _depth_ndjson_path(self) -> Optional[Path]:
        if not self.window.video_file:
            return None
        video_path = Path(self.window.video_file)
        return video_path.parent / video_path.stem / "depth.ndjson"

    def _build_depth_overlay(self, frame_rgb: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        depth = depth_map.astype(np.float32)
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            d_min, d_max = 0.0, 1.0
        else:
            d_min = float(np.percentile(finite, 1.0))
            d_max = float(np.percentile(finite, 99.0))
            if d_max - d_min < 1e-6:
                d_min = float(finite.min())
                d_max = float(finite.max())
                if d_max - d_min < 1e-6:
                    d_max = d_min + 1e-6
        clipped = np.clip(depth, d_min, d_max)
        normalized = (clipped - d_min) / (d_max - d_min + 1e-9)
        vis_u8 = (normalized * 255).astype(np.uint8)
        color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_INFERNO)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        overlay = color.astype(np.uint8)
        alpha = np.full(
            (overlay.shape[0], overlay.shape[1], 1), 200, dtype=np.uint8)
        return np.concatenate([overlay, alpha], axis=2)

    def _load_depth_overlay_from_record(
        self, record: Dict[str, object], frame_rgb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        depth_info = (record.get("otherData") or {}).get("depth_map")
        if not depth_info:
            return None
        image_data = depth_info.get("image_data")
        if not image_data:
            return None
        scale = depth_info.get("scale") or {}
        d_min = float(scale.get("min", 0.0))
        d_max = float(scale.get("max", 1.0))
        try:
            img_bytes = base64.b64decode(image_data)
        except Exception:
            return None
        buffer = io.BytesIO(img_bytes)
        quantized = imageio.imread(buffer)
        if quantized.ndim == 3:
            quantized = quantized[..., 0]
        quantized = quantized.astype(np.float32)
        depth = quantized / 65535.0
        depth = depth * (d_max - d_min) + d_min
        if frame_rgb is None:
            frame_rgb = self._current_frame_rgb()
        if frame_rgb is None:
            return None
        height, width = frame_rgb.shape[:2]
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height),
                               interpolation=cv2.INTER_LINEAR)
        return self._build_depth_overlay(frame_rgb, depth)

    def _current_frame_rgb(self) -> Optional[np.ndarray]:
        pixmap = getattr(self.window.canvas, "pixmap", None)
        if pixmap is None or pixmap.isNull():
            return None
        qimg = pixmap.toImage()
        try:
            return convert_qt_image_to_rgb_cv_image(qimg).copy()
        except Exception:
            return None

    def _restore_canvas_frame(self) -> None:
        if not self.window.filename:
            return
        qimg = QtGui.QImage(self.window.filename)
        if qimg.isNull():
            return
        self.window.canvas.loadPixmap(
            QtGui.QPixmap.fromImage(qimg), clear_shapes=False)
        if self.window.video_loader:
            with contextlib.suppress(Exception):
                self.window.video_loader.load_frame(self.window.frame_number)
        self.window.loadPredictShapes(self.window.frame_number,
                                      self.window.filename)

    def _handle_video_depth_finished(
        self,
        result,
        *,
        output_dir: str,
        worker_thread: QtCore.QThread,
    ) -> None:
        with contextlib.suppress(Exception):
            if self.window.canvas:
                self.window.canvas.setDepthPreviewOverlay(None)
        worker_thread.quit()
        worker_thread.wait()
        self._video_depth_worker = None
        self._video_depth_worker_thread = None

        if isinstance(result, Exception):
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("Video Depth Anything"),
                self.window.tr(
                    "Video Depth Anything failed:\n%s") % str(result),
            )
            self.window.statusBar().showMessage(
                self.window.tr("Video Depth Anything failed."), 5000
            )
            return

        self.load_depth_ndjson_records()
        self._depth_preview_active = False
        self._restore_canvas_frame()
        QtWidgets.QMessageBox.information(
            self.window,
            self.window.tr("Video Depth Anything"),
            self.window.tr("Depth outputs saved to %s.") % output_dir,
        )
        self.window.statusBar().showMessage(
            self.window.tr("Video Depth Anything complete."), 5000
        )
