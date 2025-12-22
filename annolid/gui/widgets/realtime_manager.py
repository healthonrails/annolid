from __future__ import annotations

import contextlib
import json
import logging
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools import mask as maskUtils  # type: ignore
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets import RealtimeControlWidget
from annolid.gui.workers import PerceptionProcessWorker, RealtimeSubscriberWorker
from annolid.realtime.perception import Config as RealtimeConfig
from annolid.gui.shape import Shape, MaskShape
from annolid.utils.logger import logger


class RealtimeManager(QtCore.QObject):
    """Encapsulates realtime inference dialog, workers, and overlays."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.realtime_perception_worker = None
        self.realtime_subscriber_worker = None
        self.realtime_running = False
        self._realtime_connect_address = None
        self._realtime_shapes: List[Shape] = []
        self.realtime_log_enabled = False
        self.realtime_log_fp = None
        self.realtime_log_path = None

        # Dialog + control widget
        self.realtime_control_dialog = QtWidgets.QDialog(window)
        self.realtime_control_dialog.setWindowTitle(
            window.tr("Realtime Control"))
        self.realtime_control_dialog.setModal(False)
        self.realtime_control_widget = RealtimeControlWidget(
            parent=self.realtime_control_dialog,
            config=window._config,
        )
        self.realtime_control_widget.start_requested.connect(
            self._handle_realtime_start_request)
        self.realtime_control_widget.stop_requested.connect(
            self.stop_realtime_inference)
        dialog_layout = QtWidgets.QVBoxLayout(self.realtime_control_dialog)
        dialog_layout.setContentsMargins(10, 10, 10, 10)
        dialog_layout.addWidget(self.realtime_control_widget)
        self.realtime_control_dialog.resize(420, 560)
        self.realtime_control_widget.set_status_text(
            window.tr("Realtime idle."))

    # ------------------------------------------------------------------ UI helpers
    def show_control_dialog(self) -> None:
        self.realtime_control_dialog.show()
        self.realtime_control_dialog.raise_()
        self.realtime_control_dialog.activateWindow()

    # ------------------------------------------------------------------ Start/Stop
    def _handle_realtime_start_request(
        self,
        realtime_config: RealtimeConfig,
        extras: Dict[str, Any],
    ):
        self.show_control_dialog()
        if self.realtime_perception_worker is not None:
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("Realtime Inference"),
                self.window.tr("A realtime session is already running."),
            )
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.window.tr("Realtime session already running."))
            return

        publisher = realtime_config.publisher_address
        if publisher:
            try:
                with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    sock.settimeout(0.5)
                    host, port = self._resolve_tcp_endpoint(publisher)
                    bind_result = sock.connect_ex((host, port))
                    if bind_result == 0:
                        raise RuntimeError(
                            self.window.tr("Publisher port %1 is already in use.").replace("%1", str(port)))
            except RuntimeError:
                message = self.window.tr(
                    "Publisher address %1 is already in use.").replace("%1", publisher)
                QtWidgets.QMessageBox.warning(
                    self.window,
                    self.window.tr("Realtime Inference"),
                    message,
                )
                self.realtime_control_widget.set_running(False)
                self.realtime_control_widget.set_status_text(message)
                return
            except Exception:
                pass

        try:
            self.start_realtime_inference(realtime_config, extras)
        except Exception as exc:
            logger.error("Failed to start realtime inference: %s",
                         exc, exc_info=True)
            self.realtime_running = False
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("Realtime Inference"),
                self.window.tr(
                    "Unable to start realtime inference: %s") % str(exc),
            )
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.window.tr("Failed to start realtime inference."))

    def start_realtime_inference(self,
                                 realtime_config: RealtimeConfig,
                                 extras: Dict[str, Any]):
        resolved_model = self._resolve_model_path(
            realtime_config.model_base_name)
        if resolved_model is not None and self._validate_model_file(resolved_model):
            realtime_config.model_base_name = str(resolved_model)
        else:
            # Let Ultralytics handle downloading the model automatically.
            logger.info(
                "Realtime model %s not found locally or failed validation; "
                "allowing Ultralytics to download it automatically.",
                realtime_config.model_base_name,
            )

        self.realtime_control_widget.set_running(True)
        self._realtime_connect_address = extras.get(
            "subscriber_address", "tcp://127.0.0.1:5555")

        self.realtime_running = True
        self._realtime_shapes = []
        self.realtime_log_fp = None
        self.realtime_log_path = None
        self.realtime_log_enabled = bool(extras.get("log_enabled", False))

        status_message = self.window.tr("Realtime inference starting with %s") \
            % realtime_config.model_base_name

        if self.realtime_log_enabled:
            try:
                log_path = self._prepare_realtime_log_path(
                    extras.get("log_path", ""))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self.realtime_log_fp = open(log_path, "a", encoding="utf-8")
                self.realtime_log_path = log_path
                logger.info(
                    "Realtime detections will be logged to %s", log_path)
                status_message += f" (logging to {log_path})"
            except Exception as exc:
                logger.error("Failed to open realtime NDJSON log: %s",
                             exc, exc_info=True)
                self.realtime_log_fp = None
                self.realtime_log_path = None
                self.realtime_log_enabled = False
                status_message += self.window.tr(" (logging disabled)")
        else:
            self.realtime_log_fp = None
            self.realtime_log_path = None

        self.window.statusBar().showMessage(status_message)
        self.realtime_control_widget.set_status_text(status_message)

        self.realtime_perception_worker = PerceptionProcessWorker(
            config=realtime_config,
            parent=self.window,
        )
        self.realtime_perception_worker.error.connect(
            self._on_realtime_error)
        self.realtime_perception_worker.stopped.connect(
            self._on_realtime_stopped)
        self.realtime_perception_worker.start()

        self.realtime_subscriber_worker = RealtimeSubscriberWorker(
            self._realtime_connect_address)
        self.realtime_subscriber_worker.frame_received.connect(
            self._on_realtime_frame)
        self.realtime_subscriber_worker.status_received.connect(
            self._on_realtime_status)
        self.realtime_subscriber_worker.error.connect(
            self._on_realtime_error)
        self.realtime_subscriber_worker.start()

    def stop_realtime_inference(self):
        if self.realtime_perception_worker is None and not self.realtime_running:
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.window.tr("Realtime inference stopped."))
            return

        self.realtime_control_widget.set_stopping()
        self.realtime_control_widget.set_status_text(
            self.window.tr("Stopping realtime inference…"))
        self.window.statusBar().showMessage(
            self.window.tr("Stopping realtime inference…"))
        self._shutdown_realtime_subscriber()

        worker = self.realtime_perception_worker
        if worker is not None:
            worker.request_stop()
            return

        self._finalize_realtime_shutdown()

    # ------------------------------------------------------------------ Helpers
    def _prepare_realtime_log_path(self, requested_path: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if requested_path:
            path = Path(requested_path).expanduser()
            if path.is_dir() or not path.suffix:
                path = path / f"realtime_{timestamp}.ndjson"
        else:
            path = Path.home() / "annolid_realtime_logs" / \
                f"realtime_{timestamp}.ndjson"
        return path.resolve()

    def _resolve_tcp_endpoint(self, address: str) -> Tuple[str, int]:
        if not address.startswith("tcp://"):
            raise ValueError(f"Unsupported address format: {address}")
        host_port = address[len("tcp://"):].strip()
        if host_port.startswith("*:"):
            host = "127.0.0.1"
            port_part = host_port[2:]
        elif host_port.count(":") >= 1:
            host, port_part = host_port.rsplit(":", 1)
            if host in ("*", "0.0.0.0"):
                host = "127.0.0.1"
        else:
            raise ValueError(f"Invalid tcp address: {address}")
        port = int(port_part)
        return host, port

    def _resolve_model_path(self, model_name: str) -> Optional[Path]:
        """Find a local model file to avoid network downloads."""
        name = str(model_name or "").strip()
        if not name:
            return None
        candidate_paths = []
        p = Path(name).expanduser()
        candidate_paths.append(p)
        candidate_paths.append(Path.cwd() / name)
        # Common Ultralytics/torch cache paths.
        home = Path.home()
        candidate_paths.append(home / ".cache" / "ultralytics" / name)
        candidate_paths.append(home / ".cache" / "torch" /
                               "hub" / "checkpoints" / name)
        for path in candidate_paths:
            if path.exists():
                return path
        return None

    def _validate_model_file(self, path: Path) -> bool:
        """Basic sanity checks to catch empty/corrupt checkpoint files."""
        try:
            if not path.exists():
                return False
            if path.stat().st_size < 100_000:  # heuristically too small
                return False
            return True
        except Exception:
            return False

    def _decode_mask(self, mask_data, width: int, height: int):
        if not mask_data:
            return None

        encoding = (mask_data.get("encoding") or "").lower()

        try:
            if encoding in {"coco_rle", "rle"}:
                counts = mask_data.get("counts")
                if counts is None:
                    return None
                rle = {
                    "size": mask_data.get("size") or [height, width],
                    "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
                }
                mask = maskUtils.decode(rle)
                if mask is None:
                    return None
                if mask.shape[1] != width or mask.shape[0] != height:
                    mask = cv2.resize(mask.astype(np.uint8),
                                      (width, height),
                                      interpolation=cv2.INTER_NEAREST)
                return mask.astype(bool)

            if encoding == "polygon":
                points = np.array(mask_data.get("points")
                                  or [], dtype=np.float32)
                if points.size == 0:
                    return None
                pts = points.copy()
                pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                return mask.astype(bool)

            if encoding == "bitmap":
                data = mask_data.get("data")
                if isinstance(data, list):
                    arr = np.array(data, dtype=np.uint8)
                    if arr.size == width * height:
                        return arr.reshape((height, width)).astype(bool)

        except Exception as exc:
            logger.debug("Failed to decode realtime mask: %s",
                         exc, exc_info=True)

        return None

    def _convert_detections_to_shapes(self,
                                      detections: List[dict],
                                      width: int,
                                      height: int) -> List[Shape]:
        from labelme import utils  # local import to avoid circular on startup

        shapes: List[Shape] = []
        if not detections:
            return shapes

        for detection in detections:
            label = str(detection.get("behavior", "") or "")
            base_color_rgb = self.window._get_rgb_by_label(
                label) or (0, 255, 0)
            base_color = QtGui.QColor(
                int(base_color_rgb[0]),
                int(base_color_rgb[1]),
                int(base_color_rgb[2]),
                255
            )
            fill_color = QtGui.QColor(
                base_color.red(),
                base_color.green(),
                base_color.blue(),
                60
            )

            bbox = detection.get("bbox_normalized") or []
            if len(bbox) != 4:
                bbox = None

            rect_shape = None
            if bbox:
                x1 = max(0.0, min(1.0, float(bbox[0]))) * width
                y1 = max(0.0, min(1.0, float(bbox[1]))) * height
                x2 = max(0.0, min(1.0, float(bbox[2]))) * width
                y2 = max(0.0, min(1.0, float(bbox[3]))) * height

                if x2 > x1 and y2 > y1:
                    rect_shape = Shape(
                        label=label,
                        shape_type="rectangle",
                        flags={"source": "realtime"},
                        description="realtime",
                    )
                    rect_shape.points = [
                        QtCore.QPointF(x1, y1),
                        QtCore.QPointF(x2, y2),
                    ]
                    rect_shape.point_labels = [1, 1]
                    rect_shape.fill = True
                    rect_shape.line_color = QtGui.QColor(base_color)
                    rect_shape.fill_color = QtGui.QColor(fill_color)
                    rect_shape.select_line_color = QtGui.QColor(
                        255, 255, 255, 255)
                    rect_shape.select_fill_color = QtGui.QColor(
                        base_color.red(),
                        base_color.green(),
                        base_color.blue(),
                        160
                    )
                    rect_shape.other_data["confidence"] = float(
                        detection.get("confidence", 0.0))
                    rect_shape.other_data["source"] = "realtime"
                    rect_shape.other_data["frame_timestamp"] = detection.get(
                        "timestamp")
                    shapes.append(rect_shape)

            mask_data = detection.get("mask")
            if mask_data:
                mask = self._decode_mask(mask_data, width, height)
                if mask is not None:
                    mask_shape = MaskShape(
                        label=label,
                        flags={"source": "realtime"},
                        description="realtime_mask",
                    )
                    mask_shape.mask_color = np.array(
                        [base_color.red(), base_color.green(),
                         base_color.blue(), 64],
                        dtype=np.uint8
                    )
                    mask_shape.boundary_color = np.array(
                        [base_color.red(), base_color.green(),
                         base_color.blue(), 180],
                        dtype=np.uint8
                    )
                    mask_shape.mask = mask
                    mask_shape.scale = 1.0
                    mask_shape.other_data = dict(
                        rect_shape.other_data if rect_shape else {})
                    mask_shape.other_data["confidence"] = float(
                        detection.get("confidence", 0.0))
                    shapes.append(mask_shape)

            keypoints = detection.get("keypoints")
            if keypoints:
                try:
                    kp_array = np.array(
                        keypoints, dtype=np.float32).reshape(-1, 2)
                except ValueError:
                    kp_array = np.array(keypoints, dtype=np.float32)
                    if kp_array.ndim == 1:
                        kp_array = kp_array.reshape(-1, 2)
                if kp_array.size > 0 and kp_array.shape[1] == 2:
                    points_shape = Shape(
                        label=f"{label}_keypoints",
                        shape_type="points",
                        flags={"source": "realtime"},
                        description="realtime_keypoints",
                    )
                    points_shape.points = []
                    points_shape.point_labels = []
                    for point in kp_array:
                        px = max(0.0, min(1.0, float(point[0]))) * width
                        py = max(0.0, min(1.0, float(point[1]))) * height
                        points_shape.points.append(QtCore.QPointF(px, py))
                        points_shape.point_labels.append(1)
                    points_shape.line_color = QtGui.QColor(base_color)
                    points_shape.vertex_fill_color = QtGui.QColor(
                        base_color.red(),
                        base_color.green(),
                        base_color.blue(),
                        255
                    )
                    shapes.append(points_shape)

        return shapes

    # ------------------------------------------------------------------ slots
    @QtCore.Slot(object, dict, list)
    def _on_realtime_frame(self, qimage, metadata, detections):
        if not self.realtime_running:
            return

        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.window.canvas.loadPixmap(pixmap, clear_shapes=False)
        shapes = self._convert_detections_to_shapes(
            detections, pixmap.width(), pixmap.height())
        if hasattr(self.window.canvas, "setRealtimeShapes"):
            self.window.canvas.setRealtimeShapes(shapes)
        self._realtime_shapes = shapes

        if self.realtime_log_fp:
            try:
                record = {
                    "timestamp": time.time(),
                    "frame_metadata": metadata,
                    "detections": detections,
                }
                json.dump(record, self.realtime_log_fp)
                self.realtime_log_fp.write("\n")
                self.realtime_log_fp.flush()
            except Exception as exc:
                logger.error("Failed to write realtime NDJSON record: %s",
                             exc, exc_info=True)
                with contextlib.suppress(Exception):
                    self.realtime_log_fp.close()
                self.realtime_log_fp = None
                self.realtime_log_path = None
                self.realtime_log_enabled = False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Realtime detections for frame %s: %d",
                         metadata.get("frame_index"),
                         len(detections))

        self.window.canvas.update()

        frame_index = metadata.get("frame_index")
        detection_count = len(shapes)
        self.window.statusBar().showMessage(
            self.window.tr("Realtime frame %s — detections: %d")
            % (frame_index if frame_index is not None else "?",
               detection_count))
        self.realtime_control_widget.set_status_text(
            self.window.tr("Frame %s — detections: %d")
            % (frame_index if frame_index is not None else "?",
               detection_count))

    @QtCore.Slot(dict)
    def _on_realtime_status(self, status):
        if not isinstance(status, dict):
            return
        event_name = status.get("event") or "status"
        message = self.window.tr("Realtime %s: %s") % (
            event_name,
            status.get("recording_state", status.get("message", "")),
        )
        self.window.statusBar().showMessage(message)
        self.realtime_control_widget.set_status_text(message)

    @QtCore.Slot(str)
    def _on_realtime_error(self, message: str):
        logger.error("Realtime error: %s", message)
        self.realtime_control_widget.set_status_text(
            self.window.tr("Realtime error: %s") % message)
        QtWidgets.QMessageBox.critical(
            self.window,
            self.window.tr("Realtime Inference Error"),
            str(message),
        )
        self.stop_realtime_inference()

    @QtCore.Slot()
    def _on_realtime_stopped(self):
        self.realtime_perception_worker = None
        self._finalize_realtime_shutdown()

    # ------------------------------------------------------------------ teardown
    def _shutdown_realtime_subscriber(self):
        if self.realtime_subscriber_worker is not None:
            self.realtime_subscriber_worker.stop()
            self.realtime_subscriber_worker.wait(500)
            self.realtime_subscriber_worker = None

    def _finalize_realtime_shutdown(self):
        self._shutdown_realtime_subscriber()
        self.realtime_running = False
        self.realtime_perception_worker = None
        self.realtime_control_widget.set_running(False)
        if hasattr(self.window.canvas, "setRealtimeShapes"):
            self.window.canvas.setRealtimeShapes([])
        self._realtime_shapes = []
        if self.realtime_log_fp:
            with contextlib.suppress(Exception):
                self.realtime_log_fp.flush()
                self.realtime_log_fp.close()
            self.realtime_log_fp = None
            self.realtime_log_path = None
        message = self.window.tr("Realtime inference stopped.")
        self.window.statusBar().showMessage(message)
        self.realtime_control_widget.set_status_text(message)
        # Restore persisted shapes for the current frame.
        try:
            self.window.loadPredictShapes(
                getattr(self.window, "frame_number", 0),
                getattr(self.window, "filename", None),
            )
        except Exception:
            pass
