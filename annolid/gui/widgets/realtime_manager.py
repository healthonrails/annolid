from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from pycocotools import mask as maskUtils  # type: ignore
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets import RealtimeControlWidget
from annolid.gui.widgets.bot_explain import _resolve_chat_widget
from annolid.gui.workers import PerceptionProcessWorker, RealtimeSubscriberWorker
from annolid.gui.shape import Shape, MaskShape
from annolid.utils.logger import logger
from annolid.utils.log_paths import resolve_annolid_realtime_logs_root

if TYPE_CHECKING:
    from annolid.realtime.config import Config as RealtimeConfig


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
        self._classify_eye_blinks = False
        self._blink_ear_threshold = 0.21
        self._blink_min_consecutive_frames = 2
        self._blink_state: Dict[str, Any] = {
            "closed_frames": 0,
            "blink_count": 0,
            "eyes_closed": False,
        }
        self._last_realtime_model_name = ""
        self._last_realtime_camera_source = ""
        self._last_realtime_viewer_type = ""
        self._last_realtime_rtsp_transport = "auto"
        self._bot_report_enabled = False
        self._bot_report_interval_sec = 5.0
        self._bot_watch_labels: set[str] = set()
        self._bot_email_report = False
        self._bot_email_to = ""
        self._bot_email_min_interval_sec = 60.0
        self._bot_last_report_ts = 0.0
        self._bot_last_email_request_ts = 0.0
        self._bot_last_attempt_ts = 0.0
        self._bot_last_busy_log_ts = 0.0
        self._bot_report_signature = ""
        self._bot_report_signature_ts = 0.0
        self._bot_report_dedup_window_sec = 20.0
        self._bot_event_log_fp = None
        self._bot_event_log_path = None

        # Dock + control widget
        self.realtime_control_dock = QtWidgets.QDockWidget(
            window.tr("Realtime Control"), window
        )
        self.realtime_control_dock.setObjectName("realtimeControlDock")
        self.realtime_control_widget = RealtimeControlWidget(
            parent=self.realtime_control_dock,
            config=window._config,
        )
        self.realtime_control_widget.start_requested.connect(
            self._handle_realtime_start_request
        )
        self.realtime_control_widget.stop_requested.connect(
            self.stop_realtime_inference
        )
        self.realtime_control_dock.setWidget(self.realtime_control_widget)
        self.realtime_control_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        # We don't add it to the window yet; show_control_dialog will handle it.
        self.realtime_control_dock.hide()
        self.realtime_control_widget.set_status_text(window.tr("Realtime idle."))

        self.realtime_control_dock.visibilityChanged.connect(
            self._on_dock_visibility_changed
        )

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        """Handle dock visibility changes to restore layout when closed."""
        if not visible and not self.realtime_running:
            self.window.set_unrelated_docks_visible(True)

    # ------------------------------------------------------------------ UI helpers
    def show_control_dialog(self) -> None:
        """Show the realtime control dock and hide other docks for focus."""
        # Ensure it's docked on the right side.
        self.window.addDockWidget(
            QtCore.Qt.RightDockWidgetArea, self.realtime_control_dock
        )

        if not self.realtime_control_dock.isVisible():
            self.window.set_unrelated_docks_visible(
                False, exclude=[self.realtime_control_dock]
            )
            self.realtime_control_dock.show()

        self.realtime_control_dock.raise_()

    # ------------------------------------------------------------------ Start/Stop
    def _handle_realtime_start_request(
        self,
        realtime_config: "RealtimeConfig",
        extras: Dict[str, Any],
    ):
        if not bool(extras.get("suppress_control_dock", False)):
            self.show_control_dialog()
        if self.realtime_perception_worker is not None:
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("Realtime Inference"),
                self.window.tr("A realtime session is already running."),
            )
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.window.tr("Realtime session already running.")
            )
            return

        publisher = realtime_config.publisher_address
        if publisher:
            try:
                with contextlib.closing(
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ) as sock:
                    sock.settimeout(0.5)
                    host, port = self._resolve_tcp_endpoint(publisher)
                    bind_result = sock.connect_ex((host, port))
                    if bind_result == 0:
                        raise RuntimeError(
                            self.window.tr(
                                "Publisher port %1 is already in use."
                            ).replace("%1", str(port))
                        )
            except RuntimeError:
                message = self.window.tr(
                    "Publisher address %1 is already in use."
                ).replace("%1", publisher)
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
            logger.error("Failed to start realtime inference: %s", exc, exc_info=True)
            self.realtime_running = False
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("Realtime Inference"),
                self.window.tr("Unable to start realtime inference: %s") % str(exc),
            )
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.window.tr("Failed to start realtime inference.")
            )

    def start_realtime_inference(
        self, realtime_config: "RealtimeConfig", extras: Dict[str, Any]
    ):
        resolved_model = self._resolve_model_path(realtime_config.model_base_name)
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
            "subscriber_address", "tcp://127.0.0.1:5555"
        )
        self._last_realtime_model_name = str(realtime_config.model_base_name or "")
        self._last_realtime_camera_source = str(realtime_config.camera_index or "")
        self._last_realtime_viewer_type = str(extras.get("viewer_type", ""))
        self._last_realtime_rtsp_transport = str(
            extras.get("rtsp_transport", "auto") or "auto"
        )

        self.realtime_running = True
        self._realtime_shapes = []
        self.realtime_log_fp = None
        self.realtime_log_path = None
        self.realtime_log_enabled = bool(extras.get("log_enabled", False))
        self._classify_eye_blinks = bool(extras.get("classify_eye_blinks", False))
        self._blink_ear_threshold = float(extras.get("blink_ear_threshold", 0.21))
        self._blink_ear_threshold = max(0.05, min(0.6, self._blink_ear_threshold))
        self._blink_min_consecutive_frames = int(
            extras.get("blink_min_consecutive_frames", 2)
        )
        self._blink_min_consecutive_frames = max(
            1, min(30, self._blink_min_consecutive_frames)
        )
        self._blink_state = {
            "closed_frames": 0,
            "blink_count": 0,
            "eyes_closed": False,
        }
        self._bot_report_enabled = bool(extras.get("bot_report_enabled", False))
        try:
            self._bot_report_interval_sec = max(
                1.0, float(extras.get("bot_report_interval_sec", 5.0))
            )
        except Exception:
            self._bot_report_interval_sec = 5.0
        self._bot_watch_labels = self._normalize_bot_watch_labels(
            extras.get("bot_watch_labels", [])
        )
        self._bot_email_report = bool(extras.get("bot_email_report", False))
        self._bot_email_to = str(extras.get("bot_email_to", "") or "").strip()
        try:
            self._bot_email_min_interval_sec = max(
                10.0,
                float(extras.get("bot_email_min_interval_sec", 60.0) or 60.0),
            )
        except Exception:
            self._bot_email_min_interval_sec = 60.0
        self._bot_last_report_ts = 0.0
        self._bot_last_email_request_ts = 0.0
        self._bot_last_attempt_ts = 0.0
        self._bot_last_busy_log_ts = 0.0
        self._bot_report_signature = ""
        self._bot_report_signature_ts = 0.0

        if self._bot_report_enabled:
            if self._bot_email_report and not self._bot_email_to:
                logger.warning(
                    "Realtime bot email reporting is enabled but no recipient is set."
                )
            labels_text = (
                ", ".join(sorted(self._bot_watch_labels))
                if self._bot_watch_labels
                else "(any detection)"
            )
            logger.info(
                "Realtime bot reporting enabled: interval=%ss labels=%s email=%s recipient=%s email_min_interval=%ss",
                int(self._bot_report_interval_sec),
                labels_text,
                "on" if self._bot_email_report else "off",
                self._bot_email_to or "(none)",
                int(self._bot_email_min_interval_sec),
            )
            self._open_bot_event_log()

        status_message = (
            self.window.tr("Realtime inference starting with %s")
            % realtime_config.model_base_name
        )

        if self.realtime_log_enabled:
            try:
                log_path = self._prepare_realtime_log_path(extras.get("log_path", ""))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self.realtime_log_fp = open(log_path, "a", encoding="utf-8")
                self.realtime_log_path = log_path
                logger.info("Realtime detections will be logged to %s", log_path)
                status_message += f" (logging to {log_path})"
            except Exception as exc:
                logger.error(
                    "Failed to open realtime NDJSON log: %s", exc, exc_info=True
                )
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
        self.realtime_perception_worker.error.connect(self._on_realtime_error)
        self.realtime_perception_worker.stopped.connect(self._on_realtime_stopped)
        self.realtime_perception_worker.start()

        self.realtime_subscriber_worker = RealtimeSubscriberWorker(
            self._realtime_connect_address
        )
        self.realtime_subscriber_worker.frame_received.connect(self._on_realtime_frame)
        self.realtime_subscriber_worker.status_received.connect(
            self._on_realtime_status
        )
        self.realtime_subscriber_worker.error.connect(self._on_realtime_error)
        self.realtime_subscriber_worker.start()

        viewer_type = extras.get("viewer_type", "pyqt")
        if viewer_type == "threejs":
            threejs_manager = getattr(self.window, "threejs_manager", None)
            if threejs_manager:
                viewer = threejs_manager.ensure_threejs_viewer()
                if viewer:
                    viewer.init_viewer(
                        enable_eye_control=extras.get("enable_eye_control", False),
                        enable_hand_control=extras.get("enable_hand_control", False),
                    )
                self.window._set_active_view("threejs")
        else:
            self.window._set_active_view("canvas")

    def stop_realtime_inference(self):
        if self.realtime_perception_worker is None and not self.realtime_running:
            self.realtime_control_widget.set_running(False)
            self.realtime_control_widget.set_status_text(
                self.window.tr("Realtime inference stopped.")
            )
            return

        self.realtime_control_widget.set_stopping()
        self.realtime_control_widget.set_status_text(
            self.window.tr("Stopping realtime inference…")
        )
        self.window.statusBar().showMessage(
            self.window.tr("Stopping realtime inference…")
        )
        self._shutdown_realtime_subscriber()

        worker = self.realtime_perception_worker
        if worker is not None:
            worker.request_stop()
            if not worker.wait(2500):  # Wait up to 2.5s
                logger.warning(
                    "Realtime perception worker did not stop gracefully in time."
                )
            self.realtime_perception_worker = None

        self._finalize_realtime_shutdown()

    # ------------------------------------------------------------------ Helpers
    def _prepare_realtime_log_path(self, requested_path: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if requested_path:
            path = Path(requested_path).expanduser()
            if path.is_dir() or not path.suffix:
                path = path / f"realtime_{timestamp}.ndjson"
        else:
            path = resolve_annolid_realtime_logs_root() / f"realtime_{timestamp}.ndjson"
        return path.resolve()

    def _resolve_tcp_endpoint(self, address: str) -> Tuple[str, int]:
        if not address.startswith("tcp://"):
            raise ValueError(f"Unsupported address format: {address}")
        host_port = address[len("tcp://") :].strip()
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
        from annolid.yolo import get_ultralytics_weights_cache_dir

        home = Path.home()
        candidate_paths.append(get_ultralytics_weights_cache_dir() / name)
        candidate_paths.append(home / ".cache" / "ultralytics" / name)
        candidate_paths.append(home / ".cache" / "torch" / "hub" / "checkpoints" / name)
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
                    "counts": counts.encode("utf-8")
                    if isinstance(counts, str)
                    else counts,
                }
                mask = maskUtils.decode(rle)
                if mask is None:
                    return None
                if mask.shape[1] != width or mask.shape[0] != height:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                return mask.astype(bool)

            if encoding == "polygon":
                points = np.array(mask_data.get("points") or [], dtype=np.float32)
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
            logger.debug("Failed to decode realtime mask: %s", exc, exc_info=True)

        return None

    @staticmethod
    def _normalize_bot_watch_labels(watch_labels: Any) -> set[str]:
        source = watch_labels
        normalized: set[str] = set()
        if isinstance(source, str):
            values = [p.strip() for p in source.split(",") if p.strip()]
        elif isinstance(source, list):
            values = [str(v).strip() for v in source if str(v).strip()]
        else:
            values = []
        for value in values:
            normalized.add(value.lower())
        return normalized

    @staticmethod
    def _detection_label(detection: Dict[str, Any]) -> str:
        return str(detection.get("behavior", "") or "").strip()

    def _filter_bot_report_detections(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []
        if not self._bot_watch_labels:
            return list(detections)
        matched: List[Dict[str, Any]] = []
        for detection in detections:
            label = self._detection_label(detection).lower()
            if not label:
                continue
            if label in self._bot_watch_labels or any(
                label.startswith(f"{watched}_") for watched in self._bot_watch_labels
            ):
                matched.append(detection)
        return matched

    def _build_detection_signature(self, detections: List[Dict[str, Any]]) -> str:
        if not detections:
            return ""
        labels = [self._detection_label(d).lower() or "unknown" for d in detections]
        counts = Counter(labels)
        pairs = [f"{label}:{count}" for label, count in sorted(counts.items())]
        return "|".join(pairs)

    def _should_send_bot_report(
        self, now_ts: float, detections: List[Dict[str, Any]]
    ) -> bool:
        if not self._bot_report_enabled:
            return False
        if not detections:
            return False
        if now_ts - self._bot_last_report_ts < self._bot_report_interval_sec:
            return False
        if now_ts - self._bot_last_attempt_ts < 1.0:
            return False
        signature = self._build_detection_signature(detections)
        if (
            signature
            and signature == self._bot_report_signature
            and now_ts - self._bot_report_signature_ts
            < self._bot_report_dedup_window_sec
        ):
            return False
        return True

    def _save_bot_report_frame(self, qimage, frame_index: object) -> Optional[str]:
        if qimage is None:
            return None
        try:
            suffix = f"_frame{frame_index}" if frame_index is not None else ""
            fd, temp_path = tempfile.mkstemp(
                prefix=f"annolid_bot_realtime{suffix}_", suffix=".jpg"
            )
            Path(temp_path).unlink(missing_ok=True)
            try:
                if fd >= 0:
                    os.close(fd)
            except Exception:
                pass
            if qimage.save(temp_path, "JPG", 85):
                return temp_path
        except Exception as exc:
            logger.debug("Failed to save bot report frame: %s", exc, exc_info=True)
        return None

    def _send_detection_report_to_bot(
        self,
        metadata: Dict[str, Any],
        detections: List[Dict[str, Any]],
        image_path: Optional[str],
        *,
        request_email: bool,
    ) -> bool:
        widget, err = _resolve_chat_widget(self.window)
        if widget is None:
            logger.debug("Unable to resolve Annolid Bot widget: %s", err)
            self._write_bot_event_log(
                "bot_unavailable",
                {
                    "error": err,
                    "matched_detections": len(detections),
                    "frame_index": metadata.get("frame_index"),
                },
            )
            return False
        if bool(getattr(widget, "is_streaming_chat", False)):
            now_ts = time.time()
            if now_ts - self._bot_last_busy_log_ts >= 10.0:
                logger.info(
                    "Annolid Bot is busy; deferring realtime report (matched=%d).",
                    len(detections),
                )
                self._bot_last_busy_log_ts = now_ts
            self._write_bot_event_log(
                "bot_busy",
                {
                    "matched_detections": len(detections),
                    "frame_index": metadata.get("frame_index"),
                },
            )
            return False

        prompt_input = getattr(widget, "prompt_text_edit", None)
        send_chat = getattr(widget, "chat_with_model", None)
        set_image_path = getattr(widget, "set_image_path", None)
        register_temp_image = getattr(widget, "register_managed_temp_image", None)
        set_chat_mode = getattr(widget, "set_next_chat_mode", None)
        if prompt_input is None or not callable(send_chat):
            self._write_bot_event_log(
                "bot_input_unavailable",
                {"frame_index": metadata.get("frame_index")},
            )
            return False

        labels = [self._detection_label(d).lower() or "unknown" for d in detections]
        counts = Counter(labels)
        summary_items = [f"{label}: {count}" for label, count in counts.most_common(8)]
        frame_index = metadata.get("frame_index")
        frame_ts = metadata.get("timestamp") or time.time()
        prompt = (
            "Realtime detection digest.\n"
            f"Frame: {frame_index if frame_index is not None else '?'}\n"
            f"Timestamp: {frame_ts}\n"
            f"Matched detections: {len(detections)}\n"
            f"Labels: {', '.join(summary_items) if summary_items else 'none'}\n\n"
            "Please analyze recent activity and write a concise report with notable events,"
            " possible behavior interpretation, and recommended follow-up checks.\n"
            "If tool calls are available, you may use them."
        )
        if request_email:
            to_hint = self._bot_email_to or "configured recipient"
            prompt += (
                f"\nAfter analysis, call the `email` tool to send the report to {to_hint}."
                "\nUse subject: Realtime detection report."
                "\nIn your chat reply, confirm whether email tool call succeeded."
            )

        try:
            if image_path and callable(set_image_path):
                set_image_path(image_path)
                if callable(register_temp_image):
                    register_temp_image(image_path)
                # Keep default mode so agent tooling (e.g. email) remains available.
                if callable(set_chat_mode):
                    set_chat_mode("default")
            prompt_input.setPlainText(prompt)
            prompt_input.setFocus()
            send_chat()
            self._write_bot_event_log(
                "report_sent_to_bot",
                {
                    "frame_index": frame_index,
                    "matched_detections": len(detections),
                    "labels": dict(counts),
                    "image_attached": bool(image_path),
                    "email_requested": bool(request_email),
                    "email_to": self._bot_email_to,
                },
            )
            return True
        except Exception as exc:
            logger.debug(
                "Failed to send realtime report to bot: %s", exc, exc_info=True
            )
            self._write_bot_event_log(
                "send_exception",
                {
                    "error": str(exc),
                    "frame_index": frame_index,
                    "matched_detections": len(detections),
                },
            )
            return False

    def _open_bot_event_log(self) -> None:
        if self._bot_event_log_fp is not None:
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = (
                resolve_annolid_realtime_logs_root()
                / f"realtime_bot_events_{timestamp}.ndjson"
            ).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._bot_event_log_fp = open(path, "a", encoding="utf-8")
            self._bot_event_log_path = path
            logger.info("Realtime bot event logging to %s", path)
        except Exception as exc:
            logger.warning("Failed to open realtime bot event log: %s", exc)
            self._bot_event_log_fp = None
            self._bot_event_log_path = None

    def _write_bot_event_log(self, event: str, payload: Dict[str, Any]) -> None:
        fp = self._bot_event_log_fp
        if fp is None:
            return
        try:
            entry = {
                "timestamp": time.time(),
                "event": str(event or "").strip(),
                "payload": payload or {},
            }
            json.dump(entry, fp)
            fp.write("\n")
            fp.flush()
        except Exception:
            pass

    def _convert_detections_to_shapes(
        self, detections: List[dict], width: int, height: int
    ) -> List[Shape]:
        shapes: List[Shape] = []
        if not detections:
            return shapes

        for detection in detections:
            label = str(detection.get("behavior", "") or "")
            base_color_rgb = self.window._get_rgb_by_label(label) or (0, 255, 0)
            base_color = QtGui.QColor(
                int(base_color_rgb[0]),
                int(base_color_rgb[1]),
                int(base_color_rgb[2]),
                255,
            )
            fill_color = QtGui.QColor(
                base_color.red(), base_color.green(), base_color.blue(), 60
            )

            bbox_pixels = detection.get("bbox_pixels") or []
            if len(bbox_pixels) != 4:
                bbox_pixels = None

            bbox_norm = detection.get("bbox_normalized") or []
            if len(bbox_norm) != 4:
                bbox_norm = None

            rect_shape = None
            if bbox_pixels or bbox_norm:
                if bbox_pixels:
                    x1 = max(0.0, min(float(width), float(bbox_pixels[0])))
                    y1 = max(0.0, min(float(height), float(bbox_pixels[1])))
                    x2 = max(0.0, min(float(width), float(bbox_pixels[2])))
                    y2 = max(0.0, min(float(height), float(bbox_pixels[3])))
                else:
                    x1 = max(0.0, min(1.0, float(bbox_norm[0]))) * width
                    y1 = max(0.0, min(1.0, float(bbox_norm[1]))) * height
                    x2 = max(0.0, min(1.0, float(bbox_norm[2]))) * width
                    y2 = max(0.0, min(1.0, float(bbox_norm[3]))) * height

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
                    rect_shape.select_line_color = QtGui.QColor(255, 255, 255, 255)
                    rect_shape.select_fill_color = QtGui.QColor(
                        base_color.red(), base_color.green(), base_color.blue(), 160
                    )
                    rect_shape.other_data["confidence"] = float(
                        detection.get("confidence", 0.0)
                    )
                    rect_shape.other_data["source"] = "realtime"
                    rect_shape.other_data["frame_timestamp"] = detection.get(
                        "timestamp"
                    )
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
                        [base_color.red(), base_color.green(), base_color.blue(), 64],
                        dtype=np.uint8,
                    )
                    mask_shape.boundary_color = np.array(
                        [base_color.red(), base_color.green(), base_color.blue(), 180],
                        dtype=np.uint8,
                    )
                    mask_shape.mask = mask
                    mask_shape.scale = 1.0
                    mask_shape.other_data = dict(
                        rect_shape.other_data if rect_shape else {}
                    )
                    mask_shape.other_data["confidence"] = float(
                        detection.get("confidence", 0.0)
                    )
                    shapes.append(mask_shape)

            keypoints_pixels = detection.get("keypoints_pixels")
            keypoints_norm = detection.get("keypoints")
            keypoints = (
                keypoints_pixels if keypoints_pixels is not None else keypoints_norm
            )
            keypoints_are_pixels = keypoints_pixels is not None
            if keypoints:
                try:
                    kp_array = np.array(keypoints, dtype=np.float32).reshape(-1, 2)
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
                        if keypoints_are_pixels:
                            px = max(0.0, min(float(width), float(point[0])))
                            py = max(0.0, min(float(height), float(point[1])))
                        else:
                            px = max(0.0, min(1.0, float(point[0]))) * width
                            py = max(0.0, min(1.0, float(point[1]))) * height
                        points_shape.points.append(QtCore.QPointF(px, py))
                        points_shape.point_labels.append(1)
                    points_shape.line_color = QtGui.QColor(base_color)
                    points_shape.vertex_fill_color = QtGui.QColor(
                        base_color.red(), base_color.green(), base_color.blue(), 255
                    )
                    shapes.append(points_shape)

        return shapes

    @staticmethod
    def _point_from_keypoints(keypoints: Any, idx: int) -> Optional[np.ndarray]:
        if not isinstance(keypoints, list):
            return None
        if idx < 0 or idx >= len(keypoints):
            return None
        point = keypoints[idx]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            return np.array([float(point[0]), float(point[1])], dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def _eye_aspect_ratio(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
        p5: np.ndarray,
        p6: np.ndarray,
    ) -> float:
        horiz = float(np.linalg.norm(p1 - p4))
        if horiz <= 1e-6:
            return 0.0
        vert = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
        return vert / (2.0 * horiz)

    def _classify_eye_blink(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        keypoints = detection.get("keypoints_pixels") or detection.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) < 388:
            return detection

        # MediaPipe face mesh indices for EAR computation.
        right_idx = (33, 160, 158, 133, 153, 144)
        left_idx = (362, 385, 387, 263, 373, 380)
        right_pts = [self._point_from_keypoints(keypoints, idx) for idx in right_idx]
        left_pts = [self._point_from_keypoints(keypoints, idx) for idx in left_idx]
        if any(pt is None for pt in right_pts) or any(pt is None for pt in left_pts):
            return detection

        right_ear = self._eye_aspect_ratio(*right_pts)  # type: ignore[arg-type]
        left_ear = self._eye_aspect_ratio(*left_pts)  # type: ignore[arg-type]
        ear = float((right_ear + left_ear) / 2.0)

        closed = ear < float(self._blink_ear_threshold)
        state = self._blink_state
        behavior = "eyes_open"
        if closed:
            state["closed_frames"] = int(state.get("closed_frames", 0)) + 1
            state["eyes_closed"] = True
            behavior = "eyes_closed"
        else:
            was_closed = bool(state.get("eyes_closed", False))
            closed_frames = int(state.get("closed_frames", 0))
            if was_closed and closed_frames >= int(self._blink_min_consecutive_frames):
                state["blink_count"] = int(state.get("blink_count", 0)) + 1
                behavior = "eye_blink"
            state["closed_frames"] = 0
            state["eyes_closed"] = False

        updated = dict(detection)
        updated["behavior"] = behavior
        metadata = dict(updated.get("metadata") or {})
        metadata["eye_aspect_ratio"] = ear
        metadata["eyes_closed"] = bool(closed)
        metadata["blink_count"] = int(state.get("blink_count", 0))
        metadata["blink_ear_threshold"] = float(self._blink_ear_threshold)
        updated["metadata"] = metadata
        return updated

    def _apply_behavior_classification(self, detections: List[dict]) -> List[dict]:
        if not self._classify_eye_blinks:
            return list(detections or [])
        transformed: List[dict] = []
        for detection in detections or []:
            payload = dict(detection or {})
            label = str(payload.get("behavior", "") or "").lower()
            keypoint_labels = payload.get("keypoint_labels") or []
            is_face = "face" in label or (
                isinstance(keypoint_labels, list)
                and bool(keypoint_labels)
                and str(keypoint_labels[0]).startswith("face_")
            )
            if is_face:
                payload = self._classify_eye_blink(payload)
            transformed.append(payload)
        return transformed

    # ------------------------------------------------------------------ slots
    @QtCore.Slot(object, dict, list)
    def _on_realtime_frame(self, qimage, metadata, detections):
        if not self.realtime_running:
            return

        effective_detections = self._apply_behavior_classification(detections)
        matched_bot_detections = self._filter_bot_report_detections(
            effective_detections
        )
        shapes = []
        if qimage is not None:
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.window.canvas.loadPixmap(pixmap, clear_shapes=False)
            shapes = self._convert_detections_to_shapes(
                effective_detections, pixmap.width(), pixmap.height()
            )
            if hasattr(self.window.canvas, "setRealtimeShapes"):
                self.window.canvas.setRealtimeShapes(shapes)
            self._realtime_shapes = shapes
        else:
            # Metadata-only update (e.g. for Eye Control in Three.js)
            self._realtime_shapes = []

        # If Three.js viewer is visible, send the data there too
        threejs_manager = getattr(self.window, "threejs_manager", None)
        if threejs_manager:
            viewer = threejs_manager.viewer_widget()
            if viewer and viewer.isVisible():
                viewer.update_realtime_data(qimage, effective_detections)

        if self.realtime_log_fp:
            try:
                record = {
                    "timestamp": time.time(),
                    "frame_metadata": metadata,
                    "detections": effective_detections,
                }
                json.dump(record, self.realtime_log_fp)
                self.realtime_log_fp.write("\n")
                self.realtime_log_fp.flush()
            except Exception as exc:
                logger.error(
                    "Failed to write realtime NDJSON record: %s", exc, exc_info=True
                )
                with contextlib.suppress(Exception):
                    self.realtime_log_fp.close()
                self.realtime_log_fp = None
                self.realtime_log_path = None
                self.realtime_log_enabled = False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Realtime detections for frame %s: %d",
                metadata.get("frame_index"),
                len(effective_detections),
            )

        self.window.canvas.update()

        frame_index = metadata.get("frame_index")
        detection_count = len(shapes)
        self.window.statusBar().showMessage(
            self.window.tr("Realtime frame %s — detections: %d")
            % (frame_index if frame_index is not None else "?", detection_count)
        )
        self.realtime_control_widget.set_status_text(
            self.window.tr("Frame %s — detections: %d")
            % (frame_index if frame_index is not None else "?", detection_count)
        )

        now_ts = time.time()
        if self._should_send_bot_report(now_ts, matched_bot_detections):
            self._bot_last_attempt_ts = now_ts
            image_path = self._save_bot_report_frame(qimage, frame_index)
            request_email = bool(self._bot_email_report) and (
                now_ts - self._bot_last_email_request_ts
                >= self._bot_email_min_interval_sec
            )
            sent = self._send_detection_report_to_bot(
                metadata if isinstance(metadata, dict) else {},
                matched_bot_detections,
                image_path,
                request_email=request_email,
            )
            if sent:
                self._bot_last_report_ts = now_ts
                self._bot_report_signature = self._build_detection_signature(
                    matched_bot_detections
                )
                self._bot_report_signature_ts = now_ts
                if request_email:
                    self._bot_last_email_request_ts = now_ts
                logger.info(
                    "Realtime report sent to Annolid Bot (frame=%s matched=%d).",
                    frame_index if frame_index is not None else "?",
                    len(matched_bot_detections),
                )

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
            self.window.tr("Realtime error: %s") % message
        )
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
        self._classify_eye_blinks = False
        self._bot_report_enabled = False
        self._bot_watch_labels = set()
        self._bot_email_report = False
        self._bot_email_to = ""
        self._bot_email_min_interval_sec = 60.0
        self._bot_last_report_ts = 0.0
        self._bot_last_email_request_ts = 0.0
        self._bot_last_attempt_ts = 0.0
        self._bot_last_busy_log_ts = 0.0
        self._bot_report_signature = ""
        self._bot_report_signature_ts = 0.0
        if self._bot_event_log_fp:
            with contextlib.suppress(Exception):
                self._bot_event_log_fp.flush()
                self._bot_event_log_fp.close()
            self._bot_event_log_fp = None
            self._bot_event_log_path = None
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
