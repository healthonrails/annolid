"""GUI action slots for shapes, tracking, realtime streams, logs, and approvals."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List

from qtpy import QtCore, QtGui, QtWidgets

from annolid.datasets.labelme_collection import default_label_index_path
from annolid.gui.models_registry import (
    get_model_unavailable_reason,
    get_runtime_model_registry,
)
from annolid.gui.realtime_launch import (
    build_realtime_launch_payload,
    resolve_realtime_model_weight,
)
from annolid.services.chat_realtime import redact_camera_source_for_bot_display
from annolid.utils.log_paths import (
    resolve_annolid_logs_root,
    resolve_annolid_realtime_logs_root,
)
from annolid.utils.runs import shared_runs_root


def _safe_stream_source_for_bot(source: str) -> str:
    return redact_camera_source_for_bot_display(source)


def _log_targets_for_bot() -> Dict[str, Path]:
    logs_root = resolve_annolid_logs_root()
    return {
        "logs": logs_root,
        "realtime": resolve_annolid_realtime_logs_root(),
        "runs": shared_runs_root(),
        "label_index": default_label_index_path().parent,
        "app": logs_root / "app",
    }


def _is_path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _is_probably_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except Exception:
        return True
    return b"\x00" in chunk


def _tail_text_from_file(
    path: Path, *, tail_lines: int, max_chars: int
) -> tuple[str, bool]:
    file_size = int(path.stat().st_size)
    if file_size <= 0:
        return "", False
    chunk_size = 8192
    max_bytes_soft = max(16384, int(max_chars) * 4)
    max_bytes_hard = min(max(max_bytes_soft, int(max_chars) * 8), 8 * 1024 * 1024)
    cursor = file_size
    data = b""
    with path.open("rb") as handle:
        while cursor > 0:
            read_size = min(chunk_size, cursor)
            cursor -= read_size
            handle.seek(cursor)
            data = handle.read(read_size) + data
            if data.count(b"\n") >= int(tail_lines) and len(data) >= max_bytes_soft:
                break
            if len(data) >= max_bytes_hard:
                break
            if chunk_size < 65536:
                chunk_size *= 2
    truncated = bool(cursor > 0 or len(data) >= max_bytes_hard)
    text = data.decode("utf-8", errors="replace")
    tail_text = "\n".join(text.splitlines()[-int(tail_lines) :])
    if len(tail_text) > int(max_chars):
        tail_text = tail_text[-int(max_chars) :]
        truncated = True
    return tail_text, truncated


class AIChatActionsMixin:
    """Mixin encapsulating bot-invoked GUI action slots and approval prompts."""

    @QtCore.Slot(str)
    def request_tool_approval_json(self, payload_json: str = "") -> None:
        """Show an interactive confirmation modal for destructive agent actions."""
        try:
            payload = json.loads(str(payload_json or "{}"))
        except Exception as exc:
            self._set_bot_action_result(
                "tool_approval",
                {"ok": False, "approved": False, "error": f"Invalid payload: {exc}"},
            )
            return
        if not isinstance(payload, dict):
            self._set_bot_action_result(
                "tool_approval",
                {"ok": False, "approved": False, "error": "Invalid payload type."},
            )
            return

        tool_name = str(payload.get("tool_name") or "").strip()
        reason = str(payload.get("reason") or "Run a destructive action").strip()
        args_preview = str(payload.get("args_preview") or "").strip()
        normalized_tool = tool_name.lower()

        if normalized_tool and normalized_tool in self._approved_destructive_tools:
            self._set_bot_action_result(
                "tool_approval",
                {"ok": True, "approved": True, "scope": "session", "tool": tool_name},
            )
            return

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Confirm Agent Action")
        box.setText(
            f"Annolid Bot requests permission to run '{tool_name or 'unknown'}'.\n\n"
            f"Action: {reason}"
        )
        if args_preview:
            box.setInformativeText(f"Parameters:\n{args_preview[:600]}")
        approve_once_btn = box.addButton(
            "Approve Once", QtWidgets.QMessageBox.AcceptRole
        )
        approve_session_btn = box.addButton(
            "Approve for Session", QtWidgets.QMessageBox.YesRole
        )
        deny_btn = box.addButton("Deny", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(deny_btn)
        box.setEscapeButton(deny_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is approve_session_btn:
            if normalized_tool:
                self._approved_destructive_tools.add(normalized_tool)
            self._set_bot_action_result(
                "tool_approval",
                {"ok": True, "approved": True, "scope": "session", "tool": tool_name},
            )
        elif clicked is approve_once_btn:
            self._set_bot_action_result(
                "tool_approval",
                {"ok": True, "approved": True, "scope": "once", "tool": tool_name},
            )
        else:
            self._set_bot_action_result(
                "tool_approval",
                {"ok": True, "approved": False, "scope": "denied", "tool": tool_name},
            )

    @QtCore.Slot(str, str, bool, int)
    def bot_list_shapes(
        self,
        label_contains: str = "",
        shape_type: str = "",
        selected_only: bool = False,
        max_results: int = 200,
    ) -> None:
        host = self.host_window_widget or self.window()
        canvas = getattr(host, "canvas", None)
        shapes = list(getattr(canvas, "shapes", []) or [])
        selected = list(getattr(canvas, "selectedShapes", []) or [])
        selected_ids = {id(shape) for shape in selected}
        text_filter = str(label_contains or "").strip().lower()
        type_filter = str(shape_type or "").strip().lower()
        limit = max(1, min(int(max_results), 500))

        entries: List[Dict[str, Any]] = []
        for idx, shape in enumerate(shapes):
            label = str(getattr(shape, "label", "") or "").strip()
            shape_type_name = (
                str(getattr(shape, "shape_type", "") or "").strip().lower()
            )
            is_selected = id(shape) in selected_ids
            if selected_only and not is_selected:
                continue
            if text_filter and text_filter not in label.lower():
                continue
            if type_filter and type_filter != shape_type_name:
                continue
            points = list(getattr(shape, "points", []) or [])
            entries.append(
                {
                    "index": int(idx),
                    "label": label,
                    "shape_type": shape_type_name,
                    "selected": bool(is_selected),
                    "visible": bool(getattr(shape, "visible", True)),
                    "num_points": int(len(points)),
                }
            )
            if len(entries) >= limit:
                break

        payload = {
            "ok": True,
            "total_shapes": int(len(shapes)),
            "selected_count": int(len(selected_ids)),
            "returned_count": int(len(entries)),
            "shapes": entries,
            "label_contains": str(label_contains or ""),
            "shape_type": str(shape_type or ""),
            "selected_only": bool(selected_only),
        }
        self._set_bot_action_result("list_shapes", payload)
        self.status_label.setText(f"Listed {len(entries)} shape(s).")

    @QtCore.Slot(str, str, int, bool)
    def bot_select_shapes(
        self,
        label_contains: str = "",
        shape_type: str = "",
        max_select: int = 20,
        clear_existing: bool = True,
    ) -> None:
        host = self.host_window_widget or self.window()
        canvas = getattr(host, "canvas", None)
        if canvas is None:
            payload = {"ok": False, "error": "Canvas is unavailable."}
            self._set_bot_action_result("select_shapes", payload)
            self.status_label.setText("Bot action failed: canvas unavailable.")
            return

        shapes = list(getattr(canvas, "shapes", []) or [])
        text_filter = str(label_contains or "").strip().lower()
        type_filter = str(shape_type or "").strip().lower()
        limit = max(1, min(int(max_select), 200))
        matched: List[Any] = []
        for shape in shapes:
            label = str(getattr(shape, "label", "") or "").strip()
            shape_type_name = (
                str(getattr(shape, "shape_type", "") or "").strip().lower()
            )
            if text_filter and text_filter not in label.lower():
                continue
            if type_filter and type_filter != shape_type_name:
                continue
            matched.append(shape)
            if len(matched) >= limit:
                break

        if not matched:
            payload = {
                "ok": False,
                "error": "No matching shapes found.",
                "label_contains": str(label_contains or ""),
                "shape_type": str(shape_type or ""),
            }
            self._set_bot_action_result("select_shapes", payload)
            self.status_label.setText("No matching shapes found.")
            return

        selected = matched
        if not bool(clear_existing):
            current = list(getattr(canvas, "selectedShapes", []) or [])
            merged: List[Any] = []
            seen: set[int] = set()
            for shape in [*current, *matched]:
                shape_id = id(shape)
                if shape_id in seen:
                    continue
                seen.add(shape_id)
                merged.append(shape)
            selected = merged

        try:
            canvas.selectShapes(selected)
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
            self._set_bot_action_result("select_shapes", payload)
            self.status_label.setText(f"Bot action failed: {exc}")
            return

        sync_selection = getattr(host, "shapeSelectionChanged", None)
        if callable(sync_selection):
            try:
                sync_selection(selected)
            except Exception:
                pass

        set_view = getattr(host, "_set_active_view", None)
        if callable(set_view):
            try:
                set_view("canvas")
            except Exception:
                pass

        payload = {
            "ok": True,
            "selected_count": int(len(selected)),
            "selected_labels": [
                str(getattr(shape, "label", "") or "").strip() for shape in selected
            ],
            "label_contains": str(label_contains or ""),
            "shape_type": str(shape_type or ""),
            "clear_existing": bool(clear_existing),
        }
        self._set_bot_action_result("select_shapes", payload)
        self.status_label.setText(f"Selected {len(selected)} shape(s).")

    @QtCore.Slot(str)
    def bot_set_selected_shape_label(self, new_label: str) -> None:
        host = self.host_window_widget or self.window()
        canvas = getattr(host, "canvas", None)
        selected = list(getattr(canvas, "selectedShapes", []) or [])
        label_text = str(new_label or "").strip()
        if not label_text:
            payload = {"ok": False, "error": "new_label is required."}
            self._set_bot_action_result("set_selected_shape_label", payload)
            self.status_label.setText("Bot action failed: empty label.")
            return
        if not selected:
            payload = {"ok": False, "error": "No selected shapes to relabel."}
            self._set_bot_action_result("set_selected_shape_label", payload)
            self.status_label.setText("Bot action failed: no selected shapes.")
            return

        updated = 0
        for shape in selected:
            try:
                setattr(shape, "label", label_text)
                updated += 1
            except Exception:
                continue

        refresh_items = getattr(host, "_refresh_label_list_items_for_shapes", None)
        if callable(refresh_items):
            try:
                refresh_items(selected)
            except Exception:
                pass

        rebuild_unique = getattr(host, "_rebuild_unique_label_list", None)
        if callable(rebuild_unique):
            try:
                rebuild_unique()
            except Exception:
                pass

        set_dirty = getattr(host, "setDirty", None)
        if callable(set_dirty):
            try:
                set_dirty()
            except Exception:
                pass

        if canvas is not None:
            try:
                canvas.update()
            except Exception:
                pass

        payload = {
            "ok": bool(updated > 0),
            "updated_count": int(updated),
            "new_label": label_text,
        }
        self._set_bot_action_result("set_selected_shape_label", payload)
        if updated > 0:
            self.status_label.setText(
                f"Updated label to '{label_text}' for {updated} shape(s)."
            )
        else:
            self.status_label.setText("Bot action failed: no shapes were updated.")

    @QtCore.Slot()
    def bot_delete_selected_shapes(self) -> None:
        host = self.host_window_widget or self.window()
        canvas = getattr(host, "canvas", None)
        selected = list(getattr(canvas, "selectedShapes", []) or [])
        if not selected:
            payload = {"ok": False, "error": "No selected shapes to delete."}
            self._set_bot_action_result("delete_selected_shapes", payload)
            self.status_label.setText("Bot action failed: no selected shapes.")
            return

        delete_selected = getattr(host, "deleteSelectedShapes", None)
        if not callable(delete_selected):
            payload = {"ok": False, "error": "Shape deletion is unavailable."}
            self._set_bot_action_result("delete_selected_shapes", payload)
            self.status_label.setText("Bot action failed: delete action unavailable.")
            return

        before = int(len(getattr(canvas, "shapes", []) or []))
        try:
            delete_selected()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
            self._set_bot_action_result("delete_selected_shapes", payload)
            self.status_label.setText(f"Bot action failed: {exc}")
            return
        after = int(len(getattr(canvas, "shapes", []) or []))
        deleted_count = max(0, before - after)
        payload = {"ok": True, "deleted_count": int(deleted_count)}
        self._set_bot_action_result("delete_selected_shapes", payload)
        self.status_label.setText(f"Deleted {deleted_count} shape(s).")

    @QtCore.Slot(str, bool)
    def bot_set_ai_text_prompt(self, text: str, use_countgd: bool = False) -> None:
        host = self.host_window_widget or self.window()
        widget = getattr(host, "aiRectangle", None)
        prompt_input = getattr(widget, "_aiRectanglePrompt", None)
        if prompt_input is None:
            self.status_label.setText(
                "Bot action failed: AI text prompt UI unavailable."
            )
            return
        try:
            prompt_text = str(text or "").strip()
            prompt_input.setText(prompt_text)
            countgd_checkbox = getattr(widget, "_useCountGDCheckbox", None)
            if countgd_checkbox is not None:
                countgd_checkbox.setChecked(bool(use_countgd))
            self.status_label.setText("AI text prompt updated by bot.")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_configure_ai_text_prompt(self, payload_json: str) -> None:
        host = self.host_window_widget or self.window()
        widget = getattr(host, "aiRectangle", None)
        if widget is None:
            self.status_label.setText(
                "Bot action failed: AI text prompt UI unavailable."
            )
            return
        try:
            options = json.loads(str(payload_json or "{}"))
            if not isinstance(options, dict):
                return
            if "box_threshold" in options and hasattr(widget, "set_box_threshold"):
                widget.set_box_threshold(float(options["box_threshold"]))
            if "output_mode" in options and hasattr(widget, "set_output_mode"):
                widget.set_output_mode(str(options["output_mode"]))
            if "replace_existing" in options and hasattr(
                widget, "set_replace_existing"
            ):
                widget.set_replace_existing(bool(options["replace_existing"]))
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot()
    def bot_run_ai_text_segmentation(self) -> None:
        host = self.host_window_widget or self.window()
        run_segmentation = getattr(host, "_grounding_sam", None)
        if not callable(run_segmentation):
            self.status_label.setText(
                "Bot action failed: text-prompt segmentation is unavailable."
            )
            return
        try:
            run_segmentation()
            self.status_label.setText("Started AI text-prompt segmentation.")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str, str, str, bool, str, int)
    def bot_segment_track_video(
        self,
        video_path: str,
        text_prompt: str,
        mode: str = "track",
        use_countgd: bool = False,
        model_name: str = "",
        to_frame: int = -1,
    ) -> None:
        self._set_bot_action_result(
            "segment_track_video",
            {
                "ok": False,
                "error": "Workflow did not complete.",
                "mode": str(mode or "track").strip().lower(),
                "path": str(video_path or "").strip(),
                "text_prompt": str(text_prompt or "").strip(),
            },
        )
        host = self.host_window_widget or self.window()
        open_video = getattr(host, "openVideo", None)
        run_segmentation = getattr(host, "_grounding_sam", None)
        save_file = getattr(host, "saveFile", None)
        run_track = getattr(host, "predict_from_next_frame", None)
        set_frame = getattr(host, "set_frame_number", None)

        if not callable(open_video) or not callable(run_segmentation):
            self._set_bot_action_result(
                "segment_track_video",
                {
                    "ok": False,
                    "error": "Required video/segmentation workflow is unavailable.",
                },
            )
            self.status_label.setText(
                "Bot action failed: required video/segmentation workflow is unavailable."
            )
            return

        path = str(video_path or "").strip()
        prompt_text = str(text_prompt or "").strip()
        mode_norm = str(mode or "track").strip().lower()
        if not path or not prompt_text:
            self._set_bot_action_result(
                "segment_track_video",
                {"ok": False, "error": "video_path and text_prompt are required."},
            )
            self.status_label.setText(
                "Bot action failed: video path and prompt are required."
            )
            return
        if mode_norm not in {"segment", "track"}:
            self._set_bot_action_result(
                "segment_track_video",
                {"ok": False, "error": "mode must be 'segment' or 'track'."},
            )
            self.status_label.setText(
                "Bot action failed: mode must be 'segment' or 'track'."
            )
            return

        try:
            open_video(from_video_list=True, video_path=path, programmatic_call=True)

            if callable(set_frame):
                set_frame(0)
            if not self._wait_for_canvas_pixmap(timeout_ms=4000):
                self._set_bot_action_result(
                    "segment_track_video",
                    {
                        "ok": False,
                        "error": "Timed out waiting for the first video frame to load.",
                    },
                )
                self.status_label.setText(
                    "Bot action failed: video frame not ready for segmentation."
                )
                return

            widget = getattr(host, "aiRectangle", None)
            prompt_input = getattr(widget, "_aiRectanglePrompt", None)
            if prompt_input is None:
                self._set_bot_action_result(
                    "segment_track_video",
                    {"ok": False, "error": "AI text prompt UI unavailable."},
                )
                self.status_label.setText(
                    "Bot action failed: AI text prompt UI unavailable."
                )
                return
            prompt_input.setText(prompt_text)
            countgd_checkbox = getattr(widget, "_useCountGDCheckbox", None)
            if countgd_checkbox is not None:
                countgd_checkbox.setChecked(bool(use_countgd))

            canvas = getattr(host, "canvas", None)
            before_count = len(getattr(canvas, "shapes", []) or [])
            run_segmentation()
            after_count = len(getattr(canvas, "shapes", []) or [])
            if after_count <= before_count:
                self._set_bot_action_result(
                    "segment_track_video",
                    {
                        "ok": False,
                        "error": (
                            "No polygons were generated from the prompt. "
                            "Try another prompt, enable CountGD, or verify the frame."
                        ),
                    },
                )
                self.status_label.setText("Bot action failed: no polygons generated.")
                return
            if callable(save_file):
                save_file()

            if mode_norm == "track":
                selected_model = str(model_name or "").strip() or "Cutie"
                combo = getattr(host, "_selectAiModelComboBox", None)
                if combo is not None:
                    index = combo.findText(selected_model, QtCore.Qt.MatchContains)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                    else:
                        combo.setCurrentText(selected_model)

                if not callable(run_track):
                    self._set_bot_action_result(
                        "segment_track_video",
                        {"ok": False, "error": "Tracking is unavailable."},
                    )
                    self.status_label.setText(
                        "Segmentation saved, but tracking is unavailable."
                    )
                    return
                total_frames = int(getattr(host, "num_frames", 0) or 0)
                target_frame = int(to_frame)
                if target_frame < 1:
                    target_frame = max(1, total_frames - 1) if total_frames > 0 else 60
                run_track(to_frame=target_frame)
                self._set_bot_action_result(
                    "segment_track_video",
                    {
                        "ok": True,
                        "mode": "track",
                        "path": path,
                        "text_prompt": prompt_text,
                        "to_frame": target_frame,
                        "model_name": selected_model,
                        "use_countgd": bool(use_countgd),
                    },
                )
                self.status_label.setText(
                    f"Started tracking '{prompt_text}' to frame {target_frame}."
                )
                return

            self._set_bot_action_result(
                "segment_track_video",
                {
                    "ok": True,
                    "mode": "segment",
                    "path": path,
                    "text_prompt": prompt_text,
                    "use_countgd": bool(use_countgd),
                },
            )
            self.status_label.setText(
                f"Segmented and saved '{prompt_text}' on frame 0."
            )
        except Exception as exc:
            self._set_bot_action_result(
                "segment_track_video",
                {"ok": False, "error": str(exc)},
            )
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(
        str,
        str,
        str,
        float,
        str,
        str,
        bool,
        float,
        int,
        str,
    )
    def bot_start_realtime_stream(
        self,
        camera_source: str = "",
        model_name: str = "",
        target_behaviors_csv: str = "",
        confidence_threshold: float = -1.0,
        viewer_type: str = "threejs",
        rtsp_transport: str = "auto",
        classify_eye_blinks: bool = False,
        blink_ear_threshold: float = -1.0,
        blink_min_consecutive_frames: int = -1,
        start_options_json: str = "",
    ) -> None:
        self._set_bot_action_result(
            "start_realtime_stream",
            {"ok": False, "error": "Realtime stream did not start."},
        )
        host = self.host_window_widget or self.window()
        resolved_camera_source = str(camera_source or "").strip()
        if not resolved_camera_source:
            host_video = str(getattr(host, "video_file", "") or "").strip()
            if host_video and Path(host_video).exists():
                resolved_camera_source = host_video
        manager = getattr(host, "realtime_manager", None)
        if manager is None:
            self._set_bot_action_result(
                "start_realtime_stream",
                {"ok": False, "error": "Realtime manager is unavailable."},
            )
            self.status_label.setText(
                "Bot action failed: realtime manager unavailable."
            )
            return
        try:
            start_options: Dict[str, Any] = {}
            if start_options_json:
                try:
                    raw_options = str(start_options_json or "").strip()
                    if len(raw_options) > 4096:
                        raise ValueError("start options payload too large")
                    parsed_options = json.loads(raw_options)
                    if isinstance(parsed_options, dict):
                        start_options = dict(parsed_options)
                except Exception:
                    start_options = {}

            def _as_bool(value: Any, default: bool = False) -> bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    text = value.strip().lower()
                    if text in {"1", "true", "yes", "on"}:
                        return True
                    if text in {"0", "false", "no", "off"}:
                        return False
                return bool(default)

            def _as_float(
                value: Any,
                default: float,
                low: float | None = None,
                high: float | None = None,
            ) -> float:
                try:
                    parsed = float(value)
                except Exception:
                    parsed = float(default)
                if low is not None:
                    parsed = max(float(low), parsed)
                if high is not None:
                    parsed = min(float(high), parsed)
                return parsed

            safe_bot_report_enabled = _as_bool(
                start_options.get("bot_report_enabled", False), False
            )
            safe_bot_report_interval_sec = _as_float(
                start_options.get("bot_report_interval_sec", 5.0),
                5.0,
                1.0,
                3600.0,
            )
            safe_bot_watch_labels_csv = str(
                start_options.get("bot_watch_labels_csv", "") or ""
            ).strip()
            if len(safe_bot_watch_labels_csv) > 512:
                safe_bot_watch_labels_csv = safe_bot_watch_labels_csv[:512]
            safe_bot_email_report = _as_bool(
                start_options.get("bot_email_report", False), False
            )
            safe_bot_email_to = str(start_options.get("bot_email_to", "") or "").strip()
            if len(safe_bot_email_to) > 256:
                safe_bot_email_to = safe_bot_email_to[:256]
            safe_save_detection_segments = _as_bool(
                start_options.get("save_detection_segments", False), False
            )
            safe_detection_segment_targets_csv = str(
                start_options.get("detection_segment_targets_csv", "animal,car,person")
                or "animal,car,person"
            ).strip()
            if len(safe_detection_segment_targets_csv) > 512:
                safe_detection_segment_targets_csv = safe_detection_segment_targets_csv[
                    :512
                ]
            safe_detection_segment_output_dir = str(
                start_options.get("detection_segment_output_dir", "") or ""
            ).strip()
            if len(safe_detection_segment_output_dir) > 2048:
                safe_detection_segment_output_dir = safe_detection_segment_output_dir[
                    :2048
                ]
            safe_detection_segment_prebuffer_sec = _as_float(
                start_options.get("detection_segment_prebuffer_sec", 2.0),
                2.0,
                0.0,
                30.0,
            )
            safe_detection_segment_postbuffer_sec = _as_float(
                start_options.get("detection_segment_postbuffer_sec", 3.0),
                3.0,
                0.0,
                30.0,
            )
            safe_detection_segment_min_duration_sec = _as_float(
                start_options.get("detection_segment_min_duration_sec", 1.0),
                1.0,
                0.0,
                120.0,
            )
            safe_detection_segment_max_duration_sec = _as_float(
                start_options.get("detection_segment_max_duration_sec", 120.0),
                120.0,
                1.0,
                3600.0,
            )
            if (
                safe_detection_segment_max_duration_sec
                <= safe_detection_segment_min_duration_sec
            ):
                safe_detection_segment_max_duration_sec = min(
                    3600.0, safe_detection_segment_min_duration_sec + 1.0
                )

            realtime_config, extras = build_realtime_launch_payload(
                camera_source=resolved_camera_source,
                model_name=model_name,
                target_behaviors_csv=target_behaviors_csv,
                confidence_threshold=confidence_threshold,
                viewer_type=viewer_type,
                enable_eye_control=False,
                enable_hand_control=False,
                classify_eye_blinks=bool(classify_eye_blinks),
                blink_ear_threshold=blink_ear_threshold,
                blink_min_consecutive_frames=blink_min_consecutive_frames,
                rtsp_transport=rtsp_transport,
                bot_report_enabled=safe_bot_report_enabled,
                bot_report_interval_sec=safe_bot_report_interval_sec,
                bot_watch_labels=safe_bot_watch_labels_csv,
                bot_email_report=safe_bot_email_report,
                bot_email_to=safe_bot_email_to,
                save_detection_segments=safe_save_detection_segments,
                detection_segment_targets_csv=safe_detection_segment_targets_csv,
                detection_segment_output_dir=safe_detection_segment_output_dir,
                detection_segment_prebuffer_sec=safe_detection_segment_prebuffer_sec,
                detection_segment_postbuffer_sec=safe_detection_segment_postbuffer_sec,
                detection_segment_min_duration_sec=safe_detection_segment_min_duration_sec,
                detection_segment_max_duration_sec=safe_detection_segment_max_duration_sec,
                suppress_control_dock=True,
            )
            model_weight = resolve_realtime_model_weight(model_name)
            camera_value = realtime_config.camera_index

            start_handler = getattr(manager, "_handle_realtime_start_request", None)
            if callable(start_handler):
                start_handler(realtime_config, extras)
            else:
                starter = getattr(manager, "start_realtime_inference", None)
                if not callable(starter):
                    raise RuntimeError("Realtime start API is unavailable.")
                starter(realtime_config, extras)

            self._set_bot_action_result(
                "start_realtime_stream",
                {
                    "ok": True,
                    "model_name": model_weight,
                    "camera_source": _safe_stream_source_for_bot(str(camera_value)),
                    "viewer_type": str(extras["viewer_type"]),
                    "rtsp_transport": str(rtsp_transport or "auto"),
                    "classify_eye_blinks": bool(classify_eye_blinks),
                    "bot_report_enabled": bool(extras.get("bot_report_enabled", False)),
                    "bot_report_interval_sec": float(
                        extras.get("bot_report_interval_sec", 5.0)
                    ),
                    "bot_watch_labels": list(extras.get("bot_watch_labels", [])),
                    "bot_email_report": bool(extras.get("bot_email_report", False)),
                    "bot_email_to": str(extras.get("bot_email_to", "")),
                    "save_detection_segments": bool(
                        extras.get("save_detection_segments", False)
                    ),
                    "detection_segment_targets": list(
                        realtime_config.detection_segment_targets or []
                    ),
                    "detection_segment_output_dir": str(
                        realtime_config.detection_segment_output_dir or ""
                    ),
                },
            )
            self.status_label.setText(
                f"Realtime started with {model_weight} on source {camera_value}."
            )
        except Exception as exc:
            self._set_bot_action_result(
                "start_realtime_stream",
                {"ok": False, "error": str(exc)},
            )
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot()
    def bot_get_realtime_status(self) -> None:
        host = self.host_window_widget or self.window()
        manager = getattr(host, "realtime_manager", None)
        if manager is None:
            self._set_bot_action_result(
                "get_realtime_status",
                {"ok": False, "error": "Realtime manager is unavailable."},
            )
            return
        payload = {
            "ok": True,
            "running": bool(getattr(manager, "realtime_running", False)),
            "camera_source": _safe_stream_source_for_bot(
                str(getattr(manager, "_last_realtime_camera_source", "") or "")
            ),
            "model_name": str(getattr(manager, "_last_realtime_model_name", "") or ""),
            "viewer_type": str(
                getattr(manager, "_last_realtime_viewer_type", "") or ""
            ),
            "rtsp_transport": str(
                getattr(manager, "_last_realtime_rtsp_transport", "") or "auto"
            ),
            "subscriber_address": _safe_stream_source_for_bot(
                str(getattr(manager, "_realtime_connect_address", "") or "")
            ),
            "detections_log_path": str(getattr(manager, "realtime_log_path", "") or ""),
            "bot_event_log_path": str(
                getattr(manager, "_bot_event_log_path", "") or ""
            ),
            "status_text": str(
                getattr(manager.realtime_control_widget, "status_label", None).text()
                if getattr(manager, "realtime_control_widget", None) is not None
                and getattr(manager.realtime_control_widget, "status_label", None)
                is not None
                else ""
            ),
        }
        self._set_bot_action_result("get_realtime_status", payload)

    @QtCore.Slot()
    def bot_list_realtime_models(self) -> None:
        models: list[dict[str, str]] = []
        host = self.host_window_widget or self.window()
        settings = getattr(host, "settings", None)
        config = getattr(host, "_config", None)
        for item in get_runtime_model_registry(config=config, settings=settings):
            unavailable_reason = get_model_unavailable_reason(item)
            models.append(
                {
                    "id": str(item.identifier),
                    "display_name": str(item.display_name),
                    "weight_file": str(item.weight_file),
                    "available": str(unavailable_reason is None).lower(),
                    "unavailable_reason": str(unavailable_reason or ""),
                }
            )
        self._set_bot_action_result(
            "list_realtime_models",
            {"ok": True, "count": len(models), "models": models},
        )

    @QtCore.Slot()
    def bot_list_realtime_logs(self) -> None:
        host = self.host_window_widget or self.window()
        manager = getattr(host, "realtime_manager", None)
        if manager is None:
            self._set_bot_action_result(
                "list_realtime_logs",
                {"ok": False, "error": "Realtime manager is unavailable."},
            )
            return
        detections = str(getattr(manager, "realtime_log_path", "") or "")
        bot_events = str(getattr(manager, "_bot_event_log_path", "") or "")
        self._set_bot_action_result(
            "list_realtime_logs",
            {
                "ok": True,
                "detections_log_path": detections,
                "bot_event_log_path": bot_events,
                "available": bool(detections or bot_events),
            },
        )

    @QtCore.Slot()
    def bot_list_logs(self) -> None:
        entries = []
        for name, path in _log_targets_for_bot().items():
            resolved = Path(path).expanduser().resolve()
            exists = resolved.exists()
            entries.append(
                {
                    "target": name,
                    "path": str(resolved),
                    "exists": bool(exists),
                    "is_dir": bool(exists and resolved.is_dir()),
                }
            )
        self._set_bot_action_result(
            "list_logs",
            {"ok": True, "count": len(entries), "logs": entries},
        )

    @QtCore.Slot(str)
    def bot_open_log_folder(self, target: str) -> None:
        key = str(target or "").strip().lower()
        path = _log_targets_for_bot().get(key)
        if path is None:
            self._set_bot_action_result(
                "open_log_folder",
                {"ok": False, "error": f"Unsupported log target: {target}"},
            )
            return
        folder = Path(path).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        ok = bool(
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
        )
        self._set_bot_action_result(
            "open_log_folder",
            {"ok": ok, "target": key, "path": str(folder)},
        )

    @QtCore.Slot(str)
    def bot_remove_log_folder(self, target: str) -> None:
        key = str(target or "").strip().lower()
        path = _log_targets_for_bot().get(key)
        if path is None:
            self._set_bot_action_result(
                "remove_log_folder",
                {"ok": False, "error": f"Unsupported log target: {target}"},
            )
            return
        folder = Path(path).expanduser().resolve()
        if not folder.exists():
            self._set_bot_action_result(
                "remove_log_folder",
                {"ok": True, "target": key, "path": str(folder), "removed": False},
            )
            return
        try:
            shutil.rmtree(folder)
        except Exception as exc:
            self._set_bot_action_result(
                "remove_log_folder",
                {
                    "ok": False,
                    "target": key,
                    "path": str(folder),
                    "error": str(exc),
                },
            )
            return
        self._set_bot_action_result(
            "remove_log_folder",
            {"ok": True, "target": key, "path": str(folder), "removed": True},
        )

    @QtCore.Slot(str, str, int, bool, str, bool)
    def bot_list_log_files(
        self,
        target: str,
        pattern: str,
        limit: int,
        recursive: bool,
        sort_by: str,
        descending: bool,
    ) -> None:
        key = str(target or "").strip().lower()
        roots = _log_targets_for_bot()
        root = roots.get(key)
        if root is None:
            self._set_bot_action_result(
                "list_log_files",
                {"ok": False, "error": f"Unsupported log target: {target}"},
            )
            return
        path = Path(root).expanduser().resolve()
        if not path.exists():
            self._set_bot_action_result(
                "list_log_files",
                {"ok": True, "target": key, "root": str(path), "count": 0, "files": []},
            )
            return
        safe_pattern = str(pattern or "*").strip() or "*"
        max_items = max(1, min(5000, int(limit or 200)))
        do_recursive = bool(recursive)
        sort_key = str(sort_by or "name").strip().lower()
        if sort_key not in {"name", "mtime", "size"}:
            sort_key = "name"
        reverse_order = bool(descending)
        max_scan = 50000
        files = []
        scanned = 0
        truncated_scan = False
        if do_recursive:
            candidate_iter = (
                Path(dirpath) / filename
                for dirpath, _, filenames in os.walk(path)
                for filename in filenames
            )
        else:
            candidate_iter = (
                Path(entry.path) for entry in os.scandir(path) if entry.is_file()
            )
        for fp in candidate_iter:
            scanned += 1
            if scanned > max_scan:
                truncated_scan = True
                break
            try:
                rel = fp.resolve().relative_to(path)
                rel_text = str(rel)
                if not (
                    fnmatch.fnmatch(fp.name, safe_pattern)
                    or fnmatch.fnmatch(rel_text, safe_pattern)
                ):
                    continue
                st = fp.stat()
                files.append(
                    {
                        "path": str(fp.resolve()),
                        "relative_path": rel_text,
                        "size": int(st.st_size),
                        "mtime_ns": int(st.st_mtime_ns),
                    }
                )
            except Exception:
                continue
        if sort_key == "mtime":
            files.sort(
                key=lambda item: int(item.get("mtime_ns", 0)), reverse=reverse_order
            )
        elif sort_key == "size":
            files.sort(key=lambda item: int(item.get("size", 0)), reverse=reverse_order)
        else:
            files.sort(
                key=lambda item: str(item.get("relative_path", "")).lower(),
                reverse=reverse_order,
            )
        limited = len(files) > max_items
        if limited:
            files = files[:max_items]
        self._set_bot_action_result(
            "list_log_files",
            {
                "ok": True,
                "target": key,
                "root": str(path),
                "count": len(files),
                "files": files,
                "pattern": safe_pattern,
                "limit": max_items,
                "recursive": do_recursive,
                "sort_by": sort_key,
                "descending": reverse_order,
                "scanned_files": scanned,
                "truncated_scan": truncated_scan,
                "limited": limited,
            },
        )

    @QtCore.Slot(str, int, int)
    def bot_read_log_file(self, path: str, max_chars: int, tail_lines: int) -> None:
        raw_path = str(path or "").strip()
        if not raw_path:
            self._set_bot_action_result(
                "read_log_file", {"ok": False, "error": "path is required"}
            )
            return
        target_path = Path(raw_path).expanduser().resolve()
        roots = [
            Path(p).expanduser().resolve() for p in _log_targets_for_bot().values()
        ]
        if not any(_is_path_within_root(target_path, root) for root in roots):
            self._set_bot_action_result(
                "read_log_file",
                {"ok": False, "error": "path must be under Annolid log roots"},
            )
            return
        if not target_path.exists() or not target_path.is_file():
            self._set_bot_action_result(
                "read_log_file",
                {"ok": False, "error": f"log file not found: {target_path}"},
            )
            return
        cap_chars = max(200, min(200000, int(max_chars or 12000)))
        cap_lines = max(1, min(100000, int(tail_lines or 200)))
        if _is_probably_binary_file(target_path):
            self._set_bot_action_result(
                "read_log_file",
                {
                    "ok": False,
                    "error": "file appears to be binary",
                    "path": str(target_path),
                },
            )
            return
        try:
            tail_text, truncated = _tail_text_from_file(
                target_path,
                tail_lines=cap_lines,
                max_chars=cap_chars,
            )
        except Exception as exc:
            self._set_bot_action_result(
                "read_log_file",
                {"ok": False, "error": str(exc), "path": str(target_path)},
            )
            return
        returned_line_count = tail_text.count("\n") + (1 if tail_text else 0)
        self._set_bot_action_result(
            "read_log_file",
            {
                "ok": True,
                "path": str(target_path),
                "tail_lines": cap_lines,
                "max_chars": cap_chars,
                "line_count": returned_line_count,
                "truncated": bool(truncated),
                "file_size": int(target_path.stat().st_size),
                "content": tail_text,
            },
        )

    @QtCore.Slot(str, str, str, bool, bool, int, int)
    def bot_search_logs(
        self,
        query: str,
        target: str,
        pattern: str,
        case_sensitive: bool,
        use_regex: bool,
        max_matches: int,
        max_files: int,
    ) -> None:
        term = str(query or "").strip()
        if not term:
            self._set_bot_action_result(
                "search_logs", {"ok": False, "error": "query is required"}
            )
            return
        key = str(target or "logs").strip().lower() or "logs"
        roots = _log_targets_for_bot()
        root = roots.get(key)
        if root is None:
            self._set_bot_action_result(
                "search_logs",
                {"ok": False, "error": f"Unsupported log target: {target}"},
            )
            return
        base = Path(root).expanduser().resolve()
        if not base.exists():
            self._set_bot_action_result(
                "search_logs",
                {
                    "ok": True,
                    "target": key,
                    "root": str(base),
                    "matches": [],
                    "match_count": 0,
                    "scanned_files": 0,
                },
            )
            return
        safe_pattern = str(pattern or "*").strip() or "*"
        use_case_sensitive = bool(case_sensitive)
        use_re = bool(use_regex)
        max_m = max(1, min(5000, int(max_matches or 100)))
        max_f = max(1, min(2000, int(max_files or 50)))
        matcher = None
        if use_re:
            try:
                flags = 0 if use_case_sensitive else re.IGNORECASE
                matcher = re.compile(term, flags)
            except re.error as exc:
                self._set_bot_action_result(
                    "search_logs", {"ok": False, "error": f"invalid regex: {exc}"}
                )
                return
        matches = []
        scanned = 0
        max_bytes_per_file = 2 * 1024 * 1024
        for fp in sorted(base.rglob("*")):
            if not fp.is_file():
                continue
            rel = str(fp.resolve().relative_to(base))
            if not (
                fnmatch.fnmatch(fp.name, safe_pattern)
                or fnmatch.fnmatch(rel, safe_pattern)
            ):
                continue
            if _is_probably_binary_file(fp):
                continue
            scanned += 1
            if scanned > max_f:
                break
            try:
                consumed = 0
                with fp.open("r", encoding="utf-8", errors="replace") as handle:
                    for idx, line in enumerate(handle, start=1):
                        consumed += len(line.encode("utf-8", errors="ignore"))
                        if consumed > max_bytes_per_file:
                            break
                        candidate = line.rstrip("\n")
                        if matcher is not None:
                            found = matcher.search(candidate) is not None
                        elif use_case_sensitive:
                            found = term in candidate
                        else:
                            found = term.lower() in candidate.lower()
                        if not found:
                            continue
                        matches.append(
                            {
                                "path": str(fp.resolve()),
                                "relative_path": rel,
                                "line": int(idx),
                                "text": str(candidate[:500]),
                            }
                        )
                        if len(matches) >= max_m:
                            break
            except Exception:
                continue
            if len(matches) >= max_m:
                break
        self._set_bot_action_result(
            "search_logs",
            {
                "ok": True,
                "target": key,
                "root": str(base),
                "query": term,
                "pattern": safe_pattern,
                "case_sensitive": use_case_sensitive,
                "use_regex": use_re,
                "match_count": len(matches),
                "scanned_files": min(scanned, max_f),
                "matches": matches,
                "max_matches": max_m,
                "max_files": max_f,
            },
        )

    @QtCore.Slot()
    def bot_stop_realtime_stream(self) -> None:
        self._set_bot_action_result(
            "stop_realtime_stream",
            {"ok": False, "error": "Realtime stream did not stop."},
        )
        host = self.host_window_widget or self.window()
        stopper = getattr(host, "stop_realtime_inference", None)
        if not callable(stopper):
            self._set_bot_action_result(
                "stop_realtime_stream",
                {"ok": False, "error": "Realtime stop API is unavailable."},
            )
            self.status_label.setText("Bot action failed: realtime stop unavailable.")
            return
        try:
            stopper()
            self._set_bot_action_result("stop_realtime_stream", {"ok": True})
            self.status_label.setText("Realtime stream stopped.")
        except Exception as exc:
            self._set_bot_action_result(
                "stop_realtime_stream",
                {"ok": False, "error": str(exc)},
            )
            self.status_label.setText(f"Bot action failed: {exc}")
