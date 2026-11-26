"""Streaming turn lifecycle, snapshot attachment, bus draining, and audio helpers for AIChatWidget."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import time
from typing import Any, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.ai_chat_backend import StreamingChatTask
from annolid.gui.widgets.ai_chat_slash_mixin import _extract_slash_selection_state
from annolid.services.chat_backend_support import (
    ERROR_TYPE_CANCELLED,
    TURN_STATUS_CANCELLED,
)
from annolid.services.chat_bus import (
    InboundMessage,
    OutboundMessage,
    decode_outbound_chat_event,
)


def _widget_module_attr(name: str, default: Any) -> Any:
    mod = sys.modules.get("annolid.gui.widgets.ai_chat_widget")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


class AIChatStreamingMixin:
    """Mixin handling streaming turns, snapshots, message bus queues, and audio controls."""

    def _start_typing_indicator(self) -> None:
        self._typing_tick = 0
        self._on_typing_timer_tick()
        self._typing_timer.start()

    def _stop_typing_indicator(self) -> None:
        if self._typing_timer.isActive():
            self._typing_timer.stop()

    def _on_typing_timer_tick(self) -> None:
        if not self.is_streaming_chat:
            self._stop_typing_indicator()
            return
        self._typing_tick += 1
        self._render_progress_in_bubble()

    def _render_progress_in_bubble(self) -> None:
        if not self.is_streaming_chat:
            return
        if self._current_response_bubble is None:
            return
        lines: List[str] = []
        if self.enable_progress_stream and self._progress_lines:
            lines.extend(self._progress_lines[:32])
        if not lines:
            dots = "." * ((self._typing_tick % 3) + 1)
            lines = [f"Thinking{dots}"]
        bubble = self._current_response_bubble
        bubble.set_thinking_text(
            "\n".join(lines),
            in_progress=True,
        )
        QtCore.QTimer.singleShot(0, lambda b=bubble: self._reflow_single_bubble(b))
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom_if_following)

    def _append_thinking_progress_line(
        self,
        line: str,
        *,
        boilerplate: set[str],
    ) -> None:
        value = str(line or "").strip()
        if not value:
            return
        low = value.lower()
        has_rich_progress = any(
            str(item or "").strip().lower() not in boilerplate
            for item in self._progress_lines
        )
        if low in boilerplate and has_rich_progress:
            return
        if low not in boilerplate and self._progress_lines:
            self._progress_lines = [
                item
                for item in self._progress_lines
                if str(item or "").strip().lower() not in boilerplate
            ]

        if self._progress_lines:
            top = str(self._progress_lines[0] or "").strip()
            if value.startswith(top) and len(value) > len(top):
                self._progress_lines[0] = value
                return
            if top.startswith(value):
                return
        if value in self._progress_lines:
            self._progress_lines.remove(value)
        self._progress_lines.insert(0, value)
        if len(self._progress_lines) > 256:
            self._progress_lines = self._progress_lines[:256]

    def set_image_path(self, image_path: str, *, explicit: bool = False) -> None:
        if image_path and image_path not in self._snapshot_paths:
            self._cleanup_snapshots()
        self.image_path = image_path
        if explicit:
            self._pending_explicit_image_path = str(image_path or "").strip()
        self._update_shared_image_label(image_path)

    def set_next_chat_mode(self, mode: str) -> None:
        value = str(mode or "default").strip().lower()
        self._next_chat_mode = value or "default"

    def _consume_next_chat_mode(self) -> str:
        value = str(getattr(self, "_next_chat_mode", "default") or "default").strip()
        self._next_chat_mode = "default"
        return value.lower() or "default"

    def register_managed_temp_image(self, image_path: str) -> None:
        """Track externally-created temp images for later cleanup."""
        path = str(image_path or "").strip()
        if path and path not in self._snapshot_paths:
            self._snapshot_paths.append(path)

    def _cleanup_snapshots(self) -> None:
        stale = list(self._snapshot_paths)
        self._snapshot_paths.clear()
        for path in stale:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def _snapshot_widget_to_tempfile(
        self, widget: Optional[QtWidgets.QWidget], prefix: str
    ) -> Optional[str]:
        if widget is None:
            return None
        try:
            pixmap = widget.grab()
        except Exception:
            return None
        if pixmap is None or pixmap.isNull():
            return None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".png")
            os.close(fd)
            if not pixmap.save(tmp_path, "PNG"):
                os.remove(tmp_path)
                return None
            self._snapshot_paths.append(tmp_path)
            return tmp_path
        except Exception:
            return None

    def _snapshot_canvas_to_tempfile(self) -> Optional[str]:
        return self._snapshot_widget_to_tempfile(self.canvas_widget, "annolid_canvas_")

    def _snapshot_window_to_tempfile(self) -> Optional[str]:
        host = self.host_window_widget or self.window()
        return self._snapshot_widget_to_tempfile(host, "annolid_window_")

    def _clear_attached_image(self) -> None:
        self._pending_explicit_image_path = ""
        self.image_path = ""
        self._cleanup_snapshots()
        self._update_shared_image_label("")
        self.status_label.setText("Removed attached image.")

    def _update_shared_image_label(self, image_path: Optional[str]) -> None:
        has_valid_image = bool(image_path and os.path.exists(image_path))
        if has_valid_image:
            basename = os.path.basename(str(image_path))
            self.shared_image_label.setText(f"Shared image: {basename}")
            if hasattr(self, "attachment_chip_label"):
                self.attachment_chip_label.setText(f"Attached image: {basename}")
                self.attachment_chip_label.setToolTip(str(image_path))
            if hasattr(self, "attachment_chip_widget"):
                self.attachment_chip_widget.setVisible(
                    bool(self._pending_explicit_image_path)
                )
        else:
            self.shared_image_label.setText("Shared image: none")
            if hasattr(self, "attachment_chip_label"):
                self.attachment_chip_label.setText("")
                self.attachment_chip_label.setToolTip("")
            if hasattr(self, "attachment_chip_widget"):
                self.attachment_chip_widget.setVisible(False)

    def _share_canvas_now(self) -> None:
        image_path = self._snapshot_canvas_to_tempfile()
        if image_path:
            self.set_image_path(image_path, explicit=True)
            self._add_bubble(
                "Annolid Bot",
                f"Canvas snapshot attached: {os.path.basename(image_path)}",
                is_user=False,
            )

    def _share_window_now(self) -> None:
        image_path = self._snapshot_window_to_tempfile()
        if image_path:
            self.set_image_path(image_path, explicit=True)
            self._add_bubble(
                "Annolid Bot",
                f"Window snapshot attached: {os.path.basename(image_path)}",
                is_user=False,
            )

    def _prepare_chat_image(self, *, chat_mode: str = "default") -> Optional[str]:
        explicit_path = str(
            getattr(self, "_pending_explicit_image_path", "") or ""
        ).strip()
        self._pending_explicit_image_path = ""
        if explicit_path and os.path.exists(explicit_path):
            self.image_path = explicit_path
            self._update_shared_image_label(explicit_path)
            return explicit_path
        if str(chat_mode or "").strip().lower() == "vision_describe":
            current = str(getattr(self, "image_path", "") or "").strip()
            if current and os.path.exists(current):
                return current
        use_window = self.attach_window_checkbox.isChecked()
        use_canvas = self.attach_canvas_checkbox.isChecked()
        image_path = None
        if use_window:
            image_path = self._snapshot_window_to_tempfile()
        elif use_canvas:
            image_path = self._snapshot_canvas_to_tempfile()
        if image_path:
            self.set_image_path(image_path)
        return image_path

    def _snapshot_host_track_fallback_model(self) -> str:
        host = getattr(self, "host_window_widget", None)
        if host is None:
            with contextlib.suppress(Exception):
                host = self.window()
        combo = getattr(host, "_selectAiModelComboBox", None)
        if combo is None:
            return ""
        try:
            current = str(combo.currentText() or "").strip()
            if current and current.lower() != "sam3":
                return current
            for index in range(int(combo.count())):
                item = str(combo.itemText(index) or "").strip()
                if item and item.lower() != "sam3":
                    return item
        except Exception:
            return ""
        return ""

    def chat_with_model(self) -> None:
        self._refresh_runtime_llm_settings()
        raw_prompt = self.prompt_text_edit.toPlainText().strip()
        if not raw_prompt:
            return
        extract_slash = _widget_module_attr(
            "_extract_slash_selection_state", _extract_slash_selection_state
        )
        slash_state = extract_slash(raw_prompt)
        clean_prompt = str(slash_state.get("clean_prompt") or "").strip()
        selected_skill_names = list(slash_state.get("selected_skill_names") or [])
        selected_tool_names = list(slash_state.get("selected_tool_names") or [])
        slash_commands = list(slash_state.get("slash_commands") or [])
        if bool(slash_state.get("open_capabilities")) and not clean_prompt:
            self._open_agent_capabilities_dialog()
            self.status_label.setText("Opened agent capabilities.")
            return
        if not clean_prompt:
            if selected_skill_names or selected_tool_names:
                self.status_label.setText(
                    "Selected skills/tools applied. Add a prompt to send."
                )
            return
        if not self._ensure_provider_ready():
            return

        if not self._prompt_history or self._prompt_history[-1] != raw_prompt:
            self._prompt_history.append(raw_prompt)
            if len(self._prompt_history) > 100:
                self._prompt_history = self._prompt_history[-100:]
        self._prompt_history_index = -1
        self._prompt_history_draft = ""
        self._auto_follow_scroll = True

        self._add_bubble("You", raw_prompt, is_user=True)
        self.prompt_text_edit.clear()

        chat_mode = self._consume_next_chat_mode()
        chat_image_path = self._prepare_chat_image(chat_mode=chat_mode)
        if (
            hasattr(self, "attachment_chip_widget")
            and not self._pending_explicit_image_path
        ):
            self.attachment_chip_widget.setVisible(False)
        ui_prepared = False
        if not self._chat_task_active and not self.is_streaming_chat:
            self._prepare_streaming_turn_ui(raw_prompt)
            ui_prepared = True
        else:
            pending = int(self._chat_message_bus.inbound.qsize()) + 1
            self.status_label.setText(f"Queued message ({pending} pending)")

        inbound_cls = _widget_module_attr("InboundMessage", InboundMessage)
        inbound_metadata = {
            "provider": self.selected_provider,
            "model": self.selected_model,
            "session_id": self.session_id,
            "chat_mode": chat_mode,
            "show_tool_trace": bool(self.tool_trace_checkbox.isChecked()),
            "enable_web_tools": bool(self.allow_web_tools_checkbox.isChecked()),
            "settings": dict(self.llm_settings or {}),
            "image_path": str(chat_image_path or self.image_path or ""),
            "host_track_fallback_model": self._snapshot_host_track_fallback_model(),
            "ui_prepared": bool(ui_prepared),
            "raw_prompt": raw_prompt,
            "selected_skill_names": selected_skill_names,
            "selected_tool_names": selected_tool_names,
            "slash_commands": slash_commands,
        }
        inbound = inbound_cls(
            channel="gui",
            sender_id="gui_user",
            chat_id=self.session_id,
            content=clean_prompt,
            media=(
                [str(chat_image_path or self.image_path or "")]
                if str(chat_image_path or self.image_path or "").strip()
                else []
            ),
            metadata=inbound_metadata,
        )
        self._chat_message_bus.inbound.put_nowait(inbound)
        if not self._chat_inbound_bus_timer.isActive():
            self._chat_inbound_bus_timer.start()

    def _prepare_streaming_turn_ui(self, prompt: str) -> None:
        self._last_user_prompt = str(prompt or "").strip()
        self._latest_progress_text = ""
        self._has_streamed_response_chunk = False
        self._progress_lines = []
        assistant_name = self._assistant_display_name()
        self.status_label.setText("")
        self._current_response_bubble = self._add_bubble(
            assistant_name,
            "",
            is_user=False,
            allow_regenerate=True,
            allow_stop=True,
        )
        self._current_response_bubble.set_thinking_text("Thinking.", in_progress=True)
        self.is_streaming_chat = True
        self.send_button.setEnabled(bool(self.prompt_text_edit.toPlainText().strip()))
        self.send_button.setToolTip("Queue follow-up message (Ctrl+Enter)")
        if hasattr(self, "stop_button"):
            self.stop_button.setVisible(True)
        self._start_typing_indicator()

    def _drain_inbound_bus_messages(self) -> None:
        if self._chat_task_active:
            return
        try:
            inbound = self._chat_message_bus.inbound.get_nowait()
        except asyncio.QueueEmpty:
            self._chat_inbound_bus_timer.stop()
            return
        inbound_cls = _widget_module_attr("InboundMessage", InboundMessage)
        task_cls = _widget_module_attr("StreamingChatTask", StreamingChatTask)
        inbound_meta = dict(getattr(inbound, "metadata", {}) or {})
        if not bool(inbound_meta.get("ui_prepared", False)):
            self._prepare_streaming_turn_ui(str(getattr(inbound, "content", "") or ""))
            inbound_meta["ui_prepared"] = True
            inbound = inbound_cls(
                channel=str(getattr(inbound, "channel", "gui") or "gui"),
                sender_id=str(getattr(inbound, "sender_id", "gui_user") or "gui_user"),
                chat_id=str(
                    getattr(inbound, "chat_id", self.session_id) or self.session_id
                ),
                content=str(getattr(inbound, "content", "") or ""),
                media=list(getattr(inbound, "media", []) or []),
                metadata=inbound_meta,
            )
        self._chat_task_active = True
        task = task_cls(
            widget=self,
            prompt=str(getattr(inbound, "content", "") or ""),
            session_store=self._session_store,
            inbound=inbound,
        )
        self._active_turn_id = str(getattr(task, "turn_id", "") or "").strip()
        self.thread_pool.start(task)
        self._active_chat_task = task
        if self._chat_message_bus.inbound.qsize() <= 0:
            self._chat_inbound_bus_timer.stop()

    @QtCore.Slot(str)
    def enqueue_outbound_bus_message(self, payload_text: str) -> None:
        outbound_cls = _widget_module_attr("OutboundMessage", OutboundMessage)
        self._chat_message_bus.outbound.put_nowait(
            outbound_cls(
                channel="gui-ui",
                chat_id=self.session_id,
                content=str(payload_text or ""),
                metadata={"source": "ai_chat_backend"},
            )
        )
        if not self._chat_bus_timer.isActive():
            self._chat_bus_timer.start()

    def _drain_outbound_bus_messages(self) -> None:
        processed = 0
        max_events_per_tick = 64
        while processed < max_events_per_tick:
            try:
                outbound = self._chat_message_bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                self._chat_bus_timer.stop()
                return
            processed += 1
            self.consume_outbound_chat_event(str(outbound.content or ""))
        if self._chat_message_bus.outbound.qsize() <= 0:
            self._chat_bus_timer.stop()

    @QtCore.Slot(str)
    def consume_outbound_chat_event(self, payload_text: str) -> None:
        decoder = _widget_module_attr(
            "decode_outbound_chat_event", decode_outbound_chat_event
        )
        event = decoder(payload_text)
        if event is None:
            return
        event_turn_id = str(getattr(event, "turn_id", "") or "").strip()
        active_turn_id = str(getattr(self, "_active_turn_id", "") or "").strip()
        if event_turn_id and active_turn_id and event_turn_id != active_turn_id:
            return
        if event_turn_id and not active_turn_id and not self._chat_task_active:
            return
        dedupe_key = str(getattr(event, "idempotency_key", "") or "").strip()
        if dedupe_key:
            if dedupe_key in self._seen_event_keys:
                return
            self._seen_event_keys[dedupe_key] = time.monotonic()
            self._seen_event_keys.move_to_end(dedupe_key)
            while len(self._seen_event_keys) > int(self._seen_event_key_limit):
                self._seen_event_keys.popitem(last=False)
        if event.kind == "chunk":
            self.stream_chat_chunk(event.text)
            return
        if event.kind == "progress":
            self.stream_chat_progress(event.text)
            return
        error_type = str(getattr(event, "error_type", "") or "").strip().lower()
        turn_status = str(getattr(event, "turn_status", "") or "").strip().lower()
        self.update_chat_response(
            event.text,
            bool(event.is_error),
            error_type=error_type,
            turn_status=turn_status,
        )

    @QtCore.Slot(str)
    def stream_chat_chunk(self, chunk: str) -> None:
        if self._current_response_bubble is None:
            return
        if not self._has_streamed_response_chunk:
            self._has_streamed_response_chunk = True
            self._current_response_bubble.set_text("")
        bubble = self._current_response_bubble
        bubble.append_text(chunk)
        QtCore.QTimer.singleShot(0, lambda b=bubble: self._reflow_single_bubble(b))
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom_if_following)

    @QtCore.Slot(str)
    def stream_chat_progress(self, update: str) -> None:
        text = str(update or "").strip()
        if not text:
            return
        self._latest_progress_text = text
        boilerplate = {
            "starting agent loop",
            "loading tools and context",
            "prepared system prompt",
            "received model response",
        }
        parts = [part.strip() for part in text.splitlines() if part.strip()]
        if not parts:
            return
        for part in parts:
            self._append_thinking_progress_line(part, boilerplate=boilerplate)
        self._render_progress_in_bubble()

    @QtCore.Slot(str, bool)
    def update_chat_response(
        self,
        message: str,
        is_error: bool,
        *,
        error_type: str = "",
        turn_status: str = "",
    ) -> None:
        text = str(message or "")
        now = time.monotonic()
        if (
            self._current_response_bubble is None
            and text.strip()
            and text.strip() == self._last_final_message_text
            and bool(is_error) == self._last_final_message_is_error
            and (now - float(self._last_final_message_ts)) <= 2.0
        ):
            return

        if self._current_response_bubble is None:
            bubble = self._add_bubble(
                "Assistant",
                text,
                is_user=False,
            )
            self._current_response_bubble = bubble

        self._current_response_bubble.finish_thinking(collapse=True)
        if is_error:
            if self._has_streamed_response_chunk:
                current = self._current_response_bubble.text()
                self._current_response_bubble.set_text((current + "\n" + text).strip())
            else:
                self._current_response_bubble.set_text(text or "Error")
            if str(error_type or "").strip():
                self.status_label.setText(f"Error ({str(error_type).strip()})")
            else:
                self.status_label.setText("Error")
        elif text:
            self._current_response_bubble.set_text(text)
            status = str(turn_status or "").strip().lower()
            if status == TURN_STATUS_CANCELLED:
                self.status_label.setText("Cancelled")
            elif status:
                self.status_label.setText(status.capitalize())
            else:
                self.status_label.setText("Done")
        else:
            self.status_label.setText("Done")

        self.is_streaming_chat = False
        self._stop_typing_indicator()
        self._latest_progress_text = ""
        self._has_streamed_response_chunk = False
        self._chat_task_active = False
        if hasattr(self, "stop_button"):
            self.stop_button.setVisible(False)
        self._on_prompt_text_changed()
        self._active_chat_task = None
        self._active_turn_id = ""
        finished_bubble = self._current_response_bubble
        if finished_bubble is not None:
            finished_bubble.set_stop_visible(False)
        self._current_response_bubble = None
        self._current_progress_bubble = None
        self._progress_lines = []
        self._last_final_message_text = text.strip()
        self._last_final_message_is_error = bool(is_error)
        self._last_final_message_ts = now
        QtCore.QTimer.singleShot(
            0, lambda b=finished_bubble: self._reflow_single_bubble(b)
        )
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom_if_following)
        if self._chat_message_bus.inbound.qsize() > 0:
            if not self._chat_inbound_bus_timer.isActive():
                self._chat_inbound_bus_timer.start()
            QtCore.QTimer.singleShot(0, self._drain_inbound_bus_messages)

    def _stop_running_response(self) -> None:
        if not self._chat_task_active and not self.is_streaming_chat:
            return
        was_pre_drain = not self._chat_task_active and self.is_streaming_chat
        if was_pre_drain:
            # Drop the prepared inbound turn before the 40ms timer starts it.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._chat_message_bus.inbound.get_nowait()
        task = self._active_chat_task
        if task is not None and hasattr(task, "request_cancel"):
            with contextlib.suppress(Exception):
                task.request_cancel()
        self.status_label.setText("Stopping...")
        if self._current_response_bubble is not None:
            self._current_response_bubble.set_stop_visible(False)
        self.update_chat_response(
            "Stopped by user.",
            False,
            error_type=ERROR_TYPE_CANCELLED,
            turn_status=TURN_STATUS_CANCELLED,
        )

    def _last_assistant_text(self) -> str:
        for bubble in reversed(self._iter_chat_bubbles()):
            if bubble.property("role") == "assistant":
                text = bubble.text().strip()
                if text:
                    return text
        return ""

    def read_last_reply_async(self) -> None:
        if self._audio_controller is None:
            return
        self._audio_controller.read_last_reply_async()

    def speak_text_async(self, text: str) -> None:
        if self._audio_controller is None:
            return
        self._audio_controller.speak_text_async(text)

    def toggle_recording(self) -> None:
        if self._audio_controller is None:
            return
        self._audio_controller.toggle_recording()
