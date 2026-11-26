"""Composer options, attachments, drag-drop, prompt history, quick actions, and Zulip helpers for AIChatWidget."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, List

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.ai_chat_zulip import (
    _ZulipSendTask,
    build_zulip_draft_target,
    missing_zulip_config_fields,
)
from annolid.infrastructure.agent_config import (
    get_agent_config_path as get_config_path,
    load_agent_config as load_config,
)
from annolid.utils.llm_settings import save_llm_settings

_CHAT_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)


def _widget_module_attr(name: str, default: Any) -> Any:
    mod = sys.modules.get("annolid.gui.widgets.ai_chat_widget")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


class AIChatComposerMixin:
    """Mixin managing composer controls, attachments, quick prompts, and Zulip drafts."""

    def _build_zulip_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame(self)
        panel.setObjectName("zulipDraftPanel")
        panel.setVisible(False)

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)

        self.zulip_target_type_combo = QtWidgets.QComboBox(panel)
        self.zulip_target_type_combo.addItem("Stream", userData="stream")
        self.zulip_target_type_combo.addItem("Direct Message", userData="dm")
        top_row.addWidget(self.zulip_target_type_combo, 0)

        self.zulip_stream_edit = QtWidgets.QLineEdit(panel)
        self.zulip_stream_edit.setPlaceholderText("Stream")
        top_row.addWidget(self.zulip_stream_edit, 1)

        self.zulip_topic_edit = QtWidgets.QLineEdit(panel)
        self.zulip_topic_edit.setPlaceholderText("Topic")
        top_row.addWidget(self.zulip_topic_edit, 1)

        self.zulip_recipients_edit = QtWidgets.QLineEdit(panel)
        self.zulip_recipients_edit.setPlaceholderText(
            "email@example.com, teammate@example.com"
        )
        self.zulip_recipients_edit.setVisible(False)
        top_row.addWidget(self.zulip_recipients_edit, 1)

        self.zulip_defaults_button = QtWidgets.QPushButton("Defaults", panel)
        self.zulip_defaults_button.setObjectName("quickActionButton")
        top_row.addWidget(self.zulip_defaults_button, 0)

        layout.addLayout(top_row)

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(6)

        self.zulip_last_reply_button = QtWidgets.QPushButton("Use Last Reply", panel)
        self.zulip_last_reply_button.setObjectName("quickActionButton")
        self.zulip_last_reply_button.setToolTip(
            "Copy the latest Annolid Bot reply into the current draft."
        )
        bottom_row.addWidget(self.zulip_last_reply_button, 0)

        config_path_fn = _widget_module_attr("get_config_path", get_config_path)
        self.zulip_info_label = QtWidgets.QLabel(
            f"Uses the current draft in the prompt box. Configure Zulip in {config_path_fn()}",
            panel,
        )
        self.zulip_info_label.setObjectName("chatStatusLabel")
        self.zulip_info_label.setWordWrap(True)
        bottom_row.addWidget(self.zulip_info_label, 1)

        self.zulip_send_button = QtWidgets.QPushButton("Send to Zulip", panel)
        self.zulip_send_button.setObjectName("sendButton")
        bottom_row.addWidget(self.zulip_send_button, 0)

        layout.addLayout(bottom_row)
        return panel

    def _create_hidden_toggles(self, layout: QtWidgets.QLayout) -> None:
        self.attach_canvas_checkbox = QtWidgets.QCheckBox("Attach canvas", self)
        self.attach_canvas_checkbox.setChecked(True)
        self.attach_canvas_checkbox.setVisible(False)

        self.attach_window_checkbox = QtWidgets.QCheckBox("Attach window", self)
        self.attach_window_checkbox.setVisible(False)

        self.tool_trace_checkbox = QtWidgets.QCheckBox("Trace", self)
        self.tool_trace_checkbox.setChecked(False)
        self.tool_trace_checkbox.setVisible(False)

        self.allow_web_tools_checkbox = QtWidgets.QCheckBox("Allow web", self)
        self.allow_web_tools_checkbox.setChecked(True)
        self.allow_web_tools_checkbox.setVisible(False)

        layout.addWidget(self.attach_canvas_checkbox)
        layout.addWidget(self.attach_window_checkbox)
        layout.addWidget(self.tool_trace_checkbox)
        layout.addWidget(self.allow_web_tools_checkbox)

        # Shared image label kept for backward compatibility; visible pill mirrors it.
        self.shared_image_label = QtWidgets.QLabel("", self)
        self.shared_image_label.setVisible(False)
        layout.addWidget(self.shared_image_label)

        # Expose context & tool toggles cleanly in the options popup menu.
        self.options_menu = QtWidgets.QMenu(self)
        self._bind_option_menu_action(
            "Auto-attach canvas snapshot", self.attach_canvas_checkbox
        )
        self._bind_option_menu_action(
            "Auto-attach full window snapshot", self.attach_window_checkbox
        )
        self.options_menu.addSeparator()
        self._bind_option_menu_action("Show tool call trace", self.tool_trace_checkbox)
        self._bind_option_menu_action(
            "Allow web & search tools", self.allow_web_tools_checkbox
        )
        self.options_button.setMenu(self.options_menu)

    def _bind_option_menu_action(
        self, title: str, checkbox: QtWidgets.QCheckBox
    ) -> QtWidgets.QAction:
        action = self.options_menu.addAction(title)
        action.setCheckable(True)
        action.setChecked(checkbox.isChecked())
        action.toggled.connect(
            lambda checked, cb=checkbox: (
                cb.setChecked(bool(checked))
                if cb.isChecked() != bool(checked)
                else None
            )
        )
        checkbox.toggled.connect(
            lambda checked, act=action: (
                act.setChecked(bool(checked))
                if act.isChecked() != bool(checked)
                else None
            )
        )
        return action

    def _attach_paths(self, paths: List[str]) -> None:
        image_suffixes = _widget_module_attr(
            "_CHAT_IMAGE_SUFFIXES", _CHAT_IMAGE_SUFFIXES
        )
        attached_names: List[str] = []
        prompt_notes: List[str] = []
        for raw in paths:
            path = str(raw or "").strip()
            if not path:
                continue
            suffix = Path(path).suffix.lower()
            name = os.path.basename(path) or path
            attached_names.append(name)
            if suffix in image_suffixes:
                self.set_image_path(path, explicit=True)
            else:
                prompt_notes.append(f"[Attached: {path}]")
        if prompt_notes:
            current = self.prompt_text_edit.toPlainText()
            prefix = f"{current}\n" if current else ""
            self.prompt_text_edit.setPlainText(prefix + "\n".join(prompt_notes))
        if attached_names:
            self.status_label.setText(f"Attached: {', '.join(attached_names)}")

    def _attach_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Attach File or Image",
            "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff);;Videos (*.mp4 *.avi *.mov *.mkv);;Documents (*.pdf *.csv *.json *.txt *.md)",
        )
        if path:
            self._attach_paths([path])

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            local_files = [
                u.toLocalFile()
                for u in mime.urls()
                if u.isLocalFile() and u.toLocalFile()
            ]
            if local_files:
                event.acceptProposedAction()
                return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            local_files = [
                u.toLocalFile()
                for u in mime.urls()
                if u.isLocalFile()
                and u.toLocalFile()
                and os.path.exists(u.toLocalFile())
            ]
            if local_files:
                self._attach_paths(local_files)
                self.prompt_text_edit.setFocus()
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    def _toggle_zulip_panel(self) -> None:
        visible = not bool(self.zulip_panel.isVisible())
        self.zulip_panel.setVisible(visible)
        if visible:
            self._load_zulip_defaults()
            if self.zulip_target_type_combo.currentData() == "dm":
                self.zulip_recipients_edit.setFocus()
            else:
                self.zulip_stream_edit.setFocus()
            self.status_label.setText("Zulip draft panel opened.")
        else:
            self.status_label.setText("Zulip draft panel hidden.")

    def _sync_zulip_target_field_visibility(self) -> None:
        target_type = str(self.zulip_target_type_combo.currentData() or "stream")
        is_dm = target_type == "dm"
        self.zulip_stream_edit.setVisible(not is_dm)
        self.zulip_topic_edit.setVisible(not is_dm)
        self.zulip_recipients_edit.setVisible(is_dm)

    def _load_zulip_defaults(self) -> None:
        load_cfg = _widget_module_attr("load_config", load_config)
        cfg = load_cfg().tools.zulip
        if self.zulip_target_type_combo.currentData() == "dm":
            return
        if not self.zulip_stream_edit.text().strip():
            self.zulip_stream_edit.setText(str(cfg.stream or "").strip())
        if not self.zulip_topic_edit.text().strip():
            self.zulip_topic_edit.setText(str(cfg.topic or "").strip())

    def _use_last_reply_for_zulip(self) -> None:
        text = self._last_assistant_text()
        if not text:
            self.status_label.setText("No assistant reply available for Zulip.")
            return
        self.prompt_text_edit.setPlainText(text)
        self.prompt_text_edit.setFocus()
        self.status_label.setText("Loaded last assistant reply into the draft.")

    def _send_current_draft_to_zulip(self) -> None:
        if self._zulip_send_active:
            self.status_label.setText("Zulip send already in progress.")
            return
        content = self.prompt_text_edit.toPlainText().strip()
        if not content:
            self.status_label.setText("Write a draft before sending to Zulip.")
            return

        load_cfg = _widget_module_attr("load_config", load_config)
        get_cfg_path = _widget_module_attr("get_config_path", get_config_path)
        missing_fn = _widget_module_attr(
            "missing_zulip_config_fields", missing_zulip_config_fields
        )
        build_target = _widget_module_attr(
            "build_zulip_draft_target", build_zulip_draft_target
        )
        zulip_task_cls = _widget_module_attr("_ZulipSendTask", _ZulipSendTask)

        zulip_cfg = load_cfg().tools.zulip.to_dict()
        missing = missing_fn(zulip_cfg)
        if missing:
            detail = ", ".join(missing)
            self.status_label.setText(f"Configure Zulip first: {detail}")
            QtWidgets.QMessageBox.warning(
                self,
                "Zulip not configured",
                f"Annolid Bot needs a valid Zulip configuration before it can send.\n\nMissing: {detail}\n\nConfig file: {get_cfg_path()}",
            )
            return

        try:
            target = build_target(
                str(self.zulip_target_type_combo.currentData() or "stream"),
                stream=self.zulip_stream_edit.text(),
                topic=self.zulip_topic_edit.text(),
                recipients=self.zulip_recipients_edit.text(),
                default_stream=str(zulip_cfg.get("stream") or ""),
                default_topic=str(zulip_cfg.get("topic") or ""),
            )
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        self._zulip_send_active = True
        self._zulip_target_summary = target.summary
        self.zulip_send_button.setEnabled(False)
        self.status_label.setText(f"Sending to {target.summary}...")
        self.thread_pool.start(
            zulip_task_cls(
                self,
                config=zulip_cfg,
                chat_id=target.chat_id,
                content=content,
                target_summary=target.summary,
            )
        )

    @QtCore.Slot(bool, str, str)
    def _finish_zulip_send(
        self, success: bool, detail: str, target_summary: str
    ) -> None:
        self._zulip_send_active = False
        self.zulip_send_button.setEnabled(True)
        summary = str(target_summary or self._zulip_target_summary or "Zulip").strip()
        if success:
            self.status_label.setText(f"Sent draft to {summary}.")
            self.prompt_text_edit.clear()
        else:
            self.status_label.setText(f"Zulip send failed: {detail}")

    def _apply_quick_action(self, text: str) -> None:
        prompt_text = str(text or "").strip()
        if not prompt_text:
            return
        existing_text = self.prompt_text_edit.toPlainText().rstrip()
        if existing_text:
            merged_text = f"{existing_text}\n{prompt_text}"
        else:
            merged_text = prompt_text
        self.prompt_text_edit.setPlainText(merged_text)
        cursor = self.prompt_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.prompt_text_edit.setTextCursor(cursor)
        self.prompt_text_edit.setFocus()
        self._on_prompt_text_changed()

    def _refresh_quick_action_buttons(self) -> None:
        while self.quick_actions_layout.count():
            item = self.quick_actions_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget in (
                    getattr(self, "add_quick_action_button", None),
                    getattr(self, "remove_quick_action_button", None),
                    getattr(self, "behavior_label_preset_button", None),
                ):
                    continue
                widget.deleteLater()
        self.quick_action_buttons = []
        if self._selected_quick_action_index is not None and (
            self._selected_quick_action_index < 0
            or self._selected_quick_action_index >= len(self._quick_actions)
        ):
            self._selected_quick_action_index = None
        for idx, (label, prompt) in enumerate(self._quick_actions):
            btn = QtWidgets.QPushButton(label, self)
            btn.setObjectName("quickActionButton")
            btn.setToolTip(prompt)
            btn.setCheckable(True)
            btn.setChecked(idx == self._selected_quick_action_index)
            btn.clicked.connect(
                lambda _checked=False, i=idx: self._on_quick_action_clicked(i)
            )
            self.quick_action_buttons.append(btn)
            self.quick_actions_layout.addWidget(btn, 0)

        self.quick_actions_layout.addStretch(1)
        self.quick_actions_layout.addWidget(self.behavior_label_preset_button)
        self.quick_actions_layout.addWidget(self.add_quick_action_button)
        self.quick_actions_layout.addWidget(self.remove_quick_action_button)
        self._set_remove_quick_action_enabled(
            self._selected_quick_action_index is not None
        )

    def _run_behavior_label_preset_one_second(self) -> None:
        host = self.host_window_widget or self.window()
        video_path = str(getattr(host, "video_file", "") or "").strip()
        if not video_path:
            self.status_label.setText(
                "Load a video first, then run 'Label 1s Behaviors'."
            )
            return

        labels = self._labels_from_schema_or_flags()
        if labels:
            draft = (
                f"label behavior in {video_path} "
                f"with labels {', '.join(labels)} from defined list every 1s"
            )
        else:
            draft = f"label behavior in {video_path} from defined list every 1s"
        self._apply_quick_action(draft)
        self.status_label.setText(
            "Drafted 1s behavior labeling command. Edit if needed, then press Send."
        )

    def _on_quick_action_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._quick_actions):
            return

        self._selected_quick_action_index = index

        for idx, btn in enumerate(self.quick_action_buttons):
            btn.setChecked(idx == index)

        self._set_remove_quick_action_enabled(True)
        self._persist_quick_actions_to_settings()
        self._apply_quick_action(self._quick_actions[index][1])

    def _add_quick_action(self) -> None:
        label, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add Quick Prompt",
            "Button label:",
        )
        label_text = str(label or "").strip()
        if not ok or not label_text:
            return
        prompt, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Add Quick Prompt",
            "Prompt text:",
            "",
        )
        prompt_text = str(prompt or "").strip()
        if not ok or not prompt_text:
            return
        self._quick_actions.append((label_text, prompt_text))
        self._selected_quick_action_index = len(self._quick_actions) - 1
        self._refresh_quick_action_buttons()
        self._persist_quick_actions_to_settings()

    def _remove_selected_quick_action(self) -> None:
        index = self._selected_quick_action_index
        if index is None or index < 0 or index >= len(self._quick_actions):
            QtWidgets.QMessageBox.information(
                self,
                "Quick Prompts",
                "Select a quick prompt button to remove.",
            )
            return
        self._quick_actions.pop(index)
        if self._quick_actions:
            self._selected_quick_action_index = min(index, len(self._quick_actions) - 1)
        else:
            self._selected_quick_action_index = None
        self._refresh_quick_action_buttons()
        self._persist_quick_actions_to_settings()

    def _load_quick_actions_from_settings(self) -> None:
        settings = self.llm_settings if isinstance(self.llm_settings, dict) else {}
        ui_block = settings.get("ui")
        if not isinstance(ui_block, dict):
            self._quick_actions = self._default_quick_actions()
            self._selected_quick_action_index = None
            return
        raw_items = ui_block.get("chat_quick_actions")
        parsed: List[tuple[str, str]] = []
        if isinstance(raw_items, list):
            for item in raw_items:
                label = ""
                prompt = ""
                if isinstance(item, dict):
                    label = str(item.get("label") or "").strip()
                    prompt = str(item.get("prompt") or "").strip()
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    label = str(item[0] or "").strip()
                    prompt = str(item[1] or "").strip()
                if label and prompt:
                    parsed.append((label, prompt))
        self._quick_actions = parsed or self._default_quick_actions()
        raw_index = ui_block.get("chat_selected_quick_action")
        try:
            idx = int(raw_index) if raw_index is not None else -1
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(self._quick_actions):
            self._selected_quick_action_index = idx
        else:
            self._selected_quick_action_index = None

    def _persist_quick_actions_to_settings(self) -> None:
        if not isinstance(self.llm_settings, dict):
            return
        ui_block = self.llm_settings.get("ui")
        if not isinstance(ui_block, dict):
            ui_block = {}
        ui_block["chat_quick_actions"] = [
            {"label": label, "prompt": prompt} for label, prompt in self._quick_actions
        ]
        ui_block["chat_selected_quick_action"] = (
            int(self._selected_quick_action_index)
            if self._selected_quick_action_index is not None
            else -1
        )
        self.llm_settings["ui"] = ui_block
        save_fn = _widget_module_attr("save_llm_settings", save_llm_settings)
        try:
            save_fn(self.llm_settings)
        except Exception:
            pass

    def _set_remove_quick_action_enabled(self, enabled: bool) -> None:
        button = getattr(self, "remove_quick_action_button", None)
        if button is None:
            return
        try:
            button.setEnabled(bool(enabled))
        except RuntimeError:
            return

    def _sync_prompt_input_height(self) -> None:
        doc = self.prompt_text_edit.document()
        metrics = QtGui.QFontMetrics(self.prompt_text_edit.font())
        line_height = max(18, int(metrics.lineSpacing()))
        block_count = max(1, int(doc.blockCount()) if doc is not None else 1)
        visual_lines = (
            max(block_count, int(doc.size().height()))
            if doc is not None
            else block_count
        )
        target = min(156, max(48, visual_lines * line_height + 22))
        if self.prompt_text_edit.height() != target:
            self.prompt_text_edit.setFixedHeight(target)

    def _navigate_prompt_history(self, delta: int) -> bool:
        if not self._prompt_history:
            return False
        if self._prompt_history_index < 0:
            if delta >= 0:
                return False
            self._prompt_history_draft = self.prompt_text_edit.toPlainText()
            self._prompt_history_index = len(self._prompt_history) - 1
        else:
            next_idx = self._prompt_history_index + int(delta)
            if next_idx < 0:
                self._prompt_history_index = 0
            elif next_idx >= len(self._prompt_history):
                self._prompt_history_index = -1
                self.prompt_text_edit.setPlainText(self._prompt_history_draft)
                cursor = self.prompt_text_edit.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                self.prompt_text_edit.setTextCursor(cursor)
                return True
            else:
                self._prompt_history_index = next_idx
        self.prompt_text_edit.setPlainText(
            self._prompt_history[self._prompt_history_index]
        )
        cursor = self.prompt_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.prompt_text_edit.setTextCursor(cursor)
        return True
