from __future__ import annotations

import os
import tempfile
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QThreadPool

from annolid.gui.widgets.ai_chat_backend import StreamingChatTask, clear_chat_session
from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog
from annolid.gui.widgets.provider_registry import ProviderRegistry
from annolid.utils.llm_settings import (
    has_provider_api_key,
    load_llm_settings,
    save_llm_settings,
)
from annolid.utils.tts_settings import default_tts_settings, load_tts_settings


class _ChatBubble(QtWidgets.QFrame):
    """Single chat bubble row with timestamp and sender styling."""

    def __init__(
        self,
        sender: str,
        text: str,
        *,
        is_user: bool,
        on_speak=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sender = sender
        self._is_user = is_user
        self._ts = datetime.now().strftime("%H:%M")
        self._on_speak = on_speak

        self.setObjectName("chatBubble")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setProperty("role", "user" if is_user else "assistant")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self.sender_label = QtWidgets.QLabel(sender, self)
        self.sender_label.setObjectName("sender")
        layout.addWidget(self.sender_label)

        self.message_label = QtWidgets.QLabel(text, self)
        self.message_label.setObjectName("message")
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.message_label)

        self.meta_label = QtWidgets.QLabel(self._ts, self)
        self.meta_label.setObjectName("meta")
        self.meta_label.setAlignment(QtCore.Qt.AlignVCenter)

        self.speak_button = QtWidgets.QPushButton("Speak", self)
        self.speak_button.setObjectName("bubbleSpeakButton")
        self.speak_button.setToolTip("Read this message aloud")
        self.speak_button.setCursor(QtCore.Qt.PointingHandCursor)
        style = self.style()
        icon = style.standardIcon(QtWidgets.QStyle.SP_MediaVolume)
        if icon.isNull():
            icon = QtGui.QIcon.fromTheme("audio-volume-high")
        self.speak_button.setIcon(icon)
        self.speak_button.setText("")
        self.speak_button.setFixedSize(24, 22)
        self.speak_button.setIconSize(QtCore.QSize(14, 14))
        self.speak_button.clicked.connect(self._speak)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self.speak_button, 0, QtCore.Qt.AlignLeft)
        footer.addStretch(1)
        footer.addWidget(self.meta_label, 0, QtCore.Qt.AlignRight)
        layout.addLayout(footer)

    def append_text(self, chunk: str) -> None:
        self.message_label.setText(self.message_label.text() + chunk)

    def set_text(self, text: str) -> None:
        self.message_label.setText(text)

    def text(self) -> str:
        return self.message_label.text()

    def _speak(self) -> None:
        if callable(self._on_speak):
            self._on_speak(self.text())


class AIChatWidget(QtWidgets.QWidget):
    """Annolid Bot chat UI for local/cloud models with visual sharing and streaming."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.llm_settings = load_llm_settings()
        self._providers = ProviderRegistry(self.llm_settings, save_llm_settings)
        self.provider_labels: Dict[str, str] = {
            "ollama": "Ollama (local)",
            "openai": "OpenAI GPT",
            "openrouter": "OpenRouter",
            "gemini": "Google Gemini",
        }
        self.selected_provider = self._providers.current_provider()
        self.available_models = self._providers.available_models(self.selected_provider)
        self.selected_model = self._providers.resolve_initial_model(
            self.selected_provider,
            self.available_models,
        )
        self._suppress_model_updates = False
        self._suppress_provider_updates = False
        self.canvas_widget: Optional[QtWidgets.QWidget] = None
        self.host_window_widget: Optional[QtWidgets.QWidget] = None
        self.image_path: str = ""
        self._snapshot_paths: List[str] = []
        self._current_response_bubble: Optional[_ChatBubble] = None
        self.is_streaming_chat = False
        self.is_recording = False
        self.session_id = "gui:annolid_bot:default"
        self._applying_theme_styles = False
        self.thread_pool = QThreadPool()
        self._asr_pipeline = None
        self._asr_lock = threading.Lock()

        self._build_ui()
        self._apply_theme_styles()
        self._update_model_selector()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header_bar = QtWidgets.QHBoxLayout()
        self.provider_chip_label = QtWidgets.QLabel(self)
        self.provider_chip_label.setObjectName("botChipLabel")
        header_bar.addWidget(self.provider_chip_label)

        self.model_chip_label = QtWidgets.QLabel(self)
        self.model_chip_label.setObjectName("botChipLabel")
        header_bar.addWidget(self.model_chip_label)

        self.session_chip_label = QtWidgets.QLabel(self)
        self.session_chip_label.setObjectName("botChipLabel")
        header_bar.addWidget(self.session_chip_label, 1)

        self.clear_chat_button = QtWidgets.QPushButton("Clear Chat", self)
        self.clear_chat_button.setObjectName("clearChatButton")
        header_bar.addWidget(self.clear_chat_button, 0)

        self.tool_trace_checkbox = QtWidgets.QCheckBox("Show tool trace", self)
        self.tool_trace_checkbox.setChecked(False)
        self.tool_trace_checkbox.setObjectName("toolTraceCheckbox")
        header_bar.addWidget(self.tool_trace_checkbox, 0)
        root.addLayout(header_bar)

        top_bar = QtWidgets.QHBoxLayout()
        self.provider_selector = QtWidgets.QComboBox(self)
        for key, label in self.provider_labels.items():
            self.provider_selector.addItem(label, userData=key)
        idx = self.provider_selector.findData(self.selected_provider)
        if idx >= 0:
            self.provider_selector.setCurrentIndex(idx)
        top_bar.addWidget(self.provider_selector, 2)

        self.model_selector = QtWidgets.QComboBox(self)
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        top_bar.addWidget(self.model_selector, 3)

        self.configure_button = QtWidgets.QPushButton("Configure…", self)
        top_bar.addWidget(self.configure_button, 0)
        root.addLayout(top_bar)

        share_bar = QtWidgets.QHBoxLayout()
        self.attach_canvas_checkbox = QtWidgets.QCheckBox("Attach canvas", self)
        self.attach_canvas_checkbox.setChecked(False)
        self.attach_window_checkbox = QtWidgets.QCheckBox("Attach window", self)
        share_bar.addWidget(self.attach_canvas_checkbox)
        share_bar.addWidget(self.attach_window_checkbox)

        self.share_canvas_button = QtWidgets.QPushButton("Share Canvas", self)
        self.share_window_button = QtWidgets.QPushButton("Share Window", self)
        share_bar.addWidget(self.share_canvas_button)
        share_bar.addWidget(self.share_window_button)
        share_bar.addStretch(1)
        root.addLayout(share_bar)

        self.shared_image_label = QtWidgets.QLabel("Shared image: none", self)
        self.shared_image_label.setObjectName("sharedImageLabel")
        root.addWidget(self.shared_image_label)

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.chat_container = QtWidgets.QWidget(self.scroll_area)
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(8)
        self.chat_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_container)
        root.addWidget(self.scroll_area, 1)

        input_bar = QtWidgets.QHBoxLayout()
        self.prompt_text_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_text_edit.setPlaceholderText("Message Annolid Bot…")
        self.prompt_text_edit.setFixedHeight(74)
        input_bar.addWidget(self.prompt_text_edit, 1)

        side_buttons = QtWidgets.QVBoxLayout()
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.talk_button = QtWidgets.QPushButton("Talk", self)
        side_buttons.addWidget(self.send_button)
        side_buttons.addWidget(self.talk_button)
        side_buttons.addStretch(1)
        input_bar.addLayout(side_buttons)
        root.addLayout(input_bar)

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setObjectName("chatStatusLabel")
        root.addWidget(self.status_label)

        self.provider_selector.currentIndexChanged.connect(self.on_provider_changed)
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        self.model_selector.editTextChanged.connect(self.on_model_text_edited)
        line_edit = self.model_selector.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self.on_model_editing_finished)
        self.configure_button.clicked.connect(self.open_llm_settings_dialog)
        self.send_button.clicked.connect(self.chat_with_model)
        self.share_canvas_button.clicked.connect(self._share_canvas_now)
        self.share_window_button.clicked.connect(self._share_window_now)
        self.talk_button.clicked.connect(self.toggle_recording)
        self.clear_chat_button.clicked.connect(self.clear_chat_conversation)
        self._refresh_header_chips()

    @staticmethod
    def _mix_colors(
        a: QtGui.QColor, b: QtGui.QColor, ratio: float = 0.5
    ) -> QtGui.QColor:
        r = max(0.0, min(1.0, float(ratio)))
        inv = 1.0 - r
        return QtGui.QColor(
            int(a.red() * inv + b.red() * r),
            int(a.green() * inv + b.green() * r),
            int(a.blue() * inv + b.blue() * r),
        )

    def _apply_theme_styles(self) -> None:
        if self._applying_theme_styles:
            return
        self._applying_theme_styles = True
        pal = self.palette()
        base = pal.color(QtGui.QPalette.Base)
        window = pal.color(QtGui.QPalette.Window)
        text = pal.color(QtGui.QPalette.Text)
        mid = pal.color(QtGui.QPalette.Mid)
        highlight = pal.color(QtGui.QPalette.Highlight)
        button = pal.color(QtGui.QPalette.Button)

        bubble_user = self._mix_colors(base, highlight, 0.28)
        bubble_assistant = self._mix_colors(base, window, 0.45)
        bubble_meta = self._mix_colors(text, window, 0.55)
        area_bg = self._mix_colors(window, base, 0.35)
        input_bg = base
        hover_bg = self._mix_colors(button, highlight, 0.18)

        try:
            self.setStyleSheet(
                f"""
                QWidget {{
                    background: {window.name()};
                    color: {text.name()};
                }}
                QScrollArea {{
                    border: 1px solid {mid.name()};
                    border-radius: 10px;
                    background: {area_bg.name()};
                }}
                QPlainTextEdit {{
                    border: 1px solid {mid.name()};
                    border-radius: 10px;
                    background: {input_bg.name()};
                    padding: 8px;
                    font-size: 13px;
                }}
                QPushButton {{
                    border: 1px solid {mid.name()};
                    border-radius: 8px;
                    background: {button.name()};
                    padding: 6px 10px;
                }}
                QPushButton:hover {{
                    background: {hover_bg.name()};
                }}
                QLabel#sharedImageLabel, QLabel#chatStatusLabel {{
                    color: {bubble_meta.name()};
                    font-size: 11px;
                }}
                QLabel#botChipLabel {{
                    border: 1px solid {mid.name()};
                    border-radius: 10px;
                    background: {area_bg.name()};
                    color: {bubble_meta.name()};
                    padding: 2px 8px;
                    font-size: 10px;
                    font-weight: 600;
                }}
                QPushButton#clearChatButton {{
                    border: 1px solid {mid.name()};
                    border-radius: 8px;
                    background: {button.name()};
                    padding: 5px 10px;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QCheckBox#toolTraceCheckbox {{
                    color: {bubble_meta.name()};
                    font-size: 11px;
                    padding: 1px 4px;
                }}
                QFrame#chatBubble[role="user"] {{
                    background-color: {bubble_user.name()};
                    border-radius: 14px;
                    padding: 8px 10px;
                }}
                QFrame#chatBubble[role="assistant"] {{
                    background-color: {bubble_assistant.name()};
                    border-radius: 14px;
                    padding: 8px 10px;
                    border: 1px solid {mid.name()};
                }}
                QLabel#sender {{
                    color: {bubble_meta.name()};
                    font-size: 11px;
                    font-weight: 600;
                }}
                QLabel#message {{
                    color: {text.name()};
                    font-size: 13px;
                }}
                QLabel#meta {{
                    color: {bubble_meta.name()};
                    font-size: 10px;
                }}
                QPushButton#bubbleSpeakButton {{
                    border: 1px solid {mid.name()};
                    border-radius: 6px;
                    background: {button.name()};
                    padding: 2px 8px;
                    min-height: 18px;
                    font-size: 10px;
                }}
                """
            )
        finally:
            self._applying_theme_styles = False

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
            QtCore.QEvent.StyleChange,
        ):
            self._apply_theme_styles()
        super().changeEvent(event)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _add_bubble(self, sender: str, text: str, *, is_user: bool) -> _ChatBubble:
        row = QtWidgets.QHBoxLayout()
        bubble = _ChatBubble(
            sender,
            text,
            is_user=is_user,
            on_speak=self.speak_text_async,
            parent=self.chat_container,
        )
        if is_user:
            row.addStretch(1)
            row.addWidget(bubble, 0, QtCore.Qt.AlignRight)
        else:
            row.addWidget(bubble, 0, QtCore.Qt.AlignLeft)
            row.addStretch(1)

        # Insert before trailing stretch item.
        insert_idx = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertLayout(insert_idx, row)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        return bubble

    def _assistant_display_name(self) -> str:
        model_name = str(self.selected_model or "").strip()
        if model_name:
            return f"Annolid Bot ({model_name})"
        return "Annolid Bot"

    def _ensure_provider_ready(self) -> bool:
        if self.selected_provider == "openai" and not has_provider_api_key(
            self.llm_settings, "openai"
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "OpenAI API key required",
                "Please add your OpenAI API key in the AI Model Settings dialog.",
            )
            return False
        if self.selected_provider == "gemini" and not has_provider_api_key(
            self.llm_settings, "gemini"
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Gemini API key required",
                "Please add your Gemini API key in the AI Model Settings dialog.",
            )
            return False
        if self.selected_provider == "openrouter" and not has_provider_api_key(
            self.llm_settings, "openrouter"
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "OpenRouter API key required",
                "Please add your OpenRouter API key in the AI Model Settings dialog.",
            )
            return False
        return True

    def _persist_state(self) -> None:
        self._providers.set_current_provider(self.selected_provider)
        self._providers.remember_last_model(self.selected_provider, self.selected_model)

    def _update_model_selector(self) -> None:
        self._suppress_model_updates = True
        try:
            self.model_selector.blockSignals(True)
            self.model_selector.clear()
            if self.available_models:
                self.model_selector.addItems(self.available_models)
            if self.selected_model:
                self.model_selector.setCurrentText(self.selected_model)
        finally:
            self.model_selector.blockSignals(False)
            self._suppress_model_updates = False

    def on_provider_changed(self, index: int) -> None:
        if self._suppress_provider_updates:
            return
        provider = self.provider_selector.itemData(index)
        if not provider:
            return
        self.selected_provider = provider
        self._providers.set_current_provider(provider)
        self.available_models = self._providers.available_models(provider)
        if self.selected_model not in self.available_models:
            self.selected_model = self._providers.resolve_initial_model(
                provider, self.available_models
            )
        self._update_model_selector()
        self._persist_state()
        self._refresh_header_chips()

    def on_model_changed(self, index: int) -> None:
        if self._suppress_model_updates:
            return
        self.selected_model = self.model_selector.itemText(index).strip()
        self._persist_state()
        self._refresh_header_chips()

    def on_model_text_edited(self, text: str) -> None:
        if self._suppress_model_updates:
            return
        self.selected_model = text.strip()
        self._refresh_header_chips()

    def on_model_editing_finished(self) -> None:
        if self._suppress_model_updates:
            return
        text = self.model_selector.currentText().strip()
        if not text:
            return
        self.selected_model = text
        if text not in self.available_models:
            self.available_models.append(text)
            self._update_model_selector()
            self.model_selector.setCurrentText(text)
        self._persist_state()
        self._refresh_header_chips()

    def open_llm_settings_dialog(self) -> None:
        dialog = LLMSettingsDialog(self, settings=dict(self.llm_settings))
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        settings = dialog.get_settings()
        settings.setdefault("last_models", self.llm_settings.get("last_models", {}))
        self.llm_settings = settings
        self._providers = ProviderRegistry(self.llm_settings, save_llm_settings)
        new_provider = self.llm_settings.get("provider", self.selected_provider)
        self._suppress_provider_updates = True
        try:
            provider_index = self.provider_selector.findData(new_provider)
            if provider_index != -1:
                self.provider_selector.setCurrentIndex(provider_index)
            else:
                self.selected_provider = new_provider
        finally:
            self._suppress_provider_updates = False
        self.selected_provider = self.provider_selector.currentData()
        self.available_models = self._providers.available_models(self.selected_provider)
        self.selected_model = self._providers.resolve_initial_model(
            self.selected_provider,
            self.available_models,
        )
        self._update_model_selector()
        self._persist_state()
        self._refresh_header_chips()

    def set_provider_and_model(self, provider: str, model: str = "") -> None:
        provider = (provider or "").strip().lower()
        if not provider:
            return
        provider_index = self.provider_selector.findData(provider)
        if provider_index != -1:
            self.provider_selector.setCurrentIndex(provider_index)
        else:
            self.selected_provider = provider
            self._providers.set_current_provider(provider)
            self.available_models = self._providers.available_models(provider)
        if model:
            model = model.strip()
            if model and model not in self.available_models:
                self.available_models.append(model)
            self.selected_model = model
            self._update_model_selector()
            self.model_selector.setCurrentText(model)
        self._persist_state()
        self._refresh_header_chips()

    def set_default_visual_share_mode(
        self, *, attach_canvas: bool = True, attach_window: bool = False
    ) -> None:
        self.attach_canvas_checkbox.setChecked(bool(attach_canvas))
        self.attach_window_checkbox.setChecked(bool(attach_window))

    def set_session_id(self, session_id: str) -> None:
        session_text = str(session_id or "").strip()
        if session_text:
            self.session_id = session_text
            self._refresh_header_chips()

    def _refresh_header_chips(self) -> None:
        provider_text = self.provider_labels.get(
            self.selected_provider, str(self.selected_provider or "unknown")
        )
        model_text = str(self.selected_model or "unknown")
        session_text = str(self.session_id or "default")
        self.provider_chip_label.setText(f"Provider: {provider_text}")
        self.model_chip_label.setText(f"Model: {model_text}")
        self.session_chip_label.setText(f"Session: {session_text}")

    def clear_chat_conversation(self) -> None:
        if self.is_streaming_chat:
            self.status_label.setText("Wait for current response to finish.")
            return
        clear_chat_session(self.session_id)
        self._clear_chat_bubbles()
        self.status_label.setText("Conversation cleared.")

    def _clear_chat_bubbles(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item is None:
                continue
            layout = item.layout()
            widget = item.widget()
            if layout is not None:
                while layout.count():
                    sub = layout.takeAt(0)
                    if sub is None:
                        continue
                    w = sub.widget()
                    if w is not None:
                        w.deleteLater()
                layout.deleteLater()
            elif widget is not None:
                widget.deleteLater()
        self._current_response_bubble = None
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def set_canvas(self, canvas: Optional[QtWidgets.QWidget]) -> None:
        self.canvas_widget = canvas

    def set_host_window(self, window: Optional[QtWidgets.QWidget]) -> None:
        self.host_window_widget = window

    def set_image_path(self, image_path: str) -> None:
        if image_path and image_path not in self._snapshot_paths:
            self._cleanup_snapshots()
        self.image_path = image_path
        self._update_shared_image_label(image_path)

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

    def _update_shared_image_label(self, image_path: Optional[str]) -> None:
        if image_path and os.path.exists(image_path):
            self.shared_image_label.setText(
                f"Shared image: {os.path.basename(image_path)}"
            )
        else:
            self.shared_image_label.setText("Shared image: none")

    def _share_canvas_now(self) -> None:
        image_path = self._snapshot_canvas_to_tempfile()
        if image_path:
            self.set_image_path(image_path)
            self._add_bubble(
                "Annolid Bot",
                f"Canvas snapshot attached: {os.path.basename(image_path)}",
                is_user=False,
            )

    def _share_window_now(self) -> None:
        image_path = self._snapshot_window_to_tempfile()
        if image_path:
            self.set_image_path(image_path)
            self._add_bubble(
                "Annolid Bot",
                f"Window snapshot attached: {os.path.basename(image_path)}",
                is_user=False,
            )

    def _prepare_chat_image(self) -> Optional[str]:
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

    def chat_with_model(self) -> None:
        if self.is_streaming_chat:
            return
        raw_prompt = self.prompt_text_edit.toPlainText().strip()
        if not raw_prompt:
            return
        if not self._ensure_provider_ready():
            return

        chat_image_path = self._prepare_chat_image()
        self._add_bubble("You", raw_prompt, is_user=True)
        assistant_name = self._assistant_display_name()
        self._current_response_bubble = self._add_bubble(
            assistant_name,
            "",
            is_user=False,
        )

        self.prompt_text_edit.clear()
        self.send_button.setEnabled(False)
        self.is_streaming_chat = True
        self.status_label.setText(f"Talking to {self.selected_model}…")

        task = StreamingChatTask(
            prompt=raw_prompt,
            image_path=chat_image_path or self.image_path,
            widget=self,
            model=self.selected_model,
            provider=self.selected_provider,
            settings=self.llm_settings,
            session_id=self.session_id,
            show_tool_trace=self.tool_trace_checkbox.isChecked(),
        )
        self.thread_pool.start(task)

    @QtCore.Slot(str)
    def stream_chat_chunk(self, chunk: str) -> None:
        if self._current_response_bubble is None:
            return
        self._current_response_bubble.append_text(chunk)
        self._scroll_to_bottom()

    @QtCore.Slot(str, bool)
    def update_chat_response(self, message: str, is_error: bool) -> None:
        if self._current_response_bubble is None:
            bubble = self._add_bubble(
                "Assistant",
                message or "",
                is_user=False,
            )
            self._current_response_bubble = bubble

        if is_error:
            current = self._current_response_bubble.text()
            self._current_response_bubble.set_text((current + "\n" + message).strip())
            self.status_label.setText("Error")
        elif message:
            self._current_response_bubble.set_text(message)
            self.status_label.setText("Done")
        else:
            self.status_label.setText("Done")

        self.send_button.setEnabled(True)
        self.is_streaming_chat = False
        self._current_response_bubble = None
        self._scroll_to_bottom()

    def _last_assistant_text(self) -> str:
        # Find last non-user bubble text.
        for i in range(self.chat_layout.count() - 1, -1, -1):
            item = self.chat_layout.itemAt(i)
            if item is None:
                continue
            row = item.layout()
            if row is None:
                continue
            for j in range(row.count()):
                widget = row.itemAt(j).widget()
                if (
                    isinstance(widget, _ChatBubble)
                    and widget.property("role") == "assistant"
                ):
                    text = widget.text().strip()
                    if text:
                        return text
        return ""

    def _tts_settings_snapshot(self) -> Dict[str, object]:
        settings = load_tts_settings()
        defaults = default_tts_settings()
        return {
            "engine": settings.get("engine", defaults.get("engine", "auto")),
            "voice": settings.get("voice", defaults["voice"]),
            "pocket_voice": settings.get(
                "pocket_voice", defaults.get("pocket_voice", "alba")
            ),
            "pocket_prompt_path": settings.get(
                "pocket_prompt_path", defaults.get("pocket_prompt_path", "")
            ),
            "pocket_speed": settings.get(
                "pocket_speed", defaults.get("pocket_speed", 1.0)
            ),
            "lang": settings.get("lang", defaults["lang"]),
            "speed": settings.get("speed", defaults["speed"]),
            "chatterbox_voice_path": settings.get(
                "chatterbox_voice_path", defaults.get("chatterbox_voice_path", "")
            ),
            "chatterbox_dtype": settings.get(
                "chatterbox_dtype", defaults.get("chatterbox_dtype", "fp32")
            ),
            "chatterbox_max_new_tokens": settings.get(
                "chatterbox_max_new_tokens",
                defaults.get("chatterbox_max_new_tokens", 1024),
            ),
            "chatterbox_repetition_penalty": settings.get(
                "chatterbox_repetition_penalty",
                defaults.get("chatterbox_repetition_penalty", 1.2),
            ),
            "chatterbox_apply_watermark": settings.get(
                "chatterbox_apply_watermark",
                defaults.get("chatterbox_apply_watermark", False),
            ),
        }

    def read_last_reply_async(self) -> None:
        text = self._last_assistant_text()
        if not text:
            self.status_label.setText("No assistant reply to read.")
            return
        self.speak_text_async(text)

    def speak_text_async(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            self.status_label.setText("No text to read.")
            return

        def _run_tts() -> None:
            try:
                from annolid.utils.audio_playback import play_audio_buffer
                from annolid.agents.tts_router import synthesize_tts

                audio_data = synthesize_tts(text, self._tts_settings_snapshot())
                if not audio_data:
                    raise RuntimeError("No audio generated.")
                samples, sample_rate = audio_data
                played = play_audio_buffer(samples, sample_rate, blocking=True)
                if not played:
                    raise RuntimeError("No usable audio device found.")
                QtCore.QMetaObject.invokeMethod(
                    self.status_label,
                    "setText",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "Reply read aloud."),
                )
            except Exception as exc:
                QtCore.QMetaObject.invokeMethod(
                    self.status_label,
                    "setText",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, f"Read failed: {exc}"),
                )

        threading.Thread(target=_run_tts, daemon=True).start()

    def toggle_recording(self) -> None:
        if not self.is_recording:
            self.is_recording = True
            self.talk_button.setText("Stop")
            self.status_label.setText("Listening…")
            threading.Thread(target=self._record_voice, daemon=True).start()
        else:
            self.is_recording = False
            self.talk_button.setText("Talk")
            self.status_label.setText("Processing speech…")

    def _record_voice(self) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            QtCore.QMetaObject.invokeMethod(
                self.status_label,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(
                    str,
                    "Audio recording deps missing. Install: pip install sounddevice soundfile",
                ),
            )
            QtCore.QMetaObject.invokeMethod(
                self.talk_button,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "Talk"),
            )
            self.is_recording = False
            return

        sample_rate = 16000
        channels = 1
        audio_chunks: List[np.ndarray] = []

        def _audio_callback(indata, frames, stream_time, status) -> None:
            del frames, stream_time
            if status:
                return
            audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                callback=_audio_callback,
            ):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as exc:
            QtCore.QMetaObject.invokeMethod(
                self.status_label,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Mic capture failed: {exc}"),
            )
            QtCore.QMetaObject.invokeMethod(
                self.talk_button,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "Talk"),
            )
            self.is_recording = False
            return

        final_text = ""
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0).reshape(-1)
            if np.abs(audio_data).max(initial=0.0) < 1e-4:
                final_text = ""
            else:
                fd, audio_path = tempfile.mkstemp(prefix="annolid_talk_", suffix=".wav")
                os.close(fd)
                try:
                    sf.write(audio_path, audio_data, sample_rate)
                    final_text = self._transcribe_with_whisper_tiny(audio_path)
                except Exception as exc:
                    QtCore.QMetaObject.invokeMethod(
                        self.status_label,
                        "setText",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, f"Transcription failed: {exc}"),
                    )
                    QtCore.QMetaObject.invokeMethod(
                        self.talk_button,
                        "setText",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, "Talk"),
                    )
                    self.is_recording = False
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except OSError:
                        pass
                    return
                finally:
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except OSError:
                        pass

        if final_text:
            QtCore.QMetaObject.invokeMethod(
                self.prompt_text_edit,
                "setPlainText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, final_text),
            )
            QtCore.QMetaObject.invokeMethod(
                self.status_label,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "Speech captured. Review and send."),
            )
        else:
            QtCore.QMetaObject.invokeMethod(
                self.status_label,
                "setText",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, "No speech captured."),
            )
        QtCore.QMetaObject.invokeMethod(
            self.talk_button,
            "setText",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, "Talk"),
        )
        self.is_recording = False

    def _get_asr_pipeline(self):
        with self._asr_lock:
            if self._asr_pipeline is not None:
                return self._asr_pipeline
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "ASR deps missing. Install: pip install transformers torch"
                ) from exc
            device = -1
            if torch.cuda.is_available():
                device = 0
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device,
            )
            return self._asr_pipeline

    def _transcribe_with_whisper_tiny(self, audio_path: str) -> str:
        asr = self._get_asr_pipeline()
        result = asr(
            audio_path,
            return_timestamps=False,
            generate_kwargs={"task": "transcribe"},
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()
