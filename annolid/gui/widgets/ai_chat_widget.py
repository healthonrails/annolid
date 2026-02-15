from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QThreadPool

from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.gui.realtime_launch import (
    build_realtime_launch_payload,
    resolve_realtime_model_weight,
)
from annolid.gui.widgets.ai_chat_audio_controller import ChatAudioController
from annolid.gui.widgets.ai_chat_backend import StreamingChatTask, clear_chat_session
from annolid.gui.widgets.ai_chat_session_dialog import ChatSessionManagerDialog
from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog
from annolid.gui.widgets.provider_registry import ProviderRegistry
from annolid.utils.llm_settings import (
    has_provider_api_key,
    load_llm_settings,
    provider_kind,
    save_llm_settings,
)


class _ChatBubble(QtWidgets.QFrame):
    """Single chat bubble row with timestamp and sender styling."""

    def __init__(
        self,
        sender: str,
        text: str,
        *,
        is_user: bool,
        on_speak=None,
        on_copy=None,
        on_regenerate=None,
        allow_regenerate: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sender = sender
        self._is_user = is_user
        self._ts = datetime.now().strftime("%H:%M")
        self._on_speak = on_speak
        self._on_copy = on_copy
        self._on_regenerate = on_regenerate
        self._allow_regenerate = bool(allow_regenerate)
        self._raw_text = str(text or "")
        self._preferred_text_width = 560
        self._manual_text_width: Optional[int] = None
        self._resize_drag_active = False
        self._drag_anchor_global_x = 0
        self._drag_start_text_width = 0
        self._edge_handle_px = 12
        self._syncing_message_height = False

        self.setObjectName("chatBubble")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        # Ensure QSS background colors (e.g. user bubble green) always paint.
        self.setAutoFillBackground(True)
        self.setProperty("role", "user" if is_user else "assistant")
        self._apply_role_palette()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self.sender_label = QtWidgets.QLabel(sender, self)
        self.sender_label.setObjectName("sender")
        layout.addWidget(self.sender_label)

        self.message_view = QtWidgets.QTextBrowser(self)
        self.message_view.setObjectName("messageView")
        self.message_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.message_view.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.message_view.setAutoFillBackground(False)
        self.message_view.setOpenExternalLinks(True)
        self.message_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.message_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.message_view.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents
        )
        self.message_view.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse
        )
        self.message_doc = QtGui.QTextDocument(self.message_view)
        self.message_doc.setDocumentMargin(0)
        text_option = self.message_doc.defaultTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.message_doc.setDefaultTextOption(text_option)
        self._apply_message_doc_styles()
        self.message_view.setDocument(self.message_doc)
        viewport = self.message_view.viewport()
        viewport.setAutoFillBackground(False)
        viewport.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        viewport.setStyleSheet("background: transparent;")
        doc_layout = self.message_doc.documentLayout()
        if doc_layout is not None:
            doc_layout.documentSizeChanged.connect(self._sync_message_height)
        layout.addWidget(self.message_view)
        self._render_markdown(self._raw_text)

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

        self.copy_button = QtWidgets.QPushButton("", self)
        self.copy_button.setObjectName("bubbleCopyButton")
        self.copy_button.setToolTip("Copy message text")
        self.copy_button.setCursor(QtCore.Qt.PointingHandCursor)
        copy_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_FileDialogDetailedView
        )
        if copy_icon.isNull():
            copy_icon = QtGui.QIcon.fromTheme("edit-copy")
        self.copy_button.setIcon(copy_icon)
        self.copy_button.setFixedSize(24, 22)
        self.copy_button.setIconSize(QtCore.QSize(14, 14))
        self.copy_button.clicked.connect(self._copy_text)

        self.regenerate_button = QtWidgets.QPushButton("", self)
        self.regenerate_button.setObjectName("bubbleRegenerateButton")
        self.regenerate_button.setToolTip("Regenerate this reply")
        self.regenerate_button.setCursor(QtCore.Qt.PointingHandCursor)
        regen_icon = self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        if regen_icon.isNull():
            regen_icon = QtGui.QIcon.fromTheme("view-refresh")
        self.regenerate_button.setIcon(regen_icon)
        self.regenerate_button.setFixedSize(24, 22)
        self.regenerate_button.setIconSize(QtCore.QSize(14, 14))
        self.regenerate_button.setVisible(
            (not self._is_user) and self._allow_regenerate
        )
        self.regenerate_button.clicked.connect(self._regenerate)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addWidget(self.speak_button, 0, QtCore.Qt.AlignLeft)
        footer.addWidget(self.copy_button, 0, QtCore.Qt.AlignLeft)
        footer.addWidget(self.regenerate_button, 0, QtCore.Qt.AlignLeft)
        footer.addStretch(1)
        footer.addWidget(self.meta_label, 0, QtCore.Qt.AlignRight)
        layout.addLayout(footer)

    def _apply_role_palette(self) -> None:
        palette = self.palette()
        if self._is_user:
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#145C4C"))
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#F3FFF8"))
            palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#F3FFF8"))
        else:
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#1f252f"))
        self.setPalette(palette)

    def _apply_message_doc_styles(self) -> None:
        if self._is_user:
            self.message_doc.setDefaultStyleSheet(
                """
                body, p, li, ul, ol, blockquote {
                    color: #F3FFF8;
                }
                p { margin: 0 0 6px 0; }
                p:last-child { margin-bottom: 0; }
                pre {
                    background: rgba(0,0,0,0.18);
                    border: 1px solid rgba(255,255,255,0.16);
                    border-radius: 8px;
                    padding: 8px;
                    margin: 4px 0 4px 0;
                    color: #F3FFF8;
                }
                code {
                    background: rgba(0,0,0,0.16);
                    border-radius: 4px;
                    padding: 1px 3px;
                    color: #F3FFF8;
                }
                a {
                    color: #9BD2FF;
                    text-decoration: none;
                }
                """
            )
        else:
            self.message_doc.setDefaultStyleSheet(
                """
                p { margin: 0 0 6px 0; }
                p:last-child { margin-bottom: 0; }
                pre {
                    background: rgba(0,0,0,0.22);
                    border: 1px solid rgba(255,255,255,0.14);
                    border-radius: 8px;
                    padding: 8px;
                    margin: 4px 0 4px 0;
                }
                code {
                    background: rgba(0,0,0,0.16);
                    border-radius: 4px;
                    padding: 1px 3px;
                }
                a {
                    color: #7ed1ff;
                    text-decoration: none;
                }
                """
            )

    def append_text(self, chunk: str) -> None:
        self._raw_text += str(chunk or "")
        self._render_markdown(self._raw_text)

    def set_text(self, text: str) -> None:
        self._raw_text = str(text or "")
        self._render_markdown(self._raw_text)

    def text(self) -> str:
        return str(self.message_doc.toPlainText() or "").strip()

    def _render_markdown(self, text: str) -> None:
        content = str(text or "")
        if hasattr(self.message_doc, "setMarkdown"):
            self.message_doc.setMarkdown(content)
        else:
            self.message_doc.setPlainText(content)
        self._sync_message_height()

    def _sync_message_height(self, _doc_size: Optional[QtCore.QSizeF] = None) -> None:
        if self._syncing_message_height:
            return
        self._syncing_message_height = True
        try:
            available_width = max(220, self.width() - 24)
            text_width = min(max(260, int(self._preferred_text_width)), available_width)
            if int(self.message_doc.textWidth()) != int(text_width):
                self.message_doc.setTextWidth(float(text_width))
            self.message_doc.adjustSize()
            doc_height = int(self.message_doc.size().height())
            target = max(22, doc_height + 8)
            self.message_view.setMinimumHeight(target)
            self.message_view.setMaximumHeight(16777215)
        finally:
            self._syncing_message_height = False

    def set_message_width(self, width: int) -> None:
        self._preferred_text_width = max(260, int(width))
        self._sync_message_height()

    def apply_layout_width(self, text_width: int, bubble_max_width: int) -> None:
        max_text_width = max(260, int(bubble_max_width) - 24)
        if self._manual_text_width is not None:
            self._preferred_text_width = min(
                max(260, int(self._manual_text_width)), max_text_width
            )
            fixed_width = max(320, self._preferred_text_width + 24)
            self.setMinimumWidth(fixed_width)
            self.setMaximumWidth(fixed_width)
        else:
            if self._is_user:
                # Keep user bubbles compact and close to message size.
                ideal = int(self.message_doc.idealWidth()) if self.message_doc else 320
                self._preferred_text_width = min(max(260, ideal + 12), max_text_width)
                fixed_width = min(
                    max(280, self._preferred_text_width + 24), int(bubble_max_width)
                )
            else:
                self._preferred_text_width = min(
                    max(260, int(text_width)), max_text_width
                )
                fixed_width = max(320, int(bubble_max_width))
            self.setMinimumWidth(fixed_width)
            self.setMaximumWidth(fixed_width)
        self._sync_message_height()

    def _is_resize_handle_hit(self, pos: QtCore.QPoint) -> bool:
        local_x = int(pos.x())
        if self._is_user:
            return local_x <= self._edge_handle_px
        return local_x >= max(0, self.width() - self._edge_handle_px)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._is_resize_handle_hit(
            event.pos()
        ):
            self._resize_drag_active = True
            self._drag_anchor_global_x = int(event.globalX())
            self._drag_start_text_width = int(self._preferred_text_width)
            self.setCursor(QtCore.Qt.SizeHorCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._resize_drag_active:
            delta_x = int(event.globalX()) - self._drag_anchor_global_x
            if self._is_user:
                new_width = self._drag_start_text_width - delta_x
            else:
                new_width = self._drag_start_text_width + delta_x
            self._manual_text_width = max(260, int(new_width))
            self._preferred_text_width = self._manual_text_width
            self._sync_message_height()
            event.accept()
            return
        if self._is_resize_handle_hit(event.pos()):
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._resize_drag_active and event.button() == QtCore.Qt.LeftButton:
            self._resize_drag_active = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_message_height()

    def _speak(self) -> None:
        if callable(self._on_speak):
            self._on_speak(self.text())

    def _copy_text(self) -> None:
        if callable(self._on_copy):
            self._on_copy(self.text())

    def _regenerate(self) -> None:
        if callable(self._on_regenerate):
            self._on_regenerate(self.text())


class AIChatWidget(QtWidgets.QWidget):
    """Annolid Bot chat UI for local/cloud models with visual sharing and streaming."""

    @staticmethod
    def _new_startup_session_id() -> str:
        return f"gui:annolid_bot:{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.llm_settings = load_llm_settings()
        self._providers = ProviderRegistry(self.llm_settings, save_llm_settings)
        self.provider_labels: Dict[str, str] = self._providers.labels()
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
        self._bot_action_result: Dict[str, Any] = {}
        self.session_id = self._new_startup_session_id()
        self._applying_theme_styles = False
        self.thread_pool = QThreadPool()
        self._audio_controller: Optional[ChatAudioController] = None
        self._session_manager = AgentSessionManager()
        self._session_store = PersistentSessionStore(self._session_manager)
        self._max_prompt_chars = 4000
        self._last_user_prompt: str = ""
        self._typing_tick = 0
        self._typing_timer = QtCore.QTimer(self)
        self._typing_timer.setInterval(350)
        self._typing_timer.timeout.connect(self._on_typing_timer_tick)
        self.enable_progress_stream = True
        self._bot_action_results: Dict[str, Dict[str, Any]] = {}
        self._quick_actions: List[tuple[str, str]] = [
            (
                "Start Blink Stream",
                "open stream with model mediapipe face and classify eye blinks",
            ),
            ("Stop Stream", "stop realtime stream"),
            ("Summarize Context", "Summarize what we are doing in this session."),
            ("Review Memory", "Review long-term memory and list key facts."),
        ]
        self._selected_quick_action_index: Optional[int] = None
        self._current_progress_bubble: Optional[_ChatBubble] = None
        self._progress_lines: List[str] = []

        self._build_ui()
        self._apply_theme_styles()
        self._update_model_selector()
        self._load_session_history_into_bubbles(self.session_id)
        self._refresh_header_chips()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)
        root.addLayout(self._build_header_bar())
        root.addWidget(self._build_chat_area(), 1)
        root.addLayout(self._build_quick_actions_row())
        root.addWidget(self._build_composer_panel())
        root.addWidget(self._build_status_label())
        self._audio_controller = ChatAudioController(
            status_label=self.status_label,
            talk_button=self.talk_button,
            prompt_text_edit=self.prompt_text_edit,
            get_last_assistant_text=self._last_assistant_text,
        )
        self._wire_ui_signals()
        self._refresh_header_chips()

    def _build_header_bar(self) -> QtWidgets.QHBoxLayout:
        header_bar = QtWidgets.QHBoxLayout()
        header_bar.setSpacing(8)

        self.back_button = QtWidgets.QToolButton(self)
        self.back_button.setObjectName("chatTopIconButton")
        self.back_button.setToolTip("Back")
        self.back_button.setEnabled(False)
        self._set_button_icon(
            self.back_button,
            QtWidgets.QStyle.SP_ArrowBack,
            "go-previous",
        )
        header_bar.addWidget(self.back_button, 0)

        self.bot_icon_label = QtWidgets.QLabel(self)
        self.bot_icon_label.setObjectName("chatBotIconLabel")
        self.bot_icon_label.setFixedSize(34, 34)
        self.bot_icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self._set_bot_icon()
        header_bar.addWidget(self.bot_icon_label, 0)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(1)
        self.chat_title_label = QtWidgets.QLabel("Annolid Bot", self)
        self.chat_title_label.setObjectName("chatTitleLabel")
        title_col.addWidget(self.chat_title_label)

        self.session_chip_label = QtWidgets.QLabel(self)
        self.session_chip_label.setObjectName("chatSubtitleLabel")
        title_col.addWidget(self.session_chip_label)
        header_bar.addLayout(title_col, 1)

        self.clear_chat_button = QtWidgets.QToolButton(self)
        self.clear_chat_button.setObjectName("chatTopIconButton")
        self.clear_chat_button.setToolTip("Clear conversation")
        header_bar.addWidget(self.clear_chat_button, 0)

        self.sessions_button = QtWidgets.QToolButton(self)
        self.sessions_button.setObjectName("chatTopIconButton")
        self.sessions_button.setToolTip("Manage sessions")
        header_bar.addWidget(self.sessions_button, 0)

        self.configure_button = QtWidgets.QToolButton(self)
        self.configure_button.setObjectName("chatTopIconButton")
        self.configure_button.setToolTip("Configure providers and defaults")
        header_bar.addWidget(self.configure_button, 0)
        return header_bar

    def _build_provider_bar(self) -> QtWidgets.QHBoxLayout:
        top_bar = QtWidgets.QHBoxLayout()
        self.provider_selector = QtWidgets.QComboBox(self)
        self._populate_provider_selector()
        idx = self.provider_selector.findData(self.selected_provider)
        if idx >= 0:
            self.provider_selector.setCurrentIndex(idx)
        top_bar.addWidget(self.provider_selector, 2)

        self.model_selector = QtWidgets.QComboBox(self)
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        top_bar.addWidget(self.model_selector, 3)
        return top_bar

    def _populate_provider_selector(self) -> None:
        self.provider_labels = self._providers.labels()
        self.provider_selector.blockSignals(True)
        try:
            self.provider_selector.clear()
            for key, label in self.provider_labels.items():
                self.provider_selector.addItem(label, userData=key)
        finally:
            self.provider_selector.blockSignals(False)

    def _build_share_bar(self) -> QtWidgets.QHBoxLayout:
        share_bar = QtWidgets.QHBoxLayout()
        share_bar.setSpacing(6)
        self.attach_canvas_checkbox = QtWidgets.QCheckBox("Attach canvas", self)
        self.attach_canvas_checkbox.setChecked(False)
        self.attach_canvas_checkbox.setObjectName("chatInlineToggle")
        self.attach_window_checkbox = QtWidgets.QCheckBox("Attach window", self)
        self.attach_window_checkbox.setObjectName("chatInlineToggle")
        self.tool_trace_checkbox = QtWidgets.QCheckBox("Trace", self)
        self.tool_trace_checkbox.setChecked(False)
        self.tool_trace_checkbox.setObjectName("chatInlineToggle")
        self.allow_web_tools_checkbox = QtWidgets.QCheckBox("Allow web", self)
        self.allow_web_tools_checkbox.setChecked(True)
        self.allow_web_tools_checkbox.setObjectName("chatInlineToggle")
        self.allow_web_tools_checkbox.setToolTip(
            "Allow web_search/web_fetch tools for this chat turn."
        )
        share_bar.addWidget(self.attach_canvas_checkbox)
        share_bar.addWidget(self.attach_window_checkbox)
        share_bar.addWidget(self.tool_trace_checkbox)
        share_bar.addWidget(self.allow_web_tools_checkbox)

        self.share_canvas_button = QtWidgets.QToolButton(self)
        self.share_canvas_button.setText("")
        self.share_canvas_button.setObjectName("chatComposerIconButton")
        self.share_canvas_button.setToolTip("Capture the current canvas and attach it.")
        self.share_window_button = QtWidgets.QToolButton(self)
        self.share_window_button.setText("")
        self.share_window_button.setObjectName("chatComposerIconButton")
        self.share_window_button.setToolTip("Capture the window and attach it.")
        share_bar.addWidget(self.share_canvas_button)
        share_bar.addWidget(self.share_window_button)
        share_bar.addStretch(1)
        return share_bar

    def _build_shared_image_label(self) -> QtWidgets.QLabel:
        self.shared_image_label = QtWidgets.QLabel("Shared image: none", self)
        self.shared_image_label.setObjectName("sharedImageLabel")
        return self.shared_image_label

    def _build_chat_area(self) -> QtWidgets.QScrollArea:
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.chat_container = QtWidgets.QWidget(self.scroll_area)
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(8)
        self.empty_state_label = QtWidgets.QLabel(
            "Start a conversation with Annolid Bot.\nTip: press Ctrl+Enter to send quickly.",
            self.chat_container,
        )
        self.empty_state_label.setObjectName("chatEmptyState")
        self.empty_state_label.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_state_label.setWordWrap(True)
        self.chat_layout.addWidget(self.empty_state_label, 0, QtCore.Qt.AlignCenter)
        self.chat_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_container)
        return self.scroll_area

    def _build_input_bar(self) -> QtWidgets.QHBoxLayout:
        input_bar = QtWidgets.QHBoxLayout()
        input_bar.setSpacing(8)
        self.prompt_text_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_text_edit.setPlaceholderText("Message Annolid Bot…")
        self.prompt_text_edit.setFixedHeight(118)
        self.prompt_text_edit.setToolTip("Type a message. Use Ctrl+Enter to send.")
        input_bar.addWidget(self.prompt_text_edit, 1)

        side_buttons = QtWidgets.QVBoxLayout()
        side_buttons.setSpacing(8)
        self.send_button = QtWidgets.QToolButton(self)
        self.send_button.setObjectName("sendButton")
        self.send_button.setText("")
        self.send_button.setToolTip("Send message (Ctrl+Enter).")
        self.send_button.setFixedSize(42, 42)

        self.talk_button = QtWidgets.QToolButton(self)
        self.talk_button.setObjectName("talkButton")
        self.talk_button.setText("")
        self.talk_button.setToolTip("Record voice input.")
        self.talk_button.setFixedSize(36, 36)
        self._set_button_icon(
            self.send_button,
            QtWidgets.QStyle.SP_ArrowForward,
            "go-up",
        )
        self._set_button_icon(
            self.talk_button,
            QtWidgets.QStyle.SP_MediaPlay,
            "audio-input-microphone",
        )
        side_buttons.addWidget(self.send_button)
        side_buttons.addWidget(self.talk_button)
        side_buttons.addStretch(1)
        input_bar.addLayout(side_buttons)
        return input_bar

    def _build_composer_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame(self)
        panel.setObjectName("chatComposerPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addLayout(self._build_input_bar())
        layout.addLayout(self._build_provider_bar())
        layout.addLayout(self._build_share_bar())
        layout.addWidget(self._build_shared_image_label())
        layout.addWidget(self._build_prompt_meta_row())
        return panel

    def _build_quick_actions_row(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.quick_action_buttons: List[QtWidgets.QPushButton] = []
        self.quick_actions_layout = QtWidgets.QHBoxLayout()
        self.quick_actions_layout.setSpacing(6)
        row.addLayout(self.quick_actions_layout, 1)
        self.add_quick_action_button = QtWidgets.QToolButton(self)
        self.add_quick_action_button.setObjectName("chatComposerIconButton")
        self.add_quick_action_button.setText("+")
        self.add_quick_action_button.setToolTip("Add quick prompt")
        self.add_quick_action_button.clicked.connect(self._add_quick_action)
        row.addWidget(self.add_quick_action_button, 0)
        self.remove_quick_action_button = QtWidgets.QToolButton(self)
        self.remove_quick_action_button.setObjectName("chatComposerIconButton")
        self.remove_quick_action_button.setText("-")
        self.remove_quick_action_button.setToolTip("Remove selected quick prompt")
        self.remove_quick_action_button.clicked.connect(
            self._remove_selected_quick_action
        )
        row.addWidget(self.remove_quick_action_button, 0)
        self._refresh_quick_action_buttons()
        return row

    def _build_prompt_meta_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(self)
        row.setObjectName("promptMetaRow")
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.prompt_hint_label = QtWidgets.QLabel("Ctrl+Enter to send", row)
        self.prompt_hint_label.setObjectName("promptHintLabel")
        layout.addWidget(self.prompt_hint_label, 0)
        layout.addStretch(1)
        self.prompt_count_label = QtWidgets.QLabel("0/4000", row)
        self.prompt_count_label.setObjectName("promptCountLabel")
        layout.addWidget(self.prompt_count_label, 0)
        return row

    def _build_status_label(self) -> QtWidgets.QLabel:
        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setObjectName("chatStatusLabel")
        return self.status_label

    def _wire_ui_signals(self) -> None:
        self.provider_selector.currentIndexChanged.connect(self.on_provider_changed)
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        self.model_selector.editTextChanged.connect(self.on_model_text_edited)
        self.prompt_text_edit.textChanged.connect(self._on_prompt_text_changed)
        self.prompt_text_edit.installEventFilter(self)
        line_edit = self.model_selector.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self.on_model_editing_finished)
        self.configure_button.clicked.connect(self.open_llm_settings_dialog)
        self.send_button.clicked.connect(self.chat_with_model)
        self.share_canvas_button.clicked.connect(self._share_canvas_now)
        self.share_window_button.clicked.connect(self._share_window_now)
        self.talk_button.clicked.connect(self.toggle_recording)
        self.clear_chat_button.clicked.connect(self.clear_chat_conversation)
        self.sessions_button.clicked.connect(self.open_session_manager_dialog)
        self._set_button_icon(
            self.clear_chat_button,
            QtWidgets.QStyle.SP_BrowserStop,
            "edit-clear",
        )
        self._set_button_icon(
            self.sessions_button,
            QtWidgets.QStyle.SP_FileDialogDetailedView,
            "view-list-details",
        )
        self._set_button_icon(
            self.configure_button,
            QtWidgets.QStyle.SP_FileDialogInfoView,
            "preferences-system",
        )
        self._set_button_icon(
            self.share_canvas_button,
            QtWidgets.QStyle.SP_FileDialogNewFolder,
            "insert-image",
        )
        self._set_button_icon(
            self.share_window_button,
            QtWidgets.QStyle.SP_DesktopIcon,
            "window-new",
        )
        self._on_prompt_text_changed()

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
        self.remove_quick_action_button.setEnabled(
            self._selected_quick_action_index is not None
        )

    def _on_quick_action_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._quick_actions):
            return
        self._selected_quick_action_index = index
        for idx, btn in enumerate(self.quick_action_buttons):
            btn.setChecked(idx == index)
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

    @staticmethod
    def _set_button_icon(
        button: QtWidgets.QAbstractButton,
        style_icon: QtWidgets.QStyle.StandardPixmap,
        theme_icon: str = "",
    ) -> None:
        style = button.style()
        icon = style.standardIcon(style_icon)
        if icon.isNull() and theme_icon:
            icon = QtGui.QIcon.fromTheme(theme_icon)
        if not icon.isNull():
            button.setIcon(icon)

    def _set_bot_icon(self) -> None:
        icon = self._resolve_bot_icon()
        pixmap = icon.pixmap(QtCore.QSize(26, 26))
        if not pixmap.isNull():
            self.bot_icon_label.setPixmap(pixmap)

    def _resolve_bot_icon(self) -> QtGui.QIcon:
        icon_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "icons", "icon_annolid.png")
        )
        if os.path.exists(icon_path):
            icon = QtGui.QIcon(icon_path)
            if not icon.isNull():
                return icon
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        if icon.isNull():
            icon = QtGui.QIcon.fromTheme("applications-science")
        return icon

    def _on_prompt_text_changed(self) -> None:
        text = self.prompt_text_edit.toPlainText()
        total = len(text)
        self.prompt_count_label.setText(f"{total}/{self._max_prompt_chars}")
        self.prompt_count_label.setProperty(
            "limitReached", total >= self._max_prompt_chars
        )
        self.prompt_count_label.style().unpolish(self.prompt_count_label)
        self.prompt_count_label.style().polish(self.prompt_count_label)
        can_send = bool(text.strip()) and not self.is_streaming_chat
        self.send_button.setEnabled(can_send)

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
        dots = "." * ((self._typing_tick % 3) + 1)
        self._typing_tick += 1
        self.status_label.setText(f"{self._assistant_display_name()} is typing{dots}")

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self.prompt_text_edit
            and event.type() == QtCore.QEvent.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter)
            and bool(event.modifiers() & QtCore.Qt.ControlModifier)
        ):
            self.chat_with_model()
            return True
        return super().eventFilter(watched, event)

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
        try:
            self.setStyleSheet(
                """
                QWidget {{
                    background: #111317;
                    color: #e7e8ea;
                }}
                QComboBox {{
                    border: 1px solid #31353c;
                    border-radius: 10px;
                    background: #1a1d23;
                    color: #e7e8ea;
                    min-height: 24px;
                    padding: 3px 8px;
                }}
                QScrollArea {{
                    border: 1px solid #2f333b;
                    border-radius: 14px;
                    background: #151920;
                }}
                QPlainTextEdit {{
                    border: 1px solid #343943;
                    border-radius: 14px;
                    background: #1d2129;
                    padding: 10px;
                    font-size: 14px;
                }}
                QFrame#chatComposerPanel {{
                    border: 1px solid #2f333b;
                    border-radius: 22px;
                    background: #1a1d23;
                }}
                QToolButton#chatTopIconButton {{
                    border: 1px solid transparent;
                    border-radius: 14px;
                    background: transparent;
                    min-width: 28px;
                    min-height: 28px;
                }}
                QToolButton#chatTopIconButton:hover {{
                    background: #232832;
                    border-color: #343943;
                }}
                QLabel#chatTitleLabel {{
                    color: #f4f5f6;
                    font-size: 27px;
                    font-weight: 700;
                }}
                QLabel#chatSubtitleLabel {{
                    color: #9ea4af;
                    font-size: 11px;
                }}
                QLabel#chatBotIconLabel {{
                    border: 1px solid #353b47;
                    border-radius: 17px;
                    background: #1f2530;
                    padding: 2px;
                }}
                QPushButton:hover {{
                    background: #2b3038;
                }}
                QToolButton#chatComposerIconButton, QToolButton#talkButton {{
                    border: 1px solid #3a404b;
                    border-radius: 16px;
                    background: #252a33;
                    min-width: 32px;
                    min-height: 32px;
                }}
                QToolButton#chatComposerIconButton:hover, QToolButton#talkButton:hover {{
                    background: #313744;
                }}
                QPushButton#quickActionButton {{
                    border: 1px solid #373d47;
                    border-radius: 14px;
                    padding: 4px 12px;
                    font-size: 12px;
                    font-weight: 600;
                    background: #2a2f38;
                }}
                QPushButton#quickActionButton:hover {{
                    background: #363d49;
                }}
                QLabel#sharedImageLabel, QLabel#chatStatusLabel {{
                    color: #9ea4af;
                    font-size: 11px;
                }}
                QLabel#chatEmptyState {{
                    border: 1px dashed #3b4049;
                    border-radius: 14px;
                    background: #181c23;
                    color: #a9afba;
                    padding: 16px;
                    margin: 20px 12px;
                    font-size: 12px;
                }}
                QWidget#promptMetaRow {{
                    background: transparent;
                }}
                QLabel#promptHintLabel, QLabel#promptCountLabel {{
                    color: #8f96a2;
                    font-size: 11px;
                }}
                QLabel#promptCountLabel[limitReached="true"] {{
                    color: #ff6a57;
                    font-weight: 700;
                }}
                QCheckBox#chatInlineToggle {{
                    color: #9aa1ac;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QToolButton#sendButton {{
                    border: 1px solid #6f7684;
                    border-radius: 21px;
                    background: #8d939e;
                    color: #111317;
                    font-weight: 800;
                }}
                QToolButton#sendButton:disabled {{
                    border-color: #4b505a;
                    background: #4a4f58;
                    color: #7f8591;
                }}
                QToolButton#sendButton:hover {{
                    background: #a1a7b2;
                }}
                QFrame#chatBubble[role="user"] {{
                    background-color: #145C4C;
                    border-radius: 16px;
                    padding: 9px 11px;
                    border: 1px solid #1d7a65;
                }}
                QFrame#chatBubble[role="assistant"] {{
                    background-color: #1f252f;
                    border-radius: 16px;
                    padding: 9px 11px;
                    border: 1px solid #2f3946;
                }}
                QLabel#sender {{
                    color: #9ea4af;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QTextBrowser#messageView {{
                    color: #e9ebee;
                    font-size: 13px;
                    background: transparent;
                    border: none;
                    selection-background-color: #2f84ff;
                    selection-color: #f5f7fa;
                }}
                QFrame#chatBubble[role="user"] QLabel#sender {{
                    color: #C2E9DD;
                }}
                QFrame#chatBubble[role="user"] QTextBrowser#messageView {{
                    color: #F3FFF8;
                }}
                QFrame#chatBubble[role="user"] QLabel#meta {{
                    color: #C2E9DD;
                }}
                QFrame#chatBubble[role="user"] QPushButton#bubbleSpeakButton,
                QFrame#chatBubble[role="user"] QPushButton#bubbleCopyButton,
                QFrame#chatBubble[role="user"] QPushButton#bubbleRegenerateButton {{
                    border: 1px solid #2a8b74;
                    background: #196a57;
                    color: #F3FFF8;
                }}
                QFrame#chatBubble[role="user"] QPushButton#bubbleSpeakButton:hover,
                QFrame#chatBubble[role="user"] QPushButton#bubbleCopyButton:hover,
                QFrame#chatBubble[role="user"] QPushButton#bubbleRegenerateButton:hover {{
                    background: #1f7661;
                }}
                QFrame#chatBubble[progress="true"] {{
                    background-color: #181d27;
                    border: 1px solid #2a3140;
                }}
                QFrame#chatBubble[progress="true"] QLabel#sender {{
                    color: #8db7ff;
                }}
                QFrame#chatBubble[progress="true"] QTextBrowser#messageView {{
                    color: #bfcae2;
                    font-size: 12px;
                }}
                QLabel#meta {{
                    color: #8e95a1;
                    font-size: 10px;
                }}
                QPushButton#bubbleSpeakButton, QPushButton#bubbleCopyButton, QPushButton#bubbleRegenerateButton {{
                    border: 1px solid #424855;
                    border-radius: 6px;
                    background: #2a3039;
                    padding: 2px 6px;
                    min-height: 18px;
                    font-size: 10px;
                }}
                QPushButton#bubbleSpeakButton:hover, QPushButton#bubbleCopyButton:hover, QPushButton#bubbleRegenerateButton:hover {{
                    background: #353c48;
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

    def _iter_chat_bubbles(self) -> List[_ChatBubble]:
        bubbles: List[_ChatBubble] = []
        for idx in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(idx)
            if item is None:
                continue
            row_layout = None
            row_widget = item.widget()
            if row_widget is not None:
                row_layout = row_widget.layout()
            if row_layout is None:
                row_layout = item.layout()
            if row_layout is None:
                continue
            for j in range(row_layout.count()):
                widget = row_layout.itemAt(j).widget()
                if isinstance(widget, _ChatBubble):
                    bubbles.append(widget)
        return bubbles

    def _bubble_max_width(self) -> int:
        viewport = self.scroll_area.viewport() if self.scroll_area is not None else None
        base_width = (
            int(viewport.width()) if viewport is not None else int(self.width())
        )
        return max(320, int(base_width) - 12)

    def _reflow_chat_bubbles(self) -> None:
        max_width = self._bubble_max_width()
        text_width = max_width - 24
        for bubble in self._iter_chat_bubbles():
            bubble.apply_layout_width(text_width, max_width)

    def _add_bubble(
        self,
        sender: str,
        text: str,
        *,
        is_user: bool,
        allow_regenerate: bool = False,
    ) -> _ChatBubble:
        # Use a row widget (not just a bare layout) so the row reliably expands
        # to the full scroll area width, letting bubbles fill the dock width.
        row_widget = QtWidgets.QWidget(self.chat_container)
        row_widget.setObjectName("chatBubbleRow")
        row_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        row = QtWidgets.QHBoxLayout(row_widget)
        row.setContentsMargins(6, 1, 6, 1)
        row.setSpacing(0)
        bubble = _ChatBubble(
            sender,
            text,
            is_user=is_user,
            on_speak=self.speak_text_async,
            on_copy=self._copy_message_text,
            on_regenerate=self._regenerate_from_bubble,
            allow_regenerate=allow_regenerate,
            parent=self.chat_container,
        )
        bubble.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        bubble.apply_layout_width(
            self._bubble_max_width() - 24, self._bubble_max_width()
        )
        if is_user:
            row.addStretch(1)
            row.addWidget(bubble, 0, QtCore.Qt.AlignRight)
        else:
            row.addWidget(bubble, 0, QtCore.Qt.AlignLeft)
            row.addStretch(1)

        # Insert before trailing stretch item.
        insert_idx = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertWidget(insert_idx, row_widget)
        self._reflow_chat_bubbles()
        self._update_empty_state_visibility()
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        return bubble

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)

    def _copy_message_text(self, text: str) -> None:
        clipboard = QtGui.QGuiApplication.clipboard()
        clipboard.setText(str(text or ""))
        self.status_label.setText("Message copied.")

    def _regenerate_from_bubble(self, _text: str) -> None:
        prompt = str(self._last_user_prompt or "").strip()
        if not prompt:
            self.status_label.setText("No prompt available to regenerate.")
            return
        if self.is_streaming_chat:
            self.status_label.setText("Wait for current response to finish.")
            return
        self.prompt_text_edit.setPlainText(prompt)
        self.chat_with_model()

    def _assistant_display_name(self) -> str:
        model_name = str(self.selected_model or "").strip()
        if model_name:
            return f"Annolid Bot ({model_name})"
        return "Annolid Bot"

    def _ensure_provider_ready(self) -> bool:
        kind = provider_kind(self.llm_settings, self.selected_provider)
        if kind in {"openai_compat", "gemini"} and not has_provider_api_key(
            self.llm_settings, self.selected_provider
        ):
            label = self.provider_labels.get(
                self.selected_provider, self.selected_provider
            )
            QtWidgets.QMessageBox.warning(
                self,
                f"{label} API key required",
                f"Please configure an API key for {label} in AI Model Settings.",
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
        self._populate_provider_selector()
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
        self.selected_provider = self.provider_selector.currentData() or new_provider
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
            self._load_session_history_into_bubbles(self.session_id)
            self._refresh_header_chips()

    def _refresh_header_chips(self) -> None:
        provider_text = self.provider_labels.get(
            self.selected_provider, str(self.selected_provider or "unknown")
        )
        model_text = str(self.selected_model or "unknown")
        session_text = str(self.session_id or "default")
        self.chat_title_label.setText(model_text)
        self.session_chip_label.setText(f"{provider_text} · {session_text}")

    def clear_chat_conversation(self) -> None:
        if self.is_streaming_chat:
            self.status_label.setText("Wait for current response to finish.")
            return
        self._session_store.clear_session(self.session_id)
        clear_chat_session(self.session_id)
        self._clear_chat_bubbles()
        self.status_label.setText("Conversation cleared.")
        self._refresh_header_chips()

    def _clear_chat_bubbles(self) -> None:
        while self.chat_layout.count() > 2:
            item = self.chat_layout.takeAt(1)
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
        self._update_empty_state_visibility()
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _update_empty_state_visibility(self) -> None:
        bubble_count = 0
        for idx in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(idx)
            if item is None:
                continue
            row = item.layout()
            if row is None:
                continue
            for j in range(row.count()):
                widget = row.itemAt(j).widget()
                if isinstance(widget, _ChatBubble):
                    bubble_count += 1
        self.empty_state_label.setVisible(bubble_count == 0)

    def _load_session_history_into_bubbles(self, session_id: str) -> None:
        self._clear_chat_bubbles()
        try:
            history = self._session_store.get_history(str(session_id or ""))
        except Exception:
            history = []
        for msg in history[-80:]:
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                self._add_bubble("You", content, is_user=True)
            elif role == "assistant":
                self._add_bubble(self._assistant_display_name(), content, is_user=False)
            elif role == "system":
                self._add_bubble("System", content, is_user=False)

    def open_session_manager_dialog(self) -> None:
        dialog = ChatSessionManagerDialog(
            session_manager=self._session_manager,
            session_store=self._session_store,
            active_session_id=self.session_id,
            parent=self,
        )
        dialog.sessionSwitched.connect(self._on_session_switched)
        dialog.sessionCreated.connect(self._on_session_created)
        dialog.sessionCleared.connect(self._on_session_cleared)
        dialog.sessionDeleted.connect(self._on_session_deleted)
        dialog.exec_()
        self._refresh_header_chips()

    @QtCore.Slot(str)
    def _on_session_switched(self, session_id: str) -> None:
        key = str(session_id or "").strip()
        if not key:
            return
        self.set_session_id(key)
        self.status_label.setText(f"Switched to session: {key}")

    @QtCore.Slot(str)
    def _on_session_created(self, session_id: str) -> None:
        key = str(session_id or "").strip()
        if not key:
            return
        self.set_session_id(key)
        self.status_label.setText(f"Created session: {key}")
        self._refresh_header_chips()

    @QtCore.Slot(str)
    def _on_session_cleared(self, session_id: str) -> None:
        key = str(session_id or "").strip()
        if not key:
            return
        if key == self.session_id:
            self._load_session_history_into_bubbles(key)
        self.status_label.setText(f"Cleared session: {key}")
        self._refresh_header_chips()

    @QtCore.Slot(str)
    def _on_session_deleted(self, session_id: str) -> None:
        key = str(session_id or "").strip()
        if not key:
            return
        if key == self.session_id:
            self._clear_chat_bubbles()
        self.status_label.setText(f"Deleted session: {key}")
        self._refresh_header_chips()

    def set_canvas(self, canvas: Optional[QtWidgets.QWidget]) -> None:
        self.canvas_widget = canvas

    def set_host_window(self, window: Optional[QtWidgets.QWidget]) -> None:
        self.host_window_widget = window

    def get_bot_action_result(self, action: str) -> Dict[str, Any]:
        payload = self._bot_action_results.get(str(action), {})
        return dict(payload or {})

    def _canvas_pixmap_ready(self) -> bool:
        host = self.host_window_widget or self.window()
        canvas = getattr(host, "canvas", None)
        pixmap = getattr(canvas, "pixmap", None)
        return bool(
            pixmap is not None and hasattr(pixmap, "isNull") and not pixmap.isNull()
        )

    def _wait_for_canvas_pixmap(self, timeout_ms: int = 3000) -> bool:
        deadline = QtCore.QDeadlineTimer(int(max(1, timeout_ms)))
        while not deadline.hasExpired():
            if self._canvas_pixmap_ready():
                return True
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        return self._canvas_pixmap_ready()

    @staticmethod
    def _normalize_behavior_labels(labels: List[str]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for raw in labels:
            value = str(raw or "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        return normalized

    def _labels_from_schema_or_flags(self) -> List[str]:
        host = self.host_window_widget or self.window()
        labels: List[str] = []
        schema = getattr(host, "project_schema", None)
        behaviors = getattr(schema, "behaviors", None)
        if isinstance(behaviors, list):
            for behavior in behaviors:
                code = str(getattr(behavior, "code", "") or "").strip()
                if code:
                    labels.append(code)
        if not labels:
            flags = getattr(host, "flags", None)
            if isinstance(flags, dict):
                labels.extend([str(k).strip() for k in flags.keys() if str(k).strip()])
        return self._normalize_behavior_labels(labels)

    @staticmethod
    def _behavior_ranges_from_events(events: List[object]) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, List[object]] = {}
        for event in events:
            behavior = str(getattr(event, "behavior", "") or "").strip()
            if not behavior:
                continue
            subject = getattr(event, "subject", None)
            key = (behavior, subject)
            grouped.setdefault(key, []).append(event)
        ranges: List[Dict[str, Any]] = []
        for (behavior, subject), group in grouped.items():
            group_sorted = sorted(
                group,
                key=lambda evt: (
                    int(getattr(evt, "frame", 0)),
                    str(getattr(evt, "event", "")),
                ),
            )
            open_frame: Optional[int] = None
            for evt in group_sorted:
                label = str(getattr(evt, "event", "") or "").strip().lower()
                frame = int(getattr(evt, "frame", 0))
                if label == "start":
                    open_frame = frame
                elif label == "end":
                    start = frame if open_frame is None else open_frame
                    end = frame
                    if end < start:
                        start, end = end, start
                    ranges.append(
                        {
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "subject": subject,
                            "seed_behavior": behavior,
                        }
                    )
                    open_frame = None
            if open_frame is not None:
                ranges.append(
                    {
                        "start_frame": int(open_frame),
                        "end_frame": int(open_frame),
                        "subject": subject,
                        "seed_behavior": behavior,
                    }
                )
        return sorted(
            ranges, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]))
        )

    @staticmethod
    def _extract_label_from_model_text(
        text: str, labels: List[str]
    ) -> tuple[str, float]:
        raw = str(text or "").strip()
        if not labels:
            return ("Agent", 0.0)

        # Prefer structured JSON responses if present, including fenced blocks.
        json_candidate = ""
        if raw.startswith("{"):
            json_candidate = raw
        else:
            fence_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fence_match:
                json_candidate = str(fence_match.group(1) or "").strip()
            else:
                brace_match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
                if brace_match:
                    json_candidate = str(brace_match.group(1) or "").strip()
        if json_candidate:
            try:
                payload = json.loads(json_candidate)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                label_val = str(payload.get("label") or "").strip()
                conf_val = payload.get("confidence", 0.0)
                try:
                    conf = float(conf_val)
                except Exception:
                    conf = 0.0
                if label_val:
                    for candidate in labels:
                        if candidate.lower() == label_val.lower():
                            return candidate, max(0.0, min(1.0, conf))

        lowered = raw.lower()
        for candidate in labels:
            if candidate.lower() in lowered:
                return candidate, 0.6
        return labels[0], 0.2

    def _save_behavior_timestamps_csv(self, host: object) -> Dict[str, Any]:
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is None:
            return {"ok": False, "error": "Behavior controller unavailable."}
        video_file = str(getattr(host, "video_file", "") or "").strip()
        if not video_file:
            return {"ok": False, "error": "No video file is loaded."}

        video_path = Path(video_file)
        output_path = video_path.with_name(f"{video_path.stem}_timestamps.csv")
        rows = behavior_controller.export_rows(
            timestamp_fallback=lambda evt: self._estimate_recording_time(evt.frame)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["Trial time", "Recording time", "Subject", "Behavior", "Event"]
            )
            for row in rows:
                writer.writerow(row)
        return {"ok": True, "path": str(output_path), "rows": len(rows)}

    @QtCore.Slot(str)
    def bot_open_video(self, video_path: str) -> None:
        path = str(video_path or "").strip()
        if not path:
            self.status_label.setText("Bot action failed: empty video path.")
            return
        host = self.host_window_widget or self.window()
        open_video = getattr(host, "openVideo", None)
        if not callable(open_video):
            self.status_label.setText("Bot action failed: video loader is unavailable.")
            return
        try:
            open_video(from_video_list=True, video_path=path, programmatic_call=True)
            self.status_label.setText(f"Opened video: {os.path.basename(path)}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    def _set_bot_action_result(self, *args: Any, **kwargs: Any) -> None:
        """
        Backward-compatible bot action result setter.
        Supports both call styles:
        - _set_bot_action_result(payload_dict)
        - _set_bot_action_result(action_name, payload_dict)
        """
        # Preferred path: two positional args => (action, payload)
        action = ""
        payload: Dict[str, Any] = {}
        if len(args) >= 2:
            action = str(args[0] or "").strip()
            payload = dict(args[1] or {})
        elif len(args) == 1:
            value = args[0]
            if isinstance(value, dict):
                payload = dict(value or {})
                action = str(payload.get("action") or "").strip()
            else:
                action = str(value or "").strip()
                payload = dict(kwargs.get("result") or {})
        else:
            action = str(kwargs.get("action") or "").strip()
            payload = dict(kwargs.get("result") or kwargs.get("payload") or {})

        if action:
            self._bot_action_results[action] = dict(payload)
        self._bot_action_result = dict(payload)

    def _resolve_web_manager(self):
        host = self.host_window_widget or self.window()
        return getattr(host, "web_manager", None)

    def _resolve_pdf_manager(self):
        host = self.host_window_widget or self.window()
        return getattr(host, "pdf_manager", None)

    @QtCore.Slot(str)
    def bot_open_pdf(self, pdf_path: str = "") -> None:
        host = self.host_window_widget or self.window()
        pdf_import_widget = getattr(host, "pdf_import_widget", None)
        open_pdf = getattr(pdf_import_widget, "open_pdf", None)
        open_pdf_path = getattr(pdf_import_widget, "open_pdf_path", None)
        if not callable(open_pdf):
            self.status_label.setText("Bot action failed: PDF opener is unavailable.")
            return
        try:
            resolved = str(pdf_path or "").strip()
            if resolved and callable(open_pdf_path):
                ok = bool(open_pdf_path(resolved))
                if ok:
                    self.status_label.setText(
                        f"Opened PDF: {os.path.basename(resolved)}"
                    )
                else:
                    self.status_label.setText("Bot action failed: could not open PDF.")
                return
            # Reuse the exact same flow as File > Open PDF...
            open_pdf()
            self.status_label.setText("Opened PDF picker.")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_open_url(self, url: str) -> None:
        target_url = str(url or "").strip()
        if not target_url:
            self.status_label.setText("Bot action failed: empty URL.")
            return
        parsed = QtCore.QUrl(target_url)
        if parsed.isValid() and parsed.scheme().lower() in {"http", "https", "file"}:
            normalized = parsed.toString()
        else:
            local_path = Path(target_url).expanduser()
            if local_path.exists() and local_path.is_file():
                normalized = QtCore.QUrl.fromLocalFile(str(local_path)).toString()
                parsed = QtCore.QUrl(normalized)
            else:
                self.status_label.setText(
                    "Bot action failed: invalid URL or file path."
                )
                return
        host = self.host_window_widget or self.window()
        show_web = getattr(host, "show_web_in_viewer", None)
        if callable(show_web) and bool(show_web(normalized)):
            self.status_label.setText(f"Opened URL in canvas: {normalized}")
            return
        opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Bot action failed: could not open URL.")

    @QtCore.Slot(str)
    def bot_open_in_browser(self, url: str) -> None:
        target_url = str(url or "").strip()
        if not target_url:
            self.status_label.setText("Bot action failed: empty URL.")
            return
        parsed = QtCore.QUrl(target_url)
        if not parsed.isValid() or parsed.scheme().lower() not in {"http", "https"}:
            self.status_label.setText("Bot action failed: invalid URL.")
            return
        opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            self.status_label.setText(f"Opened URL in browser: {target_url}")
            return
        self.status_label.setText("Bot action failed: could not open URL.")

    @QtCore.Slot(int)
    def bot_web_get_dom_text(self, max_chars: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.get_page_text(max_chars=int(max_chars))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Captured page text." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_web_get_state(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.get_web_state()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)

    @QtCore.Slot()
    def bot_pdf_get_state(self) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_state()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)

    @QtCore.Slot(int, int)
    def bot_pdf_get_text(self, max_chars: int, pages: int) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_text(max_chars=int(max_chars), pages=int(pages))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Captured PDF text." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int, int)
    def bot_pdf_find_sections(self, max_sections: int, max_pages: int) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_sections(
                max_sections=int(max_sections),
                max_pages=int(max_pages),
            )
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Detected PDF sections." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(str)
    def bot_web_click(self, selector: str) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.click_selector(str(selector or ""))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Clicked selector." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(str, str, bool)
    def bot_web_type(self, selector: str, text: str, submit: bool) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.type_selector(
                str(selector or ""), str(text or ""), submit=bool(submit)
            )
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Typed into selector." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int)
    def bot_web_scroll(self, delta_y: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.scroll_by(delta_y=int(delta_y))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Scrolled page." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_web_find_forms(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result(payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.find_forms()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result(payload)
        self.status_label.setText(
            "Inspected forms." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int)
    def bot_set_frame(self, frame_index: int) -> None:
        host = self.host_window_widget or self.window()
        setter = getattr(host, "set_frame_number", None)
        if not callable(setter):
            self.status_label.setText(
                "Bot action failed: frame navigation unavailable."
            )
            return
        try:
            setter(int(frame_index))
            self.status_label.setText(f"Moved to frame {int(frame_index)}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_set_chat_prompt(self, text: str) -> None:
        value = str(text or "").strip()
        self.prompt_text_edit.setPlainText(value)
        self.prompt_text_edit.setFocus()
        self.status_label.setText("Chat prompt updated by bot.")

    @QtCore.Slot()
    def bot_send_chat_prompt(self) -> None:
        if self.is_streaming_chat:
            self.status_label.setText("Wait for current response to finish.")
            return
        self.chat_with_model()

    @QtCore.Slot(str, str)
    def bot_set_chat_model(self, provider: str, model: str) -> None:
        self.set_provider_and_model(str(provider or ""), str(model or ""))
        self.status_label.setText("Bot chat provider/model updated.")

    @QtCore.Slot(str)
    def bot_select_annotation_model(self, model_name: str) -> None:
        value = str(model_name or "").strip()
        if not value:
            self.status_label.setText("Bot action failed: empty model name.")
            return
        host = self.host_window_widget or self.window()
        combo = getattr(host, "_selectAiModelComboBox", None)
        if combo is None:
            self.status_label.setText("Bot action failed: model selector unavailable.")
            return
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentText(value)
        self.status_label.setText(f"Selected annotation model: {value}")

    @QtCore.Slot(int)
    def bot_track_next_frames(self, to_frame: int) -> None:
        host = self.host_window_widget or self.window()
        run_track = getattr(host, "predict_from_next_frame", None)
        if not callable(run_track):
            self.status_label.setText("Bot action failed: tracking is unavailable.")
            return
        try:
            run_track(to_frame=int(to_frame))
            self.status_label.setText(f"Started tracking to frame {int(to_frame)}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

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

    @QtCore.Slot(str, str, str, int, int, str, bool, str, str, str)
    def bot_label_behavior_segments(
        self,
        video_path: str = "",
        behavior_labels_csv: str = "",
        segment_mode: str = "timeline",
        segment_frames: int = 60,
        max_segments: int = 120,
        subject: str = "Agent",
        overwrite_existing: bool = False,
        llm_profile: str = "",
        llm_provider: str = "",
        llm_model: str = "",
    ) -> None:
        self._set_bot_action_result(
            "label_behavior_segments",
            {"ok": False, "error": "Labeling did not complete."},
        )
        host = self.host_window_widget or self.window()
        open_video = getattr(host, "openVideo", None)
        set_frame = getattr(host, "set_frame_number", None)
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is None or not callable(set_frame):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior timeline APIs are unavailable."},
            )
            self.status_label.setText(
                "Bot action failed: behavior timeline unavailable."
            )
            return

        try:
            video_text = str(video_path or "").strip()
            if video_text:
                if not callable(open_video):
                    raise RuntimeError("Video opening is unavailable.")
                open_video(
                    from_video_list=True,
                    video_path=video_text,
                    programmatic_call=True,
                )
            if not self._wait_for_canvas_pixmap(timeout_ms=4000):
                raise RuntimeError("Timed out waiting for video frame.")

            labels = self._normalize_behavior_labels(
                [
                    p.strip()
                    for p in str(behavior_labels_csv or "").split(",")
                    if p.strip()
                ]
            )
            if not labels:
                labels = self._labels_from_schema_or_flags()
            if not labels:
                raise RuntimeError(
                    "No behavior labels provided/found. Define behavior schema or pass labels."
                )

            mode = str(segment_mode or "timeline").strip().lower()
            total_frames = int(getattr(host, "num_frames", 0) or 0)
            if total_frames <= 0:
                raise RuntimeError("No video is loaded.")
            max_segments = max(1, int(max_segments))
            segment_frames = max(1, int(segment_frames))

            intervals: List[Dict[str, Any]] = []
            if mode == "timeline":
                events = list(behavior_controller.iter_events())
                intervals = self._behavior_ranges_from_events(events)
            if not intervals:
                mode = "uniform"
                last = total_frames - 1
                start = 0
                while start <= last:
                    end = min(last, start + segment_frames - 1)
                    intervals.append(
                        {
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "subject": None,
                            "seed_behavior": None,
                        }
                    )
                    start = end + 1
            intervals = intervals[:max_segments]
            if not intervals:
                raise RuntimeError("No segments available for labeling.")

            from annolid.core.models.adapters.llm_chat import LLMChatAdapter
            from annolid.core.models.base import ModelRequest

            adapter = LLMChatAdapter(
                profile=str(llm_profile or "").strip() or None,
                provider=str(llm_provider or "").strip() or None,
                model=str(llm_model or "").strip() or None,
                persist=False,
            )
            predictions: List[Dict[str, Any]] = []
            inference_cache: Dict[int, tuple[str, float]] = {}
            skipped_segments = 0
            label_options = ", ".join(labels)
            with adapter:
                for item in intervals:
                    start_frame = int(item["start_frame"])
                    end_frame = int(item["end_frame"])
                    mid = int((start_frame + end_frame) // 2)
                    cached = inference_cache.get(mid)
                    if cached is None:
                        set_frame(mid)
                        if not self._wait_for_canvas_pixmap(timeout_ms=1200):
                            skipped_segments += 1
                            continue
                        image_path = self._snapshot_canvas_to_tempfile()
                        if not image_path:
                            skipped_segments += 1
                            continue
                        try:
                            prompt = (
                                "Classify behavior in this video segment. "
                                f"Choose exactly one label from: {label_options}. "
                                "Return strict JSON only: "
                                '{"label":"<one label>","confidence":0.0}'
                            )
                            resp = adapter.predict(
                                ModelRequest(
                                    task="caption",
                                    image_path=image_path,
                                    text=prompt,
                                    params={"temperature": 0.0, "max_tokens": 90},
                                )
                            )
                            raw = str(
                                resp.text or (resp.output or {}).get("text") or ""
                            ).strip()
                            cached = self._extract_label_from_model_text(raw, labels)
                            inference_cache[mid] = cached
                        except Exception:
                            skipped_segments += 1
                            continue
                        finally:
                            try:
                                if image_path and os.path.exists(image_path):
                                    os.remove(image_path)
                            except OSError:
                                pass
                    label, confidence = cached
                    predictions.append(
                        {
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "subject": item.get("subject"),
                            "label": label,
                            "confidence": confidence,
                        }
                    )

            if not predictions:
                raise RuntimeError(
                    "Model did not return usable labels for any segment."
                )

            if bool(overwrite_existing):
                behavior_controller.clear_behavior_data()

            def _timestamp_provider(frame: int) -> Optional[float]:
                fps = getattr(host, "fps", None)
                if fps is None or float(fps) <= 0:
                    return None
                return float(frame) / float(fps)

            default_subject = str(subject or "").strip() or None
            for pred in predictions:
                behavior_controller.create_interval(
                    behavior=str(pred["label"]),
                    start_frame=int(pred["start_frame"]),
                    end_frame=int(pred["end_frame"]),
                    subject=pred.get("subject") or default_subject,
                    timestamp_provider=_timestamp_provider,
                )

            refresh_log = getattr(host, "_refresh_behavior_log", None)
            if callable(refresh_log):
                refresh_log()
            timeline_panel = getattr(host, "timeline_panel", None)
            refresh_timeline = getattr(
                timeline_panel, "refresh_from_behavior_controller", None
            )
            if callable(refresh_timeline):
                refresh_timeline()
            timestamp_result = self._save_behavior_timestamps_csv(host)
            if not bool(timestamp_result.get("ok", False)):
                raise RuntimeError(
                    str(
                        timestamp_result.get("error")
                        or "Failed to save behavior timestamps."
                    )
                )

            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": True,
                    "mode": mode,
                    "labeled_segments": len(predictions),
                    "evaluated_segments": len(intervals),
                    "skipped_segments": int(skipped_segments),
                    "labels_used": labels,
                    "timestamps_csv": str(timestamp_result.get("path") or ""),
                    "timestamps_rows": int(timestamp_result.get("rows") or 0),
                },
            )
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s); saved timestamps to "
                f"{Path(str(timestamp_result.get('path') or '')).name}."
            )
        except Exception as exc:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": str(exc)},
            )
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str, str, str, float, str, bool, float, int)
    def bot_start_realtime_stream(
        self,
        camera_source: str = "",
        model_name: str = "",
        target_behaviors_csv: str = "",
        confidence_threshold: float = -1.0,
        viewer_type: str = "threejs",
        classify_eye_blinks: bool = False,
        blink_ear_threshold: float = -1.0,
        blink_min_consecutive_frames: int = -1,
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
                    "camera_source": str(camera_value),
                    "viewer_type": str(extras["viewer_type"]),
                    "classify_eye_blinks": bool(classify_eye_blinks),
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

        self._last_user_prompt = raw_prompt
        chat_image_path = self._prepare_chat_image()
        self._add_bubble("You", raw_prompt, is_user=True)
        assistant_name = self._assistant_display_name()
        self._current_progress_bubble = self._add_bubble(
            f"{assistant_name} • Thinking",
            "Preparing response...",
            is_user=False,
        )
        self._current_progress_bubble.setProperty("progress", True)
        self._current_progress_bubble.style().unpolish(self._current_progress_bubble)
        self._current_progress_bubble.style().polish(self._current_progress_bubble)
        self._progress_lines = ["Preparing response..."]
        self._current_response_bubble = self._add_bubble(
            assistant_name,
            "",
            is_user=False,
            allow_regenerate=True,
        )

        self.prompt_text_edit.clear()
        self.send_button.setEnabled(False)
        self.is_streaming_chat = True
        self._start_typing_indicator()

        task = StreamingChatTask(
            prompt=raw_prompt,
            image_path=chat_image_path or self.image_path,
            widget=self,
            model=self.selected_model,
            provider=self.selected_provider,
            settings=self.llm_settings,
            session_id=self.session_id,
            show_tool_trace=self.tool_trace_checkbox.isChecked(),
            enable_web_tools=self.allow_web_tools_checkbox.isChecked(),
        )
        self.thread_pool.start(task)

    @QtCore.Slot(str)
    def stream_chat_chunk(self, chunk: str) -> None:
        if self._current_response_bubble is None:
            return
        self._current_response_bubble.append_text(chunk)
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    @QtCore.Slot(str)
    def stream_chat_progress(self, update: str) -> None:
        line = str(update or "").strip()
        if not line:
            return
        if self._current_progress_bubble is None:
            self._current_progress_bubble = self._add_bubble(
                f"{self._assistant_display_name()} • Thinking",
                line,
                is_user=False,
            )
            self._current_progress_bubble.setProperty("progress", True)
            self._current_progress_bubble.style().unpolish(
                self._current_progress_bubble
            )
            self._current_progress_bubble.style().polish(self._current_progress_bubble)
            self._progress_lines = [line]
        elif not self._progress_lines or self._progress_lines[-1] != line:
            self._progress_lines.append(line)
        if self._current_progress_bubble is not None:
            compact = self._progress_lines[-8:]
            text = "\n".join(f"- {item}" for item in compact)
            self._current_progress_bubble.set_text(text)
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

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

        self.is_streaming_chat = False
        self._stop_typing_indicator()
        self._on_prompt_text_changed()
        self._current_response_bubble = None
        self._current_progress_bubble = None
        self._progress_lines = []
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

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
