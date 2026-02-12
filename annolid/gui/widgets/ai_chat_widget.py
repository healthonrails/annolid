from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QThreadPool

from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
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

    def append_text(self, chunk: str) -> None:
        self.message_label.setText(self.message_label.text() + chunk)

    def set_text(self, text: str) -> None:
        self.message_label.setText(text)

    def text(self) -> str:
        return self.message_label.text()

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
        self.session_id = "gui:annolid_bot:default"
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
        share_bar.addWidget(self.attach_canvas_checkbox)
        share_bar.addWidget(self.attach_window_checkbox)
        share_bar.addWidget(self.tool_trace_checkbox)

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
        actions = [
            ("Summarize Context", "Summarize what we are doing in this session."),
            ("Help Me Track", "Help me track the next frames in this video."),
            ("Review Memory", "Review long-term memory and list key facts."),
        ]
        for label, prompt in actions:
            btn = QtWidgets.QPushButton(label, self)
            btn.setObjectName("quickActionButton")
            btn.setToolTip(prompt)
            btn.clicked.connect(
                lambda _checked=False, p=prompt: self._apply_quick_action(p)
            )
            self.quick_action_buttons.append(btn)
            row.addWidget(btn, 0)
        row.addStretch(1)
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
        self.prompt_text_edit.setPlainText(str(text or "").strip())
        self.prompt_text_edit.setFocus()
        self._on_prompt_text_changed()

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
                    background-color: #2a2f39;
                    border-radius: 14px;
                    padding: 8px 10px;
                    border: 1px solid #3a404b;
                }}
                QFrame#chatBubble[role="assistant"] {{
                    background-color: #1c2128;
                    border-radius: 14px;
                    padding: 8px 10px;
                    border: 1px solid #303640;
                }}
                QLabel#sender {{
                    color: #9ea4af;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QLabel#message {{
                    color: #e9ebee;
                    font-size: 13px;
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

    def _add_bubble(
        self,
        sender: str,
        text: str,
        *,
        is_user: bool,
        allow_regenerate: bool = False,
    ) -> _ChatBubble:
        row = QtWidgets.QHBoxLayout()
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
        if is_user:
            row.addStretch(1)
            row.addWidget(bubble, 0, QtCore.Qt.AlignRight)
        else:
            row.addWidget(bubble, 0, QtCore.Qt.AlignLeft)
            row.addStretch(1)

        # Insert before trailing stretch item.
        insert_idx = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertLayout(insert_idx, row)
        self._update_empty_state_visibility()
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        return bubble

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

        self.is_streaming_chat = False
        self._stop_typing_indicator()
        self._on_prompt_text_changed()
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
