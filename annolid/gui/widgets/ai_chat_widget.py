from __future__ import annotations

import contextlib
import logging
import sys as sys
from collections import OrderedDict
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from qtpy import QtCore, QtGui, QtWidgets
from annolid.gui.qt_compat import configure_compact_combo, palette_color_role
from qtpy.QtCore import QThreadPool

from annolid.infrastructure.agent_workspace import (
    get_agent_workspace_path as get_agent_workspace_path,
)
from annolid.services.chat_backend_support import (
    ERROR_TYPE_CANCELLED as ERROR_TYPE_CANCELLED,
    TURN_STATUS_CANCELLED as TURN_STATUS_CANCELLED,
)
from annolid.services.chat_bus import (
    InboundMessage as InboundMessage,
    MessageBus,
    OutboundMessage as OutboundMessage,
    decode_outbound_chat_event as decode_outbound_chat_event,
)
from annolid.services.chat_session import (
    AgentSessionManager,
    get_chat_session_store,
)
from annolid.gui.widgets.ai_chat_audio_controller import ChatAudioController
from annolid.gui.widgets.ai_chat_backend import (
    StreamingChatTask as StreamingChatTask,
    clear_chat_session,
)
from annolid.gui.widgets.ai_chat_actions_mixin import (
    AIChatActionsMixin,
    _is_path_within_root as _is_path_within_root,
    _is_probably_binary_file as _is_probably_binary_file,
    _log_targets_for_bot as _log_targets_for_bot,
    _safe_stream_source_for_bot as _safe_stream_source_for_bot,
    _tail_text_from_file as _tail_text_from_file,
)
from annolid.gui.widgets.ai_chat_behavior_mixin import AIChatBehaviorMixin
from annolid.gui.widgets.ai_chat_composer_mixin import (
    AIChatComposerMixin,
    _CHAT_IMAGE_SUFFIXES as _CHAT_IMAGE_SUFFIXES,
    get_config_path as get_config_path,
    load_config as load_config,
)
from annolid.gui.widgets.ai_chat_slash_mixin import (
    AIChatSlashMixin,
    _compose_slash_selection_draft as _compose_slash_selection_draft,
    _dedupe_preserve_order as _dedupe_preserve_order,
    _extract_slash_selection_state as _extract_slash_selection_state,
    _split_slash_value_list as _split_slash_value_list,
)
from annolid.gui.widgets.ai_chat_streaming_mixin import AIChatStreamingMixin
from annolid.gui.widgets.ai_chat_theme import build_chat_widget_stylesheet
from annolid.gui.widgets.ai_chat_viewer_mixin import AIChatViewerCitationsMixin
from annolid.gui.widgets.ai_chat_bubble import (
    _ChatBubble,
    _emoji_supported as _emoji_supported,
    _symbol_text,
)
from annolid.gui.widgets.ai_chat_session_dialog import ChatSessionManagerDialog
from annolid.gui.widgets.ai_chat_zulip import (
    _ZulipSendTask as _ZulipSendTask,
    build_zulip_draft_target as build_zulip_draft_target,
    missing_zulip_config_fields as missing_zulip_config_fields,
)
from annolid.gui.widgets.behavior_slash_dialog import (
    BehaviorSlashDialog as BehaviorSlashDialog,
)
from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog
from annolid.gui.widgets.provider_registry import ProviderRegistry
from annolid.gui.widgets.provider_runtime_sync import (
    refresh_runtime_llm_settings as refresh_runtime_provider_settings,
)
from annolid.gui.widgets.track_slash_dialog import (
    TrackSlashDialog,
    build_track_slash_command,
)
from annolid.gui.workers import FlexibleWorker
from annolid.utils.llm_provider_catalog import (
    inspect_provider_credential_status,
    is_loopback_base_url,
)
from annolid.utils.llm_settings import (
    has_provider_api_key,
    load_llm_settings,
    provider_kind,
    save_llm_settings,
)

logger = logging.getLogger(__name__)


class AIChatWidget(
    AIChatStreamingMixin,
    AIChatComposerMixin,
    AIChatViewerCitationsMixin,
    AIChatSlashMixin,
    AIChatBehaviorMixin,
    AIChatActionsMixin,
    QtWidgets.QWidget,
):
    """Annolid Bot chat UI for local/cloud models with visual sharing and streaming."""

    @staticmethod
    def _new_startup_session_id() -> str:
        return f"gui:annolid_bot:{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    @staticmethod
    def _default_quick_actions() -> List[tuple[str, str]]:
        return [
            (
                "Start Blink Stream",
                "open stream with model mediapipe face and classify eye blinks",
            ),
            ("Stop Stream", "stop realtime stream"),
            ("Summarize Context", "Summarize what we are doing in this session."),
        ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("AIChatWidget")

        self._active_search_topic = ""
        self._active_search_auto_draft = False
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
        self._chat_task_active = False
        self._active_chat_task: Optional[StreamingChatTask] = None
        self._active_turn_id: str = ""
        self._pending_explicit_image_path: str = ""
        self._bot_action_result: Dict[str, Any] = {}
        self._approved_destructive_tools: set[str] = set()
        self.session_id = self._new_startup_session_id()
        self._applying_theme_styles = False
        self.thread_pool = QThreadPool()
        self._audio_controller: Optional[ChatAudioController] = None
        self._session_store = get_chat_session_store()
        self._session_manager = (
            getattr(self._session_store, "manager", None)
            or getattr(self._session_store, "_manager", None)
            or AgentSessionManager()
        )
        self._max_prompt_chars = 4000
        self._last_user_prompt: str = ""
        self._next_chat_mode: str = "default"
        self._typing_tick = 0
        self._typing_timer = QtCore.QTimer(self)
        self._typing_timer.setInterval(350)
        self._typing_timer.timeout.connect(self._on_typing_timer_tick)
        self._chat_message_bus = MessageBus()
        self._chat_inbound_bus_timer = QtCore.QTimer(self)
        self._behavior_label_thread: Optional[QtCore.QThread] = None
        self._behavior_label_worker: Optional[FlexibleWorker] = None
        self._behavior_label_run_context: Dict[str, Any] = {}
        self._chat_inbound_bus_timer.setInterval(40)
        self._chat_inbound_bus_timer.timeout.connect(self._drain_inbound_bus_messages)
        self._chat_bus_timer = QtCore.QTimer(self)
        self._chat_bus_timer.setInterval(40)
        self._chat_bus_timer.timeout.connect(self._drain_outbound_bus_messages)
        self.enable_progress_stream = self._resolve_enable_progress_stream(
            self.llm_settings
        )
        self._latest_progress_text: str = ""
        self._has_streamed_response_chunk = False
        self._bot_action_results: Dict[str, Dict[str, Any]] = {}
        self._quick_actions: List[tuple[str, str]] = self._default_quick_actions()
        self._selected_quick_action_index: Optional[int] = None
        self._load_quick_actions_from_settings()
        self._current_progress_bubble: Optional[_ChatBubble] = None
        self._progress_lines: List[str] = []
        self._last_final_message_text: str = ""
        self._last_final_message_is_error: bool = False
        self._last_final_message_ts: float = 0.0
        self._zulip_send_active = False
        self._zulip_target_summary = ""
        self._seen_event_keys: "OrderedDict[str, float]" = OrderedDict()
        self._seen_event_key_limit = 512
        self._selected_capability_chip_widgets: List[QtWidgets.QToolButton] = []
        self._suggested_capability_chip_widgets: List[QtWidgets.QToolButton] = []
        self._slash_hint_widgets: List[QtWidgets.QToolButton] = []
        self._chip_drag_source: Optional[QtWidgets.QWidget] = None
        self._chip_drag_start_pos: Optional[QtCore.QPoint] = None
        self._prompt_history: List[str] = []
        self._prompt_history_index: int = -1
        self._prompt_history_draft: str = ""
        self._auto_follow_scroll: bool = True

        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_theme_styles()
        self._update_model_selector()
        self._load_session_history_into_bubbles(self.session_id)
        self._refresh_header_chips()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 1. Header
        root.addWidget(self._build_header_bar())

        # 2. Chat Area
        self.chat_area = self._build_chat_area()
        root.addWidget(self.chat_area, 1)

        # Compact jump-to-latest strip (shown when scrolled up during conversation)
        jump_row = QtWidgets.QHBoxLayout()
        jump_row.setContentsMargins(8, 2, 8, 2)
        jump_row.addStretch(1)
        self.jump_to_latest_button = QtWidgets.QPushButton("Jump to latest ↓", self)
        self.jump_to_latest_button.setObjectName("jumpToLatestButton")
        self.jump_to_latest_button.setVisible(False)
        self.jump_to_latest_button.clicked.connect(self._jump_to_latest_messages)
        jump_row.addWidget(self.jump_to_latest_button, 0)
        jump_row.addStretch(1)
        root.addLayout(jump_row)

        # 3. Input Area Container
        self.input_container = self._build_input_container()
        root.addWidget(self.input_container)

        self._audio_controller = ChatAudioController(
            status_label=self.status_label,
            talk_button=self.talk_button,
            prompt_text_edit=self.prompt_text_edit,
            get_last_assistant_text=self._last_assistant_text,
        )
        self._wire_ui_signals()
        self._refresh_header_chips()

    def _build_header_bar(self) -> QtWidgets.QWidget:
        header_widget = QtWidgets.QWidget(self)
        header_widget.setObjectName("headerBar")
        header_vbox = QtWidgets.QVBoxLayout(header_widget)
        header_vbox.setContentsMargins(12, 10, 12, 10)
        header_vbox.setSpacing(6)

        # Row 1: Bot icon, title/session summary, and compact action buttons
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        self.bot_icon_label = QtWidgets.QLabel(self)
        self.bot_icon_label.setObjectName("chatBotIconLabel")
        self.bot_icon_label.setFixedSize(28, 28)
        self.bot_icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self._set_bot_icon()
        top_row.addWidget(self.bot_icon_label, 0, QtCore.Qt.AlignTop)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)

        self.chat_title_label = QtWidgets.QLabel("Annolid Bot", self)
        self.chat_title_label.setObjectName("chatTitleLabel")
        self.chat_title_label.setMinimumWidth(0)
        self.chat_title_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        title_col.addWidget(self.chat_title_label)

        self.session_chip_label = QtWidgets.QLabel(self)
        self.session_chip_label.setObjectName("chatSubtitleLabel")
        self.session_chip_label.setMinimumWidth(0)
        self.session_chip_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        title_col.addWidget(self.session_chip_label)

        self.capability_summary_label = QtWidgets.QLabel(self)
        self.capability_summary_label.setObjectName("chatCapabilitySummaryLabel")
        self.capability_summary_label.setWordWrap(False)
        self.capability_summary_label.setMinimumWidth(0)
        self.capability_summary_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        title_col.addWidget(self.capability_summary_label)

        top_row.addLayout(title_col, 1)

        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(2)

        self.new_chat_button = self._create_header_button(
            "document-new", "New chat session", QtWidgets.QStyle.SP_FileIcon
        )
        actions_row.addWidget(self.new_chat_button)

        self.clear_chat_button = self._create_header_button(
            "edit-clear", "Clear conversation", QtWidgets.QStyle.SP_BrowserStop
        )
        actions_row.addWidget(self.clear_chat_button)

        self.sessions_button = self._create_header_button(
            "view-list-details",
            "Manage sessions",
            QtWidgets.QStyle.SP_FileDialogDetailedView,
        )
        actions_row.addWidget(self.sessions_button)

        self.capabilities_button = self._create_header_button(
            "help-contents",
            "Inspect agent tools and skills",
            QtWidgets.QStyle.SP_MessageBoxInformation,
        )
        actions_row.addWidget(self.capabilities_button)

        self.configure_button = self._create_header_button(
            "preferences-system", "Settings", QtWidgets.QStyle.SP_FileDialogInfoView
        )
        actions_row.addWidget(self.configure_button)

        top_row.addLayout(actions_row, 0)
        header_vbox.addLayout(top_row)

        # Row 2: Full-width Provider & Model selectors (readable even in narrow docks)
        provider_model_row = QtWidgets.QHBoxLayout()
        provider_model_row.setContentsMargins(0, 0, 0, 0)
        provider_model_row.setSpacing(6)

        self.provider_selector = QtWidgets.QComboBox(self)
        self.provider_selector.setObjectName("chatProviderSelector")
        self.provider_selector.setMinimumWidth(76)
        configure_compact_combo(
            self.provider_selector,
            min_chars=8,
            popup_min_width=210,
        )
        self._populate_provider_selector()
        provider_index = self.provider_selector.findData(self.selected_provider)
        if provider_index != -1:
            self.provider_selector.setCurrentIndex(provider_index)
        provider_model_row.addWidget(self.provider_selector, 0)

        self.model_selector = QtWidgets.QComboBox(self)
        self.model_selector.setObjectName("chatModelSelector")
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.model_selector.setMinimumWidth(84)
        configure_compact_combo(
            self.model_selector,
            min_chars=10,
            popup_min_width=280,
        )
        provider_model_row.addWidget(self.model_selector, 1)
        header_vbox.addLayout(provider_model_row)

        return header_widget

    def _build_provider_bar(self) -> QtWidgets.QHBoxLayout:
        # Compatibility method - elements are now in header/input
        # We return a dummy layout to satisfy any callers if they exist
        # though ideally we shouldn't have any external callers.
        return QtWidgets.QHBoxLayout()

    def _populate_provider_selector(self) -> None:
        self.provider_labels = self._providers.labels()
        self.provider_selector.blockSignals(True)
        try:
            self.provider_selector.clear()
            for key, label in self.provider_labels.items():
                self.provider_selector.addItem(label, userData=key)
        finally:
            self.provider_selector.blockSignals(False)

    def _refresh_runtime_llm_settings(self) -> None:
        """Reload LLM settings so provider/model changes apply on the next turn."""
        refresh_runtime_provider_settings(
            self, after_refresh=self._after_runtime_settings_refresh
        )

    def _after_runtime_settings_refresh(self) -> None:
        self.enable_progress_stream = self._resolve_enable_progress_stream(
            self.llm_settings
        )
        self._refresh_header_chips()

    @staticmethod
    def _resolve_enable_progress_stream(settings: Dict[str, Any]) -> bool:
        agent_cfg = settings.get("agent")
        if not isinstance(agent_cfg, dict):
            return True
        return bool(agent_cfg.get("enable_progress_stream", True))

    def _create_header_button(self, theme_icon, tooltip, style_icon):
        btn = QtWidgets.QToolButton(self)
        btn.setObjectName("chatTopIconButton")
        btn.setToolTip(tooltip)
        self._set_button_icon(btn, style_icon, theme_icon)
        return btn

    def _build_chat_area(self) -> QtWidgets.QScrollArea:
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.chat_container = QtWidgets.QWidget(self.scroll_area)
        self.chat_container.setObjectName("chatContainer")
        self.chat_layout = QtWidgets.QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(16)

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

    def _build_input_container(self) -> QtWidgets.QFrame:
        container = QtWidgets.QFrame(self)
        container.setObjectName("inputBarContainer")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # 1. Quick Actions inside a compact horizontal scroll strip
        self.quick_actions_scroll = QtWidgets.QScrollArea(self)
        self.quick_actions_scroll.setObjectName("quickActionsScroll")
        self.quick_actions_scroll.setWidgetResizable(True)
        self.quick_actions_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.quick_actions_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.quick_actions_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.quick_actions_scroll.setMaximumHeight(38)
        self.quick_actions_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        quick_actions_body = QtWidgets.QWidget(self.quick_actions_scroll)
        self.quick_actions_scroll.setWidget(quick_actions_body)
        self.quick_actions_layout = QtWidgets.QHBoxLayout(quick_actions_body)
        self.quick_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.quick_actions_layout.setSpacing(6)
        self.quick_action_buttons = []

        self.add_quick_action_button = QtWidgets.QToolButton(self)
        self.add_quick_action_button.setObjectName("chatInputButton")
        self.add_quick_action_button.setText("➕")
        self.add_quick_action_button.setToolTip("Add quick prompt")
        self.add_quick_action_button.clicked.connect(self._add_quick_action)

        self.remove_quick_action_button = QtWidgets.QToolButton(self)
        self.remove_quick_action_button.setObjectName("chatInputButton")
        self.remove_quick_action_button.setText("➖")
        self.remove_quick_action_button.setToolTip("Remove selected quick prompt")
        self.remove_quick_action_button.clicked.connect(
            self._remove_selected_quick_action
        )

        self.behavior_label_preset_button = QtWidgets.QPushButton(
            "Label 1s Behaviors", self
        )
        self.behavior_label_preset_button.setObjectName("quickActionButton")
        self.behavior_label_preset_button.setToolTip(
            "Auto-label the current video in 1-second segments using 3-frame VLM voting."
        )
        self.behavior_label_preset_button.clicked.connect(
            self._run_behavior_label_preset_one_second
        )

        layout.addWidget(self.quick_actions_scroll)
        self._refresh_quick_action_buttons()

        self.zulip_panel = self._build_zulip_panel()
        layout.addWidget(self.zulip_panel)

        # Visible attachment pill when an image/snapshot is explicitly attached
        self.attachment_chip_widget = QtWidgets.QFrame(self)
        self.attachment_chip_widget.setObjectName("attachmentChipBar")
        self.attachment_chip_widget.setVisible(False)
        attach_row = QtWidgets.QHBoxLayout(self.attachment_chip_widget)
        attach_row.setContentsMargins(6, 2, 6, 2)
        attach_row.setSpacing(6)
        self.attachment_chip_label = QtWidgets.QLabel("", self.attachment_chip_widget)
        self.attachment_chip_label.setObjectName("attachmentChipLabel")
        self.attachment_chip_label.setMinimumWidth(0)
        self.attachment_chip_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        attach_row.addWidget(self.attachment_chip_label, 1)
        self.clear_attachment_button = QtWidgets.QToolButton(
            self.attachment_chip_widget
        )
        self.clear_attachment_button.setObjectName("attachmentClearButton")
        self.clear_attachment_button.setText("×")
        self.clear_attachment_button.setToolTip("Remove attached image")
        self.clear_attachment_button.setFixedSize(20, 20)
        self.clear_attachment_button.clicked.connect(self._clear_attached_image)
        attach_row.addWidget(self.clear_attachment_button, 0)
        layout.addWidget(self.attachment_chip_widget)

        # 2. Full-width Prompt Input + Compact Composer Action Bar beneath
        self.prompt_text_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_text_edit.setPlaceholderText("Message Annolid Bot...")
        self.prompt_text_edit.setMinimumWidth(120)
        self.prompt_text_edit.setMinimumHeight(48)
        self.prompt_text_edit.setMaximumHeight(156)
        self.prompt_text_edit.setFixedHeight(54)
        self.prompt_text_edit.setToolTip(
            "Type a message (Ctrl+Enter to send, Alt+Up/Down for history, / for commands, drop files to attach)."
        )
        layout.addWidget(self.prompt_text_edit)

        input_row = QtWidgets.QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)

        tools_layout = QtWidgets.QHBoxLayout()
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(2)

        self.attach_file_button = self._create_input_icon(
            _symbol_text("📎", "F"), "Attach file or image"
        )
        self.share_canvas_button = self._create_input_icon(
            _symbol_text("🎨", "C"), "Share Canvas"
        )
        self.share_window_button = self._create_input_icon(
            _symbol_text("🪟", "W"), "Share Window"
        )
        self.citation_button = self._create_input_icon(
            _symbol_text("📚", "R"), "Manage citations"
        )
        self.zulip_button = self._create_input_icon(
            _symbol_text("✉", "@"), "Draft and send to Zulip"
        )
        self.options_button = self._create_input_icon(
            _symbol_text("⚙", "O"), "Bot context & tool options"
        )
        self.options_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        tools_layout.addWidget(self.attach_file_button)
        tools_layout.addWidget(self.share_canvas_button)
        tools_layout.addWidget(self.share_window_button)
        tools_layout.addWidget(self.citation_button)
        tools_layout.addWidget(self.zulip_button)
        tools_layout.addWidget(self.options_button)
        input_row.addLayout(tools_layout)
        input_row.addStretch(1)

        send_layout = QtWidgets.QHBoxLayout()
        send_layout.setContentsMargins(0, 0, 0, 0)
        send_layout.setSpacing(4)

        self.talk_button = QtWidgets.QToolButton(self)
        self.talk_button.setObjectName("talkButton")
        self.talk_button.setText(_symbol_text("🎤", "Mic"))
        self.talk_button.setToolTip("Record voice input")
        self.talk_button.setMinimumSize(32, 32)
        self.talk_button.setFixedHeight(32)
        self.talk_button.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )

        self.stop_button = QtWidgets.QToolButton(self)
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setText(_symbol_text("⏹", "Stop"))
        self.stop_button.setToolTip("Stop current response (Esc)")
        self.stop_button.setMinimumSize(32, 32)
        self.stop_button.setFixedHeight(32)
        self.stop_button.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )
        self.stop_button.setVisible(False)

        self.send_button = QtWidgets.QToolButton(self)
        self.send_button.setObjectName("sendButton")
        self.send_button.setText(_symbol_text("🚀", ">"))
        self.send_button.setToolTip("Send message (Ctrl+Enter)")
        self.send_button.setMinimumSize(32, 32)
        self.send_button.setFixedHeight(32)
        self.send_button.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )

        send_layout.addWidget(self.talk_button)
        send_layout.addWidget(self.stop_button)
        send_layout.addWidget(self.send_button)
        input_row.addLayout(send_layout)

        layout.addLayout(input_row)

        slash_hint_row = QtWidgets.QHBoxLayout()
        slash_hint_row.setContentsMargins(4, 0, 4, 0)
        self.slash_hint_label = QtWidgets.QLabel("Quick suggestions", self)
        self.slash_hint_label.setObjectName("chatStatusLabel")
        slash_hint_row.addWidget(self.slash_hint_label, 0)
        slash_hint_row.addStretch(1)
        layout.addLayout(slash_hint_row)

        self.slash_hint_scroll = QtWidgets.QScrollArea(self)
        self.slash_hint_scroll.setObjectName("slashHintScroll")
        self.slash_hint_scroll.setWidgetResizable(True)
        self.slash_hint_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.slash_hint_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.slash_hint_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.slash_hint_scroll.setMaximumHeight(48)
        self.slash_hint_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        slash_hint_body = QtWidgets.QWidget(self.slash_hint_scroll)
        self.slash_hint_scroll.setWidget(slash_hint_body)
        self.slash_hint_layout = QtWidgets.QHBoxLayout(slash_hint_body)
        self.slash_hint_layout.setContentsMargins(0, 0, 0, 0)
        self.slash_hint_layout.setSpacing(6)
        self.slash_hint_layout.addStretch(1)
        layout.addWidget(self.slash_hint_scroll)

        # Capability chips for slash-selected skills/tools
        chip_header_row = QtWidgets.QHBoxLayout()
        chip_header_row.setContentsMargins(4, 0, 4, 0)
        self.capability_chip_label = QtWidgets.QLabel("Selected capabilities", self)
        self.capability_chip_label.setObjectName("chatStatusLabel")
        chip_header_row.addWidget(self.capability_chip_label, 0)
        chip_header_row.addStretch(1)
        self.clear_capability_chips_button = QtWidgets.QToolButton(self)
        self.clear_capability_chips_button.setObjectName("chatInputButton")
        self.clear_capability_chips_button.setText("✕")
        self.clear_capability_chips_button.setToolTip("Clear selected skills and tools")
        self.clear_capability_chips_button.clicked.connect(
            self._clear_selected_capabilities
        )
        chip_header_row.addWidget(self.clear_capability_chips_button, 0)
        layout.addLayout(chip_header_row)

        self.capability_chip_scroll = QtWidgets.QScrollArea(self)
        self.capability_chip_scroll.setObjectName("capabilityChipScroll")
        self.capability_chip_scroll.setWidgetResizable(True)
        self.capability_chip_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.capability_chip_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.capability_chip_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.capability_chip_scroll.setMaximumHeight(54)
        self.capability_chip_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        chip_scroll_body = QtWidgets.QWidget(self.capability_chip_scroll)
        self.capability_chip_scroll.setWidget(chip_scroll_body)
        self.capability_chip_layout = QtWidgets.QHBoxLayout(chip_scroll_body)
        self.capability_chip_layout.setContentsMargins(0, 0, 0, 0)
        self.capability_chip_layout.setSpacing(6)
        self.capability_chip_layout.addStretch(1)
        layout.addWidget(self.capability_chip_scroll)

        suggested_row = QtWidgets.QHBoxLayout()
        suggested_row.setContentsMargins(4, 0, 4, 0)
        self.suggested_chip_label = QtWidgets.QLabel("Suggested skills", self)
        self.suggested_chip_label.setObjectName("chatStatusLabel")
        suggested_row.addWidget(self.suggested_chip_label, 0)
        suggested_row.addStretch(1)
        layout.addLayout(suggested_row)

        self.suggested_chip_scroll = QtWidgets.QScrollArea(self)
        self.suggested_chip_scroll.setObjectName("suggestedChipScroll")
        self.suggested_chip_scroll.setWidgetResizable(True)
        self.suggested_chip_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.suggested_chip_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.suggested_chip_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.suggested_chip_scroll.setMaximumHeight(54)
        self.suggested_chip_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        suggested_chip_body = QtWidgets.QWidget(self.suggested_chip_scroll)
        self.suggested_chip_scroll.setWidget(suggested_chip_body)
        self.suggested_chip_layout = QtWidgets.QHBoxLayout(suggested_chip_body)
        self.suggested_chip_layout.setContentsMargins(0, 0, 0, 0)
        self.suggested_chip_layout.setSpacing(6)
        self.suggested_chip_layout.addStretch(1)
        layout.addWidget(self.suggested_chip_scroll)

        # 3. Meta / Status Row
        meta_row = QtWidgets.QHBoxLayout()
        meta_row.setContentsMargins(4, 0, 4, 0)
        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setObjectName("chatStatusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        meta_row.addWidget(self.status_label, 1)

        self.prompt_count_label = QtWidgets.QLabel("0/4000", self)
        self.prompt_count_label.setObjectName("promptCountLabel")
        meta_row.addWidget(self.prompt_count_label, 0)

        layout.addLayout(meta_row)

        # Hidden logic toggles (keep objects for logic compatibility)
        self._create_hidden_toggles(layout)

        return container

    def _create_input_icon(self, text_icon, tooltip):
        btn = QtWidgets.QToolButton(self)
        btn.setObjectName("chatInputButton")
        btn.setText(text_icon)
        btn.setToolTip(tooltip)
        btn.setFixedSize(28, 28)
        # Use a larger font for emojis
        font = btn.font()
        font.setPointSize(14)
        btn.setFont(font)
        return btn

    def _wire_ui_signals(self) -> None:
        self.provider_selector.currentIndexChanged.connect(self.on_provider_changed)
        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        self.model_selector.editTextChanged.connect(self.on_model_text_edited)
        self.prompt_text_edit.textChanged.connect(self._on_prompt_text_changed)
        self.prompt_text_edit.installEventFilter(self)
        scroll_bar = self.scroll_area.verticalScrollBar()
        if scroll_bar is not None:
            scroll_bar.valueChanged.connect(self._on_chat_scroll_changed)
        line_edit = self.model_selector.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self.on_model_editing_finished)
        self.new_chat_button.clicked.connect(self.start_new_chat_session)
        self.capabilities_button.clicked.connect(self.open_agent_capabilities_dialog)
        self.configure_button.clicked.connect(self.open_llm_settings_dialog)
        self.send_button.clicked.connect(self.chat_with_model)
        self.stop_button.clicked.connect(self._stop_running_response)
        self.share_canvas_button.clicked.connect(self._share_canvas_now)
        self.share_window_button.clicked.connect(self._share_window_now)
        self.citation_button.clicked.connect(self._open_citation_manager)
        self.zulip_button.clicked.connect(self._toggle_zulip_panel)
        self.zulip_target_type_combo.currentIndexChanged.connect(
            self._sync_zulip_target_field_visibility
        )
        self.zulip_defaults_button.clicked.connect(self._load_zulip_defaults)
        self.zulip_last_reply_button.clicked.connect(self._use_last_reply_for_zulip)
        self.zulip_send_button.clicked.connect(self._send_current_draft_to_zulip)
        self.attach_file_button.clicked.connect(self._attach_file)
        self.talk_button.clicked.connect(self.toggle_recording)
        self.clear_chat_button.clicked.connect(self._confirm_clear_chat_conversation)
        self.sessions_button.clicked.connect(self.open_session_manager_dialog)
        self._on_prompt_text_changed()
        self._sync_zulip_target_field_visibility()

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
        self._sync_prompt_input_height()
        has_text = bool(text.strip())
        self.send_button.setEnabled(has_text)
        self.send_button.setToolTip(
            "Queue follow-up message (Ctrl+Enter)"
            if self.is_streaming_chat
            else "Send message (Ctrl+Enter)"
        )
        stop_btn = getattr(self, "stop_button", None)
        if stop_btn is not None:
            stop_btn.setVisible(bool(self.is_streaming_chat))
        zulip_button = getattr(self, "zulip_send_button", None)
        if zulip_button is not None:
            zulip_button.setEnabled(has_text and not self._zulip_send_active)
        self._update_slash_completion_ui()
        self._refresh_selected_capability_chips()

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        try:
            if watched in self._selected_capability_chip_widgets and isinstance(
                event, QtGui.QMouseEvent
            ):
                if (
                    event.type() == QtCore.QEvent.MouseButtonPress
                    and event.button() == QtCore.Qt.LeftButton
                ):
                    self._chip_drag_source = (
                        watched if isinstance(watched, QtWidgets.QWidget) else None
                    )
                    self._chip_drag_start_pos = event.globalPosition().toPoint()
                elif (
                    event.type() == QtCore.QEvent.MouseButtonRelease
                    and event.button() == QtCore.Qt.LeftButton
                ):
                    source = self._chip_drag_source
                    self._chip_drag_source = None
                    self._chip_drag_start_pos = None
                    if (
                        source is not None
                        and source is not watched
                        and isinstance(source, QtWidgets.QWidget)
                        and isinstance(watched, QtWidgets.QWidget)
                    ):
                        if self._reorder_selected_capability(source, watched):
                            return True
            if (
                isinstance(event, QtGui.QKeyEvent)
                and watched in self._chip_widgets_in_order()
                and event.type() == QtCore.QEvent.KeyPress
            ):
                key = event.key()
                role = (
                    str(getattr(watched, "property", lambda _k: "")("role") or "")
                    .strip()
                    .lower()
                )  # type: ignore[attr-defined]
                cap_kind = (
                    str(
                        getattr(watched, "property", lambda _k: "")("capability_kind")
                        or ""
                    )
                    .strip()
                    .lower()
                )  # type: ignore[attr-defined]
                cap_name = str(
                    getattr(watched, "property", lambda _k: "")("capability_name") or ""
                ).strip()  # type: ignore[attr-defined]
                if key == QtCore.Qt.Key_Escape:
                    self.prompt_text_edit.setFocus()
                    return True
                if key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
                    if role == "selected" and cap_kind == "skill":
                        self._remove_capability_chip("skill", cap_name)
                        return True
                    if role == "selected" and cap_kind == "tool":
                        self._remove_capability_chip("tool", cap_name)
                        return True
                if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up):
                    if self._focus_next_capability_chip(
                        watched if isinstance(watched, QtWidgets.QWidget) else None,
                        -1,
                    ):
                        return True
                if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
                    if self._focus_next_capability_chip(
                        watched if isinstance(watched, QtWidgets.QWidget) else None,
                        1,
                    ):
                        return True
                if key in (
                    QtCore.Qt.Key_Return,
                    QtCore.Qt.Key_Enter,
                    QtCore.Qt.Key_Space,
                ):
                    if role == "selected" and cap_kind == "skill":
                        self._remove_capability_chip("skill", cap_name)
                        return True
                    if role == "selected" and cap_kind == "tool":
                        self._remove_capability_chip("tool", cap_name)
                        return True
                    if role == "suggested" and cap_kind == "skill":
                        self._add_skill_capability_chip(cap_name)
                        return True
            if (
                watched is self.prompt_text_edit
                and event.type() == QtCore.QEvent.KeyPress
                and isinstance(event, QtGui.QKeyEvent)
            ):
                key = event.key()
                modifiers = event.modifiers()
                if (
                    modifiers & QtCore.Qt.ControlModifier
                    and modifiers & QtCore.Qt.AltModifier
                ):
                    if key == QtCore.Qt.Key_Left:
                        if self._focus_capability_chip(kind="selected"):
                            return True
                    if key == QtCore.Qt.Key_Right:
                        if self._focus_capability_chip(kind="suggested"):
                            return True
                    if key == QtCore.Qt.Key_Backspace:
                        self._clear_selected_capabilities()
                        return True
                completer = getattr(self, "_slash_completer", None)
                popup = completer.popup() if completer is not None else None
                if popup is not None and popup.isVisible():
                    if key in (
                        QtCore.Qt.Key_Return,
                        QtCore.Qt.Key_Enter,
                        QtCore.Qt.Key_Tab,
                    ):
                        self._apply_slash_completion(popup.currentIndex())
                        return True
                    if key == QtCore.Qt.Key_Escape:
                        self._hide_slash_completion_ui()
                        return True
                    if key == QtCore.Qt.Key_Down:
                        self._move_slash_completion_selection(1)
                        return True
                    if key == QtCore.Qt.Key_Up:
                        self._move_slash_completion_selection(-1)
                        return True
                if key == QtCore.Qt.Key_Escape and self.is_streaming_chat:
                    self._stop_running_response()
                    return True
                if (modifiers & QtCore.Qt.AltModifier) and not (
                    modifiers & QtCore.Qt.ControlModifier
                ):
                    if key == QtCore.Qt.Key_Up and self._navigate_prompt_history(-1):
                        return True
                    if key == QtCore.Qt.Key_Down and self._navigate_prompt_history(1):
                        return True
                if modifiers == QtCore.Qt.NoModifier:
                    if key == QtCore.Qt.Key_Up and (
                        not self.prompt_text_edit.toPlainText().strip()
                        or self._prompt_history_index >= 0
                    ):
                        if self._navigate_prompt_history(-1):
                            return True
                    if (
                        key == QtCore.Qt.Key_Down
                        and self._prompt_history_index >= 0
                        and self._navigate_prompt_history(1)
                    ):
                        return True
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and bool(
                    event.modifiers() & QtCore.Qt.ControlModifier
                ):
                    self.chat_with_model()
                    return True
        except Exception:
            return False
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
            palette = self.palette()
            is_dark = palette.color(palette_color_role("Window")).lightness() < 128
            self.setStyleSheet(build_chat_widget_stylesheet(is_dark))
            self._is_dark_theme = is_dark
            if hasattr(self, "chat_layout"):
                for bubble in self._iter_chat_bubbles():
                    if hasattr(bubble, "apply_theme"):
                        bubble.apply_theme(is_dark=is_dark)
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

    def _is_near_bottom(self, threshold_px: int = 48) -> bool:
        bar = self.scroll_area.verticalScrollBar()
        if bar is None:
            return True
        return int(bar.maximum() - bar.value()) <= int(threshold_px)

    def _on_chat_scroll_changed(self, _value: int = 0) -> None:
        near_bottom = self._is_near_bottom()
        self._auto_follow_scroll = near_bottom
        jump_btn = getattr(self, "jump_to_latest_button", None)
        if jump_btn is not None:
            has_bubbles = len(self._iter_chat_bubbles()) > 0
            jump_btn.setVisible(bool(has_bubbles and not near_bottom))

    def _jump_to_latest_messages(self) -> None:
        self._auto_follow_scroll = True
        self._scroll_to_bottom()
        jump_btn = getattr(self, "jump_to_latest_button", None)
        if jump_btn is not None:
            jump_btn.setVisible(False)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())
        self._auto_follow_scroll = True
        jump_btn = getattr(self, "jump_to_latest_button", None)
        if jump_btn is not None:
            jump_btn.setVisible(False)

    def _scroll_to_bottom_if_following(self) -> None:
        if self._auto_follow_scroll or self._is_near_bottom():
            self._scroll_to_bottom()
        else:
            jump_btn = getattr(self, "jump_to_latest_button", None)
            if jump_btn is not None:
                jump_btn.setVisible(True)

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

    def _reflow_single_bubble(self, bubble: Optional[_ChatBubble]) -> None:
        if bubble is None:
            return
        try:
            max_width = self._bubble_max_width()
            bubble.apply_layout_width(max_width - 24, max_width)
        except RuntimeError:
            return

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
        history_index: Optional[int] = None,
        history_message_id: str = "",
        history_role: str = "",
        allow_regenerate: bool = False,
        allow_stop: bool = False,
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
            on_stop=self._stop_running_response,
            on_delete=self._confirm_delete_bubble,
            on_open_link=self._open_chat_link_default,
            on_open_link_in_browser=self._open_chat_link_in_browser,
            allow_regenerate=allow_regenerate,
            allow_stop=allow_stop,
            parent=self.chat_container,
        )
        # Make bubbles full width as requested
        bubble.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        bubble.apply_layout_width(
            self._bubble_max_width() - 24, self._bubble_max_width()
        )
        bubble.setProperty(
            "history_index",
            int(history_index) if history_index is not None else -1,
        )
        bubble.setProperty("history_message_id", str(history_message_id or "").strip())
        bubble.setProperty(
            "history_role",
            str(history_role or ("user" if is_user else "assistant")),
        )
        # Remove addStretch and Alignment to force full width
        row.addWidget(bubble)

        # Insert before trailing stretch item.
        insert_idx = max(0, self.chat_layout.count() - 1)
        self.chat_layout.insertWidget(insert_idx, row_widget)
        self._reflow_single_bubble(bubble)
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

    def _confirm_delete_bubble(self, bubble: _ChatBubble) -> None:
        if not isinstance(bubble, _ChatBubble):
            return
        if self.is_streaming_chat and bubble is self._current_response_bubble:
            self.status_label.setText("Wait for current response to finish.")
            return

        role = str(bubble.property("history_role") or "").strip().lower()
        role_label = "message" if role not in {"user", "assistant", "system"} else role
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Message",
            f"Delete this {role_label} message?\n\nThis action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        deleted_idx = int(bubble.property("history_index") or -1)
        message_id = str(bubble.property("history_message_id") or "").strip()
        persisted_deleted = False
        if message_id or deleted_idx >= 0:
            persisted_deleted = bool(
                self._session_store.delete_history_message(
                    self.session_id,
                    message_id=message_id,
                    history_index=deleted_idx,
                    expected_role=str(bubble.property("history_role") or ""),
                    expected_content=str(bubble.raw_text() or "").strip(),
                )
            )
            if not persisted_deleted:
                self.status_label.setText(
                    "Message changed in history; reload session and try again."
                )
                return

        removed = self._remove_bubble_from_chat_layout(bubble)
        if not removed:
            return
        if persisted_deleted and deleted_idx >= 0:
            self._shift_visible_history_indices_after_delete(deleted_idx)
        if bubble is self._current_response_bubble:
            self._current_response_bubble = None
        if message_id or deleted_idx >= 0:
            self.status_label.setText("Message deleted.")
        else:
            self.status_label.setText("Message removed from chat view.")
        self._update_empty_state_visibility()
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _remove_bubble_from_chat_layout(self, bubble: _ChatBubble) -> bool:
        for idx in range(self.chat_layout.count()):
            item = self.chat_layout.itemAt(idx)
            if item is None:
                continue
            row_widget = item.widget()
            if row_widget is None:
                continue
            row_layout = row_widget.layout()
            if row_layout is None:
                continue
            found = False
            for j in range(row_layout.count()):
                if row_layout.itemAt(j).widget() is bubble:
                    found = True
                    break
            if not found:
                continue
            removed_item = self.chat_layout.takeAt(idx)
            removed_widget = removed_item.widget()
            if removed_widget is not None:
                removed_widget.deleteLater()
            return True
        return False

    def _shift_visible_history_indices_after_delete(self, deleted_idx: int) -> None:
        for row_bubble in self._iter_chat_bubbles():
            current_idx = int(row_bubble.property("history_index") or -1)
            if current_idx > deleted_idx:
                row_bubble.setProperty("history_index", current_idx - 1)

    def _prompt_for_assistant_bubble(self, bubble: _ChatBubble) -> str:
        bubbles = self._iter_chat_bubbles()
        try:
            idx = bubbles.index(bubble)
        except ValueError:
            idx = -1
        if idx > 0:
            for prev in reversed(bubbles[:idx]):
                if prev.property("role") == "user":
                    text = (prev.raw_text() or prev.text()).strip()
                    if text:
                        return text
        return str(self._last_user_prompt or "").strip()

    def _regenerate_from_bubble(self, target: Any) -> None:
        if isinstance(target, _ChatBubble):
            prompt = self._prompt_for_assistant_bubble(target)
        else:
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
        if kind in {
            "openai_compat",
            "anthropic",
            "gemini",
        } and not has_provider_api_key(self.llm_settings, self.selected_provider):
            if kind == "openai_compat":
                provider_cfg = self.llm_settings.get(self.selected_provider, {}) or {}
                provider_def = (
                    self.llm_settings.get("provider_definitions", {}) or {}
                ).get(self.selected_provider, {}) or {}
                base_url = str(
                    provider_cfg.get("base_url")
                    or provider_def.get("base_url_default")
                    or ""
                ).strip()
                if is_loopback_base_url(base_url):
                    return True
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
        self.enable_progress_stream = self._resolve_enable_progress_stream(
            self.llm_settings
        )
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
        self._load_quick_actions_from_settings()
        self._refresh_quick_action_buttons()
        self._persist_quick_actions_to_settings()
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
        if hasattr(self, "provider_selector"):
            try:
                status = inspect_provider_credential_status(
                    self.llm_settings,
                    self.selected_provider,
                    dotenv_values={},
                )
                self.provider_selector.setToolTip(
                    f"AI Provider: {provider_text}\n{status.summary}"
                )
            except Exception:
                self.provider_selector.setToolTip(f"AI Provider: {provider_text}")

    def start_new_chat_session(self) -> None:
        if self.is_streaming_chat or self._chat_task_active:
            self.status_label.setText(
                "Stop or wait for the current response before starting a new session."
            )
            return
        new_id = self._new_startup_session_id()
        if new_id == self.session_id:
            new_id = f"{new_id}_next"
        self.session_id = new_id
        self._approved_destructive_tools.clear()
        self._clear_chat_bubbles()
        self.status_label.setText("Started new chat session.")
        self._refresh_header_chips()
        self.prompt_text_edit.setFocus()

    def _confirm_clear_chat_conversation(self) -> None:
        if self.is_streaming_chat:
            self.status_label.setText("Wait for current response to finish.")
            return
        if len(self._iter_chat_bubbles()) > 0:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Clear Conversation",
                "Clear all messages in the current session?\n\n"
                "Tip: Use New Chat to start a fresh session without clearing history.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        self.clear_chat_conversation()

    def clear_chat_conversation(self) -> None:
        if self.is_streaming_chat:
            self.status_label.setText("Wait for current response to finish.")
            return
        self._session_store.clear_session(self.session_id)
        clear_chat_session(self.session_id)
        self._approved_destructive_tools.clear()
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
        self.empty_state_label.setVisible(len(self._iter_chat_bubbles()) == 0)

    def _load_session_history_into_bubbles(self, session_id: str) -> None:
        self._approved_destructive_tools.clear()
        self._clear_chat_bubbles()
        try:
            history = self._session_store.get_history(str(session_id or ""))
        except Exception:
            history = []
        start_idx = max(0, len(history) - 80)
        for idx, msg in enumerate(history[start_idx:], start=start_idx):
            role = str(msg.get("role") or "")
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                self._add_bubble(
                    "You",
                    content,
                    is_user=True,
                    history_index=idx,
                    history_message_id=str(msg.get("message_id") or ""),
                    history_role=role,
                )
            elif role == "assistant":
                self._add_bubble(
                    self._assistant_display_name(),
                    content,
                    is_user=False,
                    history_index=idx,
                    history_message_id=str(msg.get("message_id") or ""),
                    history_role=role,
                    allow_regenerate=True,
                )
            elif role == "system":
                self._add_bubble(
                    "System",
                    content,
                    is_user=False,
                    history_index=idx,
                    history_message_id=str(msg.get("message_id") or ""),
                    history_role=role,
                )

    def open_session_manager_dialog(self) -> None:
        if self.is_streaming_chat or self._chat_task_active:
            self.status_label.setText(
                "Stop or wait for the current response before managing sessions."
            )
            return
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
        if self.is_streaming_chat or self._chat_task_active:
            return
        key = str(session_id or "").strip()
        if not key:
            return
        self.set_session_id(key)
        self.status_label.setText(f"Switched to session: {key}")

    @QtCore.Slot(str)
    def _on_session_created(self, session_id: str) -> None:
        if self.is_streaming_chat or self._chat_task_active:
            return
        key = str(session_id or "").strip()
        if not key:
            return
        self.set_session_id(key)
        self.status_label.setText(f"Created session: {key}")
        self._refresh_header_chips()

    @QtCore.Slot(str)
    def _on_session_cleared(self, session_id: str) -> None:
        if self.is_streaming_chat or self._chat_task_active:
            return
        key = str(session_id or "").strip()
        if not key:
            return
        if key == self.session_id:
            self._load_session_history_into_bubbles(key)
        self.status_label.setText(f"Cleared session: {key}")
        self._refresh_header_chips()

    @QtCore.Slot(str)
    def _on_session_deleted(self, session_id: str) -> None:
        if self.is_streaming_chat or self._chat_task_active:
            return
        key = str(session_id or "").strip()
        if not key:
            return
        if key == self.session_id:
            self.session_id = self._new_startup_session_id()
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

    def shutdown(self, *, wait_ms: int = 3000) -> bool:
        """Stop background workers, audio threads, timers, and clean temp snapshots."""
        try:
            self._stop_behavior_label_worker(wait_ms=max(1000, int(wait_ms)))
        except TimeoutError:
            return False

        task = getattr(self, "_active_chat_task", None)
        if task is not None and hasattr(task, "request_cancel"):
            with contextlib.suppress(Exception):
                task.request_cancel()
        self._active_chat_task = None
        self._active_turn_id = ""
        self._chat_task_active = False
        self.is_streaming_chat = False

        audio = getattr(self, "_audio_controller", None)
        if audio is not None and hasattr(audio, "shutdown"):
            with contextlib.suppress(Exception):
                audio.shutdown()

        for timer_attr in (
            "_typing_timer",
            "_chat_inbound_bus_timer",
            "_chat_bus_timer",
        ):
            timer = getattr(self, timer_attr, None)
            if timer is not None:
                with contextlib.suppress(RuntimeError):
                    timer.stop()

        pool = getattr(self, "thread_pool", None)
        if pool is not None:
            with contextlib.suppress(Exception):
                pool.waitForDone(max(250, min(int(wait_ms), 2000)))

        with contextlib.suppress(Exception):
            self._cleanup_snapshots()
        return True

    def closeEvent(self, event) -> None:
        if not self.shutdown(wait_ms=5000):
            event.ignore()
            self.status_label.setText(
                "Behavior labeling is stopping. Close again after it finishes."
            )
            return
        super().closeEvent(event)

    def _track_slash_defaults(self) -> Dict[str, Any]:
        host = self.host_window_widget or self.window()
        active_video_path = str(getattr(host, "video_file", "") or "").strip()
        prompt_text = str(self.prompt_text_edit.toPlainText() or "").strip()
        if prompt_text.startswith("/track"):
            prompt_text = ""
        model_names: list[str] = []
        combo = getattr(host, "_selectAiModelComboBox", None)
        if combo is not None:
            try:
                model_names = [
                    str(combo.itemText(index) or "").strip()
                    for index in range(int(combo.count()))
                    if str(combo.itemText(index) or "").strip()
                ]
            except Exception:
                model_names = []
            current_model = str(combo.currentText() or "").strip()
        else:
            current_model = ""
        if not model_names:
            model_names = list(self.available_models or [])
        if not current_model:
            current_model = str(self.selected_model or "").strip()
        if current_model and current_model not in model_names:
            model_names.append(current_model)
        return {
            "video_path": active_video_path,
            "prompt": prompt_text,
            "model_names": model_names,
            "selected_model": current_model,
            "mode": "track",
            "use_countgd": False,
            "to_frame": 0,
        }

    @QtCore.Slot()
    def bot_open_track_slash_dialog(self) -> None:
        defaults = self._track_slash_defaults()
        dialog = TrackSlashDialog(
            self,
            video_path=str(defaults.get("video_path") or ""),
            prompt=str(defaults.get("prompt") or ""),
            model_names=list(defaults.get("model_names") or []),
            selected_model=str(defaults.get("selected_model") or ""),
            mode=str(defaults.get("mode") or "track"),
            use_countgd=bool(defaults.get("use_countgd", False)),
            to_frame=int(defaults.get("to_frame") or 0),
            bot_provider=str(self.selected_provider or ""),
            bot_model=str(self.selected_model or ""),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._set_bot_action_result(
                "open_track_dialog",
                {"ok": False, "cancelled": True},
            )
            self.status_label.setText("Track setup canceled.")
            return

        values = dialog.values()
        command = build_track_slash_command(values)
        self.prompt_text_edit.setPlainText(command)
        self.prompt_text_edit.setFocus()
        payload = {
            "ok": True,
            "command": command,
            "video_path": str(values.get("video_path") or ""),
            "text_prompt": str(values.get("text_prompt") or ""),
            "model_name": str(values.get("model_name") or ""),
            "mode": str(values.get("mode") or "track"),
            "use_countgd": bool(values.get("use_countgd", False)),
            "to_frame": int(values.get("to_frame") or 0) or None,
        }
        self._set_bot_action_result("open_track_dialog", payload)
        self.status_label.setText("Prepared guided /track command.")

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
