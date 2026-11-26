from __future__ import annotations

import asyncio
import contextlib
import logging
import sys as sys
from collections import OrderedDict
import os
import tempfile
import time
import webbrowser
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from qtpy import QtCore, QtGui, QtWidgets
from annolid.gui.qt_compat import configure_compact_combo, palette_color_role
from qtpy.QtCore import QThreadPool

from annolid.infrastructure.agent_config import (
    get_agent_config_path as get_config_path,
    load_agent_config as load_config,
)
from annolid.infrastructure.agent_workspace import (
    get_agent_workspace_path as get_agent_workspace_path,
)
from annolid.services.chat_backend_support import (
    ERROR_TYPE_CANCELLED,
    TURN_STATUS_CANCELLED,
)
from annolid.services.chat_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    decode_outbound_chat_event,
)
from annolid.services.chat_session import (
    AgentSessionManager,
    get_chat_session_store,
)
from annolid.gui.widgets.ai_chat_audio_controller import ChatAudioController
from annolid.gui.widgets.ai_chat_backend import (
    StreamingChatTask,
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
from annolid.gui.widgets.ai_chat_slash_mixin import (
    AIChatSlashMixin,
    _compose_slash_selection_draft as _compose_slash_selection_draft,
    _dedupe_preserve_order as _dedupe_preserve_order,
    _extract_slash_selection_state as _extract_slash_selection_state,
    _split_slash_value_list as _split_slash_value_list,
)
from annolid.gui.widgets.ai_chat_bubble import (
    _ChatBubble,
    _emoji_supported as _emoji_supported,
    _symbol_text,
)
from annolid.gui.widgets.ai_chat_session_dialog import ChatSessionManagerDialog
from annolid.gui.widgets.ai_chat_zulip import (
    _ZulipSendTask,
    build_zulip_draft_target,
    missing_zulip_config_fields,
)
from annolid.gui.widgets.citation_manager_widget import CitationManagerDialog
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
from annolid.utils.llm_settings import (
    has_provider_api_key,
    load_llm_settings,
    provider_kind,
    save_llm_settings,
)
from annolid.utils.citations import (
    BibEntry,
    load_bibtex,
    merge_validated_fields,
    save_bibtex,
    upsert_entry,
    validate_basic_citation_fields,
    validate_citation_metadata,
)
from annolid.services.citation_verify import (
    build_citation_verification_report,
    write_citation_verification_report,
)

logger = logging.getLogger(__name__)


class AIChatWidget(
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
            getattr(self._session_store, "manager", None) or AgentSessionManager()
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
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)

        # Bot Icon
        self.bot_icon_label = QtWidgets.QLabel(self)
        self.bot_icon_label.setObjectName("chatBotIconLabel")
        self.bot_icon_label.setFixedSize(32, 32)
        self.bot_icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self._set_bot_icon()
        header_layout.addWidget(self.bot_icon_label)

        # Title & Model Info
        title_col = QtWidgets.QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)

        self.chat_title_label = QtWidgets.QLabel("Annolid Bot", self)
        self.chat_title_label.setObjectName("chatTitleLabel")
        self.chat_title_label.setObjectName("chatTitleLabel")
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

        # Model Selector (Compact)
        self.model_selector = QtWidgets.QComboBox(self)
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.model_selector.setMinimumWidth(90)
        configure_compact_combo(
            self.model_selector,
            min_chars=12,
            popup_min_width=280,
        )
        # We need to populate this later, but for now add to layout
        title_col.addWidget(self.model_selector)

        header_layout.addLayout(title_col, 1)

        # Spacer
        header_layout.addStretch(0)

        # Session & Settings Actions
        self.clear_chat_button = self._create_header_button(
            "edit-clear", "Clear conversation", QtWidgets.QStyle.SP_BrowserStop
        )
        header_layout.addWidget(self.clear_chat_button)

        self.sessions_button = self._create_header_button(
            "view-list-details",
            "Manage sessions",
            QtWidgets.QStyle.SP_FileDialogDetailedView,
        )
        header_layout.addWidget(self.sessions_button)

        self.configure_button = self._create_header_button(
            "preferences-system", "Settings", QtWidgets.QStyle.SP_FileDialogInfoView
        )
        header_layout.addWidget(self.configure_button)

        # Hidden provider selector (we keep it for logic compatibility but hide it or effectively replace it)
        # Actually, let's keep it but maybe invisible if we only care about model?
        # For now, let's add it to the header but compact.
        self.provider_selector = QtWidgets.QComboBox(self)
        self.provider_selector.setFixedWidth(0)  # Hide visually but keep object
        self.provider_selector.setVisible(False)
        self._populate_provider_selector()

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

        # 2. Full-width Prompt Input + Compact Composer Action Bar beneath
        self.prompt_text_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_text_edit.setPlaceholderText("Message Annolid Bot...")
        self.prompt_text_edit.setMinimumWidth(120)
        self.prompt_text_edit.setFixedHeight(54)
        self.prompt_text_edit.setToolTip(
            "Type a message. Use Ctrl+Enter to send. Type / to pick skills, tools, and commands."
        )
        layout.addWidget(self.prompt_text_edit)

        input_row = QtWidgets.QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(4)

        tools_layout = QtWidgets.QHBoxLayout()
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(2)

        self.attach_file_button = self._create_input_icon(
            _symbol_text("📎", "F"), "Attach file"
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

        tools_layout.addWidget(self.attach_file_button)
        tools_layout.addWidget(self.share_canvas_button)
        tools_layout.addWidget(self.share_window_button)
        tools_layout.addWidget(self.citation_button)
        tools_layout.addWidget(self.zulip_button)
        input_row.addLayout(tools_layout)
        input_row.addStretch(1)

        send_layout = QtWidgets.QHBoxLayout()
        send_layout.setContentsMargins(0, 0, 0, 0)
        send_layout.setSpacing(4)

        self.talk_button = QtWidgets.QToolButton(self)
        self.talk_button.setObjectName("talkButton")
        self.talk_button.setText(_symbol_text("🎤", "Mic"))
        self.talk_button.setToolTip("Record voice input")
        self.talk_button.setFixedSize(32, 32)

        self.send_button = QtWidgets.QToolButton(self)
        self.send_button.setObjectName("sendButton")
        self.send_button.setText(_symbol_text("🚀", ">"))
        self.send_button.setToolTip("Send message (Ctrl+Enter)")
        self.send_button.setFixedSize(32, 32)

        send_layout.addWidget(self.talk_button)
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

        self.zulip_info_label = QtWidgets.QLabel(
            f"Uses the current draft in the prompt box. Configure Zulip in {get_config_path()}",
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

    def _create_hidden_toggles(self, layout):
        # We might want to expose these via a menu later, but for now defaults
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

        # Shared image label logic expects this object
        self.shared_image_label = QtWidgets.QLabel("", self)
        self.shared_image_label.setVisible(False)
        layout.addWidget(self.shared_image_label)

    def _attach_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Attach File",
            "",
            "All Files (*)",
        )
        if path:
            # For now, just append to prompt with a note
            # In a real implementation we would add it to a list of attachments
            current = self.prompt_text_edit.toPlainText()
            if current:
                current += "\n"
            self.prompt_text_edit.setPlainText(current + f"[Attached: {path}]")

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
        self.clear_chat_button.clicked.connect(self.clear_chat_conversation)
        self.sessions_button.clicked.connect(self.open_session_manager_dialog)
        self._on_prompt_text_changed()
        self._sync_zulip_target_field_visibility()

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
        cfg = load_config().tools.zulip
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

        zulip_cfg = load_config().tools.zulip.to_dict()
        missing = missing_zulip_config_fields(zulip_cfg)
        if missing:
            detail = ", ".join(missing)
            self.status_label.setText(f"Configure Zulip first: {detail}")
            QtWidgets.QMessageBox.warning(
                self,
                "Zulip not configured",
                f"Annolid Bot needs a valid Zulip configuration before it can send.\n\nMissing: {detail}\n\nConfig file: {get_config_path()}",
            )
            return

        try:
            target = build_zulip_draft_target(
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
            _ZulipSendTask(
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

        # If already selected, maybe toggle off?
        # But user wants to click and then click - to remove.
        # So we must keep it selected.
        self._selected_quick_action_index = index

        for idx, btn in enumerate(self.quick_action_buttons):
            btn.setChecked(idx == index)

        # Update the removal button state immediately
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
        try:
            save_llm_settings(self.llm_settings)
        except Exception:
            pass

    def _set_remove_quick_action_enabled(self, enabled: bool) -> None:
        button = getattr(self, "remove_quick_action_button", None)
        if button is None:
            return
        try:
            button.setEnabled(bool(enabled))
        except RuntimeError:
            # Dialog/widget can be shutting down while queued callbacks still arrive.
            return

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
        zulip_button = getattr(self, "zulip_send_button", None)
        if zulip_button is not None:
            zulip_button.setEnabled(bool(text.strip()) and not self._zulip_send_active)
        self._update_slash_completion_ui()
        self._refresh_selected_capability_chips()

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
            # Keep latest provider/model reasoning visible at the top.
            lines.extend(self._progress_lines[:32])
        if not lines:
            dots = "." * ((self._typing_tick % 3) + 1)
            lines = [f"Thinking{dots}"]
        self._current_response_bubble.set_thinking_text(
            "\n".join(lines),
            in_progress=True,
        )
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

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
            # Some providers stream reasoning as a progressively longer prefix.
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

            # Theme-aware colors
            if is_dark:
                bg_main = "#111317"
                bg_input = "#1a1d23"
                fg_main = "#e7e8ea"
                border_main = "#31353c"
                title_fg = "#f4f5f6"
                subtitle_fg = "#9ea4af"
                bubble_user_bg = "#2b5c54"
                bubble_user_border = "#3a756b"
                bubble_assistant_bg = "#1f2228"
                bubble_assistant_border = "#31353e"
            else:
                bg_main = "#ffffff"
                bg_input = "#f1f3f4"
                fg_main = "#202124"
                border_main = "#dadce0"
                title_fg = "#1a73e8"
                subtitle_fg = "#5f6368"
                bubble_user_bg = "#e8f0fe"
                bubble_user_border = "#c6dafc"
                bubble_assistant_bg = "#f1f3f4"
                bubble_assistant_border = "#e0e0e0"

            self.setStyleSheet(
                f"""
                QWidget#AIChatWidget {{
                    background: {bg_main};
                    color: {fg_main};
                    border: none;
                    outline: none;
                }}
                /* Specific containers to avoid broad QWidget styling */
                #chatContainer, #headerBar, #inputBarContainer {{
                    background: {bg_main};
                    color: {fg_main};
                    border: none;
                }}
                QComboBox {{
                    border: 1px solid {border_main};
                    border-radius: 8px;
                    background: {bg_input};
                    color: {fg_main};
                    min-height: 24px;
                    padding: 3px 8px;
                }}
                QLineEdit {{
                    border: 1px solid {border_main};
                    border-radius: 8px;
                    background: {bg_input};
                    color: {fg_main};
                    min-height: 24px;
                    padding: 3px 8px;
                }}
                QScrollArea {{
                    border: none;
                    background: {bg_main};
                }}
                QScrollArea > QWidget > QWidget {{
                    border: none;
                    background: {bg_main};
                }}
                QPlainTextEdit {{
                    border: 1px solid {border_main};
                    border-radius: 12px;
                    background: {bg_input};
                    padding: 8px;
                    font-size: 14px;
                    color: {fg_main};
                }}
                QPlainTextEdit:focus {{
                     border: 1px solid {title_fg};
                }}
                QLabel#chatTitleLabel {{
                    color: {title_fg};
                    font-size: 18px;
                    font-weight: 700;
                    background: transparent;
                }}
                QLabel#chatSubtitleLabel {{
                    color: {subtitle_fg};
                    font-size: 12px;
                    background: transparent;
                }}
                QLabel#chatCapabilitySummaryLabel {{
                    color: {fg_main};
                    font-size: 11px;
                    background: rgba(128, 128, 128, 0.10);
                    border: 1px solid {border_main};
                    border-radius: 10px;
                    padding: 2px 8px;
                }}
                QPushButton#quickActionButton {{
                    border: 1px solid {border_main};
                    border-radius: 14px;
                    padding: 6px 12px;
                    font-size: 12px;
                    background: {bg_input};
                    color: {fg_main};
                }}
                QPushButton#quickActionButton:hover {{
                    background: {bubble_assistant_border};
                }}
                QPushButton#sendButton {{
                    border: 1px solid {bubble_user_border};
                    border-radius: 12px;
                    padding: 6px 12px;
                    background: {bubble_user_bg};
                    color: {fg_main};
                    font-weight: 600;
                }}
                QPushButton#sendButton:disabled {{
                    color: {subtitle_fg};
                }}
                /* Input Bar Icons */
                QToolButton#chatInputButton {{
                    background: transparent;
                    border: none;
                    border-radius: 6px;
                }}
                QToolButton#chatInputButton:hover {{
                    background: rgba(128, 128, 128, 0.15);
                }}
                /* Chat Bubbles */
                QFrame#chatBubble {{
                     border-radius: 12px;
                }}
                QFrame#chatBubble[role="user"] {{
                    background-color: {bubble_user_bg};
                    border: 1px solid {bubble_user_border};
                    margin-left: 40px;
                }}
                QFrame#chatBubble[role="assistant"] {{
                    background-color: {bubble_assistant_bg};
                    border: 1px solid {bubble_assistant_border};
                    margin-right: 40px;
                }}
                /* Bubble Actions */
                QPushButton#bubbleActionButton {{
                    background: transparent;
                    border: none;
                    border-radius: 4px;
                    padding: 2px;
                }}
                QPushButton#bubbleActionButton:hover {{
                     background: rgba(128, 128, 128, 0.1);
                }}
                QToolButton#thinkingToggleButton {{
                    border: none;
                    background: transparent;
                    color: {subtitle_fg};
                    font-size: 11px;
                    font-weight: 600;
                    text-align: left;
                    padding: 2px 0;
                }}
                QToolButton#thinkingToggleButton:hover {{
                    color: {fg_main};
                }}
                QTextBrowser#thinkingView {{
                    border: 1px solid {border_main};
                    border-radius: 8px;
                    background: rgba(128, 128, 128, 0.08);
                    padding: 6px 8px;
                }}
                QLabel#sender {{
                    color: {subtitle_fg};
                    font-size: 11px;
                    font-weight: 600;
                    margin-bottom: 2px;
                    background: transparent;
                }}
                QLabel#meta {{
                    color: {subtitle_fg};
                    font-size: 10px;
                    background: transparent;
                }}
                /* Floating Input Bar */
                QFrame#inputBarContainer {{
                    background: {bg_input};
                    border-top: 1px solid {border_main};
                    border-bottom-left-radius: 12px;
                    border-bottom-right-radius: 12px;
                }}
                QFrame#zulipDraftPanel {{
                    background: {bg_main};
                    border: 1px solid {border_main};
                    border-radius: 12px;
                }}
                QLabel#chatStatusLabel {{
                    color: {subtitle_fg};
                    font-size: 11px;
                    background: transparent;
                }}
                """
            )
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

    def _default_citation_bib_path(self) -> Path:
        return get_agent_workspace_path() / "citations.bib"

    @staticmethod
    def _extract_doi_for_citation(text: str) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        match = re.search(
            r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b",
            raw,
            flags=re.IGNORECASE,
        )
        return str(match.group(1) or "").rstrip(").,;!?") if match else ""

    @staticmethod
    def _extract_year_for_citation(text: str) -> str:
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(text or ""))
        return years[0] if years else ""

    @staticmethod
    def _normalize_citation_key_for_ui(
        title: str, year: str, fallback: str = "paper"
    ) -> str:
        text = str(title or fallback or "paper").strip().lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        stem = "_".join(tokens[:3]) if tokens else "paper"
        yr = str(year or "").strip()
        key = f"{stem}_{yr}" if yr else stem
        key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", key).strip("_")
        return key or "paper"

    def _citation_entry_from_active_pdf(self) -> Optional[BibEntry]:
        self.bot_pdf_get_state()
        state = self.get_bot_action_result("pdf_get_state") or dict(
            self._bot_action_result or {}
        )
        if not bool(state.get("ok")) or not bool(state.get("has_pdf")):
            return None
        path = str(state.get("path") or "").strip()
        title = str(state.get("title") or "").strip()
        if title.lower().endswith(".pdf"):
            title = title[:-4]
        self.bot_pdf_get_text(9000, 2)
        text_payload = self.get_bot_action_result("pdf_get_text") or dict(
            self._bot_action_result or {}
        )
        text = str(text_payload.get("text") or "")
        doi = self._extract_doi_for_citation(text)
        year = self._extract_year_for_citation(text)
        fields: Dict[str, str] = {
            "title": title or Path(path).stem.replace("_", " "),
            "note": "Saved from active Annolid PDF viewer.",
            "source_path": path,
        }
        if year:
            fields["year"] = year
        if doi:
            fields["doi"] = doi
            fields["url"] = f"https://doi.org/{doi}"
        arxiv_match = re.search(
            r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if arxiv_match:
            fields["archiveprefix"] = "arXiv"
            fields["eprint"] = str(arxiv_match.group(1) or "").strip()
        key = self._normalize_citation_key_for_ui(
            fields["title"], fields.get("year", "")
        )
        return BibEntry(entry_type="article", key=key, fields=fields)

    def _citation_entry_from_active_web(self) -> Optional[BibEntry]:
        self.bot_web_get_state()
        state = self.get_bot_action_result("web_get_state") or dict(
            self._bot_action_result or {}
        )
        if not bool(state.get("ok")) or not bool(state.get("has_page")):
            return None
        url = str(state.get("url") or "").strip()
        if not url:
            return None
        title = str(state.get("title") or "").strip()
        self.bot_web_get_dom_text(9000)
        text_payload = self.get_bot_action_result("web_get_dom_text") or dict(
            self._bot_action_result or {}
        )
        text = str(text_payload.get("text") or "")
        doi = self._extract_doi_for_citation(f"{url}\n{text}")
        year = self._extract_year_for_citation(text)
        fields: Dict[str, str] = {
            "title": title or "Web page citation",
            "url": url,
            "note": "Saved from active Annolid web viewer.",
        }
        if year:
            fields["year"] = year
        if doi:
            fields["doi"] = doi
        arxiv_match = re.search(
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
            url,
            flags=re.IGNORECASE,
        )
        if arxiv_match:
            fields["archiveprefix"] = "arXiv"
            fields["eprint"] = str(arxiv_match.group(1) or "").strip()
        key = self._normalize_citation_key_for_ui(
            fields["title"], fields.get("year", "")
        )
        return BibEntry(entry_type="article", key=key, fields=fields)

    def _save_citation_from_active_context(
        self,
        *,
        source: str = "auto",
        key: str = "",
        bib_path: Optional[Path] = None,
        validate_before_save: bool = True,
        strict_validation: bool = False,
        verify_after_save: bool = False,
    ) -> Dict[str, Any]:
        source_norm = str(source or "auto").strip().lower()
        if source_norm not in {"auto", "pdf", "web"}:
            source_norm = "auto"
        entry: Optional[BibEntry] = None
        used_source = source_norm
        if source_norm in {"auto", "pdf"}:
            entry = self._citation_entry_from_active_pdf()
            if entry:
                used_source = "pdf"
        if entry is None and source_norm in {"auto", "web"}:
            entry = self._citation_entry_from_active_web()
            if entry:
                used_source = "web"
        if entry is None:
            return {
                "ok": False,
                "error": "No active PDF/web context found to create citation.",
            }
        basic_errors = validate_basic_citation_fields(
            {
                "__key__": str(key or "").strip(),
                "year": str(entry.fields.get("year") or ""),
                "doi": str(entry.fields.get("doi") or ""),
            }
        )
        if basic_errors:
            return {"ok": False, "error": " ".join(basic_errors)}
        chosen_key = str(key or "").strip()
        if chosen_key:
            entry.key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", chosen_key).strip("_")
        validation: Dict[str, Any] = {
            "checked": False,
            "verified": False,
            "provider": "",
            "score": 0.0,
            "message": "",
            "candidate": {},
        }
        if bool(validate_before_save):
            validation = validate_citation_metadata(entry.fields, timeout_s=1.8)
            entry.fields = merge_validated_fields(
                entry.fields, validation, replace_when_confident=True
            )
            if bool(strict_validation) and not bool(validation.get("verified")):
                return {
                    "ok": False,
                    "error": (
                        "Citation validation failed strict mode. "
                        + str(validation.get("message") or "")
                    ).strip(),
                    "validation": validation,
                }
            if not chosen_key:
                candidate_key = str(
                    dict(validation.get("candidate") or {}).get("__bibkey__") or ""
                ).strip()
                if candidate_key:
                    entry.key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", candidate_key).strip(
                        "_"
                    )
                else:
                    entry.key = self._normalize_citation_key_for_ui(
                        str(entry.fields.get("title") or "").strip(),
                        str(entry.fields.get("year") or "").strip(),
                    )
        target_bib = bib_path or self._default_citation_bib_path()
        target_bib = Path(target_bib).expanduser()
        target_bib.parent.mkdir(parents=True, exist_ok=True)
        entries = load_bibtex(target_bib) if target_bib.exists() else []
        updated, created = upsert_entry(entries, entry)
        save_bibtex(target_bib, updated, sort_keys=True)
        payload: Dict[str, Any] = {
            "ok": True,
            "created": bool(created),
            "key": entry.key,
            "bib_file": str(target_bib),
            "source": used_source,
            "validation": validation,
        }
        if bool(verify_after_save):
            report = build_citation_verification_report(
                key=str(entry.key or ""),
                bib_file=str(target_bib),
                source=used_source,
                fields=dict(entry.fields),
                validation=validation,
            )
            payload["verification"] = dict(report.get("verification") or {})
            report_path = write_citation_verification_report(
                report,
                reports_dir=target_bib.parent
                / ".annolid_cache"
                / "citation_verification",
                report_stem=f"{target_bib.stem}_{entry.key}",
            )
            payload["verification_report"] = str(report_path)
        return payload

    def _open_citation_manager(self) -> None:
        dialog = CitationManagerDialog(
            default_bib_path_getter=self._default_citation_bib_path,
            save_from_context=self._save_citation_from_active_context,
            parent=self,
        )
        dialog.exec_()

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

    @QtCore.Slot(str)
    def bot_open_image(self, image_path: str) -> None:
        path_text = str(image_path or "").strip()
        if not path_text:
            self.status_label.setText("Bot action failed: empty image path.")
            return
        resolved = Path(path_text).expanduser()
        if not resolved.exists() or not resolved.is_file():
            self.status_label.setText(
                f"Bot action failed: image not found: {path_text}"
            )
            return
        host = self.host_window_widget or self.window()
        load_file = getattr(host, "loadFile", None)
        if not callable(load_file):
            self.status_label.setText("Bot action failed: image loader is unavailable.")
            return
        try:
            load_file(str(resolved))
            set_view = getattr(host, "_set_active_view", None)
            if callable(set_view):
                set_view("canvas")
            self.status_label.setText(f"Opened image on canvas: {resolved.name}")
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

    def _normalize_url_for_open(self, url: str) -> Optional[QtCore.QUrl]:
        value = str(url or "").strip()
        if not value:
            return None
        parsed = QtCore.QUrl(value)
        if parsed.isValid() and parsed.scheme().lower() in {"http", "https", "file"}:
            return parsed
        local_path = Path(value).expanduser()
        if local_path.exists() and local_path.is_file():
            return QtCore.QUrl.fromLocalFile(str(local_path))
        if "://" not in value:
            fallback = QtCore.QUrl(f"https://{value}")
            if fallback.isValid() and fallback.scheme().lower() in {"http", "https"}:
                return fallback
        return None

    def _open_url_in_web_viewer(self, parsed: QtCore.QUrl) -> bool:
        normalized = str(parsed.toString() or "").strip()
        if not normalized:
            return False
        host = self.host_window_widget or self.window()
        show_web = getattr(host, "show_web_in_viewer", None)
        if callable(show_web) and bool(show_web(normalized)):
            return True
        manager = self._resolve_web_manager()
        if manager is not None and hasattr(manager, "show_url_in_viewer"):
            try:
                return bool(manager.show_url_in_viewer(normalized))
            except Exception:
                return False
        return False

    def _open_chat_link_default(self, url: str) -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Invalid URL.")
            return
        normalized = str(parsed.toString() or "").strip()
        if self._open_url_in_web_viewer(parsed):
            self.status_label.setText(
                f"Opened URL in embedded web viewer: {normalized}"
            )
            return
        if QtGui.QDesktopServices.openUrl(parsed):
            self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Could not open URL.")

    def _open_chat_link_in_browser(self, url: str, mode: str = "default") -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Invalid URL.")
            return
        normalized = str(parsed.toString() or "").strip()
        request = str(mode or "default").strip().lower()
        opened = False
        if request == "new_tab":
            opened = bool(webbrowser.open(normalized, new=2, autoraise=True))
        elif request == "new_window":
            opened = bool(webbrowser.open(normalized, new=1, autoraise=True))
        else:
            opened = QtGui.QDesktopServices.openUrl(parsed)
        if not opened:
            opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            if request == "new_tab":
                self.status_label.setText(
                    f"Requested opening in browser new tab: {normalized}"
                )
            elif request == "new_window":
                self.status_label.setText(
                    f"Requested opening in browser new window: {normalized}"
                )
            else:
                self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Could not open URL in browser.")

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
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Bot action failed: invalid URL or file path.")
            return
        normalized = str(parsed.toString() or "").strip()
        if self._open_url_in_web_viewer(parsed):
            self.status_label.setText(f"Opened URL in canvas: {normalized}")
            return
        opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            self.status_label.setText(f"Opened URL in browser: {normalized}")
            return
        self.status_label.setText("Bot action failed: could not open URL.")

    @QtCore.Slot(str)
    def bot_open_in_browser(self, url: str) -> None:
        parsed = self._normalize_url_for_open(url)
        if parsed is None:
            self.status_label.setText("Bot action failed: invalid URL.")
            return
        target_url = str(parsed.toString() or "").strip()
        opened = QtGui.QDesktopServices.openUrl(parsed)
        if opened:
            self.status_label.setText(f"Opened URL in browser: {target_url}")
            return
        self.status_label.setText("Bot action failed: could not open URL.")

    @staticmethod
    def _normalize_threejs_example_input(example: str) -> str:
        raw = str(example or "").strip().lower()
        if not raw:
            return "two_mice_html"
        mapping = {
            "helix": "helix_points_csv",
            "helix_points_csv": "helix_points_csv",
            "wave": "wave_surface_obj",
            "wave_surface_obj": "wave_surface_obj",
            "sphere": "sphere_points_ply",
            "sphere_points_ply": "sphere_points_ply",
            "brain": "brain_viewer_html",
            "brain_viewer_html": "brain_viewer_html",
            "two_mice": "two_mice_html",
            "two_mice_html": "two_mice_html",
            "fly_brain": "fly_brain_mouse_html",
            "fly_brain_mouse": "fly_brain_mouse_html",
            "fly_brain_mouse_html": "fly_brain_mouse_html",
            "rat": "rat_ncap_html",
            "rat_ncap": "rat_ncap_html",
            "rat_ncap_html": "rat_ncap_html",
            "rat_ncap_simulation_json": "rat_ncap_simulation_json",
        }
        normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if normalized in mapping:
            return mapping[normalized]
        if "rat" in raw:
            return "rat_ncap_html"
        if "fly" in raw and ("brain" in raw or "mouse" in raw or "mice" in raw):
            return "fly_brain_mouse_html"
        if "two" in raw and "mice" in raw:
            return "two_mice_html"
        if "brain" in raw:
            return "brain_viewer_html"
        if "helix" in raw:
            return "helix_points_csv"
        if "wave" in raw:
            return "wave_surface_obj"
        if "sphere" in raw:
            return "sphere_points_ply"
        return "two_mice_html"

    @QtCore.Slot(str)
    def bot_open_threejs_example(self, example_id: str = "") -> None:
        host = self.host_window_widget or self.window()
        open_example = getattr(host, "open_threejs_example", None)
        if not callable(open_example):
            self.status_label.setText(
                "Bot action failed: Three.js examples unavailable."
            )
            return
        resolved = self._normalize_threejs_example_input(example_id)
        try:
            open_example(resolved)
            self.status_label.setText(f"Opened Three.js example: {resolved}")
        except Exception as exc:
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_open_threejs(self, path_or_url: str) -> None:
        target = str(path_or_url or "").strip()
        if not target:
            self.status_label.setText("Bot action failed: provide a 3D path or URL.")
            return
        host = self.host_window_widget or self.window()
        manager = getattr(host, "threejs_manager", None)
        if manager is None:
            ensure_threejs_manager = getattr(host, "ensure_threejs_manager", None)
            if callable(ensure_threejs_manager):
                try:
                    manager = ensure_threejs_manager()
                except Exception:
                    manager = None
        if manager is None:
            self.status_label.setText(
                "Bot action failed: Three.js viewer is unavailable."
            )
            return

        local_path = Path(target).expanduser()
        try:
            is_local_zarr = (
                local_path.exists()
                and local_path.is_dir()
                and local_path.suffix.lower() == ".zarr"
            )
            if local_path.exists() and (local_path.is_file() or is_local_zarr):
                suffix = local_path.suffix.lower()
                if suffix in {".html", ".htm", ".xhtml"}:
                    url = QtCore.QUrl.fromLocalFile(str(local_path)).toString()
                    if bool(manager.show_url_in_viewer(url)):
                        self.status_label.setText(
                            f"Opened Three.js URL: {local_path.name}"
                        )
                        return
                elif bool(manager.show_model_in_viewer(local_path)):
                    self.status_label.setText(
                        f"Opened Three.js model: {local_path.name}"
                    )
                    return
        except Exception:
            pass

        parsed = self._normalize_url_for_open(target)
        if parsed is not None:
            normalized = str(parsed.toString() or "").strip()
            scheme = str(parsed.scheme() or "").lower()
            if normalized and scheme in {"http", "https", "file"}:
                try:
                    if bool(manager.show_url_in_viewer(normalized)):
                        self.status_label.setText(f"Opened Three.js URL: {normalized}")
                        return
                except Exception:
                    pass

        self.status_label.setText("Bot action failed: could not open Three.js content.")

    @QtCore.Slot(int)
    def bot_web_get_dom_text(self, max_chars: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_get_dom_text", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.get_page_text(max_chars=int(max_chars))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_get_dom_text", payload)
        self.status_label.setText(
            "Captured page text." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_web_get_state(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_get_state", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.get_web_state()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_get_state", payload)

    @QtCore.Slot(int)
    def bot_web_capture_screenshot(self, max_width: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_capture_screenshot", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.capture_screenshot(max_width=int(max_width))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_capture_screenshot", payload)
        self.status_label.setText(
            "Captured web screenshot." if payload.get("ok") else "Bot action failed."
        )

    @QtCore.Slot(int)
    def bot_web_describe_view(self, max_width: int) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_describe_view", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.describe_current_view(max_width=int(max_width))
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_describe_view", payload)
        self.status_label.setText(
            "Submitted screenshot for description."
            if payload.get("ok")
            else "Bot action failed."
        )

    @QtCore.Slot()
    def bot_pdf_get_state(self) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result("pdf_get_state", payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_state()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("pdf_get_state", payload)

    @QtCore.Slot(int, int)
    @QtCore.Slot(int, int, int)
    def bot_pdf_get_text(self, max_chars: int, pages: int, start_page: int = 0) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result("pdf_get_text", payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_text(
                max_chars=int(max_chars),
                pages=int(pages),
                start_page=int(start_page or 0),
            )
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("pdf_get_text", payload)
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

    @QtCore.Slot()
    def bot_web_save_current(self) -> None:
        manager = self._resolve_web_manager()
        if manager is None:
            payload = {"ok": False, "error": "Embedded web manager is unavailable."}
            self._set_bot_action_result("web_save_current", payload)
            self.status_label.setText("Bot action failed: web manager unavailable.")
            return
        try:
            payload = manager.save_current_page()
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        self._set_bot_action_result("web_save_current", payload)
        if payload.get("ok"):
            if payload.get("queued"):
                self.status_label.setText("Saving web content in background...")
            else:
                saved_path = str(payload.get("last_saved_path") or "").strip()
                self.status_label.setText(
                    f"Saved web content: {os.path.basename(saved_path)}"
                    if saved_path
                    else "Saved web content."
                )
            return
        self.status_label.setText("Bot action failed.")

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

    # ---- Research Intent Handling ----

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
        slash_state = _extract_slash_selection_state(raw_prompt)
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

        self._add_bubble("You", raw_prompt, is_user=True)
        self.prompt_text_edit.clear()

        chat_mode = self._consume_next_chat_mode()
        chat_image_path = self._prepare_chat_image(chat_mode=chat_mode)
        ui_prepared = False
        if not self._chat_task_active:
            self._prepare_streaming_turn_ui(raw_prompt)
            ui_prepared = True
        else:
            pending = int(self._chat_message_bus.inbound.qsize()) + 1
            self.status_label.setText(f"Queued message ({pending} pending)")

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
        inbound = InboundMessage(
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
        self.send_button.setEnabled(False)
        self.is_streaming_chat = True
        self._start_typing_indicator()

    def _drain_inbound_bus_messages(self) -> None:
        if self._chat_task_active:
            return
        try:
            inbound = self._chat_message_bus.inbound.get_nowait()
        except asyncio.QueueEmpty:
            self._chat_inbound_bus_timer.stop()
            return
        inbound_meta = dict(getattr(inbound, "metadata", {}) or {})
        if not bool(inbound_meta.get("ui_prepared", False)):
            self._prepare_streaming_turn_ui(str(getattr(inbound, "content", "") or ""))
            inbound_meta["ui_prepared"] = True
            inbound = InboundMessage(
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
        task = StreamingChatTask(
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
        self._chat_message_bus.outbound.put_nowait(
            OutboundMessage(
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
        event = decode_outbound_chat_event(payload_text)
        if event is None:
            return
        event_turn_id = str(getattr(event, "turn_id", "") or "").strip()
        active_turn_id = str(getattr(self, "_active_turn_id", "") or "").strip()
        if event_turn_id and active_turn_id and event_turn_id != active_turn_id:
            # Drop stale chunks/progress/final events from a cancelled or previous turn.
            return
        if event_turn_id and not active_turn_id and not self._chat_task_active:
            # Drop delayed events arriving after a turn was already stopped/finished.
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
        self._current_response_bubble.append_text(chunk)
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

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
            # Guard against duplicate final events emitted for the same turn.
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
        self._on_prompt_text_changed()
        self._active_chat_task = None
        self._active_turn_id = ""
        if self._current_response_bubble is not None:
            self._current_response_bubble.set_stop_visible(False)
        self._current_response_bubble = None
        self._current_progress_bubble = None
        self._progress_lines = []
        self._last_final_message_text = text.strip()
        self._last_final_message_is_error = bool(is_error)
        self._last_final_message_ts = now
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        if self._chat_message_bus.inbound.qsize() > 0:
            if not self._chat_inbound_bus_timer.isActive():
                self._chat_inbound_bus_timer.start()
            QtCore.QTimer.singleShot(0, self._drain_inbound_bus_messages)

    def _stop_running_response(self) -> None:
        if not self._chat_task_active:
            return
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
        # Find last non-user bubble text.
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
