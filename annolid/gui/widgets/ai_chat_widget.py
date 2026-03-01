from __future__ import annotations

import asyncio
import contextlib
import csv
import ipaddress
import fnmatch
from collections import OrderedDict
import os
import tempfile
import time
import webbrowser
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QThreadPool

from annolid.core.agent.bus import InboundMessage, MessageBus, OutboundMessage
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.datasets.labelme_collection import default_label_index_path
from annolid.gui.realtime_launch import (
    build_realtime_launch_payload,
    resolve_realtime_model_weight,
)
from annolid.gui.models_registry import (
    get_model_unavailable_reason,
    get_runtime_model_registry,
)
from annolid.gui.widgets.ai_chat_audio_controller import ChatAudioController
from annolid.gui.widgets.ai_chat_backend import (
    StreamingChatTask,
    clear_chat_session,
)
from annolid.core.agent.gui_backend.session_io import decode_outbound_chat_event
from annolid.core.agent.gui_backend.turn_state import (
    ERROR_TYPE_CANCELLED,
    TURN_STATUS_CANCELLED,
)
from annolid.gui.widgets.ai_chat_session_dialog import ChatSessionManagerDialog
from annolid.gui.widgets.citation_manager_widget import CitationManagerDialog
from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog
from annolid.gui.widgets.provider_registry import ProviderRegistry
from annolid.gui.widgets.provider_runtime_sync import (
    refresh_runtime_llm_settings as refresh_runtime_provider_settings,
)
from annolid.utils.log_paths import (
    resolve_annolid_logs_root,
    resolve_annolid_realtime_logs_root,
)
from annolid.utils.runs import shared_runs_root
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


def _safe_stream_source_for_bot(source: str) -> str:
    text = str(source or "").strip()
    if not text or text.isdigit() or "://" not in text:
        return text
    try:
        parts = urlsplit(text)
    except Exception:
        return text
    host = str(parts.hostname or "").strip()
    if not host:
        return text
    redact = host.lower() == "localhost"
    if not redact:
        try:
            ip_obj = ipaddress.ip_address(host)
            redact = bool(
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
            )
        except Exception:
            redact = False
    safe_netloc = parts.netloc
    if "@" in safe_netloc:
        safe_netloc = safe_netloc.split("@", 1)[1]
    if not redact:
        return urlunsplit(
            (parts.scheme, safe_netloc, parts.path, parts.query, parts.fragment)
        )
    port = f":{parts.port}" if parts.port else ""
    replacement = f"<private-host>{port}"
    if parts.scheme.lower() in {"rtp", "udp"}:
        replacement = f"@<private-host>{port}"
    return urlunsplit(
        (parts.scheme, replacement, parts.path, parts.query, parts.fragment)
    )


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
        on_stop=None,
        on_open_link=None,
        on_open_link_in_browser=None,
        allow_regenerate: bool = False,
        allow_stop: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sender = sender
        self._is_user = is_user
        self._ts = datetime.now().strftime("%H:%M")
        self._on_speak = on_speak
        self._on_copy = on_copy
        self._on_regenerate = on_regenerate
        self._on_stop = on_stop
        self._on_open_link = on_open_link
        self._on_open_link_in_browser = on_open_link_in_browser
        self._allow_regenerate = bool(allow_regenerate)
        self._allow_stop = bool(allow_stop)
        self._raw_text = str(text or "")
        # self._preferred_text_width = 600 # Removed for full width
        self._manual_text_width: Optional[int] = None
        self._resize_drag_active = False
        self._drag_anchor_global_x = 0
        self._drag_start_text_width = 0
        self._edge_handle_px = 12
        self._syncing_message_height = False

        self.setObjectName("chatBubble")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setProperty("role", "user" if is_user else "assistant")

        # Main layout for the bubble
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 10, 12, 10)
        self.main_layout.setSpacing(4)

        # Header: Sender + Timestamp + Buttons (hidden by default, show on hover?)
        # For now, keep buttons in a footer or header. Let's try header for cleaner look?
        # Actually footer is standard for "actions".

        # Sender Label
        if not self._is_user:
            self.sender_label = QtWidgets.QLabel(sender, self)
            self.sender_label.setObjectName("sender")
            self.main_layout.addWidget(self.sender_label)

        # Message Body
        self.message_view = QtWidgets.QTextBrowser(self)
        self.message_view.setObjectName("messageView")
        self.message_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.message_view.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.message_view.setAutoFillBackground(False)
        self.message_view.setOpenExternalLinks(False)
        self.message_view.setOpenLinks(False)
        self.message_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.message_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.message_view.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents
        )
        self.message_view.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse
        )
        self.message_view.anchorClicked.connect(self._handle_anchor_clicked)
        self.message_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.message_view.customContextMenuRequested.connect(
            self._show_message_context_menu
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

        self.main_layout.addWidget(self.message_view)
        self._render_markdown(self._raw_text)

        # Footer: Actions + Timestamp
        self.footer_layout = QtWidgets.QHBoxLayout()
        self.footer_layout.setContentsMargins(0, 4, 0, 0)
        self.footer_layout.setSpacing(8)

        # Action Buttons
        self.actions_layout = QtWidgets.QHBoxLayout()
        self.actions_layout.setSpacing(4)

        self.speak_button = self._create_action_button("🔊", "Read aloud")
        self.copy_button = self._create_action_button("📋", "Copy text")
        self.regenerate_button = self._create_action_button("🔄", "Regenerate")
        self.stop_button = self._create_action_button("⏹", "Stop running")
        self.regenerate_button.setVisible(
            (not self._is_user) and self._allow_regenerate
        )
        self.stop_button.setVisible((not self._is_user) and self._allow_stop)

        # Connect callbacks
        self.speak_button.clicked.connect(self._speak)
        self.copy_button.clicked.connect(self._copy_text)
        self.regenerate_button.clicked.connect(self._regenerate)
        self.stop_button.clicked.connect(self._stop)

        self.actions_layout.addWidget(self.speak_button)
        self.actions_layout.addWidget(self.copy_button)
        self.actions_layout.addWidget(self.regenerate_button)
        self.actions_layout.addWidget(self.stop_button)
        self.actions_layout.addStretch(1)

        self.footer_layout.addLayout(self.actions_layout)

        # Timestamp
        self.meta_label = QtWidgets.QLabel(self._ts, self)
        self.meta_label.setObjectName("meta")
        self.meta_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.footer_layout.addWidget(self.meta_label)

        self.main_layout.addLayout(self.footer_layout)

        # Determine theme styling
        self._apply_role_palette()

    def _create_action_button(self, text_icon, tooltip):
        btn = QtWidgets.QPushButton(text_icon, self)
        btn.setObjectName("bubbleActionButton")
        btn.setToolTip(tooltip)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        # Styling for "cool" flat look
        btn.setStyleSheet("""
            QPushButton#bubbleActionButton {
                background: transparent;
                border: none;
                color: #808080;
                font-size: 14px;
                padding: 4px;
                border-radius: 4px;
            }
            QPushButton#bubbleActionButton:hover {
                background: rgba(255, 255, 255, 0.1);
                color: #E0E0E0;
            }
            QPushButton#bubbleActionButton:pressed {
                background: rgba(255, 255, 255, 0.2);
            }
        """)
        btn.setFixedSize(28, 28)
        return btn

    def _apply_role_palette(self) -> None:
        # We handle most coloring in QSS now based on property "role"
        pass

    def _apply_message_doc_styles(self) -> None:
        # Enhanced CSS for internal text browser
        base_css = """
            body, p, li, ul, ol, blockquote {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 13px;
                line-height: 1.4;
            }
            p { margin: 0 0 8px 0; }
            p:last-child { margin-bottom: 0; }
            a { text-decoration: none; font-weight: 500; }
            pre {
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
                font-size: 12px;
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
            }
            code {
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
                font-size: 12px;
                padding: 2px 4px;
                border-radius: 4px;
            }
        """

        if self._is_user:
            self.message_doc.setDefaultStyleSheet(
                base_css
                + """
                body, p, li, ul, ol, blockquote { color: #FFFFFF; }
                a { color: #d0e8ff; }
                pre { background: rgba(0,0,0,0.2); color: #FFFFFF; border: 1px solid rgba(255,255,255,0.2); }
                code { background: rgba(0,0,0,0.2); color: #FFFFFF; }
                """
            )
        else:
            self.message_doc.setDefaultStyleSheet(
                base_css
                + """
                body, p, li, ul, ol, blockquote { color: #E0E0E0; }
                a { color: #64B5F6; }
                pre { background: #2b2d31; color: #E0E0E0; border: 1px solid #3f4148; }
                code { background: #2f3136; color: #E0E0E0; }
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
            # We want the bubble text to be comfortable reading width, but fill if needed.
            # Max width constraint is handled by parent layout/Application.

            # The message view needs to resize to fit its content height
            # Force update text width to match viewport
            vp_width = self.message_view.viewport().width()
            if vp_width > 0:
                self.message_doc.setTextWidth(float(vp_width))

            doc_height = int(self.message_doc.size().height())

            # Add padding: top/bottom margins of layout (10+10) + spacing + safety
            target_height = doc_height + 25

            # Calculate max allowed height (e.g. 60% of window height) to prevent huge bubbles
            # from taking over the entire screen.
            parent_window = self.window()
            max_limit = 600
            if parent_window:
                max_limit = int(max(parent_window.height() * 0.6, 200))

            if target_height > max_limit:
                self.message_view.setFixedHeight(max_limit)
                self.message_view.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAsNeeded
                )
            else:
                self.message_view.setFixedHeight(target_height)
                self.message_view.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAlwaysOff
                )

        finally:
            self._syncing_message_height = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_message_height()

    def set_message_width(self, width: int) -> None:
        # Simplified: The layout handles width now mostly, but we can hint the doc.
        if self.message_doc:
            self.message_doc.setTextWidth(float(width))

    def apply_layout_width(self, text_width: int, bubble_max_width: int) -> None:
        # This is called by parent to enforce max widths
        if self.message_doc:
            self.message_doc.setTextWidth(float(text_width))
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

    def _stop(self) -> None:
        if callable(self._on_stop):
            self._on_stop()

    def set_stop_visible(self, visible: bool) -> None:
        self.stop_button.setVisible(bool(visible))

    def _handle_anchor_clicked(self, url: QtCore.QUrl) -> None:
        target = str(url.toString() if isinstance(url, QtCore.QUrl) else url).strip()
        if not target:
            return
        if callable(self._on_open_link):
            self._on_open_link(target)

    def _show_message_context_menu(self, position: QtCore.QPoint) -> None:
        menu = self.message_view.createStandardContextMenu()
        link = str(self.message_view.anchorAt(position) or "").strip()
        if link:
            menu.addSeparator()

            open_viewer_action = menu.addAction("Open Link in Annolid Web Viewer")
            open_viewer_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_viewer(value)
            )

            open_browser_action = menu.addAction("Open Link in Default Browser")
            open_browser_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_browser(
                    value, "default"
                )
            )

            open_tab_action = menu.addAction("Open Link in Default Browser (New Tab)")
            open_tab_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_browser(
                    value, "new_tab"
                )
            )

            open_window_action = menu.addAction(
                "Open Link in Default Browser (New Window)"
            )
            open_window_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_browser(
                    value, "new_window"
                )
            )

            copy_link_action = menu.addAction("Copy Link Address")
            copy_link_action.triggered.connect(
                lambda _checked=False,
                value=link: QtGui.QGuiApplication.clipboard().setText(value)
            )

        menu.exec_(self.message_view.viewport().mapToGlobal(position))

    def _open_link_in_viewer(self, url: str) -> None:
        target = str(url or "").strip()
        if not target:
            return
        if callable(self._on_open_link):
            self._on_open_link(target)

    def _open_link_in_browser(self, url: str, mode: str) -> None:
        target = str(url or "").strip()
        if not target:
            return
        if callable(self._on_open_link_in_browser):
            self._on_open_link_in_browser(target, str(mode or "default"))

    # Simplified drag resizing - removed for now to clean up, unless strictly needed.
    # It was a bit complex and rarely used feature in chat widgets.


class AIChatWidget(QtWidgets.QWidget):
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
        self._bot_action_result: Dict[str, Any] = {}
        self.session_id = self._new_startup_session_id()
        self._applying_theme_styles = False
        self.thread_pool = QThreadPool()
        self._audio_controller: Optional[ChatAudioController] = None
        self._session_manager = AgentSessionManager()
        self._session_store = PersistentSessionStore(self._session_manager)
        self._max_prompt_chars = 4000
        self._last_user_prompt: str = ""
        self._next_chat_mode: str = "default"
        self._typing_tick = 0
        self._typing_timer = QtCore.QTimer(self)
        self._typing_timer.setInterval(350)
        self._typing_timer.timeout.connect(self._on_typing_timer_tick)
        self._chat_message_bus = MessageBus()
        self._chat_inbound_bus_timer = QtCore.QTimer(self)
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
        self._seen_event_keys: "OrderedDict[str, float]" = OrderedDict()
        self._seen_event_key_limit = 512

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
        title_col.addWidget(self.session_chip_label)

        # Model Selector (Compact)
        self.model_selector = QtWidgets.QComboBox(self)
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.model_selector.setMinimumWidth(150)
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
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # 1. Quick Actions (Horizontal Scroll if needed, or just row)
        self.quick_actions_layout = QtWidgets.QHBoxLayout()
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

        layout.addLayout(self.quick_actions_layout)
        self._refresh_quick_action_buttons()

        # 2. Main Input Row (Attach | Text | Send/Mic)
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(10)

        # Attach / Tools Group
        tools_layout = QtWidgets.QHBoxLayout()
        tools_layout.setSpacing(2)

        self.attach_file_button = self._create_input_icon("📎", "Attach file")
        self.share_canvas_button = self._create_input_icon("🎨", "Share Canvas")
        self.share_window_button = self._create_input_icon("🪟", "Share Window")
        self.citation_button = self._create_input_icon("📚", "Manage citations")

        tools_layout.addWidget(self.attach_file_button)
        tools_layout.addWidget(self.share_canvas_button)
        tools_layout.addWidget(self.share_window_button)
        tools_layout.addWidget(self.citation_button)
        input_row.addLayout(tools_layout)

        # Text Input
        self.prompt_text_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_text_edit.setPlaceholderText("Message Annolid Bot...")
        self.prompt_text_edit.setFixedHeight(50)
        self.prompt_text_edit.setToolTip("Type a message. Use Ctrl+Enter to send.")
        input_row.addWidget(self.prompt_text_edit, 1)

        # Send / Talk Group
        send_layout = QtWidgets.QHBoxLayout()
        send_layout.setSpacing(4)

        self.talk_button = QtWidgets.QToolButton(self)
        self.talk_button.setObjectName("talkButton")
        self.talk_button.setText("🎤")
        self.talk_button.setToolTip("Record voice input")
        self.talk_button.setFixedSize(36, 36)

        self.send_button = QtWidgets.QToolButton(self)
        self.send_button.setObjectName("sendButton")
        self.send_button.setText("🚀")
        self.send_button.setToolTip("Send message (Ctrl+Enter)")
        self.send_button.setFixedSize(36, 36)

        send_layout.addWidget(self.talk_button)
        send_layout.addWidget(self.send_button)
        input_row.addLayout(send_layout)

        layout.addLayout(input_row)

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

    def _create_hidden_toggles(self, layout):
        # We might want to expose these via a menu later, but for now defaults
        self.attach_canvas_checkbox = QtWidgets.QCheckBox("Attach canvas", self)
        self.attach_canvas_checkbox.setChecked(False)
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

    def _share_canvas_now(self) -> None:
        # Toggle checkbox and update state (if available)
        if not self.attach_canvas_checkbox.isChecked():
            self.attach_canvas_checkbox.setChecked(True)
        QtWidgets.QMessageBox.information(
            self, "Canvas Attached", "Canvas image will be sent with your next message."
        )

    def _share_window_now(self) -> None:
        if not self.attach_window_checkbox.isChecked():
            self.attach_window_checkbox.setChecked(True)
        QtWidgets.QMessageBox.information(
            self,
            "Window Attached",
            "Window screenshot will be sent with your next message.",
        )

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
        self.attach_file_button.clicked.connect(self._attach_file)
        self.talk_button.clicked.connect(self.toggle_recording)
        self.clear_chat_button.clicked.connect(self.clear_chat_conversation)
        self.sessions_button.clicked.connect(self.open_session_manager_dialog)
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
                if widget in (
                    getattr(self, "add_quick_action_button", None),
                    getattr(self, "remove_quick_action_button", None),
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
        self.quick_actions_layout.addWidget(self.add_quick_action_button)
        self.quick_actions_layout.addWidget(self.remove_quick_action_button)
        self._set_remove_quick_action_enabled(
            self._selected_quick_action_index is not None
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
        if self._has_streamed_response_chunk:
            return
        dots = "." * ((self._typing_tick % 3) + 1)
        lines: List[str] = [f"Thinking{dots}"]
        if self.enable_progress_stream and self._progress_lines:
            for line in self._progress_lines[-3:]:
                lines.append(f"- {line}")
        self._current_response_bubble.set_text("\n".join(lines))
        QtCore.QTimer.singleShot(0, self._reflow_chat_bubbles)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

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
            palette = self.palette()
            is_dark = palette.color(QtGui.QPalette.Window).lightness() < 128

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
                QLabel#chatStatusLabel {{
                    color: {subtitle_fg};
                    font-size: 11px;
                    background: transparent;
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
        return {
            "ok": True,
            "created": bool(created),
            "key": entry.key,
            "bib_file": str(target_bib),
            "source": used_source,
            "validation": validation,
        }

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
        }
        normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if normalized in mapping:
            return mapping[normalized]
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
            self.status_label.setText(
                "Bot action failed: Three.js viewer is unavailable."
            )
            return

        local_path = Path(target).expanduser()
        try:
            if local_path.exists() and local_path.is_file():
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
    def bot_pdf_get_text(self, max_chars: int, pages: int) -> None:
        manager = self._resolve_pdf_manager()
        if manager is None:
            payload = {"ok": False, "error": "PDF manager is unavailable."}
            self._set_bot_action_result("pdf_get_text", payload)
            self.status_label.setText("Bot action failed: PDF manager unavailable.")
            return
        try:
            payload = manager.get_pdf_text(max_chars=int(max_chars), pages=int(pages))
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

    def set_image_path(self, image_path: str) -> None:
        if image_path and image_path not in self._snapshot_paths:
            self._cleanup_snapshots()
        self.image_path = image_path
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
        self._refresh_runtime_llm_settings()
        raw_prompt = self.prompt_text_edit.toPlainText().strip()
        if not raw_prompt:
            return
        if not self._ensure_provider_ready():
            return

        self._add_bubble("You", raw_prompt, is_user=True)
        self.prompt_text_edit.clear()
        chat_mode = self._consume_next_chat_mode()
        chat_image_path = self._prepare_chat_image()
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
            "ui_prepared": bool(ui_prepared),
        }
        inbound = InboundMessage(
            channel="gui",
            sender_id="gui_user",
            chat_id=self.session_id,
            content=raw_prompt,
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
            "Thinking.",
            is_user=False,
            allow_regenerate=True,
            allow_stop=True,
        )
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
            inbound=inbound,
        )
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
        line = str(update or "").strip()
        if not line:
            return
        self._latest_progress_text = line
        if not self._progress_lines or self._progress_lines[-1] != line:
            self._progress_lines.append(line)
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
