from __future__ import annotations

import asyncio
import contextlib
import csv
import ipaddress
import fnmatch
import logging
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
from annolid.gui.qt_compat import palette_color_role
from qtpy.QtCore import QThreadPool

from annolid.infrastructure.agent_config import (
    get_agent_config_path as get_config_path,
    load_agent_config as load_config,
)
from annolid.infrastructure.agent_workspace import get_agent_workspace_path
from annolid.services.chat_backend_support import (
    ERROR_TYPE_CANCELLED,
    TURN_STATUS_CANCELLED,
)
from annolid.services.chat_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    ZulipChannel,
    decode_outbound_chat_event,
)
from annolid.services.chat_session import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.services import (
    build_root_slash_completion_entries,
    describe_agent_capabilities,
    matches_slash_completion_search,
)
from annolid.behavior import prompting as behavior_prompting
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
from annolid.gui.widgets.ai_chat_session_dialog import ChatSessionManagerDialog
from annolid.gui.widgets.ai_chat_zulip import (
    build_zulip_draft_target,
    missing_zulip_config_fields,
)
from annolid.gui.widgets.citation_manager_widget import CitationManagerDialog
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
from annolid.services.citation_verify import (
    build_citation_verification_report,
    write_citation_verification_report,
)

logger = logging.getLogger(__name__)


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


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _split_slash_value_list(raw: str) -> List[str]:
    names = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", str(raw or ""))
    return _dedupe_preserve_order(names)


def _extract_slash_selection_state(prompt: str) -> Dict[str, Any]:
    raw = str(prompt or "")
    selected_skill_names: List[str] = []
    selected_tool_names: List[str] = []
    selected_capabilities: List[Dict[str, str]] = []
    slash_commands: List[str] = []
    open_capabilities = False
    retained_lines: List[str] = []

    for line in raw.splitlines():
        stripped = str(line or "").strip()
        if not stripped.startswith("/"):
            retained_lines.append(line)
            continue
        match = re.match(r"^/([A-Za-z0-9_-]+)(?:\s+([\s\S]+))?$", stripped)
        if not match:
            retained_lines.append(line)
            continue
        command = str(match.group(1) or "").strip().lower()
        args = str(match.group(2) or "").strip()
        if command in {"skill", "skills"}:
            names = _split_slash_value_list(args)
            selected_skill_names.extend(names)
            for name in names:
                selected_capabilities.append({"kind": "skill", "name": name})
            slash_commands.append(f"/{command}{(' ' + args) if args else ''}".strip())
            if not args:
                open_capabilities = True
            continue
        if command in {"tool", "tools"}:
            names = _split_slash_value_list(args)
            selected_tool_names.extend(names)
            for name in names:
                selected_capabilities.append({"kind": "tool", "name": name})
            slash_commands.append(f"/{command}{(' ' + args) if args else ''}".strip())
            if not args:
                open_capabilities = True
            continue
        if command in {"capabilities", "caps"}:
            slash_commands.append(f"/{command}")
            open_capabilities = True
            continue
        retained_lines.append(line)

    clean_prompt = "\n".join(retained_lines).strip()
    return {
        "clean_prompt": clean_prompt,
        "selected_skill_names": _dedupe_preserve_order(selected_skill_names),
        "selected_tool_names": _dedupe_preserve_order(selected_tool_names),
        "selected_capabilities": [
            {
                "kind": str(item.get("kind") or "").strip().lower(),
                "name": str(item.get("name") or "").strip(),
            }
            for item in selected_capabilities
            if str(item.get("kind") or "").strip().lower() in {"skill", "tool"}
            and str(item.get("name") or "").strip()
        ],
        "slash_commands": slash_commands,
        "open_capabilities": open_capabilities,
    }


def _compose_slash_selection_draft(
    clean_prompt: str,
    *,
    selected_skill_names: List[str],
    selected_tool_names: List[str],
    selected_capabilities: Optional[List[Dict[str, str]]] = None,
) -> str:
    control_lines: List[str] = []
    ordered_entries = list(selected_capabilities or [])
    if ordered_entries:
        for item in ordered_entries:
            kind = str(item.get("kind") or "").strip().lower()
            name = str(item.get("name") or "").strip()
            if kind in {"skill", "tool"} and name:
                control_lines.append(f"/{kind} {name}")
    else:
        for name in _dedupe_preserve_order(selected_skill_names):
            control_lines.append(f"/skill {name}")
        for name in _dedupe_preserve_order(selected_tool_names):
            control_lines.append(f"/tool {name}")
    body = str(clean_prompt or "").strip()
    parts = list(control_lines)
    if body:
        parts.append(body)
    return "\n".join(parts).strip()


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


class _ZulipSendTask(QtCore.QRunnable):
    def __init__(
        self,
        widget: "AIChatWidget",
        *,
        config: Dict[str, Any],
        chat_id: str,
        content: str,
        target_summary: str,
    ) -> None:
        super().__init__()
        self._widget = widget
        self._config = dict(config or {})
        self._chat_id = str(chat_id or "").strip()
        self._content = str(content or "")
        self._target_summary = str(target_summary or "").strip()

    def run(self) -> None:  # pragma: no cover
        try:
            asyncio.run(self._send())
        except Exception as exc:
            QtCore.QMetaObject.invokeMethod(
                self._widget,
                "_finish_zulip_send",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, False),
                QtCore.Q_ARG(str, str(exc)),
                QtCore.Q_ARG(str, self._target_summary),
            )
        else:
            QtCore.QMetaObject.invokeMethod(
                self._widget,
                "_finish_zulip_send",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(bool, True),
                QtCore.Q_ARG(str, ""),
                QtCore.Q_ARG(str, self._target_summary),
            )

    async def _send(self) -> None:
        channel = ZulipChannel(self._config, MessageBus())
        await channel.send(
            OutboundMessage(
                channel="zulip",
                chat_id=self._chat_id,
                content=self._content,
                metadata={"source": "annolid_bot_ui"},
            )
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
        on_stop=None,
        on_delete=None,
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
        self._on_delete = on_delete
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
        self.delete_button = self._create_action_button("🗑", "Delete message")
        self.regenerate_button.setVisible(
            (not self._is_user) and self._allow_regenerate
        )
        self.stop_button.setVisible((not self._is_user) and self._allow_stop)

        # Connect callbacks
        self.speak_button.clicked.connect(self._speak)
        self.copy_button.clicked.connect(self._copy_text)
        self.regenerate_button.clicked.connect(self._regenerate)
        self.stop_button.clicked.connect(self._stop)
        self.delete_button.clicked.connect(self._delete)

        self.actions_layout.addWidget(self.speak_button)
        self.actions_layout.addWidget(self.copy_button)
        self.actions_layout.addWidget(self.regenerate_button)
        self.actions_layout.addWidget(self.stop_button)
        self.actions_layout.addWidget(self.delete_button)
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

    def raw_text(self) -> str:
        return str(self._raw_text or "").strip()

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

    def _delete(self) -> None:
        if callable(self._on_delete):
            self._on_delete(self)

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
        if callable(self._on_delete):
            menu.addSeparator()
            delete_action = menu.addAction("Delete Message")
            delete_action.triggered.connect(self._delete)

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
        title_col.addWidget(self.session_chip_label)

        self.capability_summary_label = QtWidgets.QLabel(self)
        self.capability_summary_label.setObjectName("chatCapabilitySummaryLabel")
        self.capability_summary_label.setWordWrap(False)
        title_col.addWidget(self.capability_summary_label)

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
        self.zulip_button = self._create_input_icon("✉️", "Draft and send to Zulip")

        tools_layout.addWidget(self.attach_file_button)
        tools_layout.addWidget(self.share_canvas_button)
        tools_layout.addWidget(self.share_window_button)
        tools_layout.addWidget(self.citation_button)
        tools_layout.addWidget(self.zulip_button)
        input_row.addLayout(tools_layout)

        self.zulip_panel = self._build_zulip_panel()
        layout.addWidget(self.zulip_panel)

        # Text Input
        self.prompt_text_edit = QtWidgets.QPlainTextEdit(self)
        self.prompt_text_edit.setPlaceholderText("Message Annolid Bot...")
        self.prompt_text_edit.setFixedHeight(50)
        self.prompt_text_edit.setToolTip(
            "Type a message. Use Ctrl+Enter to send. Type / to pick skills, tools, and commands."
        )
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

    def _current_slash_capability_state(self) -> Dict[str, Any]:
        text = self.prompt_text_edit.toPlainText()
        return _extract_slash_selection_state(text)

    def _selected_capability_entries(self) -> List[Dict[str, str]]:
        state = self._current_slash_capability_state()
        entries = list(state.get("selected_capabilities") or [])
        if entries:
            return [
                {
                    "kind": str(item.get("kind") or "").strip().lower(),
                    "name": str(item.get("name") or "").strip(),
                }
                for item in entries
                if str(item.get("kind") or "").strip().lower() in {"skill", "tool"}
                and str(item.get("name") or "").strip()
            ]
        skills = list(state.get("selected_skill_names") or [])
        tools = list(state.get("selected_tool_names") or [])
        return [
            *[{"kind": "skill", "name": name} for name in skills],
            *[{"kind": "tool", "name": name} for name in tools],
        ]

    def _refresh_selected_capability_chips(self) -> None:
        layout = getattr(self, "capability_chip_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addStretch(1)
        entries = self._selected_capability_entries()
        chips: List[QtWidgets.QToolButton] = []

        def add_chip(kind: str, name: str) -> None:
            chip = QtWidgets.QToolButton(self)
            chip.setObjectName("capabilityChipButton")
            chip.setText(f"{name}  ×")
            chip.setToolTip(f"Click to remove {kind} '{name}'")
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setAutoRaise(True)
            chip.setFocusPolicy(QtCore.Qt.StrongFocus)
            chip.setStyleSheet(self._selected_capability_chip_style(kind))
            chip.clicked.connect(
                lambda _checked=False,
                _kind=kind,
                _name=name: self._remove_capability_chip(_kind, _name)
            )
            chip.setProperty("role", "selected")
            chip.setProperty("capability_kind", kind)
            chip.setProperty("capability_name", name)
            chip.setProperty("kind", kind)
            chip.installEventFilter(self)
            chips.append(chip)
            layout.insertWidget(layout.count() - 1, chip)

        for entry in entries:
            add_chip(str(entry.get("kind") or "skill"), str(entry.get("name") or ""))
        self._selected_capability_chip_widgets = chips
        has_chips = bool(chips)
        self.capability_chip_scroll.setVisible(has_chips)
        self.capability_chip_label.setVisible(has_chips)
        self.clear_capability_chips_button.setVisible(has_chips)
        self._refresh_suggested_skill_chips()
        self._refresh_inline_slash_hints()

    def _refresh_suggested_skill_chips(self) -> None:
        layout = getattr(self, "suggested_chip_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addStretch(1)
        state = self._current_slash_capability_state()
        prompt_hint = str(state.get("clean_prompt") or "").strip()
        if not prompt_hint:
            self._suggested_capability_chip_widgets = []
            self.suggested_chip_scroll.setVisible(False)
            self.suggested_chip_label.setVisible(False)
            self._refresh_header_capability_summary()
            return
        payload = self._load_slash_capabilities_payload(task_hint=prompt_hint)
        skill_pool = dict(payload.get("skill_pool") or {})
        suggestions = list(skill_pool.get("suggested_skills") or [])
        if not suggestions:
            self._suggested_capability_chip_widgets = []
            self.suggested_chip_scroll.setVisible(False)
            self.suggested_chip_label.setVisible(False)
            self._refresh_header_capability_summary()
            return
        selected_names = {
            str(name or "").strip().lower()
            for name in (
                list(state.get("selected_skill_names") or [])
                + list(state.get("selected_tool_names") or [])
            )
            if str(name or "").strip()
        }
        chips: List[QtWidgets.QToolButton] = []
        for row in suggestions[:8]:
            name = str(row.get("name") or "").strip()
            if not name or name.lower() in selected_names:
                continue
            chip = QtWidgets.QToolButton(self)
            chip.setObjectName("suggestedSkillChipButton")
            score = float(row.get("score") or 0.0)
            chip.setText(f"{name}  +")
            chip.setToolTip(
                f"Click to add skill '{name}' ({row.get('strategy') or 'suggested'}, score={score:.2f})"
            )
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setAutoRaise(True)
            chip.setFocusPolicy(QtCore.Qt.StrongFocus)
            chip.setStyleSheet(self._suggested_skill_chip_style())
            chip.clicked.connect(
                lambda _checked=False, _name=name: self._add_skill_capability_chip(
                    _name
                )
            )
            chip.setProperty("role", "suggested")
            chip.setProperty("capability_kind", "skill")
            chip.setProperty("capability_name", name)
            chip.installEventFilter(self)
            chips.append(chip)
            layout.insertWidget(layout.count() - 1, chip)
        self._suggested_capability_chip_widgets = chips
        has_chips = bool(chips)
        self.suggested_chip_scroll.setVisible(has_chips)
        self.suggested_chip_label.setVisible(has_chips)
        self._refresh_header_capability_summary()

    def _refresh_inline_slash_hints(self) -> None:
        layout = getattr(self, "slash_hint_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addStretch(1)
        context = self._slash_completion_context()
        if not context:
            self._slash_hint_widgets = []
            self.slash_hint_scroll.setVisible(False)
            self.slash_hint_label.setVisible(False)
            return
        entries = self._build_slash_completion_entries(context)
        if not entries:
            self._slash_hint_widgets = []
            self.slash_hint_scroll.setVisible(False)
            self.slash_hint_label.setVisible(False)
            return
        chips: List[QtWidgets.QToolButton] = []
        for row in entries[:5]:
            display = str(row.get("display") or "").strip()
            if not display:
                continue
            chip = QtWidgets.QToolButton(self)
            chip.setObjectName("slashHintChipButton")
            kind = str(row.get("kind") or "").strip().lower()
            action = str(row.get("action") or "").strip().lower()
            chip.setText(display)
            description = str(row.get("description") or "").strip()
            tip_parts = [description] if description else []
            if kind:
                tip_parts.insert(0, kind)
            chip.setToolTip(" - ".join(tip_parts) if tip_parts else display)
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setAutoRaise(True)
            chip.setFocusPolicy(QtCore.Qt.StrongFocus)
            chip.setStyleSheet(self._slash_hint_chip_style(kind))
            chip.clicked.connect(
                lambda _checked=False,
                _row=dict(row): self._apply_slash_completion_entry(_row)
            )
            chip.setProperty("role", "slash_hint")
            chip.setProperty("completion_kind", kind)
            chip.setProperty("completion_action", action)
            chip.installEventFilter(self)
            chips.append(chip)
            layout.insertWidget(layout.count() - 1, chip)
        self._slash_hint_widgets = chips
        has_chips = bool(chips)
        self.slash_hint_scroll.setVisible(has_chips)
        self.slash_hint_label.setVisible(has_chips)

    def _selected_capability_chip_style(self, kind: str) -> str:
        kind_token = str(kind or "").strip().lower()
        if kind_token == "skill":
            return """
                QToolButton#capabilityChipButton {
                    border: 1px solid rgba(43, 166, 124, 0.55);
                    border-radius: 12px;
                    padding: 5px 10px;
                    margin: 0px;
                    background: rgba(43, 166, 124, 0.16);
                    color: palette(text);
                    font-weight: 600;
                }
                QToolButton#capabilityChipButton:hover {
                    background: rgba(43, 166, 124, 0.24);
                }
                QToolButton#capabilityChipButton:pressed {
                    background: rgba(43, 166, 124, 0.34);
                }
            """
        if kind_token == "tool":
            return """
                QToolButton#capabilityChipButton {
                    border: 1px solid rgba(78, 141, 245, 0.55);
                    border-radius: 12px;
                    padding: 5px 10px;
                    margin: 0px;
                    background: rgba(78, 141, 245, 0.16);
                    color: palette(text);
                    font-weight: 600;
                }
                QToolButton#capabilityChipButton:hover {
                    background: rgba(78, 141, 245, 0.24);
                }
                QToolButton#capabilityChipButton:pressed {
                    background: rgba(78, 141, 245, 0.34);
                }
            """
        return """
            QToolButton#capabilityChipButton {
                border: 1px solid rgba(120, 130, 145, 0.45);
                border-radius: 12px;
                padding: 5px 10px;
                margin: 0px;
                background: rgba(120, 130, 145, 0.14);
                color: palette(text);
                font-weight: 600;
            }
            QToolButton#capabilityChipButton:hover {
                background: rgba(120, 130, 145, 0.22);
            }
            QToolButton#capabilityChipButton:pressed {
                background: rgba(120, 130, 145, 0.30);
            }
        """

    def _suggested_skill_chip_style(self) -> str:
        return """
            QToolButton#suggestedSkillChipButton {
                border: 1px dashed rgba(175, 125, 55, 0.60);
                border-radius: 12px;
                padding: 5px 10px;
                margin: 0px;
                background: rgba(175, 125, 55, 0.10);
                color: palette(text);
            }
            QToolButton#suggestedSkillChipButton:hover {
                background: rgba(175, 125, 55, 0.18);
            }
            QToolButton#suggestedSkillChipButton:pressed {
                background: rgba(175, 125, 55, 0.28);
            }
        """

    def _slash_hint_chip_style(self, kind: str) -> str:
        kind_token = str(kind or "").strip().lower()
        if kind_token in {"skill", "skills"}:
            border = "rgba(43, 166, 124, 0.42)"
            fill = "rgba(43, 166, 124, 0.12)"
        elif kind_token in {"tool", "tools"}:
            border = "rgba(78, 141, 245, 0.42)"
            fill = "rgba(78, 141, 245, 0.12)"
        else:
            border = "rgba(175, 125, 55, 0.42)"
            fill = "rgba(175, 125, 55, 0.10)"
        return f"""
            QToolButton#slashHintChipButton {{
                border: 1px solid {border};
                border-radius: 12px;
                padding: 4px 10px;
                margin: 0px;
                background: {fill};
                color: palette(text);
                font-weight: 600;
            }}
            QToolButton#slashHintChipButton:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            QToolButton#slashHintChipButton:pressed {{
                background: rgba(255, 255, 255, 0.14);
            }}
        """

    def _refresh_header_capability_summary(self) -> None:
        label = getattr(self, "capability_summary_label", None)
        if label is None:
            return
        state = self._current_slash_capability_state()
        skills = int(len(state.get("selected_skill_names") or []))
        tools = int(len(state.get("selected_tool_names") or []))
        total = skills + tools
        if total <= 0:
            label.setText("Selected: none")
        else:
            label.setText(
                f"Selected: {skills} skill{'s' if skills != 1 else ''} · {tools} tool{'s' if tools != 1 else ''}"
            )

    def _chip_widgets_in_order(self) -> List[QtWidgets.QAbstractButton]:
        return [
            *list(getattr(self, "_selected_capability_chip_widgets", []) or []),
            *list(getattr(self, "_suggested_capability_chip_widgets", []) or []),
            *list(getattr(self, "_slash_hint_widgets", []) or []),
        ]

    def _focus_capability_chip(self, *, kind: str = "selected") -> bool:
        if str(kind or "").strip().lower() == "suggested":
            widgets = list(
                getattr(self, "_suggested_capability_chip_widgets", []) or []
            )
        else:
            widgets = list(getattr(self, "_selected_capability_chip_widgets", []) or [])
        if not widgets:
            return False
        try:
            widgets[0].setFocus()
            return True
        except Exception:
            return False

    def _focus_next_capability_chip(
        self, current: Optional[QtWidgets.QWidget], delta: int
    ) -> bool:
        widgets = self._chip_widgets_in_order()
        if not widgets:
            return False
        current_index = -1
        if current is not None:
            try:
                current_index = widgets.index(current)  # type: ignore[arg-type]
            except ValueError:
                current_index = -1
        next_index = (
            0
            if current_index < 0
            else max(0, min(len(widgets) - 1, current_index + int(delta)))
        )
        try:
            widgets[next_index].setFocus()
            return True
        except Exception:
            return False

    def _remove_capability_chip(self, kind: str, name: str) -> None:
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        skill_key = str(name or "").strip().lower()
        entries = [
            item
            for item in self._selected_capability_entries()
            if str(item.get("name") or "").strip().lower() != skill_key
        ]
        skill_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "skill"
        ]
        tool_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "tool"
        ]
        draft = _compose_slash_selection_draft(
            clean_prompt,
            selected_skill_names=skill_names,
            selected_tool_names=tool_names,
            selected_capabilities=entries,
        )
        self.prompt_text_edit.setPlainText(draft)
        self.prompt_text_edit.setFocus()
        self.status_label.setText(f"Removed {kind}: {name}")

    def _add_skill_capability_chip(self, name: str) -> None:
        skill_name = str(name or "").strip()
        if not skill_name:
            return
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        entries = [
            item
            for item in self._selected_capability_entries()
            if str(item.get("name") or "").strip().lower() != skill_name.lower()
        ]
        entries.append({"kind": "skill", "name": skill_name})
        skill_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "skill"
        ]
        tool_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "tool"
        ]
        draft = _compose_slash_selection_draft(
            clean_prompt,
            selected_skill_names=skill_names,
            selected_tool_names=tool_names,
            selected_capabilities=entries,
        )
        self.prompt_text_edit.setPlainText(draft)
        self.prompt_text_edit.setFocus()
        self.status_label.setText(f"Added suggested skill: {skill_name}")

    def _clear_selected_capabilities(self) -> None:
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        self.prompt_text_edit.setPlainText(clean_prompt)
        self.prompt_text_edit.setFocus()
        self.status_label.setText("Cleared selected skills and tools.")

    def _reorder_selected_capability(
        self, source: QtWidgets.QWidget, target: QtWidgets.QWidget
    ) -> bool:
        if source is target:
            return False
        source_kind = str(source.property("capability_kind") or "").strip().lower()
        source_name = str(source.property("capability_name") or "").strip()
        target_name = str(target.property("capability_name") or "").strip()
        if not source_kind or not source_name or not target_name:
            return False
        entries = self._selected_capability_entries()
        source_index = next(
            (
                idx
                for idx, item in enumerate(entries)
                if str(item.get("kind") or "").strip().lower() == source_kind
                and str(item.get("name") or "").strip().lower() == source_name.lower()
            ),
            -1,
        )
        target_index = next(
            (
                idx
                for idx, item in enumerate(entries)
                if str(item.get("name") or "").strip().lower() == target_name.lower()
            ),
            -1,
        )
        if source_index < 0 or target_index < 0 or source_index == target_index:
            return False
        moved = entries.pop(source_index)
        if source_index < target_index:
            target_index -= 1
        entries.insert(target_index, moved)
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        skill_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "skill"
        ]
        tool_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "tool"
        ]
        draft = _compose_slash_selection_draft(
            clean_prompt,
            selected_skill_names=skill_names,
            selected_tool_names=tool_names,
            selected_capabilities=entries,
        )
        self.prompt_text_edit.setPlainText(draft)
        self.prompt_text_edit.setFocus()
        self.status_label.setText(
            f"Moved {source_kind}: {source_name} before {target_name}"
        )
        return True

    def _load_slash_capabilities_payload(
        self, *, task_hint: str = "", top_k: int = 5
    ) -> Dict[str, Any]:
        workspace = str(get_agent_workspace_path())
        normalized_hint = str(task_hint or "").strip()
        cache_key = (
            workspace,
            self.selected_provider,
            self.selected_model,
            normalized_hint,
            int(top_k),
        )
        if getattr(self, "_slash_capabilities_cache_key", None) == cache_key:
            cached = getattr(self, "_slash_capabilities_cache", None)
            if isinstance(cached, dict):
                return cached
        try:
            payload = describe_agent_capabilities(
                workspace=workspace,
                provider=self.selected_provider,
                model=self.selected_model,
                task_hint=normalized_hint,
                top_k=int(top_k),
            )
        except Exception as exc:
            logger.warning("Slash capability load failed: %s", exc)
            payload = {
                "workspace": workspace,
                "provider": self.selected_provider,
                "model": self.selected_model,
                "tool_pool": {
                    "workspace": workspace,
                    "provider": self.selected_provider,
                    "model": self.selected_model,
                    "counts": {"registered": 0, "allowed": 0, "denied": 0},
                    "allowed_tools": [],
                    "denied_tools": [],
                },
                "skill_pool": {
                    "workspace": workspace,
                    "skill_pool": {"counts": {}, "preview": []},
                    "suggested_skills": [],
                },
                "summary": {},
            }
        self._slash_capabilities_cache_key = cache_key
        self._slash_capabilities_cache = payload
        return payload

    def _slash_completion_context(self) -> Dict[str, Any]:
        cursor = self.prompt_text_edit.textCursor()
        block = cursor.block()
        line = str(block.text() or "")
        cursor_pos = max(0, int(cursor.position() - block.position()))
        prefix = line[:cursor_pos]
        match = re.search(
            r"(^|\s)([\/@](?P<command>[A-Za-z0-9_-]*)(?:\s+(?P<args>[^\n]*))?)$",
            prefix,
        )
        if not match:
            return {}
        token = str(match.group(2) or "")
        command = str(match.group("command") or "").strip().lower()
        args = str(match.group("args") or "").strip()
        token_start = max(0, len(prefix) - len(token))
        mode = "root"
        if command in {"skill", "skills"}:
            mode = "skill"
        elif command in {"tool", "tools"}:
            mode = "tool"
        return {
            "mode": mode,
            "command": command,
            "args": args,
            "token": token,
            "token_start": block.position() + token_start,
            "token_end": cursor.position(),
            "search_prefix": token.lower().lstrip("/@"),
        }

    def _build_slash_completion_entries(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        payload = self._load_slash_capabilities_payload()
        entries: List[Dict[str, Any]] = []

        def add_entry(
            *,
            display: str,
            search: str,
            insert: str,
            kind: str,
            action: str = "",
            description: str = "",
        ) -> None:
            entries.append(
                {
                    "display": display,
                    "search": search,
                    "insert": insert,
                    "kind": kind,
                    "action": action,
                    "description": description,
                }
            )

        mode = str(context.get("mode") or "root").strip().lower()
        search_prefix = str(context.get("search_prefix") or "").strip().lower()

        if mode == "root":
            entries.extend(build_root_slash_completion_entries())
        elif mode == "skill":
            skill_pool = dict(payload.get("skill_pool") or {})
            nested_pool = dict(skill_pool.get("skill_pool") or {})
            preview_rows = list(nested_pool.get("preview") or [])
            suggested = {
                str(row.get("name") or "").strip().lower()
                for row in list(skill_pool.get("suggested_skills") or [])
            }
            ordered_rows = []
            if suggested:
                ordered_rows.extend(
                    row
                    for row in preview_rows
                    if str(row.get("name") or "").strip().lower() in suggested
                )
            ordered_rows.extend(row for row in preview_rows if row not in ordered_rows)
            for row in ordered_rows[:25]:
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                description = str(row.get("description") or "").strip()
                insert = f"/skill {name} "
                search = insert.rstrip().lower()
                add_entry(
                    display=f"Skill: {name}",
                    search=search,
                    insert=insert,
                    kind="skill",
                    description=description,
                )
        elif mode == "tool":
            tool_pool = dict(payload.get("tool_pool") or {})
            tool_names = [
                str(name or "").strip()
                for name in list(tool_pool.get("allowed_tools") or [])
                if str(name or "").strip()
            ]
            for name in tool_names[:40]:
                insert = f"/tool {name} "
                add_entry(
                    display=f"Tool: {name}",
                    search=insert.rstrip().lower(),
                    insert=insert,
                    kind="tool",
                    description="User-selected tool hint",
                )

        query = search_prefix
        if query:
            entries = [
                row
                for row in entries
                if matches_slash_completion_search(str(row.get("search") or ""), query)
            ]
        return entries

    def _ensure_slash_completion_ui(self) -> None:
        if getattr(self, "_slash_completion_model", None) is not None:
            return
        self._slash_completion_model = QtGui.QStandardItemModel(self)
        self._slash_completer = QtWidgets.QCompleter(self._slash_completion_model, self)
        self._slash_completer.setWidget(self.prompt_text_edit)
        self._slash_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._slash_completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        self._slash_completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self._slash_completer.setCompletionRole(QtCore.Qt.UserRole)
        popup = self._slash_completer.popup()
        popup.setAlternatingRowColors(True)
        popup.setUniformItemSizes(True)
        popup.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        popup.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        popup.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._slash_completer.activated.connect(self._apply_slash_completion)

    def _update_slash_completion_ui(self) -> None:
        self._ensure_slash_completion_ui()
        context = self._slash_completion_context()
        if not context:
            self._hide_slash_completion_ui()
            self._refresh_inline_slash_hints()
            return
        entries = self._build_slash_completion_entries(context)
        self._slash_completion_context_state = context
        if not entries:
            self._hide_slash_completion_ui()
            self._refresh_inline_slash_hints()
            return
        model = self._slash_completion_model
        assert model is not None
        model.clear()
        for row in entries:
            item = QtGui.QStandardItem(str(row.get("display") or ""))
            item.setData(str(row.get("search") or ""), QtCore.Qt.UserRole)
            item.setData(str(row.get("insert") or ""), QtCore.Qt.UserRole + 1)
            item.setData(str(row.get("kind") or ""), QtCore.Qt.UserRole + 2)
            item.setData(str(row.get("action") or ""), QtCore.Qt.UserRole + 3)
            item.setData(str(row.get("description") or ""), QtCore.Qt.UserRole + 4)
            model.appendRow(item)
        completer = self._slash_completer
        assert completer is not None
        completer.setCompletionPrefix(str(context.get("search_prefix") or ""))
        rect = self.prompt_text_edit.cursorRect()
        rect.setWidth(
            max(
                rect.width(),
                self.prompt_text_edit.viewport().width() // 2,
            )
        )
        completer.complete(rect)
        popup = completer.popup()
        if popup.model() is not None and popup.model().rowCount() > 0:
            popup.setCurrentIndex(popup.model().index(0, 0))
        self._refresh_inline_slash_hints()

    def _hide_slash_completion_ui(self) -> None:
        completer = getattr(self, "_slash_completer", None)
        if completer is None:
            return
        popup = completer.popup()
        if popup is not None and popup.isVisible():
            popup.hide()

    def _apply_slash_completion(self, completion: Any) -> None:
        if not completion:
            return
        model_index = None
        if isinstance(completion, QtCore.QModelIndex):
            model_index = completion
        else:
            completer = getattr(self, "_slash_completer", None)
            if completer is not None:
                popup = completer.popup()
                if popup is not None:
                    model_index = popup.currentIndex()
        if model_index is None or not model_index.isValid():
            return
        row = {
            "kind": str(model_index.data(QtCore.Qt.UserRole + 2) or "").strip().lower(),
            "action": str(model_index.data(QtCore.Qt.UserRole + 3) or "")
            .strip()
            .lower(),
            "insert": str(model_index.data(QtCore.Qt.UserRole + 1) or "").strip(),
            "display": str(model_index.data(QtCore.Qt.DisplayRole) or "").strip(),
            "description": str(model_index.data(QtCore.Qt.UserRole + 4) or "").strip(),
        }
        self._apply_slash_completion_entry(row)

    def _apply_slash_completion_entry(self, row: Dict[str, Any]) -> None:
        kind = str(row.get("kind") or "").strip().lower()
        action = str(row.get("action") or "").strip().lower()
        insert_text = str(row.get("insert") or "").strip()
        display = str(row.get("display") or "").strip()
        if action == "open_capabilities":
            self._hide_slash_completion_ui()
            self._open_agent_capabilities_dialog()
            return
        if action == "open_track_dialog":
            self._hide_slash_completion_ui()
            self.bot_open_track_slash_dialog()
            return
        context = dict(getattr(self, "_slash_completion_context_state", {}) or {})
        token_start = int(context.get("token_start") or 0)
        token_end = int(context.get("token_end") or 0)
        cursor = self.prompt_text_edit.textCursor()
        cursor.beginEditBlock()
        cursor.setPosition(token_start)
        cursor.setPosition(token_end, QtGui.QTextCursor.KeepAnchor)
        cursor.insertText(insert_text)
        cursor.endEditBlock()
        self.prompt_text_edit.setTextCursor(cursor)
        self._hide_slash_completion_ui()
        if kind in {"skill", "tool"}:
            self.status_label.setText(f"Selected {kind}: {display}")

    def _move_slash_completion_selection(self, delta: int) -> None:
        completer = getattr(self, "_slash_completer", None)
        if completer is None:
            return
        popup = completer.popup()
        if popup is None or not popup.isVisible():
            return
        model = popup.model()
        if model is None:
            return
        row_count = int(model.rowCount())
        if row_count <= 0:
            return
        current = popup.currentIndex()
        row = current.row() if current.isValid() else 0
        row = max(0, min(row_count - 1, row + int(delta)))
        popup.setCurrentIndex(model.index(row, 0))

    @QtCore.Slot()
    def open_agent_capabilities_dialog(self) -> None:
        self._open_agent_capabilities_dialog()

    def _open_agent_capabilities_dialog(self) -> None:
        dialog = getattr(self, "_agent_capabilities_dialog", None)
        if dialog is None:
            from annolid.gui.widgets.agent_capabilities_dialog import (
                AgentCapabilitiesDialog,
            )

            dialog = AgentCapabilitiesDialog(self)
            self._agent_capabilities_dialog = dialog
        try:
            dialog.refresh()
        except Exception:
            pass
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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

    @staticmethod
    def _is_placeholder_behavior_label(label: str) -> bool:
        value = str(label or "").strip().lower()
        if not value:
            return True
        if re.fullmatch(r"behavior[_\-\s]?\d+", value):
            return True
        return value in {"behavior", "behaviour", "label", "placeholder"}

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
        flag_widget = getattr(host, "flag_widget", None)
        if flag_widget is not None:
            try:
                existing_flag_names = getattr(
                    flag_widget, "_get_existing_flag_names", None
                )
                if callable(existing_flag_names):
                    labels.extend(
                        [
                            str(name).strip()
                            for name in dict(existing_flag_names() or {}).keys()
                            if str(name).strip()
                        ]
                    )
            except Exception:
                pass
        flags_controller = getattr(host, "flags_controller", None)
        if flags_controller is not None:
            try:
                pinned = getattr(flags_controller, "pinned_flags", None)
                if isinstance(pinned, dict):
                    labels.extend(
                        [
                            str(name).strip()
                            for name in pinned.keys()
                            if str(name).strip()
                        ]
                    )
            except Exception:
                pass
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is not None:
            try:
                timeline_behaviors = list(
                    getattr(behavior_controller, "behavior_names", lambda: set())()
                    if callable(getattr(behavior_controller, "behavior_names", None))
                    else getattr(behavior_controller, "behavior_names", set())
                )
            except Exception:
                timeline_behaviors = []
            labels.extend(
                [str(name).strip() for name in timeline_behaviors if str(name).strip()]
            )
        normalized = self._normalize_behavior_labels(labels)
        filtered = [
            name for name in normalized if not self._is_placeholder_behavior_label(name)
        ]
        return filtered

    def _resolve_segment_label_candidates(
        self,
        explicit_labels: List[str],
        *,
        use_defined_behavior_list: bool,
    ) -> List[str]:
        defined_labels = self._labels_from_schema_or_flags()
        normalized_explicit = self._normalize_behavior_labels(explicit_labels)
        if not use_defined_behavior_list:
            return normalized_explicit or defined_labels
        if not defined_labels:
            return normalized_explicit
        if not normalized_explicit:
            return defined_labels

        defined_lookup = {label.lower(): label for label in defined_labels}
        intersection: List[str] = []
        for label in normalized_explicit:
            mapped = defined_lookup.get(label.lower())
            if mapped:
                intersection.append(mapped)
        return self._normalize_behavior_labels(intersection or defined_labels)

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
            return ("", 0.0)

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
                label_val = str(
                    payload.get("label")
                    or payload.get("behavior")
                    or payload.get("classification")
                    or payload.get("prediction")
                    or ""
                ).strip()
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

    @staticmethod
    def _uniform_segment_frame_indices(
        start_frame: int, end_frame: int, sample_count: int
    ) -> List[int]:
        start = int(start_frame)
        end = int(end_frame)
        if end < start:
            start, end = end, start
        if start == end:
            return [start]
        count = max(1, int(sample_count))
        span = end - start + 1
        count = min(count, span)
        if count == span:
            return list(range(start, end + 1))
        step = (end - start) / float(max(1, count - 1))
        frames = [int(round(start + (idx * step))) for idx in range(count)]
        return sorted(set(max(start, min(end, frame)) for frame in frames))

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

    def _save_behavior_segment_labeling_log(
        self,
        host: object,
        *,
        mode: str,
        labels_used: List[str],
        segment_frames: int,
        segment_seconds: float,
        sample_frames_per_segment: int,
        evaluated_segments: int,
        skipped_segments: int,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        video_file = str(getattr(host, "video_file", "") or "").strip()
        if not video_file:
            return {"ok": False, "error": "No video file is loaded."}
        try:
            video_path = Path(video_file)
            output_path = video_path.with_name(
                f"{video_path.stem}_behavior_segment_labels.json"
            )
            payload: Dict[str, Any] = {
                "video_path": str(video_path),
                "mode": str(mode or "uniform"),
                "labels_used": list(labels_used),
                "segment_frames": int(segment_frames),
                "segment_seconds": float(segment_seconds),
                "sample_frames_per_segment": int(sample_frames_per_segment),
                "evaluated_segments": int(evaluated_segments),
                "labeled_segments": int(len(predictions)),
                "skipped_segments": int(skipped_segments),
                "predictions": list(predictions),
                "generated_at": datetime.now().isoformat(),
            }
            output_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"ok": True, "path": str(output_path), "rows": len(predictions)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _run_behavior_segment_vlm_worker(
        self,
        *,
        video_path: str,
        intervals: List[Dict[str, Any]],
        labels: List[str],
        sample_frames_per_segment: int,
        llm_profile: str,
        llm_provider: str,
        llm_model: str,
        stop_event=None,
        pred_worker=None,
    ) -> Dict[str, Any]:
        from annolid.core.models.adapters.llm_chat import LLMChatAdapter
        from annolid.core.models.base import ModelRequest

        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video for segment labeling: {video_path}"
            )
        inference_cache: Dict[int, tuple[str, float]] = {}
        predictions: List[Dict[str, Any]] = []
        skipped_segments = 0

        try:
            adapter = LLMChatAdapter(
                profile=str(llm_profile or "").strip() or None,
                provider=str(llm_provider or "").strip() or None,
                model=str(llm_model or "").strip() or None,
                persist=False,
            )
            with (
                adapter,
                tempfile.TemporaryDirectory(
                    prefix="annolid_behavior_segment_"
                ) as tmp_dir,
            ):
                total = max(1, len(intervals))
                for idx, item in enumerate(intervals, start=1):
                    if stop_event is not None and bool(stop_event.is_set()):
                        break
                    start_frame = int(item["start_frame"])
                    end_frame = int(item["end_frame"])
                    probe_frames = self._uniform_segment_frame_indices(
                        start_frame, end_frame, sample_frames_per_segment
                    )
                    frame_votes: Dict[str, int] = {}
                    frame_confidences: Dict[str, float] = {}
                    successful_samples = 0
                    for probe_frame in probe_frames:
                        cached = inference_cache.get(probe_frame)
                        if cached is None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(probe_frame))
                            ok, frame = cap.read()
                            if not ok or frame is None:
                                continue
                            image_path = str(
                                Path(tmp_dir) / f"frame_{int(probe_frame):09d}.png"
                            )
                            if not cv2.imwrite(image_path, frame):
                                continue
                            try:
                                segment_text = (
                                    f"segment frames {int(start_frame)}-{int(end_frame)}, "
                                    f"representative frame {int(probe_frame)}"
                                )
                                prompt = behavior_prompting.build_behavior_classification_prompt(
                                    behavior_labels=labels,
                                    segment_label=segment_text,
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
                                cached = self._extract_label_from_model_text(
                                    raw, labels
                                )
                                inference_cache[probe_frame] = cached
                            except Exception:
                                continue
                            finally:
                                try:
                                    if os.path.exists(image_path):
                                        os.remove(image_path)
                                except OSError:
                                    pass
                        label, confidence = cached
                        if not str(label or "").strip():
                            continue
                        frame_votes[label] = int(frame_votes.get(label, 0)) + 1
                        frame_confidences[label] = float(
                            frame_confidences.get(label, 0.0)
                        ) + float(confidence)
                        successful_samples += 1

                    if successful_samples <= 0 or not frame_votes:
                        skipped_segments += 1
                        progress_value = int((idx * 100) / total)
                        if pred_worker is not None:
                            pred_worker.report_preview(
                                {
                                    "index": int(idx),
                                    "total": int(total),
                                    "start_frame": int(start_frame),
                                    "end_frame": int(end_frame),
                                    "status": "skipped",
                                    "progress": int(progress_value),
                                }
                            )
                            pred_worker.report_progress(progress_value)
                        continue

                    sorted_labels = sorted(
                        frame_votes.keys(),
                        key=lambda key: (
                            -int(frame_votes.get(key, 0)),
                            -float(frame_confidences.get(key, 0.0)),
                            key.lower(),
                        ),
                    )
                    best_label = str(sorted_labels[0])
                    best_conf = float(frame_confidences.get(best_label, 0.0)) / float(
                        max(1, frame_votes.get(best_label, 1))
                    )
                    predictions.append(
                        {
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "subject": item.get("subject"),
                            "label": best_label,
                            "confidence": best_conf,
                        }
                    )
                    progress_value = int((idx * 100) / total)
                    if pred_worker is not None:
                        pred_worker.report_preview(
                            {
                                "index": int(idx),
                                "total": int(total),
                                "status": "labeled",
                                "progress": int(progress_value),
                                "prediction": dict(predictions[-1]),
                            }
                        )
                        pred_worker.report_progress(progress_value)
        finally:
            cap.release()

        return {
            "predictions": predictions,
            "skipped_segments": int(skipped_segments),
            "processed_segments": int(len(intervals)),
            "cancelled": bool(stop_event is not None and stop_event.is_set()),
        }

    def _clear_behavior_label_run_context(self) -> None:
        self._behavior_label_run_context = {}
        self._behavior_label_worker = None
        self._behavior_label_thread = None

    def _behavior_label_timestamp_provider(self, host: object):
        def _timestamp_provider(frame: int) -> Optional[float]:
            local_fps = getattr(host, "fps", None)
            if local_fps is None or float(local_fps) <= 0:
                return None
            return float(frame) / float(local_fps)

        return _timestamp_provider

    def _refresh_behavior_panels(self, host: object) -> None:
        refresh_log = getattr(host, "_refresh_behavior_log", None)
        if callable(refresh_log):
            refresh_log()
        timeline_panel = getattr(host, "timeline_panel", None)
        refresh_timeline = getattr(
            timeline_panel, "refresh_from_behavior_controller", None
        )
        if callable(refresh_timeline):
            refresh_timeline()

    def _commit_behavior_label_prediction(
        self,
        context: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        host = context.get("host")
        behavior_controller = context.get("behavior_controller")
        if host is None or behavior_controller is None:
            return None

        label = str(prediction.get("label") or "").strip()
        if not label:
            return None

        start_frame = int(prediction.get("start_frame") or 0)
        end_frame = int(prediction.get("end_frame") or start_frame)
        subject_value = prediction.get("subject")
        default_subject = context.get("default_subject")
        resolved_subject = subject_value or default_subject
        timestamp_provider = context.get("timestamp_provider")
        behavior_controller.create_interval(
            behavior=label,
            start_frame=start_frame,
            end_frame=end_frame,
            subject=resolved_subject,
            timestamp_provider=timestamp_provider,
        )

        normalized_prediction = {
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "subject": resolved_subject,
            "label": label,
            "confidence": float(prediction.get("confidence") or 0.0),
        }
        context_predictions = list(context.get("predictions") or [])
        context_predictions.append(normalized_prediction)
        context["predictions"] = context_predictions

        self._refresh_behavior_panels(host)
        return normalized_prediction

    def _save_behavior_segment_progress(
        self, *, force_timestamps: bool = False
    ) -> Dict[str, Any]:
        context = dict(self._behavior_label_run_context or {})
        if not context:
            return {"ok": False, "error": "Behavior labeling run context is missing."}

        host = context.get("host")
        if host is None:
            return {"ok": False, "error": "Behavior labeling host is unavailable."}

        predictions = list(context.get("predictions") or [])
        timestamp_result: Dict[str, Any] = {}
        processed = int(context.get("processed_segments") or 0)
        if force_timestamps or (processed > 0 and processed % 10 == 0):
            timestamp_result = self._save_behavior_timestamps_csv(host)

        behavior_log_result = self._save_behavior_segment_labeling_log(
            host,
            mode=str(context.get("mode") or "uniform"),
            labels_used=list(context.get("labels") or []),
            segment_frames=int(context.get("segment_frames") or 1),
            segment_seconds=float(context.get("segment_seconds") or 0.0),
            sample_frames_per_segment=int(
                context.get("sample_frames_per_segment") or 1
            ),
            evaluated_segments=int(context.get("evaluated_segments") or 0),
            skipped_segments=int(context.get("skipped_segments") or 0),
            predictions=predictions,
        )
        return {
            "ok": bool(behavior_log_result.get("ok", False)),
            "timestamp_result": timestamp_result,
            "behavior_log_result": behavior_log_result,
        }

    def _behavior_catalog_host(self) -> object:
        return self.host_window_widget or self.window()

    def _behavior_catalog_action(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        host = self._behavior_catalog_host()
        action_name = str(action or "").strip().lower()
        if not action_name:
            return {"ok": False, "error": "Behavior catalog action is required."}
        save = bool(kwargs.pop("save", True))

        if action_name == "list":
            getter = getattr(host, "list_behavior_catalog", None)
            if callable(getter):
                return dict(getter())
            schema = getattr(host, "project_schema", None)
            if schema is None:
                return {"ok": False, "error": "No project schema is loaded."}
            return {
                "ok": True,
                "count": len(schema.behaviors),
                "behavior_catalog": [
                    {
                        "code": behavior.code,
                        "name": behavior.name,
                        "description": behavior.description or "",
                        "category_id": behavior.category_id or "",
                        "modifier_ids": list(behavior.modifier_ids or []),
                        "key_binding": behavior.key_binding or "",
                        "is_state": bool(behavior.is_state),
                        "exclusive_with": list(behavior.exclusive_with or []),
                    }
                    for behavior in schema.behaviors
                ],
            }

        if action_name == "save":
            saver = getattr(host, "save_behavior_catalog", None)
            if callable(saver):
                return dict(saver())
            return {"ok": False, "error": "Behavior catalog saver is unavailable."}

        if action_name == "create":
            creator = getattr(host, "create_behavior_catalog_item", None)
            if callable(creator):
                return dict(creator(save=save, **kwargs))
            return {"ok": False, "error": "Behavior catalog creator is unavailable."}

        if action_name == "update":
            updater = getattr(host, "update_behavior_catalog_item", None)
            if callable(updater):
                code = str(kwargs.pop("code", "") or "").strip()
                if not code:
                    return {"ok": False, "error": "Behavior code is required."}
                return dict(updater(code=code, updates=kwargs, save=save))
            return {"ok": False, "error": "Behavior catalog updater is unavailable."}

        if action_name == "delete":
            deleter = getattr(host, "delete_behavior_catalog_item", None)
            if callable(deleter):
                code = str(kwargs.get("code", "") or "").strip()
                if not code:
                    return {"ok": False, "error": "Behavior code is required."}
                return dict(deleter(code=code, save=save))
            return {"ok": False, "error": "Behavior catalog deleter is unavailable."}

        return {
            "ok": False,
            "error": f"Unsupported behavior catalog action: {action_name}",
        }

    @QtCore.Slot(int)
    def _on_behavior_label_progress(self, progress_value: int) -> None:
        context = dict(self._behavior_label_run_context or {})
        if not context:
            return
        pct = max(0, min(100, int(progress_value)))
        processed = int(context.get("processed_segments") or 0)
        total = int(context.get("evaluated_segments") or 0)
        self.status_label.setText(
            f"Behavior labeling running in background… {pct}% ({processed}/{max(1, total)} segments)."
        )

    @QtCore.Slot(object)
    def _on_behavior_label_preview(self, payload: object) -> None:
        context = self._behavior_label_run_context
        if not context:
            return
        if not isinstance(payload, dict):
            return

        host = context.get("host")
        behavior_controller = context.get("behavior_controller")
        if host is None or behavior_controller is None:
            return

        status = str(payload.get("status") or "").strip().lower()
        context["processed_segments"] = int(payload.get("index") or 0)
        context["skipped_segments"] = int(context.get("skipped_segments") or 0)
        if status == "skipped":
            context["skipped_segments"] = int(context["skipped_segments"]) + 1
            progress_value = max(0, min(100, int(payload.get("progress") or 0)))
            self.status_label.setText(
                f"Skipped segment {context['processed_segments']}/{context.get('evaluated_segments')}; {progress_value}% complete."
            )
            self._save_behavior_segment_progress(force_timestamps=False)
            return

        prediction = payload.get("prediction")
        if not isinstance(prediction, dict):
            return
        normalized_prediction = self._commit_behavior_label_prediction(
            context, prediction
        )
        if normalized_prediction is None:
            return

        save_result = self._save_behavior_segment_progress(force_timestamps=False)
        behavior_log_result = dict(save_result.get("behavior_log_result") or {})

        progress_value = max(0, min(100, int(payload.get("progress") or 0)))
        total = int(context.get("evaluated_segments") or 0)
        self.status_label.setText(
            f"Labeled segment {context['processed_segments']}/{max(1, total)} as '{normalized_prediction['label']}' ({progress_value}%)."
        )
        self._set_bot_action_result(
            "label_behavior_segments",
            {
                "ok": True,
                "queued": False,
                "in_progress": True,
                "mode": str(context.get("mode") or "uniform"),
                "labeled_segments": len(context.get("predictions") or []),
                "evaluated_segments": int(context.get("evaluated_segments") or 0),
                "skipped_segments": int(context.get("skipped_segments") or 0),
                "segment_frames": int(context.get("segment_frames") or 1),
                "segment_seconds": float(context.get("segment_seconds") or 0.0),
                "sample_frames_per_segment": int(
                    context.get("sample_frames_per_segment") or 1
                ),
                "use_defined_behavior_list": bool(
                    context.get("use_defined_behavior_list", True)
                ),
                "labels_used": list(context.get("labels") or []),
                "behavior_log_json": str(behavior_log_result.get("path") or ""),
                "behavior_log_rows": int(behavior_log_result.get("rows") or 0),
            },
        )

    @QtCore.Slot(object)
    def _on_behavior_label_finished(self, result: object) -> None:
        context = dict(self._behavior_label_run_context or {})
        self._clear_behavior_label_run_context()
        if not context:
            return

        host = context.get("host")
        behavior_controller = context.get("behavior_controller")
        if host is None or behavior_controller is None:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior labeling context was lost."},
            )
            self.status_label.setText(
                "Bot action failed: behavior labeling context lost."
            )
            return

        if isinstance(result, Exception):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": str(result)},
            )
            self.status_label.setText(f"Bot action failed: {result}")
            return

        payload = dict(result or {})
        if bool(payload.get("cancelled")):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior labeling was cancelled."},
            )
            self.status_label.setText("Behavior labeling cancelled.")
            return

        predictions = list(context.get("predictions") or [])
        if not predictions:
            fallback_predictions = list(payload.get("predictions") or [])
            if fallback_predictions:
                for pred in fallback_predictions:
                    normalized_prediction = self._commit_behavior_label_prediction(
                        context, pred
                    )
                    if normalized_prediction is not None:
                        predictions.append(normalized_prediction)
                context["predictions"] = predictions

        if not predictions:
            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": False,
                    "error": "Model did not return usable labels for any segment.",
                },
            )
            self.status_label.setText("Bot action failed: no usable segment labels.")
            return

        context["skipped_segments"] = max(
            int(context.get("skipped_segments") or 0),
            int(payload.get("skipped_segments") or 0),
        )
        self._behavior_label_run_context = context
        save_result = self._save_behavior_segment_progress(force_timestamps=True)
        self._behavior_label_run_context = {}
        timestamp_result = dict(save_result.get("timestamp_result") or {})
        if not bool(timestamp_result.get("ok", False)):
            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": False,
                    "error": str(
                        timestamp_result.get("error")
                        or "Failed to save behavior timestamps."
                    ),
                },
            )
            self.status_label.setText("Bot action failed: timestamp export.")
            return

        behavior_log_result = dict(save_result.get("behavior_log_result") or {})
        self._set_bot_action_result(
            "label_behavior_segments",
            {
                "ok": True,
                "mode": str(context.get("mode") or "uniform"),
                "queued": False,
                "in_progress": False,
                "labeled_segments": len(predictions),
                "evaluated_segments": int(context.get("evaluated_segments") or 0),
                "skipped_segments": int(context.get("skipped_segments") or 0),
                "segment_frames": int(context.get("segment_frames") or 1),
                "segment_seconds": float(context.get("segment_seconds") or 0.0),
                "sample_frames_per_segment": int(
                    context.get("sample_frames_per_segment") or 1
                ),
                "use_defined_behavior_list": bool(
                    context.get("use_defined_behavior_list", True)
                ),
                "labels_used": list(context.get("labels") or []),
                "timestamps_csv": str(timestamp_result.get("path") or ""),
                "timestamps_rows": int(timestamp_result.get("rows") or 0),
                "behavior_log_json": str(behavior_log_result.get("path") or ""),
                "behavior_log_rows": int(behavior_log_result.get("rows") or 0),
            },
        )
        timestamps_name = Path(str(timestamp_result.get("path") or "")).name
        behavior_log_name = Path(str(behavior_log_result.get("path") or "")).name
        if behavior_log_name:
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s); saved {timestamps_name} and {behavior_log_name}."
            )
        else:
            self.status_label.setText(
                f"Labeled {len(predictions)} segment(s); saved timestamps to {timestamps_name}."
            )

    @QtCore.Slot(str)
    def bot_manage_behavior_catalog_json(self, payload_json: str = "") -> None:
        try:
            payload = json.loads(str(payload_json or "{}"))
        except Exception as exc:
            self._set_bot_action_result(
                "behavior_catalog",
                {"ok": False, "error": f"Invalid JSON payload: {exc}"},
            )
            self.status_label.setText(
                "Behavior catalog action failed: invalid payload."
            )
            return

        if not isinstance(payload, dict):
            self._set_bot_action_result(
                "behavior_catalog",
                {"ok": False, "error": "Invalid JSON payload type."},
            )
            self.status_label.setText(
                "Behavior catalog action failed: invalid payload."
            )
            return

        action = str(payload.get("action") or "list").strip().lower()
        kwargs = {
            "code": str(payload.get("code") or "").strip(),
            "name": str(payload.get("name") or "").strip() or None,
            "description": str(payload.get("description") or "").strip() or None,
            "category_id": str(payload.get("category_id") or "").strip() or None,
            "modifier_ids": [
                str(item).strip()
                for item in (payload.get("modifier_ids") or [])
                if str(item).strip()
            ],
            "key_binding": str(payload.get("key_binding") or "").strip() or None,
            "is_state": payload.get("is_state"),
            "exclusive_with": [
                str(item).strip()
                for item in (payload.get("exclusive_with") or [])
                if str(item).strip()
            ],
            "save": bool(payload.get("save", True)),
        }
        result = self._behavior_catalog_action(action, **kwargs)
        if action == "create" and result.get("ok", False):
            message = str(result.get("message") or "Behavior created.")
            self.status_label.setText(message)
        elif action == "update" and result.get("ok", False):
            message = str(result.get("message") or "Behavior updated.")
            self.status_label.setText(message)
        elif action == "delete" and result.get("ok", False):
            message = str(result.get("message") or "Behavior deleted.")
            self.status_label.setText(message)
        elif action == "save" and result.get("ok", False):
            path_text = str(result.get("path") or "").strip()
            if path_text:
                self.status_label.setText(
                    f"Behavior catalog saved to {Path(path_text).name}."
                )
            else:
                self.status_label.setText("Behavior catalog saved.")
        elif action == "list" and result.get("ok", False):
            self.status_label.setText(
                f"Behavior catalog has {int(result.get('count') or 0)} item(s)."
            )
        self._set_bot_action_result("behavior_catalog", result)

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

    @QtCore.Slot(str, str, bool, str, int, float, int, int, str, bool, str, str, str)
    def bot_label_behavior_segments(
        self,
        video_path: str = "",
        behavior_labels_csv: str = "",
        use_defined_behavior_list: bool = True,
        segment_mode: str = "timeline",
        segment_frames: int = 60,
        segment_seconds: float = 0.0,
        sample_frames_per_segment: int = 3,
        max_segments: int = 120,
        subject: str = "Agent",
        overwrite_existing: bool = False,
        llm_profile: str = "",
        llm_provider: str = "",
        llm_model: str = "",
    ) -> None:
        host = self.host_window_widget or self.window()
        open_video = getattr(host, "openVideo", None)
        behavior_controller = getattr(host, "behavior_controller", None)
        if behavior_controller is None:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Behavior timeline APIs are unavailable."},
            )
            self.status_label.setText(
                "Bot action failed: behavior timeline unavailable."
            )
            return

        try:
            if (
                self._behavior_label_thread is not None
                and self._behavior_label_thread.isRunning()
            ):
                self._set_bot_action_result(
                    "label_behavior_segments",
                    {
                        "ok": False,
                        "error": "Behavior labeling is already running. Wait for completion or cancel the current run.",
                    },
                )
                self.status_label.setText(
                    "Behavior labeling already running. Please wait."
                )
                return

            video_text = str(video_path or "").strip()
            if video_text:
                if not callable(open_video):
                    raise RuntimeError("Video opening is unavailable.")
                open_video(
                    from_video_list=True,
                    video_path=video_text,
                    programmatic_call=True,
                )
            resolved_video_path = str(getattr(host, "video_file", "") or "").strip()
            if not resolved_video_path:
                resolved_video_path = video_text
            if not resolved_video_path:
                raise RuntimeError("No video is loaded.")

            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(resolved_video_path))
            try:
                total_frames = int(getattr(host, "num_frames", 0) or 0)
                fps = float(getattr(host, "fps", 0.0) or 0.0)
                if total_frames <= 0 and cap.isOpened():
                    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
                if fps <= 0.0 and cap.isOpened():
                    fps = max(0.0, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
            finally:
                cap.release()

            explicit_labels = self._normalize_behavior_labels(
                [
                    p.strip()
                    for p in str(behavior_labels_csv or "").split(",")
                    if p.strip()
                ]
            )
            labels = self._resolve_segment_label_candidates(
                explicit_labels,
                use_defined_behavior_list=bool(use_defined_behavior_list),
            )
            if not labels:
                raise RuntimeError(
                    "No behavior labels provided/found. Define behaviors in schema/flags/timeline or pass labels."
                )

            mode = str(segment_mode or "timeline").strip().lower()
            if total_frames <= 0:
                raise RuntimeError("No video is loaded.")
            max_segments = max(1, int(max_segments))
            segment_frames = max(1, int(segment_frames))
            segment_seconds = max(0.0, float(segment_seconds or 0.0))
            if mode == "uniform" and segment_seconds > 0.0 and fps > 0.0:
                segment_frames = max(1, int(round(segment_seconds * fps)))
            sample_frames_per_segment = max(1, int(sample_frames_per_segment))

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

            if bool(overwrite_existing):
                behavior_controller.clear_behavior_data()

            self._set_bot_action_result(
                "label_behavior_segments",
                {
                    "ok": True,
                    "queued": True,
                    "in_progress": True,
                    "mode": mode,
                    "evaluated_segments": len(intervals),
                    "segment_frames": int(segment_frames),
                    "segment_seconds": float(segment_seconds),
                    "sample_frames_per_segment": int(sample_frames_per_segment),
                    "use_defined_behavior_list": bool(use_defined_behavior_list),
                    "labels_used": labels,
                },
            )
            self.status_label.setText(
                f"Queued behavior labeling for {len(intervals)} segment(s). Running in background..."
            )

            self._behavior_label_run_context = {
                "host": host,
                "behavior_controller": behavior_controller,
                "mode": mode,
                "labels": list(labels),
                "segment_frames": int(segment_frames),
                "segment_seconds": float(segment_seconds),
                "sample_frames_per_segment": int(sample_frames_per_segment),
                "evaluated_segments": int(len(intervals)),
                "processed_segments": 0,
                "skipped_segments": 0,
                "predictions": [],
                "use_defined_behavior_list": bool(use_defined_behavior_list),
                "default_subject": str(subject or "").strip() or None,
                "timestamp_provider": self._behavior_label_timestamp_provider(host),
            }

            thread = QtCore.QThread(self)
            worker = FlexibleWorker(
                self._run_behavior_segment_vlm_worker,
                video_path=str(resolved_video_path),
                intervals=list(intervals),
                labels=list(labels),
                sample_frames_per_segment=int(sample_frames_per_segment),
                llm_profile=str(llm_profile or ""),
                llm_provider=str(llm_provider or ""),
                llm_model=str(llm_model or ""),
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
            worker.progress_signal.connect(
                self._on_behavior_label_progress, QtCore.Qt.QueuedConnection
            )
            worker.preview_signal.connect(
                self._on_behavior_label_preview, QtCore.Qt.QueuedConnection
            )
            worker.finished_signal.connect(
                self._on_behavior_label_finished, QtCore.Qt.QueuedConnection
            )
            worker.finished_signal.connect(thread.quit, QtCore.Qt.QueuedConnection)
            worker.finished_signal.connect(
                worker.deleteLater, QtCore.Qt.QueuedConnection
            )
            thread.finished.connect(thread.deleteLater)

            self._behavior_label_worker = worker
            self._behavior_label_thread = thread
            thread.start()
        except Exception as exc:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": str(exc)},
            )
            self.status_label.setText(f"Bot action failed: {exc}")

    @QtCore.Slot(str)
    def bot_label_behavior_segments_json(self, payload_json: str = "") -> None:
        try:
            payload = json.loads(str(payload_json or "{}"))
        except Exception as exc:
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": f"Invalid JSON payload: {exc}"},
            )
            self.status_label.setText("Bot action failed: invalid labeling payload.")
            return

        if not isinstance(payload, dict):
            self._set_bot_action_result(
                "label_behavior_segments",
                {"ok": False, "error": "Invalid JSON payload type."},
            )
            self.status_label.setText("Bot action failed: invalid labeling payload.")
            return

        self.bot_label_behavior_segments(
            video_path=str(payload.get("video_path") or ""),
            behavior_labels_csv=str(payload.get("behavior_labels_csv") or ""),
            use_defined_behavior_list=bool(
                payload.get("use_defined_behavior_list", True)
            ),
            segment_mode=str(payload.get("segment_mode") or "timeline"),
            segment_frames=int(payload.get("segment_frames") or 60),
            segment_seconds=float(payload.get("segment_seconds") or 0.0),
            sample_frames_per_segment=int(
                payload.get("sample_frames_per_segment") or 3
            ),
            max_segments=int(payload.get("max_segments") or 120),
            subject=str(payload.get("subject") or "Agent"),
            overwrite_existing=bool(payload.get("overwrite_existing", False)),
            llm_profile=str(payload.get("llm_profile") or ""),
            llm_provider=str(payload.get("llm_provider") or ""),
            llm_model=str(payload.get("llm_model") or ""),
        )

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

    # ---- Research Intent Handling ----

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
