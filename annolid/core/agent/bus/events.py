from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _normalize_scope(scope: str) -> str:
    value = str(scope or "").strip().lower()
    allowed = {
        "main",
        "per-peer",
        "per-channel-peer",
        "per-account-channel-peer",
    }
    return value if value in allowed else ""


def _clean_segment(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip().replace(":", "_")
    if text:
        return text
    return fallback


def resolve_session_key(
    *,
    channel: str,
    chat_id: str,
    sender_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    default_dm_scope: str = "main",
    default_main_key: str = "",
) -> str:
    """Resolve a stable session key with optional DM isolation scopes.

    Metadata hints:
    - conversation_type/chat_type/is_dm
    - session_dm_scope or dm_scope
    - peer_id, channel_key/channel_id, account_id/workspace_id/tenant_id
    """
    base_channel = _clean_segment(channel, "channel")
    base_chat = _clean_segment(chat_id, "main")
    meta = dict(metadata or {})

    conversation_type = str(
        meta.get("conversation_type") or meta.get("chat_type") or ""
    ).strip()
    is_dm = bool(meta.get("is_dm")) or conversation_type.lower() in {
        "dm",
        "direct",
        "direct_message",
        "private",
    }
    if not is_dm:
        return f"{base_channel}:{base_chat}"

    scope = _normalize_scope(meta.get("session_dm_scope") or meta.get("dm_scope"))
    if not scope:
        scope = _normalize_scope(default_dm_scope)
    if not scope:
        return f"{base_channel}:{base_chat}"

    peer = _clean_segment(meta.get("peer_id") or sender_id or chat_id, "unknown")
    channel_key = _clean_segment(meta.get("channel_key") or meta.get("channel_id"))
    if not channel_key:
        channel_key = base_chat
    main_key = _clean_segment(meta.get("main_session_key") or default_main_key)
    if not main_key:
        main_key = base_chat

    if scope == "main":
        return f"{base_channel}:{main_key}"
    if scope == "per-peer":
        return f"{base_channel}:dm:{peer}"
    if scope == "per-channel-peer":
        return f"{base_channel}:{channel_key}:dm:{peer}"
    if scope == "per-account-channel-peer":
        account = _clean_segment(
            meta.get("account_id") or meta.get("workspace_id") or meta.get("tenant_id")
        )
        if account:
            return f"{base_channel}:{account}:{channel_key}:dm:{peer}"
        return f"{base_channel}:{channel_key}:dm:{peer}"
    return f"{base_channel}:{base_chat}"


@dataclass(frozen=True)
class InboundMessage:
    """Inbound chat event produced by channels/adapters."""

    channel: str
    sender_id: str
    chat_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    media: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        return self.resolved_session_key()

    def resolved_session_key(
        self,
        *,
        default_dm_scope: str = "main",
        default_main_key: str = "",
    ) -> str:
        return resolve_session_key(
            channel=self.channel,
            chat_id=self.chat_id,
            sender_id=self.sender_id,
            metadata=self.metadata,
            default_dm_scope=default_dm_scope,
            default_main_key=default_main_key,
        )


@dataclass(frozen=True)
class OutboundMessage:
    """Outbound chat event produced by agent runtime."""

    channel: str
    chat_id: str
    content: str
    reply_to: Optional[str] = None
    media: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
