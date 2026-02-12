from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


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
        return f"{self.channel}:{self.chat_id}"


@dataclass(frozen=True)
class OutboundMessage:
    """Outbound chat event produced by agent runtime."""

    channel: str
    chat_id: str
    content: str
    reply_to: Optional[str] = None
    media: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
