from annolid.utils.logger import logger
from abc import ABC, abstractmethod
from typing import Any, Optional

from annolid.core.agent.bus import InboundMessage, MessageBus, OutboundMessage


class BaseChannel(ABC):
    """Base channel interface for bus-driven channel adapters."""

    name: str = "base"

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start receiving channel events."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop channel resources."""

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send an outbound bus message via channel."""

    def is_allowed(self, sender_id: str) -> bool:
        if isinstance(self.config, dict):
            allow_list = self.config.get("allow_from")
        else:
            allow_list = getattr(self.config, "allow_from", None)
        if not allow_list:
            return True
        sender = str(sender_id).lower()
        for allowed in allow_list:
            if str(allowed).lower() in sender:
                return True
        return False

    async def _handle_message(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.is_allowed(sender_id):
            logger.warning(
                "Channel %s: Message from %s blocked by allow_from list.",
                self.name,
                sender_id,
            )
            return False
        normalized_meta = self._normalize_session_metadata(
            sender_id=sender_id,
            chat_id=chat_id,
            metadata=metadata,
        )
        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=str(content),
            media=list(media or []),
            metadata=normalized_meta,
        )
        await self.bus.publish_inbound(msg)
        return True

    def _normalize_session_metadata(
        self,
        *,
        sender_id: str,
        chat_id: str,
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        sender = str(sender_id or "")
        chat = str(chat_id or "")
        merged.setdefault("peer_id", sender or chat)
        merged.setdefault("channel_key", chat or sender)

        conversation_type = str(
            merged.get("conversation_type") or merged.get("chat_type") or ""
        ).strip()
        if conversation_type:
            merged.setdefault("conversation_type", conversation_type.lower())
        is_dm_raw = merged.get("is_dm")

        if "is_dm" not in merged:
            lowered = conversation_type.lower()
            if lowered in {"dm", "direct", "direct_message", "private"}:
                merged["is_dm"] = True
            elif lowered in {"group", "channel", "room", "thread"}:
                merged["is_dm"] = False
            else:
                merged["is_dm"] = bool(sender and chat and sender == chat)
        is_dm = bool(merged.get("is_dm"))
        if not conversation_type:
            merged["conversation_type"] = "dm" if is_dm else "channel"
        elif is_dm and conversation_type.lower() not in {
            "dm",
            "direct",
            "direct_message",
            "private",
        }:
            merged["conversation_type"] = "dm"

        if is_dm_raw is None:
            merged["is_dm"] = is_dm
        return merged

    @property
    def is_running(self) -> bool:
        return bool(self._running)
