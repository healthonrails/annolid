from __future__ import annotations

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
        sender = str(sender_id)
        if sender in allow_list:
            return True
        if "|" in sender:
            for token in sender.split("|"):
                if token and token in allow_list:
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
            return False
        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=str(content),
            media=list(media or []),
            metadata=dict(metadata or {}),
        )
        await self.bus.publish_inbound(msg)
        return True

    @property
    def is_running(self) -> bool:
        return bool(self._running)
