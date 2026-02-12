from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


class WhatsAppChannel(BaseChannel):
    """Dependency-light WhatsApp adapter for bus integration."""

    name = "whatsapp"

    def __init__(
        self,
        config: Any,
        bus,
        *,
        send_callback: Optional[SendCallback] = None,
    ):
        super().__init__(config, bus)
        self._send_callback = send_callback
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        self._running = True
        self._stop_event.clear()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    async def send(self, msg: OutboundMessage) -> None:
        if self._send_callback is None:
            return
        ret = self._send_callback(msg)
        if asyncio.iscoroutine(ret):
            await ret

    async def ingest(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        return await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata=metadata,
        )
