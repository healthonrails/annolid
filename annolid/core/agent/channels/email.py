from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


class EmailChannel(BaseChannel):
    """Dependency-light email adapter for bus integration."""

    name = "email"

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
        sender_email: str,
        subject: str,
        body: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        content = f"Email received.\nFrom: {sender_email}\nSubject: {subject}\n\n{body}"
        merged = dict(metadata or {})
        merged.setdefault("subject", subject)
        merged.setdefault("sender_email", sender_email)
        return await self._handle_message(
            sender_id=sender_email,
            chat_id=sender_email,
            content=content,
            metadata=merged,
        )
