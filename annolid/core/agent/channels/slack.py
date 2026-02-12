from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


class SlackChannel(BaseChannel):
    """Dependency-light Slack adapter for bus integration."""

    name = "slack"

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
        if isinstance(config, dict):
            self._bot_user_id = str(config.get("bot_user_id", "") or "")
        else:
            self._bot_user_id = str(getattr(config, "bot_user_id", "") or "")

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

    def strip_bot_mention(self, text: str) -> str:
        if not text or not self._bot_user_id:
            return text
        return re.sub(rf"<@{re.escape(self._bot_user_id)}>\s*", "", text).strip()

    async def ingest(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        clean = self.strip_bot_mention(content)
        return await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=clean,
            media=media,
            metadata=metadata,
        )
