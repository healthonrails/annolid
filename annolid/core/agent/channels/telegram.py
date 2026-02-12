from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


def markdown_to_telegram_html(text: str) -> str:
    """Convert lightweight markdown into Telegram-safe HTML."""
    if not text:
        return ""

    code_blocks: list[str] = []

    def _save_block(match: re.Match[str]) -> str:
        code_blocks.append(match.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    inline_codes: list[str] = []

    def _save_inline(match: re.Match[str]) -> str:
        inline_codes.append(match.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", _save_block, text)
    text = re.sub(r"`([^`]+)`", _save_inline, text)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*(.*)$", r"\1", text, flags=re.MULTILINE)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    for i, code in enumerate(inline_codes):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")
    for i, code in enumerate(code_blocks):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")
    return text


class TelegramChannel(BaseChannel):
    """Dependency-light Telegram adapter for Annolid bus integration."""

    name = "telegram"

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
