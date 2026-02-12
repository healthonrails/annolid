from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Dict, Optional

from annolid.core.agent.bus import MessageBus, OutboundMessage

from .base import BaseChannel
from .discord import DiscordChannel
from .email import EmailChannel
from .slack import SlackChannel
from .telegram import TelegramChannel
from .whatsapp import WhatsAppChannel


class ChannelManager:
    """Manage enabled channel adapters and outbound routing."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        channels_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.bus = bus
        self.channels_config = dict(channels_config or {})
        self.channels: Dict[str, BaseChannel] = {}
        self._logger = logger or logging.getLogger("annolid.agent.channels")
        self._dispatch_task: Optional[asyncio.Task[None]] = None

        self._init_from_config()

    def _is_enabled(self, key: str) -> bool:
        cfg = self.channels_config.get(key, {})
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", False))
        return bool(getattr(cfg, "enabled", False))

    def _init_from_config(self) -> None:
        mapping = {
            "telegram": TelegramChannel,
            "whatsapp": WhatsAppChannel,
            "discord": DiscordChannel,
            "email": EmailChannel,
            "slack": SlackChannel,
        }
        for name, cls in mapping.items():
            if not self._is_enabled(name):
                continue
            cfg = self.channels_config.get(name, {})
            try:
                self.channels[name] = cls(cfg, self.bus)
            except Exception as exc:
                self._logger.warning("Skipping channel %s: %s", name, exc)

    def register_channel(self, channel: BaseChannel) -> None:
        self.channels[channel.name] = channel

    async def start_all(self) -> None:
        if self._dispatch_task is None or self._dispatch_task.done():
            self._dispatch_task = asyncio.create_task(self._dispatch_outbound())
        tasks = [asyncio.create_task(ch.start()) for ch in self.channels.values()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        for channel in self.channels.values():
            with suppress(Exception):
                await channel.stop()
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._dispatch_task
            self._dispatch_task = None

    async def _dispatch_outbound(self) -> None:
        while True:
            try:
                msg = await self.bus.consume_outbound()
                await self._send_to_channel(msg)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Outbound dispatch failure: %s", exc)

    async def _send_to_channel(self, msg: OutboundMessage) -> None:
        channel = self.channels.get(msg.channel)
        if channel is None:
            self._logger.warning("Unknown outbound channel: %s", msg.channel)
            return
        await channel.send(msg)

    def get_channel(self, name: str) -> Optional[BaseChannel]:
        return self.channels.get(name)

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        return list(self.channels.keys())
