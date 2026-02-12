from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Awaitable, Callable, Dict, List, Optional

from .events import InboundMessage, OutboundMessage

OutboundCallback = Callable[[OutboundMessage], Awaitable[None]]


class MessageBus:
    """
    Async bus for decoupled inbound/outbound agent messaging.

    Notes:
    - `publish_*` and `consume_*` provide direct queue access.
    - `start_dispatcher()` forwards outbound queue messages to subscribed callbacks.
    """

    def __init__(self) -> None:
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._outbound_subscribers: Dict[str, List[OutboundCallback]] = {}
        self._running = False
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._logger = logging.getLogger("annolid.agent.bus")

    async def publish_inbound(self, msg: InboundMessage) -> None:
        await self.inbound.put(msg)

    async def consume_inbound(
        self, *, timeout_s: Optional[float] = None
    ) -> InboundMessage:
        if timeout_s is None:
            return await self.inbound.get()
        return await asyncio.wait_for(self.inbound.get(), timeout=timeout_s)

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        await self.outbound.put(msg)

    async def consume_outbound(
        self, *, timeout_s: Optional[float] = None
    ) -> OutboundMessage:
        if timeout_s is None:
            return await self.outbound.get()
        return await asyncio.wait_for(self.outbound.get(), timeout=timeout_s)

    def subscribe_outbound(self, channel: str, callback: OutboundCallback) -> None:
        channel_key = str(channel or "").strip().lower()
        if not channel_key:
            raise ValueError("channel is required for outbound subscription")
        callbacks = self._outbound_subscribers.setdefault(channel_key, [])
        callbacks.append(callback)

    async def start_dispatcher(self) -> None:
        if (
            self._running
            and self._dispatch_task is not None
            and not self._dispatch_task.done()
        ):
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop_dispatcher(self) -> None:
        self._running = False
        task = self._dispatch_task
        self._dispatch_task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                msg = await self.outbound.get()
                callbacks = self._outbound_subscribers.get(msg.channel.lower(), [])
                for callback in callbacks:
                    try:
                        await callback(msg)
                    except Exception as exc:
                        self._logger.error(
                            "Outbound callback failed for channel=%s: %s",
                            msg.channel,
                            exc,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Dispatcher loop error: %s", exc)

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
