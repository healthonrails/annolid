from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from contextlib import suppress
from typing import Optional

from ..loop import AgentLoop
from .events import InboundMessage, OutboundMessage
from .queue import MessageBus


class AgentBusService:
    """Bridge inbound bus messages to AgentLoop and publish outbound replies."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        loop: AgentLoop,
        max_idempotency_cache: int = 512,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.bus = bus
        self.loop = loop
        self._logger = logger or logging.getLogger("annolid.agent.bus.service")
        self._max_idempotency_cache = max(16, int(max_idempotency_cache))
        self._idempotency_cache: OrderedDict[str, OutboundMessage] = OrderedDict()
        self._outbound_seq = 0
        self._state_version = 0
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if self._running and self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        while self._running:
            try:
                inbound = await self.bus.consume_inbound()
                await self._process_inbound(inbound)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Bus service loop error: %s", exc)

    async def _process_inbound(self, inbound: InboundMessage) -> None:
        cache_key = self._idempotency_cache_key(inbound)
        if cache_key:
            cached = self._idempotency_cache.get(cache_key)
            if cached is not None:
                replay = self._annotate_outbound(
                    OutboundMessage(
                        channel=cached.channel,
                        chat_id=cached.chat_id,
                        content=cached.content,
                        reply_to=cached.reply_to,
                        media=list(cached.media or []),
                        metadata={
                            **dict(cached.metadata or {}),
                            "idempotency_replay": True,
                        },
                    )
                )
                await self.bus.publish_outbound(replay)
                return
        try:
            result = await self.loop.run(
                inbound.content,
                session_id=inbound.session_key,
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                media=list(inbound.media or []),
            )
            outbound = OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=result.content,
                metadata={
                    "iterations": result.iterations,
                    "tool_runs": len(result.tool_runs),
                    "stopped_reason": result.stopped_reason,
                },
            )
        except Exception as exc:
            outbound = OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=f"Error: {exc}",
                metadata={"error": True},
            )
        outbound = self._annotate_outbound(outbound)
        self._store_idempotency(cache_key, outbound)
        await self.bus.publish_outbound(outbound)

    def _idempotency_cache_key(self, inbound: InboundMessage) -> str:
        raw = inbound.metadata.get("idempotency_key")
        key = str(raw or "").strip()
        if not key:
            return ""
        return f"{inbound.session_key}:{key}"

    def _store_idempotency(self, cache_key: str, outbound: OutboundMessage) -> None:
        if not cache_key:
            return
        self._idempotency_cache[cache_key] = outbound
        self._idempotency_cache.move_to_end(cache_key)
        while len(self._idempotency_cache) > self._max_idempotency_cache:
            self._idempotency_cache.popitem(last=False)

    def _annotate_outbound(self, outbound: OutboundMessage) -> OutboundMessage:
        self._outbound_seq += 1
        self._state_version += 1
        meta = dict(outbound.metadata or {})
        meta.setdefault("seq", self._outbound_seq)
        meta.setdefault("state_version", self._state_version)
        return OutboundMessage(
            channel=outbound.channel,
            chat_id=outbound.chat_id,
            content=outbound.content,
            reply_to=outbound.reply_to,
            media=list(outbound.media or []),
            metadata=meta,
        )
