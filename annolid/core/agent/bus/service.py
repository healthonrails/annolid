from __future__ import annotations

import asyncio
import logging
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
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.bus = bus
        self.loop = loop
        self._logger = logger or logging.getLogger("annolid.agent.bus.service")
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
        await self.bus.publish_outbound(outbound)
