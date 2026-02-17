from __future__ import annotations

import asyncio
from annolid.utils.logger import logger
from collections import OrderedDict
from contextlib import suppress
from typing import Any, Optional

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
        default_dm_scope: str = "main",
        default_main_session_key: str = "",
    ) -> None:
        self.bus = bus
        self.loop = loop
        self._logger = logger
        self._max_idempotency_cache = max(16, int(max_idempotency_cache))
        self._default_dm_scope = str(default_dm_scope or "").strip() or "main"
        self._default_main_session_key = str(default_main_session_key or "").strip()
        self._idempotency_cache: OrderedDict[str, OutboundMessage] = OrderedDict()
        self._outbound_seq = 0
        self._state_version = 0
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None

    @classmethod
    def from_agent_config(
        cls,
        *,
        bus: MessageBus,
        loop: AgentLoop,
        agent_config: Any,
        max_idempotency_cache: int = 512,
    ) -> "AgentBusService":
        dm_scope = "main"
        main_key = "main"
        try:
            defaults = agent_config.agents.defaults
            session = getattr(defaults, "session", None)
            if session is not None:
                dm_scope = str(getattr(session, "dm_scope", dm_scope) or dm_scope)
                main_key = str(
                    getattr(session, "main_session_key", main_key) or main_key
                )
        except Exception:
            pass
        return cls(
            bus=bus,
            loop=loop,
            max_idempotency_cache=max_idempotency_cache,
            default_dm_scope=dm_scope,
            default_main_session_key=main_key,
        )

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
                self._logger.info(
                    "New message from %s via %s",
                    inbound.sender_id,
                    inbound.channel,
                )
                await self._process_inbound(inbound)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Bus service loop error: %s", exc)

    async def _process_inbound(self, inbound: InboundMessage) -> None:
        session_key = self._resolve_session_key(inbound)
        cache_key = self._idempotency_cache_key(inbound, session_key=session_key)
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
                session_id=session_key,
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                media=list(inbound.media or []),
            )
            outbound_meta = {
                "iterations": result.iterations,
                "tool_runs": len(result.tool_runs),
                "stopped_reason": result.stopped_reason,
            }
            if inbound.metadata.get("subject"):
                outbound_meta["subject"] = inbound.metadata["subject"]

            outbound = OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=result.content,
                metadata=outbound_meta,
            )
        except Exception as exc:
            outbound = OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=f"Error: {exc}",
                metadata={"error": True},
            )
        normalized = str(outbound.content or "").strip()
        if not normalized:
            if str(inbound.channel or "").strip().lower() == "email":
                fallback = self._build_empty_email_fallback(inbound)
                meta = dict(outbound.metadata or {})
                meta["empty_reply_fallback"] = True
                outbound = OutboundMessage(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=fallback,
                    reply_to=outbound.reply_to,
                    media=list(outbound.media or []),
                    metadata=meta,
                )
                self._logger.warning(
                    "Generated empty email reply; substituted fallback text for %s",
                    outbound.chat_id,
                )
            else:
                self._logger.warning(
                    "Generated empty outbound reply; skipping publish channel=%s chat=%s",
                    outbound.channel,
                    outbound.chat_id,
                )
                return
        elif normalized != outbound.content:
            outbound = OutboundMessage(
                channel=outbound.channel,
                chat_id=outbound.chat_id,
                content=normalized,
                reply_to=outbound.reply_to,
                media=list(outbound.media or []),
                metadata=dict(outbound.metadata or {}),
            )
        outbound = self._annotate_outbound(outbound)
        self._store_idempotency(cache_key, outbound)
        self._logger.info(
            "Publishing reply to %s via %s", outbound.chat_id, outbound.channel
        )
        await self.bus.publish_outbound(outbound)

    @staticmethod
    def _build_empty_email_fallback(inbound: InboundMessage) -> str:
        subject = str((inbound.metadata or {}).get("subject") or "").strip()
        if subject:
            return (
                "I received your email"
                f" about '{subject}', but I couldn't generate a complete reply. "
                "Please resend with a bit more detail and I will try again."
            )
        return (
            "I received your email, but I couldn't generate a complete reply. "
            "Please resend with a bit more detail and I will try again."
        )

    def _resolve_session_key(self, inbound: InboundMessage) -> str:
        return inbound.resolved_session_key(
            default_dm_scope=self._default_dm_scope,
            default_main_key=self._default_main_session_key,
        )

    def _idempotency_cache_key(
        self,
        inbound: InboundMessage,
        *,
        session_key: str,
    ) -> str:
        raw = inbound.metadata.get("idempotency_key")
        key = str(raw or "").strip()
        if not key:
            return ""
        return f"{session_key}:{key}"

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
