from __future__ import annotations

import asyncio
import ipaddress
from annolid.utils.logger import logger
from collections import OrderedDict, deque
from contextlib import suppress
import hashlib
import re
import time
from typing import Any, Deque, Optional
from urllib.parse import urlsplit, urlunsplit

from ..loop import AgentLoop
from ..gui_backend.turn_state import (
    ERROR_TYPE_INTERNAL,
    ERROR_TYPE_NONE,
    ERROR_TYPE_TRANSPORT,
    TURN_STATUS_COMPLETED,
    TURN_STATUS_FAILED,
    TURN_STATUS_RUNNING,
)
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
        max_parallel_sessions: int = 1,
        max_pending_messages: int = 2048,
        collapse_superseded_pending: bool = True,
        transient_retry_attempts: int = 2,
        transient_retry_initial_backoff_s: float = 0.5,
        transient_retry_max_backoff_s: float = 4.0,
        default_dm_scope: str = "main",
        default_main_session_key: str = "",
    ) -> None:
        self.bus = bus
        self.loop = loop
        self._logger = logger
        self._max_idempotency_cache = max(16, int(max_idempotency_cache))
        self._max_parallel_sessions = max(1, int(max_parallel_sessions))
        self._max_pending_messages = max(1, int(max_pending_messages))
        self._collapse_superseded_pending = bool(collapse_superseded_pending)
        self._transient_retry_attempts = max(0, int(transient_retry_attempts))
        self._transient_retry_initial_backoff_s = max(
            0.0, float(transient_retry_initial_backoff_s)
        )
        self._transient_retry_max_backoff_s = max(
            self._transient_retry_initial_backoff_s,
            float(transient_retry_max_backoff_s),
        )
        self._default_dm_scope = str(default_dm_scope or "").strip() or "main"
        self._default_main_session_key = str(default_main_session_key or "").strip()
        self._idempotency_cache: OrderedDict[str, OutboundMessage] = OrderedDict()
        self._outbound_dedupe_cache: OrderedDict[str, float] = OrderedDict()
        self._pending_by_session: dict[str, Deque[InboundMessage]] = {}
        self._runnable_sessions: Deque[str] = deque()
        self._active_sessions: set[str] = set()
        self._scheduled_count = 0
        self._scheduler_condition = asyncio.Condition()
        self._outbound_seq = 0
        self._state_version = 0
        self._running = False
        self._dispatcher_task: Optional[asyncio.Task[None]] = None
        self._worker_tasks: list[asyncio.Task[None]] = []

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
        max_parallel_sessions = 1
        max_pending_messages = 2048
        collapse_superseded_pending = True
        transient_retry_attempts = 2
        transient_retry_initial_backoff_s = 0.5
        transient_retry_max_backoff_s = 4.0
        try:
            defaults = agent_config.agents.defaults
            session = getattr(defaults, "session", None)
            if session is not None:
                dm_scope = str(getattr(session, "dm_scope", dm_scope) or dm_scope)
                main_key = str(
                    getattr(session, "main_session_key", main_key) or main_key
                )
            max_parallel_sessions = max(
                1, int(getattr(defaults, "max_parallel_sessions", 1))
            )
            max_pending_messages = max(
                1, int(getattr(defaults, "max_pending_messages", 2048))
            )
            collapse_superseded_pending = bool(
                getattr(defaults, "collapse_superseded_pending", True)
            )
            transient_retry_attempts = max(
                0, int(getattr(defaults, "transient_retry_attempts", 2))
            )
            transient_retry_initial_backoff_s = max(
                0.0, float(getattr(defaults, "transient_retry_initial_backoff_s", 0.5))
            )
            transient_retry_max_backoff_s = max(
                transient_retry_initial_backoff_s,
                float(getattr(defaults, "transient_retry_max_backoff_s", 4.0)),
            )
        except Exception:
            pass
        return cls(
            bus=bus,
            loop=loop,
            max_idempotency_cache=max_idempotency_cache,
            max_parallel_sessions=max_parallel_sessions,
            max_pending_messages=max_pending_messages,
            collapse_superseded_pending=collapse_superseded_pending,
            transient_retry_attempts=transient_retry_attempts,
            transient_retry_initial_backoff_s=transient_retry_initial_backoff_s,
            transient_retry_max_backoff_s=transient_retry_max_backoff_s,
            default_dm_scope=dm_scope,
            default_main_session_key=main_key,
        )

    async def start(self) -> None:
        if self._running and self._dispatcher_task is not None:
            return
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._run())
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self._max_parallel_sessions)
        ]

    async def stop(self) -> None:
        self._running = False
        async with self._scheduler_condition:
            self._scheduler_condition.notify_all()
        task = self._dispatcher_task
        self._dispatcher_task = None
        if task is not None:
            task.cancel()
        worker_tasks = list(self._worker_tasks)
        self._worker_tasks = []
        for worker in worker_tasks:
            worker.cancel()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task
        for worker in worker_tasks:
            with suppress(asyncio.CancelledError):
                await worker

    async def _run(self) -> None:
        while self._running:
            try:
                inbound = await self.bus.consume_inbound()
                session_key = self._resolve_session_key(inbound)
                self._logger.info(
                    "Queued message from %s via %s session=%s",
                    inbound.sender_id,
                    inbound.channel,
                    session_key,
                )
                await self._enqueue_for_scheduling(
                    inbound=inbound, session_key=session_key
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Bus service loop error: %s", exc)

    async def _worker_loop(self, worker_index: int) -> None:
        while self._running:
            try:
                scheduled = await self._dequeue_scheduled_item()
                if scheduled is None:
                    continue
                session_key, inbound = scheduled
                self._logger.debug(
                    "Worker %d handling session=%s channel=%s chat=%s",
                    worker_index,
                    session_key,
                    inbound.channel,
                    inbound.chat_id,
                )
                try:
                    await self._process_inbound(inbound, session_key=session_key)
                finally:
                    await self._complete_scheduled_item(session_key)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Worker loop error: %s", exc)

    async def _enqueue_for_scheduling(
        self, *, inbound: InboundMessage, session_key: str
    ) -> None:
        overflowed = False
        collapsed = 0
        async with self._scheduler_condition:
            if self._scheduled_count >= self._max_pending_messages:
                overflowed = True
            else:
                queue = self._pending_by_session.setdefault(session_key, deque())
                if (
                    self._collapse_superseded_pending
                    and not bool((inbound.metadata or {}).get("queue_keep_all"))
                    and session_key in self._active_sessions
                    and len(queue) > 0
                ):
                    collapsed = len(queue)
                    queue.clear()
                    self._scheduled_count = max(0, self._scheduled_count - collapsed)
                queue.append(inbound)
                self._scheduled_count += 1
                if (
                    session_key not in self._active_sessions
                    and session_key not in self._runnable_sessions
                ):
                    self._runnable_sessions.append(session_key)
                self._scheduler_condition.notify()
        if overflowed:
            await self.bus.publish_outbound(
                self._annotate_outbound(
                    OutboundMessage(
                        channel=inbound.channel,
                        chat_id=inbound.chat_id,
                        content="System busy, please retry shortly.",
                        metadata={
                            "error": True,
                            "error_type": ERROR_TYPE_TRANSPORT,
                            "turn_status": TURN_STATUS_FAILED,
                            "dropped_by_scheduler": True,
                        },
                    )
                )
            )
            self._logger.warning(
                "Dropped inbound due scheduler overflow session=%s channel=%s chat=%s",
                session_key,
                inbound.channel,
                inbound.chat_id,
            )
        elif collapsed > 0:
            self._logger.info(
                "Collapsed %d superseded pending prompts session=%s",
                collapsed,
                session_key,
            )

    async def _dequeue_scheduled_item(self) -> Optional[tuple[str, InboundMessage]]:
        async with self._scheduler_condition:
            while self._running:
                if self._runnable_sessions:
                    session_key = self._runnable_sessions.popleft()
                    queue = self._pending_by_session.get(session_key)
                    if queue:
                        inbound = queue.popleft()
                        self._scheduled_count = max(0, self._scheduled_count - 1)
                        self._active_sessions.add(session_key)
                        return (session_key, inbound)
                    self._pending_by_session.pop(session_key, None)
                    self._active_sessions.discard(session_key)
                    continue
                await self._scheduler_condition.wait()
            return None

    async def _complete_scheduled_item(self, session_key: str) -> None:
        async with self._scheduler_condition:
            self._active_sessions.discard(session_key)
            queue = self._pending_by_session.get(session_key)
            if queue and len(queue) > 0:
                self._runnable_sessions.append(session_key)
                self._scheduler_condition.notify()
            else:
                self._pending_by_session.pop(session_key, None)

    async def _process_inbound(
        self,
        inbound: InboundMessage,
        *,
        session_key: Optional[str] = None,
    ) -> None:
        resolved_session_key = str(
            session_key or ""
        ).strip() or self._resolve_session_key(inbound)
        cache_key = self._idempotency_cache_key(
            inbound, session_key=resolved_session_key
        )
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
            last_progress: str = ""

            async def _on_progress(content: str) -> None:
                nonlocal last_progress
                text = self._sanitize_outbound_content(content)
                if not text or text == last_progress:
                    return
                last_progress = text
                await self._publish_intermediate_progress(
                    inbound=inbound,
                    content=text,
                )

            result = await self._run_with_transient_retry(
                inbound=inbound,
                resolved_session_key=resolved_session_key,
                on_progress=_on_progress,
            )
            outbound_meta = {
                "iterations": result.iterations,
                "tool_runs": len(result.tool_runs),
                "stopped_reason": result.stopped_reason,
                "turn_status": TURN_STATUS_COMPLETED,
                "error_type": ERROR_TYPE_NONE,
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
                metadata={
                    "error": True,
                    "error_type": (
                        ERROR_TYPE_TRANSPORT
                        if self._is_transient_failure(exc)
                        else ERROR_TYPE_INTERNAL
                    ),
                    "turn_status": TURN_STATUS_FAILED,
                },
            )
        normalized = self._sanitize_outbound_content(outbound.content)
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
        if self._is_duplicate_outbound(outbound):
            self._logger.warning(
                "Skipping duplicate outbound message channel=%s chat=%s",
                outbound.channel,
                outbound.chat_id,
            )
            return
        self._store_idempotency(cache_key, outbound)
        self._logger.info(
            "Publishing reply to %s via %s", outbound.chat_id, outbound.channel
        )
        await self.bus.publish_outbound(outbound)

    async def _publish_intermediate_progress(
        self,
        *,
        inbound: InboundMessage,
        content: str,
    ) -> None:
        progress_meta = {
            "intermediate": True,
            "progress": True,
            "turn_status": TURN_STATUS_RUNNING,
            "error_type": ERROR_TYPE_NONE,
        }
        if inbound.metadata.get("subject"):
            progress_meta["subject"] = inbound.metadata["subject"]
        outbound = self._annotate_outbound(
            OutboundMessage(
                channel=inbound.channel,
                chat_id=inbound.chat_id,
                content=str(content),
                metadata=progress_meta,
            )
        )
        await self.bus.publish_outbound(outbound)

    @staticmethod
    def _sanitize_outbound_content(content: str | None) -> str:
        raw = str(content or "")
        if not raw:
            return ""
        without_think = re.sub(r"<think>[\s\S]*?</think>", "", raw)
        return AgentBusService._redact_sensitive_text(str(without_think or "").strip())

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
        meta = self._redact_sensitive_metadata(meta)
        meta.setdefault("seq", self._outbound_seq)
        meta.setdefault("state_version", self._state_version)
        return OutboundMessage(
            channel=outbound.channel,
            chat_id=outbound.chat_id,
            content=self._sanitize_outbound_content(outbound.content),
            reply_to=outbound.reply_to,
            media=list(outbound.media or []),
            metadata=meta,
        )

    @staticmethod
    def _is_private_host(host: str) -> bool:
        text = str(host or "").strip().strip("[]").lower()
        if not text:
            return False
        if text in {"localhost", "127.0.0.1", "::1"}:
            return True
        if text.endswith(".local") or text.endswith(".lan"):
            return True
        try:
            ip = ipaddress.ip_address(text)
            return bool(
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            )
        except ValueError:
            return False

    @classmethod
    def _redact_sensitive_url(cls, raw_url: str) -> str:
        text = str(raw_url or "").strip()
        if "://" not in text:
            return text
        with suppress(Exception):
            parts = urlsplit(text)
            host = str(parts.hostname or "")
            if cls._is_private_host(host):
                port = f":{parts.port}" if parts.port else ""
                redacted_netloc = f"<private-host>{port}"
                return urlunsplit(
                    (
                        parts.scheme,
                        redacted_netloc,
                        parts.path,
                        parts.query,
                        parts.fragment,
                    )
                )
        return text

    @classmethod
    def _redact_sensitive_text(cls, text: str) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        pattern = re.compile(
            r"\b(?:https?|rtsp|rtsps|rtp|udp|tcp|srt)://[^\s<>\]\[)\"']+",
            re.IGNORECASE,
        )
        return pattern.sub(lambda m: cls._redact_sensitive_url(m.group(0)), raw)

    @classmethod
    def _redact_sensitive_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                k = str(key)
                if k.lower() in cls._SENSITIVE_META_KEYS:
                    redacted[k] = "<redacted>"
                else:
                    redacted[k] = cls._redact_sensitive_metadata(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact_sensitive_metadata(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._redact_sensitive_metadata(v) for v in value)
        if isinstance(value, str):
            return cls._redact_sensitive_text(value)
        return value

    def _is_duplicate_outbound(self, outbound: OutboundMessage) -> bool:
        meta = dict(outbound.metadata or {})
        if bool(meta.get("idempotency_replay")):
            return False
        digest_src = "|".join(
            [
                str(outbound.channel or ""),
                str(outbound.chat_id or ""),
                str(outbound.content or ""),
                str(bool(meta.get("intermediate", False))),
                str(meta.get("turn_status") or ""),
                str(meta.get("error_type") or ""),
            ]
        )
        key = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()
        now = time.monotonic()
        stamp = self._outbound_dedupe_cache.get(key)
        self._outbound_dedupe_cache[key] = now
        self._outbound_dedupe_cache.move_to_end(key)
        while len(self._outbound_dedupe_cache) > self._max_idempotency_cache:
            self._outbound_dedupe_cache.popitem(last=False)
        if stamp is None:
            return False
        return (now - float(stamp)) <= 1.0

    async def _run_with_transient_retry(
        self,
        *,
        inbound: InboundMessage,
        resolved_session_key: str,
        on_progress,
    ):
        attempts = 0
        backoff = self._transient_retry_initial_backoff_s
        while True:
            try:
                return await self.loop.run(
                    inbound.content,
                    session_id=resolved_session_key,
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    media=list(inbound.media or []),
                    on_progress=on_progress,
                    inbound_metadata=dict(inbound.metadata or {}),
                )
            except Exception as exc:
                if (
                    not self._is_transient_failure(exc)
                    or attempts >= self._transient_retry_attempts
                ):
                    raise
                attempts += 1
                delay = min(backoff, self._transient_retry_max_backoff_s)
                self._logger.warning(
                    "Transient run failure session=%s attempt=%d/%d backoff=%.2fs error=%s",
                    resolved_session_key,
                    attempts,
                    self._transient_retry_attempts,
                    delay,
                    exc,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                backoff = max(delay * 2.0, self._transient_retry_initial_backoff_s)

    @staticmethod
    def _is_transient_failure(exc: Exception) -> bool:
        if isinstance(
            exc, (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError)
        ):
            return True
        text = str(exc or "").strip().lower()
        if not text:
            return False
        transient_markers = (
            "timeout",
            "timed out",
            "temporarily unavailable",
            "temporary failure",
            "rate limit",
            "too many requests",
            "connection reset",
            "connection refused",
            "try again",
            "service unavailable",
            "overloaded",
        )
        return any(marker in text for marker in transient_markers)

    _SENSITIVE_META_KEYS = frozenset(
        {
            "peer_id",
            "account_id",
            "workspace_id",
            "tenant_id",
            "channel_key",
            "idempotency_key",
        }
    )
