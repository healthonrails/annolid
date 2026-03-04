from __future__ import annotations

import asyncio
import base64
from collections import deque
import contextlib
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage
from annolid.utils.logger import logger

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


class ZulipChannel(BaseChannel):
    """Zulip adapter with lightweight polling and outbound send support."""

    name = "zulip"

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
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._last_message_id: int = 0
        self._anchor_initialized = False
        self._processed_message_ids: set[int] = set()
        self._processed_message_order: deque[int] = deque()
        raw_max = self._cfg("max_processed_ids", 4096)
        self._max_processed_ids: int = max(256, int(raw_max or 4096))
        self._log_skip_reasons = bool(self._cfg("log_skip_reasons", False))
        self._unread_backfill_enabled = bool(self._cfg("unread_backfill_enabled", True))
        self._unread_backfill_on_empty_only = bool(
            self._cfg("unread_backfill_on_empty_only", True)
        )
        self._unread_backfill_limit = max(
            1, min(200, int(self._cfg("unread_backfill_limit", 50) or 50))
        )
        self._unread_backfill_cooldown_s = max(
            0, int(self._cfg("unread_backfill_cooldown_s", 300) or 300)
        )
        self._last_unread_backfill_at = 0.0
        self._cursor_state_path = self._resolve_cursor_state_path()
        self._cursor_state_loaded = False
        self._cursor_state_has_checkpoint = False
        self._startup_unix_s = int(time.time())
        self._missing_config_warned = False

    async def start(self) -> None:
        if self._running:
            return
        self._load_cursor_state_once()
        self._running = True
        self._stop_event.clear()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def send(self, msg: OutboundMessage) -> None:
        content = str(msg.content or "").strip()
        if not content:
            logger.warning("Skipping empty Zulip reply to %s", msg.chat_id)
            return
        if self._send_callback:
            ret = self._send_callback(msg)
            if asyncio.iscoroutine(ret):
                await ret
            return
        await self._send_via_api(msg)

    async def _poll_loop(self) -> None:
        last_interval: Optional[int] = None
        while self._running:
            interval = self._resolve_poll_interval_seconds()
            if interval != last_interval:
                logger.info("Zulip poll interval set to %ss", interval)
                last_interval = interval
            try:
                await self._poll_messages()
            except Exception as exc:
                logger.error("Zulip polling failure: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                continue

    def _resolve_poll_interval_seconds(self) -> int:
        raw = int(self._cfg("polling_interval", 30) or 30)
        return max(5, raw)

    def _cfg(self, key: str, default: Any = "") -> Any:
        aliases = {
            "server_url": ("serverUrl",),
            "api_key": ("apiKey",),
            "polling_interval": ("pollingInterval",),
            "allow_from": ("allowFrom",),
            "cursor_state_path": ("cursorStatePath",),
            "max_processed_ids": ("maxProcessedIds",),
            "log_skip_reasons": ("logSkipReasons",),
            "bot_name": ("botName",),
            "unread_backfill_enabled": ("unreadBackfillEnabled",),
            "unread_backfill_on_empty_only": ("unreadBackfillOnEmptyOnly",),
            "unread_backfill_limit": ("unreadBackfillLimit",),
            "unread_backfill_cooldown_s": ("unreadBackfillCooldownS",),
        }
        if isinstance(self.config, dict):
            if key in self.config:
                return self.config.get(key, default)
            for alias in aliases.get(key, ()):
                if alias in self.config:
                    return self.config.get(alias, default)
            return default
        value = getattr(self.config, key, None)
        if value is not None:
            return value
        for alias in aliases.get(key, ()):
            value = getattr(self.config, alias, None)
            if value is not None:
                return value
        return default

    async def _poll_messages(self) -> None:
        self._load_cursor_state_once()
        base_url = str(self._cfg("server_url", "")).strip().rstrip("/")
        user = str(self._cfg("user", "")).strip()
        api_key = str(self._cfg("api_key", "")).strip()
        if not (base_url and user and api_key):
            if not self._missing_config_warned:
                logger.warning("Zulip polling skipped: missing server_url/user/api_key")
                self._missing_config_warned = True
            return
        self._missing_config_warned = False

        stream = str(self._cfg("stream", "") or "").strip() or "*"
        topic = str(self._cfg("topic", "") or "").strip() or "*"
        logger.info(
            "Polling Zulip for %s stream=%s topic=%s anchor=%s",
            user,
            stream,
            topic,
            self._last_message_id,
        )

        if not self._anchor_initialized:
            await self._initialize_anchor(base_url=base_url, user=user, api_key=api_key)
            if not self._anchor_initialized:
                return

        data = await asyncio.to_thread(
            self._request_json,
            method="GET",
            base_url=base_url,
            path="/api/v1/messages",
            user=user,
            api_key=api_key,
            params={
                "anchor": self._last_message_id,
                "num_before": 0,
                "num_after": 100,
                "apply_markdown": False,
                "client_gravatar": False,
                "narrow": json.dumps(self._build_narrow()),
                "include_anchor": False,
            },
        )
        if str(data.get("result")) != "success":
            logger.warning(
                "Zulip polling response error: %s",
                str(data.get("msg") or "unknown error"),
            )
            return
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            return
        main_messages = list(messages)
        unread_messages: list[dict[str, Any]] = []

        if self._should_run_unread_backfill(main_count=len(main_messages)):
            unread_messages = await self._fetch_unread_backfill(
                base_url=base_url,
                user=user,
                api_key=api_key,
            )
            if unread_messages:
                logger.info(
                    "Zulip unread backfill found=%d limit=%d",
                    len(unread_messages),
                    self._unread_backfill_limit,
                )
        messages = [*main_messages, *unread_messages]
        unread_backfill_ids = {
            int(item.get("id") or 0)
            for item in unread_messages
            if isinstance(item, dict)
        }

        bot_email = str(user).lower()
        highest_seen_id = self._last_message_id
        state_changed = False
        skipped_count = 0
        handled_count = 0
        skip_reason_counts: dict[str, int] = {}

        def _count_skip(reason: str) -> None:
            skip_reason_counts[reason] = int(skip_reason_counts.get(reason) or 0) + 1

        for item in messages:
            if not isinstance(item, dict):
                continue
            message_id = int(item.get("id") or 0)
            if message_id <= 0:
                continue
            highest_seen_id = max(highest_seen_id, message_id)
            if message_id in self._processed_message_ids:
                reason = "duplicate"
                self._log_skip(message_id=message_id, reason=reason)
                skipped_count += 1
                _count_skip(reason)
                continue
            sender_email = str(item.get("sender_email") or "").strip().lower()
            if not sender_email or sender_email == bot_email:
                reason = "self_or_missing_sender"
                self._log_skip(message_id=message_id, reason=reason)
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                skipped_count += 1
                _count_skip(reason)
                continue
            if self._is_read_message(item):
                reason = "read"
                self._log_skip(message_id=message_id, reason=reason)
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                skipped_count += 1
                _count_skip(reason)
                continue
            is_unread_backfill_item = message_id in unread_backfill_ids
            if (not is_unread_backfill_item) and self._is_historical_message(item):
                reason = "historical"
                self._log_skip(message_id=message_id, reason=reason)
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                skipped_count += 1
                _count_skip(reason)
                continue
            message_type = str(item.get("type") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not content:
                reason = "empty_content"
                self._log_skip(message_id=message_id, reason=reason)
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                skipped_count += 1
                _count_skip(reason)
                continue
            if message_type == "stream":
                stream_name = str(item.get("display_recipient") or "").strip()
                topic_name = str(item.get("subject") or item.get("topic") or "").strip()
                if not self._should_process_stream_message(
                    topic=topic_name,
                    content=content,
                    bot_user=user,
                ):
                    reason = "topic_mismatch"
                    self._log_skip(message_id=message_id, reason=reason)
                    state_changed = (
                        self._mark_processed_message(message_id) or state_changed
                    )
                    skipped_count += 1
                    _count_skip(reason)
                    continue
                chat_id = f"stream:{stream_name}:{topic_name}".rstrip(":")
                content = self._strip_bot_mentions(content, bot_user=user)
                metadata = {
                    "zulip_message_id": message_id,
                    "idempotency_key": f"zulip:{message_id}",
                    "zulip_stream": stream_name,
                    "zulip_topic": topic_name,
                    "conversation_type": "channel",
                    "is_dm": False,
                }
            else:
                recipients = []
                display = item.get("display_recipient")
                if isinstance(display, list):
                    for recipient in display:
                        if isinstance(recipient, dict):
                            email = str(recipient.get("email") or "").strip().lower()
                            if email:
                                recipients.append(email)
                if not recipients:
                    recipients = [sender_email]
                chat_id = f"pm:{','.join(sorted(set(recipients)))}"
                metadata = {
                    "zulip_message_id": message_id,
                    "idempotency_key": f"zulip:{message_id}",
                    "conversation_type": "dm",
                    "is_dm": True,
                    "zulip_recipients": recipients,
                }
            if not self.is_allowed(sender_email):
                reason = "allow_from_blocked"
                logger.warning(
                    "Channel %s: Message from %s blocked by allow_from list.",
                    self.name,
                    sender_email,
                )
                self._log_skip(message_id=message_id, reason=reason)
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                skipped_count += 1
                _count_skip(reason)
                continue
            handled = await self._handle_message(
                sender_id=sender_email,
                chat_id=chat_id,
                content=content,
                metadata=metadata,
            )
            if handled:
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                handled_count += 1
            else:
                reason = "filtered_or_rejected"
                self._log_skip(message_id=message_id, reason=reason)
                state_changed = (
                    self._mark_processed_message(message_id) or state_changed
                )
                skipped_count += 1
                _count_skip(reason)
        next_last = max(self._last_message_id, highest_seen_id)
        if next_last != self._last_message_id:
            self._last_message_id = next_last
            state_changed = True
        if state_changed:
            self._save_cursor_state()
        skip_reason_text = (
            ",".join(
                f"{reason}:{count}"
                for reason, count in sorted(skip_reason_counts.items())
            )
            if skip_reason_counts
            else "none"
        )
        logger.info(
            "Zulip poll complete user=%s fetched_main=%d fetched_unread=%d handled=%d skipped=%d skip_reasons=%s anchor=%s",
            user,
            len(main_messages),
            len(unread_messages),
            handled_count,
            skipped_count,
            skip_reason_text,
            self._last_message_id,
        )

    async def _send_via_api(self, msg: OutboundMessage) -> None:
        base_url = str(self._cfg("server_url", "")).strip().rstrip("/")
        user = str(self._cfg("user", "")).strip()
        api_key = str(self._cfg("api_key", "")).strip()
        if not (base_url and user and api_key):
            logger.warning("Zulip send skipped: missing server_url/user/api_key")
            return
        payload = self._build_send_payload(msg)
        if payload is None:
            logger.warning(
                "Zulip send skipped: unsupported chat target %s", msg.chat_id
            )
            return

        data = await asyncio.to_thread(
            self._request_json,
            method="POST",
            base_url=base_url,
            path="/api/v1/messages",
            user=user,
            api_key=api_key,
            form=payload,
        )
        if str(data.get("result")) != "success":
            logger.error("Zulip send failure: %s", str(data.get("msg") or "unknown"))

    def _build_send_payload(self, msg: OutboundMessage) -> Optional[dict[str, str]]:
        content = str(msg.content or "")
        chat = str(msg.chat_id or "").strip()
        if chat.startswith("stream:"):
            parts = chat.split(":", 2)
            if len(parts) < 3:
                return None
            stream = str(parts[1]).strip()
            topic = str(parts[2]).strip() or "annolid"
            if not stream:
                return None
            return {
                "type": "stream",
                "to": stream,
                "topic": topic,
                "content": content,
            }
        if chat.startswith("pm:"):
            recipients_raw = str(chat[3:] or "").strip()
            if not recipients_raw:
                return None
            recipients = [
                item.strip() for item in recipients_raw.split(",") if item.strip()
            ]
            if not recipients:
                return None
            if len(recipients) == 1:
                to_value = recipients[0]
            else:
                to_value = json.dumps(sorted(set(recipients)))
            return {
                "type": "private",
                "to": to_value,
                "content": content,
            }
        # Fallback: direct private message to chat_id (single recipient).
        if chat:
            return {
                "type": "private",
                "to": chat,
                "content": content,
            }
        return None

    def _build_narrow(self) -> list[dict[str, str]]:
        narrow: list[dict[str, str]] = []
        stream = str(self._cfg("stream", "")).strip()
        if stream:
            narrow.append({"operator": "stream", "operand": stream})
        return narrow

    def _build_unread_backfill_narrow(self) -> list[dict[str, str]]:
        narrow = self._build_narrow()
        narrow.append({"operator": "is", "operand": "unread"})
        return narrow

    def _should_run_unread_backfill(self, *, main_count: int) -> bool:
        if not self._unread_backfill_enabled:
            return False
        if self._unread_backfill_on_empty_only and main_count > 0:
            return False
        now = time.time()
        if (
            self._unread_backfill_cooldown_s > 0
            and (now - self._last_unread_backfill_at) < self._unread_backfill_cooldown_s
        ):
            return False
        self._last_unread_backfill_at = now
        return True

    async def _fetch_unread_backfill(
        self,
        *,
        base_url: str,
        user: str,
        api_key: str,
    ) -> list[dict[str, Any]]:
        data = await asyncio.to_thread(
            self._request_json,
            method="GET",
            base_url=base_url,
            path="/api/v1/messages",
            user=user,
            api_key=api_key,
            params={
                "anchor": "newest",
                "num_before": self._unread_backfill_limit,
                "num_after": 0,
                "apply_markdown": False,
                "client_gravatar": False,
                "narrow": json.dumps(self._build_unread_backfill_narrow()),
                "include_anchor": False,
            },
        )
        if str(data.get("result")) != "success":
            logger.warning(
                "Zulip unread backfill error: %s",
                str(data.get("msg") or "unknown error"),
            )
            return []
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            return []
        return [m for m in messages if isinstance(m, dict)]

    async def _initialize_anchor(
        self, *, base_url: str, user: str, api_key: str
    ) -> None:
        data = await asyncio.to_thread(
            self._request_json,
            method="GET",
            base_url=base_url,
            path="/api/v1/messages",
            user=user,
            api_key=api_key,
            params={
                "anchor": "newest",
                "num_before": 0,
                "num_after": 1,
                "apply_markdown": False,
                "client_gravatar": False,
                "narrow": json.dumps(self._build_narrow()),
                "include_anchor": True,
            },
        )
        if str(data.get("result")) != "success":
            logger.warning(
                "Zulip initial anchor sync failed: %s",
                str(data.get("msg") or "unknown error"),
            )
            return
        self._anchor_initialized = True
        messages = data.get("messages") or []
        if isinstance(messages, list):
            state_changed = False
            for item in messages:
                if isinstance(item, dict):
                    message_id = int(item.get("id") or 0)
                    if message_id > self._last_message_id:
                        self._last_message_id = message_id
                        state_changed = (
                            self._mark_processed_message(message_id) or state_changed
                        )
            if state_changed:
                self._save_cursor_state()

    def _request_json(
        self,
        *,
        method: str,
        base_url: str,
        path: str,
        user: str,
        api_key: str,
        params: Optional[dict[str, Any]] = None,
        form: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        token = _basic_auth_token(user, api_key)
        url = f"{base_url}{path}"
        if params:
            query = urllib.parse.urlencode(
                {
                    str(key): self._encode_query_value(value)
                    for key, value in params.items()
                }
            )
            url = f"{url}?{query}"
        body = None
        headers = {
            "Authorization": f"Basic {token}",
            "User-Agent": "annolid-zulip-channel/1.0",
        }
        if form is not None:
            body = urllib.parse.urlencode(form).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            with contextlib.suppress(Exception):
                payload = exc.read().decode("utf-8", errors="replace")
                data = json.loads(payload or "{}")
                if isinstance(data, dict):
                    return data
            return {"result": "error", "msg": f"HTTP {int(exc.code)}"}
        except Exception as exc:
            return {"result": "error", "msg": str(exc)}
        try:
            data = json.loads(raw or "{}")
        except Exception:
            return {"result": "error", "msg": "Invalid JSON response"}
        if not isinstance(data, dict):
            return {"result": "error", "msg": "Invalid JSON payload type"}
        return data

    @staticmethod
    def _encode_query_value(value: Any) -> str:
        if isinstance(value, bool):
            # Zulip expects JSON booleans in query params (true/false), not Python's
            # stringified booleans (True/False).
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _is_read_message(item: dict[str, Any]) -> bool:
        flags = item.get("flags")
        if not isinstance(flags, list):
            return False
        return "read" in {str(flag).strip().lower() for flag in flags}

    def _mark_processed_message(self, message_id: int) -> bool:
        if message_id <= 0 or message_id in self._processed_message_ids:
            return False
        self._processed_message_ids.add(message_id)
        self._processed_message_order.append(message_id)
        changed = True
        while len(self._processed_message_order) > self._max_processed_ids:
            old = self._processed_message_order.popleft()
            self._processed_message_ids.discard(old)
        return changed

    def _resolve_cursor_state_path(self) -> str:
        raw = str(self._cfg("cursor_state_path", "") or "").strip()
        if raw:
            return os.path.expanduser(raw)
        server_url = str(self._cfg("server_url", "") or "").strip().lower()
        user = str(self._cfg("user", "") or "").strip().lower()
        stream = str(self._cfg("stream", "") or "").strip().lower()
        topic = str(self._cfg("topic", "") or "").strip().lower()
        scope = "|".join((server_url, user, stream, topic))
        digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()[:12]
        return os.path.expanduser(
            f"~/.annolid/agent/channels/zulip_cursor_{digest}.json"
        )

    def _load_cursor_state_once(self) -> None:
        if self._cursor_state_loaded:
            return
        self._cursor_state_loaded = True
        self._load_cursor_state()

    def _load_cursor_state(self) -> None:
        path = self._cursor_state_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            last_message_id = int(data.get("last_message_id") or 0)
            if last_message_id > self._last_message_id:
                self._last_message_id = last_message_id
                self._cursor_state_has_checkpoint = True
            ids = data.get("processed_ids")
            if isinstance(ids, list):
                for value in ids[-self._max_processed_ids :]:
                    message_id = int(value or 0)
                    if message_id > 0:
                        self._mark_processed_message(message_id)
                        self._cursor_state_has_checkpoint = True
        except Exception as exc:
            logger.warning("Zulip cursor state load failed path=%s err=%s", path, exc)

    def _save_cursor_state(self) -> None:
        path = self._cursor_state_path
        if not path:
            return
        payload = {
            "last_message_id": int(self._last_message_id),
            "processed_ids": list(self._processed_message_order),
        }
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=directory or None,
                delete=False,
                prefix=".zulip_cursor_",
                suffix=".json",
            ) as tmp:
                json.dump(payload, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = tmp.name
            os.replace(tmp_path, path)
        except Exception as exc:
            with contextlib.suppress(Exception):
                if "tmp_path" in locals() and tmp_path:
                    os.unlink(tmp_path)
            logger.warning("Zulip cursor state save failed path=%s err=%s", path, exc)

    def _is_historical_message(self, item: dict[str, Any]) -> bool:
        # If we don't have a persisted checkpoint yet, ignore pre-start messages.
        if self._cursor_state_has_checkpoint:
            return False
        timestamp = int(item.get("timestamp") or 0)
        if timestamp <= 0:
            return False
        # Keep a small grace window for slight clock skew.
        return timestamp < (self._startup_unix_s - 5)

    def _log_skip(self, *, message_id: int, reason: str) -> None:
        if not self._log_skip_reasons:
            return
        logger.debug("Zulip skip message id=%s reason=%s", message_id, reason)

    def _should_process_stream_message(
        self,
        *,
        topic: str,
        content: str,
        bot_user: str,
    ) -> bool:
        configured_topic = str(self._cfg("topic", "") or "").strip()
        if not configured_topic:
            return True
        if topic.strip().lower() == configured_topic.lower():
            return True
        return self._contains_bot_mention(content, bot_user=bot_user)

    def _contains_bot_mention(self, content: str, *, bot_user: str) -> bool:
        text = str(content or "")
        if not text:
            return False
        lowered = text.lower()
        for candidate in self._bot_name_candidates(bot_user):
            if not candidate:
                continue
            name = candidate.lower()
            if f"@**{name}**" in lowered or f"@{name}" in lowered:
                return True
        return False

    def _strip_bot_mentions(self, content: str, *, bot_user: str) -> str:
        text = str(content or "")
        if not text:
            return ""
        cleaned = text
        for candidate in self._bot_name_candidates(bot_user):
            if not candidate:
                continue
            escaped = re.escape(candidate)
            cleaned = re.sub(
                rf"@(?:\*\*{escaped}\*\*|{escaped})",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        return str(cleaned).strip()

    def _bot_name_candidates(self, bot_user: str) -> list[str]:
        configured = str(self._cfg("bot_name", "") or "").strip()
        candidates: list[str] = []
        if configured:
            candidates.append(configured)
        local = str(bot_user or "").split("@", 1)[0].strip()
        if local:
            candidates.append(local)
            spaced = re.sub(r"[-_.]+", " ", local).strip()
            if spaced:
                candidates.append(spaced)
                candidates.append(spaced.title())
        seen: set[str] = set()
        unique: list[str] = []
        for item in candidates:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique


def _basic_auth_token(user: str, api_key: str) -> str:
    value = f"{user}:{api_key}".encode("utf-8", errors="ignore")

    return base64.b64encode(value).decode("ascii")
