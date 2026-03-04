from __future__ import annotations

import asyncio
import base64
import contextlib
import json
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

    async def start(self) -> None:
        if self._running:
            return
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
        base_url = str(self._cfg("server_url", "")).strip().rstrip("/")
        user = str(self._cfg("user", "")).strip()
        api_key = str(self._cfg("api_key", "")).strip()
        if not (base_url and user and api_key):
            logger.debug("Zulip polling skipped: missing server_url/user/api_key")
            return

        if not self._anchor_initialized:
            await self._initialize_anchor(base_url=base_url, user=user, api_key=api_key)
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

        bot_email = str(user).lower()
        for item in messages:
            if not isinstance(item, dict):
                continue
            message_id = int(item.get("id") or 0)
            self._last_message_id = max(self._last_message_id, message_id)
            sender_email = str(item.get("sender_email") or "").strip().lower()
            if not sender_email or sender_email == bot_email:
                continue
            message_type = str(item.get("type") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if message_type == "stream":
                stream_name = str(item.get("display_recipient") or "").strip()
                topic_name = str(item.get("subject") or item.get("topic") or "").strip()
                chat_id = f"stream:{stream_name}:{topic_name}".rstrip(":")
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
            await self._handle_message(
                sender_id=sender_email,
                chat_id=chat_id,
                content=content,
                metadata=metadata,
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
        topic = str(self._cfg("topic", "")).strip()
        if stream:
            narrow.append({"operator": "stream", "operand": stream})
        if topic:
            narrow.append({"operator": "topic", "operand": topic})
        return narrow

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
        self._anchor_initialized = True
        if str(data.get("result")) != "success":
            logger.warning(
                "Zulip initial anchor sync failed: %s",
                str(data.get("msg") or "unknown error"),
            )
            return
        messages = data.get("messages") or []
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, dict):
                    message_id = int(item.get("id") or 0)
                    if message_id > self._last_message_id:
                        self._last_message_id = message_id

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


def _basic_auth_token(user: str, api_key: str) -> str:
    value = f"{user}:{api_key}".encode("utf-8", errors="ignore")

    return base64.b64encode(value).decode("ascii")
