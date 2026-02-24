from __future__ import annotations

import asyncio
import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage
from annolid.utils.logger import logger

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


class WhatsAppChannel(BaseChannel):
    """Dependency-light WhatsApp adapter for bus integration."""

    name = "whatsapp"

    SELF_LABELS = {"message yourself", "self", "me", "you"}
    MAX_BRIDGE_CONTENT_CHARS = 16_000
    MAX_MEDIA_ITEMS = 8
    MAX_MEDIA_REF_CHARS = 4_096

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
        self._bridge_ws = None

        cfg = config if isinstance(config, dict) else None
        self._phone_number_id = self._cfg_str(cfg, "phone_number_id", "")
        self._preview_url = self._cfg_bool(cfg, "preview_url", False)
        self._access_token = self._cfg_str(cfg, "access_token", "")
        self._api_version = self._cfg_str(cfg, "api_version", "v22.0")
        self._verify_token = self._cfg_str(cfg, "verify_token", "")
        self._bridge_url = self._cfg_str(cfg, "bridge_url", "")
        self._bridge_token = self._cfg_str(cfg, "bridge_token", "")
        self._api_base = self._cfg_str(
            cfg,
            "api_base",
            "https://graph.facebook.com",
        )
        self._ingest_outgoing_messages = self._cfg_bool(
            cfg, "ingest_outgoing_messages", False
        )
        self._ingest_self_messages = self._cfg_bool(cfg, "ingest_self_messages", False)

        # Best-effort echo guard for bridge events that mirror recently sent text.
        self._recent_sent_contents: list[tuple[float, str]] = []

    def _cfg_str(self, cfg: Optional[dict[str, Any]], key: str, default: str) -> str:
        if cfg is not None:
            return str(cfg.get(key, default) or default)
        return str(getattr(self.config, key, default) or default)

    def _cfg_bool(self, cfg: Optional[dict[str, Any]], key: str, default: bool) -> bool:
        if cfg is not None:
            return bool(cfg.get(key, default))
        return bool(getattr(self.config, key, default))

    async def start(self) -> None:
        self._running = True
        self._stop_event.clear()
        if self._bridge_url:
            await self._run_bridge_loop()
            return
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        ws = self._bridge_ws
        self._bridge_ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                logger.debug("Failed closing WhatsApp bridge socket", exc_info=True)

    async def send(self, msg: OutboundMessage) -> None:
        if self._send_callback is not None:
            ret = self._send_callback(msg)
            if asyncio.iscoroutine(ret):
                await ret
            self._remember_sent_content(msg.content)
            return

        if self._bridge_url:
            sendable_media = self._filter_sendable_bridge_media(msg.media or [])
            if sendable_media:
                await self._send_bridge_media(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=msg.content,
                        reply_to=msg.reply_to,
                        media=sendable_media,
                        metadata=dict(msg.metadata or {}),
                    )
                )
            elif str(msg.content or "").strip():
                await self._send_bridge_text(msg)
            else:
                logger.warning(
                    "WhatsApp bridge outbound dropped: media refs are not local files and no fallback text was provided"
                )
            self._remember_sent_content(msg.content)
            return

        if self._access_token and self._phone_number_id:
            ok, code, detail = await asyncio.to_thread(self._send_cloud_api_text, msg)
            if ok:
                self._remember_sent_content(msg.content)
                logger.info(
                    "WhatsApp Cloud API outbound sent chat_id=%s status=%s",
                    msg.chat_id,
                    code,
                )
            else:
                logger.error(
                    "WhatsApp Cloud API outbound failed chat_id=%s status=%s detail=%s",
                    msg.chat_id,
                    code,
                    detail,
                )
            return

        logger.warning(
            "WhatsApp outbound dropped: no callback, no bridge, and missing Cloud API credentials"
        )

    def is_allowed(self, sender_id: str) -> bool:
        """Permit explicit self-label senders even when allow_from is configured."""
        sender = str(sender_id or "").strip().lower()
        if sender in self.SELF_LABELS or "(you)" in sender:
            return True
        return super().is_allowed(sender_id)

    def _remember_sent_content(self, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        now = asyncio.get_running_loop().time()
        self._recent_sent_contents.append((now, text))
        cutoff = now - 30.0
        self._recent_sent_contents = [
            (ts, c) for ts, c in self._recent_sent_contents if ts >= cutoff
        ]

    def _is_recent_echo(self, content: str) -> bool:
        text = str(content or "").strip()
        if not text:
            return False
        now = asyncio.get_running_loop().time()
        for ts, sent in self._recent_sent_contents:
            if sent == text and now - ts <= 15.0:
                return True
        return False

    @staticmethod
    def _filter_sendable_bridge_media(media: list[str]) -> list[str]:
        sendable: list[str] = []
        for item in media:
            candidate = str(item or "").strip()
            if not candidate:
                continue
            # Ignore virtual bridge refs such as wa-bridge-media:image:...
            if candidate.startswith("wa-bridge-media:") or candidate.startswith(
                "wa-media:"
            ):
                continue
            path_obj = Path(candidate).expanduser()
            try:
                if path_obj.exists() and path_obj.is_file():
                    sendable.append(str(path_obj.resolve()))
            except Exception:
                continue
        return sendable

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

    async def ingest_webhook_payload(self, payload: dict[str, Any]) -> int:
        """Parse WhatsApp Cloud API webhook payload and ingest message events."""
        if not isinstance(payload, dict):
            logger.warning("WhatsApp webhook payload ignored: not a JSON object")
            return 0

        entries = payload.get("entry", [])
        ingested = 0
        logger.info(
            "WhatsApp webhook ingest begin object=%s entries=%s",
            payload.get("object"),
            len(entries) if isinstance(entries, list) else 0,
        )

        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            ingested += await self._ingest_webhook_entry(entry)

        logger.info("WhatsApp webhook ingest complete ingested=%s", ingested)
        return ingested

    async def _ingest_webhook_entry(self, entry: dict[str, Any]) -> int:
        ingested = 0
        for change in entry.get("changes", []):
            if not isinstance(change, dict):
                continue
            field_name = str(change.get("field") or "").strip().lower()
            if field_name != "messages":
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue

            phone_number_id, display_phone_number = self._webhook_metadata(value)
            profile_name_by_waid = self._webhook_profile_map(value)
            messages = value.get("messages", [])
            for message in messages if isinstance(messages, list) else []:
                parsed = self._parse_incoming_message(
                    message=message,
                    default_phone_number_id=phone_number_id,
                    default_display_phone_number=display_phone_number,
                    profile_name_by_waid=profile_name_by_waid,
                )
                if parsed is None:
                    continue
                if await self.ingest(**parsed):
                    ingested += 1
        return ingested

    @staticmethod
    def _webhook_metadata(value: dict[str, Any]) -> tuple[str, str]:
        metadata_block = value.get("metadata")
        if not isinstance(metadata_block, dict):
            return "", ""
        phone_number_id = str(metadata_block.get("phone_number_id", "") or "").strip()
        display_phone_number = str(
            metadata_block.get("display_phone_number", "") or ""
        ).strip()
        return phone_number_id, display_phone_number

    @staticmethod
    def _webhook_profile_map(value: dict[str, Any]) -> dict[str, str]:
        profile_name_by_waid: dict[str, str] = {}
        contacts = value.get("contacts")
        if not isinstance(contacts, list):
            return profile_name_by_waid
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            waid = str(contact.get("wa_id", "") or "").strip()
            if not waid:
                continue
            profile = contact.get("profile")
            if isinstance(profile, dict):
                profile_name_by_waid[waid] = str(profile.get("name", "") or "").strip()
        return profile_name_by_waid

    def build_cloud_api_payload(self, msg: OutboundMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": str(msg.chat_id),
            "type": "text",
            "text": {
                "preview_url": bool(self._preview_url),
                "body": str(msg.content or ""),
            },
        }
        if self._phone_number_id:
            payload["phone_number_id"] = self._phone_number_id
        if msg.reply_to:
            payload["context"] = {"message_id": str(msg.reply_to)}
        return payload

    def verify_webhook_challenge(
        self,
        *,
        mode: str,
        verify_token: str,
        challenge: str,
    ) -> Optional[str]:
        if str(mode).strip() != "subscribe":
            return None
        expected = str(self._verify_token or "").strip()
        if not expected:
            return None
        if str(verify_token or "").strip() != expected:
            return None
        return str(challenge or "")

    def _send_cloud_api_text(
        self,
        msg: OutboundMessage,
        *,
        timeout_s: float = 15.0,
    ) -> tuple[bool, int, str]:
        if not self._access_token:
            return False, 0, "Missing WhatsApp access token"
        if not self._phone_number_id:
            return False, 0, "Missing WhatsApp phone_number_id"

        payload = self.build_cloud_api_payload(msg)
        api_base = str(self._api_base or "https://graph.facebook.com").rstrip("/")
        version = str(self._api_version or "v22.0").strip() or "v22.0"
        url = f"{api_base}/{version}/{self._phone_number_id}/messages"

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
                return True, int(getattr(response, "status", 200) or 200), body
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return False, int(exc.code or 0), body
        except Exception as exc:
            return False, 0, str(exc)

    def _parse_incoming_message(
        self,
        *,
        message: Any,
        default_phone_number_id: str,
        default_display_phone_number: str,
        profile_name_by_waid: dict[str, str],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(message, dict):
            return None

        sender_id = str(message.get("from", "") or "").strip()
        if not sender_id:
            return None

        if self._is_outgoing_message(message) and not self._ingest_outgoing_messages:
            return None
        if self._is_self_sender(sender_id, default_display_phone_number):
            if not self._ingest_self_messages:
                return None

        msg_type = str(message.get("type", "text") or "text").strip().lower()
        content = self._extract_message_content(message, msg_type)
        media = self._extract_media_ids(message, msg_type)
        if not content and not media:
            return None

        message_id = str(message.get("id", "") or "").strip()
        metadata: dict[str, Any] = {"conversation_type": "dm", "is_dm": True}
        if msg_type:
            metadata["whatsapp_message_type"] = msg_type
        if message_id:
            metadata["whatsapp_message_id"] = message_id
            metadata["idempotency_key"] = message_id
        if default_phone_number_id:
            metadata["phone_number_id"] = default_phone_number_id

        profile_name = profile_name_by_waid.get(sender_id, "")
        if profile_name:
            metadata["profile_name"] = profile_name

        context_block = message.get("context")
        if isinstance(context_block, dict):
            reply_to = str(context_block.get("id", "") or "").strip()
            if reply_to:
                metadata["reply_to"] = reply_to

        if not content and media:
            content = "[Media message]"
        if media:
            metadata["has_media"] = True

        return {
            "sender_id": sender_id,
            "chat_id": sender_id,
            "content": content,
            "media": media,
            "metadata": metadata,
        }

    def _extract_message_content(self, message: dict[str, Any], msg_type: str) -> str:
        text_block = message.get("text")
        if isinstance(text_block, dict):
            body = str(text_block.get("body", "") or "").strip()
            if body:
                return body

        if msg_type == "button":
            button = message.get("button")
            if isinstance(button, dict):
                label = str(button.get("text", "") or "").strip()
                if label:
                    return label

        interactive = message.get("interactive")
        if isinstance(interactive, dict):
            button_reply = interactive.get("button_reply")
            if isinstance(button_reply, dict):
                title = str(button_reply.get("title", "") or "").strip()
                if title:
                    return title
            list_reply = interactive.get("list_reply")
            if isinstance(list_reply, dict):
                title = str(list_reply.get("title", "") or "").strip()
                if title:
                    return title

        # Some media message types include captions in their own object.
        media_block = message.get(msg_type)
        if isinstance(media_block, dict):
            caption = str(media_block.get("caption", "") or "").strip()
            if caption:
                return caption

        return ""

    def _extract_media_ids(self, message: dict[str, Any], msg_type: str) -> list[str]:
        media_block = message.get(msg_type)
        if not isinstance(media_block, dict):
            return []
        media_id = str(media_block.get("id", "") or "").strip()
        if not media_id:
            return []
        return [f"wa-media:{media_id}"]

    async def _run_bridge_loop(self) -> None:
        try:
            import websockets
        except Exception as exc:
            logger.error("WhatsApp bridge mode requires `websockets` package: %s", exc)
            await self._stop_event.wait()
            return

        while self._running:
            try:
                async with websockets.connect(self._bridge_url) as ws:
                    self._bridge_ws = ws
                    logger.info("Connected to WhatsApp bridge: %s", self._bridge_url)
                    if self._bridge_token:
                        await ws.send(
                            json.dumps({"type": "auth", "token": self._bridge_token})
                        )
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle_bridge_message(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WhatsApp bridge connection error: %s", exc)
                if self._running:
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        pass
            finally:
                self._bridge_ws = None

    async def _send_bridge_text(self, msg: OutboundMessage) -> None:
        ws = self._bridge_ws
        if ws is None:
            logger.warning("WhatsApp bridge not connected; dropping outbound message")
            return
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "send",
                        "to": str(msg.chat_id),
                        "text": str(msg.content or ""),
                    }
                )
            )
        except Exception as exc:
            logger.error("WhatsApp bridge send error: %s", exc)

    async def _send_bridge_media(self, msg: OutboundMessage) -> None:
        ws = self._bridge_ws
        if ws is None:
            logger.warning(
                "WhatsApp bridge not connected; dropping outbound media message"
            )
            return
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "send_media",
                        "to": str(msg.chat_id),
                        "media": list(msg.media or []),
                        "caption": str(msg.content or ""),
                    }
                )
            )
        except Exception as exc:
            logger.error("WhatsApp bridge media send error: %s", exc)

    async def _handle_bridge_message(self, raw: Any) -> None:
        try:
            data = json.loads(str(raw))
        except Exception:
            logger.warning("Invalid WhatsApp bridge payload: %r", raw)
            return
        if not isinstance(data, dict):
            return

        msg_type = str(data.get("type", "") or "").strip().lower()
        if msg_type == "message":
            parsed = self._parse_bridge_message(data)
            if parsed is None:
                return
            await self.ingest(**parsed)
            return

        if msg_type == "qr":
            qr_payload = str(data.get("qr", "") or "").strip()
            if qr_payload and qr_payload != "open_browser_and_scan":
                logger.info(
                    "WhatsApp bridge QR payload received. If browser is not visible, set bridge_headless=false."
                )
                self._log_qr_ascii(qr_payload)
            else:
                logger.info(
                    "WhatsApp bridge is waiting for QR scan. Check the WhatsApp Web browser window."
                )
            return

        if msg_type == "status":
            logger.info("WhatsApp bridge status: %s", data.get("status"))
            return

        if msg_type == "error":
            logger.error("WhatsApp bridge error: %s", data.get("error"))

    def _parse_bridge_message(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        content = str(data.get("content", "") or "").strip()
        if content and len(content) > self.MAX_BRIDGE_CONTENT_CHARS:
            content = content[: self.MAX_BRIDGE_CONTENT_CHARS]

        sender = str(data.get("pn") or data.get("sender") or "").strip()
        if not sender:
            return None

        sender_id = sender.split("@", 1)[0] if "@" in sender else sender
        chat_id = str(data.get("chat_id") or sender).strip() or sender
        is_group = bool(data.get("isGroup", False))
        direction = str(data.get("direction", "in") or "in").strip().lower()
        is_outgoing = self._is_outgoing_message(data)
        message_id = str(data.get("id", "") or "").strip()
        media = self._extract_bridge_media(data, message_id=message_id)

        if not content and media:
            media_type = str(data.get("media_type", "") or "").strip().lower()
            content = f"[{media_type} message]" if media_type else "[Media message]"

        if is_outgoing and not self._ingest_outgoing_messages:
            return None
        # In shared/self chats, both user prompts and bot replies can appear as outgoing.
        # Keep outgoing ingest opt-in, then suppress only recent bot echoes.
        if is_outgoing and self._is_recent_echo(content):
            return None
        if not content:
            return None

        metadata = {
            "message_id": message_id,
            "timestamp": data.get("timestamp"),
            "is_group": is_group,
            "conversation_type": "group" if is_group else "dm",
            "is_dm": not is_group,
            "bridge_direction": direction,
        }
        media_type = str(data.get("media_type", "") or "").strip().lower()
        if media:
            metadata["has_media"] = True
        if media_type:
            metadata["media_type"] = media_type
            metadata["whatsapp_message_type"] = media_type
        return {
            "sender_id": sender_id,
            "chat_id": chat_id,
            "content": content,
            "media": media,
            "metadata": metadata,
        }

    def _extract_bridge_media(
        self, data: dict[str, Any], *, message_id: str
    ) -> list[str]:
        raw_media = data.get("media")
        refs: list[str] = []
        if isinstance(raw_media, str):
            raw_media = [raw_media]
        if isinstance(raw_media, list):
            for item in raw_media:
                if len(refs) >= self.MAX_MEDIA_ITEMS:
                    break
                if isinstance(item, str):
                    ref = item.strip()
                elif isinstance(item, dict):
                    ref = str(item.get("id") or item.get("url") or "").strip()
                else:
                    ref = ""
                if not ref:
                    continue
                refs.append(ref[: self.MAX_MEDIA_REF_CHARS])
        if refs:
            return refs

        media_type = str(data.get("media_type", "") or "").strip().lower()
        if media_type and message_id:
            return [f"wa-bridge-media:{media_type}:{message_id}"]
        return []

    @staticmethod
    def _is_outgoing_message(message: dict[str, Any]) -> bool:
        flags = (
            message.get("from_me"),
            message.get("fromMe"),
            message.get("is_echo"),
            message.get("isEcho"),
            message.get("echo"),
        )
        if any(bool(flag) for flag in flags):
            return True
        direction = str(message.get("direction", "") or "").strip().lower()
        return direction in {"out", "outbound", "sent"}

    @staticmethod
    def _is_self_sender(sender_id: str, self_id: str) -> bool:
        sender_norm = WhatsAppChannel._normalize_wa_identity(sender_id)
        self_norm = WhatsAppChannel._normalize_wa_identity(self_id)
        return bool(sender_norm and self_norm and sender_norm == self_norm)

    @staticmethod
    def _normalize_wa_identity(value: str) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        if "@" in raw:
            raw = raw.split("@", 1)[0]
        return "".join(ch for ch in raw if ch.isdigit()) or raw

    @classmethod
    def _is_self_chat_label(cls, sender_id: str, chat_id: str) -> bool:
        candidates = (str(sender_id or "").lower(), str(chat_id or "").lower())
        for text in candidates:
            if not text:
                continue
            if text in cls.SELF_LABELS or "(you)" in text:
                return True
        return False

    def _log_qr_ascii(self, qr_payload: str) -> None:
        """Best-effort terminal QR rendering for bridge login."""
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(qr_payload)
            qr.make(fit=True)

            import io

            buf = io.StringIO()
            qr.print_ascii(out=buf, invert=True)
            logger.info("Scan this WhatsApp QR in your terminal:\n%s", buf.getvalue())
        except Exception:
            logger.info(
                "QR payload (install optional `qrcode` package for ASCII rendering): %s",
                qr_payload,
            )
