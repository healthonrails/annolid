from __future__ import annotations

import asyncio
import json
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
        if isinstance(config, dict):
            self._phone_number_id = str(config.get("phone_number_id", "") or "")
            self._preview_url = bool(config.get("preview_url", False))
            self._access_token = str(config.get("access_token", "") or "")
            self._api_version = str(config.get("api_version", "v22.0") or "v22.0")
            self._verify_token = str(config.get("verify_token", "") or "")
            self._bridge_url = str(config.get("bridge_url", "") or "")
            self._bridge_token = str(config.get("bridge_token", "") or "")
            self._api_base = str(
                config.get("api_base", "https://graph.facebook.com")
                or "https://graph.facebook.com"
            )
            self._ingest_outgoing_messages = bool(
                config.get("ingest_outgoing_messages", False)
            )
        else:
            self._phone_number_id = str(getattr(config, "phone_number_id", "") or "")
            self._preview_url = bool(getattr(config, "preview_url", False))
            self._access_token = str(getattr(config, "access_token", "") or "")
            self._api_version = str(getattr(config, "api_version", "v22.0") or "v22.0")
            self._verify_token = str(getattr(config, "verify_token", "") or "")
            self._bridge_url = str(getattr(config, "bridge_url", "") or "")
            self._bridge_token = str(getattr(config, "bridge_token", "") or "")
            self._api_base = str(
                getattr(config, "api_base", "https://graph.facebook.com")
                or "https://graph.facebook.com"
            )
            self._ingest_outgoing_messages = bool(
                getattr(config, "ingest_outgoing_messages", False)
            )

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
        if self._send_callback is None:
            if self._bridge_url:
                await self._send_bridge_text(msg)
                return
            if self._access_token and self._phone_number_id:
                ok, code, detail = await asyncio.to_thread(
                    self._send_cloud_api_text, msg
                )
                if ok:
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
            else:
                logger.warning(
                    "WhatsApp outbound dropped: no callback, no bridge, and missing Cloud API credentials"
                )
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

    async def ingest_webhook_payload(self, payload: dict[str, Any]) -> int:
        """Parse WhatsApp Cloud API webhook payload and ingest message events."""
        if not isinstance(payload, dict):
            logger.warning("WhatsApp webhook payload ignored: not a JSON object")
            return 0

        ingested = 0
        entries = payload.get("entry", [])
        logger.info(
            "WhatsApp webhook ingest begin object=%s entries=%s",
            payload.get("object"),
            len(entries) if isinstance(entries, list) else 0,
        )
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes", []):
                if not isinstance(change, dict):
                    continue
                field_name = str(change.get("field") or "").strip().lower()
                if field_name != "messages":
                    logger.debug(
                        "WhatsApp webhook change skipped field=%s",
                        field_name,
                    )
                    continue
                value = change.get("value")
                if not isinstance(value, dict):
                    logger.warning("WhatsApp webhook change value is not object")
                    continue
                metadata_block = value.get("metadata")
                phone_number_id = ""
                if isinstance(metadata_block, dict):
                    phone_number_id = str(
                        metadata_block.get("phone_number_id", "") or ""
                    ).strip()
                contacts = value.get("contacts")
                profile_name_by_waid: dict[str, str] = {}
                if isinstance(contacts, list):
                    for contact in contacts:
                        if not isinstance(contact, dict):
                            continue
                        waid = str(contact.get("wa_id", "") or "").strip()
                        if not waid:
                            continue
                        profile = contact.get("profile")
                        if isinstance(profile, dict):
                            profile_name_by_waid[waid] = str(
                                profile.get("name", "") or ""
                            ).strip()
                messages = value.get("messages", [])
                if not isinstance(messages, list) or not messages:
                    logger.info(
                        "WhatsApp webhook has no inbound messages (statuses=%s)",
                        len(value.get("statuses", []) or []),
                    )
                for message in messages if isinstance(messages, list) else []:
                    parsed = self._parse_incoming_message(
                        message=message,
                        default_phone_number_id=phone_number_id,
                        profile_name_by_waid=profile_name_by_waid,
                    )
                    if parsed is None:
                        logger.debug(
                            "WhatsApp webhook message skipped type=%s",
                            (message or {}).get("type")
                            if isinstance(message, dict)
                            else None,
                        )
                        continue
                    ok = await self.ingest(**parsed)
                    if ok:
                        ingested += 1
                        logger.info(
                            "WhatsApp webhook message accepted sender=%s",
                            parsed.get("sender_id"),
                        )
                    else:
                        logger.warning(
                            "WhatsApp webhook message rejected sender=%s",
                            parsed.get("sender_id"),
                        )
        logger.info("WhatsApp webhook ingest complete ingested=%s", ingested)
        return ingested

    def build_cloud_api_payload(self, msg: OutboundMessage) -> dict[str, Any]:
        """Build a WhatsApp Cloud API text payload from an outbound bus message."""
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
        """Validate Meta webhook verification handshake and return challenge."""
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
        """Send outbound message via WhatsApp Cloud API."""
        if not self._access_token:
            return False, 0, "Missing WhatsApp access token"
        if not self._phone_number_id:
            return False, 0, "Missing WhatsApp phone_number_id"
        payload = self.build_cloud_api_payload(msg)
        api_base = str(self._api_base or "https://graph.facebook.com").rstrip("/")
        version = str(self._api_version or "v22.0").strip() or "v22.0"
        url = f"{api_base}/{version}/{self._phone_number_id}/messages"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
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
        profile_name_by_waid: dict[str, str],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(message, dict):
            return None
        sender_id = str(message.get("from", "") or "").strip()
        message_id = str(message.get("id", "") or "").strip()
        if not sender_id:
            return None
        msg_type = str(message.get("type", "text") or "text").strip().lower()
        content = self._extract_message_content(message, msg_type)
        if not content:
            return None
        metadata: dict[str, Any] = {"conversation_type": "dm", "is_dm": True}
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
        media = self._extract_media_ids(message, msg_type)
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
            reply_block = interactive.get("button_reply")
            if isinstance(reply_block, dict):
                title = str(reply_block.get("title", "") or "").strip()
                if title:
                    return title
            reply_block = interactive.get("list_reply")
            if isinstance(reply_block, dict):
                title = str(reply_block.get("title", "") or "").strip()
                if title:
                    return title

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
            logger.error(
                "WhatsApp bridge mode requires `websockets` package: %s",
                exc,
            )
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
            content = str(data.get("content", "") or "").strip()
            sender = str(data.get("pn") or data.get("sender") or "").strip()
            if not content or not sender:
                return
            sender_id = sender.split("@", 1)[0] if "@" in sender else sender
            chat_id = str(data.get("chat_id") or sender).strip()
            if not chat_id:
                chat_id = sender
            is_group = bool(data.get("isGroup", False))
            direction = str(data.get("direction", "in") or "in").strip().lower()
            if direction == "out" and not self._ingest_outgoing_messages:
                logger.info(
                    "WhatsApp bridge outgoing message ignored by default sender=%s chat_id=%s",
                    sender_id,
                    chat_id,
                )
                return
            metadata = {
                "message_id": str(data.get("id", "") or ""),
                "timestamp": data.get("timestamp"),
                "is_group": is_group,
                "conversation_type": "group" if is_group else "dm",
                "is_dm": not is_group,
                "bridge_direction": direction,
            }
            logger.info(
                "WhatsApp bridge message detected direction=%s sender=%s chat_id=%s",
                direction,
                sender_id,
                chat_id,
            )
            accepted = await self.ingest(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                metadata=metadata,
            )
            if not accepted:
                logger.warning(
                    "WhatsApp bridge message rejected by allow_from sender=%s chat_id=%s",
                    sender_id,
                    chat_id,
                )
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
