import asyncio
import email
import imaplib
import os
import smtplib
from email.utils import parseaddr
from email.message import EmailMessage
from typing import Any, Awaitable, Callable, Optional

from annolid.core.agent.bus import OutboundMessage
from annolid.utils.logger import logger

from .base import BaseChannel

SendCallback = Callable[[OutboundMessage], Awaitable[None] | None]


class EmailChannel(BaseChannel):
    """Email adapter for bus integration with SMTP and IMAP support."""

    name = "email"

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

    async def _poll_loop(self) -> None:
        last_interval: Optional[int] = None
        while self._running:
            interval = self._resolve_poll_interval_seconds()
            if last_interval != interval:
                logger.info("Email IMAP poll interval set to %ss", interval)
                last_interval = interval
            try:
                await self._poll_imap()
            except Exception as exc:
                logger.error("IMAP polling failure: %s", exc)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                continue

    def _resolve_poll_interval_seconds(self) -> int:
        raw_cfg = self.config.get("polling_interval", 300)
        env_raw = (
            os.getenv("ANNOLID_EMAIL_POLL_INTERVAL_SECONDS")
            or os.getenv("NANOBOT_EMAIL_POLL_INTERVAL_SECONDS")
            or ""
        ).strip()
        raw = env_raw if env_raw else raw_cfg
        try:
            seconds = int(raw)
        except (TypeError, ValueError):
            seconds = 300
        # Keep a sane lower bound to avoid hammering IMAP servers.
        return max(10, seconds)

    async def _poll_imap(self) -> None:
        imap_host = str(self.config.get("imap_host") or "").strip()
        imap_port = int(self.config.get("imap_port", 993))
        user = str(self.config.get("user") or "").strip()
        password = str(self.config.get("password") or "").replace(" ", "").strip()

        if not all([imap_host, user, password]):
            logger.warning("IMAP not configured for background polling.")
            return

        logger.info("Polling IMAP for %s...", user)

        def _sync_poll():
            messages = []
            try:
                mail = imaplib.IMAP4_SSL(imap_host, imap_port)
                mail.login(user, password)
                mail.select("inbox")
                status, response = mail.search(None, "UNSEEN")
                if status != "OK":
                    return []

                for num in response[0].split():
                    status, msg_data = mail.fetch(num, "(RFC822)")
                    if status != "OK":
                        continue

                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    sender = msg.get("From")
                    subject = msg.get("Subject")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()

                    messages.append(
                        {"sender": sender, "subject": subject, "body": body, "num": num}
                    )
                    # Mark as seen
                    mail.store(num, "+FLAGS", "\\Seen")

                mail.close()
                mail.logout()
            except Exception as e:
                logger.error("IMAP sync poll exception: %s", e)
            return messages

        new_emails = await asyncio.to_thread(_sync_poll)
        if new_emails:
            logger.info("Found %d new email(s).", len(new_emails))

        for em in new_emails:
            logger.info("Ingesting email from %s: %s", em["sender"], em["subject"])
            await self.ingest(
                sender_email=em["sender"], subject=em["subject"], body=em["body"]
            )

    async def send(self, msg: OutboundMessage) -> None:
        content = str(msg.content or "").strip()
        if not content:
            logger.warning("Skipping empty email reply to %s", msg.chat_id)
            return
        if self._send_callback:
            ret = self._send_callback(msg)
            if asyncio.iscoroutine(ret):
                await ret
            return

        smtp_host = str(self.config.get("smtp_host") or "").strip()
        smtp_port = int(self.config.get("smtp_port", 587))
        user = str(self.config.get("user") or "").strip()
        password = str(self.config.get("password") or "").replace(" ", "").strip()

        if not all([smtp_host, user, password]):
            logger.warning("SMTP not configured, cannot send email")
            return

        async def _async_send():
            email_msg = EmailMessage()
            email_msg.set_content(content)

            orig_subject = msg.metadata.get("subject")
            if not orig_subject:
                subject = "Message from Annolid"
            elif not str(orig_subject).lower().startswith("re:"):
                subject = f"Re: {orig_subject}"
            else:
                subject = str(orig_subject)

            email_msg["Subject"] = subject
            email_msg["From"] = user
            email_msg["To"] = msg.chat_id

            def _sync_send():
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(email_msg)

            await asyncio.to_thread(_sync_send)

        try:
            await _async_send()
        except Exception as exc:
            logger.error("SMTP send failure: %s", exc)

    async def ingest(
        self,
        *,
        sender_email: str,
        subject: str,
        body: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        raw_from = str(sender_email or "").strip()
        normalized_sender = self._normalize_sender_email(raw_from)
        content = f"Email received.\nFrom: {raw_from}\nSubject: {subject}\n\n{body}"
        merged = dict(metadata or {})
        merged.setdefault("subject", subject)
        merged.setdefault("sender_email", normalized_sender)
        merged.setdefault("raw_from", raw_from)
        return await self._handle_message(
            sender_id=normalized_sender,
            chat_id=normalized_sender,
            content=content,
            metadata=merged,
        )

    def _normalize_sender_email(self, raw_from: str) -> str:
        _, addr = parseaddr(str(raw_from or ""))
        normalized = str(addr or "").strip().lower()
        if normalized:
            return normalized
        return str(raw_from or "").strip().lower()
