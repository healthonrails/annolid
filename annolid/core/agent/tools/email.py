from __future__ import annotations

import asyncio
import email
import imaplib
import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from .function_base import FunctionTool


class EmailTool(FunctionTool):
    """Tool for sending emails."""

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        imap_host: str = "",
        imap_port: int = 993,
        user: str = "",
        password: str = "",
        allowed_attachment_roots: list[str | Path] | None = None,
    ):
        self._smtp_host = str(smtp_host or "").strip()
        self._smtp_port = int(smtp_port or 587)
        self._imap_host = str(imap_host or "").strip()
        self._imap_port = int(imap_port or 993)
        self._user = str(user or "").strip()
        # Strip spaces from password too, as Google app passwords often
        # come with spaces when copy-pasted.
        self._password = str(password or "").replace(" ", "").strip()
        self._allowed_attachment_roots = [
            Path(root).expanduser().resolve()
            for root in (allowed_attachment_roots or [])
            if str(root).strip()
        ]

    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        return "Send an email to a recipient, optionally with file attachments."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "content": {"type": "string", "description": "Email body content"},
                "attachment_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional local file paths to attach",
                },
            },
            "required": ["to", "content"],
        }

    async def execute(
        self,
        to: str,
        content: str,
        subject: str = "Message from Annolid Bot",
        attachment_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if not all([self._smtp_host, self._user, self._password]):
            missing = []
            if not self._smtp_host:
                missing.append("smtp_host")
            if not self._user:
                missing.append("user")
            if not self._password:
                missing.append("password")
            return f"Error: Email tool not configured. Missing: {', '.join(missing)}"

        email_msg = EmailMessage()
        email_msg.set_content(content)
        email_msg["Subject"] = subject
        email_msg["From"] = self._user
        email_msg["To"] = to

        attachments = self._normalize_attachment_paths(attachment_paths)
        for attachment in attachments:
            valid_path, error = self._resolve_attachment_path(attachment)
            if valid_path is None:
                return f"Error sending email: {error}"
            try:
                payload = valid_path.read_bytes()
            except Exception as exc:
                return f"Error sending email: failed to read attachment {valid_path}: {exc}"
            mime_type, _encoding = mimetypes.guess_type(str(valid_path))
            if mime_type:
                maintype, subtype = mime_type.split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"
            email_msg.add_attachment(
                payload,
                maintype=maintype,
                subtype=subtype,
                filename=valid_path.name,
            )

        def _sync_send():
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                server.login(self._user, self._password)
                server.send_message(email_msg)

        try:
            await asyncio.to_thread(_sync_send)
            if attachments:
                return (
                    f"Email successfully sent to {to} "
                    f"with {len(attachments)} attachment(s)"
                )
            return f"Email successfully sent to {to}"
        except Exception as exc:
            return f"Error sending email: {exc}"

    @staticmethod
    def _normalize_attachment_paths(paths: list[str] | None) -> list[str]:
        if not isinstance(paths, list):
            return []
        normalized: list[str] = []
        for item in paths:
            path_text = str(item or "").strip()
            if path_text:
                normalized.append(path_text)
        return normalized

    def _resolve_attachment_path(self, raw_path: str) -> tuple[Path | None, str]:
        candidate = Path(raw_path).expanduser()
        try:
            resolved = candidate.resolve()
        except Exception as exc:
            return None, f"invalid attachment path {raw_path}: {exc}"
        if not resolved.exists() or not resolved.is_file():
            return None, f"attachment not found: {resolved}"
        if self._allowed_attachment_roots:
            in_allowed_root = any(
                resolved == root or root in resolved.parents
                for root in self._allowed_attachment_roots
            )
            if not in_allowed_root:
                return (
                    None,
                    f"attachment path is outside allowed roots: {resolved}",
                )
        return resolved, ""


class ListEmailsTool(FunctionTool):
    """Tool for listing recent emails."""

    def __init__(
        self,
        imap_host: str = "",
        imap_port: int = 993,
        user: str = "",
        password: str = "",
    ):
        self._imap_host = str(imap_host or "").strip()
        self._imap_port = int(imap_port or 993)
        self._user = str(user or "").strip()
        self._password = str(password or "").replace(" ", "").strip()

    @property
    def name(self) -> str:
        return "list_emails"

    @property
    def description(self) -> str:
        return "List recent emails from the inbox."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of emails to list",
                    "default": 5,
                },
                "unread_only": {
                    "type": "boolean",
                    "description": "Only list unread emails",
                    "default": True,
                },
            },
        }

    async def execute(
        self, count: int = 5, unread_only: bool = True, **kwargs: Any
    ) -> str:
        del kwargs
        if not all([self._imap_host, self._user, self._password]):
            return "Error: IMAP credentials not configured."

        def _sync_list():
            try:
                mail = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
                mail.login(self._user, self._password)
                mail.select("inbox")
                criteria = "UNSEEN" if unread_only else "ALL"
                status, response = mail.search(None, criteria)
                if status != "OK":
                    return "Error searching mailbox."

                ids = response[0].split()
                recent_ids = ids[-count:] if count > 0 else ids
                recent_ids.reverse()  # Newest first

                results = []
                for mail_id in recent_ids:
                    status, msg_data = mail.fetch(
                        mail_id, "(RFC822.SIZE BODY[HEADER.FIELDS (SUBJECT FROM DATE)])"
                    )
                    if status != "OK":
                        continue
                    msg = email.message_from_bytes(msg_data[0][1])
                    results.append(
                        f"ID: {mail_id.decode()} | From: {msg['From']} | Subject: {msg['Subject']} | Date: {msg['Date']}"
                    )

                mail.close()
                mail.logout()

                if not results:
                    return "No emails found matching criteria."
                return "\n".join(results)
            except Exception as e:
                return f"Error listing emails: {e}"

        return await asyncio.to_thread(_sync_list)


class ReadEmailTool(FunctionTool):
    """Tool for reading a specific email."""

    def __init__(
        self,
        imap_host: str = "",
        imap_port: int = 993,
        user: str = "",
        password: str = "",
    ):
        self._imap_host = str(imap_host or "").strip()
        self._imap_port = int(imap_port or 993)
        self._user = str(user or "").strip()
        self._password = str(password or "").replace(" ", "").strip()

    @property
    def name(self) -> str:
        return "read_email"

    @property
    def description(self) -> str:
        return "Read the full content of a specific email by ID."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "email_id": {
                    "type": "string",
                    "description": "The ID of the email to read (obtained from list_emails)",
                },
            },
            "required": ["email_id"],
        }

    async def execute(self, email_id: str, **kwargs: Any) -> str:
        del kwargs
        if not all([self._imap_host, self._user, self._password]):
            return "Error: IMAP credentials not configured."

        def _sync_read():
            try:
                mail = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
                mail.login(self._user, self._password)
                mail.select("inbox")
                status, msg_data = mail.fetch(str(email_id).encode(), "(RFC822)")
                if status != "OK":
                    return f"Error: Could not find email with ID {email_id}"

                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(
                                errors="replace"
                            )
                            break
                    if not body:
                        body = "[No plain text body found]"
                else:
                    body = msg.get_payload(decode=True).decode(errors="replace")

                res = [
                    f"From: {msg['From']}",
                    f"Subject: {msg['Subject']}",
                    f"Date: {msg['Date']}",
                    "-" * 20,
                    body,
                ]

                # Mark as seen
                mail.store(str(email_id).encode(), "+FLAGS", "\\Seen")

                mail.close()
                mail.logout()
                return "\n".join(res)
            except Exception as e:
                return f"Error reading email: {e}"

        return await asyncio.to_thread(_sync_read)


__all__ = ["EmailTool", "ListEmailsTool", "ReadEmailTool"]
