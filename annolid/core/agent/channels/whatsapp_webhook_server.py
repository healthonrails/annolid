from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from annolid.utils.logger import logger

from .whatsapp import WhatsAppChannel


class WhatsAppWebhookServer:
    """Minimal built-in HTTP server for WhatsApp Cloud API webhook events."""

    def __init__(
        self,
        *,
        channel: WhatsAppChannel,
        host: str = "127.0.0.1",
        port: int = 0,
        webhook_path: str = "/whatsapp/webhook",
        ingest_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.channel = channel
        self.host = str(host or "127.0.0.1")
        self.port = int(port)
        path = str(webhook_path or "/whatsapp/webhook").strip()
        if not path.startswith("/"):
            path = f"/{path}"
        self.webhook_path = path
        self.ingest_loop = ingest_loop
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> str:
        with self._lock:
            if self._httpd is not None:
                return self.webhook_url

            owner = self

            class _Handler(BaseHTTPRequestHandler):
                server_version = "AnnolidWhatsAppWebhook/1.0"

                def log_message(self, fmt: str, *args: object) -> None:
                    return

                def do_OPTIONS(self) -> None:  # noqa: N802
                    self.send_response(204)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header(
                        "Access-Control-Allow-Methods", "GET, POST, OPTIONS"
                    )
                    self.send_header("Access-Control-Allow-Headers", "Content-Type")
                    self.end_headers()

                def do_GET(self) -> None:  # noqa: N802
                    parsed = urlparse(self.path)
                    if parsed.path != owner.webhook_path:
                        logger.debug(
                            "WhatsApp webhook GET ignored path=%s from=%s",
                            parsed.path,
                            self.client_address,
                        )
                        self.send_error(404)
                        return
                    logger.info(
                        "WhatsApp webhook GET verify request from=%s path=%s",
                        self.client_address,
                        parsed.path,
                    )
                    query = parse_qs(parsed.query or "")
                    mode = (query.get("hub.mode") or [""])[0]
                    verify_token = (query.get("hub.verify_token") or [""])[0]
                    challenge = (query.get("hub.challenge") or [""])[0]
                    accepted = owner.channel.verify_webhook_challenge(
                        mode=mode,
                        verify_token=verify_token,
                        challenge=challenge,
                    )
                    if accepted is None:
                        logger.warning(
                            "WhatsApp webhook verify rejected mode=%s token_prefix=%s",
                            mode,
                            str(verify_token)[:4],
                        )
                        self.send_error(403)
                        return
                    logger.info("WhatsApp webhook verify accepted")
                    body = accepted.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_POST(self) -> None:  # noqa: N802
                    parsed = urlparse(self.path)
                    if parsed.path != owner.webhook_path:
                        logger.debug(
                            "WhatsApp webhook POST ignored path=%s from=%s",
                            parsed.path,
                            self.client_address,
                        )
                        self.send_error(404)
                        return
                    logger.info(
                        "WhatsApp webhook POST request from=%s path=%s",
                        self.client_address,
                        parsed.path,
                    )
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except Exception:
                        length = 0
                    if length <= 0 or length > 1_048_576:
                        logger.warning(
                            "WhatsApp webhook POST invalid content-length=%s",
                            length,
                        )
                        self.send_error(400)
                        return
                    raw = self.rfile.read(length)
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        logger.warning(
                            "WhatsApp webhook POST invalid JSON payload len=%s",
                            length,
                        )
                        self.send_error(400)
                        return
                    if isinstance(payload, dict):
                        logger.info(
                            "WhatsApp webhook POST payload object=%s entry_count=%s",
                            payload.get("object"),
                            len(payload.get("entry", []) or []),
                        )
                    try:
                        ingested = owner._run_ingest(payload)
                    except Exception as exc:
                        logger.exception("WhatsApp webhook ingest failed: %s", exc)
                        self.send_error(500)
                        return
                    logger.info("WhatsApp webhook ingested=%s", ingested)
                    body = json.dumps({"ok": True, "ingested": ingested}).encode(
                        "utf-8"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
            actual_port = int(getattr(self._httpd, "server_port", 0) or 0)
            self.port = actual_port
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                name="WhatsAppWebhookServer",
                daemon=True,
            )
            self._thread.start()
            logger.info("WhatsApp webhook server started on %s", self.webhook_url)
            return self.webhook_url

    def stop(self) -> None:
        with self._lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
        if httpd is not None:
            try:
                httpd.shutdown()
            finally:
                httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def webhook_url(self) -> str:
        return f"{self.base_url}{self.webhook_path}"

    def _run_ingest(self, payload: dict) -> int:
        loop = self.ingest_loop
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self.channel.ingest_webhook_payload(payload), loop
            )
            return int(fut.result(timeout=10.0))
        return int(asyncio.run(self.channel.ingest_webhook_payload(payload)))


__all__ = ["WhatsAppWebhookServer"]
