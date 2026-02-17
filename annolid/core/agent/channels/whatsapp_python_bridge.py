from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from annolid.utils.logger import logger


class _PlaywrightWhatsAppProvider:
    """Experimental WhatsApp Web provider backed by Playwright."""

    def __init__(
        self,
        *,
        session_dir: str,
        headless: bool,
        on_event,
    ) -> None:
        self._session_dir = str(session_dir or "~/.annolid/whatsapp-web-session")
        self._headless = bool(headless)
        self._on_event = on_event
        self._page = None
        self._context = None
        self._playwright = None
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._last_seen_id = ""
        self._last_qr_ref = ""
        self._recent_bot_messages: list[tuple[float, str]] = []
        self._last_activity_ts = 0.0

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:
            await self._on_event(
                {
                    "type": "error",
                    "error": (
                        "Playwright is required for embedded WhatsApp bridge. "
                        f"Install Playwright and browser runtime. ({exc})"
                    ),
                }
            )
            return

        logger.info("WhatsApp Playwright provider starting")
        await self._on_event({"type": "status", "status": "starting"})
        self._playwright = await async_playwright().start()
        user_data_dir = str(Path(self._session_dir).expanduser())
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self._headless,
        )
        page = (
            self._context.pages[0]
            if self._context.pages
            else await self._context.new_page()
        )
        self._page = page
        await page.goto("https://web.whatsapp.com", wait_until="domcontentloaded")
        await self._emit_qr_if_available(force=True)
        await self._on_event({"type": "status", "status": "awaiting_qr_scan"})
        # Wait for UI to become available after scan.
        for _ in range(180):
            if await self._is_ready():
                await self._on_event({"type": "status", "status": "connected"})
                logger.info("WhatsApp Playwright provider connected")
                self._poll_task = asyncio.create_task(self._poll_incoming())
                return
            await self._emit_qr_if_available(force=False)
            await asyncio.sleep(1.0)
        await self._on_event(
            {"type": "error", "error": "Timed out waiting for QR login"}
        )

    async def stop(self) -> None:
        poll = self._poll_task
        self._poll_task = None
        if poll is not None:
            poll.cancel()
            try:
                await poll
            except asyncio.CancelledError:
                pass
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def send_text(self, to: str, text: str) -> None:
        if self._page is None:
            raise RuntimeError("WhatsApp Web is not initialized")
        raw_target = str(to or "").strip()
        target = raw_target.split("@", 1)[0]
        target = "".join(ch for ch in target if ch.isdigit())
        message = str(text or "")
        if target:
            send_url = f"https://web.whatsapp.com/send?phone={quote(target)}&text={quote(message)}"
            await self._page.goto(send_url, wait_until="domcontentloaded")
            # Wait for compose box and send by Enter.
            box = await self._page.wait_for_selector(
                'div[contenteditable="true"][data-tab]', timeout=15000
            )
            await box.press("Enter")
        else:
            # Fallback for chats resolved by WhatsApp title (self-chat / business label):
            # send in the currently open chat composer.
            logger.warning(
                "WhatsApp bridge send fallback: non-numeric recipient=%r, sending to active chat",
                raw_target,
            )
            composer = await self._page.wait_for_selector(
                'footer div[contenteditable="true"][data-tab]',
                timeout=15000,
            )
            await composer.click()
            await composer.fill("")
            await composer.type(message)
            await composer.press("Enter")
        self._remember_bot_message(message)

    async def _is_ready(self) -> bool:
        if self._page is None:
            return False
        script = """
        () => {
          const app = document.querySelector('#app');
          if (!app) return false;
          const hasSearch = !!document.querySelector('div[contenteditable="true"][data-tab]');
          const hasSidebar = !!document.querySelector('aside');
          return hasSearch || hasSidebar;
        }
        """
        try:
            return bool(await self._page.evaluate(script))
        except Exception:
            return False

    async def _poll_incoming(self) -> None:
        while True:
            try:
                if self._page is None:
                    await asyncio.sleep(2.0)
                    continue
                payload = await self._page.evaluate(
                    """
                    () => {
                      const titleNode = document.querySelector('header span[title]');
                      const sender = titleNode ? titleNode.getAttribute('title') : '';
                      const wraps = [...document.querySelectorAll('div.copyable-text[data-pre-plain-text], div[data-id] div.copyable-text')];
                      const lastWrap = wraps[wraps.length - 1];
                      if (!lastWrap) return null;
                      const bubble = lastWrap.closest('div.message-in, div.message-out');
                      const wrapWithId = lastWrap.closest('div[data-id]');
                      const id = (wrapWithId && wrapWithId.getAttribute('data-id')) || lastWrap.getAttribute('data-id') || '';
                      let jid = '';
                      if (id) {
                        const m = id.match(/(\\d+@(?:c\\.us|s\\.us|c\\.whatsapp\\.net|s\\.whatsapp\\.net))/);
                        if (m && m[1]) jid = m[1];
                      }
                      let pn = '';
                      if (jid) {
                        pn = jid.split('@')[0].replace(/\\D+/g, '');
                      }
                      const textNode =
                        lastWrap.querySelector('span.selectable-text') ||
                        lastWrap.querySelector('span[dir="ltr"]') ||
                        lastWrap.querySelector('span[dir="auto"]');
                      const text = (textNode ? textNode.innerText : lastWrap.innerText || '').trim();
                      if (!text) return null;
                      const direction = bubble && bubble.classList.contains('message-out') ? 'out' : 'in';
                      return { id, text, sender, jid, pn, direction };
                    }
                    """
                )
                if isinstance(payload, dict):
                    msg_id = str(payload.get("id", "") or "")
                    text = str(payload.get("text", "") or "")
                    sender_title = str(payload.get("sender", "") or "")
                    sender_jid = str(payload.get("jid", "") or "")
                    sender_pn = str(payload.get("pn", "") or "")
                    direction = (
                        str(payload.get("direction", "in") or "in").strip().lower()
                    )
                    if direction == "out" and self._is_recent_bot_message(text):
                        await asyncio.sleep(2.0)
                        continue
                    sender = sender_jid or sender_pn or sender_title
                    chat_id = sender_pn or sender_jid or sender_title
                    key = msg_id or f"{sender}:{text}"
                    if key and key != self._last_seen_id:
                        self._last_seen_id = key
                        self._last_activity_ts = asyncio.get_running_loop().time()
                        logger.info(
                            "WhatsApp Playwright detected message direction=%s sender=%s chat_id=%s",
                            direction,
                            sender or sender_title,
                            chat_id,
                        )
                        await self._on_event(
                            {
                                "type": "message",
                                "id": key,
                                "sender": sender,
                                "chat_id": chat_id,
                                "pn": sender_pn,
                                "content": text,
                                "timestamp": int(asyncio.get_running_loop().time()),
                                "isGroup": False,
                                "direction": direction,
                            }
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("WhatsApp Playwright poll error: %s", exc)
            now = asyncio.get_running_loop().time()
            if now - self._last_activity_ts > 60.0:
                self._last_activity_ts = now
                logger.info(
                    "WhatsApp Playwright polling active; no new messages detected in the last 60s"
                )
            await asyncio.sleep(2.0)

    def _remember_bot_message(self, text: str) -> None:
        content = str(text or "").strip()
        if not content:
            return
        now = asyncio.get_running_loop().time()
        self._recent_bot_messages.append((now, content))
        cutoff = now - 30.0
        self._recent_bot_messages = [
            (ts, body) for ts, body in self._recent_bot_messages if ts >= cutoff
        ]

    def _is_recent_bot_message(self, text: str) -> bool:
        content = str(text or "").strip()
        if not content:
            return False
        now = asyncio.get_running_loop().time()
        keep: list[tuple[float, str]] = []
        matched = False
        for ts, body in self._recent_bot_messages:
            if now - ts > 30.0:
                continue
            if not matched and body == content and now - ts <= 15.0:
                matched = True
                continue
            keep.append((ts, body))
        self._recent_bot_messages = keep
        return matched

    async def _emit_qr_if_available(self, *, force: bool) -> None:
        if self._page is None:
            return
        try:
            qr_ref = await self._page.evaluate(
                """
                () => {
                  const node = document.querySelector('div[data-ref]');
                  if (!node) return '';
                  return String(node.getAttribute('data-ref') || '');
                }
                """
            )
        except Exception:
            qr_ref = ""
        qr_ref = str(qr_ref or "").strip()
        if not qr_ref:
            if force:
                await self._on_event({"type": "qr", "qr": "open_browser_and_scan"})
            return
        if not force and qr_ref == self._last_qr_ref:
            return
        self._last_qr_ref = qr_ref
        logger.info("WhatsApp Playwright QR payload refreshed")
        await self._on_event({"type": "qr", "qr": qr_ref})


class WhatsAppPythonBridge:
    """Embedded Python WebSocket bridge for WhatsAppChannel bridge mode."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 3001,
        token: str = "",
        session_dir: str = "~/.annolid/whatsapp-web-session",
        headless: bool = False,
    ) -> None:
        self.host = str(host or "127.0.0.1")
        self.port = int(port or 3001)
        self.token = str(token or "")
        self.session_dir = str(session_dir or "~/.annolid/whatsapp-web-session")
        self.headless = bool(headless)
        self._clients: set[Any] = set()
        self._server = None
        self._provider_task: Optional[asyncio.Task[None]] = None
        self._provider_started = False
        self._recent_events: list[dict[str, Any]] = []
        self._provider = _PlaywrightWhatsAppProvider(
            session_dir=self.session_dir,
            headless=self.headless,
            on_event=self._broadcast,
        )

    @property
    def bridge_url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    async def start(self) -> str:
        try:
            import websockets
        except Exception as exc:
            raise RuntimeError(f"websockets package is required for bridge mode: {exc}")
        self._server = await websockets.serve(self._on_client, self.host, self.port)
        logger.info("Embedded WhatsApp Python bridge started on %s", self.bridge_url)
        return self.bridge_url

    async def stop(self) -> None:
        self._provider_started = False
        task = self._provider_task
        self._provider_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._provider.stop()
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

    async def _on_client(self, websocket) -> None:
        authed = not self.token
        self._clients.add(websocket)
        try:
            await self._ensure_provider_started()
            async for raw in websocket:
                data = self._parse_json(raw)
                if data is None:
                    continue
                msg_type = str(data.get("type", "") or "").strip().lower()
                if not authed:
                    if msg_type == "auth" and str(data.get("token", "")) == self.token:
                        authed = True
                    else:
                        await websocket.close(code=4003, reason="Invalid bridge token")
                        return
                    continue
                if msg_type == "send":
                    to = str(data.get("to", "") or "")
                    text = str(data.get("text", "") or "")
                    try:
                        await self._provider.send_text(to, text)
                        await websocket.send(json.dumps({"type": "sent", "to": to}))
                    except Exception as exc:
                        await websocket.send(
                            json.dumps({"type": "error", "error": str(exc)})
                        )
        finally:
            self._clients.discard(websocket)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        if isinstance(event, dict):
            self._recent_events.append(dict(event))
            self._recent_events = self._recent_events[-20:]
        if not self._clients:
            return
        payload = json.dumps(event)
        dead = []
        for client in self._clients:
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    async def _ensure_provider_started(self) -> None:
        if self._provider_started:
            # Replay recent state to newly connected client(s).
            if self._clients and self._recent_events:
                for event in self._recent_events[-5:]:
                    payload = json.dumps(event)
                    for client in list(self._clients):
                        try:
                            await client.send(payload)
                        except Exception:
                            self._clients.discard(client)
            return
        self._provider_started = True

        async def _runner() -> None:
            try:
                await self._provider.start()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._broadcast({"type": "error", "error": str(exc)})

        self._provider_task = asyncio.create_task(_runner())

    def _parse_json(self, raw: Any) -> Optional[dict[str, Any]]:
        try:
            payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload


__all__ = ["WhatsAppPythonBridge"]
