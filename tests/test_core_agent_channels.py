from __future__ import annotations

import asyncio
import builtins
from unittest.mock import patch

import pytest

from annolid.core.agent.bus import MessageBus, OutboundMessage
from annolid.core.agent.channels import (
    ChannelManager,
    SlackChannel,
    TelegramChannel,
    WhatsAppChannel,
    markdown_to_telegram_html,
)
from annolid.core.agent.channels.whatsapp_python_bridge import (
    _PlaywrightWhatsAppProvider,
)


def test_markdown_to_telegram_html_basic() -> None:
    text = "**bold** _it_ `x` [site](https://example.com)"
    rendered = markdown_to_telegram_html(text)
    assert "<b>bold</b>" in rendered
    assert "<i>it</i>" in rendered
    assert "<code>x</code>" in rendered
    assert 'href="https://example.com"' in rendered


def test_telegram_channel_allowlist_and_ingest() -> None:
    async def _run() -> None:
        bus = MessageBus()
        cfg = {"allow_from": ["alice"]}
        channel = TelegramChannel(cfg, bus)
        ok = await channel.ingest(
            sender_id="alice",
            chat_id="chat1",
            content="hello",
        )
        denied = await channel.ingest(
            sender_id="bob",
            chat_id="chat1",
            content="blocked",
        )
        assert ok is True
        assert denied is False
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.content == "hello"
        assert inbound.channel == "telegram"

    asyncio.run(_run())


def test_slack_strip_bot_mention() -> None:
    bus = MessageBus()
    ch = SlackChannel({"bot_user_id": "U123"}, bus)
    assert ch.strip_bot_mention("<@U123> hi there") == "hi there"


def test_channel_manager_dispatches_outbound() -> None:
    async def _run() -> None:
        bus = MessageBus()
        seen: list[str] = []

        async def _send(msg: OutboundMessage) -> None:
            seen.append(f"{msg.channel}:{msg.chat_id}:{msg.content}")

        manager = ChannelManager(bus=bus, channels_config={})
        manager.register_channel(TelegramChannel({}, bus, send_callback=_send))

        dispatch_task = asyncio.create_task(manager._dispatch_outbound())
        try:
            await bus.publish_outbound(
                OutboundMessage(channel="telegram", chat_id="c1", content="pong")
            )
            for _ in range(20):
                if seen:
                    break
                await asyncio.sleep(0.01)
            assert seen == ["telegram:c1:pong"]
        finally:
            dispatch_task.cancel()
            try:
                await dispatch_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_channel_ingest_infers_dm_metadata() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = TelegramChannel({}, bus)
        ok = await channel.ingest(
            sender_id="alice",
            chat_id="alice",
            content="hello",
        )
        assert ok is True
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.metadata.get("is_dm") is True
        assert inbound.metadata.get("conversation_type") == "dm"
        assert inbound.metadata.get("peer_id") == "alice"
        assert inbound.metadata.get("channel_key") == "alice"

    asyncio.run(_run())


def test_channel_ingest_preserves_explicit_dm_hints() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = SlackChannel({}, bus)
        ok = await channel.ingest(
            sender_id="U123",
            chat_id="C999",
            content="hello",
            metadata={
                "chat_type": "direct_message",
                "channel_id": "D111",
                "account_id": "workspace-1",
            },
        )
        assert ok is True
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.metadata.get("is_dm") is True
        assert inbound.metadata.get("conversation_type") == "direct_message"
        assert inbound.metadata.get("channel_key") == "C999"
        assert inbound.metadata.get("channel_id") == "D111"
        assert inbound.metadata.get("account_id") == "workspace-1"

    asyncio.run(_run())


def test_whatsapp_channel_ingest_webhook_payload() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({}, bus)
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123"},
                                "contacts": [
                                    {
                                        "wa_id": "15551234567",
                                        "profile": {"name": "Alice"},
                                    }
                                ],
                                "messages": [
                                    {
                                        "id": "wamid.ABCD",
                                        "from": "15551234567",
                                        "type": "text",
                                        "text": {"body": "hello from whatsapp"},
                                    }
                                ],
                            },
                        }
                    ]
                }
            ]
        }
        ingested = await channel.ingest_webhook_payload(payload)
        assert ingested == 1
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.channel == "whatsapp"
        assert inbound.sender_id == "15551234567"
        assert inbound.chat_id == "15551234567"
        assert inbound.content == "hello from whatsapp"
        assert inbound.metadata.get("idempotency_key") == "wamid.ABCD"
        assert inbound.metadata.get("phone_number_id") == "123"
        assert inbound.metadata.get("profile_name") == "Alice"
        assert inbound.metadata.get("conversation_type") == "dm"
        assert inbound.metadata.get("is_dm") is True

    asyncio.run(_run())


def test_whatsapp_channel_build_cloud_api_payload() -> None:
    bus = MessageBus()
    channel = WhatsAppChannel(
        {"phone_number_id": "12345", "preview_url": True},
        bus,
    )
    payload = channel.build_cloud_api_payload(
        OutboundMessage(
            channel="whatsapp",
            chat_id="15551234567",
            content="bot reply",
            reply_to="wamid.REF",
        )
    )
    assert payload["messaging_product"] == "whatsapp"
    assert payload["phone_number_id"] == "12345"
    assert payload["to"] == "15551234567"
    assert payload["type"] == "text"
    assert payload["text"]["body"] == "bot reply"
    assert payload["text"]["preview_url"] is True
    assert payload["context"]["message_id"] == "wamid.REF"


def test_whatsapp_channel_verify_webhook_challenge() -> None:
    bus = MessageBus()
    channel = WhatsAppChannel({"verify_token": "secret-token"}, bus)
    ok = channel.verify_webhook_challenge(
        mode="subscribe",
        verify_token="secret-token",
        challenge="12345",
    )
    denied = channel.verify_webhook_challenge(
        mode="subscribe",
        verify_token="wrong",
        challenge="12345",
    )
    assert ok == "12345"
    assert denied is None


def test_whatsapp_channel_send_cloud_api_text() -> None:
    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self) -> bytes:
            return b'{"messages":[{"id":"wamid.sent"}]}'

    bus = MessageBus()
    channel = WhatsAppChannel(
        {
            "access_token": "token",
            "phone_number_id": "123",
            "api_version": "v22.0",
            "api_base": "https://graph.facebook.com",
        },
        bus,
    )
    with patch("urllib.request.urlopen", return_value=_FakeResponse()) as mocked:
        ok, code, body = channel._send_cloud_api_text(
            OutboundMessage(
                channel="whatsapp",
                chat_id="15551234567",
                content="hello",
            )
        )
    assert mocked.called
    assert ok is True
    assert code == 200
    assert "wamid.sent" in body


def test_whatsapp_channel_bridge_message_ingest() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        await channel._handle_bridge_message(
            '{"type":"message","id":"m1","sender":"15551234567@s.whatsapp.net","content":"hi","timestamp":123,"isGroup":false}'
        )
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.channel == "whatsapp"
        assert inbound.sender_id == "15551234567"
        assert inbound.chat_id == "15551234567@s.whatsapp.net"
        assert inbound.content == "hi"
        assert inbound.metadata.get("message_id") == "m1"
        assert inbound.metadata.get("is_dm") is True

    asyncio.run(_run())


def test_whatsapp_channel_bridge_prefers_pn_sender() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        await channel._handle_bridge_message(
            '{"type":"message","id":"m2","sender":"My Name","pn":"15551234567","content":"hello","timestamp":123,"isGroup":false}'
        )
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.sender_id == "15551234567"
        assert inbound.chat_id == "15551234567"
        assert inbound.content == "hello"

    asyncio.run(_run())


def test_whatsapp_channel_bridge_prefers_chat_id_when_present() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        await channel._handle_bridge_message(
            '{"type":"message","id":"m2b","sender":"Business Account","pn":"15551234567","chat_id":"Message yourself","content":"hello","timestamp":123,"isGroup":false}'
        )
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.sender_id == "15551234567"
        assert inbound.chat_id == "Message yourself"
        assert inbound.content == "hello"

    asyncio.run(_run())


def test_whatsapp_channel_bridge_self_message_direction_out() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        await channel._handle_bridge_message(
            '{"type":"message","id":"m3","sender":"15551234567","content":"self ping","direction":"out","timestamp":123,"isGroup":false}'
        )
        with pytest.raises(asyncio.TimeoutError):
            await bus.consume_inbound(timeout_s=0.2)

    asyncio.run(_run())


def test_whatsapp_channel_bridge_self_message_direction_out_when_enabled() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel(
            {"bridge_url": "ws://127.0.0.1:3001", "ingest_outgoing_messages": True},
            bus,
        )
        await channel._handle_bridge_message(
            '{"type":"message","id":"m3","sender":"15551234567","content":"self ping","direction":"out","timestamp":123,"isGroup":false}'
        )
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.sender_id == "15551234567"
        assert inbound.metadata.get("bridge_direction") == "out"

    asyncio.run(_run())


def test_whatsapp_channel_send_bridge_text() -> None:
    async def _run() -> None:
        sent: list[str] = []

        class _Ws:
            async def send(self, payload: str) -> None:
                sent.append(payload)

        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        channel._bridge_ws = _Ws()
        await channel.send(
            OutboundMessage(channel="whatsapp", chat_id="chat-1", content="pong")
        )
        assert sent
        assert '"type": "send"' in sent[0]
        assert '"to": "chat-1"' in sent[0]
        assert '"text": "pong"' in sent[0]

    asyncio.run(_run())


def test_whatsapp_channel_bridge_loop_without_websockets() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        channel._running = True
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "websockets" or name.startswith("websockets."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_blocked_import),
            patch("annolid.core.agent.channels.whatsapp.logger.error") as mocked_error,
        ):
            task = asyncio.create_task(channel._run_bridge_loop())
            await asyncio.sleep(0.05)
            channel._stop_event.set()
            await asyncio.wait_for(task, timeout=1.0)
            assert mocked_error.called

    asyncio.run(_run())


def test_whatsapp_channel_log_qr_without_qrcode() -> None:
    bus = MessageBus()
    channel = WhatsAppChannel({}, bus)
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "qrcode" or name.startswith("qrcode."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=_blocked_import),
        patch("annolid.core.agent.channels.whatsapp.logger.info") as mocked_info,
    ):
        channel._log_qr_ascii("dummy-qr")
        assert mocked_info.called


def test_whatsapp_python_provider_start_without_playwright() -> None:
    async def _run() -> None:
        events: list[dict[str, str]] = []

        async def _on_event(evt: dict[str, str]) -> None:
            events.append(evt)

        provider = _PlaywrightWhatsAppProvider(
            session_dir="~/.annolid/whatsapp-web-session",
            headless=True,
            on_event=_on_event,
        )
        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "playwright.async_api" or name.startswith("playwright"):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            await provider.start()
        assert events
        assert events[0].get("type") == "error"
        assert "Playwright is required" in str(events[0].get("error", ""))

    asyncio.run(_run())
