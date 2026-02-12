from __future__ import annotations

import asyncio

from annolid.core.agent.bus import MessageBus, OutboundMessage
from annolid.core.agent.channels import (
    ChannelManager,
    SlackChannel,
    TelegramChannel,
    markdown_to_telegram_html,
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
