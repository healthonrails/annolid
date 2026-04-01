from __future__ import annotations

import asyncio
import json
import builtins
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from annolid.core.agent.bus import MessageBus, OutboundMessage
from annolid.core.agent.channels import (
    ChannelManager,
    EmailChannel,
    SlackChannel,
    TelegramChannel,
    WhatsAppChannel,
    ZulipChannel,
    markdown_to_telegram_html,
)
from annolid.core.agent.channels.whatsapp_python_bridge import (
    _PlaywrightWhatsAppProvider,
    WhatsAppPythonBridge,
)


@pytest.fixture(autouse=True)
def _isolate_home_for_zulip_cursor_state(monkeypatch, tmp_path: Path):
    """Prevent default Zulip cursor files in ~/.annolid from leaking across tests."""
    monkeypatch.setenv("HOME", str(tmp_path))


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


def test_zulip_channel_ingest_with_allowlist() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel({"allow_from": ["alice@example.com"]}, bus)
        ok = await channel._handle_message(
            sender_id="alice@example.com",
            chat_id="stream:general:annolid",
            content="hello from zulip",
            metadata={"conversation_type": "channel"},
        )
        denied = await channel._handle_message(
            sender_id="bob@example.com",
            chat_id="stream:general:annolid",
            content="blocked",
            metadata={"conversation_type": "channel"},
        )
        assert ok is True
        assert denied is False
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.channel == "zulip"
        assert inbound.sender_id == "alice@example.com"
        assert inbound.chat_id == "stream:general:annolid"

    asyncio.run(_run())


def test_channel_manager_skips_local_only_gui_outbound_without_warning(caplog) -> None:
    async def _run() -> None:
        bus = MessageBus()
        manager = ChannelManager(bus=bus, channels_config={})
        with caplog.at_level("WARNING", logger="annolid.agent.channels"):
            await manager._send_to_channel(
                OutboundMessage(channel="gui", chat_id="annolid_bot", content="ok")
            )
        assert "Unknown outbound channel: gui" not in caplog.text

    asyncio.run(_run())


def test_channel_manager_initializes_zulip_when_enabled() -> None:
    bus = MessageBus()
    manager = ChannelManager(
        bus=bus,
        channels_config={
            "zulip": {
                "enabled": True,
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "x",
            }
        },
    )
    assert "zulip" in manager.enabled_channels


def test_zulip_channel_build_send_payload_for_private_multiple() -> None:
    bus = MessageBus()
    channel = ZulipChannel({}, bus)
    payload = channel._build_send_payload(
        OutboundMessage(
            channel="zulip",
            chat_id="pm:alice@example.com,bob@example.com,alice@example.com",
            content="hello",
        )
    )
    assert payload is not None
    assert payload["type"] == "private"
    recipients = json.loads(payload["to"])
    assert recipients == ["alice@example.com", "bob@example.com"]


def test_zulip_channel_uses_camel_case_config_aliases() -> None:
    bus = MessageBus()
    channel = ZulipChannel(
        {
            "enabled": True,
            "serverUrl": "https://zulip.example.com",
            "user": "bot@zulip.example.com",
            "apiKey": "k",
            "pollingInterval": 15,
            "cursorStatePath": "/tmp/zulip_cursor.json",
            "maxProcessedIds": 2048,
            "logSkipReasons": True,
            "botName": "Annolid Bot",
            "unreadBackfillEnabled": True,
            "unreadBackfillOnEmptyOnly": True,
            "unreadBackfillLimit": 25,
            "unreadBackfillCooldownS": 60,
        },
        bus,
    )
    assert channel._cfg("server_url") == "https://zulip.example.com"
    assert channel._cfg("api_key") == "k"
    assert channel._resolve_poll_interval_seconds() == 15
    assert channel._cfg("cursor_state_path") == "/tmp/zulip_cursor.json"
    assert int(channel._cfg("max_processed_ids")) == 2048
    assert bool(channel._cfg("log_skip_reasons")) is True
    assert channel._cfg("bot_name") == "Annolid Bot"
    assert bool(channel._cfg("unread_backfill_enabled")) is True
    assert bool(channel._cfg("unread_backfill_on_empty_only")) is True
    assert int(channel._cfg("unread_backfill_limit")) == 25
    assert int(channel._cfg("unread_backfill_cooldown_s")) == 60


def test_zulip_channel_uses_default_cursor_state_path_when_not_configured() -> None:
    bus = MessageBus()
    channel = ZulipChannel(
        {
            "server_url": "https://zulip.example.com",
            "user": "bot@zulip.example.com",
            "stream": "Annolid",
            "topic": "general chat",
        },
        bus,
    )
    path = channel._cursor_state_path
    assert path
    assert path.endswith(".json")
    assert "zulip_cursor_" in path


def test_zulip_channel_request_json_handles_http_error_payload() -> None:
    bus = MessageBus()
    channel = ZulipChannel({}, bus)

    class _FakeHttpError(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__(
                url="https://zulip.example.com/api/v1/messages",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=None,
            )

        def read(self) -> bytes:
            return b'{"result":"error","msg":"invalid narrow operator"}'

    with patch("urllib.request.urlopen", side_effect=_FakeHttpError()):
        data = channel._request_json(
            method="GET",
            base_url="https://zulip.example.com",
            path="/api/v1/messages",
            user="bot@zulip.example.com",
            api_key="k",
            params={"anchor": "newest"},
        )
    assert data["result"] == "error"
    assert "invalid narrow operator" in data["msg"]


def test_zulip_channel_first_poll_only_initializes_anchor() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
            },
            bus,
        )

        def _fake_request_json(**kwargs):
            anchor = kwargs.get("params", {}).get("anchor")
            if anchor == "newest":
                return {
                    "result": "success",
                    "messages": [{"id": 123, "sender_email": "bot@zulip.example.com"}],
                }
            return {"result": "success", "messages": []}

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert channel._anchor_initialized is True
        assert channel._last_message_id == 123
        with pytest.raises(asyncio.TimeoutError):
            await bus.consume_inbound(timeout_s=0.1)

    asyncio.run(_run())


def test_zulip_channel_anchor_init_failure_does_not_mark_initialized(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
                "cursor_state_path": str(
                    tmp_path / "zulip_cursor_anchor_init_fail.json"
                ),
            },
            bus,
        )

        def _fake_request_json(**kwargs):
            del kwargs
            return {"result": "error", "msg": "temporary failure"}

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert channel._anchor_initialized is False
        assert channel._last_message_id == 0

    asyncio.run(_run())


def test_zulip_channel_skips_read_and_duplicate_messages() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
            },
            bus,
        )
        channel._anchor_initialized = True

        calls: list[int] = []

        async def _fake_handle_message(**kwargs):
            calls.append(int((kwargs.get("metadata") or {}).get("zulip_message_id", 0)))
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 200,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "already read",
                        "flags": ["read"],
                    },
                    {
                        "id": 201,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "process me",
                        "flags": [],
                    },
                    {
                        "id": 201,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "duplicate",
                        "flags": [],
                    },
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert calls == [201]
        assert channel._last_message_id == 201

    asyncio.run(_run())


def test_zulip_channel_persists_cursor_state_and_skips_old_on_restart(tmp_path) -> None:
    async def _run() -> None:
        state_path = tmp_path / "zulip_cursor.json"
        config = {
            "server_url": "https://zulip.example.com",
            "user": "bot@zulip.example.com",
            "api_key": "k",
            "cursor_state_path": str(state_path),
        }
        bus = MessageBus()
        channel = ZulipChannel(config, bus)
        channel._anchor_initialized = True

        first_calls: list[int] = []

        async def _fake_handle_message_first(**kwargs):
            first_calls.append(
                int((kwargs.get("metadata") or {}).get("zulip_message_id", 0))
            )
            return True

        channel._handle_message = _fake_handle_message_first  # type: ignore[assignment]

        def _fake_request_json_first(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 301,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "first message",
                        "flags": [],
                    }
                ],
            }

        channel._request_json = _fake_request_json_first  # type: ignore[assignment]
        await channel._poll_messages()
        assert first_calls == [301]
        assert state_path.exists()

        bus2 = MessageBus()
        channel2 = ZulipChannel(config, bus2)
        channel2._anchor_initialized = True

        second_calls: list[int] = []

        async def _fake_handle_message_second(**kwargs):
            second_calls.append(
                int((kwargs.get("metadata") or {}).get("zulip_message_id", 0))
            )
            return True

        channel2._handle_message = _fake_handle_message_second  # type: ignore[assignment]

        def _fake_request_json_second(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 301,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "old replay",
                        "flags": [],
                    },
                    {
                        "id": 302,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "new message",
                        "flags": [],
                    },
                ],
            }

        channel2._request_json = _fake_request_json_second  # type: ignore[assignment]
        await channel2._poll_messages()
        assert second_calls == [302]
        assert channel2._last_message_id == 302

    asyncio.run(_run())


def test_zulip_channel_skips_historical_messages_without_checkpoint(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
                "cursor_state_path": str(
                    tmp_path / "zulip_cursor_historical_skip.json"
                ),
            },
            bus,
        )
        channel._anchor_initialized = True
        channel._cursor_state_has_checkpoint = False
        channel._startup_unix_s = 2_000_000

        calls: list[int] = []

        async def _fake_handle_message(**kwargs):
            calls.append(int((kwargs.get("metadata") or {}).get("zulip_message_id", 0)))
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 401,
                        "timestamp": 1_999_000,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "old message",
                        "flags": [],
                    },
                    {
                        "id": 402,
                        "timestamp": 2_000_000,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "new message",
                        "flags": [],
                    },
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert calls == [402]

    asyncio.run(_run())


def test_zulip_channel_logs_skip_reasons_when_enabled(tmp_path: Path) -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
                "log_skip_reasons": True,
                "cursor_state_path": str(tmp_path / "zulip_cursor_skip_reasons.json"),
            },
            bus,
        )
        channel._anchor_initialized = True
        channel._cursor_state_has_checkpoint = False
        channel._startup_unix_s = 2_000_000

        async def _fake_handle_message(**kwargs):
            del kwargs
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 501,
                        "timestamp": 2_000_000,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "",
                        "flags": [],
                    },
                    {
                        "id": 502,
                        "timestamp": 1_999_000,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "old",
                        "flags": [],
                    },
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        with patch("annolid.core.agent.channels.zulip.logger.debug") as mock_debug:
            await channel._poll_messages()
        debug_calls = " | ".join(
            str(args[0]) % args[1:] if args else ""
            for args, _kwargs in mock_debug.call_args_list
        )
        assert "Zulip skip message id=501 reason=empty_content" in debug_calls
        assert "Zulip skip message id=502 reason=historical" in debug_calls

    asyncio.run(_run())


def test_zulip_channel_topic_mismatch_requires_mention() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "annolid-bot@zulip.example.com",
                "api_key": "k",
                "topic": "bot",
                "bot_name": "Annolid Bot",
            },
            bus,
        )
        channel._anchor_initialized = True
        channel._cursor_state_has_checkpoint = True
        calls: list[str] = []

        async def _fake_handle_message(**kwargs):
            calls.append(str(kwargs.get("content") or ""))
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 601,
                        "timestamp": 2_000_000,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "No mention here",
                        "flags": [],
                    },
                    {
                        "id": 602,
                        "timestamp": 2_000_001,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "general chat",
                        "content": "@**Annolid Bot** please help",
                        "flags": [],
                    },
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert calls == ["please help"]

    asyncio.run(_run())


def test_zulip_channel_first_poll_fetches_after_anchor_init() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "annolid-bot@zulip.example.com",
                "api_key": "k",
            },
            bus,
        )
        calls: list[int] = []
        request_anchors: list[Any] = []
        channel._cursor_state_has_checkpoint = True

        async def _fake_handle_message(**kwargs):
            calls.append(int((kwargs.get("metadata") or {}).get("zulip_message_id", 0)))
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            params = kwargs.get("params", {})
            anchor = params.get("anchor")
            request_anchors.append(anchor)
            if anchor == "newest":
                return {"result": "success", "messages": [{"id": 700}]}
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 701,
                        "timestamp": 2_000_010,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "bot",
                        "content": "hello",
                        "flags": [],
                    }
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert "newest" in request_anchors
        assert 700 in request_anchors
        assert calls == [701]

    asyncio.run(_run())


def test_zulip_channel_request_json_encodes_bool_query_as_json_lowercase() -> None:
    bus = MessageBus()
    channel = ZulipChannel({}, bus)
    captured_url = {"value": ""}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self) -> bytes:
            return b'{"result":"success","messages":[]}'

    def _fake_urlopen(req, timeout=30):
        del timeout
        captured_url["value"] = str(getattr(req, "full_url", "") or "")
        return _FakeResponse()

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        data = channel._request_json(
            method="GET",
            base_url="https://zulip.example.com",
            path="/api/v1/messages",
            user="bot@zulip.example.com",
            api_key="k",
            params={
                "anchor": "newest",
                "apply_markdown": False,
                "include_anchor": True,
            },
        )
    assert data["result"] == "success"
    assert "apply_markdown=false" in captured_url["value"]
    assert "include_anchor=true" in captured_url["value"]


def test_zulip_channel_logs_poll_cycle_summary() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
                "stream": "Annolid",
                "topic": "bot",
            },
            bus,
        )
        channel._anchor_initialized = True
        channel._cursor_state_has_checkpoint = True

        async def _fake_handle_message(**kwargs):
            del kwargs
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 801,
                        "timestamp": 2_000_000,
                        "sender_email": "alice@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "bot",
                        "content": "hello",
                        "flags": [],
                    }
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        with patch("annolid.core.agent.channels.zulip.logger.info") as mock_info:
            await channel._poll_messages()
        info_messages = " | ".join(
            str(args[0]) % args[1:] if args else ""
            for args, _kwargs in mock_info.call_args_list
        )
        assert (
            "Polling Zulip for bot@zulip.example.com stream=Annolid topic=bot"
            in info_messages
        )
        assert (
            "Zulip poll complete user=bot@zulip.example.com fetched_main=1 fetched_unread=0 handled=1 skipped=0 skip_reasons=none"
            in info_messages
        )

    asyncio.run(_run())


def test_zulip_channel_missing_credentials_warn_once() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel({}, bus)
        with patch("annolid.core.agent.channels.zulip.logger.warning") as mock_warning:
            await channel._poll_messages()
            await channel._poll_messages()
        warning_messages = [
            str(args[0]) % args[1:] if args else ""
            for args, _kwargs in mock_warning.call_args_list
        ]
        hits = [m for m in warning_messages if "missing server_url/user/api_key" in m]
        assert len(hits) == 1

    asyncio.run(_run())


def test_zulip_channel_unread_backfill_processes_older_unread() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
                "stream": "Annolid",
                "unread_backfill_enabled": True,
                "unread_backfill_on_empty_only": True,
                "unread_backfill_limit": 10,
                "unread_backfill_cooldown_s": 0,
            },
            bus,
        )
        channel._anchor_initialized = True
        channel._cursor_state_has_checkpoint = False
        channel._startup_unix_s = 2_000_000

        calls: list[int] = []

        async def _fake_handle_message(**kwargs):
            calls.append(int((kwargs.get("metadata") or {}).get("zulip_message_id", 0)))
            return True

        channel._handle_message = _fake_handle_message  # type: ignore[assignment]

        def _fake_request_json(**kwargs):
            params = kwargs.get("params", {})
            anchor = params.get("anchor")
            narrow = str(params.get("narrow") or "")
            if anchor == "newest" and '"unread"' in narrow:
                return {
                    "result": "success",
                    "messages": [
                        {
                            "id": 901,
                            "timestamp": 1_999_000,
                            "sender_email": "alice@example.com",
                            "type": "stream",
                            "display_recipient": "Annolid",
                            "subject": "bot",
                            "content": "older unread",
                            "flags": [],
                        }
                    ],
                }
            return {"result": "success", "messages": []}

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        await channel._poll_messages()
        assert calls == [901]

    asyncio.run(_run())


def test_zulip_channel_allow_from_blocked_reason_in_summary() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = ZulipChannel(
            {
                "server_url": "https://zulip.example.com",
                "user": "bot@zulip.example.com",
                "api_key": "k",
                "allow_from": ["trusted@example.com"],
            },
            bus,
        )
        channel._anchor_initialized = True
        channel._cursor_state_has_checkpoint = True

        def _fake_request_json(**kwargs):
            del kwargs
            return {
                "result": "success",
                "messages": [
                    {
                        "id": 951,
                        "timestamp": 2_000_000,
                        "sender_email": "other@example.com",
                        "type": "stream",
                        "display_recipient": "Annolid",
                        "subject": "bot",
                        "content": "hello",
                        "flags": [],
                    }
                ],
            }

        channel._request_json = _fake_request_json  # type: ignore[assignment]
        with patch("annolid.core.agent.channels.zulip.logger.info") as mock_info:
            await channel._poll_messages()
        info_messages = " | ".join(
            str(args[0]) % args[1:] if args else ""
            for args, _kwargs in mock_info.call_args_list
        )
        assert "skip_reasons=allow_from_blocked:1" in info_messages

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


def test_email_channel_allowlist_matches_from_header_address() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = EmailChannel({"allow_from": ["googlecloud@google.com"]}, bus)
        ok = await channel.ingest(
            sender_email="Google Cloud <googlecloud@google.com>",
            subject="Welcome",
            body="Hello",
        )
        assert ok is True
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.sender_id == "googlecloud@google.com"
        assert inbound.chat_id == "googlecloud@google.com"
        assert inbound.metadata.get("sender_email") == "googlecloud@google.com"
        assert (
            inbound.metadata.get("raw_from") == "Google Cloud <googlecloud@google.com>"
        )

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


def test_whatsapp_channel_ingest_webhook_skips_self_sender_by_default() -> None:
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
                                "metadata": {
                                    "phone_number_id": "123",
                                    "display_phone_number": "+1 (555) 123-4567",
                                },
                                "messages": [
                                    {
                                        "id": "wamid.SELF",
                                        "from": "15551234567",
                                        "type": "text",
                                        "text": {"body": "my own message"},
                                    }
                                ],
                            },
                        }
                    ]
                }
            ]
        }
        ingested = await channel.ingest_webhook_payload(payload)
        assert ingested == 0
        with pytest.raises(asyncio.TimeoutError):
            await bus.consume_inbound(timeout_s=0.2)

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


def test_whatsapp_channel_bridge_media_only_message_ingest() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        await channel._handle_bridge_message(
            '{"type":"message","id":"m-media-1","sender":"15551234567@s.whatsapp.net","media_type":"image","media":["wa-bridge-media:image:m-media-1"],"content":"","timestamp":123,"isGroup":false}'
        )
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.channel == "whatsapp"
        assert inbound.content == "[image message]"
        assert inbound.media == ["wa-bridge-media:image:m-media-1"]
        assert inbound.metadata.get("has_media") is True
        assert inbound.metadata.get("media_type") == "image"

    asyncio.run(_run())


def test_whatsapp_channel_self_chat_bypasses_allowlist() -> None:
    async def _run() -> None:
        bus = MessageBus()
        # allow_from does NOT include "Message yourself"
        channel = WhatsAppChannel(
            {"bridge_url": "ws://127.0.0.1:3001", "allow_from": ["someone-else"]},
            bus,
        )
        # This sender should be allowed due to the override
        ok = await channel.ingest(
            sender_id="Message yourself",
            chat_id="Message yourself",
            content="hello self",
        )
        assert ok is True
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.sender_id == "Message yourself"

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


def test_whatsapp_channel_bridge_self_message_from_me_ignored() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        await channel._handle_bridge_message(
            '{"type":"message","id":"m3b","sender":"15551234567","chat_id":"15551234567","content":"self reply","fromMe":true,"direction":"in","timestamp":123,"isGroup":false}'
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


def test_whatsapp_channel_bridge_loop_incompatible_websockets_version() -> None:
    async def _run() -> None:
        import types

        bus = MessageBus()
        channel = WhatsAppChannel({"bridge_url": "ws://127.0.0.1:3001"}, bus)
        channel._running = True
        channel._stop_event.set()
        real_import = builtins.__import__

        fake_ws = types.SimpleNamespace(__version__="8.1")

        def _patched_import(name, *args, **kwargs):
            if name == "websockets":
                return fake_ws
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_patched_import),
            patch("annolid.core.agent.channels.whatsapp.logger.error") as mocked_error,
        ):
            await asyncio.wait_for(channel._run_bridge_loop(), timeout=1.0)
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


def test_whatsapp_python_bridge_start_rejects_incompatible_websockets() -> None:
    async def _run() -> None:
        import types

        bridge = WhatsAppPythonBridge()
        real_import = builtins.__import__

        fake_ws = types.SimpleNamespace(__version__="8.1")

        def _patched_import(name, *args, **kwargs):
            if name == "websockets":
                return fake_ws
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_patched_import):
            with pytest.raises(RuntimeError) as excinfo:
                await bridge.start()
        assert "websockets>=12" in str(excinfo.value)

    asyncio.run(_run())


def test_whatsapp_python_provider_dispatch_media_send_uses_caption_enter() -> None:
    async def _run() -> None:
        async def _on_event(_: dict[str, str]) -> None:
            return

        provider = _PlaywrightWhatsAppProvider(
            session_dir="~/.annolid/whatsapp-web-session",
            headless=True,
            on_event=_on_event,
        )

        page = MagicMock()
        provider._page = page
        page.keyboard = MagicMock()
        page.keyboard.press = AsyncMock()
        page.wait_for_selector = AsyncMock()

        sent = {"done": False}
        caption_box = MagicMock()

        async def _caption_press(key: str) -> None:
            assert key == "Enter"
            sent["done"] = True

        caption_box.press = AsyncMock(side_effect=_caption_press)

        async def _query_selector(selector: str):
            if selector == provider.SELECTORS["caption_box"]:
                return None if sent["done"] else caption_box
            if selector == provider.SELECTORS["footer_composer"]:
                return MagicMock() if sent["done"] else None
            if selector == provider.SELECTORS["media_preview"]:
                return None
            return None

        page.query_selector = AsyncMock(side_effect=_query_selector)
        provider._click_with_fallback = AsyncMock()

        await provider._dispatch_media_send("Message yourself")

        assert caption_box.press.called
        assert not page.keyboard.press.called
        assert not page.wait_for_selector.called

    asyncio.run(_run())


def test_whatsapp_python_provider_dispatch_media_send_falls_back_to_button() -> None:
    async def _run() -> None:
        async def _on_event(_: dict[str, str]) -> None:
            return

        provider = _PlaywrightWhatsAppProvider(
            session_dir="~/.annolid/whatsapp-web-session",
            headless=True,
            on_event=_on_event,
        )

        page = MagicMock()
        provider._page = page
        page.keyboard = MagicMock()
        page.keyboard.press = AsyncMock(side_effect=RuntimeError("enter failed"))

        sent = {"done": False}
        caption_box = MagicMock()
        caption_box.press = AsyncMock(side_effect=RuntimeError("caption enter failed"))
        send_btn = MagicMock()

        async def _query_selector(selector: str):
            if selector == provider.SELECTORS["caption_box"]:
                return None if sent["done"] else caption_box
            if selector == provider.SELECTORS["footer_composer"]:
                return MagicMock() if sent["done"] else None
            if selector == provider.SELECTORS["media_preview"]:
                return MagicMock() if not sent["done"] else None
            return None

        async def _click_send_btn(*args, **kwargs) -> None:
            sent["done"] = True

        page.query_selector = AsyncMock(side_effect=_query_selector)
        page.wait_for_selector = AsyncMock(return_value=send_btn)
        provider._click_with_fallback = AsyncMock(side_effect=_click_send_btn)

        await provider._dispatch_media_send("Message yourself")

        assert page.wait_for_selector.called
        assert provider._click_with_fallback.called

    asyncio.run(_run())


def test_whatsapp_channel_bridge_send_media(tmp_path: Path) -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel(
            {"bridge_url": "ws://127.0.0.1:3001"},
            bus,
        )

        # Mock the websocket
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        channel._bridge_ws = mock_ws

        media_file = tmp_path / "test.png"
        media_file.write_bytes(b"fake-image")
        msg = OutboundMessage(
            channel="whatsapp",
            chat_id="12345",
            content="here is an image",
            media=[str(media_file)],
        )

        await channel.send(msg)

        assert mock_ws.send.called
        sent_args = mock_ws.send.call_args[0][0]
        data = json.loads(sent_args)
        assert data["type"] == "send_media"
        assert data["to"] == "12345"
        assert data["media"] == [str(media_file.resolve())]
        assert data["caption"] == "here is an image"

    asyncio.run(_run())


def test_whatsapp_channel_bridge_virtual_media_ref_falls_back_to_text() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel(
            {"bridge_url": "ws://127.0.0.1:3001"},
            bus,
        )
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock()
        channel._bridge_ws = mock_ws

        msg = OutboundMessage(
            channel="whatsapp",
            chat_id="Message yourself",
            content="camera stream is reachable",
            media=["wa-bridge-media:image:true_123_out"],
        )
        await channel.send(msg)

        assert mock_ws.send.called
        payload = json.loads(mock_ws.send.call_args[0][0])
        assert payload["type"] == "send"
        assert payload["text"] == "camera stream is reachable"

    asyncio.run(_run())


def test_whatsapp_channel_bridge_self_outgoing_ingested_when_enabled() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel(
            {"bridge_url": "ws://127.0.0.1:3001", "ingest_outgoing_messages": True},
            bus,
        )
        await channel._handle_bridge_message(
            '{"type":"message","id":"m-self-out","sender":"Message yourself","chat_id":"Message yourself","content":"what do you see from camera?","direction":"out","timestamp":123,"isGroup":false}'
        )
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.sender_id == "Message yourself"
        assert inbound.chat_id == "Message yourself"
        assert inbound.content == "what do you see from camera?"
        assert inbound.metadata.get("bridge_direction") == "out"

    asyncio.run(_run())


def test_whatsapp_channel_bridge_self_outgoing_echo_suppressed() -> None:
    async def _run() -> None:
        bus = MessageBus()
        channel = WhatsAppChannel(
            {"bridge_url": "ws://127.0.0.1:3001", "ingest_outgoing_messages": True},
            bus,
        )
        channel._remember_sent_content("bot echo")
        await channel._handle_bridge_message(
            '{"type":"message","id":"m-self-out-echo","sender":"Message yourself","chat_id":"Message yourself","content":"bot echo","direction":"out","timestamp":123,"isGroup":false}'
        )
        with pytest.raises(asyncio.TimeoutError):
            await bus.consume_inbound(timeout_s=0.2)

    asyncio.run(_run())


def test_whatsapp_channel_webhook_video_with_caption_includes_media_metadata() -> None:
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
                                "messages": [
                                    {
                                        "id": "wamid.VIDEO1",
                                        "from": "15551234567",
                                        "type": "video",
                                        "video": {
                                            "id": "media123",
                                            "caption": "see this clip",
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        }
        ingested = await channel.ingest_webhook_payload(payload)
        assert ingested == 1
        inbound = await bus.consume_inbound(timeout_s=0.2)
        assert inbound.content == "see this clip"
        assert inbound.media == ["wa-media:media123"]
        assert inbound.metadata.get("whatsapp_message_type") == "video"
        assert inbound.metadata.get("has_media") is True

    asyncio.run(_run())
