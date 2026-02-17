from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from annolid.core.agent.bus import (
    AgentBusService,
    EventFrame,
    InboundMessage,
    MessageBus,
    OutboundMessage,
    ProtocolValidationError,
    RequestFrame,
    ResponseFrame,
    parse_frame,
)
from annolid.core.agent.config import AgentConfig, SessionRoutingConfig
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


def test_message_bus_publish_consume_and_dispatch() -> None:
    async def _run() -> None:
        bus = MessageBus()

        inbound = InboundMessage(
            channel="slack",
            sender_id="u1",
            chat_id="c1",
            content="hello",
        )
        await bus.publish_inbound(inbound)
        got_inbound = await bus.consume_inbound(timeout_s=0.2)
        assert got_inbound.session_key == "slack:c1"

        seen: list[str] = []

        async def _subscriber(msg: OutboundMessage) -> None:
            seen.append(msg.content)

        bus.subscribe_outbound("slack", _subscriber)
        await bus.start_dispatcher()
        try:
            await bus.publish_outbound(
                OutboundMessage(channel="slack", chat_id="c1", content="reply")
            )
            for _ in range(20):
                if seen:
                    break
                await asyncio.sleep(0.01)
            assert seen == ["reply"]
        finally:
            await bus.stop_dispatcher()

    asyncio.run(_run())


def test_agent_bus_service_processes_inbound_to_outbound() -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del tools, model
        last = messages[-1]
        return {"content": f"echo:{last.get('content', '')}"}

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop)
        await svc.start()
        try:
            await bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="alice",
                    chat_id="chat-1",
                    content="ping",
                )
            )
            out = await bus.consume_outbound(timeout_s=1.0)
            assert out.channel == "telegram"
            assert out.chat_id == "chat-1"
            assert out.content == "echo:ping"
            assert out.metadata.get("iterations") == 1
            assert int(out.metadata.get("seq") or 0) >= 1
            assert int(out.metadata.get("state_version") or 0) >= 1
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_substitutes_empty_email_reply_with_fallback() -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        return {"content": "   "}

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop)
        await svc.start()
        try:
            await bus.publish_inbound(
                InboundMessage(
                    channel="email",
                    sender_id="alice@example.com",
                    chat_id="alice@example.com",
                    content="papers",
                    metadata={"subject": "Papers"},
                )
            )
            out = await bus.consume_outbound(timeout_s=1.0)
            assert out.channel == "email"
            assert out.chat_id == "alice@example.com"
            assert bool(str(out.content or "").strip())
            assert "Papers" in out.content
            assert bool(out.metadata.get("empty_reply_fallback")) is True
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_idempotency_replays_cached_result() -> None:
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        state["calls"] += 1
        return {"content": f"run:{state['calls']}"}

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop)
        await svc.start()
        try:
            for _ in range(2):
                await bus.publish_inbound(
                    InboundMessage(
                        channel="telegram",
                        sender_id="alice",
                        chat_id="chat-1",
                        content="ping",
                        metadata={"idempotency_key": "abc-123"},
                    )
                )
            out1 = await bus.consume_outbound(timeout_s=1.0)
            out2 = await bus.consume_outbound(timeout_s=1.0)
            assert out1.content == "run:1"
            assert out2.content == "run:1"
            assert bool(out2.metadata.get("idempotency_replay")) is True
            assert state["calls"] == 1
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_parse_frame_validates_req_res_event() -> None:
    req = parse_frame(
        {
            "type": "req",
            "id": "r1",
            "method": "chat.send",
            "params": {"content": "hi"},
            "idempotency_key": "k1",
        }
    )
    assert isinstance(req, RequestFrame)
    assert req.id == "r1"
    assert req.idempotency_key == "k1"

    res = parse_frame({"type": "res", "id": "r1", "ok": True, "payload": {"x": 1}})
    assert isinstance(res, ResponseFrame)
    assert res.ok is True

    evt = parse_frame({"type": "event", "event": "agent.delta", "payload": {"x": 1}})
    assert isinstance(evt, EventFrame)
    assert evt.event == "agent.delta"


def test_parse_frame_rejects_invalid_payloads() -> None:
    try:
        _ = parse_frame({"type": "req", "id": "", "method": "x", "params": {}})
        assert False, "Expected ProtocolValidationError"
    except ProtocolValidationError:
        pass
    try:
        _ = parse_frame({"type": "event", "event": "", "payload": {}})
        assert False, "Expected ProtocolValidationError"
    except ProtocolValidationError:
        pass


def test_inbound_message_session_key_supports_dm_scope() -> None:
    dm_msg = InboundMessage(
        channel="telegram",
        sender_id="alice",
        chat_id="dm-room",
        content="hello",
        metadata={
            "conversation_type": "dm",
            "session_dm_scope": "per-channel-peer",
            "channel_key": "private-thread",
            "peer_id": "peer-42",
        },
    )
    assert dm_msg.session_key == "telegram:private-thread:dm:peer-42"

    account_dm = InboundMessage(
        channel="slack",
        sender_id="alice",
        chat_id="im-1",
        content="hello",
        metadata={
            "conversation_type": "dm",
            "session_dm_scope": "per-account-channel-peer",
            "account_id": "workspace-a",
            "channel_key": "dm-channel",
            "peer_id": "U123",
        },
    )
    assert account_dm.session_key == "slack:workspace-a:dm-channel:dm:U123"


def test_agent_bus_service_idempotency_isolated_by_peer_scope() -> None:
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        state["calls"] += 1
        return {"content": f"run:{state['calls']}"}

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop, default_dm_scope="per-peer")
        await svc.start()
        try:
            for sender in ("alice", "bob"):
                await bus.publish_inbound(
                    InboundMessage(
                        channel="telegram",
                        sender_id=sender,
                        chat_id="shared-dm-chat",
                        content="ping",
                        metadata={
                            "conversation_type": "dm",
                            "idempotency_key": "abc-123",
                        },
                    )
                )
            out1 = await bus.consume_outbound(timeout_s=1.0)
            out2 = await bus.consume_outbound(timeout_s=1.0)
            assert out1.content == "run:1"
            assert out2.content == "run:2"
            assert bool(out2.metadata.get("idempotency_replay")) is False
            assert state["calls"] == 2
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_uses_session_defaults_from_config() -> None:
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        state["calls"] += 1
        return {"content": f"run:{state['calls']}"}

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        cfg = AgentConfig()
        cfg.agents.defaults.session = SessionRoutingConfig(dm_scope="per-peer")
        svc = AgentBusService.from_agent_config(bus=bus, loop=loop, agent_config=cfg)
        await svc.start()
        try:
            for sender in ("alice", "bob"):
                await bus.publish_inbound(
                    InboundMessage(
                        channel="telegram",
                        sender_id=sender,
                        chat_id="shared-dm-chat",
                        content="ping",
                        metadata={
                            "conversation_type": "dm",
                            "idempotency_key": "abc-123",
                        },
                    )
                )
            out1 = await bus.consume_outbound(timeout_s=1.0)
            out2 = await bus.consume_outbound(timeout_s=1.0)
            assert out1.content == "run:1"
            assert out2.content == "run:2"
            assert state["calls"] == 2
        finally:
            await svc.stop()

    asyncio.run(_run())
