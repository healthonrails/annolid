from __future__ import annotations

import asyncio
from types import MethodType
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
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


class _EchoTool(FunctionTool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo tool."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"tool:{kwargs.get('text', '')}"


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
            assert out.metadata.get("turn_status") == "completed"
            assert out.metadata.get("error_type") == "none"
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_strips_model_think_blocks_from_outbound() -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        return {"content": "<think>internal reasoning</think>\n\nCamera OK."}

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
                    content="check camera",
                )
            )
            out = await bus.consume_outbound(timeout_s=1.0)
            assert out.content == "Camera OK."
            assert "<think>" not in out.content
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_streams_intermediate_progress() -> None:
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "Inspecting local files",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        return {"content": "done"}

    async def _run() -> None:
        bus = MessageBus()
        registry = FunctionToolRegistry()
        registry.register(_EchoTool())
        loop = AgentLoop(
            tools=registry,
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
            progress = await bus.consume_outbound(timeout_s=1.0)
            final = await bus.consume_outbound(timeout_s=1.0)

            assert progress.content == "Inspecting local files"
            assert bool(progress.metadata.get("intermediate")) is True
            assert bool(progress.metadata.get("progress")) is True
            assert final.content == "done"
            assert bool(final.metadata.get("intermediate")) is False
            assert final.metadata.get("iterations") == 2
            assert final.metadata.get("turn_status") == "completed"
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


def test_agent_bus_service_assigns_error_taxonomy_for_failures() -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        raise RuntimeError("boom")

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
            assert out.content.startswith("Error:")
            assert out.metadata.get("turn_status") == "failed"
            assert out.metadata.get("error_type") == "internal_error"
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
        cfg.agents.defaults.max_parallel_sessions = 3
        cfg.agents.defaults.max_pending_messages = 1024
        cfg.agents.defaults.collapse_superseded_pending = False
        cfg.agents.defaults.transient_retry_attempts = 5
        cfg.agents.defaults.transient_retry_initial_backoff_s = 0.25
        cfg.agents.defaults.transient_retry_max_backoff_s = 3.0
        svc = AgentBusService.from_agent_config(bus=bus, loop=loop, agent_config=cfg)
        assert svc._max_parallel_sessions == 3
        assert svc._max_pending_messages == 1024
        assert svc._collapse_superseded_pending is False
        assert svc._transient_retry_attempts == 5
        assert svc._transient_retry_initial_backoff_s == 0.25
        assert svc._transient_retry_max_backoff_s == 3.0
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


def test_agent_bus_service_scheduler_overflow_returns_transport_error() -> None:
    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=lambda *_args, **_kwargs: {"content": "unused"},
            model="fake",
        )
        svc = AgentBusService(
            bus=bus,
            loop=loop,
            max_parallel_sessions=1,
            max_pending_messages=1,
        )

        async def _fake_process(
            self: AgentBusService,
            inbound: InboundMessage,
            *,
            session_key: str | None = None,
        ) -> None:
            del session_key
            await asyncio.sleep(0.2)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=str(inbound.content),
                )
            )

        svc._process_inbound = MethodType(_fake_process, svc)
        await svc.start()
        try:
            for idx in range(6):
                await bus.publish_inbound(
                    InboundMessage(
                        channel="telegram",
                        sender_id=f"alice-{idx}",
                        chat_id=f"chat-{idx}",
                        content=f"m-{idx}",
                    )
                )

            messages: list[OutboundMessage] = []
            for _ in range(6):
                try:
                    msg = await bus.consume_outbound(timeout_s=1.0)
                    messages.append(msg)
                except TimeoutError:
                    break

            dropped = [
                m
                for m in messages
                if bool((m.metadata or {}).get("dropped_by_scheduler"))
            ]
            assert dropped
            assert all(
                (m.metadata or {}).get("error_type") == "transport_error"
                for m in dropped
            )
            assert all(
                (m.metadata or {}).get("turn_status") == "failed" for m in dropped
            )
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_scheduler_serializes_same_session() -> None:
    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=lambda *_args, **_kwargs: {"content": "unused"},
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop, max_parallel_sessions=2)

        active_by_session: dict[str, int] = {}
        max_active_by_session: dict[str, int] = {}

        async def _fake_process(
            self: AgentBusService,
            inbound: InboundMessage,
            *,
            session_key: str | None = None,
        ) -> None:
            key = str(session_key or "missing")
            active = active_by_session.get(key, 0) + 1
            active_by_session[key] = active
            max_active_by_session[key] = max(max_active_by_session.get(key, 0), active)
            await asyncio.sleep(0.03)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=str(inbound.content),
                )
            )
            active_by_session[key] = max(0, active_by_session.get(key, 1) - 1)

        svc._process_inbound = MethodType(_fake_process, svc)
        await svc.start()
        try:
            await bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="alice",
                    chat_id="chat-1",
                    content="first",
                )
            )
            await bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="alice",
                    chat_id="chat-1",
                    content="second",
                )
            )
            out1 = await bus.consume_outbound(timeout_s=1.0)
            out2 = await bus.consume_outbound(timeout_s=1.0)
            assert out1.content == "first"
            assert out2.content == "second"
            assert max(max_active_by_session.values() or [0]) == 1
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_scheduler_runs_sessions_in_parallel() -> None:
    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=lambda *_args, **_kwargs: {"content": "unused"},
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop, max_parallel_sessions=2)

        active = 0
        max_active = 0

        async def _fake_process(
            self: AgentBusService,
            inbound: InboundMessage,
            *,
            session_key: str | None = None,
        ) -> None:
            del session_key
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=str(inbound.content),
                )
            )
            active = max(0, active - 1)

        svc._process_inbound = MethodType(_fake_process, svc)
        await svc.start()
        try:
            await bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="alice",
                    chat_id="chat-1",
                    content="first",
                )
            )
            await bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="bob",
                    chat_id="chat-2",
                    content="second",
                )
            )
            _ = await bus.consume_outbound(timeout_s=1.0)
            _ = await bus.consume_outbound(timeout_s=1.0)
            assert max_active >= 2
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_collapses_superseded_pending_prompts() -> None:
    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=lambda *_args, **_kwargs: {"content": "unused"},
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop, max_parallel_sessions=1)

        async def _fake_process(
            self: AgentBusService,
            inbound: InboundMessage,
            *,
            session_key: str | None = None,
        ) -> None:
            del session_key
            await asyncio.sleep(0.05)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=str(inbound.content),
                )
            )

        svc._process_inbound = MethodType(_fake_process, svc)
        await svc.start()
        try:
            await bus.publish_inbound(
                InboundMessage(
                    channel="gui",
                    sender_id="u1",
                    chat_id="chat-1",
                    content="first",
                )
            )
            await asyncio.sleep(0.01)
            await bus.publish_inbound(
                InboundMessage(
                    channel="gui",
                    sender_id="u1",
                    chat_id="chat-1",
                    content="second",
                )
            )
            await bus.publish_inbound(
                InboundMessage(
                    channel="gui",
                    sender_id="u1",
                    chat_id="chat-1",
                    content="third",
                )
            )
            out1 = await bus.consume_outbound(timeout_s=1.0)
            out2 = await bus.consume_outbound(timeout_s=1.0)
            assert out1.content == "first"
            assert out2.content == "third"
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_preserves_pending_when_queue_keep_all() -> None:
    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=lambda *_args, **_kwargs: {"content": "unused"},
            model="fake",
        )
        svc = AgentBusService(bus=bus, loop=loop, max_parallel_sessions=1)

        async def _fake_process(
            self: AgentBusService,
            inbound: InboundMessage,
            *,
            session_key: str | None = None,
        ) -> None:
            del session_key
            await asyncio.sleep(0.03)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=str(inbound.content),
                )
            )

        svc._process_inbound = MethodType(_fake_process, svc)
        await svc.start()
        try:
            for text in ("first", "second", "third"):
                await bus.publish_inbound(
                    InboundMessage(
                        channel="gui",
                        sender_id="u1",
                        chat_id="chat-1",
                        content=text,
                        metadata={"queue_keep_all": True},
                    )
                )
            out = [await bus.consume_outbound(timeout_s=1.0) for _ in range(3)]
            assert [m.content for m in out] == ["first", "second", "third"]
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_retries_transient_failures_with_backoff() -> None:
    calls = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("provider timeout")
        return {"content": "ok-after-retry"}

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        svc = AgentBusService(
            bus=bus,
            loop=loop,
            transient_retry_attempts=2,
            transient_retry_initial_backoff_s=0.01,
            transient_retry_max_backoff_s=0.02,
        )
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
            assert out.content == "ok-after-retry"
            assert calls["n"] == 2
        finally:
            await svc.stop()

    asyncio.run(_run())


def test_agent_bus_service_marks_exhausted_transient_as_transport_error() -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        raise TimeoutError("still timing out")

    async def _run() -> None:
        bus = MessageBus()
        loop = AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
        )
        svc = AgentBusService(
            bus=bus,
            loop=loop,
            transient_retry_attempts=1,
            transient_retry_initial_backoff_s=0.01,
            transient_retry_max_backoff_s=0.02,
        )
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
            assert out.content.startswith("Error:")
            assert out.metadata.get("error_type") == "transport_error"
            assert out.metadata.get("turn_status") == "failed"
        finally:
            await svc.stop()

    asyncio.run(_run())
