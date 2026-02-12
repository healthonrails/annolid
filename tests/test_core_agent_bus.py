from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from annolid.core.agent.bus import (
    AgentBusService,
    InboundMessage,
    MessageBus,
    OutboundMessage,
)
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
        finally:
            await svc.stop()

    asyncio.run(_run())
