from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from annolid.core.agent.loop import AgentLoop, AgentMemoryConfig
from annolid.core.agent.tools.function_builtin import WebSearchTool
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


class _SearchLikeTool(FunctionTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for online information and return brief results."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or question.",
                }
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"search:{kwargs.get('query', '')}"


class _MathLikeTool(FunctionTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Compute arithmetic expressions and return numeric results."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like 2+2.",
                }
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"calc:{kwargs.get('expression', '')}"


def test_agent_loop_runs_tool_then_finishes() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del tools, model
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        assert any(m.get("role") == "tool" for m in messages)
        return {"content": "done", "tool_calls": []}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("hi"))
    assert result.content == "done"
    assert result.iterations == 2
    assert len(result.tool_runs) == 1
    assert result.tool_runs[0].result == "tool:hello"


def test_agent_loop_handles_string_tool_arguments() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
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
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "echo",
                            "arguments": '{"text": "json"}',
                        },
                    }
                ],
            }
        return {"content": "ok", "tool_calls": []}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("hello"))
    assert result.content == "ok"
    assert result.tool_runs[0].arguments["text"] == "json"


def test_agent_loop_respects_max_iterations() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        return {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"text": "x"}}],
        }

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        max_iterations=2,
    )
    result = asyncio.run(loop.run("start"))
    assert result.stopped_reason == "max_iterations"
    assert result.iterations == 2


def test_agent_loop_persists_history_per_session() -> None:
    registry = FunctionToolRegistry()
    state = {"calls": 0, "seen_prev_assistant": False}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del tools, model
        state["calls"] += 1
        if state["calls"] == 2:
            state["seen_prev_assistant"] = any(
                m.get("role") == "assistant" and m.get("content") == "first-reply"
                for m in messages
            )
        return {"content": "first-reply" if state["calls"] == 1 else "second-reply"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("first", session_id="s1"))
    _ = asyncio.run(loop.run("second", session_id="s1"))
    assert state["seen_prev_assistant"] is True
    hist = loop.get_session_history("s1")
    assert len(hist) == 4
    assert hist[-1]["content"] == "second-reply"


def test_agent_loop_memory_facts_injected() -> None:
    registry = FunctionToolRegistry()
    observed = {"has_memory_fact": False}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del tools, model
        observed["has_memory_fact"] = any(
            m.get("role") == "system"
            and "preferred_language: zh" in str(m.get("content") or "")
            for m in messages
        )
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        memory_config=AgentMemoryConfig(
            enabled=True, include_facts_in_system_prompt=True
        ),
    )
    loop.remember("s2", "preferred_language", "zh")
    _ = asyncio.run(loop.run("hello", session_id="s2"))
    assert observed["has_memory_fact"] is True
    assert loop.recall("s2", "preferred_language") == "zh"


def test_agent_loop_does_not_persist_empty_assistant_reply() -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        return {"content": ""}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("hello", session_id="s-empty"))
    assert result.content == ""
    assert loop.get_session_history("s-empty") == []


def test_agent_loop_sanitizes_and_deduplicates_tool_calls() -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    raw_calls = [
        {"id": "c1", "name": "echo", "arguments": {"text": "x"}},
        {"id": "c1", "name": "echo", "arguments": {"text": "x"}},
        {"id": "", "name": "echo", "arguments": '{"text":"y"}'},
        {"id": "c2", "name": "", "arguments": {}},
    ]
    cleaned = loop._sanitize_tool_calls(raw_calls)
    assert len(cleaned) == 2
    assert cleaned[0]["id"] == "c1"
    assert cleaned[0]["name"] == "echo"
    assert cleaned[1]["name"] == "echo"
    assert cleaned[1]["arguments"]["text"] == "y"


def test_agent_loop_stops_on_repeated_identical_tool_cycles() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
        return {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"text": "x"}}],
        }

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        max_iterations=8,
    )
    result = asyncio.run(loop.run("start"))
    assert result.stopped_reason == "repeated_tool_calls"
    assert result.iterations == 3
    assert "stalled" in result.content


def test_agent_loop_toolcall_web_search_example() -> None:
    registry = FunctionToolRegistry()
    registry.register(WebSearchTool(api_key=""))
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del model
        state["n"] += 1
        if state["n"] == 1:
            assert any(
                isinstance(t.get("function"), Mapping)
                and t["function"].get("name") == "web_search"
                for t in tools
            )
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search_1",
                        "name": "web_search",
                        "arguments": {"query": "annolid github"},
                    }
                ],
            }
        assert any(m.get("role") == "tool" for m in messages)
        return {"content": "Search step completed."}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("Find Annolid online."))
    assert result.content == "Search step completed."
    assert len(result.tool_runs) == 1
    assert result.tool_runs[0].name == "web_search"
    assert "BRAVE_API_KEY not configured" in result.tool_runs[0].result


def test_agent_loop_selects_relevant_tools_by_description() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SearchLikeTool())
    registry.register(_MathLikeTool())
    registry.register(_EchoTool())
    observed = {"tool_names": []}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, model
        observed["tool_names"] = [
            str((t.get("function") or {}).get("name") or "") for t in tools
        ]
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("Please search online for annolid docs"))
    assert "web_search" in observed["tool_names"]
    assert "calculator" not in observed["tool_names"]


def test_agent_loop_uses_compact_default_tool_subset_for_low_signal_prompt() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SearchLikeTool())
    registry.register(_MathLikeTool())
    registry.register(_EchoTool())

    def _make_tmp_tool(index: int) -> FunctionTool:
        class _TmpTool(FunctionTool):
            @property
            def name(self) -> str:  # type: ignore[override]
                return f"tool_{index}"

            @property
            def description(self) -> str:
                return f"Tool number {index}"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}, "required": []}

            async def execute(self, **kwargs: Any) -> str:
                del kwargs
                return "ok"

        return _TmpTool()

    for idx in range(8):
        registry.register(_make_tmp_tool(idx))

    observed = {"tool_names": []}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, model
        observed["tool_names"] = [
            str((t.get("function") or {}).get("name") or "") for t in tools
        ]
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("hi"))
    assert len(observed["tool_names"]) <= 6
    assert "echo" in observed["tool_names"] or "web_search" in observed["tool_names"]
