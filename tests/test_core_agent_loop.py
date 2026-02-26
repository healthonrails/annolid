from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from annolid.core.agent.loop import AgentLoop, AgentMemoryConfig
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
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


class _BrowserLikeTool(FunctionTool):
    @property
    def name(self) -> str:
        return "mcp_browser_navigate"

    @property
    def description(self) -> str:
        return "Navigate browser to a URL and inspect page content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"browser:{kwargs.get('url', '')}"


class _SlowTool(FunctionTool):
    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "Slow tool."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        await asyncio.sleep(0.05)
        return "slow-done"


class _ExecTool(FunctionTool):
    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute shell command."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"exec:{kwargs.get('command', '')}"


class _EmailTool(FunctionTool):
    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        return "Send email."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"email:{kwargs.get('to', '')}"


def test_agent_loop_runs_tool_then_finishes() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
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


def test_agent_loop_blocks_high_risk_tool_combo_without_explicit_intent() -> None:
    registry = FunctionToolRegistry()
    registry.register(_ExecTool())
    registry.register(_EmailTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "exec", "arguments": {"command": "ls"}}
                ],
            }
        if state["n"] == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "email",
                        "arguments": {"to": "a@example.com", "body": "hello"},
                    }
                ],
            }
        return {"content": "done"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("please do the task"))
    assert result.content == "done"
    assert len(result.tool_runs) == 2
    assert result.tool_runs[0].result.startswith("exec:")
    assert "Blocked by safety policy" in result.tool_runs[1].result


def test_agent_loop_allows_high_risk_tool_combo_with_explicit_intent() -> None:
    registry = FunctionToolRegistry()
    registry.register(_ExecTool())
    registry.register(_EmailTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "exec", "arguments": {"command": "ls"}}
                ],
            }
        if state["n"] == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "email",
                        "arguments": {"to": "a@example.com", "body": "hello"},
                    }
                ],
            }
        return {"content": "done"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(
        loop.run("intent:high-risk please run command and email summary")
    )
    assert result.content == "done"
    assert len(result.tool_runs) == 2
    assert result.tool_runs[0].result.startswith("exec:")
    assert result.tool_runs[1].result.startswith("email:")


def test_agent_loop_allows_high_risk_combo_when_guard_disabled() -> None:
    registry = FunctionToolRegistry()
    registry.register(_ExecTool())
    registry.register(_EmailTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "exec", "arguments": {"command": "ls"}}
                ],
            }
        if state["n"] == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "email",
                        "arguments": {"to": "a@example.com", "body": "hello"},
                    }
                ],
            }
        return {"content": "done"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        strict_runtime_tool_guard=False,
    )
    result = asyncio.run(loop.run("please do the task"))
    assert result.content == "done"
    assert len(result.tool_runs) == 2
    assert result.tool_runs[0].result.startswith("exec:")
    assert result.tool_runs[1].result.startswith("email:")


def test_agent_loop_redacts_session_values_in_contextual_prompt(tmp_path: Path) -> None:
    observed = {"system": ""}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        observed["system"] = str(messages[0].get("content") or "")
        return {"content": "ok"}

    loop = AgentLoop(
        tools=FunctionToolRegistry(),
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
    )
    _ = asyncio.run(
        loop.run(
            "hello", channel="email", chat_id="person@example.com", use_memory=False
        )
    )
    assert "person@example.com" not in observed["system"]
    assert "pe***n@example.com" in observed["system"]


def test_agent_loop_emits_intermediate_progress_for_tool_calls() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}
    progress_updates: list[str] = []

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "<think>internal</think>Need a tool call.",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        return {"content": "done"}

    async def _on_progress(text: str) -> None:
        progress_updates.append(text)

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("hi", on_progress=_on_progress))
    assert result.content == "done"
    assert progress_updates == ["Need a tool call."]


def test_agent_loop_progress_uses_tool_hint_when_content_empty() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}
    progress_updates: list[str] = []

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        return {"content": "done"}

    async def _on_progress(text: str) -> None:
        progress_updates.append(text)

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("hi", on_progress=_on_progress))
    assert progress_updates == ['echo("hello")']


def test_agent_loop_progress_accepts_sync_callback() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}
    progress_updates: list[str] = []

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        return {"content": "done"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("hi", on_progress=progress_updates.append))
    assert progress_updates == ['echo("hello")']


def test_agent_loop_timeout_raises_explicit_message() -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        await asyncio.sleep(0.05)
        return {"content": "late"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        llm_timeout_seconds=0.01,
    )
    try:
        _ = asyncio.run(loop.run("hi"))
        assert False, "Expected timeout"
    except TimeoutError as exc:
        assert "LLM timed out after" in str(exc)


def test_agent_loop_tool_timeout_returns_error_result() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SlowTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "slow_tool", "arguments": {}}],
            }
        return {"content": "done"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        tool_timeout_seconds=0.01,
    )
    result = asyncio.run(loop.run("start"))
    assert result.content == "done"
    assert len(result.tool_runs) == 1
    assert "timed out after" in result.tool_runs[0].result


def test_agent_loop_handles_string_tool_arguments() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
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


def test_agent_loop_remember_persists_long_term_memory(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
    )
    loop.remember("s3", "favorite_animal", "mouse")

    memory_path = tmp_path / "memory" / "MEMORY.md"
    assert memory_path.exists()
    content = memory_path.read_text(encoding="utf-8")
    assert "- favorite_animal: mouse" in content


def test_agent_loop_remember_prompt_persists_note_to_long_term_memory(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "Noted."}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
    )
    _ = asyncio.run(loop.run("remember you are annolid bot", session_id="s4"))

    memory_path = tmp_path / "memory" / "MEMORY.md"
    assert memory_path.exists()
    content = memory_path.read_text(encoding="utf-8")
    assert "- you are annolid bot" in content


def test_agent_loop_does_not_persist_empty_assistant_reply() -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": ""}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("hello", session_id="s-empty"))
    assert result.content == ""
    assert loop.get_session_history("s-empty") == []


def test_agent_loop_consolidates_large_history_into_history_file(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        state["calls"] += 1
        if (
            len(messages) == 1
            and messages[0].get("role") == "system"
            and "Consolidate the archived chat transcript"
            in str(messages[0].get("content") or "")
        ):
            return {
                "content": (
                    '{"history_entry":"[2026-01-01 10:00] '
                    'Consolidated earlier turns.","memory_update":"- user_pref: fast"}'
                )
            }
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
        memory_config=AgentMemoryConfig(
            enabled=True,
            max_history_messages=64,
            memory_window=6,
            include_facts_in_system_prompt=True,
        ),
    )
    for i in range(4):
        _ = asyncio.run(loop.run(f"turn-{i}", session_id="c1"))

    _ = asyncio.run(loop.run("trigger consolidate", session_id="c1"))
    history_path = tmp_path / "memory" / "HISTORY.md"
    assert history_path.exists()
    assert "Consolidated earlier turns" in history_path.read_text(encoding="utf-8")
    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "user_pref: fast" in memory_text
    assert len(loop.get_session_history("c1")) <= 10
    assert state["calls"] >= 2


def test_agent_loop_consolidation_accepts_save_memory_tool_call(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del model, on_token
        state["calls"] += 1
        if (
            len(messages) == 1
            and messages[0].get("role") == "system"
            and "Consolidate the archived chat transcript"
            in str(messages[0].get("content") or "")
        ):
            assert any(
                isinstance(t.get("function"), Mapping)
                and t["function"].get("name") == "save_memory"
                for t in tools
            )
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "mem_1",
                        "name": "save_memory",
                        "arguments": {
                            "history_entry": "[2026-01-01 11:00] Consolidated via tool call.",
                            "memory_update": "- project: annolid",
                        },
                    }
                ],
            }
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
        memory_config=AgentMemoryConfig(
            enabled=True,
            max_history_messages=64,
            memory_window=6,
            include_facts_in_system_prompt=True,
        ),
    )
    for i in range(5):
        _ = asyncio.run(loop.run(f"turn-{i}", session_id="c2"))

    history_path = tmp_path / "memory" / "HISTORY.md"
    assert history_path.exists()
    assert "Consolidated via tool call." in history_path.read_text(encoding="utf-8")
    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "project: annolid" in memory_text
    assert state["calls"] >= 2


def test_agent_loop_consolidation_parses_json_after_think_block(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        state["calls"] += 1
        if (
            len(messages) == 1
            and messages[0].get("role") == "system"
            and "Consolidate the archived chat transcript"
            in str(messages[0].get("content") or "")
        ):
            return {
                "content": (
                    "<think>internal note</think>\n"
                    '{"history_entry":"[2026-01-01 12:00] Parsed after think.",'
                    '"memory_update":"- camera: wireless"}'
                )
            }
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
        memory_config=AgentMemoryConfig(
            enabled=True,
            max_history_messages=64,
            memory_window=6,
            include_facts_in_system_prompt=True,
        ),
    )
    for i in range(5):
        _ = asyncio.run(loop.run(f"turn-{i}", session_id="c3"))

    history_path = tmp_path / "memory" / "HISTORY.md"
    assert history_path.exists()
    assert "Parsed after think." in history_path.read_text(encoding="utf-8")
    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "camera: wireless" in memory_text
    assert state["calls"] >= 2


def test_agent_loop_consolidation_skips_llm_for_short_transcript(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()
    state = {"calls": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        state["calls"] += 1
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
        memory_config=AgentMemoryConfig(
            enabled=True,
            max_history_messages=64,
            memory_window=4,
            include_facts_in_system_prompt=True,
        ),
    )
    short_history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ]
    kept = asyncio.run(
        loop._consolidate_memory(session_id="short", history=short_history)
    )
    assert len(kept) == 2
    assert state["calls"] == 0
    history_path = tmp_path / "memory" / "HISTORY.md"
    assert history_path.exists()
    history_text = history_path.read_text(encoding="utf-8")
    assert "USER: a" in history_text or "Session history consolidated." in history_text


def test_agent_loop_consolidation_flushes_archive_to_daily_memory_before_compaction(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        if (
            len(messages) == 1
            and messages[0].get("role") == "system"
            and "Consolidate the archived chat transcript"
            in str(messages[0].get("content") or "")
        ):
            raise RuntimeError("forced consolidation failure")
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
        memory_config=AgentMemoryConfig(
            enabled=True,
            max_history_messages=64,
            memory_window=4,
            include_facts_in_system_prompt=True,
        ),
    )
    history = [
        {"role": "user", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
        {"role": "user", "content": "gamma"},
        {"role": "assistant", "content": "delta"},
        {"role": "user", "content": "epsilon"},
    ]
    kept = asyncio.run(loop._consolidate_memory(session_id="flush-1", history=history))
    assert len(kept) == 2

    today_file = tmp_path / "memory" / f"{date.today().strftime('%Y-%m-%d')}.md"
    assert today_file.exists()
    daily_text = today_file.read_text(encoding="utf-8")
    assert "Pre-compaction Memory Flush" in daily_text
    assert "USER: alpha" in daily_text


def test_agent_loop_persists_tools_used_metadata() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        assert any(m.get("role") == "tool" for m in messages)
        return {"content": "done"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("hi", session_id="tools-meta"))
    hist = loop.get_session_history("tools-meta")
    assert hist[-1]["role"] == "assistant"
    assert hist[-1]["tools_used"] == ["echo"]


def test_agent_loop_sanitizes_and_deduplicates_tool_calls() -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del model, on_token
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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, model, on_token
        observed["tool_names"] = [
            str((t.get("function") or {}).get("name") or "") for t in tools
        ]
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("Please search online for annolid docs"))
    assert "web_search" in observed["tool_names"]
    assert "calculator" not in observed["tool_names"]


def test_agent_loop_prefers_browser_tool_for_web_intent_queries() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SearchLikeTool())
    registry.register(_BrowserLikeTool())
    registry.register(_MathLikeTool())
    observed = {"tool_names": []}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, model, on_token
        observed["tool_names"] = [
            str((t.get("function") or {}).get("name") or "") for t in tools
        ]
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("what is the weather today?"))
    assert observed["tool_names"]
    assert observed["tool_names"][0] == "mcp_browser_navigate"


def test_agent_loop_can_disable_browser_first_for_web() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SearchLikeTool())
    registry.register(_BrowserLikeTool())
    registry.register(_MathLikeTool())
    observed = {"tool_names": []}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, model, on_token
        observed["tool_names"] = [
            str((t.get("function") or {}).get("name") or "") for t in tools
        ]
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        browser_first_for_web=False,
    )
    _ = asyncio.run(loop.run("what is the weather today?"))
    assert observed["tool_names"]
    assert observed["tool_names"][0] == "web_search"


def test_agent_loop_inserts_post_tool_system_guidance() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0, "saw_guidance": False}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "hello"}}
                ],
            }
        state["saw_guidance"] = any(
            m.get("role") == "system"
            and "Use the tool results to decide the next best action."
            in str(m.get("content") or "")
            for m in messages
        )
        assert not any(
            m.get("role") == "user"
            and "Reflect on the results and decide next steps."
            in str(m.get("content") or "")
            for m in messages
        )
        return {"content": "done"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("hi"))
    assert result.content == "done"
    assert state["saw_guidance"] is True


def test_agent_loop_can_disable_post_tool_guidance() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    state = {"n": 0, "saw_guidance": False}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        state["n"] += 1
        if state["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "echo", "arguments": {"text": "x"}}
                ],
            }
        state["saw_guidance"] = any(
            m.get("role") == "system"
            and "Use the tool results to decide the next best action."
            in str(m.get("content") or "")
            for m in messages
        )
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        interleave_post_tool_guidance=False,
    )
    result = asyncio.run(loop.run("start"))
    assert result.content == "ok"
    assert state["saw_guidance"] is False


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
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, model, on_token
        observed["tool_names"] = [
            str((t.get("function") or {}).get("name") or "") for t in tools
        ]
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    _ = asyncio.run(loop.run("hi"))
    assert len(observed["tool_names"]) <= 6
    assert "echo" in observed["tool_names"] or "web_search" in observed["tool_names"]


def test_agent_loop_records_memory_telemetry_and_turn_counters(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "ok"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
        memory_config=AgentMemoryConfig(
            enabled=True,
            max_history_messages=64,
            memory_window=4,
            include_facts_in_system_prompt=True,
        ),
    )
    for i in range(4):
        _ = asyncio.run(loop.run(f"turn-{i}", session_id="telemetry"))
    meta = loop._memory_store.get_session_metadata("telemetry")  # type: ignore[attr-defined]
    assert int(meta.get("turn_counter") or 0) >= 4
    assert int(meta.get("next_consolidation_turn") or 0) >= 1
    rows = list(meta.get("memory_telemetry") or [])
    assert rows


def test_agent_loop_records_inbound_outbound_events_for_replay(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "ok"}

    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        memory_store=store,
    )
    _ = asyncio.run(loop.run("hello", session_id="replay-1"))
    rows = store.replay_events("replay-1", limit=10)
    assert len(rows) >= 2
    directions = {str(item.get("direction") or "") for item in rows}
    assert "inbound" in directions
    assert "outbound" in directions


def test_agent_loop_captures_anonymized_run_trace(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "done", "tool_calls": []}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
    )
    result = asyncio.run(loop.run("hello user@example.com", session_id="s1"))
    assert result.content == "done"
    trace_path = tmp_path / "eval" / "run_traces.ndjson"
    assert trace_path.exists()
    payload = trace_path.read_text(encoding="utf-8")
    assert "trace_id" in payload
    assert "user@example.com" not in payload


def test_agent_loop_shadow_routing_mode_writes_shadow_log(
    tmp_path: Path, monkeypatch
) -> None:
    registry = FunctionToolRegistry()
    registry.register(_SearchLikeTool())
    registry.register(_MathLikeTool())
    monkeypatch.setenv("ANNOLID_AGENT_SHADOW_MODE", "1")
    monkeypatch.setenv("ANNOLID_AGENT_SHADOW_ROUTING_POLICY", "default")

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "done", "tool_calls": []}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        workspace=str(tmp_path),
    )
    _ = asyncio.run(loop.run("search web", session_id="s-shadow"))
    shadow_path = tmp_path / "eval" / "shadow_routing.ndjson"
    assert shadow_path.exists()
