from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

from annolid.core.agent.context_compaction import plan_context_compaction
from annolid.core.agent.gui_backend.response_finalize import (
    sanitize_final_response_text,
)
from annolid.core.agent.gui_backend.telemetry import wrap_tool_callback
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.providers.ollama_utils import (
    collect_ollama_stream,
    normalize_messages_for_ollama,
    parse_ollama_tool_calls,
)
from annolid.core.agent.providers.openai_codex_provider import (
    _convert_messages as codex_convert_messages,
)
from annolid.core.agent.providers.openai_compat import OpenAICompatProvider
from annolid.core.agent.providers.unified_provider import UnifiedLLMProvider
from annolid.core.agent.tool_call_utils import (
    extract_text_tool_calls,
    normalize_tool_arguments,
    strip_tool_call_markup,
)
from annolid.core.agent.tool_execution import (
    ThinkTagStreamParser,
    compact_tool_result_for_llm,
)
from annolid.core.agent.tool_selector import (
    compile_tool_index,
    score_tool_schema,
    select_relevant_tool_definitions,
    tokenize_text,
)
from annolid.core.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.core.agent.tools.git import GitCliTool, GitHubCliTool
from annolid.core.agent.tools.mcp import MCPToolWrapper, _validate_mcp_server_url
from annolid.core.agent.tools.mcp_browser import (
    McpBrowserScrollTool,
    _call_mcp_tool,
)
from annolid.core.agent.tools.policy import (
    ToolPermissionContext,
    classify_destructive_tool_call,
    resolve_allowed_tools,
)
from annolid.core.agent.config.schema import ToolsConfig


class _StrictTool(FunctionTool):
    @property
    def name(self) -> str:
        return "strict_tool"

    @property
    def description(self) -> str:
        return "Strict schema validation test tool."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": r"^[A-Z]{2}-\d+$"},
                "ratio": {"type": "number"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        return json.dumps({"ok": True, "kwargs": kwargs})


class _SimpleTool(FunctionTool):
    def __init__(
        self,
        tool_name: str,
        *,
        description: str = "Simple test tool.",
        read_only: bool = False,
        delay: float = 0.0,
        events: Optional[list[tuple[str, str, float]]] = None,
    ) -> None:
        self._name = tool_name
        self._description = description
        self._read_only = read_only
        self._delay = delay
        self._events = events

    @property
    def name(self) -> str:
        return self._name

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }

    async def execute(self, value: str = "", **kwargs: Any) -> str:
        del kwargs
        if self._events is not None:
            self._events.append((self._name, "start", time.perf_counter()))
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._events is not None:
            self._events.append((self._name, "end", time.perf_counter()))
        return f"{self._name}:{value}"


def test_normalize_tool_arguments_repairs_near_json_and_fails_closed() -> None:
    assert normalize_tool_arguments('```json\n{"path": "a.txt"}\n```') == {
        "path": "a.txt"
    }
    assert normalize_tool_arguments('{"path": "a.txt",}') == {"path": "a.txt"}
    assert normalize_tool_arguments('"{\\"path\\": \\"b.txt\\"}"') == {"path": "b.txt"}
    # Garbage input must still fail closed to _raw so broken args aren't silently dropped.
    assert normalize_tool_arguments("{broken") == {"_raw": "{broken"}
    assert normalize_tool_arguments("not json at all") == {"_raw": "not json at all"}


def test_extract_and_strip_qwen_and_kimi_tool_call_markup() -> None:
    qwen = (
        "Let me check.\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'
    )
    calls = extract_text_tool_calls(qwen)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"] == {"path": "README.md"}
    assert strip_tool_call_markup(qwen) == "Let me check."

    kimi = (
        "Checking\n"
        "<|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.list_dir:0<|tool_call_argument_begin|>{"path": "."}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    kimi_calls = extract_text_tool_calls(kimi)
    assert len(kimi_calls) == 1
    assert kimi_calls[0]["name"] == "list_dir"
    assert kimi_calls[0]["arguments"] == {"path": "."}
    assert strip_tool_call_markup(kimi) == "Checking"


def test_function_tool_validation_enforces_additional_properties_finite_pattern_unique() -> (
    None
):
    tool = _StrictTool()
    assert (
        tool.validate_params({"code": "AB-12", "ratio": 0.5, "tags": ["a", "b"]}) == []
    )

    # Non-dict params should return an error rather than raising AttributeError.
    non_dict_errors = tool.validate_params("oops")  # type: ignore[arg-type]
    assert non_dict_errors and "must be an object" in non_dict_errors[0]

    extra_errors = tool.validate_params({"code": "AB-12", "unknown_arg": 1})
    assert any("unexpected parameter unknown_arg" in err for err in extra_errors)

    nan_errors = tool.validate_params({"ratio": math.nan})
    assert any("finite number" in err for err in nan_errors)

    pat_errors = tool.validate_params({"code": "bad"})
    assert any("must match pattern" in err for err in pat_errors)

    dup_errors = tool.validate_params({"tags": ["x", "x"]})
    assert any("must be unique" in err for err in dup_errors)


def test_function_registry_raw_json_rejection_fuzzy_hint_and_helpers() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SimpleTool("read_file", read_only=True))
    registry.register(_SimpleTool("write_file"))

    assert registry.canonical_name("read_file") == "read_file"
    assert len(registry.values()) == 2
    filtered = registry.get_definitions(allowed_names=["read_file"])
    assert [item["function"]["name"] for item in filtered] == ["read_file"]

    missing_msg = asyncio.run(registry.execute("read_fil", {}))
    assert "not found" in missing_msg
    assert "Did you mean: read_file" in missing_msg

    bad_json_msg = asyncio.run(registry.execute("read_file", {"_raw": "{broken"}))
    assert "Invalid JSON arguments" in bad_json_msg


def test_policy_destructive_classification_and_additive_groups() -> None:
    assert classify_destructive_tool_call("gh_cli", {"allow_mutation": False}) is None
    assert (
        classify_destructive_tool_call("gh_cli", {"allow_mutation": True}) is not None
    )
    assert (
        classify_destructive_tool_call("annolid_train_start", {"allow_mutation": True})
        is not None
    )
    assert classify_destructive_tool_call("exec_process", {"action": "poll"}) is None
    assert (
        classify_destructive_tool_call("exec_process", {"action": "kill"}) is not None
    )

    cfg = ToolsConfig(profile="minimal", allow=["group:memory", "group:training"])
    resolved = resolve_allowed_tools(
        all_tool_names=[
            "gui_context",
            "memory_search",
            "memory_get",
            "annolid_train_start",
            "exec",
        ],
        tools_cfg=cfg,
        provider="openai",
        model="gpt-5",
    )
    assert {"memory_search", "memory_get", "annolid_train_start"}.issubset(
        resolved.allowed_tools
    )
    assert "exec" not in resolved.allowed_tools


def test_agent_loop_denied_exec_does_not_poison_subsequent_email() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SimpleTool("exec"))
    registry.register(_SimpleTool("email"))
    turn = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        turn["n"] += 1
        if turn["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "exec", "arguments": {"value": "ls"}}
                ],
            }
        if turn["n"] == 2:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c2", "name": "email", "arguments": {"value": "hi"}}
                ],
            }
        return {"content": "all done"}

    loop = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        tool_approval_callback=lambda name, args, reason: False,
    )
    result = asyncio.run(loop.run("run task"))
    assert result.content == "all done"
    assert "denied by user confirmation" in result.tool_runs[0].result
    # Because exec was NOT executed, email must NOT be blocked by high-risk guard.
    assert result.tool_runs[1].result == "email:hi"


def test_agent_loop_blocks_exec_with_spawn_behavior_subagent() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SimpleTool("exec"))
    registry.register(_SimpleTool("spawn_behavior_subagent"))
    turn = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        turn["n"] += 1
        if turn["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "exec", "arguments": {"value": "pwd"}}
                ],
            }
        if turn["n"] == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "spawn_behavior_subagent",
                        "arguments": {"value": "bg"},
                    }
                ],
            }
        return {"content": "done"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("inspect and spawn"))
    assert result.tool_runs[0].result == "exec:pwd"
    assert "Blocked by safety policy" in result.tool_runs[1].result


def test_agent_loop_runs_read_only_tool_batch_in_parallel_and_preserves_order() -> None:
    events: list[tuple[str, str, float]] = []
    registry = FunctionToolRegistry()
    registry.register(
        _SimpleTool("reader_a", read_only=True, delay=0.12, events=events)
    )
    registry.register(
        _SimpleTool("reader_b", read_only=True, delay=0.12, events=events)
    )
    registry.register(
        _SimpleTool("writer_c", read_only=False, delay=0.05, events=events)
    )
    turn = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        turn["n"] += 1
        if turn["n"] == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "reader_a", "arguments": {"value": "1"}},
                    {"id": "c2", "name": "reader_b", "arguments": {"value": "2"}},
                    {"id": "c3", "name": "writer_c", "arguments": {"value": "3"}},
                ],
            }
        return {"content": "ok"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("read both then write"))
    assert [r.name for r in result.tool_runs] == ["reader_a", "reader_b", "writer_c"]
    assert [r.result for r in result.tool_runs] == [
        "reader_a:1",
        "reader_b:2",
        "writer_c:3",
    ]
    starts = {name: ts for name, kind, ts in events if kind == "start"}
    ends = {name: ts for name, kind, ts in events if kind == "end"}
    # Both read-only tools must overlap in time, while writer_c starts after both finish.
    assert starts["reader_b"] < ends["reader_a"]
    assert starts["writer_c"] >= max(ends["reader_a"], ends["reader_b"]) - 1e-4


def test_context_compaction_never_leaves_orphaned_tool_messages() -> None:
    messages: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    # Create 20 assistant(tool_calls) + tool pairs (40 messages) + 12 user/assistant turns.
    for i in range(20):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": "read_file",
                "content": f"res_{i}",
            }
        )
    for i in range(6):
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})

    # Choose a non-system window that lands right on a `tool` message (odd index).
    decision = plan_context_compaction(
        messages, max_messages=30, max_non_system_messages=25
    )
    non_sys = [m for m in decision.messages if m["role"] != "system"]
    assert non_sys
    assert non_sys[0]["role"] != "tool"
    seen_ids: set[str] = set()
    for msg in non_sys:
        if msg["role"] == "assistant":
            for tc in msg.get("tool_calls") or []:
                seen_ids.add(tc["id"])
        elif msg["role"] == "tool":
            assert msg["tool_call_id"] in seen_ids


def test_tool_selector_snake_case_no_short_false_positives_and_companions() -> None:
    assert "github" in tokenize_text("mcp_github_list_issues")
    assert "mcp_github_list_issues" in tokenize_text("mcp_github_list_issues")

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "annolid_track_qc",
                "description": "Compute tracking quality control metrics.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_github_list_issues",
                "description": "List GitHub repository issues.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List directory entries.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file text.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit text in a file.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    index = compile_tool_index(schemas)
    annolid_entry = next(e for e in index if e.name == "annolid_track_qc")
    # Short token "id" must NOT match "annolid_track_qc" via substring.
    assert score_tool_schema(annolid_entry, ["id"]) == 0

    selected = select_relevant_tool_definitions(
        all_tool_definitions=schemas,
        tool_index=index,
        default_tool_definitions=schemas[:3],
        user_message_text="list github issues",
        messages=[],
    )
    assert selected[0]["function"]["name"] == "mcp_github_list_issues"

    # Follow-up after `read_file` should retain companion `edit_file`.
    followup = select_relevant_tool_definitions(
        all_tool_definitions=schemas,
        tool_index=index,
        default_tool_definitions=schemas[:3],
        user_message_text="now update that section",
        messages=[{"role": "tool", "name": "read_file", "content": "hello"}],
    )
    followup_names = {s["function"]["name"] for s in followup}
    assert "read_file" in followup_names
    assert "edit_file" in followup_names


def test_compact_tool_result_handles_output_and_results_lists_valid_json() -> None:
    huge_output = json.dumps({"ok": True, "output": "x" * 20000})
    compacted = compact_tool_result_for_llm(
        name="git_diff", result=huge_output, limit=2000
    )
    parsed = json.loads(compacted)
    assert parsed["truncated_for_llm"] is True
    assert parsed["original_output_length"] == 20000
    assert len(compacted) <= 2300

    huge_list = json.dumps(
        {
            "query": "foo",
            "results": [{"line": i, "text": "y" * 200} for i in range(200)],
        }
    )
    compacted_list = compact_tool_result_for_llm(
        name="code_search", result=huge_list, limit=2500
    )
    parsed_list = json.loads(compacted_list)
    assert parsed_list["truncated_for_llm"] is True
    assert parsed_list["original_results_count"] == 200
    assert len(compacted_list) <= 2800


def test_think_tag_stream_parser_handles_split_tags() -> None:
    parser = ThinkTagStreamParser()
    chunks = ["Hello ", "<thin", "k>deep ", "thought</thin", "k> world", "!"]
    for ch in chunks:
        parser.feed(ch)
    parser.flush()
    assert parser.thought_text == "deep thought"
    assert parser.output_text == "Hello  world!"


def test_agent_loop_qwen_tool_call_and_unknown_tool_recovery() -> None:
    registry = FunctionToolRegistry()
    registry.register(_SimpleTool("read_file", read_only=True))
    turn = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        turn["n"] += 1
        if turn["n"] == 1:
            # Hallucinated tool name -> should trigger one-shot self-correction.
            return {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "nonexistent_tool", "arguments": {}}
                ],
            }
        if turn["n"] == 2:
            assert any(
                m.get("role") == "system"
                and "nonexistent_tool" in str(m.get("content"))
                for m in messages
            )
            return {
                "content": '<tool_call>{"name": "read_file", "arguments": {"value": "ok"}}</tool_call>',
            }
        return {"content": "Recovered answer"}

    loop = AgentLoop(tools=registry, llm_callable=fake_llm, model="fake")
    result = asyncio.run(loop.run("read it"))
    assert result.content == "Recovered answer"
    assert len(result.tool_runs) == 1
    assert result.tool_runs[0].name == "read_file"
    assert result.tool_runs[0].result == "read_file:ok"


def test_agent_loop_close_unregisters_mcp_tools_and_subagent_factory_propagates_security() -> (
    None
):
    registry = FunctionToolRegistry()
    registry.register(_SimpleTool("mcp_demo_ping"))
    perm_ctx = ToolPermissionContext.from_iterables(deny_names=["exec"])
    approval_cb = lambda name, args, reason: False  # noqa: E731

    loop = AgentLoop(
        tools=registry,
        llm_callable=lambda *a, **k: asyncio.sleep(0, result={"content": "ok"}),
        model="fake",
        tool_timeout_seconds=12.5,
        tool_permission_context=perm_ctx,
        tool_approval_callback=approval_cb,
    )
    loop._mcp_registered_tool_names.add("mcp_demo_ping")
    asyncio.run(loop.close())
    assert not registry.has("mcp_demo_ping")

    child = loop._build_subagent_loop_factory()()
    assert child._tool_timeout_seconds == 12.5
    assert child._tool_permission_context is perm_ctx
    assert child._tool_approval_callback is approval_cb


def test_provider_fixes_codex_ollama_anthropic_and_openai_compat() -> None:
    # 1. Codex concatenates multiple system prompts instead of overwriting.
    sys_prompt, items = codex_convert_messages(
        [
            {"role": "system", "content": "Base instructions."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Post-tool guidance."},
        ]
    )
    assert "Base instructions." in sys_prompt
    assert "Post-tool guidance." in sys_prompt
    assert len(items) == 1

    # 2. Ollama normalizer preserves assistant tool_calls (as dict args) & tool_name.
    ollama_msgs = normalize_messages_for_ollama(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "contents",
            },
        ]
    )
    assert ollama_msgs[0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert ollama_msgs[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.txt"}
    assert ollama_msgs[1]["tool_name"] == "read_file"

    # 3. Ollama stream collector does not clobber distinct idless tool calls across chunks.
    stream_parts = [
        {
            "message": {
                "tool_calls": [{"function": {"name": "a", "arguments": {"x": 1}}}]
            }
        },
        {
            "message": {
                "tool_calls": [{"function": {"name": "b", "arguments": {"y": 2}}}]
            }
        },
    ]
    _, collected_calls, _ = collect_ollama_stream(stream_parts, parse_ollama_tool_calls)
    assert [c["name"] for c in collected_calls] == ["a", "b"]
    assert len({c["id"] for c in collected_calls}) == 2

    # 4. Anthropic conversion translates image_url, skips empty assistant messages, starts with user.
    unified = UnifiedLLMProvider(default_model="anthropic/claude-3-5-sonnet")
    sys_text, anthropic_msgs = unified._convert_messages_for_anthropic(
        [
            {"role": "system", "content": "System A"},
            {"role": "assistant", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            },
        ]
    )
    assert sys_text == "System A"
    assert anthropic_msgs[0]["role"] == "user"
    assert anthropic_msgs[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }

    # 5. OpenAI-compat parses legacy function_call and defaults missing tc.id.
    compat = OpenAICompatProvider.__new__(OpenAICompatProvider)
    legacy_resp = compat._parse_response(
        {
            "choices": [
                {
                    "finish_reason": "function_call",
                    "message": {
                        "content": "",
                        "function_call": {
                            "name": "read_file",
                            "arguments": '{"path": "x.txt",}',
                        },
                    },
                }
            ]
        }
    )
    assert len(legacy_resp.tool_calls) == 1
    assert legacy_resp.tool_calls[0].id == "call_0"
    assert legacy_resp.tool_calls[0].arguments == {"path": "x.txt"}


@pytest.mark.anyio
async def test_unified_provider_empty_stream_returns_retryable_error() -> None:
    provider = UnifiedLLMProvider(default_model="test")

    class _FakeCompletions:
        async def create(self, **kwargs: Any):
            async def _stream():
                chunk = SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="", reasoning_content=None, tool_calls=None
                            ),
                            finish_reason="stop",
                        )
                    ]
                )
                yield chunk

            return _stream()

    provider._ensure_openai_client = lambda: SimpleNamespace(  # type: ignore[method-assign]
        chat=SimpleNamespace(completions=_FakeCompletions())
    )
    res = await provider.chat(messages=[], on_token=lambda t: None)
    assert res.finish_reason == "error"
    assert res.error_kind == "empty"
    assert res.error_should_retry is True


def test_filesystem_tools_range_replace_all_max_entries_and_atomic_write(
    tmp_path: Path,
) -> None:
    writer = WriteFileTool(allowed_dir=tmp_path)
    reader = ReadFileTool(allowed_dir=tmp_path)
    editor = EditFileTool(allowed_dir=tmp_path)
    lister = ListDirTool(allowed_dir=tmp_path)

    assert reader.read_only is True
    assert lister.read_only is True

    target = tmp_path / "sample.txt"
    asyncio.run(writer.execute(path=str(target), content="line1\nfoo\nline3\nfoo\n"))
    assert asyncio.run(reader.execute(path=str(target), start_line=2, end_line=3)) == (
        "foo\nline3"
    )
    warn = asyncio.run(editor.execute(path=str(target), old_text="foo", new_text="bar"))
    assert "Warning:" in warn
    ok = asyncio.run(
        editor.execute(
            path=str(target), old_text="foo", new_text="bar", replace_all=True
        )
    )
    assert "Successfully edited" in ok
    assert target.read_text(encoding="utf-8") == "line1\nbar\nline3\nbar\n"

    for idx in range(5):
        (tmp_path / f"f_{idx}.txt").write_text("x", encoding="utf-8")
    listed = asyncio.run(lister.execute(path=str(tmp_path), max_entries=3))
    assert "more entries omitted" in listed


def test_mcp_and_mcp_browser_and_git_and_telemetry_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. MCP URL validation rejects non-http(s) schemes.
    assert _validate_mcp_server_url("https://example.com/mcp") == ""
    assert _validate_mcp_server_url("file:///etc/passwd") != ""

    # 2. MCPToolWrapper surfaces isError=True even when TextContent is present & summarizes images.
    fake_types = SimpleNamespace(
        TextContent=type("TextContent", (), {}),
        ImageContent=type("ImageContent", (), {}),
        EmbeddedResource=type("EmbeddedResource", (), {}),
    )
    text_block = fake_types.TextContent()
    text_block.text = "Tool failed on server"
    img_block = fake_types.ImageContent()
    img_block.mimeType = "image/png"
    img_block.data = "A" * 5000

    class _FakeSession:
        async def call_tool(self, name: str, arguments: dict[str, Any]):
            del name, arguments
            return SimpleNamespace(content=[text_block, img_block], isError=True)

    import sys

    fake_mcp_mod = SimpleNamespace(types=fake_types)
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp_mod)
    wrapper = MCPToolWrapper(
        _FakeSession(),
        "demo",
        SimpleNamespace(
            name="ping", description="ping", inputSchema={"type": "object"}
        ),
    )
    mcp_out = asyncio.run(wrapper.execute())
    assert mcp_out.startswith("Error: ")
    assert "Image content: mimeType=image/png" in mcp_out
    assert "AAAAA" not in mcp_out

    # 3. _call_mcp_tool respects isError=True & McpBrowserScrollTool uses PageUp for negative delta_y.
    err_payload = asyncio.run(_call_mcp_tool(_FakeSession(), "browser_click", {}))
    assert err_payload["ok"] is False
    assert "Tool failed on server" in err_payload["error"]

    recorded_keys: list[str] = []

    class _ScrollSession:
        async def call_tool(self, name: str, args: dict[str, Any]):
            if name == "browser_press_key":
                recorded_keys.append(args["key"])
            ok_block = fake_types.TextContent()
            ok_block.text = "scrolled"
            return SimpleNamespace(content=[ok_block], isError=False)

    monkeypatch.setattr(
        "annolid.core.agent.tools.mcp_browser._ensure_mcp_browser_session",
        lambda servers: asyncio.sleep(0, result=_ScrollSession()),
    )
    scroll_tool = McpBrowserScrollTool()
    asyncio.run(scroll_tool.execute(delta_y=-400))
    asyncio.run(scroll_tool.execute(delta_y=400))
    assert recorded_keys == ["PageUp", "PageDown"]

    # 4. GitCliTool / GitHubCliTool catch unbalanced quotes cleanly.
    git_res = json.loads(
        asyncio.run(
            GitCliTool(allowed_dir=tmp_path).execute(
                command='git log --grep="unterminated', repo_path=str(tmp_path)
            )
        )
    )
    assert "Invalid git command syntax" in git_res["error"]

    gh_res = json.loads(
        asyncio.run(
            GitHubCliTool(allowed_dir=tmp_path).execute(
                command='gh pr view "unterminated', repo_path=str(tmp_path)
            )
        )
    )
    assert "Invalid gh command syntax" in gh_res["error"]

    # 5. wrap_tool_callback awaits async callbacks before emitting Finished/Tool failed.
    events: list[str] = []

    async def _async_gui_cb(ok: bool) -> dict[str, Any]:
        events.append("body_start")
        await asyncio.sleep(0.01)
        events.append("body_end")
        return {"ok": ok}

    wrapped = wrap_tool_callback(
        name="gui_open_url", callback=_async_gui_cb, emit_progress=events.append
    )
    asyncio.run(wrapped(True))
    assert events == [
        "Running tool: gui open url",
        "body_start",
        "body_end",
        "Finished tool: gui open url",
    ]

    events.clear()
    asyncio.run(wrapped(False))
    assert events[-1] == "Tool failed: gui open url"

    # 6. sanitize_final_response_text strips Qwen <tool_call> markup too.
    assert (
        sanitize_final_response_text(
            'Done.<tool_call>{"name":"read_file","arguments":{}}</tool_call>'
        )
        == "Done."
    )
