from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import inspect
import json
import os
from annolid.utils.logger import logger
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from annolid.utils.llm_settings import resolve_agent_runtime_config, resolve_llm_config

from .context import AgentContextBuilder
from .eval.telemetry import RunTraceStore
from .memory import AgentMemoryStore
from .memory_store.flush import append_pre_compaction_flush
from .providers import LiteLLMProvider, OpenAICompatProvider, resolve_openai_compat
from .tools import FunctionToolRegistry
from .tools.function_builtin import (
    AutomationSchedulerTool,
    CancelTaskTool,
    CronTool,
    ListTasksTool,
    MessageTool,
    SpawnTool,
)
from annolid.core.agent.tools.swarm_tool import SwarmTool

if TYPE_CHECKING:  # pragma: no cover
    from .subagent import SubagentManager

LLMCallable = Callable[
    [
        Sequence[Mapping[str, Any]],
        Sequence[Mapping[str, Any]],
        str,
        Optional[Callable[[str], None]],
    ],
    Awaitable[Mapping[str, Any]],
]
ProgressCallback = Callable[[str], Any]


@dataclass(frozen=True)
class AgentToolRun:
    call_id: str
    name: str
    arguments: Dict[str, Any]
    result: str


@dataclass(frozen=True)
class AgentLoopResult:
    content: str
    messages: Sequence[Dict[str, Any]]
    iterations: int
    tool_runs: Sequence[AgentToolRun] = field(default_factory=tuple)
    media: Sequence[str] = field(default_factory=tuple)
    stopped_reason: str = "done"


@dataclass(frozen=True)
class AgentMemoryConfig:
    enabled: bool = True
    max_history_messages: int = 24
    memory_window: int = 50
    include_facts_in_system_prompt: bool = True


@dataclass(frozen=True)
class _ToolSchemaIndex:
    schema: Dict[str, Any]
    name: str
    desc: str
    tokens: frozenset[str]


class InMemorySessionStore:
    """Thread-safe in-process memory store for chat history and facts."""

    def __init__(self) -> None:
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._facts: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = RLock()

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(m) for m in self._history.get(session_id, [])]

    def append_history(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_messages: int,
    ) -> None:
        with self._lock:
            self._history[session_id].extend([dict(m) for m in messages])
            max_keep = max(1, int(max_messages))
            if len(self._history[session_id]) > max_keep:
                self._history[session_id] = self._history[session_id][-max_keep:]

    def clear_history(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)

    def get_facts(self, session_id: str) -> Dict[str, str]:
        with self._lock:
            return dict(self._facts.get(session_id, {}))

    def set_fact(self, session_id: str, key: str, value: str) -> None:
        with self._lock:
            self._facts[session_id][str(key)] = str(value)

    def delete_fact(self, session_id: str, key: str) -> bool:
        with self._lock:
            facts = self._facts.get(session_id)
            if not facts or key not in facts:
                return False
            facts.pop(key, None)
            return True

    def clear_facts(self, session_id: str) -> None:
        with self._lock:
            self._facts.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._history.pop(session_id, None)
            self._facts.pop(session_id, None)
            self._metadata.pop(session_id, None)

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metadata.get(session_id, {}))

    def update_session_metadata(
        self, session_id: str, updates: Mapping[str, Any]
    ) -> None:
        with self._lock:
            meta = self._metadata[session_id]
            for raw_key, raw_value in dict(updates or {}).items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                meta[key] = raw_value


class SessionStoreProtocol(Protocol):
    def get_history(self, session_id: str) -> List[Dict[str, Any]]: ...
    def append_history(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_messages: int,
    ) -> None: ...
    def clear_history(self, session_id: str) -> None: ...
    def get_facts(self, session_id: str) -> Dict[str, str]: ...
    def set_fact(self, session_id: str, key: str, value: str) -> None: ...
    def delete_fact(self, session_id: str, key: str) -> bool: ...
    def clear_facts(self, session_id: str) -> None: ...
    def clear_session(self, session_id: str) -> None: ...
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]: ...
    def update_session_metadata(
        self, session_id: str, updates: Mapping[str, Any]
    ) -> None: ...


class AgentLoop:
    """OpenAI-compatible async tool loop inspired by nanobot/agent/loop.py.

    This loop is stateless by default: pass existing history in `history`.
    """

    _MIN_CONSOLIDATION_TRANSCRIPT_CHARS = 64
    _HIGH_RISK_INTENT_MARKERS = (
        "intent:high-risk",
        "intent:high_risk",
        "allow:high-risk",
        "allow_high_risk",
        "unsafe:high-risk",
    )
    _HIGH_RISK_TOOL_MESSAGING = frozenset(
        {"email", "list_emails", "read_email", "message", "camera_snapshot"}
    )
    _HIGH_RISK_TOOL_AUTOMATION = frozenset({"cron", "automation_schedule", "spawn"})

    def __init__(
        self,
        *,
        tools: FunctionToolRegistry,
        llm_callable: Optional[LLMCallable] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
        max_iterations: Optional[int] = None,
        memory_config: Optional[AgentMemoryConfig] = None,
        memory_store: Optional[SessionStoreProtocol] = None,
        workspace: Optional[str] = None,
        allowed_read_roots: Optional[Sequence[str | Path]] = None,
        context_builder: Optional[AgentContextBuilder] = None,
        subagent_manager: Optional["SubagentManager"] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        interleave_post_tool_guidance: bool = True,
        llm_timeout_seconds: Optional[float] = None,
        tool_timeout_seconds: Optional[float] = None,
        browser_first_for_web: bool = True,
        strict_runtime_tool_guard: bool = True,
    ) -> None:
        self._tools = tools
        runtime_cfg = resolve_agent_runtime_config(profile=profile)
        resolved_max_iterations = (
            runtime_cfg.max_tool_iterations
            if max_iterations is None
            else max(1, int(max_iterations))
        )
        self._max_iterations = resolved_max_iterations
        self._default_temperature = float(runtime_cfg.temperature)
        self._logger = logger
        self._memory_config = memory_config or AgentMemoryConfig(
            max_history_messages=runtime_cfg.max_history_messages,
            memory_window=runtime_cfg.memory_window,
        )
        self._memory_store = memory_store or InMemorySessionStore()
        self._workspace = workspace
        self._allowed_read_roots = tuple(str(p) for p in (allowed_read_roots or ()))
        self._context_builder = context_builder
        self._subagent_manager = subagent_manager
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: Optional[AsyncExitStack] = None
        self._mcp_connected = False
        self._interleave_post_tool_guidance = bool(interleave_post_tool_guidance)
        self._llm_timeout_seconds = (
            float(llm_timeout_seconds)
            if llm_timeout_seconds is not None and float(llm_timeout_seconds) > 0
            else None
        )
        self._tool_timeout_seconds = (
            float(tool_timeout_seconds)
            if tool_timeout_seconds is not None and float(tool_timeout_seconds) > 0
            else None
        )
        self._browser_first_for_web = bool(browser_first_for_web)
        self._strict_runtime_tool_guard = bool(strict_runtime_tool_guard)
        self._provider_impl: Optional[Any] = None
        self._shadow_mode_enabled = str(
            os.getenv("ANNOLID_AGENT_SHADOW_MODE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._shadow_routing_policy = (
            str(os.getenv("ANNOLID_AGENT_SHADOW_ROUTING_POLICY", "default")).strip()
            or "default"
        )

        self._provider = provider
        self._model_override = model
        self._profile = profile
        self._llm_callable = llm_callable
        self._cached_tool_signature: Optional[tuple[tuple[str, str], ...]] = None
        self._cached_tool_index: List[_ToolSchemaIndex] = []
        self._cached_default_tools: List[Dict[str, Any]] = []

        if self._llm_callable is None:
            (
                self._llm_callable,
                self._resolved_model,
                self._provider_impl,
            ) = self._build_default_llm_callable(
                profile=profile,
                provider=provider,
                model=model,
            )
        else:
            self._resolved_model = model or "unknown"

        if self._context_builder is None and self._workspace:
            self._context_builder = AgentContextBuilder(Path(self._workspace))

        self._wire_tools()

    @property
    def model(self) -> str:
        return self._resolved_model

    async def run(
        self,
        user_message: str,
        *,
        session_id: str = "default",
        history: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        use_memory: Optional[bool] = None,
        channel: Optional[str] = None,
        chat_id: Optional[str] = None,
        media: Optional[List[str]] = None,
        skill_names: Optional[List[str]] = None,
        on_progress: Optional[ProgressCallback] = None,
        inbound_metadata: Optional[Mapping[str, Any]] = None,
    ) -> AgentLoopResult:
        run_started = time.perf_counter()
        llm_total_ms = 0.0
        tool_exec_total_ms = 0.0
        tool_call_count = 0
        message_build_ms = 0.0
        mcp_connect_ms = 0.0
        memory_enabled = (
            self._memory_config.enabled if use_memory is None else bool(use_memory)
        )
        inbound_meta = dict(inbound_metadata or {})
        turn_id = str(inbound_meta.get("turn_id") or "").strip()
        parent_id = str(inbound_meta.get("parent_id") or "").strip()

        turn_seq = self._increment_session_turn_counter(session_id)
        if not turn_id:
            turn_id = f"turn-{turn_seq}"
        self._record_session_event(
            session_id=session_id,
            direction="inbound",
            kind="user",
            turn_id=turn_id,
            payload={"text": str(user_message or "")},
        )
        self._set_tool_context(channel=channel, chat_id=chat_id)
        self._logger.info(
            "Processing message from %s on %s using model %s",
            chat_id,
            channel,
            self.model,
        )
        connect_started = time.perf_counter()
        await self._connect_mcp()
        mcp_connect_ms = (time.perf_counter() - connect_started) * 1000.0
        try:
            build_started = time.perf_counter()
            user_message_text = str(user_message)
            messages = await self._build_initial_messages(
                user_message_text=user_message_text,
                session_id=session_id,
                system_prompt=system_prompt,
                memory_enabled=memory_enabled,
                channel=channel,
                chat_id=chat_id,
                media=media,
                skill_names=skill_names,
                history=history,
                turn_seq=turn_seq,
            )
            message_build_ms = (time.perf_counter() - build_started) * 1000.0

            tool_runs: List[AgentToolRun] = []
            messages_media: List[str] = list(media or [])
            executed_tools: set[str] = set()
            final_content = ""
            stopped_reason = "done"
            explicit_high_risk_intent = self._has_explicit_high_risk_intent(
                user_message_text,
                inbound_metadata=inbound_metadata,
            )
            all_tool_definitions, tool_signature, tool_index, default_tools = (
                self._prepare_tool_selection()
            )
            repeated_tool_cycles = 0
            last_tool_cycle_signature: Optional[tuple[str, ...]] = None
            last_iteration = 0

            for iteration in range(1, self._max_iterations + 1):
                last_iteration = iteration
                tool_definitions = self._select_relevant_tool_definitions(
                    all_tool_definitions=all_tool_definitions,
                    tool_index=tool_index,
                    default_tool_definitions=default_tools,
                    user_message_text=user_message_text,
                    messages=messages,
                )
                if self._shadow_mode_enabled:
                    candidate = (
                        default_tools
                        if self._shadow_routing_policy == "default"
                        else all_tool_definitions
                    )
                    self._capture_shadow_routing(
                        session_id=session_id,
                        primary=tool_definitions,
                        candidate=candidate,
                        iteration=iteration,
                    )
                active_node_id = "planner"
                if session_id and session_id.startswith("swarm:"):
                    active_node_id = session_id.split(":", 1)[1].lower()

                token_buffer = []
                thought_buffer = []
                is_thinking = False
                last_update = time.perf_counter()

                def _on_llm_token(token: str) -> None:
                    nonlocal is_thinking, last_update

                    if "<think>" in token:
                        is_thinking = True
                        token = token.replace("<think>", "")
                    if "</think>" in token:
                        is_thinking = False
                        token = token.replace("</think>", "")

                    if is_thinking:
                        thought_buffer.append(token)
                    else:
                        token_buffer.append(token)

                    # Throttled visualizer update (every 500ms or so)
                    now = time.perf_counter()
                    if now - last_update > 0.5:
                        last_update = now
                        current_thought = "".join(thought_buffer)
                        current_output = "".join(token_buffer)

                        short_thought = (
                            (current_thought[-100:] + "...")
                            if len(current_thought) > 100
                            else current_thought
                        )
                        short_output = (
                            (current_output[-100:] + "...")
                            if len(current_output) > 100
                            else current_output
                        )

                        try:
                            from annolid.gui.widgets.threejs_viewer_server import (
                                update_swarm_node,
                            )

                            update_swarm_node(
                                active_node_id,
                                "active",
                                f"Output: {short_output}"
                                if short_output
                                else "Thinking...",
                                thinking=short_thought,
                                parent=parent_id.lower() if parent_id else "",
                            )
                        except Exception:
                            pass

                response, llm_elapsed_ms = await self._execute_llm_cycle(
                    session_id=session_id,
                    iteration=iteration,
                    messages=messages,
                    tool_definitions=tool_definitions,
                    on_token=_on_llm_token,
                )
                llm_total_ms += llm_elapsed_ms

                assistant_text = str(response.get("content") or "")
                reasoning = str(response.get("reasoning_content") or "").strip()
                if reasoning and not assistant_text.startswith("<think>"):
                    assistant_text = (
                        f"<think>\n{reasoning}\n</think>\n\n{assistant_text}".strip()
                    )
                    response["content"] = assistant_text

                tool_calls = self._sanitize_tool_calls(
                    self._extract_tool_calls(response)
                )
                self._logger.info(
                    "annolid-bot profile iteration session=%s model=%s iteration=%d llm_ms=%.1f tool_calls=%d",
                    session_id,
                    self.model,
                    iteration,
                    llm_elapsed_ms,
                    len(tool_calls),
                )

                if tool_calls:
                    if on_progress is not None:
                        progress_text = self._strip_think(
                            assistant_text
                        ) or self._tool_hint(tool_calls)
                        if progress_text:
                            try:
                                await self._dispatch_progress(
                                    on_progress, progress_text
                                )
                            except Exception as exc:
                                self._logger.warning(
                                    "on_progress callback failed: %s",
                                    exc,
                                )
                    tools_str = ", ".join(
                        str(tc.get("name") or "") for tc in tool_calls
                    )
                    active_node_id = "planner"
                    if session_id and session_id.startswith("swarm:"):
                        active_node_id = session_id.split(":", 1)[1].lower()

                    try:
                        from annolid.gui.widgets.threejs_viewer_server import (
                            update_swarm_node,
                        )

                        update_swarm_node(
                            active_node_id,
                            "active",
                            f"Executing: {tools_str}",
                            thinking=reasoning or self._strip_think(assistant_text),
                        )
                    except Exception:
                        pass

                    tool_cycle_signature = tuple(
                        f"{call.get('name', '')}:{json.dumps(call.get('arguments', {}), ensure_ascii=False, sort_keys=True)}"
                        for call in tool_calls
                    )
                    if tool_cycle_signature == last_tool_cycle_signature:
                        repeated_tool_cycles += 1
                    else:
                        repeated_tool_cycles = 0
                        last_tool_cycle_signature = tool_cycle_signature
                    if repeated_tool_cycles >= 2:
                        stopped_reason = "repeated_tool_calls"
                        final_content = (
                            "Agent tool loop stalled on repeated identical tool calls. "
                            "Please revise the prompt or switch model."
                        )
                        break

                    # Clean up assistant text to remove raw tool call syntax before saving to history,
                    # so the LLM doesn't get confused by both raw text tools and native JSON `tool_calls`.
                    clean_assistant_text = assistant_text
                    if "<|tool_call_begin|>" in clean_assistant_text:
                        import re

                        clean_assistant_text = re.sub(
                            r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>",
                            "",
                            clean_assistant_text,
                            flags=re.DOTALL,
                        )
                        clean_assistant_text = re.sub(
                            r"<\|tool_calls_section_end\|>", "", clean_assistant_text
                        )
                        clean_assistant_text = clean_assistant_text.strip()

                    messages.append(
                        {
                            "role": "assistant",
                            "content": clean_assistant_text,
                            "tool_calls": [
                                self._to_openai_tool_call(tc) for tc in tool_calls
                            ],
                        }
                    )

                    cycle_exec_ms, cycle_call_count = await self._execute_tool_cycle(
                        session_id=session_id,
                        iteration=iteration,
                        tool_calls=tool_calls,
                        messages=messages,
                        tool_runs=tool_runs,
                        media_list=messages_media,
                        executed_tools=executed_tools,
                        explicit_high_risk_intent=explicit_high_risk_intent,
                    )

                    try:
                        from annolid.gui.widgets.threejs_viewer_server import (
                            update_swarm_node,
                        )

                        update_swarm_node(active_node_id, "idle", "Awaiting Tasks")
                    except Exception:
                        pass

                    tool_exec_total_ms += cycle_exec_ms
                    tool_call_count += cycle_call_count
                    continue
                else:
                    # Fallback for local models that output raw string tokens instead of JSON arrays
                    active_node_id = "planner"
                    if session_id and session_id.startswith("swarm:"):
                        active_node_id = session_id.split(":", 1)[1].lower()

                    import re

                    text_tools = re.findall(
                        r"(?:functions\.|<\|tool_call_begin\|>\s*)([a-zA-Z0-9_]+)",
                        assistant_text,
                    )
                    if text_tools:
                        tools_str = ", ".join(set(text_tools))
                        try:
                            from annolid.gui.widgets.threejs_viewer_server import (
                                update_swarm_node,
                            )

                            update_swarm_node(
                                active_node_id, "active", f"Executing: {tools_str}"
                            )
                        except Exception:
                            pass
                    try:
                        from annolid.gui.widgets.threejs_viewer_server import (
                            update_swarm_node,
                        )

                        update_swarm_node(active_node_id, "idle", "Awaiting Tasks")
                    except Exception:
                        pass

                repeated_tool_cycles = 0
                last_tool_cycle_signature = None

                final_content = self._strip_think(assistant_text)
                if memory_enabled and str(final_content).strip():
                    tools_used = self._extract_tools_used(tool_runs)
                    self._memory_store.append_history(
                        session_id,
                        [
                            {"role": "user", "content": user_message_text},
                            {
                                "role": "assistant",
                                "content": str(final_content),
                                "tools_used": tools_used,
                            },
                        ],
                        max_messages=self._history_persist_limit(),
                    )
                    self._sync_memory_layers(
                        session_id=session_id,
                        reason="append_turn_history",
                        turn_id=turn_id,
                    )
                self._record_session_event(
                    session_id=session_id,
                    direction="outbound",
                    kind="assistant",
                    turn_id=turn_id,
                    payload={"text": str(final_content or "")},
                )
                result = AgentLoopResult(
                    content=final_content,
                    messages=messages,
                    iterations=iteration,
                    tool_runs=tuple(tool_runs),
                    media=tuple(messages_media),
                    stopped_reason=stopped_reason,
                )
                self._capture_anonymized_run_trace(
                    session_id=session_id,
                    channel=channel,
                    chat_id=chat_id,
                    user_message_text=user_message_text,
                    result=result,
                    turn_id=turn_id,
                )
                return result

            if stopped_reason == "done":
                stopped_reason = "max_iterations"
                if not final_content:
                    if tool_runs:
                        tool_names = ", ".join([r.name for r in tool_runs])
                        final_content = (
                            f"Execution stopped after reaching the maximum number of iterations ({iteration}). "
                            f"The following tools were used during this process: [{tool_names}]. "
                            "Please refine your request or provide further guidance."
                        )
                    else:
                        final_content = (
                            "Reached max iterations before producing a final response."
                        )
            if memory_enabled:
                tools_used = self._extract_tools_used(tool_runs)
                self._memory_store.append_history(
                    session_id,
                    [
                        {"role": "user", "content": user_message_text},
                        {
                            "role": "assistant",
                            "content": str(final_content),
                            "tools_used": tools_used,
                        },
                    ],
                    max_messages=self._history_persist_limit(),
                )
                self._sync_memory_layers(
                    session_id=session_id,
                    reason="append_turn_history",
                    turn_id=turn_id,
                )

            # Update visualizer with final output if in a swarm
            if session_id and session_id.startswith("swarm:"):
                active_node_id = session_id.split(":", 1)[1].lower()
                try:
                    from annolid.gui.widgets.threejs_viewer_server import (
                        update_swarm_node,
                    )

                    # Clean up output for display (strip thinking tags)
                    clean_output = str(final_content or "")
                    thinking_block = ""
                    if "<think>" in clean_output and "</think>" in clean_output:
                        parts = clean_output.split("</think>", 1)
                        thinking_block = parts[0].replace("<think>", "").strip()
                        clean_output = parts[1].strip()

                    update_swarm_node(
                        active_node_id,
                        "idle",
                        "Awaiting Tasks",
                        thinking=thinking_block[:100] + "..."
                        if len(thinking_block) > 100
                        else thinking_block,
                        output=clean_output[:150] + "..."
                        if len(clean_output) > 150
                        else clean_output,
                    )
                except Exception:
                    pass

            result = AgentLoopResult(
                content=final_content,
                messages=messages,
                iterations=last_iteration or self._max_iterations,
                tool_runs=tuple(tool_runs),
                stopped_reason=stopped_reason,
            )
            self._record_session_event(
                session_id=session_id,
                direction="outbound",
                kind="assistant",
                turn_id=turn_id,
                payload={"text": str(final_content or "")},
            )
            self._capture_anonymized_run_trace(
                session_id=session_id,
                channel=channel,
                chat_id=chat_id,
                user_message_text=user_message_text,
                result=result,
                turn_id=turn_id,
            )
            return result
        finally:
            total_ms = (time.perf_counter() - run_started) * 1000.0
            known_ms = (
                mcp_connect_ms + message_build_ms + llm_total_ms + tool_exec_total_ms
            )
            other_ms = max(0.0, total_ms - known_ms)
            buckets = {
                "llm": llm_total_ms,
                "tool_exec": tool_exec_total_ms,
                "mcp_connect": mcp_connect_ms,
                "message_build": message_build_ms,
                "other": other_ms,
            }
            bottleneck_name, bottleneck_ms = max(
                buckets.items(), key=lambda item: item[1]
            )
            self._logger.info(
                "annolid-bot profile summary session=%s model=%s total_ms=%.1f mcp_connect_ms=%.1f message_build_ms=%.1f llm_total_ms=%.1f tool_exec_total_ms=%.1f tool_calls=%d",
                session_id,
                self.model,
                total_ms,
                mcp_connect_ms,
                message_build_ms,
                llm_total_ms,
                tool_exec_total_ms,
                tool_call_count,
            )
            self._logger.info(
                "annolid-bot profile bottleneck session=%s model=%s bottleneck=%s bottleneck_ms=%.1f other_ms=%.1f",
                session_id,
                self.model,
                bottleneck_name,
                bottleneck_ms,
                other_ms,
            )
            await self._disconnect_mcp()

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        from .tools.mcp import connect_mcp_servers

        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        try:
            await connect_mcp_servers(self._mcp_servers, self._tools, self._mcp_stack)
        except Exception:
            try:
                await self._mcp_stack.aclose()
            except Exception:
                pass
            self._mcp_stack = None
            raise
        self._mcp_connected = True

    async def _execute_llm_cycle(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: Sequence[Mapping[str, Any]],
        tool_definitions: Sequence[Mapping[str, Any]],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[Dict[str, Any], float]:
        llm_started = time.perf_counter()
        llm_call = self._llm_callable(messages, tool_definitions, self.model, on_token)
        try:
            if self._llm_timeout_seconds is not None:
                response = await asyncio.wait_for(
                    llm_call, timeout=self._llm_timeout_seconds
                )
            else:
                response = await llm_call
        except asyncio.TimeoutError as exc:
            llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
            self._logger.warning(
                "annolid-bot profile iteration timeout session=%s model=%s iteration=%d llm_ms=%.1f timeout_s=%.1f",
                session_id,
                self.model,
                iteration,
                llm_elapsed_ms,
                float(self._llm_timeout_seconds or 0.0),
            )
            raise TimeoutError(
                f"LLM timed out after {float(self._llm_timeout_seconds or 0.0):.1f}s "
                f"(iteration={iteration}, model={self.model})"
            ) from exc
        llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
        return dict(response), llm_elapsed_ms

    async def _execute_tool_cycle(
        self,
        *,
        session_id: str,
        iteration: int,
        tool_calls: Sequence[Mapping[str, Any]],
        messages: List[Dict[str, Any]],
        tool_runs: List[AgentToolRun],
        media_list: List[str],
        executed_tools: set[str],
        explicit_high_risk_intent: bool,
    ) -> tuple[float, int]:
        cycle_exec_ms = 0.0
        cycle_call_count = 0
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            raw_args = call.get("arguments")
            args = self._normalize_args(raw_args)
            tool_started = time.perf_counter()
            try:
                block_reason = self._high_risk_block_reason(
                    tool_name=name,
                    executed_tools=executed_tools,
                    explicit_high_risk_intent=explicit_high_risk_intent,
                )
                if block_reason:
                    result = f"Error: {block_reason}"
                else:
                    if self._tool_timeout_seconds is not None:
                        result = await asyncio.wait_for(
                            self._tools.execute(name, args),
                            timeout=self._tool_timeout_seconds,
                        )
                    else:
                        result = await self._tools.execute(name, args)
            except asyncio.TimeoutError:
                result = (
                    f"Error: Tool '{name}' timed out after "
                    f"{float(self._tool_timeout_seconds or 0.0):.1f}s"
                )
                self._logger.warning(
                    "annolid-bot tool timeout session=%s model=%s iteration=%d tool=%s timeout_s=%.1f",
                    session_id,
                    self.model,
                    iteration,
                    name,
                    float(self._tool_timeout_seconds or 0.0),
                )
            tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0
            cycle_exec_ms += tool_elapsed_ms
            cycle_call_count += 1
            executed_tools.add(name)
            if tool_elapsed_ms >= 500.0:
                self._logger.info(
                    "annolid-bot profile tool session=%s model=%s iteration=%d tool=%s elapsed_ms=%.1f",
                    session_id,
                    self.model,
                    iteration,
                    name,
                    tool_elapsed_ms,
                )
            tool_runs.append(
                AgentToolRun(
                    call_id=call_id,
                    name=name,
                    arguments=dict(args),
                    result=str(result),
                )
            )
            # Intercept snapshots for media delivery if not sent via email
            if name in {
                "check_stream_source",
                "gui_check_stream_source",
                "camera_snapshot",
            }:
                try:
                    res_data = json.loads(str(result or "{}"))
                    if isinstance(res_data, dict) and res_data.get("ok"):
                        # Extract snapshot_path from various possible locations
                        # (Top-level, inside 'payload', or inside 'steps/capture')
                        def find_snapshot_path(d: Any) -> str:
                            if not isinstance(d, dict):
                                return ""
                            # 1. Direct hit
                            path = str(d.get("snapshot_path") or "").strip()
                            if path:
                                return path
                            # 2. Check common ancestors
                            for key in ("payload", "steps", "capture"):
                                if key in d:
                                    path = find_snapshot_path(d[key])
                                    if path:
                                        return path
                            return ""

                        snapshot_path = find_snapshot_path(res_data)
                        email_to = str(
                            res_data.get("email_to")
                            or res_data.get("payload", {}).get("email_to")
                            or ""
                        ).strip()

                        if snapshot_path and not email_to:
                            # Ensure absolute path for the delivery channel
                            path_obj = Path(snapshot_path)
                            if not path_obj.is_absolute():
                                base = Path(self._workspace or Path.cwd()).resolve()
                                path_obj = (base / path_obj).resolve()
                            media_list.append(str(path_obj))
                except Exception:
                    pass
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": str(result),
                }
            )
        self._append_post_tool_guidance(messages)
        return cycle_exec_ms, cycle_call_count

    async def _build_initial_messages(
        self,
        *,
        user_message_text: str,
        session_id: str,
        system_prompt: Optional[str],
        memory_enabled: bool,
        channel: Optional[str],
        chat_id: Optional[str],
        media: Optional[List[str]],
        skill_names: Optional[List[str]],
        history: Optional[Sequence[Mapping[str, Any]]],
        turn_seq: int = 0,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        self._persist_long_term_memory_note_from_user_text(user_message_text)
        memory_history: List[Dict[str, Any]] = []
        memory_facts: Dict[str, str] = {}
        if memory_enabled:
            memory_history = self._memory_store.get_history(session_id)
            memory_facts = self._memory_store.get_facts(session_id)
            if self._should_consolidate_memory(
                session_id=session_id,
                history_len=len(memory_history),
                turn_seq=turn_seq,
            ):
                memory_history = await self._consolidate_memory(
                    session_id=session_id,
                    history=memory_history,
                )
                self._set_next_consolidation_turn(
                    session_id=session_id,
                    turn_seq=turn_seq,
                )
            else:
                self._record_memory_telemetry(
                    session_id=session_id,
                    outcome="not_due",
                    history_len=len(memory_history),
                    archive_len=0,
                    keep_len=len(memory_history),
                    elapsed_ms=0.0,
                )
            self._sync_memory_layers(
                session_id=session_id,
                reason="pre_prompt_build",
            )

        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        elif self._context_builder is not None:
            contextual = self._context_builder.build_system_prompt(
                skill_names=skill_names
            )
            if channel and chat_id:
                contextual += (
                    "\n\n## Current Session\n"
                    f"Channel: {self._context_builder.redact_session_value(channel)}\n"
                    f"Chat ID: {self._context_builder.redact_session_value(chat_id)}"
                )
            messages.append({"role": "system", "content": contextual})
        if (
            memory_enabled
            and memory_facts
            and self._memory_config.include_facts_in_system_prompt
        ):
            messages.append(
                {
                    "role": "system",
                    "content": self._format_memory_facts(memory_facts),
                }
            )
        working_memory = self._get_store_text(
            "get_working_memory",
            session_id=session_id,
        )
        if memory_enabled and working_memory.strip():
            messages.append(
                {
                    "role": "system",
                    "content": "Working memory summary (latest turns):\n"
                    + working_memory.strip(),
                }
            )
        if memory_enabled and memory_history:
            messages.extend(memory_history)
        if history:
            messages.extend([dict(m) for m in history])
        if self._context_builder is not None and media:
            user_payload = self._context_builder.build_user_content(
                user_message_text, media
            )
            messages.append({"role": "user", "content": user_payload})
        else:
            messages.append({"role": "user", "content": user_message_text})
        return messages

    def _prepare_tool_selection(
        self,
    ) -> tuple[
        List[Dict[str, Any]],
        tuple[tuple[str, str], ...],
        List[_ToolSchemaIndex],
        List[Dict[str, Any]],
    ]:
        all_tool_definitions = self._tools.get_definitions()
        tool_signature = self._tool_signature(all_tool_definitions)
        tool_index = self._get_cached_tool_index(all_tool_definitions, tool_signature)
        default_tools = self._get_cached_default_tools(tool_index, tool_signature)
        return all_tool_definitions, tool_signature, tool_index, default_tools

    def _capture_shadow_routing(
        self,
        *,
        session_id: str,
        primary: Sequence[Mapping[str, Any]],
        candidate: Sequence[Mapping[str, Any]],
        iteration: int,
    ) -> None:
        if not self._shadow_mode_enabled or not self._workspace:
            return
        try:
            store = RunTraceStore(Path(self._workspace))
            primary_names = [
                str((row.get("function") or {}).get("name") or "")
                for row in primary
                if isinstance(row, Mapping)
            ]
            candidate_names = [
                str((row.get("function") or {}).get("name") or "")
                for row in candidate
                if isinstance(row, Mapping)
            ]
            store.capture_shadow_routing(
                session_id=session_id,
                primary_tools=primary_names,
                candidate_tools=candidate_names,
                policy_name=self._shadow_routing_policy,
                metadata={"iteration": int(iteration)},
            )
        except Exception:
            return

    @classmethod
    def _has_explicit_high_risk_intent(
        cls,
        user_message: str,
        *,
        inbound_metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        text = str(user_message or "").strip().lower()
        if any(marker in text for marker in cls._HIGH_RISK_INTENT_MARKERS):
            return True
        if re.search(
            r"\b(i|we)\s+(explicitly\s+)?(allow|approve|authorize|consent)\b.*\b(high[\s_-]?risk|dangerous)\b",
            text,
        ):
            return True
        meta = dict(inbound_metadata or {})
        if bool(meta.get("high_risk_intent")):
            return True
        for key in ("allow", "allow_patterns", "intent_markers"):
            raw = meta.get(key)
            if isinstance(raw, (list, tuple, set)):
                joined = " ".join(str(v) for v in raw)
            else:
                joined = str(raw or "")
            lowered = joined.lower()
            if any(marker in lowered for marker in cls._HIGH_RISK_INTENT_MARKERS):
                return True
        return False

    def _high_risk_block_reason(
        self,
        *,
        tool_name: str,
        executed_tools: set[str],
        explicit_high_risk_intent: bool,
    ) -> str:
        if not self._strict_runtime_tool_guard:
            return ""
        if explicit_high_risk_intent:
            return ""
        candidate = set(executed_tools)
        candidate.add(str(tool_name or "").strip())
        has_exec = "exec" in candidate
        has_messaging = bool(candidate.intersection(self._HIGH_RISK_TOOL_MESSAGING))
        has_automation = bool(candidate.intersection(self._HIGH_RISK_TOOL_AUTOMATION))
        if has_exec and (has_messaging or has_automation):
            return (
                "Blocked by safety policy: exec cannot be combined with "
                "messaging/automation tools without explicit high-risk intent."
            )
        if "read_file" in candidate and has_messaging and has_automation:
            return (
                "Blocked by safety policy: read_file + messaging + automation "
                "requires explicit high-risk intent."
            )
        return ""

    async def _disconnect_mcp(self) -> None:
        if self._mcp_stack is None:
            self._mcp_connected = False
            return
        try:
            await self._mcp_stack.aclose()
        except Exception:
            pass
        self._mcp_stack = None
        self._mcp_connected = False

    def remember(self, session_id: str, key: str, value: str) -> None:
        self._memory_store.set_fact(session_id, key, value)
        if not self._workspace:
            return
        try:
            memory = AgentMemoryStore(Path(self._workspace))
            existing = memory.read_long_term().rstrip()
            entry = f"- {str(key).strip()}: {str(value).strip()}".strip()
            if not entry or entry == "- :":
                return
            if existing:
                memory.write_long_term(existing + "\n" + entry + "\n")
            else:
                memory.write_long_term(entry + "\n")
        except Exception:
            # Best-effort long-term persistence should not break the turn flow.
            return

    def _persist_long_term_memory_note_from_user_text(self, text: str) -> None:
        if not self._workspace:
            return
        match = self._REMEMBER_NOTE_RE.match(str(text or ""))
        if match is None:
            return
        note = str(match.group(1) or "").strip()
        if not note:
            return
        line = f"- {note}"
        try:
            memory = AgentMemoryStore(Path(self._workspace))
            existing = memory.read_long_term().rstrip()
            existing_lines = {ln.strip() for ln in existing.splitlines() if ln.strip()}
            if line in existing_lines:
                return
            if existing:
                memory.write_long_term(existing + "\n" + line + "\n")
            else:
                memory.write_long_term(line + "\n")
        except Exception:
            return

    def recall(self, session_id: str, key: Optional[str] = None) -> Any:
        facts = self._memory_store.get_facts(session_id)
        if key is None:
            return facts
        return facts.get(key)

    def forget(self, session_id: str, key: Optional[str] = None) -> bool:
        if key is None:
            self._memory_store.clear_facts(session_id)
            return True
        return self._memory_store.delete_fact(session_id, key)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self._memory_store.get_history(session_id)

    def clear_memory(self, session_id: str) -> None:
        self._memory_store.clear_session(session_id)

    async def close(self) -> None:
        provider = self._provider_impl
        self._provider_impl = None
        if provider is not None:
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                await close_fn()
        await self._disconnect_mcp()

    def set_subagent_manager(self, manager: Optional["SubagentManager"]) -> None:
        self._subagent_manager = manager
        self._wire_tools()

    @staticmethod
    def _format_memory_facts(facts: Mapping[str, str]) -> str:
        lines = ["Session memory facts (use when relevant):"]
        for key, value in facts.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _history_persist_limit(self) -> int:
        return max(
            4,
            int(self._memory_config.max_history_messages),
            int(self._memory_config.memory_window) * 2,
        )

    @staticmethod
    def _extract_tools_used(tool_runs: Sequence[AgentToolRun]) -> List[str]:
        if not tool_runs:
            return []
        ordered: List[str] = []
        seen: set[str] = set()
        for run in tool_runs:
            name = str(run.name or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    async def _consolidate_memory(
        self,
        *,
        session_id: str,
        history: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        window = max(4, int(self._memory_config.memory_window))
        if len(history) <= window:
            return [dict(m) for m in history]
        keep_count = min(10, max(2, window // 2))
        cursor = self._get_last_consolidated_cursor(
            session_id=session_id,
            history_len=len(history),
        )
        archive_end = max(0, len(history) - keep_count)
        keep = [dict(m) for m in history[archive_end:]]
        archive = [dict(m) for m in history[cursor:archive_end]]
        outcome = "noop"
        if not archive:
            if len(history) > len(keep):
                self._replace_session_history(session_id, keep)
                self._set_last_consolidated_cursor(session_id=session_id, value=0)
            outcome = "no_archive"
            self._logger.info(
                "memory consolidation session=%s outcome=%s history_len=%d archive_len=%d keep_len=%d cursor=%d elapsed_ms=%.1f",
                session_id,
                outcome,
                len(history),
                0,
                len(keep),
                cursor,
                (time.perf_counter() - started) * 1000.0,
            )
            return keep

        if not self._workspace:
            self._replace_session_history(session_id, keep)
            self._set_last_consolidated_cursor(session_id=session_id, value=0)
            outcome = "no_workspace"
            self._logger.info(
                "memory consolidation session=%s outcome=%s history_len=%d archive_len=%d keep_len=%d cursor=%d elapsed_ms=%.1f",
                session_id,
                outcome,
                len(history),
                len(archive),
                len(keep),
                cursor,
                (time.perf_counter() - started) * 1000.0,
            )
            return keep

        transcript = self._format_consolidation_transcript(archive)
        try:
            memory = AgentMemoryStore(Path(self._workspace))
            self._flush_archive_to_daily_log(
                memory=memory,
                session_id=session_id,
                transcript=transcript,
                archive_len=len(archive),
            )
            old_long_term = memory.read_long_term()
            if self._should_skip_memory_consolidation_llm(transcript=transcript):
                history_entry = self._fallback_history_entry(transcript)
                if history_entry:
                    memory.append_history(history_entry)
                    self._append_memory_audit(
                        session_id=session_id,
                        scope="working_memory",
                        mutation="consolidation_history_entry",
                        reason="append consolidation history entry",
                        before="",
                        after=history_entry,
                    )
                self._set_last_consolidated_cursor(session_id=session_id, value=0)
                outcome = "skipped_short_transcript"
                self._logger.info(
                    "memory consolidation session=%s outcome=%s history_len=%d archive_len=%d keep_len=%d cursor=%d transcript_chars=%d elapsed_ms=%.1f",
                    session_id,
                    outcome,
                    len(history),
                    len(archive),
                    len(keep),
                    cursor,
                    len(transcript),
                    (time.perf_counter() - started) * 1000.0,
                )
                self._replace_session_history(session_id, keep)
                return keep
            payload = await self._request_memory_consolidation(
                transcript=transcript,
                current_memory=old_long_term,
            )
            history_entry = self._normalize_history_entry(
                str(payload.get("history_entry") or "").strip()
            )
            if history_entry:
                memory.append_history(history_entry)
            updated_memory = str(payload.get("memory_update") or "")
            if self._is_valid_memory_update(old_long_term, updated_memory):
                if updated_memory != old_long_term:
                    memory.write_long_term(updated_memory)
                    self._append_memory_audit(
                        session_id=session_id,
                        scope="long_term_memory",
                        mutation="consolidation_memory_update",
                        reason="apply LLM memory consolidation update",
                        before=old_long_term,
                        after=updated_memory,
                    )
            self._replace_session_history(session_id, keep)
            self._set_last_consolidated_cursor(session_id=session_id, value=0)
            outcome = "llm_consolidated"
        except Exception as exc:
            self._replace_session_history(session_id, keep)
            self._set_last_consolidated_cursor(session_id=session_id, value=0)
            outcome = "failed"
            self._logger.warning(
                "memory consolidation failed for session=%s: %s", session_id, exc
            )
        self._logger.info(
            "memory consolidation session=%s outcome=%s history_len=%d archive_len=%d keep_len=%d cursor=%d elapsed_ms=%.1f",
            session_id,
            outcome,
            len(history),
            len(archive),
            len(keep),
            cursor,
            (time.perf_counter() - started) * 1000.0,
        )
        self._record_memory_telemetry(
            session_id=session_id,
            outcome=outcome,
            history_len=len(history),
            archive_len=len(archive),
            keep_len=len(keep),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return keep

    def _flush_archive_to_daily_log(
        self,
        *,
        memory: AgentMemoryStore,
        session_id: str,
        transcript: str,
        archive_len: int,
    ) -> None:
        _ = append_pre_compaction_flush(
            store=memory,
            session_id=session_id,
            transcript=transcript,
            archive_len=archive_len,
            max_chars=6000,
        )

    def _should_consolidate_memory(
        self,
        *,
        session_id: str,
        history_len: int,
        turn_seq: int,
    ) -> bool:
        window = max(4, int(self._memory_config.memory_window))
        if history_len <= max(2, window):
            return False
        next_turn = self._get_session_meta_int(
            session_id=session_id,
            key="next_consolidation_turn",
            default=1,
        )
        return int(turn_seq) >= int(next_turn)

    def _set_next_consolidation_turn(self, *, session_id: str, turn_seq: int) -> None:
        spacing = max(2, int(self._memory_config.memory_window) // 3)
        self._update_session_meta(
            session_id=session_id,
            updates={"next_consolidation_turn": int(turn_seq) + int(spacing)},
        )

    def _increment_session_turn_counter(self, session_id: str) -> int:
        current = self._get_session_meta_int(
            session_id=session_id,
            key="turn_counter",
            default=0,
        )
        nxt = int(current) + 1
        self._update_session_meta(session_id=session_id, updates={"turn_counter": nxt})
        return nxt

    def _get_session_meta_int(self, *, session_id: str, key: str, default: int) -> int:
        getter = getattr(self._memory_store, "get_session_metadata", None)
        if not callable(getter):
            return int(default)
        try:
            meta = getter(session_id)
        except Exception:
            return int(default)
        if not isinstance(meta, Mapping):
            return int(default)
        try:
            return int(meta.get(str(key), default) or default)
        except Exception:
            return int(default)

    def _update_session_meta(
        self, *, session_id: str, updates: Mapping[str, Any]
    ) -> None:
        setter = getattr(self._memory_store, "update_session_metadata", None)
        if not callable(setter):
            return
        try:
            setter(session_id, dict(updates or {}))
        except Exception:
            return

    def _record_memory_telemetry(
        self,
        *,
        session_id: str,
        outcome: str,
        history_len: int,
        archive_len: int,
        keep_len: int,
        elapsed_ms: float,
    ) -> None:
        getter = getattr(self._memory_store, "get_session_metadata", None)
        setter = getattr(self._memory_store, "update_session_metadata", None)
        if not callable(getter) or not callable(setter):
            return
        try:
            meta = getter(session_id)
        except Exception:
            return
        rows = []
        if isinstance(meta, Mapping):
            rows = list(meta.get("memory_telemetry") or [])
        rows.append(
            {
                "timestamp": datetime.now().isoformat(),
                "outcome": str(outcome or "").strip().lower(),
                "history_len": int(history_len),
                "archive_len": int(archive_len),
                "keep_len": int(keep_len),
                "elapsed_ms": float(elapsed_ms),
            }
        )
        if len(rows) > 200:
            rows = rows[-200:]
        try:
            setter(session_id, {"memory_telemetry": rows})
        except Exception:
            return

    def _append_memory_audit(
        self,
        *,
        session_id: str,
        scope: str,
        mutation: str,
        reason: str,
        before: str,
        after: str,
    ) -> None:
        appender = getattr(self._memory_store, "append_memory_audit", None)
        if not callable(appender):
            return
        try:
            appender(
                session_id,
                scope=scope,
                mutation=mutation,
                reason=reason,
                before=before,
                after=after,
            )
        except Exception:
            return

    def _sync_memory_layers(
        self,
        *,
        session_id: str,
        reason: str,
        turn_id: str = "",
    ) -> None:
        history = self._memory_store.get_history(session_id)
        facts = self._memory_store.get_facts(session_id)
        working_lines: List[str] = []
        for msg in history[-8:]:
            role = str(msg.get("role") or "").strip().lower()
            content = str(msg.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            working_lines.append(f"{role}: {content}")
        working_payload = "\n".join(working_lines).strip()
        long_term_lines = [
            f"{k}: {v}" for k, v in sorted(facts.items()) if str(k).strip()
        ]
        long_term_payload = "\n".join(long_term_lines).strip()
        self._set_store_text(
            "set_working_memory",
            session_id=session_id,
            text=working_payload,
            reason=reason,
            turn_id=turn_id,
        )
        self._set_store_text(
            "set_long_term_memory",
            session_id=session_id,
            text=long_term_payload,
            reason=reason,
            turn_id=turn_id,
        )

    def _set_store_text(
        self,
        method_name: str,
        *,
        session_id: str,
        text: str,
        reason: str,
        turn_id: str = "",
    ) -> None:
        method = getattr(self._memory_store, method_name, None)
        if not callable(method):
            return
        try:
            method(
                session_id,
                text,
                reason=reason,
                turn_id=turn_id,
            )
        except Exception:
            return

    def _record_session_event(
        self,
        *,
        session_id: str,
        direction: str,
        kind: str,
        payload: Mapping[str, Any],
        turn_id: str = "",
        event_id: str = "",
        idempotency_key: str = "",
    ) -> None:
        recorder = getattr(self._memory_store, "record_event", None)
        if not callable(recorder):
            return
        try:
            recorder(
                session_id,
                direction=str(direction or "").strip().lower() or "event",
                kind=str(kind or "").strip().lower() or "event",
                payload=dict(payload or {}),
                turn_id=str(turn_id or "").strip(),
                event_id=str(event_id or "").strip(),
                idempotency_key=str(idempotency_key or "").strip(),
            )
        except Exception:
            return

    def _capture_anonymized_run_trace(
        self,
        *,
        session_id: str,
        channel: Optional[str],
        chat_id: Optional[str],
        user_message_text: str,
        result: AgentLoopResult,
        turn_id: str,
    ) -> None:
        if not self._workspace:
            return
        try:
            store = RunTraceStore(Path(self._workspace))
            store.capture_run(
                session_id=session_id,
                channel=channel,
                chat_id=chat_id,
                user_message=user_message_text,
                assistant_response=str(result.content or ""),
                tool_names=[run.name for run in result.tool_runs],
                metadata={
                    "turn_id": str(turn_id or ""),
                    "iterations": int(result.iterations),
                    "stopped_reason": str(result.stopped_reason or ""),
                },
            )
        except Exception:
            return

    def _get_store_text(self, method_name: str, *, session_id: str) -> str:
        method = getattr(self._memory_store, method_name, None)
        if not callable(method):
            return ""
        try:
            return str(method(session_id) or "")
        except Exception:
            return ""

    def _get_last_consolidated_cursor(
        self, *, session_id: str, history_len: int
    ) -> int:
        default = 0
        getter = getattr(self._memory_store, "get_session_metadata", None)
        if not callable(getter):
            return default
        try:
            meta = getter(session_id)
        except Exception:
            return default
        if not isinstance(meta, Mapping):
            return default
        try:
            cursor = int(meta.get("last_consolidated", 0) or 0)
        except Exception:
            return default
        return max(0, min(cursor, max(0, int(history_len))))

    def _set_last_consolidated_cursor(self, *, session_id: str, value: int) -> None:
        setter = getattr(self._memory_store, "update_session_metadata", None)
        if not callable(setter):
            return
        try:
            setter(session_id, {"last_consolidated": int(max(0, value))})
        except Exception:
            return

    def _replace_session_history(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        self._memory_store.clear_history(session_id)
        if not messages:
            return
        self._memory_store.append_history(
            session_id,
            [dict(m) for m in messages],
            max_messages=self._history_persist_limit(),
        )

    def _format_consolidation_transcript(
        self, history: Sequence[Mapping[str, Any]]
    ) -> str:
        lines: List[str] = []
        for item in history:
            role = str(item.get("role") or "unknown").upper()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            tools_used = item.get("tools_used")
            if isinstance(tools_used, (list, tuple)):
                used = [str(t).strip() for t in tools_used if str(t).strip()]
                if used:
                    lines.append(f"{role} [tools: {', '.join(used)}]: {content}")
                    continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    async def _request_memory_consolidation(
        self,
        *,
        transcript: str,
        current_memory: str,
    ) -> Dict[str, Any]:
        prompt = (
            "Consolidate the archived chat transcript into two outputs.\n"
            "Return strict JSON with keys: history_entry, memory_update.\n"
            "- history_entry: one compact grep-friendly line prefixed with "
            "[YYYY-MM-DD HH:MM].\n"
            "- memory_update: full updated MEMORY.md text. Keep facts stable and "
            "do not invent details.\n"
            "- If no long-term facts changed, return current memory unchanged.\n\n"
            f"Current MEMORY.md:\n{current_memory.strip()}\n\n"
            f"Archived Transcript:\n{transcript}"
        )
        save_memory_tool = [
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": (
                        "Save memory consolidation with history_entry and memory_update."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "history_entry": {
                                "type": "string",
                                "description": (
                                    "One grep-friendly summary line prefixed with "
                                    "[YYYY-MM-DD HH:MM]."
                                ),
                            },
                            "memory_update": {
                                "type": "string",
                                "description": (
                                    "Full updated MEMORY.md markdown text."
                                ),
                            },
                        },
                        "required": ["history_entry", "memory_update"],
                    },
                },
            }
        ]

        resp = await self._llm_callable(
            [{"role": "system", "content": prompt}],
            save_memory_tool,
            self.model,
        )
        for call in self._sanitize_tool_calls(self._extract_tool_calls(resp)):
            if str(call.get("name") or "").strip() != "save_memory":
                continue
            args = self._normalize_args(call.get("arguments"))
            if isinstance(args, Mapping):
                return dict(args)
        content = self._strip_think(str(resp.get("content") or "").strip())
        parsed = self._try_parse_json_object(content)
        if parsed is None:
            return {
                "history_entry": self._fallback_history_entry(transcript),
                "memory_update": current_memory,
            }
        return parsed

    def _should_skip_memory_consolidation_llm(self, *, transcript: str) -> bool:
        return len(str(transcript or "").strip()) < int(
            self._MIN_CONSOLIDATION_TRANSCRIPT_CHARS
        )

    @staticmethod
    def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return dict(parsed)
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return dict(parsed)
        except Exception:
            return None
        return None

    def _fallback_history_entry(self, transcript: str) -> str:
        lines = [ln.strip() for ln in str(transcript or "").splitlines() if ln.strip()]
        summary = " | ".join(lines[:2])[:500]
        if not summary:
            summary = "Session history consolidated."
        return self._normalize_history_entry(summary)

    @staticmethod
    def _normalize_history_entry(entry: str) -> str:
        text = str(entry or "").strip()
        if not text:
            return ""
        if not text.startswith("["):
            stamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
            text = f"{stamp} {text}"
        return text

    @staticmethod
    def _is_valid_memory_update(previous: str, candidate: str) -> bool:
        new_text = str(candidate or "")
        old_text = str(previous or "")
        if new_text == old_text:
            return True
        if not new_text.strip():
            return not old_text.strip()
        if old_text.strip():
            old_len = len(old_text.strip())
            new_len = len(new_text.strip())
            if new_len < max(40, int(old_len * 0.2)):
                return False
            if new_len > int(old_len * 4.0):
                return False
        return True

    def _wire_tools(self) -> None:
        self._set_tool_context(channel="cli", chat_id="direct")
        if self._subagent_manager is None and (
            isinstance(self._tools.get("spawn"), SpawnTool)
            or isinstance(self._tools.get("list_tasks"), ListTasksTool)
            or isinstance(self._tools.get("cancel_task"), CancelTaskTool)
        ):
            self._subagent_manager = self._create_default_subagent_manager()

        spawn_tool = self._tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool) and self._subagent_manager is not None:
            spawn_tool.set_spawn_callback(self._subagent_manager.spawn)

        list_tasks_tool = self._tools.get("list_tasks")
        if (
            isinstance(list_tasks_tool, ListTasksTool)
            and self._subagent_manager is not None
        ):
            list_tasks_tool._list_tasks_callback = self._subagent_manager.list_tasks

        cancel_task_tool = self._tools.get("cancel_task")
        if (
            isinstance(cancel_task_tool, CancelTaskTool)
            and self._subagent_manager is not None
        ):
            cancel_task_tool._cancel_callback = self._subagent_manager.cancel

        swarm_tool = self._tools.get("run_swarm")
        if isinstance(swarm_tool, SwarmTool):
            swarm_tool.set_swarm_callback(self._create_swarm_manager_and_run)

    def _set_tool_context(
        self,
        *,
        channel: Optional[str],
        chat_id: Optional[str],
    ) -> None:
        resolved_channel = channel or "cli"
        resolved_chat_id = chat_id or "direct"
        message_tool = self._tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(resolved_channel, resolved_chat_id)

        cron_tool = self._tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(resolved_channel, resolved_chat_id)

        spawn_tool = self._tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(resolved_channel, resolved_chat_id)

        automation_tool = self._tools.get("automation_schedule")
        if isinstance(automation_tool, AutomationSchedulerTool):
            automation_tool.set_context(resolved_channel, resolved_chat_id)

    def _create_default_subagent_manager(self) -> Optional["SubagentManager"]:
        try:
            from .subagent import SubagentManager, build_subagent_tools_registry
        except Exception:
            return None

        def _loop_factory() -> Awaitable["AgentLoop"]:
            async def _create():
                tools = await build_subagent_tools_registry(
                    Path(self._workspace) if self._workspace else None,
                    allowed_read_roots=self._allowed_read_roots,
                    mcp_servers=self._mcp_servers,
                )
                return AgentLoop(
                    tools=tools,
                    llm_callable=self._llm_callable,
                    model=self.model,
                    max_iterations=min(self._max_iterations, 10),
                    workspace=self._workspace,
                    allowed_read_roots=self._allowed_read_roots,
                    mcp_servers=self._mcp_servers,
                    llm_timeout_seconds=self._llm_timeout_seconds,
                    strict_runtime_tool_guard=self._strict_runtime_tool_guard,
                )

            return _create()

        workspace_path = Path(self._workspace) if self._workspace else None
        return SubagentManager(loop_factory=_loop_factory, workspace=workspace_path)

    async def _create_swarm_manager_and_run(
        self, task: str, max_turns: int = 5, agents: list[str] | None = None
    ) -> str:
        try:
            from .swarm import SwarmAgent, SwarmManager
        except Exception as e:
            return f"Cannot initialize swarm: {e}"

        def _loop_factory() -> "AgentLoop":
            # For brevity in the swarm, limit iterations heavily per turn
            return AgentLoop(
                tools=self._tools,
                llm_callable=self._llm_callable,
                model=self.model,
                max_iterations=min(self._max_iterations, 3),
                workspace=self._workspace,
                allowed_read_roots=self._allowed_read_roots,
                mcp_servers=self._mcp_servers,
                llm_timeout_seconds=self._llm_timeout_seconds,
                strict_runtime_tool_guard=self._strict_runtime_tool_guard,
            )

        manager = SwarmManager()

        roles = agents or []
        if not roles:
            # Look for "(Role1, Role2, ...)" or "(Role1 x Role2)" anywhere in the task
            import re

            match = re.search(r"\(([^)]+)\)", task)
            if match:
                raw_roles = match.group(1)
                # Normalize separators: split by comma, x, &, or " and "
                normalized = re.sub(r"\s+(?:x|&|and)\s+", ", ", raw_roles)
                roles = [r.strip() for r in normalized.split(",") if r.strip()]
            elif "specialized agents (" in task:
                match = re.search(r"specialized agents \((.*?)\)", task)
                if match:
                    roles = [r.strip() for r in match.group(1).split(",")]
            elif "agents:" in task.lower():
                # agents: Role1, Role2
                match = re.search(r"agents:\s*(.*?)(?:\n|$)", task, re.IGNORECASE)
                if match:
                    roles = [r.strip() for r in match.group(1).split(",")]

        if not roles:
            roles = ["Planner", "Researcher", "Coder", "Reviewer"]

        for role in roles:
            prompt = f"You are the {role}. "
            if "planner" in role.lower():
                prompt += "Break down the task into steps. Read files if needed. Guide the others."
            elif "researcher" in role.lower():
                prompt += "Find facts, patterns, and relevant files using list_dir and read_file."
            elif "coder" in role.lower() or "implement" in role.lower():
                prompt += "Write and edit code to fulfill the task using your tools."
            elif "reviewer" in role.lower() or "security" in role.lower():
                prompt += "Check the work so far. If complete, say 'TASK COMPLETE'."
            else:
                prompt += "Contribute to the task based on your specialized knowledge."

            manager.register_agent(SwarmAgent(role, role, prompt, _loop_factory))

        return await manager.run_swarm(task, max_turns=max_turns)

    _TOOL_SELECTOR_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "use",
        "with",
        "you",
        "your",
    }
    _DEFAULT_TOOL_SELECTION_MAX = 6
    _WEB_INTENT_TOKENS = frozenset(
        {
            "search",
            "web",
            "website",
            "threejs",
            "three",
            "3d",
            "url",
            "weather",
            "forecast",
            "temperature",
            "news",
            "price",
            "stock",
            "latest",
            "current",
            "live",
            "today",
        }
    )
    _DEFAULT_TOOL_PRIORITY = (
        "mcp_browser",
        "mcp_browser_navigate",
        "mcp_browser_snapshot",
        "mcp_browser_type",
        "mcp_browser_click",
        "mcp_browser_wait",
        "gui_start_realtime_stream",
        "gui_stop_realtime_stream",
        "gui_label_behavior_segments",
        "gui_segment_track_video",
        "gui_open_video",
        "gui_open_url",
        "gui_open_in_browser",
        "gui_open_threejs",
        "gui_open_threejs_example",
        "clawhub_search_skills",
        "clawhub_install_skill",
        "gui_web_get_dom_text",
        "gui_web_click",
        "gui_web_type",
        "gui_web_scroll",
        "gui_web_find_forms",
        "gui_web_run_steps",
        "gui_open_pdf",
        "gui_set_frame",
        "gui_set_ai_text_prompt",
        "gui_run_ai_text_segmentation",
        "gui_track_next_frames",
        "gui_context",
        "run_swarm",
        "read_file",
        "list_dir",
        "video_info",
        "video_sample_frames",
        "open_pdf",
        "extract_pdf_text",
        "download_pdf",
        "web_search",
    )
    _REMEMBER_NOTE_RE = re.compile(
        r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*$",
        flags=re.IGNORECASE,
    )
    _POST_TOOL_SYSTEM_GUIDANCE = (
        "Use the tool results to decide the next best action. "
        "Either call another tool with concrete arguments or provide a concise "
        "final answer. For live web search requests, prefer MCP browser workflow "
        "by default: navigate to a search engine, type the query, snapshot/parse "
        "the results, then continue. For GUI web tasks, prefer `gui_web_run_steps` "
        "before claiming browsing limits. Do not reveal private chain-of-thought."
    )
    _BROWSER_WORKFLOW_TOOL_NAMES = frozenset(
        {
            "mcp_browser",
            "mcp_browser_navigate",
            "mcp_browser_snapshot",
            "mcp_browser_type",
            "mcp_browser_click",
            "mcp_browser_wait",
            "mcp_browser_screenshot",
            "mcp_browser_scroll",
        }
    )
    _TOKENIZE_RE = re.compile(r"[a-zA-Z0-9_]+")

    @classmethod
    def _tokenize_text(cls, text: str) -> List[str]:
        raw = cls._TOKENIZE_RE.findall(str(text or "").lower())
        return [t for t in raw if len(t) > 1 and t not in cls._TOOL_SELECTOR_STOPWORDS]

    @classmethod
    def _build_tool_tokens(cls, schema: Mapping[str, Any]) -> set[str]:
        fn = schema.get("function")
        if not isinstance(fn, Mapping):
            return set()
        parts: List[str] = [
            str(fn.get("name") or ""),
            str(fn.get("description") or ""),
        ]
        params = fn.get("parameters")
        if isinstance(params, Mapping):
            props = params.get("properties")
            if isinstance(props, Mapping):
                for key, value in props.items():
                    parts.append(str(key))
                    if isinstance(value, Mapping):
                        parts.append(str(value.get("description") or ""))
                        enum_values = value.get("enum")
                        if isinstance(enum_values, list):
                            parts.extend(str(item) for item in enum_values)
        tokens: set[str] = set()
        for part in parts:
            tokens.update(cls._tokenize_text(part))
        return tokens

    @classmethod
    def _score_tool_schema(
        cls,
        tool_entry: _ToolSchemaIndex,
        query_tokens: Sequence[str],
    ) -> int:
        if not query_tokens:
            return 0
        score = 0
        for token in query_tokens:
            if token in tool_entry.tokens:
                score += 2
            if token and token in tool_entry.name:
                score += 3
            if token and token in tool_entry.desc:
                score += 1
        return score

    @classmethod
    def _compile_tool_index(
        cls,
        all_tool_definitions: Sequence[Mapping[str, Any]],
    ) -> List[_ToolSchemaIndex]:
        compiled: List[_ToolSchemaIndex] = []
        for schema in all_tool_definitions:
            schema_dict = dict(schema)
            fn = schema_dict.get("function")
            if not isinstance(fn, Mapping):
                continue
            compiled.append(
                _ToolSchemaIndex(
                    schema=schema_dict,
                    name=str(fn.get("name") or "").lower(),
                    desc=str(fn.get("description") or "").lower(),
                    tokens=frozenset(cls._build_tool_tokens(schema_dict)),
                )
            )
        return compiled

    @staticmethod
    def _tool_signature(
        all_tool_definitions: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[str, str], ...]:
        pairs: List[tuple[str, str]] = []
        for schema in all_tool_definitions:
            fn = schema.get("function")
            if not isinstance(fn, Mapping):
                continue
            pairs.append(
                (
                    str(fn.get("name") or ""),
                    str(fn.get("description") or ""),
                )
            )
        return tuple(sorted(pairs))

    def _get_cached_tool_index(
        self,
        all_tool_definitions: Sequence[Mapping[str, Any]],
        signature: tuple[tuple[str, str], ...],
    ) -> List[_ToolSchemaIndex]:
        if self._cached_tool_signature == signature and len(
            self._cached_tool_index
        ) == len(signature):
            return list(self._cached_tool_index)
        compiled = self._compile_tool_index(all_tool_definitions)
        self._cached_tool_signature = signature
        self._cached_tool_index = list(compiled)
        self._cached_default_tools = []
        return compiled

    def _get_cached_default_tools(
        self,
        tool_index: Sequence[_ToolSchemaIndex],
        signature: tuple[tuple[str, str], ...],
    ) -> List[Dict[str, Any]]:
        if self._cached_tool_signature == signature and self._cached_default_tools:
            return list(self._cached_default_tools)
        tools = [entry.schema for entry in tool_index]
        defaults = self._select_default_tool_definitions(tools)
        self._cached_default_tools = list(defaults)
        return defaults

    def _select_relevant_tool_definitions(
        self,
        *,
        all_tool_definitions: Sequence[Mapping[str, Any]],
        tool_index: Sequence[_ToolSchemaIndex],
        default_tool_definitions: Sequence[Mapping[str, Any]],
        user_message_text: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        tools = [entry.schema for entry in tool_index] or [
            dict(t) for t in all_tool_definitions
        ]
        if len(tools) <= 2:
            return [dict(t) for t in tools]
        user_tokens = self._tokenize_text(user_message_text)
        if not user_tokens:
            return [dict(t) for t in default_tool_definitions]

        # Use recent interaction context to improve follow-up turns.
        tail_text_parts: List[str] = []
        for msg in messages[-6:]:
            role = str(msg.get("role") or "")
            if role in {"assistant", "tool"}:
                tail_text_parts.append(str(msg.get("content") or ""))
        tail_tokens = self._tokenize_text(" ".join(tail_text_parts))
        query_tokens = list(dict.fromkeys([*user_tokens, *tail_tokens]))

        scored: List[tuple[int, Dict[str, Any]]] = []
        entries = list(tool_index) or self._compile_tool_index(tools)
        web_intent = bool(self._WEB_INTENT_TOKENS.intersection(query_tokens))
        for entry in entries:
            score = self._score_tool_schema(entry, query_tokens)
            if self._browser_first_for_web and web_intent:
                if entry.name in self._BROWSER_WORKFLOW_TOOL_NAMES:
                    score += 24
                elif "browser" in entry.name or "browser" in entry.desc:
                    score += 12
                if entry.name == "web_search":
                    score = max(1, score - 6)
            if score > 0:
                scored.append((score, entry.schema))
        if not scored:
            return [dict(t) for t in default_tool_definitions]

        scored.sort(
            key=lambda item: (
                -item[0],
                str((item[1].get("function") or {}).get("name") or ""),
            )
        )
        max_tools = min(6, max(3, len(scored)))
        selected = [schema for _, schema in scored[:max_tools]]
        selected_names = {
            str((schema.get("function") or {}).get("name") or "") for schema in selected
        }
        if (
            "read_file" in self._tools
            and "read_file" not in selected_names
            and len(selected) < max_tools
        ):
            read_file_schema = next(
                (
                    schema
                    for schema in tools
                    if str((schema.get("function") or {}).get("name") or "")
                    == "read_file"
                ),
                None,
            )
            if read_file_schema is not None:
                selected.append(read_file_schema)
        return selected or [dict(t) for t in default_tool_definitions]

    @classmethod
    def _select_default_tool_definitions(
        cls,
        tools: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if len(tools) <= cls._DEFAULT_TOOL_SELECTION_MAX:
            return [dict(t) for t in tools]

        by_name: Dict[str, Dict[str, Any]] = {}
        for schema in tools:
            fn = schema.get("function")
            if not isinstance(fn, Mapping):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            by_name.setdefault(name, dict(schema))

        selected: List[Dict[str, Any]] = []
        selected_names: set[str] = set()
        for name in cls._DEFAULT_TOOL_PRIORITY:
            schema = by_name.get(name)
            if schema is None:
                continue
            selected.append(schema)
            selected_names.add(name)
            if len(selected) >= cls._DEFAULT_TOOL_SELECTION_MAX:
                return selected

        remaining_names = sorted(
            name for name in by_name.keys() if name not in selected_names
        )
        for name in remaining_names:
            selected.append(by_name[name])
            if len(selected) >= cls._DEFAULT_TOOL_SELECTION_MAX:
                break
        return selected or [dict(t) for t in tools[: cls._DEFAULT_TOOL_SELECTION_MAX]]

    def _sanitize_tool_calls(
        self, tool_calls: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in tool_calls:
            call_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            args = self._normalize_args(item.get("arguments"))
            if not call_id:
                call_id = f"call_{len(deduped)}"
            signature = (
                f"{call_id}:{name}:"
                f"{json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)}"
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append({"id": call_id, "name": name, "arguments": args})
        return deduped

    def _append_post_tool_guidance(self, messages: List[Dict[str, Any]]) -> None:
        if not self._interleave_post_tool_guidance:
            return
        if messages:
            last = messages[-1]
            if (
                str(last.get("role") or "") == "system"
                and str(last.get("content") or "").strip()
                == self._POST_TOOL_SYSTEM_GUIDANCE
            ):
                return
        messages.append({"role": "system", "content": self._POST_TOOL_SYSTEM_GUIDANCE})

    def _extract_tool_calls(self, response: Mapping[str, Any]) -> List[Dict[str, Any]]:
        raw_calls = response.get("tool_calls") or []
        normalized: List[Dict[str, Any]] = []
        for item in raw_calls:
            if not isinstance(item, Mapping):
                continue
            call_id = str(item.get("id") or "")
            name = str(item.get("name") or "")
            args = item.get("arguments", {})
            if not name:
                function = item.get("function")
                if isinstance(function, Mapping):
                    name = str(function.get("name") or "")
                    args = function.get("arguments", args)
            if not call_id:
                call_id = f"call_{len(normalized)}"
            if not name:
                continue
            normalized.append({"id": call_id, "name": name, "arguments": args})

        if not normalized:
            content = str(response.get("content") or "")
            import re
            import json

            matches = re.finditer(
                r"<\|tool_call_begin\|>\s*(?:functions\.)?([a-zA-Z0-9_]+)(?::[a-zA-Z0-9_-]+)?\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
                content,
                re.DOTALL,
            )
            for i, match in enumerate(matches):
                name = match.group(1).strip()
                args_str = match.group(2).strip()
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"_raw": args_str}
                normalized.append(
                    {"id": f"call_parsed_{i}", "name": name, "arguments": args}
                )
        return normalized

    def _to_openai_tool_call(self, call: Mapping[str, Any]) -> Dict[str, Any]:
        args = call.get("arguments")
        if isinstance(args, str):
            args_json = args
        else:
            args_json = json.dumps(args or {}, ensure_ascii=False)
        return {
            "id": str(call.get("id") or ""),
            "type": "function",
            "function": {
                "name": str(call.get("name") or ""),
                "arguments": args_json,
            },
        }

    @staticmethod
    def _strip_think(text: str | None) -> str:
        """Remove <think>...</think> sections from model content."""
        raw = str(text or "")
        if not raw:
            return ""
        return re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

    @staticmethod
    def _tool_hint(tool_calls: Sequence[Mapping[str, Any]]) -> str:
        """Compact string describing one or more tool calls."""

        def _format_call(call: Mapping[str, Any]) -> str:
            name = str(call.get("name") or "").strip() or "tool"
            args = call.get("arguments")
            first_value: Any = None
            if isinstance(args, Mapping) and args:
                try:
                    first_value = next(iter(args.values()))
                except Exception:
                    first_value = None
            if isinstance(first_value, str):
                preview = first_value[:40]
                if len(first_value) > 40:
                    preview = f"{preview}..."
                return f'{name}("{preview}")'
            return name

        return ", ".join(_format_call(call) for call in tool_calls if call)

    @staticmethod
    async def _dispatch_progress(
        callback: ProgressCallback,
        content: str,
    ) -> None:
        """Dispatch progress update to sync or async callback."""
        outcome = callback(content)
        if inspect.isawaitable(outcome):
            await outcome

    @staticmethod
    def _normalize_args(raw_args: Any) -> Dict[str, Any]:
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if isinstance(raw_args, str):
            text = raw_args.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return dict(parsed)
            except Exception:
                return {"_raw": raw_args}
            return {"_raw": raw_args}
        return {"_raw": raw_args}

    def _build_default_llm_callable(
        self,
        *,
        profile: Optional[str],
        provider: Optional[str],
        model: Optional[str],
    ) -> tuple[LLMCallable, str, Any]:
        cfg = resolve_llm_config(
            profile=profile,
            provider=provider,
            model=model,
            persist=False,
        )
        provider_name = str(cfg.provider or "").strip().lower()
        openai_compat_names = {"openai", "ollama", "openrouter", "aihubmix", "vllm"}
        model_name = str(cfg.model)
        if provider_name in openai_compat_names:
            resolved = resolve_openai_compat(cfg)
            model_name = resolved.model
            provider_impl = OpenAICompatProvider(resolved=resolved)
        else:
            provider_impl = LiteLLMProvider(
                provider_name=provider_name or None,
                api_key=cfg.params.get("api_key"),
                api_base=cfg.params.get("base_url") or cfg.params.get("host"),
                default_model=model_name,
                extra_headers=cfg.params.get("extra_headers"),
            )

        async def _call(
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]],
            model_id: str,
            on_token: Optional[Callable[[str], None]] = None,
        ) -> Mapping[str, Any]:
            resp = await provider_impl.chat(
                messages=list(messages),
                tools=list(tools) if tools else None,
                model=model_id,
                temperature=self._default_temperature,
                on_token=on_token,
            )
            tool_calls: List[Dict[str, Any]] = []
            for tc in resp.tool_calls:
                tool_calls.append(
                    {"id": tc.id, "name": tc.name, "arguments": dict(tc.arguments)}
                )
            return {
                "content": resp.content or "",
                "tool_calls": tool_calls,
                "finish_reason": resp.finish_reason,
                "usage": dict(resp.usage),
                "reasoning_content": resp.reasoning_content,
            }

        return _call, model_name, provider_impl
