from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
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

from annolid.utils.llm_settings import (
    load_llm_settings,
    provider_kind,
    resolve_agent_runtime_config,
    resolve_llm_config,
)

from .context import AgentContextBuilder
from .eval.telemetry import RunTraceStore
from .meta_learning import AgentMetaLearner
from .memory import AgentMemoryStore
from .memory_store.flush import append_pre_compaction_flush
from .providers import (
    CodexCLIProvider,
    UnifiedLLMProvider,
    OpenAICodexProvider,
    OpenAICompatProvider,
    raise_for_error_response,
    resolve_codex_cli,
    resolve_openai_codex,
    resolve_openai_compat,
)
from .context_compaction import (
    ContextCompactionDecision,
    has_unresolved_tool_tasks,
    plan_context_compaction,
)
from .tool_execution import (
    ThinkTagStreamParser,
    build_tool_only_fallback_answer,
    can_skip_empty_final_repair,
    compact_tool_result_for_llm,
    extract_snapshot_media_path,
    extract_tool_result_preview,
    extract_tool_result_preview_from_payload,
    is_concurrency_safe_tool_call,
    partition_tool_calls_for_execution,
    repair_tool_arguments_for_execution,
    tool_result_looks_user_ready,
)
from .tool_selector import (
    ANALYTICS_INTENT_TOKENS,
    ANALYTICS_TOOL_NAMES,
    BROWSER_WORKFLOW_TOOL_NAMES,
    DEFAULT_TOOL_PRIORITY,
    DEFAULT_TOOL_SELECTION_MAX,
    TOOL_SELECTOR_STOPWORDS,
    ToolSchemaIndex as _ToolSchemaIndex,
    VCS_INTENT_TOKENS,
    VCS_TOOL_NAMES,
    WEB_INTENT_TOKENS,
    build_tool_tokens,
    compile_tool_index,
    score_tool_schema,
    select_default_tool_definitions,
    select_relevant_tool_definitions,
    tokenize_text,
    tool_signature,
)
from .tools import FunctionToolRegistry
from .tools.policy import (
    HIGH_RISK_INTENT_MARKERS,
    HIGH_RISK_TOOL_AUTOMATION,
    HIGH_RISK_TOOL_MESSAGING,
    HIGH_RISK_TOOL_RUNTIME_EXEC,
    classify_destructive_tool_call,
)
from .observability import emit_governance_event
from .tools.function_builtin import (
    AutomationSchedulerTool,
    CancelTaskTool,
    CronTool,
    ListTasksTool,
    MessageTool,
    SpawnBehaviorSubagentTool,
    SpawnTool,
)
from annolid.core.agent.tools.swarm_tool import SwarmTool
from annolid.core.agent.swarm_budget import resolve_swarm_turn_budget
from .tool_call_utils import (
    extract_text_tool_calls,
    normalize_tool_arguments,
    sanitize_tool_call_id,
    sanitize_tool_call_id_segment,
    sanitize_tool_call_requests,
    sanitize_tool_name,
    strip_tool_call_markup,
)

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
    _HIGH_RISK_INTENT_MARKERS = tuple(sorted(HIGH_RISK_INTENT_MARKERS))
    _HIGH_RISK_TOOL_MESSAGING = HIGH_RISK_TOOL_MESSAGING
    _HIGH_RISK_TOOL_RUNTIME_EXEC = HIGH_RISK_TOOL_RUNTIME_EXEC
    _HIGH_RISK_TOOL_AUTOMATION = HIGH_RISK_TOOL_AUTOMATION
    _HIGH_RISK_TOOL_RUNTIME_AUTOMATION = frozenset(
        {"automation_schedule", "spawn", "spawn_behavior_subagent"}
    )
    _FINAL_ANSWER_REPAIR_PROMPT = (
        "Provide the final user-facing answer now using only the tool results above. "
        "Do not call any more tools. Keep it concise, concrete, and directly actionable."
    )
    _MAX_TOOL_RESULT_CHARS_FOR_LLM = 32000
    _MAX_CONCURRENT_READ_ONLY_TOOLS = 4
    _CONTEXT_COMPACT_MAX_MESSAGES = 48
    _CONTEXT_COMPACT_MAX_NON_SYSTEM_MESSAGES = 36

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
        tool_permission_context: Optional[Any] = None,
        tool_approval_callback: Optional[
            Callable[[str, Dict[str, Any], str], Any]
        ] = None,
        max_selected_tools: Optional[int] = None,
    ) -> None:
        self._tools = tools
        self._tool_permission_context = tool_permission_context
        self._tool_approval_callback = tool_approval_callback
        self._max_selected_tools = max_selected_tools
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
        self._mcp_connecting = False
        self._mcp_ref_count = 0
        self._mcp_registered_tool_names: set[str] = set()
        self._mcp_lock = asyncio.Lock()
        self._exclusive_mcp_run_lock = asyncio.Lock()
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
        self._cached_tool_signature: Optional[tuple[tuple[str, str, str], ...]] = None
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
        self._meta_learner: AgentMetaLearner | None = None
        if self._workspace:
            self._meta_learner = AgentMetaLearner(Path(self._workspace))

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
        async with self._mcp_run_guard():
            return await self._run_impl(
                user_message,
                session_id=session_id,
                history=history,
                system_prompt=system_prompt,
                use_memory=use_memory,
                channel=channel,
                chat_id=chat_id,
                media=media,
                skill_names=skill_names,
                on_progress=on_progress,
                inbound_metadata=inbound_metadata,
            )

    async def _run_impl(
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
        empty_final_repair_passes = 0
        message_build_ms = 0.0
        mcp_connect_ms = 0.0
        loop_result: Optional[AgentLoopResult] = None
        context_compaction_runs = 0
        context_messages_trimmed = 0
        context_compaction_skipped_active_tasks = 0
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
            user_persisted_early = False
            if memory_enabled and user_message_text.strip():
                try:
                    self._memory_store.append_history(
                        session_id,
                        [{"role": "user", "content": user_message_text}],
                        max_messages=self._history_persist_limit(),
                    )
                    user_persisted_early = True
                    self._sync_memory_layers(
                        session_id=session_id,
                        reason="append_user_early",
                        turn_id=turn_id,
                    )
                except Exception as exc:
                    self._logger.warning(
                        "annolid-bot early user persistence failed session=%s turn=%s err=%s",
                        session_id,
                        turn_id,
                        exc,
                    )

            def _rollback_early_user_only_turn_if_needed(
                *,
                final_text: str,
                tool_runs_for_turn: Sequence[AgentToolRun],
            ) -> None:
                if not (memory_enabled and user_persisted_early):
                    return
                if str(final_text or "").strip():
                    return
                if tool_runs_for_turn:
                    return
                try:
                    history_rows = list(self._memory_store.get_history(session_id))
                    if not history_rows:
                        return
                    last = dict(history_rows[-1] or {})
                    if str(last.get("role") or "") != "user":
                        return
                    if str(last.get("content") or "") != user_message_text:
                        return
                    remaining = [dict(row) for row in history_rows[:-1]]
                    self._memory_store.clear_history(session_id)
                    if remaining:
                        self._memory_store.append_history(
                            session_id,
                            remaining,
                            max_messages=self._history_persist_limit(),
                        )
                    self._sync_memory_layers(
                        session_id=session_id,
                        reason="rollback_user_early_empty",
                        turn_id=turn_id,
                    )
                except Exception as exc:
                    self._logger.warning(
                        "annolid-bot early user rollback failed session=%s turn=%s err=%s",
                        session_id,
                        turn_id,
                        exc,
                    )

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
            empty_final_repair_used = False
            unknown_tool_recovery_used = False

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

                think_parser = ThinkTagStreamParser()
                last_update = time.perf_counter()
                last_progress_at = 0.0
                last_progress_text = ""
                pending_progress_text = ""
                runtime_loop = asyncio.get_running_loop()

                def _schedule_progress(text: str) -> None:
                    nonlocal last_progress_text, last_progress_at, pending_progress_text
                    if on_progress is None or not text or text == last_progress_text:
                        return
                    last_progress_text = text
                    pending_progress_text = ""
                    last_progress_at = time.perf_counter()
                    progress_task = runtime_loop.create_task(
                        self._dispatch_progress(on_progress, text)
                    )

                    def _log_progress_failure(task: asyncio.Task[Any]) -> None:
                        with suppress(Exception):
                            err = task.exception()
                            if err is not None:
                                self._logger.warning(
                                    "on_progress callback failed: %s",
                                    err,
                                )

                    progress_task.add_done_callback(_log_progress_failure)

                def _on_llm_token(token: str) -> None:
                    nonlocal last_update, pending_progress_text
                    raw_token = str(token or "")
                    if not raw_token:
                        return
                    thought_delta, output_delta = think_parser.feed(raw_token)
                    if not thought_delta and not output_delta:
                        return

                    now = time.perf_counter()
                    if on_progress is not None:
                        progress_text = (
                            think_parser.thought_text.strip()
                            or think_parser.output_text.strip()
                        )
                        if progress_text and progress_text != last_progress_text:
                            if (
                                last_progress_at == 0.0
                                or (now - last_progress_at) >= 0.05
                            ):
                                _schedule_progress(progress_text)
                            else:
                                pending_progress_text = progress_text

                    # Throttled visualizer update (every 500ms or so)
                    if now - last_update > 0.5:
                        last_update = now
                        current_thought = think_parser.thought_text
                        current_output = think_parser.output_text

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
                        self._notify_swarm_visualizer(
                            active_node_id,
                            "active",
                            f"Output: {short_output}"
                            if short_output
                            else "Thinking...",
                            thinking=short_thought,
                            parent=parent_id.lower() if parent_id else "",
                            context="streaming",
                        )

                compaction = self._plan_context_compaction(messages)
                if compaction.skipped_due_to_active_tasks:
                    context_compaction_skipped_active_tasks += 1
                compacted_messages = list(compaction.messages)
                trimmed_count = int(compaction.trimmed_count)
                if trimmed_count > 0:
                    messages = compacted_messages
                    context_compaction_runs += 1
                    context_messages_trimmed += trimmed_count
                elif (
                    not compaction.skipped_due_to_active_tasks
                    and compacted_messages != messages
                ):
                    messages = compacted_messages
                    context_compaction_runs += 1

                llm_cycle_started = time.perf_counter()
                try:
                    response, llm_elapsed_ms = await self._execute_llm_cycle(
                        session_id=session_id,
                        iteration=iteration,
                        messages=messages,
                        tool_definitions=tool_definitions,
                        on_token=_on_llm_token,
                    )
                finally:
                    think_parser.flush()
                    if on_progress is not None:
                        flushed_text = (
                            pending_progress_text
                            or think_parser.thought_text.strip()
                            or think_parser.output_text.strip()
                        )
                        if flushed_text and flushed_text != last_progress_text:
                            _schedule_progress(flushed_text)
                    llm_total_ms += (time.perf_counter() - llm_cycle_started) * 1000.0

                assistant_text = str(response.get("content") or "")
                reasoning = str(response.get("reasoning_content") or "").strip()
                if reasoning and not assistant_text.startswith("<think>"):
                    assistant_text = (
                        f"<think>\n{reasoning}\n</think>\n\n{assistant_text}".strip()
                    )
                    response["content"] = assistant_text

                raw_extracted_calls = self._extract_tool_calls(response)
                tool_calls = self._sanitize_tool_calls(raw_extracted_calls)
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
                        progress_text = (
                            self._extract_think(assistant_text)
                            or self._strip_think(assistant_text)
                            or self._tool_hint(tool_calls)
                        )
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
                    self._notify_swarm_visualizer(
                        active_node_id,
                        "active",
                        f"Executing: {tools_str}",
                        thinking=reasoning or self._strip_think(assistant_text),
                        context="tool",
                    )

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
                    clean_assistant_text = strip_tool_call_markup(assistant_text)

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

                    self._notify_swarm_visualizer(
                        active_node_id, "idle", "Awaiting Tasks", context="idle"
                    )

                    tool_exec_total_ms += cycle_exec_ms
                    tool_call_count += cycle_call_count
                    continue
                else:
                    # One-shot self-correction when the model hallucinated an unregistered
                    # (but syntactically valid) tool name and emitted no user-facing text.
                    stripped_visible = self._strip_think(
                        strip_tool_call_markup(assistant_text)
                    )
                    unknown_valid_names = [
                        clean
                        for clean in (
                            sanitize_tool_name(c.get("name"))
                            for c in raw_extracted_calls
                        )
                        if clean
                    ]
                    if (
                        unknown_valid_names
                        and not stripped_visible
                        and not unknown_tool_recovery_used
                        and iteration < self._max_iterations
                        and self._tools.tool_names
                    ):
                        unknown_tool_recovery_used = True
                        available_names = [
                            str((t.get("function") or {}).get("name") or "")
                            for t in tool_definitions
                            if isinstance(t, Mapping)
                        ]
                        available_names = [n for n in available_names if n] or list(
                            self._tools.tool_names[:20]
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    f"Unavailable tool call(s) ignored: {', '.join(unknown_valid_names)}. "
                                    f"Use ONLY available tools ({', '.join(available_names)}) "
                                    "or answer directly without tools."
                                ),
                            }
                        )
                        continue

                    text_tools = re.findall(
                        r"(?:functions\.|<\|tool_call_begin\|>\s*)([a-zA-Z0-9_]+)",
                        assistant_text,
                    )
                    if text_tools:
                        tools_str = ", ".join(set(text_tools))
                        self._notify_swarm_visualizer(
                            active_node_id,
                            "active",
                            f"Executing: {tools_str}",
                            context="text-tool",
                        )
                    self._notify_swarm_visualizer(
                        active_node_id,
                        "idle",
                        "Awaiting Tasks",
                        context="text-tool reset",
                    )

                repeated_tool_cycles = 0
                last_tool_cycle_signature = None

                final_content = self._strip_think(
                    strip_tool_call_markup(assistant_text)
                )
                if (
                    not str(final_content or "").strip()
                    and tool_runs
                    and not empty_final_repair_used
                ):
                    if self._can_skip_empty_final_repair(tool_runs=tool_runs):
                        final_content = self._build_tool_only_fallback_answer(
                            tool_runs=tool_runs
                        )
                        self._logger.info(
                            "annolid-bot empty final response session=%s model=%s iteration=%d; synthesized from tool output without repair pass",
                            session_id,
                            self.model,
                            iteration,
                        )
                    else:
                        empty_final_repair_used = True
                        empty_final_repair_passes += 1
                        self._logger.info(
                            "annolid-bot empty final response session=%s model=%s iteration=%d; requesting no-tool finalization pass",
                            session_id,
                            self.model,
                            iteration,
                        )
                        repair_started = time.perf_counter()
                        try:
                            (
                                final_content,
                                repair_llm_ms,
                            ) = await self._repair_empty_final_answer(
                                session_id=session_id,
                                iteration=iteration,
                                messages=messages,
                                on_token=_on_llm_token,
                            )
                        finally:
                            llm_total_ms += (
                                time.perf_counter() - repair_started
                            ) * 1000.0
                        if str(final_content or "").strip():
                            self._logger.info(
                                "annolid-bot empty final response repaired session=%s model=%s iteration=%d repair_llm_ms=%.1f",
                                session_id,
                                self.model,
                                iteration,
                                repair_llm_ms,
                            )
                        else:
                            final_content = self._build_tool_only_fallback_answer(
                                tool_runs=tool_runs
                            )
                            self._logger.warning(
                                "annolid-bot synthesized fallback final response session=%s model=%s iteration=%d tool_runs=%d",
                                session_id,
                                self.model,
                                iteration,
                                len(tool_runs),
                            )
                if memory_enabled and str(final_content).strip():
                    tools_used = self._extract_tools_used(tool_runs)
                    turn_messages: List[Dict[str, Any]] = []
                    if not user_persisted_early:
                        turn_messages.append(
                            {"role": "user", "content": user_message_text}
                        )
                    turn_messages.append(
                        {
                            "role": "assistant",
                            "content": str(final_content),
                            "tools_used": tools_used,
                        }
                    )
                    self._memory_store.append_history(
                        session_id,
                        turn_messages,
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
                _rollback_early_user_only_turn_if_needed(
                    final_text=str(final_content or ""),
                    tool_runs_for_turn=tool_runs,
                )
                result = AgentLoopResult(
                    content=final_content,
                    messages=messages,
                    iterations=iteration,
                    tool_runs=tuple(tool_runs),
                    media=tuple(messages_media),
                    stopped_reason=stopped_reason,
                )
                loop_result = result
                self._capture_anonymized_run_trace(
                    session_id=session_id,
                    channel=channel,
                    chat_id=chat_id,
                    user_message_text=user_message_text,
                    result=result,
                    turn_id=turn_id,
                )
                self._record_meta_learning_turn(
                    session_id=session_id,
                    user_message_text=user_message_text,
                    result=result,
                    tool_runs=tool_runs,
                    empty_final_repair_used=empty_final_repair_used,
                    llm_total_ms=llm_total_ms,
                    run_started=run_started,
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
                turn_messages: List[Dict[str, Any]] = []
                if not user_persisted_early:
                    turn_messages.append({"role": "user", "content": user_message_text})
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": str(final_content),
                        "tools_used": tools_used,
                    }
                )
                self._memory_store.append_history(
                    session_id,
                    turn_messages,
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
                clean_output = str(final_content or "")
                thinking_block = ""
                if "<think>" in clean_output and "</think>" in clean_output:
                    parts = clean_output.split("</think>", 1)
                    thinking_block = parts[0].replace("<think>", "").strip()
                    clean_output = parts[1].strip()
                self._notify_swarm_visualizer(
                    active_node_id,
                    "idle",
                    "Awaiting Tasks",
                    thinking=thinking_block[:100] + "..."
                    if len(thinking_block) > 100
                    else thinking_block,
                    output=clean_output[:150] + "..."
                    if len(clean_output) > 150
                    else clean_output,
                    context="final",
                )

            result = AgentLoopResult(
                content=final_content,
                messages=messages,
                iterations=last_iteration or self._max_iterations,
                tool_runs=tuple(tool_runs),
                media=tuple(messages_media),
                stopped_reason=stopped_reason,
            )
            loop_result = result
            self._record_session_event(
                session_id=session_id,
                direction="outbound",
                kind="assistant",
                turn_id=turn_id,
                payload={"text": str(final_content or "")},
            )
            _rollback_early_user_only_turn_if_needed(
                final_text=str(final_content or ""),
                tool_runs_for_turn=tool_runs,
            )
            self._capture_anonymized_run_trace(
                session_id=session_id,
                channel=channel,
                chat_id=chat_id,
                user_message_text=user_message_text,
                result=result,
                turn_id=turn_id,
            )
            self._record_meta_learning_turn(
                session_id=session_id,
                user_message_text=user_message_text,
                result=result,
                tool_runs=tool_runs,
                empty_final_repair_used=empty_final_repair_used,
                llm_total_ms=llm_total_ms,
                run_started=run_started,
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
                "annolid-bot profile summary session=%s model=%s total_ms=%.1f mcp_connect_ms=%.1f message_build_ms=%.1f llm_total_ms=%.1f tool_exec_total_ms=%.1f tool_calls=%d empty_repair_passes=%d",
                session_id,
                self.model,
                total_ms,
                mcp_connect_ms,
                message_build_ms,
                llm_total_ms,
                tool_exec_total_ms,
                tool_call_count,
                empty_final_repair_passes,
            )
            self._logger.info(
                "annolid-bot profile bottleneck session=%s model=%s bottleneck=%s bottleneck_ms=%.1f other_ms=%.1f",
                session_id,
                self.model,
                bottleneck_name,
                bottleneck_ms,
                other_ms,
            )
            self._record_turn_snapshot(
                session_id=session_id,
                turn_id=turn_id,
                result=loop_result,
                total_ms=total_ms,
                mcp_connect_ms=mcp_connect_ms,
                message_build_ms=message_build_ms,
                llm_total_ms=llm_total_ms,
                tool_exec_total_ms=tool_exec_total_ms,
                tool_call_count=tool_call_count,
                empty_final_repair_passes=empty_final_repair_passes,
                bottleneck_name=bottleneck_name,
                bottleneck_ms=bottleneck_ms,
                other_ms=other_ms,
                context_compaction_runs=context_compaction_runs,
                context_messages_trimmed=context_messages_trimmed,
                context_compaction_skipped_active_tasks=context_compaction_skipped_active_tasks,
            )
            await self._disconnect_mcp()

    def _record_meta_learning_turn(
        self,
        *,
        session_id: str,
        user_message_text: str,
        result: AgentLoopResult,
        tool_runs: Sequence[AgentToolRun],
        empty_final_repair_used: bool,
        llm_total_ms: float,
        run_started: float,
    ) -> None:
        learner = self._meta_learner
        if learner is None:
            return
        try:
            meta = learner.record_turn(
                session_id=session_id,
                user_text=user_message_text,
                assistant_text=str(result.content or ""),
                tool_runs=tool_runs,
                stopped_reason=result.stopped_reason,
                empty_repair_used=empty_final_repair_used,
                llm_total_ms=llm_total_ms,
                total_ms=(time.perf_counter() - run_started) * 1000.0,
            )
        except Exception as exc:
            self._logger.warning(
                "annolid-bot meta-learning record failed session=%s err=%s",
                session_id,
                exc,
            )
            return
        evolved = meta.get("evolved_skills", []) if isinstance(meta, Mapping) else []
        if evolved:
            self._logger.info(
                "annolid-bot meta-learning evolved_skills session=%s count=%d skills=%s",
                session_id,
                len(evolved),
                ",".join(str(s) for s in evolved),
            )

    @asynccontextmanager
    async def _mcp_run_guard(self):
        if not self._requires_exclusive_mcp_run():
            yield
            return
        async with self._exclusive_mcp_run_lock:
            yield

    def _unregister_mcp_tools(self) -> None:
        if not self._mcp_registered_tool_names:
            return
        for name in list(self._mcp_registered_tool_names):
            self._tools.unregister(name)
        self._mcp_registered_tool_names.clear()
        self._cached_tool_signature = None

    def _finalize_mcp_registration(self, before_names: set[str]) -> None:
        added = set(self._tools.tool_names) - before_names
        if self._tool_permission_context is not None:
            for name in list(added):
                if self._tool_permission_context.blocks(name):
                    self._tools.unregister(name)
                    added.discard(name)
        self._mcp_registered_tool_names.update(added)
        if added:
            self._cached_tool_signature = None

    async def _connect_mcp(self) -> None:
        """Acquire a shared MCP connection for the current run."""
        if not self._mcp_servers:
            return
        from .tools.mcp import connect_mcp_servers

        if not self._can_share_mcp_session():
            self._unregister_mcp_tools()
            before_names = set(self._tools.tool_names)
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            try:
                await connect_mcp_servers(
                    self._mcp_servers, self._tools, self._mcp_stack
                )
            except Exception:
                self._unregister_mcp_tools()
                for name in set(self._tools.tool_names) - before_names:
                    self._tools.unregister(name)
                try:
                    await self._mcp_stack.aclose()
                except Exception as close_exc:
                    self._logger.debug(
                        "MCP cleanup after connection failure failed: %s",
                        close_exc,
                    )
                self._mcp_stack = None
                raise
            self._finalize_mcp_registration(before_names)
            self._mcp_connected = True
            self._mcp_connecting = False
            self._mcp_ref_count = 1
            return

        async with self._mcp_lock:
            self._mcp_ref_count += 1
            if self._mcp_connected:
                return
            self._unregister_mcp_tools()
            before_names = set(self._tools.tool_names)
            self._mcp_connecting = True
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            try:
                await connect_mcp_servers(
                    self._mcp_servers, self._tools, self._mcp_stack
                )
            except Exception:
                self._unregister_mcp_tools()
                for name in set(self._tools.tool_names) - before_names:
                    self._tools.unregister(name)
                try:
                    await self._mcp_stack.aclose()
                except Exception as close_exc:
                    self._logger.debug(
                        "Shared MCP cleanup after connection failure failed: %s",
                        close_exc,
                    )
                self._mcp_stack = None
                self._mcp_ref_count = max(0, self._mcp_ref_count - 1)
                raise
            finally:
                self._mcp_connecting = False
            self._finalize_mcp_registration(before_names)
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

    def _notify_swarm_visualizer(
        self,
        node_id: str,
        status: str,
        label: str,
        *,
        context: str = "",
        **kwargs: Any,
    ) -> None:
        try:
            from annolid.gui.widgets.threejs_viewer_server import update_swarm_node

            update_swarm_node(node_id, status, label, **kwargs)
        except Exception as exc:
            self._logger.debug(
                "Swarm visualizer %s update failed: %s",
                context or "node",
                exc,
            )

    def _canonical_tool_name(self, name: str) -> str:
        resolver = getattr(self._tools, "canonical_name", None)
        if callable(resolver):
            resolved = str(resolver(name) or "").strip()
            if resolved:
                return resolved
        return str(name or "").strip()

    async def _run_single_tool_call(
        self,
        *,
        session_id: str,
        iteration: int,
        name: str,
        canonical_name: str,
        args: Dict[str, Any],
        executed_tools: set[str],
        explicit_high_risk_intent: bool,
    ) -> tuple[str, float, bool]:
        """Execute a single tool call after policy, high-risk, and approval checks.

        Returns `(result_text, elapsed_ms, actually_executed)`.
        """
        tool_started = time.perf_counter()
        block_reason = ""
        try:
            if self._tool_permission_context is not None and (
                self._tool_permission_context.blocks(canonical_name)
                or (
                    canonical_name != name
                    and self._tool_permission_context.blocks(name)
                )
            ):
                block_reason = f"Tool '{name}' is blocked by tool permission policy."
            if not block_reason:
                block_reason = self._high_risk_block_reason(
                    tool_name=canonical_name,
                    executed_tools=executed_tools,
                    explicit_high_risk_intent=explicit_high_risk_intent,
                )
            if not block_reason and self._tool_approval_callback is not None:
                risk_reason = classify_destructive_tool_call(canonical_name, args)
                if risk_reason:
                    approved = False
                    try:
                        decision = self._tool_approval_callback(
                            canonical_name, dict(args), risk_reason
                        )
                        if inspect.isawaitable(decision):
                            decision = await decision
                        approved = bool(decision)
                    except Exception as exc:
                        self._logger.warning(
                            "Destructive tool approval callback failed for %s; denying execution: %s",
                            canonical_name,
                            exc,
                        )
                        approved = False
                    with suppress(Exception):
                        emit_governance_event(
                            event_type="tool_approval",
                            action=canonical_name,
                            outcome="approved" if approved else "denied",
                            actor="user",
                            details={
                                "session_id": session_id,
                                "risk_reason": risk_reason,
                            },
                        )
                    if not approved:
                        block_reason = (
                            f"Tool '{name}' was denied by user confirmation "
                            f"({risk_reason})."
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
        return str(result), tool_elapsed_ms, not bool(block_reason)

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

        prepared: List[tuple[str, str, str, Dict[str, Any], bool]] = []
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            canonical_name = self._canonical_tool_name(name)
            args = self._normalize_args(call.get("arguments"))
            args = self._repair_tool_arguments_for_execution(
                name=canonical_name,
                args=args,
                messages=messages,
            )
            tool_obj = self._tools.get(canonical_name)
            is_parallel = is_concurrency_safe_tool_call(
                canonical_name=canonical_name,
                tool_obj=tool_obj,
                args=args,
            )
            prepared.append((call_id, name, canonical_name, args, is_parallel))

        batches = partition_tool_calls_for_execution([item[4] for item in prepared])
        sem = asyncio.Semaphore(max(1, int(self._MAX_CONCURRENT_READ_ONLY_TOOLS)))

        def _record_result(
            *,
            call_id: str,
            name: str,
            canonical_name: str,
            args: Dict[str, Any],
            result: str,
            tool_elapsed_ms: float,
            was_executed: bool,
        ) -> None:
            nonlocal cycle_exec_ms, cycle_call_count
            cycle_exec_ms += tool_elapsed_ms
            cycle_call_count += 1
            if was_executed:
                executed_tools.add(canonical_name)
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
                    result=result,
                )
            )
            try:
                media_path = extract_snapshot_media_path(
                    name=canonical_name,
                    result=result,
                    workspace=self._workspace,
                )
                if media_path:
                    media_list.append(media_path)
            except Exception as exc:
                self._logger.debug(
                    "Tool result media extraction failed for %s: %s",
                    name,
                    exc,
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": self._compact_tool_result_for_llm(
                        name=name, result=result
                    ),
                }
            )

        for is_parallel_batch, indices in batches:
            if not is_parallel_batch:
                idx = indices[0]
                call_id, name, canonical_name, args, _ = prepared[idx]
                result, elapsed_ms, was_executed = await self._run_single_tool_call(
                    session_id=session_id,
                    iteration=iteration,
                    name=name,
                    canonical_name=canonical_name,
                    args=args,
                    executed_tools=executed_tools,
                    explicit_high_risk_intent=explicit_high_risk_intent,
                )
                _record_result(
                    call_id=call_id,
                    name=name,
                    canonical_name=canonical_name,
                    args=args,
                    result=result,
                    tool_elapsed_ms=elapsed_ms,
                    was_executed=was_executed,
                )
                continue

            snapshot_executed = set(executed_tools)

            async def _run_bounded(idx: int) -> tuple[str, float, bool]:
                _, item_name, item_canonical, item_args, _ = prepared[idx]
                async with sem:
                    return await self._run_single_tool_call(
                        session_id=session_id,
                        iteration=iteration,
                        name=item_name,
                        canonical_name=item_canonical,
                        args=item_args,
                        executed_tools=snapshot_executed,
                        explicit_high_risk_intent=explicit_high_risk_intent,
                    )

            outcomes = await asyncio.gather(*(_run_bounded(i) for i in indices))
            for idx, (result, elapsed_ms, was_executed) in zip(indices, outcomes):
                call_id, name, canonical_name, args, _ = prepared[idx]
                _record_result(
                    call_id=call_id,
                    name=name,
                    canonical_name=canonical_name,
                    args=args,
                    result=result,
                    tool_elapsed_ms=elapsed_ms,
                    was_executed=was_executed,
                )

        self._append_post_tool_guidance(messages)
        return cycle_exec_ms, cycle_call_count

    def _compact_tool_result_for_llm(self, *, name: str, result: str) -> str:
        return compact_tool_result_for_llm(
            name=name,
            result=result,
            limit=int(self._MAX_TOOL_RESULT_CHARS_FOR_LLM),
        )

    @staticmethod
    def _build_tool_only_fallback_answer(*, tool_runs: Sequence[AgentToolRun]) -> str:
        return build_tool_only_fallback_answer(tool_runs)

    @staticmethod
    def _extract_tool_result_preview(raw_result: Any) -> str:
        return extract_tool_result_preview(raw_result)

    @staticmethod
    def _extract_tool_result_preview_from_payload(payload: Any) -> str:
        return extract_tool_result_preview_from_payload(payload)

    def _repair_tool_arguments_for_execution(
        self,
        *,
        name: str,
        args: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        repaired, repaired_prompt = repair_tool_arguments_for_execution(
            name=name,
            args=args,
            messages=messages,
        )
        if repaired_prompt:
            self._logger.info(
                "annolid-bot repaired missing web_search query from prompt"
            )
        return repaired

    @staticmethod
    def _tool_result_looks_user_ready(tool_name: str, preview: str) -> bool:
        return tool_result_looks_user_ready(tool_name, preview)

    @staticmethod
    def _can_skip_empty_final_repair(*, tool_runs: Sequence[AgentToolRun]) -> bool:
        return can_skip_empty_final_repair(tool_runs)

    async def _repair_empty_final_answer(
        self,
        *,
        session_id: str,
        iteration: int,
        messages: List[Dict[str, Any]],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, float]:
        repair_prompt = {
            "role": "user",
            "content": self._FINAL_ANSWER_REPAIR_PROMPT,
        }
        repair_messages = [*messages, repair_prompt]
        response, llm_elapsed_ms = await self._execute_llm_cycle(
            session_id=session_id,
            iteration=iteration,
            messages=repair_messages,
            tool_definitions=[],
            on_token=on_token,
        )
        assistant_text = str(response.get("content") or "")
        reasoning = str(response.get("reasoning_content") or "").strip()
        if reasoning and not assistant_text.startswith("<think>"):
            assistant_text = (
                f"<think>\n{reasoning}\n</think>\n\n{assistant_text}".strip()
            )
        final_content = self._strip_think(assistant_text)
        messages.append(repair_prompt)
        messages.append({"role": "assistant", "content": assistant_text})
        return final_content, llm_elapsed_ms

    @classmethod
    def _compact_context_messages(
        cls, messages: Sequence[Mapping[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        decision = cls._plan_context_compaction(messages)
        return list(decision.messages), int(decision.trimmed_count)

    @classmethod
    def _plan_context_compaction(
        cls, messages: Sequence[Mapping[str, Any]]
    ) -> ContextCompactionDecision:
        return plan_context_compaction(
            messages,
            max_messages=cls._CONTEXT_COMPACT_MAX_MESSAGES,
            max_non_system_messages=cls._CONTEXT_COMPACT_MAX_NON_SYSTEM_MESSAGES,
        )

    @classmethod
    def _has_unresolved_tool_tasks(cls, messages: Sequence[Mapping[str, Any]]) -> bool:
        del cls
        return has_unresolved_tool_tasks(messages)

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

        resolved_system_prompt = self._resolve_system_prompt_content(
            system_prompt=system_prompt,
            user_message_text=user_message_text,
            skill_names=skill_names,
            channel=channel,
            chat_id=chat_id,
        )
        if resolved_system_prompt:
            messages.append({"role": "system", "content": resolved_system_prompt})
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

    def _resolve_system_prompt_content(
        self,
        *,
        system_prompt: Optional[str],
        user_message_text: str,
        skill_names: Optional[List[str]],
        channel: Optional[str],
        chat_id: Optional[str],
    ) -> str:
        base_prompt = str(system_prompt or "").strip()
        if self._context_builder is None:
            return base_prompt
        contextual = self._context_builder.build_system_prompt(
            skill_names=skill_names,
            task_hint=user_message_text,
        )
        if channel and chat_id:
            contextual += (
                "\n\n## Current Session\n"
                f"Channel: {self._context_builder.redact_session_value(channel)}\n"
                f"Chat ID: {self._context_builder.redact_session_value(chat_id)}"
            )
        contextual = str(contextual or "").strip()
        if base_prompt and contextual:
            return f"{base_prompt}\n\n---\n\n{contextual}"
        return base_prompt or contextual

    def _prepare_tool_selection(
        self,
    ) -> tuple[
        List[Dict[str, Any]],
        tuple[tuple[str, str, str], ...],
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
        has_exec = bool(candidate.intersection(self._HIGH_RISK_TOOL_RUNTIME_EXEC))
        has_messaging = bool(candidate.intersection(self._HIGH_RISK_TOOL_MESSAGING))
        has_automation = bool(candidate.intersection(self._HIGH_RISK_TOOL_AUTOMATION))
        has_runtime_automation = bool(
            candidate.intersection(self._HIGH_RISK_TOOL_RUNTIME_AUTOMATION)
        )
        if has_exec and (has_messaging or has_automation):
            return (
                "Blocked by safety policy: runtime execution cannot be combined with "
                "messaging/automation tools without explicit high-risk intent."
            )
        if "read_file" in candidate and has_messaging and has_runtime_automation:
            return (
                "Blocked by safety policy: read_file + messaging + runtime automation "
                "requires explicit high-risk intent."
            )
        return ""

    async def _disconnect_mcp(self) -> None:
        if not self._can_share_mcp_session():
            stack = self._mcp_stack
            self._mcp_stack = None
            self._mcp_connected = False
            self._mcp_connecting = False
            self._mcp_ref_count = 0
            if stack is None:
                return
            try:
                await stack.aclose()
            except Exception as exc:
                self._logger.debug("MCP disconnect cleanup failed: %s", exc)
            return

        async with self._mcp_lock:
            if self._mcp_ref_count > 0:
                self._mcp_ref_count -= 1
            if self._mcp_ref_count > 0:
                return
            stack = self._mcp_stack
            self._mcp_stack = None
            self._mcp_connected = False
            self._mcp_connecting = False
        if stack is None:
            return
        try:
            await stack.aclose()
        except Exception as exc:
            self._logger.debug("Shared MCP disconnect cleanup failed: %s", exc)

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
        async with self._mcp_lock:
            self._unregister_mcp_tools()
            self._mcp_ref_count = 0
            stack = self._mcp_stack
            self._mcp_stack = None
            self._mcp_connected = False
            self._mcp_connecting = False
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as exc:
                self._logger.debug("Agent loop MCP close failed: %s", exc)

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
            # Preserve unconsolidated session history when consolidation fails
            # so user context is not silently lost.
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
        return history if outcome == "failed" else keep

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

    def _record_turn_snapshot(
        self,
        *,
        session_id: str,
        turn_id: str,
        result: Optional[AgentLoopResult],
        total_ms: float,
        mcp_connect_ms: float,
        message_build_ms: float,
        llm_total_ms: float,
        tool_exec_total_ms: float,
        tool_call_count: int,
        empty_final_repair_passes: int,
        bottleneck_name: str,
        bottleneck_ms: float,
        other_ms: float,
        context_compaction_runs: int,
        context_messages_trimmed: int,
        context_compaction_skipped_active_tasks: int = 0,
    ) -> None:
        recorder = getattr(self._memory_store, "record_turn_snapshot", None)
        if not callable(recorder):
            return
        tool_runs = list(result.tool_runs) if result is not None else []
        tool_names = sorted(
            {
                str(run.name).strip()
                for run in tool_runs
                if str(getattr(run, "name", "")).strip()
            }
        )
        content_preview = ""
        if result is not None:
            content_preview = str(result.content or "").strip().replace("\n", " ")
            if len(content_preview) > 280:
                content_preview = content_preview[:280].rstrip() + "..."
        usage_by_tool: Dict[str, int] = {}
        for run in tool_runs:
            name = str(getattr(run, "name", "")).strip()
            if not name:
                continue
            usage_by_tool[name] = usage_by_tool.get(name, 0) + 1
        payload = {
            "session_id": str(session_id or ""),
            "turn_id": str(turn_id or ""),
            "provider": str(self._provider or ""),
            "model": str(self.model or ""),
            "stopped_reason": (
                str(result.stopped_reason or "") if result is not None else "aborted"
            ),
            "iterations": int(result.iterations if result is not None else 0),
            "tool_call_count": int(max(0, int(tool_call_count))),
            "tool_run_count": len(tool_runs),
            "tools_invoked": tool_names,
            "tool_usage_counts": usage_by_tool,
            "durations_ms": {
                "total": float(total_ms),
                "mcp_connect": float(mcp_connect_ms),
                "message_build": float(message_build_ms),
                "llm_total": float(llm_total_ms),
                "tool_exec_total": float(tool_exec_total_ms),
                "other": float(other_ms),
                "bottleneck": {
                    "name": str(bottleneck_name or ""),
                    "duration_ms": float(bottleneck_ms),
                },
            },
            "empty_final_repair_passes": int(max(0, int(empty_final_repair_passes))),
            "context_compaction": {
                "runs": int(max(0, int(context_compaction_runs))),
                "messages_trimmed": int(max(0, int(context_messages_trimmed))),
                "skipped_active_tasks": int(
                    max(0, int(context_compaction_skipped_active_tasks))
                ),
            },
            "final_content_preview": content_preview,
        }
        try:
            recorder(session_id, payload=payload, limit=200)
        except Exception as exc:
            self._logger.warning(
                "annolid-bot turn snapshot record failed session=%s err=%s",
                session_id,
                exc,
            )
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

    def _requires_exclusive_mcp_run(self) -> bool:
        return bool(self._mcp_servers) and not self._can_share_mcp_session()

    def _can_share_mcp_session(self) -> bool:
        if not self._mcp_servers:
            return False
        for raw_cfg in self._mcp_servers.values():
            command = str(getattr(raw_cfg, "command", "") or "").strip()
            url = str(getattr(raw_cfg, "url", "") or "").strip()
            if not command and isinstance(raw_cfg, Mapping):
                command = str(raw_cfg.get("command", "") or "").strip()
                url = str(raw_cfg.get("url", "") or "").strip()
            if command:
                return False
            if not url:
                return False
        return True

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

        llm_call = self._llm_callable(
            [{"role": "system", "content": prompt}],
            save_memory_tool,
            self.model,
        )
        if self._llm_timeout_seconds is not None:
            resp = await asyncio.wait_for(llm_call, timeout=self._llm_timeout_seconds)
        else:
            resp = await llm_call
        for call in self._sanitize_tool_calls(
            self._extract_tool_calls(resp),
            allowed_tool_names=("save_memory",),
        ):
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
        except (json.JSONDecodeError, TypeError):
            parsed = None
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
        runtime_router = None
        has_subagent_tool = (
            isinstance(self._tools.get("spawn"), SpawnTool)
            or isinstance(
                self._tools.get("spawn_behavior_subagent"),
                SpawnBehaviorSubagentTool,
            )
            or isinstance(self._tools.get("list_tasks"), ListTasksTool)
            or isinstance(self._tools.get("cancel_task"), CancelTaskTool)
        )
        if self._subagent_manager is None and has_subagent_tool:
            self._subagent_manager = self._create_default_subagent_manager()
        if self._subagent_manager is not None and has_subagent_tool:
            runtime_router = self._create_default_runtime_router()

        spawn_tool = self._tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool) and runtime_router is not None:
            spawn_tool.set_spawn_callback(runtime_router.spawn)
        spawn_behavior_tool = self._tools.get("spawn_behavior_subagent")
        if (
            isinstance(spawn_behavior_tool, SpawnBehaviorSubagentTool)
            and runtime_router is not None
        ):
            spawn_behavior_tool.set_spawn_callback(runtime_router.spawn)

        list_tasks_tool = self._tools.get("list_tasks")
        if isinstance(list_tasks_tool, ListTasksTool) and runtime_router is not None:
            list_tasks_tool._list_tasks_callback = runtime_router.list_tasks

        cancel_task_tool = self._tools.get("cancel_task")
        if isinstance(cancel_task_tool, CancelTaskTool) and runtime_router is not None:
            cancel_task_tool._cancel_callback = runtime_router.cancel

        swarm_tool = self._tools.get("run_swarm")
        if isinstance(swarm_tool, SwarmTool):
            swarm_tool.set_swarm_callback(self._create_swarm_manager_and_run)

        # Inject loop factory to any tools that support it (like custom Swarm constructors)
        loop_factory = self._build_subagent_loop_factory()
        if hasattr(self._tools, "values"):
            tool_items = self._tools.values()
        elif hasattr(self._tools, "_tools"):
            tool_items = self._tools._tools.values()
        else:
            tool_items = []

        for t in tool_items:
            if hasattr(t, "set_loop_factory"):
                t.set_loop_factory(loop_factory)

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
        spawn_behavior_tool = self._tools.get("spawn_behavior_subagent")
        if isinstance(spawn_behavior_tool, SpawnBehaviorSubagentTool):
            spawn_behavior_tool.set_context(resolved_channel, resolved_chat_id)

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
                if self._tool_permission_context is not None:
                    for tool_name in list(tools.tool_names):
                        if self._tool_permission_context.blocks(tool_name):
                            tools.unregister(tool_name)
                return AgentLoop(
                    tools=tools,
                    llm_callable=self._llm_callable,
                    model=self.model,
                    max_iterations=min(self._max_iterations, 10),
                    workspace=self._workspace,
                    allowed_read_roots=self._allowed_read_roots,
                    mcp_servers=self._mcp_servers,
                    llm_timeout_seconds=self._llm_timeout_seconds,
                    tool_timeout_seconds=self._tool_timeout_seconds,
                    strict_runtime_tool_guard=self._strict_runtime_tool_guard,
                    tool_permission_context=self._tool_permission_context,
                )

            return _create()

        workspace_path = Path(self._workspace) if self._workspace else None
        return SubagentManager(loop_factory=_loop_factory, workspace=workspace_path)

    def _create_default_runtime_router(self) -> Optional[Any]:
        if self._subagent_manager is None:
            return None
        try:
            from .acp import get_acp_runtime_manager
            from .subagent import RuntimeSessionRouter
        except Exception:
            return None
        workspace_path = Path(self._workspace) if self._workspace else None
        return RuntimeSessionRouter(
            subagent_manager=self._subagent_manager,
            acp_manager=get_acp_runtime_manager(),
            workspace=workspace_path,
        )

    def _build_subagent_loop_factory(self) -> Callable[[], "AgentLoop"]:
        def _loop_factory() -> "AgentLoop":
            # Give each swarm turn enough room to think and use tools so we
            # don't force the outer swarm to burn extra turns on partial work.
            return AgentLoop(
                tools=self._tools,
                llm_callable=self._llm_callable,
                model=self.model,
                max_iterations=min(self._max_iterations, 5),
                workspace=self._workspace,
                allowed_read_roots=self._allowed_read_roots,
                mcp_servers=self._mcp_servers,
                llm_timeout_seconds=self._llm_timeout_seconds,
                tool_timeout_seconds=self._tool_timeout_seconds,
                strict_runtime_tool_guard=self._strict_runtime_tool_guard,
                tool_permission_context=self._tool_permission_context,
                tool_approval_callback=self._tool_approval_callback,
                max_selected_tools=self._max_selected_tools,
            )

        return _loop_factory

    async def _create_swarm_manager_and_run(
        self, task: str, max_turns: int = 8, agents: list[str] | None = None
    ) -> str:
        try:
            from .swarm import SwarmAgent, SwarmManager
        except Exception as e:
            return f"Cannot initialize swarm: {e}"

        _loop_factory = self._build_subagent_loop_factory()

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

        resolved_turns = resolve_swarm_turn_budget(
            task,
            max_turns,
            agent_count=len(roles),
        )
        return await manager.run_swarm(task, max_turns=resolved_turns)

    _TOOL_SELECTOR_STOPWORDS = set(TOOL_SELECTOR_STOPWORDS)
    _DEFAULT_TOOL_SELECTION_MAX = DEFAULT_TOOL_SELECTION_MAX
    _WEB_INTENT_TOKENS = WEB_INTENT_TOKENS
    _VCS_INTENT_TOKENS = VCS_INTENT_TOKENS
    _VCS_TOOL_NAMES = VCS_TOOL_NAMES
    _ANALYTICS_INTENT_TOKENS = ANALYTICS_INTENT_TOKENS
    _ANALYTICS_TOOL_NAMES = ANALYTICS_TOOL_NAMES
    _DEFAULT_TOOL_PRIORITY = DEFAULT_TOOL_PRIORITY
    _REMEMBER_NOTE_RE = re.compile(
        r"^\s*(?:please\s+)?remember(?:\s+that)?\s+(.+?)\s*$",
        flags=re.IGNORECASE,
    )
    _POST_TOOL_SYSTEM_GUIDANCE = (
        "Use the tool results to decide the next best action. "
        "Either call another tool with concrete arguments or provide a concise "
        "final answer. For live web search requests, prefer native `web_search` "
        "or `web_fetch` first, and use MCP browser workflow as fallback: "
        "navigate to a search engine, type the query, snapshot/parse the results, "
        "then continue. For GUI web tasks, prefer `gui_web_run_steps`; "
        "for visual description of the current web view, prefer `gui_web_describe_view`. "
        "before claiming browsing limits. Do not reveal private chain-of-thought."
    )
    _BROWSER_WORKFLOW_TOOL_NAMES = BROWSER_WORKFLOW_TOOL_NAMES
    _TOKENIZE_RE = re.compile(r"[a-zA-Z0-9_]+")

    @classmethod
    def _tokenize_text(cls, text: str) -> List[str]:
        del cls
        return tokenize_text(text)

    @classmethod
    def _build_tool_tokens(cls, schema: Mapping[str, Any]) -> set[str]:
        del cls
        return build_tool_tokens(schema)

    @classmethod
    def _score_tool_schema(
        cls,
        tool_entry: _ToolSchemaIndex,
        query_tokens: Sequence[str],
    ) -> int:
        del cls
        return score_tool_schema(tool_entry, query_tokens)

    @classmethod
    def _compile_tool_index(
        cls,
        all_tool_definitions: Sequence[Mapping[str, Any]],
    ) -> List[_ToolSchemaIndex]:
        del cls
        return compile_tool_index(all_tool_definitions)

    @staticmethod
    def _tool_signature(
        all_tool_definitions: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[str, str, str], ...]:
        return tool_signature(all_tool_definitions)

    def _get_cached_tool_index(
        self,
        all_tool_definitions: Sequence[Mapping[str, Any]],
        signature: tuple[tuple[str, str, str], ...],
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
        signature: tuple[tuple[str, str, str], ...],
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
        registered_names = (
            set(self._tools.tool_names)
            if hasattr(self._tools, "tool_names")
            and isinstance(self._tools.tool_names, (list, tuple, set))
            else None
        )
        return select_relevant_tool_definitions(
            all_tool_definitions=all_tool_definitions,
            tool_index=tool_index,
            default_tool_definitions=default_tool_definitions,
            user_message_text=user_message_text,
            messages=messages,
            registered_tool_names=registered_names,
            browser_first_for_web=self._browser_first_for_web,
            max_selected_tools=self._max_selected_tools,
        )

    @classmethod
    def _select_default_tool_definitions(
        cls,
        tools: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        return select_default_tool_definitions(
            tools,
            max_tools=cls._DEFAULT_TOOL_SELECTION_MAX,
        )

    def _sanitize_tool_calls(
        self,
        tool_calls: Sequence[Mapping[str, Any]],
        *,
        allowed_tool_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        names = set(self._tools.tool_names)
        for name in allowed_tool_names or ():
            clean = sanitize_tool_name(name)
            if clean:
                names.add(clean)
        return sanitize_tool_call_requests(
            tool_calls,
            allowed_tool_names=sorted(names),
        )

    def _is_registered_tool_name(self, name: str) -> bool:
        tool_name = sanitize_tool_name(name)
        return bool(tool_name) and self._tools.has(tool_name)

    @classmethod
    def _sanitize_tool_call_id(cls, raw_id: Any, *, default_index: int) -> str:
        return sanitize_tool_call_id(raw_id, default_index=default_index)

    @classmethod
    def _sanitize_tool_call_id_segment(cls, value: Any) -> str:
        return sanitize_tool_call_id_segment(value)

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
            legacy = response.get("function_call")
            if isinstance(legacy, Mapping):
                legacy_name = str(legacy.get("name") or "").strip()
                if legacy_name:
                    normalized.append(
                        {
                            "id": "call_0",
                            "name": legacy_name,
                            "arguments": legacy.get("arguments", {}),
                        }
                    )

        if not normalized:
            content = str(response.get("content") or "")
            normalized.extend(extract_text_tool_calls(content))
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
    def _extract_think(text: str | None) -> str:
        """Return concatenated <think>...</think> content when present."""
        raw = str(text or "")
        if not raw:
            return ""
        blocks = re.findall(r"<think>([\s\S]*?)</think>", raw)
        if not blocks:
            return ""
        return "\n".join(str(block or "").strip() for block in blocks if block).strip()

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
        return normalize_tool_arguments(raw_args)

    def _build_default_llm_callable(
        self,
        *,
        profile: Optional[str],
        provider: Optional[str],
        model: Optional[str],
    ) -> tuple[LLMCallable, str, Any]:
        try:
            cfg = resolve_llm_config(
                profile=profile,
                provider=provider,
                model=model,
                persist=False,
            )
            provider_name = str(cfg.provider or "").strip().lower()
            model_name = str(cfg.model)
            resolved_kind = provider_kind(load_llm_settings(), provider_name)
            if provider_name == "openai_codex" or resolved_kind == "openai_codex":
                resolved = resolve_openai_codex(cfg)
                model_name = resolved.model
                provider_impl = OpenAICodexProvider(resolved=resolved)
            elif provider_name == "codex_cli" or resolved_kind == "codex_cli":
                resolved = resolve_codex_cli(cfg)
                model_name = resolved.model
                provider_impl = CodexCLIProvider(resolved=resolved)
            elif provider_name == "anthropic" or resolved_kind == "anthropic":
                api_key = str(cfg.params.get("api_key") or "").strip()
                if not api_key:
                    raise ValueError(
                        "Anthropic provider requires API key for tool-calling chat."
                    )
                base_url = str(cfg.params.get("base_url") or "").strip()
                if base_url.rstrip("/") == "https://api.anthropic.com":
                    base_url = ""
                provider_impl = UnifiedLLMProvider(
                    provider_name="anthropic",
                    api_key=api_key,
                    api_base=base_url or None,
                    default_model=model_name,
                    extra_headers=cfg.params.get("extra_headers"),
                )
            else:
                try:
                    resolved = resolve_openai_compat(cfg)
                    model_name = resolved.model
                    provider_impl = OpenAICompatProvider(resolved=resolved)
                except ValueError:
                    provider_impl = UnifiedLLMProvider(
                        provider_name=provider_name or None,
                        api_key=cfg.params.get("api_key"),
                        api_base=cfg.params.get("base_url") or cfg.params.get("host"),
                        default_model=model_name,
                        extra_headers=cfg.params.get("extra_headers"),
                    )
        except (ValueError, KeyError, ImportError) as exc:
            error_text = str(exc)
            self._logger.warning(
                "Failed to resolve LLM configuration for %s: %s. "
                "The agent loop will start but LLM requests will fail until configured.",
                provider or "default",
                error_text,
            )
            model_name = model or "unconfigured"
            # Return a placeholder provider_impl that doesn't crash on chat()
            provider_impl = UnifiedLLMProvider(
                provider_name="placeholder",
                default_model=model_name,
            )

            async def _error_call(
                messages: Sequence[Mapping[str, Any]],
                tools: Sequence[Mapping[str, Any]],
                model_id: str,
                on_token: Optional[Callable[[str], None]] = None,
            ) -> Mapping[str, Any]:
                error_msg = (
                    f"LLM is not configured correctly: {error_text}. "
                    "Please check your LLM settings in the preferences."
                )
                if on_token:
                    on_token(error_msg)
                return {
                    "content": error_msg,
                    "tool_calls": [],
                    "finish_reason": "error",
                    "usage": {},
                    "reasoning_content": None,
                }

            return _error_call, model_name, provider_impl

        async def _call(
            messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]],
            model_id: str,
            on_token: Optional[Callable[[str], None]] = None,
        ) -> Mapping[str, Any]:
            resp = await provider_impl.chat_with_retry(
                messages=list(messages),
                tools=list(tools) if tools else None,
                model=model_id,
                temperature=self._default_temperature,
                on_token=on_token,
            )
            raise_for_error_response(
                resp,
                provider=provider_name,
                model=model_id,
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
