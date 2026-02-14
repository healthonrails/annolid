from __future__ import annotations

import json
import logging
import re
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
from .memory import AgentMemoryStore
from .providers import LiteLLMProvider, OpenAICompatProvider, resolve_openai_compat
from .tools import FunctionToolRegistry
from .tools.function_builtin import CronTool, MessageTool, SpawnTool

if TYPE_CHECKING:  # pragma: no cover
    from .subagent import SubagentManager

LLMCallable = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], str],
    Awaitable[Mapping[str, Any]],
]


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


class AgentLoop:
    """OpenAI-compatible async tool loop inspired by nanobot/agent/loop.py.

    This loop is stateless by default: pass existing history in `history`.
    """

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
        interleave_post_tool_guidance: bool = True,
        logger: Optional[logging.Logger] = None,
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
        self._logger = logger or logging.getLogger("annolid.agent.loop")
        self._memory_config = memory_config or AgentMemoryConfig(
            max_history_messages=runtime_cfg.max_history_messages,
            memory_window=runtime_cfg.memory_window,
        )
        self._memory_store = memory_store or InMemorySessionStore()
        self._workspace = workspace
        self._allowed_read_roots = tuple(str(p) for p in (allowed_read_roots or ()))
        self._context_builder = context_builder
        self._subagent_manager = subagent_manager
        self._interleave_post_tool_guidance = bool(interleave_post_tool_guidance)
        self._provider_impl: Optional[Any] = None

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
    ) -> AgentLoopResult:
        memory_enabled = (
            self._memory_config.enabled if use_memory is None else bool(use_memory)
        )
        self._set_tool_context(channel=channel, chat_id=chat_id)
        messages: List[Dict[str, Any]] = []
        user_message_text = str(user_message)
        self._persist_long_term_memory_note_from_user_text(user_message_text)
        memory_history: List[Dict[str, Any]] = []
        memory_facts: Dict[str, str] = {}
        if memory_enabled:
            memory_history = self._memory_store.get_history(session_id)
            memory_facts = self._memory_store.get_facts(session_id)
            if len(memory_history) > max(2, int(self._memory_config.memory_window)):
                memory_history = await self._consolidate_memory(
                    session_id=session_id,
                    history=memory_history,
                )

        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        elif self._context_builder is not None:
            contextual = self._context_builder.build_system_prompt(
                skill_names=skill_names
            )
            if channel and chat_id:
                contextual += (
                    f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
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

        tool_runs: List[AgentToolRun] = []
        final_content = ""
        stopped_reason = "done"
        all_tool_definitions = self._tools.get_definitions()
        tool_signature = self._tool_signature(all_tool_definitions)
        tool_index = self._get_cached_tool_index(all_tool_definitions, tool_signature)
        default_tools = self._get_cached_default_tools(tool_index, tool_signature)
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
            response = await self._llm_callable(
                messages,
                tool_definitions,
                self.model,
            )
            assistant_text = str(response.get("content") or "")
            tool_calls = self._sanitize_tool_calls(self._extract_tool_calls(response))

            if tool_calls:
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

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "tool_calls": [
                            self._to_openai_tool_call(tc) for tc in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    call_id = str(call.get("id") or "")
                    name = str(call.get("name") or "")
                    raw_args = call.get("arguments")
                    args = self._normalize_args(raw_args)
                    result = await self._tools.execute(name, args)
                    tool_runs.append(
                        AgentToolRun(
                            call_id=call_id,
                            name=name,
                            arguments=dict(args),
                            result=str(result),
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": str(result),
                        }
                    )
                self._append_post_tool_guidance(messages)
                continue

            repeated_tool_cycles = 0
            last_tool_cycle_signature = None

            final_content = assistant_text
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
            return AgentLoopResult(
                content=final_content,
                messages=messages,
                iterations=iteration,
                tool_runs=tuple(tool_runs),
                stopped_reason=stopped_reason,
            )

        if stopped_reason == "done":
            stopped_reason = "max_iterations"
            if not final_content:
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
        return AgentLoopResult(
            content=final_content,
            messages=messages,
            iterations=last_iteration or self._max_iterations,
            tool_runs=tuple(tool_runs),
            stopped_reason=stopped_reason,
        )

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
        if provider is None:
            return
        close_fn = getattr(provider, "close", None)
        if callable(close_fn):
            await close_fn()

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
        window = max(4, int(self._memory_config.memory_window))
        if len(history) <= window:
            return [dict(m) for m in history]
        keep_count = min(10, max(2, window // 2))
        archive = [dict(m) for m in history[:-keep_count]]
        keep = [dict(m) for m in history[-keep_count:]]
        if not archive:
            return keep

        self._replace_session_history(session_id, keep)
        if not self._workspace:
            return keep

        try:
            memory = AgentMemoryStore(Path(self._workspace))
            old_long_term = memory.read_long_term()
            transcript = self._format_consolidation_transcript(archive)
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
        except Exception as exc:
            self._logger.warning(
                "memory consolidation failed for session=%s: %s", session_id, exc
            )
        return keep

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
        resp = await self._llm_callable(
            [{"role": "system", "content": prompt}],
            [],
            self.model,
        )
        content = str(resp.get("content") or "").strip()
        parsed = self._try_parse_json_object(content)
        if parsed is None:
            return {
                "history_entry": self._fallback_history_entry(transcript),
                "memory_update": current_memory,
            }
        return parsed

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
        if self._subagent_manager is None and isinstance(
            self._tools.get("spawn"), SpawnTool
        ):
            self._subagent_manager = self._create_default_subagent_manager()
        spawn_tool = self._tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool) and self._subagent_manager is not None:
            spawn_tool.set_spawn_callback(self._subagent_manager.spawn)

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

    def _create_default_subagent_manager(self) -> Optional["SubagentManager"]:
        try:
            from .subagent import SubagentManager, build_subagent_tools_registry
        except Exception:
            return None

        def _loop_factory() -> "AgentLoop":
            tools = build_subagent_tools_registry(
                Path(self._workspace) if self._workspace else None,
                allowed_read_roots=self._allowed_read_roots,
            )
            return AgentLoop(
                tools=tools,
                llm_callable=self._llm_callable,
                model=self.model,
                max_iterations=min(self._max_iterations, 10),
                workspace=self._workspace,
                allowed_read_roots=self._allowed_read_roots,
            )

        workspace_path = Path(self._workspace) if self._workspace else None
        return SubagentManager(loop_factory=_loop_factory, workspace=workspace_path)

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
    _DEFAULT_TOOL_PRIORITY = (
        "gui_start_realtime_stream",
        "gui_stop_realtime_stream",
        "gui_label_behavior_segments",
        "gui_segment_track_video",
        "gui_open_video",
        "gui_open_url",
        "gui_open_in_browser",
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
        "final answer. Do not reveal private chain-of-thought."
    )

    @classmethod
    def _tokenize_text(cls, text: str) -> List[str]:
        raw = re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())
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
        for entry in entries:
            score = self._score_tool_schema(entry, query_tokens)
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
        ) -> Mapping[str, Any]:
            resp = await provider_impl.chat(
                messages=list(messages),
                tools=list(tools) if tools else None,
                model=model_id,
                temperature=self._default_temperature,
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
