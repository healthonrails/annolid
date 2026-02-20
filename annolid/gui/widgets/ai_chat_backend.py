from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import logging
import time
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from qtpy import QtCore
from qtpy.QtCore import QRunnable

from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.config import load_config
from annolid.core.agent.memory import AgentMemoryStore
from annolid.core.agent.providers import (
    ollama_mark_plain_mode,
    ollama_plain_mode_decrement,
    ollama_plain_mode_remaining,
    recover_with_plain_ollama_reply,
)
from annolid.core.agent.providers.background_chat import (
    OLLAMA_PLAIN_MODE_COOLDOWN_TURNS as _PROVIDER_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS,
)
from annolid.core.agent.providers.background_chat import (
    _OLLAMA_FORCE_PLAIN_CACHE as _PROVIDER_OLLAMA_FORCE_PLAIN_CACHE,
)
from annolid.core.agent.providers.background_chat import (
    _OLLAMA_TOOL_SUPPORT_CACHE as _PROVIDER_OLLAMA_TOOL_SUPPORT_CACHE,
)
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.tools import (
    FunctionToolRegistry,
)
from annolid.core.agent.tools.policy import resolve_allowed_tools
from annolid.core.agent.tools.clawhub import (
    clawhub_install_skill,
    clawhub_search_skills,
)
from annolid.core.agent.tools.pdf import DownloadPdfTool
from annolid.core.agent.tools.filesystem import RenameFileTool
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.core.agent.gui_backend.commands import (
    looks_like_local_access_refusal,
    parse_direct_gui_command,
    prompt_may_need_tools,
)
from annolid.core.agent.gui_backend.context_blocks import (
    build_live_pdf_context_prompt_block,
    build_live_web_context_prompt_block,
)
from annolid.core.agent.gui_backend.context_setup import (
    load_execution_prerequisites as load_gui_execution_prerequisites,
    prepare_context_tools as prepare_gui_context_tools,
)
from annolid.core.agent.gui_backend.fallbacks import (
    candidate_web_urls_for_prompt,
    extract_page_text_from_web_steps,
    try_browser_search_fallback,
    try_open_page_content_fallback,
    try_open_pdf_content_fallback,
    try_web_fetch_fallback,
)
from annolid.core.agent.gui_backend.heuristics import (
    build_extractive_summary,
    contains_hint,
    extract_web_urls,
    looks_like_knowledge_gap_response,
    looks_like_open_pdf_suggestion,
    looks_like_open_url_suggestion,
    looks_like_url_request,
    looks_like_web_access_refusal,
    prompt_may_need_mcp,
    should_attach_live_pdf_context,
    should_attach_live_web_context,
    topic_tokens,
)
from annolid.core.agent.gui_backend.paths import (
    build_pdf_search_roots,
    build_workspace_roots,
    extract_pdf_path_candidates,
    extract_video_path_candidates,
    find_video_by_basename_in_roots,
    list_available_pdfs_in_roots,
    resolve_pdf_path_for_roots,
    resolve_video_path_for_roots,
)
from annolid.core.agent.gui_backend.router import execute_direct_gui_command
from annolid.core.agent.gui_backend.runtime_flow import (
    emit_agent_loop_result,
    maybe_handle_ollama_plain_mode,
)
from annolid.core.agent.gui_backend.tool_registration import register_chat_gui_tools
from annolid.core.agent.gui_backend.tool_handlers_web_pdf import (
    pdf_find_sections as gui_pdf_find_sections,
    pdf_get_state as gui_pdf_get_state,
    pdf_get_text as gui_pdf_get_text,
    web_click as gui_web_click,
    web_find_forms as gui_web_find_forms,
    web_get_dom_text as gui_web_get_dom_text,
    web_get_state as gui_web_get_state,
    web_run_steps as gui_web_run_steps,
    web_scroll as gui_web_scroll,
    web_type as gui_web_type,
)
from annolid.core.agent.gui_backend.tool_handlers_openers import (
    extract_first_web_url as gui_extract_first_web_url,
    open_in_browser_tool as gui_open_in_browser_tool,
    open_pdf_tool as gui_open_pdf_tool,
    open_url_tool as gui_open_url_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_video import (
    open_video_tool as gui_open_video_tool,
    resolve_video_path_for_gui_tool as gui_resolve_video_path_for_gui_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_video_workflow import (
    label_behavior_segments_tool as gui_label_behavior_segments_tool,
    segment_track_video_tool as gui_segment_track_video_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_chat_controls import (
    run_ai_text_segmentation_tool as gui_run_ai_text_segmentation_tool,
    select_annotation_model_tool as gui_select_annotation_model_tool,
    send_chat_prompt_tool as gui_send_chat_prompt_tool,
    set_ai_text_prompt_tool as gui_set_ai_text_prompt_tool,
    set_chat_model_tool as gui_set_chat_model_tool,
    set_chat_prompt_tool as gui_set_chat_prompt_tool,
    set_frame_tool as gui_set_frame_tool,
    track_next_frames_tool as gui_track_next_frames_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_arxiv import (
    arxiv_search_tool as gui_arxiv_search_tool,
    list_local_pdfs as gui_list_local_pdfs,
    safe_run_arxiv_search as gui_safe_run_arxiv_search,
)
from annolid.core.agent.gui_backend.tool_handlers_citations import (
    add_citation_raw_tool as gui_add_citation_raw_tool,
    citation_fields_from_pdf_state as gui_citation_fields_from_pdf_state,
    citation_fields_from_web_state as gui_citation_fields_from_web_state,
    extract_doi as gui_extract_doi,
    extract_year as gui_extract_year,
    list_citations_tool as gui_list_citations_tool,
    normalize_citation_key as gui_normalize_citation_key,
    resolve_bib_output_path as gui_resolve_bib_output_path,
    save_citation_tool as gui_save_citation_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_filesystem import (
    rename_file_tool as gui_rename_file_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_realtime import (
    start_realtime_stream_tool as gui_start_realtime_stream_tool,
    stop_realtime_stream_tool as gui_stop_realtime_stream_tool,
)
from annolid.core.agent.gui_backend.session_io import (
    emit_chunk as gui_emit_chunk,
    emit_final as gui_emit_final,
    emit_progress as gui_emit_progress,
    load_history_messages as gui_load_history_messages,
    persist_turn as gui_persist_turn,
)
from annolid.core.agent.gui_backend.provider_fallback import (
    format_provider_config_error as gui_format_provider_config_error,
    is_provider_config_error as gui_is_provider_config_error,
    is_provider_timeout_error as gui_is_provider_timeout_error,
    run_provider_fallback as gui_run_provider_fallback,
)
from annolid.core.agent.gui_backend.provider_runtime import (
    has_image_context as gui_has_image_context,
    run_fast_mode as gui_run_fast_mode,
    run_fast_provider_chat as gui_run_fast_provider_chat,
    run_gemini_provider_chat as gui_run_gemini_provider_chat,
    run_ollama_chat as gui_run_ollama_chat,
    run_openai_chat as gui_run_openai_chat,
)
from annolid.core.agent.gui_backend.provider_dependencies import (
    format_dependency_error as gui_format_dependency_error,
    provider_dependency_error as gui_provider_dependency_error,
)
from annolid.core.agent.gui_backend.ollama_adapter import (
    build_gui_ollama_llm_callable,
    collect_gui_ollama_stream,
    extract_gui_ollama_text,
    format_gui_tool_trace,
    normalize_gui_messages_for_ollama,
    parse_gui_ollama_tool_calls,
)
from annolid.core.agent.gui_backend.runtime_config import (
    agent_loop_llm_timeout_seconds as gui_agent_loop_llm_timeout_seconds,
    agent_loop_tool_timeout_seconds as gui_agent_loop_tool_timeout_seconds,
    browser_first_for_web as gui_browser_first_for_web,
    fallback_retry_timeout_seconds as gui_fallback_retry_timeout_seconds,
    fallback_timeout_retry_seconds as gui_fallback_timeout_retry_seconds,
    fast_mode_timeout_seconds as gui_fast_mode_timeout_seconds,
    ollama_agent_plain_timeout_seconds as gui_ollama_agent_plain_timeout_seconds,
    ollama_agent_tool_timeout_seconds as gui_ollama_agent_tool_timeout_seconds,
    ollama_plain_recovery_nudge_timeout_seconds as gui_ollama_plain_recovery_nudge_timeout_seconds,
    ollama_plain_recovery_timeout_seconds as gui_ollama_plain_recovery_timeout_seconds,
)
from annolid.core.agent.gui_backend.widget_bridge import (
    build_gui_context_payload as gui_build_gui_context_payload,
    get_widget_action_result as gui_get_widget_action_result,
    invoke_widget_json_slot as gui_invoke_widget_json_slot,
    invoke_widget_slot as gui_invoke_widget_slot,
)
from annolid.core.agent.gui_backend.response_finalize import (
    apply_direct_gui_fallback as gui_apply_direct_gui_fallback,
    apply_empty_ollama_recovery as gui_apply_empty_ollama_recovery,
    apply_pdf_response_fallback as gui_apply_pdf_response_fallback,
    apply_web_response_fallbacks as gui_apply_web_response_fallbacks,
    ensure_non_empty_final_text as gui_ensure_non_empty_final_text,
    should_apply_web_refusal_fallback as gui_should_apply_web_refusal_fallback,
)
from annolid.core.agent.gui_backend.telemetry import (
    log_agent_result as gui_log_agent_result,
    log_runtime_timeouts as gui_log_runtime_timeouts,
    wrap_tool_callback as gui_wrap_tool_callback,
)
from annolid.core.agent.gui_backend.direct_commands import (
    execute_direct_gui_command as gui_execute_direct_gui_command,
    run_awaitable_sync as gui_run_awaitable_sync,
)
from annolid.core.agent.gui_backend.prompt_builder import (
    PromptBuildInputs,
    build_compact_system_prompt as build_gui_compact_system_prompt,
)
from annolid.utils.llm_settings import resolve_agent_runtime_config
from annolid.utils.citations import (
    BibEntry,
    entry_to_dict,
    load_bibtex,
    merge_validated_fields,
    parse_bibtex,
    save_bibtex,
    search_entries,
    upsert_entry,
    validate_basic_citation_fields,
    validate_citation_metadata,
)
from annolid.utils.logger import logger

# Backward-compat alias used by tests that monkeypatch invokeMethod.
QMetaObject = QtCore.QMetaObject


_SESSION_STORE: Optional[PersistentSessionStore] = None
_GUI_ALWAYS_DISABLED_TOOLS = {"cron", "spawn", "message"}
_GUI_WEB_TOOLS = {"web_search", "web_fetch"}
# Backward-compat aliases retained for tests/internal callers that reference
# backend module globals directly.
_OLLAMA_TOOL_SUPPORT_CACHE = _PROVIDER_OLLAMA_TOOL_SUPPORT_CACHE
_OLLAMA_FORCE_PLAIN_CACHE = _PROVIDER_OLLAMA_FORCE_PLAIN_CACHE
_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS = _PROVIDER_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS
_PROMPT_FILE_CACHE_LOCK = Lock()
_PROMPT_FILE_CACHE: Dict[Tuple[str, int], Tuple[int, int, str]] = {}
_SKILL_DIR_CACHE_LOCK = Lock()
_SKILL_DIR_CACHE: Dict[str, Tuple[int, List[str]]] = {}


def _get_session_store() -> PersistentSessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = PersistentSessionStore(AgentSessionManager())
    return _SESSION_STORE


def _read_text_limited_cached(path: Path, max_chars: int) -> str:
    key = (str(path), int(max_chars))
    try:
        stat = path.stat()
    except Exception:
        return ""
    stamp = int(getattr(stat, "st_mtime_ns", 0))
    size = int(getattr(stat, "st_size", -1))
    with _PROMPT_FILE_CACHE_LOCK:
        cached = _PROMPT_FILE_CACHE.get(key)
    if cached is not None:
        cached_stamp, cached_size, cached_text = cached
        if cached_stamp == stamp and cached_size == size:
            return cached_text
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    value = str(text or "").strip()
    if len(value) > max_chars:
        value = value[:max_chars].rstrip() + "\n...[truncated]"
    with _PROMPT_FILE_CACHE_LOCK:
        _PROMPT_FILE_CACHE[key] = (stamp, size, value)
    return value


def _list_workspace_skill_names_cached(skills_dir: Path) -> List[str]:
    key = str(skills_dir)
    try:
        stamp = int(getattr(skills_dir.stat(), "st_mtime_ns", 0))
    except Exception:
        return []
    with _SKILL_DIR_CACHE_LOCK:
        cached = _SKILL_DIR_CACHE.get(key)
    if cached is not None:
        cached_stamp, cached_names = cached
        if cached_stamp == stamp:
            return list(cached_names)
    try:
        names = sorted(
            p.name
            for p in skills_dir.iterdir()
            if p.is_dir() and (p / "SKILL.md").exists()
        )
    except Exception:
        return []
    with _SKILL_DIR_CACHE_LOCK:
        _SKILL_DIR_CACHE[key] = (stamp, list(names))
    return names


def clear_chat_session(session_id: str) -> None:
    """Clear persisted chat history/facts for a specific GUI session."""
    _get_session_store().clear_session(str(session_id or "gui:annolid_bot:default"))


@dataclass(frozen=True)
class _AgentExecutionContext:
    workspace: Path
    allowed_read_roots: List[str]
    tools: FunctionToolRegistry
    system_prompt: str


class StreamingChatTask(QRunnable):
    """Stream a chat response from the selected provider back to a widget."""

    def __init__(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        widget=None,
        model: str = "llama3.2-vision:latest",
        provider: str = "ollama",
        settings: Optional[Dict[str, Any]] = None,
        session_id: str = "gui:annolid_bot:default",
        session_store: Optional[PersistentSessionStore] = None,
        show_tool_trace: bool = False,
        enable_web_tools: bool = True,
        chat_mode: str = "default",
    ):
        super().__init__()
        self.prompt = prompt
        self.image_path = image_path
        self.widget = widget
        self.model = model
        self.provider = provider
        self.settings = settings or {}
        self.session_id = str(session_id or "gui:annolid_bot:default")
        self.session_store = session_store or _get_session_store()
        self.show_tool_trace = bool(show_tool_trace)
        self.enable_web_tools = bool(enable_web_tools)
        self.chat_mode = str(chat_mode or "default").strip().lower() or "default"
        self.workspace = get_agent_workspace_path()
        self.workspace_memory = AgentMemoryStore(self.workspace)
        runtime_cfg = resolve_agent_runtime_config(profile="playground")
        self.max_history_messages = int(runtime_cfg.max_history_messages)
        self._last_progress_update: str = ""

    def run(self) -> None:
        """Execute the chat task flow."""
        logger.info(
            "annolid-bot turn start session=%s provider=%s model=%s prompt_chars=%d",
            self.session_id,
            self.provider,
            self.model,
            len(str(self.prompt or "")),
        )
        dep_error = self._provider_dependency_error()
        if dep_error:
            self._emit_progress("Provider dependency check failed")
            logger.warning(
                "annolid-bot dependency check failed session=%s provider=%s model=%s error=%s",
                self.session_id,
                self.provider,
                self.model,
                dep_error,
            )
            self._emit_final(dep_error, is_error=True)
            logger.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=dependency_missing",
                self.session_id,
                self.provider,
                self.model,
            )
            return
        try:
            if self._should_run_fast_mode():
                self._emit_progress(f"Starting fast mode ({self.chat_mode})")
                try:
                    self._run_fast_mode()
                except Exception as fast_exc:
                    if self._is_provider_config_error(fast_exc):
                        logger.warning(
                            "annolid-bot fast mode unavailable due provider config; falling back to agent loop session=%s provider=%s model=%s error=%s",
                            self.session_id,
                            self.provider,
                            self.model,
                            fast_exc,
                        )
                        self._emit_progress(
                            "Fast mode unavailable, using standard path"
                        )
                        self._run_agent_loop()
                        logger.info(
                            "annolid-bot turn stop session=%s provider=%s model=%s status=ok_fast_fallback",
                            self.session_id,
                            self.provider,
                            self.model,
                        )
                        return
                    raise
                logger.info(
                    "annolid-bot turn stop session=%s provider=%s model=%s status=ok_fast_mode",
                    self.session_id,
                    self.provider,
                    self.model,
                )
                return
            self._emit_progress("Starting agent loop")
            self._run_agent_loop()
            logger.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=ok",
                self.session_id,
                self.provider,
                self.model,
            )
        except Exception as exc:
            if self._is_provider_config_error(exc):
                message = self._format_provider_config_error(str(exc))
                logger.warning(
                    "annolid-bot provider config error session=%s provider=%s model=%s error=%s",
                    self.session_id,
                    self.provider,
                    self.model,
                    exc,
                )
                self._emit_final(message, is_error=True)
                logger.info(
                    "annolid-bot turn stop session=%s provider=%s model=%s status=config_error",
                    self.session_id,
                    self.provider,
                    self.model,
                )
                return
            if isinstance(exc, ImportError):
                message = self._format_dependency_error(str(exc))
                logger.warning(
                    "annolid-bot agent dependency missing session=%s provider=%s model=%s error=%s",
                    self.session_id,
                    self.provider,
                    self.model,
                    exc,
                )
                self._emit_final(message, is_error=True)
                logger.info(
                    "annolid-bot turn stop session=%s provider=%s model=%s status=dependency_missing",
                    self.session_id,
                    self.provider,
                    self.model,
                )
                return
            logger.warning(
                "annolid-bot agent loop failed; trying provider fallback session=%s provider=%s model=%s error=%s",
                self.session_id,
                self.provider,
                self.model,
                exc,
            )
            self._run_provider_fallback(exc)

    def _run_provider_fallback(self, original_error: Exception) -> None:
        gui_run_provider_fallback(
            original_error=original_error,
            settings=self.settings,
            provider=self.provider,
            model=self.model,
            session_id=self.session_id,
            fallback_timeout_retry_seconds=self._fallback_timeout_retry_seconds,
            fallback_retry_timeout_seconds=self._fallback_retry_timeout_seconds,
            run_ollama=self._run_ollama,
            run_openai=lambda provider_name, timeout_s, max_tokens: self._run_openai(
                provider_name=provider_name,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
            ),
            run_gemini=self._run_gemini,
            emit_progress=self._emit_progress,
            emit_final=lambda message, is_error: self._emit_final(
                message, is_error=is_error
            ),
            format_dependency_error=self._format_dependency_error,
            logger=logger,
        )

    @staticmethod
    def _is_provider_config_error(exc: Exception | str) -> bool:
        return gui_is_provider_config_error(exc)

    def _format_provider_config_error(self, raw_error: str) -> str:
        return gui_format_provider_config_error(raw_error, provider=self.provider)

    @staticmethod
    def _is_provider_timeout_error(exc: Exception | str) -> bool:
        return gui_is_provider_timeout_error(exc)

    def _has_image_context(self) -> bool:
        return gui_has_image_context(self.image_path)

    def _should_run_fast_mode(self) -> bool:
        if self.chat_mode == "vision_describe":
            return self._has_image_context()
        return False

    def _run_fast_mode(self) -> None:
        gui_run_fast_mode(
            chat_mode=self.chat_mode,
            run_fast_provider_chat=lambda include_image,
            include_history: self._run_fast_provider_chat(
                include_image=include_image,
                include_history=include_history,
            ),
        )

    def _run_fast_provider_chat(
        self, *, include_image: bool, include_history: bool
    ) -> None:
        gui_run_fast_provider_chat(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            provider=self.provider,
            settings=self.settings,
            include_image=include_image,
            include_history=include_history,
            load_history_messages=self._load_history_messages,
            fast_mode_timeout_seconds=self._fast_mode_timeout_seconds,
            emit_progress=self._emit_progress,
            emit_chunk=self._emit_chunk,
            emit_final=lambda message, is_error: self._emit_final(
                message, is_error=is_error
            ),
            persist_turn=lambda user_text, assistant_text: self._persist_turn(
                user_text, assistant_text
            ),
        )

    def _fast_mode_timeout_seconds(self) -> float:
        return gui_fast_mode_timeout_seconds(self.settings)

    def _agent_loop_llm_timeout_seconds(self, *, prompt_needs_tools: bool) -> float:
        return gui_agent_loop_llm_timeout_seconds(
            self.settings, prompt_needs_tools=prompt_needs_tools
        )

    def _ollama_agent_tool_timeout_seconds(self) -> float:
        return gui_ollama_agent_tool_timeout_seconds(self.settings)

    def _agent_loop_tool_timeout_seconds(self) -> float:
        return gui_agent_loop_tool_timeout_seconds(
            self.settings, provider=self.provider
        )

    def _browser_first_for_web(self) -> bool:
        return gui_browser_first_for_web(self.settings)

    def _ollama_agent_plain_timeout_seconds(self) -> float:
        return gui_ollama_agent_plain_timeout_seconds(self.settings)

    def _ollama_plain_recovery_timeout_seconds(self) -> float:
        return gui_ollama_plain_recovery_timeout_seconds(self.settings)

    def _ollama_plain_recovery_nudge_timeout_seconds(self) -> float:
        return gui_ollama_plain_recovery_nudge_timeout_seconds(self.settings)

    def _fallback_retry_timeout_seconds(self) -> float:
        return gui_fallback_retry_timeout_seconds(self.settings)

    def _fallback_timeout_retry_seconds(self) -> float:
        return gui_fallback_timeout_retry_seconds(
            self.settings,
            prompt_needs_tools=self._prompt_may_need_tools(self.prompt),
        )

    def _provider_dependency_error(self) -> Optional[str]:
        return gui_provider_dependency_error(
            settings=self.settings, provider=self.provider
        )

    def _format_dependency_error(self, raw_error: str) -> str:
        return gui_format_dependency_error(
            raw_error=raw_error,
            settings=self.settings,
            provider=self.provider,
        )

    def _emit_chunk(self, chunk: str) -> None:
        gui_emit_chunk(widget=self.widget, chunk=chunk)

    def _emit_progress(self, update: str) -> None:
        self._last_progress_update = gui_emit_progress(
            widget=self.widget,
            update=update,
            last_progress_update=self._last_progress_update,
        )

    def _emit_final(self, message: str, *, is_error: bool) -> None:
        gui_emit_final(
            widget=self.widget,
            message=message,
            is_error=is_error,
            emit_progress_cb=self._emit_progress,
        )

    def _load_history_messages(self) -> List[Dict[str, Any]]:
        return gui_load_history_messages(
            session_store=self.session_store,
            session_id=self.session_id,
            max_history_messages=self.max_history_messages,
        )

    def _persist_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        persist_session_history: bool = True,
    ) -> None:
        gui_persist_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            session_id=self.session_id,
            session_store=self.session_store,
            max_history_messages=self.max_history_messages,
            workspace_memory=self.workspace_memory,
            persist_session_history=persist_session_history,
        )

    def _run_agent_loop(self) -> None:
        asyncio.run(self._run_agent_loop_async())

    async def _run_agent_loop_async(self) -> bool:
        # Check for direct command match first (e.g. "open video path/to/file.mp4")
        if await self._try_execute_direct_gui_command():
            return True

        prompt_needs_tools = self._prompt_may_need_tools(self.prompt)
        context = await self._build_agent_execution_context(
            include_tools=prompt_needs_tools
        )

        if maybe_handle_ollama_plain_mode(
            provider=self.provider,
            model=self.model,
            prompt=self.prompt,
            show_tool_trace=self.show_tool_trace,
            prompt_may_need_tools=self._prompt_may_need_tools,
            plain_mode_remaining=ollama_plain_mode_remaining,
            plain_mode_decrement=ollama_plain_mode_decrement,
            recover_with_plain_reply=self._recover_with_plain_ollama_reply,
            persist_turn=lambda user, assistant: self._persist_turn(user, assistant),
            emit_final=lambda message, is_error: self._emit_final(
                message, is_error=is_error
            ),
            logger=logger,
        ):
            return True

        mcp_servers = self._resolve_mcp_servers(
            context=context, prompt_needs_tools=prompt_needs_tools
        )
        self._log_runtime_timeouts(prompt_needs_tools=prompt_needs_tools)
        loop = self._build_agent_loop_instance(
            context=context,
            prompt_needs_tools=prompt_needs_tools,
            mcp_servers=mcp_servers,
        )
        media = self._build_media_payload()

        result = await loop.run(
            self.prompt,
            session_id=self.session_id,
            channel="gui",
            chat_id="annolid_bot",
            media=media,
            system_prompt=context.system_prompt,
            on_progress=self._emit_progress,
        )
        self._emit_progress("Received model response")
        (
            text,
            used_recovery,
            used_direct_gui_fallback,
        ) = await self._finalize_agent_text_async(
            result,
            tools=context.tools,
        )
        self._log_agent_result(result, used_recovery, used_direct_gui_fallback)
        emit_agent_loop_result(
            prompt=self.prompt,
            text=text,
            persist_turn=lambda user, assistant: self._persist_turn(
                user, assistant, persist_session_history=False
            ),
            emit_final=lambda message, is_error: self._emit_final(
                message, is_error=is_error
            ),
        )
        return False

    def _resolve_mcp_servers(
        self,
        *,
        context: _AgentExecutionContext,
        prompt_needs_tools: bool,
    ) -> Dict[str, Any]:
        configured_mcp_servers = self.settings.get("tools", {}).get("mcp_servers", {})
        mcp_needed = prompt_needs_tools and self._prompt_may_need_mcp(self.prompt)
        if mcp_needed and len(context.tools) > 0:
            return configured_mcp_servers
        if configured_mcp_servers:
            logger.info(
                "annolid-bot skipping mcp connect for no-mcp-intent prompt session=%s model=%s",
                self.session_id,
                self.model,
            )
        return {}

    def _log_runtime_timeouts(self, *, prompt_needs_tools: bool) -> None:
        gui_log_runtime_timeouts(
            logger=logger,
            session_id=self.session_id,
            model=self.model,
            loop_llm_s=self._agent_loop_llm_timeout_seconds(
                prompt_needs_tools=prompt_needs_tools
            ),
            loop_tool_s=self._agent_loop_tool_timeout_seconds(),
            ollama_tool_s=self._ollama_agent_tool_timeout_seconds(),
            ollama_plain_s=self._ollama_agent_plain_timeout_seconds(),
            recover_s=self._ollama_plain_recovery_timeout_seconds(),
            recover_nudge_s=self._ollama_plain_recovery_nudge_timeout_seconds(),
        )

    def _build_agent_loop_instance(
        self,
        *,
        context: _AgentExecutionContext,
        prompt_needs_tools: bool,
        mcp_servers: Dict[str, Any],
    ) -> AgentLoop:
        return AgentLoop(
            tools=context.tools,
            llm_callable=self._resolve_loop_llm_callable(),
            provider=self.provider,
            model=self.model,
            profile="playground",
            memory_store=self.session_store,
            workspace=str(context.workspace),
            allowed_read_roots=context.allowed_read_roots,
            mcp_servers=mcp_servers,
            llm_timeout_seconds=self._agent_loop_llm_timeout_seconds(
                prompt_needs_tools=prompt_needs_tools
            ),
            tool_timeout_seconds=self._agent_loop_tool_timeout_seconds(),
            browser_first_for_web=self._browser_first_for_web(),
        )

    def _build_media_payload(self) -> Optional[List[str]]:
        if self.image_path and os.path.exists(self.image_path):
            return [self.image_path]
        return None

    async def _try_execute_direct_gui_command(self) -> bool:
        """Attempt to execute a direct GUI command if the prompt matches a pattern."""
        if not self.prompt:
            return False
        try:
            direct_command_text = await self._execute_direct_gui_command(self.prompt)
        except Exception as exc:
            logger.warning(
                "annolid-bot direct gui command failed session=%s model=%s error=%s",
                self.session_id,
                self.model,
                exc,
            )
            return False
        if not direct_command_text:
            return False
        self._emit_progress("Executed direct GUI command")
        logger.info(
            "annolid-bot direct gui command handled session=%s model=%s",
            self.session_id,
            self.model,
        )
        self._persist_turn(self.prompt, direct_command_text)
        self._emit_final(direct_command_text, is_error=False)
        return True

    async def _build_agent_execution_context(
        self, *, include_tools: bool = True
    ) -> _AgentExecutionContext:
        self._emit_progress("Loading tools and context")
        profile_t0 = time.perf_counter()
        workspace, agent_cfg, allowed_read_roots, t_after_workspace, t_after_config = (
            self._load_execution_prerequisites(profile_t0)
        )
        (
            tools,
            policy_profile,
            policy_source,
            t_after_register,
            t_after_policy,
            t_before_register,
            t_before_policy,
        ) = await self._prepare_context_tools(
            include_tools=include_tools,
            workspace=workspace,
            allowed_read_roots=allowed_read_roots,
            agent_cfg=agent_cfg,
        )
        system_prompt = self._build_compact_system_prompt(
            workspace,
            allowed_read_roots=allowed_read_roots,
            allow_web_tools=bool(include_tools and self.enable_web_tools),
            include_workspace_docs=bool(include_tools),
        )
        t_after_prompt = time.perf_counter()
        self._emit_progress("Prepared system prompt")
        logger.info(
            "annolid-bot agent config session=%s model=%s tools=%d read_roots=%d profile=%s policy_source=%s prompt_chars=%d",
            self.session_id,
            self.model,
            len(tools),
            len(allowed_read_roots),
            policy_profile,
            policy_source,
            len(system_prompt),
        )
        logger.info(
            "annolid-bot profile context session=%s model=%s workspace_ms=%.1f config_ms=%.1f register_tools_ms=%.1f policy_ms=%.1f prompt_ms=%.1f total_ms=%.1f",
            self.session_id,
            self.model,
            (t_after_workspace - profile_t0) * 1000.0,
            (t_after_config - t_after_workspace) * 1000.0,
            (t_after_register - t_before_register) * 1000.0,
            (t_after_policy - t_before_policy) * 1000.0,
            (t_after_prompt - t_after_policy) * 1000.0,
            (t_after_prompt - profile_t0) * 1000.0,
        )
        context_buckets = {
            "workspace": (t_after_workspace - profile_t0) * 1000.0,
            "config": (t_after_config - t_after_workspace) * 1000.0,
            "register_tools": (t_after_register - t_before_register) * 1000.0,
            "policy": (t_after_policy - t_before_policy) * 1000.0,
            "prompt_build": (t_after_prompt - t_after_policy) * 1000.0,
        }
        context_bottleneck_name, context_bottleneck_ms = max(
            context_buckets.items(), key=lambda item: item[1]
        )
        logger.info(
            "annolid-bot profile context-bottleneck session=%s model=%s bottleneck=%s bottleneck_ms=%.1f",
            self.session_id,
            self.model,
            context_bottleneck_name,
            context_bottleneck_ms,
        )
        return _AgentExecutionContext(
            workspace=workspace,
            allowed_read_roots=allowed_read_roots,
            tools=tools,
            system_prompt=system_prompt,
        )

    def _load_execution_prerequisites(
        self,
        profile_t0: float,
    ) -> Tuple[Path, Any, List[str], float, float]:
        del profile_t0
        prepared = load_gui_execution_prerequisites()
        return (
            prepared.workspace,
            prepared.agent_cfg,
            prepared.allowed_read_roots,
            prepared.t_after_workspace,
            prepared.t_after_config,
        )

    async def _prepare_context_tools(
        self,
        *,
        include_tools: bool,
        workspace: Path,
        allowed_read_roots: List[str],
        agent_cfg: Any,
    ) -> Tuple[FunctionToolRegistry, str, str, float, float, float, float]:
        prepared = await prepare_gui_context_tools(
            include_tools=include_tools,
            workspace=workspace,
            allowed_read_roots=allowed_read_roots,
            agent_cfg=agent_cfg,
            register_gui_tools=self._register_gui_tools,
            provider=self.provider,
            model=self.model,
            enable_web_tools=self.enable_web_tools,
            always_disabled_tools=_GUI_ALWAYS_DISABLED_TOOLS,
            web_tools=_GUI_WEB_TOOLS,
            resolve_policy=resolve_allowed_tools,
        )
        return (
            prepared.tools,
            prepared.policy_profile,
            prepared.policy_source,
            prepared.t_after_register,
            prepared.t_after_policy,
            prepared.t_before_register,
            prepared.t_before_policy,
        )

    def _register_gui_tools(self, tools: FunctionToolRegistry) -> None:
        register_chat_gui_tools(
            tools,
            context_callback=self._build_gui_context_payload,
            image_path_callback=self._get_shared_image_path,
            wrap_tool_callback=self._wrap_tool_callback,
            handlers={
                "open_video": self._tool_gui_open_video,
                "open_url": self._tool_gui_open_url,
                "open_in_browser": self._tool_gui_open_in_browser,
                "web_get_dom_text": self._tool_gui_web_get_dom_text,
                "web_click": self._tool_gui_web_click,
                "web_type": self._tool_gui_web_type,
                "web_scroll": self._tool_gui_web_scroll,
                "web_find_forms": self._tool_gui_web_find_forms,
                "web_run_steps": self._tool_gui_web_run_steps,
                "open_pdf": self._tool_gui_open_pdf,
                "pdf_get_state": self._tool_gui_pdf_get_state,
                "pdf_get_text": self._tool_gui_pdf_get_text,
                "pdf_find_sections": self._tool_gui_pdf_find_sections,
                "set_frame": self._tool_gui_set_frame,
                "set_prompt": self._tool_gui_set_chat_prompt,
                "send_prompt": self._tool_gui_send_chat_prompt,
                "set_chat_model": self._tool_gui_set_chat_model,
                "select_annotation_model": self._tool_gui_select_annotation_model,
                "track_next_frames": self._tool_gui_track_next_frames,
                "set_ai_text_prompt": self._tool_gui_set_ai_text_prompt,
                "run_ai_text_segmentation": self._tool_gui_run_ai_text_segmentation,
                "segment_track_video": self._tool_gui_segment_track_video,
                "label_behavior_segments": self._tool_gui_label_behavior_segments,
                "start_realtime_stream": self._tool_gui_start_realtime_stream,
                "stop_realtime_stream": self._tool_gui_stop_realtime_stream,
                "arxiv_search": self._tool_gui_arxiv_search,
                "list_pdfs": self._tool_gui_list_pdfs,
                "save_citation": self._tool_gui_save_citation,
            },
        )

    def _resolve_loop_llm_callable(self) -> Optional[Callable[..., Any]]:
        if self.provider == "ollama":
            return self._build_ollama_llm_callable()
        return None

    def _finalize_agent_text(
        self,
        result: Any,
        *,
        tools: Optional[FunctionToolRegistry] = None,
    ) -> Tuple[str, bool, bool]:
        # Backward-compatible sync entrypoint used by tests and legacy callers.
        return self._run_async(self._finalize_agent_text_async(result, tools=tools))

    async def _finalize_agent_text_async(
        self,
        result: Any,
        *,
        tools: Optional[FunctionToolRegistry] = None,
    ) -> Tuple[str, bool, bool]:
        text = str(getattr(result, "content", "") or "").strip()
        tool_run_count = len(getattr(result, "tool_runs", ()) or ())
        (
            text,
            used_direct_gui_fallback,
            direct_gui_text,
        ) = await self._apply_direct_gui_fallback(text, tool_run_count=tool_run_count)
        text = await self._apply_web_response_fallbacks(text, tools=tools)
        text = self._apply_pdf_response_fallback(text)
        text, used_recovery = self._apply_empty_ollama_recovery(
            text,
            used_direct_gui_fallback=used_direct_gui_fallback,
            direct_gui_text=direct_gui_text,
        )
        text = self._ensure_non_empty_final_text(text)
        if self.show_tool_trace:
            trace = self._format_tool_trace(getattr(result, "tool_runs", ()) or ())
            text = f"{text}\n\n{trace}".strip()
        self._emit_progress("Finalizing response")
        return text, used_recovery, used_direct_gui_fallback

    async def _apply_direct_gui_fallback(
        self,
        text: str,
        *,
        tool_run_count: int,
    ) -> Tuple[str, bool, str]:
        return await gui_apply_direct_gui_fallback(
            text=text,
            provider=self.provider,
            tool_run_count=tool_run_count,
            prompt=self.prompt,
            execute_direct_gui_command=self._execute_direct_gui_command,
            looks_like_local_access_refusal=self._looks_like_local_access_refusal,
            logger=logger,
            session_id=self.session_id,
            model=self.model,
        )

    async def _apply_web_response_fallbacks(
        self,
        text: str,
        *,
        tools: Optional[FunctionToolRegistry],
    ) -> str:
        return await gui_apply_web_response_fallbacks(
            text=text,
            prompt=self.prompt,
            tools=tools,
            enable_web_tools=self.enable_web_tools,
            looks_like_open_url_suggestion=self._looks_like_open_url_suggestion,
            should_apply_web_refusal_fallback_cb=self._should_apply_web_refusal_fallback,
            try_open_page_content_fallback=self._try_open_page_content_fallback,
            try_browser_search_fallback=self._try_browser_search_fallback,
            try_web_fetch_fallback=self._try_web_fetch_fallback,
        )

    def _should_apply_web_refusal_fallback(self, text: str) -> bool:
        return gui_should_apply_web_refusal_fallback(
            text,
            looks_like_web_access_refusal=self._looks_like_web_access_refusal,
            looks_like_knowledge_gap_response=self._looks_like_knowledge_gap_response,
        )

    def _apply_pdf_response_fallback(self, text: str) -> str:
        return gui_apply_pdf_response_fallback(
            text,
            looks_like_local_access_refusal=self._looks_like_local_access_refusal,
            looks_like_open_pdf_suggestion=self._looks_like_open_pdf_suggestion,
            try_open_pdf_content_fallback=self._try_open_pdf_content_fallback,
        )

    def _apply_empty_ollama_recovery(
        self,
        text: str,
        *,
        used_direct_gui_fallback: bool,
        direct_gui_text: str,
    ) -> Tuple[str, bool]:
        return gui_apply_empty_ollama_recovery(
            text,
            provider=self.provider,
            model=self.model,
            used_direct_gui_fallback=used_direct_gui_fallback,
            direct_gui_text=direct_gui_text,
            recover_with_plain_ollama_reply=self._recover_with_plain_ollama_reply,
            ollama_mark_plain_mode=ollama_mark_plain_mode,
        )

    def _ensure_non_empty_final_text(self, text: str) -> str:
        return gui_ensure_non_empty_final_text(
            text,
            provider=self.provider,
            model=self.model,
        )

    def _log_agent_result(
        self,
        result: Any,
        used_recovery: bool,
        used_direct_gui_fallback: bool,
    ) -> None:
        gui_log_agent_result(
            logger=logger,
            session_id=self.session_id,
            provider=self.provider,
            model=self.model,
            result=result,
            used_recovery=used_recovery,
            used_direct_gui_fallback=used_direct_gui_fallback,
        )

    def _wrap_tool_callback(
        self, name: str, callback: Callable[..., Any]
    ) -> Callable[..., Any]:
        return gui_wrap_tool_callback(
            name=name,
            callback=callback,
            emit_progress=self._emit_progress,
        )

    def _get_shared_image_path(self) -> str:
        return str(self.image_path or "")

    def _build_gui_context_payload(self) -> Dict[str, Any]:
        return gui_build_gui_context_payload(
            session_id=self.session_id,
            provider=self.provider,
            model=self.model,
            prompt=self.prompt,
            image_path=str(self.image_path or ""),
            widget=self.widget,
            web_state_getter=self._tool_gui_web_get_state,
            pdf_state_getter=self._tool_gui_pdf_get_state,
        )

    def _invoke_widget_slot(self, slot_name: str, *qargs: Any) -> bool:
        return gui_invoke_widget_slot(
            widget=self.widget,
            session_id=self.session_id,
            slot_name=slot_name,
            qargs=qargs,
            logger=logger,
        )

    def _invoke_widget_json_slot(self, slot_name: str, *qargs: Any) -> Dict[str, Any]:
        return gui_invoke_widget_json_slot(
            widget=self.widget,
            invoke_slot=self._invoke_widget_slot,
            slot_name=slot_name,
            qargs=qargs,
        )

    def _tool_gui_open_video(self, path: str) -> Dict[str, Any]:
        return gui_open_video_tool(
            path,
            resolve_video_path=self._resolve_video_path_for_gui_tool,
            invoke_open_video=lambda video_path: self._invoke_widget_slot(
                "bot_open_video", QtCore.Q_ARG(str, str(video_path))
            ),
        )

    async def _tool_gui_open_url(self, url: str) -> Dict[str, Any]:
        return await gui_open_url_tool(
            url,
            extract_first_web_url_fn=self._extract_first_web_url,
            emit_progress=self._emit_progress,
            run_arxiv_search=lambda query: self._safe_run_arxiv_search(query=query),
            invoke_open_url=lambda target_url: self._invoke_widget_slot(
                "bot_open_url", QtCore.Q_ARG(str, target_url)
            ),
        )

    def _tool_gui_open_in_browser(self, url: str) -> Dict[str, Any]:
        return gui_open_in_browser_tool(
            url,
            extract_first_web_url_fn=self._extract_first_web_url,
            invoke_open_in_browser=lambda target_url: self._invoke_widget_slot(
                "bot_open_in_browser", QtCore.Q_ARG(str, target_url)
            ),
        )

    def _tool_gui_web_get_dom_text(self, max_chars: int = 8000) -> Dict[str, Any]:
        return gui_web_get_dom_text(
            invoke_widget_json_slot=self._invoke_widget_json_slot,
            max_chars=max_chars,
        )

    def _tool_gui_web_get_state(self) -> Dict[str, Any]:
        return gui_web_get_state(invoke_widget_json_slot=self._invoke_widget_json_slot)

    def _tool_gui_web_click(self, selector: str) -> Dict[str, Any]:
        return gui_web_click(
            invoke_widget_json_slot=self._invoke_widget_json_slot,
            selector=selector,
        )

    def _tool_gui_web_type(
        self, selector: str, text: str, submit: bool = False
    ) -> Dict[str, Any]:
        return gui_web_type(
            invoke_widget_json_slot=self._invoke_widget_json_slot,
            selector=selector,
            text=text,
            submit=submit,
        )

    def _tool_gui_web_scroll(self, delta_y: int = 800) -> Dict[str, Any]:
        return gui_web_scroll(
            invoke_widget_json_slot=self._invoke_widget_json_slot,
            delta_y=delta_y,
        )

    def _tool_gui_web_find_forms(self) -> Dict[str, Any]:
        return gui_web_find_forms(invoke_widget_json_slot=self._invoke_widget_json_slot)

    async def _tool_gui_web_run_steps(
        self,
        steps: Any,
        stop_on_error: bool = True,
        max_steps: int = 12,
    ) -> Dict[str, Any]:
        return await gui_web_run_steps(
            steps=steps,
            stop_on_error=stop_on_error,
            max_steps=max_steps,
            open_url=self._tool_gui_open_url,
            open_in_browser=self._tool_gui_open_in_browser,
            get_dom_text=self._tool_gui_web_get_dom_text,
            click=self._tool_gui_web_click,
            type_text=self._tool_gui_web_type,
            scroll=self._tool_gui_web_scroll,
            find_forms=self._tool_gui_web_find_forms,
            sleep_ms=lambda ms: QtCore.QThread.msleep(ms),
        )

    @staticmethod
    def _extract_first_web_url(text: str) -> str:
        return gui_extract_first_web_url(
            text,
            extract_web_urls=StreamingChatTask._extract_web_urls,
        )

    @staticmethod
    def _extract_pdf_path_candidates(raw: str) -> List[str]:
        return extract_pdf_path_candidates(raw)

    def _resolve_pdf_path_for_gui_tool(self, raw_path: str) -> Optional[Path]:
        try:
            cfg = load_config()
            read_roots_cfg = list(getattr(cfg.tools, "allowed_read_roots", []) or [])
        except Exception:
            read_roots_cfg = []
        roots = build_workspace_roots(get_agent_workspace_path(), read_roots_cfg)
        return resolve_pdf_path_for_roots(raw_path, roots)

    def _pdf_search_roots(self) -> List[Path]:
        workspace = get_agent_workspace_path()
        try:
            cfg = load_config()
            read_roots_cfg = list(getattr(cfg.tools, "allowed_read_roots", []) or [])
        except Exception:
            read_roots_cfg = []
        return build_pdf_search_roots(workspace, read_roots_cfg)

    def _list_available_pdfs(
        self, *, limit: int = 8, max_scan: int = 40000
    ) -> List[Path]:
        return list_available_pdfs_in_roots(
            self._pdf_search_roots(),
            limit=limit,
            max_scan=max_scan,
        )

    async def _tool_gui_open_pdf(self, path: str = "") -> Dict[str, Any]:
        return await gui_open_pdf_tool(
            path,
            extract_pdf_path_candidates=self._extract_pdf_path_candidates,
            extract_web_urls=self._extract_web_urls,
            download_pdf=self._download_pdf_for_gui_tool,
            resolve_pdf_path=self._resolve_pdf_path_for_gui_tool,
            list_available_pdfs=lambda limit: self._list_available_pdfs(limit=limit),
            invoke_open_pdf=lambda resolved_path: self._invoke_widget_slot(
                "bot_open_pdf", QtCore.Q_ARG(str, str(resolved_path))
            ),
        )

    def _tool_gui_pdf_get_state(self) -> Dict[str, Any]:
        return gui_pdf_get_state(invoke_widget_json_slot=self._invoke_widget_json_slot)

    def _tool_gui_pdf_get_text(
        self, max_chars: int = 8000, pages: int = 2
    ) -> Dict[str, Any]:
        return gui_pdf_get_text(
            invoke_widget_json_slot=self._invoke_widget_json_slot,
            max_chars=max_chars,
            pages=pages,
        )

    def _tool_gui_pdf_find_sections(
        self,
        max_sections: int = 20,
        max_pages: int = 12,
    ) -> Dict[str, Any]:
        return gui_pdf_find_sections(
            invoke_widget_json_slot=self._invoke_widget_json_slot,
            max_sections=max_sections,
            max_pages=max_pages,
        )

    async def _download_pdf_for_gui_tool(self, url: str) -> Optional[Path]:
        text = str(url or "").strip()
        if not text.lower().startswith(("http://", "https://")):
            return None
        workspace = get_agent_workspace_path()
        downloader = DownloadPdfTool(allowed_dir=workspace)
        try:
            payload_raw = await downloader.execute(url=text)
            payload = json.loads(str(payload_raw or "{}"))
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("error"):
            return None
        output_path = str(payload.get("output_path") or "").strip()
        if not output_path:
            return None
        candidate = Path(output_path).expanduser()
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    @staticmethod
    def _run_async(awaitable: Any) -> Any:
        return gui_run_awaitable_sync(awaitable)

    @staticmethod
    def _extract_path_candidates(raw: str) -> List[str]:
        return extract_video_path_candidates(raw)

    def _resolve_video_path_for_gui_tool(self, raw_path: str) -> Optional[Path]:
        return gui_resolve_video_path_for_gui_tool(
            raw_path,
            widget=self.widget,
            load_config_fn=load_config,
            get_workspace_path_fn=get_agent_workspace_path,
            build_workspace_roots_fn=build_workspace_roots,
            resolve_video_path_for_roots_fn=resolve_video_path_for_roots,
        )

    @staticmethod
    def _find_video_by_basename_in_roots(
        *, basenames: set[str], roots: List[Path], max_scan: int = 30000
    ) -> Optional[Path]:
        return find_video_by_basename_in_roots(
            basenames=basenames,
            roots=roots,
            max_scan=max_scan,
        )

    def _maybe_run_direct_gui_tool_from_prompt(self, prompt: str) -> str:
        return self._run_async(self._execute_direct_gui_command(prompt))

    async def _execute_direct_gui_command(self, prompt: str) -> str:
        return await gui_execute_direct_gui_command(
            prompt=prompt,
            parse_direct_gui_command=self._parse_direct_gui_command,
            route_direct_gui_command=execute_direct_gui_command,
            handlers={
                "open_video": self._tool_gui_open_video,
                "open_url": self._tool_gui_open_url,
                "open_in_browser": self._tool_gui_open_in_browser,
                "open_pdf": self._tool_gui_open_pdf,
                "set_frame": self._tool_gui_set_frame,
                "track_next_frames": self._tool_gui_track_next_frames,
                "segment_track_video": self._tool_gui_segment_track_video,
                "label_behavior_segments": self._tool_gui_label_behavior_segments,
                "start_realtime_stream": self._tool_gui_start_realtime_stream,
                "stop_realtime_stream": self._tool_gui_stop_realtime_stream,
                "list_pdfs": self._tool_gui_list_pdfs,
                "clawhub_search_skills": self._tool_clawhub_search_skills,
                "clawhub_install_skill": self._tool_clawhub_install_skill,
                "set_chat_model": self._tool_gui_set_chat_model,
                "rename_file": self._tool_gui_rename_file,
                "list_citations": self._tool_gui_list_citations,
                "add_citation_raw": self._tool_gui_add_citation_raw,
                "save_citation": self._tool_gui_save_citation,
            },
        )

    def _parse_direct_gui_command(self, prompt: str) -> Dict[str, Any]:
        return parse_direct_gui_command(prompt)

    @staticmethod
    def _looks_like_local_access_refusal(text: str) -> bool:
        return looks_like_local_access_refusal(text)

    @staticmethod
    def _looks_like_web_access_refusal(text: str) -> bool:
        return looks_like_web_access_refusal(text)

    @staticmethod
    def _looks_like_knowledge_gap_response(text: str) -> bool:
        return looks_like_knowledge_gap_response(text)

    @staticmethod
    def _looks_like_open_url_suggestion(text: str) -> bool:
        return looks_like_open_url_suggestion(text)

    @staticmethod
    def _looks_like_open_pdf_suggestion(text: str) -> bool:
        return looks_like_open_pdf_suggestion(text)

    @staticmethod
    def _extract_web_urls(text: str) -> List[str]:
        return extract_web_urls(text)

    def _candidate_web_urls_for_prompt(self, prompt: str) -> List[str]:
        return candidate_web_urls_for_prompt(
            prompt,
            extract_web_urls=self._extract_web_urls,
            load_history_messages=self._load_history_messages,
        )

    @staticmethod
    def _build_extractive_summary(
        text: str,
        *,
        max_sentences: int = 6,
        max_chars: int = 1200,
    ) -> str:
        return build_extractive_summary(
            text, max_sentences=max_sentences, max_chars=max_chars
        )

    async def _try_web_fetch_fallback(
        self,
        prompt: str,
        tools: Optional[FunctionToolRegistry],
    ) -> str:
        return await try_web_fetch_fallback(
            prompt=prompt,
            tools=tools,
            candidate_urls_for_prompt=self._candidate_web_urls_for_prompt,
            build_summary=self._build_extractive_summary,
            emit_progress=self._emit_progress,
        )

    @staticmethod
    def _extract_page_text_from_web_steps(payload: Dict[str, Any]) -> str:
        return extract_page_text_from_web_steps(payload)

    async def _try_browser_search_fallback(
        self,
        prompt: str,
        tools: Optional[FunctionToolRegistry],
    ) -> str:
        return await try_browser_search_fallback(
            prompt=prompt,
            tools=tools,
            emit_progress=self._emit_progress,
            build_summary=self._build_extractive_summary,
        )

    def _try_open_page_content_fallback(self) -> str:
        return try_open_page_content_fallback(
            prompt=self.prompt,
            get_state=self._tool_gui_web_get_state,
            get_dom_text=self._tool_gui_web_get_dom_text,
            should_use_open_page_fallback=self._should_use_open_page_fallback,
            topic_tokens=self._topic_tokens,
            build_summary=self._build_extractive_summary,
        )

    def _try_open_pdf_content_fallback(self) -> str:
        return try_open_pdf_content_fallback(
            get_state=self._tool_gui_pdf_get_state,
            get_text=self._tool_gui_pdf_get_text,
            build_summary=self._build_extractive_summary,
        )

    def _tool_gui_set_frame(self, frame_index: int) -> Dict[str, Any]:
        return gui_set_frame_tool(
            frame_index,
            invoke_set_frame=lambda target_frame: self._invoke_widget_slot(
                "bot_set_frame", QtCore.Q_ARG(int, target_frame)
            ),
        )

    def _tool_gui_set_chat_prompt(self, text: str) -> Dict[str, Any]:
        return gui_set_chat_prompt_tool(
            text,
            invoke_set_chat_prompt=lambda prompt_text: self._invoke_widget_slot(
                "bot_set_chat_prompt", QtCore.Q_ARG(str, prompt_text)
            ),
        )

    def _tool_gui_send_chat_prompt(self) -> Dict[str, Any]:
        return gui_send_chat_prompt_tool(
            invoke_send_chat_prompt=lambda: self._invoke_widget_slot(
                "bot_send_chat_prompt"
            )
        )

    def _tool_gui_set_chat_model(self, provider: str, model: str) -> Dict[str, Any]:
        return gui_set_chat_model_tool(
            provider,
            model,
            invoke_set_chat_model=lambda provider_text,
            model_text: self._invoke_widget_slot(
                "bot_set_chat_model",
                QtCore.Q_ARG(str, provider_text),
                QtCore.Q_ARG(str, model_text),
            ),
        )

    def _tool_gui_select_annotation_model(self, model_name: str) -> Dict[str, Any]:
        return gui_select_annotation_model_tool(
            model_name,
            invoke_select_annotation_model=lambda model_text: self._invoke_widget_slot(
                "bot_select_annotation_model", QtCore.Q_ARG(str, model_text)
            ),
        )

    def _tool_gui_track_next_frames(self, to_frame: int) -> Dict[str, Any]:
        return gui_track_next_frames_tool(
            to_frame,
            invoke_track_next_frames=lambda frame: self._invoke_widget_slot(
                "bot_track_next_frames", QtCore.Q_ARG(int, frame)
            ),
        )

    def _tool_gui_set_ai_text_prompt(
        self, text: str, use_countgd: bool = False
    ) -> Dict[str, Any]:
        return gui_set_ai_text_prompt_tool(
            text,
            use_countgd=use_countgd,
            invoke_set_ai_text_prompt=lambda prompt_text,
            flag: self._invoke_widget_slot(
                "bot_set_ai_text_prompt",
                QtCore.Q_ARG(str, prompt_text),
                QtCore.Q_ARG(bool, flag),
            ),
        )

    def _tool_gui_run_ai_text_segmentation(self) -> Dict[str, Any]:
        return gui_run_ai_text_segmentation_tool(
            invoke_run_ai_text_segmentation=lambda: self._invoke_widget_slot(
                "bot_run_ai_text_segmentation"
            )
        )

    def _tool_gui_segment_track_video(
        self,
        *,
        path: str,
        text_prompt: str,
        mode: str = "track",
        use_countgd: bool = False,
        model_name: str = "",
        to_frame: Optional[int] = None,
    ) -> Dict[str, Any]:
        return gui_segment_track_video_tool(
            path=path,
            text_prompt=text_prompt,
            mode=mode,
            use_countgd=use_countgd,
            model_name=model_name,
            to_frame=to_frame,
            resolve_video_path=self._resolve_video_path_for_gui_tool,
            invoke_segment_track=lambda vpath,
            prompt,
            mode_norm,
            countgd,
            model,
            frame: self._invoke_widget_slot(
                "bot_segment_track_video",
                QtCore.Q_ARG(str, vpath),
                QtCore.Q_ARG(str, prompt),
                QtCore.Q_ARG(str, mode_norm),
                QtCore.Q_ARG(bool, countgd),
                QtCore.Q_ARG(str, model),
                QtCore.Q_ARG(int, frame),
            ),
            get_action_result=self._get_widget_action_result,
        )

    def _tool_gui_label_behavior_segments(
        self,
        *,
        path: str = "",
        behavior_labels: Any = None,
        segment_mode: str = "timeline",
        segment_frames: int = 60,
        max_segments: int = 120,
        subject: str = "Agent",
        overwrite_existing: bool = False,
        llm_profile: str = "",
        llm_provider: str = "",
        llm_model: str = "",
    ) -> Dict[str, Any]:
        return gui_label_behavior_segments_tool(
            path=path,
            behavior_labels=behavior_labels,
            segment_mode=segment_mode,
            segment_frames=segment_frames,
            max_segments=max_segments,
            subject=subject,
            overwrite_existing=overwrite_existing,
            llm_profile=llm_profile,
            llm_provider=llm_provider,
            llm_model=llm_model,
            resolve_video_path=self._resolve_video_path_for_gui_tool,
            invoke_label_behavior=lambda resolved_path,
            labels,
            mode_norm,
            frames,
            max_seg,
            subj,
            overwrite,
            profile,
            provider,
            model: self._invoke_widget_slot(
                "bot_label_behavior_segments",
                QtCore.Q_ARG(str, resolved_path),
                QtCore.Q_ARG(str, labels),
                QtCore.Q_ARG(str, mode_norm),
                QtCore.Q_ARG(int, frames),
                QtCore.Q_ARG(int, max_seg),
                QtCore.Q_ARG(str, subj),
                QtCore.Q_ARG(bool, overwrite),
                QtCore.Q_ARG(str, profile),
                QtCore.Q_ARG(str, provider),
                QtCore.Q_ARG(str, model),
            ),
            get_action_result=self._get_widget_action_result,
        )

    def _get_widget_action_result(self, action_name: str) -> Dict[str, Any]:
        return gui_get_widget_action_result(widget=self.widget, action_name=action_name)

    async def _safe_run_arxiv_search(self, query: str) -> None:
        await gui_safe_run_arxiv_search(
            query=query,
            run_arxiv_search=lambda **kwargs: self._tool_gui_arxiv_search(**kwargs),
            emit_progress=self._emit_progress,
            log_error=lambda msg: logging.error(
                "ArXiv operation crashed: %s", msg, exc_info=True
            ),
        )

    def _tool_gui_list_pdfs(self, query: str = None) -> Dict[str, Any]:
        return gui_list_local_pdfs(
            workspace=get_agent_workspace_path(),
            query=query,
            max_results=20,
        )

    @staticmethod
    def _extract_doi(text: str) -> str:
        return gui_extract_doi(text)

    @staticmethod
    def _extract_year(text: str) -> str:
        return gui_extract_year(text)

    @staticmethod
    def _normalize_citation_key(title: str, year: str, fallback: str = "paper") -> str:
        return gui_normalize_citation_key(title, year, fallback=fallback)

    def _resolve_bib_output_path(self, bib_file: str) -> Path:
        return gui_resolve_bib_output_path(
            bib_file,
            workspace=get_agent_workspace_path(),
        )

    def _citation_fields_from_pdf_state(self) -> Dict[str, Any]:
        return gui_citation_fields_from_pdf_state(
            get_pdf_state=self._tool_gui_pdf_get_state,
            get_pdf_text=self._tool_gui_pdf_get_text,
        )

    def _citation_fields_from_web_state(self) -> Dict[str, Any]:
        return gui_citation_fields_from_web_state(
            get_web_state=self._tool_gui_web_get_state,
            get_web_text=self._tool_gui_web_get_dom_text,
        )

    def _tool_gui_save_citation(
        self,
        *,
        key: str = "",
        bib_file: str = "",
        source: str = "auto",
        entry_type: str = "article",
        validate_before_save: bool = True,
        strict_validation: bool = False,
    ) -> Dict[str, Any]:
        return gui_save_citation_tool(
            key=key,
            bib_file=bib_file,
            source=source,
            entry_type=entry_type,
            validate_before_save=validate_before_save,
            strict_validation=strict_validation,
            choose_pdf_fields=self._citation_fields_from_pdf_state,
            choose_web_fields=self._citation_fields_from_web_state,
            resolve_bib_path=self._resolve_bib_output_path,
            validate_basic_fields=validate_basic_citation_fields,
            validate_metadata=lambda fields, timeout: validate_citation_metadata(
                fields, timeout_s=timeout
            ),
            merge_fields=lambda fields, validation, replace: merge_validated_fields(
                fields, validation, replace_when_confident=replace
            ),
            load_bibtex=load_bibtex,
            upsert_entry=upsert_entry,
            save_bibtex=lambda path, entries, sort_keys=True: save_bibtex(
                path, entries, sort_keys=sort_keys
            ),
            bib_entry_cls=BibEntry,
        )

    def _tool_gui_add_citation_raw(
        self,
        *,
        bibtex: str = "",
        bib_file: str = "",
    ) -> Dict[str, Any]:
        return gui_add_citation_raw_tool(
            bibtex=bibtex,
            bib_file=bib_file,
            parse_bibtex=parse_bibtex,
            resolve_bib_path=self._resolve_bib_output_path,
            load_bibtex=load_bibtex,
            upsert_entry=upsert_entry,
            save_bibtex=lambda path, entries, sort_keys=True: save_bibtex(
                path, entries, sort_keys=sort_keys
            ),
        )

    def _tool_gui_list_citations(
        self,
        *,
        bib_file: str = "",
        query: str = "",
        limit: int = 20,
    ) -> Dict[str, Any]:
        return gui_list_citations_tool(
            bib_file=bib_file,
            query=query,
            limit=limit,
            resolve_bib_path=self._resolve_bib_output_path,
            load_bibtex=load_bibtex,
            search_entries=search_entries,
            entry_to_dict=entry_to_dict,
        )

    async def _tool_clawhub_search_skills(
        self, query: str, limit: int = 5
    ) -> Dict[str, Any]:
        self._emit_progress(f"ClawHub search: {str(query or '').strip()}")
        return await clawhub_search_skills(
            str(query or ""),
            limit=int(limit or 5),
            workspace=get_agent_workspace_path(),
        )

    async def _tool_clawhub_install_skill(self, slug: str) -> Dict[str, Any]:
        self._emit_progress(f"Installing ClawHub skill: {str(slug or '').strip()}")
        return await clawhub_install_skill(
            str(slug or ""),
            workspace=get_agent_workspace_path(),
        )

    async def _tool_gui_arxiv_search(
        self, query: str, max_results: int = 1
    ) -> Dict[str, Any]:
        return await gui_arxiv_search_tool(
            query=query,
            max_results=max_results,
            workspace=Path(self.workspace),
            emit_progress=self._emit_progress,
            open_pdf=self._tool_gui_open_pdf,
        )

    def _tool_gui_start_realtime_stream(
        self,
        *,
        camera_source: str = "",
        model_name: str = "",
        target_behaviors: Any = None,
        confidence_threshold: Optional[float] = None,
        viewer_type: str = "threejs",
        classify_eye_blinks: bool = False,
        blink_ear_threshold: Optional[float] = None,
        blink_min_consecutive_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        return gui_start_realtime_stream_tool(
            camera_source=camera_source,
            model_name=model_name,
            target_behaviors=target_behaviors,
            confidence_threshold=confidence_threshold,
            viewer_type=viewer_type,
            classify_eye_blinks=classify_eye_blinks,
            blink_ear_threshold=blink_ear_threshold,
            blink_min_consecutive_frames=blink_min_consecutive_frames,
            invoke_start=lambda camera_text,
            model_text,
            targets,
            threshold,
            viewer,
            classify,
            ear_threshold,
            min_blink_frames: self._invoke_widget_slot(
                "bot_start_realtime_stream",
                QtCore.Q_ARG(str, camera_text),
                QtCore.Q_ARG(str, model_text),
                QtCore.Q_ARG(str, targets),
                QtCore.Q_ARG(float, threshold),
                QtCore.Q_ARG(str, viewer),
                QtCore.Q_ARG(bool, classify),
                QtCore.Q_ARG(float, ear_threshold),
                QtCore.Q_ARG(int, min_blink_frames),
            ),
            get_action_result=self._get_widget_action_result,
        )

    def _tool_gui_stop_realtime_stream(self) -> Dict[str, Any]:
        return gui_stop_realtime_stream_tool(
            invoke_stop=lambda: self._invoke_widget_slot("bot_stop_realtime_stream")
        )

    def _tool_gui_rename_file(
        self,
        source_path: str = "",
        new_name: str = "",
        new_path: str = "",
        use_active_file: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        workspace = get_agent_workspace_path()
        return gui_rename_file_tool(
            source_path=source_path,
            new_name=new_name,
            new_path=new_path,
            use_active_file=use_active_file,
            overwrite=overwrite,
            get_pdf_state=self._tool_gui_pdf_get_state,
            get_active_video_path=self._get_active_video_path,
            workspace=workspace,
            run_rename=lambda current, target_name, target_path, overwrite_flag: str(
                self._run_async(
                    RenameFileTool(allowed_dir=workspace).execute(
                        path=current,
                        new_name=target_name,
                        new_path=target_path,
                        overwrite=overwrite_flag,
                    )
                )
                or ""
            ),
            reopen_pdf=lambda resolved_new_path: self._invoke_widget_slot(
                "bot_open_pdf", QtCore.Q_ARG(str, str(resolved_new_path))
            ),
        )

    def _get_active_video_path(self) -> str:
        widget = self.widget
        host = getattr(widget, "host_window_widget", None) if widget else None
        return str(getattr(host, "video_file", "") or "").strip()

    def _recover_with_plain_ollama_reply(self) -> str:
        return recover_with_plain_ollama_reply(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            settings=self.settings,
            logger=logger,
            import_module=importlib.import_module,
            first_timeout_s=self._ollama_plain_recovery_timeout_seconds(),
            nudge_timeout_s=self._ollama_plain_recovery_nudge_timeout_seconds(),
        )

    def _build_compact_system_prompt(
        self,
        workspace: Path,
        *,
        allowed_read_roots: Optional[List[str]] = None,
        allow_web_tools: Optional[bool] = None,
        include_workspace_docs: bool = True,
    ) -> str:
        return build_gui_compact_system_prompt(
            inputs=PromptBuildInputs(
                workspace=workspace,
                prompt=self.prompt,
                enable_web_tools=self.enable_web_tools,
                enable_ollama_fallback=getattr(self, "enable_ollama_fallback", False),
                allowed_read_roots=allowed_read_roots,
                allow_web_tools=allow_web_tools,
                include_workspace_docs=include_workspace_docs,
                now=datetime.now(),
            ),
            read_text_limited=_read_text_limited_cached,
            list_skill_names=_list_workspace_skill_names_cached,
            should_attach_live_web_context=self._should_attach_live_web_context,
            should_attach_live_pdf_context=self._should_attach_live_pdf_context,
            build_live_web_context_prompt_block=self._build_live_web_context_prompt_block,
            build_live_pdf_context_prompt_block=self._build_live_pdf_context_prompt_block,
        )

    @staticmethod
    def _contains_hint(text: str, hints: Tuple[str, ...]) -> bool:
        return contains_hint(text, hints)

    @staticmethod
    def _looks_like_url_request(text: str) -> bool:
        return looks_like_url_request(text)

    def _should_attach_live_web_context(self, prompt: str) -> bool:
        return should_attach_live_web_context(prompt)

    def _should_attach_live_pdf_context(self, prompt: str) -> bool:
        return should_attach_live_pdf_context(prompt)

    def _should_use_open_page_fallback(self, prompt: str) -> bool:
        return self._should_attach_live_web_context(prompt)

    @staticmethod
    def _topic_tokens(text: str) -> List[str]:
        return topic_tokens(text)

    def _build_live_web_context_prompt_block(
        self, *, include_snapshot: bool = True
    ) -> str:
        return build_live_web_context_prompt_block(
            get_state=self._tool_gui_web_get_state,
            get_dom_text=self._tool_gui_web_get_dom_text,
            include_snapshot=include_snapshot,
        )

    def _build_live_pdf_context_prompt_block(
        self, *, include_snapshot: bool = True
    ) -> str:
        return build_live_pdf_context_prompt_block(
            get_state=self._tool_gui_pdf_get_state,
            get_text=self._tool_gui_pdf_get_text,
            include_snapshot=include_snapshot,
        )

    def _build_ollama_llm_callable(self):
        return build_gui_ollama_llm_callable(
            prompt=self.prompt,
            settings=self.settings,
            prompt_may_need_tools=self._prompt_may_need_tools,
            logger=logger,
            tool_request_timeout_s=self._ollama_agent_tool_timeout_seconds(),
            plain_request_timeout_s=self._ollama_agent_plain_timeout_seconds(),
        )

    @classmethod
    def _collect_ollama_stream(
        cls, stream_iter: Any
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        return collect_gui_ollama_stream(stream_iter)

    @staticmethod
    def _parse_ollama_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
        return parse_gui_ollama_tool_calls(raw_calls)

    @staticmethod
    def _normalize_messages_for_ollama(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return normalize_gui_messages_for_ollama(messages)

    @staticmethod
    def _extract_ollama_text(response: Dict[str, Any]) -> str:
        return extract_gui_ollama_text(response)

    @staticmethod
    def _format_tool_trace(tool_runs: Any) -> str:
        return format_gui_tool_trace(tool_runs)

    @staticmethod
    def _prompt_may_need_tools(prompt: str) -> bool:
        return prompt_may_need_tools(prompt)

    @staticmethod
    def _prompt_may_need_mcp(prompt: str) -> bool:
        return prompt_may_need_mcp(prompt)

    def _run_ollama(self) -> None:
        gui_run_ollama_chat(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            settings=self.settings,
            load_history_messages=self._load_history_messages,
            emit_chunk=self._emit_chunk,
            emit_final=lambda message, is_error: self._emit_final(
                message, is_error=is_error
            ),
            persist_turn=lambda user_text, assistant_text: self._persist_turn(
                user_text, assistant_text
            ),
        )

    def _run_openai(
        self,
        provider_name: str = "openai",
        *,
        timeout_s: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> None:
        user_prompt, text = gui_run_openai_chat(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            provider_name=provider_name,
            settings=self.settings,
            load_history_messages=self._load_history_messages,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
        self._persist_turn(user_prompt, text)
        self._emit_final(text, is_error=False)

    def _run_gemini(self) -> None:
        user_prompt, text = gui_run_gemini_provider_chat(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            provider_name=self.provider,
            settings=self.settings,
        )
        self._persist_turn(user_prompt, text)
        self._emit_final(text, is_error=False)
