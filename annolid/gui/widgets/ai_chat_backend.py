from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from qtpy import QtCore
from qtpy.QtCore import QMetaObject, QRunnable

from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.config import load_config
from annolid.core.agent.memory import AgentMemoryStore
from annolid.core.agent.providers import (
    build_ollama_llm_callable,
    dependency_error_for_kind,
    ollama_mark_plain_mode,
    ollama_plain_mode_decrement,
    ollama_plain_mode_remaining,
    recover_with_plain_ollama_reply,
    run_gemini_chat,
    run_ollama_streaming_chat,
    run_openai_compat_chat,
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
    register_annolid_gui_tools,
    register_nanobot_style_tools,
)
from annolid.core.agent.tools.pdf import DownloadPdfTool
from annolid.core.agent.tools.filesystem import RenameFileTool
from annolid.core.agent.tools.policy import resolve_allowed_tools
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.core.agent.gui_backend.commands import (
    looks_like_local_access_refusal,
    parse_direct_gui_command,
    prompt_may_need_tools,
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
from annolid.core.agent.providers.ollama_utils import (
    collect_ollama_stream,
    extract_ollama_text,
    format_tool_trace,
    normalize_messages_for_ollama,
    parse_ollama_tool_calls,
)
from annolid.utils.llm_settings import (
    provider_kind,
    resolve_agent_runtime_config,
)


_SESSION_STORE: Optional[PersistentSessionStore] = None
_LOGGER = logging.getLogger("annolid.bot.backend")
_GUI_ALWAYS_DISABLED_TOOLS = {"cron", "spawn", "message"}
_GUI_WEB_TOOLS = {"web_search", "web_fetch"}
_WEB_ACCESS_REFUSAL_HINTS = (
    "don't have web browsing capabilities",
    "do not have web browsing capabilities",
    "cannot directly fetch urls",
    "can't directly fetch urls",
    "i cannot directly fetch urls",
    "i can't directly fetch urls",
    "cannot browse the web",
    "can't browse the web",
    "cannot access external websites",
    "can't access external websites",
    "cannot access the internet",
    "can't access the internet",
    "no browsing capability",
)
_KNOWLEDGE_GAP_HINTS = (
    "i don't have access to",
    "i do not have access to",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i cannot determine",
    "i can't determine",
    "you can check by",
    "check a website",
    "check an app",
    "can't check",
    "cannot check",
    "isn't configured",
    "is not configured",
    "not configured",
    "web search api",
    "api key",
    "in your browser",
)
_OPEN_URL_SUGGESTION_HINTS = (
    "open your browser",
    "in your browser",
    "search for",
    "go to ",
    "visit ",
    "check weather.gov",
    "check accuweather",
)
_OPEN_PDF_SUGGESTION_HINTS = (
    "open pdf",
    "open the pdf",
    "upload pdf",
    "share the pdf",
    "provide the pdf",
    "cannot access your local file",
    "can't access your local file",
    "cannot access local file",
    "can't access local file",
)
_WEB_CONTEXT_HINTS = (
    "this page",
    "current page",
    "open page",
    "web page",
    "website",
    "site",
    "browser",
    "tab",
)
_EMBEDDED_SEARCH_URL_TEMPLATE = "https://html.duckduckgo.com/html/?q={query}"
_EMBEDDED_SEARCH_SOURCE = "DuckDuckGo search results page (embedded web viewer)."
_PDF_CONTEXT_HINTS = (
    "pdf",
    "document",
    "paper",
    "article",
    "manuscript",
)
# Backward-compat aliases retained for tests/internal callers that reference
# backend module globals directly.
_OLLAMA_TOOL_SUPPORT_CACHE = _PROVIDER_OLLAMA_TOOL_SUPPORT_CACHE
_OLLAMA_FORCE_PLAIN_CACHE = _PROVIDER_OLLAMA_FORCE_PLAIN_CACHE
_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS = _PROVIDER_OLLAMA_PLAIN_MODE_COOLDOWN_TURNS


def _get_session_store() -> PersistentSessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = PersistentSessionStore(AgentSessionManager())
    return _SESSION_STORE


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
        self.workspace = get_agent_workspace_path()
        self.workspace_memory = AgentMemoryStore(self.workspace)
        runtime_cfg = resolve_agent_runtime_config(profile="playground")
        self.max_history_messages = int(runtime_cfg.max_history_messages)
        self._last_progress_update: str = ""

    def run(self) -> None:
        self._emit_progress("Analyzing request")
        _LOGGER.info(
            "annolid-bot turn start session=%s provider=%s model=%s prompt_chars=%d",
            self.session_id,
            self.provider,
            self.model,
            len(str(self.prompt or "")),
        )
        dep_error = self._provider_dependency_error()
        if dep_error:
            self._emit_progress("Provider dependency check failed")
            _LOGGER.warning(
                "annolid-bot dependency check failed session=%s provider=%s model=%s error=%s",
                self.session_id,
                self.provider,
                self.model,
                dep_error,
            )
            self._emit_final(dep_error, is_error=True)
            _LOGGER.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=dependency_missing",
                self.session_id,
                self.provider,
                self.model,
            )
            return
        try:
            self._emit_progress("Starting agent loop")
            self._run_agent_loop()
            _LOGGER.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=ok",
                self.session_id,
                self.provider,
                self.model,
            )
        except Exception as exc:
            if isinstance(exc, ImportError):
                message = self._format_dependency_error(str(exc))
                _LOGGER.warning(
                    "annolid-bot agent dependency missing session=%s provider=%s model=%s error=%s",
                    self.session_id,
                    self.provider,
                    self.model,
                    exc,
                )
                self._emit_final(message, is_error=True)
                _LOGGER.info(
                    "annolid-bot turn stop session=%s provider=%s model=%s status=dependency_missing",
                    self.session_id,
                    self.provider,
                    self.model,
                )
                return
            _LOGGER.warning(
                "annolid-bot agent loop failed; trying provider fallback session=%s provider=%s model=%s error=%s",
                self.session_id,
                self.provider,
                self.model,
                exc,
            )
            self._run_provider_fallback(exc)

    def _run_provider_fallback(self, original_error: Exception) -> None:
        """Run legacy provider fallback when agent loop setup/execution fails."""
        try:
            # Keep backward-compatible fallback behavior if agent loop setup fails.
            self._emit_progress("Agent loop failed, trying provider fallback")
            kind = provider_kind(self.settings, self.provider)
            if kind == "ollama":
                self._run_ollama()
            elif kind == "openai_compat":
                self._run_openai(provider_name=self.provider)
            elif kind == "gemini":
                self._run_gemini()
            else:
                raise ValueError(f"Unsupported provider '{self.provider}'.")
            _LOGGER.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=fallback_ok",
                self.session_id,
                self.provider,
                self.model,
            )
        except Exception as fallback_exc:
            if isinstance(fallback_exc, ImportError):
                message = self._format_dependency_error(str(fallback_exc))
                _LOGGER.warning(
                    "annolid-bot fallback dependency missing session=%s provider=%s model=%s error=%s",
                    self.session_id,
                    self.provider,
                    self.model,
                    fallback_exc,
                )
                self._emit_final(message, is_error=True)
                _LOGGER.info(
                    "annolid-bot turn stop session=%s provider=%s model=%s status=dependency_missing",
                    self.session_id,
                    self.provider,
                    self.model,
                )
                return
            _LOGGER.exception(
                "annolid-bot fallback failed session=%s provider=%s model=%s",
                self.session_id,
                self.provider,
                self.model,
            )
            self._emit_final(
                f"Error in chat interaction: {original_error}; fallback failed: {fallback_exc}",
                is_error=True,
            )

    def _provider_dependency_error(self) -> Optional[str]:
        kind = provider_kind(self.settings, self.provider)
        return dependency_error_for_kind(kind)

    def _format_dependency_error(self, raw_error: str) -> str:
        message = str(raw_error or "").strip()
        kind = provider_kind(self.settings, self.provider)
        if kind == "openai_compat" and "openai package is required" in message:
            return (
                "OpenAI-compatible provider requires the `openai` package. "
                "Install it in your Annolid environment, for example: "
                "`.venv/bin/pip install openai`."
            )
        if kind == "gemini" and "google-generativeai" in message:
            return (
                "Gemini provider requires `google-generativeai`. "
                "Install it in your Annolid environment, for example: "
                "`.venv/bin/pip install google-generativeai`."
            )
        return message or "Required provider dependency is missing."

    def _emit_chunk(self, chunk: str) -> None:
        QMetaObject.invokeMethod(
            self.widget,
            "stream_chat_chunk",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, chunk),
        )

    def _emit_progress(self, update: str) -> None:
        if not bool(getattr(self.widget, "enable_progress_stream", False)):
            return
        text = str(update or "").strip()
        if not text or text == self._last_progress_update:
            return
        self._last_progress_update = text
        QMetaObject.invokeMethod(
            self.widget,
            "stream_chat_progress",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    def _emit_final(self, message: str, *, is_error: bool) -> None:
        if is_error:
            self._emit_progress("Response failed")
        else:
            self._emit_progress("Response ready")
        QMetaObject.invokeMethod(
            self.widget,
            "update_chat_response",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, message),
            QtCore.Q_ARG(bool, is_error),
        )

    def _load_history_messages(self) -> List[Dict[str, Any]]:
        """Load persisted chat history as role/content records."""
        if not self.session_store:
            return []
        try:
            history = self.session_store.get_history(self.session_id)
        except Exception:
            return []
        cleaned: List[Dict[str, Any]] = []
        for msg in history:
            role = str(msg.get("role") or "")
            content = msg.get("content")
            if role not in {"user", "assistant", "system"}:
                continue
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text:
                continue
            cleaned.append({"role": role, "content": text})
        keep = max(1, int(self.max_history_messages))
        return cleaned[-keep:]

    def _persist_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        persist_session_history: bool = True,
    ) -> None:
        user_msg = str(user_text or "").strip()
        assistant_msg = str(assistant_text or "").strip()
        if not user_msg and not assistant_msg:
            return
        entries: List[Dict[str, str]] = []
        if user_msg:
            entries.append({"role": "user", "content": user_msg})
        if assistant_msg:
            entries.append({"role": "assistant", "content": assistant_msg})

        if persist_session_history and self.session_store and entries:
            try:
                self.session_store.append_history(
                    self.session_id,
                    entries,
                    max_messages=self.max_history_messages,
                )
            except Exception:
                pass

        try:
            stamp = datetime.now().strftime("%H:%M:%S")
            parts: List[str] = [f"## {stamp} [{self.session_id}]"]
            if user_msg:
                parts.append(f"User: {user_msg}")
            if assistant_msg:
                parts.append(f"Assistant: {assistant_msg}")
            entry = "\n\n".join(parts)
            self.workspace_memory.append_today(entry)
            self.workspace_memory.append_history(entry)
        except Exception:
            pass

    def _run_agent_loop(self) -> None:
        asyncio.run(self._run_agent_loop_async())

    async def _run_agent_loop_async(self) -> None:
        if self._try_execute_direct_gui_command():
            return

        context = await self._build_agent_execution_context()

        if self.provider == "ollama":
            remaining_plain_turns = int(ollama_plain_mode_remaining(self.model) or 0)
            wants_tools = self._prompt_may_need_tools(self.prompt)
            if remaining_plain_turns > 0 and not wants_tools:
                updated_remaining = ollama_plain_mode_decrement(self.model)
                _LOGGER.warning(
                    "annolid-bot model is in temporary plain mode; skipping agent/tool loop model=%s remaining_turns=%d",
                    self.model,
                    updated_remaining,
                )
                text = self._recover_with_plain_ollama_reply()
                if not text:
                    text = (
                        "Model returned empty output in plain mode. "
                        f"Provider={self.provider}, model={self.model}. "
                        "Please switch to another Ollama model for Annolid Bot."
                    )
                if self.show_tool_trace:
                    text = (
                        f"{text}\n\n[Tool Trace]\n"
                        "(skipped: temporary plain-mode fallback)"
                    ).strip()
                if text.strip():
                    self._persist_turn(self.prompt, text)
                self._emit_final(text, is_error=False)
                return
            if remaining_plain_turns > 0 and wants_tools:
                _LOGGER.info(
                    "annolid-bot bypassing temporary plain mode due tool-intent prompt model=%s remaining_turns=%d",
                    self.model,
                    remaining_plain_turns,
                )

        loop = AgentLoop(
            tools=context.tools,
            llm_callable=self._resolve_loop_llm_callable(),
            provider=self.provider,
            model=self.model,
            profile="playground",
            memory_store=self.session_store,
            workspace=str(context.workspace),
            allowed_read_roots=context.allowed_read_roots,
            mcp_servers=self.settings.get("tools", {}).get("mcp_servers", {}),
        )
        media: Optional[List[str]] = None
        if self.image_path and os.path.exists(self.image_path):
            media = [self.image_path]

        result = await loop.run(
            self.prompt,
            session_id=self.session_id,
            channel="gui",
            chat_id="annolid_bot",
            media=media,
            system_prompt=context.system_prompt,
        )
        self._emit_progress("Received model response")
        text, used_recovery, used_direct_gui_fallback = self._finalize_agent_text(
            result,
            tools=context.tools,
        )
        self._log_agent_result(result, used_recovery, used_direct_gui_fallback)
        if text.strip():
            self._persist_turn(self.prompt, text, persist_session_history=False)
        self._emit_final(text, is_error=False)

    def _try_execute_direct_gui_command(self) -> bool:
        direct_command_text = self._execute_direct_gui_command(self.prompt)
        if not direct_command_text:
            return False
        self._emit_progress("Executed direct GUI command")
        _LOGGER.info(
            "annolid-bot direct gui command handled session=%s model=%s",
            self.session_id,
            self.model,
        )
        self._persist_turn(self.prompt, direct_command_text)
        self._emit_final(direct_command_text, is_error=False)
        return True

    async def _build_agent_execution_context(self) -> _AgentExecutionContext:
        self._emit_progress("Loading tools and context")
        workspace = get_agent_workspace_path()
        agent_cfg = load_config()
        allowed_read_roots = list(
            getattr(agent_cfg.tools, "allowed_read_roots", []) or []
        )
        tools = FunctionToolRegistry()
        await register_nanobot_style_tools(
            tools,
            allowed_dir=workspace,
            allowed_read_roots=allowed_read_roots,
        )
        self._register_gui_tools(tools)
        disabled_tools = set(_GUI_ALWAYS_DISABLED_TOOLS)
        if not self.enable_web_tools:
            disabled_tools.update(_GUI_WEB_TOOLS)
        for tool_name in disabled_tools:
            tools.unregister(tool_name)
        resolved_policy = resolve_allowed_tools(
            all_tool_names=tools.tool_names,
            tools_cfg=agent_cfg.tools,
            provider=self.provider,
            model=self.model,
        )
        for tool_name in list(tools.tool_names):
            if tool_name not in resolved_policy.allowed_tools:
                tools.unregister(tool_name)
        system_prompt = self._build_compact_system_prompt(
            workspace, allowed_read_roots=allowed_read_roots
        )
        self._emit_progress("Prepared system prompt")
        _LOGGER.info(
            "annolid-bot agent config session=%s model=%s tools=%d read_roots=%d profile=%s policy_source=%s prompt_chars=%d",
            self.session_id,
            self.model,
            len(tools),
            len(allowed_read_roots),
            resolved_policy.profile,
            resolved_policy.source,
            len(system_prompt),
        )
        return _AgentExecutionContext(
            workspace=workspace,
            allowed_read_roots=allowed_read_roots,
            tools=tools,
            system_prompt=system_prompt,
        )

    def _register_gui_tools(self, tools: FunctionToolRegistry) -> None:
        register_annolid_gui_tools(
            tools,
            context_callback=self._build_gui_context_payload,
            image_path_callback=self._get_shared_image_path,
            open_video_callback=self._wrap_tool_callback(
                "open_video", self._tool_gui_open_video
            ),
            open_url_callback=self._wrap_tool_callback(
                "open_url", self._tool_gui_open_url
            ),
            open_in_browser_callback=self._wrap_tool_callback(
                "open_in_browser", self._tool_gui_open_in_browser
            ),
            web_get_dom_text_callback=self._wrap_tool_callback(
                "web_get_dom_text", self._tool_gui_web_get_dom_text
            ),
            web_click_callback=self._wrap_tool_callback(
                "web_click", self._tool_gui_web_click
            ),
            web_type_callback=self._wrap_tool_callback(
                "web_type", self._tool_gui_web_type
            ),
            web_scroll_callback=self._wrap_tool_callback(
                "web_scroll", self._tool_gui_web_scroll
            ),
            web_find_forms_callback=self._wrap_tool_callback(
                "web_find_forms", self._tool_gui_web_find_forms
            ),
            web_run_steps_callback=self._wrap_tool_callback(
                "web_run_steps", self._tool_gui_web_run_steps
            ),
            open_pdf_callback=self._wrap_tool_callback(
                "open_pdf", self._tool_gui_open_pdf
            ),
            pdf_get_state_callback=self._wrap_tool_callback(
                "pdf_get_state", self._tool_gui_pdf_get_state
            ),
            pdf_get_text_callback=self._wrap_tool_callback(
                "pdf_get_text", self._tool_gui_pdf_get_text
            ),
            pdf_find_sections_callback=self._wrap_tool_callback(
                "pdf_find_sections", self._tool_gui_pdf_find_sections
            ),
            set_frame_callback=self._wrap_tool_callback(
                "set_frame", self._tool_gui_set_frame
            ),
            set_prompt_callback=self._wrap_tool_callback(
                "set_prompt", self._tool_gui_set_chat_prompt
            ),
            send_prompt_callback=self._wrap_tool_callback(
                "send_prompt", self._tool_gui_send_chat_prompt
            ),
            set_chat_model_callback=self._wrap_tool_callback(
                "set_chat_model", self._tool_gui_set_chat_model
            ),
            select_annotation_model_callback=self._wrap_tool_callback(
                "select_annotation_model", self._tool_gui_select_annotation_model
            ),
            track_next_frames_callback=self._wrap_tool_callback(
                "track_next_frames", self._tool_gui_track_next_frames
            ),
            set_ai_text_prompt_callback=self._wrap_tool_callback(
                "set_ai_text_prompt", self._tool_gui_set_ai_text_prompt
            ),
            run_ai_text_segmentation_callback=self._wrap_tool_callback(
                "run_ai_text_segmentation", self._tool_gui_run_ai_text_segmentation
            ),
            segment_track_video_callback=self._wrap_tool_callback(
                "segment_track_video", self._tool_gui_segment_track_video
            ),
            label_behavior_segments_callback=self._wrap_tool_callback(
                "label_behavior_segments", self._tool_gui_label_behavior_segments
            ),
            start_realtime_stream_callback=self._wrap_tool_callback(
                "start_realtime_stream", self._tool_gui_start_realtime_stream
            ),
            stop_realtime_stream_callback=self._wrap_tool_callback(
                "stop_realtime_stream", self._tool_gui_stop_realtime_stream
            ),
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
        text = str(getattr(result, "content", "") or "").strip()
        tool_run_count = len(getattr(result, "tool_runs", ()) or ())
        used_recovery = False
        used_direct_gui_fallback = False
        direct_gui_text = ""
        if self.provider == "ollama" and tool_run_count == 0:
            direct_gui_text = self._maybe_run_direct_gui_tool_from_prompt(self.prompt)
            used_direct_gui_fallback = bool(direct_gui_text)
            if used_direct_gui_fallback:
                _LOGGER.info(
                    "annolid-bot direct gui fallback executed session=%s model=%s",
                    self.session_id,
                    self.model,
                )
                if not text or self._looks_like_local_access_refusal(text):
                    text = direct_gui_text
        if self.enable_web_tools:
            if self._looks_like_open_url_suggestion(text):
                open_page_fallback = self._try_open_page_content_fallback()
                if open_page_fallback:
                    text = open_page_fallback
            refusal_or_gap = self._looks_like_web_access_refusal(
                text
            ) or self._looks_like_knowledge_gap_response(text)
            if refusal_or_gap:
                open_page_fallback = self._try_open_page_content_fallback()
                if open_page_fallback:
                    text = open_page_fallback
                else:
                    browser_fallback = self._try_browser_search_fallback(
                        self.prompt, tools
                    )
                    if browser_fallback:
                        text = browser_fallback
                    else:
                        web_fallback = self._try_web_fetch_fallback(self.prompt, tools)
                        if web_fallback:
                            text = web_fallback
        if self._looks_like_local_access_refusal(
            text
        ) or self._looks_like_open_pdf_suggestion(text):
            open_pdf_fallback = self._try_open_pdf_content_fallback()
            if open_pdf_fallback:
                text = open_pdf_fallback
        # Final safety net: if the model still returns empty after our in-call retry,
        # attempt a single plain Ollama stream request (no tools) and use it.
        if not text and self.provider == "ollama":
            if used_direct_gui_fallback and direct_gui_text:
                text = direct_gui_text
            else:
                text = self._recover_with_plain_ollama_reply()
                used_recovery = bool(text)
            if used_recovery:
                ollama_mark_plain_mode(self.model)
        if not text:
            text = (
                "Model returned empty output after multiple attempts. "
                f"Provider={self.provider}, model={self.model}. "
                "Please switch to another Ollama model for Annolid Bot."
            )
        if self.show_tool_trace:
            trace = self._format_tool_trace(getattr(result, "tool_runs", ()) or ())
            text = f"{text}\n\n{trace}".strip()
        self._emit_progress("Finalizing response")
        return text, used_recovery, used_direct_gui_fallback

    def _log_agent_result(
        self,
        result: Any,
        used_recovery: bool,
        used_direct_gui_fallback: bool,
    ) -> None:
        _LOGGER.info(
            "annolid-bot agent result session=%s provider=%s model=%s iterations=%s tool_runs=%d",
            self.session_id,
            self.provider,
            self.model,
            getattr(result, "iterations", "?"),
            len(getattr(result, "tool_runs", ()) or ()),
        )
        if used_recovery:
            _LOGGER.info(
                "annolid-bot recovered empty agent reply with plain ollama answer session=%s model=%s",
                self.session_id,
                self.model,
            )
        if used_direct_gui_fallback:
            _LOGGER.info(
                "annolid-bot responded from direct gui fallback session=%s model=%s",
                self.session_id,
                self.model,
            )

    def _wrap_tool_callback(
        self, name: str, callback: Callable[..., Any]
    ) -> Callable[..., Any]:
        label = str(name or "tool").replace("_", " ")

        def _wrapped(*args, **kwargs):
            self._emit_progress(f"Running tool: {label}")
            result = callback(*args, **kwargs)
            if isinstance(result, dict):
                if bool(result.get("ok")):
                    self._emit_progress(f"Finished tool: {label}")
                else:
                    self._emit_progress(f"Tool failed: {label}")
            else:
                self._emit_progress(f"Finished tool: {label}")
            return result

        return _wrapped

    def _get_shared_image_path(self) -> str:
        return str(self.image_path or "")

    def _build_gui_context_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "prompt_chars": len(str(self.prompt or "")),
            "image_path": str(self.image_path or ""),
            "has_image": bool(self.image_path),
        }
        widget = self.widget
        if widget is not None:
            payload["attach_canvas"] = bool(
                getattr(
                    getattr(widget, "attach_canvas_checkbox", None),
                    "isChecked",
                    lambda: False,
                )()
            )
            payload["attach_window"] = bool(
                getattr(
                    getattr(widget, "attach_window_checkbox", None),
                    "isChecked",
                    lambda: False,
                )()
            )
            host = getattr(widget, "host_window_widget", None)
            if host is not None:
                for key in ("video_file", "filename", "frame_number"):
                    with_context = getattr(host, key, None)
                    if with_context not in (None, ""):
                        payload[key] = with_context
            web_state = self._tool_gui_web_get_state()
            if isinstance(web_state, dict):
                payload["web"] = web_state
            pdf_state = self._tool_gui_pdf_get_state()
            if isinstance(pdf_state, dict):
                payload["pdf"] = pdf_state
        return payload

    def _invoke_widget_slot(self, slot_name: str, *qargs: Any) -> bool:
        widget = self.widget
        if widget is None:
            return False
        try:
            invoked = QMetaObject.invokeMethod(
                widget,
                slot_name,
                QtCore.Qt.BlockingQueuedConnection,
                *qargs,
            )
            # Depending on Qt binding/runtime, invokeMethod may return either
            # bool or None on success. Treat non-exception as success.
            if isinstance(invoked, bool):
                return invoked
            return True
        except Exception as exc:
            _LOGGER.warning(
                "annolid-bot gui slot invoke failed session=%s slot=%s error=%s",
                self.session_id,
                slot_name,
                exc,
            )
            return False

    def _invoke_widget_json_slot(self, slot_name: str, *qargs: Any) -> Dict[str, Any]:
        widget = self.widget
        if widget is not None:
            try:
                setattr(widget, "_bot_action_result", {})
            except Exception:
                pass
        ok = self._invoke_widget_slot(slot_name, *qargs)
        if not ok:
            return {"ok": False, "error": f"Failed to run GUI action: {slot_name}"}
        if widget is not None:
            payload = getattr(widget, "_bot_action_result", None)
            if isinstance(payload, dict) and payload:
                return dict(payload)
        return {"ok": True}

    def _tool_gui_open_video(self, path: str) -> Dict[str, Any]:
        video_path = self._resolve_video_path_for_gui_tool(path)
        if video_path is None:
            raw_text = str(path or "").strip()
            basename = Path(raw_text).name if raw_text else ""
            return {
                "ok": False,
                "error": "Video not found from provided path/text.",
                "input": raw_text,
                "basename": basename,
                "hint": (
                    "Provide an absolute path, or a filename located in workspace/read-roots."
                ),
            }
        ok = self._invoke_widget_slot(
            "bot_open_video", QtCore.Q_ARG(str, str(video_path))
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue GUI video open action"}
        return {"ok": True, "queued": True, "path": str(video_path)}

    def _tool_gui_open_url(self, url: str) -> Dict[str, Any]:
        raw_text = str(url or "").strip()
        target_url = self._extract_first_web_url(raw_text)
        if not target_url:
            candidate = raw_text
            lowered = candidate.lower()
            for prefix in ("open ", "load ", "show ", "open url ", "open file "):
                if lowered.startswith(prefix):
                    candidate = candidate[len(prefix) :].strip()
                    lowered = candidate.lower()
                    break
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists() and candidate_path.is_file():
                target_url = str(candidate_path)
        if not target_url:
            return {
                "ok": False,
                "error": "URL or local file path not found in provided text.",
                "input": raw_text,
                "hint": (
                    "Provide a URL (e.g. google.com) or an existing local file path "
                    "(e.g. /path/to/file.html)."
                ),
            }
        lower_target = target_url.lower()
        if not (
            lower_target.startswith(("http://", "https://", "file://"))
            or Path(target_url).expanduser().is_file()
        ):
            return {
                "ok": False,
                "error": "Only http(s) URLs or existing local files are supported.",
                "url": target_url,
            }
        ok = self._invoke_widget_slot("bot_open_url", QtCore.Q_ARG(str, target_url))
        if not ok:
            return {"ok": False, "error": "Failed to queue GUI URL open action"}
        return {"ok": True, "queued": True, "url": target_url}

    def _tool_gui_open_in_browser(self, url: str) -> Dict[str, Any]:
        target_url = self._extract_first_web_url(url)
        if not target_url:
            return {
                "ok": False,
                "error": "URL not found in provided text.",
                "input": str(url or "").strip(),
                "hint": "Provide a URL, for example google.com or https://example.org.",
            }
        ok = self._invoke_widget_slot(
            "bot_open_in_browser", QtCore.Q_ARG(str, target_url)
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue browser open action"}
        return {"ok": True, "queued": True, "url": target_url}

    def _tool_gui_web_get_dom_text(self, max_chars: int = 8000) -> Dict[str, Any]:
        limit = max(200, min(int(max_chars or 8000), 200000))
        payload = self._invoke_widget_json_slot(
            "bot_web_get_dom_text", QtCore.Q_ARG(int, limit)
        )
        if "max_chars" not in payload:
            payload["max_chars"] = limit
        return payload

    def _tool_gui_web_get_state(self) -> Dict[str, Any]:
        return self._invoke_widget_json_slot("bot_web_get_state")

    def _tool_gui_web_click(self, selector: str) -> Dict[str, Any]:
        value = str(selector or "").strip()
        if not value:
            return {"ok": False, "error": "selector is required"}
        payload = self._invoke_widget_json_slot(
            "bot_web_click", QtCore.Q_ARG(str, value)
        )
        if "selector" not in payload:
            payload["selector"] = value
        return payload

    def _tool_gui_web_type(
        self, selector: str, text: str, submit: bool = False
    ) -> Dict[str, Any]:
        selector_text = str(selector or "").strip()
        if not selector_text:
            return {"ok": False, "error": "selector is required"}
        payload = self._invoke_widget_json_slot(
            "bot_web_type",
            QtCore.Q_ARG(str, selector_text),
            QtCore.Q_ARG(str, str(text or "")),
            QtCore.Q_ARG(bool, bool(submit)),
        )
        if "selector" not in payload:
            payload["selector"] = selector_text
        return payload

    def _tool_gui_web_scroll(self, delta_y: int = 800) -> Dict[str, Any]:
        value = int(delta_y or 0)
        payload = self._invoke_widget_json_slot(
            "bot_web_scroll", QtCore.Q_ARG(int, value)
        )
        if "delta_y" not in payload and "deltaY" not in payload:
            payload["delta_y"] = value
        return payload

    def _tool_gui_web_find_forms(self) -> Dict[str, Any]:
        return self._invoke_widget_json_slot("bot_web_find_forms")

    def _tool_gui_web_run_steps(
        self,
        steps: Any,
        stop_on_error: bool = True,
        max_steps: int = 12,
    ) -> Dict[str, Any]:
        if not isinstance(steps, list) or not steps:
            return {"ok": False, "error": "steps must be a non-empty list"}
        limit = max(1, min(int(max_steps or 12), 50))
        if len(steps) > limit:
            return {
                "ok": False,
                "error": f"Too many steps ({len(steps)}), max_steps={limit}",
            }

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        halt_on_error = bool(stop_on_error)

        for idx, raw_step in enumerate(steps):
            if not isinstance(raw_step, dict):
                payload = {"ok": False, "error": "step must be an object"}
                results.append({"index": idx, "action": "", "result": payload})
                errors.append({"index": idx, "error": payload["error"]})
                if halt_on_error:
                    break
                continue

            action = str(raw_step.get("action") or "").strip().lower()
            if not action:
                payload = {"ok": False, "error": "step.action is required"}
                results.append({"index": idx, "action": action, "result": payload})
                errors.append({"index": idx, "error": payload["error"]})
                if halt_on_error:
                    break
                continue

            if action == "open_url":
                payload = self._tool_gui_open_url(str(raw_step.get("url") or ""))
            elif action == "open_in_browser":
                payload = self._tool_gui_open_in_browser(str(raw_step.get("url") or ""))
            elif action in {"get_text", "dom_text", "snapshot"}:
                payload = self._tool_gui_web_get_dom_text(
                    int(raw_step.get("max_chars") or 8000)
                )
            elif action == "click":
                payload = self._tool_gui_web_click(str(raw_step.get("selector") or ""))
            elif action == "type":
                payload = self._tool_gui_web_type(
                    str(raw_step.get("selector") or ""),
                    str(raw_step.get("text") or ""),
                    submit=bool(raw_step.get("submit", False)),
                )
            elif action == "scroll":
                payload = self._tool_gui_web_scroll(int(raw_step.get("delta_y") or 800))
            elif action == "find_forms":
                payload = self._tool_gui_web_find_forms()
            elif action == "wait":
                wait_ms = max(0, min(int(raw_step.get("wait_ms") or 500), 60000))
                QtCore.QThread.msleep(wait_ms)
                payload = {"ok": True, "wait_ms": wait_ms}
            else:
                payload = {"ok": False, "error": f"Unsupported action: {action}"}

            results.append({"index": idx, "action": action, "result": payload})
            if not bool(payload.get("ok")):
                errors.append(
                    {
                        "index": idx,
                        "action": action,
                        "error": str(payload.get("error") or "step failed"),
                    }
                )
                if halt_on_error:
                    break

        return {
            "ok": len(errors) == 0,
            "steps_requested": len(steps),
            "steps_run": len(results),
            "stop_on_error": halt_on_error,
            "results": results,
            "errors": errors,
        }

    @staticmethod
    def _extract_first_web_url(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        candidates = StreamingChatTask._extract_web_urls(raw)
        if raw.lower().startswith(("http://", "https://")) and raw not in candidates:
            candidates.insert(0, raw.rstrip(").,;!?"))
        if not candidates:
            domain_match = re.search(
                r"\b(?:www\.)?[a-z0-9][a-z0-9\-]{0,62}"
                r"(?:\.[a-z0-9][a-z0-9\-]{0,62})+(?::\d+)?(?:/[^\s<>\"]*)?",
                raw,
                flags=re.IGNORECASE,
            )
            if domain_match:
                domain_url = str(domain_match.group(0) or "").strip().rstrip(").,;!?")
                if domain_url:
                    return f"https://{domain_url}"
            return ""
        return str(candidates[0] or "").strip()

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

    def _tool_gui_open_pdf(self, path: str = "") -> Dict[str, Any]:
        path_text = str(path or "").strip()
        path_candidates = (
            self._extract_pdf_path_candidates(path_text) if path_text else []
        )
        generic_url_candidates = self._extract_web_urls(path_text) if path_text else []
        has_explicit_pdf_path = bool(path_candidates)
        resolved_path: Optional[Path] = None
        if has_explicit_pdf_path or generic_url_candidates:
            url_candidate = next(
                (
                    candidate
                    for candidate in path_candidates
                    if str(candidate).lower().startswith(("http://", "https://"))
                ),
                "",
            )
            if not url_candidate and generic_url_candidates:
                url_candidate = str(generic_url_candidates[0] or "").strip()
            if url_candidate:
                resolved_path = self._download_pdf_for_gui_tool(url_candidate)
            if resolved_path is None:
                resolved_path = self._resolve_pdf_path_for_gui_tool(path_text)
        if (has_explicit_pdf_path or generic_url_candidates) and resolved_path is None:
            return {
                "ok": False,
                "error": "PDF not found or URL did not resolve to a PDF.",
                "input": path_text,
                "hint": (
                    "Provide an absolute/local PDF path, or a URL that serves application/pdf."
                ),
            }
        if resolved_path is None:
            available = self._list_available_pdfs(limit=8)
            if not available:
                return {
                    "ok": False,
                    "error": (
                        "No PDF files found in workspace/read-roots. "
                        "Download a PDF first or provide a path."
                    ),
                }
            if len(available) > 1:
                choices = [str(path) for path in available]
                return {
                    "ok": False,
                    "error": (
                        "Multiple PDFs are available. Please specify which PDF to open."
                    ),
                    "choices": choices,
                }
            resolved_path = available[0]

        ok = self._invoke_widget_slot(
            "bot_open_pdf", QtCore.Q_ARG(str, str(resolved_path))
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue GUI PDF open action"}
        return {"ok": True, "queued": True, "path": str(resolved_path)}

    def _tool_gui_pdf_get_state(self) -> Dict[str, Any]:
        return self._invoke_widget_json_slot("bot_pdf_get_state")

    def _tool_gui_pdf_get_text(
        self, max_chars: int = 8000, pages: int = 2
    ) -> Dict[str, Any]:
        limit = max(200, min(int(max_chars or 8000), 200000))
        pages_limit = max(1, min(int(pages or 2), 5))
        payload = self._invoke_widget_json_slot(
            "bot_pdf_get_text",
            QtCore.Q_ARG(int, limit),
            QtCore.Q_ARG(int, pages_limit),
        )
        if "max_chars" not in payload:
            payload["max_chars"] = limit
        if "pages" not in payload:
            payload["pages"] = pages_limit
        return payload

    def _tool_gui_pdf_find_sections(
        self,
        max_sections: int = 20,
        max_pages: int = 12,
    ) -> Dict[str, Any]:
        sections_limit = max(1, min(int(max_sections or 20), 200))
        pages_limit = max(1, min(int(max_pages or 12), 100))
        payload = self._invoke_widget_json_slot(
            "bot_pdf_find_sections",
            QtCore.Q_ARG(int, sections_limit),
            QtCore.Q_ARG(int, pages_limit),
        )
        if "max_sections" not in payload:
            payload["max_sections"] = sections_limit
        if "max_pages" not in payload:
            payload["max_pages"] = pages_limit
        return payload

    def _download_pdf_for_gui_tool(self, url: str) -> Optional[Path]:
        text = str(url or "").strip()
        if not text.lower().startswith(("http://", "https://")):
            return None
        workspace = get_agent_workspace_path()
        downloader = DownloadPdfTool(allowed_dir=workspace)
        try:
            payload_raw = self._run_async(downloader.execute(url=text))
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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()

    @staticmethod
    def _extract_path_candidates(raw: str) -> List[str]:
        return extract_video_path_candidates(raw)

    def _resolve_video_path_for_gui_tool(self, raw_path: str) -> Optional[Path]:
        try:
            cfg = load_config()
            read_roots_cfg = list(getattr(cfg.tools, "allowed_read_roots", []) or [])
        except Exception:
            read_roots_cfg = []
        roots = build_workspace_roots(get_agent_workspace_path(), read_roots_cfg)
        active_video_raw: str = ""
        try:
            widget = self.widget
            host = getattr(widget, "host_window_widget", None) if widget else None
            if host is None and widget is not None:
                host_getter = getattr(widget, "window", None)
                if callable(host_getter):
                    host = host_getter()
            active_video_raw = str(getattr(host, "video_file", "") or "").strip()
            if active_video_raw:
                candidate_video = Path(active_video_raw).expanduser()
                if candidate_video.exists():
                    roots.append(candidate_video.parent)
        except Exception:
            active_video_raw = ""
        return resolve_video_path_for_roots(
            raw_path,
            roots,
            active_video_raw=active_video_raw,
            max_scan=30000,
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
        return self._execute_direct_gui_command(prompt)

    def _execute_direct_gui_command(self, prompt: str) -> str:
        command = self._parse_direct_gui_command(prompt)
        return execute_direct_gui_command(
            command,
            open_video=self._tool_gui_open_video,
            open_url=self._tool_gui_open_url,
            open_in_browser=self._tool_gui_open_in_browser,
            open_pdf=self._tool_gui_open_pdf,
            set_frame=self._tool_gui_set_frame,
            track_next_frames=self._tool_gui_track_next_frames,
            segment_track_video=self._tool_gui_segment_track_video,
            label_behavior_segments=self._tool_gui_label_behavior_segments,
            start_realtime_stream=self._tool_gui_start_realtime_stream,
            stop_realtime_stream=self._tool_gui_stop_realtime_stream,
            set_chat_model=self._tool_gui_set_chat_model,
            rename_file=self._tool_gui_rename_file,
        )

    def _parse_direct_gui_command(self, prompt: str) -> Dict[str, Any]:
        return parse_direct_gui_command(prompt)

    @staticmethod
    def _looks_like_local_access_refusal(text: str) -> bool:
        return looks_like_local_access_refusal(text)

    @staticmethod
    def _looks_like_web_access_refusal(text: str) -> bool:
        value = str(text or "").lower()
        if not value:
            return False
        return any(hint in value for hint in _WEB_ACCESS_REFUSAL_HINTS)

    @staticmethod
    def _looks_like_knowledge_gap_response(text: str) -> bool:
        value = str(text or "").lower()
        if not value:
            return False
        return any(hint in value for hint in _KNOWLEDGE_GAP_HINTS)

    @staticmethod
    def _looks_like_open_url_suggestion(text: str) -> bool:
        value = str(text or "").lower()
        if not value:
            return False
        return any(hint in value for hint in _OPEN_URL_SUGGESTION_HINTS)

    @staticmethod
    def _looks_like_open_pdf_suggestion(text: str) -> bool:
        value = str(text or "").lower()
        if not value:
            return False
        return any(hint in value for hint in _OPEN_PDF_SUGGESTION_HINTS)

    @staticmethod
    def _extract_web_urls(text: str) -> List[str]:
        raw = str(text or "")
        if not raw:
            return []
        candidates = re.findall(r"https?://[^\s<>\"]+", raw, flags=re.IGNORECASE)
        urls: List[str] = []
        for item in candidates:
            cleaned = str(item or "").strip().rstrip(").,;!?")
            if cleaned and cleaned not in urls:
                urls.append(cleaned)
        return urls

    def _candidate_web_urls_for_prompt(self, prompt: str) -> List[str]:
        urls = self._extract_web_urls(prompt)
        if urls:
            return urls
        history = self._load_history_messages()
        # Prefer the most recent user-provided URL when the current turn uses
        # references like "that page" without repeating the link.
        for msg in reversed(history):
            if str(msg.get("role") or "") != "user":
                continue
            content = str(msg.get("content") or "")
            if not content:
                continue
            from_msg = self._extract_web_urls(content)
            if from_msg:
                return from_msg
        return []

    @staticmethod
    def _build_extractive_summary(
        text: str,
        *,
        max_sentences: int = 6,
        max_chars: int = 1200,
    ) -> str:
        source = " ".join(str(text or "").split()).strip()
        if not source:
            return ""
        chunks = re.split(r"(?<=[.!?])\s+", source)
        picked: List[str] = []
        total = 0
        for chunk in chunks:
            sentence = str(chunk or "").strip()
            if not sentence:
                continue
            if picked and total + 1 + len(sentence) > max_chars:
                break
            if not picked and len(sentence) > max_chars:
                picked.append(sentence[: max_chars - 3].rstrip() + "...")
                break
            picked.append(sentence)
            total += len(sentence) + 1
            if len(picked) >= max_sentences:
                break
        return " ".join(picked).strip()

    def _try_web_fetch_fallback(
        self,
        prompt: str,
        tools: Optional[FunctionToolRegistry],
    ) -> str:
        registry = tools
        if registry is None:
            return ""
        if not registry.has("web_fetch"):
            return ""
        urls = self._candidate_web_urls_for_prompt(prompt)
        if not urls:
            return ""
        target_url = urls[0]
        try:
            self._emit_progress("Retrying with web_fetch")
            payload_raw = self._run_async(
                registry.execute(
                    "web_fetch",
                    {"url": target_url, "extractMode": "text", "maxChars": 12000},
                )
            )
        except Exception:
            return ""
        try:
            payload = json.loads(str(payload_raw or "{}"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return ""
        if payload.get("error"):
            return ""
        page_text = str(payload.get("text") or "").strip()
        if not page_text:
            return ""
        summary = self._build_extractive_summary(page_text)
        if not summary:
            return ""
        source_url = str(payload.get("finalUrl") or target_url).strip() or target_url
        return (
            f"Summary of {source_url}:\n{summary}\n\n"
            f"Source: {source_url}\n"
            "(Generated via web_fetch fallback after a browsing-capability refusal.)"
        )

    @staticmethod
    def _extract_page_text_from_web_steps(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        for item in payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("action") or "").lower() not in {
                "get_text",
                "dom_text",
                "snapshot",
            }:
                continue
            result_payload = item.get("result")
            if not isinstance(result_payload, dict):
                continue
            text_value = str(result_payload.get("text") or "").strip()
            if text_value:
                return text_value
        return ""

    def _try_browser_search_fallback(
        self,
        prompt: str,
        tools: Optional[FunctionToolRegistry],
    ) -> str:
        registry = tools
        if registry is None:
            return ""
        if not registry.has("gui_web_run_steps"):
            return ""
        query = " ".join(str(prompt or "").split()).strip()
        if not query:
            return ""
        if len(query) > 280:
            query = query[:280].rstrip()
        encoded_query = quote_plus(query)
        steps = [
            {
                "action": "open_url",
                "url": _EMBEDDED_SEARCH_URL_TEMPLATE.format(query=encoded_query),
            },
            {"action": "wait", "wait_ms": 1200},
            {"action": "get_text", "max_chars": 9000},
        ]
        try:
            self._emit_progress("Retrying with browser search workflow")
            payload_raw = self._run_async(
                registry.execute(
                    "gui_web_run_steps",
                    {"steps": steps, "stop_on_error": True, "max_steps": 12},
                )
            )
        except Exception:
            return ""

        try:
            payload = json.loads(str(payload_raw or "{}"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict) or payload.get("error"):
            return ""
        if not bool(payload.get("ok")):
            return ""
        page_text = self._extract_page_text_from_web_steps(payload)
        if not page_text:
            return ""
        summary = self._build_extractive_summary(
            page_text, max_sentences=8, max_chars=1400
        )
        if not summary:
            return ""
        return (
            f"Web lookup via embedded browser:\n{summary}\n\n"
            f"Source: {_EMBEDDED_SEARCH_SOURCE}"
        )

    def _try_open_page_content_fallback(self) -> str:
        state = self._tool_gui_web_get_state()
        if not isinstance(state, dict):
            return ""
        if not bool(state.get("ok")) or not bool(state.get("has_page")):
            return ""
        if not self._should_use_open_page_fallback(self.prompt):
            prompt_tokens = set(self._topic_tokens(self.prompt))
            page_hint_text = " ".join(
                [
                    str(state.get("title") or ""),
                    str(state.get("url") or ""),
                ]
            )
            page_tokens = set(self._topic_tokens(page_hint_text))
            if not (prompt_tokens and page_tokens and (prompt_tokens & page_tokens)):
                return ""
        page_payload = self._tool_gui_web_get_dom_text(max_chars=9000)
        if not isinstance(page_payload, dict) or not bool(page_payload.get("ok")):
            return ""
        page_text = str(page_payload.get("text") or "").strip()
        if not page_text:
            return ""
        summary = self._build_extractive_summary(
            page_text, max_sentences=8, max_chars=1400
        )
        if not summary:
            return ""
        url = str(page_payload.get("url") or state.get("url") or "").strip()
        title = str(page_payload.get("title") or state.get("title") or "").strip()
        source = title or url or "active embedded web page"
        return f"Using the currently open page ({source}):\n{summary}"

    def _try_open_pdf_content_fallback(self) -> str:
        state = self._tool_gui_pdf_get_state()
        if not isinstance(state, dict):
            return ""
        if not bool(state.get("ok")) or not bool(state.get("has_pdf")):
            return ""
        pdf_payload = self._tool_gui_pdf_get_text(max_chars=9000, pages=2)
        if not isinstance(pdf_payload, dict) or not bool(pdf_payload.get("ok")):
            return ""
        pdf_text = str(pdf_payload.get("text") or "").strip()
        if not pdf_text:
            return ""
        summary = self._build_extractive_summary(
            pdf_text, max_sentences=8, max_chars=1400
        )
        if not summary:
            return ""
        title = str(pdf_payload.get("title") or state.get("title") or "").strip()
        path = str(pdf_payload.get("path") or state.get("path") or "").strip()
        source = title or path or "active PDF"
        return f"Using the currently open PDF ({source}):\n{summary}"

    def _tool_gui_set_frame(self, frame_index: int) -> Dict[str, Any]:
        target_frame = int(frame_index)
        if target_frame < 0:
            return {"ok": False, "error": "frame_index must be >= 0"}
        ok = self._invoke_widget_slot("bot_set_frame", QtCore.Q_ARG(int, target_frame))
        if not ok:
            return {"ok": False, "error": "Failed to queue frame action"}
        return {"ok": True, "queued": True, "frame_index": target_frame}

    def _tool_gui_set_chat_prompt(self, text: str) -> Dict[str, Any]:
        prompt_text = str(text or "").strip()
        if not prompt_text:
            return {"ok": False, "error": "text is required"}
        ok = self._invoke_widget_slot(
            "bot_set_chat_prompt", QtCore.Q_ARG(str, prompt_text)
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue prompt update"}
        return {"ok": True, "queued": True, "chars": len(prompt_text)}

    def _tool_gui_send_chat_prompt(self) -> Dict[str, Any]:
        ok = self._invoke_widget_slot("bot_send_chat_prompt")
        if not ok:
            return {"ok": False, "error": "Failed to queue chat send action"}
        return {"ok": True, "queued": True}

    def _tool_gui_set_chat_model(self, provider: str, model: str) -> Dict[str, Any]:
        provider_text = str(provider or "").strip().lower()
        model_text = str(model or "").strip()
        if not provider_text:
            return {"ok": False, "error": "provider is required"}
        if not model_text:
            return {"ok": False, "error": "model is required"}
        ok = self._invoke_widget_slot(
            "bot_set_chat_model",
            QtCore.Q_ARG(str, provider_text),
            QtCore.Q_ARG(str, model_text),
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue provider/model update"}
        return {
            "ok": True,
            "queued": True,
            "provider": provider_text,
            "model": model_text,
        }

    def _tool_gui_select_annotation_model(self, model_name: str) -> Dict[str, Any]:
        model_text = str(model_name or "").strip()
        if not model_text:
            return {"ok": False, "error": "model_name is required"}
        ok = self._invoke_widget_slot(
            "bot_select_annotation_model", QtCore.Q_ARG(str, model_text)
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue model selection"}
        return {"ok": True, "queued": True, "model_name": model_text}

    def _tool_gui_track_next_frames(self, to_frame: int) -> Dict[str, Any]:
        frame = int(to_frame)
        if frame < 1:
            return {"ok": False, "error": "to_frame must be >= 1"}
        ok = self._invoke_widget_slot("bot_track_next_frames", QtCore.Q_ARG(int, frame))
        if not ok:
            return {"ok": False, "error": "Failed to queue tracking action"}
        return {"ok": True, "queued": True, "to_frame": frame}

    def _tool_gui_set_ai_text_prompt(
        self, text: str, use_countgd: bool = False
    ) -> Dict[str, Any]:
        prompt_text = str(text or "").strip()
        if not prompt_text:
            return {"ok": False, "error": "text is required"}
        ok = self._invoke_widget_slot(
            "bot_set_ai_text_prompt",
            QtCore.Q_ARG(str, prompt_text),
            QtCore.Q_ARG(bool, bool(use_countgd)),
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue AI prompt update"}
        return {
            "ok": True,
            "queued": True,
            "text": prompt_text,
            "use_countgd": bool(use_countgd),
        }

    def _tool_gui_run_ai_text_segmentation(self) -> Dict[str, Any]:
        ok = self._invoke_widget_slot("bot_run_ai_text_segmentation")
        if not ok:
            return {"ok": False, "error": "Failed to queue AI text segmentation action"}
        return {"ok": True, "queued": True}

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
        video_path = self._resolve_video_path_for_gui_tool(path)
        if video_path is None:
            return {
                "ok": False,
                "error": "Video not found from provided path/text.",
                "input": str(path or "").strip(),
            }
        prompt_text = str(text_prompt or "").strip()
        if not prompt_text:
            return {"ok": False, "error": "text_prompt is required"}
        mode_norm = str(mode or "track").strip().lower()
        if mode_norm not in {"segment", "track"}:
            return {"ok": False, "error": "mode must be 'segment' or 'track'"}
        target_frame = -1 if to_frame is None else int(to_frame)
        if target_frame != -1 and target_frame < 1:
            return {"ok": False, "error": "to_frame must be >= 1"}

        resolved_model = str(model_name or "").strip()
        if mode_norm == "track" and not resolved_model:
            resolved_model = "Cutie"

        ok = self._invoke_widget_slot(
            "bot_segment_track_video",
            QtCore.Q_ARG(str, str(video_path)),
            QtCore.Q_ARG(str, prompt_text),
            QtCore.Q_ARG(str, mode_norm),
            QtCore.Q_ARG(bool, bool(use_countgd)),
            QtCore.Q_ARG(str, resolved_model),
            QtCore.Q_ARG(int, target_frame),
        )
        if not ok:
            return {
                "ok": False,
                "error": "Failed to queue segment/track workflow action",
            }
        widget_result: Dict[str, Any] = {}
        try:
            widget = self.widget
            getter = getattr(widget, "get_bot_action_result", None) if widget else None
            if callable(getter):
                payload = getter("segment_track_video")
                if isinstance(payload, dict):
                    widget_result = payload
        except Exception:
            widget_result = {}

        if widget_result:
            if not bool(widget_result.get("ok", False)):
                return {
                    "ok": False,
                    "error": str(
                        widget_result.get("error")
                        or "Segment/track workflow failed in GUI."
                    ),
                    "path": str(video_path),
                    "basename": Path(video_path).name,
                    "text_prompt": prompt_text,
                    "mode": mode_norm,
                }
            return {
                "ok": True,
                "path": str(video_path),
                "basename": Path(video_path).name,
                "text_prompt": prompt_text,
                "mode": str(widget_result.get("mode") or mode_norm),
                "use_countgd": bool(use_countgd),
                "model_name": str(widget_result.get("model_name") or resolved_model),
                "to_frame": (
                    widget_result.get("to_frame")
                    if widget_result.get("to_frame") is not None
                    else (None if target_frame == -1 else target_frame)
                ),
            }
        return {
            "ok": True,
            "queued": True,
            "path": str(video_path),
            "basename": Path(video_path).name,
            "text_prompt": prompt_text,
            "mode": mode_norm,
            "use_countgd": bool(use_countgd),
            "model_name": resolved_model,
            "to_frame": None if target_frame == -1 else target_frame,
        }

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
        resolved_path = None
        if str(path or "").strip():
            resolved_path = self._resolve_video_path_for_gui_tool(str(path))
            if resolved_path is None:
                return {
                    "ok": False,
                    "error": "Video not found from provided path/text.",
                    "input": str(path or "").strip(),
                }

        labels: list[str] = []
        if isinstance(behavior_labels, list):
            labels = [str(v).strip() for v in behavior_labels if str(v).strip()]
        elif isinstance(behavior_labels, str):
            labels = [p.strip() for p in behavior_labels.split(",") if p.strip()]

        mode_norm = str(segment_mode or "timeline").strip().lower()
        if mode_norm not in {"timeline", "uniform"}:
            return {
                "ok": False,
                "error": "segment_mode must be 'timeline' or 'uniform'",
            }
        frames = max(1, int(segment_frames))
        max_seg = max(1, int(max_segments))
        ok = self._invoke_widget_slot(
            "bot_label_behavior_segments",
            QtCore.Q_ARG(str, str(resolved_path) if resolved_path else ""),
            QtCore.Q_ARG(str, ",".join(labels)),
            QtCore.Q_ARG(str, mode_norm),
            QtCore.Q_ARG(int, frames),
            QtCore.Q_ARG(int, max_seg),
            QtCore.Q_ARG(str, str(subject or "Agent")),
            QtCore.Q_ARG(bool, bool(overwrite_existing)),
            QtCore.Q_ARG(str, str(llm_profile or "")),
            QtCore.Q_ARG(str, str(llm_provider or "")),
            QtCore.Q_ARG(str, str(llm_model or "")),
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue behavior labeling action"}

        widget_result: Dict[str, Any] = {}
        try:
            widget = self.widget
            getter = getattr(widget, "get_bot_action_result", None) if widget else None
            if callable(getter):
                payload = getter("label_behavior_segments")
                if isinstance(payload, dict):
                    widget_result = payload
        except Exception:
            widget_result = {}
        if widget_result:
            if not bool(widget_result.get("ok", False)):
                return {
                    "ok": False,
                    "error": str(
                        widget_result.get("error")
                        or "Behavior segment labeling failed in GUI."
                    ),
                }
            return {
                "ok": True,
                "mode": str(widget_result.get("mode") or mode_norm),
                "labeled_segments": int(widget_result.get("labeled_segments") or 0),
                "evaluated_segments": int(widget_result.get("evaluated_segments") or 0),
                "skipped_segments": int(widget_result.get("skipped_segments") or 0),
                "labels_used": list(widget_result.get("labels_used") or labels),
                "timestamps_csv": str(widget_result.get("timestamps_csv") or ""),
                "timestamps_rows": int(widget_result.get("timestamps_rows") or 0),
            }
        return {"ok": True, "queued": True, "mode": mode_norm}

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
        model_text = str(model_name or "").strip()
        camera_text = str(camera_source or "").strip()
        viewer = str(viewer_type or "threejs").strip().lower()
        if viewer not in {"pyqt", "threejs"}:
            viewer = "threejs"

        targets: list[str] = []
        if isinstance(target_behaviors, list):
            targets = [str(v).strip() for v in target_behaviors if str(v).strip()]
        elif isinstance(target_behaviors, str):
            targets = [p.strip() for p in target_behaviors.split(",") if p.strip()]

        threshold = None
        if confidence_threshold is not None:
            try:
                threshold = float(confidence_threshold)
            except Exception:
                return {
                    "ok": False,
                    "error": "confidence_threshold must be a float in [0, 1].",
                }
            threshold = max(0.0, min(1.0, threshold))

        ear_threshold = None
        if blink_ear_threshold is not None:
            try:
                ear_threshold = float(blink_ear_threshold)
            except Exception:
                return {"ok": False, "error": "blink_ear_threshold must be a float."}
            ear_threshold = max(0.05, min(0.6, ear_threshold))

        min_blink_frames = None
        if blink_min_consecutive_frames is not None:
            try:
                min_blink_frames = int(blink_min_consecutive_frames)
            except Exception:
                return {
                    "ok": False,
                    "error": "blink_min_consecutive_frames must be an integer.",
                }
            min_blink_frames = max(1, min(30, min_blink_frames))

        ok = self._invoke_widget_slot(
            "bot_start_realtime_stream",
            QtCore.Q_ARG(str, camera_text),
            QtCore.Q_ARG(str, model_text),
            QtCore.Q_ARG(str, ",".join(targets)),
            QtCore.Q_ARG(float, threshold if threshold is not None else -1.0),
            QtCore.Q_ARG(str, viewer),
            QtCore.Q_ARG(bool, bool(classify_eye_blinks)),
            QtCore.Q_ARG(float, ear_threshold if ear_threshold is not None else -1.0),
            QtCore.Q_ARG(int, min_blink_frames if min_blink_frames is not None else -1),
        )
        if not ok:
            return {"ok": False, "error": "Failed to queue realtime start action"}
        widget_result: Dict[str, Any] = {}
        try:
            widget = self.widget
            getter = getattr(widget, "get_bot_action_result", None) if widget else None
            if callable(getter):
                payload = getter("start_realtime_stream")
                if isinstance(payload, dict):
                    widget_result = payload
        except Exception:
            widget_result = {}
        if widget_result:
            if not bool(widget_result.get("ok", False)):
                return {
                    "ok": False,
                    "error": str(
                        widget_result.get("error") or "Realtime stream failed to start."
                    ),
                }
            return {
                "ok": True,
                "model_name": str(widget_result.get("model_name") or model_text),
                "camera_source": str(
                    widget_result.get("camera_source") or camera_text or "0"
                ),
                "viewer_type": str(widget_result.get("viewer_type") or viewer),
                "classify_eye_blinks": bool(
                    widget_result.get("classify_eye_blinks", classify_eye_blinks)
                ),
            }
        return {
            "ok": True,
            "queued": True,
            "model_name": model_text,
            "camera_source": camera_text or "0",
            "viewer_type": viewer,
            "classify_eye_blinks": bool(classify_eye_blinks),
        }

    def _tool_gui_stop_realtime_stream(self) -> Dict[str, Any]:
        ok = self._invoke_widget_slot("bot_stop_realtime_stream")
        if not ok:
            return {"ok": False, "error": "Failed to queue realtime stop action"}
        return {"ok": True, "queued": True}

    def _tool_gui_rename_file(
        self,
        source_path: str = "",
        new_name: str = "",
        new_path: str = "",
        use_active_file: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        source_text = str(source_path or "").strip()
        target_name = str(new_name or "").strip()
        target_path = str(new_path or "").strip()
        if not target_name and not target_path:
            return {
                "ok": False,
                "error": "Provide a new_name or new_path for rename.",
            }

        current_path: Optional[Path] = None
        if source_text:
            current_path = Path(source_text).expanduser()
        elif bool(use_active_file):
            pdf_state = self._tool_gui_pdf_get_state()
            if isinstance(pdf_state, dict) and bool(pdf_state.get("ok")):
                active_pdf_path = str(pdf_state.get("path") or "").strip()
                if active_pdf_path:
                    current_path = Path(active_pdf_path).expanduser()
            if current_path is None:
                widget = self.widget
                host = getattr(widget, "host_window_widget", None) if widget else None
                active_video = str(getattr(host, "video_file", "") or "").strip()
                if active_video:
                    current_path = Path(active_video).expanduser()
        if current_path is None:
            return {
                "ok": False,
                "error": "No source file provided and no active file found.",
            }
        if not current_path.exists() or not current_path.is_file():
            return {
                "ok": False,
                "error": f"Source file is missing: {current_path}",
            }

        if target_name and not Path(target_name).suffix and current_path.suffix:
            target_name = f"{target_name}{current_path.suffix}"

        workspace = get_agent_workspace_path()
        renamer = RenameFileTool(allowed_dir=workspace)
        result_text = str(
            self._run_async(
                renamer.execute(
                    path=str(current_path),
                    new_name=target_name,
                    new_path=target_path,
                    overwrite=bool(overwrite),
                )
            )
            or ""
        )
        if not result_text.startswith("Successfully renamed "):
            return {
                "ok": False,
                "error": result_text or "Rename failed.",
                "path": str(current_path),
                "requested_new_name": target_name,
                "requested_new_path": target_path,
            }

        if target_path:
            resolved_new_path = Path(target_path).expanduser()
        else:
            resolved_new_path = current_path.with_name(target_name)
        reopened = False
        if resolved_new_path.exists() and resolved_new_path.suffix.lower() == ".pdf":
            reopened = bool(
                self._invoke_widget_slot(
                    "bot_open_pdf", QtCore.Q_ARG(str, str(resolved_new_path))
                )
            )

        return {
            "ok": True,
            "renamed": True,
            "old_path": str(current_path),
            "new_path": str(resolved_new_path),
            "reopened": reopened,
        }

    def _recover_with_plain_ollama_reply(self) -> str:
        return recover_with_plain_ollama_reply(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            settings=self.settings,
            logger=_LOGGER,
            import_module=importlib.import_module,
        )

    @staticmethod
    def _read_text_limited(path: Path, max_chars: int) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "\n...[truncated]"

    def _build_compact_system_prompt(
        self, workspace: Path, *, allowed_read_roots: Optional[List[str]] = None
    ) -> str:
        short_prompt = len(str(self.prompt or "").strip()) <= 80
        parts: List[str] = [
            "You are Annolid Bot. Be concise, practical, and return plain text answers."
        ]
        if self.enable_web_tools:
            parts.append(
                "Web tools are enabled (`web_search`, `web_fetch`). "
                "When a user asks about a URL or web page, use web tools to retrieve "
                "content before answering. Do not claim you cannot browse."
            )
            parts.append(
                "Do not assume the currently open embedded page is relevant. "
                "Use it only when the user explicitly asks about the open/current page "
                "or references that page URL/topic."
            )
            live_web_context = self._build_live_web_context_prompt_block(
                include_snapshot=self._should_attach_live_web_context(self.prompt)
            )
            if live_web_context:
                parts.append(live_web_context)
        live_pdf_context = self._build_live_pdf_context_prompt_block(
            include_snapshot=self._should_attach_live_pdf_context(self.prompt)
        )
        if live_pdf_context:
            parts.append(live_pdf_context)
        roots = [str(r).strip() for r in (allowed_read_roots or []) if str(r).strip()]
        if roots:
            parts.append(
                "Readable paths include workspace plus configured read roots. "
                "Do not claim a path is inaccessible before trying the relevant tool."
            )
            parts.append(
                "# Allowed Read Roots\n" + "\n".join(f"- {root}" for root in roots[:20])
            )
        agents_limit = 900 if short_prompt else 1600
        memory_limit = 500 if short_prompt else 900
        agents_md = self._read_text_limited(workspace / "AGENTS.md", agents_limit)
        if agents_md:
            parts.append(f"# Workspace Instructions\n{agents_md}")
        memory_md = self._read_text_limited(
            workspace / "memory" / "MEMORY.md", memory_limit
        )
        if memory_md:
            parts.append(f"# Long-term Memory\n{memory_md}")
        skills_dir = workspace / "skills"
        if skills_dir.exists():
            names = sorted(
                p.name
                for p in skills_dir.iterdir()
                if p.is_dir() and (p / "SKILL.md").exists()
            )
            if names:
                preview = ", ".join(names[:15])
                if len(names) > 15:
                    preview += ", ..."
                parts.append(
                    "Available skills exist in workspace. Use `read_file` to inspect a "
                    f"skill before using it. Skills: {preview}"
                )
        return "\n\n".join(parts)

    @staticmethod
    def _contains_hint(text: str, hints: Tuple[str, ...]) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        return any(h in lowered for h in hints)

    @staticmethod
    def _looks_like_url_request(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if "http://" in lowered or "https://" in lowered or "www." in lowered:
            return True
        return bool(
            re.search(
                r"\b[a-z0-9][a-z0-9\-]{0,62}(?:\.[a-z0-9][a-z0-9\-]{0,62})+\b",
                lowered,
            )
        )

    def _should_attach_live_web_context(self, prompt: str) -> bool:
        return self._contains_hint(
            prompt, _WEB_CONTEXT_HINTS
        ) or self._looks_like_url_request(prompt)

    def _should_attach_live_pdf_context(self, prompt: str) -> bool:
        return self._contains_hint(prompt, _PDF_CONTEXT_HINTS)

    def _should_use_open_page_fallback(self, prompt: str) -> bool:
        return self._should_attach_live_web_context(prompt)

    @staticmethod
    def _topic_tokens(text: str) -> List[str]:
        raw = re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())
        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "for",
            "of",
            "in",
            "on",
            "at",
            "is",
            "are",
            "my",
            "me",
            "your",
            "you",
            "check",
            "current",
            "latest",
            "today",
        }
        return [t for t in raw if len(t) > 2 and t not in stop]

    def _build_live_web_context_prompt_block(
        self, *, include_snapshot: bool = True
    ) -> str:
        state = self._tool_gui_web_get_state()
        if not isinstance(state, dict):
            return ""
        if not bool(state.get("ok")):
            return ""
        if not bool(state.get("has_page")):
            return "No embedded web page is currently open."
        url = str(state.get("url") or "").strip()
        title = str(state.get("title") or "").strip()
        snapshot_block = "Visible text snapshot: [omitted to save tokens]"
        if include_snapshot:
            page_payload = self._tool_gui_web_get_dom_text(max_chars=1200)
            if isinstance(page_payload, dict) and bool(page_payload.get("ok")):
                text = str(page_payload.get("text") or "").strip()
                if len(text) > 600:
                    text = text[:600].rstrip() + "\n...[truncated]"
                url = str(page_payload.get("url") or url).strip()
                title = str(page_payload.get("title") or title).strip()
                snapshot_block = f"Visible text snapshot:\n{text or '[empty]'}"
            else:
                snapshot_block = "Visible text snapshot unavailable."
        return (
            "# Active Embedded Web Page\n"
            f"URL: {url or '[unknown]'}\n"
            f"Title: {title or '[unknown]'}\n"
            f"{snapshot_block}"
        )

    def _build_live_pdf_context_prompt_block(
        self, *, include_snapshot: bool = True
    ) -> str:
        state = self._tool_gui_pdf_get_state()
        if not isinstance(state, dict):
            return ""
        if not bool(state.get("ok")) or not bool(state.get("has_pdf")):
            return ""
        title = str(state.get("title") or "").strip()
        path = str(state.get("path") or "").strip()
        page = int(state.get("current_page") or 0)
        total = int(state.get("total_pages") or 0)
        snapshot_block = "Text snapshot: [omitted to save tokens]"
        if include_snapshot:
            payload = self._tool_gui_pdf_get_text(max_chars=1200, pages=1)
            if isinstance(payload, dict) and bool(payload.get("ok")):
                text = str(payload.get("text") or "").strip()
                if len(text) > 600:
                    text = text[:600].rstrip() + "\n...[truncated]"
                title = str(payload.get("title") or title).strip()
                path = str(payload.get("path") or path).strip()
                page = int(payload.get("current_page") or page or 0)
                total = int(payload.get("total_pages") or total or 0)
                snapshot_block = f"Text snapshot:\n{text or '[empty]'}"
            else:
                snapshot_block = "Text snapshot unavailable."
        return (
            "# Active PDF\n"
            f"Path: {path or '[unknown]'}\n"
            f"Title: {title or '[unknown]'}\n"
            f"Page: {page}/{total}\n"
            f"{snapshot_block}"
        )

    def _build_ollama_llm_callable(self):
        return build_ollama_llm_callable(
            prompt=self.prompt,
            settings=self.settings,
            parse_tool_calls=self._parse_ollama_tool_calls,
            normalize_messages=self._normalize_messages_for_ollama,
            extract_text=self._extract_ollama_text,
            prompt_may_need_tools=self._prompt_may_need_tools,
            logger=_LOGGER,
            import_module=importlib.import_module,
        )

    @classmethod
    def _collect_ollama_stream(
        cls, stream_iter: Any
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        return collect_ollama_stream(stream_iter, cls._parse_ollama_tool_calls)

    @staticmethod
    def _parse_ollama_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
        return parse_ollama_tool_calls(raw_calls)

    @staticmethod
    def _normalize_messages_for_ollama(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return normalize_messages_for_ollama(messages)

    @staticmethod
    def _extract_ollama_text(response: Dict[str, Any]) -> str:
        return extract_ollama_text(response)

    @staticmethod
    def _format_tool_trace(tool_runs: Any) -> str:
        return format_tool_trace(tool_runs)

    @staticmethod
    def _prompt_may_need_tools(prompt: str) -> bool:
        return prompt_may_need_tools(prompt)

    def _run_ollama(self) -> None:
        run_ollama_streaming_chat(
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

    def _run_openai(self, provider_name: str = "openai") -> None:
        user_prompt, text = run_openai_compat_chat(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            provider_name=provider_name,
            settings=self.settings,
            load_history_messages=self._load_history_messages,
        )
        self._persist_turn(user_prompt, text)
        self._emit_final(text, is_error=False)

    def _run_gemini(self) -> None:
        user_prompt, text = run_gemini_chat(
            prompt=self.prompt,
            image_path=self.image_path,
            model=self.model,
            provider_name=self.provider,
            settings=self.settings,
        )
        self._persist_turn(user_prompt, text)
        self._emit_final(text, is_error=False)
