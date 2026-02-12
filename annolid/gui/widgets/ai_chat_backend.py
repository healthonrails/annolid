from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qtpy import QtCore
from qtpy.QtCore import QMetaObject, QRunnable

from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.config import load_config
from annolid.core.agent.providers import OpenAICompatProvider, resolve_openai_compat
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.tools import FunctionToolRegistry, register_nanobot_style_tools
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.utils.llm_settings import LLMConfig, resolve_agent_runtime_config


_SESSION_STORE: Optional[PersistentSessionStore] = None
_LOGGER = logging.getLogger("annolid.bot.backend")
_OLLAMA_TOOL_SUPPORT_CACHE: Dict[str, bool] = {}
_OLLAMA_FORCE_PLAIN_CACHE: Dict[str, bool] = {}
_GUI_DISABLED_TOOLS = {"web_search", "web_fetch", "cron", "spawn", "message"}


def _get_session_store() -> PersistentSessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = PersistentSessionStore(AgentSessionManager())
    return _SESSION_STORE


def clear_chat_session(session_id: str) -> None:
    """Clear persisted chat history/facts for a specific GUI session."""
    _get_session_store().clear_session(str(session_id or "gui:annolid_bot:default"))


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
        runtime_cfg = resolve_agent_runtime_config(profile="playground")
        self.max_history_messages = int(runtime_cfg.max_history_messages)

    def run(self) -> None:
        _LOGGER.info(
            "annolid-bot turn start session=%s provider=%s model=%s prompt_chars=%d",
            self.session_id,
            self.provider,
            self.model,
            len(str(self.prompt or "")),
        )
        dep_error = self._provider_dependency_error()
        if dep_error:
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
            try:
                # Keep backward-compatible fallback behavior if agent loop setup fails.
                if self.provider == "ollama":
                    self._run_ollama()
                elif self.provider == "openai":
                    self._run_openai()
                elif self.provider == "openrouter":
                    self._run_openai(provider_name="openrouter")
                elif self.provider == "gemini":
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
                    f"Error in chat interaction: {exc}; fallback failed: {fallback_exc}",
                    is_error=True,
                )

    def _provider_dependency_error(self) -> Optional[str]:
        if self.provider in {"openai", "openrouter"}:
            if importlib.util.find_spec("openai") is None:
                return (
                    "OpenAI/OpenRouter provider requires the `openai` package. "
                    "Install it in your Annolid environment, for example: "
                    "`.venv/bin/pip install openai`."
                )
        if self.provider == "gemini":
            if importlib.util.find_spec("google.generativeai") is None:
                return (
                    "Gemini provider requires `google-generativeai`. "
                    "Install it in your Annolid environment, for example: "
                    "`.venv/bin/pip install google-generativeai`."
                )
        return None

    def _format_dependency_error(self, raw_error: str) -> str:
        message = str(raw_error or "").strip()
        if (
            self.provider in {"openai", "openrouter"}
            and "openai package is required" in message
        ):
            return (
                "OpenAI/OpenRouter provider requires the `openai` package. "
                "Install it in your Annolid environment, for example: "
                "`.venv/bin/pip install openai`."
            )
        if self.provider == "gemini" and "google-generativeai" in message:
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

    def _emit_final(self, message: str, *, is_error: bool) -> None:
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

    def _persist_turn(self, user_text: str, assistant_text: str) -> None:
        if not self.session_store:
            return
        user_msg = str(user_text or "").strip()
        assistant_msg = str(assistant_text or "").strip()
        if not user_msg and not assistant_msg:
            return
        entries: List[Dict[str, str]] = []
        if user_msg:
            entries.append({"role": "user", "content": user_msg})
        if assistant_msg:
            entries.append({"role": "assistant", "content": assistant_msg})
        if not entries:
            return
        try:
            self.session_store.append_history(
                self.session_id,
                entries,
                max_messages=self.max_history_messages,
            )
        except Exception:
            return

    def _run_agent_loop(self) -> None:
        workspace = get_agent_workspace_path()
        agent_cfg = load_config()
        allowed_read_roots = list(
            getattr(agent_cfg.tools, "allowed_read_roots", []) or []
        )
        tools = FunctionToolRegistry()
        register_nanobot_style_tools(
            tools,
            allowed_dir=workspace,
            allowed_read_roots=allowed_read_roots,
        )
        for tool_name in _GUI_DISABLED_TOOLS:
            tools.unregister(tool_name)
        system_prompt = self._build_compact_system_prompt(
            workspace, allowed_read_roots=allowed_read_roots
        )
        _LOGGER.info(
            "annolid-bot agent config session=%s model=%s tools=%d read_roots=%d prompt_chars=%d",
            self.session_id,
            self.model,
            len(tools),
            len(allowed_read_roots),
            len(system_prompt),
        )
        if self.provider == "ollama" and _OLLAMA_FORCE_PLAIN_CACHE.get(
            self.model, False
        ):
            _LOGGER.warning(
                "annolid-bot model is in forced plain mode; skipping agent/tool loop model=%s",
                self.model,
            )
            text = self._recover_with_plain_ollama_reply()
            if not text:
                text = (
                    "Model returned empty output in plain mode. "
                    f"Provider={self.provider}, model={self.model}. "
                    "Please switch to another Ollama model for Annolid Bot."
                )
            if self.show_tool_trace:
                text = f"{text}\n\n[Tool Trace]\n(skipped: model plain-mode fallback)".strip()
            if text.strip():
                self._persist_turn(self.prompt, text)
            self._emit_final(text, is_error=False)
            return

        llm_callable = None
        if self.provider == "ollama":
            llm_callable = self._build_ollama_llm_callable()

        loop = AgentLoop(
            tools=tools,
            llm_callable=llm_callable,
            provider=self.provider,
            model=self.model,
            profile="playground",
            memory_store=self.session_store,
            workspace=str(workspace),
            allowed_read_roots=allowed_read_roots,
        )
        media: Optional[List[str]] = None
        if self.image_path and os.path.exists(self.image_path):
            media = [self.image_path]

        result = asyncio.run(
            loop.run(
                self.prompt,
                session_id=self.session_id,
                channel="gui",
                chat_id="annolid_bot",
                media=media,
                system_prompt=system_prompt,
            )
        )
        text = str(getattr(result, "content", "") or "").strip()
        used_recovery = False
        # Final safety net: if the model still returns empty after our in-call retry,
        # attempt a single plain Ollama stream request (no tools) and use it.
        if not text and self.provider == "ollama":
            text = self._recover_with_plain_ollama_reply()
            used_recovery = bool(text)
            if used_recovery:
                _OLLAMA_FORCE_PLAIN_CACHE[self.model] = True
        if not text:
            text = (
                "Model returned empty output after multiple attempts. "
                f"Provider={self.provider}, model={self.model}. "
                "Please switch to another Ollama model for Annolid Bot."
            )
        if self.show_tool_trace:
            trace = self._format_tool_trace(getattr(result, "tool_runs", ()) or ())
            text = f"{text}\n\n{trace}".strip()
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
            self._persist_turn(self.prompt, text)
        self._emit_final(text, is_error=False)

    def _recover_with_plain_ollama_reply(self) -> str:
        host = str(self.settings.get("ollama", {}).get("host") or "").strip()
        try:
            ollama_module = importlib.import_module("ollama")
        except ImportError:
            return ""

        user_message: Dict[str, Any] = {
            "role": "user",
            "content": str(self.prompt or ""),
        }
        if self.image_path and os.path.exists(self.image_path):
            user_message["images"] = [self.image_path]

        # Keep this recovery intentionally simple: one streaming attempt, no tool schema.
        def _run_stream_once(extra_nudge: bool = False) -> str:
            prev_host_present = "OLLAMA_HOST" in os.environ
            prev_host_value = os.environ.get("OLLAMA_HOST")
            try:
                if host:
                    os.environ["OLLAMA_HOST"] = host
                else:
                    os.environ.pop("OLLAMA_HOST", None)
                msgs = [user_message]
                if extra_nudge:
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                "Reply with plain text in one short paragraph."
                            ),
                        }
                    )
                stream_iter = ollama_module.chat(
                    model=self.model,
                    messages=msgs,
                    stream=True,
                )
                chunks: List[str] = []
                for part in stream_iter:
                    if "message" in part and "content" in part["message"]:
                        chunk = str(part["message"]["content"] or "")
                        if chunk:
                            chunks.append(chunk)
                return "".join(chunks).strip()
            finally:
                if prev_host_present and prev_host_value is not None:
                    os.environ["OLLAMA_HOST"] = prev_host_value
                else:
                    os.environ.pop("OLLAMA_HOST", None)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                text_stream = executor.submit(_run_stream_once, False).result(
                    timeout=35
                )
            _LOGGER.info(
                "annolid-bot plain ollama stream recovery model=%s content_chars=%d",
                self.model,
                len(text_stream),
            )
            if text_stream:
                return text_stream
            with ThreadPoolExecutor(max_workers=1) as executor:
                text_stream_nudge = executor.submit(_run_stream_once, True).result(
                    timeout=35
                )
            _LOGGER.info(
                "annolid-bot plain ollama stream-nudge recovery model=%s content_chars=%d",
                self.model,
                len(text_stream_nudge),
            )
            return text_stream_nudge
        except FutureTimeoutError:
            _LOGGER.warning(
                "annolid-bot plain ollama recovery timed out model=%s",
                self.model,
            )
            return ""
        except Exception as exc:
            _LOGGER.warning(
                "annolid-bot plain ollama recovery failed model=%s error=%s",
                self.model,
                exc,
            )
            return ""

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
        parts: List[str] = [
            "You are Annolid Bot. Be concise, practical, and return plain text answers."
        ]
        roots = [str(r).strip() for r in (allowed_read_roots or []) if str(r).strip()]
        if roots:
            parts.append(
                "Readable paths include workspace plus configured read roots. "
                "Do not claim a path is inaccessible before trying the relevant tool."
            )
            parts.append(
                "# Allowed Read Roots\n" + "\n".join(f"- {root}" for root in roots[:20])
            )
        agents_md = self._read_text_limited(workspace / "AGENTS.md", 2400)
        if agents_md:
            parts.append(f"# Workspace Instructions\n{agents_md}")
        memory_md = self._read_text_limited(workspace / "memory" / "MEMORY.md", 1400)
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

    def _build_ollama_llm_callable(self):
        host = str(self.settings.get("ollama", {}).get("host") or "").strip()
        try:
            ollama_module = importlib.import_module("ollama")
        except ImportError as exc:
            raise ImportError(
                "The python 'ollama' package is required for Ollama agent mode."
            ) from exc

        def _coerce_tool_calls(tool_calls_payload: Any) -> List[Dict[str, Any]]:
            """Accept either raw Ollama tool_calls or already-normalized tool calls."""
            if not isinstance(tool_calls_payload, list):
                return []
            # If a payload already looks like our normalized shape, keep it.
            if tool_calls_payload and all(
                isinstance(item, dict) and "name" in item for item in tool_calls_payload
            ):
                return [
                    dict(item) for item in tool_calls_payload if isinstance(item, dict)
                ]
            return self._parse_ollama_tool_calls(tool_calls_payload)

        async def _call(
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]],
            model_id: str,
        ) -> Dict[str, Any]:
            prepared = self._normalize_messages_for_ollama(messages)
            supports_tools = _OLLAMA_TOOL_SUPPORT_CACHE.get(model_id, True)
            effective_tools = (
                [dict(t) for t in tools] if (tools and supports_tools) else None
            )
            _LOGGER.info(
                "annolid-bot ollama request model=%s effective_tools_sent=%d supports_tools=%s",
                model_id,
                len(effective_tools or []),
                supports_tools,
            )

            def _invoke_chat_stream(
                tools_payload: Optional[List[Dict[str, Any]]],
            ) -> Dict[str, Any]:
                prev_host_present = "OLLAMA_HOST" in os.environ
                prev_host_value = os.environ.get("OLLAMA_HOST")
                try:
                    if host:
                        os.environ["OLLAMA_HOST"] = host
                    else:
                        os.environ.pop("OLLAMA_HOST", None)
                    stream_iter = ollama_module.chat(
                        model=model_id,
                        messages=prepared,
                        tools=tools_payload,
                        stream=True,
                    )
                    content, tool_calls, done_reason = self._collect_ollama_stream(
                        stream_iter
                    )
                    return {
                        "done_reason": done_reason,
                        "message": {"content": content, "tool_calls": tool_calls},
                    }
                finally:
                    if prev_host_present and prev_host_value is not None:
                        os.environ["OLLAMA_HOST"] = prev_host_value
                    else:
                        os.environ.pop("OLLAMA_HOST", None)

            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(_invoke_chat_stream, effective_tools), 60
                )
            except Exception as exc:
                msg = str(exc)
                if "400" in msg and effective_tools:
                    _LOGGER.warning(
                        "annolid-bot ollama tool-call request rejected; retrying without tools model=%s error=%s",
                        model_id,
                        exc,
                    )
                    _OLLAMA_TOOL_SUPPORT_CACHE[model_id] = False

                    def _invoke_chat_without_tools() -> Dict[str, Any]:
                        return _invoke_chat_stream(None)

                    response = await asyncio.wait_for(
                        asyncio.to_thread(_invoke_chat_without_tools), 60
                    )
                else:
                    raise
            msg = dict(response.get("message") or {})
            tool_calls = _coerce_tool_calls(msg.get("tool_calls"))
            content = self._extract_ollama_text(response)
            _LOGGER.info(
                "annolid-bot ollama raw response model=%s done_reason=%s content_chars=%d tool_calls=%d",
                model_id,
                str(response.get("done_reason") or ""),
                len(content),
                len(tool_calls),
            )
            if tool_calls:
                _OLLAMA_TOOL_SUPPORT_CACHE[model_id] = True
                _OLLAMA_FORCE_PLAIN_CACHE[model_id] = False
            if not content.strip() and not tool_calls:
                # Empirically: some "cloud" models return empty content when a tools schema is provided,
                # but produce text when tools are omitted. Do a single fast retry without tools and cache.
                if effective_tools is not None:
                    _LOGGER.warning(
                        "annolid-bot ollama returned empty content with tools; retrying once without tools model=%s",
                        model_id,
                    )
                    _OLLAMA_TOOL_SUPPORT_CACHE[model_id] = False
                    response2 = await asyncio.wait_for(
                        asyncio.to_thread(_invoke_chat_stream, None), 60
                    )
                    msg2 = dict(response2.get("message") or {})
                    tool_calls2 = _coerce_tool_calls(msg2.get("tool_calls"))
                    content2 = self._extract_ollama_text(response2)
                    _LOGGER.info(
                        "annolid-bot ollama no-tools retry model=%s done_reason=%s content_chars=%d tool_calls=%d",
                        model_id,
                        str(response2.get("done_reason") or ""),
                        len(content2),
                        len(tool_calls2),
                    )
                    if content2.strip() or tool_calls2:
                        response = response2
                        tool_calls = tool_calls2
                        content = content2
                if not content.strip() and not tool_calls:
                    _LOGGER.warning(
                        "annolid-bot ollama returned empty content (tools=%s) model=%s",
                        bool(effective_tools),
                        model_id,
                    )
                    if effective_tools is None:
                        _OLLAMA_FORCE_PLAIN_CACHE[model_id] = True
            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": str(response.get("done_reason") or "stop"),
                "usage": {},
                "reasoning_content": None,
            }

        return _call

    @classmethod
    def _collect_ollama_stream(
        cls, stream_iter: Any
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Collect non-streaming output from an Ollama stream iterator.

        Some models/providers return empty content for stream=False; stream=True is more reliable.
        """
        chunks: List[str] = []
        tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
        done_reason = "stop"
        for part in stream_iter:
            if not isinstance(part, dict):
                continue
            done_reason = str(part.get("done_reason") or done_reason)
            msg = part.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content:
                    chunks.append(content)
                raw_tool_calls = msg.get("tool_calls")
                if raw_tool_calls:
                    for call in cls._parse_ollama_tool_calls(raw_tool_calls):
                        call_id = str(call.get("id") or f"call_{len(tool_calls_by_id)}")
                        tool_calls_by_id[call_id] = call
        return "".join(chunks).strip(), list(tool_calls_by_id.values()), done_reason

    @staticmethod
    def _parse_ollama_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        for idx, item in enumerate(list(raw_calls or [])):
            if not isinstance(item, dict):
                continue
            fn = item.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    args = parsed if isinstance(parsed, dict) else {"_raw": args}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            elif not isinstance(args, dict):
                args = {"_raw": args}
            call_id = str(item.get("id") or f"ollama_call_{idx}")
            tool_calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": dict(args),
                }
            )
        return tool_calls

    @staticmethod
    def _normalize_messages_for_ollama(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "")
            content = msg.get("content")
            out: Dict[str, Any] = {"role": role}
            if isinstance(content, list):
                text_parts: List[str] = []
                images: List[Any] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text") or ""))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url") or {}
                        if isinstance(image_url, dict):
                            url = str(image_url.get("url") or "")
                            if url.startswith("data:image/") and ";base64," in url:
                                try:
                                    images.append(
                                        base64.b64decode(url.split(";base64,", 1)[1])
                                    )
                                except Exception:
                                    continue
                out["content"] = "\n".join([p for p in text_parts if p]).strip()
                if images:
                    out["images"] = images
            else:
                out["content"] = str(content or "")
                existing_images = msg.get("images")
                if isinstance(existing_images, list) and existing_images:
                    out["images"] = list(existing_images)
            normalized.append(out)
        return normalized

    @staticmethod
    def _extract_ollama_text(response: Dict[str, Any]) -> str:
        msg = response.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    elif isinstance(item, str) and item.strip():
                        parts.append(item)
                if parts:
                    return "\n".join(parts).strip()
            thinking = msg.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                return thinking
            text = msg.get("text")
            if isinstance(text, str) and text.strip():
                return text
            output_text = msg.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text
            output = msg.get("output")
            if isinstance(output, str) and output.strip():
                return output

        fallback = response.get("response")
        if isinstance(fallback, str) and fallback.strip():
            return fallback
        top_text = response.get("text")
        if isinstance(top_text, str) and top_text.strip():
            return top_text
        output_text_top = response.get("output_text")
        if isinstance(output_text_top, str) and output_text_top.strip():
            return output_text_top
        return ""

    @staticmethod
    def _format_tool_trace(tool_runs: Any) -> str:
        lines: List[str] = []
        for run in tool_runs:
            name = str(getattr(run, "name", "") or "").strip()
            args = getattr(run, "arguments", {}) or {}
            result = str(getattr(run, "result", "") or "").strip()
            if not name:
                continue
            lines.append(f"- `{name}` args={args}")
            if result:
                lines.append(f"  -> {result}")
        if not lines:
            return "[Tool Trace]\n(no tool calls)"
        return "[Tool Trace]\n" + "\n".join(lines)

    def _run_ollama(self) -> None:
        ollama_module = globals().get("ollama")
        if ollama_module is None:
            try:
                ollama_module = importlib.import_module("ollama")
            except ImportError as exc:
                raise ImportError(
                    "The python 'ollama' package is not installed."
                ) from exc
            globals()["ollama"] = ollama_module

        host = self.settings.get("ollama", {}).get("host")
        prev_host_present = "OLLAMA_HOST" in os.environ
        prev_host_value = os.environ.get("OLLAMA_HOST")
        try:
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            messages = self._load_history_messages()
            user_message = {"role": "user", "content": self.prompt}
            if self.image_path and os.path.exists(self.image_path):
                user_message["images"] = [self.image_path]
            messages.append(user_message)

            stream = ollama_module.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )
            full_response = ""
            for part in stream:
                if "message" in part and "content" in part["message"]:
                    chunk = part["message"]["content"]
                    full_response += chunk
                    self._emit_chunk(chunk)
                elif "error" in part:
                    self._emit_final(f"Stream error: {part['error']}", is_error=True)
                    return

            if not full_response.strip():
                self._emit_final("No response from Ollama.", is_error=True)
            else:
                self._persist_turn(self.prompt, full_response)
                self._emit_final("", is_error=False)
        finally:
            if prev_host_present and prev_host_value is not None:
                os.environ["OLLAMA_HOST"] = prev_host_value
            else:
                os.environ.pop("OLLAMA_HOST", None)

    def _run_openai(self, provider_name: str = "openai") -> None:
        provider_key = str(provider_name or "openai").strip().lower()
        provider_block = dict(self.settings.get(provider_key, {}))
        cfg = LLMConfig(
            provider=provider_key,
            model=self.model,
            params=provider_block,
        )
        resolved = resolve_openai_compat(cfg)
        provider = OpenAICompatProvider(resolved=resolved)

        user_prompt = self.prompt
        if self.image_path and os.path.exists(self.image_path):
            user_prompt += (
                f"\n\n[Note: Image context available at {self.image_path}. "
                "Use this visual context in your response.]"
            )

        messages = self._load_history_messages()
        messages.append({"role": "user", "content": user_prompt})

        async def _chat_once() -> str:
            model_lower = (self.model or "").lower()
            # Keep existing behavior: avoid temperature for GPT-5 models.
            temperature = 0.7 if "gpt-5" not in model_lower else None
            resp = await provider.chat(
                messages=messages,
                model=self.model,
                temperature=temperature,
            )
            return str(resp.content or "")

        text = asyncio.run(_chat_once())
        self._persist_turn(user_prompt, text)
        self._emit_final(text, is_error=False)

    def _run_gemini(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for Gemini providers."
            ) from exc

        config = self.settings.get("gemini", {})
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is missing. Configure it in settings.")

        genai.configure(api_key=api_key)
        model_name = self.model or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)

        user_prompt = self.prompt
        if self.image_path and os.path.exists(self.image_path):
            user_prompt += (
                f"\n\n[Note: Image context available at {self.image_path}. "
                "Use this visual context in your response.]"
            )

        result = model.generate_content(user_prompt)
        text = getattr(result, "text", "") or ""
        self._persist_turn(user_prompt, text)
        self._emit_final(text, is_error=False)
