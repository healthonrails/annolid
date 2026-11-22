"""PDF summarization and caching mixin for StreamingChatTask."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

from qtpy import QtCore

from annolid.infrastructure.agent_workspace import get_agent_workspace_path
from annolid.services.chat_agent_core import DownloadPdfTool
from annolid.services.chat_backend_support import gui_summarize_active_pdf_with_cache

logger = logging.getLogger(__name__)


def _backend_attr(name: str, default: Any) -> Any:
    mod = sys.modules.get("annolid.gui.widgets.ai_chat_backend")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default


class StreamingChatPdfMixin:
    """Mixin encapsulating PDF summarization, caching, and web-PDF helpers."""

    def _tool_gui_pdf_summarize(
        self,
        path: str = "",
        max_pages: int = 80,
        max_extract_chars: int = 350000,
    ) -> Dict[str, Any]:
        normalized_max_pages = max(1, min(int(max_pages or 80), 200))
        normalized_max_extract_chars = max(
            10000, min(int(max_extract_chars or 350000), 2000000)
        )
        attempted_pdf_paths: set[str] = set()
        path_text = str(path or "").strip()
        if path_text:
            resolved_explicit = self._resolve_pdf_path_for_gui_tool(path_text)
            if resolved_explicit is not None and resolved_explicit.exists():
                canonical = self._canonical_existing_path(resolved_explicit)
                if canonical:
                    attempted_pdf_paths.add(canonical)
                summary_text = self._summarize_pdf_path_with_cache(
                    resolved_explicit,
                    max_pages=normalized_max_pages,
                    max_extract_chars=normalized_max_extract_chars,
                )
                if summary_text:
                    return {
                        "ok": True,
                        "path": str(resolved_explicit),
                        "title": str(resolved_explicit.name),
                        "summary": summary_text,
                        "text": summary_text,
                    }
            opened = self._run_async(self._tool_gui_open_pdf(path_text))
            if isinstance(opened, dict) and bool(opened.get("error")):
                return {
                    "ok": False,
                    "error": str(opened.get("error") or "Failed to open PDF."),
                    "open_pdf": opened,
                }

        summary_text = self._summarize_active_pdf_with_cache_compat(
            max_pages=normalized_max_pages,
            max_extract_chars=normalized_max_extract_chars,
        )
        if summary_text:
            state = self._tool_gui_pdf_get_state()
            return {
                "ok": True,
                "path": str(state.get("path") or ""),
                "title": str(state.get("title") or ""),
                "summary": summary_text,
                "text": summary_text,
            }
        active_state = dict(self._tool_gui_pdf_get_state() or {})
        active_canonical = self._canonical_existing_path(
            str(active_state.get("path") or "")
        )
        if active_canonical:
            attempted_pdf_paths.add(active_canonical)

        source = self._resolve_pdf_summary_source(path_text=path_text)
        source_mode = str(source.get("mode") or "").strip()
        if source_mode == "pdf_path":
            pdf_path = source.get("path")
            if isinstance(pdf_path, Path):
                canonical = self._canonical_existing_path(pdf_path)
                if canonical and canonical in attempted_pdf_paths:
                    pdf_path = None
            if isinstance(pdf_path, Path):
                summary_text = self._summarize_pdf_path_with_cache(
                    pdf_path,
                    max_pages=normalized_max_pages,
                    max_extract_chars=normalized_max_extract_chars,
                )
                if summary_text:
                    return {
                        "ok": True,
                        "path": str(pdf_path),
                        "title": str(pdf_path.name),
                        "summary": summary_text,
                        "text": summary_text,
                    }
        elif source_mode == "cache_md":
            cache_path = source.get("path")
            if isinstance(cache_path, Path):
                summary_text = self._summarize_cached_pdf_markdown(cache_path)
                if summary_text:
                    return {
                        "ok": True,
                        "path": "",
                        "title": str(source.get("title") or ""),
                        "summary": summary_text,
                        "text": summary_text,
                    }

        prepared = self._prepare_active_pdf_for_summary_from_web()
        if bool(prepared.get("ok")):
            summary_text = self._summarize_active_pdf_with_cache_compat(
                max_pages=normalized_max_pages,
                max_extract_chars=normalized_max_extract_chars,
            )
            if summary_text:
                state = self._tool_gui_pdf_get_state()
                return {
                    "ok": True,
                    "path": str(state.get("path") or ""),
                    "title": str(state.get("title") or ""),
                    "summary": summary_text,
                    "text": summary_text,
                }

        source_hint = str(source.get("hint") or "").strip()
        if bool(prepared.get("error")) and source_hint:
            source_hint = (
                f"{source_hint}; web_prepare={str(prepared.get('error') or '')}"
            )
        if not summary_text:
            return {
                "ok": False,
                "error": (
                    "Failed to summarize PDF from active view, session context, "
                    "or cached extraction."
                ),
                "source_hint": source_hint,
                "prepare_pdf": prepared if bool(prepared.get("error")) else {},
            }
        return {"ok": False, "error": "Failed to summarize PDF."}

    def _summarize_pdf_path_with_cache(
        self,
        pdf_path: Path,
        *,
        max_pages: int,
        max_extract_chars: int,
    ) -> str:
        candidate = Path(pdf_path).expanduser()
        if not candidate.exists() or not candidate.is_file():
            return ""
        return _backend_attr(
            "gui_summarize_active_pdf_with_cache", gui_summarize_active_pdf_with_cache
        )(
            workspace=self.workspace,
            get_pdf_state=lambda: {
                "ok": True,
                "has_pdf": True,
                "path": str(candidate),
                "title": str(candidate.name),
            },
            build_summary=self._build_extractive_summary,
            build_llm_summary=self._build_llm_pdf_summary,
            require_llm_summary=True,
            max_pages=max_pages,
            max_extract_chars=max_extract_chars,
            on_telemetry=self._log_pdf_summary_telemetry,
        )

    @staticmethod
    def _extract_cached_markdown_paths(text: str) -> List[Path]:
        value = str(text or "")
        if not value.strip():
            return []
        paths: list[Path] = []
        for match in re.findall(r"Cached extraction:\s*([^\n\r]+)", value, flags=re.I):
            candidate = Path(str(match or "").strip().strip("\"'`")).expanduser()
            if str(candidate).lower().endswith(".md"):
                paths.append(candidate)
        for match in re.findall(
            r"(?:~?/|/)[^\n\r]+?pdf_text_cache[^\n\r]+?\.md",
            value,
            flags=re.I,
        ):
            candidate = Path(str(match or "").strip().strip("\"'`")).expanduser()
            paths.append(candidate)
        deduped: list[Path] = []
        seen: set[str] = set()
        for path_obj in paths:
            key = str(path_obj)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path_obj)
        return deduped

    def _resolve_pdf_summary_source(self, *, path_text: str = "") -> Dict[str, Any]:
        active_state = dict(self._tool_gui_pdf_get_state() or {})
        active_path = str(active_state.get("path") or "").strip()
        if (
            bool(active_state.get("ok"))
            and bool(active_state.get("has_pdf"))
            and active_path
        ):
            candidate = Path(active_path).expanduser()
            if candidate.exists() and candidate.is_file():
                return {"mode": "pdf_path", "path": candidate, "hint": "active_pdf"}

        if path_text:
            resolved = self._resolve_pdf_path_for_gui_tool(path_text)
            if resolved is not None and resolved.exists() and resolved.is_file():
                return {"mode": "pdf_path", "path": resolved, "hint": "explicit_path"}

        context_blobs: list[str] = [str(self.prompt or "").strip()]
        history = self._load_history_messages()
        for message in reversed(history[-80:]):
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "").strip()
            if content:
                context_blobs.append(content)

        for blob in context_blobs:
            for raw in self._extract_pdf_path_candidates(blob):
                resolved = self._resolve_pdf_path_for_gui_tool(raw)
                if resolved is None:
                    try:
                        candidate = Path(
                            str(raw or "").strip().strip("\"'`")
                        ).expanduser()
                    except Exception:
                        candidate = None
                    if candidate is not None:
                        try:
                            if candidate.exists() and candidate.is_file():
                                resolved = candidate
                        except (OSError, ValueError):
                            resolved = None
                if resolved is not None and resolved.exists() and resolved.is_file():
                    return {
                        "mode": "pdf_path",
                        "path": resolved,
                        "hint": "session_pdf_path",
                    }

        for blob in context_blobs:
            for cache_path in self._extract_cached_markdown_paths(blob):
                if cache_path.exists() and cache_path.is_file():
                    return {
                        "mode": "cache_md",
                        "path": cache_path,
                        "title": str(cache_path.name),
                        "hint": "session_cached_md",
                    }

        cache_dir = self.workspace / "pdf_text_cache"
        if cache_dir.exists() and cache_dir.is_dir():
            try:
                cache_files = sorted(
                    cache_dir.glob("*.md"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                    reverse=True,
                )
            except Exception:
                cache_files = []
            if cache_files:
                return {
                    "mode": "cache_md",
                    "path": cache_files[0],
                    "title": str(cache_files[0].name),
                    "hint": "latest_cached_md",
                }

        available = self._list_available_pdfs(limit=1)
        if available:
            candidate = available[0]
            if candidate.exists() and candidate.is_file():
                return {
                    "mode": "pdf_path",
                    "path": candidate,
                    "hint": "recent_local_pdf",
                }
        return {"mode": "", "hint": "no_pdf_source"}

    def _summarize_cached_pdf_markdown(self, cache_path: Path) -> str:
        candidate = Path(cache_path).expanduser()
        if not candidate.exists() or not candidate.is_file():
            return ""
        try:
            raw_text = str(candidate.read_text(encoding="utf-8") or "").strip()
        except Exception:
            return ""
        if not raw_text:
            return ""
        source_pdf_path = ""
        source_match = re.search(
            r"^\s*Source:\s*(.+?\.pdf)\s*$",
            raw_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if source_match:
            source_pdf_path = str(source_match.group(1) or "").strip()
        summary = self._build_llm_pdf_summary(
            raw_text,
            max_chars=1800,
            pdf_path=source_pdf_path,
        )
        if not summary:
            return ""
        source_label = Path(source_pdf_path).name if source_pdf_path else candidate.stem
        return (
            f"Summary of cached PDF ({source_label}):\n"
            f"{summary}\n\n"
            f"Cached extraction: {candidate}"
        )

    @staticmethod
    def _extract_pdf_path_from_web_save_payload(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        saved_path = str(payload.get("saved_path") or "").strip()
        if saved_path.lower().endswith(".pdf") and Path(saved_path).exists():
            return saved_path
        if bool(payload.get("queued")):
            return ""
        saved_kind = str(
            payload.get("last_saved_kind") or payload.get("kind") or ""
        ).lower()
        last_saved_path = str(payload.get("last_saved_path") or "").strip()
        if (
            saved_kind == "pdf"
            and last_saved_path.lower().endswith(".pdf")
            and Path(last_saved_path).exists()
        ):
            return last_saved_path
        return ""

    def _prepare_active_pdf_for_summary_from_web(self) -> Dict[str, Any]:
        web_state = dict(self._tool_gui_web_get_state() or {})
        if not bool(web_state.get("ok")) or not bool(web_state.get("has_page")):
            return {
                "ok": False,
                "error": "No active web page available for PDF preparation.",
            }

        source_url = str(web_state.get("url") or "").strip()
        if source_url:
            opened_from_url = self._run_async(self._tool_gui_open_pdf(source_url))
            if isinstance(opened_from_url, dict) and bool(opened_from_url.get("ok")):
                state_after_url_open = dict(self._tool_gui_pdf_get_state() or {})
                if bool(state_after_url_open.get("ok")) and bool(
                    state_after_url_open.get("has_pdf")
                ):
                    return {
                        "ok": True,
                        "mode": "open_url",
                        "path": str(state_after_url_open.get("path") or ""),
                    }

        initial_save = dict(self._tool_gui_web_save_current() or {})
        saved_pdf_path = self._extract_pdf_path_from_web_save_payload(initial_save)
        if not saved_pdf_path and bool(initial_save.get("queued")):
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                QtCore.QThread.msleep(125)
                polled = dict(self._tool_gui_web_save_current() or {})
                saved_pdf_path = self._extract_pdf_path_from_web_save_payload(polled)
                if saved_pdf_path:
                    break
                if not bool(polled.get("queued")):
                    break
        if not saved_pdf_path:
            return {
                "ok": False,
                "error": "Could not save a PDF from the active web viewer.",
                "web_save": initial_save,
            }

        opened_saved = self._run_async(self._tool_gui_open_pdf(saved_pdf_path))
        if isinstance(opened_saved, dict) and bool(opened_saved.get("error")):
            return {
                "ok": False,
                "error": str(opened_saved.get("error") or "Failed to open saved PDF."),
                "web_save": initial_save,
                "open_pdf": opened_saved,
            }
        state_after_save_open = dict(self._tool_gui_pdf_get_state() or {})
        if not bool(state_after_save_open.get("ok")) or not bool(
            state_after_save_open.get("has_pdf")
        ):
            return {
                "ok": False,
                "error": "Saved PDF could not be activated for summarization.",
                "web_save": initial_save,
                "path": saved_pdf_path,
            }
        return {
            "ok": True,
            "mode": "saved_pdf",
            "path": str(state_after_save_open.get("path") or saved_pdf_path),
        }

    async def _download_pdf_for_gui_tool(self, url: str) -> Optional[Path]:
        text = str(url or "").strip()
        if not text.lower().startswith(("http://", "https://")):
            return None
        workspace = _backend_attr(
            "get_agent_workspace_path", get_agent_workspace_path
        )()
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

    def _log_pdf_summary_telemetry(self, telemetry: Dict[str, Any]) -> None:
        payload = dict(telemetry or {})
        if not payload:
            return
        ok = bool(payload.get("ok"))
        if ok:
            logger.info(
                "annolid-bot pdf summarize session=%s model=%s ok=1 cache_hit=%d pages=%s text_chars=%s extract_ms=%.1f summary_ms=%.1f total_ms=%.1f path=%s cache=%s",
                self.session_id,
                self.model,
                int(bool(payload.get("cache_hit"))),
                int(payload.get("pages_scanned") or 0),
                int(payload.get("text_chars") or 0),
                float(payload.get("extract_ms") or 0.0),
                float(payload.get("summary_ms") or 0.0),
                float(payload.get("total_ms") or 0.0),
                str(payload.get("pdf_path") or ""),
                str(payload.get("cache_path") or ""),
            )
            return
        logger.warning(
            "annolid-bot pdf summarize session=%s model=%s ok=0 reason=%s total_ms=%.1f path=%s",
            self.session_id,
            self.model,
            str(payload.get("reason") or "unknown"),
            float(payload.get("total_ms") or 0.0),
            str(payload.get("pdf_path") or ""),
        )
