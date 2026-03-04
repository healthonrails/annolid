from __future__ import annotations

import asyncio
from typing import Any, Optional

from annolid.core.agent.gui_backend.response_finalize import (
    apply_web_response_fallbacks,
)


def test_apply_web_response_fallbacks_logs_discover_stage_events() -> None:
    events: list[tuple[str, str, str]] = []
    calls: list[str] = []

    def _record_event(stage: str, step: str, outcome: str) -> None:
        events.append((stage, step, outcome))

    async def _run() -> str:
        return await apply_web_response_fallbacks(
            text="",
            prompt="check NVDA stock price",
            tools=None,
            tool_run_count=0,
            enable_web_tools=True,
            looks_like_open_url_suggestion=lambda _text: False,
            should_apply_web_refusal_fallback_cb=lambda _text: True,
            should_force_web_fallback_cb=lambda _prompt, _text: True,
            try_open_page_content_fallback=lambda: "",
            try_browser_search_fallback=_browser_fallback(calls),
            try_web_search_fallback=_web_search_fallback(calls, "search-result"),
            try_web_fetch_fallback=_web_fetch_fallback(calls),
            log_web_fallback_event=_record_event,
        )

    result = asyncio.run(_run())
    assert result == "search-result"
    assert ("discover", "web_search", "hit") in events
    assert calls == ["web_search"]


def test_apply_web_response_fallbacks_prefers_fetch_for_read_stage() -> None:
    calls: list[str] = []

    async def _run() -> str:
        return await apply_web_response_fallbacks(
            text="",
            prompt="summarize this page content",
            tools=None,
            tool_run_count=0,
            enable_web_tools=True,
            looks_like_open_url_suggestion=lambda _text: False,
            should_apply_web_refusal_fallback_cb=lambda _text: True,
            should_force_web_fallback_cb=lambda _prompt, _text: True,
            try_open_page_content_fallback=lambda: "",
            try_browser_search_fallback=_browser_fallback(calls),
            try_web_search_fallback=_web_search_fallback(calls, ""),
            try_web_fetch_fallback=_web_fetch_fallback(calls, "fetch-result"),
            log_web_fallback_event=None,
        )

    result = asyncio.run(_run())
    assert result == "fetch-result"
    assert calls == ["web_fetch"]


def _browser_fallback(calls: list[str], result: str = ""):
    async def _fn(_prompt: str, _tools: Optional[Any]) -> str:
        calls.append("browser")
        return result

    return _fn


def _web_search_fallback(calls: list[str], result: str = ""):
    async def _fn(_prompt: str, _tools: Optional[Any]) -> str:
        calls.append("web_search")
        return result

    return _fn


def _web_fetch_fallback(calls: list[str], result: str = ""):
    async def _fn(_prompt: str, _tools: Optional[Any]) -> str:
        calls.append("web_fetch")
        return result

    return _fn
