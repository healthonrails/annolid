from __future__ import annotations

import asyncio
from typing import Any, Optional

from annolid.core.agent.gui_backend.fallbacks import (
    try_browser_search_fallback,
    try_web_fetch_fallback,
    try_web_search_fallback,
)
from annolid.core.agent.web_prompt_utils import (
    derive_web_lookup_prompt_from_messages,
    normalize_web_lookup_prompt,
)
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


def test_apply_web_response_fallbacks_uses_stage_order() -> None:
    scenarios = [
        (
            "check stock price",
            ["web_search", "web_fetch", "browser"],
        ),
        (
            "summarize this page content",
            ["web_fetch", "web_search", "browser"],
        ),
        (
            "click the button on this website",
            ["browser", "web_search", "web_fetch"],
        ),
    ]

    for prompt, expected_order in scenarios:
        calls: list[str] = []

        async def _run() -> str:
            return await apply_web_response_fallbacks(
                text="I don't have web browsing capabilities.",
                prompt=prompt,
                tools=None,
                tool_run_count=0,
                enable_web_tools=True,
                looks_like_open_url_suggestion=lambda _text: False,
                should_apply_web_refusal_fallback_cb=lambda _text: True,
                should_force_web_fallback_cb=lambda _prompt, _text: True,
                try_open_page_content_fallback=lambda: "",
                try_browser_search_fallback=_browser_fallback(calls),
                try_web_search_fallback=_web_search_fallback(calls, ""),
                try_web_fetch_fallback=_web_fetch_fallback(calls, ""),
                log_web_fallback_event=None,
            )

        result = asyncio.run(_run())
        assert result == "I don't have web browsing capabilities."
        assert calls == expected_order


def test_web_lookup_fallbacks_strip_slash_controls_and_log_repairs() -> None:
    events: list[str] = []
    captures: dict[str, Any] = {}

    class _Registry:
        def __init__(self) -> None:
            self.web_search_payloads: list[dict[str, Any]] = []
            self.browser_payloads: list[dict[str, Any]] = []

        def has(self, name: str) -> bool:
            return name in {"web_search", "web_fetch", "gui_web_run_steps"}

        async def execute(self, name: str, params: dict[str, Any]) -> str:
            captures.setdefault(name, []).append(dict(params))
            if name == "web_search":
                return "Results for: check today's weather"
            if name == "gui_web_run_steps":
                return (
                    '{"ok": true, "results": [{"action": "get_text", "result": '
                    '{"text": "Current conditions in Ithaca NY: 39 F"}}]}'
                )
            if name == "web_fetch":
                return '{"text": "Current conditions in Ithaca NY: 39 F", "ok": true}'
            return ""

    registry = _Registry()

    async def _run() -> None:
        text = await try_web_search_fallback(
            prompt="/skill weather\nCheck today's weather",
            tools=registry,  # type: ignore[arg-type]
            emit_progress=events.append,
        )
        assert "Results for:" in text

        text = await try_browser_search_fallback(
            prompt="/skill weather\nCheck today's weather",
            tools=registry,  # type: ignore[arg-type]
            emit_progress=events.append,
            build_summary=lambda value, **kwargs: value,
        )
        assert "Web lookup via embedded browser" in text

        text = await try_web_fetch_fallback(
            prompt="/skill weather\nhttps://example.com/weather",
            tools=registry,  # type: ignore[arg-type]
            candidate_urls_for_prompt=lambda prompt: [prompt]
            if "example.com" in prompt
            else [],
            build_summary=lambda value, **kwargs: value,
            emit_progress=events.append,
        )
        assert "Summary of https://example.com/weather" in text

    asyncio.run(_run())
    assert any("Repairing web_search prompt context" in item for item in events)
    assert any("Repairing browser search prompt context" in item for item in events)
    assert any("Repairing web_fetch prompt context" in item for item in events)
    assert captures["web_search"][0]["query"] == "Check today's weather"
    browser_steps = captures["gui_web_run_steps"][0]["steps"]
    assert "Check+today%27s+weather" in browser_steps[0]["url"]
    assert captures["web_fetch"][0]["url"] == "https://example.com/weather"


def test_shared_web_prompt_normalizer_strips_control_lines() -> None:
    normalized, repaired = normalize_web_lookup_prompt(
        "/skill weather\n/tool cron\nCheck today's weather"
    )
    assert repaired is True
    assert normalized == "Check today's weather"

    derived, repaired_from_messages = derive_web_lookup_prompt_from_messages(
        [
            {
                "role": "user",
                "content": "/skill weather\nCheck today's weather",
            }
        ]
    )
    assert repaired_from_messages is True
    assert derived == "Check today's weather"


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
