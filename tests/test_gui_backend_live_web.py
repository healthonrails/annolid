import asyncio

from annolid.core.agent.gui_backend.live_web import (
    contextualize_live_web_prompt,
    run_tool_first_live_web_response,
)


def test_contextualize_weather_uses_recent_history_location() -> None:
    result = contextualize_live_web_prompt(
        "weather",
        history_messages=[
            {"role": "user", "content": "weather in Ithaca NY"},
            {"role": "assistant", "content": "Ithaca weather today: 39 F."},
        ],
        memory_text="",
    )
    assert result.prompt == "weather in Ithaca NY"
    assert result.source == "location"


def test_contextualize_weather_uses_memory_location_when_history_has_none() -> None:
    result = contextualize_live_web_prompt(
        "check weather",
        history_messages=[],
        memory_text="default location: Ithaca NY",
    )
    assert result.prompt == "weather in Ithaca NY"
    assert result.source == "location"


def test_contextualize_weather_falls_back_to_near_me() -> None:
    result = contextualize_live_web_prompt(
        "weather",
        history_messages=[],
        memory_text="",
    )
    assert result.prompt == "weather near me"
    assert result.source == "near_me"


def test_weather_tool_first_failure_does_not_repeat_generic_plan() -> None:
    calls: list[str] = []

    async def _miss(name: str) -> str:
        calls.append(name)
        return ""

    async def _generic_plan(*_args) -> str:
        raise AssertionError("weather route should not repeat the generic plan")

    class _Logger:
        def info(self, *_args, **_kwargs) -> None:
            return None

    result = asyncio.run(
        run_tool_first_live_web_response(
            prompt="weather in Ithaca NY",
            tools=None,
            enable_web_tools=True,
            apply_web_response_fallbacks=_generic_plan,
            try_browser_search_fallback=lambda *_args: _miss("browser"),
            try_web_search_fallback=lambda *_args: _miss("web_search"),
            try_web_fetch_fallback=lambda *_args: _miss("web_fetch"),
            sanitize_text=lambda text: text,
            log_web_fallback_event=lambda *_args: None,
            emit_progress=lambda _message: None,
            logger=_Logger(),
            session_id="test",
            model="test-model",
        )
    )

    assert result.text == ""
    assert result.attempted is True
    assert result.route == "weather"
    assert calls == ["web_search", "browser", "web_fetch"]
