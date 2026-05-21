from annolid.core.agent.gui_backend.live_web import contextualize_live_web_prompt


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
