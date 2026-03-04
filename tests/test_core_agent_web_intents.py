from annolid.core.agent.web_intents import (
    FAST_WEB_DATA_TOKENS,
    LIVE_WEB_INTENT_TOKENS,
    has_fast_web_data_intent,
    has_live_web_intent,
    tokenize_intent_text,
)


def test_tokenize_intent_text_normalizes_case_and_symbols() -> None:
    tokens = tokenize_intent_text("Check NVDA stock price now!")
    assert "check" in tokens
    assert "nvda" in tokens
    assert "stock" in tokens
    assert "price" in tokens


def test_fast_web_data_intent_detects_market_terms() -> None:
    assert has_fast_web_data_intent("check NVDA stock price") is True
    assert has_fast_web_data_intent(["latest", "nasdaq", "quote"]) is True
    assert has_fast_web_data_intent("summarize this local file") is False


def test_live_web_intent_detects_weather_and_news_terms() -> None:
    assert has_live_web_intent("weather in Ithaca today") is True
    assert has_live_web_intent(["latest", "news"]) is True
    assert has_live_web_intent("rename this file") is False


def test_intent_token_sets_include_expected_keywords() -> None:
    assert "weather" in LIVE_WEB_INTENT_TOKENS
    assert "price" in LIVE_WEB_INTENT_TOKENS
    assert "price" in FAST_WEB_DATA_TOKENS
    assert "ticker" in FAST_WEB_DATA_TOKENS
