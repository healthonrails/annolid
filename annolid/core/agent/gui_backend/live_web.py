from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple

from annolid.core.agent.web_intents import (
    WEATHER_INTENT_TOKENS,
    has_fast_web_data_intent,
    tokenize_intent_text,
)


WebLookupCallback = Callable[[str, Optional[Any]], Awaitable[str]]


@dataclass(frozen=True)
class ToolFirstWebResult:
    text: str = ""
    attempted: bool = False
    route: str = ""
    step: str = ""


@dataclass(frozen=True)
class ContextualWebPrompt:
    prompt: str
    source: str = ""


def is_weather_prompt(prompt: str) -> bool:
    return bool(set(tokenize_intent_text(prompt)).intersection(WEATHER_INTENT_TOKENS))


def is_bare_weather_prompt(prompt: str) -> bool:
    tokens = set(tokenize_intent_text(prompt))
    return bool(tokens) and tokens <= set(WEATHER_INTENT_TOKENS)


def contextualize_live_web_prompt(
    prompt: str,
    *,
    history_messages: Sequence[Any] = (),
    memory_text: str = "",
) -> ContextualWebPrompt:
    value = str(prompt or "").strip()
    if not is_weather_prompt(value) or _weather_prompt_has_location(value):
        return ContextualWebPrompt(prompt=value)
    location = _extract_recent_weather_location(
        history_messages
    ) or _extract_memory_location(memory_text)
    if location:
        return ContextualWebPrompt(prompt=f"weather in {location}", source="location")
    if _is_under_specified_weather_prompt(value):
        return ContextualWebPrompt(prompt="weather near me", source="near_me")
    return ContextualWebPrompt(prompt=value)


def _is_under_specified_weather_prompt(prompt: str) -> bool:
    tokens = set(tokenize_intent_text(prompt))
    if not tokens.intersection(WEATHER_INTENT_TOKENS):
        return False
    return not _weather_prompt_has_location(prompt)


def _weather_prompt_has_location(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    if "near me" in text or "my location" in text:
        return True
    return bool(
        re.search(
            r"\b(?:in|for|at|near|around)\s+[a-z][a-z0-9 .,'-]{2,}\b",
            text,
        )
    )


def _extract_recent_weather_location(history_messages: Sequence[Any]) -> str:
    for message in reversed(list(history_messages)[-20:]):
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "").strip()
        if not content or not is_weather_prompt(content):
            continue
        location = _extract_location_phrase(content)
        if location:
            return location
    return ""


def _extract_memory_location(memory_text: str) -> str:
    text = str(memory_text or "")
    patterns = (
        r"\b(?:user|default|home|current)\s+location\s*[:=-]\s*([A-Za-z][A-Za-z0-9 .,'-]{2,80})",
        r"\b(?:user|default|home|current)\s+city\s*[:=-]\s*([A-Za-z][A-Za-z0-9 .,'-]{2,80})",
        r"\b(?:i live in|i am in|i'm in|my location is)\s+([A-Za-z][A-Za-z0-9 .,'-]{2,80})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_location(match.group(1))
    return ""


def _extract_location_phrase(text: str) -> str:
    match = re.search(
        r"\b(?:weather|forecast|temperature)\b[\s\S]{0,80}?\b(?:in|for|at|near|around)\s+([A-Za-z][A-Za-z0-9 .,'-]{2,80})",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b(?:in|for|at|near|around)\s+([A-Za-z][A-Za-z0-9 .,'-]{2,80})[\s\S]{0,80}?\b(?:weather|forecast|temperature)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    if not match:
        return ""
    return _clean_location(match.group(1))


def _clean_location(value: str) -> str:
    text = " ".join(str(value or "").split()).strip(" .,'\"")
    text = re.split(r"\b(?:today|tomorrow|now|please|forecast|weather)\b", text, 1)[0]
    return text.strip(" .,'\"")


def is_tool_first_live_web_prompt(prompt: str) -> bool:
    tokens = tokenize_intent_text(prompt)
    token_set = set(tokens)
    if token_set.intersection(WEATHER_INTENT_TOKENS):
        return True
    if has_fast_web_data_intent(tokens):
        return True
    if "news" in token_set:
        return True
    web_action_tokens = {"search", "check", "find", "lookup", "look", "get"}
    freshness_tokens = {"latest", "current", "today", "live"}
    return bool(
        token_set.intersection(web_action_tokens)
        and token_set.intersection(freshness_tokens)
    )


def tool_first_live_web_error_message(prompt: str) -> str:
    if is_weather_prompt(prompt) and is_bare_weather_prompt(prompt):
        return (
            "I couldn't retrieve weather from the available Annolid tools. "
            "I tried a local-context lookup first; verify browser/web tools and retry."
        )
    return (
        "I couldn't retrieve current web data with the available Annolid tools. "
        "Verify web tools/network access, then retry or switch provider if the "
        "model keeps returning empty output."
    )


async def run_tool_first_live_web_response(
    *,
    prompt: str,
    tools: Optional[Any],
    enable_web_tools: bool,
    apply_web_response_fallbacks: Callable[[str, Optional[Any], int], Awaitable[str]],
    try_browser_search_fallback: WebLookupCallback,
    try_web_search_fallback: WebLookupCallback,
    try_web_fetch_fallback: WebLookupCallback,
    sanitize_text: Callable[[str], str],
    log_web_fallback_event: Callable[[str, str, str], None],
    emit_progress: Callable[[str], None],
    logger: Any,
    session_id: str,
    model: str,
) -> ToolFirstWebResult:
    if not enable_web_tools or not is_tool_first_live_web_prompt(prompt):
        return ToolFirstWebResult()

    if is_weather_prompt(prompt):
        result = await _run_ordered_web_route(
            prompt=prompt,
            tools=tools,
            route="weather",
            steps=(
                ("browser", try_browser_search_fallback),
                ("web_search", try_web_search_fallback),
                ("web_fetch", try_web_fetch_fallback),
            ),
            sanitize_text=sanitize_text,
            log_web_fallback_event=log_web_fallback_event,
        )
        if result.text:
            logger.info(
                "annolid-bot weather skill lookup succeeded session=%s model=%s step=%s text_chars=%d",
                session_id,
                model,
                result.step,
                len(result.text),
            )
            emit_progress("Answered with weather tools")
            return result

    text = await apply_web_response_fallbacks("", tools, 0)
    text = sanitize_text(text)
    if text:
        logger.info(
            "annolid-bot tool-first live web response session=%s model=%s prompt_chars=%d text_chars=%d",
            session_id,
            model,
            len(prompt),
            len(text),
        )
        emit_progress("Answered with web tools")
        return ToolFirstWebResult(
            text=text,
            attempted=True,
            route="generic",
            step="fallback_plan",
        )
    return ToolFirstWebResult(attempted=True, route="generic")


async def _run_ordered_web_route(
    *,
    prompt: str,
    tools: Optional[Any],
    route: str,
    steps: Sequence[Tuple[str, WebLookupCallback]],
    sanitize_text: Callable[[str], str],
    log_web_fallback_event: Callable[[str, str, str], None],
) -> ToolFirstWebResult:
    for step, callback in steps:
        text = await callback(prompt, tools)
        text = sanitize_text(text)
        if text:
            log_web_fallback_event(route, step, "hit")
            return ToolFirstWebResult(
                text=text,
                attempted=True,
                route=route,
                step=step,
            )
        log_web_fallback_event(route, step, "miss")
    return ToolFirstWebResult(attempted=True, route=route)
