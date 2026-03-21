from __future__ import annotations

import re
from typing import Any, Callable, Optional, Tuple

from annolid.core.agent.web_intents import (
    classify_web_access_stage,
    has_fast_web_data_intent,
    has_live_web_intent,
    tokenize_intent_text,
)


def should_apply_web_refusal_fallback(
    text: str,
    *,
    looks_like_web_access_refusal: Callable[[str], bool],
    looks_like_knowledge_gap_response: Callable[[str], bool],
) -> bool:
    return looks_like_web_access_refusal(text) or looks_like_knowledge_gap_response(
        text
    )


def apply_pdf_response_fallback(
    text: str,
    *,
    prompt: str,
    tool_run_count: int,
    looks_like_local_access_refusal: Callable[[str], bool],
    looks_like_open_pdf_suggestion: Callable[[str], bool],
    looks_like_pdf_read_promise: Callable[[str], bool],
    looks_like_pdf_phrase_miss_response: Callable[[str], bool],
    looks_like_pdf_summary_request: Callable[[str], bool],
    try_open_pdf_content_fallback: Callable[[], str],
) -> str:
    needs_pdf_read_recovery = bool(
        not str(text or "").strip() and looks_like_pdf_summary_request(prompt)
    )
    has_pdf_refusal_like_text = bool(
        looks_like_local_access_refusal(text)
        or looks_like_open_pdf_suggestion(text)
        or looks_like_pdf_read_promise(text)
        or looks_like_pdf_phrase_miss_response(text)
    )
    if (
        tool_run_count > 0
        and not needs_pdf_read_recovery
        and not has_pdf_refusal_like_text
    ):
        return text
    if not needs_pdf_read_recovery and not has_pdf_refusal_like_text:
        return text
    open_pdf_fallback = try_open_pdf_content_fallback()
    return open_pdf_fallback or text


def apply_empty_ollama_recovery(
    text: str,
    *,
    provider: str,
    model: str,
    used_direct_gui_fallback: bool,
    direct_gui_text: str,
    recover_with_plain_ollama_reply: Callable[[], str],
    ollama_mark_plain_mode: Callable[[str], None],
) -> Tuple[str, bool]:
    if text or provider != "ollama":
        return text, False
    # Final safety net: if the model still returns empty after our in-call retry,
    # attempt a single plain Ollama stream request (no tools) and use it.
    if used_direct_gui_fallback and direct_gui_text:
        return direct_gui_text, False
    recovered_text = recover_with_plain_ollama_reply()
    used_recovery = bool(recovered_text)
    if used_recovery:
        ollama_mark_plain_mode(model)
    return recovered_text, used_recovery


def ensure_non_empty_final_text(text: str, *, provider: str, model: str) -> str:
    if text:
        return text
    provider_name = str(provider or "").strip().lower()
    if provider_name == "ollama":
        suggestion = (
            "Please retry once. If this persists, increase Ollama timeout values in "
            "AI Model Settings and verify the local Ollama server is responsive."
        )
    else:
        suggestion = (
            "Please retry, then try a different model/provider if this persists."
        )
    return (
        "Model returned empty output after multiple attempts. "
        f"Provider={provider}, model={model}. " + suggestion
    )


def sanitize_final_response_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"<think>[\s\S]*?</think>", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"<\|tool_calls_section_begin\|>[\s\S]*?<\|tool_calls_section_end\|>",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<\|tool_call_begin\|>[\s\S]*?<\|tool_call_end\|>", "", value)
    value = re.sub(r"<\|tool_call_argument_begin\|>", "", value)
    value = re.sub(r"<\|tool_calls_section_(?:begin|end)\|>", "", value)
    value = re.sub(r"<\|tool_call_(?:begin|end)\|>", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


async def apply_direct_gui_fallback(
    *,
    text: str,
    provider: str,
    tool_run_count: int,
    prompt: str,
    execute_direct_gui_command: Callable[[str], Any],
    looks_like_local_access_refusal: Callable[[str], bool],
    logger: Any,
    session_id: str,
    model: str,
) -> Tuple[str, bool, str]:
    should_try = tool_run_count == 0 or looks_like_local_access_refusal(text)
    if not should_try:
        return text, False, ""
    direct_gui_text = await execute_direct_gui_command(prompt)
    used_direct_gui_fallback = bool(direct_gui_text)
    if used_direct_gui_fallback:
        logger.info(
            "annolid-bot direct gui fallback executed session=%s model=%s",
            session_id,
            model,
        )
        if not text or looks_like_local_access_refusal(text):
            return direct_gui_text, True, direct_gui_text
    return text, used_direct_gui_fallback, direct_gui_text


async def apply_web_response_fallbacks(
    *,
    text: str,
    prompt: str,
    tools: Optional[Any],
    tool_run_count: int,
    enable_web_tools: bool,
    looks_like_open_url_suggestion: Callable[[str], bool],
    should_apply_web_refusal_fallback_cb: Callable[[str], bool],
    should_force_web_fallback_cb: Callable[[str, str], bool],
    try_open_page_content_fallback: Callable[[], str],
    try_browser_search_fallback: Callable[[str, Optional[Any]], Any],
    try_web_search_fallback: Callable[[str, Optional[Any]], Any],
    try_web_fetch_fallback: Callable[[str, Optional[Any]], Any],
    log_web_fallback_event: Optional[Callable[[str, str, str], None]] = None,
) -> str:
    def _log(step: str, outcome: str) -> None:
        if log_web_fallback_event:
            log_web_fallback_event(access_stage, step, outcome)

    result = text
    if tool_run_count > 0:
        return result
    prompt_tokens = tokenize_intent_text(prompt)
    prompt_needs_live_web = has_live_web_intent(prompt_tokens)
    prompt_prefers_fast_web_tools = has_fast_web_data_intent(prompt_tokens)
    access_stage = classify_web_access_stage(prompt_tokens)
    _log(
        "decision",
        (
            "start"
            f" live_web={int(prompt_needs_live_web)}"
            f" fast_data={int(prompt_prefers_fast_web_tools)}"
            f" has_text={int(bool(str(result or '').strip()))}"
        ),
    )
    should_force_fallback = bool(
        should_force_web_fallback_cb(prompt, result)
        or (prompt_needs_live_web and not str(result or "").strip())
    )
    if enable_web_tools and looks_like_open_url_suggestion(result):
        open_page_fallback = try_open_page_content_fallback()
        if open_page_fallback:
            _log("open_page_suggestion", "hit")
            result = open_page_fallback
        else:
            _log("open_page_suggestion", "miss")
    if not should_force_fallback and not should_apply_web_refusal_fallback_cb(result):
        _log("decision", "skip_no_refusal_or_force")
        return result
    open_page_fallback = try_open_page_content_fallback()
    if open_page_fallback:
        _log("open_page_refusal", "hit")
        return open_page_fallback
    _log("open_page_refusal", "miss")
    if enable_web_tools and (
        prompt_prefers_fast_web_tools or access_stage == "discover"
    ):
        search_fallback = await try_web_search_fallback(prompt, tools)
        if search_fallback:
            _log("web_search", "hit")
            return search_fallback
        _log("web_search", "miss")
        web_fallback = await try_web_fetch_fallback(prompt, tools)
        if web_fallback:
            _log("web_fetch", "hit")
            return web_fallback
        _log("web_fetch", "miss")
    if access_stage == "read" and enable_web_tools:
        web_fallback = await try_web_fetch_fallback(prompt, tools)
        if web_fallback:
            _log("web_fetch", "hit")
            return web_fallback
        _log("web_fetch", "miss")
        search_fallback = await try_web_search_fallback(prompt, tools)
        if search_fallback:
            _log("web_search", "hit")
            return search_fallback
        _log("web_search", "miss")
    # Browser MCP fallback is allowed even when web_search/web_fetch are disabled.
    browser_fallback = await try_browser_search_fallback(prompt, tools)
    if browser_fallback:
        _log("browser", "hit")
        return browser_fallback
    _log("browser", "miss")
    if access_stage == "interact" and enable_web_tools:
        # For interactive requests, fall back to data tools if browser path is unavailable.
        search_fallback = await try_web_search_fallback(prompt, tools)
        if search_fallback:
            _log("web_search", "hit")
            return search_fallback
        _log("web_search", "miss")
        web_fallback = await try_web_fetch_fallback(prompt, tools)
        if web_fallback:
            _log("web_fetch", "hit")
            return web_fallback
        _log("web_fetch", "miss")
    if enable_web_tools:
        search_fallback = await try_web_search_fallback(prompt, tools)
        if search_fallback:
            _log("web_search", "hit")
            return search_fallback
        _log("web_search", "miss")
        web_fallback = await try_web_fetch_fallback(prompt, tools)
        if web_fallback:
            _log("web_fetch", "hit")
            return web_fallback
        _log("web_fetch", "miss")
    _log("decision", "return_original")
    return result
