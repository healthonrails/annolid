from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


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
    tool_run_count: int,
    looks_like_local_access_refusal: Callable[[str], bool],
    looks_like_open_pdf_suggestion: Callable[[str], bool],
    try_open_pdf_content_fallback: Callable[[], str],
) -> str:
    if tool_run_count > 0:
        return text
    if not (
        looks_like_local_access_refusal(text) or looks_like_open_pdf_suggestion(text)
    ):
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
    suggestion = (
        "Please switch to another Ollama model for Annolid Bot."
        if provider_name == "ollama"
        else "Please retry, then try a different model/provider if this persists."
    )
    return (
        "Model returned empty output after multiple attempts. "
        f"Provider={provider}, model={model}. " + suggestion
    )


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
    if provider != "ollama" or tool_run_count != 0:
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
    try_open_page_content_fallback: Callable[[], str],
    try_browser_search_fallback: Callable[[str, Optional[Any]], Any],
    try_web_fetch_fallback: Callable[[str, Optional[Any]], Any],
) -> str:
    result = text
    if tool_run_count > 0:
        return result
    if enable_web_tools and looks_like_open_url_suggestion(result):
        open_page_fallback = try_open_page_content_fallback()
        if open_page_fallback:
            result = open_page_fallback
    if not should_apply_web_refusal_fallback_cb(result):
        return result
    open_page_fallback = try_open_page_content_fallback()
    if open_page_fallback:
        return open_page_fallback
    # Browser MCP fallback is allowed even when web_search/web_fetch are disabled.
    browser_fallback = await try_browser_search_fallback(prompt, tools)
    if browser_fallback:
        return browser_fallback
    if enable_web_tools:
        web_fallback = await try_web_fetch_fallback(prompt, tools)
        if web_fallback:
            return web_fallback
    return result
