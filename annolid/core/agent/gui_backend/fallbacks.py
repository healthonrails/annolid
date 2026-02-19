from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus

from annolid.core.agent.gui_backend.heuristics import (
    EMBEDDED_SEARCH_SOURCE,
    EMBEDDED_SEARCH_URL_TEMPLATE,
)
from annolid.core.agent.tools import FunctionToolRegistry


def candidate_web_urls_for_prompt(
    prompt: str,
    *,
    extract_web_urls: Callable[[str], List[str]],
    load_history_messages: Callable[[], List[Dict[str, Any]]],
) -> List[str]:
    urls = extract_web_urls(prompt)
    if urls:
        return urls
    history = load_history_messages()
    for msg in reversed(history):
        if str(msg.get("role") or "") != "user":
            continue
        content = str(msg.get("content") or "")
        if not content:
            continue
        from_msg = extract_web_urls(content)
        if from_msg:
            return from_msg
    return []


async def try_web_fetch_fallback(
    *,
    prompt: str,
    tools: Optional[FunctionToolRegistry],
    candidate_urls_for_prompt: Callable[[str], List[str]],
    build_summary: Callable[..., str],
    emit_progress: Callable[[str], None],
) -> str:
    registry = tools
    if registry is None:
        return ""
    if not registry.has("web_fetch"):
        return ""
    urls = candidate_urls_for_prompt(prompt)
    if not urls:
        return ""
    target_url = urls[0]
    try:
        emit_progress("Retrying with web_fetch")
        payload_raw = await registry.execute(
            "web_fetch",
            {"url": target_url, "extractMode": "text", "maxChars": 12000},
        )
    except Exception:
        return ""
    try:
        payload = json.loads(str(payload_raw or "{}"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("error"):
        return ""
    page_text = str(payload.get("text") or "").strip()
    if not page_text:
        return ""
    summary = build_summary(page_text)
    if not summary:
        return ""
    source_url = str(payload.get("finalUrl") or target_url).strip() or target_url
    return (
        f"Summary of {source_url}:\n{summary}\n\n"
        f"Source: {source_url}\n"
        "(Generated via web_fetch fallback after a browsing-capability refusal.)"
    )


def extract_page_text_from_web_steps(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("action") or "").lower() not in {
            "get_text",
            "dom_text",
            "snapshot",
        }:
            continue
        result_payload = item.get("result")
        if not isinstance(result_payload, dict):
            continue
        text_value = str(result_payload.get("text") or "").strip()
        if text_value:
            return text_value
    return ""


async def try_browser_search_fallback(
    *,
    prompt: str,
    tools: Optional[FunctionToolRegistry],
    emit_progress: Callable[[str], None],
    build_summary: Callable[..., str],
) -> str:
    registry = tools
    if registry is None:
        return ""
    if not registry.has("gui_web_run_steps"):
        return ""
    query = " ".join(str(prompt or "").split()).strip()
    if not query:
        return ""
    if len(query) > 280:
        query = query[:280].rstrip()
    encoded_query = quote_plus(query)
    steps = [
        {
            "action": "open_url",
            "url": EMBEDDED_SEARCH_URL_TEMPLATE.format(query=encoded_query),
        },
        {"action": "wait", "wait_ms": 1200},
        {"action": "get_text", "max_chars": 9000},
    ]
    try:
        emit_progress("Retrying with browser search workflow")
        payload_raw = await registry.execute(
            "gui_web_run_steps",
            {"steps": steps, "stop_on_error": True, "max_steps": 12},
        )
    except Exception:
        return ""
    try:
        payload = json.loads(str(payload_raw or "{}"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict) or payload.get("error"):
        return ""
    if not bool(payload.get("ok")):
        return ""
    page_text = extract_page_text_from_web_steps(payload)
    if not page_text:
        return ""
    summary = build_summary(page_text, max_sentences=8, max_chars=1400)
    if not summary:
        return ""
    return f"Web lookup via embedded browser:\n{summary}\n\nSource: {EMBEDDED_SEARCH_SOURCE}"


def try_open_page_content_fallback(
    *,
    prompt: str,
    get_state: Callable[[], Dict[str, Any]],
    get_dom_text: Callable[..., Dict[str, Any]],
    should_use_open_page_fallback: Callable[[str], bool],
    topic_tokens: Callable[[str], List[str]],
    build_summary: Callable[..., str],
) -> str:
    state = get_state()
    if not isinstance(state, dict):
        return ""
    if not bool(state.get("ok")) or not bool(state.get("has_page")):
        return ""
    if not should_use_open_page_fallback(prompt):
        prompt_tokens = set(topic_tokens(prompt))
        page_hint_text = " ".join(
            [
                str(state.get("title") or ""),
                str(state.get("url") or ""),
            ]
        )
        page_tokens = set(topic_tokens(page_hint_text))
        if not (prompt_tokens and page_tokens and (prompt_tokens & page_tokens)):
            return ""
    page_payload = get_dom_text(max_chars=9000)
    if not isinstance(page_payload, dict) or not bool(page_payload.get("ok")):
        return ""
    page_text = str(page_payload.get("text") or "").strip()
    if not page_text:
        return ""
    summary = build_summary(page_text, max_sentences=8, max_chars=1400)
    if not summary:
        return ""
    url = str(page_payload.get("url") or state.get("url") or "").strip()
    title = str(page_payload.get("title") or state.get("title") or "").strip()
    source = title or url or "active embedded web page"
    return f"Using the currently open page ({source}):\n{summary}"


def try_open_pdf_content_fallback(
    *,
    get_state: Callable[[], Dict[str, Any]],
    get_text: Callable[..., Dict[str, Any]],
    build_summary: Callable[..., str],
) -> str:
    state = get_state()
    if not isinstance(state, dict):
        return ""
    if not bool(state.get("ok")) or not bool(state.get("has_pdf")):
        return ""
    pdf_payload = get_text(max_chars=9000, pages=2)
    if not isinstance(pdf_payload, dict) or not bool(pdf_payload.get("ok")):
        return ""
    pdf_text = str(pdf_payload.get("text") or "").strip()
    if not pdf_text:
        return ""
    summary = build_summary(pdf_text, max_sentences=8, max_chars=1400)
    if not summary:
        return ""
    title = str(pdf_payload.get("title") or state.get("title") or "").strip()
    path = str(pdf_payload.get("path") or state.get("path") or "").strip()
    source = title or path or "active PDF"
    return f"Using the currently open PDF ({source}):\n{summary}"
