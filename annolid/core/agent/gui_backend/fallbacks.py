from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus

from annolid.core.agent.gui_backend.heuristics import (
    EMBEDDED_SEARCH_SOURCE,
    EMBEDDED_SEARCH_URL_TEMPLATE,
    classify_unresolved_tool_promise,
)
from annolid.core.agent.web_prompt_utils import normalize_web_lookup_prompt
from annolid.core.agent.tools import FunctionToolRegistry
from annolid.core.agent.tools.web_runtime import sanitize_web_error


_LOCAL_SEARCH_STOPWORDS = {
    "about",
    "annolid",
    "check",
    "codebase",
    "could",
    "feature",
    "features",
    "find",
    "handle",
    "handles",
    "how",
    "improve",
    "like",
    "look",
    "recent",
    "repo",
    "repository",
    "search",
    "source",
    "tell",
    "that",
    "the",
    "this",
    "what",
    "where",
    "will",
    "with",
}


@dataclass(frozen=True)
class WebFallbackResult:
    """Structured outcome from one live-web recovery step."""

    step: str
    status: str
    text: str = ""
    detail: str = ""

    @property
    def attempted(self) -> bool:
        return self.status not in {"unavailable", "not_applicable"}

    @property
    def succeeded(self) -> bool:
        return self.status == "success" and bool(self.text)


def _web_fallback_result(
    step: str,
    status: str,
    *,
    text: str = "",
    detail: object = "",
) -> WebFallbackResult:
    return WebFallbackResult(
        step=step,
        status=status,
        text=str(text or "").strip(),
        detail=sanitize_web_error(detail, limit=300).strip() if detail else "",
    )


def _derive_local_search_query(prompt: str, fallback_text: str = "") -> str:
    combined = f"{prompt}\n{fallback_text}"
    for pattern in (r"`([^`]{2,80})`", r"['\"]([^'\"]{2,80})['\"]"):
        for match in re.findall(pattern, combined):
            query = " ".join(str(match).split()).strip()
            if query:
                return query
    symbolic = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*(?:[_\-.][a-zA-Z0-9]+)+\b", combined)
    if symbolic:
        return symbolic[0].strip(".,:;")
    lowered = combined.lower()
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", lowered)
        if token not in _LOCAL_SEARCH_STOPWORDS
    ]
    if not tokens:
        return ""
    return tokens[0]


def _format_local_search_payload(payload: Dict[str, Any]) -> str:
    results = payload.get("results")
    if not isinstance(results, list):
        return ""
    lines = []
    for row in results[:8]:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "").strip()
        line = int(row.get("line") or 0)
        text = str(row.get("text") or "").strip()
        if not path or not line:
            continue
        lines.append(f"- {path}:{line}: {text[:180]}")
    if not lines:
        return ""
    query = str(payload.get("query") or "").strip()
    count = int(payload.get("count") or len(lines))
    truncated = " (truncated)" if bool(payload.get("truncated")) else ""
    return (
        f"I searched the local workspace for `{query}` and found {count} match(es)"
        f"{truncated}:\n" + "\n".join(lines)
    )


async def try_local_search_fallback(
    *,
    prompt: str,
    fallback_text: str,
    tools: Optional[FunctionToolRegistry],
    emit_progress: Callable[[str], None],
) -> str:
    registry = tools
    if registry is None or not registry.has("code_search"):
        return ""
    intent = classify_unresolved_tool_promise(fallback_text)
    if intent is None or intent.kind != "local_search":
        return ""
    query = _derive_local_search_query(prompt, fallback_text)
    if not query:
        return ""
    try:
        emit_progress("Converting local-search promise into code_search")
        payload_raw = await registry.execute(
            "code_search",
            {
                "query": query,
                "path": ".",
                "glob": "*",
                "max_results": 8,
                "context_lines": 0,
            },
        )
    except Exception:
        return ""
    try:
        payload = json.loads(str(payload_raw or "{}"))
    except Exception:
        return ""
    if not isinstance(payload, dict) or payload.get("error"):
        return ""
    return _format_local_search_payload(payload)


async def try_code_search_fallback(
    *,
    prompt: str,
    fallback_text: str,
    tools: Optional[FunctionToolRegistry],
    emit_progress: Callable[[str], None],
) -> str:
    return await try_local_search_fallback(
        prompt=prompt,
        fallback_text=fallback_text,
        tools=tools,
        emit_progress=emit_progress,
    )


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
    result = await try_web_fetch_fallback_result(
        prompt=prompt,
        tools=tools,
        candidate_urls_for_prompt=candidate_urls_for_prompt,
        build_summary=build_summary,
        emit_progress=emit_progress,
    )
    return result.text


async def try_web_fetch_fallback_result(
    *,
    prompt: str,
    tools: Optional[FunctionToolRegistry],
    candidate_urls_for_prompt: Callable[[str], List[str]],
    build_summary: Callable[..., str],
    emit_progress: Callable[[str], None],
) -> WebFallbackResult:
    registry = tools
    if registry is None:
        return _web_fallback_result(
            "web_fetch", "unavailable", detail="Tool registry is unavailable."
        )
    if not registry.has("web_fetch"):
        return _web_fallback_result(
            "web_fetch", "unavailable", detail="web_fetch is not registered."
        )
    normalized_prompt, repaired = normalize_web_lookup_prompt(prompt)
    if repaired:
        emit_progress("Repairing web_fetch prompt context")
    urls = candidate_urls_for_prompt(normalized_prompt)
    if not urls:
        return _web_fallback_result(
            "web_fetch",
            "not_applicable",
            detail="No URL was found in the prompt or recent user history.",
        )
    target_url = urls[0]
    try:
        emit_progress("Retrying with web_fetch")
        payload_raw = await registry.execute(
            "web_fetch",
            {"url": target_url, "extractMode": "text", "maxChars": 12000},
        )
    except Exception as exc:
        return _web_fallback_result("web_fetch", "error", detail=exc)
    try:
        payload = json.loads(str(payload_raw or "{}"))
    except Exception as exc:
        return _web_fallback_result(
            "web_fetch",
            "invalid_response",
            detail=f"web_fetch returned invalid JSON: {exc}",
        )
    if not isinstance(payload, dict):
        return _web_fallback_result(
            "web_fetch",
            "invalid_response",
            detail="web_fetch returned a non-object response.",
        )
    if payload.get("error"):
        return _web_fallback_result("web_fetch", "error", detail=payload.get("error"))
    page_text = str(payload.get("text") or "").strip()
    if not page_text:
        return _web_fallback_result(
            "web_fetch", "empty", detail="web_fetch returned no page text."
        )
    summary = build_summary(page_text)
    if not summary:
        return _web_fallback_result(
            "web_fetch",
            "empty",
            detail="The fetched page did not contain summarizable text.",
        )
    source_url = str(payload.get("finalUrl") or target_url).strip() or target_url
    return _web_fallback_result(
        "web_fetch",
        "success",
        text=(
            f"Summary of {source_url}:\n{summary}\n\n"
            f"Source: {source_url}\n"
            "(Generated via web_fetch fallback after a browsing-capability refusal.)"
        ),
    )


async def try_web_search_fallback(
    *,
    prompt: str,
    tools: Optional[FunctionToolRegistry],
    emit_progress: Callable[[str], None],
) -> str:
    result = await try_web_search_fallback_result(
        prompt=prompt,
        tools=tools,
        emit_progress=emit_progress,
    )
    return result.text


async def try_web_search_fallback_result(
    *,
    prompt: str,
    tools: Optional[FunctionToolRegistry],
    emit_progress: Callable[[str], None],
) -> WebFallbackResult:
    registry = tools
    if registry is None:
        return _web_fallback_result(
            "web_search", "unavailable", detail="Tool registry is unavailable."
        )
    if not registry.has("web_search"):
        return _web_fallback_result(
            "web_search", "unavailable", detail="web_search is not registered."
        )
    query, repaired = normalize_web_lookup_prompt(prompt)
    query = " ".join(str(query or "").split()).strip()
    if not query:
        return _web_fallback_result(
            "web_search", "not_applicable", detail="The search query is empty."
        )
    if len(query) > 280:
        query = query[:280].rstrip()
    try:
        if repaired:
            emit_progress("Repairing web_search prompt context")
        emit_progress("Retrying with web_search")
        payload_raw = await registry.execute(
            "web_search",
            {"query": query, "count": 5},
        )
    except Exception as exc:
        return _web_fallback_result("web_search", "error", detail=exc)
    text = str(payload_raw or "").strip()
    if not text:
        return _web_fallback_result(
            "web_search", "empty", detail="web_search returned no output."
        )
    lowered = text.lower()
    if lowered.startswith("error:"):
        return _web_fallback_result(
            "web_search", "error", detail=text.partition(":")[2].strip()
        )
    if lowered.startswith("no results for:"):
        return _web_fallback_result("web_search", "no_results", detail=text)
    return _web_fallback_result("web_search", "success", text=text)


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
    result = await try_browser_search_fallback_result(
        prompt=prompt,
        tools=tools,
        emit_progress=emit_progress,
        build_summary=build_summary,
    )
    return result.text


async def try_browser_search_fallback_result(
    *,
    prompt: str,
    tools: Optional[FunctionToolRegistry],
    emit_progress: Callable[[str], None],
    build_summary: Callable[..., str],
) -> WebFallbackResult:
    registry = tools
    if registry is None:
        return _web_fallback_result(
            "browser", "unavailable", detail="Tool registry is unavailable."
        )
    if not registry.has("gui_web_run_steps"):
        return _web_fallback_result(
            "browser",
            "unavailable",
            detail="Embedded browser automation is not registered.",
        )
    query, repaired = normalize_web_lookup_prompt(prompt)
    query = " ".join(str(query or "").split()).strip()
    if not query:
        return _web_fallback_result(
            "browser", "not_applicable", detail="The browser search query is empty."
        )
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
        if repaired:
            emit_progress("Repairing browser search prompt context")
        emit_progress("Retrying with browser search workflow")
        payload_raw = await registry.execute(
            "gui_web_run_steps",
            {"steps": steps, "stop_on_error": True, "max_steps": 12},
        )
    except Exception as exc:
        return _web_fallback_result("browser", "error", detail=exc)
    try:
        payload = json.loads(str(payload_raw or "{}"))
    except Exception as exc:
        return _web_fallback_result(
            "browser",
            "invalid_response",
            detail=f"Embedded browser returned invalid JSON: {exc}",
        )
    if not isinstance(payload, dict):
        return _web_fallback_result(
            "browser",
            "invalid_response",
            detail="Embedded browser returned a non-object response.",
        )
    if payload.get("error"):
        return _web_fallback_result("browser", "error", detail=payload.get("error"))
    if not bool(payload.get("ok")):
        return _web_fallback_result(
            "browser",
            "error",
            detail="Embedded browser workflow returned ok=false.",
        )
    page_text = extract_page_text_from_web_steps(payload)
    if not page_text:
        return _web_fallback_result(
            "browser", "empty", detail="Embedded browser returned no page text."
        )
    summary = build_summary(page_text, max_sentences=8, max_chars=1400)
    if not summary:
        return _web_fallback_result(
            "browser",
            "empty",
            detail="The browser page did not contain summarizable text.",
        )
    return _web_fallback_result(
        "browser",
        "success",
        text=(
            f"Web lookup via embedded browser:\n{summary}\n\n"
            f"Source: {EMBEDDED_SEARCH_SOURCE}"
        ),
    )


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
