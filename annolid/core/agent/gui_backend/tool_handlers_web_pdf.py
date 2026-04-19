from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List

from qtpy import QtCore


def web_get_dom_text(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    max_chars: int = 8000,
) -> Dict[str, Any]:
    limit = max(200, min(int(max_chars or 8000), 200000))
    payload = invoke_widget_json_slot("bot_web_get_dom_text", QtCore.Q_ARG(int, limit))
    if "max_chars" not in payload:
        payload["max_chars"] = limit
    return payload


def web_get_state(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    return invoke_widget_json_slot("bot_web_get_state")


def web_capture_screenshot(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    max_width: int = 1600,
) -> Dict[str, Any]:
    limit = max(320, min(int(max_width or 1600), 4096))
    payload = invoke_widget_json_slot(
        "bot_web_capture_screenshot",
        QtCore.Q_ARG(int, limit),
    )
    if "max_width" not in payload:
        payload["max_width"] = limit
    return payload


def web_describe_view(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    max_width: int = 1600,
) -> Dict[str, Any]:
    limit = max(320, min(int(max_width or 1600), 4096))
    payload = invoke_widget_json_slot(
        "bot_web_describe_view",
        QtCore.Q_ARG(int, limit),
    )
    if "max_width" not in payload:
        payload["max_width"] = limit
    return payload


def web_extract_structured(
    *,
    get_state: Callable[[], Dict[str, Any]],
    get_dom_text: Callable[[int], Dict[str, Any]],
    fields: List[str] | None = None,
    regex_overrides: Dict[str, str] | None = None,
    selector_hints: Dict[str, str] | None = None,
    extraction_mode: str = "auto",
    max_chars: int = 9000,
    include_excerpt: bool = True,
) -> Dict[str, Any]:
    requested_fields = [
        str(item).strip() for item in (fields or []) if str(item).strip()
    ]
    if not requested_fields:
        requested_fields = ["title", "summary"]
    limit = max(200, min(int(max_chars or 9000), 50000))

    state = dict(get_state() or {})
    if not bool(state.get("ok")) or not bool(state.get("has_page")):
        return {"ok": False, "error": "No active embedded web page"}

    text_payload = dict(get_dom_text(limit) or {})
    if not bool(text_payload.get("ok")):
        return {
            "ok": False,
            "error": str(text_payload.get("error") or "Failed to read page text"),
        }
    raw_text = str(text_payload.get("text") or "").strip()
    if not raw_text:
        return {"ok": False, "error": "Active page has no readable text content"}

    normalized = " ".join(raw_text.split()).strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    line_index = [line.lower() for line in lines]
    fields_out: Dict[str, str] = {}
    overrides = dict(regex_overrides or {})
    hints = {
        str(k).strip(): str(v).strip()
        for k, v in dict(selector_hints or {}).items()
        if str(k).strip() and str(v).strip()
    }
    mode = str(extraction_mode or "auto").strip().lower()
    if mode not in {"auto", "regex", "hint"}:
        mode = "auto"

    def _extract_from_hint(hint: str) -> str:
        # Accept CSS-like hint text and map it into robust keyword matching.
        hint_tokens = re.findall(r"[a-z0-9]+", str(hint or "").lower())
        hint_tokens = [tok for tok in hint_tokens if len(tok) > 2]
        if not hint_tokens:
            return ""
        for idx, line in enumerate(line_index):
            if all(tok in line for tok in hint_tokens):
                candidate = lines[idx]
                if ":" in candidate:
                    candidate = candidate.split(":", 1)[1].strip() or candidate
                return candidate[:500].strip()
        return ""

    for field in requested_fields:
        key = str(field).strip()
        lower_key = key.lower()
        if lower_key in {"title", "page_title"}:
            value = str(text_payload.get("title") or state.get("title") or "").strip()
            if value:
                fields_out[key] = value
                continue
        if lower_key in {"url", "source", "page_url"}:
            value = str(text_payload.get("url") or state.get("url") or "").strip()
            if value:
                fields_out[key] = value
                continue
        if lower_key in {"summary", "excerpt"}:
            snippet = normalized[:500].strip()
            if snippet:
                fields_out[key] = snippet
                continue

        override = str(overrides.get(key, "") or "").strip()
        if override and mode in {"auto", "regex"}:
            try:
                match = re.search(override, raw_text, flags=re.IGNORECASE | re.DOTALL)
            except re.error:
                match = None
            if match:
                picked = str(
                    match.group(1) if match.groups() else match.group(0)
                ).strip()
                if picked:
                    fields_out[key] = " ".join(picked.split())
                    continue

        hint = str(hints.get(key, "") or "").strip()
        if hint and mode in {"auto", "hint"}:
            picked = _extract_from_hint(hint)
            if picked:
                fields_out[key] = picked
                continue

        key_tokens = [
            tok for tok in re.findall(r"[a-z0-9]+", lower_key) if len(tok) > 2
        ]
        best_line = ""
        for idx, line in enumerate(line_index):
            if key_tokens and not any(tok in line for tok in key_tokens):
                continue
            best_line = lines[idx]
            break
        if not best_line and lines:
            best_line = lines[0]
        fields_out[key] = best_line[:500].strip()

    payload: Dict[str, Any] = {
        "ok": True,
        "url": str(text_payload.get("url") or state.get("url") or "").strip(),
        "title": str(text_payload.get("title") or state.get("title") or "").strip(),
        "fields": fields_out,
        "max_chars": limit,
    }
    if include_excerpt:
        payload["excerpt"] = normalized[:1200]
    return payload


def web_click(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    selector: str,
) -> Dict[str, Any]:
    value = str(selector or "").strip()
    if not value:
        return {"ok": False, "error": "selector is required"}
    payload = invoke_widget_json_slot("bot_web_click", QtCore.Q_ARG(str, value))
    if "selector" not in payload:
        payload["selector"] = value
    return payload


def web_type(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    selector: str,
    text: str,
    submit: bool = False,
) -> Dict[str, Any]:
    selector_text = str(selector or "").strip()
    if not selector_text:
        return {"ok": False, "error": "selector is required"}
    payload = invoke_widget_json_slot(
        "bot_web_type",
        QtCore.Q_ARG(str, selector_text),
        QtCore.Q_ARG(str, str(text or "")),
        QtCore.Q_ARG(bool, bool(submit)),
    )
    if "selector" not in payload:
        payload["selector"] = selector_text
    return payload


def web_scroll(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    delta_y: int = 800,
) -> Dict[str, Any]:
    value = int(delta_y or 0)
    payload = invoke_widget_json_slot("bot_web_scroll", QtCore.Q_ARG(int, value))
    if "delta_y" not in payload and "deltaY" not in payload:
        payload["delta_y"] = value
    return payload


def web_find_forms(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    return invoke_widget_json_slot("bot_web_find_forms")


def web_save_current(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    return invoke_widget_json_slot("bot_web_save_current")


async def web_run_steps(
    *,
    steps: Any,
    stop_on_error: bool,
    max_steps: int,
    open_url: Callable[[str], Awaitable[Dict[str, Any]]],
    open_in_browser: Callable[[str], Dict[str, Any]],
    get_dom_text: Callable[[int], Dict[str, Any]],
    click: Callable[[str], Dict[str, Any]],
    type_text: Callable[[str, str, bool], Dict[str, Any]],
    scroll: Callable[[int], Dict[str, Any]],
    find_forms: Callable[[], Dict[str, Any]],
    capture_screenshot: Callable[[int], Dict[str, Any]],
    describe_view: Callable[[int], Dict[str, Any]],
    sleep_ms: Callable[[int], None],
) -> Dict[str, Any]:
    if not isinstance(steps, list) or not steps:
        return {"ok": False, "error": "steps must be a non-empty list"}
    limit = max(1, min(int(max_steps or 12), 50))
    if len(steps) > limit:
        return {
            "ok": False,
            "error": f"Too many steps ({len(steps)}), max_steps={limit}",
        }

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    halt_on_error = bool(stop_on_error)

    for idx, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            payload = {"ok": False, "error": "step must be an object"}
            results.append({"index": idx, "action": "", "result": payload})
            errors.append({"index": idx, "error": payload["error"]})
            if halt_on_error:
                break
            continue

        action = str(raw_step.get("action") or "").strip().lower()
        if not action:
            payload = {"ok": False, "error": "step.action is required"}
            results.append({"index": idx, "action": action, "result": payload})
            errors.append({"index": idx, "error": payload["error"]})
            if halt_on_error:
                break
            continue

        if action == "open_url":
            payload = await open_url(str(raw_step.get("url") or ""))
        elif action == "open_in_browser":
            payload = open_in_browser(str(raw_step.get("url") or ""))
        elif action in {"get_text", "dom_text", "snapshot"}:
            payload = get_dom_text(int(raw_step.get("max_chars") or 8000))
        elif action == "click":
            payload = click(str(raw_step.get("selector") or ""))
        elif action == "type":
            payload = type_text(
                str(raw_step.get("selector") or ""),
                str(raw_step.get("text") or ""),
                bool(raw_step.get("submit", False)),
            )
        elif action == "scroll":
            payload = scroll(int(raw_step.get("delta_y") or 800))
        elif action == "find_forms":
            payload = find_forms()
        elif action in {"capture_screenshot", "screenshot"}:
            payload = capture_screenshot(int(raw_step.get("max_width") or 1600))
        elif action in {"describe_view", "describe_screenshot"}:
            payload = describe_view(int(raw_step.get("max_width") or 1600))
        elif action == "wait":
            wait_ms = max(0, min(int(raw_step.get("wait_ms") or 500), 60000))
            sleep_ms(wait_ms)
            payload = {"ok": True, "wait_ms": wait_ms}
        else:
            payload = {"ok": False, "error": f"Unsupported action: {action}"}

        results.append({"index": idx, "action": action, "result": payload})
        if not bool(payload.get("ok")):
            errors.append(
                {
                    "index": idx,
                    "action": action,
                    "error": str(payload.get("error") or "step failed"),
                }
            )
            if halt_on_error:
                break

    return {
        "ok": len(errors) == 0,
        "steps_requested": len(steps),
        "steps_run": len(results),
        "stop_on_error": halt_on_error,
        "results": results,
        "errors": errors,
    }


def pdf_get_state(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    return invoke_widget_json_slot("bot_pdf_get_state")


def pdf_get_text(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    max_chars: int = 8000,
    pages: int = 2,
    start_page: int = 0,
    path: str = "",
) -> Dict[str, Any]:
    del path
    limit = max(200, min(int(max_chars or 8000), 200000))
    pages_limit = max(1, min(int(pages or 2), 5))
    start_page_value = max(0, int(start_page or 0))
    payload = invoke_widget_json_slot(
        "bot_pdf_get_text",
        QtCore.Q_ARG(int, limit),
        QtCore.Q_ARG(int, pages_limit),
        QtCore.Q_ARG(int, start_page_value),
    )
    if "max_chars" not in payload:
        payload["max_chars"] = limit
    if "pages" not in payload:
        payload["pages"] = pages_limit
    if "start_page" not in payload:
        payload["start_page"] = start_page_value
    return payload


def pdf_find_sections(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
    max_sections: int = 20,
    max_pages: int = 12,
) -> Dict[str, Any]:
    sections_limit = max(1, min(int(max_sections or 20), 200))
    pages_limit = max(1, min(int(max_pages or 12), 100))
    payload = invoke_widget_json_slot(
        "bot_pdf_find_sections",
        QtCore.Q_ARG(int, sections_limit),
        QtCore.Q_ARG(int, pages_limit),
    )
    if "max_sections" not in payload:
        payload["max_sections"] = sections_limit
    if "max_pages" not in payload:
        payload["max_pages"] = pages_limit
    return payload
