from __future__ import annotations

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
) -> Dict[str, Any]:
    limit = max(200, min(int(max_chars or 8000), 200000))
    pages_limit = max(1, min(int(pages or 2), 5))
    payload = invoke_widget_json_slot(
        "bot_pdf_get_text",
        QtCore.Q_ARG(int, limit),
        QtCore.Q_ARG(int, pages_limit),
    )
    if "max_chars" not in payload:
        payload["max_chars"] = limit
    if "pages" not in payload:
        payload["pages"] = pages_limit
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
