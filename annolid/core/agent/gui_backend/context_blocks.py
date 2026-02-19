from __future__ import annotations

from typing import Any, Callable, Dict


def build_live_web_context_prompt_block(
    *,
    get_state: Callable[[], Dict[str, Any]],
    get_dom_text: Callable[..., Dict[str, Any]],
    include_snapshot: bool = True,
) -> str:
    state = get_state()
    if not isinstance(state, dict):
        return ""
    if not bool(state.get("ok")):
        return ""
    if not bool(state.get("has_page")):
        return "No embedded web page is currently open."
    url = str(state.get("url") or "").strip()
    title = str(state.get("title") or "").strip()
    snapshot_block = "Visible text snapshot: [omitted to save tokens]"
    if include_snapshot:
        page_payload = get_dom_text(max_chars=1200)
        if isinstance(page_payload, dict) and bool(page_payload.get("ok")):
            text = str(page_payload.get("text") or "").strip()
            if len(text) > 600:
                text = text[:600].rstrip() + "\n...[truncated]"
            url = str(page_payload.get("url") or url).strip()
            title = str(page_payload.get("title") or title).strip()
            snapshot_block = f"Visible text snapshot:\n{text or '[empty]'}"
        else:
            snapshot_block = "Visible text snapshot unavailable."
    return (
        "# Active Embedded Web Page\n"
        f"URL: {url or '[unknown]'}\n"
        f"Title: {title or '[unknown]'}\n"
        f"{snapshot_block}"
    )


def build_live_pdf_context_prompt_block(
    *,
    get_state: Callable[[], Dict[str, Any]],
    get_text: Callable[..., Dict[str, Any]],
    include_snapshot: bool = True,
) -> str:
    state = get_state()
    if not isinstance(state, dict):
        return ""
    if not bool(state.get("ok")) or not bool(state.get("has_pdf")):
        return ""
    title = str(state.get("title") or "").strip()
    path = str(state.get("path") or "").strip()
    page = int(state.get("current_page") or 0)
    total = int(state.get("total_pages") or 0)
    snapshot_block = "Text snapshot: [omitted to save tokens]"
    if include_snapshot:
        payload = get_text(max_chars=1200, pages=1)
        if isinstance(payload, dict) and bool(payload.get("ok")):
            text = str(payload.get("text") or "").strip()
            if len(text) > 600:
                text = text[:600].rstrip() + "\n...[truncated]"
            title = str(payload.get("title") or title).strip()
            path = str(payload.get("path") or path).strip()
            page = int(payload.get("current_page") or page or 0)
            total = int(payload.get("total_pages") or total or 0)
            snapshot_block = f"Text snapshot:\n{text or '[empty]'}"
        else:
            snapshot_block = "Text snapshot unavailable."
    return (
        "# Active PDF\n"
        f"Path: {path or '[unknown]'}\n"
        f"Title: {title or '[unknown]'}\n"
        f"Page: {page}/{total}\n"
        f"{snapshot_block}"
    )
