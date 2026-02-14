from __future__ import annotations

from typing import Tuple

from qtpy import QtWidgets


_MAX_EXPLAIN_CHARS = 4000


def explain_selection_with_annolid_bot(
    owner: QtWidgets.QWidget,
    selected_text: str,
    *,
    source_hint: str = "",
) -> Tuple[bool, str]:
    text = " ".join(str(selected_text or "").split()).strip()
    if not text:
        return False, "No selected text to explain."
    if len(text) > _MAX_EXPLAIN_CHARS:
        text = text[:_MAX_EXPLAIN_CHARS].rstrip() + "…"

    host = owner.window() if owner is not None else None
    if host is None:
        return False, "Unable to locate host window."

    # Ensure the bot dock exists/visible.
    open_bot = getattr(host, "open_annolid_bot_dock", None)
    if callable(open_bot):
        try:
            open_bot()
        except Exception:
            pass

    manager = getattr(host, "ai_chat_manager", None)
    if manager is None:
        return False, "Annolid Bot is unavailable in this session."
    widget = getattr(manager, "ai_chat_widget", None)
    if widget is None and hasattr(manager, "show_chat_dock"):
        try:
            manager.show_chat_dock()
        except Exception:
            pass
        widget = getattr(manager, "ai_chat_widget", None)
    if widget is None:
        return False, "Unable to open Annolid Bot."
    if bool(getattr(widget, "is_streaming_chat", False)):
        return False, "Annolid Bot is currently responding. Please try again."
    prompt_input = getattr(widget, "prompt_text_edit", None)
    send_chat = getattr(widget, "chat_with_model", None)
    if prompt_input is None or not callable(send_chat):
        return False, "Annolid Bot input is unavailable."

    prefix = "Please explain the following selected text"
    if source_hint:
        prefix += f" (source: {source_hint})"
    prompt = f"{prefix} in clear, concise terms:\n\n{text}"

    try:
        prompt_input.setPlainText(prompt)
        prompt_input.setFocus()
        send_chat()
    except Exception as exc:
        return False, f"Failed to send prompt to Annolid Bot: {exc}"
    return True, "Sent selected text to Annolid Bot for explanation."
