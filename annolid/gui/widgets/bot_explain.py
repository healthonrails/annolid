from __future__ import annotations

from typing import Tuple

from qtpy import QtWidgets


_MAX_EXPLAIN_CHARS = 4000


def _resolve_chat_widget(owner: QtWidgets.QWidget):
    host = owner.window() if owner is not None else None
    if host is None:
        return None, "Unable to locate host window."

    open_bot = getattr(host, "open_annolid_bot_dock", None)
    if callable(open_bot):
        try:
            open_bot()
        except Exception:
            pass

    manager = getattr(host, "ai_chat_manager", None)
    if manager is None:
        return None, "Annolid Bot is unavailable in this session."
    widget = getattr(manager, "ai_chat_widget", None)
    if widget is None and hasattr(manager, "show_chat_dock"):
        try:
            manager.show_chat_dock()
        except Exception:
            pass
        widget = getattr(manager, "ai_chat_widget", None)
    if widget is None:
        return None, "Unable to open Annolid Bot."
    return widget, ""


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

    widget, err = _resolve_chat_widget(owner)
    if widget is None:
        return False, err or "Unable to open Annolid Bot."
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


def explain_image_with_annolid_bot(
    owner: QtWidgets.QWidget,
    image_path: str,
    *,
    source_hint: str = "",
    image_url: str = "",
) -> Tuple[bool, str]:
    path = str(image_path or "").strip()
    if not path:
        return False, "No image selected to explain."

    widget, err = _resolve_chat_widget(owner)
    if widget is None:
        return False, err or "Unable to open Annolid Bot."
    if bool(getattr(widget, "is_streaming_chat", False)):
        return False, "Annolid Bot is currently responding. Please try again."
    prompt_input = getattr(widget, "prompt_text_edit", None)
    send_chat = getattr(widget, "chat_with_model", None)
    set_image_path = getattr(widget, "set_image_path", None)
    register_temp_image = getattr(widget, "register_managed_temp_image", None)
    set_chat_mode = getattr(widget, "set_next_chat_mode", None)
    if prompt_input is None or not callable(send_chat) or not callable(set_image_path):
        return False, "Annolid Bot input is unavailable."

    source_note = ""
    if source_hint:
        source_note = f" from: {source_hint}"
    if image_url:
        source_note += f"\nImage URL: {image_url}"
    prompt = (
        "Please describe this attached web image in clear, concise terms. "
        "Include key objects and relevant context."
        f"{source_note}"
    )

    try:
        set_image_path(path)
        if callable(register_temp_image):
            register_temp_image(path)
        if callable(set_chat_mode):
            set_chat_mode("vision_describe")
        prompt_input.setPlainText(prompt)
        prompt_input.setFocus()
        send_chat()
    except Exception as exc:
        return False, f"Failed to send image to Annolid Bot: {exc}"
    return True, "Sent selected image to Annolid Bot for description."
