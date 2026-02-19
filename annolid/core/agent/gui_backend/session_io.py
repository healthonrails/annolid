from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from qtpy import QtCore
from qtpy.QtCore import QMetaObject


def emit_chunk(*, widget: Any, chunk: str) -> None:
    QMetaObject.invokeMethod(
        widget,
        "stream_chat_chunk",
        QtCore.Qt.QueuedConnection,
        QtCore.Q_ARG(str, chunk),
    )


def emit_progress(
    *,
    widget: Any,
    update: str,
    last_progress_update: Optional[str],
) -> Optional[str]:
    if not bool(getattr(widget, "enable_progress_stream", False)):
        return last_progress_update
    text = str(update or "").strip()
    if not text or text == last_progress_update:
        return last_progress_update
    QMetaObject.invokeMethod(
        widget,
        "stream_chat_progress",
        QtCore.Qt.QueuedConnection,
        QtCore.Q_ARG(str, text),
    )
    return text


def emit_final(
    *,
    widget: Any,
    message: str,
    is_error: bool,
    emit_progress_cb,
) -> None:
    if is_error:
        emit_progress_cb("Response failed")
    else:
        emit_progress_cb("Response ready")
    QMetaObject.invokeMethod(
        widget,
        "update_chat_response",
        QtCore.Qt.QueuedConnection,
        QtCore.Q_ARG(str, message),
        QtCore.Q_ARG(bool, is_error),
    )


def load_history_messages(
    *,
    session_store: Any,
    session_id: str,
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    """Load persisted chat history as role/content records."""
    if not session_store:
        return []
    try:
        history = session_store.get_history(session_id)
    except Exception:
        return []
    cleaned: List[Dict[str, Any]] = []
    for msg in history:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        cleaned.append({"role": role, "content": text})
    keep = max(1, int(max_history_messages))
    return cleaned[-keep:]


def persist_turn(
    *,
    user_text: str,
    assistant_text: str,
    session_id: str,
    session_store: Any,
    max_history_messages: int,
    workspace_memory: Any,
    persist_session_history: bool = True,
) -> None:
    user_msg = str(user_text or "").strip()
    assistant_msg = str(assistant_text or "").strip()
    if not user_msg and not assistant_msg:
        return
    entries: List[Dict[str, str]] = []
    if user_msg:
        entries.append({"role": "user", "content": user_msg})
    if assistant_msg:
        entries.append({"role": "assistant", "content": assistant_msg})

    if persist_session_history and session_store and entries:
        try:
            session_store.append_history(
                session_id,
                entries,
                max_messages=max_history_messages,
            )
        except Exception:
            pass

    try:
        stamp = datetime.now().strftime("%H:%M:%S")
        parts: List[str] = [f"## {stamp} [{session_id}]"]
        if user_msg:
            parts.append(f"User: {user_msg}")
        if assistant_msg:
            parts.append(f"Assistant: {assistant_msg}")
        entry = "\n\n".join(parts)
        workspace_memory.append_today(entry)
        workspace_memory.append_history(entry)
    except Exception:
        pass
