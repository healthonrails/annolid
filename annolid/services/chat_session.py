"""Service helpers for GUI chat session storage and session IO."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.gui_backend.session_io import (
    emit_chunk as gui_emit_chunk,
    emit_final as gui_emit_final,
    emit_progress as gui_emit_progress,
    load_history_messages as gui_load_history_messages,
    persist_turn as gui_persist_turn,
)

_SESSION_STORE: PersistentSessionStore | None = None


def get_chat_session_store() -> PersistentSessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = PersistentSessionStore(AgentSessionManager())
    return _SESSION_STORE


def clear_chat_session(
    session_id: str,
    *,
    session_store: PersistentSessionStore | None = None,
) -> None:
    store = session_store or get_chat_session_store()
    store.clear_session(str(session_id or "gui:annolid_bot:default"))


def delete_chat_history_message(
    session_id: str,
    *,
    message_id: str = "",
    history_index: int = -1,
    expected_role: str = "",
    expected_content: str = "",
    session_store: PersistentSessionStore | None = None,
) -> bool:
    store = session_store or get_chat_session_store()
    return bool(
        store.delete_history_message(
            str(session_id or "gui:annolid_bot:default"),
            message_id=str(message_id or ""),
            history_index=int(history_index),
            expected_role=str(expected_role or ""),
            expected_content=str(expected_content or ""),
        )
    )


def emit_chat_chunk(*, widget: Any, chunk: str, turn_id: str = "") -> None:
    gui_emit_chunk(widget=widget, chunk=chunk, turn_id=turn_id)


def emit_chat_progress(
    *,
    widget: Any,
    update: str,
    last_progress_update: str | None,
    turn_id: str = "",
    turn_status: str = "",
) -> str | None:
    return gui_emit_progress(
        widget=widget,
        update=update,
        last_progress_update=last_progress_update,
        turn_id=turn_id,
        turn_status=turn_status,
    )


def emit_chat_final(
    *,
    widget: Any,
    message: str,
    is_error: bool,
    emit_progress_cb: Any,
    turn_id: str = "",
    turn_status: str = "",
    error_type: str = "",
) -> None:
    gui_emit_final(
        widget=widget,
        message=message,
        is_error=is_error,
        emit_progress_cb=emit_progress_cb,
        turn_id=turn_id,
        turn_status=turn_status,
        error_type=error_type,
    )


def load_chat_history_messages(
    *,
    session_store: Any,
    session_id: str,
    max_history_messages: int,
) -> list[dict[str, Any]]:
    return gui_load_history_messages(
        session_store=session_store,
        session_id=session_id,
        max_history_messages=max_history_messages,
    )


def persist_chat_turn(
    *,
    user_text: str,
    assistant_text: str,
    session_id: str,
    session_store: Any,
    max_history_messages: int,
    workspace_memory: Any,
    persist_session_history: bool = True,
) -> None:
    gui_persist_turn(
        user_text=user_text,
        assistant_text=assistant_text,
        session_id=session_id,
        session_store=session_store,
        max_history_messages=max_history_messages,
        workspace_memory=workspace_memory,
        persist_session_history=persist_session_history,
    )


__all__ = [
    "AgentSessionManager",
    "PersistentSessionStore",
    "clear_chat_session",
    "delete_chat_history_message",
    "emit_chat_chunk",
    "emit_chat_final",
    "emit_chat_progress",
    "get_chat_session_store",
    "load_chat_history_messages",
    "persist_chat_turn",
]
