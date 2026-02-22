from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from qtpy import QtCore
from qtpy.QtCore import QMetaObject
from .turn_state import (
    ERROR_TYPE_INTERNAL,
    ERROR_TYPE_NONE,
    TURN_STATUS_COMPLETED,
    TURN_STATUS_FAILED,
    TURN_STATUS_QUEUED,
    TURN_STATUS_RUNNING,
    normalize_error_type,
    normalize_turn_status,
)


@dataclass(frozen=True)
class OutboundChatEvent:
    direction: str
    kind: str
    text: str = ""
    is_error: bool = False
    turn_id: str = ""
    turn_status: str = ""
    error_type: str = ""
    event_id: str = ""
    idempotency_key: str = ""


def encode_outbound_chat_event(event: OutboundChatEvent) -> str:
    direction = str(event.direction or "").strip().lower() or "outbound"
    kind = str(event.kind or "").strip().lower()
    turn_id = str(event.turn_id or "").strip()
    event_id = str(event.event_id or "").strip()
    idempotency_key = str(event.idempotency_key or "").strip()
    if not idempotency_key:
        digest_src = "|".join(
            [
                direction,
                kind,
                turn_id,
                str(bool(event.is_error)),
                str(event.text or ""),
            ]
        )
        idempotency_key = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()
    payload = {
        "direction": direction,
        "kind": kind,
        "text": str(event.text or ""),
        "is_error": bool(event.is_error),
        "turn_id": turn_id,
        "turn_status": normalize_turn_status(
            str(event.turn_status or "").strip().lower(), default=TURN_STATUS_QUEUED
        ),
        "error_type": normalize_error_type(
            str(event.error_type or "").strip().lower(), default=ERROR_TYPE_NONE
        ),
        "event_id": event_id,
        "idempotency_key": idempotency_key,
    }
    return json.dumps(payload, ensure_ascii=False)


def decode_outbound_chat_event(payload_text: str) -> Optional[OutboundChatEvent]:
    raw = str(payload_text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    direction = str(payload.get("direction") or "").strip().lower() or "outbound"
    kind = str(payload.get("kind") or "").strip().lower()
    if kind not in {"chunk", "progress", "final"}:
        return None
    return OutboundChatEvent(
        direction=direction,
        kind=kind,
        text=str(payload.get("text") or ""),
        is_error=bool(payload.get("is_error", False)),
        turn_id=str(payload.get("turn_id") or "").strip(),
        turn_status=normalize_turn_status(
            str(payload.get("turn_status") or "").strip().lower(),
            default=TURN_STATUS_QUEUED,
        ),
        error_type=normalize_error_type(
            str(payload.get("error_type") or "").strip().lower(),
            default=ERROR_TYPE_NONE,
        ),
        event_id=str(payload.get("event_id") or "").strip(),
        idempotency_key=str(payload.get("idempotency_key") or "").strip(),
    )


def _emit_outbound_chat_event(widget: Any, event: OutboundChatEvent) -> None:
    if widget is None:
        return
    payload_text = encode_outbound_chat_event(event)
    bus_enqueue_slot = getattr(widget, "enqueue_outbound_bus_message", None)
    if callable(bus_enqueue_slot):
        invoked = QMetaObject.invokeMethod(
            widget,
            "enqueue_outbound_bus_message",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, payload_text),
        )
        if bool(invoked):
            return
    consume_slot = getattr(widget, "consume_outbound_chat_event", None)
    if callable(consume_slot):
        invoked = QMetaObject.invokeMethod(
            widget,
            "consume_outbound_chat_event",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, payload_text),
        )
        if bool(invoked):
            return

    # Backward-compatible fallback for old widgets.
    if event.kind == "chunk":
        QMetaObject.invokeMethod(
            widget,
            "stream_chat_chunk",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, event.text),
        )
    elif event.kind == "progress":
        QMetaObject.invokeMethod(
            widget,
            "stream_chat_progress",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, event.text),
        )
    elif event.kind == "final":
        QMetaObject.invokeMethod(
            widget,
            "update_chat_response",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, event.text),
            QtCore.Q_ARG(bool, bool(event.is_error)),
        )


def emit_chunk(*, widget: Any, chunk: str, turn_id: str = "") -> None:
    _emit_outbound_chat_event(
        widget,
        OutboundChatEvent(
            direction="outbound",
            kind="chunk",
            text=str(chunk or ""),
            turn_id=str(turn_id or "").strip(),
            turn_status=TURN_STATUS_RUNNING,
        ),
    )


def emit_progress(
    *,
    widget: Any,
    update: str,
    last_progress_update: Optional[str],
    turn_id: str = "",
    turn_status: str = TURN_STATUS_RUNNING,
) -> Optional[str]:
    if not bool(getattr(widget, "enable_progress_stream", False)):
        return last_progress_update
    text = str(update or "").strip()
    if not text or text == last_progress_update:
        return last_progress_update
    _emit_outbound_chat_event(
        widget,
        OutboundChatEvent(
            direction="outbound",
            kind="progress",
            text=text,
            turn_id=str(turn_id or "").strip(),
            turn_status=normalize_turn_status(
                str(turn_status or "").strip().lower(), default=TURN_STATUS_RUNNING
            ),
        ),
    )
    return text


def emit_final(
    *,
    widget: Any,
    message: str,
    is_error: bool,
    emit_progress_cb,
    turn_id: str = "",
    turn_status: str = "",
    error_type: str = "",
) -> None:
    if is_error:
        emit_progress_cb("Response failed")
    else:
        emit_progress_cb("Response ready")
    status_text = normalize_turn_status(
        str(turn_status or "").strip().lower(),
        default=TURN_STATUS_FAILED if is_error else TURN_STATUS_COMPLETED,
    )
    normalized_error_type = normalize_error_type(
        str(error_type or "").strip().lower(),
        default=ERROR_TYPE_INTERNAL if is_error else ERROR_TYPE_NONE,
    )
    _emit_outbound_chat_event(
        widget,
        OutboundChatEvent(
            direction="outbound",
            kind="final",
            text=str(message or ""),
            is_error=bool(is_error),
            turn_id=str(turn_id or "").strip(),
            turn_status=status_text,
            error_type=normalized_error_type,
        ),
    )


def load_history_messages(
    *,
    session_store: Any,
    session_id: str,
    max_history_messages: int,
) -> List[Dict[str, Any]]:
    """Load persisted chat history as role/content records.

    Preserves tool-call metadata when present so resumed sessions can keep
    assistant-tool continuity across turns.
    """
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
        entry: Dict[str, Any] = {"role": role, "content": text}
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            entry["tool_calls"] = tool_calls
        tool_call_id = msg.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            entry["tool_call_id"] = tool_call_id.strip()
        name = msg.get("name")
        if isinstance(name, str) and name.strip():
            entry["name"] = name.strip()
        cleaned.append(entry)
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
    if session_store:
        _record_session_event(
            session_store=session_store,
            session_id=session_id,
            direction="inbound",
            kind="user",
            payload={"text": user_msg},
        )
        _record_session_event(
            session_store=session_store,
            session_id=session_id,
            direction="outbound",
            kind="assistant",
            payload={"text": assistant_msg},
        )

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


def _record_session_event(
    *,
    session_store: Any,
    session_id: str,
    direction: str,
    kind: str,
    payload: Dict[str, Any],
) -> None:
    if session_store is None:
        return
    recorder = getattr(session_store, "record_event", None)
    if not callable(recorder):
        return
    try:
        recorder(
            session_id,
            direction=str(direction or "").strip().lower() or "outbound",
            kind=str(kind or "").strip().lower() or "event",
            payload=dict(payload or {}),
        )
    except Exception:
        return


def replay_session_debug_events(
    *,
    session_store: Any,
    session_id: str,
    direction: str = "",
    limit: int = 200,
) -> List[Dict[str, Any]]:
    if not session_store:
        return []
    replay = getattr(session_store, "replay_events", None)
    if not callable(replay):
        return []
    try:
        rows = replay(
            session_id,
            direction=str(direction or "").strip().lower(),
            limit=max(1, int(limit)),
        )
    except Exception:
        return []
    normalized: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in list(rows or []):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        key = "|".join(
            [
                str(row.get("direction") or "").strip().lower(),
                str(row.get("kind") or "").strip().lower(),
                str(row.get("event_id") or "").strip(),
                str(row.get("idempotency_key") or "").strip(),
                str(
                    (row.get("payload") or {}).get("text")
                    if isinstance(row.get("payload"), dict)
                    else ""
                ),
            ]
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        payload = row.get("payload")
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            if len(text) > 1024:
                payload = dict(payload)
                payload["text"] = text[:1024] + "\n...[truncated]..."
                row["payload"] = payload
        normalized.append(row)
    return normalized


def format_replay_as_text(events: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in events:
        ts = str(item.get("timestamp") or "").strip()
        direction = str(item.get("direction") or "").strip().lower()
        kind = str(item.get("kind") or "").strip().lower()
        payload = item.get("payload")
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("text") or "").strip()
        stamp = ts if ts else "-"
        header = f"[{stamp}] {direction}:{kind}".strip()
        if text:
            lines.append(f"{header} {text}")
        else:
            lines.append(header)
    return "\n".join(lines).strip()
