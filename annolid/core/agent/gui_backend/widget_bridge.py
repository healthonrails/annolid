from __future__ import annotations

from typing import Any, Dict

from qtpy import QtCore
from qtpy.QtCore import QMetaObject


def build_gui_context_payload(
    *,
    session_id: str,
    provider: str,
    model: str,
    prompt: str,
    image_path: str,
    widget: Any,
    web_state_getter,
    pdf_state_getter,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "provider": provider,
        "model": model,
        "prompt_chars": len(str(prompt or "")),
        "image_path": str(image_path or ""),
        "has_image": bool(image_path),
    }
    if widget is not None:
        payload["attach_canvas"] = bool(
            getattr(
                getattr(widget, "attach_canvas_checkbox", None),
                "isChecked",
                lambda: False,
            )()
        )
        payload["attach_window"] = bool(
            getattr(
                getattr(widget, "attach_window_checkbox", None),
                "isChecked",
                lambda: False,
            )()
        )
        host = getattr(widget, "host_window_widget", None)
        if host is not None:
            for key in ("video_file", "filename", "frame_number"):
                with_context = getattr(host, key, None)
                if with_context not in (None, ""):
                    payload[key] = with_context
        web_state = web_state_getter()
        if isinstance(web_state, dict):
            payload["web"] = web_state
        pdf_state = pdf_state_getter()
        if isinstance(pdf_state, dict):
            payload["pdf"] = pdf_state
    return payload


def invoke_widget_slot(
    *,
    widget: Any,
    session_id: str,
    slot_name: str,
    qargs: tuple[Any, ...],
    logger: Any,
) -> bool:
    if widget is None:
        return False
    try:
        invoked = QMetaObject.invokeMethod(
            widget,
            slot_name,
            QtCore.Qt.BlockingQueuedConnection,
            *qargs,
        )
        # Depending on Qt binding/runtime, invokeMethod may return either
        # bool or None on success. Treat non-exception as success.
        if isinstance(invoked, bool):
            return invoked
        return True
    except Exception as exc:
        logger.warning(
            "annolid-bot gui slot invoke failed session=%s slot=%s error=%s",
            session_id,
            slot_name,
            exc,
        )
        return False


def invoke_widget_json_slot(
    *,
    widget: Any,
    invoke_slot,
    slot_name: str,
    qargs: tuple[Any, ...],
) -> Dict[str, Any]:
    if widget is not None:
        try:
            setattr(widget, "_bot_action_result", {})
        except Exception:
            pass
    ok = invoke_slot(slot_name, *qargs)
    if not ok:
        return {"ok": False, "error": f"Failed to run GUI action: {slot_name}"}
    if widget is not None:
        payload = getattr(widget, "_bot_action_result", None)
        if isinstance(payload, dict) and payload:
            return dict(payload)
    return {"ok": True}


def get_widget_action_result(*, widget: Any, action_name: str) -> Dict[str, Any]:
    try:
        getter = getattr(widget, "get_bot_action_result", None) if widget else None
        if callable(getter):
            payload = getter(action_name)
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {}
