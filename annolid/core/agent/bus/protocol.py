from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Union

PROTOCOL_VERSION = 1


class ProtocolValidationError(ValueError):
    """Raised when a bus protocol frame is malformed."""


@dataclass(frozen=True)
class RequestFrame:
    type: str
    id: str
    method: str
    params: dict[str, Any]
    idempotency_key: str | None = None


@dataclass(frozen=True)
class ResponseFrame:
    type: str
    id: str
    ok: bool
    payload: Any = None
    error: Any = None


@dataclass(frozen=True)
class EventFrame:
    type: str
    event: str
    payload: Any
    seq: int | None = None
    stateVersion: int | None = None


Frame = Union[RequestFrame, ResponseFrame, EventFrame]


def parse_frame(payload: str | bytes | Mapping[str, Any]) -> Frame:
    raw: Any
    if isinstance(payload, (str, bytes)):
        try:
            raw = json.loads(payload)
        except Exception as exc:
            raise ProtocolValidationError(f"Invalid JSON frame: {exc}") from exc
    else:
        raw = dict(payload)
    if not isinstance(raw, dict):
        raise ProtocolValidationError("Frame must be a JSON object.")

    frame_type = str(raw.get("type") or "").strip().lower()
    if frame_type == "req":
        frame_id = str(raw.get("id") or "").strip()
        method = str(raw.get("method") or "").strip()
        params_raw = raw.get("params", {})
        if not frame_id:
            raise ProtocolValidationError("Request frame requires non-empty id.")
        if not method:
            raise ProtocolValidationError("Request frame requires non-empty method.")
        if params_raw is None:
            params_raw = {}
        if not isinstance(params_raw, dict):
            raise ProtocolValidationError("Request frame params must be an object.")
        idempotency_key = raw.get("idempotency_key")
        idem = str(idempotency_key).strip() if idempotency_key is not None else None
        return RequestFrame(
            type="req",
            id=frame_id,
            method=method,
            params=dict(params_raw),
            idempotency_key=idem or None,
        )
    if frame_type == "res":
        frame_id = str(raw.get("id") or "").strip()
        if not frame_id:
            raise ProtocolValidationError("Response frame requires non-empty id.")
        if "ok" not in raw:
            raise ProtocolValidationError("Response frame requires ok boolean field.")
        ok = bool(raw.get("ok"))
        return ResponseFrame(
            type="res",
            id=frame_id,
            ok=ok,
            payload=raw.get("payload"),
            error=raw.get("error"),
        )
    if frame_type == "event":
        event_name = str(raw.get("event") or "").strip()
        if not event_name:
            raise ProtocolValidationError("Event frame requires non-empty event.")
        seq_raw = raw.get("seq")
        state_raw = raw.get("stateVersion")
        seq = int(seq_raw) if seq_raw is not None else None
        state_version = int(state_raw) if state_raw is not None else None
        return EventFrame(
            type="event",
            event=event_name,
            payload=raw.get("payload"),
            seq=seq,
            stateVersion=state_version,
        )
    raise ProtocolValidationError(f"Unsupported frame type: {frame_type!r}")
