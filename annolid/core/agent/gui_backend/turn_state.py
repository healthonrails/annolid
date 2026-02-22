from __future__ import annotations

from dataclasses import dataclass

TURN_STATUS_QUEUED = "queued"
TURN_STATUS_RUNNING = "running"
TURN_STATUS_CANCELLING = "cancelling"
TURN_STATUS_CANCELLED = "cancelled"
TURN_STATUS_COMPLETED = "completed"
TURN_STATUS_FAILED = "failed"

ERROR_TYPE_NONE = "none"
ERROR_TYPE_USER = "user_error"
ERROR_TYPE_TOOL = "tool_error"
ERROR_TYPE_TRANSPORT = "transport_error"
ERROR_TYPE_POLICY = "policy_error"
ERROR_TYPE_INTERNAL = "internal_error"
ERROR_TYPE_CANCELLED = "cancelled"

_VALID_TURN_STATUS = {
    TURN_STATUS_QUEUED,
    TURN_STATUS_RUNNING,
    TURN_STATUS_CANCELLING,
    TURN_STATUS_CANCELLED,
    TURN_STATUS_COMPLETED,
    TURN_STATUS_FAILED,
}
_VALID_ERROR_TYPES = {
    ERROR_TYPE_NONE,
    ERROR_TYPE_USER,
    ERROR_TYPE_TOOL,
    ERROR_TYPE_TRANSPORT,
    ERROR_TYPE_POLICY,
    ERROR_TYPE_INTERNAL,
    ERROR_TYPE_CANCELLED,
}


def normalize_turn_status(value: str, *, default: str = TURN_STATUS_QUEUED) -> str:
    text = str(value or "").strip().lower()
    if text in _VALID_TURN_STATUS:
        return text
    return str(default or TURN_STATUS_QUEUED)


def normalize_error_type(value: str, *, default: str = ERROR_TYPE_NONE) -> str:
    text = str(value or "").strip().lower()
    if text in _VALID_ERROR_TYPES:
        return text
    return str(default or ERROR_TYPE_NONE)


@dataclass(frozen=True)
class TurnState:
    turn_id: str
    status: str = TURN_STATUS_QUEUED
    error_type: str = ERROR_TYPE_NONE

    def normalized(self) -> "TurnState":
        return TurnState(
            turn_id=str(self.turn_id or "").strip(),
            status=normalize_turn_status(self.status),
            error_type=normalize_error_type(self.error_type),
        )
