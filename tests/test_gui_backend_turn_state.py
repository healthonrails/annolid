from __future__ import annotations

from annolid.core.agent.gui_backend.turn_state import (
    ERROR_TYPE_INTERNAL,
    ERROR_TYPE_NONE,
    TURN_STATUS_FAILED,
    TURN_STATUS_QUEUED,
    TurnState,
    normalize_error_type,
    normalize_turn_status,
)


def test_turn_state_normalizers_use_defaults_for_unknown_values() -> None:
    assert (
        normalize_turn_status("weird", default=TURN_STATUS_QUEUED) == TURN_STATUS_QUEUED
    )
    assert normalize_error_type("bad", default=ERROR_TYPE_NONE) == ERROR_TYPE_NONE


def test_turn_state_dataclass_normalized() -> None:
    state = TurnState(turn_id=" t1 ", status="FAILED", error_type="INTERNAL_ERROR")
    normalized = state.normalized()
    assert normalized.turn_id == "t1"
    assert normalized.status == TURN_STATUS_FAILED
    assert normalized.error_type == ERROR_TYPE_INTERNAL
