from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class GovernanceEvent:
    timestamp: str
    event_type: str
    action: str
    outcome: str
    actor: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "action": self.action,
            "outcome": self.outcome,
            "actor": self.actor,
            "details": dict(self.details),
        }


def build_governance_event(
    *,
    event_type: str,
    action: str,
    outcome: str = "ok",
    actor: str = "system",
    details: Mapping[str, Any] | None = None,
) -> GovernanceEvent:
    return GovernanceEvent(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        event_type=str(event_type or "unknown").strip().lower() or "unknown",
        action=str(action or "unknown").strip().lower() or "unknown",
        outcome=str(outcome or "ok").strip().lower() or "ok",
        actor=str(actor or "system").strip().lower() or "system",
        details=dict(details or {}),
    )
