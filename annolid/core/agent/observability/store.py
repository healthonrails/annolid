from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional

from annolid.core.agent.utils import get_agent_data_path

from .events import GovernanceEvent, build_governance_event


class GovernanceEventStore:
    """Append-only NDJSON governance event store."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = self._resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @staticmethod
    def _resolve_path(path: Optional[Path]) -> Path:
        if path is not None:
            return Path(path).expanduser().resolve()
        env_path = str(os.getenv("ANNOLID_GOVERNANCE_EVENTS_PATH") or "").strip()
        if env_path:
            return Path(env_path).expanduser().resolve()
        return get_agent_data_path() / "governance" / "events.ndjson"

    def append(self, event: GovernanceEvent) -> Dict[str, Any]:
        row = event.to_dict()
        text = json.dumps(row, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return row

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        for raw in lines[-max(1, int(limit)) :]:
            text = str(raw or "").strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(dict(item))
        return rows


def emit_governance_event(
    *,
    event_type: str,
    action: str,
    outcome: str = "ok",
    actor: str = "system",
    details: Mapping[str, Any] | None = None,
    store_path: Optional[Path] = None,
) -> Dict[str, Any]:
    try:
        store = GovernanceEventStore(path=store_path)
        event = build_governance_event(
            event_type=event_type,
            action=action,
            outcome=outcome,
            actor=actor,
            details=details,
        )
        return store.append(event)
    except Exception:
        return {}
