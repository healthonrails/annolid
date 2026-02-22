from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .utils import get_sessions_path


def _now_iso() -> str:
    return datetime.now().isoformat()


def _encode_key(key: str) -> str:
    raw = str(key).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return encoded or "session"


@dataclass
class AgentSession:
    """Conversation session persisted as JSONL (metadata + message lines)."""

    key: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Mapping[str, Any]) -> None:
        msg = dict(message)
        msg.setdefault("timestamp", _now_iso())
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def clear_messages(self) -> None:
        self.messages = []
        self.updated_at = datetime.now()


class AgentSessionManager:
    """Manage persistent agent sessions on disk."""

    def __init__(
        self,
        workspace: Optional[Path] = None,
        *,
        sessions_dir: Optional[Path] = None,
    ) -> None:
        if workspace is not None:
            base = Path(workspace) / ".annolid" / "sessions"
        else:
            base = sessions_dir or get_sessions_path()
        self.sessions_dir = Path(base).expanduser()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, AgentSession] = {}
        self._lock = RLock()

    def _session_path(self, key: str) -> Path:
        return self.sessions_dir / f"{_encode_key(key)}.jsonl"

    def get_or_create(self, key: str) -> AgentSession:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            session = self._load(key) or AgentSession(key=key)
            self._cache[key] = session
            return session

    def _load(self, key: str) -> Optional[AgentSession]:
        path = self._session_path(key)
        if not path.exists():
            return None
        try:
            messages: List[Dict[str, Any]] = []
            facts: Dict[str, str] = {}
            metadata: Dict[str, Any] = {}
            created_at: Optional[datetime] = None
            updated_at: Optional[datetime] = None
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    data = json.loads(text)
                    if data.get("_type") == "metadata":
                        metadata = dict(data.get("metadata") or {})
                        facts = {
                            str(k): str(v)
                            for k, v in dict(data.get("facts") or {}).items()
                        }
                        created_text = data.get("created_at")
                        updated_text = data.get("updated_at")
                        if created_text:
                            created_at = datetime.fromisoformat(str(created_text))
                        if updated_text:
                            updated_at = datetime.fromisoformat(str(updated_text))
                        continue
                    if isinstance(data, dict):
                        messages.append(dict(data))
            return AgentSession(
                key=key,
                messages=messages,
                facts=facts,
                created_at=created_at or datetime.now(),
                updated_at=updated_at or datetime.now(),
                metadata=metadata,
            )
        except Exception:
            return None

    def save(self, session: AgentSession) -> None:
        with self._lock:
            path = self._session_path(session.key)
            tmp_path = path.with_name(f"{path.name}.tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                meta_line = {
                    "_type": "metadata",
                    "key": session.key,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "facts": dict(session.facts),
                    "metadata": dict(session.metadata),
                    "message_count": len(session.messages),
                }
                fh.write(json.dumps(meta_line, ensure_ascii=False) + "\n")
                for msg in session.messages:
                    fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
            tmp_path.replace(path)
            self._cache[session.key] = session

    def delete(self, key: str) -> bool:
        with self._lock:
            self._cache.pop(key, None)
            path = self._session_path(key)
            if path.exists():
                path.unlink()
                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if not first:
                    continue
                data = json.loads(first)
                if data.get("_type") != "metadata":
                    continue
                rows.append(
                    {
                        "key": str(data.get("key") or ""),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "message_count": int(data.get("message_count") or 0),
                        "path": str(path),
                    }
                )
            except Exception:
                continue
        return sorted(
            rows, key=lambda item: str(item.get("updated_at") or ""), reverse=True
        )

    def get_session_overview(self, key: str) -> Dict[str, Any]:
        session = self.get_or_create(str(key or ""))
        return {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "fact_count": len(session.facts),
            "facts": dict(session.facts),
            "metadata": dict(session.metadata),
            "path": str(self._session_path(session.key)),
        }

    def get_session_metadata(self, key: str) -> Dict[str, Any]:
        session = self.get_or_create(str(key or ""))
        return dict(session.metadata)

    def update_session_metadata(self, key: str, updates: Mapping[str, Any]) -> None:
        session = self.get_or_create(str(key or ""))
        changed = False
        for raw_key, raw_value in dict(updates or {}).items():
            meta_key = str(raw_key or "").strip()
            if not meta_key:
                continue
            if session.metadata.get(meta_key) == raw_value:
                continue
            session.metadata[meta_key] = raw_value
            changed = True
        if changed:
            session.updated_at = datetime.now()
            self.save(session)


class PersistentSessionStore:
    """Session store adapter for AgentLoop with disk persistence."""

    _DEFAULT_WORKING_MEMORY_MAX_CHARS = 8192
    _DEFAULT_LONG_TERM_MEMORY_MAX_CHARS = 32768
    _DEFAULT_MEMORY_AUDIT_MAX_ENTRIES = 200
    _DEFAULT_EVENT_LOG_MAX_ENTRIES = 500

    def __init__(
        self,
        manager: AgentSessionManager,
        *,
        working_memory_max_chars: int = _DEFAULT_WORKING_MEMORY_MAX_CHARS,
        long_term_memory_max_chars: int = _DEFAULT_LONG_TERM_MEMORY_MAX_CHARS,
        memory_audit_max_entries: int = _DEFAULT_MEMORY_AUDIT_MAX_ENTRIES,
        event_log_max_entries: int = _DEFAULT_EVENT_LOG_MAX_ENTRIES,
    ):
        self._manager = manager
        self._lock = RLock()
        self._working_memory_max_chars = max(32, int(working_memory_max_chars))
        self._long_term_memory_max_chars = max(64, int(long_term_memory_max_chars))
        self._memory_audit_max_entries = max(10, int(memory_audit_max_entries))
        self._event_log_max_entries = max(20, int(event_log_max_entries))

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            return [dict(m) for m in session.messages]

    def append_history(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_messages: int,
    ) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            session.messages = self._compact_messages(session.messages)
            for message in messages:
                msg = dict(message)
                role = str(msg.get("role") or "")
                content = msg.get("content")
                if role in {"user", "assistant", "system"}:
                    if not isinstance(content, str) or not content.strip():
                        continue
                session.add_message(msg)
            keep = max(1, int(max_messages))
            if len(session.messages) > keep:
                session.messages = session.messages[-keep:]
            self._manager.save(session)

    def clear_history(self, session_id: str) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            session.clear_messages()
            self._manager.save(session)

    def get_facts(self, session_id: str) -> Dict[str, str]:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            return dict(session.facts)

    def set_fact(self, session_id: str, key: str, value: str) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            before = str(session.facts.get(str(key), ""))
            after = str(value)
            session.facts[str(key)] = after
            session.updated_at = datetime.now()
            self._append_memory_audit_entry(
                session,
                scope="facts",
                mutation="set_fact",
                reason=f"set key={str(key)}",
                before=before,
                after=after,
            )
            self._manager.save(session)

    def delete_fact(self, session_id: str, key: str) -> bool:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            if key not in session.facts:
                return False
            before = str(session.facts.get(str(key), ""))
            session.facts.pop(key, None)
            session.updated_at = datetime.now()
            self._append_memory_audit_entry(
                session,
                scope="facts",
                mutation="delete_fact",
                reason=f"delete key={str(key)}",
                before=before,
                after="",
            )
            self._manager.save(session)
            return True

    def clear_facts(self, session_id: str) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            before = "\n".join(
                f"{k}: {v}" for k, v in dict(session.facts or {}).items()
            ).strip()
            session.facts = {}
            session.updated_at = datetime.now()
            self._append_memory_audit_entry(
                session,
                scope="facts",
                mutation="clear_facts",
                reason="clear all fact entries",
                before=before,
                after="",
            )
            self._manager.save(session)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._manager.delete(session_id)

    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._manager.get_session_metadata(session_id)

    def update_session_metadata(
        self, session_id: str, updates: Mapping[str, Any]
    ) -> None:
        with self._lock:
            self._manager.update_session_metadata(session_id, updates)

    def get_working_memory(self, session_id: str) -> str:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            payload = str(session.metadata.get("working_memory") or "")
            return self._truncate_text(payload, self._working_memory_max_chars)

    def set_working_memory(
        self,
        session_id: str,
        text: str,
        *,
        reason: str = "",
        turn_id: str = "",
    ) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            before = str(session.metadata.get("working_memory") or "")
            after = self._truncate_text(str(text or ""), self._working_memory_max_chars)
            session.metadata["working_memory"] = after
            session.updated_at = datetime.now()
            self._append_memory_audit_entry(
                session,
                scope="working_memory",
                mutation="set_working_memory",
                reason=reason or "set working memory",
                before=before,
                after=after,
                turn_id=turn_id,
            )
            self._manager.save(session)

    def get_long_term_memory(self, session_id: str) -> str:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            payload = str(session.metadata.get("long_term_memory") or "")
            return self._truncate_text(payload, self._long_term_memory_max_chars)

    def set_long_term_memory(
        self,
        session_id: str,
        text: str,
        *,
        reason: str = "",
        turn_id: str = "",
    ) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            before = str(session.metadata.get("long_term_memory") or "")
            after = self._truncate_text(
                str(text or ""), self._long_term_memory_max_chars
            )
            session.metadata["long_term_memory"] = after
            session.updated_at = datetime.now()
            self._append_memory_audit_entry(
                session,
                scope="long_term_memory",
                mutation="set_long_term_memory",
                reason=reason or "set long-term memory",
                before=before,
                after=after,
                turn_id=turn_id,
            )
            self._manager.save(session)

    def get_memory_audit_trail(
        self, session_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            rows = list(session.metadata.get("memory_audit_trail") or [])
            keep = max(1, int(limit))
            normalized: List[Dict[str, Any]] = []
            for item in rows[-keep:]:
                if isinstance(item, Mapping):
                    normalized.append(dict(item))
            return normalized

    def append_memory_audit(
        self,
        session_id: str,
        *,
        scope: str,
        mutation: str,
        reason: str,
        before: str = "",
        after: str = "",
        turn_id: str = "",
    ) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            self._append_memory_audit_entry(
                session,
                scope=scope,
                mutation=mutation,
                reason=reason,
                before=before,
                after=after,
                turn_id=turn_id,
            )
            session.updated_at = datetime.now()
            self._manager.save(session)

    def record_event(
        self,
        session_id: str,
        *,
        direction: str,
        kind: str,
        payload: Mapping[str, Any],
        turn_id: str = "",
        event_id: str = "",
        idempotency_key: str = "",
    ) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            events = list(session.metadata.get("event_log") or [])
            events.append(
                {
                    "timestamp": _now_iso(),
                    "direction": str(direction or "").strip().lower() or "outbound",
                    "kind": str(kind or "").strip().lower() or "event",
                    "turn_id": str(turn_id or "").strip(),
                    "event_id": str(event_id or "").strip(),
                    "idempotency_key": str(idempotency_key or "").strip(),
                    "payload": dict(payload or {}),
                }
            )
            if len(events) > self._event_log_max_entries:
                events = events[-self._event_log_max_entries :]
            session.metadata["event_log"] = events
            session.updated_at = datetime.now()
            self._manager.save(session)

    def replay_events(
        self,
        session_id: str,
        *,
        direction: str = "",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            events = list(session.metadata.get("event_log") or [])
            selected: List[Dict[str, Any]] = []
            want_direction = str(direction or "").strip().lower()
            for item in events:
                if not isinstance(item, Mapping):
                    continue
                row = dict(item)
                if (
                    want_direction
                    and str(row.get("direction") or "").lower() != want_direction
                ):
                    continue
                selected.append(row)
            keep = max(1, int(limit))
            return selected[-keep:]

    def record_automation_task_run(
        self,
        session_id: str,
        *,
        task_name: str,
        status: str,
        detail: str = "",
    ) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            runs = list(session.metadata.get("automation_runs") or [])
            runs.append(
                {
                    "task_name": str(task_name or "").strip(),
                    "status": str(status or "").strip().lower(),
                    "detail": str(detail or "").strip(),
                    "timestamp": _now_iso(),
                }
            )
            # Keep bounded history in metadata.
            if len(runs) > 100:
                runs = runs[-100:]
            session.metadata["automation_runs"] = runs
            session.updated_at = datetime.now()
            self._manager.save(session)

    @staticmethod
    def _compact_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for message in messages:
            msg = dict(message)
            role = str(msg.get("role") or "")
            content = msg.get("content")
            if role in {"user", "assistant", "system"}:
                if not isinstance(content, str) or not content.strip():
                    continue
            cleaned.append(msg)
        return cleaned

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        text = str(value or "")
        if len(text) <= int(max_chars):
            return text
        keep = max(32, int(max_chars) - 32)
        return text[:keep] + "\n...[truncated]..."

    def _append_memory_audit_entry(
        self,
        session: AgentSession,
        *,
        scope: str,
        mutation: str,
        reason: str,
        before: str,
        after: str,
        turn_id: str = "",
    ) -> None:
        rows = list(session.metadata.get("memory_audit_trail") or [])
        rows.append(
            {
                "timestamp": _now_iso(),
                "scope": str(scope or "").strip().lower(),
                "mutation": str(mutation or "").strip().lower(),
                "reason": str(reason or "").strip(),
                "turn_id": str(turn_id or "").strip(),
                "before_chars": len(str(before or "")),
                "after_chars": len(str(after or "")),
            }
        )
        if len(rows) > self._memory_audit_max_entries:
            rows = rows[-self._memory_audit_max_entries :]
        session.metadata["memory_audit_trail"] = rows
