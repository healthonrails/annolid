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
        del workspace  # kept for API compatibility/future workspace scoping
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


class PersistentSessionStore:
    """Session store adapter for AgentLoop with disk persistence."""

    def __init__(self, manager: AgentSessionManager):
        self._manager = manager
        self._lock = RLock()

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
            session.facts[str(key)] = str(value)
            session.updated_at = datetime.now()
            self._manager.save(session)

    def delete_fact(self, session_id: str, key: str) -> bool:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            if key not in session.facts:
                return False
            session.facts.pop(key, None)
            session.updated_at = datetime.now()
            self._manager.save(session)
            return True

    def clear_facts(self, session_id: str) -> None:
        with self._lock:
            session = self._manager.get_or_create(session_id)
            session.facts = {}
            session.updated_at = datetime.now()
            self._manager.save(session)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._manager.delete(session_id)

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
