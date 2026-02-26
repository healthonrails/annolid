from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(value: str, *, salt: str = "") -> str:
    data = f"{salt}|{value}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _redact_text(text: str, *, max_chars: int = 240) -> str:
    value = str(text or "")
    value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>", value)
    value = re.sub(r"\b\d{8,}\b", "<number>", value)
    value = re.sub(r"https?://\S+", "<url>", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > max(16, int(max_chars)):
        return value[: max(16, int(max_chars))].rstrip()
    return value


@dataclass(frozen=True)
class TraceCaptureConfig:
    enabled: bool = True
    hash_salt: str = ""
    max_text_chars: int = 240

    @classmethod
    def from_env(cls) -> "TraceCaptureConfig":
        enabled_raw = str(os.getenv("ANNOLID_AGENT_TRACE_CAPTURE", "1")).strip().lower()
        enabled = enabled_raw not in {"0", "false", "no", "off"}
        salt = str(os.getenv("ANNOLID_AGENT_TRACE_SALT", "")).strip()
        max_chars_raw = str(
            os.getenv("ANNOLID_AGENT_TRACE_MAX_TEXT_CHARS", "240")
        ).strip()
        try:
            max_chars = max(64, int(max_chars_raw))
        except Exception:
            max_chars = 240
        return cls(enabled=enabled, hash_salt=salt, max_text_chars=max_chars)


class RunTraceStore:
    """Anonymized trace/feedback store under workspace/eval."""

    def __init__(
        self, workspace: Path, *, config: Optional[TraceCaptureConfig] = None
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.config = config or TraceCaptureConfig.from_env()
        self.eval_dir = self.workspace / "eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.eval_dir / "run_traces.ndjson"
        self.feedback_path = self.eval_dir / "feedback.ndjson"
        self.shadow_path = self.eval_dir / "shadow_routing.ndjson"

    @staticmethod
    def _append_ndjson(path: Path, row: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(row)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return payload

    @staticmethod
    def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = str(line or "").strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(dict(item))
        return rows

    def capture_run(
        self,
        *,
        session_id: str,
        channel: Optional[str],
        chat_id: Optional[str],
        user_message: str,
        assistant_response: str,
        tool_names: Iterable[str],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            return {}
        user_preview = _redact_text(user_message, max_chars=self.config.max_text_chars)
        answer_preview = _redact_text(
            assistant_response,
            max_chars=self.config.max_text_chars,
        )
        trace_id = _stable_hash(
            f"{session_id}|{user_preview}|{answer_preview}",
            salt=self.config.hash_salt,
        )
        payload = {
            "trace_id": trace_id,
            "timestamp": _utc_now(),
            "session_id_hash": _stable_hash(session_id, salt=self.config.hash_salt),
            "channel_hash": _stable_hash(
                str(channel or ""), salt=self.config.hash_salt
            ),
            "chat_id_hash": _stable_hash(
                str(chat_id or ""), salt=self.config.hash_salt
            ),
            "user_message_preview": user_preview,
            "assistant_response_preview": answer_preview,
            "tool_names": sorted(
                str(name).strip() for name in tool_names if str(name).strip()
            ),
            "metadata": dict(metadata or {}),
        }
        return self._append_ndjson(self.traces_path, payload)

    def capture_feedback(
        self,
        *,
        session_id: str,
        rating: int,
        comment: str = "",
        trace_id: Optional[str] = None,
        expected_substring: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        rating_value = int(rating)
        if rating_value < -1:
            rating_value = -1
        if rating_value > 1:
            rating_value = 1
        payload = {
            "timestamp": _utc_now(),
            "trace_id": str(trace_id or "").strip(),
            "session_id_hash": _stable_hash(session_id, salt=self.config.hash_salt),
            "rating": rating_value,
            "comment_preview": _redact_text(
                comment, max_chars=self.config.max_text_chars
            ),
            "expected_substring": _redact_text(
                expected_substring,
                max_chars=min(120, self.config.max_text_chars),
            ),
            "metadata": dict(metadata or {}),
        }
        return self._append_ndjson(self.feedback_path, payload)

    def capture_shadow_routing(
        self,
        *,
        session_id: str,
        primary_tools: Iterable[str],
        candidate_tools: Iterable[str],
        policy_name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        primary = sorted(
            {str(name).strip() for name in primary_tools if str(name).strip()}
        )
        candidate = sorted(
            {str(name).strip() for name in candidate_tools if str(name).strip()}
        )
        payload = {
            "timestamp": _utc_now(),
            "session_id_hash": _stable_hash(session_id, salt=self.config.hash_salt),
            "policy_name": str(policy_name or "default"),
            "primary_tools": primary,
            "candidate_tools": candidate,
            "added": [name for name in candidate if name not in set(primary)],
            "removed": [name for name in primary if name not in set(candidate)],
            "metadata": dict(metadata or {}),
        }
        return self._append_ndjson(self.shadow_path, payload)

    def load_traces(self) -> List[Dict[str, Any]]:
        return self._read_ndjson(self.traces_path)

    def load_feedback(self) -> List[Dict[str, Any]]:
        return self._read_ndjson(self.feedback_path)


def build_regression_eval_rows(
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
    min_abs_rating: int = 1,
) -> List[Dict[str, Any]]:
    by_trace: Dict[str, Dict[str, Any]] = {}
    for row in trace_rows:
        trace_id = str(row.get("trace_id") or "").strip()
        if trace_id:
            by_trace[trace_id] = dict(row)

    out: List[Dict[str, Any]] = []
    for feedback in feedback_rows:
        trace_id = str(feedback.get("trace_id") or "").strip()
        if not trace_id or trace_id not in by_trace:
            continue
        rating = int(feedback.get("rating") or 0)
        if abs(rating) < max(0, int(min_abs_rating)):
            continue
        trace = by_trace[trace_id]
        user_message = str(trace.get("user_message_preview") or "").strip()
        expected = str(feedback.get("expected_substring") or "").strip()
        if rating > 0 and not expected:
            expected = str(trace.get("assistant_response_preview") or "").strip()
        if not user_message or not expected:
            continue
        out.append(
            {
                "trace_id": trace_id,
                "user_message": user_message,
                "expected_substring": expected,
                "metadata": {
                    "rating": rating,
                    "feedback_comment": str(feedback.get("comment_preview") or ""),
                    "source": "run_trace_feedback",
                },
            }
        )
    return out
