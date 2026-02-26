from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class EvalTrace:
    trace_id: str
    user_message: str
    expected_substring: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EvalResponse:
    trace_id: str
    content: str
    metadata: Dict[str, Any]


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    rows.append(dict(item))
        elif isinstance(payload, Mapping):
            rows.append(dict(payload))
        return rows
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def load_eval_traces(path: Path) -> List[EvalTrace]:
    rows = _load_rows(Path(path).expanduser().resolve())
    traces: List[EvalTrace] = []
    for idx, row in enumerate(rows):
        trace_id = str(
            row.get("trace_id")
            or row.get("id")
            or row.get("turn_id")
            or f"trace_{idx:05d}"
        ).strip()
        user_message = str(row.get("user_message") or row.get("message") or "").strip()
        expected = str(
            row.get("expected_substring")
            or row.get("expected")
            or row.get("must_contain")
            or ""
        ).strip()
        metadata = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "trace_id",
                "id",
                "turn_id",
                "user_message",
                "message",
                "expected_substring",
                "expected",
                "must_contain",
            }
        }
        traces.append(
            EvalTrace(
                trace_id=trace_id,
                user_message=user_message,
                expected_substring=expected,
                metadata=metadata,
            )
        )
    return traces


def load_eval_responses(path: Path) -> Dict[str, EvalResponse]:
    rows = _load_rows(Path(path).expanduser().resolve())
    out: Dict[str, EvalResponse] = {}
    for idx, row in enumerate(rows):
        trace_id = str(
            row.get("trace_id")
            or row.get("id")
            or row.get("turn_id")
            or f"trace_{idx:05d}"
        ).strip()
        content = str(row.get("content") or row.get("response") or "").strip()
        metadata = {
            k: v
            for k, v in row.items()
            if k not in {"trace_id", "id", "turn_id", "content", "response"}
        }
        out[trace_id] = EvalResponse(
            trace_id=trace_id, content=content, metadata=metadata
        )
    return out
