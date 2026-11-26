from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .tool_call_utils import strip_tool_call_markup
from .tools.policy import (
    CONCURRENCY_SAFE_READ_ONLY_TOOLS,
    HIGH_RISK_TOOL_RUNTIME_AUTOMATION,
    HIGH_RISK_TOOL_RUNTIME_EXEC,
    classify_destructive_tool_call,
)
from .web_prompt_utils import derive_web_lookup_prompt_from_messages

_SNAPSHOT_TOOL_NAMES = frozenset(
    {"check_stream_source", "gui_check_stream_source", "camera_snapshot"}
)
_COMPACTABLE_STRING_KEYS = (
    "text",
    "output",
    "stdout",
    "stderr",
    "content",
    "message",
    "stdout_tail",
)
_COMPACTABLE_LIST_KEYS = (
    "results",
    "items",
    "entries",
    "lines",
    "matches",
    "files",
)


class ThinkTagStreamParser:
    """Incremental parser for `<think>...</think>` tags split across arbitrary chunks."""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""
        self._thought_chunks: List[str] = []
        self._output_chunks: List[str] = []

    @property
    def thought_text(self) -> str:
        return "".join(self._thought_chunks)

    @property
    def output_text(self) -> str:
        return "".join(self._output_chunks)

    @staticmethod
    def _trailing_partial_tag_len(text: str, tag: str) -> int:
        max_len = min(len(text), len(tag) - 1)
        for length in range(max_len, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def feed(self, chunk: str) -> Tuple[str, str]:
        raw = str(chunk or "")
        if not raw:
            return "", ""
        buf = self._pending + raw
        self._pending = ""
        thought_delta: List[str] = []
        output_delta: List[str] = []

        while buf:
            target = self._CLOSE_TAG if self._in_think else self._OPEN_TAG
            idx = buf.find(target)
            if idx >= 0:
                prefix = buf[:idx]
                if prefix:
                    if self._in_think:
                        self._thought_chunks.append(prefix)
                        thought_delta.append(prefix)
                    else:
                        self._output_chunks.append(prefix)
                        output_delta.append(prefix)
                buf = buf[idx + len(target) :]
                self._in_think = not self._in_think
                continue

            partial_len = self._trailing_partial_tag_len(buf, target)
            if partial_len:
                emit = buf[:-partial_len]
                self._pending = buf[-partial_len:]
            else:
                emit = buf
            if emit:
                if self._in_think:
                    self._thought_chunks.append(emit)
                    thought_delta.append(emit)
                else:
                    self._output_chunks.append(emit)
                    output_delta.append(emit)
            break

        return "".join(thought_delta), "".join(output_delta)

    def flush(self) -> Tuple[str, str]:
        if not self._pending:
            return "", ""
        leftover = self._pending
        self._pending = ""
        if self._in_think:
            self._thought_chunks.append(leftover)
            return leftover, ""
        self._output_chunks.append(leftover)
        return "", leftover


def compact_tool_result_for_llm(
    *,
    name: str,
    result: str,
    limit: int = 32000,
) -> str:
    """Compact oversized tool output while preserving valid JSON structure when possible."""
    del name
    text = str(result or "")
    max_chars = max(512, int(limit))
    if len(text) <= max_chars:
        return text

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if isinstance(payload, dict):
        mutated = False
        # 1. Trim large string fields (prioritizing `text` for legacy contract).
        reserve = 400
        for key in _COMPACTABLE_STRING_KEYS:
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                continue
            max_field = max(1000, max_chars - reserve)
            if len(value) > max_field:
                payload[key] = value[:max_field]
                payload["truncated_for_llm"] = True
                payload[f"original_{key}_length"] = len(value)
                mutated = True

        # 2. Trim large list fields if serialized payload still exceeds budget.
        serialized = json.dumps(payload, ensure_ascii=False)
        if len(serialized) > max_chars + 300:
            for key in _COMPACTABLE_LIST_KEYS:
                value = payload.get(key)
                if not isinstance(value, list) or len(value) <= 10:
                    continue
                orig_count = len(value)
                keep = min(len(value), 50)
                while keep >= 5:
                    payload[key] = value[:keep]
                    payload["truncated_for_llm"] = True
                    payload[f"original_{key}_count"] = orig_count
                    mutated = True
                    serialized = json.dumps(payload, ensure_ascii=False)
                    if len(serialized) <= max_chars + 300:
                        break
                    keep = keep // 2
                if len(serialized) <= max_chars + 300:
                    break

        # 3. Final shrink pass across all string fields if still over budget.
        if len(serialized) > max_chars + 300:
            str_keys = [
                k for k, v in payload.items() if isinstance(v, str) and len(v) > 512
            ]
            if str_keys:
                per_key_cap = max(256, (max_chars - reserve) // max(1, len(str_keys)))
                for key in str_keys:
                    val = str(payload[key])
                    if len(val) > per_key_cap:
                        payload.setdefault(f"original_{key}_length", len(val))
                        payload[key] = val[:per_key_cap]
                        payload["truncated_for_llm"] = True
                        mutated = True
                serialized = json.dumps(payload, ensure_ascii=False)

        if mutated and len(serialized) <= max_chars + 300:
            return serialized

    omitted = max(0, len(text) - max_chars)
    return (
        text[:max_chars]
        + f"\n...[tool output truncated for LLM; omitted {omitted} chars]"
    )


def is_concurrency_safe_tool_call(
    *,
    canonical_name: str,
    tool_obj: Any,
    args: Mapping[str, Any],
) -> bool:
    """Return True if a tool call is side-effect-free and safe to run concurrently."""
    name = str(canonical_name or "").strip().lower()
    if not name:
        return False
    # Never run GUI bridge callbacks or shared browser-tab actions concurrently.
    if name.startswith("gui_") or name.startswith("mcp_browser"):
        return False
    if name in HIGH_RISK_TOOL_RUNTIME_EXEC or name in HIGH_RISK_TOOL_RUNTIME_AUTOMATION:
        return False
    if classify_destructive_tool_call(name, dict(args or {})) is not None:
        return False
    if bool(getattr(tool_obj, "read_only", False)):
        return True
    return name in CONCURRENCY_SAFE_READ_ONLY_TOOLS


def partition_tool_calls_for_execution(
    flags: Sequence[bool],
) -> List[Tuple[bool, List[int]]]:
    """Partition ordered tool indices into parallel read-only vs sequential batches."""
    batches: List[Tuple[bool, List[int]]] = []
    current_parallel: List[int] = []

    def _flush_parallel() -> None:
        nonlocal current_parallel
        if not current_parallel:
            return
        batches.append((len(current_parallel) > 1, current_parallel))
        current_parallel = []

    for idx, is_parallel in enumerate(flags):
        if is_parallel:
            current_parallel.append(idx)
        else:
            _flush_parallel()
            batches.append((False, [idx]))
    _flush_parallel()
    return batches


def extract_snapshot_media_path(
    *,
    name: str,
    result: str,
    workspace: Optional[str],
) -> Optional[str]:
    """Extract an absolute snapshot path from camera/stream check tool output."""
    if name not in _SNAPSHOT_TOOL_NAMES:
        return None
    res_data = json.loads(str(result or "{}"))
    if not isinstance(res_data, dict) or not res_data.get("ok"):
        return None

    def _find_snapshot_path(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        path = str(data.get("snapshot_path") or "").strip()
        if path:
            return path
        for key in ("payload", "steps", "capture"):
            if key in data:
                nested = _find_snapshot_path(data[key])
                if nested:
                    return nested
        return ""

    snapshot_path = _find_snapshot_path(res_data)
    payload_dict = res_data.get("payload")
    email_to = str(
        res_data.get("email_to")
        or (payload_dict.get("email_to") if isinstance(payload_dict, dict) else "")
        or ""
    ).strip()
    if not snapshot_path or email_to:
        return None

    path_obj = Path(snapshot_path)
    if not path_obj.is_absolute():
        base = Path(workspace or Path.cwd()).resolve()
        path_obj = (base / path_obj).resolve()
    return str(path_obj)


def extract_tool_result_preview_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("error", "text", "content", "message", "result", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                nested = extract_tool_result_preview_from_payload(value)
                if nested:
                    return nested
        if isinstance(payload.get("items"), list):
            parts: list[str] = []
            for item in payload.get("items")[:3]:
                nested = extract_tool_result_preview_from_payload(item)
                if nested:
                    parts.append(nested)
            if parts:
                return "\n".join(parts)
    if isinstance(payload, list):
        parts = []
        for item in payload[:3]:
            nested = extract_tool_result_preview_from_payload(item)
            if nested:
                parts.append(nested)
        if parts:
            return "\n".join(parts)
    return ""


def extract_tool_result_preview(raw_result: Any) -> str:
    raw = str(raw_result or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
        preview = extract_tool_result_preview_from_payload(payload)
    except Exception:
        preview = raw
    preview = re.sub(r"<think>[\s\S]*?</think>", "", str(preview or ""))
    preview = strip_tool_call_markup(preview)
    preview = re.sub(r"\s+", " ", preview).strip()
    return preview


def tool_result_looks_user_ready(tool_name: str, preview: str) -> bool:
    name = str(tool_name or "").strip().lower()
    text = str(preview or "").strip()
    if not text:
        return False
    if name in {
        "web_search",
        "gui_web_get_dom_text",
        "gui_pdf_get_text",
        "gui_pdf_summarize",
    }:
        return True
    if name in {"read_file", "extract_pdf_text"} and len(text) >= 80:
        return True
    if text.lower().startswith(("results for:", "current conditions", "weather")):
        return True
    return False


def build_tool_only_fallback_answer(tool_runs: Sequence[Any]) -> str:
    if not tool_runs:
        return ""
    last = tool_runs[-1]
    preview = extract_tool_result_preview(last.result)
    if preview:
        if tool_result_looks_user_ready(last.name, preview):
            return preview[:800]
        return (
            f"I ran `{last.name}` and recovered the tool output below because the "
            "model returned an empty final response.\n\n"
            f"{preview[:800]}"
        )
    return (
        f"I ran `{last.name}` but the model returned an empty final response and "
        "the tool did not provide readable output. Please retry."
    )


def can_skip_empty_final_repair(tool_runs: Sequence[Any]) -> bool:
    raw = str(os.getenv("ANNOLID_AGENT_EMPTY_FINAL_FASTPATH", "1")).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if not tool_runs:
        return False
    last = tool_runs[-1]
    preview = extract_tool_result_preview(last.result)
    if not preview:
        return False
    return tool_result_looks_user_ready(last.name, preview)


def repair_tool_arguments_for_execution(
    *,
    name: str,
    args: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], bool]:
    repaired = dict(args or {})
    tool_name = str(name or "").strip().lower()
    if tool_name != "web_search":
        return repaired, False
    query = str(repaired.get("query") or repaired.get("q") or "").strip()
    if query:
        repaired["query"] = query
        return repaired, False
    derived_query, repaired_prompt = derive_web_lookup_prompt_from_messages(messages)
    if derived_query:
        repaired["query"] = derived_query
        return repaired, bool(repaired_prompt)
    return repaired, False


__all__ = [
    "ThinkTagStreamParser",
    "build_tool_only_fallback_answer",
    "can_skip_empty_final_repair",
    "compact_tool_result_for_llm",
    "extract_snapshot_media_path",
    "extract_tool_result_preview",
    "extract_tool_result_preview_from_payload",
    "is_concurrency_safe_tool_call",
    "partition_tool_calls_for_execution",
    "repair_tool_arguments_for_execution",
    "tool_result_looks_user_ready",
]
