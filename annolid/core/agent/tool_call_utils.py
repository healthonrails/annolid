from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def sanitize_tool_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text or not _TOOL_NAME_RE.match(text):
        return ""
    return text


def sanitize_tool_call_id(raw_id: Any, *, default_index: int = 0) -> str:
    text = str(raw_id or "").strip()
    if not text:
        return f"call_{default_index}"
    if "|" in text:
        left, right = text.split("|", 1)
        left_clean = sanitize_tool_call_id_segment(left)
        right_clean = sanitize_tool_call_id_segment(right)
        if left_clean and right_clean:
            return f"{left_clean}|{right_clean}"
        if left_clean:
            return left_clean
        if right_clean:
            return right_clean
        return f"call_{default_index}"
    cleaned = sanitize_tool_call_id_segment(text)
    return cleaned or f"call_{default_index}"


def sanitize_tool_call_id_segment(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return ""
    trimmed = text[:64]
    return trimmed if _TOOL_CALL_ID_RE.match(trimmed) else trimmed


def normalize_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        text = raw_args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {"_raw": raw_args}
        if isinstance(parsed, dict):
            return dict(parsed)
        return {"_raw": raw_args}
    return {"_raw": raw_args}


def sanitize_tool_call_requests(
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    allowed_tool_names: Optional[Iterable[str]] = None,
    max_calls: Optional[int] = None,
    dedupe: bool = True,
) -> List[Dict[str, Any]]:
    allowed_names: Optional[Set[str]] = None
    if allowed_tool_names is not None:
        allowed_names = {
            clean
            for clean in (sanitize_tool_name(name) for name in allowed_tool_names)
            if clean
        }

    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    limit = None if max_calls is None else max(0, int(max_calls))
    for item in tool_calls:
        name = sanitize_tool_name(item.get("name"))
        if not name:
            continue
        if allowed_names is not None and name not in allowed_names:
            continue
        arguments = normalize_tool_arguments(item.get("arguments"))
        call_id = sanitize_tool_call_id(item.get("id"), default_index=len(cleaned))
        if dedupe:
            signature = (
                f"{call_id}:{name}:"
                f"{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"
            )
            if signature in seen:
                continue
            seen.add(signature)
        cleaned.append({"id": call_id, "name": name, "arguments": arguments})
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


def tool_names_from_schemas(tools: Sequence[Mapping[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    for schema in tools:
        function = schema.get("function")
        if not isinstance(function, Mapping):
            continue
        name = sanitize_tool_name(function.get("name"))
        if name:
            names.add(name)
    return names
