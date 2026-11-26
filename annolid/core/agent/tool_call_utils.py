from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MARKDOWN_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?:functions\.)?([A-Za-z0-9_-]+)"
    r"(?::([A-Za-z0-9_-]+))?\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_KIMI_TOOL_CALL_STRIP_RE = re.compile(
    r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


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


def _try_parse_json_object(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return dict(parsed)
    # Local/OpenAI-compat models occasionally double-encode tool arguments as a JSON string.
    if isinstance(parsed, str):
        inner_text = parsed.strip()
        if inner_text.startswith("{"):
            try:
                inner = json.loads(inner_text)
            except Exception:
                return None
            if isinstance(inner, dict):
                return dict(inner)
    return None


def normalize_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    if isinstance(raw_args, str):
        text = raw_args.strip()
        if not text:
            return {}
        parsed = _try_parse_json_object(text)
        if parsed is not None:
            return parsed

        # Strip surrounding markdown code fences (e.g. ```json { ... } ```).
        fenced_match = _MARKDOWN_FENCE_RE.match(text)
        candidate = fenced_match.group(1).strip() if fenced_match else text
        if candidate != text:
            parsed = _try_parse_json_object(candidate)
            if parsed is not None:
                return parsed

        # Repair trailing commas before closing braces/brackets.
        no_trailing = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if no_trailing != candidate:
            parsed = _try_parse_json_object(no_trailing)
            if parsed is not None:
                return parsed

        # Optional json_repair recovery for near-JSON objects (single quotes,
        # unescaped newlines, missing closing brace). Require an opening brace
        # and key/value separator so arbitrary garbage still fails closed to _raw.
        if candidate.startswith("{") and ":" in candidate:
            try:
                import json_repair  # type: ignore

                repaired = json_repair.loads(candidate)
                if isinstance(repaired, dict) and repaired:
                    return dict(repaired)
            except Exception:
                pass
        return {"_raw": raw_args}
    return {"_raw": raw_args}


def extract_text_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract text-encoded tool calls (Kimi and Qwen/Hermes markup formats)."""
    source = str(text or "")
    if not source:
        return []
    found: List[Dict[str, Any]] = []

    if "<|tool_call_begin|>" in source:
        for idx, match in enumerate(_KIMI_TOOL_CALL_RE.finditer(source)):
            name = str(match.group(1) or "").strip()
            call_idx = str(match.group(2) or idx)
            raw_args = str(match.group(3) or "").strip()
            if not name:
                continue
            found.append(
                {
                    "id": f"call_{call_idx}",
                    "name": name,
                    "arguments": normalize_tool_arguments(raw_args),
                }
            )

    if "<tool_call>" in source.lower():
        for match in _QWEN_TOOL_CALL_RE.finditer(source):
            body = str(match.group(1) or "").strip()
            if not body:
                continue
            payload = normalize_tool_arguments(body)
            if "_raw" in payload and len(payload) == 1:
                continue
            name = str(payload.get("name") or "").strip()
            if not name:
                continue
            raw_args: Any = payload.get("arguments")
            if raw_args is None and "parameters" in payload:
                raw_args = payload.get("parameters")
            elif raw_args is None and "args" in payload:
                raw_args = payload.get("args")
            call_id = str(payload.get("id") or f"call_{len(found)}")
            found.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": normalize_tool_arguments(raw_args),
                }
            )

    return found


def strip_tool_call_markup(text: str) -> str:
    """Remove Kimi and Qwen/Hermes inline tool-call markup from assistant text."""
    cleaned = str(text or "")
    if not cleaned:
        return ""
    if "<|tool_call" in cleaned:
        cleaned = _KIMI_TOOL_CALL_STRIP_RE.sub("", cleaned)
        cleaned = cleaned.replace("<|tool_calls_section_begin|>", "")
        cleaned = cleaned.replace("<|tool_calls_section_end|>", "")
    if "<tool_call>" in cleaned.lower():
        cleaned = _QWEN_TOOL_CALL_RE.sub("", cleaned)
    return cleaned.strip()


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
