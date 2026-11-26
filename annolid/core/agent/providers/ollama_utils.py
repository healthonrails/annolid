from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from annolid.utils.logger import logger

from ..tool_call_utils import normalize_tool_arguments, sanitize_tool_call_requests


def _tool_call_signature(call: Mapping[str, Any]) -> str:
    return (
        f"{call.get('name', '')}:"
        f"{json.dumps(call.get('arguments', {}), ensure_ascii=False, sort_keys=True, default=str)}"
    )


def merge_streamed_ollama_tool_calls(
    tool_calls_by_id: Dict[str, Dict[str, Any]],
    incoming_calls: Iterable[Mapping[str, Any]],
    *,
    seen_signatures: Optional[set[str]] = None,
) -> None:
    """Merge streamed Ollama tool calls without overwriting distinct idless calls."""
    signatures = seen_signatures if seen_signatures is not None else set()
    for raw_call in incoming_calls:
        call = dict(raw_call)
        call_id = str(call.get("id") or "").strip()
        sig = _tool_call_signature(call)
        if call_id in tool_calls_by_id:
            existing = tool_calls_by_id[call_id]
            if _tool_call_signature(existing) == sig:
                continue
            # Ollama often omits tool call IDs, so parse_ollama_tool_calls assigns
            # `ollama_call_0` on every chunk. Allocate a fresh ID rather than
            # silently clobbering an earlier distinct call.
            if call_id.startswith(("ollama_call_", "call_")):
                if sig in signatures:
                    continue
                next_idx = len(tool_calls_by_id)
                while f"ollama_call_{next_idx}" in tool_calls_by_id:
                    next_idx += 1
                call_id = f"ollama_call_{next_idx}"
                call["id"] = call_id
        elif sig in signatures and call_id.startswith(("ollama_call_", "call_")):
            continue
        tool_calls_by_id[call_id] = call
        signatures.add(sig)


def collect_ollama_stream(
    stream_iter: Any,
    parse_tool_calls,
    *,
    allowed_tool_names: Optional[Iterable[str]] = None,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Collect non-streaming output from an Ollama stream iterator."""
    chunks: List[str] = []
    tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
    seen_signatures: set[str] = set()
    done_reason = "stop"
    for part in stream_iter:
        if not isinstance(part, dict):
            continue
        done_reason = str(part.get("done_reason") or done_reason)
        msg = part.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content:
                chunks.append(content)
            raw_tool_calls = msg.get("tool_calls")
            if raw_tool_calls:
                sanitized = sanitize_tool_call_requests(
                    parse_tool_calls(raw_tool_calls),
                    allowed_tool_names=allowed_tool_names,
                )
                merge_streamed_ollama_tool_calls(
                    tool_calls_by_id,
                    sanitized,
                    seen_signatures=seen_signatures,
                )
    return "".join(chunks).strip(), list(tool_calls_by_id.values()), done_reason


def parse_ollama_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for idx, item in enumerate(list(raw_calls or [])):
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        args = normalize_tool_arguments(fn.get("arguments", {}))
        call_id = str(item.get("id") or f"ollama_call_{idx}")
        tool_calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": dict(args),
            }
        )
    return tool_calls


def _normalize_ollama_tool_calls(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for item in list(raw_tool_calls or []):
        if not isinstance(item, Mapping):
            continue
        fn = item.get("function")
        if isinstance(fn, Mapping):
            name = str(fn.get("name") or "").strip()
            raw_args = fn.get("arguments", {})
        else:
            name = str(item.get("name") or "").strip()
            raw_args = item.get("arguments", {})
        if not name:
            continue
        entry: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "arguments": normalize_tool_arguments(raw_args),
            },
        }
        call_id = str(item.get("id") or "").strip()
        if call_id:
            entry["id"] = call_id
        converted.append(entry)
    return converted


def normalize_messages_for_ollama(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        out: Dict[str, Any] = {"role": role}
        if isinstance(content, list):
            text_parts: List[str] = []
            images: List[Any] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url") or {}
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url") or "")
                        if url.startswith("data:image/") and ";base64," in url:
                            try:
                                images.append(
                                    base64.b64decode(url.split(";base64,", 1)[1])
                                )
                            except (ValueError, binascii.Error):
                                logger.debug(
                                    "Ignoring malformed base64 Ollama image content."
                                )
            out["content"] = "\n".join([p for p in text_parts if p]).strip()
            if images:
                out["images"] = images
        else:
            out["content"] = str(content or "")
            existing_images = msg.get("images")
            if isinstance(existing_images, list) and existing_images:
                out["images"] = list(existing_images)

        if role == "assistant":
            raw_tool_calls = msg.get("tool_calls")
            if raw_tool_calls:
                normalized_calls = _normalize_ollama_tool_calls(raw_tool_calls)
                if normalized_calls:
                    out["tool_calls"] = normalized_calls
        elif role == "tool":
            tool_name = str(msg.get("tool_name") or msg.get("name") or "").strip()
            if tool_name:
                out["tool_name"] = tool_name

        normalized.append(out)
    return normalized


def extract_ollama_text(response: Dict[str, Any]) -> str:
    msg = response.get("message") or {}
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                elif isinstance(item, str) and item.strip():
                    parts.append(item)
            if parts:
                return "\n".join(parts).strip()
        thinking = msg.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            return thinking
        text = msg.get("text")
        if isinstance(text, str) and text.strip():
            return text
        output_text = msg.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output = msg.get("output")
        if isinstance(output, str) and output.strip():
            return output

    fallback = response.get("response")
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    top_text = response.get("text")
    if isinstance(top_text, str) and top_text.strip():
        return top_text
    output_text_top = response.get("output_text")
    if isinstance(output_text_top, str) and output_text_top.strip():
        return output_text_top
    return ""


def format_tool_trace(tool_runs: Any) -> str:
    lines: List[str] = []
    for run in tool_runs:
        name = str(getattr(run, "name", "") or "").strip()
        args = getattr(run, "arguments", {}) or {}
        result = str(getattr(run, "result", "") or "").strip()
        if not name:
            continue
        lines.append(f"- `{name}` args={args}")
        if result:
            lines.append(f"  -> {result}")
    if not lines:
        return "[Tool Trace]\n(no tool calls)"
    return "[Tool Trace]\n" + "\n".join(lines)
