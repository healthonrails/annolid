from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Tuple


def collect_ollama_stream(
    stream_iter: Any, parse_tool_calls
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Collect non-streaming output from an Ollama stream iterator."""
    chunks: List[str] = []
    tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
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
                for call in parse_tool_calls(raw_tool_calls):
                    call_id = str(call.get("id") or f"call_{len(tool_calls_by_id)}")
                    tool_calls_by_id[call_id] = call
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
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                args = parsed if isinstance(parsed, dict) else {"_raw": args}
            except json.JSONDecodeError:
                args = {"_raw": args}
        elif not isinstance(args, dict):
            args = {"_raw": args}
        call_id = str(item.get("id") or f"ollama_call_{idx}")
        tool_calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": dict(args),
            }
        )
    return tool_calls


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
                            except Exception:
                                continue
            out["content"] = "\n".join([p for p in text_parts if p]).strip()
            if images:
                out["images"] = images
        else:
            out["content"] = str(content or "")
            existing_images = msg.get("images")
            if isinstance(existing_images, list) and existing_images:
                out["images"] = list(existing_images)
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
