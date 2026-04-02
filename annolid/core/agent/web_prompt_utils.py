from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_WEB_LOOKUP_CONTROL_LINE_RE = re.compile(
    r"^/(skill|skills|tool|tools|capabilities|caps)\b(?:\s+.*)?$",
    re.IGNORECASE,
)


def normalize_web_lookup_prompt(prompt: str) -> tuple[str, bool]:
    raw = str(prompt or "")
    body_lines: list[str] = []
    repaired = False
    for line in raw.splitlines():
        stripped = str(line or "").strip()
        if not stripped:
            body_lines.append(line)
            continue
        if _WEB_LOOKUP_CONTROL_LINE_RE.match(stripped):
            repaired = True
            continue
        body_lines.append(line)
    normalized = "\n".join(body_lines).strip()
    if normalized != raw.strip():
        repaired = True
    return normalized, repaired


def derive_web_lookup_prompt_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, bool]:
    for msg in reversed(messages):
        if str(msg.get("role") or "").strip().lower() != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            text = "\n".join(text_parts)
        else:
            text = str(content or "")
        return normalize_web_lookup_prompt(text)
    return "", False
