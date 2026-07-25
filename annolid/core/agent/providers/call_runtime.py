from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Mapping, Optional


ERROR_DETAIL_LIMIT = 2000
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
RETRYABLE_ERROR_KINDS = frozenset(
    {"connection", "empty", "overloaded", "rate_limit", "server_error", "timeout"}
)
NON_RETRYABLE_QUOTA_MARKERS = (
    "billing hard limit",
    "billing_hard_limit",
    "credit balance",
    "insufficient balance",
    "insufficient quota",
    "insufficient_balance",
    "insufficient_quota",
    "out of credits",
    "payment required",
    "payment_required",
    "quota exceeded",
    "quota_exceeded",
    "quota exhausted",
    "quota_exhausted",
)
TRANSIENT_ERROR_MARKERS = (
    "connection refused",
    "connection reset",
    "gateway timeout",
    "overloaded",
    "rate limit",
    "server error",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
)
_SECRET_SUBSTITUTIONS = (
    (
        re.compile(
            r"(?i)\b(authorization)\b(\s*[:=]\s*)(?:bearer\s+)?"
            r"([^\s,;\"']+)"
        ),
        r"\1\2<redacted>",
    ),
    (
        re.compile(
            r"(?i)\b(api[_ -]?key(?:\s+provided)?|access[_ -]?token|"
            r"refresh[_ -]?token|password|secret)\b"
            r"(\s*[:=]\s*)[\"']?([^\s,;\"'&]+)"
        ),
        r"\1\2<redacted>",
    ),
    (
        re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|key)=)([^&#\s]+)"),
        r"\1<redacted>",
    ),
    (
        re.compile(r"(?i)(https?://[^:/\s]+:)([^@\s/]+)(@)"),
        r"\1<redacted>\3",
    ),
    (
        re.compile(r"(?i)\b(?:sk-(?:or-)?|nvapi-|ghp_)[a-z0-9_-]{12,}\b"),
        "<redacted>",
    ),
)
_PROVIDER_MESSAGE_KEYS = frozenset(
    {"content", "name", "role", "tool_call_id", "tool_calls"}
)


@dataclass(frozen=True)
class ProviderErrorDetails:
    message: str
    status_code: Optional[int]
    kind: Optional[str]
    error_type: Optional[str]
    error_code: Optional[str]
    retry_after_s: Optional[float]
    should_retry: Optional[bool]


def sanitize_provider_error(value: Any, *, limit: int = ERROR_DETAIL_LIMIT) -> str:
    """Return bounded provider error text with common credential shapes redacted."""
    text = str(value or "").strip()
    for pattern, replacement in _SECRET_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    max_chars = max(128, int(limit))
    if len(text) > max_chars:
        text = f"{text[:max_chars]}…"
    return text


def _strip_internal_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_internal_metadata(item)
            for key, item in value.items()
            if str(key) != "_meta"
        }
    if isinstance(value, list):
        return [_strip_internal_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_internal_metadata(item) for item in value]
    return value


def sanitize_openai_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy messages into a portable OpenAI-compatible request shape."""
    sanitized: List[Dict[str, Any]] = []
    for raw_message in messages:
        clean = {
            key: _strip_internal_metadata(value)
            for key, value in dict(raw_message).items()
            if key in _PROVIDER_MESSAGE_KEYS
        }
        if (
            str(clean.get("role") or "") == "assistant"
            and clean.get("tool_calls")
            and ("content" not in clean or clean.get("content") == "")
        ):
            clean["content"] = None
        sanitized.append(clean)
    return sanitized


def sanitize_tool_schemas(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [_strip_internal_metadata(dict(tool)) for tool in tools]


def _header_value(headers: Any, name: str) -> Any:
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is None:
            value = getter(name.title())
        if value is not None:
            return value
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return value
    return None


def _retry_after_seconds(headers: Any) -> Optional[float]:
    with suppress(TypeError, ValueError):
        retry_ms = _header_value(headers, "retry-after-ms")
        if retry_ms is not None:
            value = float(retry_ms) / 1000.0
            if value > 0:
                return value

    retry_after = _header_value(headers, "retry-after")
    if retry_after is None:
        return None
    text = str(retry_after).strip()
    if not text:
        return None
    with suppress(TypeError, ValueError):
        value = float(text)
        if value > 0:
            return value
    try:
        retry_at = parsedate_to_datetime(text)
    except Exception:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    remaining = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
    return max(0.0, remaining)


def _error_type_code(payload: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(payload, str):
        with suppress(Exception):
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                payload = parsed
    if not isinstance(payload, Mapping):
        return None, None
    error = payload.get("error")
    type_value = payload.get("type")
    code_value = payload.get("code")
    if isinstance(error, Mapping):
        type_value = error.get("type") or type_value
        code_value = error.get("code") or code_value
    error_type = str(type_value or "").strip().lower() or None
    error_code = str(code_value or "").strip().lower() or None
    return error_type, error_code


def classify_provider_exception(exc: Exception) -> ProviderErrorDetails:
    """Extract portable retry metadata from common provider SDK exceptions."""
    response = getattr(exc, "response", None)
    status_value = getattr(exc, "status_code", None)
    if status_value is None and response is not None:
        status_value = getattr(response, "status_code", None)
    status_code: Optional[int] = None
    with suppress(TypeError, ValueError):
        if status_value is not None:
            status_code = int(status_value)

    headers = getattr(exc, "headers", None)
    if headers is None and response is not None:
        headers = getattr(response, "headers", None)
    payload = getattr(exc, "body", None)
    if payload is None and response is not None:
        payload = getattr(response, "body", None)
    error_type, error_code = _error_type_code(payload)

    detail = sanitize_provider_error(exc) or exc.__class__.__name__
    lower = " ".join(
        part
        for part in (
            detail.lower(),
            str(error_type or ""),
            str(error_code or ""),
            exc.__class__.__name__.lower(),
        )
        if part
    )
    if status_code is None:
        match = re.search(r"\b([45]\d\d)\b", lower)
        if match:
            status_code = int(match.group(1))

    if any(marker in lower for marker in NON_RETRYABLE_QUOTA_MARKERS):
        error_kind = "quota"
    elif "timeout" in lower or "timed out" in lower:
        error_kind = "timeout"
    elif "connection" in lower or "network" in lower:
        error_kind = "connection"
    elif status_code in {401, 403} or any(
        marker in lower
        for marker in ("authentication", "invalid api key", "invalid_api_key")
    ):
        error_kind = "authentication"
    elif status_code == 429 or any(
        marker in lower for marker in ("rate limit", "too many requests")
    ):
        error_kind = "rate_limit"
    elif status_code is not None and status_code >= 500:
        error_kind = "server_error"
    elif any(
        marker in lower
        for marker in ("context length", "context_length", "maximum context")
    ):
        error_kind = "context_length"
    elif status_code in {400, 404, 422} or isinstance(exc, ValueError):
        error_kind = "invalid_request"
    else:
        error_kind = None

    should_retry_header = str(_header_value(headers, "x-should-retry") or "").lower()
    should_retry: Optional[bool] = None
    if should_retry_header == "true":
        should_retry = True
    elif should_retry_header == "false":
        should_retry = False
    if error_kind == "quota":
        should_retry = False

    return ProviderErrorDetails(
        message=detail,
        status_code=status_code,
        kind=error_kind,
        error_type=error_type,
        error_code=error_code,
        retry_after_s=_retry_after_seconds(headers),
        should_retry=should_retry,
    )
