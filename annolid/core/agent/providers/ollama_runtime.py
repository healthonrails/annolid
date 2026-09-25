"""Context allocation and stream normalization shared by Ollama chat paths."""

from __future__ import annotations

import re
from typing import Any, Iterator, Mapping

from .base import ProviderCallError


DEFAULT_NUM_CTX = 16384
DEFAULT_MAX_NUM_CTX = 32768


def is_context_length_error(error: Any) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "exceed_context_size_error",
            "exceeds the available context size",
            "context length exceeded",
            "maximum context length",
        )
    )


def is_tool_support_error(error: Any) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "does not support tools",
            "doesn't support tools",
            "tools are not supported",
        )
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"ollama.{name} must be a positive integer")
    try:
        parsed = int(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"ollama.{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"ollama.{name} must be a positive integer")
    return parsed


def stream_ollama_chat(
    ollama: Any,
    *,
    settings: Mapping[str, Any],
    model: str,
    messages: Any,
    logger: Any,
    tools: Any = None,
) -> Iterator[dict[str, Any]]:
    """Retry a rejected prompt once with more context, never replay partial output.

    Preserve messages and tool schemas verbatim. The ceiling bounds automatic
    growth; it is not a claim about the model's supported context capacity.
    """
    config = settings.get("ollama") or {}
    num_ctx = _positive_int(config.get("num_ctx", DEFAULT_NUM_CTX), "num_ctx")
    max_num_ctx = _positive_int(
        config.get("max_num_ctx", max(DEFAULT_MAX_NUM_CTX, num_ctx)), "max_num_ctx"
    )
    if max_num_ctx < num_ctx:
        raise ValueError(
            "ollama.max_num_ctx must be greater than or equal to ollama.num_ctx"
        )
    for attempt in range(2):
        stream = None
        emitted = False
        try:
            logger.info(
                "annolid-bot ollama context model=%s num_ctx=%d max_num_ctx=%d attempt=%d",
                model,
                num_ctx,
                max_num_ctx,
                attempt + 1,
            )
            stream = ollama.chat(
                model=model,
                messages=messages,
                tools=tools,
                stream=True,
                options={"num_ctx": num_ctx},
            )
            for part in stream:
                if not isinstance(part, Mapping) and callable(
                    getattr(part, "model_dump", None)
                ):
                    part = part.model_dump()
                if not isinstance(part, Mapping):
                    raise TypeError("Ollama returned an unsupported stream response")
                if part.get("error"):
                    raise RuntimeError(str(part["error"]))
                message = part.get("message") or {}
                # A stream may emit empty metadata before rejecting the prompt.
                emitted = emitted or bool(
                    message.get("content")
                    or message.get("thinking")
                    or message.get("tool_calls")
                )
                yield dict(part)
            return
        except Exception as exc:
            if not is_context_length_error(exc):
                raise
            match = re.search(r'["\']?n_prompt_tokens["\']?\s*:\s*(\d+)', str(exc))
            if match is None:
                match = re.search(r"request\s*\((\d+)\s+tokens\)", str(exc))
            required = int(match.group(1)) + 1024 if match else num_ctx * 2
            target = max(num_ctx + 1024, ((required + 1023) // 1024) * 1024)
            if attempt or emitted or target > max_num_ctx:
                raise ProviderCallError(
                    f"Ollama context limit exceeded for {model} (num_ctx={num_ctx}, "
                    f"max_num_ctx={max_num_ctx}). Shorten the conversation or configure "
                    "ollama.num_ctx and ollama.max_num_ctx within your model and memory "
                    f"limits. Original error: {exc}",
                    provider="ollama",
                    model=model,
                    error_kind="context_length",
                    retryable=False,
                ) from exc
            logger.warning(
                "annolid-bot retrying Ollama context overflow model=%s num_ctx=%d next_num_ctx=%d",
                model,
                num_ctx,
                target,
            )
            num_ctx = target
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
