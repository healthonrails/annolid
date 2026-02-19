from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple

from annolid.core.agent.providers import build_ollama_llm_callable
from annolid.core.agent.providers.ollama_utils import (
    collect_ollama_stream,
    extract_ollama_text,
    format_tool_trace,
    normalize_messages_for_ollama,
    parse_ollama_tool_calls,
)


def build_gui_ollama_llm_callable(
    *,
    prompt: str,
    settings: Dict[str, Any],
    prompt_may_need_tools: Callable[[str], bool],
    logger: Any,
    tool_request_timeout_s: float,
    plain_request_timeout_s: float,
):
    return build_ollama_llm_callable(
        prompt=prompt,
        settings=settings,
        parse_tool_calls=parse_ollama_tool_calls,
        normalize_messages=normalize_messages_for_ollama,
        extract_text=extract_ollama_text,
        prompt_may_need_tools=prompt_may_need_tools,
        logger=logger,
        import_module=importlib.import_module,
        tool_request_timeout_s=tool_request_timeout_s,
        plain_request_timeout_s=plain_request_timeout_s,
    )


def collect_gui_ollama_stream(
    stream_iter: Any,
) -> Tuple[str, List[Dict[str, Any]], str]:
    return collect_ollama_stream(stream_iter, parse_ollama_tool_calls)


def parse_gui_ollama_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    return parse_ollama_tool_calls(raw_calls)


def normalize_gui_messages_for_ollama(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return normalize_messages_for_ollama(messages)


def extract_gui_ollama_text(response: Dict[str, Any]) -> str:
    return extract_ollama_text(response)


def format_gui_tool_trace(tool_runs: Any) -> str:
    return format_tool_trace(tool_runs)
