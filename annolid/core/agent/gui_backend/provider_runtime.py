from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from annolid.core.agent.providers import (
    run_gemini_chat,
    run_ollama_streaming_chat,
    run_openai_compat_chat,
)
from annolid.utils.llm_settings import provider_kind


def has_image_context(image_path: Optional[str]) -> bool:
    path = str(image_path or "").strip()
    return bool(path and os.path.exists(path))


def run_fast_mode(
    *,
    chat_mode: str,
    run_fast_provider_chat: Callable[[bool, bool], None],
) -> None:
    if chat_mode == "vision_describe":
        run_fast_provider_chat(True, False)
        return
    raise ValueError(f"Unsupported fast mode '{chat_mode}'.")


def run_fast_provider_chat(
    *,
    prompt: str,
    image_path: Optional[str],
    model: str,
    provider: str,
    settings: Dict[str, Any],
    include_image: bool,
    include_history: bool,
    load_history_messages: Callable[[], List[Dict[str, Any]]],
    fast_mode_timeout_seconds: Callable[[], float],
    emit_progress: Callable[[str], None],
    emit_chunk: Callable[[str], None],
    emit_final: Callable[[str, bool], None],
    persist_turn: Callable[[str, str], None],
) -> None:
    image = image_path if include_image else ""
    load_history = load_history_messages if include_history else (lambda: [])
    timeout_s = fast_mode_timeout_seconds()
    kind = provider_kind(settings, provider)
    emit_progress(f"Fast mode provider call ({kind})")
    if kind == "ollama":
        run_ollama_streaming_chat(
            prompt=prompt,
            image_path=image,
            model=model,
            settings=settings,
            load_history_messages=load_history,
            emit_chunk=emit_chunk,
            emit_final=lambda message, is_error: emit_final(message, bool(is_error)),
            persist_turn=lambda user_text, assistant_text: persist_turn(
                user_text, assistant_text
            ),
        )
        return
    if kind == "openai_compat":
        user_prompt, text = run_openai_compat_chat(
            prompt=prompt,
            image_path=image,
            model=model,
            provider_name=provider,
            settings=settings,
            load_history_messages=load_history,
            max_tokens=320 if include_image else 640,
            timeout_s=timeout_s,
        )
        persist_turn(user_prompt, text)
        emit_final(text, False)
        return
    if kind == "gemini":
        user_prompt, text = run_gemini_chat(
            prompt=prompt,
            image_path=image,
            model=model,
            provider_name=provider,
            settings=settings,
        )
        persist_turn(user_prompt, text)
        emit_final(text, False)
        return
    raise ValueError(f"Unsupported provider '{provider}' in fast mode.")


def run_ollama_chat(
    *,
    prompt: str,
    image_path: Optional[str],
    model: str,
    settings: Dict[str, Any],
    load_history_messages: Callable[[], List[Dict[str, Any]]],
    emit_chunk: Callable[[str], None],
    emit_final: Callable[[str, bool], None],
    persist_turn: Callable[[str, str], None],
) -> None:
    run_ollama_streaming_chat(
        prompt=prompt,
        image_path=image_path,
        model=model,
        settings=settings,
        load_history_messages=load_history_messages,
        emit_chunk=emit_chunk,
        emit_final=lambda message, is_error: emit_final(message, bool(is_error)),
        persist_turn=lambda user_text, assistant_text: persist_turn(
            user_text, assistant_text
        ),
    )


def run_openai_chat(
    *,
    prompt: str,
    image_path: Optional[str],
    model: str,
    provider_name: str,
    settings: Dict[str, Any],
    load_history_messages: Callable[[], List[Dict[str, Any]]],
    timeout_s: Optional[float] = None,
    max_tokens: int = 4096,
) -> Tuple[str, str]:
    return run_openai_compat_chat(
        prompt=prompt,
        image_path=image_path,
        model=model,
        provider_name=provider_name,
        settings=settings,
        load_history_messages=load_history_messages,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )


def run_gemini_provider_chat(
    *,
    prompt: str,
    image_path: Optional[str],
    model: str,
    provider_name: str,
    settings: Dict[str, Any],
) -> Tuple[str, str]:
    return run_gemini_chat(
        prompt=prompt,
        image_path=image_path,
        model=model,
        provider_name=provider_name,
        settings=settings,
    )
