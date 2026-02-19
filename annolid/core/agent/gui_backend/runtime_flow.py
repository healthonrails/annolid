from __future__ import annotations

from typing import Callable


def maybe_handle_ollama_plain_mode(
    *,
    provider: str,
    model: str,
    prompt: str,
    show_tool_trace: bool,
    prompt_may_need_tools: Callable[[str], bool],
    plain_mode_remaining: Callable[[str], int],
    plain_mode_decrement: Callable[[str], int],
    recover_with_plain_reply: Callable[[], str],
    persist_turn: Callable[[str, str], None],
    emit_final: Callable[[str, bool], None],
    logger: object,
) -> bool:
    if provider != "ollama":
        return False

    remaining_plain_turns = int(plain_mode_remaining(model) or 0)
    wants_tools = bool(prompt_may_need_tools(prompt))
    if remaining_plain_turns > 0 and not wants_tools:
        updated_remaining = plain_mode_decrement(model)
        logger.warning(  # type: ignore[attr-defined]
            "annolid-bot model is in temporary plain mode; skipping agent/tool loop model=%s remaining_turns=%d",
            model,
            updated_remaining,
        )
        text = recover_with_plain_reply()
        if not text:
            text = (
                "Model returned empty output in plain mode. "
                f"Provider={provider}, model={model}. "
                "Please switch to another Ollama model for Annolid Bot."
            )
        if show_tool_trace:
            text = (
                f"{text}\n\n[Tool Trace]\n(skipped: temporary plain-mode fallback)"
            ).strip()
        if text.strip():
            persist_turn(prompt, text)
        emit_final(text, False)
        return True

    if remaining_plain_turns > 0 and wants_tools:
        logger.info(  # type: ignore[attr-defined]
            "annolid-bot bypassing temporary plain mode due tool-intent prompt model=%s remaining_turns=%d",
            model,
            remaining_plain_turns,
        )
    return False


def emit_agent_loop_result(
    *,
    prompt: str,
    text: str,
    persist_turn: Callable[[str, str], None],
    emit_final: Callable[[str, bool], None],
) -> None:
    if text.strip():
        persist_turn(prompt, text)
    emit_final(text, False)
