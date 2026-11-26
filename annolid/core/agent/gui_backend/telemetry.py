from __future__ import annotations

import inspect
from typing import Any, Callable


def log_runtime_timeouts(
    *,
    logger: Any,
    session_id: str,
    model: str,
    loop_llm_s: float,
    loop_tool_s: float,
    ollama_tool_s: float,
    ollama_plain_s: float,
    recover_s: float,
    recover_nudge_s: float,
) -> None:
    logger.info(
        "annolid-bot runtime timeouts session=%s model=%s loop_llm_s=%.1f loop_tool_s=%.1f ollama_tool_s=%.1f ollama_plain_s=%.1f recover_s=%.1f recover_nudge_s=%.1f",
        session_id,
        model,
        loop_llm_s,
        loop_tool_s,
        ollama_tool_s,
        ollama_plain_s,
        recover_s,
        recover_nudge_s,
    )


def log_agent_result(
    *,
    logger: Any,
    session_id: str,
    provider: str,
    model: str,
    result: Any,
    used_recovery: bool,
    used_direct_gui_fallback: bool,
) -> None:
    logger.info(
        "annolid-bot agent result session=%s provider=%s model=%s iterations=%s tool_runs=%d",
        session_id,
        provider,
        model,
        getattr(result, "iterations", "?"),
        len(getattr(result, "tool_runs", ()) or ()),
    )
    if used_recovery:
        logger.info(
            "annolid-bot recovered empty agent reply with plain ollama answer session=%s model=%s",
            session_id,
            model,
        )
    if used_direct_gui_fallback:
        logger.info(
            "annolid-bot responded from direct gui fallback session=%s model=%s",
            session_id,
            model,
        )


def wrap_tool_callback(
    *,
    name: str,
    callback: Callable[..., Any],
    emit_progress: Callable[[str], None],
) -> Callable[..., Any]:
    label = str(name or "tool").replace("_", " ")

    def _emit_outcome(result: Any) -> None:
        if isinstance(result, dict):
            if bool(result.get("ok")):
                emit_progress(f"Finished tool: {label}")
            else:
                emit_progress(f"Tool failed: {label}")
        else:
            emit_progress(f"Finished tool: {label}")

    if inspect.iscoroutinefunction(callback):

        async def _wrapped_async(*args, **kwargs):
            emit_progress(f"Running tool: {label}")
            try:
                result = await callback(*args, **kwargs)
            except Exception:
                emit_progress(f"Tool failed: {label}")
                raise
            _emit_outcome(result)
            return result

        return _wrapped_async

    def _wrapped(*args, **kwargs):
        emit_progress(f"Running tool: {label}")
        try:
            result = callback(*args, **kwargs)
        except Exception:
            emit_progress(f"Tool failed: {label}")
            raise
        if inspect.isawaitable(result):

            async def _await_result():
                try:
                    awaited = await result
                except Exception:
                    emit_progress(f"Tool failed: {label}")
                    raise
                _emit_outcome(awaited)
                return awaited

            return _await_result()
        _emit_outcome(result)
        return result

    return _wrapped
