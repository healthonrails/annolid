from __future__ import annotations

from typing import Any, Callable

from annolid.utils.llm_settings import provider_kind


def is_provider_config_error(exc: Exception | str) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "requires api key",
            "api key is missing",
            "missing api key",
            "invalid api key",
            "unsupported provider/model for agent loop",
        )
    )


def format_provider_config_error(raw_error: str, *, provider: str) -> str:
    message = str(raw_error or "").strip()
    if is_provider_config_error(message):
        return (
            f"Provider configuration error for '{provider}'. {message} "
            "Open AI Model Settings and configure the provider API key/base URL."
        )
    return message or "Provider configuration is invalid."


def is_provider_timeout_error(exc: Exception | str) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return (
        "timed out" in text
        or "timeout" in text
        or "gateway timeout" in text
        or "504" in text
    )


def run_provider_fallback(
    *,
    original_error: Exception,
    settings: dict[str, Any],
    provider: str,
    model: str,
    session_id: str,
    fallback_timeout_retry_seconds: Callable[[], float],
    fallback_retry_timeout_seconds: Callable[[], float],
    run_ollama: Callable[[], None],
    run_openai: Callable[[str, float, int], None],
    run_gemini: Callable[[], None],
    emit_progress: Callable[[str], None],
    emit_final: Callable[[str, bool], None],
    format_dependency_error: Callable[[str], str],
    logger: Any,
) -> None:
    """Run legacy provider fallback when agent loop setup/execution fails."""
    try:
        # Keep backward-compatible fallback behavior if agent loop setup fails.
        emit_progress("Agent loop failed, trying provider fallback")
        kind = provider_kind(settings, provider)
        if kind == "openai_compat" and is_provider_timeout_error(original_error):
            retry_timeout_s = fallback_timeout_retry_seconds()
            emit_progress("Provider timeout detected; running one bounded retry")
            run_openai(provider, retry_timeout_s, 512)
            logger.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=fallback_timeout_retry_ok",
                session_id,
                provider,
                model,
            )
            return
        if kind == "ollama":
            run_ollama()
        elif kind == "openai_compat":
            run_openai(provider, fallback_retry_timeout_seconds(), 512)
        elif kind == "gemini":
            run_gemini()
        else:
            raise ValueError(f"Unsupported provider '{provider}'.")
        logger.info(
            "annolid-bot turn stop session=%s provider=%s model=%s status=fallback_ok",
            session_id,
            provider,
            model,
        )
    except Exception as fallback_exc:
        if is_provider_config_error(fallback_exc):
            message = format_provider_config_error(str(fallback_exc), provider=provider)
            logger.warning(
                "annolid-bot fallback provider config error session=%s provider=%s model=%s error=%s",
                session_id,
                provider,
                model,
                fallback_exc,
            )
            emit_final(message, True)
            logger.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=config_error",
                session_id,
                provider,
                model,
            )
            return
        if isinstance(fallback_exc, ImportError):
            message = format_dependency_error(str(fallback_exc))
            logger.warning(
                "annolid-bot fallback dependency missing session=%s provider=%s model=%s error=%s",
                session_id,
                provider,
                model,
                fallback_exc,
            )
            emit_final(message, True)
            logger.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=dependency_missing",
                session_id,
                provider,
                model,
            )
            return
        if is_provider_timeout_error(fallback_exc):
            message = (
                f"Provider request timed out for '{provider}:{model}'. "
                "Try again, use a smaller prompt, reduce tool usage, or increase "
                "timeouts in Settings → Agent Runtime."
            )
            logger.warning(
                "annolid-bot fallback timed out session=%s provider=%s model=%s error=%s",
                session_id,
                provider,
                model,
                fallback_exc,
            )
            emit_final(message, True)
            logger.info(
                "annolid-bot turn stop session=%s provider=%s model=%s status=timeout",
                session_id,
                provider,
                model,
            )
            return
        logger.exception(
            "annolid-bot fallback failed session=%s provider=%s model=%s",
            session_id,
            provider,
            model,
        )
        emit_final(
            f"Error in chat interaction: {original_error}; fallback failed: {fallback_exc}",
            True,
        )
