from __future__ import annotations

from typing import Any, Dict, Optional

from annolid.core.agent.providers import dependency_error_for_kind
from annolid.utils.llm_settings import provider_kind


def provider_dependency_error(
    *, settings: Dict[str, Any], provider: str
) -> Optional[str]:
    kind = provider_kind(settings, provider)
    return dependency_error_for_kind(kind)


def format_dependency_error(
    *,
    raw_error: str,
    settings: Dict[str, Any],
    provider: str,
) -> str:
    message = str(raw_error or "").strip()
    lowered = message.lower()
    kind = provider_kind(settings, provider)
    if kind == "openai_compat" and (
        "openai package is required" in lowered or "no module named 'openai'" in lowered
    ):
        return (
            "OpenAI-compatible provider requires the `openai` package. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install openai`."
        )
    if kind == "openai_codex" and (
        "oauth_cli_kit" in lowered or "oauth-cli-kit" in lowered
    ):
        return (
            "OpenAI Codex provider requires `oauth_cli_kit`. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install oauth-cli-kit`."
        )
    if kind == "openai_codex" and "httpx" in lowered:
        return (
            "OpenAI Codex provider requires `httpx`. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install httpx`."
        )
    if kind == "codex_cli" and "codex executable" in lowered:
        return (
            "Codex CLI provider requires the `codex` executable. "
            "Install Codex and ensure `codex` is available on PATH."
        )
    if kind == "anthropic" and (
        "anthropic package is required" in lowered
        or "no module named 'anthropic'" in lowered
    ):
        return (
            "Anthropic provider requires the `anthropic` package. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install anthropic`."
        )
    if kind == "gemini" and (
        "google-generativeai" in lowered
        or "google.generativeai" in lowered
        or "no module named 'google'" in lowered
    ):
        return (
            "Gemini provider requires `google-generativeai`. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install google-generativeai`."
        )
    return message or "Required provider dependency is missing."
