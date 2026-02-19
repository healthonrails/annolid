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
    kind = provider_kind(settings, provider)
    if kind == "openai_compat" and "openai package is required" in message:
        return (
            "OpenAI-compatible provider requires the `openai` package. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install openai`."
        )
    if kind == "gemini" and "google-generativeai" in message:
        return (
            "Gemini provider requires `google-generativeai`. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install google-generativeai`."
        )
    return message or "Required provider dependency is missing."
