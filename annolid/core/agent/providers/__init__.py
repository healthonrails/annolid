"""Agent provider abstraction layer."""

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .litellm_provider import LiteLLMProvider
from .openai_compat import (
    OpenAICompatProvider,
    OpenAICompatResolved,
    resolve_openai_compat,
)
from .transcription import OpenAICompatTranscriptionProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCallRequest",
    "LiteLLMProvider",
    "OpenAICompatProvider",
    "OpenAICompatResolved",
    "resolve_openai_compat",
    "OpenAICompatTranscriptionProvider",
]
