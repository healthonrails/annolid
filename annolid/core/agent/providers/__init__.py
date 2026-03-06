"""Agent provider abstraction layer."""

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .background_chat import (
    build_ollama_llm_callable,
    run_codex_cli_chat,
    dependency_error_for_kind,
    ollama_clear_plain_mode,
    ollama_mark_plain_mode,
    ollama_plain_mode_decrement,
    ollama_plain_mode_remaining,
    recover_with_plain_ollama_reply,
    run_gemini_chat,
    run_openai_codex_chat,
    run_ollama_streaming_chat,
    run_openai_compat_chat,
)
from .litellm_provider import LiteLLMProvider
from .codex_cli_provider import CodexCLIProvider, CodexCLIResolved, resolve_codex_cli
from .openai_codex_provider import OpenAICodexProvider, resolve_openai_codex
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
    "dependency_error_for_kind",
    "build_ollama_llm_callable",
    "run_codex_cli_chat",
    "recover_with_plain_ollama_reply",
    "ollama_plain_mode_remaining",
    "ollama_plain_mode_decrement",
    "ollama_mark_plain_mode",
    "ollama_clear_plain_mode",
    "run_ollama_streaming_chat",
    "run_openai_compat_chat",
    "run_openai_codex_chat",
    "run_gemini_chat",
    "LiteLLMProvider",
    "CodexCLIProvider",
    "CodexCLIResolved",
    "resolve_codex_cli",
    "OpenAICodexProvider",
    "resolve_openai_codex",
    "OpenAICompatProvider",
    "OpenAICompatResolved",
    "resolve_openai_compat",
    "OpenAICompatTranscriptionProvider",
]
