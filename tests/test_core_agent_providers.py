from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from pathlib import Path
import subprocess

import pytest

import annolid.core.agent.gui_backend.provider_fallback as provider_fallback_mod
import annolid.core.agent.providers.background_chat as background_chat_mod
import annolid.core.agent.providers.openai_codex_provider as openai_codex_mod
from annolid.core.agent.providers.openai_compat import (
    OpenAICompatProvider,
    resolve_openai_compat,
)
from annolid.core.agent.providers.codex_cli_provider import (
    CodexCLIProvider,
    resolve_codex_cli,
)
import annolid.core.agent.providers.codex_cli_provider as codex_cli_mod
from annolid.core.agent.providers.openai_codex_provider import (
    OpenAICodexProvider,
    resolve_openai_codex,
)
from annolid.core.agent.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallError,
    ToolCallRequest,
    error_response_from_exception,
)
from annolid.core.agent.providers.call_runtime import (
    sanitize_openai_messages,
    sanitize_tool_schemas,
)
from annolid.core.agent.providers.unified_provider import UnifiedLLMProvider
from annolid.core.agent.providers.registry import find_by_model
from annolid.utils.llm_settings import LLMConfig


def test_provider_payload_sanitization_drops_internal_metadata_without_mutation() -> (
    None
):
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                    "_meta": {"path": "/private/secret.png"},
                }
            ],
            "tools_used": ["read_file"],
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object"},
                "_meta": {"source": "/private/tool.py"},
            },
        }
    ]

    sanitized_messages = sanitize_openai_messages(messages)
    sanitized_tools = sanitize_tool_schemas(tools)

    assert "tools_used" not in sanitized_messages[0]
    assert "_meta" not in str(sanitized_messages)
    assert "_meta" not in str(sanitized_tools)
    assert messages[0]["tools_used"] == ["read_file"]
    assert "_meta" in messages[0]["content"][0]
    assert "_meta" in tools[0]["function"]


def test_provider_exception_is_structured_and_redacted() -> None:
    class _RateLimitError(RuntimeError):
        status_code = 429
        headers = {
            "retry-after-ms": "250",
            "x-should-retry": "true",
        }
        body = {
            "error": {
                "type": "rate_limit_error",
                "code": "too_many_requests",
            }
        }

    response = error_response_from_exception(
        _RateLimitError(
            "request failed API key provided: sk-super-secret-value "
            "Authorization: Bearer bearer-secret "
            "at https://user:password@example.com/v1"
        )
    )

    assert response.finish_reason == "error"
    assert response.error_status_code == 429
    assert response.error_kind == "rate_limit"
    assert response.error_type == "rate_limit_error"
    assert response.error_code == "too_many_requests"
    assert response.error_retry_after_s == 0.25
    assert response.error_should_retry is True
    assert "sk-super-secret-value" not in str(response.content)
    assert "bearer-secret" not in str(response.content)
    assert "password@example.com" not in str(response.content)
    assert str(response.content).count("<redacted>") == 3


def test_provider_retry_is_bounded_and_recovers_transient_error() -> None:
    class _SequenceProvider(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs):  # noqa: ANN003
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="temporary outage",
                    finish_reason="error",
                    error_status_code=503,
                    error_kind="server_error",
                )
            return LLMResponse(content="ok")

        def get_default_model(self) -> str:
            return "test-model"

    provider = _SequenceProvider()
    response = __import__("asyncio").run(
        provider.chat_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            retry_delays=(0.0,),
        )
    )

    assert response.content == "ok"
    assert provider.calls == 2


def test_provider_retry_stops_after_streamed_content() -> None:
    class _StreamingFailureProvider(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs):  # noqa: ANN003
            self.calls += 1
            kwargs["on_token"]("partial")
            return LLMResponse(
                content="stream disconnected",
                finish_reason="error",
                error_kind="connection",
            )

        def get_default_model(self) -> str:
            return "test-model"

    provider = _StreamingFailureProvider()
    chunks: list[str] = []
    response = __import__("asyncio").run(
        provider.chat_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            on_token=chunks.append,
            retry_delays=(0.0,),
        )
    )

    assert response.finish_reason == "error"
    assert provider.calls == 1
    assert provider.is_retryable_response(response) is False
    assert chunks == ["partial"]


def test_provider_retry_does_not_block_on_long_retry_after() -> None:
    class _RateLimitedProvider(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs):  # noqa: ANN003
            del kwargs
            self.calls += 1
            return LLMResponse(
                content="rate limited",
                finish_reason="error",
                error_status_code=429,
                error_kind="rate_limit",
                error_retry_after_s=120.0,
            )

        def get_default_model(self) -> str:
            return "test-model"

    provider = _RateLimitedProvider()
    response = __import__("asyncio").run(
        provider.chat_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            retry_delays=(0.0,),
            max_retry_after_s=30.0,
        )
    )

    assert response.finish_reason == "error"
    assert provider.calls == 1
    assert provider.is_retryable_response(response) is False


def test_resolve_openai_compat_for_ollama() -> None:
    cfg = LLMConfig(
        provider="ollama",
        model="qwen3-vl",
        params={"host": "http://127.0.0.1:11434"},
    )
    resolved = resolve_openai_compat(cfg)
    assert resolved.provider == "ollama"
    assert resolved.api_key == "ollama"
    assert resolved.base_url.endswith("/v1")


def test_openai_codex_preserves_multimodal_tool_output() -> None:
    _, items = openai_codex_mod._convert_messages(  # noqa: SLF001
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,abc",
                            "detail": "high",
                        },
                        "_meta": {"path": "/private/image.png"},
                    },
                    {"type": "text", "text": "Rendered page"},
                    {
                        "type": "input_file",
                        "file_id": "file_123",
                        "filename": "report.pdf",
                        "_meta": {"path": "/private/report.pdf"},
                    },
                ],
            }
        ]
    )

    assert items[0]["output"] == [
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc",
            "detail": "high",
        },
        {"type": "input_text", "text": "Rendered page"},
        {
            "type": "input_file",
            "file_id": "file_123",
            "filename": "report.pdf",
        },
    ]
    assert "_meta" not in str(items[0])


def test_openai_codex_preserves_unknown_tool_lists_as_json() -> None:
    content = [
        {"type": "text", "text": "status", "code": 7},
        {"kind": "record", "value": 42},
    ]

    _, items = openai_codex_mod._convert_messages(  # noqa: SLF001
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": content,
            }
        ]
    )

    assert items[0]["output"] == json.dumps(
        content,
        ensure_ascii=False,
    )


def test_resolve_openai_compat_for_openrouter_key_prefix() -> None:
    cfg = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        params={"api_key": "sk-or-test", "base_url": ""},
    )
    resolved = resolve_openai_compat(cfg)
    assert resolved.provider == "openrouter"
    assert "openrouter.ai" in resolved.base_url


def test_provider_resolution_does_not_export_credentials_to_process_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="nvidia",
            model="nvidia/nemotron-3-ultra-550b-a55b",
            params={
                "api_key": "nvapi-configured-secret",
                "base_url": "https://integrate.api.nvidia.com/v1",
            },
        )
    )
    _ = UnifiedLLMProvider(
        provider_name="openrouter",
        api_key="sk-or-configured-secret",
        api_base="https://openrouter.ai/api/v1",
        default_model="openrouter/test-model",
    )

    assert resolved.api_key == "nvapi-configured-secret"
    assert __import__("os").environ["OPENAI_API_KEY"] == "existing-key"
    assert "NVIDIA_API_KEY" not in __import__("os").environ


def test_openai_compat_provider_parses_tool_calls() -> None:
    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            del kwargs
            tc = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="echo", arguments='{"text":"hi"}'),
            )
            msg = SimpleNamespace(content="ok", tool_calls=[tc], reasoning_content="r")
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            usage = SimpleNamespace(
                prompt_tokens=1, completion_tokens=2, total_tokens=3
            )
            return SimpleNamespace(choices=[choice], usage=usage)

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            params={"api_key": "sk-test", "base_url": "https://api.openai.com/v1"},
        )
    )
    provider = OpenAICompatProvider(
        resolved=resolved,
        client_factory=lambda _resolved: _FakeClient(),
    )
    resp = __import__("asyncio").run(
        provider.chat(messages=[{"role": "user", "content": "x"}])
    )
    assert resp.content == "ok"
    assert resp.has_tool_calls is True
    assert resp.tool_calls[0].name == "echo"
    assert resp.tool_calls[0].arguments["text"] == "hi"
    assert resp.usage["total_tokens"] == 3


def test_openai_compat_omits_image_for_known_text_only_model() -> None:
    calls: list[dict] = []

    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            calls.append(dict(kwargs))
            if "data:image" in str(kwargs.get("messages")):
                raise ValueError(
                    "Received multimodal data but multimodal processing is not "
                    "enabled. Use --enable-multimodal flag to enable multimodal "
                    "processing."
                )
            msg = SimpleNamespace(
                content="Hello!",
                tool_calls=[],
                reasoning_content=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=None,
            )

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="nvidia",
            model="nvidia/nemotron-3-ultra-550b-a55b",
            params={
                "api_key": "nvapi-test",
                "base_url": "https://integrate.api.nvidia.com/v1",
            },
        )
    )
    provider = OpenAICompatProvider(
        resolved=resolved,
        client_factory=lambda _resolved: _FakeClient(),
    )
    original_content = [
        {"type": "text", "text": "hello"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abc"},
        },
    ]

    resp = __import__("asyncio").run(
        provider.chat(
            messages=[{"role": "user", "content": original_content}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
    )

    assert resp.content == "Hello!"
    assert len(calls) == 1
    request_content = calls[0]["messages"][0]["content"]
    assert isinstance(request_content, str)
    assert "hello" in request_content
    assert "accepts text-only input" in request_content
    assert "data:image" not in request_content
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["tools"]
    assert original_content[1]["image_url"]["url"] == "data:image/png;base64,abc"


def test_openai_compat_retries_explicit_text_only_model_error_without_image() -> None:
    calls: list[dict] = []

    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise ValueError("vendor/text-model is not a multimodal model")
            msg = SimpleNamespace(
                content="Text fallback",
                tool_calls=[],
                reasoning_content=None,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                usage=None,
            )

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="openai",
            model="vendor/text-model",
            params={
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
            },
        )
    )
    provider = OpenAICompatProvider(
        resolved=resolved,
        client_factory=lambda _resolved: _FakeClient(),
    )

    resp = __import__("asyncio").run(
        provider.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                }
            ]
        )
    )

    assert resp.content == "Text fallback"
    assert len(calls) == 2
    assert "data:image" in str(calls[0]["messages"])
    assert "data:image" not in str(calls[1]["messages"])
    assert "accepts text-only input" in str(calls[1]["messages"])


def test_openai_compat_does_not_hide_multimodal_server_misconfiguration() -> None:
    calls: list[dict] = []

    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            calls.append(dict(kwargs))
            raise ValueError(
                "Received multimodal data but multimodal processing is not enabled. "
                "Use --enable-multimodal flag to enable multimodal processing."
            )

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="nvidia",
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            params={
                "api_key": "nvapi-test",
                "base_url": "https://integrate.api.nvidia.com/v1",
            },
        )
    )
    provider = OpenAICompatProvider(
        resolved=resolved,
        client_factory=lambda _resolved: _FakeClient(),
    )

    resp = __import__("asyncio").run(
        provider.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                }
            ]
        )
    )

    assert resp.finish_reason == "error"
    assert "--enable-multimodal" in str(resp.content)
    assert len(calls) == 1


def test_openai_compat_provider_handles_empty_choices() -> None:
    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            del kwargs
            return SimpleNamespace(choices=None, usage=None)

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

        async def aclose(self) -> None:
            return None

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            params={"api_key": "sk-test", "base_url": "https://api.openai.com/v1"},
        )
    )
    provider = OpenAICompatProvider(
        resolved=resolved,
        client_factory=lambda _resolved: _FakeClient(),
    )
    resp = __import__("asyncio").run(
        provider.chat(messages=[{"role": "user", "content": "x"}])
    )
    assert resp.content == "Model provider returned no response choices."
    assert resp.finish_reason == "error"
    assert resp.error_kind == "empty"
    assert resp.error_should_retry is True
    assert resp.has_tool_calls is False


def test_openai_compat_provider_parses_dict_response_and_reuses_client_until_closed() -> (
    None
):
    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            del kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text":"hi"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

    closed = {"value": False}

    class _FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

        async def aclose(self) -> None:
            closed["value"] = True

    resolved = resolve_openai_compat(
        LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            params={"api_key": "sk-test", "base_url": "https://api.openai.com/v1"},
        )
    )
    provider = OpenAICompatProvider(
        resolved=resolved,
        client_factory=lambda _resolved: _FakeClient(),
    )
    resp = __import__("asyncio").run(
        provider.chat(messages=[{"role": "user", "content": "x"}])
    )
    assert resp.content == "ok"
    assert resp.has_tool_calls is True
    assert resp.tool_calls[0].name == "echo"
    assert closed["value"] is False
    __import__("asyncio").run(provider.close())
    assert closed["value"] is True


def test_unified_provider_resolves_gateway_prefix() -> None:
    provider = UnifiedLLMProvider(
        provider_name="openrouter",
        api_key="sk-or-test",
        default_model="gpt-4o-mini",
    )
    assert provider._resolve_model("gpt-4o-mini") == "openrouter/gpt-4o-mini"


def test_unified_provider_applies_model_overrides() -> None:
    provider = UnifiedLLMProvider(
        provider_name="moonshot",
        api_key="x",
        default_model="kimi-k2.5",
    )
    payload = {"temperature": 0.2}
    provider._apply_model_overrides("moonshot/kimi-k2.5", payload)
    assert payload["temperature"] == 1.0


def test_unified_provider_sanitizes_assistant_tool_call_messages() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1"}],
            "untrusted": "drop",
        }
    ]
    sanitized = UnifiedLLMProvider._sanitize_messages(messages)
    assert len(sanitized) == 1
    assert "untrusted" not in sanitized[0]
    assert sanitized[0]["content"] is None


def test_unified_provider_parses_repairable_tool_call_arguments(monkeypatch) -> None:
    fake_repair = types.SimpleNamespace(loads=lambda _text: {"text": "hi"})
    monkeypatch.setitem(sys.modules, "json_repair", fake_repair)
    parsed = UnifiedLLMProvider._parse_tool_call_arguments('{"text":"hi",}')
    assert parsed["text"] == "hi"


def test_unified_provider_parse_response_defaults_missing_tool_call_id() -> None:
    provider = UnifiedLLMProvider(
        provider_name="openrouter",
        api_key="sk-or-test",
        default_model="gpt-4o-mini",
    )
    tc = SimpleNamespace(
        id="",
        function=SimpleNamespace(name="echo", arguments='{"text":"hi"}'),
    )
    message = SimpleNamespace(content="ok", tool_calls=[tc], reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    completion = SimpleNamespace(choices=[choice], usage=None)
    resp = provider._parse_response(completion)
    assert resp.has_tool_calls is True
    assert resp.tool_calls[0].id == "call_0"


def test_unified_provider_configures_runtime_logging() -> None:
    class _FakeRuntime:
        suppress_debug_info = False
        drop_params = False
        set_verbose = True

    UnifiedLLMProvider._runtime_logging_configured = False
    UnifiedLLMProvider._configure_runtime_logging(_FakeRuntime)
    assert _FakeRuntime.suppress_debug_info is True
    assert _FakeRuntime.drop_params is True
    assert _FakeRuntime.set_verbose is False


def test_run_openai_compat_chat_closes_provider_on_timeout(monkeypatch) -> None:
    closed = {"value": False}

    class _FakeProvider:
        async def chat(self, **kwargs):  # noqa: ANN003
            del kwargs
            await __import__("asyncio").sleep(0.2)
            return SimpleNamespace(content="")

        async def close(self) -> None:
            await __import__("asyncio").sleep(0.01)
            closed["value"] = True

    monkeypatch.setattr(
        background_chat_mod, "resolve_openai_compat", lambda _cfg: object()
    )
    monkeypatch.setattr(
        background_chat_mod, "OpenAICompatProvider", lambda resolved: _FakeProvider()
    )

    try:
        background_chat_mod.run_openai_compat_chat(
            prompt="hello",
            image_path="",
            model="fake-model",
            provider_name="openai",
            settings={"openai": {"api_key": "x", "base_url": "https://example.com/v1"}},
            load_history_messages=lambda: [],
            timeout_s=0.01,
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("Expected timeout")

    assert closed["value"] is True


def test_run_openai_compat_chat_raises_provider_error_instead_of_returning_text(
    monkeypatch,
) -> None:
    closed = {"value": False}

    class _FakeProvider:
        async def chat_with_retry(self, **kwargs):  # noqa: ANN003
            del kwargs
            return LLMResponse(
                content="service unavailable",
                finish_reason="error",
                error_status_code=503,
                error_kind="server_error",
            )

        async def close(self) -> None:
            closed["value"] = True

    monkeypatch.setattr(
        background_chat_mod, "resolve_openai_compat", lambda _cfg: object()
    )
    monkeypatch.setattr(
        background_chat_mod, "OpenAICompatProvider", lambda resolved: _FakeProvider()
    )

    with pytest.raises(ProviderCallError) as exc_info:
        background_chat_mod.run_openai_compat_chat(
            prompt="hello",
            image_path="",
            model="fake-model",
            provider_name="openai",
            settings={
                "openai": {
                    "api_key": "x",
                    "base_url": "https://example.com/v1",
                }
            },
            load_history_messages=lambda: [],
            timeout_s=1.0,
        )

    assert exc_info.value.retryable is True
    assert exc_info.value.status_code == 503
    assert closed["value"] is True


def test_inline_model_image_has_bounded_read_limit(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "large.png"
    image_path.write_bytes(b"12345")
    monkeypatch.setattr(background_chat_mod, "MAX_INLINE_MODEL_IMAGE_BYTES", 4)

    with pytest.raises(ValueError, match="model-call limit"):
        background_chat_mod._read_image_data_url(str(image_path))


def test_gui_provider_fallback_skips_non_retryable_model_error() -> None:
    provider_calls: list[str] = []
    final: list[tuple[str, bool]] = []
    error = ProviderCallError(
        "invalid multimodal request",
        provider="openai",
        model="test-model",
        error_kind="invalid_request",
        retryable=False,
    )

    provider_fallback_mod.run_provider_fallback(
        original_error=error,
        settings={"openai": {"kind": "openai_compat"}},
        provider="openai",
        model="test-model",
        session_id="test-session",
        fallback_timeout_retry_seconds=lambda: 1.0,
        fallback_retry_timeout_seconds=lambda: 1.0,
        run_ollama=lambda: provider_calls.append("ollama"),
        run_openai=lambda *_args: provider_calls.append("openai"),
        run_gemini=lambda: provider_calls.append("gemini"),
        emit_progress=lambda _text: None,
        emit_final=lambda message, is_error: final.append((message, is_error)),
        format_dependency_error=lambda message: message,
        logger=SimpleNamespace(warning=lambda *_args: None),
    )

    assert provider_calls == []
    assert final == [("invalid multimodal request", True)]


def test_gui_provider_fallback_redacts_raw_fallback_errors() -> None:
    final: list[tuple[str, bool]] = []

    def _fail_provider(*_args) -> None:
        raise RuntimeError("Authorization: Bearer secret-provider-token")

    provider_fallback_mod.run_provider_fallback(
        original_error=RuntimeError("API key: sk-original-secret-value"),
        settings={"openai": {"kind": "openai_compat"}},
        provider="openai",
        model="test-model",
        session_id="test-session",
        fallback_timeout_retry_seconds=lambda: 1.0,
        fallback_retry_timeout_seconds=lambda: 1.0,
        run_ollama=lambda: None,
        run_openai=_fail_provider,
        run_gemini=lambda: None,
        emit_progress=lambda _text: None,
        emit_final=lambda message, is_error: final.append((message, is_error)),
        format_dependency_error=lambda message: message,
        logger=SimpleNamespace(
            warning=lambda *_args: None,
            info=lambda *_args: None,
            exception=lambda *_args: None,
        ),
    )

    assert len(final) == 1
    assert final[0][1] is True
    assert "sk-original-secret-value" not in final[0][0]
    assert "secret-provider-token" not in final[0][0]
    assert final[0][0].count("<redacted>") == 2


def test_provider_registry_matches_openai_codex_explicit_prefix() -> None:
    spec = find_by_model("openai-codex/gpt-5.1-codex")
    assert spec is not None
    assert spec.name == "openai_codex"


def test_provider_registry_matches_codex_cli_explicit_prefix() -> None:
    spec = find_by_model("codex-cli/gpt-5.1-codex")
    assert spec is not None
    assert spec.name == "codex_cli"


def test_openai_codex_provider_chat_parses_sse_tool_calls() -> None:
    class _Token:
        account_id = "acct_123"
        access = "tok_123"

    async def _fake_request(
        url,
        headers,
        body,
        *,
        transport,
        timeout_seconds,
        on_token=None,  # noqa: ANN001
    ):
        assert url == "https://chatgpt.com/backend-api/codex/responses"
        assert headers["Authorization"] == "Bearer tok_123"
        assert body["model"] == "gpt-5.1-codex"
        assert transport == "sse"
        assert body["tools"][0]["name"] == "echo"
        assert timeout_seconds == 33.0
        if on_token is not None:
            on_token("hello")
        return (
            "hello",
            [ToolCallRequest(id="call_1|fc_1", name="echo", arguments={"text": "hi"})],
            "stop",
            "brief reasoning",
        )

    async def _failing_websocket_request(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("ws unavailable")

    resolved = resolve_openai_codex(
        LLMConfig(
            provider="openai_codex",
            model="openai-codex/gpt-5.1-codex",
            params={"base_url": "https://chatgpt.com/backend-api/codex/responses"},
        )
    )
    provider = OpenAICodexProvider(
        resolved=resolved,
        token_getter=lambda: _Token(),
        request_callable=_fake_request,
        websocket_request_callable=_failing_websocket_request,
    )
    streamed: list[str] = []
    resp = __import__("asyncio").run(
        provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "description": "Echo.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            timeout_seconds=33.0,
            on_token=streamed.append,
        )
    )
    assert resp.content == "hello"
    assert resp.reasoning_content == "brief reasoning"
    assert resp.has_tool_calls is True
    assert resp.tool_calls[0].name == "echo"
    assert resp.tool_calls[0].arguments["text"] == "hi"
    assert streamed == ["hello"]


def test_resolve_openai_codex_accepts_transport_override() -> None:
    resolved = resolve_openai_codex(
        LLMConfig(
            provider="openai_codex",
            model="openai-codex/gpt-5.4",
            params={
                "base_url": "https://chatgpt.com/backend-api/codex/responses",
                "transport": "sse",
            },
        )
    )
    assert resolved.model == "openai-codex/gpt-5.4"
    assert resolved.transport == "sse"
    assert resolved.websocket_url == "wss://chatgpt.com/backend-api/codex/responses"


def test_openai_codex_provider_uses_websocket_transport_when_requested() -> None:
    class _Token:
        account_id = "acct_123"
        access = "tok_123"

    async def _fake_websocket_request(
        url,
        headers,
        body,
        *,
        timeout_seconds,
        on_token=None,  # noqa: ANN001
    ):
        assert url == "wss://chatgpt.com/backend-api/codex/responses"
        assert headers["Authorization"] == "Bearer tok_123"
        assert body["model"] == "gpt-5.4"
        assert timeout_seconds == 15.0
        if on_token is not None:
            on_token("ws")
        return ("ws", [], "stop", "")

    async def _unexpected_sse_request(*args, **kwargs):  # noqa: ANN001
        raise AssertionError(
            "SSE path should not be used for explicit websocket transport"
        )

    resolved = resolve_openai_codex(
        LLMConfig(
            provider="openai_codex",
            model="openai-codex/gpt-5.4",
            params={"transport": "websocket"},
        )
    )
    provider = OpenAICodexProvider(
        resolved=resolved,
        token_getter=lambda: _Token(),
        request_callable=_unexpected_sse_request,
        websocket_request_callable=_fake_websocket_request,
    )
    streamed: list[str] = []
    resp = __import__("asyncio").run(
        provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            timeout_seconds=15.0,
            on_token=streamed.append,
        )
    )
    assert resp.content == "ws"
    assert streamed == ["ws"]


def test_openai_codex_provider_auto_transport_falls_back_to_sse() -> None:
    class _Token:
        account_id = "acct_123"
        access = "tok_123"

    calls = {"ws": 0, "sse": 0}

    async def _failing_websocket_request(
        url,
        headers,
        body,
        *,
        timeout_seconds,
        on_token=None,  # noqa: ANN001
    ):
        del url, headers, body, timeout_seconds, on_token
        calls["ws"] += 1
        raise RuntimeError("websocket unavailable")

    async def _sse_request(
        url,
        headers,
        body,
        *,
        transport,
        timeout_seconds,
        on_token=None,  # noqa: ANN001
    ):
        assert url == "https://chatgpt.com/backend-api/codex/responses"
        assert transport == "sse"
        calls["sse"] += 1
        if on_token is not None:
            on_token("sse")
        return ("sse", [], "stop", "")

    resolved = resolve_openai_codex(
        LLMConfig(
            provider="openai_codex",
            model="openai-codex/gpt-5.4",
            params={"transport": "auto"},
        )
    )
    provider = OpenAICodexProvider(
        resolved=resolved,
        token_getter=lambda: _Token(),
        request_callable=_sse_request,
        websocket_request_callable=_failing_websocket_request,
    )
    resp = __import__("asyncio").run(
        provider.chat(
            messages=[{"role": "user", "content": "hello"}], timeout_seconds=9.0
        )
    )
    assert resp.content == "sse"
    assert calls == {"ws": 1, "sse": 1}


def test_codex_cli_provider_runs_text_only_cli(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/codex")

    calls = {}

    def _fake_runner(
        *,
        cli_path,
        prompt,
        model,
        workdir,
        timeout_seconds,
        images,
        session_id,
        runtime,
        sandbox,
    ):  # noqa: ANN001
        calls["cli_path"] = cli_path
        calls["prompt"] = prompt
        calls["model"] = model
        calls["workdir"] = workdir
        calls["timeout_seconds"] = timeout_seconds
        calls["images"] = list(images)
        calls["session_id"] = session_id
        calls["runtime"] = runtime
        calls["sandbox"] = sandbox
        return "final cli reply"

    resolved = resolve_codex_cli(
        LLMConfig(
            provider="codex_cli",
            model="codex-cli/gpt-5.1-codex",
            params={"workdir": "/tmp/annolid-codex", "session_id": "gui:test-codex"},
        )
    )
    provider = CodexCLIProvider(resolved=resolved, runner=_fake_runner)
    resp = __import__("asyncio").run(
        provider.chat(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Summarize the change."},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "code_search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            timeout_seconds=42.0,
        )
    )
    assert resp.content == "final cli reply"
    assert resp.has_tool_calls is False
    assert calls["cli_path"] == "/usr/local/bin/codex"
    assert calls["model"] == "gpt-5.1-codex"
    assert calls["timeout_seconds"] == 42.0
    assert calls["workdir"] == "/tmp/annolid-codex"
    assert calls["session_id"] == "gui:test-codex"
    assert calls["runtime"] == ""
    assert calls["sandbox"] == "read-only"
    assert "tools are unavailable" in calls["prompt"].lower()


def test_codex_cli_runner_persists_and_resumes_thread_id(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_file = tmp_path / "codex_cli_sessions.json"
    monkeypatch.setattr(codex_cli_mod, "_ANNOLID_DIR", tmp_path)
    monkeypatch.setattr(codex_cli_mod, "_CODEX_CLI_SESSION_FILE", session_file)

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        assert kwargs["cwd"] == str(workspace)
        if "resume" in cmd:
            assert kwargs["env"]["ANNOLID_AGENT_RUNTIME"] == "acp"
            assert kwargs["env"]["ANNOLID_SHELL"] == "acp"
            assert kwargs["env"]["OPENCLAW_SHELL"] == "acp"
            assert kwargs["env"]["ANNOLID_ACP_SESSION_ID"] == "gui:codex-session"
        message_path = Path(cmd[cmd.index("--output-last-message") + 1])
        message_path.write_text("cli reply", encoding="utf-8")
        if "resume" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout='{"type":"turn.started"}\n',
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"type":"thread.started","thread_id":"thread_123"}\n',
            stderr="",
        )

    monkeypatch.setattr(codex_cli_mod.subprocess, "run", _fake_run)

    first = codex_cli_mod._run_codex_cli(
        cli_path="codex",
        prompt="first prompt",
        model="gpt-5.1-codex",
        workdir=str(workspace),
        timeout_seconds=30.0,
        images=[],
        session_id="gui:codex-session",
        runtime="acp",
        sandbox="workspace-write",
    )
    second = codex_cli_mod._run_codex_cli(
        cli_path="codex",
        prompt="second prompt",
        model="gpt-5.1-codex",
        workdir=str(workspace),
        timeout_seconds=30.0,
        images=[],
        session_id="gui:codex-session",
        runtime="acp",
        sandbox="workspace-write",
    )

    assert first == "cli reply"
    assert second == "cli reply"
    assert "resume" not in calls[0]
    assert calls[1][:4] == ["codex", "exec", "resume", "thread_123"]
    assert calls[0][calls[0].index("--sandbox") + 1] == "workspace-write"
    assert calls[1][calls[1].index("--sandbox") + 1] == "workspace-write"
