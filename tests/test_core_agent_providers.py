from __future__ import annotations

from types import SimpleNamespace

from annolid.core.agent.providers.openai_compat import (
    OpenAICompatProvider,
    resolve_openai_compat,
)
from annolid.core.agent.providers.litellm_provider import LiteLLMProvider
from annolid.utils.llm_settings import LLMConfig


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


def test_resolve_openai_compat_for_openrouter_key_prefix() -> None:
    cfg = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        params={"api_key": "sk-or-test", "base_url": ""},
    )
    resolved = resolve_openai_compat(cfg)
    assert resolved.provider == "openrouter"
    assert "openrouter.ai" in resolved.base_url


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


def test_litellm_provider_resolves_gateway_prefix() -> None:
    provider = LiteLLMProvider(
        provider_name="openrouter",
        api_key="sk-or-test",
        default_model="gpt-4o-mini",
    )
    assert provider._resolve_model("gpt-4o-mini") == "openrouter/gpt-4o-mini"


def test_litellm_provider_applies_model_overrides() -> None:
    provider = LiteLLMProvider(
        provider_name="moonshot",
        api_key="x",
        default_model="kimi-k2.5",
    )
    payload = {"temperature": 0.2}
    provider._apply_model_overrides("moonshot/kimi-k2.5", payload)
    assert payload["temperature"] == 1.0
