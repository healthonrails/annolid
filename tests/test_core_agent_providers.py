from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from pathlib import Path
import subprocess

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
from annolid.core.agent.providers.base import ToolCallRequest
from annolid.core.agent.providers.litellm_provider import LiteLLMProvider
from annolid.core.agent.providers.registry import find_by_model
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
    assert resp.content == ""
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


def test_litellm_provider_sanitizes_assistant_tool_call_messages() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1"}],
            "untrusted": "drop",
        }
    ]
    sanitized = LiteLLMProvider._sanitize_messages(messages)
    assert len(sanitized) == 1
    assert "untrusted" not in sanitized[0]
    assert sanitized[0]["content"] is None


def test_litellm_provider_parses_repairable_tool_call_arguments(monkeypatch) -> None:
    fake_repair = types.SimpleNamespace(loads=lambda _text: {"text": "hi"})
    monkeypatch.setitem(sys.modules, "json_repair", fake_repair)
    parsed = LiteLLMProvider._parse_tool_call_arguments('{"text":"hi",}')
    assert parsed["text"] == "hi"


def test_litellm_provider_parse_response_defaults_missing_tool_call_id() -> None:
    provider = LiteLLMProvider(
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


def test_litellm_provider_configures_runtime_logging(monkeypatch) -> None:
    class _FakeLiteLLM:
        suppress_debug_info = False
        drop_params = False
        set_verbose = True

    monkeypatch.delenv("LITELLM_LOG", raising=False)
    LiteLLMProvider._litellm_logging_configured = False
    LiteLLMProvider._configure_litellm_runtime_logging(_FakeLiteLLM)
    assert _FakeLiteLLM.suppress_debug_info is True
    assert _FakeLiteLLM.drop_params is True
    assert _FakeLiteLLM.set_verbose is False
    assert os.environ.get("LITELLM_LOG") == "ERROR"


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
    ):  # noqa: ANN001
        calls["cli_path"] = cli_path
        calls["prompt"] = prompt
        calls["model"] = model
        calls["workdir"] = workdir
        calls["timeout_seconds"] = timeout_seconds
        calls["images"] = list(images)
        calls["session_id"] = session_id
        calls["runtime"] = runtime
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
    assert calls["model"] == "gpt-5.1-codex"
    assert calls["timeout_seconds"] == 42.0
    assert calls["workdir"] == "/tmp/annolid-codex"
    assert calls["session_id"] == "gui:test-codex"
    assert calls["runtime"] == ""
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
    )

    assert first == "cli reply"
    assert second == "cli reply"
    assert "resume" not in calls[0]
    assert calls[1][:4] == ["codex", "exec", "resume", "thread_123"]
