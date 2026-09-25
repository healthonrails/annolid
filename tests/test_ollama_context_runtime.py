from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from annolid.core.agent.providers.background_chat import (
    _OLLAMA_TOOL_SUPPORT_CACHE,
    build_ollama_llm_callable,
)
from annolid.core.agent.providers.base import ProviderCallError
from annolid.core.agent.providers.ollama_runtime import stream_ollama_chat


LOG = logging.getLogger(__name__)
OVERFLOW = (
    '{"error":{"code":400,"message":"request (8032 tokens) exceeds the '
    'available context size (4096 tokens), try increasing it",'
    '"type":"exceed_context_size_error","n_prompt_tokens":8032,"n_ctx":4096}}'
    " (status code: 400)"
)


def test_default_context_accepts_reported_prompt_and_sdk_response():
    ollama = pytest.importorskip("ollama")
    calls = []
    messages = [
        {"role": "system", "content": "s" * 16893},
        {"role": "user", "content": "hi"},
    ]

    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs.get("options", {}).get("num_ctx", 4096) < 8032:
            raise RuntimeError(OVERFLOW)
        return iter(
            [
                ollama.ChatResponse(
                    message=ollama.Message(role="assistant", content="ok")
                )
            ]
        )

    rows = list(
        stream_ollama_chat(
            SimpleNamespace(chat=chat),
            settings={},
            model="m",
            messages=messages,
            logger=LOG,
        )
    )
    assert rows[0]["message"]["content"] == "ok"
    assert len(calls) == 1
    assert calls[0]["messages"] == messages
    assert calls[0]["options"] == {"num_ctx": 16384}


@pytest.mark.parametrize("stream_error", [False, True])
def test_context_retry_preserves_messages_and_tools(stream_error):
    calls = []
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question"},
    ]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if stream_error:
                return iter([{"error": OVERFLOW}])
            raise RuntimeError(OVERFLOW)
        return iter([{"message": {"content": "ok"}}])

    rows = list(
        stream_ollama_chat(
            SimpleNamespace(chat=chat),
            settings={"ollama": {"num_ctx": 4096}},
            model="m",
            messages=messages,
            tools=tools,
            logger=LOG,
        )
    )
    assert rows[-1]["message"]["content"] == "ok"
    assert [c["options"]["num_ctx"] for c in calls] == [4096, 9216]
    assert all(c["messages"] == messages and c["tools"] == tools for c in calls)


@pytest.mark.parametrize("ceiling,expected_calls", [(4096, 1), (32768, 2)])
def test_context_retry_is_bounded_and_disables_plain_fallback(ceiling, expected_calls):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(OVERFLOW)

    with pytest.raises(ProviderCallError, match="Shorten the conversation") as raised:
        list(
            stream_ollama_chat(
                SimpleNamespace(chat=chat),
                settings={"ollama": {"num_ctx": 4096, "max_num_ctx": ceiling}},
                model="m",
                messages=[],
                logger=LOG,
            )
        )
    assert len(calls) == expected_calls
    assert raised.value.retryable is False
    assert raised.value.error_kind == "context_length"


def test_context_error_after_partial_output_does_not_replay_and_closes_stream():
    calls = []
    closed = []

    def chat(**kwargs):
        calls.append(kwargs)
        try:
            yield {"message": {"content": "partial"}}
            raise RuntimeError(OVERFLOW)
        finally:
            closed.append(True)

    stream = stream_ollama_chat(
        SimpleNamespace(chat=chat), settings={}, model="m", messages=[], logger=LOG
    )
    assert next(stream)["message"]["content"] == "partial"
    with pytest.raises(ProviderCallError):
        next(stream)
    assert len(calls) == 1
    assert closed == [True]


@pytest.mark.parametrize(
    "config",
    [
        {"num_ctx": 0},
        {"num_ctx": True},
        {"num_ctx": "oops"},
        {"num_ctx": 8192, "max_num_ctx": 4096},
    ],
)
def test_invalid_context_settings_fail_before_request(config):
    def chat(**kwargs):
        pytest.fail("Invalid configuration must not reach the provider")

    with pytest.raises(ValueError, match="ollama"):
        list(
            stream_ollama_chat(
                SimpleNamespace(chat=chat),
                settings={"ollama": config},
                model="m",
                messages=[],
                logger=LOG,
            )
        )


def _build(chat):
    return build_ollama_llm_callable(
        prompt="hi",
        settings={"ollama": {"num_ctx": 4096}},
        parse_tool_calls=lambda calls: [
            {
                "id": "call_1",
                "name": c["function"]["name"],
                "arguments": c["function"]["arguments"],
            }
            for c in calls
        ],
        normalize_messages=lambda messages: messages,
        extract_text=lambda response: response["message"]["content"],
        prompt_may_need_tools=lambda prompt: True,
        logger=LOG,
        import_module=lambda name: SimpleNamespace(chat=chat),
    )


@pytest.mark.parametrize("error", [OVERFLOW, "invalid request (status code: 400)"])
def test_other_400_errors_do_not_disable_tools(error):
    calls = []
    model = "test-error-classification"
    _OLLAMA_TOOL_SUPPORT_CACHE.pop(model, None)

    def chat(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(error)

    with pytest.raises(RuntimeError):
        asyncio.run(
            _build(chat)(
                [{"role": "user", "content": "hi"}],
                [{"type": "function", "function": {"name": "read_file"}}],
                model,
            )
        )
    assert all(c["tools"] for c in calls)
    assert model not in _OLLAMA_TOOL_SUPPORT_CACHE


def test_explicit_unsupported_tools_retries_plain():
    calls = []
    model = "test-no-tools"
    _OLLAMA_TOOL_SUPPORT_CACHE.pop(model, None)

    def chat(**kwargs):
        calls.append(kwargs)
        if kwargs["tools"]:
            raise RuntimeError("model does not support tools (status code: 400)")
        return iter([{"message": {"content": "ok"}}])

    try:
        result = asyncio.run(
            _build(chat)([{"role": "user", "content": "hi"}], [{"fake": "tool"}], model)
        )
        assert result["content"] == "ok"
        assert len(calls) == 2
        assert calls[1]["tools"] is None
        assert _OLLAMA_TOOL_SUPPORT_CACHE[model] is False
    finally:
        _OLLAMA_TOOL_SUPPORT_CACHE.pop(model, None)


def test_sdk_tool_response_and_token_callbacks_run_on_event_loop():
    ollama = pytest.importorskip("ollama")
    seen = []

    def chat(**kwargs):
        return iter(
            [
                ollama.ChatResponse(
                    message=ollama.Message(
                        role="assistant",
                        content="checking",
                        tool_calls=[
                            ollama.Message.ToolCall(
                                function=ollama.Message.ToolCall.Function(
                                    name="read_file", arguments={"path": "README.md"}
                                )
                            )
                        ],
                    )
                )
            ]
        )

    async def run():
        loop = asyncio.get_running_loop()

        def on_token(token):
            seen.append((token, asyncio.get_running_loop() is loop))

        return await _build(chat)(
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "read_file"}}],
            "test-sdk",
            on_token,
        )

    result = asyncio.run(run(), debug=True)
    assert result["tool_calls"][0]["name"] == "read_file"
    assert seen == [("checking", True)]


@pytest.mark.parametrize("path", ["plain", "recovery"])
def test_plain_and_recovery_share_explicit_context(monkeypatch, path):
    from annolid.core.agent.providers import background_chat

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return iter([{"message": {"content": "ok"}}])

    module = SimpleNamespace(chat=chat)
    settings = {"ollama": {"num_ctx": 8192, "max_num_ctx": 8192}}
    if path == "plain":
        monkeypatch.setattr(
            background_chat.importlib, "import_module", lambda name: module
        )
        final = []
        background_chat.run_ollama_streaming_chat(
            prompt="hi",
            image_path="",
            model="m",
            settings=settings,
            load_history_messages=lambda: [],
            emit_chunk=lambda chunk: None,
            emit_final=lambda text, error: final.append((text, error)),
            persist_turn=lambda user, reply: None,
        )
        assert final == [("", False)]
    else:
        assert (
            background_chat.recover_with_plain_ollama_reply(
                prompt="hi",
                image_path="",
                model="m",
                settings=settings,
                logger=LOG,
                import_module=lambda name: module,
            )
            == "ok"
        )
    assert calls[0]["options"] == {"num_ctx": 8192}


def test_exhausted_context_retry_does_not_trigger_gui_plain_fallback():
    from annolid.core.agent.gui_backend.provider_fallback import run_provider_fallback

    def chat(**kwargs):
        raise RuntimeError(OVERFLOW)

    with pytest.raises(ProviderCallError) as caught:
        list(
            stream_ollama_chat(
                SimpleNamespace(chat=chat),
                settings={"ollama": {"num_ctx": 4096, "max_num_ctx": 4096}},
                model="m",
                messages=[],
                logger=LOG,
            )
        )
    calls = []
    final = []
    run_provider_fallback(
        original_error=caught.value,
        settings={},
        provider="ollama",
        model="m",
        session_id="test",
        fallback_timeout_retry_seconds=lambda: 1,
        fallback_retry_timeout_seconds=lambda: 1,
        run_ollama=lambda: calls.append("ollama"),
        run_openai=lambda *args: calls.append("openai"),
        run_gemini=lambda: calls.append("gemini"),
        emit_progress=lambda text: None,
        emit_final=lambda text, error: final.append((text, error)),
        format_dependency_error=str,
        logger=LOG,
    )
    assert calls == []
    assert final[0][1] is True
    assert "Shorten the conversation" in final[0][0]
