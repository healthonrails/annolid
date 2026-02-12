from __future__ import annotations

import asyncio

from annolid.gui.widgets.ai_chat_backend import StreamingChatTask


def test_parse_ollama_tool_calls_handles_mixed_argument_shapes() -> None:
    raw = [
        {
            "id": "call_1",
            "function": {"name": "read_file", "arguments": '{"path":"/tmp/a.txt"}'},
        },
        {
            "function": {"name": "exec", "arguments": {"command": "pwd"}},
        },
        {
            "function": {"name": "", "arguments": {}},
        },
        {
            "function": {"name": "bad", "arguments": 123},
        },
    ]

    parsed = StreamingChatTask._parse_ollama_tool_calls(raw)
    assert len(parsed) == 3
    assert parsed[0]["name"] == "read_file"
    assert parsed[0]["arguments"]["path"] == "/tmp/a.txt"
    assert parsed[1]["name"] == "exec"
    assert parsed[1]["arguments"]["command"] == "pwd"
    assert parsed[2]["name"] == "bad"
    assert parsed[2]["arguments"]["_raw"] == 123


def test_collect_ollama_stream_accumulates_content_and_tool_calls() -> None:
    stream = [
        {"message": {"content": "he"}},
        {"message": {"content": "llo"}},
        {
            "done_reason": "stop",
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "exec", "arguments": {"command": "pwd"}},
                    }
                ],
            },
        },
    ]
    content, tool_calls, done_reason = StreamingChatTask._collect_ollama_stream(stream)
    assert content == "hello"
    assert done_reason == "stop"
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "exec"


def test_collect_ollama_stream_merges_tool_calls_across_chunks() -> None:
    stream = [
        {
            "message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "a.txt"},
                        },
                    }
                ]
            }
        },
        {
            "message": {
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "exec", "arguments": {"command": "pwd"}},
                    }
                ]
            }
        },
    ]
    content, tool_calls, done_reason = StreamingChatTask._collect_ollama_stream(stream)
    assert content == ""
    assert done_reason == "stop"
    assert len(tool_calls) == 2
    names = {call["name"] for call in tool_calls}
    assert names == {"read_file", "exec"}


def test_ollama_llm_callable_preserves_stream_tool_calls(monkeypatch) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    backend._OLLAMA_TOOL_SUPPORT_CACHE.clear()

    class DummyOllama:
        def chat(self, *, model, messages, tools=None, stream=True):
            assert stream is True
            return iter(
                [
                    {
                        "done_reason": "stop",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "exec",
                                        "arguments": {"command": "pwd"},
                                    },
                                }
                            ],
                        },
                    }
                ]
            )

    monkeypatch.setattr(backend.importlib, "import_module", lambda name: DummyOllama())

    task = StreamingChatTask(
        "hi", widget=None, settings={"ollama": {}}, provider="ollama"
    )
    llm = task._build_ollama_llm_callable()
    resp = asyncio.run(
        llm([{"role": "user", "content": "hi"}], [{"fake": "tool"}], "m")
    )
    assert resp["content"] == ""
    assert len(resp["tool_calls"]) == 1
    assert resp["tool_calls"][0]["name"] == "exec"


def test_ollama_llm_callable_fast_retries_without_tools_on_empty(monkeypatch) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    backend._OLLAMA_TOOL_SUPPORT_CACHE.clear()
    calls = {"with_tools": 0, "without_tools": 0}

    class DummyOllama:
        def chat(self, *, model, messages, tools=None, stream=True):
            assert stream is True
            if tools is None:
                calls["without_tools"] += 1
                return iter([{"done_reason": "stop", "message": {"content": "ok"}}])
            calls["with_tools"] += 1
            return iter([{"done_reason": "stop", "message": {"content": ""}}])

    monkeypatch.setattr(backend.importlib, "import_module", lambda name: DummyOllama())

    task = StreamingChatTask(
        "hi", widget=None, settings={"ollama": {}}, provider="ollama"
    )
    llm = task._build_ollama_llm_callable()
    resp = asyncio.run(
        llm([{"role": "user", "content": "hi"}], [{"fake": "tool"}], "m")
    )
    assert resp["content"] == "ok"
    assert calls["with_tools"] == 1
    assert calls["without_tools"] == 1
    assert backend._OLLAMA_TOOL_SUPPORT_CACHE.get("m") is False
