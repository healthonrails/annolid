from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def test_compact_system_prompt_includes_allowed_read_roots(tmp_path: Path) -> None:
    task = StreamingChatTask("hi", widget=None)
    prompt = task._build_compact_system_prompt(
        tmp_path,
        allowed_read_roots=["/Users/chenyang/Downloads/test_annolid_videos_batch"],
    )
    assert "Allowed Read Roots" in prompt
    assert "/Users/chenyang/Downloads/test_annolid_videos_batch" in prompt


def test_build_agent_context_disables_web_tools_by_default(monkeypatch) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = []

    class _Resolved:
        profile = "full"
        source = "test"
        allowed_tools = None

    def _resolve_policy(*, all_tool_names, tools_cfg, provider, model):
        del tools_cfg, provider, model
        payload = _Resolved()
        payload.allowed_tools = set(all_tool_names)
        return payload

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "resolve_allowed_tools", _resolve_policy)

    task = StreamingChatTask("hi", widget=None, enable_web_tools=False)
    context = task._build_agent_execution_context()
    tool_names = set(context.tools.tool_names)
    assert "web_search" not in tool_names
    assert "web_fetch" not in tool_names
    assert "cron" not in tool_names
    assert "spawn" not in tool_names
    assert "message" not in tool_names


def test_build_agent_context_enables_web_tools_when_requested(monkeypatch) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = []

    class _Resolved:
        profile = "full"
        source = "test"
        allowed_tools = None

    def _resolve_policy(*, all_tool_names, tools_cfg, provider, model):
        del tools_cfg, provider, model
        payload = _Resolved()
        payload.allowed_tools = set(all_tool_names)
        return payload

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "resolve_allowed_tools", _resolve_policy)

    task = StreamingChatTask("hi", widget=None, enable_web_tools=True)
    context = task._build_agent_execution_context()
    tool_names = set(context.tools.tool_names)
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
    assert "cron" not in tool_names
    assert "spawn" not in tool_names
    assert "message" not in tool_names


def test_gui_tool_callbacks_validate_and_queue(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(tmp_path)]

    # Isolate PDF discovery to this tmp workspace so real local PDFs do not leak into the test.
    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []

    def _invoke(slot_name: str, *args):
        del args
        calls.append(slot_name)
        return True

    task._invoke_widget_slot = _invoke  # type: ignore[method-assign]

    video_file = tmp_path / "mouse.mp4"
    video_file.write_bytes(b"fake")

    open_payload = task._tool_gui_open_video(str(video_file))
    assert open_payload["ok"] is True
    assert open_payload["queued"] is True
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    open_pdf_payload = task._tool_gui_open_pdf()
    assert open_pdf_payload["ok"] is True
    assert open_pdf_payload["queued"] is True
    assert open_pdf_payload["path"] == str(pdf_file)
    open_pdf_by_path_payload = task._tool_gui_open_pdf(str(pdf_file))
    assert open_pdf_by_path_payload["ok"] is True
    assert open_pdf_by_path_payload["path"] == str(pdf_file)

    frame_payload = task._tool_gui_set_frame(42)
    assert frame_payload["ok"] is True

    prompt_payload = task._tool_gui_set_chat_prompt("describe this frame")
    assert prompt_payload["ok"] is True

    send_payload = task._tool_gui_send_chat_prompt()
    assert send_payload["ok"] is True

    model_payload = task._tool_gui_set_chat_model("ollama", "qwen3:8b")
    assert model_payload["ok"] is True

    annotation_payload = task._tool_gui_select_annotation_model("Cutie")
    assert annotation_payload["ok"] is True

    track_payload = task._tool_gui_track_next_frames(120)
    assert track_payload["ok"] is True

    ai_prompt_payload = task._tool_gui_set_ai_text_prompt("mouse", use_countgd=True)
    assert ai_prompt_payload["ok"] is True
    assert ai_prompt_payload["use_countgd"] is True

    run_ai_seg_payload = task._tool_gui_run_ai_text_segmentation()
    assert run_ai_seg_payload["ok"] is True

    workflow_payload = task._tool_gui_segment_track_video(
        path=str(video_file),
        text_prompt="mouse",
        mode="track",
        use_countgd=True,
        model_name="Cutie",
        to_frame=120,
    )
    assert workflow_payload["ok"] is True
    assert workflow_payload["mode"] == "track"
    assert workflow_payload["text_prompt"] == "mouse"

    assert calls == [
        "bot_open_video",
        "bot_open_pdf",
        "bot_open_pdf",
        "bot_set_frame",
        "bot_set_chat_prompt",
        "bot_send_chat_prompt",
        "bot_set_chat_model",
        "bot_select_annotation_model",
        "bot_track_next_frames",
        "bot_set_ai_text_prompt",
        "bot_run_ai_text_segmentation",
        "bot_segment_track_video",
    ]


def test_invoke_widget_slot_none_result_counts_as_success(monkeypatch) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    task = StreamingChatTask("hi", widget=object())

    def _invoke(*args, **kwargs):
        del args, kwargs
        return None

    monkeypatch.setattr(backend.QMetaObject, "invokeMethod", _invoke)
    ok = task._invoke_widget_slot("bot_segment_track_video")
    assert ok is True


def test_gui_open_video_rejects_missing_path(tmp_path: Path) -> None:
    task = StreamingChatTask("hi", widget=None)
    payload = task._tool_gui_open_video(str(tmp_path / "missing.mp4"))
    assert payload["ok"] is False
    assert "not found" in payload["error"].lower()


def test_gui_open_video_accepts_tool_style_text(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    video_file = tmp_path / "fish_demo.mp4"
    video_file.write_bytes(b"fake")

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]
    payload = task._tool_gui_open_video(
        'gui_open_video(path="/Users/chenyang/Downloads/test_annolid_videos_batch/fish_demo.mp4")'
    )
    assert payload["ok"] is True
    assert Path(payload["path"]).name == "fish_demo.mp4"
    assert calls == ["bot_open_video"]


def test_gui_open_video_resolves_by_basename_in_roots(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    root = tmp_path / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    video_file = root / "mouse.mp4"
    video_file.write_bytes(b"fake")

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(root)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    task._invoke_widget_slot = lambda *args, **kwargs: True  # type: ignore[method-assign]
    payload = task._tool_gui_open_video("please open mouse.mp4")
    assert payload["ok"] is True
    assert payload["path"] == str(video_file)


def test_gui_open_video_resolves_by_basename_recursively_in_allowed_root(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    root = tmp_path / "downloads"
    nested = root / "session_01" / "videos"
    nested.mkdir(parents=True, exist_ok=True)
    video_file = nested / "mouse.mp4"
    video_file.write_bytes(b"fake")

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(root)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    task._invoke_widget_slot = lambda *args, **kwargs: True  # type: ignore[method-assign]
    payload = task._tool_gui_open_video("please open mouse.mp4")
    assert payload["ok"] is True
    assert payload["path"] == str(video_file)


def test_gui_open_pdf_downloads_url_then_queues(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    downloaded_pdf = tmp_path / "downloads" / "paper.pdf"
    downloaded_pdf.parent.mkdir(parents=True, exist_ok=True)
    downloaded_pdf.write_bytes(b"%PDF-1.4 fake")

    task._download_pdf_for_gui_tool = lambda _url: downloaded_pdf  # type: ignore[method-assign]
    payload = task._tool_gui_open_pdf(
        "open pdf https://www.biorxiv.org/content/10.64898/2026.01.20.700446v2.full.pdf"
    )
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["path"] == str(downloaded_pdf)
    assert calls == ["bot_open_pdf"]


def test_extract_pdf_path_candidates_includes_url() -> None:
    task = StreamingChatTask("hi", widget=None)
    candidates = task._extract_pdf_path_candidates(
        "download and open https://example.org/a/paper.full.pdf?download=1"
    )
    assert any(candidate.startswith("https://example.org/") for candidate in candidates)


def test_resolve_video_path_uses_active_video_basename(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    external_dir = tmp_path / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    active_video = external_dir / "mouse.mp4"
    active_video.write_bytes(b"fake")

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(workspace)]

    class _Host:
        video_file = str(active_video)

    class _Widget:
        host_window_widget = _Host()

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: workspace)

    task = StreamingChatTask("hi", widget=_Widget())
    resolved = task._resolve_video_path_for_gui_tool("track mouse in video mouse.mp4")
    assert resolved == active_video


def test_resolve_video_path_uses_active_video_even_if_missing_on_disk(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    # Simulate GUI session where active video path is known but currently not resolvable.
    active_video = tmp_path / "external" / "mouse.mp4"

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(workspace)]

    class _Host:
        video_file = str(active_video)

    class _Widget:
        host_window_widget = _Host()

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: workspace)

    task = StreamingChatTask("hi", widget=_Widget())
    resolved = task._resolve_video_path_for_gui_tool("track mouse in video mouse.mp4")
    assert resolved == active_video


def test_prompt_may_need_tools_heuristic() -> None:
    assert StreamingChatTask._prompt_may_need_tools("use tool to open video") is True
    assert StreamingChatTask._prompt_may_need_tools("please list_dir workspace") is True
    assert (
        StreamingChatTask._prompt_may_need_tools("segment mouse with text prompt")
        is True
    )
    assert StreamingChatTask._prompt_may_need_tools("hello there") is False


def test_parse_direct_segment_track_video_command() -> None:
    task = StreamingChatTask("track mouse in /tmp/mouse.mp4", widget=None)
    cmd = task._parse_direct_gui_command(
        "track mouse in /tmp/mouse.mp4 to frame 120 with countgd"
    )
    assert cmd["name"] == "segment_track_video"
    assert cmd["args"]["mode"] == "track"
    assert cmd["args"]["text_prompt"] == "mouse"
    assert cmd["args"]["use_countgd"] is True
    assert cmd["args"]["to_frame"] == 120


def test_ollama_llm_callable_reprobes_tools_when_prompt_needs_tools(
    monkeypatch,
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    backend._OLLAMA_TOOL_SUPPORT_CACHE.clear()
    backend._OLLAMA_TOOL_SUPPORT_CACHE["m"] = False
    seen_tools_payloads = []

    class DummyOllama:
        def chat(self, *, model, messages, tools=None, stream=True):
            del model, messages, stream
            seen_tools_payloads.append(tools)
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
                                        "name": "read_file",
                                        "arguments": {"path": "a.txt"},
                                    },
                                }
                            ],
                        },
                    }
                ]
            )

    monkeypatch.setattr(backend.importlib, "import_module", lambda name: DummyOllama())

    task = StreamingChatTask(
        "please use tool to read a file",
        widget=None,
        settings={"ollama": {}},
        provider="ollama",
    )
    llm = task._build_ollama_llm_callable()
    resp = asyncio.run(
        llm([{"role": "user", "content": "hi"}], [{"fake": "tool"}], "m")
    )
    assert len(resp["tool_calls"]) == 1
    assert seen_tools_payloads and seen_tools_payloads[0] is not None


def test_direct_gui_fallback_opens_video_from_prompt(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    video_file = tmp_path / "fish_demo.mp4"
    video_file.write_bytes(b"fake")

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    task._invoke_widget_slot = lambda *args, **kwargs: True  # type: ignore[method-assign]
    text = task._maybe_run_direct_gui_tool_from_prompt(
        "Please open video /Users/chenyang/Downloads/test_annolid_videos_batch/fish_demo.mp4"
    )
    assert "Opened video in Annolid:" in text
    assert text.endswith("fish_demo.mp4")


def test_local_access_refusal_heuristic() -> None:
    assert (
        StreamingChatTask._looks_like_local_access_refusal(
            "As an AI, I cannot directly access your local file system."
        )
        is True
    )
    assert (
        StreamingChatTask._looks_like_local_access_refusal("Opened the video.") is False
    )


def test_web_access_refusal_heuristic() -> None:
    assert (
        StreamingChatTask._looks_like_web_access_refusal(
            "I don't have web browsing capabilities."
        )
        is True
    )
    assert (
        StreamingChatTask._looks_like_web_access_refusal(
            "I cannot directly fetch URLs. To summarize that page, share the content."
        )
        is True
    )
    assert (
        StreamingChatTask._looks_like_web_access_refusal("Fetched via web_fetch.")
        is False
    )


def test_finalize_agent_text_uses_web_fetch_fallback_on_refusal() -> None:
    class _DummyRegistry:
        def has(self, name: str) -> bool:
            return name == "web_fetch"

        async def execute(self, name: str, params: dict) -> str:
            assert name == "web_fetch"
            assert params["url"].startswith("https://")
            return json.dumps(
                {
                    "finalUrl": params["url"],
                    "text": (
                        "BrainGlobe AtlasAPI provides atlas loading, metadata, "
                        "resolution details, and structures metadata. "
                        "The atlas details page explains fields such as species, "
                        "orientation, shape, and resolution."
                    ),
                }
            )

    class _Result:
        content = "I don't have web browsing capabilities."
        tool_runs = ()

    task = StreamingChatTask(
        "summarize https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html",
        widget=None,
        enable_web_tools=True,
    )
    text, used_recovery, used_direct_gui_fallback = task._finalize_agent_text(
        _Result(),
        tools=_DummyRegistry(),  # type: ignore[arg-type]
    )
    assert used_recovery is False
    assert used_direct_gui_fallback is False
    assert (
        "Summary of https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        in text
    )
    assert (
        "Source: https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        in text
    )


def test_web_fetch_fallback_uses_recent_history_url_when_prompt_has_no_url() -> None:
    class _DummyRegistry:
        def has(self, name: str) -> bool:
            return name == "web_fetch"

        async def execute(self, name: str, params: dict) -> str:
            assert name == "web_fetch"
            assert (
                params["url"]
                == "https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
            )
            return json.dumps(
                {
                    "finalUrl": params["url"],
                    "text": "Atlas details documentation describes metadata and anatomy structures.",
                }
            )

    class _Store:
        def get_history(self, session_id: str):
            del session_id
            return [
                {
                    "role": "user",
                    "content": "summarize https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html",
                },
                {
                    "role": "assistant",
                    "content": "I cannot directly fetch URLs.",
                },
            ]

        def append_history(self, session_id: str, messages, *, max_messages: int):
            del session_id, messages, max_messages

    class _Result:
        content = "I cannot directly fetch URLs. To summarize that page, provide text."
        tool_runs = ()

    task = StreamingChatTask(
        "summarize that page",
        widget=None,
        enable_web_tools=True,
        session_store=_Store(),  # type: ignore[arg-type]
    )
    text, _, _ = task._finalize_agent_text(_Result(), tools=_DummyRegistry())  # type: ignore[arg-type]
    assert (
        "Summary of https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        in text
    )


def test_parse_direct_gui_command_variants() -> None:
    task = StreamingChatTask("hi", widget=None)
    parsed_pdf = task._parse_direct_gui_command("open pdf")
    assert parsed_pdf["name"] == "open_pdf"

    parsed_video = task._parse_direct_gui_command(
        "Please open video /tmp/fish_demo.mp4"
    )
    assert parsed_video["name"] == "open_video"

    parsed_frame = task._parse_direct_gui_command("go to frame 128")
    assert parsed_frame["name"] == "set_frame"
    assert parsed_frame["args"]["frame_index"] == 128

    parsed_track = task._parse_direct_gui_command("track to frame 400")
    assert parsed_track["name"] == "track_next_frames"
    assert parsed_track["args"]["to_frame"] == 400

    parsed_model = task._parse_direct_gui_command(
        "set chat model openrouter/nvidia/nemotron-nano-12b-v2-vl:free"
    )
    assert parsed_model["name"] == "set_chat_model"
    assert parsed_model["args"]["provider"] == "openrouter"

    parsed_behavior = task._parse_direct_gui_command(
        "segment mouse.mp4 with labels walking, rearing"
    )
    assert parsed_behavior["name"] == "label_behavior_segments"
    assert parsed_behavior["args"]["path"] == "mouse.mp4"
    assert parsed_behavior["args"]["behavior_labels"] == ["walking", "rearing"]

    parsed_non_open = task._parse_direct_gui_command(
        "segment mouse.mp4 with labels walking"
    )
    assert parsed_non_open["name"] != "open_video"

    parsed_stream = task._parse_direct_gui_command(
        "open stream with model mediapipe face and classify eye blinks"
    )
    assert parsed_stream["name"] == "start_realtime_stream"
    assert parsed_stream["args"]["model_name"] == "mediapipe_face"
    assert parsed_stream["args"]["viewer_type"] == "threejs"
    assert parsed_stream["args"]["classify_eye_blinks"] is True

    parsed_stop_stream = task._parse_direct_gui_command("stop realtime stream")
    assert parsed_stop_stream["name"] == "stop_realtime_stream"


def test_execute_direct_gui_command_routes_actions(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    video_file = tmp_path / "mouse.mp4"
    video_file.write_bytes(b"fake")
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    class _Cfg:
        class tools:  # noqa: N801
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    task._invoke_widget_slot = lambda *args, **kwargs: True  # type: ignore[method-assign]

    out_pdf = task._execute_direct_gui_command("open pdf")
    assert "Opened PDF in Annolid:" in out_pdf

    out_video = task._execute_direct_gui_command("open video mouse.mp4")
    assert "Opened video in Annolid:" in out_video

    out_frame = task._execute_direct_gui_command("set frame 5")
    assert "Moved to frame 5." == out_frame

    out_track = task._execute_direct_gui_command("track to frame 60")
    assert "Started tracking to frame 60." == out_track

    out_model = task._execute_direct_gui_command("set chat model ollama/qwen3:8b")
    assert "Updated chat model to ollama/qwen3:8b." == out_model

    out_stream = task._execute_direct_gui_command(
        "open stream with model mediapipe face and classify eye blinks"
    )
    assert "Started realtime stream with model mediapipe_face." == out_stream

    out_stop_stream = task._execute_direct_gui_command("stop realtime stream")
    assert "Stopped realtime stream." == out_stop_stream
