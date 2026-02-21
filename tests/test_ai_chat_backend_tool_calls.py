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
            email = None
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
    context = asyncio.run(task._build_agent_execution_context())
    tool_names = set(context.tools.tool_names)
    assert "web_search" not in tool_names
    assert "web_fetch" not in tool_names
    assert "cron" in tool_names
    assert "spawn" not in tool_names
    assert "message" not in tool_names


def test_build_agent_context_enables_web_tools_when_requested(monkeypatch) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
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
    context = asyncio.run(task._build_agent_execution_context())
    tool_names = set(context.tools.tool_names)
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
    assert "cron" in tool_names
    assert "spawn" not in tool_names
    assert "message" not in tool_names


def test_gui_tool_callbacks_validate_and_queue(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
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
    open_url_payload = asyncio.run(task._tool_gui_open_url("google.com"))
    assert open_url_payload["ok"] is True
    assert open_url_payload["queued"] is True
    assert open_url_payload["url"] == "https://google.com"
    open_in_browser_payload = task._tool_gui_open_in_browser("google.com")
    assert open_in_browser_payload["ok"] is True
    assert open_in_browser_payload["queued"] is True
    assert open_in_browser_payload["url"] == "https://google.com"
    web_text_payload = task._tool_gui_web_get_dom_text(1200)
    assert web_text_payload["ok"] is True
    web_click_payload = task._tool_gui_web_click("button.submit")
    assert web_click_payload["ok"] is True
    web_type_payload = task._tool_gui_web_type(
        "input[name='q']",
        "annolid",
        submit=True,
    )
    assert web_type_payload["ok"] is True
    web_scroll_payload = task._tool_gui_web_scroll(600)
    assert web_scroll_payload["ok"] is True
    web_forms_payload = task._tool_gui_web_find_forms()
    assert web_forms_payload["ok"] is True
    pdf_state_payload = task._tool_gui_pdf_get_state()
    assert pdf_state_payload["ok"] is True
    pdf_text_payload = task._tool_gui_pdf_get_text(max_chars=900, pages=2)
    assert pdf_text_payload["ok"] is True
    pdf_sections_payload = task._tool_gui_pdf_find_sections(
        max_sections=12, max_pages=6
    )
    assert pdf_sections_payload["ok"] is True
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")
    open_pdf_payload = asyncio.run(task._tool_gui_open_pdf())
    assert open_pdf_payload["ok"] is True
    assert open_pdf_payload["queued"] is True
    assert open_pdf_payload["path"] == str(pdf_file)
    open_pdf_by_path_payload = asyncio.run(task._tool_gui_open_pdf(str(pdf_file)))
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
        "bot_open_url",
        "bot_open_in_browser",
        "bot_web_get_dom_text",
        "bot_web_click",
        "bot_web_type",
        "bot_web_scroll",
        "bot_web_find_forms",
        "bot_pdf_get_state",
        "bot_pdf_get_text",
        "bot_pdf_find_sections",
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


def test_gui_label_behavior_segments_with_widget_result(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    video_file = tmp_path / "mouse.mp4"
    video_file.write_bytes(b"fake")

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    class _Widget:
        host_window_widget = None

        def get_bot_action_result(self, action_name: str):
            assert action_name == "label_behavior_segments"
            return {
                "ok": True,
                "mode": "timeline",
                "labeled_segments": 7,
                "evaluated_segments": 10,
                "skipped_segments": 3,
                "labels_used": ["groom", "eat"],
                "timestamps_csv": str(tmp_path / "segments.csv"),
                "timestamps_rows": 7,
            }

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=_Widget())
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    payload = task._tool_gui_label_behavior_segments(
        path=str(video_file),
        behavior_labels=["groom", "eat"],
        segment_mode="timeline",
        segment_frames=60,
        max_segments=100,
    )
    assert payload["ok"] is True
    assert payload["mode"] == "timeline"
    assert payload["labeled_segments"] == 7
    assert payload["timestamps_rows"] == 7
    assert calls == ["bot_label_behavior_segments"]


def test_gui_label_behavior_segments_invalid_mode() -> None:
    task = StreamingChatTask("hi", widget=None)
    payload = task._tool_gui_label_behavior_segments(segment_mode="bad")
    assert payload["ok"] is False
    assert "segment_mode" in payload["error"]


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
            email = None
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
            email = None
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
            email = None
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
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    downloaded_pdf = tmp_path / "downloads" / "paper.pdf"
    downloaded_pdf.parent.mkdir(parents=True, exist_ok=True)
    downloaded_pdf.write_bytes(b"%PDF-1.4 fake")

    async def mock_download(_url):
        return downloaded_pdf

    task._download_pdf_for_gui_tool = mock_download  # type: ignore[method-assign]
    payload = asyncio.run(
        task._tool_gui_open_pdf(
            "open pdf https://www.biorxiv.org/content/10.64898/2026.01.20.700446v2.full.pdf"
        )
    )
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["path"] == str(downloaded_pdf)
    assert calls == ["bot_open_pdf"]


def test_gui_open_pdf_downloads_non_suffix_url_then_queues(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    downloaded_pdf = tmp_path / "downloads" / "paper.pdf"
    downloaded_pdf.parent.mkdir(parents=True, exist_ok=True)
    downloaded_pdf.write_bytes(b"%PDF-1.4 fake")

    async def mock_download(_url):
        return downloaded_pdf

    task._download_pdf_for_gui_tool = mock_download  # type: ignore[method-assign]
    payload = asyncio.run(
        task._tool_gui_open_pdf("open url https://example.org/download?id=12345")
    )
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["path"] == str(downloaded_pdf)
    assert calls == ["bot_open_pdf"]


def test_gui_open_url_queues(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    payload = asyncio.run(
        task._tool_gui_open_url(
            "open this page https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        )
    )
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert (
        payload["url"]
        == "https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
    )
    assert calls == ["bot_open_url"]


def test_gui_open_url_queues_for_domain_without_scheme(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    payload = asyncio.run(task._tool_gui_open_url("google.com"))
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["url"] == "https://google.com"
    assert calls == ["bot_open_url"]


def test_gui_open_url_queues_for_local_html_file(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    html_file = tmp_path / "ai_studio_code (1).html"
    html_file.write_text("<html><body>ok</body></html>", encoding="utf-8")

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    payload = asyncio.run(task._tool_gui_open_url(f"open {html_file}"))
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["url"] == str(html_file)
    assert calls == ["bot_open_url"]


def test_gui_web_run_steps_executes_sequence(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    payload = asyncio.run(
        task._tool_gui_web_run_steps(
            [
                {"action": "open_url", "url": "google.com"},
                {"action": "click", "selector": "button.submit"},
                {"action": "scroll", "delta_y": 400},
            ],
            stop_on_error=True,
            max_steps=10,
        )
    )
    assert payload["ok"] is True
    assert payload["steps_run"] == 3
    assert calls == ["bot_open_url", "bot_web_click", "bot_web_scroll"]


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
            email = None
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
            email = None
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
    assert (
        StreamingChatTask._prompt_may_need_tools("what is the weather today?") is True
    )
    assert StreamingChatTask._prompt_may_need_tools("latest ai news") is True
    assert StreamingChatTask._prompt_may_need_tools("hello there") is False


def test_prompt_may_need_mcp_heuristic() -> None:
    assert (
        StreamingChatTask._prompt_may_need_mcp(
            "open https://example.com and click the login button"
        )
        is True
    )
    assert StreamingChatTask._prompt_may_need_mcp("use playwright to inspect page") is (
        True
    )
    assert StreamingChatTask._prompt_may_need_mcp("what is the weather today?") is True
    assert StreamingChatTask._prompt_may_need_mcp("summarize local annotations") is (
        False
    )


def test_browser_first_for_web_setting_default_and_override() -> None:
    task_default = StreamingChatTask("weather", widget=None, settings={})
    assert task_default._browser_first_for_web() is True

    task_disabled = StreamingChatTask(
        "weather",
        widget=None,
        settings={"agent": {"browser_first_for_web": False}},
    )
    assert task_disabled._browser_first_for_web() is False


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
            email = None
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
        StreamingChatTask._looks_like_web_access_refusal(
            "I apologize, but I don't have web search or web browsing capabilities available with my current tools."
        )
        is True
    )
    assert (
        StreamingChatTask._looks_like_web_access_refusal("Fetched via web_fetch.")
        is False
    )


def test_knowledge_gap_heuristic() -> None:
    assert (
        StreamingChatTask._looks_like_knowledge_gap_response(
            "I don't have access to weather data or your location information."
        )
        is True
    )
    assert (
        StreamingChatTask._looks_like_knowledge_gap_response(
            "I can access weather data."
        )
        is False
    )
    assert (
        StreamingChatTask._looks_like_knowledge_gap_response(
            "I can't check the weather right now (web search API isn't configured)."
        )
        is True
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
    assert isinstance(used_direct_gui_fallback, bool)
    assert (
        "Summary of https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        in text
    )
    assert (
        "Source: https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        in text
    )


def test_finalize_agent_text_uses_browser_search_fallback_on_knowledge_gap() -> None:
    class _DummyRegistry:
        def has(self, name: str) -> bool:
            return name == "gui_web_run_steps"

        async def execute(self, name: str, params: dict) -> str:
            assert name == "gui_web_run_steps"
            steps = params.get("steps", [])
            assert isinstance(steps, list) and steps
            assert steps[0]["action"] == "open_url"
            assert (
                steps[0]["url"]
                == "https://html.duckduckgo.com/html/?q=latest+malaria+guidance"
            )
            return json.dumps(
                {
                    "ok": True,
                    "results": [
                        {"action": "open_url", "result": {"ok": True}},
                        {"action": "wait", "result": {"ok": True}},
                        {
                            "action": "get_text",
                            "result": {
                                "ok": True,
                                "text": (
                                    "The latest WHO update reports key facts "
                                    "about malaria prevention and treatment."
                                ),
                            },
                        },
                    ],
                }
            )

    class _Result:
        content = (
            "I don't have access to that information directly. "
            "You can check by searching on the web."
        )
        tool_runs = ()

    task = StreamingChatTask(
        "latest malaria guidance",
        widget=None,
        enable_web_tools=True,
    )
    text, used_recovery, used_direct_gui_fallback = task._finalize_agent_text(
        _Result(),
        tools=_DummyRegistry(),  # type: ignore[arg-type]
    )
    assert used_recovery is False
    assert used_direct_gui_fallback is False
    assert "Web lookup via embedded browser" in text
    assert "malaria prevention and treatment" in text


def test_finalize_agent_text_uses_browser_search_fallback_without_web_tools() -> None:
    class _DummyRegistry:
        def has(self, name: str) -> bool:
            return name == "gui_web_run_steps"

        async def execute(self, name: str, params: dict) -> str:
            assert name == "gui_web_run_steps"
            steps = params.get("steps", [])
            assert isinstance(steps, list) and steps
            assert steps[0]["action"] == "open_url"
            assert (
                steps[0]["url"] == "https://html.duckduckgo.com/html/?q=latest+AI+news"
            )
            return json.dumps(
                {
                    "ok": True,
                    "results": [
                        {
                            "action": "get_text",
                            "result": {
                                "ok": True,
                                "text": "Top AI headlines today: model releases, policy updates, and hardware news.",
                            },
                        }
                    ],
                }
            )

    class _Result:
        content = (
            "I apologize, but I don't have web search or web browsing capabilities "
            "available with my current tools."
        )
        tool_runs = ()

    task = StreamingChatTask(
        "latest AI news",
        widget=None,
        enable_web_tools=False,
    )
    text, used_recovery, used_direct_gui_fallback = task._finalize_agent_text(
        _Result(),
        tools=_DummyRegistry(),  # type: ignore[arg-type]
    )
    assert used_recovery is False
    assert used_direct_gui_fallback is False
    assert "Web lookup via embedded browser" in text
    assert "Top AI headlines today" in text


def test_finalize_agent_text_uses_browser_search_fallback_when_web_api_unconfigured() -> (
    None
):
    class _DummyRegistry:
        def has(self, name: str) -> bool:
            return name == "gui_web_run_steps"

        async def execute(self, name: str, params: dict) -> str:
            assert name == "gui_web_run_steps"
            steps = params.get("steps")
            assert isinstance(steps, list)
            assert steps[0]["action"] == "open_url"
            assert (
                steps[0]["url"]
                == "https://html.duckduckgo.com/html/?q=check+weather+in+Ithaca+NY"
            )
            return json.dumps(
                {
                    "ok": True,
                    "results": [
                        {
                            "action": "get_text",
                            "result": {
                                "ok": True,
                                "text": "Ithaca weather today: 39 F, light rain, wind 6 mph.",
                            },
                        }
                    ],
                }
            )

    class _Result:
        content = (
            "I can't check the weather right now (web search API isn't configured). "
            "Search 'Ithaca NY weather' in your browser."
        )
        tool_runs = ()

    task = StreamingChatTask(
        "check weather in Ithaca NY",
        widget=None,
        enable_web_tools=True,
    )
    text, used_recovery, used_direct_gui_fallback = task._finalize_agent_text(
        _Result(),
        tools=_DummyRegistry(),  # type: ignore[arg-type]
    )
    assert used_recovery is False
    assert used_direct_gui_fallback is False
    assert "Web lookup via embedded browser" in text
    assert "Ithaca weather today" in text


def test_finalize_agent_text_prefers_open_page_content_before_browser_search() -> None:
    class _Result:
        content = (
            "I can't check the weather right now (web search API isn't configured). "
            "Search 'Ithaca NY weather' in your browser."
        )
        tool_runs = ()

    task = StreamingChatTask(
        "what is today's weather in Ithaca NY",
        widget=None,
        enable_web_tools=True,
    )
    task._tool_gui_web_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_page": True,
        "url": "https://weather.com/weather/today/l/Ithaca+NY",
        "title": "Ithaca Weather",
    }
    task._tool_gui_web_get_dom_text = lambda max_chars=8000: {  # type: ignore[method-assign]
        "ok": True,
        "url": "https://weather.com/weather/today/l/Ithaca+NY",
        "title": "Ithaca Weather",
        "text": "Current conditions in Ithaca NY: 39 F, light rain, wind 6 mph.",
    }

    text, used_recovery, used_direct_gui_fallback = task._finalize_agent_text(
        _Result(),
        tools=None,
    )
    assert used_recovery is False
    assert used_direct_gui_fallback is False
    assert "Using the currently open page" in text
    assert "Current conditions in Ithaca NY" in text


def test_build_live_web_context_prompt_block_includes_open_page_snapshot() -> None:
    task = StreamingChatTask("hi", widget=None, enable_web_tools=True)
    task._tool_gui_web_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_page": True,
        "url": "https://example.org",
        "title": "Example Domain",
    }
    task._tool_gui_web_get_dom_text = lambda max_chars=2500: {  # type: ignore[method-assign]
        "ok": True,
        "url": "https://example.org",
        "title": "Example Domain",
        "text": "Example Domain This domain is for use in illustrative examples.",
    }
    block = task._build_live_web_context_prompt_block()
    assert "Active Embedded Web Page" in block
    assert "https://example.org" in block
    assert "illustrative examples" in block


def test_finalize_agent_text_prefers_open_pdf_content_on_local_access_refusal() -> None:
    class _Result:
        content = (
            "I can't access your local file system. Please open the PDF and share it."
        )
        tool_runs = ()

    task = StreamingChatTask(
        "summarize this pdf",
        widget=None,
        enable_web_tools=True,
    )
    task._tool_gui_pdf_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_pdf": True,
        "path": "/tmp/paper.pdf",
        "title": "paper.pdf",
        "current_page": 3,
        "total_pages": 12,
    }
    task._tool_gui_pdf_get_text = lambda max_chars=8000, pages=2: {  # type: ignore[method-assign]
        "ok": True,
        "path": "/tmp/paper.pdf",
        "title": "paper.pdf",
        "current_page": 3,
        "total_pages": 12,
        "text": "This paper presents a robust segmentation method with strong benchmarks.",
    }

    text, used_recovery, used_direct_gui_fallback = task._finalize_agent_text(
        _Result(),
        tools=None,
    )
    assert used_recovery is False
    assert used_direct_gui_fallback is False
    assert "Using the currently open PDF" in text
    assert "robust segmentation method" in text


def test_build_live_pdf_context_prompt_block_includes_snapshot() -> None:
    task = StreamingChatTask("hi", widget=None, enable_web_tools=True)
    task._tool_gui_pdf_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_pdf": True,
        "path": "/tmp/paper.pdf",
        "title": "paper.pdf",
        "current_page": 3,
        "total_pages": 12,
    }
    task._tool_gui_pdf_get_text = lambda max_chars=2500, pages=2: {  # type: ignore[method-assign]
        "ok": True,
        "path": "/tmp/paper.pdf",
        "title": "paper.pdf",
        "current_page": 3,
        "total_pages": 12,
        "text": "This section describes the method and evaluation setup.",
    }
    block = task._build_live_pdf_context_prompt_block()
    assert "Active PDF" in block
    assert "/tmp/paper.pdf" in block
    assert "evaluation setup" in block


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

    parsed_open_url_direct = task._parse_direct_gui_command(
        "open https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
    )
    assert parsed_open_url_direct["name"] == "open_url"
    assert (
        parsed_open_url_direct["args"]["url"]
        == "https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
    )

    parsed_open_url = task._parse_direct_gui_command(
        "open url https://example.org/download?id=12345"
    )
    assert parsed_open_url["name"] == "open_url"

    parsed_open_url_pdf = task._parse_direct_gui_command(
        "open url https://example.org/paper.pdf?download=1"
    )
    assert parsed_open_url_pdf["name"] == "open_pdf"

    parsed_open_domain = task._parse_direct_gui_command("open google.com")
    assert parsed_open_domain["name"] == "open_url"
    assert parsed_open_domain["args"]["url"] == "https://google.com"

    parsed_open_local_html = task._parse_direct_gui_command(
        "open /Users/chenyang/Downloads/ai_studio_code (1).html"
    )
    assert parsed_open_local_html["name"] == "open_url"
    assert "ai_studio_code (1).html" in parsed_open_local_html["args"]["url"]

    parsed_open_domain_browser = task._parse_direct_gui_command(
        "open google.com in browser"
    )
    assert parsed_open_domain_browser["name"] == "open_in_browser"
    assert parsed_open_domain_browser["args"]["url"] == "https://google.com"

    parsed_domain_only = task._parse_direct_gui_command("google.com")
    assert parsed_domain_only["name"] == "open_url"
    assert parsed_domain_only["args"]["url"] == "https://google.com"

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

    parsed_label_in = task._parse_direct_gui_command(
        "label behavior in mouse.mp4 with labels rearing, walking"
    )
    assert parsed_label_in["name"] == "label_behavior_segments"
    assert parsed_label_in["args"]["path"] == "mouse.mp4"
    assert parsed_label_in["args"]["behavior_labels"] == ["rearing", "walking"]

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

    parsed_rename_pdf = task._parse_direct_gui_command(
        "rename this pdf with title A_3_Dimensional_Digital_Atlas_of_the_Starling_Brain.pdf"
    )
    assert parsed_rename_pdf["name"] == "rename_file"
    assert parsed_rename_pdf["args"]["use_active_file"] is True
    assert (
        parsed_rename_pdf["args"]["new_name"]
        == "A_3_Dimensional_Digital_Atlas_of_the_Starling_Brain.pdf"
    )

    parsed_clawhub_search = task._parse_direct_gui_command(
        "search clawhub skills for behavior labeling"
    )
    assert parsed_clawhub_search["name"] == "clawhub_search_skills"
    assert parsed_clawhub_search["args"]["query"] == "behavior labeling"

    parsed_clawhub_search_trailing = task._parse_direct_gui_command(
        "search skills on clawhub for Research Paper Writer"
    )
    assert parsed_clawhub_search_trailing["name"] == "clawhub_search_skills"
    assert parsed_clawhub_search_trailing["args"]["query"] == "research paper writer"

    parsed_clawhub_install = task._parse_direct_gui_command(
        "install skill behavior-labeler from clawhub"
    )
    assert parsed_clawhub_install["name"] == "clawhub_install_skill"
    assert parsed_clawhub_install["args"]["slug"] == "behavior-labeler"

    parsed_save_citation = task._parse_direct_gui_command(
        "save citation from pdf as annolid2024 to refs.bib"
    )
    assert parsed_save_citation["name"] == "save_citation"
    assert parsed_save_citation["args"]["source"] == "pdf"
    assert parsed_save_citation["args"]["key"] == "annolid2024"
    assert parsed_save_citation["args"]["bib_file"] == "refs.bib"
    assert parsed_save_citation["args"]["validate_before_save"] is True
    assert parsed_save_citation["args"]["strict_validation"] is False

    parsed_save_citation_strict = task._parse_direct_gui_command(
        "save citation from web to refs.bib with strict validation without validation"
    )
    assert parsed_save_citation_strict["name"] == "save_citation"
    assert parsed_save_citation_strict["args"]["source"] == "web"
    assert parsed_save_citation_strict["args"]["validate_before_save"] is False
    assert parsed_save_citation_strict["args"]["strict_validation"] is True

    parsed_add_citation_raw = task._parse_direct_gui_command(
        "add citation @article{yang2024annolid,title={Annolid},author={Yang, Chen and Cleland, Thomas A},year={2024}} to refs.bib"
    )
    assert parsed_add_citation_raw["name"] == "add_citation_raw"
    assert "@article{yang2024annolid" in parsed_add_citation_raw["args"]["bibtex"]
    assert parsed_add_citation_raw["args"]["bib_file"] == "refs.bib"

    parsed_list_citations = task._parse_direct_gui_command(
        "list citations from refs.bib for annolid"
    )
    assert parsed_list_citations["name"] == "list_citations"
    assert parsed_list_citations["args"]["bib_file"] == "refs.bib"
    assert parsed_list_citations["args"]["query"] == "annolid"


def test_execute_direct_gui_command_routes_actions(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    video_file = tmp_path / "mouse.mp4"
    video_file.write_bytes(b"fake")
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    class _Cfg:
        class tools:  # noqa: N801
            email = None
            allowed_read_roots = [str(tmp_path)]

    monkeypatch.setattr(backend, "load_config", lambda: _Cfg())
    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    task._invoke_widget_slot = lambda *args, **kwargs: True  # type: ignore[method-assign]
    task._tool_clawhub_search_skills = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": True,
        "query": kwargs.get("query", ""),
        "results": [
            {
                "slug": "behavior-labeler",
                "name": "Behavior Labeler",
                "description": "Label behavior segments from timelines.",
            }
        ],
        **kwargs,
    }
    task._tool_clawhub_install_skill = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": True,
        "slug": kwargs.get("slug", ""),
    }
    task._tool_gui_rename_file = lambda **kwargs: {  # type: ignore[method-assign]
        **kwargs,
        "ok": True,
        "old_path": str(pdf_file),
        "new_path": str(
            tmp_path / "A_3_Dimensional_Digital_Atlas_of_the_Starling_Brain.pdf"
        ),
    }
    task._tool_gui_save_citation = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": True,
        "created": True,
        "key": kwargs.get("key") or "annolid2024",
        "bib_file": str(tmp_path / "citations.bib"),
        "source": kwargs.get("source") or "auto",
    }
    task._tool_gui_add_citation_raw = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": True,
        "created": True,
        "key": "yang2024annolid",
        "bib_file": str(tmp_path / "citations.bib"),
    }
    task._tool_gui_list_citations = lambda **kwargs: {  # type: ignore[method-assign]
        "ok": True,
        "count": 1,
        "entries": [
            {
                "key": "yang2024annolid",
                "title": "Annolid: Annotate, Segment, and Track Anything You Need",
                "year": "2024",
            }
        ],
        "bib_file": str(tmp_path / "citations.bib"),
    }

    out_pdf = asyncio.run(task._execute_direct_gui_command("open pdf"))
    assert "Opened PDF in Annolid:" in out_pdf

    out_url = asyncio.run(
        task._execute_direct_gui_command(
            "open https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        )
    )
    assert (
        "Opened URL in Annolid: https://brainglobe.info/documentation/brainglobe-atlasapi/usage/atlas-details.html"
        == out_url
    )

    out_url_browser = asyncio.run(
        task._execute_direct_gui_command("open google.com in browser")
    )
    assert "Opened URL in browser: https://google.com" == out_url_browser

    out_video = asyncio.run(task._execute_direct_gui_command("open video mouse.mp4"))
    assert "Opened video in Annolid:" in out_video

    out_frame = asyncio.run(task._execute_direct_gui_command("set frame 5"))
    assert "Moved to frame 5." == out_frame

    out_track = asyncio.run(task._execute_direct_gui_command("track to frame 60"))
    assert "Started tracking to frame 60." == out_track

    out_model = asyncio.run(
        task._execute_direct_gui_command("set chat model ollama/qwen3:8b")
    )
    assert "Updated chat model to ollama/qwen3:8b." == out_model

    out_stream = asyncio.run(
        task._execute_direct_gui_command(
            "open stream with model mediapipe face and classify eye blinks"
        )
    )
    assert "Started realtime stream with model mediapipe_face." == out_stream

    out_stop_stream = asyncio.run(
        task._execute_direct_gui_command("stop realtime stream")
    )
    assert "Stopped realtime stream." == out_stop_stream

    out_rename = asyncio.run(
        task._execute_direct_gui_command(
            "rename this pdf with title A_3_Dimensional_Digital_Atlas_of_the_Starling_Brain.pdf"
        )
    )
    assert "Renamed file:" in out_rename

    out_clawhub_search = asyncio.run(
        task._execute_direct_gui_command("search clawhub skills for segmentation")
    )
    assert "behavior-labeler" in out_clawhub_search
    assert "Label behavior segments" in out_clawhub_search

    out_clawhub_install = asyncio.run(
        task._execute_direct_gui_command("install skill behavior-labeler from clawhub")
    )
    assert "Installed skill 'behavior-labeler' from ClawHub." in out_clawhub_install

    out_save_citation = asyncio.run(
        task._execute_direct_gui_command(
            "save citation from pdf as annolid2024 to citations.bib"
        )
    )
    assert "Created citation 'annolid2024'" in out_save_citation

    out_add_citation_raw = asyncio.run(
        task._execute_direct_gui_command(
            "add citation @article{yang2024annolid,title={Annolid},author={Yang, Chen and Cleland, Thomas A},year={2024}} to citations.bib"
        )
    )
    assert "Created citation 'yang2024annolid'" in out_add_citation_raw

    out_list_citations = asyncio.run(
        task._execute_direct_gui_command("list citations from citations.bib")
    )
    assert "Found 1 citation(s):" in out_list_citations
    assert "yang2024annolid" in out_list_citations


def test_tool_gui_save_citation_from_active_pdf(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)
    task = StreamingChatTask("save citation", widget=None)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    task._tool_gui_pdf_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_pdf": True,
        "path": str(pdf_path),
        "title": "Amazing Paper.pdf",
    }
    task._tool_gui_pdf_get_text = lambda max_chars=8000, pages=2: {  # type: ignore[method-assign]
        "ok": True,
        "text": "A. Author\nAmazing Paper\n2024\nDOI:10.1000/example",
    }
    task._tool_gui_web_get_state = lambda: {"ok": True, "has_page": False}  # type: ignore[method-assign]

    payload = task._tool_gui_save_citation(source="pdf")
    assert payload["ok"] is True
    assert payload["source"] == "pdf"
    assert payload["fields"]["doi"] == "10.1000/example"
    assert Path(str(payload["bib_file"])).exists()


def test_tool_gui_add_citation_raw_upserts_entry(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend
    from annolid.utils.citations import load_bibtex

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)
    task = StreamingChatTask("add citation", widget=None)
    payload = task._tool_gui_add_citation_raw(
        bibtex=(
            "@article{yang2024annolid,"
            "title={Annolid: Annotate, Segment, and Track Anything You Need},"
            "author={Yang, Chen and Cleland, Thomas A},"
            "journal={arXiv preprint arXiv:2403.18690},"
            "year={2024}}"
        ),
        bib_file="refs.bib",
    )
    assert payload["ok"] is True
    assert payload["key"] == "yang2024annolid"
    saved = load_bibtex(tmp_path / "refs.bib")
    assert len(saved) == 1
    assert saved[0].fields["author"] == "Yang, Chen and Cleland, Thomas A"


def test_tool_gui_list_citations_returns_entries(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)
    task = StreamingChatTask("list citations", widget=None)
    refs = tmp_path / "refs.bib"
    refs.write_text(
        (
            "@article{yang2024annolid,\n"
            "  title={Annolid: Annotate, Segment, and Track Anything You Need},\n"
            "  author={Yang, Chen and Cleland, Thomas A},\n"
            "  year={2024},\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    payload = task._tool_gui_list_citations(bib_file="refs.bib", query="annolid")
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["entries"][0]["key"] == "yang2024annolid"


def test_tool_gui_save_citation_strict_validation_blocks_save(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)
    monkeypatch.setattr(
        backend,
        "validate_citation_metadata",
        lambda fields, timeout_s=1.8: {
            "checked": True,
            "verified": False,
            "provider": "crossref",
            "score": 0.25,
            "message": "weak match",
            "candidate": {},
        },
    )
    task = StreamingChatTask("save citation", widget=None)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    task._tool_gui_pdf_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_pdf": True,
        "path": str(pdf_path),
        "title": "Test Paper.pdf",
    }
    task._tool_gui_pdf_get_text = lambda max_chars=8000, pages=2: {  # type: ignore[method-assign]
        "ok": True,
        "text": "Test Paper\n2024\nDOI:10.1000/example",
    }
    task._tool_gui_web_get_state = lambda: {"ok": True, "has_page": False}  # type: ignore[method-assign]
    payload = task._tool_gui_save_citation(source="pdf", strict_validation=True)
    assert payload["ok"] is False
    assert "strict mode" in str(payload["error"]).lower()


def test_tool_gui_save_citation_skip_validation(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    calls = {"count": 0}

    def _should_not_be_called(fields, timeout_s=1.8):
        calls["count"] += 1
        return {
            "checked": True,
            "verified": True,
            "provider": "crossref",
            "score": 1.0,
            "message": "ok",
            "candidate": {},
        }

    monkeypatch.setattr(backend, "validate_citation_metadata", _should_not_be_called)
    task = StreamingChatTask("save citation", widget=None)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    task._tool_gui_pdf_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_pdf": True,
        "path": str(pdf_path),
        "title": "No Validate.pdf",
    }
    task._tool_gui_pdf_get_text = lambda max_chars=8000, pages=2: {  # type: ignore[method-assign]
        "ok": True,
        "text": "No Validate\n2024\nDOI:10.1000/example",
    }
    task._tool_gui_web_get_state = lambda: {"ok": True, "has_page": False}  # type: ignore[method-assign]
    payload = task._tool_gui_save_citation(source="pdf", validate_before_save=False)
    assert payload["ok"] is True
    assert calls["count"] == 0


def test_tool_gui_save_citation_uses_scholar_bibkey_and_fields(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)
    monkeypatch.setattr(
        backend,
        "validate_citation_metadata",
        lambda fields, timeout_s=1.8: {
            "checked": True,
            "verified": True,
            "provider": "google_scholar",
            "score": 0.68,
            "message": "matched",
            "candidate": {
                "__bibkey__": "lovell2020zebra",
                "title": "ZEBrA: Zebra finch Expression Brain Atlas",
                "author": ("Lovell, Peter V and Wirthlin, Morgan and Kaser, Taylor"),
                "journal": "Journal of Comparative Neurology",
                "year": "2020",
                "doi": "10.1002/cne.24879",
                "volume": "528",
                "number": "12",
                "pages": "2099--2131",
                "publisher": "Wiley Online Library",
            },
        },
    )
    task = StreamingChatTask("save citation", widget=None)
    task._tool_gui_web_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_page": True,
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/",
        "title": (
            "ZEBrA - Zebra finch Expression Brain Atlas: a resource for "
            "comparative molecular neuroanatomy and brain evolution studies - PMC"
        ),
    }
    task._tool_gui_web_get_dom_text = lambda max_chars=9000: {  # type: ignore[method-assign]
        "ok": True,
        "text": "DOI:10.1002/cne.24879 published 2021",
    }

    payload = task._tool_gui_save_citation(source="web")
    assert payload["ok"] is True
    assert payload["key"] == "lovell2020zebra"
    assert payload["fields"]["title"].startswith("ZEBrA:")
    assert payload["fields"]["year"] == "2020"
    assert payload["fields"]["volume"] == "528"


def test_tool_gui_clawhub_search_install_payload(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)
    task = StreamingChatTask("hi", widget=None)
    progress: list[str] = []
    task._emit_progress = lambda message: progress.append(message)  # type: ignore[method-assign]

    async def _fake_search(query, *, limit=5, workspace=None):
        return {
            "ok": True,
            "query": query,
            "limit": limit,
            "workspace": str(workspace),
        }

    async def _fake_install(slug, *, workspace=None):
        return {
            "ok": True,
            "slug": slug,
            "workspace": str(workspace),
            "skills_dir": str(Path(str(workspace)) / "skills"),
            "restart_hint": "Start a new Annolid Bot session to load newly installed skills.",
        }

    monkeypatch.setattr(backend, "clawhub_search_skills", _fake_search)
    monkeypatch.setattr(backend, "clawhub_install_skill", _fake_install)

    search_payload = asyncio.run(task._tool_clawhub_search_skills("behavior", limit=3))
    assert search_payload["ok"] is True
    assert search_payload["query"] == "behavior"
    assert search_payload["limit"] == 3
    assert search_payload["workspace"] == str(tmp_path)
    assert "ClawHub search: behavior" in progress

    install_payload = asyncio.run(task._tool_clawhub_install_skill("my-skill"))
    assert install_payload["ok"] is True
    assert install_payload["slug"] == "my-skill"
    assert install_payload["workspace"] == str(tmp_path)
    assert install_payload["skills_dir"] == str(tmp_path / "skills")
    assert "restart_hint" in install_payload


def test_tool_gui_rename_file_uses_active_pdf(monkeypatch, tmp_path: Path) -> None:
    import annolid.gui.widgets.ai_chat_backend as backend

    pdf_file = tmp_path / "Microsoft_Word_-_Manual_docx.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(backend, "get_agent_workspace_path", lambda: tmp_path)

    task = StreamingChatTask("hi", widget=None)
    task._tool_gui_pdf_get_state = lambda: {  # type: ignore[method-assign]
        "ok": True,
        "has_pdf": True,
        "path": str(pdf_file),
        "title": pdf_file.name,
    }
    calls: list[str] = []
    task._invoke_widget_slot = lambda slot_name, *args: calls.append(slot_name) or True  # type: ignore[method-assign]

    payload = task._tool_gui_rename_file(
        new_name="A_3_Dimensional_Digital_Atlas_of_the_Starling_Brain.pdf",
        use_active_file=True,
    )
    assert payload["ok"] is True
    assert payload["renamed"] is True
    assert (
        Path(payload["new_path"]).name
        == "A_3_Dimensional_Digital_Atlas_of_the_Starling_Brain.pdf"
    )
    assert calls == ["bot_open_pdf"]
