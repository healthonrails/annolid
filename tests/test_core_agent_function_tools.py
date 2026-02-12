from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_builtin import (
    CronTool,
    DownloadUrlTool,
    EditFileTool,
    ExecTool,
    ExtractPdfImagesTool,
    ExtractPdfTextTool,
    ListDirTool,
    ReadFileTool,
    WebSearchTool,
    WriteFileTool,
    register_nanobot_style_tools,
)
from annolid.core.agent.tools.function_video import (
    VideoInfoTool,
    VideoProcessSegmentsTool,
    VideoSampleFramesTool,
    VideoSegmentTool,
)
from annolid.core.agent.tools.function_gui import register_annolid_gui_tools
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


class _EchoTool(FunctionTool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo text."

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs) -> str:
        return str(kwargs.get("text", ""))


def _write_test_video(path: Path, *, fps: float = 10.0, frames: int = 8) -> None:
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(int(frames)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = (idx * 20) % 255
            writer.write(frame)
    finally:
        writer.release()


def test_function_registry_validate_and_execute() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    bad = asyncio.run(registry.execute("echo", {"text": 123}))
    assert "Invalid parameters" in bad
    ok = asyncio.run(registry.execute("echo", {"text": "hi"}))
    assert ok == "hi"


def test_filesystem_tools_round_trip(tmp_path: Path) -> None:
    write = WriteFileTool(allowed_dir=tmp_path)
    read = ReadFileTool(allowed_dir=tmp_path)
    edit = EditFileTool(allowed_dir=tmp_path)
    list_dir = ListDirTool(allowed_dir=tmp_path)
    file_path = tmp_path / "note.txt"

    wrote = asyncio.run(write.execute(path=str(file_path), content="hello"))
    assert "Successfully wrote" in wrote
    text = asyncio.run(read.execute(path=str(file_path)))
    assert text == "hello"
    edited = asyncio.run(
        edit.execute(path=str(file_path), old_text="hello", new_text="world")
    )
    assert "Successfully edited" in edited
    listed = asyncio.run(list_dir.execute(path=str(tmp_path)))
    assert "note.txt" in listed


def test_read_file_rejects_pdf_with_actionable_message(tmp_path: Path) -> None:
    tool = ReadFileTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(tool.execute(path=str(pdf_path)))
    assert "extract_pdf_text" in result


def test_extract_pdf_text_tool_uses_fitz_backend(tmp_path: Path, monkeypatch) -> None:
    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self._text

    class _FakeDoc:
        def __init__(self, pages: list[_FakePage]) -> None:
            self._pages = pages

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, idx: int) -> _FakePage:
            return self._pages[idx]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeFitz:
        @staticmethod
        def open(path: str) -> _FakeDoc:
            del path
            return _FakeDoc([_FakePage("Intro"), _FakePage("Results")])

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz)
    tool = ExtractPdfTextTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(
        tool.execute(path=str(pdf_path), start_page=1, max_pages=2, max_chars=1000)
    )
    payload = json.loads(result)
    assert payload["backend"] == "pymupdf"
    assert payload["pages_read"] == 2
    assert "Intro" in payload["text"]
    assert "Results" in payload["text"]


def test_extract_pdf_images_tool_renders_pages(tmp_path: Path, monkeypatch) -> None:
    class _FakePixmap:
        def __init__(self, content: bytes) -> None:
            self._content = content

        def save(self, path: str) -> None:
            Path(path).write_bytes(self._content)

    class _FakePage:
        def __init__(self, number: int) -> None:
            self._number = number

        def get_pixmap(self, matrix=None, alpha=False):
            del matrix, alpha
            return _FakePixmap(f"page-{self._number}".encode("utf-8"))

    class _FakeDoc:
        def __init__(self, pages: list[_FakePage]) -> None:
            self._pages = pages

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, idx: int) -> _FakePage:
            return self._pages[idx]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeFitz:
        @staticmethod
        def Matrix(x: float, y: float):
            return (x, y)

        @staticmethod
        def open(path: str) -> _FakeDoc:
            del path
            return _FakeDoc([_FakePage(1), _FakePage(2)])

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz)
    tool = ExtractPdfImagesTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(
        tool.execute(path=str(pdf_path), start_page=1, max_pages=2, dpi=144)
    )
    payload = json.loads(result)
    assert payload["pages_rendered"] == 2
    for image_path in payload["images"]:
        image_file = Path(image_path)
        assert image_file.exists()
        assert image_file.suffix == ".png"


def test_video_info_tool_reads_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=8.0, frames=6)
    tool = VideoInfoTool(allowed_dir=tmp_path)
    result = asyncio.run(tool.execute(path=str(video_path)))
    payload = json.loads(result)
    assert payload["total_frames"] == 6
    assert payload["fps"] > 0
    assert payload["width"] == 64
    assert payload["height"] == 48


def test_video_sample_frames_tool_stream_mode(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=10.0, frames=10)
    tool = VideoSampleFramesTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            path=str(video_path),
            mode="stream",
            start_frame=2,
            step=2,
            max_frames=3,
        )
    )
    payload = json.loads(result)
    assert payload["count"] == 3
    frame_indices = [item["frame_index"] for item in payload["frames"]]
    assert frame_indices == [2, 4, 6]
    for item in payload["frames"]:
        assert Path(item["image_path"]).exists()


def test_video_segment_tool_exports_frame_range(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=10.0, frames=12)
    out_path = tmp_path / "tiny_seg.avi"
    tool = VideoSegmentTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            path=str(video_path),
            output_path=str(out_path),
            start_frame=3,
            end_frame=6,
            overwrite=True,
        )
    )
    payload = json.loads(result)
    assert payload["frames_written"] == 4
    assert out_path.exists()


def test_video_process_segments_tool_exports_multiple_ranges(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=10.0, frames=12)
    tool = VideoProcessSegmentsTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            path=str(video_path),
            segments=[
                {"start_frame": 0, "end_frame": 2},
                {"start_sec": 0.3, "end_sec": 0.5},
            ],
            overwrite=True,
        )
    )
    payload = json.loads(result)
    assert payload["segments_processed"] == 2
    assert len(payload["results"]) == 2
    for item in payload["results"]:
        assert item["frames_written"] > 0
        assert Path(item["output_path"]).exists()


def test_video_tools_allow_external_read_root_but_write_to_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    video_path = external / "mouse.avi"
    _write_test_video(video_path, fps=10.0, frames=8)

    info_tool = VideoInfoTool(allowed_dir=workspace, allowed_read_roots=[str(external)])
    info_result = asyncio.run(info_tool.execute(path=str(video_path)))
    info_payload = json.loads(info_result)
    assert info_payload["total_frames"] == 8

    sample_tool = VideoSampleFramesTool(
        allowed_dir=workspace, allowed_read_roots=[str(external)]
    )
    sample_result = asyncio.run(
        sample_tool.execute(path=str(video_path), mode="stream", step=2, max_frames=2)
    )
    sample_payload = json.loads(sample_result)
    assert sample_payload["count"] == 2
    for frame in sample_payload["frames"]:
        image_path = Path(frame["image_path"])
        assert image_path.exists()
        assert str(image_path).startswith(str(workspace))


def test_exec_tool_guard_blocks_dangerous() -> None:
    tool = ExecTool()
    result = asyncio.run(tool.execute(command="rm -rf /tmp/foo"))
    assert "blocked by safety guard" in result


def test_web_search_tool_without_key_reports_config_error() -> None:
    tool = WebSearchTool(api_key="")
    result = asyncio.run(tool.execute(query="annolid"))
    assert "BRAVE_API_KEY not configured" in result


def test_cron_tool_add_list_remove(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    added = asyncio.run(
        tool.execute(action="add", message="ping", every_seconds=30, cron_expr=None)
    )
    assert "Created job" in added
    listed = asyncio.run(tool.execute(action="list"))
    assert "Scheduled jobs" in listed
    job_id = added.split("id: ")[-1].rstrip(")")
    removed = asyncio.run(tool.execute(action="remove", job_id=job_id))
    assert f"Removed job {job_id}" == removed


def test_register_nanobot_style_tools(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()
    register_nanobot_style_tools(registry, allowed_dir=tmp_path)
    assert registry.has("read_file")
    assert registry.has("extract_pdf_text")
    assert registry.has("extract_pdf_images")
    assert registry.has("video_info")
    assert registry.has("video_sample_frames")
    assert registry.has("video_segment")
    assert registry.has("video_process_segments")
    assert registry.has("exec")
    assert registry.has("cron")
    assert registry.has("download_url")


def test_download_url_tool_saves_file_and_blocks_outside_dir(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakeResponse:
        status_code = 200
        headers = {"content-type": "text/plain; charset=utf-8"}
        url = "https://example.org/note.txt"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"hello "
            yield b"agent"

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, headers=None):
            del method, url, headers
            return _FakeStreamContext()

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadUrlTool(allowed_dir=tmp_path)
    out_path = tmp_path / "downloads" / "note.txt"
    result = asyncio.run(
        tool.execute(
            url="https://example.org/note.txt",
            output_path=str(out_path),
            content_type_prefixes=["text/plain"],
        )
    )
    payload = json.loads(result)
    assert payload["output_path"] == str(out_path)
    assert out_path.read_text(encoding="utf-8") == "hello agent"

    blocked = asyncio.run(
        tool.execute(
            url="https://example.org/note.txt",
            output_path=str(tmp_path.parent / "escape.txt"),
        )
    )
    blocked_payload = json.loads(blocked)
    assert "outside allowed directory" in blocked_payload["error"]


def test_register_annolid_gui_tools_and_context_payload() -> None:
    calls: list[tuple[str, object]] = []

    def _mark(name: str, value: object = None) -> dict[str, object]:
        calls.append((name, value))
        return {"ok": True}

    registry = FunctionToolRegistry()
    register_annolid_gui_tools(
        registry,
        context_callback=lambda: {"provider": "ollama", "frame_number": 12},
        image_path_callback=lambda: "/tmp/shared.png",
        open_video_callback=lambda path: _mark("open_video", path),
        set_frame_callback=lambda frame_index: _mark("set_frame", frame_index),
        set_prompt_callback=lambda text: _mark("set_prompt", text),
        send_prompt_callback=lambda: _mark("send_prompt"),
        set_chat_model_callback=lambda provider, model: _mark(
            "set_chat_model", f"{provider}:{model}"
        ),
        select_annotation_model_callback=lambda model_name: _mark(
            "select_model", model_name
        ),
        track_next_frames_callback=lambda to_frame: _mark("track", to_frame),
    )
    assert registry.has("gui_context")
    assert registry.has("gui_shared_image_path")
    assert registry.has("gui_open_video")
    assert registry.has("gui_set_frame")
    assert registry.has("gui_set_chat_prompt")
    assert registry.has("gui_send_chat_prompt")
    assert registry.has("gui_set_chat_model")
    assert registry.has("gui_select_annotation_model")
    assert registry.has("gui_track_next_frames")
    ctx = asyncio.run(registry.execute("gui_context", {}))
    ctx_payload = json.loads(ctx)
    assert ctx_payload["provider"] == "ollama"
    image = asyncio.run(registry.execute("gui_shared_image_path", {}))
    image_payload = json.loads(image)
    assert image_payload["image_path"] == "/tmp/shared.png"
    result = asyncio.run(registry.execute("gui_open_video", {"path": "/tmp/a.mp4"}))
    assert json.loads(result)["ok"] is True
    asyncio.run(registry.execute("gui_set_frame", {"frame_index": 3}))
    asyncio.run(registry.execute("gui_set_chat_prompt", {"text": "describe this"}))
    asyncio.run(registry.execute("gui_send_chat_prompt", {}))
    asyncio.run(
        registry.execute(
            "gui_set_chat_model", {"provider": "ollama", "model": "qwen3:8b"}
        )
    )
    asyncio.run(
        registry.execute(
            "gui_select_annotation_model", {"model_name": "Segment Anything 2"}
        )
    )
    asyncio.run(registry.execute("gui_track_next_frames", {"to_frame": 120}))
    assert calls == [
        ("open_video", "/tmp/a.mp4"),
        ("set_frame", 3),
        ("set_prompt", "describe this"),
        ("send_prompt", None),
        ("set_chat_model", "ollama:qwen3:8b"),
        ("select_model", "Segment Anything 2"),
        ("track", 120),
    ]
