from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
import re

from annolid.core.agent.config import CalendarToolConfig
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_builtin import (
    CodeExplainTool,
    CodeSearchTool,
    CronTool,
    DownloadPdfTool,
    DownloadUrlTool,
    EditFileTool,
    ExecTool,
    ExtractPdfImagesTool,
    ExtractPdfTextTool,
    GitDiffTool,
    GitHubPrChecksTool,
    GitHubPrStatusTool,
    GitLogTool,
    GitStatusTool,
    ListDirTool,
    MemoryGetTool,
    MemorySetTool,
    MemorySearchTool,
    OpenPdfTool,
    ReadFileTool,
    RenameFileTool,
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
from annolid.core.agent.tools.mcp import MCPToolWrapper


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


def test_rename_file_tool_rename_and_overwrite(tmp_path: Path) -> None:
    writer = WriteFileTool(allowed_dir=tmp_path)
    renamer = RenameFileTool(allowed_dir=tmp_path)
    src = tmp_path / "old.pdf"
    dst = tmp_path / "new.pdf"
    conflict = tmp_path / "conflict.pdf"

    asyncio.run(writer.execute(path=str(src), content="v1"))
    asyncio.run(writer.execute(path=str(conflict), content="v2"))

    denied = asyncio.run(
        renamer.execute(path=str(src), new_path=str(conflict), overwrite=False)
    )
    assert "Target already exists" in denied

    renamed = asyncio.run(
        renamer.execute(path=str(src), new_path=str(dst), overwrite=False)
    )
    assert "Successfully renamed" in renamed
    assert not src.exists()
    assert dst.exists()

    replaced = asyncio.run(
        renamer.execute(path=str(conflict), new_path=str(dst), overwrite=True)
    )
    assert "Successfully renamed" in replaced
    assert not conflict.exists()
    assert dst.read_text(encoding="utf-8") == "v2"


def test_rename_file_tool_rejects_invalid_new_name(tmp_path: Path) -> None:
    writer = WriteFileTool(allowed_dir=tmp_path)
    renamer = RenameFileTool(allowed_dir=tmp_path)
    src = tmp_path / "paper.pdf"
    asyncio.run(writer.execute(path=str(src), content="pdf"))
    result = asyncio.run(
        renamer.execute(path=str(src), new_name="nested/path/illegal.pdf")
    )
    assert "new_name must be a base name" in result


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


def test_open_pdf_tool_uses_extract_pdf_text_backend(
    tmp_path: Path, monkeypatch
) -> None:
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
            return _FakeDoc([_FakePage("Page One"), _FakePage("Page Two")])

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz)
    tool = OpenPdfTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(tool.execute(path=str(pdf_path), start_page=1, max_pages=1))
    payload = json.loads(result)
    assert payload["backend"] == "pymupdf"
    assert payload["pages_read"] == 1
    assert payload["text"] == "Page One"


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


def test_cron_tool_add_one_time_at_iso_datetime(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    at_value = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    added = asyncio.run(
        tool.execute(
            action="add",
            message="one-shot",
            every_seconds=None,
            cron_expr=None,
            at=at_value,
        )
    )
    assert "Created job" in added
    listed = asyncio.run(tool.execute(action="list"))
    assert "Scheduled jobs" in listed
    assert "at=" in listed


def test_memory_search_and_get_tools(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text(
        "# Long-term\n\nPreferred species: zebrafish\n",
        encoding="utf-8",
    )
    (memory_dir / "2026-02-11.md").write_text(
        "# 2026-02-11\n\nReviewed tracking thresholds.\n",
        encoding="utf-8",
    )

    search_tool = MemorySearchTool(workspace=tmp_path)
    result = asyncio.run(search_tool.execute(query="zebrafish preference", top_k=3))
    payload = json.loads(result)
    assert payload["count"] >= 1
    assert any(item["path"] == "memory/MEMORY.md" for item in payload["results"])

    get_tool = MemoryGetTool(workspace=tmp_path)
    got = asyncio.run(get_tool.execute(path="MEMORY.md", start_line=1, end_line=2))
    got_payload = json.loads(got)
    assert got_payload["path"] == "memory/MEMORY.md"
    assert "# Long-term" in got_payload["content"]

    blocked = asyncio.run(get_tool.execute(path="../secret.md"))
    blocked_payload = json.loads(blocked)
    assert "allowed" in blocked_payload["error"]


def test_memory_set_tool_writes_long_term_memory(tmp_path: Path) -> None:
    set_tool = MemorySetTool(workspace=tmp_path)
    first = asyncio.run(set_tool.execute(key="preferred_species", value="zebrafish"))
    first_payload = json.loads(first)
    assert first_payload["ok"] is True
    assert first_payload["path"] == "memory/MEMORY.md"

    second = asyncio.run(set_tool.execute(note="Use higher threshold for arena C"))
    second_payload = json.loads(second)
    assert second_payload["ok"] is True

    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- preferred_species: zebrafish" in memory_text
    assert "- Use higher threshold for arena C" in memory_text


def test_code_search_tool_finds_matches_with_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "module.py"
    target.write_text(
        "def load_config(path):\n"
        "    return path\n"
        "\n"
        "def save_config(path, content):\n"
        "    return content\n",
        encoding="utf-8",
    )
    tool = CodeSearchTool(allowed_dir=workspace)
    result = asyncio.run(
        tool.execute(
            query="config",
            path=str(workspace),
            glob="*.py",
            context_lines=1,
            max_results=10,
        )
    )
    payload = json.loads(result)
    assert payload["count"] >= 2
    assert payload["truncated"] is False
    first = payload["results"][0]
    assert first["path"] == "module.py"
    assert "context" in first


def test_code_explain_tool_describes_module_and_symbol(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "analyzer.py"
    target.write_text(
        '"""Behavior analysis helpers."""\n'
        "import json\n"
        "\n"
        "class Runner:\n"
        '    """Executes processing."""\n'
        "    def run(self, value):\n"
        "        return json.dumps(value)\n"
        "\n"
        "def normalize(data):\n"
        "    return str(data).strip()\n",
        encoding="utf-8",
    )
    tool = CodeExplainTool(allowed_dir=workspace)

    module_result = asyncio.run(tool.execute(path=str(target)))
    module_payload = json.loads(module_result)
    assert module_payload["module_docstring"] == "Behavior analysis helpers."
    assert any(item["name"] == "Runner" for item in module_payload["classes"])
    assert any(item["name"] == "normalize" for item in module_payload["functions"])

    symbol_result = asyncio.run(
        tool.execute(path=str(target), symbol="Runner.run", include_source=True)
    )
    symbol_payload = json.loads(symbol_result)
    assert symbol_payload["kind"] == "function"
    assert symbol_payload["name"] == "run"
    assert "json.dumps" in symbol_payload["calls"]
    assert "def run" in symbol_payload["source"]


def test_git_tools_status_diff_log(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available in this environment")
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "annolid@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Annolid Bot"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    tracked = repo / "tracked.txt"
    tracked.write_text("line1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    tracked.write_text("line1\nline2\n", encoding="utf-8")

    status_tool = GitStatusTool(allowed_dir=repo)
    status_result = asyncio.run(status_tool.execute(repo_path=str(repo)))
    status_payload = json.loads(status_result)
    assert status_payload["exit_code"] == 0
    assert "tracked.txt" in status_payload["output"]

    diff_tool = GitDiffTool(allowed_dir=repo)
    diff_result = asyncio.run(diff_tool.execute(repo_path=str(repo)))
    diff_payload = json.loads(diff_result)
    assert diff_payload["exit_code"] == 0
    assert "+line2" in diff_payload["output"]

    log_tool = GitLogTool(allowed_dir=repo)
    log_result = asyncio.run(log_tool.execute(repo_path=str(repo), max_count=5))
    log_payload = json.loads(log_result)
    assert log_payload["exit_code"] == 0
    assert "initial" in log_payload["output"]


def test_github_tools_report_missing_gh_cli(tmp_path: Path, monkeypatch) -> None:
    async def _missing_command(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("gh")

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _missing_command,
    )
    status_tool = GitHubPrStatusTool(allowed_dir=tmp_path)
    checks_tool = GitHubPrChecksTool(allowed_dir=tmp_path)

    status_result = asyncio.run(status_tool.execute(repo_path=str(tmp_path)))
    checks_result = asyncio.run(checks_tool.execute(repo_path=str(tmp_path)))
    status_payload = json.loads(status_result)
    checks_payload = json.loads(checks_result)
    assert "Command not found: gh" in status_payload["error"]
    assert "Command not found: gh" in checks_payload["error"]


def test_register_nanobot_style_tools(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()
    asyncio.run(register_nanobot_style_tools(registry, allowed_dir=tmp_path))
    assert registry.has("read_file")
    assert registry.has("rename_file")
    assert registry.has("code_search")
    assert registry.has("code_explain")
    assert registry.has("git_status")
    assert registry.has("git_diff")
    assert registry.has("git_log")
    assert registry.has("github_pr_status")
    assert registry.has("github_pr_checks")
    assert registry.has("memory_search")
    assert registry.has("memory_get")
    assert registry.has("memory_set")
    assert registry.has("extract_pdf_text")
    assert registry.has("open_pdf")
    assert registry.has("extract_pdf_images")
    assert registry.has("video_info")
    assert registry.has("video_sample_frames")
    assert registry.has("video_segment")
    assert registry.has("video_process_segments")
    assert registry.has("exec")
    assert registry.has("cron")
    assert registry.has("download_url")
    assert registry.has("download_pdf")
    assert registry.has("clawhub_search_skills")
    assert registry.has("clawhub_install_skill")


def test_register_nanobot_style_tools_skips_calendar_when_deps_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = FunctionToolRegistry()
    monkeypatch.setattr(
        "annolid.core.agent.tools.nanobot.GoogleCalendarTool.is_available",
        lambda: False,
    )
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            calendar_cfg=CalendarToolConfig(enabled=True, provider="google"),
        )
    )
    assert registry.has("google_calendar") is False


def test_mcp_tool_wrapper_sanitizes_name_and_schema() -> None:
    class _ToolDef:
        name = "search.web"
        description = "Search"
        inputSchema = {"properties": {"query": {"type": "string"}}}

    wrapper = MCPToolWrapper(
        session=object(),
        server_name="weather-server:v1",
        tool_def=_ToolDef(),
    )
    assert re.match(r"^[A-Za-z0-9_]+$", wrapper.name)
    assert len(wrapper.name) <= 64
    assert wrapper.parameters["type"] == "object"
    assert "query" in wrapper.parameters["properties"]


def test_mcp_tool_wrapper_execute_falls_back_to_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TextContent:
        def __init__(self, text: str) -> None:
            self.text = text

    fake_mcp = types.SimpleNamespace(
        types=types.SimpleNamespace(TextContent=_TextContent)
    )
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)

    class _ToolDef:
        name = "tool"
        description = "Tool"
        inputSchema = {"type": "object", "properties": {}}

    class _Session:
        async def call_tool(self, name: str, arguments: dict) -> object:
            assert name == "tool"
            assert arguments == {"x": 1}
            return types.SimpleNamespace(
                content=[],
                structuredContent={"ok": True, "value": 1},
                isError=False,
            )

    wrapper = MCPToolWrapper(
        session=_Session(),
        server_name="s",
        tool_def=_ToolDef(),
    )
    payload = asyncio.run(wrapper.execute(x=1))
    assert json.loads(payload) == {"ok": True, "value": 1}


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


def test_download_pdf_tool_enforces_pdf_content_type(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://example.org/paper.pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeTextResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://example.org/not-a-pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"<html>hello</html>"

    class _FakeStreamContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

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
            del method, headers
            if str(url).endswith(".pdf"):
                return _FakeStreamContext(_FakePdfResponse())
            return _FakeStreamContext(_FakeTextResponse())

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    out_path = tmp_path / "downloads" / "paper.pdf"
    ok = asyncio.run(
        tool.execute(url="https://example.org/paper.pdf", output_path=str(out_path))
    )
    ok_payload = json.loads(ok)
    assert ok_payload["is_pdf"] is True
    assert Path(ok_payload["output_path"]).exists()

    bad = asyncio.run(
        tool.execute(
            url="https://example.org/not-a-pdf",
            output_path=str(tmp_path / "downloads" / "bad.pdf"),
        )
    )
    bad_payload = json.loads(bad)
    assert "not allowed" in str(bad_payload.get("error", ""))


def test_download_pdf_tool_renames_generic_pdf_filename(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://example.org/pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakePdfResponse()

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

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    monkeypatch.setattr(
        tool,
        "_extract_pdf_title",
        lambda _path: "Neural Circuit Dynamics in Mouse Cortex",
    )
    result = asyncio.run(tool.execute(url="https://example.org/pdf"))
    payload = json.loads(result)

    output_path = Path(str(payload["output_path"]))
    assert payload["is_pdf"] is True
    assert payload["renamed"] is True
    assert output_path.name == "Neural_Circuit_Dynamics_in_Mouse_Cortex.pdf"
    assert output_path.exists()
    assert not (tmp_path / "downloads" / "pdf.pdf").exists()


def test_download_pdf_tool_renames_non_generic_when_title_differs(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://www.biorxiv.org/content/10.64898/2026.01.20.700446v2.full.pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakePdfResponse()

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

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    monkeypatch.setattr(
        tool,
        "_extract_pdf_title",
        lambda _path: "A Better Paper Title",
    )
    result = asyncio.run(
        tool.execute(
            url="https://www.biorxiv.org/content/10.64898/2026.01.20.700446v2.full.pdf"
        )
    )
    payload = json.loads(result)

    output_path = Path(str(payload["output_path"]))
    assert payload["is_pdf"] is True
    assert payload["renamed"] is True
    assert output_path.name == "A_Better_Paper_Title.pdf"
    assert output_path.exists()
    assert not (tmp_path / "downloads" / "2026.01.20.700446v2.full.pdf").exists()


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
        open_url_callback=lambda url: _mark("open_url", url),
        open_in_browser_callback=lambda url: _mark("open_in_browser", url),
        web_get_dom_text_callback=lambda max_chars=8000: _mark(
            "web_get_dom_text", max_chars
        ),
        web_click_callback=lambda selector: _mark("web_click", selector),
        web_type_callback=lambda selector, text, submit=False: _mark(
            "web_type", {"selector": selector, "text": text, "submit": bool(submit)}
        ),
        web_scroll_callback=lambda delta_y=800: _mark("web_scroll", delta_y),
        web_find_forms_callback=lambda: _mark("web_find_forms"),
        web_run_steps_callback=lambda steps, stop_on_error=True, max_steps=12: _mark(
            "web_run_steps",
            {
                "steps": steps,
                "stop_on_error": bool(stop_on_error),
                "max_steps": int(max_steps),
            },
        ),
        open_pdf_callback=lambda path="": _mark("open_pdf", path or None),
        pdf_get_state_callback=lambda: _mark("pdf_get_state"),
        pdf_get_text_callback=lambda max_chars=8000, pages=2: _mark(
            "pdf_get_text", {"max_chars": int(max_chars), "pages": int(pages)}
        ),
        pdf_find_sections_callback=lambda max_sections=20, max_pages=12: _mark(
            "pdf_find_sections",
            {"max_sections": int(max_sections), "max_pages": int(max_pages)},
        ),
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
        set_ai_text_prompt_callback=lambda text, use_countgd=False: _mark(
            "set_ai_text_prompt", f"{text}|{bool(use_countgd)}"
        ),
        run_ai_text_segmentation_callback=lambda: _mark("run_ai_text_segmentation"),
        segment_track_video_callback=lambda **kwargs: _mark(
            "segment_track_video", kwargs
        ),
        label_behavior_segments_callback=lambda **kwargs: _mark(
            "label_behavior_segments", kwargs
        ),
        start_realtime_stream_callback=lambda **kwargs: _mark(
            "start_realtime_stream", kwargs
        ),
        stop_realtime_stream_callback=lambda: _mark("stop_realtime_stream"),
    )
    assert registry.has("gui_context")
    assert registry.has("gui_shared_image_path")
    assert registry.has("gui_open_video")
    assert registry.has("gui_open_url")
    assert registry.has("gui_open_in_browser")
    assert registry.has("gui_web_get_dom_text")
    assert registry.has("gui_web_click")
    assert registry.has("gui_web_type")
    assert registry.has("gui_web_scroll")
    assert registry.has("gui_web_find_forms")
    assert registry.has("gui_web_run_steps")
    assert registry.has("gui_open_pdf")
    assert registry.has("gui_pdf_get_state")
    assert registry.has("gui_pdf_get_text")
    assert registry.has("gui_pdf_find_sections")
    assert registry.has("gui_set_frame")
    assert registry.has("gui_set_chat_prompt")
    assert registry.has("gui_send_chat_prompt")
    assert registry.has("gui_set_chat_model")
    assert registry.has("gui_select_annotation_model")
    assert registry.has("gui_track_next_frames")
    assert registry.has("gui_set_ai_text_prompt")
    assert registry.has("gui_run_ai_text_segmentation")
    assert registry.has("gui_segment_track_video")
    assert registry.has("gui_label_behavior_segments")
    assert registry.has("gui_start_realtime_stream")
    assert registry.has("gui_stop_realtime_stream")
    ctx = asyncio.run(registry.execute("gui_context", {}))
    ctx_payload = json.loads(ctx)
    assert ctx_payload["provider"] == "ollama"
    image = asyncio.run(registry.execute("gui_shared_image_path", {}))
    image_payload = json.loads(image)
    assert image_payload["image_path"] == "/tmp/shared.png"
    result = asyncio.run(registry.execute("gui_open_video", {"path": "/tmp/a.mp4"}))
    assert json.loads(result)["ok"] is True
    open_url = asyncio.run(
        registry.execute("gui_open_url", {"url": "https://example.org"})
    )
    assert json.loads(open_url)["ok"] is True
    open_in_browser = asyncio.run(
        registry.execute("gui_open_in_browser", {"url": "https://example.org"})
    )
    assert json.loads(open_in_browser)["ok"] is True
    web_get_dom_text = asyncio.run(
        registry.execute("gui_web_get_dom_text", {"max_chars": 1200})
    )
    assert json.loads(web_get_dom_text)["ok"] is True
    web_click = asyncio.run(
        registry.execute("gui_web_click", {"selector": "button.submit"})
    )
    assert json.loads(web_click)["ok"] is True
    web_type = asyncio.run(
        registry.execute(
            "gui_web_type",
            {"selector": "input[name='q']", "text": "annolid", "submit": True},
        )
    )
    assert json.loads(web_type)["ok"] is True
    web_scroll = asyncio.run(registry.execute("gui_web_scroll", {"delta_y": 600}))
    assert json.loads(web_scroll)["ok"] is True
    web_find_forms = asyncio.run(registry.execute("gui_web_find_forms", {}))
    assert json.loads(web_find_forms)["ok"] is True
    web_run_steps = asyncio.run(
        registry.execute(
            "gui_web_run_steps",
            {
                "steps": [{"action": "open_url", "url": "https://example.org"}],
                "stop_on_error": True,
                "max_steps": 5,
            },
        )
    )
    assert json.loads(web_run_steps)["ok"] is True
    open_pdf = asyncio.run(registry.execute("gui_open_pdf", {}))
    assert json.loads(open_pdf)["ok"] is True
    open_pdf_with_path = asyncio.run(
        registry.execute("gui_open_pdf", {"path": "/tmp/paper.pdf"})
    )
    assert json.loads(open_pdf_with_path)["ok"] is True
    pdf_state = asyncio.run(registry.execute("gui_pdf_get_state", {}))
    assert json.loads(pdf_state)["ok"] is True
    pdf_text = asyncio.run(
        registry.execute("gui_pdf_get_text", {"max_chars": 1200, "pages": 2})
    )
    assert json.loads(pdf_text)["ok"] is True
    pdf_sections = asyncio.run(
        registry.execute("gui_pdf_find_sections", {"max_sections": 10, "max_pages": 8})
    )
    assert json.loads(pdf_sections)["ok"] is True
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
    asyncio.run(
        registry.execute(
            "gui_set_ai_text_prompt",
            {"text": "mouse", "use_countgd": True},
        )
    )
    asyncio.run(registry.execute("gui_run_ai_text_segmentation", {}))
    asyncio.run(
        registry.execute(
            "gui_segment_track_video",
            {
                "path": "/tmp/a.mp4",
                "text_prompt": "mouse",
                "mode": "track",
                "to_frame": 120,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_label_behavior_segments",
            {
                "path": "/tmp/a.mp4",
                "behavior_labels": ["walking", "eating"],
                "segment_mode": "uniform",
                "segment_frames": 30,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_start_realtime_stream",
            {
                "camera_source": "0",
                "model_name": "mediapipe_face",
                "classify_eye_blinks": True,
            },
        )
    )
    asyncio.run(registry.execute("gui_stop_realtime_stream", {}))
    assert calls == [
        ("open_video", "/tmp/a.mp4"),
        ("open_url", "https://example.org"),
        ("open_in_browser", "https://example.org"),
        ("web_get_dom_text", 1200),
        ("web_click", "button.submit"),
        (
            "web_type",
            {"selector": "input[name='q']", "text": "annolid", "submit": True},
        ),
        ("web_scroll", 600),
        ("web_find_forms", None),
        (
            "web_run_steps",
            {
                "steps": [{"action": "open_url", "url": "https://example.org"}],
                "stop_on_error": True,
                "max_steps": 5,
            },
        ),
        ("open_pdf", None),
        ("open_pdf", "/tmp/paper.pdf"),
        ("pdf_get_state", None),
        ("pdf_get_text", {"max_chars": 1200, "pages": 2}),
        ("pdf_find_sections", {"max_sections": 10, "max_pages": 8}),
        ("set_frame", 3),
        ("set_prompt", "describe this"),
        ("send_prompt", None),
        ("set_chat_model", "ollama:qwen3:8b"),
        ("select_model", "Segment Anything 2"),
        ("track", 120),
        ("set_ai_text_prompt", "mouse|True"),
        ("run_ai_text_segmentation", None),
        (
            "segment_track_video",
            {
                "path": "/tmp/a.mp4",
                "text_prompt": "mouse",
                "mode": "track",
                "to_frame": 120,
            },
        ),
        (
            "label_behavior_segments",
            {
                "path": "/tmp/a.mp4",
                "behavior_labels": ["walking", "eating"],
                "segment_mode": "uniform",
                "segment_frames": 30,
            },
        ),
        (
            "start_realtime_stream",
            {
                "camera_source": "0",
                "model_name": "mediapipe_face",
                "classify_eye_blinks": True,
            },
        ),
        ("stop_realtime_stream", None),
    ]
