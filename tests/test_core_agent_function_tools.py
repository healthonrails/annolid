from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

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
