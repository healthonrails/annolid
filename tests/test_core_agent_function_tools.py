from __future__ import annotations

import asyncio
from pathlib import Path

from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_builtin import (
    CronTool,
    EditFileTool,
    ExecTool,
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
    assert registry.has("exec")
    assert registry.has("cron")
