from __future__ import annotations

import pytest

from annolid.core.agent.tools.base import ToolContext, ToolError
from annolid.core.agent.tools.registry import ToolRegistry
from annolid.core.agent.tools.utility import (
    CalculatorTool,
    DateTimeTool,
    TextStatsTool,
    register_builtin_utility_tools,
)


def _ctx(tmp_path):
    return ToolContext(video_path=tmp_path / "v.mp4", results_dir=tmp_path, run_id="r1")


def test_calculator_tool_basic_math(tmp_path) -> None:
    tool = CalculatorTool()
    out = tool.run(_ctx(tmp_path), "2 + 3 * 4")
    assert out.result == 14.0


def test_calculator_tool_rejects_unsafe_expressions(tmp_path) -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolError):
        tool.run(_ctx(tmp_path), "__import__('os').system('echo hacked')")


def test_datetime_tool_supports_utc_offset(tmp_path) -> None:
    tool = DateTimeTool()
    out = tool.run(_ctx(tmp_path), {"utc_offset": "+08:00"})
    assert out.utc_offset == "+08:00"
    assert "T" in out.iso


def test_text_stats_tool_counts_words_and_lines(tmp_path) -> None:
    tool = TextStatsTool()
    out = tool.run(_ctx(tmp_path), "hello world\nnext line")
    assert out.words == 4
    assert out.lines == 2


def test_register_builtin_utility_tools(tmp_path) -> None:
    registry = ToolRegistry()
    register_builtin_utility_tools(registry)
    assert registry.has("calculator")
    assert registry.has("datetime")
    assert registry.has("text_stats")
    assert registry.create("text_stats").run(_ctx(tmp_path), "x").characters == 1
