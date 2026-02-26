from __future__ import annotations

import asyncio
import json
import sys
import types

from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.core.agent.tools.mcp_browser import (
    McpBrowserNavigateTool,
    McpBrowserTool,
    _validate_browser_url,
    register_mcp_browser_tools,
)


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict) -> object:
        self.calls.append((name, dict(args)))
        return types.SimpleNamespace(content=[_FakeTextContent(f"{name}:ok")])


def test_validate_browser_url_blocks_unsafe_schemes() -> None:
    good, err = _validate_browser_url("example.org")
    assert err == ""
    assert good == "https://example.org"

    _, err_file = _validate_browser_url("file:///etc/passwd")
    assert "Unsupported URL scheme" in err_file

    _, err_js = _validate_browser_url("javascript:alert(1)")
    assert "Unsupported URL scheme" in err_js


def test_mcp_browser_navigate_rejects_file_scheme() -> None:
    tool = McpBrowserNavigateTool(mcp_servers={})
    out = asyncio.run(tool.execute(url="file:///etc/passwd"))
    payload = json.loads(out)
    assert "Unsupported URL scheme" in str(payload.get("error") or "")


def test_unified_mcp_browser_tool_executes_actions(monkeypatch) -> None:
    fake_mcp = types.SimpleNamespace(
        types=types.SimpleNamespace(TextContent=_FakeTextContent)
    )
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)

    session = _FakeSession()
    closed = {"value": False}

    async def _fake_ensure(_servers):
        return session

    async def _fake_close():
        closed["value"] = True

    monkeypatch.setattr(
        "annolid.core.agent.tools.mcp_browser._ensure_mcp_browser_session",
        _fake_ensure,
    )
    monkeypatch.setattr(
        "annolid.core.agent.tools.mcp_browser._close_mcp_browser_session",
        _fake_close,
    )

    tool = McpBrowserTool(mcp_servers={})
    started = json.loads(asyncio.run(tool.execute(action="start")))
    assert started["ok"] is True

    navigated = json.loads(
        asyncio.run(tool.execute(action="navigate", url="example.org"))
    )
    assert navigated["ok"] is True
    assert navigated["url"] == "https://example.org"

    acted = json.loads(
        asyncio.run(
            tool.execute(
                action="act",
                operation="click",
                ref="e12",
                element="Submit",
            )
        )
    )
    assert acted["ok"] is True
    assert any(call[0] == "browser_click" for call in session.calls)

    stopped = json.loads(asyncio.run(tool.execute(action="stop")))
    assert stopped["ok"] is True
    assert closed["value"] is True


def test_register_mcp_browser_tools_registers_unified_tool() -> None:
    registry = FunctionToolRegistry()
    register_mcp_browser_tools(registry, mcp_servers={})
    assert registry.has("mcp_browser")
    assert registry.has("mcp_browser_navigate")
