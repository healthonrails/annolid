"""MCP Browser tools using Playwright MCP server for standalone browser automation.

These tools provide browser automation capabilities independent of the GUI,
using the MCP protocol to connect to a Playwright MCP server.
"""

import json
from typing import Any, Dict, Optional

from annolid.utils.logger import logger

from .function_base import FunctionTool
from .function_registry import FunctionToolRegistry

# Global session and browser state for MCP browser tools
_mcp_browser_session: Optional[Any] = None
_mcp_browser_stack: Optional[Any] = None


async def _ensure_mcp_browser_session(mcp_servers: Dict[str, Any]) -> Optional[Any]:
    """Ensure MCP browser session is connected."""
    global _mcp_browser_session, _mcp_browser_stack

    if _mcp_browser_session is not None:
        return _mcp_browser_session

    # Look for Playwright MCP server configuration
    playwright_cfg = None
    for name, cfg in mcp_servers.items():
        if "playwright" in name.lower():
            playwright_cfg = cfg
            break

    if playwright_cfg is None:
        logger.warning("No Playwright MCP server configured")
        return None

    try:
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # Handle config as dict or object
        if hasattr(playwright_cfg, "to_dict"):
            c = playwright_cfg
        else:
            from annolid.core.agent.config.schema import MCPServerConfig

            c = MCPServerConfig.from_dict(playwright_cfg)

        if c.command:
            params = StdioServerParameters(
                command=c.command, args=c.args, env=c.env or None
            )
            _mcp_browser_stack = AsyncExitStack()
            read, write = await _mcp_browser_stack.enter_async_context(
                stdio_client(params)
            )
        elif c.url:
            from mcp.client.streamable_http import streamable_http_client

            _mcp_browser_stack = AsyncExitStack()
            read, write, _ = await _mcp_browser_stack.enter_async_context(
                streamable_http_client(c.url)
            )
        else:
            logger.warning("MCP browser: no command or url configured")
            return None

        _mcp_browser_session = await _mcp_browser_stack.enter_async_context(
            ClientSession(read, write)
        )
        await _mcp_browser_session.initialize()
        logger.info("MCP browser session connected")
        return _mcp_browser_session

    except ImportError:
        logger.warning("mcp package not installed")
        return None
    except Exception as e:
        logger.error("Failed to connect to MCP browser: %s", e)
        return None


async def _close_mcp_browser_session() -> None:
    """Close the MCP browser session."""
    global _mcp_browser_session, _mcp_browser_stack

    if _mcp_browser_stack is not None:
        try:
            await _mcp_browser_stack.aclose()
        except Exception:
            pass
    _mcp_browser_session = None
    _mcp_browser_stack = None


class McpBrowserNavigateTool(FunctionTool):
    """Navigate to a URL using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_navigate"

    @property
    def description(self) -> str:
        return "Navigate to a URL in the browser using Playwright MCP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url", "")
        if not url:
            return json.dumps({"error": "url is required"})

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            result = await session.call_tool("browser_navigate", {"url": url})
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return json.dumps({"ok": True, "message": "\n".join(parts)})
            return json.dumps({"ok": True, "url": url})
        except Exception as e:
            return json.dumps({"error": str(e)})


class McpBrowserClickTool(FunctionTool):
    """Click an element using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_click"

    @property
    def description(self) -> str:
        return "Click an element on the page using Playwright MCP. Requires a page snapshot first."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Element reference from page snapshot",
                },
                "element": {
                    "type": "string",
                    "description": "Human-readable element description",
                },
            },
            "required": ["ref"],
        }

    async def execute(self, **kwargs: Any) -> str:
        ref = kwargs.get("ref", "")
        element = kwargs.get("element", "")

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            result = await session.call_tool(
                "browser_click", {"ref": ref, "element": element}
            )
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return json.dumps({"ok": True, "message": "\n".join(parts)})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})


class McpBrowserTypeTool(FunctionTool):
    """Type text into an element using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_type"

    @property
    def description(self) -> str:
        return "Type text into an input element using Playwright MCP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Element reference from page snapshot",
                },
                "text": {"type": "string", "description": "Text to type"},
                "submit": {
                    "type": "boolean",
                    "description": "Whether to submit after typing",
                },
                "slowly": {
                    "type": "boolean",
                    "description": "Type one character at a time",
                },
            },
            "required": ["ref", "text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        ref = kwargs.get("ref", "")
        text = kwargs.get("text", "")
        submit = kwargs.get("submit", False)
        slowly = kwargs.get("slowly", False)

        if not ref or not text:
            return json.dumps({"error": "ref and text are required"})

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            result = await session.call_tool(
                "browser_type",
                {
                    "ref": ref,
                    "text": text,
                    "submit": submit,
                    "slowly": slowly,
                },
            )
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return json.dumps({"ok": True, "message": "\n".join(parts)})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})


class McpBrowserSnapshotTool(FunctionTool):
    """Get a snapshot of the current page using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_snapshot"

    @property
    def description(self) -> str:
        return "Get an accessibility snapshot of the current page using Playwright MCP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Optional filename to save snapshot",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        filename = kwargs.get("filename", "")

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            args = {}
            if filename:
                args["filename"] = filename

            result = await session.call_tool("browser_snapshot", args)
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return "\n".join(parts)
            return "(no content)"
        except Exception as e:
            return json.dumps({"error": str(e)})


class McpBrowserScreenshotTool(FunctionTool):
    """Take a screenshot using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_screenshot"

    @property
    def description(self) -> str:
        return "Take a screenshot of the current page using Playwright MCP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["png", "jpeg"], "default": "png"},
                "filename": {
                    "type": "string",
                    "description": "Optional filename to save screenshot",
                },
                "fullPage": {"type": "boolean", "description": "Capture full page"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        screenshot_type = kwargs.get("type", "png")
        filename = kwargs.get("filename", "")
        full_page = kwargs.get("fullPage", False)

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            result = await session.call_tool(
                "browser_take_screenshot",
                {
                    "type": screenshot_type,
                    "filename": filename,
                    "fullPage": full_page,
                },
            )
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return json.dumps({"ok": True, "message": "\n".join(parts)})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})


class McpBrowserScrollTool(FunctionTool):
    """Scroll the page using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_scroll"

    @property
    def description(self) -> str:
        return "Scroll the page using Playwright MCP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "delta_y": {
                    "type": "integer",
                    "description": "Vertical scroll delta in pixels",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        # MCP browser uses press_key for scrolling
        _ = kwargs.get("delta_y", 800)

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            # Use PageDown key for scrolling
            result = await session.call_tool("browser_press_key", {"key": "PageDown"})
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return json.dumps({"ok": True, "message": "\n".join(parts)})
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"error": str(e)})


class McpBrowserCloseTool(FunctionTool):
    """Close the browser using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_close"

    @property
    def description(self) -> str:
        return "Close the browser using Playwright MCP."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        await _close_mcp_browser_session()
        return json.dumps({"ok": True, "message": "Browser closed"})


class McpBrowserWaitTool(FunctionTool):
    """Wait for text or time using Playwright MCP."""

    def __init__(self, mcp_servers: Optional[Dict[str, Any]] = None):
        self._mcp_servers = mcp_servers or {}

    @property
    def name(self) -> str:
        return "mcp_browser_wait"

    @property
    def description(self) -> str:
        return "Wait for text to appear or disappear, or wait for a specified time."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to wait for"},
                "textGone": {
                    "type": "string",
                    "description": "Text to wait for to disappear",
                },
                "time": {"type": "integer", "description": "Time to wait in seconds"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        text = kwargs.get("text", "")
        text_gone = kwargs.get("textGone", "")
        wait_time = kwargs.get("time", 0)

        session = await _ensure_mcp_browser_session(self._mcp_servers)
        if session is None:
            return json.dumps({"error": "MCP browser session not available"})

        try:
            from mcp import types

            args = {}
            if text:
                args["text"] = text
            if text_gone:
                args["textGone"] = text_gone
            if wait_time:
                args["time"] = wait_time

            result = await session.call_tool("browser_wait_for", args)
            parts = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    parts.append(block.text)

            if parts:
                return "\n".join(parts)
            return "(no content)"
        except Exception as e:
            return json.dumps({"error": str(e)})


def register_mcp_browser_tools(
    registry: FunctionToolRegistry,
    mcp_servers: Optional[Dict[str, Any]] = None,
) -> None:
    """Register MCP browser tools for standalone browser automation.

    These tools connect to a Playwright MCP server and provide browser
    automation capabilities without requiring the GUI to be running.
    """
    registry.register(McpBrowserNavigateTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserClickTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserTypeTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserSnapshotTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserScreenshotTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserScrollTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserCloseTool(mcp_servers=mcp_servers))
    registry.register(McpBrowserWaitTool(mcp_servers=mcp_servers))
    logger.info("Registered MCP browser tools")
