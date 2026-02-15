"""MCP client: connects to MCP servers and wraps their tools as native annolid tools."""

import hashlib
import json
import re
from contextlib import AsyncExitStack
from typing import Any, Dict

from loguru import logger

from .function_base import FunctionTool
from .function_registry import FunctionToolRegistry


class MCPToolWrapper(FunctionTool):
    """Wraps a single MCP server tool as an annolid FunctionTool."""

    def __init__(self, session, server_name: str, tool_def):
        self._session = session
        self._original_name = tool_def.name
        self._name = _build_wrapper_tool_name(server_name, tool_def.name)
        self._description = tool_def.description or tool_def.name
        raw_schema = tool_def.inputSchema
        if isinstance(raw_schema, dict):
            self._parameters = dict(raw_schema)
            self._parameters.setdefault("type", "object")
            self._parameters.setdefault("properties", {})
        else:
            self._parameters = {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types

        result = await self._session.call_tool(self._original_name, arguments=kwargs)
        parts = []
        for block in getattr(result, "content", []) or []:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))

        if parts:
            return "\n".join(parts).strip()

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                return json.dumps(structured, ensure_ascii=False)
            except Exception:
                return str(structured)

        if getattr(result, "isError", False):
            return "Error: MCP tool call returned an error with no text payload."
        return "(no output)"


def _sanitize_tool_name_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "tool"


def _build_wrapper_tool_name(server_name: str, tool_name: str) -> str:
    base = f"mcp_{_sanitize_tool_name_part(server_name)}_{_sanitize_tool_name_part(tool_name)}"
    if len(base) <= 64:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{digest}"[:64]


async def connect_mcp_servers(
    mcp_servers: Dict[str, Any], registry: FunctionToolRegistry, stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.warning("mcp package not installed, skipping MCP servers")
        return

    for name, cfg in mcp_servers.items():
        try:
            # cfg can be MCPServerConfig object or a dict from config
            if hasattr(cfg, "to_dict"):
                c = cfg
            else:
                # Handle raw dict if passed
                from annolid.core.agent.config.schema import MCPServerConfig

                c = MCPServerConfig.from_dict(cfg)

            if c.command:
                params = StdioServerParameters(
                    command=c.command, args=c.args, env=c.env or None
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif c.url:
                from mcp.client.streamable_http import streamable_http_client

                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(c.url)
                )
            else:
                logger.warning(
                    f"MCP server '{name}': no command or url configured, skipping"
                )
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def)
                registry.register(wrapper)
                logger.debug(
                    f"MCP: registered tool '{wrapper.name}' from server '{name}'"
                )

            logger.info(
                "MCP server '{}': connected, {} tools registered",
                name,
                len(tools.tools),
            )
        except Exception as e:
            logger.error("MCP server '{}': failed to connect: {}", name, e)
