"""MCP client: connects to MCP servers and wraps their tools as native annolid tools."""

import hashlib
import json
import re
from contextlib import AsyncExitStack
from typing import Any, Dict

from annolid.utils.logger import logger

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
            text_payload = "\n".join(parts).strip()
            # If the payload looks like massive HTML, strip it to save tokens
            if len(text_payload) > 20000 and (
                "<html" in text_payload.lower() or "<body" in text_payload.lower()
            ):
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(text_payload, "html.parser")
                    text_payload = soup.get_text(separator="\n", strip=True)
                except ImportError:
                    from .common import _normalize, _strip_tags

                    text_payload = _normalize(_strip_tags(text_payload))

            # Cap extremely long outputs
            if len(text_payload) > 50000:
                text_payload = (
                    text_payload[:50000]
                    + "\n\n[WARNING: MCP tool response was truncated because it exceeded 50,000 characters. "
                    "Please refine your query to return more specific data (e.g., using document.body.innerText "
                    "rather than full HTML elements) to avoid context limit errors.]"
                )
            return text_payload

        def _truncate_json(data: Any, max_len: int = 50) -> Any:
            if isinstance(data, list):
                if len(data) > max_len:
                    return [_truncate_json(x, max_len) for x in data[:max_len]] + [
                        f"... ({len(data) - max_len} more items truncated)"
                    ]
                return [_truncate_json(x, max_len) for x in data]
            elif isinstance(data, dict):
                return {k: _truncate_json(v, max_len) for k, v in data.items()}
            return data

        structured = getattr(result, "structuredContent", None)
        if hasattr(result, "content") and not parts and not structured:
            # Some MCP servers pass JSON evaluation data in a non-text block, try grabbing raw dictionary from content if possible
            try:
                for block in getattr(result, "content", []) or []:
                    if hasattr(block, "model_dump"):
                        structured = block.model_dump()
                        break
            except Exception:
                pass

        if structured is not None:
            try:
                struct_data = _truncate_json(structured, max_len=50)
                struct_text = json.dumps(struct_data, ensure_ascii=False, indent=2)
                if len(struct_text) > 40000:
                    struct_data = _truncate_json(structured, max_len=15)
                    struct_text = json.dumps(struct_data, ensure_ascii=False, indent=2)
                return struct_text
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
                    "MCP server '%s': no command or url configured, skipping", name
                )
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def)
                registry.register(wrapper)
                logger.debug(
                    "MCP: registered tool '%s' from server '%s'", wrapper.name, name
                )

            logger.info(
                "MCP server '%s': connected, %d tools registered",
                name,
                len(tools.tools),
            )
        except Exception as e:
            logger.error("MCP server '%s': failed to connect: %s", name, e)
