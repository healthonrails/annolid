"""MCP client: connects to MCP servers and wraps their tools as native annolid tools."""

import asyncio
import hashlib
import json
import re
from contextlib import AsyncExitStack
from typing import Any, Dict
from urllib.parse import urlsplit

from annolid.utils.logger import logger

from .function_base import FunctionTool
from .function_registry import FunctionToolRegistry
from .resolve import ensure_node_env, resolve_command

_DEFAULT_MCP_CONNECT_TIMEOUT_S = 30.0
_MCP_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def _validate_mcp_server_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return "empty URL"
    parsed = urlsplit(value)
    scheme = str(parsed.scheme or "").strip().lower()
    if scheme not in _MCP_ALLOWED_URL_SCHEMES:
        return f"unsupported URL scheme {scheme or '(none)'!r}"
    if not str(parsed.netloc or "").strip():
        return "URL must include a hostname"
    return ""


def _format_mcp_content_block(block: Any, types_module: Any) -> str:
    text_type = getattr(types_module, "TextContent", None)
    if text_type is not None and isinstance(block, text_type):
        return str(getattr(block, "text", "") or "")
    image_type = getattr(types_module, "ImageContent", None)
    block_type = str(getattr(block, "type", "") or "").strip().lower()
    if (
        image_type is not None and isinstance(block, image_type)
    ) or block_type == "image":
        mime = str(getattr(block, "mimeType", "unknown") or "unknown")
        data_len = len(str(getattr(block, "data", "") or ""))
        return f"[Image content: mimeType={mime}, base64_chars={data_len}]"
    resource_type = getattr(types_module, "EmbeddedResource", None)
    if (
        resource_type is not None and isinstance(block, resource_type)
    ) or block_type == "resource":
        resource = getattr(block, "resource", None)
        uri = str(getattr(resource, "uri", "") or "")
        text = getattr(resource, "text", None)
        if isinstance(text, str) and text.strip():
            return f"[Resource {uri}]\n{text}" if uri else text
        blob = getattr(resource, "blob", None)
        if blob:
            return (
                f"[Binary resource {uri or '(unnamed)'}: base64_chars={len(str(blob))}]"
            )
        return f"[Resource {uri or '(empty)'}]"
    rendered = str(block)
    if len(rendered) > 2000:
        return rendered[:2000] + "... [truncated non-text MCP block]"
    return rendered


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
        is_error = bool(getattr(result, "isError", False))
        parts = []
        for block in getattr(result, "content", []) or []:
            formatted = _format_mcp_content_block(block, types)
            if formatted:
                parts.append(formatted)

        def _with_error_prefix(payload_text: str) -> str:
            if not is_error:
                return payload_text
            if payload_text.lower().startswith("error:"):
                return payload_text
            return f"Error: {payload_text}"

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
            return _with_error_prefix(text_payload)

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
                return _with_error_prefix(struct_text)
            except Exception:
                return _with_error_prefix(str(structured))

        if is_error:
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
    digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{base[:55]}_{digest}"[:64]


async def connect_mcp_servers(
    mcp_servers: Dict[str, Any],
    registry: FunctionToolRegistry,
    stack: AsyncExitStack,
    *,
    connect_timeout_seconds: float = _DEFAULT_MCP_CONNECT_TIMEOUT_S,
) -> None:
    """Connect to configured MCP servers and register their tools."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.warning("mcp package not installed, skipping MCP servers")
        return

    timeout_s = max(
        1.0, float(connect_timeout_seconds or _DEFAULT_MCP_CONNECT_TIMEOUT_S)
    )

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
                resolved_cmd = resolve_command(c.command)
                if resolved_cmd is None:
                    logger.warning(
                        "MCP server '%s': command '%s' not found on PATH, skipping",
                        name,
                        c.command,
                    )
                    continue

                # Combine custom env with augmented node env
                server_env = ensure_node_env()
                if c.env:
                    server_env.update(c.env)

                params = StdioServerParameters(
                    command=resolved_cmd, args=c.args, env=server_env
                )
                read, write = await asyncio.wait_for(
                    stack.enter_async_context(stdio_client(params)),
                    timeout=timeout_s,
                )
            elif c.url:
                url_error = _validate_mcp_server_url(c.url)
                if url_error:
                    logger.warning(
                        "MCP server '%s': invalid URL (%s), skipping",
                        name,
                        url_error,
                    )
                    continue
                from mcp.client.streamable_http import streamable_http_client

                read, write, _ = await asyncio.wait_for(
                    stack.enter_async_context(streamable_http_client(c.url)),
                    timeout=timeout_s,
                )
            else:
                logger.warning(
                    "MCP server '%s': no command or url configured, skipping", name
                )
                continue

            session = await asyncio.wait_for(
                stack.enter_async_context(ClientSession(read, write)),
                timeout=timeout_s,
            )
            await asyncio.wait_for(session.initialize(), timeout=timeout_s)

            tools = await asyncio.wait_for(session.list_tools(), timeout=timeout_s)
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
