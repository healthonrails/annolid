from __future__ import annotations

import logging
import re
from typing import Any

from .function_base import FunctionTool

logger = logging.getLogger(__name__)


_PATH_ARGUMENTS = ("path", "file_path", "filename", "output_path", "new_path")
_ABS_PATH_RE = re.compile(r"(?P<path>/(?:[^\s\"'`;|<>])+)")
_WORKSPACE_VIOLATION_MARKERS = (
    "outside allowed directory",
    "outside allowed read roots",
    "outside the configured workspace",
    "path outside working dir",
    "hard policy boundary",
)


def _workspace_violation_signature(
    tool_name: str,
    params: dict[str, Any],
) -> str | None:
    for key in _PATH_ARGUMENTS:
        value = str(params.get(key) or "").strip()
        if value.startswith("/"):
            return _normalize_signature_path(value)

    if tool_name in {"exec", "exec_start"}:
        command = str(params.get("command") or params.get("cmd") or "")
        match = _ABS_PATH_RE.search(command)
        if match:
            return _normalize_signature_path(match.group("path"))
        working_dir = str(params.get("working_dir") or params.get("workdir") or "")
        if working_dir.startswith("/"):
            return _normalize_signature_path(working_dir)
    return None


def _normalize_signature_path(path: str) -> str:
    return str(path or "").strip().rstrip(".,:;)").lower()


def _is_workspace_violation_result(result: object) -> bool:
    text = str(result or "").lower()
    return any(marker in text for marker in _WORKSPACE_VIOLATION_MARKERS)


class FunctionToolRegistry:
    """Nanobot-style dynamic registry for function-call tools."""

    def __init__(self) -> None:
        self._tools: dict[str, FunctionTool] = {}
        self._workspace_violation_counts: dict[str, int] = {}

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        return str(name or "").strip()

    def register(self, tool: FunctionTool, *, replace: bool = False) -> bool:
        name = self._normalize_tool_name(tool.name)
        if not name:
            raise ValueError("Tool name must be a non-empty string.")
        existing = self._tools.get(name)
        if existing is not None and not replace:
            logger.warning(
                "Tool name collision: %s from %s skipped; existing %s kept.",
                name,
                type(tool).__name__,
                type(existing).__name__,
            )
            return False
        if existing is not None:
            logger.warning(
                "Tool name collision: %s from %s replaces existing %s.",
                name,
                type(tool).__name__,
                type(existing).__name__,
            )
        self._tools[name] = tool
        return True

    def unregister(self, name: str) -> None:
        self._tools.pop(self._normalize_tool_name(name), None)

    def get(self, name: str) -> FunctionTool | None:
        return self._tools.get(self._normalize_tool_name(name))

    def has(self, name: str) -> bool:
        return self._normalize_tool_name(name) in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        normalized_name = self._normalize_tool_name(name)
        tool = self._tools.get(normalized_name)
        if tool is None:
            return f"Error: Tool '{name}' not found"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(
                    errors
                )
            result = await tool.execute(**params)
            signature = _workspace_violation_signature(normalized_name, params)
            if signature and _is_workspace_violation_result(result):
                count = self._workspace_violation_counts.get(signature, 0) + 1
                self._workspace_violation_counts[signature] = count
                if count >= 3:
                    return (
                        "Error: refusing repeated workspace-bypass attempts for "
                        f"{signature}. This is a hard policy boundary; ask how "
                        "the user wants to proceed instead of retrying with "
                        "alternate tools or shell tricks."
                    )
            return result
        except Exception as exc:
            return f"Error executing {name}: {exc}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self._normalize_tool_name(name) in self._tools
