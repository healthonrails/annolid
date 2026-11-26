from __future__ import annotations

from collections import OrderedDict
import difflib
import logging
import re
from typing import Any, Iterable

from .function_base import FunctionTool

logger = logging.getLogger(__name__)


_PATH_ARGUMENTS = (
    "path",
    "file_path",
    "filename",
    "output_path",
    "new_path",
    "source",
    "source_path",
    "destination",
    "dest_path",
    "target",
    "target_path",
    "repo_path",
    "dataset_folder",
    "workspace",
    "directory",
    "dir_path",
    "video_path",
    "bib_file",
    "working_dir",
    "workdir",
)
_ABS_OR_HOME_PATH_RE = re.compile(r"(?P<path>(?:/|~/)(?:[^\s\"'`;|<>])+)")
_WORKSPACE_VIOLATION_MARKERS = (
    "outside allowed directory",
    "outside allowed read roots",
    "outside the configured workspace",
    "path outside working dir",
    "hard policy boundary",
)
_MAX_WORKSPACE_VIOLATION_SIGNATURES = 256


def _is_abs_or_home_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("~")


def _workspace_violation_signature(
    tool_name: str,
    params: dict[str, Any],
) -> str | None:
    for key in _PATH_ARGUMENTS:
        value = str(params.get(key) or "").strip()
        if _is_abs_or_home_path(value):
            return _normalize_signature_path(value)

    if tool_name in {"exec", "exec_start"}:
        command = str(params.get("command") or params.get("cmd") or "")
        match = _ABS_OR_HOME_PATH_RE.search(command)
        if match:
            return _normalize_signature_path(match.group("path"))
        working_dir = str(params.get("working_dir") or params.get("workdir") or "")
        if _is_abs_or_home_path(working_dir):
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
        self._aliases: dict[str, str] = {}
        self._workspace_violation_counts: OrderedDict[str, int] = OrderedDict()

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        return str(name or "").strip()

    def _resolve_tool_name(self, name: str) -> tuple[str, bool]:
        normalized = self._normalize_tool_name(name)
        if not normalized:
            return "", False
        if normalized in self._tools:
            return normalized, False
        mapped = self._aliases.get(normalized)
        if mapped and mapped in self._tools:
            return mapped, True
        return normalized, False

    def canonical_name(self, name: str) -> str:
        """Resolve an alias to its canonical registered name, or return normalized name."""
        canonical, _ = self._resolve_tool_name(name)
        return canonical

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
            for alias, target in list(self._aliases.items()):
                if target == name:
                    self._aliases.pop(alias, None)
        self._tools[name] = tool
        # If a previous alias shared this canonical name, canonical wins.
        self._aliases.pop(name, None)
        raw_aliases = getattr(tool, "aliases", ()) or ()
        for raw_alias in raw_aliases:
            alias = self._normalize_tool_name(str(raw_alias))
            if not alias or alias == name:
                continue
            if alias in self._tools:
                continue
            if alias in self._aliases and not replace:
                continue
            self._aliases[alias] = name
        return True

    def unregister(self, name: str) -> None:
        canonical, _ = self._resolve_tool_name(name)
        if not canonical:
            return
        self._tools.pop(canonical, None)
        for alias, target in list(self._aliases.items()):
            if target == canonical:
                self._aliases.pop(alias, None)

    def get(self, name: str) -> FunctionTool | None:
        canonical, _ = self._resolve_tool_name(name)
        return self._tools.get(canonical)

    def has(self, name: str) -> bool:
        canonical, _ = self._resolve_tool_name(name)
        return canonical in self._tools

    def values(self) -> list[FunctionTool]:
        return list(self._tools.values())

    def reset_violation_counts(self) -> None:
        self._workspace_violation_counts.clear()

    def get_definitions(
        self,
        *,
        allowed_names: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if allowed_names is None:
            return [tool.to_schema() for tool in self._tools.values()]
        allowed = {
            self.canonical_name(str(name))
            for name in allowed_names
            if str(name or "").strip()
        }
        return [
            tool.to_schema() for name, tool in self._tools.items() if name in allowed
        ]

    def _record_workspace_violation(self, signature: str) -> int:
        count = self._workspace_violation_counts.get(signature, 0) + 1
        self._workspace_violation_counts[signature] = count
        self._workspace_violation_counts.move_to_end(signature)
        while (
            len(self._workspace_violation_counts) > _MAX_WORKSPACE_VIOLATION_SIGNATURES
        ):
            self._workspace_violation_counts.popitem(last=False)
        return count

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        normalized_name = self._normalize_tool_name(name)
        canonical_name, via_alias = self._resolve_tool_name(normalized_name)
        tool = self._tools.get(canonical_name)
        if tool is None:
            candidates = list(self._tools.keys()) + list(self._aliases.keys())
            suggestions = (
                difflib.get_close_matches(normalized_name, candidates, n=3, cutoff=0.6)
                if normalized_name and candidates
                else []
            )
            hint = f". Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            return f"Error: Tool '{name}' not found{hint}"
        if via_alias:
            logger.warning(
                "Deprecated tool alias '%s' invoked; use '%s' instead.",
                normalized_name,
                canonical_name,
            )

        if not isinstance(params, dict):
            return (
                f"Error: Invalid parameters for tool '{name}': "
                f"parameters must be a JSON object, got {type(params).__name__}"
            )

        raw_schema = getattr(tool, "parameters", None)
        schema_props = (
            raw_schema.get("properties", {}) if isinstance(raw_schema, dict) else {}
        )
        if (
            "_raw" in params
            and isinstance(schema_props, dict)
            and "_raw" not in schema_props
        ):
            raw_preview = str(params.get("_raw") or "")[:160]
            return (
                f"Error: Invalid JSON arguments for tool '{name}': "
                f"could not parse arguments into a JSON object (raw={raw_preview!r}). "
                "Retry with a valid JSON object matching the tool schema."
            )

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(
                    errors
                )
            result = await tool.execute(**params)
            signature = _workspace_violation_signature(canonical_name, params)
            if signature and _is_workspace_violation_result(result):
                count = self._record_workspace_violation(signature)
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
        canonical, _ = self._resolve_tool_name(name)
        return canonical in self._tools
