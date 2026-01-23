from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from .base import Tool


ToolFactory = Callable[[Dict[str, object]], Tool[Any, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tool_id: str
    config: Dict[str, object] = field(default_factory=dict)


class ToolRegistry:
    """Registry for pluggable tools (Phase 3)."""

    def __init__(self) -> None:
        self._factories: Dict[str, ToolFactory] = {}

    def register(self, tool_id: str, factory: ToolFactory) -> None:
        if not tool_id:
            raise ValueError("tool_id must be non-empty.")
        self._factories[tool_id] = factory

    def has(self, tool_id: str) -> bool:
        return tool_id in self._factories

    def create(
        self, tool_id: str, config: Optional[Dict[str, object]] = None
    ) -> Tool[Any, Any]:
        if tool_id not in self._factories:
            raise KeyError(f"Tool not registered: {tool_id}")
        return self._factories[tool_id](dict(config or {}))

    def available(self) -> Sequence[str]:
        return sorted(self._factories.keys())


def build_pipeline(
    registry: ToolRegistry,
    specs: Iterable[Dict[str, object]],
) -> Sequence[Tool[Any, Any]]:
    tools: list[Tool[Any, Any]] = []
    for spec in specs:
        tool_id = str(spec.get("tool") or spec.get("tool_id") or "")
        if not tool_id:
            raise ValueError("Tool spec missing tool_id.")
        config = spec.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("Tool spec 'config' must be a dict.")
        tools.append(registry.create(tool_id, dict(config)))
    return tools
