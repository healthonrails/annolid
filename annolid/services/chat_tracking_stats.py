"""Service helpers for GUI chat tracking-stats analysis tools."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.gui_backend.tool_handlers_tracking_stats import (
    analyze_tracking_stats_tool as gui_analyze_tracking_stats_tool,
)


def analyze_chat_tracking_stats_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_analyze_tracking_stats_tool(**kwargs)


__all__ = ["analyze_chat_tracking_stats_tool"]
