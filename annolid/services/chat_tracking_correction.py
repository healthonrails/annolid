"""Service helpers for GUI chat tracking-correction tools."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.gui_backend.tool_handlers_tracking_correction import (
    correct_tracking_ndjson_tool as gui_correct_tracking_ndjson_tool,
)


def correct_chat_tracking_ndjson_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_correct_tracking_ndjson_tool(**kwargs)


__all__ = ["correct_chat_tracking_ndjson_tool"]
