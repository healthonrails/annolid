"""Service helpers for GUI chat video actions and workflows."""

from __future__ import annotations

import json
from typing import Any

from annolid.core.agent.gui_backend.tool_handlers_video import (
    open_video_tool as gui_open_video_tool,
    resolve_video_path_for_gui_tool as gui_resolve_video_path_for_gui_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_video_workflow import (
    behavior_catalog_tool as gui_behavior_catalog_tool,
    label_behavior_segments_tool as gui_label_behavior_segments_tool,
    process_video_behaviors_tool as gui_process_video_behaviors_tool,
    segment_track_video_tool as gui_segment_track_video_tool,
)
from annolid.core.agent.gui_backend.direct_commands import run_awaitable_sync
from annolid.core.agent.tools.function_sam3 import Sam3AgentVideoTrackTool


def open_chat_video_tool(path: str, **kwargs: Any) -> dict[str, object]:
    return gui_open_video_tool(path, **kwargs)


def resolve_chat_video_path_for_gui_tool(raw_path: str, **kwargs: Any):
    return gui_resolve_video_path_for_gui_tool(raw_path, **kwargs)


def segment_track_chat_video_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_segment_track_video_tool(**kwargs)


def sam3_agent_video_track_tool(
    *,
    allowed_dir=None,
    allowed_read_roots=None,
    **kwargs: Any,
) -> dict[str, Any]:
    tool = Sam3AgentVideoTrackTool(
        allowed_dir=allowed_dir,
        allowed_read_roots=allowed_read_roots,
    )
    result = run_awaitable_sync(tool.execute(**kwargs))
    if isinstance(result, dict):
        return result
    try:
        parsed = json.loads(str(result or ""))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def label_chat_behavior_segments_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_label_behavior_segments_tool(**kwargs)


def behavior_catalog_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_behavior_catalog_tool(**kwargs)


def process_chat_video_behaviors_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_process_video_behaviors_tool(**kwargs)


__all__ = [
    "behavior_catalog_tool",
    "label_chat_behavior_segments_tool",
    "open_chat_video_tool",
    "process_chat_video_behaviors_tool",
    "resolve_chat_video_path_for_gui_tool",
    "sam3_agent_video_track_tool",
    "segment_track_chat_video_tool",
]
