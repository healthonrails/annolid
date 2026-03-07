"""Service helpers for GUI chat video actions and workflows."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.gui_backend.tool_handlers_video import (
    open_video_tool as gui_open_video_tool,
    resolve_video_path_for_gui_tool as gui_resolve_video_path_for_gui_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_video_workflow import (
    label_behavior_segments_tool as gui_label_behavior_segments_tool,
    segment_track_video_tool as gui_segment_track_video_tool,
)


def open_chat_video_tool(path: str, **kwargs: Any) -> dict[str, object]:
    return gui_open_video_tool(path, **kwargs)


def resolve_chat_video_path_for_gui_tool(raw_path: str, **kwargs: Any):
    return gui_resolve_video_path_for_gui_tool(raw_path, **kwargs)


def segment_track_chat_video_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_segment_track_video_tool(**kwargs)


def label_chat_behavior_segments_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_label_behavior_segments_tool(**kwargs)


__all__ = [
    "label_chat_behavior_segments_tool",
    "open_chat_video_tool",
    "resolve_chat_video_path_for_gui_tool",
    "segment_track_chat_video_tool",
]
