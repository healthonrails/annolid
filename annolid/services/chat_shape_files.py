"""Service helpers for GUI chat annotation shape-file actions."""

from __future__ import annotations

from annolid.core.agent.gui_backend.tool_handlers_shape_files import (
    delete_shapes_in_annotation_tool,
    list_shapes_in_annotation_tool,
    relabel_shapes_in_annotation_tool,
)

__all__ = [
    "delete_shapes_in_annotation_tool",
    "list_shapes_in_annotation_tool",
    "relabel_shapes_in_annotation_tool",
]
