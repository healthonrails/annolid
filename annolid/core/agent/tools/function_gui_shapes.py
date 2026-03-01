from __future__ import annotations

from typing import Any, Optional

from .function_gui_base import ActionCallback, _run_callback
from .function_base import FunctionTool


class GuiListShapesTool(FunctionTool):
    def __init__(self, list_shapes_callback: Optional[ActionCallback] = None):
        self._list_shapes_callback = list_shapes_callback

    @property
    def name(self) -> str:
        return "gui_list_shapes"

    @property
    def description(self) -> str:
        return "List current canvas shapes with optional filters (label/type/selected)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label_contains": {"type": "string"},
                "shape_type": {"type": "string"},
                "selected_only": {"type": "boolean"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._list_shapes_callback, **kwargs)


class GuiSelectShapesTool(FunctionTool):
    def __init__(self, select_shapes_callback: Optional[ActionCallback] = None):
        self._select_shapes_callback = select_shapes_callback

    @property
    def name(self) -> str:
        return "gui_select_shapes"

    @property
    def description(self) -> str:
        return (
            "Select canvas shapes by label substring and/or shape type for follow-up "
            "operations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label_contains": {"type": "string"},
                "shape_type": {"type": "string"},
                "max_select": {"type": "integer", "minimum": 1, "maximum": 200},
                "clear_existing": {"type": "boolean"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._select_shapes_callback, **kwargs)


class GuiSetSelectedShapeLabelTool(FunctionTool):
    def __init__(
        self, set_selected_shape_label_callback: Optional[ActionCallback] = None
    ):
        self._set_selected_shape_label_callback = set_selected_shape_label_callback

    @property
    def name(self) -> str:
        return "gui_set_selected_shape_label"

    @property
    def description(self) -> str:
        return "Set label for currently selected canvas shape(s)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"new_label": {"type": "string", "minLength": 1}},
            "required": ["new_label"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._set_selected_shape_label_callback, **kwargs)


class GuiDeleteSelectedShapesTool(FunctionTool):
    def __init__(
        self, delete_selected_shapes_callback: Optional[ActionCallback] = None
    ):
        self._delete_selected_shapes_callback = delete_selected_shapes_callback

    @property
    def name(self) -> str:
        return "gui_delete_selected_shapes"

    @property
    def description(self) -> str:
        return "Delete currently selected shape(s) from canvas and label list."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._delete_selected_shapes_callback)


class GuiListShapesInAnnotationTool(FunctionTool):
    def __init__(
        self, list_shapes_in_annotation_callback: Optional[ActionCallback] = None
    ):
        self._list_shapes_in_annotation_callback = list_shapes_in_annotation_callback

    @property
    def name(self) -> str:
        return "gui_list_shapes_in_annotation"

    @property
    def description(self) -> str:
        return "List shapes from a LabelMe JSON file or annotation-store NDJSON record."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "frame": {"type": "integer", "minimum": 0},
                "label_contains": {"type": "string"},
                "exact_label": {"type": "string"},
                "shape_type": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
                "include_points": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._list_shapes_in_annotation_callback, **kwargs)


class GuiRelabelShapesInAnnotationTool(FunctionTool):
    def __init__(
        self, relabel_shapes_in_annotation_callback: Optional[ActionCallback] = None
    ):
        self._relabel_shapes_in_annotation_callback = (
            relabel_shapes_in_annotation_callback
        )

    @property
    def name(self) -> str:
        return "gui_relabel_shapes_in_annotation"

    @property
    def description(self) -> str:
        return "Rename shape labels in a LabelMe JSON file or annotation-store record."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_label": {"type": "string", "minLength": 1},
                "new_label": {"type": "string", "minLength": 1},
                "frame": {"type": "integer", "minimum": 0},
                "shape_type": {"type": "string"},
                "apply_all_frames": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["path", "old_label", "new_label"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(
            self._relabel_shapes_in_annotation_callback, **kwargs
        )


class GuiDeleteShapesInAnnotationTool(FunctionTool):
    def __init__(
        self, delete_shapes_in_annotation_callback: Optional[ActionCallback] = None
    ):
        self._delete_shapes_in_annotation_callback = (
            delete_shapes_in_annotation_callback
        )

    @property
    def name(self) -> str:
        return "gui_delete_shapes_in_annotation"

    @property
    def description(self) -> str:
        return (
            "Delete shapes from a LabelMe JSON file or annotation-store record "
            "using label/type filters."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "frame": {"type": "integer", "minimum": 0},
                "label_contains": {"type": "string"},
                "exact_label": {"type": "string"},
                "shape_type": {"type": "string"},
                "max_delete": {"type": "integer", "minimum": 1, "maximum": 1000000},
                "apply_all_frames": {"type": "boolean"},
                "delete_all": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._delete_shapes_in_annotation_callback, **kwargs)
