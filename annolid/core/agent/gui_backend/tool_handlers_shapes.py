from __future__ import annotations

from typing import Any, Callable, Dict


_ALLOWED_SHAPE_TYPES = {
    "polygon",
    "rectangle",
    "circle",
    "line",
    "point",
    "linestrip",
    "mask",
}


def _normalize_shape_type(shape_type: str) -> str:
    return str(shape_type or "").strip().lower()


def list_shapes_tool(
    *,
    label_contains: str = "",
    shape_type: str = "",
    selected_only: bool = False,
    max_results: int = 200,
    invoke_widget_json_slot: Callable[[str, str, str, bool, int], Dict[str, Any]],
) -> Dict[str, Any]:
    type_text = _normalize_shape_type(shape_type)
    if type_text and type_text not in _ALLOWED_SHAPE_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported shape_type: {shape_type}",
            "allowed_shape_types": sorted(_ALLOWED_SHAPE_TYPES),
        }
    limit = max(1, min(int(max_results), 500))
    return invoke_widget_json_slot(
        "bot_list_shapes",
        str(label_contains or ""),
        type_text,
        bool(selected_only),
        limit,
    )


def select_shapes_tool(
    *,
    label_contains: str = "",
    shape_type: str = "",
    max_select: int = 20,
    clear_existing: bool = True,
    invoke_widget_json_slot: Callable[[str, str, str, int, bool], Dict[str, Any]],
) -> Dict[str, Any]:
    text_filter = str(label_contains or "").strip()
    type_text = _normalize_shape_type(shape_type)
    if not text_filter and not type_text:
        return {"ok": False, "error": "Provide label_contains or shape_type."}
    if type_text and type_text not in _ALLOWED_SHAPE_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported shape_type: {shape_type}",
            "allowed_shape_types": sorted(_ALLOWED_SHAPE_TYPES),
        }
    limit = max(1, min(int(max_select), 200))
    return invoke_widget_json_slot(
        "bot_select_shapes",
        text_filter,
        type_text,
        limit,
        bool(clear_existing),
    )


def set_selected_shape_label_tool(
    *,
    new_label: str,
    invoke_widget_json_slot: Callable[[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    label_text = str(new_label or "").strip()
    if not label_text:
        return {"ok": False, "error": "new_label is required"}
    return invoke_widget_json_slot("bot_set_selected_shape_label", label_text)


def delete_selected_shapes_tool(
    *,
    invoke_widget_json_slot: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    return invoke_widget_json_slot("bot_delete_selected_shapes")
