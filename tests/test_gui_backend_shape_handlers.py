from __future__ import annotations

from annolid.core.agent.gui_backend.tool_handlers_shapes import (
    delete_selected_shapes_tool,
    list_shapes_tool,
    select_shapes_tool,
    set_selected_shape_label_tool,
)


def test_list_shapes_tool_validates_shape_type() -> None:
    payload = list_shapes_tool(
        shape_type="triangle",
        invoke_widget_json_slot=lambda *args: {"ok": True},
    )
    assert payload["ok"] is False
    assert "Unsupported shape_type" in str(payload.get("error") or "")


def test_select_shapes_tool_requires_filter() -> None:
    payload = select_shapes_tool(
        invoke_widget_json_slot=lambda *args: {"ok": True},
    )
    assert payload["ok"] is False
    assert "Provide label_contains or shape_type" in str(payload.get("error") or "")


def test_shape_tools_queue_expected_slots() -> None:
    calls: list[str] = []

    def _invoke(slot_name: str, *args):
        calls.append(slot_name)
        return {"ok": True, "slot": slot_name, "args_count": len(args)}

    list_payload = list_shapes_tool(
        label_contains="nose",
        shape_type="point",
        selected_only=True,
        max_results=3,
        invoke_widget_json_slot=_invoke,
    )
    assert list_payload["ok"] is True
    assert list_payload["slot"] == "bot_list_shapes"

    select_payload = select_shapes_tool(
        label_contains="mouse",
        shape_type="polygon",
        max_select=4,
        clear_existing=False,
        invoke_widget_json_slot=_invoke,
    )
    assert select_payload["ok"] is True
    assert select_payload["slot"] == "bot_select_shapes"

    relabel_payload = set_selected_shape_label_tool(
        new_label="animal",
        invoke_widget_json_slot=_invoke,
    )
    assert relabel_payload["ok"] is True
    assert relabel_payload["slot"] == "bot_set_selected_shape_label"

    delete_payload = delete_selected_shapes_tool(invoke_widget_json_slot=_invoke)
    assert delete_payload["ok"] is True
    assert delete_payload["slot"] == "bot_delete_selected_shapes"

    assert calls == [
        "bot_list_shapes",
        "bot_select_shapes",
        "bot_set_selected_shape_label",
        "bot_delete_selected_shapes",
    ]
