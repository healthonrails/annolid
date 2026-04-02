from __future__ import annotations

import asyncio

from annolid.services.chat_widget_bridge import (
    build_chat_gui_context_payload,
    execute_chat_direct_gui_command,
    get_chat_widget_action_result,
    invoke_chat_widget_json_slot,
    invoke_chat_widget_slot,
    run_chat_awaitable_sync,
)


def test_chat_widget_bridge_wrappers(monkeypatch) -> None:
    import annolid.services.chat_widget_bridge as bridge_mod

    captured = {}

    monkeypatch.setattr(
        bridge_mod,
        "gui_build_gui_context_payload",
        lambda **kwargs: {"session_id": kwargs["session_id"]},
    )
    monkeypatch.setattr(
        bridge_mod,
        "gui_invoke_widget_slot",
        lambda **kwargs: kwargs["slot_name"] == "ok_slot",
    )
    monkeypatch.setattr(
        bridge_mod,
        "gui_invoke_widget_json_slot",
        lambda **kwargs: {
            "ok": True,
            "slot": kwargs["slot_name"],
            "transport_error": False,
        },
    )
    monkeypatch.setattr(
        bridge_mod,
        "gui_get_widget_action_result",
        lambda **kwargs: {"action": kwargs["action_name"]},
    )
    monkeypatch.setattr(
        bridge_mod,
        "gui_run_awaitable_sync",
        lambda awaitable: asyncio.run(awaitable),
    )

    async def _fake_execute(**kwargs):
        captured.update(kwargs)
        return {"message": "done", "payload": {}}

    monkeypatch.setattr(bridge_mod, "gui_execute_direct_gui_command", _fake_execute)
    monkeypatch.setattr(bridge_mod, "gui_route_direct_gui_command", object())

    assert build_chat_gui_context_payload(session_id="s1") == {"session_id": "s1"}
    assert (
        invoke_chat_widget_slot(
            widget=None, session_id="s", slot_name="ok_slot", qargs=(), logger=None
        )
        is True
    )
    assert invoke_chat_widget_json_slot(
        widget=None, invoke_slot=lambda *_a, **_k: True, slot_name="json_slot", qargs=()
    ) == {
        "ok": True,
        "transport_error": False,
        "slot": "json_slot",
    }
    assert get_chat_widget_action_result(widget=None, action_name="refresh") == {
        "action": "refresh"
    }
    assert run_chat_awaitable_sync(asyncio.sleep(0, result="ok")) == "ok"

    result = asyncio.run(
        execute_chat_direct_gui_command(
            prompt="open video foo.mp4",
            parse_direct_gui_command=lambda prompt: {
                "name": "open_video",
                "args": {"path": prompt},
            },
            handlers={"open_video": object()},
        )
    )
    assert result["message"] == "done"
    assert captured["prompt"] == "open video foo.mp4"
    assert callable(captured["parse_direct_gui_command"])
    assert captured["handlers"]["open_video"] is not None


def test_invoke_chat_widget_json_slot_marks_transport_failures() -> None:
    payload = invoke_chat_widget_json_slot(
        widget=None,
        invoke_slot=lambda *_a, **_k: False,
        slot_name="json_slot",
        qargs=(),
    )
    assert payload["ok"] is False
    assert payload["transport_error"] is True
    assert "Failed to run GUI action" in str(payload.get("error") or "")
