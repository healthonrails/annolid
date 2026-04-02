from __future__ import annotations

import os

from qtpy import QtCore, QtGui, QtWidgets

from annolid.core.agent.gui_backend.commands import parse_direct_gui_command
from annolid.gui.widgets.ai_chat_backend import _append_selected_capabilities_prompt
from annolid.gui.widgets.ai_chat_widget import _compose_slash_selection_draft
from annolid.gui.widgets.ai_chat_widget import _extract_slash_selection_state
from annolid.gui.widgets.ai_chat_widget import AIChatWidget


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp() -> QtWidgets.QApplication:
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_extract_slash_selection_state_strips_control_lines() -> None:
    state = _extract_slash_selection_state(
        "/skill weather\n/tool cron\nCheck today's weather"
    )

    assert state["clean_prompt"] == "Check today's weather"
    assert state["selected_skill_names"] == ["weather"]
    assert state["selected_tool_names"] == ["cron"]
    assert state["open_capabilities"] is False


def test_extract_slash_selection_state_opens_capabilities_for_empty_command() -> None:
    state = _extract_slash_selection_state("/capabilities")

    assert state["clean_prompt"] == ""
    assert state["open_capabilities"] is True


def test_append_selected_capabilities_prompt_adds_selected_sections() -> None:
    prompt = _append_selected_capabilities_prompt(
        "Base prompt",
        selected_skill_names=["weather"],
        selected_tool_names=["cron"],
    )

    assert "## Selected Skills" in prompt
    assert "`weather`" in prompt
    assert "## Selected Tools" in prompt
    assert "`cron`" in prompt


def test_compose_slash_selection_draft_orders_controls_before_prompt() -> None:
    draft = _compose_slash_selection_draft(
        "Check today's weather",
        selected_skill_names=["weather"],
        selected_tool_names=["cron"],
    )

    assert draft.splitlines() == [
        "/skill weather",
        "/tool cron",
        "Check today's weather",
    ]


def test_slash_selection_state_preserves_control_order() -> None:
    state = _extract_slash_selection_state("/tool cron\n/skill weather\nRun the report")

    assert state["selected_capabilities"] == [
        {"kind": "tool", "name": "cron"},
        {"kind": "skill", "name": "weather"},
    ]

    draft = _compose_slash_selection_draft(
        "Run the report",
        selected_skill_names=["weather"],
        selected_tool_names=["cron"],
        selected_capabilities=state["selected_capabilities"],
    )

    assert draft.splitlines() == [
        "/tool cron",
        "/skill weather",
        "Run the report",
    ]


def test_parse_direct_gui_command_routes_capabilities_command() -> None:
    command = parse_direct_gui_command("/capabilities")

    assert command == {"name": "open_agent_capabilities", "args": {}}


def test_capability_chips_and_shortcuts_route_actions(monkeypatch) -> None:
    _ensure_qapp()
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.load_llm_settings",
        lambda: {
            "provider": "ollama",
            "last_models": {"ollama": "test-model"},
            "ollama": {"preferred_models": ["test-model"]},
        },
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.available_providers",
        lambda self: ["ollama"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.labels",
        lambda self: {"ollama": "Ollama"},
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.current_provider",
        lambda self: "ollama",
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.available_models",
        lambda self, provider: ["test-model"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.resolve_initial_model",
        lambda self, provider, available_models: "test-model",
    )
    monkeypatch.setattr(
        AIChatWidget,
        "_load_session_history_into_bubbles",
        lambda self, session_id: None,
    )
    monkeypatch.setattr(AIChatWidget, "_update_model_selector", lambda self: None)
    monkeypatch.setattr(AIChatWidget, "_refresh_header_chips", lambda self: None)
    monkeypatch.setattr(
        AIChatWidget, "_load_quick_actions_from_settings", lambda self: None
    )

    widget = None
    widget = AIChatWidget()
    try:
        monkeypatch.setattr(
            AIChatWidget,
            "_load_slash_capabilities_payload",
            lambda self, **kwargs: {
                "skill_pool": {
                    "suggested_skills": [
                        {
                            "name": "forecast",
                            "score": 0.94,
                            "strategy": "lexical",
                            "source": "builtin",
                        },
                        {
                            "name": "weather",
                            "score": 0.90,
                            "strategy": "lexical",
                            "source": "builtin",
                        },
                    ]
                }
            },
        )

        widget.prompt_text_edit.setPlainText(
            "/skill weather\n/tool cron\nCheck today's weather"
        )
        QtWidgets.QApplication.processEvents()

        assert [
            chip.property("capability_name")
            for chip in widget._selected_capability_chip_widgets
        ] == [
            "weather",
            "cron",
        ]
        assert [
            chip.property("capability_name")
            for chip in widget._suggested_capability_chip_widgets
        ] == [
            "forecast",
        ]

        selected_chip = widget._selected_capability_chip_widgets[0]

        widget.prompt_text_edit.setPlainText("@wea")
        cursor = widget.prompt_text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        widget.prompt_text_edit.setTextCursor(cursor)
        QtWidgets.QApplication.processEvents()
        context = widget._slash_completion_context()
        assert context["search_prefix"] == "wea"

        widget.prompt_text_edit.setPlainText(
            "/skill weather\n/tool cron\nCheck today's weather"
        )
        QtWidgets.QApplication.processEvents()
        selected_chip = widget._selected_capability_chip_widgets[0]

        focus_calls = []
        monkeypatch.setattr(
            widget,
            "_focus_capability_chip",
            lambda kind="selected": focus_calls.append(kind) or True,
        )
        clear_calls = []
        monkeypatch.setattr(
            widget,
            "_clear_selected_capabilities",
            lambda: clear_calls.append(True),
        )

        left_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Left,
            QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier,
        )
        assert widget.eventFilter(widget.prompt_text_edit, left_event) is True
        assert focus_calls == ["selected"]

        backspace_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Backspace,
            QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier,
        )
        assert widget.eventFilter(widget.prompt_text_edit, backspace_event) is True
        assert clear_calls == [True]

        removed = []
        monkeypatch.setattr(
            widget,
            "_remove_capability_chip",
            lambda kind, name: removed.append((kind, name)),
        )
        delete_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Delete,
            QtCore.Qt.NoModifier,
        )
        assert widget.eventFilter(selected_chip, delete_event) is True
        assert removed == [("skill", "weather")]

        added = []
        monkeypatch.setattr(
            widget,
            "_add_skill_capability_chip",
            lambda name: added.append(name),
        )
        suggested_chip = widget._suggested_capability_chip_widgets[0]
        enter_event = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Return,
            QtCore.Qt.NoModifier,
        )
        assert widget.eventFilter(suggested_chip, enter_event) is True
        assert added == ["forecast"]
    finally:
        if widget is not None:
            widget.close()
