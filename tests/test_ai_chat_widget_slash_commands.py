from __future__ import annotations

import os

from qtpy import QtCore, QtGui, QtWidgets

from annolid.core.agent.gui_backend.commands import parse_direct_gui_command
from annolid.gui.widgets.ai_chat_backend import _append_selected_capabilities_prompt
from annolid.gui.widgets.ai_chat_widget import _compose_slash_selection_draft
from annolid.gui.widgets.ai_chat_widget import _extract_slash_selection_state
from annolid.gui.widgets.ai_chat_widget import AIChatWidget
from annolid.gui.widgets.track_slash_dialog import TrackSlashDialog
from annolid.gui.widgets.track_slash_dialog import build_track_slash_command


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


def test_build_track_slash_command_uses_structured_gui_fields() -> None:
    command = build_track_slash_command(
        {
            "video_path": "/tmp/mouse demo.mp4",
            "text_prompt": "mouse and feeder",
            "model_name": "Cutie",
            "mode": "track",
            "use_countgd": True,
            "to_frame": 400,
        }
    )

    assert command.startswith("/track ")
    assert "video='/tmp/mouse demo.mp4'" in command
    assert "prompt='mouse and feeder'" in command
    assert "model=Cutie" in command
    assert "use_countgd=true" in command
    assert "to_frame=400" in command


def test_track_slash_defaults_use_host_ai_model_dropdown(monkeypatch) -> None:
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

    widget = AIChatWidget()
    try:
        host = QtWidgets.QWidget()
        combo = QtWidgets.QComboBox(host)
        combo.addItems(["Cutie", "SAM3", "CoWTracker"])
        combo.setCurrentText("SAM3")
        host._selectAiModelComboBox = combo  # type: ignore[attr-defined]
        host.video_file = "/tmp/video.mp4"  # type: ignore[attr-defined]
        widget.host_window_widget = host
        widget.prompt_text_edit.setPlainText("track mouse")
        widget.selected_model = "test-model"

        defaults = widget._track_slash_defaults()

        assert defaults["model_names"] == ["Cutie", "SAM3", "CoWTracker"]
        assert defaults["selected_model"] == "SAM3"
    finally:
        widget.deleteLater()


def test_track_slash_dialog_shows_bot_model_hint() -> None:
    _ensure_qapp()
    dialog = TrackSlashDialog(
        None,
        bot_provider="nvidia",
        bot_model="moonshotai/kimi-k2.5",
    )
    try:
        hint = dialog.sam3_hint_label.text()
        assert "nvidia" in hint
        assert "moonshotai/kimi-k2.5" in hint
        assert "SAM3 will reuse the current bot provider/model" in hint
    finally:
        dialog.deleteLater()


def test_extract_label_from_model_text_accepts_behavior_alias_keys() -> None:
    labels = ["walking", "grooming", "rearing"]
    label, confidence = AIChatWidget._extract_label_from_model_text(
        '{"behavior":"grooming","confidence":0.85}',
        labels,
    )
    assert label == "grooming"
    assert confidence == 0.85


def test_extract_label_from_model_text_accepts_classification_alias_key() -> None:
    labels = ["walking", "grooming", "rearing"]
    label, confidence = AIChatWidget._extract_label_from_model_text(
        '{"classification":"rearing","confidence":0.6}',
        labels,
    )
    assert label == "rearing"
    assert confidence == 0.6


def test_extract_label_from_model_text_accepts_prediction_alias_key() -> None:
    labels = ["walking", "grooming", "rearing"]
    label, confidence = AIChatWidget._extract_label_from_model_text(
        '{"prediction":"walking","confidence":0.4}',
        labels,
    )
    assert label == "walking"
    assert confidence == 0.4


def test_extract_prediction_from_model_text_parses_aggression_sub_events() -> None:
    labels = ["aggression_bout", "grooming"]
    parsed = AIChatWidget._extract_prediction_from_model_text(
        (
            '{"label":"aggression_bout","confidence":0.9,'
            '"description":"slap in the face then run away",'
            '"sub_events":{"slap in face":1,"run_away":1}}'
        ),
        labels,
    )
    assert parsed["label"] == "aggression_bout"
    assert parsed["confidence"] == 0.9
    assert parsed["aggression_sub_events"] == {
        "slap_in_face": 1,
        "run_away": 1,
    }


def test_extract_prediction_from_model_text_infers_aggression_sub_events_from_text() -> (
    None
):
    labels = ["aggression_bout", "grooming"]
    parsed = AIChatWidget._extract_prediction_from_model_text(
        "Aggression bout with initiation of bigger fights then run away.",
        labels,
    )
    assert parsed["label"] == "aggression_bout"
    assert parsed["aggression_sub_events"] == {
        "run_away": 1,
        "fight_initiation": 1,
    }


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


def test_behavior_label_preset_runs_one_second_vlm_flow(monkeypatch) -> None:
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

        class _Host:
            video_file = "/tmp/mouse.mp4"

        widget.host_window_widget = _Host()
        monkeypatch.setattr(
            widget,
            "_labels_from_schema_or_flags",
            lambda: ["walking", "rearing"],
        )

        widget._run_behavior_label_preset_one_second()

        drafted = widget.prompt_text_edit.toPlainText()
        assert "label behavior in /tmp/mouse.mp4" in drafted
        assert "with labels walking, rearing" in drafted
        assert "from defined list every 1s" in drafted
        assert "Drafted 1s behavior labeling command" in widget.status_label.text()
    finally:
        if widget is not None:
            widget.close()


def test_labels_from_schema_flags_includes_timeline_behaviors(monkeypatch) -> None:
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

        class _FlagWidget:
            @staticmethod
            def _get_existing_flag_names():
                return {"behavior_1": 0, "walk": 1}

        class _FlagsController:
            pinned_flags = {"behavior_2": False, "rear": True}

        class _BehaviorController:
            behavior_names = {"groom", "behavior_3"}

        class _Host:
            project_schema = None
            flags = {}
            flag_widget = _FlagWidget()
            flags_controller = _FlagsController()
            behavior_controller = _BehaviorController()

        widget.host_window_widget = _Host()
        labels = widget._labels_from_schema_or_flags()
        assert "groom" in labels
        assert "walk" in labels
        assert "rear" in labels
        assert "Agent" not in labels
        assert "behavior_1" not in labels
        assert "behavior_2" not in labels
        assert "behavior_3" not in labels
    finally:
        if widget is not None:
            widget.close()


def test_behavior_label_preview_applies_incremental_updates(monkeypatch) -> None:
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

    widget = AIChatWidget()
    try:

        class _BehaviorController:
            def __init__(self) -> None:
                self.intervals = []

            def create_interval(self, **kwargs):
                self.intervals.append(dict(kwargs))

        class _Host:
            fps = 30.0
            video_file = "/tmp/mouse.mp4"
            timeline_panel = None

            @staticmethod
            def _refresh_behavior_log() -> None:
                return None

        behavior_controller = _BehaviorController()
        host = _Host()
        widget._behavior_label_run_context = {
            "host": host,
            "behavior_controller": behavior_controller,
            "mode": "uniform",
            "labels": ["walking", "rearing"],
            "segment_frames": 30,
            "segment_seconds": 1.0,
            "sample_frames_per_segment": 3,
            "evaluated_segments": 3,
            "processed_segments": 0,
            "skipped_segments": 0,
            "predictions": [],
            "use_defined_behavior_list": True,
            "default_subject": "mouse",
            "timestamp_provider": widget._behavior_label_timestamp_provider(host),
        }
        calls = []

        monkeypatch.setattr(
            widget,
            "_save_behavior_segment_progress",
            lambda *, force_timestamps=False: {
                "ok": True,
                "timestamp_result": {
                    "ok": True,
                    "path": "/tmp/mouse_timestamps.csv",
                    "rows": 2,
                },
                "behavior_log_result": {
                    "ok": True,
                    "path": "/tmp/mouse_behavior_segment_labels.json",
                    "rows": 2,
                },
            },
        )
        monkeypatch.setattr(
            widget,
            "_set_bot_action_result",
            lambda action, payload: calls.append((action, dict(payload))),
        )

        widget._on_behavior_label_preview(
            {
                "index": 1,
                "total": 3,
                "status": "labeled",
                "progress": 33,
                "prediction": {
                    "start_frame": 0,
                    "end_frame": 29,
                    "subject": "mouse",
                    "label": "walking",
                    "confidence": 0.9,
                },
            }
        )

        assert len(behavior_controller.intervals) == 1
        assert behavior_controller.intervals[0]["behavior"] == "walking"
        assert behavior_controller.intervals[0]["start_frame"] == 0
        assert len(widget._behavior_label_run_context["predictions"]) == 1
        assert "Labeled segment 1/3 as 'walking'" in widget.status_label.text()
        assert calls[-1][0] == "label_behavior_segments"
        assert calls[-1][1]["in_progress"] is True

        widget._on_behavior_label_finished(
            {
                "predictions": [],
                "skipped_segments": 0,
                "processed_segments": 3,
                "cancelled": False,
            }
        )

        assert calls[-1][1]["ok"] is True
        assert calls[-1][1]["in_progress"] is False
        assert calls[-1][1]["labeled_segments"] == 1
    finally:
        widget.close()
