from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.behavior import segment_labeling as segment_labeling_module
from annolid.core.models.base import ModelResponse
from annolid.core.agent.gui_backend.commands import parse_direct_gui_command
from annolid.gui.widgets.ai_chat_backend import _append_selected_capabilities_prompt
from annolid.gui.widgets import ai_chat_widget as ai_chat_widget_module
from annolid.gui.widgets.ai_chat_widget import _compose_slash_selection_draft
from annolid.gui.widgets.ai_chat_widget import _ChatBubble
from annolid.gui.widgets.ai_chat_widget import _extract_slash_selection_state
from annolid.gui.widgets.ai_chat_widget import _symbol_text
from annolid.gui.widgets.ai_chat_widget import AIChatWidget
from annolid.gui.widgets.behavior_slash_dialog import BehaviorSlashDialog
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


def test_linux_symbol_fallbacks_fit_fixed_icon_buttons(monkeypatch) -> None:
    monkeypatch.setattr(ai_chat_widget_module.sys, "platform", "linux")

    labels = [
        _symbol_text("📎", "F"),
        _symbol_text("🎨", "C"),
        _symbol_text("🪟", "W"),
        _symbol_text("📚", "R"),
        _symbol_text("✉", "@"),
        _symbol_text("🚀", ">"),
    ]

    assert labels == ["F", "C", "W", "R", "@", ">"]
    assert all(len(label) <= 3 for label in labels)


def test_parse_direct_gui_command_routes_capabilities_command() -> None:
    command = parse_direct_gui_command("/capabilities")

    assert command == {"name": "open_agent_capabilities", "args": {}}


def test_parse_direct_gui_command_routes_behavior_dialog() -> None:
    command = parse_direct_gui_command("/behavior")

    assert command == {"name": "open_behavior_dialog", "args": {}}

    command_with_context = parse_direct_gui_command(
        "/behavior labels still, walk from defined list"
    )
    assert command_with_context == {"name": "open_behavior_dialog", "args": {}}


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


def test_behavior_slash_dialog_collects_model_labels_and_prompt(tmp_path: Path) -> None:
    _ensure_qapp()
    video_path = tmp_path / "assay.mp4"
    video_path.write_bytes(b"fake")
    dialog = BehaviorSlashDialog(
        None,
        video_path=str(video_path),
        labels=["still", "walk", "walk"],
        providers=["nvidia", "ollama"],
        provider_models={"nvidia": ["vision-model"], "ollama": ["local-model"]},
        selected_provider="nvidia",
        selected_model="vision-model",
        segment_seconds=1.0,
        frames_per_grid=9,
        max_segments=25,
        subject_term="fly",
        video_description="Head-fixed fly assay.",
        behavior_definitions="front groom: foreleg grooming",
        focus_points="legs and abdomen",
    )
    try:
        values = dialog.values()
    finally:
        dialog.deleteLater()

    assert values["video_path"] == str(video_path)
    assert values["behavior_labels"] == ["still", "walk"]
    assert values["llm_provider"] == "nvidia"
    assert values["llm_model"] == "vision-model"
    assert values["segment_seconds"] == 1.0
    assert values["sample_frames_per_segment"] == 9
    assert values["max_segments"] == 25
    assert values["subject_term"] == "fly"
    assert "Head-fixed fly" in values["video_description"]
    assert "front groom" in values["behavior_definitions"]
    assert "abdomen" in values["focus_points"]


def test_behavior_slash_defaults_use_active_video_labels_and_bot_model(
    monkeypatch,
) -> None:
    _ensure_qapp()
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.load_llm_settings",
        lambda: {
            "provider": "nvidia",
            "last_models": {"nvidia": "vision-model"},
            "nvidia": {"preferred_models": ["vision-model"]},
        },
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.available_providers",
        lambda self: ["nvidia"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.labels",
        lambda self: {"nvidia": "NVIDIA"},
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.current_provider",
        lambda self: "nvidia",
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.available_models",
        lambda self, provider: ["vision-model"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.resolve_initial_model",
        lambda self, provider, available_models: "vision-model",
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
    monkeypatch.setattr(
        AIChatWidget,
        "_labels_from_schema_or_flags",
        lambda self: ["still", "walk"],
    )

    widget = AIChatWidget()
    try:
        host = QtWidgets.QWidget()
        host.video_file = "/tmp/fly.mp4"  # type: ignore[attr-defined]
        widget.host_window_widget = host
        widget.selected_provider = "nvidia"
        widget.selected_model = "vision-model"

        defaults = widget._behavior_slash_defaults()
    finally:
        widget.deleteLater()

    assert defaults["video_path"] == "/tmp/fly.mp4"
    assert defaults["labels"] == ["still", "walk"]
    assert defaults["providers"] == ["nvidia"]
    assert defaults["provider_models"] == {"nvidia": ["vision-model"]}
    assert defaults["selected_provider"] == "nvidia"
    assert defaults["selected_model"] == "vision-model"
    assert defaults["frames_per_grid"] == 9


def test_behavior_slash_dialog_acceptance_queues_labeling(monkeypatch) -> None:
    _ensure_qapp()
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.load_llm_settings",
        lambda: {
            "provider": "nvidia",
            "last_models": {"nvidia": "vision-model"},
            "nvidia": {"preferred_models": ["vision-model"]},
        },
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.available_providers",
        lambda self: ["nvidia"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.labels",
        lambda self: {"nvidia": "NVIDIA"},
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.current_provider",
        lambda self: "nvidia",
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.available_models",
        lambda self, provider: ["vision-model"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.resolve_initial_model",
        lambda self, provider, available_models: "vision-model",
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
    monkeypatch.setattr(
        AIChatWidget,
        "_labels_from_schema_or_flags",
        lambda self: ["still", "walk"],
    )

    class _AcceptedDialog:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def values(self):
            return {
                "video_path": "/tmp/fly.mp4",
                "behavior_labels": ["still", "walk", "front groom"],
                "llm_provider": "nvidia",
                "llm_model": "vision-model",
                "segment_seconds": 1.0,
                "sample_frames_per_segment": 9,
                "max_segments": 50,
                "subject_term": "fly",
                "video_description": "Head-fixed fly assay.",
                "behavior_definitions": "front groom: foreleg grooming",
                "focus_points": "legs and abdomen",
                "overwrite_existing": False,
            }

    monkeypatch.setattr(ai_chat_widget_module, "BehaviorSlashDialog", _AcceptedDialog)

    widget = AIChatWidget()
    captured = {}
    model_route = {}
    try:
        widget.set_provider_and_model = (  # type: ignore[method-assign]
            lambda provider, model: model_route.update(
                {"provider": provider, "model": model}
            )
        )
        widget.bot_label_behavior_segments = (  # type: ignore[method-assign]
            lambda **kwargs: captured.update(kwargs)
        )

        widget.bot_open_behavior_slash_dialog()
    finally:
        widget.deleteLater()

    assert model_route == {"provider": "nvidia", "model": "vision-model"}
    assert captured["video_path"] == "/tmp/fly.mp4"
    assert captured["behavior_labels_csv"] == "still, walk, front groom"
    assert captured["use_defined_behavior_list"] is True
    assert captured["segment_mode"] == "uniform"
    assert captured["segment_seconds"] == 1.0
    assert captured["sample_frames_per_segment"] == 9
    assert captured["max_segments"] == 50
    assert captured["llm_provider"] == "nvidia"
    assert captured["llm_model"] == "vision-model"
    assert captured["subject_term"] == "fly"
    assert "Head-fixed fly" in captured["video_description"]
    assert "front groom" in captured["behavior_definitions"]
    assert "abdomen" in captured["focus_points"]


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


def test_extract_prediction_from_model_text_accepts_no_behavior() -> None:
    parsed = AIChatWidget._extract_prediction_from_model_text(
        '{"label":"no_behavior","confidence":0.88,"description":"Mouse is walking, not grooming or rearing."}',
        ["grooming", "supported rearing", "unsupported rearing"],
    )

    assert parsed["label"] == ""
    assert parsed["classification"] == "no_behavior"
    assert parsed["confidence"] == 0.88
    assert parsed["no_behavior"] is True
    assert parsed["model_label"] == "no_behavior"


def test_extract_prediction_from_model_text_preserves_no_behavior_alias() -> None:
    parsed = AIChatWidget._extract_prediction_from_model_text(
        '{"label":"background","confidence":0.72,"description":"No listed behavior is visible."}',
        ["grooming", "supported rearing", "unsupported rearing"],
    )

    assert parsed["classification"] == "no_behavior"
    assert parsed["model_label"] == "background"


def test_extract_prediction_from_model_text_accepts_label_slug_variants() -> None:
    parsed = AIChatWidget._extract_prediction_from_model_text(
        (
            '{"label":"unsupported_rearing","confidence":0.74,'
            '"description":"Mouse rears without forepaw support."}'
        ),
        ["grooming", "supported rearing", "unsupported rearing"],
    )

    assert parsed["label"] == "unsupported rearing"
    assert parsed["classification"] == "unsupported rearing"
    assert parsed["confidence"] == 0.74
    assert parsed["description"] == "Mouse rears without forepaw support."


def test_extract_prediction_from_model_text_accepts_plain_no_behavior_alias() -> None:
    parsed = AIChatWidget._extract_prediction_from_model_text(
        "No behavior.",
        ["grooming", "supported rearing", "unsupported rearing"],
    )

    assert parsed["label"] == ""
    assert parsed["classification"] == "no_behavior"
    assert parsed["no_behavior"] is True


def test_extract_prediction_from_model_text_uses_json_description_no_behavior() -> None:
    parsed = AIChatWidget._extract_prediction_from_model_text(
        (
            '{"confidence":0.67,'
            '"description":"No listed behavior is visible; the mouse is walking, not grooming."}'
        ),
        ["grooming", "supported rearing", "unsupported rearing"],
    )

    assert parsed["label"] == ""
    assert parsed["classification"] == "no_behavior"
    assert parsed["confidence"] == 0.67
    assert parsed["no_behavior"] is True
    assert "No listed behavior" in parsed["description"]


def test_extract_prediction_from_model_text_preserves_unsupported_label_description() -> (
    None
):
    parsed = AIChatWidget._extract_prediction_from_model_text(
        '{"label":"walking","confidence":0.82,"description":"Mouse walks along the wall without grooming or rearing."}',
        ["grooming", "supported rearing", "unsupported rearing"],
    )

    assert parsed["label"] == ""
    assert parsed["confidence"] == 0.82
    assert parsed["unmatched_label"] == "walking"
    assert parsed["description"] == (
        "Mouse walks along the wall without grooming or rearing."
    )


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


def test_extract_prediction_from_model_text_does_not_default_to_first_label() -> None:
    parsed = AIChatWidget._extract_prediction_from_model_text(
        "The animal is moving across the arena, but no exact class is stated.",
        ["grooming", "rearing", "walking"],
    )
    assert parsed["label"] == ""
    assert parsed["confidence"] == 0.0
    assert "moving across" in parsed["unmatched_text"]


def test_extract_prediction_from_model_text_ignores_verbose_allowed_label_echo() -> (
    None
):
    parsed = AIChatWidget._extract_prediction_from_model_text(
        (
            "Thinking Process:\n"
            "The available labels are `behavior in /tmp/mouse.mp4` and `no_behavior`.\n"
            "The mouse appears mostly stationary. Since no specific behavior is visible, "
            "I should choose between those options."
        ),
        ["behavior in /tmp/mouse.mp4"],
    )

    assert parsed["label"] == ""
    assert parsed["confidence"] == 0.0
    assert "available labels" in parsed["unmatched_text"]


def test_chat_bubble_thinking_collapses_and_toggles() -> None:
    _ensure_qapp()
    bubble = _ChatBubble("Annolid Bot", "", is_user=False)
    try:
        bubble.show()
        QtWidgets.QApplication.processEvents()
        bubble.set_thinking_text("Loading context", in_progress=True)
        assert bubble.thinking_toggle_button is not None
        assert bubble.thinking_view is not None
        assert bubble.thinking_toggle_button.isVisible()
        assert bubble.thinking_view.isVisible()
        assert "running" in bubble.thinking_toggle_button.text().lower()

        bubble.finish_thinking(collapse=True)
        QtWidgets.QApplication.processEvents()
        assert not bubble.thinking_view.isVisible()
        assert "finished" in bubble.thinking_toggle_button.text().lower()

        bubble.toggle_thinking_content()
        QtWidgets.QApplication.processEvents()
        assert bubble.thinking_view.isVisible()
    finally:
        bubble.deleteLater()


def test_append_thinking_progress_line_prefers_newest_and_prefix_growth() -> None:
    class _Dummy:
        def __init__(self) -> None:
            self._progress_lines = []

    dummy = _Dummy()
    boilerplate = {
        "starting agent loop",
        "loading tools and context",
        "prepared system prompt",
        "received model response",
    }

    AIChatWidget._append_thinking_progress_line(
        dummy, "starting agent loop", boilerplate=boilerplate
    )
    assert dummy._progress_lines == ["starting agent loop"]

    AIChatWidget._append_thinking_progress_line(
        dummy, "Analyzing frame 1", boilerplate=boilerplate
    )
    assert dummy._progress_lines == ["Analyzing frame 1"]

    AIChatWidget._append_thinking_progress_line(
        dummy, "Analyzing frame 1 with pose features", boilerplate=boilerplate
    )
    assert dummy._progress_lines == ["Analyzing frame 1 with pose features"]

    AIChatWidget._append_thinking_progress_line(
        dummy, "Evaluating trajectory priors", boilerplate=boilerplate
    )
    assert dummy._progress_lines[0] == "Evaluating trajectory priors"
    assert dummy._progress_lines[1] == "Analyzing frame 1 with pose features"


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


def test_behavior_label_resolution_preserves_explicit_defined_list(monkeypatch) -> None:
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
            widget,
            "_labels_from_schema_or_flags",
            lambda: ["walking", "grooming", "rearing", "STILL"],
        )

        labels = widget._resolve_segment_label_candidates(
            ["still", "walk", "front groom", "back groom", "abdomen move"],
            use_defined_behavior_list=True,
        )

        assert labels == [
            "still",
            "walk",
            "front groom",
            "back groom",
            "abdomen move",
        ]
    finally:
        if widget is not None:
            widget.close()


def test_labels_from_project_schema_prefers_names_and_normalizes_underscores():
    widget = AIChatWidget.__new__(AIChatWidget)

    class _Behavior:
        def __init__(self, code, name):
            self.code = code
            self.name = name

    class _Schema:
        behaviors = [
            _Behavior("behavior_1", "Behavior 1"),
            _Behavior(
                "behavior_in_users_chenyang_downloads_test_annolid_videos_batch_mouse_mp4",
                "behavior_in_users_chenyang_downloads_test_annolid_videos_batch_mouse_mp4",
            ),
            _Behavior("unsupported_rearing", "unsupported_rearing"),
            _Behavior("walking", "walking"),
        ]

    class _Host:
        project_schema = _Schema()

    widget.host_window_widget = _Host()

    assert widget._labels_from_schema_or_flags() == [
        "unsupported rearing",
        "walking",
    ]


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
                self.generic_marks = []

            def create_interval(self, **kwargs):
                self.intervals.append(dict(kwargs))

            def add_generic_mark(self, frame, **kwargs):
                self.generic_marks.append((int(frame), dict(kwargs)))

        class _Host:
            fps = 30.0
            video_file = "/tmp/mouse.mp4"
            timeline_panel = None
            frame_number = 30

            def __init__(self) -> None:
                self.overlay_refresh_calls = []

            @staticmethod
            def _refresh_behavior_log() -> None:
                return None

            def _refresh_behavior_overlay(self, frame_number=None) -> None:
                self.overlay_refresh_calls.append(frame_number)

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
            "skipped_predictions": [],
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

        widget._on_behavior_label_preview(
            {
                "index": 2,
                "total": 3,
                "status": "unclassified",
                "progress": 66,
                "skipped_prediction": {
                    "start_frame": 30,
                    "end_frame": 59,
                    "subject": "mouse",
                    "label": "unclassified",
                    "confidence": 0.82,
                    "description": "Mouse walks, outside the requested behavior list.",
                },
            }
        )

        assert behavior_controller.generic_marks == [
            (
                30,
                {"mark_type": "behavior_unclassified", "color": "#ff9800"},
            )
        ]
        assert len(widget._behavior_label_run_context["skipped_predictions"]) == 1
        assert host._behavior_label_skipped_overlay_records[0]["label"] == (
            "unclassified"
        )
        assert host.overlay_refresh_calls[-1] == 30

        widget._on_behavior_label_finished(
            {
                "predictions": [],
                "skipped_predictions": [],
                "skipped_segments": 1,
                "processed_segments": 3,
                "cancelled": False,
            }
        )

        assert calls[-1][1]["ok"] is True
        assert calls[-1][1]["in_progress"] is False
        assert calls[-1][1]["labeled_segments"] == 1
    finally:
        widget.close()


def test_behavior_segment_vlm_worker_uses_frame_grid_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        video_path = tmp_path / "mouse.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (64, 48),
        )
    assert writer.isOpened()
    try:
        for idx in range(12):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"walking","confidence":0.91,'
                        '"description":"mouse traverses the arena across tiles"}'
                    )
                },
                text=(
                    '{"label":"walking","confidence":0.91,'
                    '"description":"mouse traverses the arena across tiles"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 9, "subject": "mouse"}],
        labels=["grooming", "rearing", "walking"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 0
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["label"] == "walking"
    assert prediction["classification"] == "walking"
    assert prediction["description"]
    assert prediction["visual_evidence"]["type"] == "frame_grid"
    assert prediction["visual_evidence"]["frame_indices"] == [0, 4, 9]
    grid_path = Path(prediction["grid_image_path"])
    assert grid_path.exists()
    assert (
        grid_path.parent
        == video_path.parent / f"{video_path.stem}_behavior_segment_grids"
    )
    assert grid_path.name == f"{video_path.stem}_segment_000001_frames_0_9.png"
    assert prediction["visual_evidence"]["grid_image_path"] == str(grid_path)
    assert "segment frames 0-9" in prediction["grid_frame_description"]
    assert (
        prediction["visual_evidence"]["grid_frame_description"]
        == prediction["grid_frame_description"]
    )
    assert len(requests) == 1
    assert Path(requests[0].image_path) == grid_path
    assert "chronological frame grid" in requests[0].text
    assert "grooming" in requests[0].text
    assert "Annolid Bot" in str(dict(requests[0].params or {}).get("system_prompt"))


def test_behavior_segment_vlm_worker_retries_empty_json_response(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            if "response_format" in dict(request.params or {}):
                return ModelResponse(task="caption", output={"text": ""}, text="")
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"classification":"grooming","confidence":0.72,'
                        '"description":"Mouse repeatedly moves near the head."}'
                    )
                },
                text=(
                    '{"classification":"grooming","confidence":0.72,'
                    '"description":"Mouse repeatedly moves near the head."}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["grooming", "rearing"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 0
    assert len(requests) == 2
    prediction = result["predictions"][0]
    assert prediction["label"] == "grooming"
    assert prediction["classification"] == "grooming"
    assert prediction["description"] == "Mouse repeatedly moves near the head."
    assert prediction["model_description"] == "Mouse repeatedly moves near the head."
    assert prediction["description_source"] == "model"


def test_ollama_vision_capability_detection_uses_metadata() -> None:
    assert (
        segment_labeling_module._extract_ollama_vision_capability(
            {"capabilities": ["completion", "vision"]}
        )
        is True
    )
    assert (
        segment_labeling_module._extract_ollama_vision_capability(
            {"capabilities": ["completion"]}
        )
        is False
    )
    assert (
        segment_labeling_module._extract_ollama_vision_capability(
            {"details": {"families": ["gemma", "clip"]}}
        )
        is True
    )
    assert segment_labeling_module._extract_ollama_vision_capability({}) is None


def test_ollama_non_vision_routing_uses_capability_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: True,
    )
    assert (
        segment_labeling_module.is_likely_non_vision_model(
            provider="ollama", model="any-local-model"
        )
        is False
    )

    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: False,
    )
    assert (
        segment_labeling_module.is_likely_non_vision_model(
            provider="ollama", model="any-local-model"
        )
        is True
    )

    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: None,
    )
    assert (
        segment_labeling_module.is_likely_non_vision_model(
            provider="ollama", model="any-local-model"
        )
        is False
    )


def test_behavior_segment_vlm_worker_uses_single_local_vlm_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"still","confidence":0.84,'
                        '"description":"fly posture remains stable across tiles"}'
                    )
                },
                text=(
                    '{"label":"still","confidence":0.84,'
                    '"description":"fly posture remains stable across tiles"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: True,
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "fly"}],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="ollama",
        llm_model="local-vlm:latest",
    )

    assert result["skipped_segments"] == 0
    assert len(requests) == 1
    assert result["routed_to_caption_profile"] is False
    assert "/no_think" not in str(requests[0].text)
    assert dict(requests[0].params).get("response_format") is None
    assert dict(requests[0].params).get("max_tokens") == 512
    assert dict(requests[0].params).get("extra_body") == {"think": False}


def test_behavior_segment_vlm_worker_routes_ollama_text_model_to_caption_profile(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    init_kwargs = []
    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            init_kwargs.append(dict(kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"still","confidence":0.84,'
                        '"description":"fly posture remains stable across tiles"}'
                    )
                },
                text=(
                    '{"label":"still","confidence":0.84,'
                    '"description":"fly posture remains stable across tiles"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: False,
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "fly"}],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="ollama",
        llm_model="local-text-model:latest",
    )

    assert result["skipped_segments"] == 0
    assert result["routed_to_caption_profile"] is True
    assert len(requests) == 1
    assert requests[0].params.get("extra_body") is None
    assert "/no_think" not in str(requests[0].text)
    assert any(str(item.get("profile") or "") == "caption" for item in init_kwargs)
    assert all(item.get("provider") is None for item in init_kwargs)
    assert all(item.get("model") is None for item in init_kwargs)


def test_behavior_segment_vlm_worker_reports_active_progress(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"still","confidence":0.84,'
                        '"description":"fly posture remains stable across tiles"}'
                    )
                },
                text=(
                    '{"label":"still","confidence":0.84,'
                    '"description":"fly posture remains stable across tiles"}'
                ),
            )

    class _PreviewWorker:
        def __init__(self) -> None:
            self.previews = []
            self.progress = []

        def report_preview(self, payload):
            self.previews.append(dict(payload))

        def report_progress(self, progress):
            self.progress.append(int(progress))

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: True,
    )
    pred_worker = _PreviewWorker()

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "fly"}],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="ollama",
        llm_model="local-vlm:latest",
        pred_worker=pred_worker,
    )

    assert result["skipped_segments"] == 0
    statuses = [payload["status"] for payload in pred_worker.previews]
    assert statuses[:2] == ["building_grid", "model_request"]
    assert pred_worker.previews[1]["attempt"] == "local_vlm_with_image"
    assert statuses[-1] == "labeled"


def test_behavior_label_preview_updates_status_for_active_request(monkeypatch) -> None:
    _ensure_qapp()
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.load_llm_settings",
        lambda: {
            "provider": "ollama",
            "last_models": {"ollama": "local-vlm:latest"},
            "ollama": {"preferred_models": ["local-vlm:latest"]},
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
        lambda self, provider: ["local-vlm:latest"],
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_widget.ProviderRegistry.resolve_initial_model",
        lambda self, provider, available_models: "local-vlm:latest",
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
    calls = []
    try:
        monkeypatch.setattr(
            widget,
            "_set_bot_action_result",
            lambda action, payload: calls.append((action, dict(payload))),
        )
        widget._behavior_label_run_context = {
            "host": object(),
            "behavior_controller": object(),
            "mode": "uniform",
            "labels": ["still", "walk"],
            "evaluated_segments": 10,
            "resumed_segments": 0,
            "skipped_segments": 0,
            "predictions": [],
        }

        widget._on_behavior_label_preview(
            {
                "index": 2,
                "total": 10,
                "status": "model_request",
                "attempt": "json_with_image",
                "progress": 10,
                "start_frame": 70,
                "end_frame": 139,
            }
        )
    finally:
        widget.deleteLater()

    assert (
        "Labeling behavior grid 2/10 with json_with_image" in widget.status_label.text()
    )
    assert calls[-1][0] == "label_behavior_segments"
    assert calls[-1][1]["in_progress"] is True
    assert calls[-1][1]["active_status"] == "model_request"
    assert calls[-1][1]["active_segment"] == 2
    assert calls[-1][1]["active_attempt"] == "json_with_image"


def test_behavior_segment_vlm_worker_builds_description_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            return ModelResponse(
                task="caption",
                output={"text": '{"label":"still","confidence":0.7}'},
                text='{"label":"still","confidence":0.7}',
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "fly"}],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 0
    prediction = result["predictions"][0]
    assert prediction["label"] == "still"
    assert prediction["description_source"] == "fallback"
    assert prediction["model_description"] == ""
    assert "model did not provide a description" in prediction["description"]
    assert prediction["visual_evidence"]["fallback_reason"] == "description_fallback"


def test_behavior_segment_vlm_worker_pauses_after_repeated_empty_responses(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(12):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[
            {"start_frame": 0, "end_frame": 2, "subject": "fly"},
            {"start_frame": 3, "end_frame": 5, "subject": "fly"},
            {"start_frame": 6, "end_frame": 8, "subject": "fly"},
            {"start_frame": 9, "end_frame": 11, "subject": "fly"},
        ],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["empty_response_paused"] is True
    assert result["skipped_segments"] == 3
    assert result["predictions"] == []
    assert "empty text" in result["error"]
    assert len(requests) == 12


def test_behavior_segment_vlm_worker_pauses_after_one_empty_text_only_model_segment(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: False,
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[
            {"start_frame": 0, "end_frame": 3, "subject": "fly"},
            {"start_frame": 4, "end_frame": 7, "subject": "fly"},
        ],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="ollama",
        llm_model="local-text-model:latest",
    )

    assert result["routed_to_caption_profile"] is True
    assert result["empty_response_paused"] is True
    assert result["empty_response_segment_limit"] == 1
    assert result["skipped_segments"] == 1
    assert len(requests) == 2
    assert "Your previous response was empty" in requests[1].text


def test_behavior_segment_vlm_worker_appends_empty_tail_from_prior_segment(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        39.17,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(43):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = min(255, idx * 4)
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: True,
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 39, "end_frame": 42, "subject": "mouse"}],
        labels=["rearing", "walking"],
        sample_frames_per_segment=4,
        llm_profile="",
        llm_provider="ollama",
        llm_model="local-vlm:latest",
        prior_predictions=[
            {
                "start_frame": 0,
                "end_frame": 38,
                "subject": "mouse",
                "label": "walking",
                "classification": "walking",
                "confidence": 0.9,
                "description": "Mouse moves across the frame.",
            }
        ],
    )

    assert result["empty_response_paused"] is False
    assert result["skipped_segments"] == 0
    assert len(requests) == 2
    prediction = result["predictions"][0]
    assert prediction["start_frame"] == 39
    assert prediction["end_frame"] == 42
    assert prediction["label"] == "walking"
    assert prediction["confidence"] == 0.5
    assert prediction["description_source"] == "fallback"
    assert prediction["model_description"] == ""
    assert (
        prediction["visual_evidence"]["fallback_reason"]
        == "empty_response_adjacent_tail"
    )
    assert prediction["visual_evidence"]["empty_attempts"] == 2


def test_behavior_segment_vlm_worker_continues_after_local_empty_response(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(12):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        segment_labeling_module,
        "ollama_model_supports_vision",
        lambda model: True,
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[
            {"start_frame": 0, "end_frame": 3, "subject": "mouse"},
            {"start_frame": 4, "end_frame": 7, "subject": "mouse"},
        ],
        labels=["rearing", "walking"],
        sample_frames_per_segment=4,
        llm_profile="",
        llm_provider="ollama",
        llm_model="local-vlm:latest",
    )

    assert result["routed_to_caption_profile"] is False
    assert result["empty_response_paused"] is False
    assert result["empty_response_segment_limit"] == 3
    assert result["skipped_segments"] == 2
    assert result["predictions"] == []
    assert [
        request.params.get("system_prompt") is not None for request in requests
    ] == [
        True,
        True,
        True,
        True,
    ]
    assert len(requests) == 4


def test_behavior_segment_vlm_worker_skips_no_behavior_response(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(4):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"no_behavior","confidence":0.91,'
                        '"description":"Mouse is walking, not grooming or rearing."}'
                    )
                },
                text=(
                    '{"label":"no_behavior","confidence":0.91,'
                    '"description":"Mouse is walking, not grooming or rearing."}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 3, "subject": "mouse"}],
        labels=["grooming", "supported rearing", "unsupported rearing"],
        sample_frames_per_segment=4,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 1
    assert result["predictions"] == []
    assert result["skipped_predictions"][0]["label"] == "no_behavior"
    assert result["skipped_predictions"][0]["description"] == (
        "Mouse is walking, not grooming or rearing."
    )
    assert result["skipped_predictions"][0]["description_source"] == "model"
    assert (
        result["skipped_predictions"][0]["visual_evidence"]["model_label"]
        == "no_behavior"
    )
    assert result["empty_response_paused"] is False
    assert len(requests) == 1


def test_behavior_segment_vlm_worker_saves_unclassified_model_description(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(4):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 0] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"walking","confidence":0.82,'
                        '"description":"Mouse walks along the wall without grooming or rearing."}'
                    )
                },
                text=(
                    '{"label":"walking","confidence":0.82,'
                    '"description":"Mouse walks along the wall without grooming or rearing."}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 3, "subject": "mouse"}],
        labels=["grooming", "supported rearing", "unsupported rearing"],
        sample_frames_per_segment=4,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 1
    assert result["predictions"] == []
    skipped = result["skipped_predictions"][0]
    assert skipped["label"] == "unclassified"
    assert skipped["description"] == (
        "Mouse walks along the wall without grooming or rearing."
    )
    assert skipped["description_source"] == "model"
    assert skipped["visual_evidence"]["model_label"] == "walking"
    assert skipped["visual_evidence"]["skip_reason"] == "unmatched_label"


def test_behavior_segment_vlm_worker_normalizes_classification_to_label_list(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 12
            writer.write(frame)
    finally:
        writer.release()

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"walking","classification":"locomotion",'
                        '"confidence":0.78,"description":"mouse traverses cage"}'
                    )
                },
                text=(
                    '{"label":"walking","classification":"locomotion",'
                    '"confidence":0.78,"description":"mouse traverses cage"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["grooming", "rearing", "walking"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 0
    prediction = result["predictions"][0]
    assert prediction["label"] == "walking"
    assert prediction["classification"] == "walking"
    assert prediction["description"] == "mouse traverses cage"


def test_behavior_segment_vlm_worker_repair_prompt_fallback_is_label_agnostic(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            if "Your previous response was empty" in str(request.text or ""):
                return ModelResponse(
                    task="caption",
                    output={
                        "text": (
                            '{"label":"rearing","confidence":0.61,'
                            '"description":"posture remains elevated across tiles"}'
                        )
                    },
                    text=(
                        '{"label":"rearing","confidence":0.61,'
                        '"description":"posture remains elevated across tiles"}'
                    ),
                )
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["grooming", "rearing", "digging"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 0
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["label"] == "rearing"
    assert prediction["classification"] == "rearing"
    assert prediction["description"]
    assert prediction["visual_evidence"]["fallback_reason"] == "repair_prompt"
    assert all(str(req.task) == "caption" for req in requests)


def test_behavior_segment_vlm_worker_routes_known_non_vision_model_to_caption_profile(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    init_kwargs = []
    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            init_kwargs.append(dict(kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"rearing","classification":"rearing",'
                        '"confidence":0.71,"description":"mouse appears upright"}'
                    )
                },
                text=(
                    '{"label":"rearing","classification":"rearing",'
                    '"confidence":0.71,"description":"mouse appears upright"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["grooming", "rearing"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="nvidia",
        llm_model="moonshotai/kimi-k2.5",
    )

    assert result["skipped_segments"] == 0
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["label"] == "rearing"
    assert prediction["visual_evidence"]["model_routed_profile"] == "caption"
    assert prediction["visual_evidence"]["requested_provider"] == "nvidia"
    assert prediction["visual_evidence"]["requested_model"] == "moonshotai/kimi-k2.5"
    assert (
        prediction["visual_evidence"]["model_attempt"] == "caption_profile_with_image"
    )
    assert prediction["visual_evidence"]["request_interval_seconds"] == 1.0
    assert len(requests) == 1
    assert any(str(item.get("profile") or "") == "caption" for item in init_kwargs)
    assert all(item.get("provider") is None for item in init_kwargs)
    assert all(item.get("model") is None for item in init_kwargs)


def test_behavior_segment_vlm_worker_repairs_missing_caption_profile_label(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            if len(requests) == 1:
                return ModelResponse(
                    task="caption",
                    output={"text": '{"description":"Mouse appears upright."}'},
                    text='{"description":"Mouse appears upright."}',
                )
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"rearing","classification":"rearing",'
                        '"confidence":0.73,"description":"mouse appears upright"}'
                    )
                },
                text=(
                    '{"label":"rearing","classification":"rearing",'
                    '"confidence":0.73,"description":"mouse appears upright"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        AIChatWidget, "_sleep_with_stop", staticmethod(lambda *_: False)
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["grooming", "rearing"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="nvidia",
        llm_model="moonshotai/kimi-k2.6",
    )

    assert result["routed_to_caption_profile"] is True
    assert result["skipped_segments"] == 0
    assert len(requests) == 2
    assert "Your previous response was empty" in requests[1].text
    prediction = result["predictions"][0]
    assert prediction["label"] == "rearing"
    assert prediction["visual_evidence"]["model_attempt"] == (
        "caption_profile_repair_with_image"
    )
    assert prediction["visual_evidence"]["fallback_reason"] == "repair_prompt"


def test_behavior_segment_vlm_worker_infers_fly_subject_from_video_path(
    monkeypatch, tmp_path: Path
) -> None:
    video_dir = tmp_path / "head_fixed_fly" / "videos"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "2019_06_26_fly2.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            return ModelResponse(
                task="caption",
                output={
                    "text": (
                        '{"label":"still","classification":"still",'
                        '"confidence":0.91,"description":"fly remains still"}'
                    )
                },
                text=(
                    '{"label":"still","classification":"still",'
                    '"confidence":0.91,"description":"fly remains still"}'
                ),
            )

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": None}],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 0
    assert "the fly" in requests[0].text
    assert "the mouse" not in requests[0].text
    assert result["predictions"][0]["visual_evidence"]["subject_term"] == "fly"


def test_behavior_segment_vlm_worker_pauses_after_rate_limit(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "fly.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(16):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 10
            writer.write(frame)
    finally:
        writer.release()

    requests = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            requests.append(request)
            raise RuntimeError("Error code: 429 - Too Many Requests")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)
    monkeypatch.setattr(
        AIChatWidget, "_sleep_with_stop", staticmethod(lambda *_: False)
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[
            {"start_frame": 0, "end_frame": 7, "subject": None},
            {"start_frame": 8, "end_frame": 15, "subject": None},
        ],
        labels=["still", "walk"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="nvidia",
        llm_model="moonshotai/kimi-k2.5",
    )

    assert result["rate_limited"] is True
    assert result["skipped_segments"] == 1
    assert result["predictions"] == []
    assert "rate limit" in result["error"].lower()
    assert len(requests) == 1


def test_behavior_subject_term_prefers_message_hint_over_generic_path() -> None:
    widget = AIChatWidget.__new__(AIChatWidget)

    assert (
        widget._infer_behavior_subject_term(
            "/tmp/generic_recording.avi",
            None,
            explicit_subject_term="fly",
        )
        == "fly"
    )


def test_behavior_segment_vlm_worker_caption_profile_rescue_after_empty_primary(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = idx * 20
            writer.write(frame)
    finally:
        writer.release()

    init_kwargs = []

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            init_kwargs.append(dict(kwargs))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            if str(self.kwargs.get("profile") or "") == "caption":
                return ModelResponse(
                    task="caption",
                    output={
                        "text": (
                            '{"label":"rearing","classification":"rearing",'
                            '"confidence":0.66,'
                            '"description":"mouse remains upright in grid"}'
                        )
                    },
                    text=(
                        '{"label":"rearing","classification":"rearing",'
                        '"confidence":0.66,'
                        '"description":"mouse remains upright in grid"}'
                    ),
                )
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["grooming", "rearing", "digging"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="nvidia",
        llm_model="some-non-kimi-model",
    )

    assert result["skipped_segments"] == 0
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["label"] == "rearing"
    assert prediction["description_source"] == "model"
    assert prediction["visual_evidence"]["fallback_reason"] == "caption_profile_rescue"
    assert prediction["visual_evidence"]["model_attempt"] == "rescue_caption_profile"
    assert any(str(item.get("profile") or "") == "caption" for item in init_kwargs)


def test_behavior_segment_vlm_worker_skips_segment_when_model_returns_no_usable_output(
    monkeypatch, tmp_path: Path
) -> None:
    video_path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for _idx in range(8):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[..., 1] = 80
            writer.write(frame)
    finally:
        writer.release()

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def predict(self, request):
            return ModelResponse(task="caption", output={"text": ""}, text="")

    import annolid.core.models.adapters.llm_chat as llm_chat

    monkeypatch.setattr(llm_chat, "LLMChatAdapter", _FakeAdapter)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._run_behavior_segment_vlm_worker(
        video_path=str(video_path),
        intervals=[{"start_frame": 0, "end_frame": 7, "subject": "mouse"}],
        labels=["resting", "exploring", "rearing"],
        sample_frames_per_segment=3,
        llm_profile="",
        llm_provider="",
        llm_model="",
    )

    assert result["skipped_segments"] == 1
    assert len(result["predictions"]) == 0


def test_behavior_segment_log_saves_classification_and_description(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "mouse.mp4"
    video_path.write_bytes(b"fake")

    class _Host:
        video_file = str(video_path)

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._save_behavior_segment_labeling_log(
        _Host(),
        mode="uniform",
        labels_used=["grooming", "rearing"],
        segment_frames=10,
        segment_seconds=1.0,
        sample_frames_per_segment=3,
        evaluated_segments=2,
        skipped_segments=0,
        predictions=[
            {
                "start_frame": 0,
                "end_frame": 9,
                "subject": "mouse",
                "label": "rearing",
                "confidence": 0.77,
                "description": "Mouse stands upright on hind legs.",
                "grid_image_path": str(tmp_path / "grid.png"),
                "grid_frame_description": "segment frames 0-9; tiles f0, f4, f9.",
            },
            {
                "start_frame": 10,
                "end_frame": 19,
                "subject": "mouse",
                "classification": "grooming",
                "confidence": 0.64,
                "visual_evidence": {
                    "grid_image_path": str(tmp_path / "grid2.png"),
                    "grid_frame_description": "segment frames 10-19; tiles f10, f14, f19.",
                },
            },
        ],
        skipped_predictions=[
            {
                "start_frame": 20,
                "end_frame": 29,
                "subject": "mouse",
                "label": "no_behavior",
                "classification": "no_behavior",
                "confidence": 0.91,
                "description": "Mouse is walking, not grooming or rearing.",
                "model_description": "Mouse is walking, not grooming or rearing.",
                "description_source": "model",
                "grid_image_path": str(tmp_path / "grid3.png"),
                "grid_frame_description": "segment frames 20-29; tiles f20, f24, f29.",
                "visual_evidence": {"skip_reason": "no_behavior"},
            }
        ],
    )

    assert result["ok"] is True
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["classification"] == "rearing"
    assert payload["predictions"][0]["description"] == (
        "Mouse stands upright on hind legs."
    )
    assert payload["predictions"][0]["grid_image_path"].endswith("grid.png")
    assert "segment frames 0-9" in payload["predictions"][0]["grid_frame_description"]
    assert payload["predictions"][1]["classification"] == "grooming"
    assert payload["predictions"][1]["label"] == "grooming"
    assert payload["predictions"][1]["description"] == ""
    assert payload["predictions"][1]["grid_image_path"].endswith("grid2.png")
    assert "segment frames 10-19" in payload["predictions"][1]["grid_frame_description"]
    assert payload["skipped_predictions"][0]["label"] == "no_behavior"
    assert payload["skipped_predictions"][0]["description"] == (
        "Mouse is walking, not grooming or rearing."
    )
    assert payload["skipped_predictions"][0]["description_source"] == "model"


def test_load_resumable_behavior_segment_predictions_filters_allowed_labels(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fly.mp4"
    video_path.write_bytes(b"fake")
    log_path = tmp_path / "fly_behavior_segment_labels.json"
    log_path.write_text(
        json.dumps(
            {
                "segment_frames": 70,
                "segment_seconds": 1.0,
                "sample_frames_per_segment": 9,
                "predictions": [
                    {
                        "start_frame": 0,
                        "end_frame": 69,
                        "label": "still",
                        "confidence": 0.9,
                    },
                    {
                        "start_frame": 70,
                        "end_frame": 139,
                        "label": "walking",
                        "confidence": 0.8,
                    },
                    {
                        "start_frame": "bad",
                        "end_frame": 209,
                        "label": "walk",
                    },
                ],
                "skipped_predictions": [
                    {
                        "start_frame": 210,
                        "end_frame": 279,
                        "label": "no_behavior",
                        "confidence": 0.91,
                        "description": "Fly is still, not grooming.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._load_resumable_behavior_segment_predictions(
        str(video_path),
        labels=["still", "walk"],
        segment_frames=70,
        segment_seconds=1.0,
        sample_frames_per_segment=9,
    )

    assert result["ok"] is True
    assert result["path"] == str(log_path)
    assert result["predictions"] == [
        {
            "start_frame": 0,
            "end_frame": 69,
            "subject": None,
            "label": "still",
            "classification": "still",
            "confidence": 0.9,
            "description": "",
            "model_description": "",
            "description_source": "",
            "grid_image_path": "",
            "grid_frame_description": "",
            "aggression_sub_events": {},
        }
    ]
    assert result["skipped_predictions"] == [
        {
            "start_frame": 210,
            "end_frame": 279,
            "subject": None,
            "label": "no_behavior",
            "classification": "no_behavior",
            "confidence": 0.91,
            "description": "Fly is still, not grooming.",
            "model_description": "",
            "description_source": "",
            "grid_image_path": "",
            "grid_frame_description": "",
            "aggression_sub_events": {},
        }
    ]


def test_load_resumable_behavior_segment_predictions_canonicalizes_saved_labels(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "fly.mp4"
    video_path.write_bytes(b"fake")
    log_path = tmp_path / "fly_behavior_segment_labels.json"
    log_path.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "start_frame": 0,
                        "end_frame": 69,
                        "label": "unsupported_rearing",
                        "confidence": 0.8,
                    }
                ],
                "skipped_predictions": [
                    {
                        "start_frame": 70,
                        "end_frame": 139,
                        "label": "background",
                        "confidence": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    widget = AIChatWidget.__new__(AIChatWidget)
    result = widget._load_resumable_behavior_segment_predictions(
        str(video_path),
        labels=["unsupported rearing"],
        segment_frames=70,
        segment_seconds=1.0,
        sample_frames_per_segment=9,
    )

    assert result["predictions"][0]["label"] == "unsupported rearing"
    assert result["predictions"][0]["classification"] == "unsupported rearing"
    assert result["skipped_predictions"][0]["label"] == "no_behavior"
    assert result["skipped_predictions"][0]["classification"] == "no_behavior"
