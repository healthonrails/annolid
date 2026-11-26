from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets import ai_chat_widget as ai_chat_widget_mod
from annolid.gui.widgets.agent_capabilities_dialog import AgentCapabilitiesDialog
from annolid.gui.widgets.agent_run_dialog import AgentRunDialog
from annolid.gui.widgets.agent_workspace_onboarding_dialog import (
    AgentWorkspaceOnboardingDialog,
)
from annolid.gui.widgets.ai_chat_bubble import _ChatBubble
from annolid.gui.widgets.ai_chat_session_dialog import (
    ChatSessionManagerDialog,
    format_session_markdown,
)
from annolid.gui.widgets.ai_chat_composer_mixin import AIChatComposerMixin
from annolid.gui.widgets.ai_chat_streaming_mixin import AIChatStreamingMixin
from annolid.gui.widgets.ai_chat_theme import (
    build_chat_widget_stylesheet,
    chat_theme_tokens,
    contrast_ratio,
)
from annolid.gui.widgets.ai_chat_viewer_mixin import AIChatViewerCitationsMixin
from annolid.gui.widgets.ai_chat_widget import AIChatWidget
from annolid.services.chat_session import AgentSessionManager, PersistentSessionStore


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_isolated_chat_widget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[AIChatWidget, PersistentSessionStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    manager = AgentSessionManager(sessions_dir=sessions_dir)
    store = PersistentSessionStore(manager=manager)

    settings = {
        "provider": "ollama",
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "preferred_models": ["qwen2.5:7b"],
            "custom_models": [],
        },
        "last_models": {"ollama": "qwen2.5:7b"},
        "ui": {},
    }
    monkeypatch.setattr(ai_chat_widget_mod, "load_llm_settings", lambda: dict(settings))
    monkeypatch.setattr(
        ai_chat_widget_mod, "save_llm_settings", lambda _payload: tmp_path / "llm.json"
    )
    monkeypatch.setattr(ai_chat_widget_mod, "get_chat_session_store", lambda: store)
    monkeypatch.setattr(
        ai_chat_widget_mod, "get_agent_workspace_path", lambda: workspace
    )
    monkeypatch.setattr(
        ai_chat_widget_mod,
        "refresh_runtime_provider_settings",
        lambda _w, after_refresh=None: after_refresh()
        if callable(after_refresh)
        else None,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.provider_registry.ProviderRegistry._fetch_ollama_models",
        lambda _self: ["qwen2.5:7b"],
    )
    widget = AIChatWidget()
    widget.resize(520, 720)
    return widget, store


def test_viewer_citations_mixin_extracted_and_preserves_contracts(
    qapp, monkeypatch, tmp_path
) -> None:
    del qapp
    assert issubclass(AIChatWidget, AIChatViewerCitationsMixin)
    widget, store = _make_isolated_chat_widget(monkeypatch, tmp_path)
    try:
        assert widget._session_manager is store._manager
        assert (
            widget._extract_doi_for_citation("See doi: 10.1038/s41586-020-2649-2.")
            == "10.1038/s41586-020-2649-2"
        )
        assert widget._normalize_threejs_example_input("fly brain mouse") == (
            "fly_brain_mouse_html"
        )
        assert widget._default_citation_bib_path() == (
            tmp_path / "workspace" / "citations.bib"
        )

        opened: list[tuple[str, bool]] = []

        class _FakeWebManager:
            def get_web_state(self):
                return {
                    "ok": True,
                    "has_page": True,
                    "url": "https://example.org/paper",
                    "title": "Behavior Tracking Study",
                }

            def get_page_text(self, max_chars: int = 9000):
                del max_chars
                return {
                    "ok": True,
                    "text": "Published in 2025 with DOI 10.1234/annolid.2025.01",
                }

        host = SimpleNamespace(
            openVideo=lambda **kw: opened.append(
                (str(kw.get("video_path")), bool(kw.get("programmatic_call")))
            ),
            web_manager=_FakeWebManager(),
        )
        widget.set_host_window(host)
        widget.bot_open_video("/tmp/sample_clip.mp4")
        assert opened == [("/tmp/sample_clip.mp4", True)]

        saved = widget._save_citation_from_active_context(
            source="web",
            validate_before_save=False,
        )
        assert saved["ok"] is True
        assert saved["source"] == "web"
        bib_path = Path(saved["bib_file"])
        assert bib_path.exists()
        assert "10.1234/annolid.2025.01" in bib_path.read_text(encoding="utf-8")
    finally:
        widget.shutdown()
        widget.close()


def test_header_new_chat_and_clear_confirmation(qapp, monkeypatch, tmp_path) -> None:
    del qapp
    widget, store = _make_isolated_chat_widget(monkeypatch, tmp_path)
    try:
        first_session = widget.session_id
        store.append_history(
            first_session,
            [
                {"role": "user", "content": "First turn"},
                {"role": "assistant", "content": "First answer"},
            ],
            max_messages=50,
        )
        widget._load_session_history_into_bubbles(first_session)
        assert len(widget._iter_chat_bubbles()) == 2

        # Clicking Clear asks for confirmation and respects No.
        prompts: list[str] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda _parent, title, _text, *_args: (
                prompts.append(str(title)),
                QtWidgets.QMessageBox.No,
            )[1],
        )
        widget._confirm_clear_chat_conversation()
        assert prompts == ["Clear Conversation"]
        assert len(widget._iter_chat_bubbles()) == 2
        assert len(store.get_history(first_session)) == 2

        # New Chat rotates session without wiping the first session's history.
        widget.start_new_chat_session()
        assert widget.session_id != first_session
        assert len(widget._iter_chat_bubbles()) == 0
        assert len(store.get_history(first_session)) == 2

        # Confirming Clear with Yes clears the current session.
        widget._add_bubble("You", "Temp turn", is_user=True)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *_args: QtWidgets.QMessageBox.Yes,
        )
        widget._confirm_clear_chat_conversation()
        assert len(widget._iter_chat_bubbles()) == 0
        assert widget.status_label.text() == "Conversation cleared."

        # Direct clear_chat_conversation remains non-blocking for callers/tests.
        widget._add_bubble("You", "Programmatic clear", is_user=True)
        called_modal = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *_args: called_modal.append(True) or QtWidgets.QMessageBox.No,
        )
        widget.clear_chat_conversation()
        assert called_modal == []
        assert len(widget._iter_chat_bubbles()) == 0

        # Capabilities header button routes to open_agent_capabilities_dialog.
        cap_calls: list[bool] = []
        monkeypatch.setattr(
            widget,
            "_open_agent_capabilities_dialog",
            lambda: cap_calls.append(True),
        )
        widget.capabilities_button.click()
        assert cap_calls == [True]
    finally:
        widget.shutdown()
        widget.close()


def test_composer_options_attachments_history_and_stop_queue(
    qapp, monkeypatch, tmp_path
) -> None:
    widget, store = _make_isolated_chat_widget(monkeypatch, tmp_path)
    try:
        assert widget._session_manager is store._manager
        # 1. Options menu syncs bidirectionally with hidden checkboxes.
        actions = [
            act for act in widget.options_menu.actions() if not act.isSeparator()
        ]
        assert len(actions) == 4
        trace_action = actions[2]
        assert widget.tool_trace_checkbox.isChecked() is False
        assert trace_action.isChecked() is False
        trace_action.setChecked(True)
        assert widget.tool_trace_checkbox.isChecked() is True
        widget.tool_trace_checkbox.setChecked(False)
        assert trace_action.isChecked() is False

        # 2. Image attachment shows pill; clear removes it; non-image appends path note.
        img_path = tmp_path / "frame_001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("frame,behavior\n1,grooming\n", encoding="utf-8")

        widget._attach_paths([str(img_path), str(csv_path)])
        assert widget._pending_explicit_image_path == str(img_path)
        assert widget.attachment_chip_widget.isHidden() is False
        assert "frame_001.png" in widget.attachment_chip_label.text()
        assert f"[Attached: {csv_path}]" in widget.prompt_text_edit.toPlainText()

        widget._clear_attached_image()
        assert widget._pending_explicit_image_path == ""
        assert widget.image_path == ""
        assert widget.attachment_chip_widget.isHidden() is True

        # 3. Drag-and-drop attaches dropped local image file.
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(img_path))])
        drag_ev = QtGui.QDragEnterEvent(
            QtCore.QPoint(10, 10),
            QtCore.Qt.CopyAction,
            mime,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        widget.dragEnterEvent(drag_ev)
        assert drag_ev.isAccepted() is True

        drop_ev = QtGui.QDropEvent(
            QtCore.QPointF(10, 10),
            QtCore.Qt.CopyAction,
            mime,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        widget.dropEvent(drop_ev)
        assert drop_ev.isAccepted() is True
        assert widget._pending_explicit_image_path == str(img_path)
        widget._clear_attached_image()

        # 4. Prompt history navigation recalls earlier prompts and restores draft.
        widget._prompt_history = ["first question", "second question"]
        widget._prompt_history_index = -1
        widget.prompt_text_edit.setPlainText("in-progress draft")
        assert widget._navigate_prompt_history(-1) is True
        assert widget.prompt_text_edit.toPlainText() == "second question"
        assert widget._navigate_prompt_history(-1) is True
        assert widget.prompt_text_edit.toPlainText() == "first question"
        assert widget._navigate_prompt_history(1) is True
        assert widget.prompt_text_edit.toPlainText() == "second question"
        assert widget._navigate_prompt_history(1) is True
        assert widget.prompt_text_edit.toPlainText() == "in-progress draft"

        # 5. Streaming state exposes Stop button, allows Send-queueing, and Esc cancels.
        widget.prompt_text_edit.clear()
        widget._chat_task_active = True
        widget._prepare_streaming_turn_ui("Analyze behavior")
        assert widget.stop_button.isHidden() is False
        assert widget.send_button.isEnabled() is False

        widget.prompt_text_edit.setPlainText("follow-up prompt")
        qapp.processEvents()
        assert widget.send_button.isEnabled() is True
        assert "Queue" in widget.send_button.toolTip()

        esc = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Escape,
            QtCore.Qt.NoModifier,
        )
        handled = widget.eventFilter(widget.prompt_text_edit, esc)
        assert handled is True
        assert widget.is_streaming_chat is False
        assert widget.stop_button.isHidden() is True
        assert widget.status_label.text() == "Cancelled"
    finally:
        widget.shutdown()
        widget.close()


def test_scroll_follow_markdown_copy_and_historical_regenerate(
    qapp, monkeypatch, tmp_path
) -> None:
    widget, store = _make_isolated_chat_widget(monkeypatch, tmp_path)
    try:
        # 1. _ChatBubble copies raw markdown with visual check feedback.
        copied_texts: list[str] = []
        raw_md = "Result: **42 bouts**\n```python\nprint('ok')\n```"
        bubble = _ChatBubble(
            "Annolid Bot",
            raw_md,
            is_user=False,
            on_copy=lambda txt: copied_texts.append(txt),
            allow_regenerate=True,
        )
        bubble._copy_text()
        assert copied_texts == [raw_md]
        assert bubble.copy_button.text() == "✓"
        bubble.close()

        # 2. Reloaded historical assistant bubbles support per-turn regenerate.
        sid = widget.session_id
        store.append_history(
            sid,
            [
                {"role": "user", "content": "Prompt one"},
                {"role": "assistant", "content": "Reply one"},
                {"role": "user", "content": "Prompt two"},
                {"role": "assistant", "content": "Reply two"},
            ],
            max_messages=50,
        )
        widget._load_session_history_into_bubbles(sid)
        bubbles = widget._iter_chat_bubbles()
        assert len(bubbles) == 4
        first_assistant = bubbles[1]
        second_assistant = bubbles[3]
        assert first_assistant.regenerate_button.isHidden() is False
        assert second_assistant.regenerate_button.isHidden() is False
        assert widget._prompt_for_assistant_bubble(first_assistant) == "Prompt one"
        assert widget._prompt_for_assistant_bubble(second_assistant) == "Prompt two"

        sent_prompts: list[str] = []
        monkeypatch.setattr(
            widget,
            "chat_with_model",
            lambda: sent_prompts.append(widget.prompt_text_edit.toPlainText()),
        )
        first_assistant._regenerate()
        assert sent_prompts == ["Prompt one"]

        # 3. Smart scroll follow shows Jump to Latest when user scrolled up.
        bar = widget.scroll_area.verticalScrollBar()
        bar.setRange(0, 500)
        bar.setValue(100)
        widget._on_chat_scroll_changed(100)
        assert widget._auto_follow_scroll is False
        assert widget.jump_to_latest_button.isHidden() is False

        widget._scroll_to_bottom_if_following()
        assert bar.value() == 100  # did not yank user back down

        widget._jump_to_latest_messages()
        assert bar.value() == bar.maximum()
        assert widget._auto_follow_scroll is True
        assert widget.jump_to_latest_button.isHidden() is True

        # 4. Typing triggers _refresh_inline_slash_hints once per text change (no double build).
        hint_calls = 0
        orig_hints = widget._refresh_inline_slash_hints

        def _counted_hints(*args, **kwargs):
            nonlocal hint_calls
            hint_calls += 1
            return orig_hints(*args, **kwargs)

        monkeypatch.setattr(widget, "_refresh_inline_slash_hints", _counted_hints)
        widget.prompt_text_edit.setPlainText("/skill demo")
        qapp.processEvents()
        assert hint_calls == 1
    finally:
        widget.shutdown()
        widget.close()


def test_session_manager_dialog_caches_previews_and_exports_markdown(
    qapp, monkeypatch, tmp_path
) -> None:
    del qapp
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager=manager)

    store.append_history(
        "gui:alpha",
        [
            {"role": "user", "content": "Count grooming bouts"},
            {"role": "assistant", "content": "Detected 12 grooming bouts."},
        ],
        max_messages=50,
    )
    s_alpha = manager.get_or_create("gui:alpha")
    s_alpha.facts["assay"] = "open_field"
    manager.save(s_alpha)

    store.append_history(
        "gui:beta",
        [
            {"role": "user", "content": "Check social sniffing"},
            {"role": "assistant", "content": "Found 4 sniffing episodes."},
        ],
        max_messages=50,
    )

    overview_calls = 0
    orig_overview = manager.get_session_overview

    def _counted_overview(key: str):
        nonlocal overview_calls
        overview_calls += 1
        return orig_overview(key)

    monkeypatch.setattr(manager, "get_session_overview", _counted_overview)

    dlg = ChatSessionManagerDialog(
        session_manager=manager,
        session_store=store,
        active_session_id="gui:alpha",
    )
    try:
        calls_after_refresh = overview_calls
        assert calls_after_refresh == 2
        assert "[You] Count grooming bouts" in dlg.details.toPlainText()
        assert "[Bot] Detected 12 grooming bouts." in dlg.details.toPlainText()

        # Filtering by message content uses cache with zero extra disk reads.
        dlg.filter_edit.setText("sniffing")
        assert overview_calls == calls_after_refresh
        assert dlg.table.rowCount() == 1
        assert dlg._selected_key() == "gui:beta"

        # Filtering by fact value matches alpha.
        dlg.filter_edit.setText("open_field")
        assert overview_calls == calls_after_refresh
        assert dlg.table.rowCount() == 1
        assert dlg._selected_key() == "gui:alpha"

        # Filtering with no matches disables selection-dependent action buttons.
        dlg.filter_edit.setText("nonexistent_query_xyz")
        assert dlg.table.rowCount() == 0
        assert dlg.switch_btn.isEnabled() is False
        assert dlg.export_btn.isEnabled() is False
        assert dlg.clear_btn.isEnabled() is False
        assert dlg.delete_btn.isEnabled() is False

        # Export Markdown writes clean transcript atomically.
        dlg.filter_edit.setText("alpha")
        assert dlg.export_btn.isEnabled() is True
        export_path = tmp_path / "alpha_transcript.md"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(export_path), "Markdown Files (*.md)"),
        )
        dlg._export_selected_markdown()
        assert export_path.exists()
        md_text = export_path.read_text(encoding="utf-8")
        assert "# Annolid Bot Session: gui:alpha" in md_text
        assert "## Session Facts" in md_text
        assert "Count grooming bouts" in md_text
        assert "Detected 12 grooming bouts." in md_text
        assert "Exported session" in dlg.status_label.text()

        # Standalone formatter handles empty transcripts gracefully.
        empty_md = format_session_markdown({"key": "gui:empty"}, [])
        assert "No messages recorded" in empty_md
    finally:
        dlg.close()


def test_agent_capabilities_dialog_filters_copy_slash_and_caches_json(
    qapp, monkeypatch, tmp_path
) -> None:
    del qapp
    calls = 0

    def _fake_describe(*, workspace: str, task_hint: str = "", top_k: int = 5):
        nonlocal calls
        calls += 1
        return {
            "workspace": workspace,
            "summary": {"suggested_skills": 1 if task_hint else 0},
            "tool_pool": {
                "counts": {"allowed": 2, "registered": 2},
                "allowed_tools": ["read_file", "run_tracker"],
                "denied_tools": [],
                "policy_profile": "default",
                "policy_source": "builtin",
            },
            "skill_pool": {
                "skill_pool": {
                    "counts": {"available": 2, "total": 2},
                    "preview": [
                        {
                            "name": "behavior_report",
                            "available": True,
                            "source": "builtin",
                            "description": "Write scientific behavior reports.",
                        },
                        {
                            "name": "tracking_qa",
                            "available": True,
                            "source": "workspace",
                            "description": "Audit identity switches.",
                        },
                    ],
                },
                "suggested_skills": [],
            },
        }

    monkeypatch.setattr(
        "annolid.gui.widgets.agent_capabilities_dialog.describe_agent_capabilities",
        _fake_describe,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.agent_capabilities_dialog.get_agent_workspace_path",
        lambda: tmp_path,
    )
    dlg = AgentCapabilitiesDialog()
    try:
        assert calls == 1

        # Filter tools table.
        dlg.tools_filter_edit.setText("tracker")
        hidden_flags = [
            dlg.tools_table.isRowHidden(r) for r in range(dlg.tools_table.rowCount())
        ]
        assert hidden_flags.count(False) == 1

        # Filter skills table.
        dlg.skills_filter_edit.setText("identity")
        skill_hidden = [
            dlg.skills_table.isRowHidden(r) for r in range(dlg.skills_table.rowCount())
        ]
        assert skill_hidden.count(False) == 1

        # Double-click copies slash command with feedback.
        dlg.tools_filter_edit.clear()
        dlg._copy_selected_tool_slash(dlg.tools_table.item(0, 0))
        tool_name = dlg.tools_table.item(0, 0).text()
        assert QtWidgets.QApplication.clipboard().text() == f"/tool {tool_name}"
        assert f"/tool {tool_name}" in dlg.feedback_label.text()

        dlg.skills_filter_edit.clear()
        dlg._copy_selected_skill_slash(dlg.skills_table.item(0, 0))
        skill_name = dlg.skills_table.item(0, 0).text()
        assert QtWidgets.QApplication.clipboard().text() == f"/skill {skill_name}"

        # Copy JSON reuses cached payload without re-running describe_agent_capabilities.
        dlg._copy_json()
        assert calls == 1
        parsed = json.loads(QtWidgets.QApplication.clipboard().text())
        assert parsed["workspace"] == str(tmp_path)
        assert "Copied capabilities JSON" in dlg.feedback_label.text()

        # Browse workspace updates line edit and triggers refresh.
        ws2 = tmp_path / "ws2"
        ws2.mkdir()
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getExistingDirectory",
            lambda *_args: str(ws2),
        )
        dlg._browse_workspace()
        assert dlg.workspace_edit.text() == str(ws2)
        assert calls == 2
    finally:
        dlg.close()


def test_agent_workspace_onboarding_preserves_progress_and_guards_actions(
    qapp, monkeypatch, tmp_path
) -> None:
    del qapp
    missing_ws = tmp_path / "uninitialized_ws"

    monkeypatch.setattr(
        "annolid.gui.widgets.agent_workspace_onboarding_dialog.get_agent_workspace_path",
        lambda: missing_ws,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.agent_workspace_onboarding_dialog.get_agent_status",
        lambda workspace: {
            "workspace": workspace,
            "workspace_templates": {"AGENTS.md": False},
            "workspace_backup_count": 0,
            "workspace_health": {"guidance": ["Run Apply"]},
        },
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.agent_workspace_onboarding_dialog.onboard_agent_workspace",
        lambda **kw: {
            "workspace": kw["workspace"],
            "dry_run": kw["dry_run"],
            "overwrite": kw["overwrite"],
            "prune_bootstrap": kw["prune_bootstrap"],
            "files": {"AGENTS.md": "would_created" if kw["dry_run"] else "created"},
            "pruned_files": {},
            "summary": {"created": 1},
            "backup_dir": None,
            "workspace_health_after": {
                "template_missing_count": 1 if kw["dry_run"] else 0
            },
        },
    )

    dlg = AgentWorkspaceOnboardingDialog()
    try:
        # Restore button disabled when 0 backups exist.
        assert dlg.restore_btn.isEnabled() is False

        # Opening non-existent workspace shows friendly hint instead of launching desktop URL.
        opened_urls: list[str] = []
        monkeypatch.setattr(
            QtGui.QDesktopServices,
            "openUrl",
            lambda url: opened_urls.append(str(url.toString())) or True,
        )
        dlg._open_workspace_folder()
        assert opened_urls == []
        assert "does not exist yet" in dlg.summary_label.text()

        # Preview + Refresh preserves preview checkmark instead of wiping it.
        dlg._run_preview()
        assert dlg._preview_completed is True
        dlg._refresh_status()
        assert dlg._preview_completed is True
        guide_texts = [
            dlg.guide_list.item(i).text() for i in range(dlg.guide_list.count())
        ]
        assert any("✓ Preview onboarding plan" in t for t in guide_texts)

        # Switching workspace resets progress flags cleanly.
        ws2 = tmp_path / "other_ws"
        ws2.mkdir()
        dlg.workspace_edit.setText(str(ws2))
        dlg._refresh_status()
        assert dlg._preview_completed is False
        assert dlg._apply_completed is False
    finally:
        dlg.close()


def test_agent_run_dialog_completers_preserve_line_edit_contract(
    qapp, monkeypatch
) -> None:
    del qapp
    monkeypatch.setattr(
        "annolid.gui.widgets.agent_run_dialog.load_llm_settings",
        lambda: {
            "provider": "openai",
            "openai": {
                "preferred_models": ["gpt-5-mini", "gpt-4.1"],
                "custom_models": ["custom-vision"],
            },
            "anthropic": {
                "preferred_models": ["claude-sonnet-4-5"],
                "custom_models": [],
            },
            "last_models": {"openai": "gpt-4.1-mini"},
        },
    )
    dlg = AgentRunDialog(config={"llm_adapter": "llm_chat", "llm_provider": "openai"})
    try:
        assert isinstance(dlg.llm_provider_edit, QtWidgets.QLineEdit)
        assert isinstance(dlg.llm_model_edit, QtWidgets.QLineEdit)

        provider_model = dlg.llm_provider_edit.completer().model()
        providers = [
            provider_model.data(provider_model.index(r, 0))
            for r in range(provider_model.rowCount())
        ]
        assert "openai" in providers
        assert "anthropic" in providers

        model_list_model = dlg.llm_model_edit.completer().model()
        openai_models = [
            model_list_model.data(model_list_model.index(r, 0))
            for r in range(model_list_model.rowCount())
        ]
        assert openai_models[0] == "gpt-4.1-mini"
        assert "gpt-5-mini" in openai_models
        assert "custom-vision" in openai_models

        # Changing provider refreshes model completer dynamically.
        dlg.llm_provider_edit.setText("anthropic")
        model_list_model2 = dlg.llm_model_edit.completer().model()
        anthropic_models = [
            model_list_model2.data(model_list_model2.index(r, 0))
            for r in range(model_list_model2.rowCount())
        ]
        assert anthropic_models == ["claude-sonnet-4-5"]

        # Empty provider falls back to active default provider's models.
        dlg.llm_provider_edit.setText("")
        fallback_model = dlg.llm_model_edit.completer().model()
        fallback_models = [
            fallback_model.data(fallback_model.index(r, 0))
            for r in range(fallback_model.rowCount())
        ]
        assert "gpt-5-mini" in fallback_models
    finally:
        dlg.close()


def test_theme_wcag_contrast_and_light_dark_layout_rendering(
    qapp, monkeypatch, tmp_path
) -> None:
    # 1. Verify WCAG AA (>= 4.5:1) contrast for all dark and light theme color pairs.
    for is_dark in (True, False):
        tokens = chat_theme_tokens(is_dark)
        pairs = [
            ("fg_main", "bg_main"),
            ("fg_main", "bg_input"),
            ("subtitle_fg", "bg_main"),
            ("subtitle_fg", "bg_input"),
            ("title_fg", "bg_main"),
            ("bubble_user_fg", "bubble_user_bg"),
            ("fg_main", "bubble_assistant_bg"),
            ("stop_fg", "stop_bg"),
            ("attachment_fg", "attachment_bg"),
            ("danger_fg", "bg_input"),
        ]
        for fg_key, bg_key in pairs:
            ratio = contrast_ratio(tokens[fg_key], tokens[bg_key])
            assert ratio >= 4.5, (
                f"Low contrast ({ratio:.2f}:1) for {fg_key}/{bg_key} in "
                f"{'dark' if is_dark else 'light'} theme"
            )
        qss = build_chat_widget_stylesheet(is_dark)
        assert "QToolButton#sendButton" in qss
        assert "QToolButton#stopButton" in qss
        assert "QFrame#attachmentChipBar" in qss
        assert 'QLabel#promptCountLabel[limitReached="true"]' in qss

    # 2. Verify mixins are wired and light/dark offscreen rendering + narrow-dock sizing.
    assert issubclass(AIChatWidget, AIChatComposerMixin)
    assert issubclass(AIChatWidget, AIChatStreamingMixin)
    widget, store = _make_isolated_chat_widget(monkeypatch, tmp_path)
    try:
        assert widget._session_manager is store._manager
        long_img = tmp_path / ("very_long_experiment_recording_frame_" * 3 + ".png")
        long_img.write_bytes(b"\x89PNG\r\n\x1a\n")
        widget.set_image_path(str(long_img), explicit=True)
        widget.stop_button.setText("Stop")
        widget.stop_button.setVisible(True)
        widget.jump_to_latest_button.setVisible(True)
        widget._add_bubble("You", "Inspect this frame", is_user=True)
        widget._add_bubble("Annolid Bot", "**Detected** grooming bout.", is_user=False)

        for bg_hex in ("#111317", "#ffffff"):
            pal = widget.palette()
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(bg_hex))
            widget.setPalette(pal)
            widget._apply_theme_styles()
            widget.resize(340, 640)
            widget.show()
            qapp.processEvents()
            pixmap = widget.grab()
            assert not pixmap.isNull()
            assert widget.width() <= 360
            # Verify Linux fallback text "Stop" fits inside stop_button without clipping.
            fm = QtGui.QFontMetrics(widget.stop_button.font())
            assert widget.stop_button.sizeHint().width() >= fm.horizontalAdvance("Stop")
            assert widget.attachment_chip_label.toolTip() == str(long_img)
    finally:
        widget.shutdown()
        widget.close()


def test_pre_drain_stop_race_and_double_click_copy_reset(
    qapp, monkeypatch, tmp_path
) -> None:
    del qapp
    # 1. Rapid double-click Copy never sticks button text on "✓".
    bubble = _ChatBubble("Annolid Bot", "**hello**", is_user=False)
    try:
        default_icon = bubble._copy_button_default_text
        assert default_icon != "✓"
        bubble._copy_text()
        assert bubble.copy_button.text() == "✓"
        bubble._copy_text()
        assert bubble.copy_button.text() == "✓"
        bubble._restore_copy_button()
        assert bubble.copy_button.text() == default_icon
    finally:
        bubble.close()

    # 2. Clicking Stop before the 40ms inbound timer drains cleanly cancels & dequeues turn.
    widget, store = _make_isolated_chat_widget(monkeypatch, tmp_path)
    try:
        assert widget._session_manager is store._manager
        widget.attach_canvas_checkbox.setChecked(False)
        widget.prompt_text_edit.setPlainText("Slow request to cancel immediately")
        widget.chat_with_model()
        assert widget.is_streaming_chat is True
        assert widget._chat_task_active is False
        assert widget._chat_message_bus.inbound.qsize() == 1
        assert widget.stop_button.isHidden() is False

        widget._stop_running_response()
        assert widget.is_streaming_chat is False
        assert widget._chat_task_active is False
        assert widget._chat_message_bus.inbound.qsize() == 0
        assert widget.stop_button.isHidden() is True
        assert widget.status_label.text() == "Cancelled"
    finally:
        widget.shutdown()
        widget.close()
