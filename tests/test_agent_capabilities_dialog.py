from __future__ import annotations

import os

from qtpy import QtWidgets

from annolid.gui.widgets.agent_capabilities_dialog import AgentCapabilitiesDialog


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp() -> QtWidgets.QApplication:
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_agent_capabilities_dialog_renders_combined_payload(monkeypatch) -> None:
    _ensure_qapp()

    monkeypatch.setattr(
        "annolid.gui.widgets.agent_capabilities_dialog.describe_agent_capabilities",
        lambda **kwargs: {
            "workspace": str(kwargs.get("workspace") or "/tmp/ws"),
            "provider": "ollama",
            "model": "qwen3",
            "tool_pool": {
                "counts": {"registered": 3, "allowed": 2, "denied": 1},
                "allowed_tools": ["read_file", "exec"],
                "denied_tools": ["write_file"],
                "policy_profile": "strict",
                "policy_source": "runtime",
            },
            "skill_pool": {
                "skill_pool": {
                    "counts": {
                        "total": 2,
                        "available": 2,
                        "unavailable": 0,
                        "always": 1,
                    },
                    "preview": [
                        {
                            "name": "weather",
                            "available": True,
                            "source": "workspace",
                            "description": "Weather helper",
                        },
                        {
                            "name": "logs",
                            "available": True,
                            "source": "builtin",
                            "description": "Log helper",
                        },
                    ],
                    "suggested_skills": [
                        {
                            "name": "weather",
                            "score": 1.0,
                            "strategy": "lexical",
                            "source": "workspace",
                        }
                    ],
                },
                "suggested_skills": [
                    {
                        "name": "weather",
                        "score": 1.0,
                        "strategy": "lexical",
                        "source": "workspace",
                    }
                ],
            },
            "summary": {
                "registered_tools": 3,
                "available_tools": 2,
                "available_skills": 2,
                "suggested_skills": 1,
            },
        },
    )

    dialog = AgentCapabilitiesDialog()
    try:
        assert dialog.tabs.count() == 4
        assert dialog.tools_table.rowCount() == 3
        assert dialog.skills_table.rowCount() == 2
        assert "Suggested skills: 1" in dialog.summary_label.text()
        assert "weather" in dialog.suggestions_text.toPlainText()
        assert "strict" in dialog.overview_text.toPlainText()
    finally:
        dialog.close()


def test_agent_capabilities_dialog_behavior_preset_updates_task_hint(
    monkeypatch,
) -> None:
    _ensure_qapp()
    calls: list[dict[str, object]] = []

    def _fake_describe(**kwargs):
        calls.append(dict(kwargs))
        return {
            "workspace": str(kwargs.get("workspace") or "/tmp/ws"),
            "provider": "ollama",
            "model": "qwen3",
            "tool_pool": {
                "counts": {"registered": 0, "allowed": 0, "denied": 0},
                "allowed_tools": [],
                "denied_tools": [],
                "policy_profile": "default",
                "policy_source": "runtime",
            },
            "skill_pool": {
                "skill_pool": {"counts": {"total": 0, "available": 0}, "preview": []},
                "suggested_skills": [],
            },
            "summary": {
                "registered_tools": 0,
                "available_tools": 0,
                "available_skills": 0,
                "suggested_skills": 0,
            },
        }

    monkeypatch.setattr(
        "annolid.gui.widgets.agent_capabilities_dialog.describe_agent_capabilities",
        _fake_describe,
    )

    dialog = AgentCapabilitiesDialog()
    try:
        buttons = dialog.findChildren(QtWidgets.QToolButton)
        preset = next(
            btn for btn in buttons if btn.text().strip() == "Aggression Segmentation"
        )
        preset.click()
        assert (
            "Segment aggression bouts"
            in str(dialog.task_hint_edit.text() or "").strip()
        )
        assert calls
        assert (
            "Segment aggression bouts" in str(calls[-1].get("task_hint") or "").strip()
        )
    finally:
        dialog.close()
