from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from qtpy import QtWidgets

from annolid.services.agent_tooling import describe_agent_capabilities
from annolid.infrastructure.agent_workspace import get_agent_workspace_path


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _set_table_headers(table: QtWidgets.QTableWidget, headers: Iterable[str]) -> None:
    headers_list = [str(item) for item in headers]
    table.setColumnCount(len(headers_list))
    table.setHorizontalHeaderLabels(headers_list)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)


class AgentCapabilitiesDialog(QtWidgets.QDialog):
    """Visual report for combined agent tool and skill capabilities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annolid Agent Capabilities")
        self.resize(1100, 760)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Inspect the current agent tool policy and skill discovery in one place. "
            "Use a task hint to preview which skills the runtime would choose."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Workspace:"))
        self.workspace_edit = QtWidgets.QLineEdit(str(get_agent_workspace_path()))
        controls.addWidget(self.workspace_edit, 1)
        controls.addWidget(QtWidgets.QLabel("Task hint:"))
        self.task_hint_edit = QtWidgets.QLineEdit()
        self.task_hint_edit.setPlaceholderText("e.g. check today's weather")
        controls.addWidget(self.task_hint_edit, 2)
        controls.addWidget(QtWidgets.QLabel("Top K:"))
        self.top_k_spin = QtWidgets.QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(5)
        controls.addWidget(self.top_k_spin)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(self.refresh_btn)
        self.copy_btn = QtWidgets.QPushButton("Copy JSON")
        self.copy_btn.clicked.connect(self._copy_json)
        controls.addWidget(self.copy_btn)
        layout.addLayout(controls)

        self.summary_label = QtWidgets.QLabel("No data loaded yet.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.overview_page = QtWidgets.QWidget()
        self.overview_text = QtWidgets.QPlainTextEdit()
        self.overview_text.setReadOnly(True)
        overview_layout = QtWidgets.QVBoxLayout(self.overview_page)
        overview_layout.addWidget(self.overview_text)
        self.tabs.addTab(self.overview_page, "Overview")

        self.tools_page = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout(self.tools_page)
        self.tools_table = QtWidgets.QTableWidget(0, 4, self)
        _set_table_headers(
            self.tools_table,
            ["Tool", "Allowed", "Source", "Policy"],
        )
        tools_layout.addWidget(self.tools_table)
        self.tabs.addTab(self.tools_page, "Tools")

        self.skills_page = QtWidgets.QWidget()
        skills_layout = QtWidgets.QVBoxLayout(self.skills_page)
        self.skills_table = QtWidgets.QTableWidget(0, 4, self)
        _set_table_headers(
            self.skills_table,
            ["Skill", "Available", "Source", "Description"],
        )
        skills_layout.addWidget(self.skills_table)
        self.tabs.addTab(self.skills_page, "Skills")

        self.suggestions_page = QtWidgets.QWidget()
        suggestions_layout = QtWidgets.QVBoxLayout(self.suggestions_page)
        self.suggestions_text = QtWidgets.QPlainTextEdit()
        self.suggestions_text.setReadOnly(True)
        suggestions_layout.addWidget(self.suggestions_text)
        self.tabs.addTab(self.suggestions_page, "Suggestions")

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _workspace(self) -> str:
        value = self.workspace_edit.text().strip()
        return (
            str(Path(value).expanduser()) if value else str(get_agent_workspace_path())
        )

    def refresh(self) -> None:
        payload = describe_agent_capabilities(
            workspace=self._workspace(),
            task_hint=self.task_hint_edit.text().strip(),
            top_k=int(self.top_k_spin.value()),
        )
        self._render_payload(payload)

    def _render_payload(self, payload: Dict[str, Any]) -> None:
        tool_pool = dict(payload.get("tool_pool") or {})
        skill_pool = dict(payload.get("skill_pool") or {})
        summary = dict(payload.get("summary") or {})
        tool_counts = dict(tool_pool.get("counts") or {})
        skill_counts = dict(skill_pool.get("skill_pool", {}).get("counts") or {})
        self.summary_label.setText(
            " | ".join(
                [
                    f"Workspace: {payload.get('workspace')}",
                    f"Tools: {tool_counts.get('allowed', 0)}/{tool_counts.get('registered', 0)} allowed",
                    f"Skills: {skill_counts.get('available', 0)}/{skill_counts.get('total', 0)} available",
                    f"Suggested skills: {summary.get('suggested_skills', 0)}",
                ]
            )
        )
        self.overview_text.setPlainText(_json_text(payload))
        self._populate_tools_table(tool_pool)
        self._populate_skills_table(skill_pool)
        self._populate_suggestions(payload.get("skill_pool", {}))

    def _populate_tools_table(self, tool_pool: Dict[str, Any]) -> None:
        allowed = {str(item) for item in tool_pool.get("allowed_tools") or []}
        denied = {str(item) for item in tool_pool.get("denied_tools") or []}
        rows = sorted(allowed | denied)
        self.tools_table.setRowCount(len(rows))
        policy = str(tool_pool.get("policy_profile") or "")
        source = str(tool_pool.get("policy_source") or "")
        for idx, name in enumerate(rows):
            self.tools_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.tools_table.setItem(
                idx,
                1,
                QtWidgets.QTableWidgetItem("yes" if str(name) in allowed else "no"),
            )
            self.tools_table.setItem(idx, 2, QtWidgets.QTableWidgetItem(source))
            self.tools_table.setItem(idx, 3, QtWidgets.QTableWidgetItem(policy))
        self.tools_table.resizeColumnsToContents()

    def _populate_skills_table(self, skill_pool: Dict[str, Any]) -> None:
        rows: List[Dict[str, Any]] = list(
            dict(skill_pool.get("skill_pool") or {}).get("preview") or []
        )
        self.skills_table.setRowCount(len(rows))
        for idx, row in enumerate(rows):
            self.skills_table.setItem(
                idx, 0, QtWidgets.QTableWidgetItem(str(row.get("name") or ""))
            )
            self.skills_table.setItem(
                idx,
                1,
                QtWidgets.QTableWidgetItem(
                    "yes" if bool(row.get("available")) else "no"
                ),
            )
            self.skills_table.setItem(
                idx, 2, QtWidgets.QTableWidgetItem(str(row.get("source") or ""))
            )
            self.skills_table.setItem(
                idx,
                3,
                QtWidgets.QTableWidgetItem(str(row.get("description") or "")),
            )
        self.skills_table.resizeColumnsToContents()

    def _populate_suggestions(self, skill_pool: Dict[str, Any]) -> None:
        suggested: List[Dict[str, Any]] = list(skill_pool.get("suggested_skills") or [])
        if not suggested:
            self.suggestions_text.setPlainText(
                "No task hint provided or no suggestions available."
            )
            return
        lines = []
        for row in suggested:
            lines.append(
                f"- {row.get('name')} | score={float(row.get('score') or 0.0):.2f} | "
                f"strategy={row.get('strategy')} | source={row.get('source')}"
            )
        self.suggestions_text.setPlainText("\n".join(lines))

    def _copy_json(self) -> None:
        payload = describe_agent_capabilities(
            workspace=self._workspace(),
            task_hint=self.task_hint_edit.text().strip(),
            top_k=int(self.top_k_spin.value()),
        )
        QtWidgets.QApplication.clipboard().setText(_json_text(payload))
