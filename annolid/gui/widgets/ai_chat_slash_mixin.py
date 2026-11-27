"""Slash command parsing, completions, and capability chips for AIChatWidget."""

from __future__ import annotations

from collections import OrderedDict
import logging
import re
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.infrastructure.agent_workspace import get_agent_workspace_path
from annolid.services import (
    build_root_slash_completion_entries,
    describe_agent_capabilities,
    matches_slash_completion_search,
)

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _split_slash_value_list(raw: str) -> List[str]:
    names = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", str(raw or ""))
    return _dedupe_preserve_order(names)


def _extract_slash_selection_state(prompt: str) -> Dict[str, Any]:
    raw = str(prompt or "")
    selected_skill_names: List[str] = []
    selected_tool_names: List[str] = []
    selected_capabilities: List[Dict[str, str]] = []
    slash_commands: List[str] = []
    open_capabilities = False
    retained_lines: List[str] = []

    for line in raw.splitlines():
        stripped = str(line or "").strip()
        if not stripped.startswith("/"):
            retained_lines.append(line)
            continue
        match = re.match(r"^/([A-Za-z0-9_-]+)(?:\s+([\s\S]+))?$", stripped)
        if not match:
            retained_lines.append(line)
            continue
        command = str(match.group(1) or "").strip().lower()
        args = str(match.group(2) or "").strip()
        if command in {"skill", "skills"}:
            names = _split_slash_value_list(args)
            selected_skill_names.extend(names)
            for name in names:
                selected_capabilities.append({"kind": "skill", "name": name})
            slash_commands.append(f"/{command}{(' ' + args) if args else ''}".strip())
            if not args:
                open_capabilities = True
            continue
        if command in {"tool", "tools"}:
            names = _split_slash_value_list(args)
            selected_tool_names.extend(names)
            for name in names:
                selected_capabilities.append({"kind": "tool", "name": name})
            slash_commands.append(f"/{command}{(' ' + args) if args else ''}".strip())
            if not args:
                open_capabilities = True
            continue
        if command in {"capabilities", "caps"}:
            slash_commands.append(f"/{command}")
            open_capabilities = True
            continue
        retained_lines.append(line)

    clean_prompt = "\n".join(retained_lines).strip()
    return {
        "clean_prompt": clean_prompt,
        "selected_skill_names": _dedupe_preserve_order(selected_skill_names),
        "selected_tool_names": _dedupe_preserve_order(selected_tool_names),
        "selected_capabilities": [
            {
                "kind": str(item.get("kind") or "").strip().lower(),
                "name": str(item.get("name") or "").strip(),
            }
            for item in selected_capabilities
            if str(item.get("kind") or "").strip().lower() in {"skill", "tool"}
            and str(item.get("name") or "").strip()
        ],
        "slash_commands": slash_commands,
        "open_capabilities": open_capabilities,
    }


def _compose_slash_selection_draft(
    clean_prompt: str,
    *,
    selected_skill_names: List[str],
    selected_tool_names: List[str],
    selected_capabilities: Optional[List[Dict[str, str]]] = None,
) -> str:
    control_lines: List[str] = []
    ordered_entries = list(selected_capabilities or [])
    if ordered_entries:
        for item in ordered_entries:
            kind = str(item.get("kind") or "").strip().lower()
            name = str(item.get("name") or "").strip()
            if kind in {"skill", "tool"} and name:
                control_lines.append(f"/{kind} {name}")
    else:
        for name in _dedupe_preserve_order(selected_skill_names):
            control_lines.append(f"/skill {name}")
        for name in _dedupe_preserve_order(selected_tool_names):
            control_lines.append(f"/tool {name}")
    body = str(clean_prompt or "").strip()
    parts = list(control_lines)
    if body:
        parts.append(body)
    return "\n".join(parts).strip()


class AIChatSlashMixin:
    """Mixin encapsulating slash completions and capability chip controls."""

    def _current_slash_capability_state(self) -> Dict[str, Any]:
        text = self.prompt_text_edit.toPlainText()
        return _extract_slash_selection_state(text)

    def _selected_capability_entries(self) -> List[Dict[str, str]]:
        state = self._current_slash_capability_state()
        entries = list(state.get("selected_capabilities") or [])
        if entries:
            return [
                {
                    "kind": str(item.get("kind") or "").strip().lower(),
                    "name": str(item.get("name") or "").strip(),
                }
                for item in entries
                if str(item.get("kind") or "").strip().lower() in {"skill", "tool"}
                and str(item.get("name") or "").strip()
            ]
        skills = list(state.get("selected_skill_names") or [])
        tools = list(state.get("selected_tool_names") or [])
        return [
            *[{"kind": "skill", "name": name} for name in skills],
            *[{"kind": "tool", "name": name} for name in tools],
        ]

    def _refresh_selected_capability_chips(self) -> None:
        layout = getattr(self, "capability_chip_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addStretch(1)
        entries = self._selected_capability_entries()
        chips: List[QtWidgets.QToolButton] = []

        def add_chip(kind: str, name: str) -> None:
            chip = QtWidgets.QToolButton(self)
            chip.setObjectName("capabilityChipButton")
            chip.setText(f"{name}  ×")
            chip.setToolTip(f"Click to remove {kind} '{name}'")
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setAutoRaise(True)
            chip.setFocusPolicy(QtCore.Qt.StrongFocus)
            chip.setStyleSheet(self._selected_capability_chip_style(kind))
            chip.clicked.connect(
                lambda _checked=False,
                _kind=kind,
                _name=name: self._remove_capability_chip(_kind, _name)
            )
            chip.setProperty("role", "selected")
            chip.setProperty("capability_kind", kind)
            chip.setProperty("capability_name", name)
            chip.setProperty("kind", kind)
            chip.installEventFilter(self)
            chips.append(chip)
            layout.insertWidget(layout.count() - 1, chip)

        for entry in entries:
            add_chip(str(entry.get("kind") or "skill"), str(entry.get("name") or ""))
        self._selected_capability_chip_widgets = chips
        has_chips = bool(chips)
        self.capability_chip_scroll.setVisible(has_chips)
        self.capability_chip_label.setVisible(has_chips)
        self.clear_capability_chips_button.setVisible(has_chips)
        self._refresh_suggested_skill_chips()

    def _refresh_suggested_skill_chips(self) -> None:
        layout = getattr(self, "suggested_chip_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addStretch(1)
        state = self._current_slash_capability_state()
        prompt_hint = str(state.get("clean_prompt") or "").strip()
        if not prompt_hint:
            self._suggested_capability_chip_widgets = []
            self.suggested_chip_scroll.setVisible(False)
            self.suggested_chip_label.setVisible(False)
            self._refresh_header_capability_summary()
            return
        payload = self._load_slash_capabilities_payload(task_hint=prompt_hint)
        skill_pool = dict(payload.get("skill_pool") or {})
        suggestions = list(skill_pool.get("suggested_skills") or [])
        if not suggestions:
            self._suggested_capability_chip_widgets = []
            self.suggested_chip_scroll.setVisible(False)
            self.suggested_chip_label.setVisible(False)
            self._refresh_header_capability_summary()
            return
        selected_names = {
            str(name or "").strip().lower()
            for name in (
                list(state.get("selected_skill_names") or [])
                + list(state.get("selected_tool_names") or [])
            )
            if str(name or "").strip()
        }
        chips: List[QtWidgets.QToolButton] = []
        for row in suggestions[:8]:
            name = str(row.get("name") or "").strip()
            if not name or name.lower() in selected_names:
                continue
            chip = QtWidgets.QToolButton(self)
            chip.setObjectName("suggestedSkillChipButton")
            score = float(row.get("score") or 0.0)
            chip.setText(f"{name}  +")
            chip.setToolTip(
                f"Click to add skill '{name}' ({row.get('strategy') or 'suggested'}, score={score:.2f})"
            )
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setAutoRaise(True)
            chip.setFocusPolicy(QtCore.Qt.StrongFocus)
            chip.setStyleSheet(self._suggested_skill_chip_style())
            chip.clicked.connect(
                lambda _checked=False, _name=name: self._add_skill_capability_chip(
                    _name
                )
            )
            chip.setProperty("role", "suggested")
            chip.setProperty("capability_kind", "skill")
            chip.setProperty("capability_name", name)
            chip.installEventFilter(self)
            chips.append(chip)
            layout.insertWidget(layout.count() - 1, chip)
        self._suggested_capability_chip_widgets = chips
        has_chips = bool(chips)
        self.suggested_chip_scroll.setVisible(has_chips)
        self.suggested_chip_label.setVisible(has_chips)
        self._refresh_header_capability_summary()

    def _refresh_inline_slash_hints(self) -> None:
        layout = getattr(self, "slash_hint_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addStretch(1)
        context = self._slash_completion_context()
        if not context:
            self._slash_hint_widgets = []
            self.slash_hint_scroll.setVisible(False)
            self.slash_hint_label.setVisible(False)
            return
        entries = self._build_slash_completion_entries(context)
        if not entries:
            self._slash_hint_widgets = []
            self.slash_hint_scroll.setVisible(False)
            self.slash_hint_label.setVisible(False)
            return
        chips: List[QtWidgets.QToolButton] = []
        for row in entries[:5]:
            display = str(row.get("display") or "").strip()
            if not display:
                continue
            chip = QtWidgets.QToolButton(self)
            chip.setObjectName("slashHintChipButton")
            kind = str(row.get("kind") or "").strip().lower()
            action = str(row.get("action") or "").strip().lower()
            chip.setText(display)
            description = str(row.get("description") or "").strip()
            tip_parts = [description] if description else []
            if kind:
                tip_parts.insert(0, kind)
            chip.setToolTip(" - ".join(tip_parts) if tip_parts else display)
            chip.setCursor(QtCore.Qt.PointingHandCursor)
            chip.setAutoRaise(True)
            chip.setFocusPolicy(QtCore.Qt.StrongFocus)
            chip.setStyleSheet(self._slash_hint_chip_style(kind))
            chip.clicked.connect(
                lambda _checked=False,
                _row=dict(row): self._apply_slash_completion_entry(_row)
            )
            chip.setProperty("role", "slash_hint")
            chip.setProperty("completion_kind", kind)
            chip.setProperty("completion_action", action)
            chip.installEventFilter(self)
            chips.append(chip)
            layout.insertWidget(layout.count() - 1, chip)
        self._slash_hint_widgets = chips
        has_chips = bool(chips)
        self.slash_hint_scroll.setVisible(has_chips)
        self.slash_hint_label.setVisible(has_chips)

    def _selected_capability_chip_style(self, kind: str) -> str:
        kind_token = str(kind or "").strip().lower()
        if kind_token == "skill":
            return """
                QToolButton#capabilityChipButton {
                    border: 1px solid rgba(43, 166, 124, 0.55);
                    border-radius: 12px;
                    padding: 5px 10px;
                    margin: 0px;
                    background: rgba(43, 166, 124, 0.16);
                    color: palette(text);
                    font-weight: 600;
                }
                QToolButton#capabilityChipButton:hover {
                    background: rgba(43, 166, 124, 0.24);
                }
                QToolButton#capabilityChipButton:pressed {
                    background: rgba(43, 166, 124, 0.34);
                }
            """
        if kind_token == "tool":
            return """
                QToolButton#capabilityChipButton {
                    border: 1px solid rgba(78, 141, 245, 0.55);
                    border-radius: 12px;
                    padding: 5px 10px;
                    margin: 0px;
                    background: rgba(78, 141, 245, 0.16);
                    color: palette(text);
                    font-weight: 600;
                }
                QToolButton#capabilityChipButton:hover {
                    background: rgba(78, 141, 245, 0.24);
                }
                QToolButton#capabilityChipButton:pressed {
                    background: rgba(78, 141, 245, 0.34);
                }
            """
        return """
            QToolButton#capabilityChipButton {
                border: 1px solid rgba(120, 130, 145, 0.45);
                border-radius: 12px;
                padding: 5px 10px;
                margin: 0px;
                background: rgba(120, 130, 145, 0.14);
                color: palette(text);
                font-weight: 600;
            }
            QToolButton#capabilityChipButton:hover {
                background: rgba(120, 130, 145, 0.22);
            }
            QToolButton#capabilityChipButton:pressed {
                background: rgba(120, 130, 145, 0.30);
            }
        """

    def _suggested_skill_chip_style(self) -> str:
        return """
            QToolButton#suggestedSkillChipButton {
                border: 1px dashed rgba(175, 125, 55, 0.60);
                border-radius: 12px;
                padding: 5px 10px;
                margin: 0px;
                background: rgba(175, 125, 55, 0.10);
                color: palette(text);
            }
            QToolButton#suggestedSkillChipButton:hover {
                background: rgba(175, 125, 55, 0.18);
            }
            QToolButton#suggestedSkillChipButton:pressed {
                background: rgba(175, 125, 55, 0.28);
            }
        """

    def _slash_hint_chip_style(self, kind: str) -> str:
        kind_token = str(kind or "").strip().lower()
        if kind_token in {"skill", "skills"}:
            border = "rgba(43, 166, 124, 0.42)"
            fill = "rgba(43, 166, 124, 0.12)"
        elif kind_token in {"tool", "tools"}:
            border = "rgba(78, 141, 245, 0.42)"
            fill = "rgba(78, 141, 245, 0.12)"
        else:
            border = "rgba(175, 125, 55, 0.42)"
            fill = "rgba(175, 125, 55, 0.10)"
        return f"""
            QToolButton#slashHintChipButton {{
                border: 1px solid {border};
                border-radius: 12px;
                padding: 4px 10px;
                margin: 0px;
                background: {fill};
                color: palette(text);
                font-weight: 600;
            }}
            QToolButton#slashHintChipButton:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            QToolButton#slashHintChipButton:pressed {{
                background: rgba(255, 255, 255, 0.14);
            }}
        """

    def _refresh_header_capability_summary(self) -> None:
        label = getattr(self, "capability_summary_label", None)
        if label is None:
            return
        state = self._current_slash_capability_state()
        skills = int(len(state.get("selected_skill_names") or []))
        tools = int(len(state.get("selected_tool_names") or []))
        total = skills + tools
        if total <= 0:
            label.setText("Selected: none")
        else:
            label.setText(
                f"Selected: {skills} skill{'s' if skills != 1 else ''} · {tools} tool{'s' if tools != 1 else ''}"
            )

    def _chip_widgets_in_order(self) -> List[QtWidgets.QAbstractButton]:
        return [
            *list(getattr(self, "_selected_capability_chip_widgets", []) or []),
            *list(getattr(self, "_suggested_capability_chip_widgets", []) or []),
            *list(getattr(self, "_slash_hint_widgets", []) or []),
        ]

    def _focus_capability_chip(self, *, kind: str = "selected") -> bool:
        if str(kind or "").strip().lower() == "suggested":
            widgets = list(
                getattr(self, "_suggested_capability_chip_widgets", []) or []
            )
        else:
            widgets = list(getattr(self, "_selected_capability_chip_widgets", []) or [])
        if not widgets:
            return False
        try:
            widgets[0].setFocus()
            return True
        except Exception:
            return False

    def _focus_next_capability_chip(
        self, current: Optional[QtWidgets.QWidget], delta: int
    ) -> bool:
        widgets = self._chip_widgets_in_order()
        if not widgets:
            return False
        current_index = -1
        if current is not None:
            try:
                current_index = widgets.index(current)  # type: ignore[arg-type]
            except ValueError:
                current_index = -1
        next_index = (
            0
            if current_index < 0
            else max(0, min(len(widgets) - 1, current_index + int(delta)))
        )
        try:
            widgets[next_index].setFocus()
            return True
        except Exception:
            return False

    def _remove_capability_chip(self, kind: str, name: str) -> None:
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        skill_key = str(name or "").strip().lower()
        entries = [
            item
            for item in self._selected_capability_entries()
            if str(item.get("name") or "").strip().lower() != skill_key
        ]
        skill_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "skill"
        ]
        tool_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "tool"
        ]
        draft = _compose_slash_selection_draft(
            clean_prompt,
            selected_skill_names=skill_names,
            selected_tool_names=tool_names,
            selected_capabilities=entries,
        )
        self.prompt_text_edit.setPlainText(draft)
        self.prompt_text_edit.setFocus()
        self.status_label.setText(f"Removed {kind}: {name}")

    def _add_skill_capability_chip(self, name: str) -> None:
        skill_name = str(name or "").strip()
        if not skill_name:
            return
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        entries = [
            item
            for item in self._selected_capability_entries()
            if str(item.get("name") or "").strip().lower() != skill_name.lower()
        ]
        entries.append({"kind": "skill", "name": skill_name})
        skill_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "skill"
        ]
        tool_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "tool"
        ]
        draft = _compose_slash_selection_draft(
            clean_prompt,
            selected_skill_names=skill_names,
            selected_tool_names=tool_names,
            selected_capabilities=entries,
        )
        self.prompt_text_edit.setPlainText(draft)
        self.prompt_text_edit.setFocus()
        self.status_label.setText(f"Added suggested skill: {skill_name}")

    def _clear_selected_capabilities(self) -> None:
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        self.prompt_text_edit.setPlainText(clean_prompt)
        self.prompt_text_edit.setFocus()
        self.status_label.setText("Cleared selected skills and tools.")

    def _reorder_selected_capability(
        self, source: QtWidgets.QWidget, target: QtWidgets.QWidget
    ) -> bool:
        if source is target:
            return False
        source_kind = str(source.property("capability_kind") or "").strip().lower()
        source_name = str(source.property("capability_name") or "").strip()
        target_name = str(target.property("capability_name") or "").strip()
        if not source_kind or not source_name or not target_name:
            return False
        entries = self._selected_capability_entries()
        source_index = next(
            (
                idx
                for idx, item in enumerate(entries)
                if str(item.get("kind") or "").strip().lower() == source_kind
                and str(item.get("name") or "").strip().lower() == source_name.lower()
            ),
            -1,
        )
        target_index = next(
            (
                idx
                for idx, item in enumerate(entries)
                if str(item.get("name") or "").strip().lower() == target_name.lower()
            ),
            -1,
        )
        if source_index < 0 or target_index < 0 or source_index == target_index:
            return False
        moved = entries.pop(source_index)
        if source_index < target_index:
            target_index -= 1
        entries.insert(target_index, moved)
        state = self._current_slash_capability_state()
        clean_prompt = str(state.get("clean_prompt") or "").strip()
        skill_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "skill"
        ]
        tool_names = [
            str(item.get("name") or "").strip()
            for item in entries
            if str(item.get("kind") or "").strip().lower() == "tool"
        ]
        draft = _compose_slash_selection_draft(
            clean_prompt,
            selected_skill_names=skill_names,
            selected_tool_names=tool_names,
            selected_capabilities=entries,
        )
        self.prompt_text_edit.setPlainText(draft)
        self.prompt_text_edit.setFocus()
        self.status_label.setText(
            f"Moved {source_kind}: {source_name} before {target_name}"
        )
        return True

    def _load_slash_capabilities_payload(
        self, *, task_hint: str = "", top_k: int = 5
    ) -> Dict[str, Any]:
        workspace = str(get_agent_workspace_path())
        normalized_hint = str(task_hint or "").strip()
        cache_key = (
            workspace,
            self.selected_provider,
            self.selected_model,
            normalized_hint,
            int(top_k),
        )
        cache_entries = getattr(self, "_slash_capabilities_cache_entries", None)
        if isinstance(cache_entries, OrderedDict) and cache_key in cache_entries:
            cache_entries.move_to_end(cache_key)
            cached = cache_entries[cache_key]
            if isinstance(cached, dict):
                return cached
        if getattr(self, "_slash_capabilities_cache_key", None) == cache_key:
            cached = getattr(self, "_slash_capabilities_cache", None)
            if isinstance(cached, dict):
                return cached
        try:
            payload = describe_agent_capabilities(
                workspace=workspace,
                provider=self.selected_provider,
                model=self.selected_model,
                task_hint=normalized_hint,
                top_k=int(top_k),
            )
        except Exception as exc:
            logger.warning("Slash capability load failed: %s", exc)
            payload = {
                "workspace": workspace,
                "provider": self.selected_provider,
                "model": self.selected_model,
                "tool_pool": {
                    "workspace": workspace,
                    "provider": self.selected_provider,
                    "model": self.selected_model,
                    "counts": {"registered": 0, "allowed": 0, "denied": 0},
                    "allowed_tools": [],
                    "denied_tools": [],
                },
                "skill_pool": {
                    "workspace": workspace,
                    "skill_pool": {"counts": {}, "preview": []},
                    "suggested_skills": [],
                },
                "summary": {},
            }
        self._slash_capabilities_cache_key = cache_key
        self._slash_capabilities_cache = payload
        if not isinstance(cache_entries, OrderedDict):
            cache_entries = OrderedDict()
            self._slash_capabilities_cache_entries = cache_entries
        cache_entries[cache_key] = payload
        cache_entries.move_to_end(cache_key)
        while len(cache_entries) > 16:
            cache_entries.popitem(last=False)
        return payload

    def _slash_completion_context(self) -> Dict[str, Any]:
        cursor = self.prompt_text_edit.textCursor()
        block = cursor.block()
        line = str(block.text() or "")
        cursor_pos = max(0, int(cursor.position() - block.position()))
        prefix = line[:cursor_pos]
        match = re.search(
            r"(^|\s)([\/@](?P<command>[A-Za-z0-9_-]*)(?:\s+(?P<args>[^\n]*))?)$",
            prefix,
        )
        if not match:
            return {}
        token = str(match.group(2) or "")
        command = str(match.group("command") or "").strip().lower()
        args = str(match.group("args") or "").strip()
        token_start = max(0, len(prefix) - len(token))
        mode = "root"
        if command in {"skill", "skills"}:
            mode = "skill"
        elif command in {"tool", "tools"}:
            mode = "tool"
        return {
            "mode": mode,
            "command": command,
            "args": args,
            "token": token,
            "token_start": block.position() + token_start,
            "token_end": cursor.position(),
            "search_prefix": token.lower().lstrip("/@"),
        }

    def _build_slash_completion_entries(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        payload = self._load_slash_capabilities_payload()
        entries: List[Dict[str, Any]] = []

        def add_entry(
            *,
            display: str,
            search: str,
            insert: str,
            kind: str,
            action: str = "",
            description: str = "",
        ) -> None:
            entries.append(
                {
                    "display": display,
                    "search": search,
                    "insert": insert,
                    "kind": kind,
                    "action": action,
                    "description": description,
                }
            )

        mode = str(context.get("mode") or "root").strip().lower()
        search_prefix = str(context.get("search_prefix") or "").strip().lower()

        if mode == "root":
            entries.extend(build_root_slash_completion_entries())
        elif mode == "skill":
            skill_pool = dict(payload.get("skill_pool") or {})
            nested_pool = dict(skill_pool.get("skill_pool") or {})
            preview_rows = list(nested_pool.get("preview") or [])
            suggested = {
                str(row.get("name") or "").strip().lower()
                for row in list(skill_pool.get("suggested_skills") or [])
            }
            ordered_rows = []
            if suggested:
                ordered_rows.extend(
                    row
                    for row in preview_rows
                    if str(row.get("name") or "").strip().lower() in suggested
                )
            ordered_rows.extend(row for row in preview_rows if row not in ordered_rows)
            for row in ordered_rows:
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                description = str(row.get("description") or "").strip()
                insert = f"/skill {name} "
                search = insert.rstrip().lower()
                add_entry(
                    display=f"Skill: {name}",
                    search=search,
                    insert=insert,
                    kind="skill",
                    description=description,
                )
        elif mode == "tool":
            tool_pool = dict(payload.get("tool_pool") or {})
            tool_names = [
                str(name or "").strip()
                for name in list(tool_pool.get("allowed_tools") or [])
                if str(name or "").strip()
            ]
            for name in tool_names:
                insert = f"/tool {name} "
                add_entry(
                    display=f"Tool: {name}",
                    search=insert.rstrip().lower(),
                    insert=insert,
                    kind="tool",
                    description="User-selected tool hint",
                )

        query = search_prefix
        if query:
            entries = [
                row
                for row in entries
                if matches_slash_completion_search(str(row.get("search") or ""), query)
            ]
        if mode == "skill":
            return entries[:25]
        if mode == "tool":
            return entries[:40]
        return entries

    def _ensure_slash_completion_ui(self) -> None:
        if getattr(self, "_slash_completion_model", None) is not None:
            return
        self._slash_completion_model = QtGui.QStandardItemModel(self)
        self._slash_completer = QtWidgets.QCompleter(self._slash_completion_model, self)
        self._slash_completer.setWidget(self.prompt_text_edit)
        self._slash_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._slash_completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        self._slash_completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self._slash_completer.setCompletionRole(QtCore.Qt.UserRole)
        popup = self._slash_completer.popup()
        popup.setAlternatingRowColors(True)
        popup.setUniformItemSizes(True)
        popup.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        popup.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        popup.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._slash_completer.activated.connect(self._apply_slash_completion)

    def _update_slash_completion_ui(self) -> None:
        self._ensure_slash_completion_ui()
        context = self._slash_completion_context()
        if not context:
            self._hide_slash_completion_ui()
            self._refresh_inline_slash_hints()
            return
        entries = self._build_slash_completion_entries(context)
        self._slash_completion_context_state = context
        if not entries:
            self._hide_slash_completion_ui()
            self._refresh_inline_slash_hints()
            return
        model = self._slash_completion_model
        assert model is not None
        model.clear()
        for row in entries:
            item = QtGui.QStandardItem(str(row.get("display") or ""))
            # Strip leading '/' or '@' so QCompleter's MatchStartsWith matches
            # context["search_prefix"] (which also strips leading '/@').
            normalized_search = (
                str(row.get("search") or "").strip().lower().lstrip("/@")
            )
            item.setData(normalized_search, QtCore.Qt.UserRole)
            item.setData(str(row.get("insert") or ""), QtCore.Qt.UserRole + 1)
            item.setData(str(row.get("kind") or ""), QtCore.Qt.UserRole + 2)
            item.setData(str(row.get("action") or ""), QtCore.Qt.UserRole + 3)
            item.setData(str(row.get("description") or ""), QtCore.Qt.UserRole + 4)
            model.appendRow(item)
        completer = self._slash_completer
        assert completer is not None
        completer.setCompletionPrefix(
            str(context.get("search_prefix") or "").lstrip("/@")
        )
        rect = self.prompt_text_edit.cursorRect()
        rect.setWidth(
            max(
                rect.width(),
                self.prompt_text_edit.viewport().width() // 2,
            )
        )
        completer.complete(rect)
        popup = completer.popup()
        if popup.model() is not None and popup.model().rowCount() > 0:
            popup.setCurrentIndex(popup.model().index(0, 0))
        self._refresh_inline_slash_hints()

    def _hide_slash_completion_ui(self) -> None:
        completer = getattr(self, "_slash_completer", None)
        if completer is None:
            return
        popup = completer.popup()
        if popup is not None and popup.isVisible():
            popup.hide()

    def _apply_slash_completion(self, completion: Any) -> None:
        if not completion:
            return
        model_index = None
        if isinstance(completion, QtCore.QModelIndex):
            model_index = completion
        else:
            completer = getattr(self, "_slash_completer", None)
            if completer is not None:
                popup = completer.popup()
                if popup is not None:
                    model_index = popup.currentIndex()
        if model_index is None or not model_index.isValid():
            return
        row = {
            "kind": str(model_index.data(QtCore.Qt.UserRole + 2) or "").strip().lower(),
            "action": str(model_index.data(QtCore.Qt.UserRole + 3) or "")
            .strip()
            .lower(),
            "insert": str(model_index.data(QtCore.Qt.UserRole + 1) or ""),
            "display": str(model_index.data(QtCore.Qt.DisplayRole) or "").strip(),
            "description": str(model_index.data(QtCore.Qt.UserRole + 4) or "").strip(),
        }
        self._apply_slash_completion_entry(row)

    def _clear_active_slash_token(self) -> None:
        context = dict(getattr(self, "_slash_completion_context_state", {}) or {})
        token_start = int(context.get("token_start") or 0)
        token_end = int(context.get("token_end") or 0)
        if token_end <= token_start:
            return
        cursor = self.prompt_text_edit.textCursor()
        doc_len = len(self.prompt_text_edit.toPlainText())
        if token_start < 0 or token_end > doc_len:
            return
        cursor.beginEditBlock()
        cursor.setPosition(token_start)
        cursor.setPosition(token_end, QtGui.QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.endEditBlock()
        self.prompt_text_edit.setTextCursor(cursor)

    def _apply_slash_completion_entry(self, row: Dict[str, Any]) -> None:
        kind = str(row.get("kind") or "").strip().lower()
        action = str(row.get("action") or "").strip().lower()
        # Preserve intentional trailing spaces (e.g. "/skill ", "/tool ", "/track ").
        insert_text = str(row.get("insert") or "").lstrip()
        display = str(row.get("display") or "").strip()
        if action == "open_capabilities":
            self._hide_slash_completion_ui()
            self._clear_active_slash_token()
            self._open_agent_capabilities_dialog()
            return
        if action == "open_track_dialog":
            self._hide_slash_completion_ui()
            self._clear_active_slash_token()
            self.bot_open_track_slash_dialog()
            return
        if action == "open_behavior_dialog":
            self._hide_slash_completion_ui()
            self._clear_active_slash_token()
            self.bot_open_behavior_slash_dialog()
            return
        context = dict(getattr(self, "_slash_completion_context_state", {}) or {})
        token_start = int(context.get("token_start") or 0)
        token_end = int(context.get("token_end") or 0)
        cursor = self.prompt_text_edit.textCursor()
        cursor.beginEditBlock()
        cursor.setPosition(token_start)
        cursor.setPosition(token_end, QtGui.QTextCursor.KeepAnchor)
        cursor.insertText(insert_text)
        cursor.endEditBlock()
        self.prompt_text_edit.setTextCursor(cursor)
        self._hide_slash_completion_ui()
        if kind in {"skill", "tool"}:
            self.status_label.setText(f"Selected {kind}: {display}")

    def _move_slash_completion_selection(self, delta: int) -> None:
        completer = getattr(self, "_slash_completer", None)
        if completer is None:
            return
        popup = completer.popup()
        if popup is None or not popup.isVisible():
            return
        model = popup.model()
        if model is None:
            return
        row_count = int(model.rowCount())
        if row_count <= 0:
            return
        current = popup.currentIndex()
        row = current.row() if current.isValid() else 0
        row = max(0, min(row_count - 1, row + int(delta)))
        popup.setCurrentIndex(model.index(row, 0))

    @QtCore.Slot()
    def open_agent_capabilities_dialog(self) -> None:
        self._open_agent_capabilities_dialog()

    def _open_agent_capabilities_dialog(self) -> None:
        dialog = getattr(self, "_agent_capabilities_dialog", None)
        if dialog is None:
            from annolid.gui.widgets.agent_capabilities_dialog import (
                AgentCapabilitiesDialog,
            )

            dialog = AgentCapabilitiesDialog(self)
            self._agent_capabilities_dialog = dialog
        try:
            dialog.refresh()
        except Exception:
            pass
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
