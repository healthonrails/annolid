from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import yaml
from qtpy import QtCore

from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow
    from annolid.gui.widgets.flags import FlagTableWidget


class FlagsController(QtCore.QObject):
    """Manage pinned behavior flags, persistence, and widget synchronization."""

    def __init__(
        self,
        *,
        window: "AnnolidWindow",
        widget: "FlagTableWidget",
        config_path: Optional[Path] = None,
    ) -> None:
        super().__init__(widget)
        self._window = window
        self._widget = widget
        self._config_path = (
            Path(config_path)
            if config_path is not None
            else window.here.parent.resolve() / "configs" / "behaviors.yaml"
        )
        self.pinned_flags: Dict[str, bool] = {}

        widget.startButtonClicked.connect(self.start_flag)
        widget.endButtonClicked.connect(self.end_flag)
        widget.flagToggled.connect(self._handle_flag_toggled)
        widget.flagsSaved.connect(self.save_flags)
        widget.rowSelected.connect(self._handle_row_selected)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        """Load persisted flags and populate the widget."""
        flags = self._load_from_disk()
        pending = self._window.__dict__.pop("_pending_pinned_flags", None)
        if pending:
            flags.update(pending)
        self.set_flags(flags, persist=False)

    def set_flags(self, flags: Dict[str, bool], *, persist: bool = False) -> None:
        """Replace the current pinned flags and refresh UI state."""
        flags = dict(flags or {})
        self.pinned_flags.clear()
        self.pinned_flags.update(flags)
        if self.pinned_flags:
            self._widget.loadFlags(self.pinned_flags)
        else:
            self._widget.clear()
        self._update_canvas_text()
        if persist:
            self._persist_flags(self.pinned_flags.keys())

    def load_flags(self, flags: Dict[str, bool]) -> None:
        """Convenience wrapper for callers that expect load semantics."""
        self.set_flags(flags, persist=False)

    def save_flags(self, flags: Dict[str, bool]) -> None:
        """Persist the provided flags (called when user saves via the widget)."""
        logger.info("Saving pinned flags: %s", sorted(flags.keys()))
        self.set_flags(flags, persist=True)

    def start_flag(self, flag_name: str, record_event: bool = True) -> None:
        """Mark a flag as active and optionally record a start event."""
        if not flag_name:
            return
        if self._window.project_schema:
            behavior_def = self._window.project_schema.behavior_map().get(
                flag_name)
            if behavior_def and behavior_def.exclusive_with:
                active_conflicts = [
                    name
                    for name, value in self.pinned_flags.items()
                    if value and name in behavior_def.exclusive_with
                ]
                if active_conflicts:
                    message = (
                        f"Behavior '{flag_name}' conflicts with "
                        f"{', '.join(sorted(active_conflicts))}."
                    )
                    logger.warning(message)
                    self._widget._update_row_value(flag_name, False)
                    self._window.statusBar().showMessage(message, 4000)
                    return
        if record_event and self._window.seekbar:
            self._window.record_behavior_event(
                flag_name, "start", frame_number=self._window.frame_number
            )
        self._window.canvas.setBehaviorText(flag_name)
        self._window.event_type = flag_name
        self._window._update_modifier_controls_for_behavior(flag_name)
        self.pinned_flags.setdefault(flag_name, False)
        self.pinned_flags[flag_name] = True
        existing = self._widget._get_existing_flag_names()
        if flag_name not in existing:
            self._widget.loadFlags({flag_name: True})
        else:
            self._widget._update_row_value(flag_name, True)

    def end_flag(self, flag_name: str, record_event: bool = True) -> None:
        """Mark a flag as inactive and optionally record an end event."""
        if not flag_name:
            return
        if record_event and self._window.seekbar:
            self._window.record_behavior_event(
                flag_name, "end", frame_number=self._window.frame_number
            )
        if self._window.event_type == flag_name:
            self._window.event_type = None
        self._window._update_modifier_controls_for_behavior(
            self._window.event_type)
        self.pinned_flags.setdefault(flag_name, False)
        self.pinned_flags[flag_name] = False
        existing = self._widget._get_existing_flag_names()
        if flag_name not in existing:
            self._widget.loadFlags({flag_name: False})
        else:
            self._widget._update_row_value(flag_name, False)
        self._update_canvas_text()

    def apply_prompt_flags(self, flags: Dict[str, bool]) -> None:
        """Apply flags provided via prompt without persisting them."""
        if flags:
            self.set_flags(flags, persist=False)
        else:
            self.clear_flags()

    def clear_flags(self) -> None:
        """Clear all pinned flags from memory and the widget."""
        self.pinned_flags.clear()
        self._widget.clear()
        self._update_canvas_text()

    def get_active_flags(self, flags: Dict[str, bool]) -> Dict[str, bool]:
        """Return a subset containing only active flags."""
        return {name: value for name, value in (flags or {}).items() if value}

    def get_current_behavior_text(self, flags: Dict[str, bool]) -> str:
        """Return a comma-separated list of active behaviors."""
        active = self.get_active_flags(flags)
        return ",".join(sorted(active.keys()))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _handle_row_selected(self, flag_name: str) -> None:
        self._window.event_type = flag_name
        self._window._update_modifier_controls_for_behavior(flag_name)

    def _handle_flag_toggled(self, flag_name: str, state: bool) -> None:
        if state:
            self.start_flag(flag_name, record_event=True)
        else:
            self.end_flag(flag_name, record_event=True)

    def _update_canvas_text(self) -> None:
        text = self.get_current_behavior_text(self.pinned_flags)
        self._window.canvas.setBehaviorText(text)

    def _load_from_disk(self) -> Dict[str, bool]:
        try:
            with self._config_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to load pinned flags config: %s", exc)
            return {}

        names: Iterable[str] = config.get("pinned_flags", [])
        unique_names = []
        for name in names:
            if name and name not in unique_names:
                unique_names.append(name)
        return {name: False for name in unique_names}

    def _persist_flags(self, names: Iterable[str]) -> None:
        unique_names = []
        for name in names:
            if name and name not in unique_names:
                unique_names.append(name)

        config = {}
        if self._config_path.exists():
            try:
                with self._config_path.open("r", encoding="utf-8") as file:
                    config = yaml.safe_load(file) or {}
            except yaml.YAMLError as exc:
                logger.error("Failed to read pinned flags config: %s", exc)
                config = {}

        config["pinned_flags"] = unique_names
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._config_path.open("w", encoding="utf-8") as file:
                yaml.dump(config, file, default_flow_style=False)
        except Exception as exc:
            logger.error("Failed to persist pinned flags: %s", exc)
