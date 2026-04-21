from __future__ import annotations

from typing import Dict, Optional

from qtpy import QtWidgets

from annolid.gui.widgets.flags import FlagTableWidget


class FlagsOverlayMixin:
    """Flag loading and behavior-overlay synchronization helpers."""

    def loadFlags(self, flags):
        """Delegate flag loading to the flags controller."""
        from annolid.utils.labelme_flags import sanitize_labelme_flags

        self.flags_controller.load_flags(sanitize_labelme_flags(flags))

    @property
    def pinned_flags(self):
        if hasattr(self, "flags_controller"):
            return self.flags_controller.pinned_flags
        return getattr(self, "_pending_pinned_flags", {})

    @pinned_flags.setter
    def pinned_flags(self, value):
        if hasattr(self, "flags_controller"):
            self.flags_controller.set_flags(value or {}, persist=False)
        else:
            self.__dict__["_pending_pinned_flags"] = value or {}

    def _update_pinned_flag(self, behavior: str, active: bool) -> None:
        """Update a single pinned flag through the controller-backed sync path."""
        behavior = str(behavior).strip()
        if not behavior:
            return
        current = dict(self.pinned_flags or {})
        current[behavior] = bool(active)
        self.loadFlags(current)

    def _refresh_behavior_overlay(self, frame_number: Optional[int] = None) -> None:
        """Synchronize canvas label and flag widget with timeline behaviors."""
        target_frame = self.frame_number if frame_number is None else frame_number
        try:
            target_frame = int(target_frame)
        except Exception:
            target_frame = int(getattr(self, "frame_number", 0) or 0)
        active_behaviors = self.behavior_controller.active_behaviors(target_frame)
        overlay_text = ",".join(sorted(active_behaviors)) if active_behaviors else None
        # Keep canvas text in sync even when active-behavior signature has not changed.
        self.canvas.setBehaviorText(overlay_text)

        signature = tuple(sorted(str(name) for name in active_behaviors))
        if signature == getattr(self, "_last_behavior_overlay_signature", None):
            return
        setattr(self, "_last_behavior_overlay_signature", signature)

        current_flags: Dict[str, bool] = {}
        table = self.flag_widget._table
        for row in range(table.rowCount()):
            name_widget = table.cellWidget(row, FlagTableWidget.COLUMN_NAME)
            value_widget = table.cellWidget(row, FlagTableWidget.COLUMN_ACTIVE)
            if isinstance(name_widget, QtWidgets.QLineEdit) and isinstance(
                value_widget, QtWidgets.QCheckBox
            ):
                name = name_widget.text().strip()
                if name:
                    current_flags[name] = value_widget.isChecked()

        for behavior in sorted(self.behavior_controller.behavior_names):
            current_flags[behavior] = behavior in active_behaviors

        if current_flags:
            self.loadFlags(current_flags)
