from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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

    def _behavior_label_skipped_overlay_record_list(self) -> List[Dict[str, Any]]:
        """Return cached display-only behavior-label records for skipped segments."""
        video_file = str(getattr(self, "video_file", "") or "").strip()
        cached_video = str(
            getattr(self, "_behavior_label_skipped_overlay_video", "") or ""
        )
        cached_records = list(
            getattr(self, "_behavior_label_skipped_overlay_records", []) or []
        )
        cached_mtime = getattr(self, "_behavior_label_skipped_overlay_mtime_ns", None)
        if not video_file:
            return [
                dict(record) for record in cached_records if isinstance(record, dict)
            ]

        try:
            from annolid.behavior.segment_labeling import (
                behavior_segment_labeling_log_path,
                normalize_behavior_segment_prediction_for_log,
            )

            log_path = behavior_segment_labeling_log_path(video_file)
            mtime_ns = log_path.stat().st_mtime_ns if log_path.exists() else None
            if cached_video == video_file and cached_mtime == mtime_ns:
                return [
                    dict(record)
                    for record in cached_records
                    if isinstance(record, dict)
                ]
            records: List[Dict[str, Any]] = []
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                for raw_record in list(payload.get("skipped_predictions") or []):
                    if not isinstance(raw_record, dict):
                        continue
                    try:
                        records.append(
                            normalize_behavior_segment_prediction_for_log(raw_record)
                        )
                    except Exception:
                        continue
            setattr(self, "_behavior_label_skipped_overlay_video", video_file)
            setattr(self, "_behavior_label_skipped_overlay_mtime_ns", mtime_ns)
            setattr(self, "_behavior_label_skipped_overlay_records", records)
            return records
        except Exception:
            return [
                dict(record) for record in cached_records if isinstance(record, dict)
            ]

    def _behavior_label_skipped_overlay_labels_for_frame(
        self, frame_number: int
    ) -> List[str]:
        """Return display labels from skipped model outputs for one frame."""
        labels: List[str] = []
        for record in self._behavior_label_skipped_overlay_record_list():
            try:
                start_frame = int(record.get("start_frame") or 0)
                end_frame = int(record.get("end_frame") or start_frame)
            except Exception:
                continue
            if not (start_frame <= int(frame_number) <= end_frame):
                continue
            label = str(
                record.get("classification") or record.get("label") or ""
            ).strip()
            visual_evidence = record.get("visual_evidence")
            model_label = ""
            if isinstance(visual_evidence, dict):
                model_label = str(visual_evidence.get("model_label") or "").strip()
            if not model_label:
                model_label = str(record.get("model_label") or "").strip()
            if model_label and label in {"no_behavior", "unclassified"}:
                label = model_label
            if label:
                labels.append(label)
        return sorted(set(labels))

    def _refresh_behavior_overlay(self, frame_number: Optional[int] = None) -> None:
        """Synchronize canvas label and flag widget with timeline behaviors."""
        target_frame = self.frame_number if frame_number is None else frame_number
        try:
            target_frame = int(target_frame)
        except Exception:
            target_frame = int(getattr(self, "frame_number", 0) or 0)
        active_behaviors = set(self.behavior_controller.active_behaviors(target_frame))
        active_overlay_labels = {
            str(name).strip() for name in active_behaviors if str(name).strip()
        }
        overlay_labels = sorted(
            active_overlay_labels
            | set(self._behavior_label_skipped_overlay_labels_for_frame(target_frame))
        )
        overlay_text = ",".join(overlay_labels) if overlay_labels else None
        # Keep canvas text in sync even when active-behavior signature has not changed.
        self.canvas.setBehaviorText(overlay_text)

        signature = tuple(str(name) for name in overlay_labels)
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
