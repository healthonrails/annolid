from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from qtpy import QtWidgets

from annolid.gui.widgets.canvas_overlays import BehaviorOverlayItem
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

    def _should_display_behavior_subjects(self) -> bool:
        """Return True when multiple distinct subjects exist in schema or timeline."""
        subjects: set[str] = set()
        schema = getattr(self, "project_schema", None)
        if schema is not None:
            for subj in getattr(schema, "subjects", None) or ():
                name = str(
                    getattr(subj, "name", None) or getattr(subj, "id", None) or ""
                ).strip()
                if name:
                    subjects.add(name)
        ctrl = getattr(self, "behavior_controller", None)
        distinct_fn = getattr(ctrl, "distinct_subjects", None)
        if callable(distinct_fn):
            try:
                for name in distinct_fn():
                    cleaned = str(name or "").strip()
                    if cleaned:
                        subjects.add(cleaned)
            except Exception:
                pass
        return len(subjects) > 1

    def _active_frame_label_file_flags(self, managed_behaviors: set[str]) -> set[str]:
        """Return active frame-level JSON flags not already managed by the timeline."""
        label_file = getattr(self, "labelFile", None)
        raw_flags = getattr(label_file, "flags", None)
        if not isinstance(raw_flags, dict):
            return set()
        active: set[str] = set()
        for key, value in raw_flags.items():
            if not value:
                continue
            name = str(key or "").strip()
            if name and name not in managed_behaviors:
                active.add(name)
        return active

    def _on_behavior_controller_changed(self) -> None:
        """Synchronize canvas overlay and ethogram badges after timeline mutations."""
        if (
            getattr(self, "canvas", None) is None
            or getattr(self, "flag_widget", None) is None
        ):
            return
        try:
            self._refresh_behavior_overlay()
        except Exception:
            pass
        refresh_badges = getattr(self, "_refresh_ethogram_active_badges", None)
        if callable(refresh_badges):
            try:
                refresh_badges()
            except Exception:
                pass

    def _refresh_behavior_overlay(self, frame_number: Optional[int] = None) -> None:
        """Synchronize canvas label badges and flag widget with timeline behaviors."""
        if getattr(self, "_refreshing_behavior_overlay", False):
            return
        setattr(self, "_refreshing_behavior_overlay", True)
        try:
            target_frame = self.frame_number if frame_number is None else frame_number
            try:
                target_frame = int(target_frame)
            except Exception:
                target_frame = int(getattr(self, "frame_number", 0) or 0)

            controller = self.behavior_controller
            active_behaviors = set(controller.active_behaviors(target_frame))
            active_overlay_labels = {
                str(name).strip() for name in active_behaviors if str(name).strip()
            }
            managed_names = {
                str(name).strip()
                for name in (getattr(controller, "behavior_names", ()) or ())
                if str(name).strip()
            }

            show_subjects = self._should_display_behavior_subjects()
            active_subjects_map: Dict[str, List[str]] = {}
            active_subjects_fn = getattr(controller, "active_behavior_subjects", None)
            if show_subjects and callable(active_subjects_fn):
                try:
                    raw_map = active_subjects_fn(target_frame) or {}
                    for beh, subjs in raw_map.items():
                        key = str(beh or "").strip()
                        if key:
                            active_subjects_map[key] = [
                                str(s).strip() for s in (subjs or ()) if str(s).strip()
                            ]
                except Exception:
                    active_subjects_map = {}

            point_subjects_map: Dict[str, List[str]] = {}
            point_fn = getattr(controller, "point_behaviors_at_frame", None)
            if callable(point_fn):
                try:
                    for beh, subj in point_fn(target_frame) or ():
                        key = str(beh or "").strip()
                        if not key:
                            continue
                        bucket = point_subjects_map.setdefault(key, [])
                        cleaned_subj = str(subj).strip() if subj else ""
                        if cleaned_subj and cleaned_subj not in bucket:
                            bucket.append(cleaned_subj)
                except Exception:
                    point_subjects_map = {}

            file_only_active = self._active_frame_label_file_flags(managed_names)
            skipped_labels = self._behavior_label_skipped_overlay_labels_for_frame(
                target_frame
            )
            point_labels = set(point_subjects_map.keys())
            state_labels = active_overlay_labels | file_only_active
            overlay_labels = sorted(state_labels | point_labels | set(skipped_labels))
            overlay_text = ",".join(overlay_labels) if overlay_labels else None

            overlay_items: List[BehaviorOverlayItem] = []
            seen_overlay_labels: set[str] = set()
            for label in sorted(state_labels):
                seen_overlay_labels.add(label)
                subjs = active_subjects_map.get(label) if show_subjects else None
                subject_text = ", ".join(subjs) if subjs else None
                overlay_items.append(
                    BehaviorOverlayItem(
                        label=label,
                        kind="state",
                        subject=subject_text,
                    )
                )
            for label in sorted(point_labels):
                if label in seen_overlay_labels:
                    continue
                seen_overlay_labels.add(label)
                subjs = point_subjects_map.get(label) if show_subjects else None
                subject_text = ", ".join(subjs) if subjs else None
                overlay_items.append(
                    BehaviorOverlayItem(
                        label=label,
                        kind="point",
                        subject=subject_text,
                    )
                )
            for label in sorted(set(skipped_labels)):
                if label in seen_overlay_labels:
                    continue
                seen_overlay_labels.add(label)
                overlay_items.append(BehaviorOverlayItem(label=label, kind="skipped"))

            signature = (
                tuple(sorted(active_overlay_labels)),
                tuple(sorted(file_only_active)),
                tuple(sorted(managed_names)),
                tuple(str(name) for name in overlay_labels),
            )
            if signature != getattr(self, "_last_behavior_overlay_signature", None):
                setattr(self, "_last_behavior_overlay_signature", signature)

                current_flags: Dict[str, bool] = {}
                flag_widget = getattr(self, "flag_widget", None)
                table = getattr(flag_widget, "_table", None)
                if table is not None:
                    for row in range(table.rowCount()):
                        name_widget = table.cellWidget(row, FlagTableWidget.COLUMN_NAME)
                        value_widget = table.cellWidget(
                            row, FlagTableWidget.COLUMN_ACTIVE
                        )
                        if isinstance(name_widget, QtWidgets.QLineEdit) and isinstance(
                            value_widget, QtWidgets.QCheckBox
                        ):
                            name = name_widget.text().strip()
                            if name:
                                current_flags[name] = value_widget.isChecked()

                for behavior in sorted(controller.behavior_names):
                    current_flags[behavior] = behavior in active_behaviors

                for flag_name in sorted(file_only_active):
                    current_flags[flag_name] = True

                if current_flags:
                    setattr(self, "_suppress_flags_canvas_sync", True)
                    try:
                        self.loadFlags(current_flags)
                    finally:
                        setattr(self, "_suppress_flags_canvas_sync", False)

            # Update canvas text and rich overlay metadata AFTER flag-table sync
            # so loadFlags() cannot clobber skipped, point, or frame-level labels.
            self.canvas.setBehaviorText(overlay_text)
            if hasattr(self.canvas, "setBehaviorOverlayItems"):
                self.canvas.setBehaviorOverlayItems(overlay_items or None)
        finally:
            setattr(self, "_refreshing_behavior_overlay", False)
