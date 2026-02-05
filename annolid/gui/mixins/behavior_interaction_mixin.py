from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.core.behavior.spec import ProjectSchema
from annolid.gui.behavior_controller import BehaviorEvent
from annolid.utils.annotation_store import AnnotationStore
from annolid.utils.logger import logger


class BehaviorInteractionMixin:
    """Behavior controls, marks, and keyboard interaction helpers."""

    def handle_uniq_label_list_selection_change(self):
        selected_items = self.uniqLabelList.selectedItems()
        if self.seekbar is None:
            return
        if selected_items:
            self.add_highlighted_mark()
        else:
            self.add_highlighted_mark(mark_type="event_end", color="red")

    def _estimate_recording_time(self, frame_number: int) -> Optional[float]:
        """Approximate the recording timestamp (seconds) for a frame."""
        fps = getattr(self, "fps", None)
        if fps and fps > 0:
            return frame_number / float(fps)
        return frame_number / 29.97 if frame_number is not None else None

    def record_behavior_event(
        self,
        behavior: str,
        event_label: str,
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        trial_time: Optional[float] = None,
        subject: Optional[str] = None,
        modifiers: Optional[Iterable[str]] = None,
        highlight: bool = True,
    ) -> Optional[BehaviorEvent]:
        if frame_number is None:
            frame_number = self.frame_number
        if timestamp is None:
            timestamp = self._estimate_recording_time(frame_number)
        if trial_time is None:
            trial_time = timestamp

        auto_subject = False
        if subject is None:
            subject = self._subject_from_selected_shape()
            auto_subject = subject is not None
        if subject is None:
            subject = self._current_subject_name()

        if modifiers is None:
            modifiers = self._selected_modifiers_for_behavior(behavior)
        if not modifiers:
            modifiers = self._default_modifiers_for_behavior(behavior)

        category_label: Optional[str] = None
        if self.project_schema:
            behavior_def = self.project_schema.behavior_map().get(behavior)
            if behavior_def and behavior_def.category_id:
                category = self.project_schema.category_map().get(
                    behavior_def.category_id
                )
                if category:
                    category_label = category.name or category.id

        event = self.behavior_controller.record_event(
            behavior,
            event_label,
            frame_number,
            timestamp=timestamp,
            trial_time=trial_time,
            subject=subject,
            modifiers=modifiers,
            category=category_label,
            highlight=highlight,
        )
        if event is None:
            logger.warning(
                "Unrecognized behavior event label '%s' for '%s'.",
                event_label,
                behavior,
            )
            return None

        if auto_subject:
            self.statusBar().showMessage(
                self.tr("Auto-selected subject '%s' from polygon selection") % subject,
                2500,
            )

        self.pinned_flags.setdefault(behavior, False)
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.append_event(event, fps=fps_for_log)
        return event

    def _populate_behavior_controls_from_schema(
        self, schema: Optional[ProjectSchema]
    ) -> None:
        if not hasattr(self, "behavior_controls_widget"):
            return

        self._behavior_modifier_state.clear()

        if schema is None:
            self.behavior_controls_widget.clear()
            self._active_subject_name = None
            return

        stored_subject = self._active_subject_name or self.settings.value(
            "behavior/last_subject", type=str
        )
        subjects = list(schema.subjects)
        self.behavior_controls_widget.set_subjects(subjects, selected=stored_subject)
        self._active_subject_name = self.behavior_controls_widget.current_subject()

        self.behavior_controls_widget.set_modifiers(list(schema.modifiers))
        for behavior in schema.behaviors:
            if behavior.modifier_ids:
                self._behavior_modifier_state[behavior.code] = set(
                    behavior.modifier_ids
                )

        self.behavior_controls_widget.set_category_badge(None, None)
        self.behavior_controls_widget.set_modifier_states(
            [],
            allowed=self._modifier_ids_from_schema(),
        )
        self.behavior_controls_widget.show_warning(None)
        self._update_modifier_controls_for_behavior(self.event_type)

    def _sync_behavior_flags_from_schema(self, schema: Optional[ProjectSchema]) -> None:
        if schema is None or not hasattr(self, "flag_widget"):
            return
        current = dict(self.pinned_flags or {})
        flags: Dict[str, bool] = {}
        for behavior in schema.behaviors:
            if not behavior.code:
                continue
            flags[str(behavior.code)] = bool(current.get(behavior.code, False))
        self.loadFlags(flags)

    def _modifier_ids_from_schema(self) -> Set[str]:
        if not self.project_schema:
            return set()
        return {
            modifier.id for modifier in self.project_schema.modifiers if modifier.id
        }

    def _allowed_modifiers_for_behavior(self, behavior: Optional[str]) -> Set[str]:
        if not self.project_schema:
            return set()
        schema_modifiers = self._modifier_ids_from_schema()
        if not behavior:
            return schema_modifiers
        behavior_def = self.project_schema.behavior_map().get(behavior)
        if behavior_def and behavior_def.modifier_ids:
            return {
                modifier_id
                for modifier_id in behavior_def.modifier_ids
                if modifier_id in schema_modifiers
            }
        return schema_modifiers

    def _update_behavior_conflict_warning(self, behavior: Optional[str]) -> None:
        if not hasattr(self, "behavior_controls_widget"):
            return
        if not behavior or not self.project_schema:
            self.behavior_controls_widget.show_warning(None)
            return

        behavior_def = self.project_schema.behavior_map().get(behavior)
        if not behavior_def or not behavior_def.exclusive_with:
            self.behavior_controls_widget.show_warning(None)
            return

        active_conflicts = [
            name
            for name, value in (self.flags_controller.pinned_flags or {}).items()
            if value and name in behavior_def.exclusive_with and name != behavior
        ]
        if active_conflicts:
            message = self.tr(
                "Behavior '%s' excludes %s. Stop them before recording."
            ) % (
                behavior,
                ", ".join(sorted(active_conflicts)),
            )
            self.behavior_controls_widget.show_warning(message)
            self.statusBar().showMessage(message, 4000)
        else:
            self.behavior_controls_widget.show_warning(None)

    def _update_modifier_controls_for_behavior(self, behavior: Optional[str]) -> None:
        if not hasattr(self, "behavior_controls_widget"):
            return

        schema = self.project_schema
        if schema is None:
            self.behavior_controls_widget.clear()
            return

        allowed_modifiers = self._allowed_modifiers_for_behavior(behavior)
        if not behavior:
            self.behavior_controls_widget.set_category_badge(None, None)
            self.behavior_controls_widget.set_modifier_states(
                [],
                allowed=allowed_modifiers,
            )
            self.behavior_controls_widget.show_warning(None)
            return

        behavior_def = schema.behavior_map().get(behavior)
        category_label: Optional[str] = None
        category_color: Optional[str] = None
        if behavior_def and behavior_def.category_id:
            category = schema.category_map().get(behavior_def.category_id)
            if category:
                category_label = category.name or category.id
                category_color = category.color

        selected_modifiers = self._behavior_modifier_state.get(behavior)
        if selected_modifiers is None:
            if behavior_def and behavior_def.modifier_ids:
                selected_modifiers = set(
                    modifier_id
                    for modifier_id in behavior_def.modifier_ids
                    if modifier_id in allowed_modifiers
                )
                self._behavior_modifier_state[behavior] = set(selected_modifiers)
            else:
                selected_modifiers = set()
        else:
            selected_modifiers = {
                modifier_id
                for modifier_id in selected_modifiers
                if modifier_id in allowed_modifiers
            }
            self._behavior_modifier_state[behavior] = set(selected_modifiers)
            if not selected_modifiers and behavior_def and behavior_def.modifier_ids:
                selected_modifiers = {
                    modifier_id
                    for modifier_id in behavior_def.modifier_ids
                    if modifier_id in allowed_modifiers
                }
                self._behavior_modifier_state[behavior] = set(selected_modifiers)

        self.behavior_controls_widget.set_category_badge(
            category_label,
            category_color,
        )
        self.behavior_controls_widget.set_modifier_states(
            selected_modifiers,
            allowed=allowed_modifiers,
        )
        self._update_behavior_conflict_warning(behavior)

    def _on_active_subject_changed(self, subject_name: str) -> None:
        subject_name = subject_name.strip()
        if subject_name:
            self._active_subject_name = subject_name
            self.settings.setValue("behavior/last_subject", subject_name)

    def _on_modifier_toggled(self, modifier_id: str, state: bool) -> None:
        behavior = self.event_type
        if not behavior:
            return
        modifier_set = self._behavior_modifier_state.setdefault(behavior, set())
        if state:
            modifier_set.add(modifier_id)
        else:
            modifier_set.discard(modifier_id)

    def _subject_from_selected_shape(self) -> Optional[str]:
        selected = getattr(self.canvas, "selectedShapes", None)
        if not selected:
            return None
        label = selected[0].label
        if not label:
            return None
        label_name = str(label).strip()
        if not label_name:
            return None
        if self.project_schema:
            candidates = {subj.name: subj.name for subj in self.project_schema.subjects}
            candidates.update(
                {subj.id: subj.name or subj.id for subj in self.project_schema.subjects}
            )
            lowered = {key.lower(): value for key, value in candidates.items()}
            match = lowered.get(label_name.lower())
            if match:
                return match
        return label_name

    def _current_subject_name(self) -> str:
        if self._active_subject_name:
            return self._active_subject_name
        if hasattr(self, "behavior_controls_widget"):
            subject = self.behavior_controls_widget.current_subject()
            if subject:
                self._active_subject_name = subject
                return subject
        if self.project_schema and self.project_schema.subjects:
            subject = self.project_schema.subjects[0]
            return subject.name or subject.id or "Subject 1"
        return "Subject 1"

    def _selected_modifiers_for_behavior(self, behavior: Optional[str]) -> List[str]:
        if not behavior:
            return []
        selected = self._behavior_modifier_state.get(behavior)
        if selected:
            return list(selected)
        defaults = self._default_modifiers_for_behavior(behavior)
        if defaults:
            allowed = self._allowed_modifiers_for_behavior(behavior)
            if allowed:
                defaults = [
                    modifier_id for modifier_id in defaults if modifier_id in allowed
                ]
        if defaults:
            self._behavior_modifier_state[behavior] = set(defaults)
            return list(defaults)
        return []

    def _default_modifiers_for_behavior(self, behavior: Optional[str]) -> List[str]:
        if not behavior or not self.project_schema:
            return []
        behavior_def = self.project_schema.behavior_map().get(behavior)
        if behavior_def and behavior_def.modifier_ids:
            return list(behavior_def.modifier_ids)
        return []

    def _jump_to_frame_from_log(self, frame: int) -> None:
        if self.seekbar is None or self.num_frames is None:
            return
        target = max(0, min(frame, self.num_frames - 1))
        if self.seekbar.value() != target:
            self.seekbar.setValue(target)
        else:
            self.set_frame_number(target)
        self._update_embedding_query_frame()

    def _update_embedding_query_frame(self) -> None:
        if not hasattr(self, "embedding_search_widget"):
            return
        try:
            self.embedding_search_widget.set_video_path(
                Path(self.video_file) if getattr(self, "video_file", None) else None
            )
        except Exception:
            pass
        try:
            self.embedding_search_widget.set_annotation_dir(
                Path(self.annotation_dir)
                if getattr(self, "annotation_dir", None)
                else None
            )
        except Exception:
            pass
        try:
            self.embedding_search_widget.set_query_frame_index(int(self.frame_number))
        except Exception:
            pass

    def _mark_similar_frames_from_search(self, frames: list[int]) -> None:
        if not frames:
            return
        for frame in sorted(set(int(f) for f in frames)):
            try:
                self.behavior_controller.add_generic_mark(
                    frame, mark_type="frame_search", color="magenta"
                )
            except Exception:
                continue
        try:
            if self.seekbar is not None:
                self.seekbar._update_visual_positions()
        except Exception:
            pass

    def _clear_similar_frame_marks(self) -> None:
        try:
            self.behavior_controller.clear_generic_marks(mark_type="frame_search")
        except Exception:
            return
        try:
            if self.seekbar is not None:
                self.seekbar._update_visual_positions()
        except Exception:
            pass

    def _refresh_embedding_file_list(self) -> None:
        if not hasattr(self, "embedding_search_widget"):
            return
        files: list[Path] = []
        base = getattr(self, "video_results_folder", None)
        if base:
            try:
                base_path = Path(base)
                patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
                for pat in patterns:
                    files.extend(sorted(base_path.glob(pat)))
            except Exception:
                files = []
        if files:
            self.embedding_search_widget.set_files(files)
        else:
            self.embedding_search_widget.clear_files()

    def _label_frames_from_search(self, frames: list[int]) -> None:
        if not frames:
            return
        behavior = self.event_type or ""
        if not behavior:
            behavior, ok = QtWidgets.QInputDialog.getText(
                self,
                self.tr("Label Frames"),
                self.tr("Behavior label for selected frames:"),
            )
            if not ok or not behavior.strip():
                return
            behavior = behavior.strip()
        if len(frames) > 50:
            reply = QtWidgets.QMessageBox.question(
                self,
                self.tr("Label Frames"),
                self.tr("Label %s frames with '%s'?") % (len(frames), behavior),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        def _timestamp_provider(frame: int) -> Optional[float]:
            fps = self.fps if self.fps and self.fps > 0 else None
            if fps is None:
                return None
            return float(frame) / float(fps)

        subject = self._current_subject_name()
        for frame in sorted(set(int(f) for f in frames)):
            self.behavior_controller.create_interval(
                behavior=str(behavior),
                start_frame=frame,
                end_frame=frame,
                subject=subject,
                timestamp_provider=_timestamp_provider,
            )
        self._refresh_behavior_log()

    def _prepare_anchor_rerun(self) -> None:
        if not self.video_results_folder:
            return
        store = AnnotationStore.for_frame_path(
            Path(self.video_results_folder)
            / f"{self.video_results_folder.name}_000000000.json"
        )
        anchors = self._collect_anchor_frames(store)
        if not anchors:
            return
        anchor_frame = max(anchors)
        store.remove_frames_after(anchor_frame, protected_frames=anchors)
        self.statusBar().showMessage(
            self.tr("Anchor rerun: cleared frames after %s") % anchor_frame, 4000
        )

    def _collect_anchor_frames(self, store: AnnotationStore) -> list[int]:
        anchors: list[int] = []
        for frame in store.iter_frames():
            record = store.get_frame(frame) or {}
            shapes = record.get("shapes") or []
            other = record.get("otherData") or {}
            behavior_block = other.get("annolid_behavior") or {}
            events = behavior_block.get("events") or []
            confirmed_events = [
                e for e in events if isinstance(e, dict) and e.get("confirmed", True)
            ]
            if shapes or confirmed_events:
                try:
                    anchors.append(int(frame))
                except Exception:
                    continue
        return sorted(set(anchors))

    def undo_last_behavior_event(self) -> None:
        event = self.behavior_controller.pop_last_event()
        if event is None:
            return
        self.behavior_log_widget.remove_event(event.mark_key)
        if self.seekbar is not None:
            self.seekbar.setTickMarks()
        self.canvas.setBehaviorText(None)

    def _clear_behavior_events_from_log(self) -> None:
        if not self.behavior_controller.events_count:
            self.behavior_log_widget.clear()
            return
        self.behavior_controller.clear_behavior_data()
        self.behavior_log_widget.clear()
        if self.seekbar is not None:
            self.seekbar.setTickMarks()
        self.canvas.setBehaviorText(None)
        if self.pinned_flags:
            for behavior in list(self.pinned_flags.keys()):
                self.pinned_flags[behavior] = False
            self.loadFlags(self.pinned_flags)

    def add_highlighted_mark(
        self, val=None, mark_type=None, color=None, init_load=False
    ):
        """Add a non-behavior highlight mark to the slider."""
        if self.seekbar is None:
            return None

        frame_val = self.frame_number if val is None else int(val)
        return self.behavior_controller.add_generic_mark(
            frame_val,
            mark_type=mark_type,
            color=color,
            init_load=init_load,
        )

    def remove_highlighted_mark(self):
        if self.seekbar is None:
            return

        removed_behavior_keys: List[Tuple[int, str, str]] = []

        if self.behavior_controller.highlighted_mark is not None:
            if self.isPlaying:
                self.togglePlay()
            removed = self.behavior_controller.remove_highlighted_mark()
            if removed:
                if removed[0] == "behavior":
                    removed_behavior_keys.append(removed[1])  # type: ignore[index]
                if self.event_type in self.pinned_flags:
                    self.pinned_flags[self.event_type] = False
        elif self.seekbar.isMarkedVal(self.frame_number):
            removed = self.behavior_controller.remove_marks_at_value(self.frame_number)
            for kind, key in removed:
                if kind == "behavior":
                    removed_behavior_keys.append(key)  # type: ignore[arg-type]
            if removed and self.event_type in self.pinned_flags:
                self.pinned_flags[self.event_type] = False
        else:
            current_val = self.seekbar.value()
            removed_any = False
            local_removed_keys: List[Tuple[int, str, str]] = []
            for mark in list(self.seekbar.getMarks()):
                if mark.val == current_val:
                    result = self.behavior_controller.remove_mark_instance(mark)
                    removed_any = removed_any or bool(result)
                    if result and result[0] == "behavior":
                        local_removed_keys.append(result[1])  # type: ignore[arg-type]
            if removed_any:
                self.seekbar.setTickMarks()
                if self.event_type in self.pinned_flags:
                    self.pinned_flags[self.event_type] = False
                removed_behavior_keys.extend(local_removed_keys)

        for key in removed_behavior_keys:
            self.behavior_log_widget.remove_event(key)
        self.canvas.setBehaviorText(None)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.seekbar is not None:
            if event.key() == Qt.Key_Right:
                next_pos = self.seekbar.value() + 1
                self.seekbar.setValue(next_pos)
                self.seekbar.valueChanged.emit(next_pos)
            elif event.key() == Qt.Key_Left:
                prev_pos = self.seekbar.value() - 1
                self.seekbar.setValue(prev_pos)
                self.seekbar.valueChanged.emit(prev_pos)
            elif event.key() == Qt.Key_0:
                self.seekbar.setValue(0)
            elif event.key() == Qt.Key_Space:
                self.togglePlay()
            elif event.key() == Qt.Key_S:
                if self.event_type is None:
                    self.add_highlighted_mark(
                        self.frame_number,
                        mark_type=self._config["events"]["start"],
                    )
                else:
                    self.record_behavior_event(
                        self.event_type, "start", frame_number=self.frame_number
                    )
            elif event.key() == Qt.Key_E:
                if self.event_type is None:
                    self.add_highlighted_mark(
                        self.frame_number,
                        mark_type=self._config["events"]["end"],
                        color="red",
                    )
                else:
                    self.record_behavior_event(
                        self.event_type, "end", frame_number=self.frame_number
                    )
                    self.flags_controller.end_flag(self.event_type, record_event=False)
            elif event.key() == Qt.Key_R:
                self.remove_highlighted_mark()
            elif event.key() == Qt.Key_Q:
                self.seekbar.setValue(self.seekbar._val_max)
            elif event.key() == Qt.Key_1 or event.key() == Qt.Key_I:
                self.update_step_size(1)
            elif event.key() == Qt.Key_2 or event.key() == Qt.Key_F:
                self.update_step_size(self.step_size + 10)
            elif event.key() == Qt.Key_B:
                self.update_step_size(self.step_size - 10)
            elif event.key() == Qt.Key_M:
                self.update_step_size(self.step_size - 1)
            elif event.key() == Qt.Key_P:
                self.update_step_size(self.step_size + 1)
            else:
                event.ignore()

    def saveTimestampList(self):
        default_timestamp_csv_file = (
            str(os.path.dirname(self.filename)) + "_timestamps.csv"
        )
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setDefaultSuffix(".csv")
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Timestamps", default_timestamp_csv_file, "CSV files (*.csv)"
        )

        if file_path:
            rows = self.behavior_controller.export_rows(
                timestamp_fallback=lambda evt: self._estimate_recording_time(evt.frame)
            )
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Trial time", "Recording time", "Subject", "Behavior", "Event"]
                )
                for row in rows:
                    writer.writerow(row)

            QtWidgets.QMessageBox.information(
                self, "Timestamps saved", "Timestamps saved successfully!"
            )
