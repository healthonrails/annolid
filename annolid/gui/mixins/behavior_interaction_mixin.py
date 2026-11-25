from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.domain import ProjectSchema
from annolid.gui.behavior_controller import BehaviorEvent
from annolid.infrastructure import AnnotationStore
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
        comment: Optional[str] = None,
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

        if comment is None and hasattr(self, "behavior_controls_widget"):
            try:
                taken = self.behavior_controls_widget.take_event_comment()
                comment = taken or None
            except Exception:
                comment = None

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
            comment=comment,
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

        if event.event == "start":
            pinned_state = True
        elif event.event == "end":
            pinned_state = False
        else:
            pinned_state = bool(self.pinned_flags.get(behavior, False))
        self._update_pinned_flag(behavior, pinned_state)
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.append_event(event, fps=fps_for_log)
        self._refresh_ethogram_active_badges(frame_number)
        return event

    def _refresh_ethogram_active_badges(
        self, frame_number: Optional[int] = None
    ) -> None:
        controls = getattr(self, "behavior_controls_widget", None)
        ctrl = getattr(self, "behavior_controller", None)
        if (
            controls is None
            or ctrl is None
            or not hasattr(controls, "set_active_behaviors")
        ):
            return
        frame = (
            int(frame_number)
            if frame_number is not None
            else int(getattr(self, "frame_number", 0) or 0)
        )
        subj = self._current_subject_name()
        active = ctrl.active_behaviors(frame, subject=subj)
        controls.set_active_behaviors(active)

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

        if hasattr(self.behavior_controls_widget, "set_behaviors"):
            self.behavior_controls_widget.set_behaviors(
                list(schema.behaviors),
                categories=list(schema.categories),
            )

        self.behavior_controls_widget.set_modifiers(list(schema.modifiers))
        self.behavior_controls_widget.set_category_badge(None, None)
        self.behavior_controls_widget.set_modifier_states(
            [],
            allowed=self._modifier_ids_from_schema(),
        )
        self.behavior_controls_widget.show_warning(None)
        self._update_modifier_controls_for_behavior(self.event_type)
        self._refresh_ethogram_active_badges()

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

        selected_modifiers = {
            modifier_id
            for modifier_id in (self._behavior_modifier_state.get(behavior) or set())
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
            self._refresh_ethogram_active_badges()

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
            return sorted(selected)
        return []

    def set_playback_speed(self, speed: float) -> None:
        self._playback_speed = max(0.05, min(16.0, float(speed or 1.0)))
        timer = getattr(self, "timer", None)
        if getattr(self, "isPlaying", False) and timer is not None and timer.isActive():
            fps = float(getattr(self, "fps", 0.0) or 0.0)
            if fps > 0:
                timer.setInterval(
                    max(5, int(round(1000.0 / (fps * self._playback_speed))))
                )
            else:
                timer.setInterval(max(5, int(round(20.0 / self._playback_speed))))
            self._play_synchronized_video_audio(getattr(self, "frame_number", 0))
        self.statusBar().showMessage(
            self.tr("Playback speed: %gx") % self._playback_speed, 2500
        )

    def step_video_by_seconds(self, delta_seconds: float) -> None:
        if (
            getattr(self, "seekbar", None) is None
            or getattr(self, "num_frames", None) is None
        ):
            return
        fps = float(getattr(self, "fps", 0.0) or 29.97)
        delta_frames = int(round(float(delta_seconds) * fps))
        target = max(
            0,
            min(int(self.frame_number) + delta_frames, int(self.num_frames) - 1),
        )
        self._jump_to_frame_from_log(target)

    def trigger_ethogram_behavior(self, behavior_code: str) -> None:
        """BORIS-style behavior trigger: fires Point events or toggles State Start/Stop."""
        code = str(behavior_code or "").strip()
        if not code:
            return
        self.event_type = code
        self._update_modifier_controls_for_behavior(code)

        if getattr(self, "seekbar", None) is None:
            self.statusBar().showMessage(
                self.tr("Selected behavior '%s' (open a video to record events)")
                % code,
                3000,
            )
            return

        schema = getattr(self, "project_schema", None)
        bdef = schema.behavior_map().get(code) if schema is not None else None
        subj = self._current_subject_name()
        frame = int(getattr(self, "frame_number", 0) or 0)

        if bdef is not None and not bdef.is_state:
            self.record_behavior_event(code, "point", frame_number=frame, subject=subj)
            self.statusBar().showMessage(
                self.tr("Recorded POINT '%s' for %s at frame %d") % (code, subj, frame),
                2500,
            )
            return

        is_active = self.behavior_controller.is_behavior_active(
            frame, code, subject=subj
        )
        flags_ctrl = getattr(self, "flags_controller", None)
        if is_active:
            if flags_ctrl is not None:
                flags_ctrl.end_flag(code, record_event=True)
            else:
                self.record_behavior_event(
                    code, "end", frame_number=frame, subject=subj
                )
            self.statusBar().showMessage(
                self.tr("Stopped '%s' for %s at frame %d") % (code, subj, frame),
                2500,
            )
        else:
            if flags_ctrl is not None:
                flags_ctrl.start_flag(code, record_event=True)
            else:
                self.record_behavior_event(
                    code, "start", frame_number=frame, subject=subj
                )
            if self.behavior_controller.is_behavior_active(frame, code, subject=subj):
                self.statusBar().showMessage(
                    self.tr("Started '%s' for %s at frame %d") % (code, subj, frame),
                    2500,
                )
        self._refresh_ethogram_active_badges(frame)

    def _handle_boris_coding_keypress(self, event: QtGui.QKeyEvent) -> bool:
        controls = getattr(self, "behavior_controls_widget", None)
        if controls is None or not hasattr(controls, "is_coding_mode_enabled"):
            return False
        if not controls.is_coding_mode_enabled():
            return False
        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(
            focused,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
            ),
        ):
            return False

        key_text = (event.text() or "").strip()
        if not key_text:
            key_text = QtGui.QKeySequence(event.key()).toString().strip()
        if not key_text:
            return False
        norm_key = key_text.lower()

        schema = getattr(self, "project_schema", None)
        if schema is None:
            return False

        # 1) Focal subject shortcut
        for subj in schema.subjects:
            if subj.key_binding and subj.key_binding.strip().lower() == norm_key:
                target_name = subj.name or subj.id
                controls.select_subject(target_name)
                self.statusBar().showMessage(
                    self.tr("Focal subject: %s") % target_name, 2500
                )
                return True

        # 2) Behavior shortcut (State toggle or Point event)
        for beh in schema.behaviors:
            if beh.key_binding and beh.key_binding.strip().lower() == norm_key:
                self.trigger_ethogram_behavior(beh.code)
                return True

        # 3) Applicable modifier shortcut
        allowed = self._allowed_modifiers_for_behavior(self.event_type)
        for mod in schema.modifiers:
            if (
                mod.id in allowed
                and mod.key_binding
                and mod.key_binding.strip().lower() == norm_key
            ):
                btn = controls._modifier_buttons.get(mod.id)
                if btn is not None and btn.isEnabled():
                    btn.toggle()
                    return True

        return False

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
        target = max(0, min(int(frame), int(self.num_frames) - 1))
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
            default_label = ""
            search_widget = getattr(self, "embedding_search_widget", None)
            if search_widget is not None and hasattr(search_widget, "last_query_text"):
                try:
                    default_label = str(search_widget.last_query_text() or "").strip()
                except Exception:
                    default_label = ""
            behavior, ok = QtWidgets.QInputDialog.getText(
                self,
                self.tr("Label Frames"),
                self.tr("Behavior label for selected frames:"),
                text=default_label,
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
        refresh_overlay = getattr(self, "_refresh_behavior_overlay", None)
        if callable(refresh_overlay):
            refresh_overlay()
        else:
            self.canvas.setBehaviorText(None)

    def _clear_behavior_events_from_log(self) -> None:
        if not self.behavior_controller.events_count:
            self.behavior_log_widget.clear()
            return
        self.behavior_controller.clear_behavior_data()
        self.behavior_log_widget.clear()
        if self.seekbar is not None:
            self.seekbar.setTickMarks()
        if self.pinned_flags:
            self.loadFlags({behavior: False for behavior in self.pinned_flags})
        refresh_overlay = getattr(self, "_refresh_behavior_overlay", None)
        if callable(refresh_overlay):
            refresh_overlay()
        else:
            self.canvas.setBehaviorText(None)

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
                    self._update_pinned_flag(self.event_type, False)
        elif self.seekbar.isMarkedVal(self.frame_number):
            removed = self.behavior_controller.remove_marks_at_value(self.frame_number)
            for kind, key in removed:
                if kind == "behavior":
                    removed_behavior_keys.append(key)  # type: ignore[arg-type]
            if removed and self.event_type in self.pinned_flags:
                self._update_pinned_flag(self.event_type, False)
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
                    self._update_pinned_flag(self.event_type, False)
                removed_behavior_keys.extend(local_removed_keys)

        for key in removed_behavior_keys:
            self.behavior_log_widget.remove_event(key)
        refresh_overlay = getattr(self, "_refresh_behavior_overlay", None)
        if callable(refresh_overlay):
            refresh_overlay()
        else:
            self.canvas.setBehaviorText(None)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self._handle_boris_coding_keypress(event):
            event.accept()
            return
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
            if getattr(self, "filename", None)
            else "behavior_timestamps.csv"
        )
        filters = (
            "Annolid Behavior CSV (*.csv);;"
            "BORIS Project (*.boris);;"
            "BORIS Tabular Events CSV (*.tabular.csv);;"
            "BORIS Aggregated Events CSV (*.aggregated.csv)"
        )
        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save / Export Behavior Events",
            default_timestamp_csv_file,
            filters,
        )
        if not file_path:
            return

        detailed = self.behavior_controller.export_detailed_rows(
            timestamp_fallback=lambda evt: self._estimate_recording_time(evt.frame)
        )
        out_path = Path(file_path)
        from annolid.behavior.boris_io import (
            BorisObservation,
            write_annolid_events_csv,
            write_boris_aggregated_csv,
            write_boris_project,
            write_boris_tabular_csv,
        )
        from annolid.behavior.event_utils import parse_event_mapping_rows
        from annolid.domain import default_behavior_spec

        parsed_rows = parse_event_mapping_rows(detailed)
        schema = self.project_schema or default_behavior_spec()
        media_str = str(getattr(self, "video_file", "") or "")
        obs_id = Path(media_str).stem if media_str else out_path.stem

        if out_path.suffix.lower() == ".boris" or "BORIS Project" in selected_filter:
            if out_path.suffix.lower() != ".boris":
                out_path = out_path.with_suffix(".boris")
            sync_w = getattr(self, "boris_sync_widget", None)
            if sync_w is not None and hasattr(sync_w, "export_boris_media_and_plots"):
                by_player, media_offsets, ext_specs = (
                    sync_w.export_boris_media_and_plots(media_str or None)
                )
            else:
                by_player = {"1": (media_str,)} if media_str else {}
                media_offsets = {}
                ext_specs = ()
            all_media = tuple(
                p for paths in by_player.values() for p in paths if p
            ) or ((media_str,) if media_str else ())
            obs = BorisObservation(
                observation_id=obs_id,
                events=tuple(parsed_rows),
                media_files=all_media,
                fps=float(self.fps) if getattr(self, "fps", None) else None,
                media_by_player=by_player,
                media_offsets=media_offsets,
                external_data=ext_specs,
            )
            write_boris_project(
                out_path, schema, observations=[obs], project_name=obs_id
            )
        elif (
            str(out_path).lower().endswith(".tabular.csv")
            or "Tabular" in selected_filter
        ):
            write_boris_tabular_csv(
                parsed_rows,
                out_path,
                observation_id=obs_id,
                media_file=media_str,
                schema=schema,
            )
        elif (
            str(out_path).lower().endswith(".aggregated.csv")
            or "Aggregated" in selected_filter
        ):
            write_boris_aggregated_csv(
                parsed_rows,
                out_path,
                observation_id=obs_id,
                media_file=media_str,
                schema=schema,
            )
        else:
            write_annolid_events_csv(parsed_rows, out_path)

        QtWidgets.QMessageBox.information(
            self, "Events Saved", f"Behavior events saved to:\n{out_path}"
        )

    def import_boris_dialog(self) -> None:
        """Import a .boris project or BORIS CSV/TSV event table into Annolid."""
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import BORIS Project or Events"),
            str(Path(self.video_file).parent)
            if getattr(self, "video_file", None)
            else "",
            self.tr(
                "BORIS Files (*.boris *.csv *.tsv *.json);;BORIS Project (*.boris);;CSV / TSV (*.csv *.tsv)"
            ),
        )
        if not path_str:
            return
        resolved = Path(path_str)
        if resolved.suffix.lower() in {".csv", ".tsv"}:
            self._load_behavior(str(resolved))
            self.statusBar().showMessage(
                self.tr("Loaded behavior events from %s") % resolved.name, 4000
            )
            return

        from annolid.behavior.boris_io import import_boris_project

        try:
            imported = import_boris_project(resolved)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("BORIS Import Error"),
                self.tr("Failed to import BORIS project:\n%s") % exc,
            )
            return

        self.project_schema = imported.schema
        if hasattr(self, "_refresh_behavior_catalog_views"):
            self._refresh_behavior_catalog_views()
        if hasattr(self, "save_behavior_catalog"):
            self.save_behavior_catalog()

        if imported.observations:
            chosen_obs = imported.observations[0]
            if len(imported.observations) > 1:
                obs_ids = [o.observation_id for o in imported.observations]
                choice, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    self.tr("Select BORIS Observation"),
                    self.tr("Load events from observation:"),
                    obs_ids,
                    0,
                    False,
                )
                if not ok:
                    return
                chosen_obs = next(
                    (o for o in imported.observations if o.observation_id == choice),
                    imported.observations[0],
                )
            fps = (
                self.fps
                if getattr(self, "fps", None) and self.fps > 0
                else (chosen_obs.fps or 29.97)
            )
            self.behavior_controller.load_events_from_rows(
                chosen_obs.events,
                time_to_frame=lambda t: int(round(float(t) * float(fps))),
            )
            if getattr(self, "seekbar", None) is not None:
                self.behavior_controller.attach_slider(self.seekbar)
            if hasattr(self, "_refresh_behavior_log"):
                self._refresh_behavior_log()
            self._refresh_ethogram_active_badges()
            refresh_overlay = getattr(self, "_refresh_behavior_overlay", None)
            if callable(refresh_overlay):
                refresh_overlay()
            self._apply_boris_observation_sync_assets(chosen_obs, resolved.parent)

        self.statusBar().showMessage(
            self.tr("Imported BORIS project '%s' (%d behaviors, %d observations)")
            % (
                imported.project_name,
                len(imported.schema.behaviors),
                len(imported.observations),
            ),
            5000,
        )

    def export_boris_dialog(self) -> None:
        """Export current schema and active session events to .boris or BORIS CSV."""
        self.saveTimestampList()

    # ------------------------------------------------------------------ #
    # Synchronized Multi-Camera & External Data Streams Dock
    # ------------------------------------------------------------------ #
    def open_boris_sync_dock(self):
        """Open (lazily creating) the synchronized secondary cameras & external data streams dock."""
        from annolid.behavior.time_budget import extract_behavior_intervals
        from annolid.gui.widgets.boris_sync_dock import BorisSyncWidget

        dock = getattr(self, "boris_sync_dock", None)
        if dock is None:
            dock = QtWidgets.QDockWidget(
                self.tr("Synchronized Cameras & Data Streams"), self
            )
            dock.setObjectName("BorisSyncDock")
            dock.setAllowedAreas(
                Qt.BottomDockWidgetArea
                | Qt.RightDockWidgetArea
                | Qt.LeftDockWidgetArea
                | Qt.TopDockWidgetArea
            )
            sync_widget = BorisSyncWidget(dock)

            def _intervals_provider():
                ctrl = getattr(self, "behavior_controller", None)
                if ctrl is None or getattr(ctrl, "events_count", 0) == 0:
                    return ()
                rows = ctrl.export_detailed_rows(
                    timestamp_fallback=lambda evt: self._estimate_recording_time(
                        evt.frame
                    )
                )
                ivs, _ = extract_behavior_intervals(rows)
                return ivs

            sync_widget.set_intervals_provider(_intervals_provider)
            sync_widget.timeJumpRequested.connect(self._jump_to_time_from_sync_widget)
            sync_widget.audioMixChanged.connect(self._on_boris_audio_mix_changed)
            dock.setWidget(sync_widget)
            self.addDockWidget(Qt.BottomDockWidgetArea, dock)
            self.boris_sync_dock = dock
            self.boris_sync_widget = sync_widget

        dock.show()
        dock.raise_()
        self._sync_boris_media_and_streams()
        return dock

    def _on_boris_audio_mix_changed(self) -> None:
        if bool(getattr(self, "isPlaying", False)):
            self._play_synchronized_video_audio(getattr(self, "frame_number", 0))

    def _play_synchronized_video_audio(
        self, start_frame: Optional[int] = None, *, ensure_ready: bool = False
    ) -> None:
        from annolid.utils.audio_playback import play_audio_buffer, stop_audio_playback

        frame_idx = (
            int(start_frame)
            if start_frame is not None
            else int(getattr(self, "frame_number", 0) or 0)
        )
        sync_w = getattr(self, "boris_sync_widget", None)
        speed = max(0.05, float(getattr(self, "_playback_speed", 1.0) or 1.0))
        active_loader_fn = getattr(self, "_active_audio_loader", None)
        primary_loader = (
            active_loader_fn(ensure_ready=ensure_ready)
            if callable(active_loader_fn)
            else None
        )

        use_mix = (sync_w is not None and sync_w.custom_audio_mix_active()) or abs(
            speed - 1.0
        ) > 1e-6
        if not use_mix:
            if primary_loader is not None:
                primary_loader.play(start_frame=frame_idx)
            return

        primary_buf = getattr(primary_loader, "_buffer", None)
        t_start = self._estimate_recording_time(frame_idx) or 0.0
        if sync_w is not None:
            mixed, sr = sync_w.build_mixed_audio(
                primary_buf,
                float(t_start),
                playback_speed=speed,
            )
        elif primary_buf is not None:
            from annolid.core.media.audio import (
                AudioTrackMixSpec,
                mix_synchronized_audio_buffers,
            )

            mixed, sr = mix_synchronized_audio_buffers(
                [AudioTrackMixSpec(buffer=primary_buf)],
                primary_start_sec=float(t_start),
                playback_speed=speed,
            )
        else:
            return

        if mixed.size == 0:
            stop_audio_playback()
            return
        play_audio_buffer(mixed, sr, blocking=False)

    def _stop_synchronized_video_audio(self) -> None:
        from annolid.utils.audio_playback import stop_audio_playback

        active_loader_fn = getattr(self, "_active_audio_loader", None)
        loader = active_loader_fn() if callable(active_loader_fn) else None
        if loader is not None:
            loader.stop()
        else:
            stop_audio_playback()

    def _jump_to_time_from_sync_widget(self, time_s: float) -> None:
        if (
            getattr(self, "seekbar", None) is None
            or getattr(self, "num_frames", None) is None
        ):
            sync_w = getattr(self, "boris_sync_widget", None)
            if sync_w is not None:
                sync_w.sync_to_time(float(time_s), immediate=True)
            return
        fps = float(getattr(self, "fps", 0.0) or 29.97)
        target_frame = max(
            0, min(int(round(float(time_s) * fps)), int(self.num_frames) - 1)
        )
        self._jump_to_frame_from_log(target_frame)

    def _sync_boris_media_and_streams(self, frame_number: Optional[int] = None) -> None:
        sync_w = getattr(self, "boris_sync_widget", None)
        if sync_w is None:
            return
        dock = getattr(self, "boris_sync_dock", None)
        if dock is not None and not dock.isVisible():
            return
        frame = (
            int(frame_number)
            if frame_number is not None
            else int(getattr(self, "frame_number", 0) or 0)
        )
        t_sec = self._estimate_recording_time(frame)
        if t_sec is not None:
            sync_w.sync_to_time(float(t_sec))

    def _apply_boris_observation_sync_assets(
        self, obs, base_dir: Optional[Path] = None
    ) -> None:
        secondary_players = {
            pk: paths
            for pk, paths in (getattr(obs, "media_by_player", None) or {}).items()
            if str(pk) != "1" and paths
        }
        ext_specs = tuple(getattr(obs, "external_data", None) or ())
        if not secondary_players and not ext_specs:
            return

        self.open_boris_sync_dock()
        sync_w = getattr(self, "boris_sync_widget", None)
        if sync_w is None:
            return

        sync_w.clear_secondary_videos()
        sync_w.clear_external_streams()

        media_offsets = getattr(obs, "media_offsets", None) or {}
        for pk, paths in sorted(secondary_players.items(), key=lambda kv: str(kv[0])):
            raw_p = Path(paths[0]).expanduser()
            candidates = [raw_p]
            if not raw_p.is_absolute() and base_dir is not None:
                candidates.insert(0, Path(base_dir).resolve() / raw_p)
                candidates.append(Path(base_dir).resolve() / raw_p.name)
            resolved_vid = next((c for c in candidates if c.is_file()), None)
            if resolved_vid is None:
                logger.warning(
                    "Secondary BORIS media for player #%s not found on disk: %s",
                    pk,
                    paths[0],
                )
                continue
            off = float(media_offsets.get(pk, media_offsets.get(paths[0], 0.0)) or 0.0)
            try:
                sync_w.add_secondary_video(
                    resolved_vid, player_id=str(pk), offset_s=off
                )
            except Exception as exc:
                logger.warning(
                    "Failed opening secondary BORIS media '%s': %s",
                    resolved_vid.name,
                    exc,
                )

        for spec in ext_specs:
            try:
                sync_w.add_external_stream(spec, base_dir=base_dir)
            except Exception as exc:
                logger.warning(
                    "Failed loading BORIS external data stream '%s': %s",
                    spec.file_path,
                    exc,
                )
        self._sync_boris_media_and_streams()
