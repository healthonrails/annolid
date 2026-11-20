from __future__ import annotations

from typing import Optional

from qtpy import QtWidgets


class BehaviorLogMixin:
    """Behavior-event log display and editing helpers."""

    def _behavior_interval_bounds(self, event) -> Optional[tuple[int, int]]:
        interval = self._interval_for_behavior_event(event)
        if interval is None:
            return None
        start_frame, end_frame = interval
        if end_frame is None:
            end_frame = start_frame
        return int(start_frame), int(end_frame)

    def _behavior_event_key(self, event) -> tuple:
        storage_key = getattr(event, "storage_key", None)
        if storage_key is not None:
            return storage_key
        subj = getattr(event, "subject", None)
        if subj:
            return (int(event.frame), str(subj), str(event.behavior), str(event.event))
        return (int(event.frame), str(event.behavior), str(event.event))

    def _show_behavior_event_details(self, event) -> None:
        try:
            behavior = str(event.behavior)
        except Exception:
            behavior = "Unknown"
        ev_type = str(getattr(event, "event", "") or "").strip()
        subject = getattr(event, "subject", None)
        category = getattr(event, "category", None)
        modifiers = getattr(event, "modifiers", None)
        comment = getattr(event, "comment", None)
        timestamp = getattr(event, "timestamp", None)

        details = [f"Behavior: {behavior}"]
        if ev_type:
            details.append(f"Event Type: {ev_type.upper()}")
        if subject:
            details.append(f"Subject: {subject}")
        if category:
            details.append(f"Category: {category}")
        if modifiers:
            details.append(f"Modifiers: {', '.join(modifiers)}")
        if comment:
            details.append(f"Comment: {comment}")
        if timestamp is not None:
            details.append(f"Time: {float(timestamp):.3f}s")
        details.append(f"Frame: {int(getattr(event, 'frame', 0))}")

        QtWidgets.QMessageBox.information(
            self,
            self.tr("Behavior Event"),
            "\n".join(details),
        )

    def _edit_behavior_event_from_log(self, event) -> None:
        ev_kind = str(getattr(event, "event", "") or "").lower()
        is_point = ev_kind == "point"
        interval = None if is_point else self._behavior_interval_bounds(event)
        if not is_point and interval is None:
            return

        max_frame = max(int(getattr(self, "num_frames", 0) or 0) - 1, 999999)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(
            self.tr("Edit Point Event")
            if is_point
            else self.tr("Edit Behavior Interval")
        )
        form = QtWidgets.QFormLayout(dlg)

        frame_spin = QtWidgets.QSpinBox(dlg)
        frame_spin.setRange(0, max_frame)
        start_spin = QtWidgets.QSpinBox(dlg)
        start_spin.setRange(0, max_frame)
        end_spin = QtWidgets.QSpinBox(dlg)
        end_spin.setRange(0, max_frame)

        if is_point:
            frame_spin.setValue(int(getattr(event, "frame", 0)))
            form.addRow(self.tr("Frame:"), frame_spin)
        else:
            assert interval is not None
            start_frame, end_frame = interval
            start_spin.setValue(int(start_frame))
            end_spin.setValue(int(end_frame))
            form.addRow(self.tr("Start frame:"), start_spin)
            form.addRow(self.tr("End frame:"), end_spin)

        old_subj = getattr(event, "subject", None)
        subj_edit = QtWidgets.QLineEdit(str(old_subj or ""), dlg)
        mods_edit = QtWidgets.QLineEdit(
            ", ".join(getattr(event, "modifiers", ()) or ()), dlg
        )
        comment_edit = QtWidgets.QLineEdit(
            str(getattr(event, "comment", None) or ""), dlg
        )
        form.addRow(self.tr("Subject:"), subj_edit)
        form.addRow(self.tr("Modifiers (comma-separated):"), mods_edit)
        form.addRow(self.tr("Comment:"), comment_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, dlg
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        new_subject = subj_edit.text().strip() or None
        new_mods = [m.strip() for m in mods_edit.text().split(",") if m.strip()]
        new_comment = comment_edit.text().strip()

        def _timestamp_provider(frame: int) -> Optional[float]:
            fps = self.fps if getattr(self, "fps", None) and self.fps > 0 else None
            if fps is None:
                return None
            return float(frame) / float(fps)

        if is_point:
            new_frame = int(frame_spin.value())
            ts = _timestamp_provider(new_frame)
            self.behavior_controller.delete_event(self._behavior_event_key(event))
            self.behavior_controller.record_event(
                str(event.behavior),
                "point",
                new_frame,
                timestamp=ts,
                trial_time=ts,
                subject=new_subject,
                modifiers=new_mods,
                category=getattr(event, "category", None),
                confirmed=bool(getattr(event, "confirmed", True)),
                comment=new_comment or None,
            )
        else:
            assert interval is not None
            start_frame, end_frame = interval
            self.behavior_controller.update_interval(
                behavior=str(event.behavior),
                old_start=int(start_frame),
                old_end=int(end_frame) if end_frame is not None else None,
                new_start=int(start_spin.value()),
                new_end=int(end_spin.value()),
                subject=new_subject,
                old_subject=old_subj,
                modifiers=new_mods,
                comment=new_comment,
                timestamp_provider=_timestamp_provider,
            )
        self._refresh_behavior_log()
        refresh_badges = getattr(self, "_refresh_ethogram_active_badges", None)
        if callable(refresh_badges):
            refresh_badges()

    def _delete_behavior_event_from_log(self, event) -> None:
        interval = self._behavior_interval_bounds(event)
        if interval is not None:
            start_frame, end_frame = interval
            self.behavior_controller.delete_interval(
                behavior=str(event.behavior),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                subject=getattr(event, "subject", None),
            )
        else:
            self.behavior_controller.delete_event(self._behavior_event_key(event))
        self._refresh_behavior_log()
        refresh_badges = getattr(self, "_refresh_ethogram_active_badges", None)
        if callable(refresh_badges):
            refresh_badges()

    def _confirm_behavior_event_from_log(self, event) -> None:
        interval = self._behavior_interval_bounds(event)
        if interval is not None:
            start_frame, end_frame = interval
            self.behavior_controller.set_interval_confirmation(
                behavior=str(event.behavior),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                confirmed=True,
                subject=getattr(event, "subject", None),
            )
        else:
            self.behavior_controller.set_event_confirmation(
                self._behavior_event_key(event), confirmed=True
            )
        self._refresh_behavior_log()

    def _reject_behavior_event_from_log(self, event) -> None:
        interval = self._behavior_interval_bounds(event)
        if interval is not None:
            start_frame, end_frame = interval
            self.behavior_controller.set_interval_confirmation(
                behavior=str(event.behavior),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                confirmed=False,
                subject=getattr(event, "subject", None),
            )
        else:
            self.behavior_controller.set_event_confirmation(
                self._behavior_event_key(event), confirmed=False
            )
        self._refresh_behavior_log()

    def _interval_for_behavior_event(self, event):
        if str(getattr(event, "event", "") or "").lower() == "point":
            return None
        subj = getattr(event, "subject", None)
        try:
            ranges = self.behavior_controller.timeline.get_behavior_ranges(
                str(event.behavior),
                subject=subj,
            )
        except TypeError:
            ranges = self.behavior_controller.timeline.get_behavior_ranges(
                str(event.behavior)
            )
        except Exception:
            return None
        for start, end in ranges:
            end_bound = end if end is not None else start
            if int(start) <= int(event.frame) <= int(end_bound):
                return (start, end)
        return None

    def _refresh_behavior_log(self) -> None:
        if getattr(self, "behavior_log_widget", None) is None:
            return
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.set_events(
            list(self.behavior_controller.iter_events()),
            fps=fps_for_log,
        )
