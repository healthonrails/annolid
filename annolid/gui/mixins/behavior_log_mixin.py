from __future__ import annotations

from typing import Optional

from qtpy import QtWidgets


class BehaviorLogMixin:
    """Behavior-event log display and editing helpers."""

    def _show_behavior_event_details(self, event) -> None:
        try:
            behavior = str(event.behavior)
        except Exception:
            behavior = "Unknown"
        subject = getattr(event, "subject", None)
        category = getattr(event, "category", None)
        modifiers = getattr(event, "modifiers", None)
        timestamp = getattr(event, "timestamp", None)

        details = [f"Behavior: {behavior}"]
        if subject:
            details.append(f"Subject: {subject}")
        if category:
            details.append(f"Category: {category}")
        if modifiers:
            details.append(f"Modifiers: {', '.join(modifiers)}")
        if timestamp is not None:
            details.append(f"Time: {float(timestamp):.2f}s")
        details.append(f"Frame: {int(getattr(event, 'frame', 0))}")

        QtWidgets.QMessageBox.information(
            self,
            self.tr("Behavior Event"),
            "\n".join(details),
        )

    def _edit_behavior_event_from_log(self, event) -> None:
        interval = self._interval_for_behavior_event(event)
        if interval is None:
            return
        start_frame, end_frame = interval
        if end_frame is None:
            end_frame = start_frame

        new_start, ok = QtWidgets.QInputDialog.getInt(
            self,
            self.tr("Edit Interval"),
            self.tr("Start frame:"),
            value=int(start_frame),
            min=0,
            max=max(int(self.num_frames or 0) - 1, 0),
        )
        if not ok:
            return
        new_end, ok = QtWidgets.QInputDialog.getInt(
            self,
            self.tr("Edit Interval"),
            self.tr("End frame:"),
            value=int(end_frame),
            min=0,
            max=max(int(self.num_frames or 0) - 1, 0),
        )
        if not ok:
            return

        def _timestamp_provider(frame: int) -> Optional[float]:
            fps = self.fps if self.fps and self.fps > 0 else None
            if fps is None:
                return None
            return float(frame) / float(fps)

        self.behavior_controller.update_interval(
            behavior=str(event.behavior),
            old_start=int(start_frame),
            old_end=int(end_frame) if end_frame is not None else None,
            new_start=int(new_start),
            new_end=int(new_end),
            subject=getattr(event, "subject", None),
            timestamp_provider=_timestamp_provider,
        )
        self._refresh_behavior_log()

    def _delete_behavior_event_from_log(self, event) -> None:
        interval = self._interval_for_behavior_event(event)
        if interval is not None:
            start_frame, end_frame = interval
            if end_frame is None:
                end_frame = start_frame
            self.behavior_controller.delete_interval(
                behavior=str(event.behavior),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
            )
        else:
            key = (int(event.frame), str(event.behavior), str(event.event))
            self.behavior_controller.delete_event(key)
        self._refresh_behavior_log()

    def _confirm_behavior_event_from_log(self, event) -> None:
        interval = self._interval_for_behavior_event(event)
        if interval is not None:
            start_frame, end_frame = interval
            if end_frame is None:
                end_frame = start_frame
            self.behavior_controller.set_interval_confirmation(
                behavior=str(event.behavior),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                confirmed=True,
            )
        else:
            key = (int(event.frame), str(event.behavior), str(event.event))
            self.behavior_controller.set_event_confirmation(key, confirmed=True)
        self._refresh_behavior_log()

    def _reject_behavior_event_from_log(self, event) -> None:
        interval = self._interval_for_behavior_event(event)
        if interval is not None:
            start_frame, end_frame = interval
            if end_frame is None:
                end_frame = start_frame
            self.behavior_controller.set_interval_confirmation(
                behavior=str(event.behavior),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                confirmed=False,
            )
        else:
            key = (int(event.frame), str(event.behavior), str(event.event))
            self.behavior_controller.set_event_confirmation(key, confirmed=False)
        self._refresh_behavior_log()

    def _interval_for_behavior_event(self, event):
        try:
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
