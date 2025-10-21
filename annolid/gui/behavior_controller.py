from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from qtpy import QtGui

from annolid.behavior.project_schema import ProjectSchema
from annolid.behavior.event_utils import normalize_event_label
from annolid.gui.widgets.video_slider import VideoSlider, VideoSliderMark


@dataclass(frozen=True)
class BehaviorEvent:
    """Structured representation of a behavior boundary event."""

    frame: int
    behavior: str
    event: str  # canonical "start" or "end"
    timestamp: Optional[float] = None
    trial_time: Optional[float] = None
    subject: Optional[str] = None
    raw_event: Optional[str] = None

    @property
    def mark_key(self) -> Tuple[int, str, str]:
        return (self.frame, self.behavior, self.event)

    @property
    def slider_mark_type(self) -> str:
        return "event_start" if self.event == "start" else "event_end"


class BehaviorTimeline:
    """Store behavior events and derived active intervals."""

    def __init__(self) -> None:
        self._events: Dict[Tuple[int, str, str], BehaviorEvent] = {}
        self._ranges: Dict[str, List[Tuple[int, Optional[int]]]] = {}
        self._behaviors: Set[str] = set()
        self._pending_rebuild: Set[str] = set()

    def clear(self) -> None:
        self._events.clear()
        self._ranges.clear()
        self._behaviors.clear()
        self._pending_rebuild.clear()

    @property
    def events(self) -> Dict[Tuple[int, str, str], BehaviorEvent]:
        return self._events

    @property
    def behavior_names(self) -> Set[str]:
        return set(self._behaviors)

    @property
    def events_count(self) -> int:
        return len(self._events)

    def record_event(
        self,
        behavior: str,
        event_label: str,
        frame: int,
        *,
        timestamp: Optional[float] = None,
        trial_time: Optional[float] = None,
        subject: Optional[str] = None,
        raw_event: Optional[str] = None,
        rebuild: bool = True,
    ) -> Optional[BehaviorEvent]:
        canonical = event_label if event_label in {
            "start", "end"} else normalize_event_label(event_label)
        if canonical is None:
            return None

        event = BehaviorEvent(
            frame=frame,
            behavior=behavior,
            event=canonical,
            timestamp=timestamp,
            trial_time=trial_time,
            subject=subject,
            raw_event=raw_event or event_label,
        )

        if event.mark_key in self._events:
            return self._events[event.mark_key]

        self._events[event.mark_key] = event
        self._behaviors.add(behavior)
        if rebuild:
            self._rebuild_behavior(behavior)
        else:
            self._pending_rebuild.add(behavior)
        return event

    def remove_event(self, key: Tuple[int, str, str]) -> Optional[BehaviorEvent]:
        event = self._events.pop(key, None)
        if event is None:
            return None

        behavior = event.behavior
        remaining = [evt for evt in self._events.values()
                     if evt.behavior == behavior]
        if remaining:
            self._rebuild_behavior(behavior)
        else:
            self._behaviors.discard(behavior)
            self._ranges.pop(behavior, None)
        return event

    def flush_pending(self) -> None:
        if not self._pending_rebuild:
            return

        for behavior in list(self._pending_rebuild):
            self._rebuild_behavior(behavior)
        self._pending_rebuild.clear()

    def _rebuild_behavior(self, behavior: str) -> None:
        events = [evt for evt in self._events.values()
                  if evt.behavior == behavior]
        if not events:
            self._ranges.pop(behavior, None)
            return

        events.sort(key=lambda evt: (
            evt.frame, 0 if evt.event == "start" else 1))
        ranges: List[Tuple[int, Optional[int]]] = []
        open_starts: List[int] = []

        for evt in events:
            if evt.event == "start":
                open_starts.append(evt.frame)
            else:
                start_frame = open_starts.pop() if open_starts else evt.frame
                ranges.append((start_frame, evt.frame))

        for start_frame in open_starts:
            ranges.append((start_frame, None))

        self._ranges[behavior] = ranges

    def is_behavior_active(self, frame: int, behavior: str) -> bool:
        if behavior not in self._ranges:
            return False

        for start, end in self._ranges[behavior]:
            end_bound = end if end is not None else float("inf")
            if start <= frame <= end_bound:
                return True
        return False

    def active_behaviors(self, frame: int) -> Set[str]:
        return {behavior for behavior in self._ranges.keys() if self.is_behavior_active(frame, behavior)}

    def iter_events(self) -> Iterable[BehaviorEvent]:
        return iter(sorted(self._events.values(), key=lambda evt: (evt.frame, 0 if evt.event == "start" else 1, evt.behavior)))

    def iter_ranges(self) -> Iterable[Tuple[str, int, Optional[int]]]:
        for behavior, ranges in self._ranges.items():
            for start, end in ranges:
                yield behavior, start, end

    def get_behavior_ranges(self, behavior: str) -> List[Tuple[int, Optional[int]]]:
        return list(self._ranges.get(behavior, []))

    def to_export_rows(
        self,
        timestamp_fallback: Optional[Callable[[
            BehaviorEvent], Optional[float]]] = None,
    ) -> List[Tuple[float, float, str, str, str]]:
        rows: List[Tuple[float, float, str, str, str]] = []
        for event in self.iter_events():
            recording_time = event.timestamp
            if recording_time is None and timestamp_fallback is not None:
                fallback_value = timestamp_fallback(event)
                if fallback_value is not None:
                    recording_time = fallback_value
            trial_time = event.trial_time if event.trial_time is not None else recording_time
            subject = event.subject or "Subject 1"
            label = event.raw_event or (
                "state start" if event.event == "start" else "state stop")
            rows.append((trial_time if trial_time is not None else 0.0,
                         recording_time if recording_time is not None else 0.0,
                         subject,
                         event.behavior,
                         label))
        return rows


ColorType = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]


class BehaviorMarkManager:
    """Handle mapping between behavior events and slider marks."""

    def __init__(self, color_getter: Callable[[str], ColorType]) -> None:
        self._color_getter = color_getter
        self._slider: Optional[VideoSlider] = None
        self._behavior_marks: Dict[Tuple[int, str, str], VideoSliderMark] = {}
        self._interval_marks: Dict[Tuple[str, int, int], VideoSliderMark] = {}
        self._generic_marks: Dict[Tuple[int,
                                        Optional[str]], VideoSliderMark] = {}
        self._highlighted_mark: Optional[VideoSliderMark] = None
        self._highlighted_behavior_key: Optional[Tuple[int, str, str]] = None

    @property
    def slider(self) -> Optional[VideoSlider]:
        return self._slider

    def attach_slider(self, slider: Optional[VideoSlider]) -> None:
        if self._slider is slider:
            return
        self._detach_marks()
        self._slider = slider
        if self._slider is None:
            return
        self._reattach_marks()

    def _detach_marks(self) -> None:
        if self._slider is None:
            return
        for mark in list(self._behavior_marks.values()) + list(self._interval_marks.values()) + list(self._generic_marks.values()):
            try:
                self._slider.removeMark(mark)
            except Exception:
                pass

    def _reattach_marks(self) -> None:
        if self._slider is None:
            return
        for mark in self._behavior_marks.values():
            self._slider.addMark(mark, update=False)
        for mark in self._interval_marks.values():
            self._slider.addMark(mark, update=False)
        for mark in self._generic_marks.values():
            self._slider.addMark(mark, update=False)
        self._slider._update_visual_positions()

    @property
    def highlighted_mark(self) -> Optional[VideoSliderMark]:
        return self._highlighted_mark

    @property
    def highlighted_behavior_key(self) -> Optional[Tuple[int, str, str]]:
        return self._highlighted_behavior_key

    def clear_highlight(self) -> None:
        self._highlighted_mark = None
        self._highlighted_behavior_key = None

    def clear_behavior_marks(self) -> None:
        if self._slider is not None:
            for mark in list(self._behavior_marks.values()) + list(self._interval_marks.values()):
                self._slider.removeMark(mark)
            self._slider._update_visual_positions()
        self._behavior_marks.clear()
        self._interval_marks.clear()
        self.clear_highlight()

    def clear_generic_marks(self) -> None:
        if self._slider is not None:
            for mark in self._generic_marks.values():
                self._slider.removeMark(mark)
            self._slider._update_visual_positions()
        self._generic_marks.clear()

    def clear_all(self) -> None:
        self.clear_behavior_marks()
        self.clear_generic_marks()

    def set_marks(
        self,
        events: Iterable[BehaviorEvent],
        ranges: Iterable[Tuple[str, int, Optional[int]]],
        *,
        highlight_key: Optional[Tuple[int, str, str]] = None,
    ) -> None:
        if self._slider is None:
            self._behavior_marks.clear()
            self._interval_marks.clear()
            self.clear_highlight()
            return

        self.clear_behavior_marks()
        for event in events:
            self._create_behavior_mark(
                event, highlight=highlight_key == event.mark_key)
        for behavior, start, end in ranges:
            if end is None:
                continue
            self._create_interval_mark(behavior, start, end)
        self._slider._update_visual_positions()

    def add_generic_mark(
        self,
        frame: int,
        mark_type: Optional[str] = None,
        color: Optional[ColorType] = None,
        init_load: bool = False,
    ) -> Optional[VideoSliderMark]:
        if self._slider is None:
            return None
        key = (frame, mark_type)
        if key in self._generic_marks and not init_load:
            return self._generic_marks[key]

        qcolor = self._normalize_color(
            color if color is not None else (
                self._color_getter(str(mark_type)) if mark_type else "green"
            )
        )
        mark_color = self._color_tuple(qcolor)
        style = mark_type or "simple"
        mark = VideoSliderMark(mark_type=style, val=frame, _color=mark_color)
        self._generic_marks[key] = mark
        self._slider.addMark(mark)
        return mark

    def remove_generic_highlighted_mark(self) -> bool:
        if self._highlighted_mark is None or self._highlighted_behavior_key is not None:
            return False
        mark = self._highlighted_mark
        self._remove_generic_mark_instance(mark)
        self.clear_highlight()
        return True

    def remove_generic_marks_at_value(self, value: int) -> bool:
        removed = False
        for key in list(self._generic_marks.keys()):
            frame, _ = key
            if frame == value:
                self._remove_generic_mark(key)
                removed = True
        return removed

    def remove_mark_instance(self, mark: VideoSliderMark) -> Optional[Tuple[str, Tuple]]:
        key = self._find_generic_key(mark)
        if key is not None:
            self._remove_generic_mark(key)
            return ("generic", key)
        return None

    def find_behavior_key(self, mark: VideoSliderMark) -> Optional[Tuple[int, str, str]]:
        for key, stored in self._behavior_marks.items():
            if stored is mark:
                return key
        return None

    def _create_behavior_mark(self, event: BehaviorEvent, highlight: bool = False) -> None:
        qcolor = self._normalize_color(self._color_getter(event.behavior))
        color = self._color_tuple(qcolor)
        mark = VideoSliderMark(mark_type=event.slider_mark_type,
                               val=event.frame,
                               _color=color)
        self._behavior_marks[event.mark_key] = mark
        if self._slider is not None:
            self._slider.addMark(mark, update=False)
        if highlight:
            self._highlighted_mark = mark
            self._highlighted_behavior_key = event.mark_key

    def _create_interval_mark(self, behavior: str, start: int, end: int) -> None:
        qcolor = self._normalize_color(self._color_getter(behavior))
        qcolor.setAlpha(80)
        color = self._color_tuple(qcolor)
        mark = VideoSliderMark(mark_type="behavior_interval",
                               val=start,
                               end_val=end,
                               _color=color)
        self._interval_marks[(behavior, start, end)] = mark
        if self._slider is not None:
            self._slider.addMark(mark, update=False)

    def _remove_generic_mark(self, key: Tuple[int, Optional[str]]) -> None:
        mark = self._generic_marks.pop(key, None)
        if mark is not None:
            self._remove_generic_mark_instance(mark)

    def _remove_generic_mark_instance(self, mark: VideoSliderMark) -> None:
        if self._slider is not None:
            try:
                self._slider.removeMark(mark)
            except Exception:
                pass
            self._slider._update_visual_positions()
        if self._highlighted_mark is mark:
            self.clear_highlight()

    def _find_generic_key(self, mark: VideoSliderMark) -> Optional[Tuple[int, Optional[str]]]:
        for key, stored in self._generic_marks.items():
            if stored is mark:
                return key
        return None

    @staticmethod
    def _color_tuple(color: QtGui.QColor) -> Tuple[int, int, int, int]:
        return (color.red(), color.green(), color.blue(), color.alpha())

    def _normalize_color(self, value: ColorType) -> QtGui.QColor:
        if isinstance(value, str):
            return QtGui.QColor(value)
        if len(value) == 4:
            r, g, b, a = value
            return QtGui.QColor(r, g, b, a)
        r, g, b = value
        return QtGui.QColor(r, g, b)


class BehaviorController:
    """High-level orchestrator for behavior events and slider marks."""

    def __init__(self, color_getter: Callable[[str], ColorType]) -> None:
        self.timeline = BehaviorTimeline()
        self.marks = BehaviorMarkManager(color_getter)
        self._schema: Optional[ProjectSchema] = None
        self._behavior_categories: Dict[str, str] = {}
        self._subjects: Set[str] = set()

    def configure_from_schema(self, schema: Optional[ProjectSchema]) -> None:
        """Record known metadata from the project schema."""
        self._schema = schema
        self._behavior_categories = {}
        self._subjects = set()
        if schema is None:
            return
        self._subjects = {subj.name for subj in schema.subjects}
        for behavior in schema.behaviors:
            if behavior.category_id:
                self._behavior_categories[behavior.code] = behavior.category_id

    @property
    def highlighted_mark(self) -> Optional[VideoSliderMark]:
        return self.marks.highlighted_mark

    @property
    def highlighted_behavior_key(self) -> Optional[Tuple[int, str, str]]:
        return self.marks.highlighted_behavior_key

    def attach_slider(self, slider: Optional[VideoSlider]) -> None:
        self.marks.attach_slider(slider)
        if slider is not None:
            self.sync_marks()

    def sync_marks(self, highlight_key: Optional[Tuple[int, str, str]] = None) -> None:
        if self.marks.slider is None:
            return
        self.timeline.flush_pending()
        events = list(self.timeline.iter_events())
        ranges = list(self.timeline.iter_ranges())
        self.marks.set_marks(events, ranges, highlight_key=highlight_key)

    def clear(self) -> None:
        self.timeline.clear()
        self.marks.clear_all()

    def clear_behavior_data(self) -> None:
        self.timeline.clear()
        self.sync_marks()

    def record_event(
        self,
        behavior: str,
        event_label: str,
        frame: int,
        *,
        timestamp: Optional[float] = None,
        trial_time: Optional[float] = None,
        subject: Optional[str] = None,
        raw_event: Optional[str] = None,
        rebuild: bool = True,
        highlight: bool = True,
    ) -> Optional[BehaviorEvent]:
        event = self.timeline.record_event(
            behavior,
            event_label,
            frame,
            timestamp=timestamp,
            trial_time=trial_time,
            subject=subject,
            raw_event=raw_event,
            rebuild=rebuild,
        )
        if event is None:
            return None
        self.sync_marks(highlight_key=event.mark_key if highlight else None)
        return event

    def remove_highlighted_mark(self) -> Optional[Tuple[str, Tuple]]:
        key = self.marks.highlighted_behavior_key
        if key is not None:
            self.timeline.remove_event(key)
            self.sync_marks()
            return ("behavior", key)
        if self.marks.remove_generic_highlighted_mark():
            return ("generic", ())
        return None

    def remove_marks_at_value(self, value: int) -> List[Tuple[str, Tuple]]:
        removed: List[Tuple[str, Tuple]] = []
        for key, event in list(self.timeline.events.items()):
            if event.frame == value:
                self.timeline.remove_event(key)
                removed.append(("behavior", key))
        if removed:
            self.sync_marks()
            return removed
        if self.marks.remove_generic_marks_at_value(value):
            return [("generic", ())]
        return []

    def add_generic_mark(
        self,
        frame: int,
        mark_type: Optional[str] = None,
        color: Optional[ColorType] = None,
        init_load: bool = False,
    ) -> Optional[VideoSliderMark]:
        return self.marks.add_generic_mark(frame, mark_type=mark_type, color=color, init_load=init_load)

    def remove_mark_instance(self, mark: VideoSliderMark) -> Optional[Tuple[str, Tuple]]:
        key = self.marks.find_behavior_key(mark)
        if key is not None:
            self.timeline.remove_event(key)
            self.sync_marks()
            return ("behavior", key)
        return self.marks.remove_mark_instance(mark)

    def active_behaviors(self, frame: int) -> Set[str]:
        self.timeline.flush_pending()
        return self.timeline.active_behaviors(frame)

    def is_behavior_active(self, frame: int, behavior: str) -> bool:
        self.timeline.flush_pending()
        return self.timeline.is_behavior_active(frame, behavior)

    def iter_events(self):
        self.timeline.flush_pending()
        return self.timeline.iter_events()

    @property
    def behavior_names(self) -> Set[str]:
        return self.timeline.behavior_names

    @property
    def events_count(self) -> int:
        return self.timeline.events_count

    @property
    def schema(self) -> Optional[ProjectSchema]:
        return self._schema

    def delete_event(self, key: Tuple[int, str, str]) -> Optional[BehaviorEvent]:
        event = self.timeline.remove_event(key)
        if event is not None:
            self.sync_marks()
        return event

    def pop_last_event(self) -> Optional[BehaviorEvent]:
        events = list(self.iter_events())
        if not events:
            return None
        event = events[-1]
        self.timeline.remove_event(event.mark_key)
        self.sync_marks()
        return event

    def export_rows(
        self,
        timestamp_fallback: Optional[Callable[[
            BehaviorEvent], Optional[float]]] = None,
    ) -> List[Tuple[float, float, str, str, str]]:
        self.timeline.flush_pending()
        return self.timeline.to_export_rows(timestamp_fallback=timestamp_fallback)

    def load_events_from_rows(
        self,
        rows: Iterable[Tuple[Optional[float], float, Optional[str], str, str]],
        *,
        time_to_frame: Callable[[float], int],
        rebuild: bool = True,
    ) -> None:
        self.timeline.clear()
        for trial_time, recording_time, subject, behavior, event_label in rows:
            try:
                time_value = float(recording_time)
            except (TypeError, ValueError):
                continue
            frame = time_to_frame(time_value)
            trial_value: Optional[float]
            try:
                trial_value = float(
                    trial_time) if trial_time is not None else None
            except (TypeError, ValueError):
                trial_value = None
            self.record_event(
                behavior,
                event_label,
                frame,
                timestamp=time_value,
                trial_time=trial_value,
                subject=subject,
                rebuild=False,
                highlight=False,
            )

        if rebuild:
            self.timeline.flush_pending()
        self.sync_marks()
