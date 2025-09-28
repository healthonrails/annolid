from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from annolid.gui.widgets.video_slider import VideoSlider, VideoSliderMark


def normalize_event_label(event_label: Optional[str]) -> Optional[str]:
    """Return a canonical event label ("start" or "end") if recognized."""

    if not event_label:
        return None

    label = event_label.lower()
    if "start" in label or "begin" in label or "onset" in label:
        return "start"
    if "end" in label or "stop" in label or "offset" in label:
        return "end"
    return None


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


class BehaviorMarkManager:
    """Handle mapping between behavior events and slider marks."""

    def __init__(self, color_getter: Callable[[str], Union[str, Tuple[int, int, int]]]) -> None:
        self._color_getter = color_getter
        self._slider: Optional[VideoSlider] = None
        self._behavior_marks: Dict[Tuple[int, str, str], VideoSliderMark] = {}
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

        for mark in self._behavior_marks.values():
            self._slider.addMark(mark)
        for mark in self._generic_marks.values():
            self._slider.addMark(mark)

    def _detach_marks(self) -> None:
        if self._slider is None:
            return
        for mark in list(self._behavior_marks.values()) + list(self._generic_marks.values()):
            try:
                self._slider.removeMark(mark)
            except Exception:
                pass

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
            for mark in self._behavior_marks.values():
                self._slider.removeMark(mark)
        self._behavior_marks.clear()
        self.clear_highlight()

    def clear_all(self) -> None:
        self.clear_behavior_marks()
        if self._slider is not None:
            for mark in self._generic_marks.values():
                self._slider.removeMark(mark)
        self._generic_marks.clear()

    def ensure_behavior_mark(self, event: BehaviorEvent, highlight: bool = False) -> VideoSliderMark:
        mark = self._behavior_marks.get(event.mark_key)
        if mark is None:
            color = self._color_getter(event.behavior)
            mark = VideoSliderMark(mark_type=event.slider_mark_type,
                                   val=event.frame,
                                   _color=color)
            self._behavior_marks[event.mark_key] = mark
            if self._slider is not None:
                self._slider.addMark(mark)
        if highlight:
            self._highlighted_mark = mark
            self._highlighted_behavior_key = event.mark_key
        return mark

    def remove_behavior_mark(self, key: Tuple[int, str, str]) -> Optional[VideoSliderMark]:
        mark = self._behavior_marks.pop(key, None)
        if mark is None:
            return None
        if self._slider is not None:
            self._slider.removeMark(mark)
            self._slider.setTickMarks()
        if self._highlighted_behavior_key == key:
            self.clear_highlight()
        return mark

    def add_generic_mark(self,
                         frame: int,
                         mark_type: Optional[str] = None,
                         color: Optional[Union[str,
                                               Tuple[int, int, int]]] = None,
                         init_load: bool = False) -> Optional[VideoSliderMark]:
        if self._slider is None:
            return None

        key = (frame, mark_type)
        if key in self._generic_marks and not init_load:
            return self._generic_marks[key]

        resolved_color = color
        if resolved_color is None:
            if mark_type is not None:
                resolved_color = self._color_getter(str(mark_type))
            else:
                resolved_color = "green"

        style = mark_type or "simple"
        mark = VideoSliderMark(
            mark_type=style, val=frame, _color=resolved_color)
        self._generic_marks[key] = mark
        self._slider.addMark(mark)
        return mark

    def remove_mark_instance(self, mark: VideoSliderMark) -> Optional[Tuple[str, Tuple]]:
        if mark is None:
            return None

        for key, stored in list(self._behavior_marks.items()):
            if stored is mark:
                self.remove_behavior_mark(key)
                return ("behavior", key)

        for key, stored in list(self._generic_marks.items()):
            if stored is mark:
                if self._slider is not None:
                    self._slider.removeMark(mark)
                self._generic_marks.pop(key, None)
                return ("generic", key)
        return None

    def remove_highlighted_mark(self) -> Optional[Tuple[str, Tuple]]:
        if self._highlighted_mark is None:
            return None

        mark = self._highlighted_mark
        self._highlighted_mark = None

        if self._highlighted_behavior_key is not None:
            key = self._highlighted_behavior_key
            self.remove_behavior_mark(key)
            return ("behavior", key)

        # Highlighted mark was not a behavior mark; remove via instance search
        result = self.remove_mark_instance(mark)
        self.clear_highlight()
        return result

    def remove_marks_at_value(self, value: int) -> List[Tuple[str, Tuple]]:
        removed: List[Tuple[str, Tuple]] = []
        if self._slider is None:
            return removed

        for mark in list(self._slider.getMarksAtVal(value)):
            result = self.remove_mark_instance(mark)
            if result is not None:
                removed.append(result)

        if removed:
            self._slider.setTickMarks()
        return removed

    def remove_marks_matching_value(self, value: int) -> List[Tuple[str, Tuple]]:
        return self.remove_marks_at_value(value)

    def remove_marks_by_value(self, value: int) -> List[Tuple[str, Tuple]]:
        return self.remove_marks_at_value(value)


class BehaviorController:
    """High-level orchestrator for behavior events and slider marks."""

    def __init__(self, color_getter: Callable[[str], Union[str, Tuple[int, int, int]]]) -> None:
        self.timeline = BehaviorTimeline()
        self.marks = BehaviorMarkManager(color_getter)

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

    def sync_marks(self) -> None:
        self.marks.clear_behavior_marks()
        for event in self.timeline.iter_events():
            self.marks.ensure_behavior_mark(event)

    def clear(self) -> None:
        self.timeline.clear()
        self.marks.clear_all()

    def clear_behavior_data(self) -> None:
        self.timeline.clear()
        self.marks.clear_behavior_marks()

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

        self.marks.ensure_behavior_mark(event, highlight=highlight)
        return event

    def remove_highlighted_mark(self) -> Optional[Tuple[str, Tuple]]:
        result = self.marks.remove_highlighted_mark()
        if result is None:
            return None
        kind, key = result
        if kind == "behavior":
            self.timeline.remove_event(key)  # type: ignore[arg-type]
        return result

    def remove_marks_at_value(self, value: int) -> List[Tuple[str, Tuple]]:
        removed = self.marks.remove_marks_at_value(value)
        for kind, key in removed:
            if kind == "behavior":
                self.timeline.remove_event(key)  # type: ignore[arg-type]
        return removed

    def add_generic_mark(
        self,
        frame: int,
        mark_type: Optional[str] = None,
        color: Optional[Union[str, Tuple[int, int, int]]] = None,
        init_load: bool = False,
    ) -> Optional[VideoSliderMark]:
        return self.marks.add_generic_mark(frame, mark_type=mark_type, color=color, init_load=init_load)

    def remove_mark_instance(self, mark: VideoSliderMark) -> Optional[Tuple[str, Tuple]]:
        result = self.marks.remove_mark_instance(mark)
        if result is None:
            return None
        kind, key = result
        if kind == "behavior":
            self.timeline.remove_event(key)  # type: ignore[arg-type]
        return result

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

    def delete_event(self, key: Tuple[int, str, str]) -> Optional[BehaviorEvent]:
        event = self.timeline.remove_event(key)
        if event is not None:
            self.marks.remove_behavior_mark(key)
        return event

    def pop_last_event(self) -> Optional[BehaviorEvent]:
        events = list(self.iter_events())
        if not events:
            return None
        event = events[-1]
        self.marks.remove_behavior_mark(event.mark_key)
        self.timeline.remove_event(event.mark_key)
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
        self.clear_behavior_data()
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
