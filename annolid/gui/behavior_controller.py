from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from qtpy import QtGui

from annolid.behavior.event_utils import ParsedEventRow, classify_event_label
from annolid.domain import ProjectSchema, normalize_event_label
from annolid.gui.widgets.video_slider import VideoSlider, VideoSliderMark
from annolid.infrastructure import AnnotationStore


EventKeyType = Union[
    Tuple[int, str, str],
    Tuple[int, str, str, str],
    Tuple[int, str, str, str, int],
]
InternalStorageKeyType = Tuple[int, str, str, str, int]


@dataclass(frozen=True)
class BehaviorEvent:
    """Structured representation of a behavior boundary or point event."""

    frame: int
    behavior: str
    event: str  # canonical "start", "end", or "point"
    timestamp: Optional[float] = None
    trial_time: Optional[float] = None
    subject: Optional[str] = None
    raw_event: Optional[str] = None
    modifiers: Tuple[str, ...] = ()
    category: Optional[str] = None
    confirmed: bool = True
    comment: Optional[str] = None
    subframe_seq: int = 0

    @property
    def mark_key(self) -> Tuple[int, str, str]:
        return (self.frame, self.behavior, self.event)

    @property
    def storage_key(self) -> EventKeyType:
        s_norm = (self.subject or "").strip()
        if self.subframe_seq > 0:
            return (self.frame, s_norm, self.behavior, self.event, self.subframe_seq)
        return (self.frame, s_norm, self.behavior, self.event)

    @property
    def slider_mark_type(self) -> str:
        if self.event == "point":
            return "event_point"
        return "event_start" if self.event == "start" else "event_end"

    def to_store_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "behavior": self.behavior,
            "event": self.event,
            "confirmed": bool(self.confirmed),
        }
        if self.subject is not None:
            payload["subject"] = self.subject
        if self.raw_event is not None:
            payload["raw_event"] = self.raw_event
        if self.timestamp is not None:
            payload["timestamp"] = float(self.timestamp)
        if self.trial_time is not None:
            payload["trial_time"] = float(self.trial_time)
        if self.modifiers:
            payload["modifiers"] = list(self.modifiers)
        if self.category is not None:
            payload["category"] = self.category
        if self.comment:
            payload["comment"] = self.comment
        return payload


class BehaviorTimeline:
    """Store behavior events (state start/end and instantaneous point) and active intervals."""

    def __init__(self) -> None:
        self._events: Dict[InternalStorageKeyType, BehaviorEvent] = {}
        self._ranges: Dict[str, List[Tuple[int, Optional[int]]]] = {}
        self._subject_ranges: Dict[
            Tuple[str, str], List[Tuple[int, Optional[int]]]
        ] = {}
        self._behaviors: Set[str] = set()
        self._pending_rebuild: Set[str] = set()

    def clear(self) -> None:
        self._events.clear()
        self._ranges.clear()
        self._subject_ranges.clear()
        self._behaviors.clear()
        self._pending_rebuild.clear()

    @property
    def events(self) -> Dict[InternalStorageKeyType, BehaviorEvent]:
        return self._events

    @property
    def behavior_names(self) -> Set[str]:
        return set(self._behaviors)

    @property
    def events_count(self) -> int:
        return len(self._events)

    def _resolve_storage_keys(self, key: EventKeyType) -> List[InternalStorageKeyType]:
        if len(key) == 5:
            k5 = (
                int(key[0]),
                str(key[1] or "").strip(),
                str(key[2]),
                str(key[3]),
                int(key[4]),
            )
            return [k5] if k5 in self._events else []
        if len(key) == 4:
            f_val = int(key[0])
            s_val = str(key[1] or "").strip()
            beh_val = str(key[2])
            ev_val = str(key[3])
            return sorted(
                sk
                for sk in self._events.keys()
                if sk[0] == f_val
                and sk[1] == s_val
                and sk[2] == beh_val
                and sk[3] == ev_val
            )
        f_val, beh_val, ev_val = int(key[0]), str(key[1]), str(key[2])
        return sorted(
            sk
            for sk in self._events.keys()
            if sk[0] == f_val and sk[2] == beh_val and sk[3] == ev_val
        )

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
        modifiers: Optional[Iterable[str]] = None,
        category: Optional[str] = None,
        confirmed: bool = True,
        comment: Optional[str] = None,
    ) -> Optional[BehaviorEvent]:
        canonical = (
            event_label
            if event_label in {"start", "end", "point"}
            else (
                normalize_event_label(event_label) or classify_event_label(event_label)
            )
        )
        if canonical is None:
            return None

        f_int = int(frame)
        s_norm = (subject or "").strip()
        same_slot = [
            (sk, evt)
            for sk, evt in self._events.items()
            if sk[0] == f_int
            and sk[1] == s_norm
            and sk[2] == behavior
            and sk[3] == canonical
        ]
        for _sk, existing in same_slot:
            if existing.timestamp is None and timestamp is None:
                return existing
            if (
                existing.timestamp is not None
                and timestamp is not None
                and abs(float(existing.timestamp) - float(timestamp)) <= 1e-6
            ):
                return existing

        used_seqs = {sk[4] for sk, _ in same_slot}
        seq = 0 if 0 not in used_seqs else (max(used_seqs) + 1)

        event = BehaviorEvent(
            frame=f_int,
            behavior=behavior,
            event=canonical,
            timestamp=timestamp,
            trial_time=trial_time,
            subject=subject,
            raw_event=raw_event or event_label,
            modifiers=tuple(modifiers or ()),
            category=category,
            confirmed=bool(confirmed),
            comment=(str(comment).strip() or None) if comment else None,
            subframe_seq=seq,
        )

        self._events[(f_int, s_norm, behavior, canonical, seq)] = event
        self._behaviors.add(behavior)
        if rebuild:
            self._rebuild_behavior(behavior)
        else:
            self._pending_rebuild.add(behavior)
        return event

    def remove_event(self, key: EventKeyType) -> Optional[BehaviorEvent]:
        matched_keys = self._resolve_storage_keys(key)
        if not matched_keys:
            return None
        event = self._events.pop(matched_keys[-1], None)
        for extra_k in matched_keys[:-1]:
            self._events.pop(extra_k, None)
        if event is None:
            return None

        behavior = event.behavior
        remaining = [evt for evt in self._events.values() if evt.behavior == behavior]
        if remaining:
            self._rebuild_behavior(behavior)
        else:
            self._behaviors.discard(behavior)
            self._ranges.pop(behavior, None)
            for sk in [k for k in self._subject_ranges if k[1] == behavior]:
                self._subject_ranges.pop(sk, None)
        return event

    def update_event_confirmation(
        self,
        key: EventKeyType,
        *,
        confirmed: bool,
    ) -> Optional[BehaviorEvent]:
        matched_keys = self._resolve_storage_keys(key)
        if not matched_keys:
            return None
        updated_last: Optional[BehaviorEvent] = None
        for sk in matched_keys:
            event = self._events.get(sk)
            if event is None:
                continue
            updated = BehaviorEvent(
                frame=event.frame,
                behavior=event.behavior,
                event=event.event,
                timestamp=event.timestamp,
                trial_time=event.trial_time,
                subject=event.subject,
                raw_event=event.raw_event,
                modifiers=event.modifiers,
                category=event.category,
                confirmed=bool(confirmed),
                comment=event.comment,
                subframe_seq=event.subframe_seq,
            )
            self._events[sk] = updated
            updated_last = updated
        if updated_last is not None:
            self._rebuild_behavior(updated_last.behavior)
        return updated_last

    def flush_pending(self) -> None:
        if not self._pending_rebuild:
            return

        for behavior in list(self._pending_rebuild):
            self._rebuild_behavior(behavior)
        self._pending_rebuild.clear()

    def _rebuild_behavior(self, behavior: str) -> None:
        for sk in [k for k in self._subject_ranges if k[1] == behavior]:
            self._subject_ranges.pop(sk, None)

        events = [
            evt
            for evt in self._events.values()
            if evt.behavior == behavior and evt.event in {"start", "end"}
        ]
        if not events:
            self._ranges.pop(behavior, None)
            return

        by_subj: Dict[str, List[BehaviorEvent]] = {}
        for evt in events:
            s_norm = (evt.subject or "").strip()
            by_subj.setdefault(s_norm, []).append(evt)

        combined_ranges: List[Tuple[int, Optional[int]]] = []
        for s_norm, s_events in sorted(by_subj.items()):
            frames_missing_ts = {evt.frame for evt in s_events if evt.timestamp is None}
            s_events.sort(
                key=lambda evt: (
                    evt.frame,
                    (
                        round(float(evt.timestamp), 9)
                        if evt.timestamp is not None
                        and evt.frame not in frames_missing_ts
                        else 0.0
                    ),
                    0 if evt.event == "end" else 1,
                    evt.subframe_seq,
                )
            )
            s_ranges: List[Tuple[int, Optional[int]]] = []
            open_starts: List[int] = []

            for evt in s_events:
                if evt.event == "start":
                    open_starts.append(evt.frame)
                else:
                    start_frame = open_starts.pop(0) if open_starts else evt.frame
                    s_ranges.append((start_frame, evt.frame))

            for start_frame in open_starts:
                s_ranges.append((start_frame, None))

            self._subject_ranges[(s_norm, behavior)] = s_ranges
            combined_ranges.extend(s_ranges)

        combined_ranges.sort(
            key=lambda pair: (pair[0], pair[1] if pair[1] is not None else 10**12)
        )
        self._ranges[behavior] = combined_ranges

    def is_behavior_active(
        self,
        frame: int,
        behavior: str,
        *,
        subject: Optional[str] = None,
    ) -> bool:
        if subject is not None:
            s_norm = subject.strip()
            ranges = self._subject_ranges.get((s_norm, behavior), [])
            for start, end in ranges:
                end_bound = end if end is not None else float("inf")
                if start <= frame <= end_bound:
                    return True
            return False

        if behavior not in self._ranges:
            return False

        for start, end in self._ranges[behavior]:
            end_bound = end if end is not None else float("inf")
            if start <= frame <= end_bound:
                return True
        return False

    def active_behaviors(
        self,
        frame: int,
        *,
        subject: Optional[str] = None,
    ) -> Set[str]:
        if subject is not None:
            s_norm = subject.strip()
            return {
                beh
                for (subj_k, beh) in self._subject_ranges.keys()
                if subj_k == s_norm
                and self.is_behavior_active(frame, beh, subject=s_norm)
            }
        return {
            behavior
            for behavior in self._ranges.keys()
            if self.is_behavior_active(frame, behavior)
        }

    def active_behavior_subjects(self, frame: int) -> Dict[str, List[str]]:
        """Return active state behaviors mapped to sorted non-empty active subjects."""
        by_behavior: Dict[str, Set[str]] = {}
        for (subj_k, behavior), ranges in self._subject_ranges.items():
            for start, end in ranges:
                end_bound = end if end is not None else float("inf")
                if start <= frame <= end_bound:
                    subjects = by_behavior.setdefault(behavior, set())
                    if subj_k:
                        subjects.add(subj_k)
                    break
        return {
            behavior: sorted(subjects) for behavior, subjects in by_behavior.items()
        }

    def point_behaviors_at_frame(self, frame: int) -> List[Tuple[str, Optional[str]]]:
        """Return sorted unique (behavior, subject) pairs for point events at exact frame."""
        frame_int = int(frame)
        pairs: Set[Tuple[str, Optional[str]]] = set()
        for evt in self._events.values():
            if evt.frame == frame_int and evt.event == "point":
                behavior = str(evt.behavior or "").strip()
                if not behavior:
                    continue
                subj = (str(evt.subject).strip() or None) if evt.subject else None
                pairs.add((behavior, subj))
        return sorted(pairs, key=lambda item: (item[0].lower(), item[1] or ""))

    def distinct_subjects(self) -> Set[str]:
        """Return all non-empty subject names present across recorded events."""
        subjects: Set[str] = set()
        for evt in self._events.values():
            subj = str(evt.subject).strip() if evt.subject else ""
            if subj:
                subjects.add(subj)
        return subjects

    def iter_events(self) -> Iterable[BehaviorEvent]:
        order_rank = {"start": 0, "point": 1, "end": 2}
        vals = list(self._events.values())
        frames_missing_ts = {evt.frame for evt in vals if evt.timestamp is None}
        return iter(
            sorted(
                vals,
                key=lambda evt: (
                    evt.frame,
                    (
                        round(float(evt.timestamp), 9)
                        if evt.timestamp is not None
                        and evt.frame not in frames_missing_ts
                        else 0.0
                    ),
                    order_rank.get(evt.event, 1),
                    evt.subframe_seq,
                    evt.behavior,
                    evt.subject or "",
                ),
            )
        )

    def rebind_fps(
        self, fps: float, *, max_frame: Optional[int] = None
    ) -> Tuple[bool, Set[int]]:
        """Re-index timestamped events to a new FPS without losing sub-frame timestamps."""
        if fps <= 0:
            return False, set()
        snapshot = list(self.iter_events())
        if not snapshot or not any(evt.timestamp is not None for evt in snapshot):
            return False, set()

        def _compute_frame(evt: BehaviorEvent) -> int:
            if evt.timestamp is None:
                target = evt.frame
            else:
                target = int(round(float(evt.timestamp) * float(fps)))
            if max_frame is not None:
                target = max(0, min(int(max_frame), target))
            else:
                target = max(0, target)
            return target

        if all(_compute_frame(evt) == evt.frame for evt in snapshot):
            return False, set()

        dirty_frames: Set[int] = {evt.frame for evt in snapshot}
        self.clear()
        for evt in snapshot:
            nf = _compute_frame(evt)
            dirty_frames.add(nf)
            self.record_event(
                evt.behavior,
                evt.event,
                nf,
                timestamp=evt.timestamp,
                trial_time=evt.trial_time,
                subject=evt.subject,
                raw_event=evt.raw_event,
                rebuild=False,
                modifiers=evt.modifiers,
                category=evt.category,
                confirmed=evt.confirmed,
                comment=evt.comment,
            )
        self.flush_pending()
        return True, dirty_frames

    def events_at_frame(self, frame: int) -> List[BehaviorEvent]:
        return [evt for evt in self._events.values() if evt.frame == frame]

    def iter_ranges(self) -> Iterable[Tuple[str, int, Optional[int]]]:
        for behavior, ranges in self._ranges.items():
            for start, end in ranges:
                yield behavior, start, end

    def iter_point_events(self) -> Iterable[BehaviorEvent]:
        for evt in self.iter_events():
            if evt.event == "point":
                yield evt

    def get_behavior_ranges(
        self, behavior: str, *, subject: Optional[str] = None
    ) -> List[Tuple[int, Optional[int]]]:
        if subject is not None:
            return list(self._subject_ranges.get((subject.strip(), behavior), []))
        return list(self._ranges.get(behavior, []))

    def to_export_rows(
        self,
        timestamp_fallback: Optional[Callable[[BehaviorEvent], Optional[float]]] = None,
        *,
        confirmed_only: bool = True,
    ) -> List[Tuple[float, float, str, str, str]]:
        rows: List[Tuple[float, float, str, str, str]] = []
        for event in self.iter_events():
            if confirmed_only and not event.confirmed:
                continue
            recording_time = event.timestamp
            if recording_time is None and timestamp_fallback is not None:
                fallback_value = timestamp_fallback(event)
                if fallback_value is not None:
                    recording_time = fallback_value
            trial_time = (
                event.trial_time if event.trial_time is not None else recording_time
            )
            subject = event.subject or "Subject 1"
            if event.event == "point":
                label = (
                    event.raw_event
                    if classify_event_label(event.raw_event) == "point"
                    else "point"
                )
            else:
                label = event.raw_event or (
                    "state start" if event.event == "start" else "state stop"
                )
            rows.append(
                (
                    trial_time if trial_time is not None else 0.0,
                    recording_time if recording_time is not None else 0.0,
                    subject,
                    event.behavior,
                    label,
                )
            )
        return rows

    def to_detailed_export_rows(
        self,
        timestamp_fallback: Optional[Callable[[BehaviorEvent], Optional[float]]] = None,
        *,
        confirmed_only: bool = True,
    ) -> List[Dict[str, Any]]:
        detailed: List[Dict[str, Any]] = []
        for event in self.iter_events():
            if confirmed_only and not event.confirmed:
                continue
            recording_time = event.timestamp
            if recording_time is None and timestamp_fallback is not None:
                fallback_value = timestamp_fallback(event)
                if fallback_value is not None:
                    recording_time = fallback_value
            trial_time = (
                event.trial_time if event.trial_time is not None else recording_time
            )
            subject = event.subject or "Subject 1"
            if event.event == "point":
                label = (
                    event.raw_event
                    if classify_event_label(event.raw_event) == "point"
                    else "point"
                )
            else:
                label = event.raw_event or (
                    "state start" if event.event == "start" else "state stop"
                )
            detailed.append(
                {
                    "Trial time": trial_time if trial_time is not None else 0.0,
                    "Recording time": recording_time
                    if recording_time is not None
                    else 0.0,
                    "Subject": subject,
                    "Behavior": event.behavior,
                    "Event": label,
                    "Modifiers": "|".join(event.modifiers) if event.modifiers else "",
                    "Comment": event.comment or "",
                    "Category": event.category or "",
                    "Frame": int(event.frame),
                    "Confirmed": bool(event.confirmed),
                }
            )
        return detailed


ColorType = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]


class BehaviorMarkManager:
    """Handle mapping between behavior events and slider marks."""

    def __init__(self, color_getter: Callable[[str], ColorType]) -> None:
        self._color_getter = color_getter
        self._slider: Optional[VideoSlider] = None
        self._behavior_marks: Dict[Tuple[int, str, str], VideoSliderMark] = {}
        self._interval_marks: Dict[Tuple[str, int, int], VideoSliderMark] = {}
        self._generic_marks: Dict[Tuple[int, Optional[str]], VideoSliderMark] = {}
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
        for mark in (
            list(self._behavior_marks.values())
            + list(self._interval_marks.values())
            + list(self._generic_marks.values())
        ):
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
            for mark in list(self._behavior_marks.values()) + list(
                self._interval_marks.values()
            ):
                self._slider.removeMark(mark)
            self._slider._update_visual_positions()
        self._behavior_marks.clear()
        self._interval_marks.clear()
        self.clear_highlight()

    def clear_generic_marks(self, *, mark_type: Optional[str] = None) -> None:
        if mark_type is None:
            if self._slider is not None:
                for mark in self._generic_marks.values():
                    self._slider.removeMark(mark)
                self._slider._update_visual_positions()
            self._generic_marks.clear()
            return

        to_remove = [key for key in self._generic_marks.keys() if key[1] == mark_type]
        if not to_remove:
            return
        if self._slider is not None:
            for key in to_remove:
                mark = self._generic_marks.get(key)
                if mark is not None:
                    self._slider.removeMark(mark)
            self._slider._update_visual_positions()
        for key in to_remove:
            self._generic_marks.pop(key, None)

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
            self._create_behavior_mark(event, highlight=highlight_key == event.mark_key)
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
            color
            if color is not None
            else (self._color_getter(str(mark_type)) if mark_type else "green")
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

    def remove_mark_instance(
        self, mark: VideoSliderMark
    ) -> Optional[Tuple[str, Tuple]]:
        key = self._find_generic_key(mark)
        if key is not None:
            self._remove_generic_mark(key)
            return ("generic", key)
        return None

    def find_behavior_key(
        self, mark: VideoSliderMark
    ) -> Optional[Tuple[int, str, str]]:
        for key, stored in self._behavior_marks.items():
            if stored is mark:
                return key
        return None

    def _create_behavior_mark(
        self, event: BehaviorEvent, highlight: bool = False
    ) -> None:
        qcolor = self._normalize_color(self._color_getter(event.behavior))
        color = self._color_tuple(qcolor)
        mark = VideoSliderMark(
            mark_type=event.slider_mark_type, val=event.frame, _color=color
        )
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
        mark = VideoSliderMark(
            mark_type="behavior_interval", val=start, end_val=end, _color=color
        )
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

    def _find_generic_key(
        self, mark: VideoSliderMark
    ) -> Optional[Tuple[int, Optional[str]]]:
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

    BEHAVIOR_STORE_KEY = "annolid_behavior"
    BEHAVIOR_STORE_VERSION = 1

    def __init__(self, color_getter: Callable[[str], ColorType]) -> None:
        self.timeline = BehaviorTimeline()
        self.marks = BehaviorMarkManager(color_getter)
        self._schema: Optional[ProjectSchema] = None
        self._behavior_categories: Dict[str, str] = {}
        self._subjects: Set[str] = set()
        self._annotation_store: Optional[AnnotationStore] = None
        self._frame_record_provider: Optional[
            Callable[[int], Optional[Dict[str, Any]]]
        ] = None
        self._change_listeners: List[Callable[[], None]] = []

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

    def attach_annotation_store(
        self,
        store: Optional[AnnotationStore],
        *,
        frame_record_provider: Optional[
            Callable[[int], Optional[Dict[str, Any]]]
        ] = None,
    ) -> None:
        self._annotation_store = store
        self._frame_record_provider = frame_record_provider

    def add_change_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._change_listeners:
            self._change_listeners.append(callback)

    def remove_change_listener(self, callback: Callable[[], None]) -> None:
        if callback in self._change_listeners:
            self._change_listeners.remove(callback)

    def attach_annotation_store_for_video(
        self,
        results_dir: Optional[Path],
        *,
        store_name: Optional[str] = None,
        frame_record_provider: Optional[
            Callable[[int], Optional[Dict[str, Any]]]
        ] = None,
    ) -> None:
        if results_dir is None:
            self.attach_annotation_store(None)
            return
        results_dir = Path(results_dir)
        frame_stub = results_dir / f"{results_dir.name}_000000000.json"
        store = AnnotationStore.for_frame_path(frame_stub, store_name)
        self.attach_annotation_store(store, frame_record_provider=frame_record_provider)

    def clear_generic_marks(self, *, mark_type: Optional[str] = None) -> None:
        self.marks.clear_generic_marks(mark_type=mark_type)

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
        self._notify_listeners()

    def clear_behavior_data(self) -> None:
        frames = {event.frame for event in self.timeline.events.values()}
        self.timeline.clear()
        self._persist_frames(frames)
        self.sync_marks()
        self._notify_listeners()

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
        modifiers: Optional[Iterable[str]] = None,
        category: Optional[str] = None,
        confirmed: bool = True,
        comment: Optional[str] = None,
        persist: bool = True,
        notify: bool = True,
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
            modifiers=modifiers,
            category=category,
            confirmed=confirmed,
            comment=comment,
        )
        if event is None:
            return None
        self.sync_marks(highlight_key=event.mark_key if highlight else None)
        if persist:
            self._persist_frame(event.frame)
        if notify:
            self._notify_listeners()
        return event

    def remove_highlighted_mark(self) -> Optional[Tuple[str, Tuple]]:
        key = self.marks.highlighted_behavior_key
        if key is not None:
            removed_event = self.timeline.remove_event(key)
            self.sync_marks()
            if removed_event is not None:
                self._persist_frame(removed_event.frame)
                self._notify_listeners()
            return ("behavior", key)
        if self.marks.remove_generic_highlighted_mark():
            return ("generic", ())
        return None

    def remove_marks_at_value(self, value: int) -> List[Tuple[str, Tuple]]:
        removed: List[Tuple[str, Tuple]] = []
        for key, event in list(self.timeline.events.items()):
            if event.frame == value:
                self.timeline.remove_event(key)
                removed.append(("behavior", event.mark_key))
        if removed:
            self.sync_marks()
            self._persist_frame(value)
            self._notify_listeners()
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
        return self.marks.add_generic_mark(
            frame, mark_type=mark_type, color=color, init_load=init_load
        )

    def remove_mark_instance(
        self, mark: VideoSliderMark
    ) -> Optional[Tuple[str, Tuple]]:
        key = self.marks.find_behavior_key(mark)
        if key is not None:
            removed_event = self.timeline.remove_event(key)
            self.sync_marks()
            if removed_event is not None:
                self._persist_frame(removed_event.frame)
                self._notify_listeners()
            return ("behavior", key)
        return self.marks.remove_mark_instance(mark)

    def active_behaviors(
        self, frame: int, *, subject: Optional[str] = None
    ) -> Set[str]:
        self.timeline.flush_pending()
        return self.timeline.active_behaviors(frame, subject=subject)

    def active_behavior_subjects(self, frame: int) -> Dict[str, List[str]]:
        self.timeline.flush_pending()
        return self.timeline.active_behavior_subjects(frame)

    def point_behaviors_at_frame(self, frame: int) -> List[Tuple[str, Optional[str]]]:
        self.timeline.flush_pending()
        return self.timeline.point_behaviors_at_frame(frame)

    def distinct_subjects(self) -> Set[str]:
        return set(self._subjects) | self.timeline.distinct_subjects()

    def is_behavior_active(
        self,
        frame: int,
        behavior: str,
        *,
        subject: Optional[str] = None,
    ) -> bool:
        self.timeline.flush_pending()
        return self.timeline.is_behavior_active(frame, behavior, subject=subject)

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

    def delete_event(self, key: EventKeyType) -> Optional[BehaviorEvent]:
        event = self.timeline.remove_event(key)
        if event is not None:
            self.sync_marks()
            self._persist_frame(event.frame)
            self._notify_listeners()
        return event

    def set_event_confirmation(
        self,
        key: EventKeyType,
        *,
        confirmed: bool,
    ) -> Optional[BehaviorEvent]:
        event = self.timeline.update_event_confirmation(key, confirmed=confirmed)
        if event is not None:
            self.sync_marks()
            self._persist_frame(event.frame)
            self._notify_listeners()
        return event

    def set_interval_confirmation(
        self,
        *,
        behavior: str,
        start_frame: int,
        end_frame: int,
        confirmed: bool,
        subject: Optional[str] = None,
    ) -> None:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        if subject is not None:
            self.timeline.update_event_confirmation(
                (start_frame, subject, behavior, "start"), confirmed=confirmed
            )
            self.timeline.update_event_confirmation(
                (end_frame, subject, behavior, "end"), confirmed=confirmed
            )
        else:
            self.timeline.update_event_confirmation(
                (start_frame, behavior, "start"), confirmed=confirmed
            )
            self.timeline.update_event_confirmation(
                (end_frame, behavior, "end"), confirmed=confirmed
            )
        self.sync_marks()
        self._persist_frames({start_frame, end_frame})
        self._notify_listeners()

    def pop_last_event(self) -> Optional[BehaviorEvent]:
        events = list(self.iter_events())
        if not events:
            return None
        event = events[-1]
        self.timeline.remove_event(event.storage_key)
        self.sync_marks()
        self._persist_frame(event.frame)
        self._notify_listeners()
        return event

    def rebind_fps(
        self,
        fps: float,
        *,
        max_frame: Optional[int] = None,
        persist: bool = True,
    ) -> bool:
        """Recompute integer frame indices from exact float timestamps when FPS changes."""
        changed, dirty_frames = self.timeline.rebind_fps(fps, max_frame=max_frame)
        if not changed:
            return False
        self.sync_marks()
        if persist:
            self._persist_frames(dirty_frames)
        self._notify_listeners()
        return True

    def export_rows(
        self,
        timestamp_fallback: Optional[Callable[[BehaviorEvent], Optional[float]]] = None,
        *,
        confirmed_only: bool = True,
    ) -> List[Tuple[float, float, str, str, str]]:
        self.timeline.flush_pending()
        return self.timeline.to_export_rows(
            timestamp_fallback=timestamp_fallback,
            confirmed_only=confirmed_only,
        )

    def export_detailed_rows(
        self,
        timestamp_fallback: Optional[Callable[[BehaviorEvent], Optional[float]]] = None,
        *,
        confirmed_only: bool = True,
    ) -> List[Dict[str, Any]]:
        self.timeline.flush_pending()
        return self.timeline.to_detailed_export_rows(
            timestamp_fallback=timestamp_fallback,
            confirmed_only=confirmed_only,
        )

    def load_events_from_rows(
        self,
        rows: Iterable[Any],
        *,
        time_to_frame: Callable[[float], int],
        rebuild: bool = True,
        persist: bool = True,
    ) -> None:
        self.timeline.clear()
        frames_to_persist: Set[int] = set()
        cat_map = self._schema.category_map() if self._schema else {}
        beh_map = self._schema.behavior_map() if self._schema else {}

        for raw_item in rows:
            mods: Optional[Iterable[str]] = None
            comment: Optional[str] = None
            if isinstance(raw_item, ParsedEventRow):
                if raw_item.recording_time is None or not raw_item.behavior:
                    continue
                time_value = float(raw_item.recording_time)
                trial_value = (
                    float(raw_item.trial_time)
                    if raw_item.trial_time is not None
                    else time_value
                )
                subject = raw_item.subject
                behavior = raw_item.behavior
                event_label = raw_item.event_kind or raw_item.raw_event
                raw_ev = raw_item.raw_event
                mods = raw_item.modifiers or None
                comment = raw_item.comment or None
            elif isinstance(raw_item, Mapping):
                rec = raw_item.get(
                    "recording time",
                    raw_item.get("Recording time", raw_item.get("time")),
                )
                try:
                    time_value = float(rec)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                tr = raw_item.get("trial time", raw_item.get("Trial time"))
                try:
                    trial_value = float(tr) if tr is not None else time_value
                except (TypeError, ValueError):
                    trial_value = time_value
                subject = (
                    str(raw_item.get("subject", raw_item.get("Subject")) or "").strip()
                    or None
                )
                behavior = str(
                    raw_item.get("behavior", raw_item.get("Behavior")) or ""
                ).strip()
                event_label = str(
                    raw_item.get("event", raw_item.get("Event", raw_item.get("Status")))
                    or ""
                ).strip()
                raw_ev = event_label
                raw_m = raw_item.get("modifiers", raw_item.get("Modifiers"))
                if isinstance(raw_m, (list, tuple)):
                    mods = [str(x).strip() for x in raw_m if str(x).strip()]
                elif isinstance(raw_m, str) and raw_m.strip():
                    mods = [
                        x.strip()
                        for x in raw_m.replace("|", ",").split(",")
                        if x.strip()
                    ]
                comment = (
                    str(raw_item.get("comment", raw_item.get("Comment")) or "").strip()
                    or None
                )
            else:
                seq = tuple(raw_item)
                if len(seq) < 5:
                    continue
                trial_time, recording_time, subject, behavior, event_label = seq[:5]
                raw_ev = event_label
                if len(seq) >= 6 and seq[5]:
                    if isinstance(seq[5], (list, tuple)):
                        mods = [str(x).strip() for x in seq[5] if str(x).strip()]
                    else:
                        mods = [
                            x.strip()
                            for x in str(seq[5]).replace("|", ",").split(",")
                            if x.strip()
                        ]
                if len(seq) >= 7 and seq[6]:
                    comment = str(seq[6]).strip() or None
                try:
                    time_value = float(recording_time)
                except (TypeError, ValueError):
                    continue
                try:
                    trial_value = float(trial_time) if trial_time is not None else None
                except (TypeError, ValueError):
                    trial_value = None

            if not behavior or not event_label:
                continue
            frame = time_to_frame(time_value)
            category_name: Optional[str] = None
            bdef = beh_map.get(behavior)
            if bdef and bdef.category_id:
                cobj = cat_map.get(bdef.category_id)
                category_name = (cobj.name or cobj.id) if cobj else bdef.category_id

            self.record_event(
                behavior,
                event_label,
                frame,
                timestamp=time_value,
                trial_time=trial_value,
                subject=subject,
                raw_event=raw_ev,
                rebuild=False,
                highlight=False,
                modifiers=mods,
                category=category_name,
                comment=comment,
                persist=False,
                notify=False,
            )
            frames_to_persist.add(frame)

        if rebuild:
            self.timeline.flush_pending()
        self.sync_marks()
        if persist:
            self._persist_frames(frames_to_persist)
        self._notify_listeners()

    def load_events_from_store(
        self,
        store: Optional[AnnotationStore] = None,
    ) -> None:
        target_store = store or self._annotation_store
        if target_store is None:
            return

        self.timeline.clear()
        loaded_behavior_events = False
        pending_agent: Dict[tuple, int] = {}
        frames = sorted(target_store.iter_frames())
        for frame in frames:
            try:
                frame_index = int(frame)
            except (TypeError, ValueError):
                continue
            record = target_store.get_frame(frame)
            if not record:
                continue
            other = record.get("otherData") or {}
            behavior_block = other.get(self.BEHAVIOR_STORE_KEY)
            if not isinstance(behavior_block, dict):
                if not loaded_behavior_events:
                    self._load_agent_events_from_record(
                        other,
                        frame_index,
                        pending_agent,
                    )
                continue
            events = behavior_block.get("events") or []
            if not isinstance(events, list):
                if not loaded_behavior_events:
                    self._load_agent_events_from_record(
                        other,
                        frame_index,
                        pending_agent,
                    )
                continue
            loaded_behavior_events = True
            for payload in events:
                if not isinstance(payload, dict):
                    continue
                behavior = payload.get("behavior")
                event_label = payload.get("event")
                if not behavior or not event_label:
                    continue
                timestamp = _safe_float(payload.get("timestamp"))
                trial_time = _safe_float(payload.get("trial_time"))
                subject = payload.get("subject")
                raw_event = payload.get("raw_event")
                modifiers = payload.get("modifiers") or ()
                category = payload.get("category")
                confirmed = payload.get("confirmed", True)
                comment = payload.get("comment")
                self.record_event(
                    str(behavior),
                    str(event_label),
                    frame_index,
                    timestamp=timestamp,
                    trial_time=trial_time,
                    subject=str(subject) if subject is not None else None,
                    raw_event=str(raw_event) if raw_event is not None else None,
                    rebuild=False,
                    highlight=False,
                    modifiers=list(modifiers) if modifiers else None,
                    category=str(category) if category is not None else None,
                    confirmed=bool(confirmed),
                    comment=str(comment) if comment else None,
                    persist=False,
                    notify=False,
                )

        self.timeline.flush_pending()
        self.sync_marks()
        self._notify_listeners()

    def _load_agent_events_from_record(
        self,
        other: Dict[str, Any],
        frame_index: int,
        pending: Dict[tuple, int],
    ) -> None:
        active = other.get("behavior_active") or []
        completed = other.get("behavior_completed") or []
        if not isinstance(active, list):
            active = []
        if not isinstance(completed, list):
            completed = []

        for entry in active:
            if not isinstance(entry, dict):
                continue
            behavior = entry.get("code") or entry.get("behavior")
            if not behavior:
                continue
            subject = self._subject_from_track_ids(entry.get("track_ids"))
            start_frame = entry.get("start_frame")
            if start_frame is None:
                start_frame = frame_index
            key = (str(behavior), subject)
            pending.setdefault(key, int(start_frame))
            self.record_event(
                str(behavior),
                "start",
                int(start_frame),
                subject=subject,
                raw_event="agent_active",
                confirmed=False,
                rebuild=False,
                highlight=False,
                persist=False,
                notify=False,
            )

        for entry in completed:
            if not isinstance(entry, dict):
                continue
            behavior = entry.get("code") or entry.get("behavior")
            if not behavior:
                continue
            subject = self._subject_from_track_ids(entry.get("track_ids"))
            start_frame = entry.get("start_frame")
            if start_frame is None:
                key = (str(behavior), subject)
                start_frame = pending.pop(key, frame_index)
            end_frame = entry.get("end_frame")
            if end_frame is None:
                end_frame = frame_index
            self.record_event(
                str(behavior),
                "start",
                int(start_frame),
                subject=subject,
                raw_event="agent_completed",
                confirmed=False,
                rebuild=False,
                highlight=False,
                persist=False,
                notify=False,
            )
            self.record_event(
                str(behavior),
                "end",
                int(end_frame),
                subject=subject,
                raw_event="agent_completed",
                confirmed=False,
                rebuild=False,
                highlight=False,
                persist=False,
                notify=False,
            )

    @staticmethod
    def _subject_from_track_ids(track_ids: Any) -> Optional[str]:
        if not track_ids:
            return None
        if isinstance(track_ids, (list, tuple)):
            ids = [str(item) for item in track_ids if item is not None]
            if not ids:
                return None
            if len(ids) == 1:
                return ids[0]
            return "+".join(ids)
        return str(track_ids)

    def _persist_frames(self, frames: Iterable[int]) -> None:
        if self._annotation_store is None:
            return
        for frame in sorted({int(value) for value in frames if value is not None}):
            self._persist_frame(frame)

    def _persist_frame(self, frame: int) -> None:
        if self._annotation_store is None:
            return
        record = self._load_store_record(frame)
        other = dict(record.get("otherData") or {})
        events = [
            event.to_store_dict() for event in self.timeline.events_at_frame(frame)
        ]
        if events:
            other[self.BEHAVIOR_STORE_KEY] = {
                "version": self.BEHAVIOR_STORE_VERSION,
                "events": events,
            }
        else:
            other.pop(self.BEHAVIOR_STORE_KEY, None)

        record["frame"] = frame
        record["otherData"] = other
        record.setdefault("version", "annolid")
        record.setdefault("flags", {})
        record.setdefault("shapes", [])
        record.setdefault("imagePath", "")
        record.setdefault("imageHeight", None)
        record.setdefault("imageWidth", None)
        self._annotation_store.append_frame(record)

    def _load_store_record(self, frame: int) -> Dict[str, Any]:
        record: Optional[Dict[str, Any]] = None
        if self._annotation_store is not None:
            record = self._annotation_store.get_frame(frame)
        if record is None and self._frame_record_provider is not None:
            try:
                record = self._frame_record_provider(frame)
            except Exception:
                record = None
        if record:
            return dict(record)
        return {"frame": frame}

    def _notify_listeners(self) -> None:
        if not self._change_listeners:
            return
        for callback in list(self._change_listeners):
            try:
                callback()
            except Exception:
                continue

    def create_point_event(
        self,
        *,
        behavior: str,
        frame: int,
        subject: Optional[str] = None,
        modifiers: Optional[Iterable[str]] = None,
        comment: Optional[str] = None,
        category: Optional[str] = None,
        timestamp_provider: Optional[Callable[[int], Optional[float]]] = None,
    ) -> Optional[BehaviorEvent]:
        frame = int(frame)
        ts = timestamp_provider(frame) if timestamp_provider else None
        return self.record_event(
            behavior,
            "point",
            frame,
            timestamp=ts,
            trial_time=ts,
            subject=subject,
            raw_event="point",
            modifiers=modifiers,
            category=category,
            comment=comment,
        )

    def create_interval(
        self,
        *,
        behavior: str,
        start_frame: int,
        end_frame: int,
        subject: Optional[str] = None,
        modifiers: Optional[Iterable[str]] = None,
        comment: Optional[str] = None,
        category: Optional[str] = None,
        timestamp_provider: Optional[Callable[[int], Optional[float]]] = None,
    ) -> None:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        start_time = timestamp_provider(start_frame) if timestamp_provider else None
        end_time = timestamp_provider(end_frame) if timestamp_provider else None
        self.timeline.record_event(
            behavior,
            "start",
            start_frame,
            timestamp=start_time,
            trial_time=start_time,
            subject=subject,
            modifiers=modifiers,
            category=category,
            comment=comment,
            rebuild=False,
        )
        self.timeline.record_event(
            behavior,
            "end",
            end_frame,
            timestamp=end_time,
            trial_time=end_time,
            subject=subject,
            modifiers=modifiers,
            category=category,
            rebuild=False,
        )
        self.timeline.flush_pending()
        self.sync_marks()
        self._persist_frames({start_frame, end_frame})
        self._notify_listeners()

    def update_interval(
        self,
        *,
        behavior: str,
        old_start: int,
        old_end: Optional[int],
        new_start: int,
        new_end: int,
        subject: Optional[str] = None,
        old_subject: Optional[str] = None,
        modifiers: Optional[Iterable[str]] = None,
        comment: Optional[str] = None,
        timestamp_provider: Optional[Callable[[int], Optional[float]]] = None,
    ) -> None:
        new_start = int(new_start)
        new_end = int(new_end)
        if new_end < new_start:
            new_start, new_end = new_end, new_start
        lookup_subj = old_subject if old_subject is not None else subject
        start_key: EventKeyType = (
            (int(old_start), lookup_subj, behavior, "start")
            if lookup_subj is not None
            else (int(old_start), behavior, "start")
        )
        old_start_evt = self.timeline.remove_event(start_key)
        old_end_evt: Optional[BehaviorEvent] = None
        if old_end is not None:
            end_key: EventKeyType = (
                (int(old_end), lookup_subj, behavior, "end")
                if lookup_subj is not None
                else (int(old_end), behavior, "end")
            )
            old_end_evt = self.timeline.remove_event(end_key)

        eff_subject = (
            subject
            if subject is not None
            else (old_start_evt.subject if old_start_evt else None)
        )
        eff_mods = (
            tuple(modifiers)
            if modifiers is not None
            else (old_start_evt.modifiers if old_start_evt else ())
        )
        eff_comment = (
            comment
            if comment is not None
            else (
                (old_start_evt.comment if old_start_evt else None)
                or (old_end_evt.comment if old_end_evt else None)
            )
        )
        eff_cat = old_start_evt.category if old_start_evt else None
        eff_conf = old_start_evt.confirmed if old_start_evt else True

        start_time = timestamp_provider(new_start) if timestamp_provider else None
        end_time = timestamp_provider(new_end) if timestamp_provider else None
        self.timeline.record_event(
            behavior,
            "start",
            new_start,
            timestamp=start_time,
            trial_time=start_time,
            subject=eff_subject,
            modifiers=eff_mods,
            category=eff_cat,
            confirmed=eff_conf,
            comment=eff_comment,
            rebuild=False,
        )
        self.timeline.record_event(
            behavior,
            "end",
            new_end,
            timestamp=end_time,
            trial_time=end_time,
            subject=eff_subject,
            modifiers=eff_mods,
            category=eff_cat,
            confirmed=eff_conf,
            rebuild=False,
        )
        self.timeline.flush_pending()
        self.sync_marks()
        self._persist_frames({old_start, old_end, new_start, new_end})
        self._notify_listeners()

    def delete_interval(
        self,
        *,
        behavior: str,
        start_frame: int,
        end_frame: int,
        subject: Optional[str] = None,
    ) -> None:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        if subject is not None:
            self.timeline.remove_event((start_frame, subject, behavior, "start"))
            self.timeline.remove_event((end_frame, subject, behavior, "end"))
        else:
            self.timeline.remove_event((start_frame, behavior, "start"))
            self.timeline.remove_event((end_frame, behavior, "end"))
        self.timeline.flush_pending()
        self.sync_marks()
        self._persist_frames({start_frame, end_frame})
        self._notify_listeners()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
