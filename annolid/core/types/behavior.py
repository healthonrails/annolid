from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .frame import FrameRef


@dataclass(frozen=True)
class BehaviorEvent:
    frame: FrameRef
    behavior: str
    event: str  # "start" or "end"
    subject: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    category: Optional[str] = None
    track_ids: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": self.frame.to_dict(),
            "behavior": str(self.behavior),
            "event": str(self.event),
        }
        if self.subject is not None:
            payload["subject"] = str(self.subject)
        if self.modifiers:
            payload["modifiers"] = list(self.modifiers)
        if self.category is not None:
            payload["category"] = str(self.category)
        if self.track_ids:
            payload["track_ids"] = list(self.track_ids)
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


@dataclass(frozen=True)
class BehaviorSpan:
    start: FrameRef
    end: FrameRef
    behavior: str
    subject: Optional[str] = None
    track_ids: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "behavior": str(self.behavior),
        }
        if self.subject is not None:
            payload["subject"] = str(self.subject)
        if self.track_ids:
            payload["track_ids"] = list(self.track_ids)
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload
