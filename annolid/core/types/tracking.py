from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .frame import FrameRef
from .geometry import Geometry, geometry_from_dict


@dataclass(frozen=True)
class TrackObservation:
    frame: FrameRef
    geometry: Geometry
    score: Optional[float] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": self.frame.to_dict(),
            "geometry": self.geometry.to_dict(),
        }
        if self.score is not None:
            payload["score"] = float(self.score)
        if self.label is not None:
            payload["label"] = str(self.label)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TrackObservation":
        frame = FrameRef(
            frame_index=int((payload.get("frame") or {})["frame_index"]),  # type: ignore[index]
            timestamp_sec=(payload.get("frame") or {}).get("timestamp_sec"),  # type: ignore[union-attr]
            video_name=(payload.get("frame") or {}).get("video_name"),  # type: ignore[union-attr]
        )
        geometry = geometry_from_dict(payload["geometry"])  # type: ignore[arg-type]
        score = payload.get("score", None)
        label = payload.get("label", None)
        return cls(
            frame=frame,
            geometry=geometry,
            score=float(score) if score is not None else None,
            label=str(label) if label is not None else None,
        )


@dataclass(frozen=True)
class Track:
    track_id: str
    label: str
    observations: List[TrackObservation] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "track_id": str(self.track_id),
            "label": str(self.label),
            "observations": [obs.to_dict() for obs in self.observations],
        }
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload
