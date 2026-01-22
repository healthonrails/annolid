from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class FrameRef:
    """Canonical reference to a video frame."""

    frame_index: int
    timestamp_sec: Optional[float] = None
    video_name: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"frame_index": int(self.frame_index)}
        if self.timestamp_sec is not None:
            payload["timestamp_sec"] = float(self.timestamp_sec)
        if self.video_name is not None:
            payload["video_name"] = str(self.video_name)
        return payload
