from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from annolid.core.types import FrameRef, Track, TrackObservation

from .base import Instances, Tool, ToolContext


@dataclass(frozen=True)
class TrackingResult:
    tracks: Sequence[Track]


class SimpleTrackTool(Tool[Sequence[Instances], TrackingResult]):
    """Lightweight tracker that groups instances by track_id/instance_id.

    This is a deterministic placeholder that turns per-frame instances into
    Track objects without performing temporal association (Phase 4/5 can
    replace it with a real tracker).
    """

    name = "simple_track"

    def run(self, ctx: ToolContext, payload: Sequence[Instances]) -> TrackingResult:
        tracks: Dict[str, List[TrackObservation]] = {}
        labels: Dict[str, str] = {}

        for frame_instances in payload:
            for inst in frame_instances.instances:
                track_id = inst.track_id or inst.instance_id
                if track_id is None:
                    track_id = f"frame_{inst.frame.frame_index}_{len(tracks)}"
                key = str(track_id)
                labels.setdefault(key, inst.label or "instance")
                obs = TrackObservation(
                    frame=inst.frame
                    if isinstance(inst.frame, FrameRef)
                    else FrameRef(frame_index=int(inst.frame.frame_index)),
                    geometry=inst.geometry,
                    mask=inst.mask,
                    score=inst.score,
                    label=inst.label,
                )
                tracks.setdefault(key, []).append(obs)

        assembled: List[Track] = []
        for track_id, observations in tracks.items():
            assembled.append(
                Track(
                    track_id=track_id,
                    label=labels.get(track_id, "instance"),
                    observations=observations,
                )
            )
        return TrackingResult(tracks=assembled)
