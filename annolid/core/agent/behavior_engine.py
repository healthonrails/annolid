from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .track_store import TrackState


@dataclass(frozen=True)
class BehaviorEvent:
    code: str
    start_frame: int
    end_frame: Optional[int] = None
    track_ids: Tuple[str, ...] = ()
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "code": self.code,
            "start_frame": int(self.start_frame),
            "track_ids": list(self.track_ids),
        }
        if self.end_frame is not None:
            payload["end_frame"] = int(self.end_frame)
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


@dataclass(frozen=True)
class BehaviorEngineConfig:
    interaction_code: str = "interaction"
    interaction_distance: float = 50.0
    interaction_min_frames: int = 3


@dataclass(frozen=True)
class BehaviorUpdate:
    active: Sequence[BehaviorEvent]
    completed: Sequence[BehaviorEvent]


class BehaviorEngine:
    """Rule-based behavior detector (Phase 4 MVP)."""

    def __init__(
        self,
        *,
        config: Optional[BehaviorEngineConfig] = None,
        allowed_codes: Optional[Iterable[str]] = None,
    ) -> None:
        self._config = config or BehaviorEngineConfig()
        self._allowed = set(allowed_codes or [])
        self._interaction_state: Dict[Tuple[str, str], Tuple[int, int]] = {}

    def update(self, frame_index: int, tracks: Sequence[TrackState]) -> BehaviorUpdate:
        active: List[BehaviorEvent] = []
        completed: List[BehaviorEvent] = []

        pairs = _pairwise_tracks(tracks)
        seen_pairs = set()

        for t1, t2, dist in pairs:
            key = _pair_key(t1.track_id, t2.track_id)
            seen_pairs.add(key)
            if dist <= self._config.interaction_distance:
                start_frame, count = self._interaction_state.get(key, (frame_index, 0))
                count += 1
                self._interaction_state[key] = (start_frame, count)
                if count >= int(self._config.interaction_min_frames):
                    active.append(
                        BehaviorEvent(
                            code=self._interaction_code(),
                            start_frame=start_frame,
                            end_frame=None,
                            track_ids=(key[0], key[1]),
                        )
                    )
            else:
                if key in self._interaction_state:
                    start_frame, count = self._interaction_state.pop(key)
                    if count >= int(self._config.interaction_min_frames):
                        completed.append(
                            BehaviorEvent(
                                code=self._interaction_code(),
                                start_frame=start_frame,
                                end_frame=frame_index,
                                track_ids=(key[0], key[1]),
                            )
                        )

        stale_pairs = [
            pair for pair in self._interaction_state if pair not in seen_pairs
        ]
        for key in stale_pairs:
            start_frame, count = self._interaction_state.pop(key)
            if count >= int(self._config.interaction_min_frames):
                completed.append(
                    BehaviorEvent(
                        code=self._interaction_code(),
                        start_frame=start_frame,
                        end_frame=frame_index,
                        track_ids=(key[0], key[1]),
                    )
                )

        return BehaviorUpdate(active=active, completed=completed)

    def _interaction_code(self) -> str:
        code = self._config.interaction_code
        if self._allowed and code not in self._allowed:
            return next(iter(self._allowed))
        return code


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairwise_tracks(
    tracks: Sequence[TrackState],
) -> Sequence[Tuple[TrackState, TrackState, float]]:
    results: List[Tuple[TrackState, TrackState, float]] = []
    total = len(tracks)
    for i in range(total):
        for j in range(i + 1, total):
            t1 = tracks[i]
            t2 = tracks[j]
            dist = _center_distance(t1.last_bbox, t2.last_bbox)
            results.append((t1, t2, dist))
    return results


def _center_distance(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    dx = acx - bcx
    dy = acy - bcy
    return (dx * dx + dy * dy) ** 0.5
