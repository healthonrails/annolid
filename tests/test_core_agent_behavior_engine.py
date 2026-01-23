from __future__ import annotations

from annolid.core.agent.behavior_engine import BehaviorEngine, BehaviorEngineConfig
from annolid.core.agent.track_store import TrackState


def _track(track_id: str, bbox: tuple[float, float, float, float]) -> TrackState:
    return TrackState(
        track_id=track_id,
        label="mouse",
        last_frame=0,
        last_bbox=bbox,
        hits=1,
        misses=0,
    )


def test_behavior_engine_interaction_event() -> None:
    engine = BehaviorEngine(
        config=BehaviorEngineConfig(
            interaction_distance=5.0,
            interaction_min_frames=2,
        ),
        allowed_codes=["interaction"],
    )

    tracks_far = [_track("t1", (0, 0, 2, 2)), _track("t2", (20, 20, 22, 22))]
    update0 = engine.update(0, tracks_far)
    assert not update0.active
    assert not update0.completed

    tracks_close = [_track("t1", (0, 0, 2, 2)), _track("t2", (3, 3, 5, 5))]
    update1 = engine.update(1, tracks_close)
    assert not update1.active

    update2 = engine.update(2, tracks_close)
    assert update2.active
    assert update2.active[0].start_frame == 1

    update3 = engine.update(3, tracks_far)
    assert update3.completed
    assert update3.completed[0].end_frame == 3
