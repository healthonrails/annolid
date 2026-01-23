from __future__ import annotations

from annolid.core.agent.track_store import TrackStore
from annolid.core.types import BBoxGeometry, FrameRef
from annolid.core.agent.tools.base import Instance, Instances


def test_track_store_assigns_stable_ids() -> None:
    store = TrackStore(iou_threshold=0.2, max_misses=2)
    frame0 = FrameRef(frame_index=0)
    frame1 = FrameRef(frame_index=1)

    inst0 = Instance(
        frame=frame0,
        geometry=BBoxGeometry("bbox", (0.0, 0.0, 10.0, 10.0)),
        label="mouse",
    )
    inst1 = Instance(
        frame=frame1,
        geometry=BBoxGeometry("bbox", (1.0, 1.0, 11.0, 11.0)),
        label="mouse",
    )

    out0 = store.update(Instances(frame=frame0, instances=[inst0]))
    out1 = store.update(Instances(frame=frame1, instances=[inst1]))

    assert out0.instances[0].track_id is not None
    assert out1.instances[0].track_id == out0.instances[0].track_id


def test_track_store_creates_new_track_for_new_instance() -> None:
    store = TrackStore(iou_threshold=0.5, max_misses=1)
    frame0 = FrameRef(frame_index=0)

    inst0 = Instance(
        frame=frame0,
        geometry=BBoxGeometry("bbox", (0.0, 0.0, 10.0, 10.0)),
        label="mouse",
    )
    inst1 = Instance(
        frame=frame0,
        geometry=BBoxGeometry("bbox", (20.0, 20.0, 30.0, 30.0)),
        label="mouse",
    )

    out = store.update(Instances(frame=frame0, instances=[inst0, inst1]))
    ids = {inst.track_id for inst in out.instances}
    assert len(ids) == 2
