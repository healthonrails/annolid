from __future__ import annotations

import pytest

pytest.importorskip("shapely")

from annolid.tracker.cotracker.track import CoTrackerProcessor
from annolid.tracker.cowtracker.track import CoWTrackerProcessor
from annolid.tracker.point_tracking_processor import BasePointTrackingProcessor


def test_cowtracker_uses_shared_base_not_cotracker_inheritance() -> None:
    assert issubclass(CoWTrackerProcessor, BasePointTrackingProcessor)
    assert not issubclass(CoWTrackerProcessor, CoTrackerProcessor)


def test_cotracker_uses_shared_point_tracking_base() -> None:
    assert issubclass(CoTrackerProcessor, BasePointTrackingProcessor)
