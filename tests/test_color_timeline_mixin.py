from __future__ import annotations

from annolid.gui.mixins.color_timeline_mixin import ColorTimelineMixin


class _TimelinePanelStub:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_behavior_catalog(self) -> None:
        self.refresh_calls += 1


class _Host(ColorTimelineMixin):
    def __init__(self) -> None:
        self.pinned_flags = {"existing": True}
        self.timeline_panel = _TimelinePanelStub()


def test_timeline_add_behavior_updates_pinned_flags_and_refreshes_catalog() -> None:
    host = _Host()

    host._timeline_add_behavior("grooming")

    assert host.pinned_flags == {"existing": True, "grooming": False}
    assert host.timeline_panel.refresh_calls == 1
