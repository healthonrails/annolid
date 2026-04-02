from __future__ import annotations

from types import SimpleNamespace

from annolid.domain import default_behavior_spec
from annolid.gui.mixins.color_timeline_mixin import ColorTimelineMixin
from annolid.gui.mixins.project_workflow_mixin import ProjectWorkflowMixin


class _ProjectWorkflowHost(ProjectWorkflowMixin):
    def __init__(self) -> None:
        self.project_schema = default_behavior_spec()
        self.project_schema_path = None
        self.pinned_flags = {}
        self.behavior_controller = SimpleNamespace(behavior_names={"walking"})
        self.flag_widget = SimpleNamespace(
            behavior_names=lambda: ["grooming", "walking", "grooming"]
        )
        self.timeline_panel = SimpleNamespace(
            _timeline_behavior_catalog=lambda: ["walking", "rearing"]
        )
        self._refresh_count = 0
        self._saved_count = 0

    def _refresh_behavior_catalog_views(self) -> None:
        self._refresh_count += 1

    def save_behavior_catalog(self) -> dict[str, object]:
        self._saved_count += 1
        return {"ok": True, "path": "/tmp/project.annolid.json"}


def test_sync_behavior_catalog_from_ui_persists_missing_names() -> None:
    host = _ProjectWorkflowHost()

    payload = host.sync_behavior_catalog_from_ui(save=True)

    codes = [behavior.code for behavior in host.project_schema.behaviors]
    assert payload["ok"] is True
    assert set(codes) == {"behavior_1", "walking", "grooming", "rearing"}
    assert payload["created_count"] == 3
    assert payload["saved"] is True
    assert host._saved_count == 1
    assert host._refresh_count == 1


def test_timeline_add_behavior_creates_catalog_entry_before_flag_sync() -> None:
    calls: list[tuple[str, str, bool]] = []

    class _TimelineHost(ColorTimelineMixin):
        def __init__(self) -> None:
            self.pinned_flags = {}

        def create_behavior_catalog_item(self, *, code: str, name: str, save: bool):
            calls.append((code, name, save))
            return {"ok": True}

        def loadFlags(self, flags):  # noqa: N802 - GUI API uses Qt-style name
            self.pinned_flags = dict(flags)

    host = _TimelineHost()
    host._timeline_add_behavior("grooming")

    assert calls == [("grooming", "grooming", True)]
    assert host.pinned_flags["grooming"] is False
