from __future__ import annotations

from annolid.gui.mixins.color_timeline_mixin import ColorTimelineMixin


class _SettingsStub:
    def __init__(self) -> None:
        self.values = {}
        self.sync_calls = 0

    def value(self, key, default=None, type=None):
        value = self.values.get(key, default)
        if type is str and value is not None:
            return str(value)
        return value

    def setValue(self, key, value) -> None:
        self.values[key] = value

    def sync(self) -> None:
        self.sync_calls += 1


class _TimelinePanelStub:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_behavior_catalog(self) -> None:
        self.refresh_calls += 1


class _Host(ColorTimelineMixin):
    def __init__(self) -> None:
        self.pinned_flags = {"existing": True}
        self.timeline_panel = _TimelinePanelStub()
        self.settings = _SettingsStub()
        self._config = {
            "shape_color": "auto",
            "shift_auto_shape_color": 0,
            "default_shape_color": [0, 255, 0],
            "label_colors": None,
        }


def test_timeline_add_behavior_updates_pinned_flags_and_refreshes_catalog() -> None:
    host = _Host()

    host._timeline_add_behavior("grooming")

    assert host.pinned_flags == {"existing": True, "grooming": False}
    assert host.timeline_panel.refresh_calls == 1


def test_label_color_override_takes_precedence_and_persists() -> None:
    host = _Host()

    assert host._set_label_color_override("mouse_1", "#123456")

    assert host._get_rgb_by_label("mouse_1") == (18, 52, 86)
    assert (
        '"mouse_1": "#123456"'
        in host.settings.values[ColorTimelineMixin.LABEL_COLOR_OVERRIDES_KEY]
    )
    assert host.settings.sync_calls == 1


def test_label_color_overrides_restore_from_settings() -> None:
    host = _Host()
    host.settings.values[ColorTimelineMixin.LABEL_COLOR_OVERRIDES_KEY] = (
        '{"mouse_1": "#abcdef", "bad": "not-a-color"}'
    )

    host._load_label_color_overrides_from_settings()

    assert host._get_rgb_by_label("mouse_1") == (171, 205, 239)
    assert "bad" not in host._config["label_color_overrides"]


def test_reset_label_color_override_restores_auto_color() -> None:
    host = _Host()
    host._set_label_color_override("mouse_1", [10, 20, 30])
    auto_color = host._get_rgb_by_label("other_label")

    assert host._reset_label_color_override("mouse_1")

    assert host._get_rgb_by_label("mouse_1") != (10, 20, 30)
    assert host._get_rgb_by_label("other_label") == auto_color
