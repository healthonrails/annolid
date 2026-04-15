from __future__ import annotations

from annolid.gui.mixins.flags_overlay_mixin import FlagsOverlayMixin


class _BehaviorControllerStub:
    def __init__(self) -> None:
        self._active = set()
        self.behavior_names = {"contact", "chamber"}

    def active_behaviors(self, _frame_number: int):
        return set(self._active)


class _CanvasStub:
    def __init__(self) -> None:
        self.behavior_text_calls = []

    def setBehaviorText(self, text):
        self.behavior_text_calls.append(text)


class _TableStub:
    def rowCount(self) -> int:
        return 0

    def cellWidget(self, *_args, **_kwargs):
        return None


class _FlagWidgetStub:
    def __init__(self) -> None:
        self._table = _TableStub()


class _Host(FlagsOverlayMixin):
    def __init__(self) -> None:
        self.behavior_controller = _BehaviorControllerStub()
        self.frame_number = 0
        self.canvas = _CanvasStub()
        self.flag_widget = _FlagWidgetStub()
        self.loaded_flags = []
        self.flags_controller = type(
            "FlagsControllerStub",
            (),
            {
                "pinned_flags": {},
                "load_flags": lambda _self, flags: self.loaded_flags.append(
                    dict(flags)
                ),
            },
        )()


def test_refresh_behavior_overlay_skips_redundant_updates() -> None:
    host = _Host()
    host.behavior_controller._active = {"contact"}

    host._refresh_behavior_overlay()
    host._refresh_behavior_overlay()
    assert len(host.loaded_flags) == 1

    host.behavior_controller._active = {"chamber"}
    host._refresh_behavior_overlay()
    assert len(host.loaded_flags) == 2
