from __future__ import annotations

from types import SimpleNamespace

from annolid.gui.mixins.shape_editing_mixin import ShapeEditingMixin


class _DummyCanvas:
    def __init__(self, mode: str = "point") -> None:
        self.createMode = mode
        self.calls: list[tuple[str, dict]] = []

    def setLastLabel(self, text, flags):  # noqa: N802
        self.calls.append((text, flags))
        return [SimpleNamespace(label=text)]


class _DummyToggle:
    def __init__(self) -> None:
        self.enabled = None

    def setEnabled(self, value: bool) -> None:  # noqa: N802
        self.enabled = bool(value)


class _DummySequencer:
    def __init__(self, enabled: bool = True, label: str | None = "nose") -> None:
        self._enabled = enabled
        self._label = label
        self._consumed = 0

    def is_sequence_enabled(self) -> bool:
        return self._enabled

    def consume_next_label(self) -> str | None:
        self._consumed += 1
        return self._label

    def auto_save_on_click(self) -> bool:
        return True


class _DummyWindow(ShapeEditingMixin):
    def __init__(self) -> None:
        self.canvas = _DummyCanvas("point")
        self.labelList = SimpleNamespace(clearSelection=lambda: None)
        self.actions = SimpleNamespace(
            editMode=_DummyToggle(),
            undoLastPoint=_DummyToggle(),
            undo=_DummyToggle(),
        )
        self.keypoint_sequence_widget = _DummySequencer()
        self.filename = "frame_0001.json"
        self.saved = 0
        self.dirty = 0
        self.labels = []

    def addLabel(self, shape, rebuild_unique=True):  # noqa: N802
        _ = rebuild_unique
        self.labels.append(shape)

    def setDirty(self):  # noqa: N802
        self.dirty += 1

    def saveFile(self):  # noqa: N802
        self.saved += 1


def test_try_apply_keypoint_sequence_labeling_applies_label_and_saves():
    w = _DummyWindow()
    applied = w._try_apply_keypoint_sequence_labeling()

    assert applied is True
    assert w.canvas.calls == [("nose", {})]
    assert len(w.labels) == 1
    assert w.dirty == 1
    assert w.saved == 1


def test_try_apply_keypoint_sequence_labeling_skips_non_point_mode():
    w = _DummyWindow()
    w.canvas = _DummyCanvas("polygon")

    applied = w._try_apply_keypoint_sequence_labeling()

    assert applied is False
    assert w.canvas.calls == []
