from __future__ import annotations

from annolid.gui.mixins.navigation_workflow_mixin import NavigationWorkflowMixin


class _SignalStub:
    def __init__(self) -> None:
        self.connect_calls = 0

    def connect(self, _slot) -> None:
        self.connect_calls += 1


class _UniqLabelListStub:
    def __init__(self) -> None:
        self.itemSelectionChanged = _SignalStub()


class _NavigationHost(NavigationWorkflowMixin):
    def __init__(self) -> None:
        self._config = {"keep_prev": False}
        self.video_loader = object()
        self.frame_number = 0
        self.num_frames = 10
        self.step_size = 1
        self._suppress_audio_seek = False
        self.uniqLabelList = _UniqLabelListStub()
        self._uniq_label_selection_connected = False
        self.set_frame_calls: list[int] = []

    def _set_active_view(self, _view: str) -> None:
        return

    def mayContinue(self) -> bool:  # noqa: N802
        return True

    def set_frame_number(self, frame_number: int) -> None:
        self.frame_number = int(frame_number)
        self.set_frame_calls.append(int(frame_number))

    def _set_seekbar_value_without_signal(self, _value: int) -> None:
        return

    def _update_frame_display_and_emit_update(self) -> None:
        return

    def handle_uniq_label_list_selection_change(self) -> None:
        return


def test_open_next_img_connects_uniq_label_selection_signal_only_once() -> None:
    host = _NavigationHost()

    host.openNextImg()
    host.openNextImg()
    host.openNextImg()

    assert host.uniqLabelList.itemSelectionChanged.connect_calls == 1
