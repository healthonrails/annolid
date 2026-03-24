from __future__ import annotations

from qtpy import QtCore

from annolid.gui.controllers.tracking import TrackingController


class _FakeSignal:
    def __init__(self) -> None:
        self.connections: list[tuple[object, object]] = []

    def connect(self, slot, connection_type=None) -> None:
        self.connections.append((slot, connection_type))


class _FakeStatusBar:
    def showMessage(self, _message: str, _timeout: int) -> None:
        return None


class _FakeWindow(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.ui_states: list[bool] = []
        self._status = _FakeStatusBar()

    def set_tracking_ui_state(self, *, is_tracking: bool) -> None:
        self.ui_states.append(is_tracking)

    def statusBar(self) -> _FakeStatusBar:
        return self._status


class _FakeTrackingWorker:
    def __init__(self) -> None:
        self.progress = _FakeSignal()
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.video_job_started = _FakeSignal()
        self.video_job_finished = _FakeSignal()


class _FakeTrackAllWorker:
    def __init__(self) -> None:
        self.progress = _FakeSignal()
        self.finished = _FakeSignal()
        self.error = _FakeSignal()
        self.video_processing_started = _FakeSignal()
        self.video_processing_finished = _FakeSignal()

    def isRunning(self) -> bool:
        return False


def test_connect_signals_to_active_worker_uses_queued_connections() -> None:
    window = _FakeWindow()
    controller = TrackingController(window)
    worker = _FakeTrackingWorker()

    controller._connect_signals_to_active_worker(worker)

    assert worker.progress.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.finished.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.error.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.video_job_started.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.video_job_finished.connections[0][1] == QtCore.Qt.QueuedConnection


def test_register_track_all_worker_uses_queued_connections() -> None:
    window = _FakeWindow()
    controller = TrackingController(window)
    worker = _FakeTrackAllWorker()

    controller.register_track_all_worker(worker)

    assert worker.progress.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.finished.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.error.connections[0][1] == QtCore.Qt.QueuedConnection
    assert (
        worker.video_processing_started.connections[0][1] == QtCore.Qt.QueuedConnection
    )
    assert (
        worker.video_processing_finished.connections[0][1] == QtCore.Qt.QueuedConnection
    )
    assert window.ui_states[-1] is True
