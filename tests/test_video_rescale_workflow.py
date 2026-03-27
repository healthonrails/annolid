from __future__ import annotations

from qtpy import QtCore

from annolid.gui.widgets.video_rescale_workflow import VideoRescaleWorkflow


class _DummyDialog(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()


class _FakeWorker:
    def __init__(self) -> None:
        self.deleted = False

    def deleteLater(self) -> None:
        self.deleted = True


class _FakeThread:
    def __init__(self, running: bool = False) -> None:
        self._running = running
        self.deleted = False
        self.wait_called = False

    def isRunning(self) -> bool:
        return self._running

    def deleteLater(self) -> None:
        self.deleted = True

    def wait(self, _timeout: int = 0) -> None:
        self.wait_called = True
        raise AssertionError("cleanup should not wait on the thread")


def test_cleanup_thread_does_not_wait_on_finished_thread() -> None:
    workflow = VideoRescaleWorkflow(_DummyDialog())
    worker = _FakeWorker()
    thread = _FakeThread(running=False)
    workflow._worker = worker
    workflow._thread = thread

    workflow._cleanup_thread()

    assert workflow._worker is None
    assert workflow._thread is None
    assert worker.deleted is True
    assert thread.deleted is True
    assert thread.wait_called is False
