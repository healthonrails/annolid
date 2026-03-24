from __future__ import annotations

from pathlib import Path

from qtpy import QtCore

import annolid.gui.widgets.optical_flow_tool as optical_flow_tool_mod
from annolid.gui.widgets.optical_flow_tool import FlowRunSettings, OpticalFlowTool


class _FakeSignal:
    def __init__(self) -> None:
        self.connections: list[tuple[object, object]] = []

    def connect(self, slot, connection_type=None) -> None:
        self.connections.append((slot, connection_type))


class _FakeWorker:
    def __init__(self, *_args, **kwargs) -> None:
        self._kwargs = kwargs
        self.progress_signal = _FakeSignal()
        self.preview_signal = _FakeSignal()
        self.finished_signal = _FakeSignal()
        self.moved_to = None

    def moveToThread(self, thread) -> None:
        self.moved_to = thread

    def run(self) -> None:
        return None

    def deleteLater(self) -> None:
        return None


class _FakeThread:
    current = None

    def __init__(self, _parent=None) -> None:
        self.started = _FakeSignal()
        self.finished = _FakeSignal()
        self.running = False
        self.quit_called = False
        self.wait_called = False
        self.delete_later_called = False

    def start(self) -> None:
        self.running = True

    def quit(self) -> None:
        self.quit_called = True
        self.running = False

    def wait(self, _timeout: int = 0) -> None:
        self.wait_called = True

    def isRunning(self) -> bool:
        return self.running

    def deleteLater(self) -> None:
        self.delete_later_called = True

    @staticmethod
    def currentThread():
        return _FakeThread.current


class _FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, message: str, timeout: int) -> None:
        self.messages.append((message, timeout))


class _FakeWindow(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self._status = _FakeStatusBar()

    def statusBar(self) -> _FakeStatusBar:
        return self._status

    def tr(self, text: str) -> str:
        return text


def test_start_worker_uses_queued_connections(monkeypatch) -> None:
    window = _FakeWindow()
    tool = OpticalFlowTool(window)

    monkeypatch.setattr(optical_flow_tool_mod, "FlexibleWorker", _FakeWorker)
    monkeypatch.setattr(optical_flow_tool_mod.QtCore, "QThread", _FakeThread)

    settings = FlowRunSettings(
        video_path="/tmp/input.mp4", ndjson_path="/tmp/out.ndjson"
    )
    tool._start_worker(settings)

    worker = tool._worker
    thread = tool._worker_thread
    assert worker is not None
    assert thread is not None

    assert worker.progress_signal.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.preview_signal.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.finished_signal.connections[0][0] == tool._on_worker_finished
    assert worker.finished_signal.connections[0][1] == QtCore.Qt.QueuedConnection
    assert worker.finished_signal.connections[1][0] == thread.quit
    assert worker.finished_signal.connections[1][1] == QtCore.Qt.QueuedConnection
    assert thread.started.connections[0][1] == QtCore.Qt.QueuedConnection


def test_on_worker_finished_avoids_waiting_on_self_thread(monkeypatch) -> None:
    window = _FakeWindow()
    tool = OpticalFlowTool(window)
    thread = _FakeThread(window)
    thread.running = True
    _FakeThread.current = thread

    loaded_paths: list[Path] = []
    monkeypatch.setattr(
        tool,
        "_load_records_from_path",
        lambda path: loaded_paths.append(path),
    )
    monkeypatch.setattr(
        optical_flow_tool_mod.QtWidgets.QMessageBox,
        "information",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        optical_flow_tool_mod.QtWidgets.QMessageBox,
        "critical",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(optical_flow_tool_mod.QtCore, "QThread", _FakeThread)

    tool._worker_thread = thread
    tool._active_ndjson_path = Path("/tmp/run.ndjson")
    tool._on_worker_finished("ok")

    assert thread.quit_called is True
    assert thread.wait_called is False
    assert thread.delete_later_called is True
    assert loaded_paths == [Path("/tmp/run.ndjson")]
    assert tool._worker_thread is None
    assert tool._active_ndjson_path is None
