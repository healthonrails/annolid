from __future__ import annotations

import json
from pathlib import Path

from annolid.gui import workers as workers_mod
from annolid.gui.mixins.prediction_progress_mixin import PredictionProgressMixin
from annolid.gui.mixins import prediction_progress_mixin as progress_mod


class _DummyPredictionProgress(PredictionProgressMixin):
    pass


class _SeekbarStub:
    def __init__(self) -> None:
        self._marks = []
        self.blocked = False
        self.updated = False

    def getMarks(self, mark_type: str = ""):
        if not mark_type:
            return list(self._marks)
        return [mark for mark in self._marks if mark.mark_type == mark_type]

    def addMark(self, mark, update: bool = True) -> None:
        self._marks.append(mark)
        if update:
            self.updated = True

    def removeMarksByType(self, mark_type: str) -> None:
        self._marks = [mark for mark in self._marks if mark.mark_type != mark_type]

    def blockSignals(self, enabled: bool) -> None:
        self.blocked = bool(enabled)

    def update(self) -> None:
        self.updated = True


def test_finalize_prediction_progress_queues_when_off_gui_thread(monkeypatch) -> None:
    obj = _DummyPredictionProgress()
    queued = []

    def _fake_dispatch(callback):
        queued.append(callback)
        return True

    monkeypatch.setattr(obj, "_dispatch_to_gui_thread", _fake_dispatch)
    obj._finalize_prediction_progress("Track All run finished.")
    assert len(queued) == 1


def test_setup_prediction_watcher_queues_when_off_gui_thread(monkeypatch) -> None:
    obj = _DummyPredictionProgress()
    queued = []

    def _fake_dispatch(callback):
        queued.append(callback)
        return True

    monkeypatch.setattr(obj, "_dispatch_to_gui_thread", _fake_dispatch)
    obj._setup_prediction_folder_watcher("/tmp/fake_results", start_frame=5)
    assert len(queued) == 1


def test_stop_prediction_watcher_queues_when_off_gui_thread(monkeypatch) -> None:
    obj = _DummyPredictionProgress()
    queued = []

    def _fake_dispatch(callback):
        queued.append(callback)
        return True

    monkeypatch.setattr(obj, "_dispatch_to_gui_thread", _fake_dispatch)
    obj._stop_prediction_folder_watcher()
    assert len(queued) == 1


def test_flexible_worker_request_stop_is_immediate() -> None:
    class Target:
        def __init__(self) -> None:
            self.stop_requested = False

        def run(self) -> str:
            return "done"

        def request_stop(self) -> None:
            self.stop_requested = True

    target = Target()
    worker = workers_mod.FlexibleWorker(target.run)

    worker.request_stop()

    assert worker.is_stopped() is True
    assert worker.stop_event.is_set() is True
    assert target.stop_requested is True


def test_flexible_worker_reads_target_termination_policy() -> None:
    class Target:
        allow_force_thread_terminate = False

        def run(self) -> str:
            return "done"

    worker = workers_mod.FlexibleWorker(Target().run)

    assert worker.allows_force_terminate() is False


def test_force_stop_respects_worker_termination_policy(monkeypatch) -> None:
    class FakeThread:
        def __init__(self) -> None:
            self.terminated = False

        def isRunning(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminated = True

    class FakeWorker:
        stop_event = None

        def is_stopped(self) -> bool:
            return True

        def allows_force_terminate(self) -> bool:
            return False

    scheduled = []
    monkeypatch.setattr(progress_mod.QtCore, "QThread", FakeThread)
    monkeypatch.setattr(
        progress_mod.QtCore.QTimer,
        "singleShot",
        lambda delay, callback: scheduled.append((delay, callback)),
    )

    obj = _DummyPredictionProgress()
    thread = FakeThread()
    obj.seg_pred_thread = thread
    obj.pred_worker = FakeWorker()
    obj._force_stop_thread_ref = thread
    obj._force_stop_attempts = 3
    obj._prediction_stop_requested = True

    obj._force_stop_prediction_thread()

    assert thread.terminated is False
    assert scheduled and scheduled[-1][0] == 5000


def test_mark_missing_instance_frame_adds_and_deduplicates() -> None:
    obj = _DummyPredictionProgress()
    obj.seekbar = _SeekbarStub()
    obj.num_frames = 50

    obj._mark_missing_instance_frame(12)
    obj._mark_missing_instance_frame(12)

    marks = obj.seekbar.getMarks("missing_instance")
    assert len(marks) == 1
    assert int(marks[0].val) == 12


def test_clear_missing_instance_marks_removes_marks() -> None:
    obj = _DummyPredictionProgress()
    obj.seekbar = _SeekbarStub()
    obj.num_frames = 50
    obj._mark_missing_instance_frame(10)
    assert obj.seekbar.getMarks("missing_instance")

    obj._clear_missing_instance_marks()

    assert obj.seekbar.getMarks("missing_instance") == []


def test_refresh_missing_instance_marks_from_tracking_stats(tmp_path: Path) -> None:
    obj = _DummyPredictionProgress()
    obj.seekbar = _SeekbarStub()
    obj.num_frames = 40

    results_dir = tmp_path / "video_a"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats_path = results_dir / "video_a_tracking_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "version": 4,
                "frame_stats": {
                    "3": {"missing_instance_count": 1},
                    "11": {"unresolved_missing_instance_count": 2},
                    "99": {"missing_instance_count": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    obj._refresh_missing_instance_slider_marks_from_tracking_stats(results_dir)

    marked = sorted(int(mark.val) for mark in obj.seekbar.getMarks("missing_instance"))
    assert marked == [3, 11]


def test_frames_to_intervals_accepts_cached_intervals() -> None:
    intervals = _DummyPredictionProgress._frames_to_intervals([(10, 20), (1, 3)])

    assert intervals == [(1, 3), (10, 20)]
