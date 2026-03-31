from __future__ import annotations

import json
from pathlib import Path

from annolid.gui.mixins.prediction_progress_mixin import PredictionProgressMixin


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
