from __future__ import annotations

from annolid.gui.mixins.prediction_progress_mixin import PredictionProgressMixin


class _DummyPredictionProgress(PredictionProgressMixin):
    pass


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
