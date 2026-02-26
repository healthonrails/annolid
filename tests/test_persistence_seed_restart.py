from __future__ import annotations

from pathlib import Path

import annolid.gui.mixins.persistence_lifecycle_mixin as persistence_mod
from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin


class _DummyStore:
    def remove_frames_in_range(self, *_args, **_kwargs) -> int:
        return 0


class _DummyWindow(PersistenceLifecycleMixin):
    def __init__(self, results_dir: Path, frame_number: int) -> None:
        self.video_loader = object()
        self.video_results_folder = str(results_dir)
        self.frame_number = int(frame_number)
        self.seekbar = None
        self.last_known_predicted_frame = -1
        self.prediction_start_timestamp = 0.0
        self._prediction_forced_start_frame = None

    def _scan_prediction_folder(self, *_args, **_kwargs) -> None:
        return None

    def tr(self, text: str) -> str:
        return text


class _DummyDeleteFlowWindow(_DummyWindow):
    def __init__(self, results_dir: Path, frame_number: int) -> None:
        super().__init__(results_dir, frame_number)
        self._delete_seed_called = 0
        self._delete_all_called = 0

    def deletePredictionsFromSeedToNext(self):
        self._delete_seed_called += 1
        return True, 10, 20

    def deleteAllFuturePredictions(self):
        self._delete_all_called += 1


def test_collect_seed_frames_prefers_manual_label_discovery(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    window = _DummyWindow(folder, frame_number=10)

    monkeypatch.setattr(
        persistence_mod,
        "find_manual_labeled_json_files",
        lambda _p: [
            "video_results_000000010.json",
            "video_results_000000020.json",
        ],
    )
    seeds = window._collect_seed_frames(folder)
    assert seeds == {10, 20}


def test_delete_predictions_from_seed_sets_forced_restart_hint(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    # Seed frames 10 and 20. Predictions in-between should be removed.
    for frame in (10, 11, 12, 20):
        (folder / f"video_results_{frame:09d}.json").write_text("{}", encoding="utf-8")
    window = _DummyWindow(folder, frame_number=10)

    monkeypatch.setattr(
        persistence_mod,
        "find_manual_labeled_json_files",
        lambda _p: [
            "video_results_000000010.json",
            "video_results_000000020.json",
        ],
    )
    monkeypatch.setattr(
        persistence_mod.AnnotationStore,
        "for_frame_path",
        lambda _p: _DummyStore(),
    )

    removed, current_seed, next_seed = window.deletePredictionsFromSeedToNext()
    assert removed is True
    assert current_seed == 10
    assert next_seed == 20
    assert window._prediction_forced_start_frame == 11
    assert (folder / "video_results_000000011.json").exists() is False
    assert (folder / "video_results_000000012.json").exists() is False
    assert (folder / "video_results_000000010.json").exists() is True
    assert (folder / "video_results_000000020.json").exists() is True


def test_delete_file_yes_to_all_triggers_seed_and_followup_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    window = _DummyDeleteFlowWindow(folder, frame_number=10)

    class _FakeMessageBox:
        No = 0
        Yes = 1
        YesToAll = 2
        Warning = 3

        def __init__(self, *_args, **_kwargs):
            self._answer = self.YesToAll

        def setIcon(self, *_args, **_kwargs):
            return None

        def setText(self, *_args, **_kwargs):
            return None

        def setInformativeText(self, *_args, **_kwargs):
            return None

        def setStandardButtons(self, *_args, **_kwargs):
            return None

        def setDefaultButton(self, *_args, **_kwargs):
            return None

        def exec_(self):
            return self._answer

        @staticmethod
        def question(*_args, **_kwargs):
            return _FakeMessageBox.Yes

    monkeypatch.setattr(persistence_mod.QtWidgets, "QMessageBox", _FakeMessageBox)
    window.deleteFile()

    assert window._delete_seed_called == 1
    assert window._delete_all_called == 1
