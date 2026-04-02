from __future__ import annotations

import json
from pathlib import Path

import annolid.gui.mixins.persistence_lifecycle_mixin as persistence_mod
from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin


class _DummyStore:
    def __init__(self) -> None:
        self.last_after_call = None
        self.last_range_call = None

    def remove_frames_in_range(
        self, start_frame, end_frame, protected_frames=None
    ) -> int:
        self.last_range_call = (start_frame, end_frame, set(protected_frames or []))
        return 0

    def remove_frames_after(self, frame_threshold, protected_frames=None) -> int:
        self.last_after_call = (frame_threshold, set(protected_frames or []))
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
        self._delete_all_called = 0

    def deleteAllFuturePredictions(self):
        self._delete_all_called += 1


def test_collect_seed_frames_prefers_manual_label_discovery(tmp_path: Path) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    window = _DummyWindow(folder, frame_number=10)
    for frame in (10, 20):
        (folder / f"video_results_{frame:09d}.json").write_text("{}", encoding="utf-8")
        (folder / f"video_results_{frame:09d}.png").write_text("seed", encoding="utf-8")
    (folder / "video_results_000000030.json").write_text("{}", encoding="utf-8")
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
    (folder / "video_results_000000010.png").write_text("seed", encoding="utf-8")
    (folder / "video_results_000000020.png").write_text("seed", encoding="utf-8")
    window = _DummyWindow(folder, frame_number=10)
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


def test_delete_file_yes_to_all_triggers_single_future_cleanup(
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
            raise AssertionError("Follow-up confirmation should not be shown.")

    monkeypatch.setattr(persistence_mod.QtWidgets, "QMessageBox", _FakeMessageBox)
    window.deleteFile()

    assert window._delete_all_called == 1


def test_delete_all_future_predictions_preserves_manual_seed_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    window = _DummyWindow(folder, frame_number=10)

    # Current frame and mixed future predictions.
    for frame in (10, 11, 12, 13):
        (folder / f"video_results_{frame:09d}.json").write_text("{}", encoding="utf-8")

    # Manual seed pair for frame 12 should be preserved.
    (folder / "video_results_000000012.png").write_text("seed", encoding="utf-8")
    stats_path = folder / "video_results_tracking_stats.json"
    stats_path.write_text(
        """
{
  "frame_stats": {
    "11": {"missing_instance_count": 1},
    "12": {"sources": ["manual_seed"], "missing_instance_count": 1},
    "13": {"missing_instance_count": 1}
  },
  "prediction_segments": [
    {"start_frame": 11, "end_frame": 13, "status": "halted"},
    {"start_frame": 20, "end_frame": 30, "status": "processed"}
  ],
  "summary": {}
}
""".strip(),
        encoding="utf-8",
    )

    store = _DummyStore()
    monkeypatch.setattr(
        persistence_mod.AnnotationStore,
        "for_frame_path",
        lambda _p: store,
    )

    window.deleteAllFuturePredictions()

    assert (folder / "video_results_000000010.json").exists() is True
    assert (folder / "video_results_000000011.json").exists() is False
    assert (folder / "video_results_000000012.json").exists() is True
    assert (folder / "video_results_000000012.png").exists() is True
    # Keep future predictions at/after nearest future seed frame (12).
    assert (folder / "video_results_000000013.json").exists() is True
    assert store.last_after_call is None
    assert store.last_range_call == (11, 11, {12})
    assert window._prediction_forced_start_frame == 11
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    frame_stats = payload.get("frame_stats", {})
    assert "11" not in frame_stats
    assert "12" in frame_stats
    assert "13" in frame_stats


def test_delete_all_future_predictions_sets_restart_hint_even_without_removals(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    # Current frame only; no future predictions to remove.
    (folder / "video_results_000000027.json").write_text("{}", encoding="utf-8")
    (folder / "video_results_000000027.png").write_text("seed", encoding="utf-8")
    window = _DummyWindow(folder, frame_number=27)
    monkeypatch.setattr(
        persistence_mod.AnnotationStore,
        "for_frame_path",
        lambda _p: _DummyStore(),
    )

    window.deleteAllFuturePredictions()

    assert window._prediction_forced_start_frame == 28


def test_delete_all_future_predictions_without_future_seed_deletes_all_future(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "video_results"
    folder.mkdir(parents=True, exist_ok=True)
    window = _DummyWindow(folder, frame_number=10)

    for frame in (10, 11, 12, 13):
        (folder / f"video_results_{frame:09d}.json").write_text("{}", encoding="utf-8")

    # Only current seed exists; no future seed pair.
    (folder / "video_results_000000010.png").write_text("seed", encoding="utf-8")
    stats_path = folder / "video_results_tracking_stats.json"
    stats_path.write_text(
        """
{
  "frame_stats": {
    "10": {"sources": ["manual_seed"]},
    "11": {"missing_instance_count": 1},
    "12": {"missing_instance_count": 1}
  },
  "prediction_segments": [
    {"start_frame": 11, "end_frame": 12, "status": "halted"}
  ],
  "summary": {}
}
""".strip(),
        encoding="utf-8",
    )

    store = _DummyStore()
    monkeypatch.setattr(
        persistence_mod.AnnotationStore,
        "for_frame_path",
        lambda _p: store,
    )

    window.deleteAllFuturePredictions()

    assert (folder / "video_results_000000010.json").exists() is True
    assert (folder / "video_results_000000011.json").exists() is False
    assert (folder / "video_results_000000012.json").exists() is False
    assert (folder / "video_results_000000013.json").exists() is False
    assert store.last_after_call == (10, {10})
    assert store.last_range_call is None
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    frame_stats = payload.get("frame_stats", {})
    assert "10" in frame_stats
    assert "11" not in frame_stats
    assert "12" not in frame_stats
