from __future__ import annotations

import numpy as np

import annolid.data.audios as audios
from annolid.core.media.audio import AudioBuffer


def test_audio_loader_slice_seconds_matches_expected_indices(monkeypatch) -> None:
    data = np.arange(2000, dtype=np.float32)
    fake = AudioBuffer(audio_data=data, sample_rate=1000, fps=10.0)

    monkeypatch.setattr(audios.AudioBuffer, "from_file", lambda *a, **k: fake)
    loader = audios.AudioLoader("dummy.mp4", fps=10.0)

    sliced = loader.slice_seconds(0.5, 1.0)
    assert sliced.size == 500
    assert float(sliced[0]) == 500.0
    assert float(sliced[-1]) == 999.0


def test_audio_loader_play_selected_part_plays_expected_slice(monkeypatch) -> None:
    data = np.arange(2000, dtype=np.float32)
    fake = AudioBuffer(audio_data=data, sample_rate=1000, fps=10.0)

    monkeypatch.setattr(audios.AudioBuffer, "from_file", lambda *a, **k: fake)
    played = {}

    def _fake_play(samples, sample_rate: int, *, blocking: bool = False) -> bool:  # noqa: ANN001
        played["samples"] = np.asarray(samples)
        played["sr"] = int(sample_rate)
        played["blocking"] = bool(blocking)
        return True

    monkeypatch.setattr(audios, "play_audio_buffer", _fake_play)
    loader = audios.AudioLoader("dummy.mp4", fps=10.0)

    loader.play_selected_part(0.5, 1.0)
    assert played["sr"] == 1000
    assert played["blocking"] is True
    assert played["samples"].size == 500
    assert float(played["samples"][0]) == 500.0
