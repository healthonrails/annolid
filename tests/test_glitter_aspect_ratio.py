from pathlib import Path

import numpy as np

from annolid.postprocessing import glitter


def test_finalize_tracked_video_moves_without_ffmpeg_for_square_pixels(
    tmp_path, monkeypatch
):
    temp_video = tmp_path / "tracked_tmp.mp4"
    final_video = tmp_path / "tracked.mp4"
    temp_video.write_bytes(b"video")

    called = []

    def fake_run(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("ffmpeg should not be called for square pixels")

    monkeypatch.setattr(glitter.subprocess, "run", fake_run)

    result = glitter._finalize_tracked_video(temp_video, final_video, "1:1")

    assert result == final_video
    assert final_video.exists()
    assert final_video.read_bytes() == b"video"
    assert not temp_video.exists()
    assert called == []


def test_finalize_tracked_video_preserves_source_sar_with_ffmpeg(tmp_path, monkeypatch):
    temp_video = tmp_path / "tracked_tmp.mp4"
    final_video = tmp_path / "tracked.mp4"
    temp_video.write_bytes(b"video")

    def fake_run(cmd, check, capture_output, text):
        assert cmd[0] == "ffmpeg"
        assert "-vf" in cmd
        vf_index = cmd.index("-vf") + 1
        assert cmd[vf_index] == "scale=trunc(iw*29/18/2)*2:ih,setsar=1"
        output_path = Path(cmd[-1])
        output_path.write_bytes(b"remuxed")
        return None

    monkeypatch.setattr(glitter.shutil, "which", lambda name: name)
    monkeypatch.setattr(glitter.subprocess, "run", fake_run)

    result = glitter._finalize_tracked_video(temp_video, final_video, "29:18")

    assert result == final_video
    assert final_video.exists()
    assert final_video.read_bytes() == b"remuxed"
    assert not temp_video.exists()


def test_finalize_tracked_video_invalid_sar_falls_back_to_rename(tmp_path):
    temp_video = tmp_path / "tracked_tmp.mp4"
    final_video = tmp_path / "tracked.mp4"
    temp_video.write_bytes(b"video")

    result = glitter._finalize_tracked_video(temp_video, final_video, "bad")

    assert result == final_video
    assert final_video.exists()
    assert final_video.read_bytes() == b"video"
    assert not temp_video.exists()


def test_probe_sample_aspect_ratio_returns_none_when_ffprobe_missing(monkeypatch):
    monkeypatch.setattr(glitter.shutil, "which", lambda name: None)

    assert glitter._probe_sample_aspect_ratio("clip.mp4") is None


def test_apply_sample_aspect_ratio_to_frame_scales_width_for_display():
    frame = np.zeros((12, 10, 3), dtype=np.uint8)

    display = glitter._apply_sample_aspect_ratio_to_frame(frame, "29:18")

    assert display.shape[0] == 12
    assert display.shape[1] == 16


def test_apply_sample_aspect_ratio_to_frame_keeps_shape_for_square_sar():
    frame = np.zeros((12, 10, 3), dtype=np.uint8)

    display = glitter._apply_sample_aspect_ratio_to_frame(frame, "1:1")

    assert display.shape == frame.shape
