from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from annolid.data.audios import AudioLoader


def test_audio_loader_can_open_video_audio_via_ffmpeg(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not available on PATH.")

    out_path = tmp_path / "with_audio.mp4"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x48:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-shortest",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    loader = AudioLoader(str(out_path), fps=10.0)
    assert loader.sample_rate > 0
    assert loader.audio_data is not None
    assert getattr(loader.audio_data, "size", 0) > 0
    assert loader.load_audio_for_frame(0).size > 0
