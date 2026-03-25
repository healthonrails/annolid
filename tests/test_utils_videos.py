from __future__ import annotations

import subprocess
from pathlib import Path

import annolid.utils.videos as videos_mod


def test_compress_and_rescale_video_falls_back_to_supported_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 15.0)

    attempted_profiles = []

    def _fake_run(cmd, check, stdout, stderr, text):
        _ = check, stdout, stderr, text
        if "h264_videotoolbox" in cmd:
            attempted_profiles.append("h264_videotoolbox")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
        if "libx264" in cmd:
            attempted_profiles.append("libx264")
            raise subprocess.CalledProcessError(
                returncode=8,
                cmd=cmd,
                stderr="Unrecognized option 'preset'.",
            )
        attempted_profiles.append("other")
        raise AssertionError(f"Unexpected profile fallback cmd: {cmd}")

    monkeypatch.setattr(videos_mod.subprocess, "run", _fake_run)

    command_log = videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        apply_denoise=True,
    )

    assert attempted_profiles == ["h264_videotoolbox"]
    assert "clip_fix.mp4" in command_log
    assert "h264_videotoolbox" in command_log["clip_fix.mp4"]


def test_compress_and_rescale_video_returns_empty_when_all_profiles_fail(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 15.0)

    def _always_fail(cmd, check, stdout, stderr, text):
        _ = check, stdout, stderr, text
        raise subprocess.CalledProcessError(
            returncode=8,
            cmd=cmd,
            stderr="Unknown encoder",
        )

    monkeypatch.setattr(videos_mod.subprocess, "run", _always_fail)

    command_log = videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        apply_denoise=True,
    )

    assert command_log == {}


def test_compress_and_rescale_video_falls_back_when_denoise_filter_parse_fails(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 15.0)

    def _fake_run(cmd, check, stdout, stderr, text):
        _ = check, stdout, stderr, text
        vf = cmd[cmd.index("-vf") + 1]
        if "hqdn3d=" in vf:
            raise subprocess.CalledProcessError(
                returncode=8,
                cmd=cmd,
                stderr="Error parsing filterchain",
            )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(videos_mod.subprocess, "run", _fake_run)

    command_log = videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        apply_denoise=True,
    )

    assert "clip_fix.mp4" in command_log
    assert "no_denoise_fallback" not in command_log["clip_fix.mp4"]
    assert "hqdn3d=" not in command_log["clip_fix.mp4"]
    assert "scale=trunc(iw*0.5/2)*2:trunc(ih*0.5/2)*2" in command_log["clip_fix.mp4"]
