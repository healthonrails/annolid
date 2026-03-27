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

    class _FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            mapping = {
                videos_mod.cv2.CAP_PROP_FRAME_COUNT: 100,
                videos_mod.cv2.CAP_PROP_FPS: 10.0,
            }
            return mapping.get(prop, 0)

        def release(self):
            return None

    monkeypatch.setattr(videos_mod.cv2, "VideoCapture", lambda _p: _FakeCapture())

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


def test_compress_and_rescale_video_supports_single_input_video_path(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "output"
    video_path.write_bytes(b"fake")
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 24.0)

    def _fake_run(cmd, check, stdout, stderr, text):
        _ = check, stdout, stderr, text
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(videos_mod.subprocess, "run", _fake_run)

    command_log = videos_mod.compress_and_rescale_video(
        input_folder=str(tmp_path),
        output_folder=str(output_dir),
        scale_factor=0.5,
        input_video_path=str(video_path),
        fps=24.0,
    )

    assert "clip.mp4" in command_log


def test_simple_fallback_uses_even_dimensions_from_crop_region(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 15.0)

    class _FakeCapture:
        def isOpened(self):
            return True

        def get(self, _prop):
            return 0

        def release(self):
            return None

    monkeypatch.setattr(videos_mod.cv2, "VideoCapture", lambda _p: _FakeCapture())

    attempted = {"simple": False}

    def _fake_run(cmd, check, stdout, stderr, text):
        _ = check, stdout, stderr, text
        if "-vf" in cmd:
            raise subprocess.CalledProcessError(
                returncode=8,
                cmd=cmd,
                stderr="Error parsing filterchain",
            )
        attempted["simple"] = True
        assert "-s" in cmd
        dims = cmd[cmd.index("-s") + 1]
        assert dims == "620x538"
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(videos_mod.subprocess, "run", _fake_run)

    command_log = videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        crop_x=0,
        crop_y=0,
        crop_width=1242,
        crop_height=1076,
    )

    assert attempted["simple"] is True
    assert "clip.mp4" in command_log


def test_compress_and_rescale_video_reports_progress(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 15.0)
    monkeypatch.setattr(
        videos_mod.subprocess,
        "run",
        lambda cmd, check, stdout, stderr, text: subprocess.CompletedProcess(
            cmd, returncode=0, stdout="", stderr=""
        ),
    )

    events: list[tuple[int, int, str]] = []

    videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        progress_callback=lambda current, total, message: events.append(
            (current, total, message)
        ),
    )

    assert events[0][2] == "Starting"
    assert any("Processing 1/1 - clip.mp4" in message for _, _, message in events)
    assert events[-1][2] == "Complete"


def test_cancel_aware_ffmpeg_loop_does_not_timeout(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 15.0)

    class _FakeProc:
        def __init__(self) -> None:
            self.calls = 0
            self.killed = False
            self.stdout = self
            self.stderr = self

        def poll(self):
            self.calls += 1
            return 0 if self.calls >= 2 else None

        def readline(self):
            return ""

        def read(self):
            return ""

        def communicate(self):
            return ("", "")

        def wait(self, timeout=None):
            _ = timeout
            return 0

        def terminate(self):
            return None

        def kill(self):
            self.killed = True

    fake_proc = _FakeProc()

    def _fake_popen(cmd, stdout=None, stderr=None, text=None, bufsize=None):
        _ = cmd, stdout, stderr, text
        _ = bufsize
        return fake_proc

    monkeypatch.setattr(videos_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        videos_mod.select,
        "select",
        lambda readable, writable, exceptional, timeout: (readable, [], []),
    )

    command_log = videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=15.0,
        cancel_callback=lambda: False,
    )

    assert "clip.mp4" in command_log
    assert fake_proc.killed is False


def test_cancel_poll_interval_uses_environment_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANNOLID_FFMPEG_CANCEL_POLL_INTERVAL", "0.05")

    assert videos_mod._ffmpeg_cancel_poll_interval() == 0.05


def test_compress_and_rescale_video_emits_streaming_progress(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "clip.mp4").write_bytes(b"fake")

    monkeypatch.setattr(videos_mod, "get_video_fps", lambda _p: 10.0)

    class _FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            mapping = {
                videos_mod.cv2.CAP_PROP_FRAME_COUNT: 100,
                videos_mod.cv2.CAP_PROP_FPS: 10.0,
            }
            return mapping.get(prop, 0)

        def release(self):
            return None

    monkeypatch.setattr(videos_mod.cv2, "VideoCapture", lambda _p: _FakeCapture())

    class _FakePipe:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines

        def readline(self):
            if self.lines:
                return self.lines.pop(0)
            return ""

        def read(self):
            return ""

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = _FakePipe(
                [
                    "out_time_ms=0\n",
                    "progress=continue\n",
                    "out_time_ms=5000\n",
                    "progress=continue\n",
                    "out_time_ms=10000\n",
                    "progress=end\n",
                ]
            )
            self.stderr = _FakePipe([])
            self._done = False

        def poll(self):
            if self.stdout.lines:
                return None
            self._done = True
            return 0

        def communicate(self):
            return ("", "")

        def wait(self, timeout=None):
            _ = timeout
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    monkeypatch.setattr(
        videos_mod.subprocess,
        "Popen",
        lambda cmd, stdout, stderr, text, bufsize: _FakeProc(),
    )

    selects: list[list[object]] = []

    def _fake_select(readable, writable, exceptional, timeout):
        _ = writable, exceptional, timeout
        selects.append(list(readable))
        stdout = next(
            (stream for stream in readable if hasattr(stream, "readline")), None
        )
        if stdout is not None and getattr(stdout, "lines", None):
            return ([stdout], [], [])
        return ([], [], [])

    monkeypatch.setattr(videos_mod.select, "select", _fake_select)

    events: list[tuple[int, int, str]] = []
    videos_mod.compress_and_rescale_video(
        str(input_dir),
        str(output_dir),
        scale_factor=0.5,
        fps=10.0,
        cancel_callback=lambda: False,
        progress_callback=lambda current, total, message: events.append(
            (current, total, message)
        ),
    )

    assert any(0 < current < 100 for current, _, _ in events)
    assert any("processing" in message.lower() for _, _, message in events)
