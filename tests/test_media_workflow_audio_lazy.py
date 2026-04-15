from __future__ import annotations

import annolid.gui.mixins.media_workflow_mixin as media_module
from annolid.gui.mixins.media_workflow_mixin import MediaWorkflowMixin


class _MediaHost(MediaWorkflowMixin):
    def __init__(self) -> None:
        self.audio_widget = None
        self._audio_loader = None


def test_configure_audio_deferred_does_not_decode_immediately(monkeypatch) -> None:
    class _FakeAudioLoader:
        calls = 0

        def __init__(self, file_path: str, fps: float = 29.97):  # noqa: ARG002
            type(self).calls += 1

        def stop(self) -> None:
            return None

    monkeypatch.setattr(media_module, "AudioLoader", _FakeAudioLoader)
    host = _MediaHost()

    host._configure_audio_for_video("/tmp/demo.mp4", 15.0, eager=False)

    assert _FakeAudioLoader.calls == 0
    assert host._audio_loader is None
    assert getattr(host, "_pending_audio_video_path", None) == "/tmp/demo.mp4"
    assert getattr(host, "_pending_audio_fps", None) == 15.0


def test_active_audio_loader_ensure_ready_materializes_pending_loader(
    monkeypatch,
) -> None:
    host = _MediaHost()
    host._configure_audio_for_video("/tmp/demo.mp4", 20.0, eager=False)
    called = {"count": 0}

    def _fake_ensure_async() -> None:
        called["count"] += 1

    monkeypatch.setattr(host, "_ensure_audio_loader_async", _fake_ensure_async)
    loader = host._active_audio_loader(ensure_ready=True)

    assert loader is None
    assert called["count"] == 1
    assert getattr(host, "_pending_audio_video_path", None) == "/tmp/demo.mp4"
    assert getattr(host, "_pending_audio_fps", None) == 20.0


def test_audio_ready_callback_starts_audio_when_playing(monkeypatch) -> None:
    host = _MediaHost()
    host.isPlaying = True
    host.frame_number = 12
    host._pending_audio_request_token = "token-1"

    cleanup_calls = {"count": 0}

    def _fake_cleanup() -> None:
        cleanup_calls["count"] += 1

    monkeypatch.setattr(host, "_cancel_audio_loader_job", _fake_cleanup)

    class _Loader:
        def __init__(self) -> None:
            self.play_calls = []

        def play(self, start_frame=None):
            self.play_calls.append(start_frame)

        def stop(self) -> None:
            return None

    loader = _Loader()
    host._on_audio_loader_async_ready(loader, "token-1")

    assert cleanup_calls["count"] == 1
    assert host._audio_loader is loader
    assert loader.play_calls == [12]
