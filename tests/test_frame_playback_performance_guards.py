from __future__ import annotations

from annolid.gui.mixins.frame_playback_mixin import FramePlaybackMixin


class _DummyVideoLoader:
    def get_time_stamp(self):
        return 0.0


class _DummyFrameLoader:
    def __init__(self) -> None:
        self.requests: list[int] = []

    def request(self, frame_number: int) -> None:
        self.requests.append(int(frame_number))


class _DummyCaptionWidget:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def set_image_path(self, path: str) -> None:
        self.paths.append(str(path))


class _PlaybackHost(FramePlaybackMixin):
    def __init__(self) -> None:
        self.num_frames = 1000
        self.frame_number = 0
        self.timeline_panel = None
        self.isPlaying = False
        self._suppress_audio_seek = False
        self.video_results_folder = None
        self.filename = ""
        self.video_loader = _DummyVideoLoader()
        self.frame_loader = _DummyFrameLoader()
        self.caption_widget = _DummyCaptionWidget()
        self.embedding_updates = 0

    def _update_audio_playhead(self, _frame_number: int) -> None:
        return

    def _active_audio_loader(self, ensure_ready: bool = False):  # noqa: ARG002
        return None

    def _update_embedding_query_frame(self) -> None:
        self.embedding_updates += 1


def test_set_frame_number_throttles_embedding_updates_during_playback() -> None:
    host = _PlaybackHost()
    host.isPlaying = True

    host.set_frame_number(11)
    assert host.embedding_updates == 0

    host.set_frame_number(15)
    assert host.embedding_updates == 1


def test_set_frame_number_updates_embedding_when_not_playing() -> None:
    host = _PlaybackHost()
    host.isPlaying = False

    host.set_frame_number(11)
    assert host.embedding_updates == 1
