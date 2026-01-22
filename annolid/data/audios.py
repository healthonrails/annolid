from __future__ import annotations

from typing import Optional
import warnings

from annolid.core.media.audio import AudioBuffer
from annolid.utils.audio_playback import play_audio_buffer, stop_audio_playback
from annolid.utils.logger import logger

warnings.filterwarnings("ignore")


class AudioLoader:
    """Backward-compatible audio loader used by the GUI.

    The underlying buffer logic lives in `annolid.core.media.audio.AudioBuffer`
    so headless/CLI code can reuse frame alignment without importing playback
    dependencies.
    """

    def __init__(self, file_path: str, fps: float = 29.97):
        self._buffer = AudioBuffer.from_file(file_path, fps=float(fps))

    @property
    def audio_data(self):
        return self._buffer.audio_data

    @property
    def sample_rate(self) -> int:
        return int(self._buffer.sample_rate)

    @property
    def fps(self) -> float:
        return float(self._buffer.fps)

    def load_audio_for_frame(self, frame_number: int):
        return self._buffer.samples_for_frame(frame_number)

    def set_playhead_frame(self, frame_number: int) -> None:
        self._buffer.set_playhead_frame(frame_number)

    def slice_seconds(self, start_sec: float, end_sec: float):
        return self._buffer.slice_seconds(float(start_sec), float(end_sec))

    def play(self, start_frame: Optional[int] = None) -> None:
        if start_frame is not None:
            self.set_playhead_frame(start_frame)

        audio_to_play = self._buffer.slice_from_playhead()
        if audio_to_play is None or getattr(audio_to_play, "size", 0) == 0:
            return

        play_audio_buffer(audio_to_play, self.sample_rate, blocking=False)

    def play_selected_part(self, x_start: float, x_end: float) -> None:
        if x_end <= x_start:
            return
        selected_audio = self.slice_seconds(float(x_start), float(x_end))
        if selected_audio is None or getattr(selected_audio, "size", 0) == 0:
            logger.debug("Selected audio part is empty; skipping playback.")
            return
        play_audio_buffer(selected_audio, self.sample_rate, blocking=True)
        self._buffer.playhead_sample = int(round(float(x_start) * self.sample_rate))

    def stop(self) -> None:
        stop_audio_playback()
