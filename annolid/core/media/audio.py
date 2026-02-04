from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Tuple
import warnings

import numpy as np


class AudioLoadError(RuntimeError):
    pass


def _load_audio_ffmpeg(
    file_path: str,
    *,
    sample_rate: int,
    channels: int = 1,
    timeout: Optional[float] = None,
) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioLoadError("ffmpeg was not found on PATH.")

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-vn",
        "-ac",
        str(int(channels)),
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:  # pragma: no cover
        raise AudioLoadError(f"Failed to spawn ffmpeg: {exc}") from exc

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        out, err = proc.communicate()
        raise AudioLoadError("ffmpeg timed out while decoding audio.") from exc

    if proc.returncode != 0:
        detail = (err or b"").decode("utf-8", errors="replace").strip()
        raise AudioLoadError(f"ffmpeg failed to decode audio: {detail}")

    if not out:
        raise AudioLoadError("No audio samples were decoded from the file.")

    audio = np.frombuffer(out, dtype=np.float32)
    return audio, int(sample_rate)


@dataclass
class AudioBuffer:
    """Audio samples aligned to video frames (GUI-free).

    This class intentionally avoids playback concerns; playback is handled by
    `annolid.utils.audio_playback` and GUI widgets.
    """

    audio_data: np.ndarray
    sample_rate: int
    fps: float = 29.97
    playhead_sample: int = 0

    @classmethod
    def from_file(
        cls,
        file_path: str,
        *,
        fps: float = 29.97,
        sample_rate: Optional[int] = None,
        ffmpeg_sample_rate: int = 48_000,
    ) -> "AudioBuffer":
        """Load audio from an audio/video file.

        Prefers `librosa` when available, but falls back to decoding via `ffmpeg`
        (which is commonly available in Annolid environments) for video files or
        when librosa's backend cannot open the media.
        """
        data: np.ndarray
        sr: int

        # Video containers are best handled by ffmpeg to avoid librosa's
        # deprecated audioread fallback path.
        video_suffixes = {
            ".mp4",
            ".m4v",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".wmv",
            ".flv",
            ".mpg",
            ".mpeg",
            ".ts",
            ".m2ts",
        }
        looks_like_video = Path(file_path).suffix.lower() in video_suffixes

        try:
            import librosa  # type: ignore
        except Exception:
            librosa = None  # type: ignore

        if librosa is not None and not looks_like_video:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*librosa\.core\.audio\.__audioread_load.*",
                        category=FutureWarning,
                    )
                    data, sr = librosa.load(file_path, sr=sample_rate)
                return cls(
                    audio_data=np.asarray(data),
                    sample_rate=int(sr),
                    fps=float(fps) if fps and fps > 0 else 29.97,
                )
            except Exception:
                # Fall back to ffmpeg below.
                pass

        target_sr = (
            int(sample_rate) if sample_rate is not None else int(ffmpeg_sample_rate)
        )
        data, sr = _load_audio_ffmpeg(file_path, sample_rate=target_sr, channels=1)
        return cls(
            audio_data=np.asarray(data),
            sample_rate=int(sr),
            fps=float(fps) if fps and fps > 0 else 29.97,
        )

    def frame_to_sample_index(self, frame_index: int) -> int:
        frame_index = max(int(frame_index), 0)
        if self.sample_rate <= 0:
            return 0
        samples = int(round(frame_index / float(self.fps) * self.sample_rate))
        return max(0, min(samples, int(self.audio_data.shape[0])))

    def samples_for_frame(self, frame_index: int) -> np.ndarray:
        start = self.frame_to_sample_index(frame_index)
        duration_sec = 1.0 / float(self.fps) if self.fps and self.fps > 0 else 0.0
        count = int(round(duration_sec * self.sample_rate))
        end = max(start, min(start + max(count, 0), int(self.audio_data.shape[0])))
        return self.audio_data[start:end]

    def set_playhead_frame(self, frame_index: int) -> None:
        self.playhead_sample = self.frame_to_sample_index(frame_index)

    def slice_from_playhead(self) -> np.ndarray:
        start = max(0, min(int(self.playhead_sample), int(self.audio_data.shape[0])))
        return self.audio_data[start:]

    def slice_seconds(self, start_sec: float, end_sec: float) -> np.ndarray:
        if self.sample_rate <= 0:
            return self.audio_data[:0]
        start = int(round(max(float(start_sec), 0.0) * self.sample_rate))
        end = int(round(max(float(end_sec), 0.0) * self.sample_rate))
        start = max(0, min(start, int(self.audio_data.shape[0])))
        end = max(start, min(end, int(self.audio_data.shape[0])))
        return self.audio_data[start:end]

    def seconds_for_frame(self, frame_index: int) -> Optional[float]:
        if not self.fps or self.fps <= 0:
            return None
        return float(max(int(frame_index), 0)) / float(self.fps)

    def frame_time_span(
        self, frame_index: int
    ) -> Tuple[Optional[float], Optional[float]]:
        start = self.seconds_for_frame(frame_index)
        if start is None or not self.fps or self.fps <= 0:
            return None, None
        end = start + (1.0 / float(self.fps))
        return start, end
