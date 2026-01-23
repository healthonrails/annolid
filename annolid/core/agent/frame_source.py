from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

from annolid.core.media.video import CV2Video
from annolid.core.types import FrameRef

from .tools.base import FrameBatch, FrameData
from .tools.sampling import FPSampler, RandomSampler, UniformSampler


@dataclass(frozen=True)
class FrameSource:
    """Iterate over video frames using configurable sampling."""

    video_path: Path
    stride: int = 1
    target_fps: Optional[float] = None
    random_count: Optional[int] = None
    random_seed: Optional[int] = None
    random_replace: bool = False
    random_include_ends: bool = False

    def iter_frames(self) -> Iterator[FrameData]:
        video = CV2Video(self.video_path)
        try:
            total_frames = int(video.total_frames())
            fps = None
            try:
                fps = float(video.fps())
            except Exception:
                fps = None

            indices = self._select_indices(total_frames, fps=fps)
            for frame_index in indices:
                frame = video.load_frame(frame_index)
                ref = FrameRef(
                    frame_index=int(frame_index),
                    timestamp_sec=video.last_timestamp_sec(),
                )
                yield FrameData(ref=ref, image_rgb=frame, image_path=None, meta={})
        finally:
            video.release()

    def iter_batches(self, *, batch_size: int = 1) -> Iterator[FrameBatch]:
        batch: list[FrameData] = []
        for frame in self.iter_frames():
            batch.append(frame)
            if len(batch) >= batch_size:
                yield FrameBatch(frames=list(batch))
                batch.clear()
        if batch:
            yield FrameBatch(frames=list(batch))

    def _select_indices(
        self, total_frames: int, *, fps: Optional[float]
    ) -> Sequence[int]:
        if self.random_count is not None:
            sampler = RandomSampler(
                count=int(self.random_count),
                seed=self.random_seed,
                replace=bool(self.random_replace),
                include_ends=bool(self.random_include_ends),
            )
            return sampler.sample_indices(total_frames, fps=fps)

        if self.target_fps is not None:
            sampler = FPSampler(target_fps=float(self.target_fps))
            return sampler.sample_indices(total_frames, fps=fps)

        sampler = UniformSampler(step=max(1, int(self.stride)))
        return sampler.sample_indices(total_frames, fps=fps)
