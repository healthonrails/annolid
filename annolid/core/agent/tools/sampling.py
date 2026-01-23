from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


def _normalize_indices(indices: Iterable[int], total_frames: int) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for idx in indices:
        if idx < 0 or idx >= total_frames:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return sorted(ordered)


@dataclass(frozen=True)
class UniformSampler:
    step: int = 1

    def sample_indices(
        self, total_frames: int, *, fps: Optional[float] = None
    ) -> List[int]:
        step = max(1, int(self.step))
        return list(range(0, max(int(total_frames), 0), step))


@dataclass(frozen=True)
class FPSampler:
    target_fps: float

    def sample_indices(
        self, total_frames: int, *, fps: Optional[float] = None
    ) -> List[int]:
        if fps is None or fps <= 0:
            return UniformSampler(1).sample_indices(total_frames)
        if self.target_fps <= 0:
            return UniformSampler(1).sample_indices(total_frames)
        step = max(1, int(round(float(fps) / float(self.target_fps))))
        return UniformSampler(step).sample_indices(total_frames)


@dataclass(frozen=True)
class RandomSampler:
    count: int
    seed: Optional[int] = None
    replace: bool = False
    include_ends: bool = False

    def sample_indices(
        self, total_frames: int, *, fps: Optional[float] = None
    ) -> List[int]:
        total = max(int(total_frames), 0)
        if total == 0:
            return []
        rng = random.Random(self.seed)
        population = list(range(total))
        k = (
            min(max(int(self.count), 0), total)
            if not self.replace
            else max(int(self.count), 0)
        )
        if k == 0:
            return []
        if self.replace:
            picks = [rng.choice(population) for _ in range(k)]
        else:
            picks = rng.sample(population, k)
        if self.include_ends:
            picks.extend([0, total - 1])
        return _normalize_indices(picks, total)


@dataclass(frozen=True)
class MotionSampler:
    threshold: float
    min_step: int = 1

    def sample_from_scores(self, motion_scores: Sequence[float]) -> List[int]:
        total = len(motion_scores)
        if total == 0:
            return []
        step = max(1, int(self.min_step))
        picks: List[int] = []
        last_pick = -step
        for idx, score in enumerate(motion_scores):
            if idx - last_pick < step:
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value >= float(self.threshold):
                picks.append(idx)
                last_pick = idx
        return _normalize_indices(picks, total)
