from __future__ import annotations

from annolid.core.agent.tools.sampling import (
    FPSampler,
    MotionSampler,
    RandomSampler,
    UniformSampler,
)


def test_uniform_sampler() -> None:
    sampler = UniformSampler(step=3)
    assert sampler.sample_indices(10) == [0, 3, 6, 9]


def test_fps_sampler() -> None:
    sampler = FPSampler(target_fps=5)
    assert sampler.sample_indices(10, fps=30) == [0, 6]


def test_random_sampler_reproducible() -> None:
    sampler = RandomSampler(count=3, seed=123)
    assert sampler.sample_indices(10) == [0, 1, 4]


def test_random_sampler_include_ends() -> None:
    sampler = RandomSampler(count=1, seed=1, include_ends=True)
    indices = sampler.sample_indices(5)
    assert 0 in indices
    assert 4 in indices


def test_motion_sampler_threshold() -> None:
    sampler = MotionSampler(threshold=0.5, min_step=1)
    scores = [0.1, 0.6, 0.2, 0.9, 0.4]
    assert sampler.sample_from_scores(scores) == [1, 3]
