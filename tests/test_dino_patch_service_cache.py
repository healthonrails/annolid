from __future__ import annotations

import numpy as np

from annolid.gui.dino_patch_service import _CachedFeatureExtractor


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(
        self,
        image,
        *,
        color_space: str = "RGB",
        return_type: str = "torch",
        return_layer: str | None = None,
        normalize: bool = True,
    ):
        _ = (image, color_space, return_type, return_layer, normalize)
        self.calls += 1
        return np.array([self.calls], dtype=np.float32)


def test_cached_feature_extractor_reuses_same_frame_embedding() -> None:
    base = _FakeExtractor()
    cached = _CachedFeatureExtractor(base)
    image = np.zeros((12, 10, 3), dtype=np.uint8)

    first = cached.extract(image, return_type="numpy", return_layer="last")
    second = cached.extract(image, return_type="numpy", return_layer="last")

    assert base.calls == 1
    assert np.array_equal(first, second)
    stats = cached.cache_stats
    assert stats["hits"] >= 1
    assert stats["misses"] == 1


def test_cached_feature_extractor_invalidates_on_frame_change() -> None:
    base = _FakeExtractor()
    cached = _CachedFeatureExtractor(base)
    frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_b = np.ones((8, 8, 3), dtype=np.uint8) * 255

    _ = cached.extract(frame_a, return_type="numpy", return_layer="last")
    _ = cached.extract(frame_b, return_type="numpy", return_layer="last")

    assert base.calls == 2
    stats = cached.cache_stats
    assert stats["misses"] == 2


def test_cached_feature_extractor_clear_cache_forces_recompute() -> None:
    base = _FakeExtractor()
    cached = _CachedFeatureExtractor(base)
    frame = np.zeros((6, 6, 3), dtype=np.uint8)

    _ = cached.extract(frame, return_type="numpy", return_layer="last")
    cached.clear_image_cache()
    _ = cached.extract(frame, return_type="numpy", return_layer="last")

    assert base.calls == 2
