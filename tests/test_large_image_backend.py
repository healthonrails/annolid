from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile

from annolid.gui.large_image import (
    TIFF_SUFFIXES,
    OpenSlideBackend,
    TiffFileBackend,
    VipsBackend,
    available_large_image_backends,
    clear_all_large_image_caches,
    format_large_image_cache_size,
    is_large_tiff_path,
    large_image_cache_size_bytes,
    list_large_image_cache_entries,
    load_image_with_backends,
    open_large_image,
    optimized_large_image_cache_path,
    prune_large_image_caches,
    probe_large_image,
    remove_large_image_cache_file,
    sniff_large_image,
)


def test_large_tiff_suffix_detection() -> None:
    assert ".ome.tiff" in TIFF_SUFFIXES
    assert is_large_tiff_path("sample.ome.tiff")
    assert is_large_tiff_path("sample.tif")
    assert not is_large_tiff_path("sample.png")


def test_probe_and_load_tiff_with_tifffile(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.ome.tiff"
    data = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    tifffile.imwrite(image_path, data, ome=True)

    metadata = probe_large_image(image_path)
    loaded = load_image_with_backends(image_path)
    backend = open_large_image(image_path)
    sniffed = sniff_large_image(image_path)

    assert metadata is not None
    assert metadata.backend_name == "tifffile"
    assert metadata.width == 96
    assert metadata.height == 64
    assert sniffed["is_tiff_family"] is True
    assert sniffed["is_ome"] is True
    assert backend.name == "tifffile"
    assert backend.get_level_count() == 1
    assert backend.get_level_shape(0) == (96, 64)
    region = backend.read_region(10, 10, 20, 10, level=0)
    assert region.shape[:2] == (10, 20)
    assert loaded.qimage.width() == 96
    assert loaded.qimage.height() == 64
    assert loaded.metadata is not None


def test_probe_tiff_with_yxs_axes_reports_width_height_correctly(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "rgb_yxs.tif"
    data = np.zeros((64, 96, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, data)

    metadata = probe_large_image(image_path)

    assert metadata is not None
    assert metadata.width == 96
    assert metadata.height == 64
    assert metadata.channels == 3


def test_tifffile_backend_supports_page_navigation_for_multipage_tiff(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((32, 48), 10, dtype=np.uint8),
            np.full((32, 48), 200, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    backend = TiffFileBackend(image_path)
    metadata = backend.probe()
    first_region = backend.read_region(0, 0, 8, 8)
    backend.set_page(1)
    second_region = backend.read_region(0, 0, 8, 8)

    assert metadata is not None
    assert metadata.page_count == 2
    assert metadata.width == 48
    assert metadata.height == 32
    assert backend.get_page_count() == 2
    assert backend.get_current_page() == 1
    assert backend.get_level_shape(0) == (48, 32)
    assert int(first_region[0, 0]) == 10
    assert int(second_region[0, 0]) == 200


def test_tifffile_backend_supports_single_series_stack_page_navigation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "single_series_stack.tif"
    data = np.stack(
        [
            np.full((24, 36), 11, dtype=np.uint8),
            np.full((24, 36), 77, dtype=np.uint8),
            np.full((24, 36), 155, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data, metadata={"axes": "QYX"})

    backend = TiffFileBackend(image_path)
    metadata = backend.probe()
    first_region = backend.read_region(0, 0, 4, 4)
    backend.set_page(2)
    third_region = backend.read_region(0, 0, 4, 4)

    assert metadata is not None
    assert metadata.page_count == 3
    assert metadata.width == 36
    assert metadata.height == 24
    assert backend.get_page_count() == 3
    assert backend.get_current_page() == 2
    assert int(first_region[0, 0]) == 11
    assert int(third_region[0, 0]) == 155


def test_tifffile_backend_falls_back_to_pillow_for_missing_codec(monkeypatch) -> None:
    backend = TiffFileBackend("dummy.tif")
    expected = np.full((12, 8), 7, dtype=np.uint8)
    calls = {"count": 0}

    def fake_pillow(page_index: int):
        calls["count"] += 1
        assert page_index == 0
        return expected

    monkeypatch.setattr(backend, "_page_array_with_pillow", fake_pillow)

    class _FakePage:
        def asarray(self):
            raise ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")

    class _FakeTiffFile:
        def __init__(self, path):
            self.pages = [_FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    import tifffile

    monkeypatch.setattr(tifffile, "TiffFile", _FakeTiffFile)

    result = backend._page_array(0)

    assert calls["count"] == 1
    assert np.array_equal(result, expected)


def test_backend_registry_includes_tifffile_backend() -> None:
    names = [backend.name for backend in available_large_image_backends()]
    assert names[0] == "qt"
    assert "tifffile" in names
    assert isinstance(TiffFileBackend().name, str)


def test_tifffile_backend_thumbnail_is_bounded_for_large_non_pyramidal_tiff(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "large_rgb.tif"
    data = np.zeros((4096, 6144, 3), dtype=np.uint8)
    data[..., 0] = 17
    data[..., 1] = 33
    data[..., 2] = 99
    tifffile.imwrite(image_path, data)

    backend = TiffFileBackend(image_path)
    thumbnail = backend.get_thumbnail(max_size=512)

    assert max(thumbnail.shape[:2]) <= 512
    assert thumbnail.shape[2] == 3


def test_probe_large_image_adds_performance_hint_for_huge_tiff_without_vips(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "huge_rgb.tif"
    data = np.zeros((9000, 9000, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, data)

    monkeypatch.setattr(VipsBackend, "can_handle", lambda self, path: False)
    monkeypatch.setattr(OpenSlideBackend, "can_handle", lambda self, path: False)

    metadata = probe_large_image(image_path)

    assert metadata is not None
    assert metadata.recommended_backend == "tifffile"
    assert metadata.performance_hint is not None
    assert "libvips" in metadata.performance_hint.lower()


def test_open_large_image_falls_back_from_vips_to_tifffile(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "sample.ome.tiff"
    data = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    tifffile.imwrite(image_path, data, ome=True)

    monkeypatch.setattr(OpenSlideBackend, "can_handle", lambda self, path: False)
    monkeypatch.setattr(VipsBackend, "can_handle", lambda self, path: True)
    monkeypatch.setattr(
        VipsBackend,
        "open",
        lambda self, path: (_ for _ in ()).throw(RuntimeError("vips unavailable")),
    )

    backend = open_large_image(image_path)

    assert backend.name == "tifffile"


def test_large_image_cache_listing_and_cleanup(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "cache_root"
    first = cache_root / "a_111.pyramidal.tif"
    second = cache_root / "b_222.pyramidal.tif"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"x" * 16)
    second.write_bytes(b"y" * 32)

    monkeypatch.setattr(
        "annolid.io.large_image.cache.large_image_cache_root", lambda: cache_root
    )

    entries = list_large_image_cache_entries()

    assert len(entries) == 2
    assert large_image_cache_size_bytes() == 48
    assert {entry.path for entry in entries} == {first, second}
    assert format_large_image_cache_size(1536) == "1.5 KB"

    assert remove_large_image_cache_file(first) is True
    assert not first.exists()
    assert clear_all_large_image_caches() == 1
    assert not second.exists()


def test_optimized_large_image_cache_path_changes_with_source_mtime(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "flat.tif"
    source_path.write_bytes(b"abc")
    first = optimized_large_image_cache_path(source_path)
    source_path.write_bytes(b"abcdef")
    second = optimized_large_image_cache_path(source_path)

    assert first != second


def test_prune_large_image_caches_keeps_protected_newest_cache(
    tmp_path: Path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache_root"
    oldest = cache_root / "oldest.pyramidal.tif"
    middle = cache_root / "middle.pyramidal.tif"
    newest = cache_root / "newest.pyramidal.tif"
    cache_root.mkdir(parents=True, exist_ok=True)
    oldest.write_bytes(b"a" * 10)
    middle.write_bytes(b"b" * 20)
    newest.write_bytes(b"c" * 30)

    monkeypatch.setattr(
        "annolid.io.large_image.cache.large_image_cache_root", lambda: cache_root
    )

    os.utime(oldest, (10, 10))
    os.utime(middle, (20, 20))
    os.utime(newest, (30, 30))

    removed = prune_large_image_caches(max_entries=2, keep_paths=(newest,))

    assert removed == [oldest]
    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()


def test_large_tiff_sniff_and_probe_fallback_without_tifffile(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "sample.tif"
    data = np.zeros((64, 96, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, data)

    monkeypatch.setattr(TiffFileBackend, "can_handle", lambda self, path: False)
    monkeypatch.setattr(
        TiffFileBackend,
        "probe",
        lambda self, path=None: (_ for _ in ()).throw(ImportError("no tifffile")),
    )
    monkeypatch.setattr(VipsBackend, "can_handle", lambda self, path: False)
    monkeypatch.setattr(OpenSlideBackend, "can_handle", lambda self, path: False)

    sniffed = sniff_large_image(image_path)
    metadata = probe_large_image(image_path)
    loaded = load_image_with_backends(image_path)

    assert sniffed["is_tiff_family"] is True
    assert sniffed["recommended_backend"] == "qt"
    assert metadata is not None
    assert metadata.backend_name in {"qt", "pillow"}
    assert loaded.qimage.width() == 96
    assert loaded.qimage.height() == 64
