from __future__ import annotations

from pathlib import Path

import numpy as np

from annolid.gui.widgets.vtk_volume_loader import VolumeSourceLoader


class _ReaderStub:
    def __init__(self, calls):
        self._calls = calls

    def make_simple_volume_data(self, volume, spacing):
        return {
            "array": volume,
            "spacing": spacing,
            "is_out_of_core": False,
        }

    def normalize_to_float01(self, arr):
        return arr

    def vtk_image_to_numpy(self, vtk_img):
        return np.ones((2, 2, 2), dtype=np.float32)

    def is_zarr_candidate(self, path: Path) -> bool:
        self._calls["zarr"].append(("candidate", path))
        return path.name.lower().endswith(".zarr")

    def read_zarr(self, path: Path):
        self._calls["zarr"].append(("read", path))
        return {"kind": "zarr"}

    def read_dicom_series(self, path: Path):
        self._calls["dicom"].append(("read", path))
        vol = np.ones((2, 2, 2), dtype=np.float32)
        return vol, (1.0, 1.0, 1.0)

    def is_tiff_candidate(self, path: Path) -> bool:
        self._calls["tiff_eager"].append(("candidate", path))
        return path.suffix.lower() in {".tif", ".tiff"}

    def should_use_out_of_core_tiff(self, path: Path) -> bool:
        self._calls["tiff_ooc"].append(("strategy", path))
        return False

    def read_tiff_eager(self, path: Path):
        self._calls["tiff_eager"].append(("read", path))
        return {"kind": "tiff_eager"}

    def read_tiff_out_of_core(self, path: Path):
        self._calls["tiff_ooc"].append(("read", path))
        return {"kind": "tiff_ooc"}

    def read_analyze_volume(self, path: Path):
        self._calls["analyze"].append(("read", path))
        return {"kind": "analyze"}


def _make_loader():
    calls = {
        "zarr": [],
        "dicom": [],
        "analyze": [],
        "tiff_eager": [],
        "tiff_ooc": [],
        "warn": [],
    }
    return VolumeSourceLoader(readers=_ReaderStub(calls)), calls


def test_read_volume_any_prefers_zarr_for_directory(tmp_path: Path):
    zarr_dir = tmp_path / "sample.zarr"
    zarr_dir.mkdir()
    loader, calls = _make_loader()
    out = loader.read_volume_any(zarr_dir)
    assert out["kind"] == "zarr"
    assert calls["zarr"] == [("candidate", zarr_dir), ("read", zarr_dir)]


def test_read_volume_any_uses_dicom_parent_for_dcm_file(tmp_path: Path):
    dcm = tmp_path / "a.dcm"
    dcm.write_text("x", encoding="utf-8")
    loader, calls = _make_loader()
    out = loader.read_volume_any(dcm)
    assert out["is_out_of_core"] is False
    assert calls["dicom"] == [("read", tmp_path)]


def test_read_volume_any_falls_back_to_out_of_core_tiff(tmp_path: Path):
    tiff = tmp_path / "img.tif"
    tiff.write_text("x", encoding="utf-8")

    class _FailingReader(_ReaderStub):
        def read_tiff_eager(self, path: Path):
            self._calls["tiff_eager"].append(("read", path))
            raise MemoryError("oom")

    calls = {
        "zarr": [],
        "dicom": [],
        "analyze": [],
        "tiff_eager": [],
        "tiff_ooc": [],
        "warn": [],
    }
    loader = VolumeSourceLoader(readers=_FailingReader(calls))
    out = loader.read_volume_any(tiff)
    assert out["kind"] == "tiff_ooc"
    assert len(calls["tiff_eager"]) == 2
    assert len(calls["tiff_ooc"]) == 2


def test_read_volume_any_unsupported_raises_context(tmp_path: Path):
    unsupported = tmp_path / "x.foo"
    unsupported.write_text("x", encoding="utf-8")
    loader, _ = _make_loader()
    try:
        loader.read_volume_any(unsupported)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc)
        assert "Failed to read volume from" in msg
        assert "Unsupported volume format" in msg
