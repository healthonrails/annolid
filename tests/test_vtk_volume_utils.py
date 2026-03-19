from __future__ import annotations

from pathlib import Path

import numpy as np

from annolid.gui.widgets.vtk_volume_utils import (
    first_file_with_suffix,
    normalize_to_float01,
    path_matches_ext,
)


def test_path_matches_ext_handles_compound_suffix() -> None:
    assert path_matches_ext(Path("sample.nii.gz"), (".nii", ".nii.gz"))
    assert not path_matches_ext(Path("sample.nii"), (".nii.gz",))


def test_first_file_with_suffix_picks_matching_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    target = tmp_path / "b.nii.gz"
    target.write_text("x", encoding="utf-8")
    found = first_file_with_suffix(tmp_path, (".nii.gz",))
    assert found == target


def test_normalize_to_float01_handles_non_finite_values() -> None:
    arr = np.array([np.nan, np.inf, -10.0, 0.0, 100.0], dtype=np.float32)
    norm = normalize_to_float01(arr)
    assert np.isfinite(norm).all()
    assert float(norm.min()) >= 0.0
    assert float(norm.max()) <= 1.0


def test_normalize_to_float01_sanitizes_percentile_bounds() -> None:
    arr = np.array([-100.0, 0.0, 10.0, 200.0], dtype=np.float32)
    norm = normalize_to_float01(arr, lower_percentile=120.0, upper_percentile=-20.0)
    assert np.isfinite(norm).all()
    assert float(norm.min()) >= 0.0
    assert float(norm.max()) <= 1.0
    assert np.ptp(norm) > 0.0


def test_normalize_to_float01_constant_volume_returns_zeros() -> None:
    arr = np.full((3, 4, 5), 7.0, dtype=np.float32)
    norm = normalize_to_float01(arr)
    assert norm.shape == arr.shape
    assert norm.dtype == np.float32
    assert np.all(norm == 0.0)
