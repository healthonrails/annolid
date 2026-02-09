from __future__ import annotations

from pathlib import Path

import pytest

from annolid.tracker.cowtracker.cowtracker.dependencies import ensure_vggt_importable


def test_ensure_vggt_importable_points_to_vendored_tree() -> None:
    vendored_root = (
        Path(__file__).resolve().parents[1]
        / "annolid"
        / "tracker"
        / "cowtracker"
        / "cowtracker"
        / "thirdparty"
        / "vggt"
    )
    if not vendored_root.exists():
        pytest.skip("Vendored VGGT tree is not present in this checkout.")

    root = ensure_vggt_importable()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "vggt").exists()
