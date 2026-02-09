from __future__ import annotations

from pathlib import Path

from annolid.tracker.cowtracker.cowtracker.dependencies import ensure_vggt_importable


def test_ensure_vggt_importable_points_to_vendored_tree() -> None:
    root = ensure_vggt_importable()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "vggt").exists()
