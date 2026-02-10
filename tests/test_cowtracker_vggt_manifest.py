from __future__ import annotations

import pytest

from annolid.tracker.cowtracker.cowtracker.vendor.vggt_runtime import (
    REQUIRED_RUNTIME_FILES,
    load_manifest_required_files,
    missing_runtime_files,
    vendored_manifest_path,
    vendored_vggt_root,
)


def test_vggt_manifest_exists() -> None:
    assert vendored_manifest_path().exists()


def test_vggt_manifest_matches_runtime_spec() -> None:
    manifest_files = load_manifest_required_files()
    assert manifest_files, "Manifest required_runtime_files list is empty."
    assert tuple(manifest_files) == tuple(REQUIRED_RUNTIME_FILES)


def test_vggt_runtime_files_exist_when_vendored_tree_present() -> None:
    root = vendored_vggt_root()
    if not root.exists():
        pytest.skip("Vendored VGGT tree is not present in this checkout.")

    missing = missing_runtime_files(root)
    assert not missing, f"Missing vendored runtime files: {', '.join(missing)}"
