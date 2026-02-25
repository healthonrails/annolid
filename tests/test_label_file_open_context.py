from __future__ import annotations

from pathlib import Path

from annolid.gui.label_file import open as label_file_open


def test_label_file_open_closes_handle(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    handle = None

    with label_file_open(path, "w") as fh:
        handle = fh
        fh.write("{}")

    assert handle is not None
    assert handle.closed
