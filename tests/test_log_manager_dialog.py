from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.widgets.log_manager_dialog import LogManagerDialog


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_log_manager_dialog_resolves_entries(monkeypatch, tmp_path: Path) -> None:
    _ensure_qapp()
    logs_root = tmp_path / "logs"
    realtime_root = logs_root / "realtime"
    runs_root = logs_root / "runs"
    label_index_root = logs_root / "label_index"
    app_root = logs_root / "app"

    monkeypatch.setattr(
        "annolid.gui.widgets.log_manager_dialog.resolve_annolid_logs_root",
        lambda *args, **kwargs: logs_root,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.log_manager_dialog.resolve_annolid_realtime_logs_root",
        lambda: realtime_root,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.log_manager_dialog.shared_runs_root",
        lambda *args, **kwargs: runs_root,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.log_manager_dialog.default_label_index_path",
        lambda *args, **kwargs: label_index_root / "annolid_dataset.jsonl",
    )

    dialog = LogManagerDialog()
    try:
        rows = dialog._resolve_entries()
        assert rows
        keys = {str(item.get("key")) for item in rows}
        assert {"logs_root", "realtime", "label_index", "runs", "app"} <= keys
        by_key = {str(item.get("key")): item for item in rows}
        assert Path(str(by_key["logs_root"]["path"])) == logs_root
        assert Path(str(by_key["realtime"]["path"])) == realtime_root
        assert Path(str(by_key["runs"]["path"])) == runs_root
        assert Path(str(by_key["label_index"]["path"])) == label_index_root
        assert Path(str(by_key["app"]["path"])) == app_root
    finally:
        dialog.close()
