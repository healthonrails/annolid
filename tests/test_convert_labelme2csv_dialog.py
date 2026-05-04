from __future__ import annotations

import os

from qtpy import QtWidgets

from annolid.gui.widgets.convert_labelme2csv_dialog import LabelmeJsonToCsvDialog


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


def _ensure_qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_dialog_runs_conversion_with_both_outputs(monkeypatch) -> None:
    _ensure_qapp()
    dialog = LabelmeJsonToCsvDialog()
    dialog.json_folder_path = "/tmp/session"

    captured = {}

    def _fake_convert(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "/tmp/session_tracking.csv"

    monkeypatch.setattr(
        "annolid.gui.widgets.convert_labelme2csv_dialog.convert_json_to_csv",
        _fake_convert,
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)

    dialog.generate_tracking_checkbox.setChecked(True)
    dialog.generate_tracked_checkbox.setChecked(True)
    dialog.run_conversion()

    assert captured["args"] == ("/tmp/session",)
    assert captured["kwargs"]["csv_file"] == "/tmp/session_tracking.csv"
    assert captured["kwargs"]["tracked_csv_file"] == "/tmp/session_tracked.csv"
    assert captured["kwargs"]["include_tracking_output"] is True


def test_dialog_allows_tracked_only_generation(monkeypatch) -> None:
    _ensure_qapp()
    dialog = LabelmeJsonToCsvDialog()
    dialog.json_folder_path = "/tmp/session"

    captured = {}

    def _fake_convert(*args, **kwargs):
        captured["kwargs"] = kwargs
        return "/tmp/session_tracking.csv"

    monkeypatch.setattr(
        "annolid.gui.widgets.convert_labelme2csv_dialog.convert_json_to_csv",
        _fake_convert,
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)

    dialog.generate_tracking_checkbox.setChecked(False)
    dialog.generate_tracked_checkbox.setChecked(True)
    dialog.run_conversion()

    assert captured["kwargs"]["tracked_csv_file"] == "/tmp/session_tracked.csv"
    assert captured["kwargs"]["include_tracking_output"] is False
