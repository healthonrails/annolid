from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.widgets.polygon_classifier_workbench import (
    PolygonClassifierWorkbench,
)


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_polygon_classifier_workbench_assigns_train_and_test_videos(tmp_path: Path):
    _ensure_qapp()
    train_dir = tmp_path / "train_video"
    test_dir = tmp_path / "test_video"
    train_dir.mkdir()
    test_dir.mkdir()
    train_labels = tmp_path / "train_labels.csv"
    test_labels = tmp_path / "test_labels.csv"
    train_labels.write_text("frame,background\n0,1\n", encoding="utf-8")
    test_labels.write_text("frame,background\n0,1\n", encoding="utf-8")

    dialog = PolygonClassifierWorkbench()
    try:
        assert hasattr(dialog, "train_annotation_dir")
        assert hasattr(dialog, "test_annotation_dir")
        assert hasattr(dialog, "generate_train_test_btn")
        assert not dialog.generate_train_test_btn.isEnabled()

        dialog.train_annotation_dir.setPath(train_dir)
        dialog.train_label_csv.setPath(train_labels)
        dialog.test_annotation_dir.setPath(test_dir)
        dialog.test_label_csv.setPath(test_labels)
        dialog.split_output_dir.setPath(tmp_path / "out")
        dialog._refresh_validation()

        assert dialog.generate_train_test_btn.isEnabled()
    finally:
        dialog.close()
