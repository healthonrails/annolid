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
        assert hasattr(dialog, "train_video_table")
        assert hasattr(dialog, "test_video_table")
        assert hasattr(dialog, "generate_train_test_btn")
        assert not dialog.generate_train_test_btn.isEnabled()

        dialog.train_video_table.add_assignment(train_dir, train_labels)
        dialog.test_video_table.add_assignment(test_dir, test_labels)
        dialog.split_output_dir.setPath(tmp_path / "out")
        dialog._refresh_validation()

        assert dialog.generate_train_test_btn.isEnabled()
    finally:
        dialog.close()


def test_polygon_classifier_workbench_applies_model_defaults():
    _ensure_qapp()
    dialog = PolygonClassifierWorkbench()
    try:
        assert dialog.model_type.currentData() == "tcn"
        assert dialog.epochs.value() == 500
        assert dialog.learning_rate.value() == 0.0001
        assert dialog.batch_size.value() == 8
        assert dialog.window_size.value() == 1000
        assert dialog.hidden_dim.value() == 32

        dialog.model_type.setCurrentIndex(0)
        assert dialog.model_type.currentData() == "convnet"
        assert dialog.epochs.value() == 30
        assert dialog.learning_rate.value() == 0.004
    finally:
        dialog.close()
