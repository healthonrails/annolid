from __future__ import annotations

from qtpy import QtWidgets

from annolid.gui.widgets.realtime_control_widget import RealtimeControlWidget


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_realtime_control_widget_viewer_only_allows_empty_model_path() -> None:
    _ensure_qapp()
    widget = RealtimeControlWidget(config={"realtime": {"model_weight": ""}})
    try:
        widget.model_path_edit.clear()
        widget.viewer_only_check.setChecked(True)

        cfg, extras = widget._build_runtime_config()

        assert cfg.viewer_only is True
        assert cfg.publish_frames is True
        assert cfg.publish_annotated_frames is False
        assert cfg.save_detection_segments is False
        assert extras["viewer_only"] is True
        assert widget.model_combo.isEnabled() is False
        assert widget.model_path_edit.isEnabled() is False
        assert widget.publish_frames_check.isChecked() is True
        assert widget.publish_annotated_check.isEnabled() is False
    finally:
        widget.deleteLater()
