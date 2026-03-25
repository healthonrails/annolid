from __future__ import annotations

import json
import os
from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.widgets.tracking_stats_dashboard_dialog import (
    TrackingStatsDashboardWidget,
)


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_tracking_stats_dashboard_widget_runs_analysis(tmp_path: Path) -> None:
    _ensure_qapp()

    stats_path = tmp_path / "video_a" / "video_a_tracking_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "version": 4,
                "video_name": "video_a.mp4",
                "summary": {
                    "manual_frames": 4,
                    "manual_segments": [[0, 3]],
                    "bad_shape_frames": 1,
                    "bad_shape_failed_frames": 1,
                    "abnormal_segment_events": 1,
                },
                "prediction_segments": [
                    {"start_frame": 10, "end_frame": 11, "status": "halted"}
                ],
                "bad_shape_events": [
                    {
                        "frame": 11,
                        "label": "mouse",
                        "reason": "polygon_conversion_failed",
                        "resolved": False,
                        "repair_source": "",
                        "timestamp": "2026-03-25T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    widget = TrackingStatsDashboardWidget(initial_root_dir=tmp_path)
    try:
        output_dir = tmp_path / "out"
        widget.root_dir_edit.setText(str(tmp_path))
        widget.output_dir_edit.setText(str(output_dir))
        widget.run_analysis()

        assert widget.open_output_btn.isEnabled() is True
        assert widget.overview_table.rowCount() == 1
        assert widget.abnormal_table.rowCount() == 1
        assert widget.bad_shape_table.rowCount() == 1
        assert "generated" in widget.status_label.text().lower()
    finally:
        widget.close()
