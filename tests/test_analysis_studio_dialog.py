"""Headless Qt GUI tests for AnalysisStudioDialog and upgraded analytics widgets."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

from qtpy import QtWidgets

from annolid.gui.widgets.analysis_studio_dialog import AnalysisStudioDialog
from annolid.gui.widgets.file_audit_widget import FileAuditWidget
from annolid.gui.widgets.labeling_progress_dashboard import (
    LabelingProgressDashboardWidget,
)
from annolid.gui.widgets.quality_control_dialog import QualityControlDialog
from annolid.gui.widgets.tracking_stats_dashboard_dialog import (
    TrackingStatsDashboardDialog,
)

_APP = None


def _ensure_qapp() -> QtWidgets.QApplication:
    global _APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _APP = app
    return _APP


def _write_events(path: Path, rows: list[tuple[float, str, str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Recording time", "Subject", "Behavior", "Event"])
        for t, s, b, e in rows:
            w.writerow([f"{t:.2f}", s, b, e])
    return path


def test_analysis_studio_dialog_all_six_tabs_and_jump_signal(tmp_path: Path) -> None:
    _ensure_qapp()

    # Tracking CSV with a velocity jump + gap + freeze bout
    track_csv = tmp_path / "session_tracking.csv"
    lines = ["frame_number,instance_name,x1,y1,x2,y2,cx,cy,class_score"]
    for f in range(30):
        cx = 20.0 + f * 4.0 if f < 12 else (250.0 if f == 12 else 70.0)
        if f != 5:  # gap at frame 5
            lines.append(f"{f},mouse_1,{cx - 5},45,{cx + 5},55,{cx},50,0.95")
    track_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ev_a = _write_events(
        tmp_path / "session_events.csv",
        [
            (0.5, "mouse_1", "groom", "start"),
            (2.5, "mouse_1", "groom", "stop"),
            (3.0, "mouse_1", "rear", "start"),  # unpaired start -> QC issue!
        ],
    )
    ev_b = _write_events(
        tmp_path / "rater2_events.csv",
        [
            (0.5, "mouse_1", "groom", "start"),
            (1.5, "mouse_1", "groom", "stop"),
        ],
    )

    # Cohort manifest
    c1 = _write_events(
        tmp_path / "c1.csv",
        [(1.0, "m1", "attack", "start"), (3.0, "m1", "attack", "stop")],
    )
    c2 = _write_events(
        tmp_path / "c2.csv",
        [(1.0, "m1", "attack", "start"), (2.5, "m1", "attack", "stop")],
    )
    d1 = _write_events(
        tmp_path / "d1.csv",
        [(0.5, "m1", "attack", "start"), (9.5, "m1", "attack", "stop")],
    )
    d2 = _write_events(
        tmp_path / "d2.csv",
        [(0.5, "m1", "attack", "start"), (10.5, "m1", "attack", "stop")],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "recordings": [
                    {"recording_id": "c1", "events_csv": str(c1), "group": "ctrl"},
                    {"recording_id": "c2", "events_csv": str(c2), "group": "ctrl"},
                    {"recording_id": "d1", "events_csv": str(d1), "group": "drug"},
                    {"recording_id": "d2", "events_csv": str(d2), "group": "drug"},
                ]
            }
        ),
        encoding="utf-8",
    )

    dlg = AnalysisStudioDialog(initial_source=track_csv, initial_tab="track_qc")
    jumped_frames: list[int] = []
    dlg.jumpToFrameRequested.connect(jumped_frames.append)
    try:
        studio = dlg.studio
        studio.events_csv_edit.setText(str(ev_a))
        studio.fps_spin.setValue(20.0)
        studio.px_per_cm_spin.setValue(10.0)

        # Tab 1: Unified Report
        rep = studio.run_unified_report_sync()
        assert rep.html_path.is_file()
        assert studio.open_html_btn.isEnabled()

        # Tab 2: Track QC & Gaps + double-click jump
        studio.run_track_qc_sync()
        assert studio.tqc_instances_table.rowCount() == 1
        assert studio.tqc_gaps_table.rowCount() >= 1
        studio.tqc_gaps_table.cellDoubleClicked.emit(0, 0)
        assert jumped_frames[-1] == 5

        # Tab 3: Kinematics & Freezing + Homography JSON + double-click jump
        cal_file = tmp_path / "studio_homography.json"
        cal_file.write_text(
            json.dumps(
                {
                    "src_points_px": [
                        [0.0, 0.0],
                        [200.0, 0.0],
                        [200.0, 200.0],
                        [0.0, 200.0],
                    ],
                    "width_cm": 20.0,
                    "height_cm": 20.0,
                }
            ),
            encoding="utf-8",
        )
        studio.calibration_edit.setText(str(cal_file))
        kin_res = studio.run_kinematics_sync()
        assert (
            kin_res.calibration is not None and kin_res.calibration.mode == "homography"
        )
        assert studio.kin_instances_table.rowCount() == 1
        assert studio.kin_bouts_table.rowCount() >= 1
        studio.kin_bouts_table.cellDoubleClicked.emit(0, 0)
        assert len(jumped_frames) >= 2

        # Tab 4: Behavior & Ethogram QC + double-click jump
        studio.run_behavior_sync()
        assert studio.beh_summary_table.rowCount() >= 1
        assert studio.beh_qc_table.rowCount() >= 1
        studio.beh_qc_table.cellDoubleClicked.emit(0, 0)
        assert jumped_frames[-1] == 60  # 3.0s * 20 fps

        # Tab 5: IRR + double-click disagreement jump
        studio.irr_files_list.addItem(str(ev_a))
        studio.irr_files_list.addItem(str(ev_b))
        studio.run_irr_sync()
        assert studio.irr_rows_table.rowCount() >= 1
        assert studio.irr_disagreements_table.rowCount() >= 1

        # Tab 6: Cohort & Transitions
        studio.cohort_manifest_edit.setText(str(manifest))
        studio.run_cohort_sync()
        assert studio.cohort_cmp_table.rowCount() >= 1
        assert studio.cohort_summary_table.rowCount() >= 2
    finally:
        dlg.deleteLater()


def test_quality_control_file_audit_labeling_and_stats_dashboard_fixes(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    # 1. QualityControlDialog: not prematurely visible, synced default 30, lineEdit updates skip_num_frames
    qc_dlg = QualityControlDialog()
    try:
        assert not qc_dlg.isVisible()
        assert qc_dlg.skip_num_frames == 30
        assert qc_dlg.slider.value() == 30
        qc_dlg.framesLineEdit.setText("45")
        assert qc_dlg.skip_num_frames == 45
        assert qc_dlg.slider.value() == 45
    finally:
        qc_dlg.deleteLater()

    # 2. FileAuditWidget: read-only cells + CSV export
    audit = FileAuditWidget()
    try:
        assert audit.table.editTriggers() == QtWidgets.QAbstractItemView.NoEditTriggers
        jpath = tmp_path / "frame_001.json"
        jpath.write_text("{}", encoding="utf-8")
        audit.set_items(
            [
                {
                    "status": "valid",
                    "json_path": jpath,
                    "image_path": tmp_path / "frame_001.png",
                    "shape_count": 3,
                }
            ]
        )
        out_csv = audit.export_csv(tmp_path / "audit.csv")
        assert out_csv is not None and out_csv.is_file()
        assert "frame_001.json" in out_csv.read_text(encoding="utf-8")
    finally:
        audit.deleteLater()

    # 3. LabelingProgressDashboardWidget: tooltip on _BarsWidget + JSON export
    (tmp_path / "frame_001.png").write_bytes(b"png")
    (tmp_path / "frame_001.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[0, 0], [1, 1], [1, 0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dash = LabelingProgressDashboardWidget(initial_project_root=tmp_path)
    try:
        dash.apply_index_stats_snapshot(
            {
                "total_images": 1,
                "total_annotation_files": 1,
                "labeled_files": 1,
                "total_shapes": 1,
                "created_files": 1,
                "edited_files": 0,
                "coverage_percent": 100.0,
                "top_labels": [["mouse", 1]],
                "shape_type_counts": [["polygon", 1]],
                "annotator_file_counts": [["alice", 1]],
                "annotator_shape_counts": [["alice", 1]],
                "activity_last_7_days": [],
            }
        )
        assert "mouse: 1" in dash.label_bars.toolTip()
        exported = dash.export_stats_json(tmp_path / "labeling_stats.json")
        assert exported is not None and exported.is_file()
    finally:
        dash.deleteLater()

    # 4. TrackingStatsDashboardDialog: double-click emits jumpToVideoFrameRequested
    stats_file = tmp_path / "vidA_tracking_stats.json"
    stats_file.write_text(
        json.dumps(
            {
                "video_name": "vidA.mp4",
                "summary": {"manual_frames": 2},
                "bad_shape_events": [
                    {
                        "frame": 42,
                        "label": "m1",
                        "reason": "area_spike",
                        "resolved": False,
                    }
                ],
                "prediction_segments": [
                    {"start_frame": 15, "end_frame": 20, "status": "abnormal"}
                ],
            }
        ),
        encoding="utf-8",
    )
    ts_dlg = TrackingStatsDashboardDialog(initial_root_dir=tmp_path)
    jumps: list[tuple[str, int]] = []
    ts_dlg.jumpToVideoFrameRequested.connect(lambda ref, f: jumps.append((ref, f)))
    try:
        ts_dlg.dashboard.run_analysis()
        assert ts_dlg.dashboard.bad_shape_table.rowCount() == 1
        ts_dlg.dashboard.bad_shape_table.cellDoubleClicked.emit(0, 0)
        assert jumps[-1][1] == 42
    finally:
        ts_dlg.deleteLater()
