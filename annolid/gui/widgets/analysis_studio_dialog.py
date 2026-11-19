"""Unified Scientific Analysis, Statistics, QC & Reporting Studio for Annolid."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from annolid.behavior.cohort import write_cohort_comparisons_csv
from annolid.behavior.irr import write_irr_disagreements_csv
from annolid.behavior.time_budget import (
    _read_event_rows,
    compute_binned_time_budget,
    extract_behavior_intervals,
    summarize_intervals,
)
from annolid.postprocessing.kinematics_stats import write_kinematics_events_csv
from annolid.postprocessing.tracking_reports import summarize_tracking_gaps
from annolid.services.behavior_analytics import (
    run_behavior_cohort,
    run_behavior_irr,
    run_behavior_qc,
)
from annolid.services.kinematics_analytics import run_kinematics_analysis
from annolid.services.track_analytics import run_track_qc
from annolid.services.unified_reporting import (
    UnifiedReportArtifacts,
    build_unified_analysis_report,
)


class _CallableWorker(QtCore.QObject):
    """Generic background QObject worker executing a callable off the UI thread."""

    finished = QtCore.Signal(str, object, str)

    def __init__(self, task_name: str, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._task_name = str(task_name)
        self._fn = fn

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._fn()
            self.finished.emit(self._task_name, result, "")
        except Exception as exc:
            self.finished.emit(self._task_name, None, str(exc))


def _configure_table(
    table: QtWidgets.QTableWidget, headers: Sequence[str]
) -> QtWidgets.QTableWidget:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels([str(h) for h in headers])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    table.setAlternatingRowColors(True)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill_table_rows(
    table: QtWidgets.QTableWidget,
    rows: Sequence[Sequence[Any]],
    *,
    frame_targets: Optional[Sequence[Optional[int]]] = None,
) -> None:
    was_sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for r_idx, row in enumerate(rows):
        target_frame = (
            frame_targets[r_idx]
            if frame_targets is not None and r_idx < len(frame_targets)
            else None
        )
        for c_idx, cell in enumerate(row):
            txt = "—" if cell is None else str(cell)
            item = QtWidgets.QTableWidgetItem(txt)
            if target_frame is not None and c_idx == 0:
                item.setData(QtCore.Qt.UserRole, int(target_frame))
            table.setItem(r_idx, c_idx, item)
    table.resizeColumnsToContents()
    table.setSortingEnabled(was_sorting)


def _fmt_num(val: Any, digits: int = 3, suffix: str = "") -> str:
    if val is None or val == "":
        return "—"
    try:
        return f"{float(val):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


class AnalysisStudioWidget(QtWidgets.QWidget):
    """Non-blocking 6-tab workbench for QC, Kinematics, Ethograms, IRR, Cohorts, and Reports."""

    jumpToFrameRequested = QtCore.Signal(int)

    _TAB_INDEX_BY_KEY = {
        "report": 0,
        "track_qc": 1,
        "kinematics": 2,
        "behavior": 3,
        "irr": 4,
        "cohort": 5,
    }

    def __init__(
        self,
        *,
        owner_window: Optional[QtWidgets.QWidget] = None,
        initial_source: Optional[str | Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._owner_window = owner_window
        self._active_thread: Optional[QtCore.QThread] = None
        self._active_worker: Optional[_CallableWorker] = None
        self._last_report_artifacts: Optional[UnifiedReportArtifacts] = None
        self._last_kinematics_report: Any = None
        self._last_irr_report: Any = None
        self._last_cohort_report: Any = None

        self._setup_ui()
        self.apply_owner_context(override_source=initial_source)

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            """
            QFrame[kpiCard="true"] {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 8px;
            }
            QLabel[kpiTitle="true"] {
                color: palette(mid);
                font-size: 11px;
                font-weight: 600;
            }
            QLabel[kpiValue="true"] {
                color: palette(text);
                font-size: 17px;
                font-weight: 700;
            }
            """
        )
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Shared Context Header
        ctx_group = QtWidgets.QGroupBox("Session & Calibration Context", self)
        ctx_grid = QtWidgets.QGridLayout(ctx_group)
        ctx_grid.setHorizontalSpacing(8)
        ctx_grid.setVerticalSpacing(6)

        self.source_edit = QtWidgets.QLineEdit(self)
        self.source_edit.setPlaceholderText(
            "Video (.mp4), tracking CSV (*_tracking.csv), frame JSON folder, or workspace directory"
        )
        self.source_browse_btn = QtWidgets.QPushButton("Browse File…", self)
        self.source_browse_dir_btn = QtWidgets.QPushButton("Browse Folder…", self)
        self.use_open_session_btn = QtWidgets.QPushButton("Use Open Session", self)
        self.source_browse_btn.clicked.connect(self._browse_source_file)
        self.source_browse_dir_btn.clicked.connect(self._browse_source_dir)
        self.use_open_session_btn.clicked.connect(lambda: self.apply_owner_context())

        self.events_csv_edit = QtWidgets.QLineEdit(self)
        self.events_csv_edit.setPlaceholderText(
            "Optional Behavior Event CSV (leave empty to auto-discover or use active Behavior Log)"
        )
        self.events_browse_btn = QtWidgets.QPushButton("Events CSV…", self)
        self.events_browse_btn.clicked.connect(self._browse_events_csv)

        self.zone_file_edit = QtWidgets.QLineEdit(self)
        self.zone_file_edit.setPlaceholderText("Optional *_zones.json file")
        self.zone_browse_btn = QtWidgets.QPushButton("Zones JSON…", self)
        self.zone_browse_btn.clicked.connect(self._browse_zone_file)

        self.fps_spin = QtWidgets.QDoubleSpinBox(self)
        self.fps_spin.setRange(0.0, 2000.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSpecialValueText("Auto FPS")
        self.fps_spin.setValue(0.0)

        self.px_per_cm_spin = QtWidgets.QDoubleSpinBox(self)
        self.px_per_cm_spin.setRange(0.0, 10000.0)
        self.px_per_cm_spin.setDecimals(3)
        self.px_per_cm_spin.setSpecialValueText("Uncalibrated (px)")
        self.px_per_cm_spin.setValue(0.0)

        ctx_grid.addWidget(QtWidgets.QLabel("Source:"), 0, 0)
        ctx_grid.addWidget(self.source_edit, 0, 1, 1, 3)
        ctx_grid.addWidget(self.source_browse_btn, 0, 4)
        ctx_grid.addWidget(self.source_browse_dir_btn, 0, 5)
        ctx_grid.addWidget(self.use_open_session_btn, 0, 6)

        ctx_grid.addWidget(QtWidgets.QLabel("Events CSV:"), 1, 0)
        ctx_grid.addWidget(self.events_csv_edit, 1, 1)
        ctx_grid.addWidget(self.events_browse_btn, 1, 2)
        ctx_grid.addWidget(QtWidgets.QLabel("Zone JSON:"), 1, 3)
        ctx_grid.addWidget(self.zone_file_edit, 1, 4, 1, 2)
        ctx_grid.addWidget(self.zone_browse_btn, 1, 6)

        calib_row = QtWidgets.QHBoxLayout()
        calib_row.addWidget(QtWidgets.QLabel("FPS:"))
        calib_row.addWidget(self.fps_spin)
        calib_row.addSpacing(12)
        calib_row.addWidget(QtWidgets.QLabel("Scale (px/cm):"))
        calib_row.addWidget(self.px_per_cm_spin)
        calib_row.addSpacing(12)
        calib_row.addWidget(QtWidgets.QLabel("Homography JSON:"))
        self.calibration_edit = QtWidgets.QLineEdit(self)
        self.calibration_edit.setPlaceholderText(
            "Optional 4-corner / perspective homography calibration JSON"
        )
        self.calibration_browse_btn = QtWidgets.QPushButton("Calibration…", self)
        self.calibration_browse_btn.clicked.connect(self._browse_calibration_file)
        calib_row.addWidget(self.calibration_edit, 1)
        calib_row.addWidget(self.calibration_browse_btn)
        ctx_grid.addLayout(calib_row, 2, 1, 1, 6)
        root.addWidget(ctx_group)

        # Main 6-Tab Workbench
        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.addTab(self._build_report_tab(), "1. Unified Report")
        self.tabs.addTab(self._build_track_qc_tab(), "2. Track QC & Gaps")
        self.tabs.addTab(self._build_kinematics_tab(), "3. Kinematics & Freezing")
        self.tabs.addTab(self._build_behavior_tab(), "4. Behavior & Ethogram QC")
        self.tabs.addTab(self._build_irr_tab(), "5. Inter-Rater Reliability")
        self.tabs.addTab(self._build_cohort_tab(), "6. Cohort & Transitions")
        root.addWidget(self.tabs, 1)

        # Status & Busy Footer
        footer = QtWidgets.QHBoxLayout()
        self.busy_bar = QtWidgets.QProgressBar(self)
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedWidth(110)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setVisible(False)
        self.status_label = QtWidgets.QLabel(
            "Ready. Double-click any review bout, gap, freezing bout, QC alert, or IRR disagreement row to jump the main canvas to that frame.",
            self,
        )
        self.status_label.setWordWrap(True)
        footer.addWidget(self.busy_bar)
        footer.addWidget(self.status_label, 1)
        root.addLayout(footer)

    def select_tab(self, tab_key: str) -> None:
        idx = self._TAB_INDEX_BY_KEY.get(str(tab_key or "").strip().lower())
        if idx is not None:
            self.tabs.setCurrentIndex(idx)

    def _make_kpi_card(self, title: str, initial: str = "—") -> QtWidgets.QLabel:
        frame = QtWidgets.QFrame(self)
        frame.setProperty("kpiCard", True)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        t_lbl = QtWidgets.QLabel(title, frame)
        t_lbl.setProperty("kpiTitle", True)
        v_lbl = QtWidgets.QLabel(initial, frame)
        v_lbl.setProperty("kpiValue", True)
        lay.addWidget(t_lbl)
        lay.addWidget(v_lbl)
        v_lbl._card_frame = frame  # type: ignore[attr-defined]
        return v_lbl

    # ------------------------------------------------------------------
    # Tab Builders
    # ------------------------------------------------------------------
    def _build_report_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(page)

        top_bar = QtWidgets.QHBoxLayout()
        self.report_title_edit = QtWidgets.QLineEdit(page)
        self.report_title_edit.setPlaceholderText("Optional custom report title")
        self.run_report_btn = QtWidgets.QPushButton("Generate Unified Report", page)
        self.open_html_btn = QtWidgets.QPushButton("Open HTML Report", page)
        self.open_md_btn = QtWidgets.QPushButton("Open Markdown", page)
        self.open_report_dir_btn = QtWidgets.QPushButton("Open Folder", page)
        for btn in (self.open_html_btn, self.open_md_btn, self.open_report_dir_btn):
            btn.setEnabled(False)

        self.run_report_btn.clicked.connect(
            lambda: self._dispatch_task("report", self._compute_report_job)
        )
        self.open_html_btn.clicked.connect(self._open_report_html)
        self.open_md_btn.clicked.connect(self._open_report_md)
        self.open_report_dir_btn.clicked.connect(self._open_report_folder)

        top_bar.addWidget(QtWidgets.QLabel("Report Title:"))
        top_bar.addWidget(self.report_title_edit, 1)
        top_bar.addWidget(self.run_report_btn)
        top_bar.addWidget(self.open_html_btn)
        top_bar.addWidget(self.open_md_btn)
        top_bar.addWidget(self.open_report_dir_btn)
        lay.addLayout(top_bar)

        kpi_row = QtWidgets.QHBoxLayout()
        self.rep_status_val = self._make_kpi_card("Quality Gate")
        self.rep_frames_val = self._make_kpi_card("Total Frames")
        self.rep_fps_val = self._make_kpi_card("FPS")
        self.rep_instances_val = self._make_kpi_card("Instances")
        self.rep_sections_val = self._make_kpi_card("Active Sections")
        for v in (
            self.rep_status_val,
            self.rep_frames_val,
            self.rep_fps_val,
            self.rep_instances_val,
            self.rep_sections_val,
        ):
            kpi_row.addWidget(v._card_frame)  # type: ignore[attr-defined]
        lay.addLayout(kpi_row)

        self.report_preview = QtWidgets.QTextBrowser(page)
        self.report_preview.setOpenExternalLinks(True)
        lay.addWidget(self.report_preview, 1)
        return page

    def _build_track_qc_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(page)

        ctrl = QtWidgets.QHBoxLayout()
        self.expected_instances_spin = QtWidgets.QSpinBox(page)
        self.expected_instances_spin.setRange(0, 100)
        self.expected_instances_spin.setSpecialValueText("Auto")
        self.run_track_qc_btn = QtWidgets.QPushButton("Run Track QC & Gap Audit", page)
        self.run_track_qc_btn.clicked.connect(
            lambda: self._dispatch_task("track_qc", self._compute_track_qc_job)
        )
        ctrl.addWidget(QtWidgets.QLabel("Expected Instances:"))
        ctrl.addWidget(self.expected_instances_spin)
        ctrl.addWidget(self.run_track_qc_btn)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, page)
        self.tqc_instances_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Instance",
                "Observed Frames",
                "Coverage %",
                "Gaps",
                "Longest Gap (f)",
                "Median Speed (px/f)",
                "P95 Speed (px/f)",
            ],
        )
        self.tqc_instances_table.setSortingEnabled(True)

        self.tqc_review_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Rank",
                "Start Frame",
                "End Frame",
                "Priority Score",
                "Instances",
                "Issue Kinds",
                "Summary",
            ],
        )
        self.tqc_review_table.setSortingEnabled(True)
        self.tqc_review_table.cellDoubleClicked.connect(
            lambda r, _c: self._emit_row_frame(self.tqc_review_table, r, fallback_col=1)
        )

        self.tqc_gaps_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            ["Instance", "Start Frame", "End Frame", "Duration (f)", "Duration (s)"],
        )
        self.tqc_gaps_table.setSortingEnabled(True)
        self.tqc_gaps_table.cellDoubleClicked.connect(
            lambda r, _c: self._emit_row_frame(self.tqc_gaps_table, r, fallback_col=1)
        )

        splitter.addWidget(self.tqc_instances_table)
        splitter.addWidget(self.tqc_review_table)
        splitter.addWidget(self.tqc_gaps_table)
        lay.addWidget(splitter, 1)
        return page

    def _build_kinematics_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(page)

        ctrl = QtWidgets.QHBoxLayout()
        self.freeze_thresh_spin = QtWidgets.QDoubleSpinBox(page)
        self.freeze_thresh_spin.setRange(0.0, 5000.0)
        self.freeze_thresh_spin.setDecimals(2)
        self.freeze_thresh_spin.setSpecialValueText("Adaptive")
        self.min_freeze_dur_spin = QtWidgets.QDoubleSpinBox(page)
        self.min_freeze_dur_spin.setRange(0.05, 60.0)
        self.min_freeze_dur_spin.setDecimals(2)
        self.min_freeze_dur_spin.setValue(0.50)

        self.run_kinematics_btn = QtWidgets.QPushButton(
            "Compute Kinematics & Freezing", page
        )
        self.export_kin_events_btn = QtWidgets.QPushButton(
            "Export Bouts to Events CSV…", page
        )
        self.export_kin_events_btn.setEnabled(False)
        self.run_kinematics_btn.clicked.connect(
            lambda: self._dispatch_task("kinematics", self._compute_kinematics_job)
        )
        self.export_kin_events_btn.clicked.connect(self._export_kinematics_events_csv)

        ctrl.addWidget(QtWidgets.QLabel("Freeze Speed (px/s):"))
        ctrl.addWidget(self.freeze_thresh_spin)
        ctrl.addWidget(QtWidgets.QLabel("Min Freeze (s):"))
        ctrl.addWidget(self.min_freeze_dur_spin)
        ctrl.addWidget(self.run_kinematics_btn)
        ctrl.addWidget(self.export_kin_events_btn)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, page)
        self.kin_instances_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Instance",
                "Distance (px)",
                "Distance (cm)",
                "Mean Speed (px/s)",
                "Mean Speed (cm/s)",
                "P95 Speed (px/s)",
                "Freezing %",
                "Freeze Bouts",
                "Straightness",
                "Tortuosity",
            ],
        )
        self.kin_instances_table.setSortingEnabled(True)

        self.kin_bouts_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Instance",
                "Kind",
                "Start Frame",
                "End Frame",
                "Start (s)",
                "Duration (s)",
                "Mean Speed (px/s)",
            ],
        )
        self.kin_bouts_table.setSortingEnabled(True)
        self.kin_bouts_table.cellDoubleClicked.connect(
            lambda r, _c: self._emit_row_frame(self.kin_bouts_table, r, fallback_col=2)
        )

        self.kin_pairwise_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Pair",
                "Shared Frames",
                "Mean Dist (px)",
                "Min Dist (px)",
                "Mean Dist (cm)",
                "Close Contact %",
                "Velocity Corr (r)",
            ],
        )
        self.kin_pairwise_table.setSortingEnabled(True)

        splitter.addWidget(self.kin_instances_table)
        splitter.addWidget(self.kin_bouts_table)
        splitter.addWidget(self.kin_pairwise_table)
        lay.addWidget(splitter, 1)
        return page

    def _build_behavior_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(page)

        ctrl = QtWidgets.QHBoxLayout()
        self.beh_session_dur_spin = QtWidgets.QDoubleSpinBox(page)
        self.beh_session_dur_spin.setRange(0.0, 864000.0)
        self.beh_session_dur_spin.setDecimals(2)
        self.beh_session_dur_spin.setSpecialValueText("Union Span")
        self.beh_bin_spin = QtWidgets.QDoubleSpinBox(page)
        self.beh_bin_spin.setRange(0.0, 36000.0)
        self.beh_bin_spin.setDecimals(1)
        self.beh_bin_spin.setSpecialValueText("No Bins")
        self.beh_bin_spin.setValue(60.0)

        self.run_behavior_btn = QtWidgets.QPushButton(
            "Run Time Budget & Ethogram QC", page
        )
        self.run_behavior_btn.clicked.connect(
            lambda: self._dispatch_task("behavior", self._compute_behavior_job)
        )
        ctrl.addWidget(QtWidgets.QLabel("Session Duration (s):"))
        ctrl.addWidget(self.beh_session_dur_spin)
        ctrl.addWidget(QtWidgets.QLabel("Bin Width (s):"))
        ctrl.addWidget(self.beh_bin_spin)
        ctrl.addWidget(self.run_behavior_btn)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, page)
        self.beh_summary_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Subject",
                "Behavior",
                "Occurrences",
                "Total (s)",
                "Mean Bout (s)",
                "Median Bout (s)",
                "% of Session",
            ],
        )
        self.beh_summary_table.setSortingEnabled(True)

        self.beh_binned_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Bin #",
                "Window (s)",
                "Subject",
                "Behavior",
                "Occurrences",
                "Duration (s)",
                "% of Bin",
            ],
        )
        self.beh_binned_table.setSortingEnabled(True)

        self.beh_qc_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Approx Frame",
                "Severity",
                "Kind",
                "Subject",
                "Behavior",
                "Time (s)",
                "Message",
            ],
        )
        self.beh_qc_table.setSortingEnabled(True)
        self.beh_qc_table.cellDoubleClicked.connect(
            lambda r, _c: self._emit_row_frame(self.beh_qc_table, r, fallback_col=0)
        )

        splitter.addWidget(self.beh_summary_table)
        splitter.addWidget(self.beh_binned_table)
        splitter.addWidget(self.beh_qc_table)
        lay.addWidget(splitter, 1)
        return page

    def _build_irr_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(page)

        files_row = QtWidgets.QHBoxLayout()
        self.irr_files_list = QtWidgets.QListWidget(page)
        self.irr_files_list.setMaximumHeight(82)
        btn_col = QtWidgets.QVBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Rater CSVs…", page)
        clear_btn = QtWidgets.QPushButton("Clear", page)
        add_btn.clicked.connect(self._add_irr_csvs)
        clear_btn.clicked.connect(self.irr_files_list.clear)
        btn_col.addWidget(add_btn)
        btn_col.addWidget(clear_btn)
        files_row.addWidget(self.irr_files_list, 1)
        files_row.addLayout(btn_col)
        lay.addLayout(files_row)

        ctrl = QtWidgets.QHBoxLayout()
        self.irr_bin_spin = QtWidgets.QDoubleSpinBox(page)
        self.irr_bin_spin.setRange(0.01, 60.0)
        self.irr_bin_spin.setDecimals(2)
        self.irr_bin_spin.setValue(0.10)
        self.irr_tol_spin = QtWidgets.QDoubleSpinBox(page)
        self.irr_tol_spin.setRange(0.0, 60.0)
        self.irr_tol_spin.setDecimals(2)
        self.irr_tol_spin.setValue(0.50)
        self.irr_pool_chk = QtWidgets.QCheckBox("Pool subjects", page)
        self.run_irr_btn = QtWidgets.QPushButton("Compute IRR & Disagreements", page)
        self.export_irr_dis_btn = QtWidgets.QPushButton(
            "Export Disagreements CSV…", page
        )
        self.export_irr_dis_btn.setEnabled(False)
        self.run_irr_btn.clicked.connect(
            lambda: self._dispatch_task("irr", self._compute_irr_job)
        )
        self.export_irr_dis_btn.clicked.connect(self._export_irr_disagreements_csv)

        ctrl.addWidget(QtWidgets.QLabel("Bin (s):"))
        ctrl.addWidget(self.irr_bin_spin)
        ctrl.addWidget(QtWidgets.QLabel("Onset Tol (s):"))
        ctrl.addWidget(self.irr_tol_spin)
        ctrl.addWidget(self.irr_pool_chk)
        ctrl.addWidget(self.run_irr_btn)
        ctrl.addWidget(self.export_irr_dis_btn)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        kpi_row = QtWidgets.QHBoxLayout()
        self.irr_kappa_val = self._make_kpi_card("Macro Cohen κ")
        self.irr_fleiss_val = self._make_kpi_card("Macro Fleiss κ")
        self.irr_alpha_val = self._make_kpi_card("Krippendorff α")
        self.irr_f1_val = self._make_kpi_card("Bout Onset F1")
        for v in (
            self.irr_kappa_val,
            self.irr_fleiss_val,
            self.irr_alpha_val,
            self.irr_f1_val,
        ):
            kpi_row.addWidget(v._card_frame)  # type: ignore[attr-defined]
        lay.addLayout(kpi_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, page)
        self.irr_rows_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Subject",
                "Behavior",
                "Cohen κ",
                "95% CI",
                "Krippendorff α",
                "Fleiss κ",
                "PABAK",
                "Onset F1",
            ],
        )
        self.irr_rows_table.setSortingEnabled(True)

        self.irr_disagreements_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Approx Frame",
                "Subject",
                "Behavior",
                "Start (s)",
                "End (s)",
                "Duration (s)",
                "Present Raters",
                "Absent Raters",
            ],
        )
        self.irr_disagreements_table.setSortingEnabled(True)
        self.irr_disagreements_table.cellDoubleClicked.connect(
            lambda r, _c: self._emit_row_frame(
                self.irr_disagreements_table, r, fallback_col=0
            )
        )

        splitter.addWidget(self.irr_rows_table)
        splitter.addWidget(self.irr_disagreements_table)
        lay.addWidget(splitter, 1)
        return page

    def _build_cohort_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(page)

        in_row = QtWidgets.QHBoxLayout()
        self.cohort_manifest_edit = QtWidgets.QLineEdit(page)
        self.cohort_manifest_edit.setPlaceholderText(
            "Cohort manifest CSV/JSON or directory of behavior event CSVs"
        )
        manifest_btn = QtWidgets.QPushButton("Manifest…", page)
        folder_btn = QtWidgets.QPushButton("Cohort Folder…", page)
        manifest_btn.clicked.connect(self._browse_cohort_manifest)
        folder_btn.clicked.connect(self._browse_cohort_folder)

        self.cohort_meta_edit = QtWidgets.QLineEdit(page)
        self.cohort_meta_edit.setPlaceholderText("Optional metadata CSV (unblinding)")
        meta_btn = QtWidgets.QPushButton("Metadata…", page)
        meta_btn.clicked.connect(self._browse_cohort_metadata)

        in_row.addWidget(self.cohort_manifest_edit, 2)
        in_row.addWidget(manifest_btn)
        in_row.addWidget(folder_btn)
        in_row.addWidget(self.cohort_meta_edit, 1)
        in_row.addWidget(meta_btn)
        lay.addLayout(in_row)

        ctrl = QtWidgets.QHBoxLayout()
        self.cohort_unit_combo = QtWidgets.QComboBox(page)
        self.cohort_unit_combo.addItem(
            "Recording (prevents dyadic pseudoreplication)", userData="recording"
        )
        self.cohort_unit_combo.addItem("Subject", userData="subject")
        self.run_cohort_btn = QtWidgets.QPushButton(
            "Run Cohort Inferential Stats & Transitions", page
        )
        self.export_cohort_cmp_btn = QtWidgets.QPushButton(
            "Export Comparisons CSV…", page
        )
        self.export_cohort_cmp_btn.setEnabled(False)
        self.run_cohort_btn.clicked.connect(
            lambda: self._dispatch_task("cohort", self._compute_cohort_job)
        )
        self.export_cohort_cmp_btn.clicked.connect(self._export_cohort_comparisons_csv)

        ctrl.addWidget(QtWidgets.QLabel("Analysis Unit:"))
        ctrl.addWidget(self.cohort_unit_combo)
        ctrl.addWidget(self.run_cohort_btn)
        ctrl.addWidget(self.export_cohort_cmp_btn)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, page)
        self.cohort_cmp_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Behavior",
                "Endpoint",
                "Groups",
                "Test",
                "p-value",
                "BH-FDR q",
                "Hedges' g",
                "Cliff's δ",
                "Diff 95% CI",
                "Log-Rank p",
            ],
        )
        self.cohort_cmp_table.setSortingEnabled(True)

        self.cohort_summary_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            [
                "Group",
                "Behavior",
                "N Units",
                "Responders",
                "Mean Duration (s)",
                "SD (s)",
                "Median Duration (s)",
                "KM Median Latency (s)",
            ],
        )
        self.cohort_summary_table.setSortingEnabled(True)

        self.cohort_transitions_table = _configure_table(
            QtWidgets.QTableWidget(splitter),
            ["Group", "From Behavior", "To Behavior", "Count", "Probability", "Lift"],
        )
        self.cohort_transitions_table.setSortingEnabled(True)

        splitter.addWidget(self.cohort_cmp_table)
        splitter.addWidget(self.cohort_summary_table)
        splitter.addWidget(self.cohort_transitions_table)
        lay.addWidget(splitter, 1)
        return page

    # ------------------------------------------------------------------
    # Context & File Browsers
    # ------------------------------------------------------------------
    def apply_owner_context(
        self, *, override_source: Optional[str | Path] = None
    ) -> None:
        if override_source is not None and str(override_source).strip():
            self.source_edit.setText(str(Path(override_source).expanduser().resolve()))
        elif self._owner_window is not None:
            w = self._owner_window
            for attr in ("video_file", "video_results_folder", "annotation_dir"):
                raw = getattr(w, attr, None)
                if raw and str(raw).strip():
                    self.source_edit.setText(str(Path(str(raw)).expanduser()))
                    break
            zone_p = getattr(w, "zone_path", None)
            if zone_p and str(zone_p).strip():
                self.zone_file_edit.setText(str(zone_p).strip())
            fps_val = getattr(w, "fps", None)
            if fps_val:
                try:
                    self.fps_spin.setValue(max(0.0, float(fps_val)))
                except (TypeError, ValueError):
                    pass

    def _browse_source_file(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video or Tracking CSV",
            self.source_edit.text().strip() or str(Path.cwd()),
            "Videos & CSVs (*.mp4 *.avi *.mov *.mkv *.csv);;All Files (*)",
        )
        if chosen:
            self.source_edit.setText(chosen)

    def _browse_source_dir(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Frame JSON or Workspace Directory",
            self.source_edit.text().strip() or str(Path.cwd()),
        )
        if chosen:
            self.source_edit.setText(chosen)

    def _browse_events_csv(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Behavior Event CSV",
            self.events_csv_edit.text().strip() or str(Path.cwd()),
            "CSV Files (*.csv)",
        )
        if chosen:
            self.events_csv_edit.setText(chosen)

    def _browse_zone_file(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Zone JSON File",
            self.zone_file_edit.text().strip() or str(Path.cwd()),
            "JSON Files (*.json)",
        )
        if chosen:
            self.zone_file_edit.setText(chosen)

    def _browse_calibration_file(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Planar / Homography Calibration JSON",
            self.calibration_edit.text().strip() or str(Path.cwd()),
            "JSON Files (*.json)",
        )
        if chosen:
            self.calibration_edit.setText(chosen)

    def _add_irr_csvs(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select 2 or More Rater Behavior Event CSVs",
            str(Path.cwd()),
            "CSV Files (*.csv)",
        )
        existing = {
            self.irr_files_list.item(i).text()
            for i in range(self.irr_files_list.count())
        }
        for f in files:
            if f not in existing:
                self.irr_files_list.addItem(f)

    def _browse_cohort_manifest(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Cohort Manifest CSV or JSON",
            self.cohort_manifest_edit.text().strip() or str(Path.cwd()),
            "Manifest Files (*.csv *.json)",
        )
        if chosen:
            self.cohort_manifest_edit.setText(chosen)

    def _browse_cohort_folder(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Cohort Events Directory",
            self.cohort_manifest_edit.text().strip() or str(Path.cwd()),
        )
        if chosen:
            self.cohort_manifest_edit.setText(chosen)

    def _browse_cohort_metadata(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Metadata CSV",
            self.cohort_meta_edit.text().strip() or str(Path.cwd()),
            "CSV Files (*.csv)",
        )
        if chosen:
            self.cohort_meta_edit.setText(chosen)

    # ------------------------------------------------------------------
    # Parameter Extraction & Jump Helpers
    # ------------------------------------------------------------------
    def _effective_fps(self) -> Optional[float]:
        val = float(self.fps_spin.value())
        return val if val > 0 else None

    def _effective_px_per_cm(self) -> Optional[float]:
        val = float(self.px_per_cm_spin.value())
        return val if val > 0 else None

    def _effective_calibration_path(self) -> Optional[Path]:
        txt = self.calibration_edit.text().strip()
        return Path(txt).expanduser().resolve() if txt else None

    def _resolve_tracking_input(self) -> Path:
        raw = self.source_edit.text().strip()
        if not raw:
            raise ValueError(
                "Select a video, tracking CSV, or frame JSON directory first."
            )
        p = Path(raw).expanduser().resolve()
        if p.is_file() and p.suffix.lower() != ".csv":
            for suf in ("_tracking.csv", "_tracked.csv"):
                cand = p.parent / f"{p.stem}{suf}"
                if cand.is_file():
                    return cand
            cand_dir = p.with_suffix("")
            if cand_dir.is_dir():
                return cand_dir
        return p

    def _resolve_events_csv_or_temp(self) -> Path:
        explicit = self.events_csv_edit.text().strip()
        if explicit:
            p = Path(explicit).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Events CSV not found: {p}")
            return p

        # Fallback: export active BehaviorController events from main window if present
        w = self._owner_window
        beh_ctrl = getattr(w, "behavior_controller", None) if w is not None else None
        if beh_ctrl is not None and getattr(beh_ctrl, "events_count", 0) > 0:
            est = getattr(w, "_estimate_recording_time", None)
            rows = beh_ctrl.export_rows(
                timestamp_fallback=(lambda evt: est(evt.frame))
                if callable(est)
                else None
            )
            if rows:
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(
                    ["Trial time", "Recording time", "Subject", "Behavior", "Event"]
                )
                for r in rows:
                    writer.writerow(list(r))
                tmp = Path(tempfile.gettempdir()) / "annolid_active_session_events.csv"
                tmp.write_text(buf.getvalue(), encoding="utf-8")
                return tmp

        src_text = self.source_edit.text().strip()
        if src_text:
            sp = Path(src_text).expanduser().resolve()
            stem = sp.name if sp.is_dir() else sp.stem
            for suf in ("_tracking", "_tracked"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            parent = sp if sp.is_dir() else sp.parent
            for cand_name in (f"{stem}_events.csv", f"{stem}_behavior_events.csv"):
                cand = parent / cand_name
                if cand.is_file():
                    return cand

        raise ValueError(
            "Provide a Behavior Event CSV or record behavior events in the main window."
        )

    def _emit_row_frame(
        self, table: QtWidgets.QTableWidget, row: int, *, fallback_col: int = 0
    ) -> None:
        first_item = table.item(row, 0)
        if first_item is not None:
            role_val = first_item.data(QtCore.Qt.UserRole)
            if role_val is not None:
                self.jumpToFrameRequested.emit(max(0, int(role_val)))
                return
        cell = table.item(row, fallback_col)
        if cell is not None:
            try:
                self.jumpToFrameRequested.emit(max(0, int(float(cell.text().strip()))))
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Synchronous Job Computations (Callable directly in tests or QThread)
    # ------------------------------------------------------------------
    def _compute_report_job(self) -> UnifiedReportArtifacts:
        raw = self.source_edit.text().strip()
        if not raw:
            raise ValueError("Please specify a Source path first.")
        ev_text = self.events_csv_edit.text().strip()
        z_text = self.zone_file_edit.text().strip()
        title = self.report_title_edit.text().strip() or None
        return build_unified_analysis_report(
            Path(raw).expanduser().resolve(),
            events_csv=Path(ev_text).expanduser().resolve() if ev_text else None,
            zone_file=Path(z_text).expanduser().resolve() if z_text else None,
            calibration_path=self._effective_calibration_path(),
            fps=self._effective_fps(),
            px_per_cm=self._effective_px_per_cm(),
            title=title,
            include_plots=True,
        )

    def _compute_track_qc_job(self) -> Dict[str, Any]:
        track_in = self._resolve_tracking_input()
        exp = int(self.expected_instances_spin.value())
        tqc = run_track_qc(
            track_in,
            expected_instances=exp if exp > 0 else None,
            fps=self._effective_fps(),
            top_k_review=30,
        )
        raw_src = Path(self.source_edit.text().strip()).expanduser().resolve()
        gaps = summarize_tracking_gaps(raw_src, fps=self._effective_fps())
        return {"track_qc": tqc, "gaps": gaps}

    def _compute_kinematics_job(self) -> Any:
        track_in = self._resolve_tracking_input()
        freeze_thr = float(self.freeze_thresh_spin.value())
        return run_kinematics_analysis(
            track_in,
            fps=self._effective_fps(),
            px_per_cm=self._effective_px_per_cm(),
            calibration=self._effective_calibration_path(),
            freeze_speed_threshold=freeze_thr if freeze_thr > 0 else None,
            min_freeze_duration_s=float(self.min_freeze_dur_spin.value()),
        )

    def _compute_behavior_job(self) -> Dict[str, Any]:
        ev_path = self._resolve_events_csv_or_temp()
        sess_dur = float(self.beh_session_dur_spin.value())
        bin_w = float(self.beh_bin_spin.value())
        rows = _read_event_rows(ev_path)
        intervals, warnings = extract_behavior_intervals(rows)
        summary = summarize_intervals(
            intervals, session_duration=sess_dur if sess_dur > 0 else None
        )
        binned = compute_binned_time_budget(intervals, bin_w) if bin_w > 0 else []
        qc_report = run_behavior_qc(ev_path)
        return {
            "summary": summary,
            "binned": binned,
            "warnings": warnings,
            "qc": qc_report,
        }

    def _compute_irr_job(self) -> Any:
        csv_paths = [
            Path(self.irr_files_list.item(i).text()).expanduser().resolve()
            for i in range(self.irr_files_list.count())
        ]
        if len(csv_paths) < 2:
            raise ValueError("Add at least 2 rater Behavior Event CSVs.")
        return run_behavior_irr(
            csv_paths,
            bin_seconds=float(self.irr_bin_spin.value()),
            onset_tolerance_seconds=float(self.irr_tol_spin.value()),
            pool_subjects=bool(self.irr_pool_chk.isChecked()),
        )

    def _compute_cohort_job(self) -> Any:
        target_str = self.cohort_manifest_edit.text().strip()
        if not target_str:
            raise ValueError(
                "Select a Cohort Manifest file or Cohort Events directory."
            )
        target = Path(target_str).expanduser().resolve()
        meta_str = self.cohort_meta_edit.text().strip()
        meta_p = Path(meta_str).expanduser().resolve() if meta_str else None
        unit = str(self.cohort_unit_combo.currentData() or "recording")
        if target.is_dir():
            return run_behavior_cohort(
                events_csvs=[target],
                metadata_csv=meta_p,
                analysis_unit=unit,
            )
        return run_behavior_cohort(
            manifest_path=target,
            metadata_csv=meta_p,
            analysis_unit=unit,
        )

    # Direct synchronous entrypoints for deterministic tests & callers
    def run_unified_report_sync(self) -> UnifiedReportArtifacts:
        res = self._compute_report_job()
        self._apply_task_result("report", res)
        return res

    def run_track_qc_sync(self) -> Dict[str, Any]:
        res = self._compute_track_qc_job()
        self._apply_task_result("track_qc", res)
        return res

    def run_kinematics_sync(self) -> Any:
        res = self._compute_kinematics_job()
        self._apply_task_result("kinematics", res)
        return res

    def run_behavior_sync(self) -> Dict[str, Any]:
        res = self._compute_behavior_job()
        self._apply_task_result("behavior", res)
        return res

    def run_irr_sync(self) -> Any:
        res = self._compute_irr_job()
        self._apply_task_result("irr", res)
        return res

    def run_cohort_sync(self) -> Any:
        res = self._compute_cohort_job()
        self._apply_task_result("cohort", res)
        return res

    # ------------------------------------------------------------------
    # Async Worker Dispatch & Result Rendering
    # ------------------------------------------------------------------
    def _dispatch_task(self, task_name: str, fn: Callable[[], Any]) -> None:
        if not self.isVisible():
            try:
                result = fn()
                self._apply_task_result(task_name, result)
            except Exception as exc:
                self.status_label.setText(f"{task_name} failed: {exc}")
            return

        if self._active_thread is not None:
            self.status_label.setText("Another analysis task is currently running…")
            return

        self.busy_bar.setVisible(True)
        self.status_label.setText(f"Running {task_name} in background…")
        thread = QtCore.QThread(self)
        worker = _CallableWorker(task_name, fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    def _clear_worker(self) -> None:
        self._active_thread = None
        self._active_worker = None

    @QtCore.Slot(str, object, str)
    def _on_worker_finished(self, task_name: str, result: object, error: str) -> None:
        self.busy_bar.setVisible(False)
        if error:
            self.status_label.setText(f"{task_name} failed: {error}")
            QtWidgets.QMessageBox.warning(
                self, "Analysis Error", f"{task_name} failed:\n\n{error}"
            )
            return
        self._apply_task_result(task_name, result)

    def _apply_task_result(self, task_name: str, result: Any) -> None:
        eff_fps = self._effective_fps() or 30.0

        if task_name == "report" and isinstance(result, UnifiedReportArtifacts):
            self._last_report_artifacts = result
            payload = result.payload
            src = payload.get("source") or {}
            sections = payload.get("sections") or {}
            active_secs = [k for k, v in sections.items() if v is not None]
            inst_list = (
                (sections.get("kinematics") or {}).get("instances")
                or (sections.get("track_qc") or {}).get("instances")
                or []
            )
            self.rep_status_val.setText(str(payload.get("overall_status", "PASS")))
            self.rep_frames_val.setText(str(src.get("total_frames", 0)))
            self.rep_fps_val.setText(_fmt_num(src.get("fps"), 2))
            self.rep_instances_val.setText(str(len(inst_list)))
            self.rep_sections_val.setText(str(len(active_secs)))
            self.open_html_btn.setEnabled(result.html_path.is_file())
            self.open_md_btn.setEnabled(result.markdown_path.is_file())
            self.open_report_dir_btn.setEnabled(result.output_dir.is_dir())
            try:
                md_content = result.markdown_path.read_text(encoding="utf-8")
                self.report_preview.setMarkdown(md_content)
            except Exception:
                self.report_preview.setPlainText(str(result.html_path))
            self.status_label.setText(f"Unified report saved to {result.html_path}")
            return

        if task_name == "track_qc" and isinstance(result, dict):
            tqc = result["track_qc"]
            gaps = result["gaps"]
            _fill_table_rows(
                self.tqc_instances_table,
                [
                    [
                        inst.instance,
                        inst.observed_frames,
                        f"{inst.coverage_ratio * 100.0:.1f}%",
                        inst.missing_frames,
                        inst.duplicate_frames,
                        _fmt_num(inst.median_speed_px_per_frame, 2),
                        _fmt_num(inst.max_speed_px_per_frame, 2),
                    ]
                    for inst in tqc.instances
                ],
            )
            _fill_table_rows(
                self.tqc_review_table,
                [
                    [
                        b.rank,
                        b.frame_start,
                        b.frame_end,
                        _fmt_num(b.risk_score, 2),
                        ", ".join(b.instances),
                        ", ".join(b.issue_codes),
                        b.summary,
                    ]
                    for b in tqc.review_queue
                ],
                frame_targets=[int(b.frame_start) for b in tqc.review_queue],
            )
            gap_rows: List[List[Any]] = []
            gap_frames: List[int] = []
            for inst_info in gaps.get("instances") or []:
                lbl = inst_info.get("instance", "")
                for g in inst_info.get("gaps") or []:
                    sf = int(g["start_frame"])
                    gap_frames.append(sf)
                    gap_rows.append(
                        [
                            lbl,
                            sf,
                            int(g["end_frame"]),
                            int(g["duration_frames"]),
                            _fmt_num(g.get("duration_seconds"), 3),
                        ]
                    )
            _fill_table_rows(self.tqc_gaps_table, gap_rows, frame_targets=gap_frames)
            self.status_label.setText(
                f"Track QC complete: {tqc.error_count} error(s), {tqc.warning_count} warning(s), {len(tqc.review_queue)} review bout(s)."
            )
            return

        if task_name == "kinematics":
            self._last_kinematics_report = result
            self.export_kin_events_btn.setEnabled(True)
            d = result.to_dict()
            _fill_table_rows(
                self.kin_instances_table,
                [
                    [
                        r.get("instance", ""),
                        _fmt_num(r.get("total_distance_px"), 1),
                        _fmt_num(r.get("total_distance_cm"), 2),
                        _fmt_num(r.get("mean_speed_px_s"), 2),
                        _fmt_num(r.get("mean_speed_cm_s"), 2),
                        _fmt_num(r.get("p95_speed_px_s"), 2),
                        _fmt_num(r.get("freezing_pct"), 1, "%"),
                        r.get("freezing_bouts", 0),
                        _fmt_num(r.get("straightness_index"), 3),
                        _fmt_num(r.get("tortuosity"), 2),
                    ]
                    for r in (d.get("instances") or [])
                ],
            )
            bouts = d.get("bouts") or []
            _fill_table_rows(
                self.kin_bouts_table,
                [
                    [
                        b.get("instance", ""),
                        b.get("kind", ""),
                        b.get("start_frame", 0),
                        b.get("end_frame", 0),
                        _fmt_num(b.get("start_s"), 2),
                        _fmt_num(b.get("duration_s"), 2),
                        _fmt_num(b.get("mean_speed_px_s"), 2),
                    ]
                    for b in bouts
                ],
                frame_targets=[int(b.get("start_frame", 0)) for b in bouts],
            )
            _fill_table_rows(
                self.kin_pairwise_table,
                [
                    [
                        f"{p.get('instance_a', '')} ↔ {p.get('instance_b', '')}",
                        p.get("shared_frames", 0),
                        _fmt_num(p.get("mean_distance_px"), 1),
                        _fmt_num(p.get("min_distance_px"), 1),
                        _fmt_num(p.get("mean_distance_cm"), 2),
                        _fmt_num(p.get("close_contact_pct"), 1, "%"),
                        _fmt_num(p.get("velocity_correlation"), 3),
                    ]
                    for p in (d.get("pairwise") or [])
                ],
            )
            self.status_label.setText(
                f"Kinematics complete: {len(d.get('instances') or [])} instance(s), {len(bouts)} bout(s)."
            )
            return

        if task_name == "behavior" and isinstance(result, dict):
            summary = result["summary"]
            binned = result["binned"]
            qc = result["qc"]
            _fill_table_rows(
                self.beh_summary_table,
                [
                    [
                        r.subject,
                        r.behavior,
                        r.occurrences,
                        _fmt_num(r.total_duration, 3),
                        _fmt_num(r.mean_duration, 3),
                        _fmt_num(r.median_duration, 3),
                        _fmt_num(r.percent_of_session, 2, "%"),
                    ]
                    for r in summary
                ],
            )
            _fill_table_rows(
                self.beh_binned_table,
                [
                    [
                        idx,
                        f"{b.bin_start:.1f}–{b.bin_end:.1f}",
                        b.subject,
                        b.behavior,
                        "—",
                        _fmt_num(b.duration, 3),
                        _fmt_num(
                            100.0
                            * float(b.duration)
                            / max(1e-9, float(b.bin_end) - float(b.bin_start)),
                            1,
                            "%",
                        ),
                    ]
                    for idx, b in enumerate(binned)
                ],
            )
            qc_frames = [
                int(round(float(iss.timestamp or 0.0) * eff_fps)) for iss in qc.issues
            ]
            _fill_table_rows(
                self.beh_qc_table,
                [
                    [
                        frm,
                        iss.severity,
                        iss.code,
                        iss.subject or "",
                        iss.behavior or "",
                        _fmt_num(iss.timestamp, 3),
                        iss.message,
                    ]
                    for frm, iss in zip(qc_frames, qc.issues)
                ],
                frame_targets=qc_frames,
            )
            self.status_label.setText(
                f"Behavior analysis complete: {len(summary)} subject-behavior row(s), {len(qc.issues)} QC alert(s)."
            )
            return

        if task_name == "irr":
            self._last_irr_report = result
            self.export_irr_dis_btn.setEnabled(True)
            d = result.to_dict()
            self.irr_kappa_val.setText(_fmt_num(d.get("macro_cohens_kappa"), 3))
            self.irr_fleiss_val.setText(_fmt_num(d.get("macro_fleiss_kappa"), 3))
            self.irr_alpha_val.setText(_fmt_num(d.get("macro_krippendorff_alpha"), 3))
            self.irr_f1_val.setText(_fmt_num(d.get("macro_onset_f1"), 3))
            _fill_table_rows(
                self.irr_rows_table,
                [
                    [
                        r.get("subject", ""),
                        r.get("behavior", ""),
                        _fmt_num(r.get("cohens_kappa"), 3),
                        (
                            f"[{r['cohens_kappa_ci95_low']:.3f}, {r['cohens_kappa_ci95_high']:.3f}]"
                            if r.get("cohens_kappa_ci95_low") is not None
                            and r.get("cohens_kappa_ci95_high") is not None
                            else "—"
                        ),
                        _fmt_num(r.get("krippendorff_alpha"), 3),
                        _fmt_num(r.get("fleiss_kappa"), 3),
                        _fmt_num(r.get("pabak"), 3),
                        _fmt_num(r.get("onset_f1"), 3),
                    ]
                    for r in (d.get("rows") or [])
                ],
            )
            dis_list = d.get("disagreements") or []
            dis_frames = [
                int(round(float(b.get("start_s", 0.0)) * eff_fps)) for b in dis_list
            ]
            _fill_table_rows(
                self.irr_disagreements_table,
                [
                    [
                        frm,
                        b.get("subject", ""),
                        b.get("behavior", ""),
                        _fmt_num(b.get("start_s"), 2),
                        _fmt_num(b.get("end_s"), 2),
                        _fmt_num(b.get("duration_s"), 2),
                        ", ".join(b.get("raters_active") or []),
                        ", ".join(b.get("raters_inactive") or []),
                    ]
                    for frm, b in zip(dis_frames, dis_list)
                ],
                frame_targets=dis_frames,
            )
            self.status_label.setText(
                f"IRR complete across {len(d.get('raters') or [])} raters: {len(dis_list)} disagreement bout(s)."
            )
            return

        if task_name == "cohort":
            self._last_cohort_report = result
            self.export_cohort_cmp_btn.setEnabled(bool(result.comparisons))
            d = result.to_dict()
            _fill_table_rows(
                self.cohort_cmp_table,
                [
                    [
                        c.get("behavior", ""),
                        "total_duration_s",
                        " vs ".join(c.get("groups") or []),
                        c.get("parametric_test", ""),
                        _fmt_num(c.get("p_parametric"), 4),
                        _fmt_num(c.get("q_parametric_bh"), 4),
                        _fmt_num(c.get("hedges_g"), 3),
                        _fmt_num(c.get("cliffs_delta"), 3),
                        (
                            f"[{c['mean_diff_ci95_low_s']:.3f}, {c['mean_diff_ci95_high_s']:.3f}]"
                            if c.get("mean_diff_ci95_low_s") is not None
                            and c.get("mean_diff_ci95_high_s") is not None
                            else "—"
                        ),
                        _fmt_num(c.get("logrank_p_value"), 4),
                    ]
                    for c in (d.get("comparisons") or [])
                ],
            )
            _fill_table_rows(
                self.cohort_summary_table,
                [
                    [
                        g.get("group", ""),
                        g.get("behavior", ""),
                        g.get("n_subjects", 0),
                        g.get("n_responders", 0),
                        _fmt_num(g.get("mean_total_duration_s"), 3),
                        _fmt_num(g.get("sd_total_duration_s"), 3),
                        _fmt_num(g.get("median_total_duration_s"), 3),
                        _fmt_num(g.get("median_latency_censored_s"), 3),
                    ]
                    for g in (d.get("group_summaries") or [])
                ],
            )
            _fill_table_rows(
                self.cohort_transitions_table,
                [
                    [
                        t.get("group", ""),
                        t.get("from_behavior", ""),
                        t.get("to_behavior", ""),
                        t.get("count", 0),
                        _fmt_num(t.get("probability"), 3),
                        _fmt_num(t.get("lift"), 3),
                    ]
                    for t in (d.get("transitions") or [])
                ],
            )
            self.status_label.setText(
                f"Cohort analysis complete ({d.get('analysis_unit', 'recording')} unit): "
                f"{len(d.get('comparisons') or [])} comparison row(s)."
            )

    # ------------------------------------------------------------------
    # Export & Open Helpers
    # ------------------------------------------------------------------
    def _open_report_html(self) -> None:
        if (
            self._last_report_artifacts
            and self._last_report_artifacts.html_path.is_file()
        ):
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self._last_report_artifacts.html_path))
            )

    def _open_report_md(self) -> None:
        if (
            self._last_report_artifacts
            and self._last_report_artifacts.markdown_path.is_file()
        ):
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(
                    str(self._last_report_artifacts.markdown_path)
                )
            )

    def _open_report_folder(self) -> None:
        if (
            self._last_report_artifacts
            and self._last_report_artifacts.output_dir.is_dir()
        ):
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(self._last_report_artifacts.output_dir))
            )

    def _export_kinematics_events_csv(self) -> None:
        if self._last_kinematics_report is None:
            return
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Kinematics Bouts as Behavior Events CSV",
            "kinematics_events.csv",
            "CSV Files (*.csv)",
        )
        if chosen:
            write_kinematics_events_csv(self._last_kinematics_report, Path(chosen))
            self.status_label.setText(f"Exported kinematics bouts to {chosen}")

    def _export_irr_disagreements_csv(self) -> None:
        if self._last_irr_report is None:
            return
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export IRR Disagreement Bouts CSV",
            "irr_disagreements.csv",
            "CSV Files (*.csv)",
        )
        if chosen:
            write_irr_disagreements_csv(
                self._last_irr_report.disagreements, Path(chosen)
            )
            self.status_label.setText(f"Exported IRR disagreements to {chosen}")

    def _export_cohort_comparisons_csv(self) -> None:
        if self._last_cohort_report is None:
            return
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Cohort Inferential Comparisons CSV",
            "cohort_comparisons.csv",
            "CSV Files (*.csv)",
        )
        if chosen:
            write_cohort_comparisons_csv(
                self._last_cohort_report.comparisons, Path(chosen)
            )
            self.status_label.setText(f"Exported cohort comparisons to {chosen}")


class AnalysisStudioDialog(QtWidgets.QDialog):
    """Non-modal top-level dialog hosting ``AnalysisStudioWidget``."""

    jumpToFrameRequested = QtCore.Signal(int)

    def __init__(
        self,
        *,
        owner_window: Optional[QtWidgets.QWidget] = None,
        initial_source: Optional[str | Path] = None,
        initial_tab: str = "report",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annolid Analysis, Statistics & Reporting Studio")
        self.resize(1280, 860)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.studio = AnalysisStudioWidget(
            owner_window=owner_window or parent,
            initial_source=initial_source,
            parent=self,
        )
        self.studio.jumpToFrameRequested.connect(self.jumpToFrameRequested.emit)
        self.studio.select_tab(initial_tab)
        lay.addWidget(self.studio)
