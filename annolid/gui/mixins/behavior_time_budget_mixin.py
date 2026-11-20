from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from qtpy import QtGui, QtWidgets

from annolid.behavior.time_budget import (
    TimeBudgetComputationError,
    format_category_summary,
)
from annolid.services.time_budget import (
    compute_behavior_time_budget_report,
    write_behavior_time_budget_report_csv,
)
from annolid.utils.atomic_io import atomic_write_json


class BehaviorTimeBudgetMixin:
    """Behavior time-budget dialog and CSV/JSON export helpers."""

    def show_behavior_time_budget_dialog(self) -> None:
        """Summarise recorded behavior events using the time-budget report."""
        if hasattr(self.behavior_controller, "export_detailed_rows"):
            rows = self.behavior_controller.export_detailed_rows(
                timestamp_fallback=lambda evt: self._estimate_recording_time(evt.frame)
            )
        else:
            rows = self.behavior_controller.export_rows(
                timestamp_fallback=lambda evt: self._estimate_recording_time(evt.frame)
            )
        if not rows:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Behavior Time Budget"),
                self.tr("No behavior events are available to summarise."),
            )
            return

        try:
            report = compute_behavior_time_budget_report(
                rows,
                schema=self.project_schema,
            )
        except TimeBudgetComputationError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Behavior Time Budget"),
                self.tr("Unable to compute the time budget:\n%s") % exc,
            )
            return

        summary = list(report.summary)
        warnings: List[str] = list(report.warnings)
        schema_for_dialog = self.project_schema
        category_summary: List[Tuple[str, float, int]] = list(report.category_summary)
        cooccurrence_summary = list(report.cooccurrence_summary)

        if not summary:
            message = self.tr(
                "No completed start/end pairs or point events were found for the current behavior events."
            )
            if warnings:
                message += "\n\n" + self.tr("Warnings:\n") + "\n".join(warnings)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Behavior Time Budget"),
                message,
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Behavior Time Budget"))
        layout = QtWidgets.QVBoxLayout(dialog)

        tabs = QtWidgets.QTabWidget(dialog)

        # Tab 1: Sortable Interactive Table
        headers = [
            "Subject",
            "Behavior",
            "Occurrences",
            "Total (s)",
            "Mean (s)",
            "SD (s)",
            "Median (s)",
            "Min–Max (s)",
            "Rate (/min)",
            "Latency (s)",
            "Mean IEI (s)",
            "% Session",
        ]
        summary_table = QtWidgets.QTableWidget(len(summary), len(headers), tabs)
        summary_table.setHorizontalHeaderLabels(headers)
        summary_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        summary_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        summary_table.setAlternatingRowColors(True)
        for r_idx, row in enumerate(summary):
            cells = [
                row.subject,
                row.behavior,
                str(int(row.occurrences)),
                f"{row.total_duration:.3f}",
                f"{row.mean_duration:.3f}",
                f"{row.std_duration:.3f}",
                f"{row.median_duration:.3f}",
                f"{row.min_duration:.2f}–{row.max_duration:.2f}",
                f"{row.rate_per_min:.2f}",
                f"{row.latency_to_first:.3f}",
                (
                    f"{row.inter_event_interval_mean:.3f}"
                    if row.inter_event_interval_mean is not None
                    else "—"
                ),
                f"{row.percent_of_session:.2f}%",
            ]
            for c_idx, val in enumerate(cells):
                summary_table.setItem(r_idx, c_idx, QtWidgets.QTableWidgetItem(val))
        summary_table.resizeColumnsToContents()
        summary_table.horizontalHeader().setStretchLastSection(True)
        summary_table.setSortingEnabled(True)
        tabs.addTab(summary_table, self.tr("Summary Table"))

        # Tab 2: Formatted Fixed-Width Text
        report_view = QtWidgets.QPlainTextEdit(tabs)
        report_view.setReadOnly(True)
        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        report_view.setFont(fixed_font)
        report_view.setPlainText(report.table_text)
        tabs.addTab(report_view, self.tr("Formatted Text"))

        if schema_for_dialog is not None and category_summary:
            category_view = QtWidgets.QPlainTextEdit(tabs)
            category_view.setReadOnly(True)
            category_view.setFont(fixed_font)
            category_view.setPlainText(format_category_summary(category_summary))
            tabs.addTab(category_view, self.tr("Category Summary"))

        if cooccurrence_summary and report.cooccurrence_table_text:
            coocc_view = QtWidgets.QPlainTextEdit(tabs)
            coocc_view.setReadOnly(True)
            coocc_view.setFont(fixed_font)
            coocc_view.setPlainText(report.cooccurrence_table_text)
            tabs.addTab(coocc_view, self.tr("Co-occurrence"))

        layout.addWidget(tabs, 1)

        if warnings:
            warning_label = QtWidgets.QLabel(self.tr("Warnings:"))
            warning_label.setStyleSheet("font-weight: bold;")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

            warning_view = QtWidgets.QPlainTextEdit()
            warning_view.setReadOnly(True)
            warning_view.setPlainText("\n".join(warnings))
            warning_view.setMaximumHeight(130)
            warning_view.setStyleSheet(
                "background-color: #fff4e5; color: #7c2d12; border: 1px solid #f59e0b; border-radius: 4px;"
            )
            layout.addWidget(warning_view)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        save_button = button_box.addButton(
            self.tr("Save CSV…"), QtWidgets.QDialogButtonBox.ActionRole
        )
        save_json_button = button_box.addButton(
            self.tr("Save JSON…"), QtWidgets.QDialogButtonBox.ActionRole
        )
        studio_button = button_box.addButton(
            self.tr("Open Analysis Studio…"), QtWidgets.QDialogButtonBox.ActionRole
        )

        def _save_csv() -> None:
            default_name = "behavior_time_budget.csv"
            default_path = (
                str(Path(self.video_file).with_suffix(".time_budget.csv"))
                if self.video_file
                else default_name
            )
            path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.tr("Save Time-Budget CSV"),
                default_path,
                self.tr("CSV files (*.csv)"),
            )
            if not path_str:
                return
            try:
                output_path = Path(path_str)
                write_behavior_time_budget_report_csv(
                    report, output_path, schema=schema_for_dialog
                )
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Behavior Time Budget"),
                    self.tr("Failed to save CSV:\n%s") % exc,
                )
            else:
                self.statusBar().showMessage(
                    self.tr("Time-budget exported to %s") % Path(path_str).name, 4000
                )

        def _save_json() -> None:
            default_name = "behavior_time_budget.json"
            default_path = (
                str(Path(self.video_file).with_suffix(".time_budget.json"))
                if self.video_file
                else default_name
            )
            path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.tr("Save Time-Budget JSON"),
                default_path,
                self.tr("JSON files (*.json)"),
            )
            if not path_str:
                return
            payload = {
                "schema": "behavior_time_budget/v1",
                "video_file": str(self.video_file) if self.video_file else None,
                "warnings": list(warnings),
                "rows": [
                    {
                        "subject": r.subject,
                        "behavior": r.behavior,
                        "occurrences": int(r.occurrences),
                        "total_duration": round(float(r.total_duration), 6),
                        "mean_duration": round(float(r.mean_duration), 6),
                        "std_duration": round(float(r.std_duration), 6),
                        "median_duration": round(float(r.median_duration), 6),
                        "min_duration": round(float(r.min_duration), 6),
                        "max_duration": round(float(r.max_duration), 6),
                        "rate_per_min": round(float(r.rate_per_min), 6),
                        "latency_to_first": round(float(r.latency_to_first), 6),
                        "inter_event_interval_mean": (
                            round(float(r.inter_event_interval_mean), 6)
                            if r.inter_event_interval_mean is not None
                            else None
                        ),
                        "inter_event_interval_std": (
                            round(float(r.inter_event_interval_std), 6)
                            if r.inter_event_interval_std is not None
                            else None
                        ),
                        "percent_of_session": round(float(r.percent_of_session), 4),
                    }
                    for r in summary
                ],
                "cooccurrences": [
                    {
                        "behavior_a": c.behavior_a,
                        "behavior_b": c.behavior_b,
                        "overlap_bouts": int(c.overlap_bouts),
                        "overlap_duration": round(float(c.overlap_duration), 6),
                        "jaccard_index": round(float(c.jaccard_index), 6),
                    }
                    for c in cooccurrence_summary
                ],
            }
            try:
                atomic_write_json(Path(path_str), payload, indent=2, ensure_ascii=False)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Behavior Time Budget"),
                    self.tr("Failed to save JSON:\n%s") % exc,
                )
            else:
                self.statusBar().showMessage(
                    self.tr("Time-budget JSON exported to %s") % Path(path_str).name,
                    4000,
                )

        def _open_studio() -> None:
            opener = getattr(self, "open_analysis_studio", None)
            if callable(opener):
                dialog.accept()
                opener(initial_tab="behavior")

        save_button.clicked.connect(_save_csv)
        save_json_button.clicked.connect(_save_json)
        studio_button.clicked.connect(_open_studio)
        layout.addWidget(button_box)

        dialog.resize(840, 560)
        dialog.exec_()
