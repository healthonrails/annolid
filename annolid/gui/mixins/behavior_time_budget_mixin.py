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


class BehaviorTimeBudgetMixin:
    """Behavior time-budget dialog and CSV export helpers."""

    def show_behavior_time_budget_dialog(self) -> None:
        """Summarise recorded behavior events using the time-budget report."""
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

        if not summary:
            message = self.tr(
                "No completed start/end pairs were found for the current behavior events."
            )
            if warnings:
                message += "\n\n" + self.tr("Warnings:\n") + "\n".join(warnings)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Behavior Time Budget"),
                message,
            )
            return

        report_text = report.table_text

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Behavior Time Budget"))
        layout = QtWidgets.QVBoxLayout(dialog)

        report_view = QtWidgets.QPlainTextEdit()
        report_view.setReadOnly(True)
        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        report_view.setFont(fixed_font)
        report_view.setPlainText(report_text)
        layout.addWidget(report_view)

        if warnings:
            warning_label = QtWidgets.QLabel(self.tr("Warnings:"))
            warning_label.setStyleSheet("font-weight: bold;")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

            warning_view = QtWidgets.QPlainTextEdit()
            warning_view.setReadOnly(True)
            warning_view.setPlainText("\n".join(warnings))
            warning_view.setMaximumHeight(140)
            warning_view.setStyleSheet("background-color: #fff4e5;")
            layout.addWidget(warning_view)

        if schema_for_dialog is not None and category_summary:
            category_label = QtWidgets.QLabel(self.tr("Category Summary:"))
            category_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(category_label)
            category_view = QtWidgets.QPlainTextEdit()
            category_view.setReadOnly(True)
            category_view.setFont(
                QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            )
            category_view.setPlainText(format_category_summary(category_summary))
            category_view.setMaximumHeight(160)
            layout.addWidget(category_view)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        save_button = button_box.addButton(
            self.tr("Save CSV…"), QtWidgets.QDialogButtonBox.ActionRole
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

        save_button.clicked.connect(_save_csv)
        layout.addWidget(button_box)

        dialog.resize(720, 520)
        dialog.exec_()
