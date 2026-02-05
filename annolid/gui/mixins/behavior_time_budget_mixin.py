from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from qtpy import QtGui, QtWidgets

from annolid.behavior.time_budget import (
    TimeBudgetComputationError,
    compute_time_budget,
    format_category_summary,
    format_time_budget_table,
    summarize_by_category,
    write_time_budget_csv,
)
from annolid.utils.logger import logger


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

        data_rows = []
        local_warnings = []
        for trial_time, recording_time, subject, behavior, event_label in rows:
            if recording_time is None:
                local_warnings.append(
                    self.tr(
                        "Skipping '%s' event for behavior '%s' because no timestamp could be determined."
                    )
                    % (event_label or "?", behavior or "?")
                )
                continue

            data_rows.append(
                {
                    "trial time": trial_time,
                    "recording time": recording_time,
                    "subject": (subject or "").strip(),
                    "behavior": (behavior or "").strip(),
                    "event": (event_label or "").strip(),
                }
            )

        if not data_rows:
            message = self.tr(
                "No timestamped events remain after filtering out rows without timing information."
            )
            if local_warnings:
                message += "\n\n" + self.tr("Warnings:\n") + "\n".join(local_warnings)
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Behavior Time Budget"),
                message,
            )
            return

        try:
            summary, compute_warnings = compute_time_budget(data_rows)
        except TimeBudgetComputationError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("Behavior Time Budget"),
                self.tr("Unable to compute the time budget:\n%s") % exc,
            )
            return

        warnings: List[str] = local_warnings + compute_warnings
        schema_for_dialog = self.project_schema
        category_summary: List[Tuple[str, float, int]] = []
        if schema_for_dialog is not None and summary:
            try:
                category_summary = summarize_by_category(summary, schema_for_dialog)
            except Exception as exc:
                logger.warning("Failed to summarize categories: %s", exc)
                category_summary = []

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

        report_text = format_time_budget_table(summary, schema=schema_for_dialog)

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
                write_time_budget_csv(summary, output_path, schema=schema_for_dialog)
                if schema_for_dialog is not None and category_summary:
                    category_path = output_path.with_name(
                        output_path.stem + "_categories" + output_path.suffix
                    )
                    with category_path.open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.writer(handle)
                        writer.writerow(["Category", "TotalSeconds", "Occurrences"])
                        for name, total, occurrences in category_summary:
                            writer.writerow([name, f"{total:.6f}", occurrences])
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
