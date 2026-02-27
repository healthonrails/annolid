from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.annotation.batch_relabel import (
    BatchRelabelResult,
    collect_label_counts,
    run_batch_relabel,
)


class BatchRelabelDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        initial_root: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Rename Shape Labels")
        self.resize(760, 420)

        root_default = (
            str(Path(initial_root).expanduser().resolve())
            if initial_root
            else str(Path.cwd())
        )

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        root_row = QtWidgets.QHBoxLayout()
        self.root_edit = QtWidgets.QLineEdit(root_default, self)
        browse_btn = QtWidgets.QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._browse_root)
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(browse_btn)
        form.addRow("Root Folder:", root_row)

        self.old_label_edit = QtWidgets.QLineEdit(self)
        self.new_label_edit = QtWidgets.QLineEdit(self)
        form.addRow("From Label:", self.old_label_edit)
        form.addRow("To Label:", self.new_label_edit)

        self.include_json_check = QtWidgets.QCheckBox("Update LabelMe JSON files")
        self.include_json_check.setChecked(True)
        self.include_store_check = QtWidgets.QCheckBox(
            "Update AnnotationStore NDJSON files"
        )
        self.include_store_check.setChecked(True)
        form.addRow(self.include_json_check)
        form.addRow(self.include_store_check)
        layout.addLayout(form)

        btn_row = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview", self)
        self.apply_btn = QtWidgets.QPushButton("Apply Rename", self)
        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.reject)
        self.preview_btn.clicked.connect(self._on_preview)
        self.apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.apply_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.results_box = QtWidgets.QPlainTextEdit(self)
        self.results_box.setReadOnly(True)
        layout.addWidget(self.results_box, 1)

    def _browse_root(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select annotation root folder",
            self.root_edit.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.root_edit.setText(selected)

    def _validate_inputs(self) -> tuple[Path, str, str] | None:
        root = Path(self.root_edit.text().strip() or ".").expanduser()
        old = self.old_label_edit.text().strip()
        new = self.new_label_edit.text().strip()
        if not root.exists():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Folder", f"Folder not found:\n{root}"
            )
            return None
        if not old:
            QtWidgets.QMessageBox.warning(
                self, "Missing Label", "Please provide a source label."
            )
            return None
        if not new:
            QtWidgets.QMessageBox.warning(
                self, "Missing Label", "Please provide a target label."
            )
            return None
        if old == new:
            QtWidgets.QMessageBox.warning(
                self, "No-op", "Source and target labels are identical."
            )
            return None
        if (
            not self.include_json_check.isChecked()
            and not self.include_store_check.isChecked()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing Selected",
                "Enable at least one target: LabelMe JSON or AnnotationStore NDJSON.",
            )
            return None
        return root, old, new

    @staticmethod
    def _format_summary(summary: BatchRelabelResult) -> str:
        return (
            f"Root: {summary.root}\n"
            f"Labels: '{summary.old_label}' -> '{summary.new_label}'\n"
            f"Mode: {'Preview' if summary.dry_run else 'Apply'}\n\n"
            f"Shapes renamed: {summary.shapes_renamed}\n"
            f"JSON files scanned/updated: {summary.json_files_scanned}/{summary.json_files_updated}\n"
            f"Store files scanned/updated: {summary.store_files_scanned}/{summary.store_files_updated}\n"
            f"Store records updated: {summary.records_updated}\n"
        )

    @staticmethod
    def _format_histogram(
        counts: dict[str, int], *, title: str, top_k: int = 10
    ) -> str:
        if not counts:
            return f"{title}\n  (no labels found)\n"
        rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
            : max(1, int(top_k))
        ]
        vmax = max(v for _, v in rows) or 1
        lines = [title]
        for label, value in rows:
            bar_len = max(1, int(round((value / vmax) * 24)))
            lines.append(f"  {label:20s} {value:6d}  {'#' * bar_len}")
        return "\n".join(lines) + "\n"

    def _run(self, *, dry_run: bool) -> BatchRelabelResult | None:
        validated = self._validate_inputs()
        if validated is None:
            return None
        root, old, new = validated

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            before_counts = collect_label_counts(
                root=root,
                include_json_files=self.include_json_check.isChecked(),
                include_annotation_stores=self.include_store_check.isChecked(),
            )
            # keep UX responsive for medium datasets without building a full worker pipeline yet
            summary = run_batch_relabel(
                root=root,
                old_label=old,
                new_label=new,
                include_json_files=self.include_json_check.isChecked(),
                include_annotation_stores=self.include_store_check.isChecked(),
                dry_run=dry_run,
            )
            if dry_run:
                after_counts = dict(before_counts)
                moved = int(after_counts.get(old, 0))
                after_counts[new] = int(after_counts.get(new, 0)) + moved
                after_counts[old] = 0
                report = (
                    self._format_summary(summary)
                    + "\n"
                    + "Focus labels (before -> after):\n"
                    + f"  {old}: {int(before_counts.get(old, 0))} -> {int(after_counts.get(old, 0))}\n"
                    + f"  {new}: {int(before_counts.get(new, 0))} -> {int(after_counts.get(new, 0))}\n\n"
                    + self._format_histogram(before_counts, title="Top Labels (Before)")
                    + "\n"
                    + self._format_histogram(
                        after_counts, title="Top Labels (Projected After)"
                    )
                )
            else:
                after_counts = collect_label_counts(
                    root=root,
                    include_json_files=self.include_json_check.isChecked(),
                    include_annotation_stores=self.include_store_check.isChecked(),
                )
                report = (
                    self._format_summary(summary)
                    + "\n"
                    + "Focus labels (after apply):\n"
                    + f"  {old}: {int(after_counts.get(old, 0))}\n"
                    + f"  {new}: {int(after_counts.get(new, 0))}\n\n"
                    + self._format_histogram(
                        after_counts, title="Top Labels (After Apply)"
                    )
                )
            self.results_box.setPlainText(report)
            return summary
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Batch Rename Failed", str(exc))
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_preview(self) -> None:
        self._run(dry_run=True)

    def _on_apply(self) -> None:
        preview = self._run(dry_run=True)
        if preview is None:
            return
        if preview.shapes_renamed <= 0:
            QtWidgets.QMessageBox.information(
                self, "No Changes", "No matching labels found."
            )
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Confirm Batch Rename",
            (
                f"Rename {preview.shapes_renamed} shapes from "
                f"'{preview.old_label}' to '{preview.new_label}'?"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        applied = self._run(dry_run=False)
        if applied is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Batch Rename Complete",
                f"Updated {applied.shapes_renamed} shapes.",
            )
