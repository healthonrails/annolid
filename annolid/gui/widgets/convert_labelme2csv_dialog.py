from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from annolid.annotation.labelme2csv import convert_json_to_csv
from annolid.postprocessing.zone_occupancy_policy import (
    apply_zone_occupancy_policy_file,
)
from annolid.postprocessing.zone_schema import load_zone_shapes
from annolid.utils.annotation_store import load_labelme_json


class LabelmeJsonToCsvDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.json_folder_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Export Tracking CSV(s)")
        self.setMinimumWidth(680)

        root = QVBoxLayout(self)

        intro = QLabel(
            "Export tracking outputs from LabelMe JSON predictions. "
            "Optionally apply zone occupancy policy rules to create a corrected tracked CSV.",
            self,
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        root.addWidget(self._build_source_group())
        root.addWidget(self._build_output_group())
        root.addWidget(self._build_zone_policy_group())
        root.addLayout(self._build_buttons())

        self._refresh_ui_state()

    def _build_source_group(self) -> QGroupBox:
        box = QGroupBox("Source", self)
        layout = QFormLayout(box)

        folder_row = QHBoxLayout()
        self.txt_selected_folder = QLineEdit(self)
        self.txt_selected_folder.setPlaceholderText("Select JSON folder")
        self.txt_selected_folder.setReadOnly(True)
        folder_row.addWidget(self.txt_selected_folder)

        self.btn_select_folder = QPushButton("Browse…", self)
        self.btn_select_folder.clicked.connect(self.select_folder)
        folder_row.addWidget(self.btn_select_folder)
        layout.addRow("JSON folder", folder_row)

        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Outputs", self)
        layout = QVBoxLayout(box)

        self.generate_tracking_checkbox = QCheckBox("Generate *_tracking.csv", self)
        self.generate_tracking_checkbox.setChecked(True)
        self.generate_tracking_checkbox.toggled.connect(self._refresh_ui_state)
        layout.addWidget(self.generate_tracking_checkbox)

        self.generate_tracked_checkbox = QCheckBox("Generate *_tracked.csv", self)
        self.generate_tracked_checkbox.setChecked(True)
        self.generate_tracked_checkbox.toggled.connect(self._refresh_ui_state)
        layout.addWidget(self.generate_tracked_checkbox)

        self.force_rewrite_tracking_checkbox = QCheckBox(
            "Force rewrite *_tracking.csv even when all frames already exist",
            self,
        )
        self.force_rewrite_tracking_checkbox.setChecked(False)
        layout.addWidget(self.force_rewrite_tracking_checkbox)

        return box

    def _build_zone_policy_group(self) -> QGroupBox:
        box = QGroupBox("Optional Zone Policy", self)
        layout = QFormLayout(box)

        self.apply_zone_policy_checkbox = QCheckBox(
            "Apply zone occupancy policy rules to tracked CSV", self
        )
        self.apply_zone_policy_checkbox.setChecked(False)
        self.apply_zone_policy_checkbox.toggled.connect(self._refresh_ui_state)
        layout.addRow(self.apply_zone_policy_checkbox)

        policy_row = QHBoxLayout()
        self.zone_policy_path_edit = QLineEdit(self)
        self.zone_policy_path_edit.setPlaceholderText("Select zone policy JSON file")
        policy_row.addWidget(self.zone_policy_path_edit)

        self.zone_policy_browse_btn = QPushButton("Browse…", self)
        self.zone_policy_browse_btn.clicked.connect(self._choose_zone_policy_file)
        policy_row.addWidget(self.zone_policy_browse_btn)
        layout.addRow("Policy JSON", policy_row)

        zone_row = QHBoxLayout()
        self.zone_file_path_edit = QLineEdit(self)
        self.zone_file_path_edit.setPlaceholderText(
            "Optional zone JSON file (auto-detected if empty)"
        )
        zone_row.addWidget(self.zone_file_path_edit)

        self.zone_file_browse_btn = QPushButton("Browse…", self)
        self.zone_file_browse_btn.clicked.connect(self._choose_zone_file)
        zone_row.addWidget(self.zone_file_browse_btn)
        layout.addRow("Zone JSON", zone_row)

        self.export_policy_audit_checkbox = QCheckBox(
            "Export policy audit CSV (*_tracked_zone_corrected_audit.csv)", self
        )
        self.export_policy_audit_checkbox.setChecked(True)
        layout.addRow(self.export_policy_audit_checkbox)

        hint = QLabel(
            "Policy rules are applied to the tracked table only. "
            "JSON annotations and raw centroid extraction are not modified.",
            self,
        )
        hint.setWordWrap(True)
        layout.addRow(hint)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)

        self.btn_run = QPushButton("Export", self)
        self.btn_run.clicked.connect(self.run_conversion)
        row.addWidget(self.btn_run)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        return row

    def _refresh_ui_state(self) -> None:
        has_folder = bool(self.json_folder_path)
        has_output = (
            self.generate_tracking_checkbox.isChecked()
            or self.generate_tracked_checkbox.isChecked()
        )
        tracked_enabled = self.generate_tracked_checkbox.isChecked()
        policy_enabled = self.apply_zone_policy_checkbox.isChecked() and tracked_enabled

        self.btn_run.setEnabled(has_folder and has_output)
        self.apply_zone_policy_checkbox.setEnabled(tracked_enabled)
        self.zone_policy_path_edit.setEnabled(policy_enabled)
        self.zone_policy_browse_btn.setEnabled(policy_enabled)
        self.zone_file_path_edit.setEnabled(policy_enabled)
        self.zone_file_browse_btn.setEnabled(policy_enabled)
        self.export_policy_audit_checkbox.setEnabled(policy_enabled)

    @staticmethod
    def _default_tracking_csv_path(json_folder_path: str) -> Path:
        return Path(f"{json_folder_path}_tracking.csv")

    @staticmethod
    def _default_tracked_csv_path(json_folder_path: str) -> Path:
        return Path(f"{json_folder_path}_tracked.csv")

    @staticmethod
    def _default_zone_corrected_tracked_csv_path(json_folder_path: str) -> Path:
        return Path(f"{json_folder_path}_tracked_zone_corrected.csv")

    def _suggest_zone_file(self, json_folder_path: str) -> Path | None:
        folder = Path(json_folder_path)
        candidate = folder.parent / f"{folder.name}_zones.json"
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    def select_folder(self) -> None:
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setNameFilter("JSON files (*.json)")
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setWindowTitle("Select JSON folder")
        if not folder_dialog.exec_():
            return

        folder_path = folder_dialog.selectedFiles()[0]
        self.json_folder_path = folder_path
        self.txt_selected_folder.setText(folder_path)

        suggested_zone = self._suggest_zone_file(folder_path)
        if suggested_zone is not None and not self.zone_file_path_edit.text().strip():
            self.zone_file_path_edit.setText(str(suggested_zone))

        self._refresh_ui_state()

    def _choose_zone_policy_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Zone Occupancy Policy JSON",
            str(Path(self.json_folder_path).parent) if self.json_folder_path else "",
            "JSON files (*.json)",
        )
        if path:
            self.zone_policy_path_edit.setText(path)

    def _choose_zone_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Zone JSON",
            str(Path(self.json_folder_path).parent) if self.json_folder_path else "",
            "JSON files (*.json)",
        )
        if path:
            self.zone_file_path_edit.setText(path)

    def _load_zone_specs(
        self, explicit_zone_path: str | None
    ) -> tuple[list, str | None]:
        if explicit_zone_path:
            zone_path = Path(explicit_zone_path).expanduser()
        elif self.json_folder_path:
            zone_path = self._suggest_zone_file(self.json_folder_path)
            if zone_path is None:
                return [], None
        else:
            return [], None

        if zone_path is None or not zone_path.exists() or not zone_path.is_file():
            return [], None

        payload = load_labelme_json(zone_path)
        specs = load_zone_shapes(payload)
        return specs, str(zone_path)

    def _apply_zone_policy_to_tracked_csv(
        self, tracked_csv_path: Path
    ) -> tuple[Path, Path | None]:
        policy_path_raw = self.zone_policy_path_edit.text().strip()
        if not policy_path_raw:
            raise ValueError(
                "Zone policy is enabled but no policy JSON file was selected."
            )

        zone_file_raw = self.zone_file_path_edit.text().strip() or None
        zone_specs, resolved_zone_path = self._load_zone_specs(zone_file_raw)

        tracked_df = pd.read_csv(tracked_csv_path)
        result = apply_zone_occupancy_policy_file(
            tracked_df,
            zone_specs,
            policy_path_raw,
        )

        corrected_path = self._default_zone_corrected_tracked_csv_path(
            self.json_folder_path
        )
        result.dataframe.to_csv(corrected_path, index=False)

        audit_path: Path | None = None
        if self.export_policy_audit_checkbox.isChecked():
            audit_path = corrected_path.with_name(
                f"{corrected_path.stem}_audit{corrected_path.suffix}"
            )
            result.audit.to_csv(audit_path, index=False)

        if not zone_specs:
            zone_message = (
                "No zone shapes were loaded. Policy was applied only to existing zone columns "
                "already present in tracked CSV."
            )
            if resolved_zone_path:
                zone_message = (
                    f"Zone JSON loaded from {resolved_zone_path} but no valid zone shapes were found. "
                    "Policy was applied only to existing zone columns in tracked CSV."
                )
            QMessageBox.information(self, "Zone Policy", zone_message)

        return corrected_path, audit_path

    def run_conversion(self) -> None:
        if not (
            self.generate_tracking_checkbox.isChecked()
            or self.generate_tracked_checkbox.isChecked()
        ):
            QMessageBox.warning(
                self,
                "No Output Selected",
                "Select at least one output file type.",
            )
            return

        if not self.json_folder_path:
            QMessageBox.warning(self, "Missing Source", "Select a JSON folder first.")
            return

        if (
            self.apply_zone_policy_checkbox.isChecked()
            and self.generate_tracked_checkbox.isChecked()
            and not self.zone_policy_path_edit.text().strip()
        ):
            QMessageBox.warning(
                self,
                "Missing Zone Policy",
                "Select a zone policy JSON file or disable zone policy application.",
            )
            return

        try:
            tracking_csv_path = self._default_tracking_csv_path(self.json_folder_path)
            tracked_csv_path = self._default_tracked_csv_path(self.json_folder_path)

            convert_json_to_csv(
                self.json_folder_path,
                csv_file=str(tracking_csv_path),
                tracked_csv_file=(
                    str(tracked_csv_path)
                    if self.generate_tracked_checkbox.isChecked()
                    else None
                ),
                force_rewrite_tracking_csv=self.force_rewrite_tracking_checkbox.isChecked(),
                include_tracking_output=self.generate_tracking_checkbox.isChecked(),
            )

            corrected_path = None
            audit_path = None
            if (
                self.apply_zone_policy_checkbox.isChecked()
                and self.generate_tracked_checkbox.isChecked()
            ):
                corrected_path, audit_path = self._apply_zone_policy_to_tracked_csv(
                    tracked_csv_path
                )

            lines = ["Export completed successfully."]
            if self.generate_tracking_checkbox.isChecked():
                lines.append(f"- Tracking CSV: {tracking_csv_path}")
            if self.generate_tracked_checkbox.isChecked():
                lines.append(f"- Tracked CSV: {tracked_csv_path}")
            if corrected_path is not None:
                lines.append(f"- Zone-corrected tracked CSV: {corrected_path}")
            if audit_path is not None:
                lines.append(f"- Zone policy audit CSV: {audit_path}")

            QMessageBox.information(self, "Success", "\n".join(lines))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = LabelmeJsonToCsvDialog()
    dialog.show()
    sys.exit(app.exec_())
