from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.cursor_utils import set_widget_busy_cursor
from annolid.postprocessing import (
    GovernorPolicy,
    IdentityGovernorResult,
    run_identity_governor,
)
from annolid.utils.logger import logger


def _default_policy_template() -> dict[str, Any]:
    return {
        "metric_aliases": {
            "in_left": "zone.inside.left_zone",
            "in_right": "zone.inside.right_zone",
            "nearest": "distance.nearest",
            "area_px": "area",
        },
        "rules": [
            {
                "name": "alpha_when_right",
                "assign_label": "alpha",
                "conditions": [
                    {"metric": "in_right", "op": "eq", "value": True},
                    {"metric": "area_px", "op": "gte", "value": 80.0},
                ],
                "min_streak_frames": 2,
                "priority": 10,
            },
            {
                "name": "beta_when_left",
                "assign_label": "beta",
                "conditions": [
                    {"metric": "in_left", "op": "eq", "value": True},
                    {"metric": "area_px", "op": "gte", "value": 80.0},
                ],
                "min_streak_frames": 2,
                "priority": 10,
            },
        ],
        "ambiguity_conditions": [{"metric": "nearest", "op": "lte", "value": 5.0}],
        "max_backtrack_frames": 500,
        "max_forward_gap_frames": 1,
        "min_correction_span_frames": 1,
        "canonical_track_ids": {"alpha": "1", "beta": "2"},
    }


def _two_subject_zone_distance_template() -> dict[str, Any]:
    return {
        "metric_aliases": {
            "in_home": "zone.inside.home_zone",
            "in_target": "zone.inside.target_zone",
            "nearest": "distance.nearest",
            "area_px": "area",
        },
        "rules": [
            {
                "name": "subject_a_in_target_zone",
                "assign_label": "subject_a",
                "conditions": [
                    {"metric": "in_target", "op": "eq", "value": True},
                    {"metric": "area_px", "op": "gte", "value": 80},
                ],
                "min_streak_frames": 2,
                "priority": 10,
            },
            {
                "name": "subject_b_in_home_zone",
                "assign_label": "subject_b",
                "conditions": [
                    {"metric": "in_home", "op": "eq", "value": True},
                    {"metric": "area_px", "op": "gte", "value": 80},
                ],
                "min_streak_frames": 2,
                "priority": 10,
            },
        ],
        "ambiguity_conditions": [{"metric": "nearest", "op": "lte", "value": 6.0}],
        "max_backtrack_frames": 400,
        "max_forward_gap_frames": 1,
        "min_correction_span_frames": 1,
        "canonical_track_ids": {"subject_a": "1", "subject_b": "2"},
    }


def _three_vole_social_template() -> dict[str, Any]:
    return {
        "metric_aliases": {
            "in_left_social": "zone.inside.left_social_zone",
            "in_right_social": "zone.inside.right_social_zone",
            "near_stim_a": "distance.to_track.2",
            "near_stim_b": "distance.to_track.3",
            "nearest": "distance.nearest",
        },
        "interesting_labels": ["rover", "stim_a", "stim_b"],
        "rules": [
            {
                "name": "rover_near_stim_a",
                "assign_label": "rover",
                "conditions": [
                    {"metric": "in_left_social", "op": "eq", "value": True},
                    {"metric": "near_stim_a", "op": "lte", "value": 50.0},
                ],
                "min_streak_frames": 2,
                "priority": 12,
            },
            {
                "name": "stim_a_when_not_rover_side",
                "assign_label": "stim_a",
                "conditions": [
                    {"metric": "in_left_social", "op": "eq", "value": False},
                    {"metric": "near_stim_b", "op": "gte", "value": 20.0},
                ],
                "priority": 8,
            },
            {
                "name": "stim_b_when_not_rover_side",
                "assign_label": "stim_b",
                "conditions": [
                    {"metric": "in_right_social", "op": "eq", "value": True},
                    {"metric": "near_stim_a", "op": "gte", "value": 20.0},
                ],
                "priority": 8,
            },
        ],
        "ambiguity_conditions": [{"metric": "nearest", "op": "lte", "value": 7.0}],
        "max_backtrack_frames": 500,
        "max_forward_gap_frames": 1,
        "min_correction_span_frames": 2,
        "canonical_track_ids": {"rover": "1", "stim_a": "2", "stim_b": "3"},
    }


def _distance_only_template() -> dict[str, Any]:
    return {
        "rules": [
            {
                "name": "subject_a_far_from_track2",
                "assign_label": "subject_a",
                "conditions": [
                    {"metric": "distance.to_track.2", "op": "gte", "value": 40}
                ],
            },
            {
                "name": "subject_b_far_from_track1",
                "assign_label": "subject_b",
                "conditions": [
                    {"metric": "distance.to_track.1", "op": "gte", "value": 40}
                ],
            },
        ],
    }


def _policy_snippets() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "Generic identity template",
            "Balanced default template for two identities with zone + area + ambiguity constraints.",
            _default_policy_template(),
        ),
        (
            "2-subject arena (zone + distance)",
            "For two tracked subjects in home/target zones; tune zone names and thresholds.",
            _two_subject_zone_distance_template(),
        ),
        (
            "3-vole social assay",
            "Starter for rover/stim_a/stim_b social tracking; tune track IDs, zone names, and cutoffs.",
            _three_vole_social_template(),
        ),
        (
            "Distance-only fallback",
            "No zone dependency; useful when only relative spacing is reliable.",
            _distance_only_template(),
        ),
    ]


class IdentityGovernorDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        initial_annotation_dir: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Identity Governor")
        self.resize(1040, 760)
        self._last_result: IdentityGovernorResult | None = None
        self._snippets = _policy_snippets()
        self._build_ui(initial_annotation_dir=initial_annotation_dir)

    def _build_ui(self, *, initial_annotation_dir: str | Path | None) -> None:
        root_default = (
            str(Path(initial_annotation_dir).expanduser().resolve())
            if initial_annotation_dir
            else str(Path.cwd())
        )

        layout = QtWidgets.QVBoxLayout(self)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        layout.addWidget(split, 1)

        top = QtWidgets.QWidget(split)
        top_layout = QtWidgets.QVBoxLayout(top)

        intro = QtWidgets.QLabel(
            "Policy-based identity repair for tracked instances "
            "(distance, zones, area, and ambiguity-aware backtracking).",
            top,
        )
        intro.setWordWrap(True)
        top_layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        top_layout.addLayout(form)

        annotation_row = QtWidgets.QHBoxLayout()
        self.annotation_dir_edit = QtWidgets.QLineEdit(root_default, top)
        browse_annotation_btn = QtWidgets.QPushButton("Browse...", top)
        browse_annotation_btn.clicked.connect(self._browse_annotation_dir)
        annotation_row.addWidget(self.annotation_dir_edit, 1)
        annotation_row.addWidget(browse_annotation_btn)
        form.addRow("Annotation Folder:", annotation_row)

        zone_row = QtWidgets.QHBoxLayout()
        self.zone_file_edit = QtWidgets.QLineEdit(top)
        self.zone_file_edit.setPlaceholderText("(Optional) path to *_zones.json")
        browse_zone_btn = QtWidgets.QPushButton("Browse...", top)
        browse_zone_btn.clicked.connect(self._browse_zone_file)
        zone_row.addWidget(self.zone_file_edit, 1)
        zone_row.addWidget(browse_zone_btn)
        form.addRow("Zone JSON:", zone_row)

        report_row = QtWidgets.QHBoxLayout()
        self.report_path_edit = QtWidgets.QLineEdit(top)
        self.report_path_edit.setPlaceholderText(
            "(Optional) default: <annotation_dir>/identity_governor_report.json"
        )
        browse_report_btn = QtWidgets.QPushButton("Browse...", top)
        browse_report_btn.clicked.connect(self._browse_report_path)
        report_row.addWidget(self.report_path_edit, 1)
        report_row.addWidget(browse_report_btn)
        form.addRow("Report Path:", report_row)

        policy_actions = QtWidgets.QHBoxLayout()
        self.template_combo = QtWidgets.QComboBox(top)
        self.template_combo.addItems([snippet[0] for snippet in self._snippets])
        self.template_combo.currentIndexChanged.connect(self._refresh_template_hint)
        self.insert_template_btn = QtWidgets.QPushButton("Insert Template", top)
        self.insert_template_btn.clicked.connect(self._insert_selected_template)
        self.load_policy_btn = QtWidgets.QPushButton("Load Policy JSON...", top)
        self.load_policy_btn.clicked.connect(self._load_policy_json)
        self.save_policy_btn = QtWidgets.QPushButton("Save Policy JSON...", top)
        self.save_policy_btn.clicked.connect(self._save_policy_json)
        self.format_policy_btn = QtWidgets.QPushButton("Format JSON", top)
        self.format_policy_btn.clicked.connect(self._format_policy_json)
        policy_actions.addWidget(self.template_combo, 1)
        policy_actions.addWidget(self.insert_template_btn)
        policy_actions.addWidget(self.load_policy_btn)
        policy_actions.addWidget(self.save_policy_btn)
        policy_actions.addWidget(self.format_policy_btn)
        top_layout.addLayout(policy_actions)

        self.template_hint_label = QtWidgets.QLabel(top)
        self.template_hint_label.setWordWrap(True)
        self.template_hint_label.setStyleSheet("color: #4b5563;")
        top_layout.addWidget(self.template_hint_label)

        self.policy_edit = QtWidgets.QPlainTextEdit(top)
        self.policy_edit.setPlaceholderText("Paste policy JSON here...")
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.policy_edit.setFont(font)
        self.policy_edit.setPlainText(
            json.dumps(_default_policy_template(), ensure_ascii=False, indent=2)
        )
        self._refresh_template_hint()
        top_layout.addWidget(self.policy_edit, 1)

        run_bar = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview (Dry-Run)", top)
        self.preview_btn.clicked.connect(lambda: self._run(apply_changes=False))
        self.apply_btn = QtWidgets.QPushButton("Apply Fixes", top)
        self.apply_btn.clicked.connect(lambda: self._run(apply_changes=True))
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #b43f3f; color: white; font-weight: 600; }"
        )
        run_bar.addWidget(self.preview_btn)
        run_bar.addWidget(self.apply_btn)
        run_bar.addStretch(1)
        top_layout.addLayout(run_bar)

        bottom = QtWidgets.QWidget(split)
        bottom_layout = QtWidgets.QVBoxLayout(bottom)

        self.summary_label = QtWidgets.QLabel("No run yet.", bottom)
        bottom_layout.addWidget(self.summary_label)

        self.corrections_table = QtWidgets.QTableWidget(0, 8, bottom)
        self.corrections_table.setHorizontalHeaderLabels(
            [
                "Track ID",
                "From",
                "To",
                "Frame Start",
                "Frame End",
                "Rule",
                "Evidence Start",
                "Evidence End",
            ]
        )
        self.corrections_table.horizontalHeader().setStretchLastSection(True)
        self.corrections_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.corrections_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        bottom_layout.addWidget(self.corrections_table, 1)

        footer = QtWidgets.QHBoxLayout()
        self.open_report_btn = QtWidgets.QPushButton("Open Report", bottom)
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.clicked.connect(self._open_report)
        close_btn = QtWidgets.QPushButton("Close", bottom)
        close_btn.clicked.connect(self.reject)
        footer.addWidget(self.open_report_btn)
        footer.addStretch(1)
        footer.addWidget(close_btn)
        bottom_layout.addLayout(footer)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

    def _template_payload(self, index: int) -> dict[str, Any]:
        if not self._snippets:
            return _default_policy_template()
        safe_index = max(0, min(int(index), len(self._snippets) - 1))
        return dict(self._snippets[safe_index][2])

    def _refresh_template_hint(self) -> None:
        if not self._snippets:
            self.template_hint_label.setText("")
            return
        safe_index = max(
            0, min(self.template_combo.currentIndex(), len(self._snippets) - 1)
        )
        self.template_hint_label.setText(self._snippets[safe_index][1])

    def _insert_selected_template(self) -> None:
        payload = self._template_payload(self.template_combo.currentIndex())
        self.policy_edit.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def _browse_annotation_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select annotation folder",
            self.annotation_dir_edit.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.annotation_dir_edit.setText(selected)

    def _browse_zone_file(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select zone JSON",
            self.zone_file_edit.text().strip()
            or self.annotation_dir_edit.text().strip(),
            "JSON Files (*.json)",
        )
        if selected:
            self.zone_file_edit.setText(selected)

    def _browse_report_path(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save report as",
            self.report_path_edit.text().strip()
            or str(
                Path(self.annotation_dir_edit.text().strip() or ".")
                / "identity_governor_report.json"
            ),
            "JSON Files (*.json)",
        )
        if selected:
            self.report_path_edit.setText(selected)

    def _load_policy_json(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load policy JSON",
            self.annotation_dir_edit.text().strip() or str(Path.cwd()),
            "JSON Files (*.json)",
        )
        if not selected:
            return
        try:
            text = Path(selected).read_text(encoding="utf-8")
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError("Policy file must contain a JSON object.")
            self.policy_edit.setPlainText(
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Invalid Policy JSON", f"Failed to load policy:\n{exc}"
            )

    def _save_policy_json(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save policy JSON",
            self.annotation_dir_edit.text().strip() or str(Path.cwd()),
            "JSON Files (*.json)",
        )
        if not selected:
            return
        try:
            payload = self._parse_policy()
            Path(selected).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save Failed", f"Could not save policy JSON:\n{exc}"
            )

    def _format_policy_json(self) -> None:
        try:
            payload = self._parse_policy()
            self.policy_edit.setPlainText(
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Invalid JSON", f"Policy JSON is invalid:\n{exc}"
            )

    def _parse_policy(self) -> dict[str, Any]:
        raw = self.policy_edit.toPlainText().strip()
        if not raw:
            raise ValueError("Policy JSON cannot be empty.")
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"JSON parse error: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Policy must be a JSON object.")
        GovernorPolicy.from_dict(payload)
        return dict(payload)

    def _validated_inputs(
        self,
    ) -> tuple[Path, dict[str, Any], str | None, str | None] | None:
        annotation_dir = Path(
            self.annotation_dir_edit.text().strip() or "."
        ).expanduser()
        if not annotation_dir.exists() or not annotation_dir.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Folder",
                f"Annotation folder not found:\n{annotation_dir}",
            )
            return None
        if not list(annotation_dir.glob("*.json")):
            QtWidgets.QMessageBox.warning(
                self,
                "No JSON Frames",
                "The selected annotation folder has no frame JSON files.",
            )
            return None

        try:
            policy = self._parse_policy()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Policy", f"Could not parse policy JSON:\n{exc}"
            )
            return None

        zone_path = self.zone_file_edit.text().strip()
        if zone_path:
            zone_file = Path(zone_path).expanduser()
            if not zone_file.exists() or not zone_file.is_file():
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Zone File", f"Zone file not found:\n{zone_file}"
                )
                return None
            zone_value = str(zone_file)
        else:
            zone_value = None

        report_text = self.report_path_edit.text().strip()
        report_value = str(Path(report_text).expanduser()) if report_text else None
        return annotation_dir, policy, zone_value, report_value

    def _run(self, *, apply_changes: bool) -> None:
        validated = self._validated_inputs()
        if validated is None:
            return
        annotation_dir, policy, zone_file, report_path = validated

        if apply_changes:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Confirm Apply",
                (
                    "This will rewrite annotation JSON files for corrected spans.\n\n"
                    "Run apply mode now?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

        set_widget_busy_cursor(self, True)
        try:
            result = run_identity_governor(
                annotation_dir=annotation_dir,
                policy=policy,
                zone_file=zone_file,
                apply_changes=apply_changes,
                report_path=report_path,
            )
            self._render_result(result)
        except Exception as exc:
            logger.error("Identity governor failed: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Identity Governor Error",
                f"Failed to run identity governor:\n{exc}",
            )
        finally:
            set_widget_busy_cursor(self, False)

    def _render_result(self, result: IdentityGovernorResult) -> None:
        self._last_result = result
        corrections = list(result.proposed_corrections)
        self.corrections_table.setRowCount(len(corrections))
        for row_idx, correction in enumerate(corrections):
            values = [
                correction.track_id,
                correction.observed_label,
                correction.corrected_label,
                str(correction.frame_start),
                str(correction.frame_end),
                correction.rule_name,
                str(correction.rule_frame_start),
                str(correction.rule_frame_end),
            ]
            for col_idx, value in enumerate(values):
                self.corrections_table.setItem(
                    row_idx, col_idx, QtWidgets.QTableWidgetItem(str(value))
                )
        self.corrections_table.resizeColumnsToContents()

        mode_text = "Apply" if not result.dry_run else "Preview"
        self.summary_label.setText(
            (
                f"{mode_text} complete. "
                f"Corrections: {len(corrections)} | "
                f"Scanned files: {result.scanned_files} | "
                f"Scanned observations: {result.scanned_observations} | "
                f"Updated files: {result.updated_files} | "
                f"Updated shapes: {result.updated_shapes}"
            )
        )
        self.open_report_btn.setEnabled(bool(result.report_path))
        self.report_path_edit.setText(str(result.report_path))

        if result.dry_run:
            QtWidgets.QMessageBox.information(
                self,
                "Preview Complete",
                (
                    f"Preview complete.\n\n"
                    f"Proposed corrections: {len(corrections)}\n"
                    f"Report: {result.report_path}"
                ),
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Apply Complete",
                (
                    f"Identity repair complete.\n\n"
                    f"Updated files: {result.updated_files}\n"
                    f"Updated shapes: {result.updated_shapes}\n"
                    f"Report: {result.report_path}"
                ),
            )

    def _open_report(self) -> None:
        if self._last_result is None:
            return
        report_path = Path(self._last_result.report_path)
        if not report_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Missing Report", f"Report file not found:\n{report_path}"
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(report_path)))
