import sys
import os
from qtpy import QtWidgets
from qtpy.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QDialog,
    QFormLayout,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_assay_profiles import available_assay_profiles
from annolid.data.videos import CV2Video
from pathlib import Path
from annolid.utils.files import find_manual_labeled_json_files


class TrackingAnalyzerDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Zone Analysis")
        self.setGeometry(100, 100, 520, 360)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        header = QLabel(
            "Export legacy place-preference CSV, generic zone metrics, profile-aware assay summaries, or social-zone summaries from the same saved zones."
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-weight: 600; color: #2d5b88;")
        layout.addWidget(header)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Enter video path")
        self.video_path_button = QPushButton("Browse...")
        self.video_path_button.clicked.connect(self.browse_video)

        self.zone_path_edit = QLineEdit()
        self.zone_path_edit.setPlaceholderText("(Optional)Enter zone JSON path")
        self.zone_path_button = QPushButton("Browse...")
        self.zone_path_button.clicked.connect(self.browse_zone)

        self.fps_edit = QLineEdit()
        self.fps_edit.setPlaceholderText("(Optional)Enter FPS (default is 30)")

        self.latency_reference_edit = QLineEdit()
        self.latency_reference_edit.setPlaceholderText(
            "(Optional)Enter latency reference frame; default is the first analyzed frame"
        )

        self.profile_combo = QComboBox()
        self._assay_profiles = available_assay_profiles()
        for profile in self._assay_profiles:
            self.profile_combo.addItem(
                profile.title,
                userData=profile.name,
            )
        self.profile_hint_label = QLabel(
            "Selected profile controls how the summary filters zones and labels the output."
        )
        self.profile_hint_label.setWordWrap(True)
        self.profile_hint_label.setStyleSheet("color: #5d6d7e;")
        self.profile_combo.currentIndexChanged.connect(self._update_profile_hint)

        form_layout.addRow(QLabel("Video Path:"), self.video_path_edit)
        form_layout.addRow(QLabel(""), self.video_path_button)
        form_layout.addRow(QLabel("Zone JSON Path:"), self.zone_path_edit)
        form_layout.addRow(QLabel(""), self.zone_path_button)
        form_layout.addRow(QLabel("FPS:"), self.fps_edit)
        form_layout.addRow(
            QLabel("Latency Reference Frame:"), self.latency_reference_edit
        )
        form_layout.addRow(QLabel("Assay Profile:"), self.profile_combo)
        form_layout.addRow(QLabel(""), self.profile_hint_label)

        self.legacy_export_button = QPushButton("Export Legacy CSV")
        self.legacy_export_button.setToolTip(
            "Export the historical place-preference CSV with one column per zone."
        )
        self.legacy_export_button.clicked.connect(self.export_legacy_csv)

        self.generic_export_button = QPushButton("Export Zone Metrics")
        self.generic_export_button.setToolTip(
            "Export per-zone occupancy, dwell time, entries, transitions, and barrier-adjacent time."
        )
        self.generic_export_button.clicked.connect(self.export_zone_metrics_csv)

        self.assay_summary_button = QPushButton("Export Assay Summary")
        self.assay_summary_button.setToolTip(
            "Export a Markdown report plus CSV metrics that explain zone coverage, phase rules, and computed metrics."
        )
        self.assay_summary_button.clicked.connect(self.export_assay_summary)

        self.social_summary_button = QPushButton("Export Social Summary")
        self.social_summary_button.setToolTip(
            "Export a Markdown report plus CSV metrics for social-zone approach latency, door proximity, and centroid-neighbor distance."
        )
        self.social_summary_button.clicked.connect(self.export_social_summary)

        layout.addLayout(form_layout)
        actions_row = QVBoxLayout()
        actions_row.addWidget(self.legacy_export_button)
        actions_row.addWidget(self.generic_export_button)
        actions_row.addWidget(self.assay_summary_button)
        actions_row.addWidget(self.social_summary_button)
        layout.addLayout(actions_row)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self._update_profile_hint()

    def browse_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov,)"
        )
        if filename:
            self.video_path_edit.setText(filename)
            # Try to extract FPS and find corresponding zone file
            fps, zone_file = self.extract_fps_and_find_zone_file(filename)
            if fps:
                self.fps_edit.setText(str(fps))
            if zone_file:
                self.zone_path_edit.setText(zone_file)

    def browse_zone(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Zone JSON File", "", "JSON Files (*.json)"
        )
        if filename:
            self.zone_path_edit.setText(filename)

    def _selected_assay_profile(self) -> str:
        return self.profile_combo.currentData() or self.profile_combo.currentText()

    def _update_profile_hint(self, *args) -> None:
        _ = args
        profile_name = self._selected_assay_profile()
        profile = next(
            (item for item in self._assay_profiles if item.name == profile_name),
            None,
        )
        if profile is None:
            self.profile_hint_label.setText(
                "Selected profile controls how the summary filters zones and labels the output."
            )
            return
        self.profile_hint_label.setText(f"{profile.title}: {profile.description}")

    def _build_analyzer(self, assay_profile=None):
        video_path = self.video_path_edit.text().strip()
        zone_path = self.zone_path_edit.text().strip() or None
        fps_text = self.fps_edit.text().strip()
        fps = float(fps_text) if fps_text else None
        selected_profile = (
            assay_profile
            if assay_profile is not None
            else self._selected_assay_profile()
        )
        if not video_path:
            raise ValueError("A video path is required.")
        return TrackingResultsAnalyzer(
            video_path,
            zone_file=zone_path,
            fps=fps,
            assay_profile=selected_profile,
        )

    def _latency_reference_frame(self):
        text = self.latency_reference_edit.text().strip()
        if not text:
            return None
        try:
            return int(float(text))
        except Exception:
            raise ValueError("Latency reference frame must be an integer frame number.")

    def extract_fps_and_find_zone_file(self, video_path):
        fps = None
        zone_file = None
        if os.path.exists(video_path):
            fps = CV2Video(video_path).get_fps()
            # Search for corresponding zone file
            video_dir = Path(video_path).with_suffix("")
            json_files = find_manual_labeled_json_files(video_dir)
            # assume the zone shapes are in the first frame
            if json_files:
                zone_file = video_dir / sorted(json_files)[0]
        return fps, str(zone_file)

    def _run_export(
        self,
        export_method_name: str,
        *,
        status_prefix: str,
        assay_profile=None,
        latency_reference_frame=None,
    ) -> str | None:
        try:
            analyzer = self._build_analyzer(assay_profile=assay_profile)
            export_method = getattr(analyzer, export_method_name)
            if latency_reference_frame is None:
                output_path = export_method()
            else:
                output_path = export_method(
                    latency_reference_frame=latency_reference_frame
                )
        except Exception as exc:
            self.status_label.setText(f"{status_prefix} failed: {exc}")
            QtWidgets.QMessageBox.warning(self, f"{status_prefix} failed", str(exc))
            return None
        self.status_label.setText(f"{status_prefix} saved to {output_path}")
        return output_path

    def export_legacy_csv(self):
        self._run_export(
            "save_all_instances_zone_time_to_csv",
            status_prefix="Legacy place-preference export",
            assay_profile="generic",
        )

    def export_zone_metrics_csv(self):
        self._run_export(
            "save_zone_metrics_to_csv",
            status_prefix="Generic zone metrics export",
            assay_profile=self._selected_assay_profile(),
        )

    def export_assay_summary(self):
        self._run_export(
            "save_assay_summary_report",
            status_prefix="Assay summary export",
            assay_profile=self._selected_assay_profile(),
        )

    def export_social_summary(self):
        self._run_export(
            "save_social_summary_report",
            status_prefix="Social summary export",
            assay_profile=self._selected_assay_profile(),
            latency_reference_frame=self._latency_reference_frame(),
        )

    def run_analysis_without_gui(self, video_path, zone_path=None, fps=None):
        self.video_path_edit.setText(video_path)
        if zone_path is None:
            fps, zone_path = self.extract_fps_and_find_zone_file(video_path)
        if zone_path:
            self.zone_path_edit.setText(zone_path)
        if fps:
            self.fps_edit.setText(str(fps))
        self.export_zone_metrics_csv()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = TrackingAnalyzerDialog()
    dialog.exec_()
    sys.exit(app.exec_())
