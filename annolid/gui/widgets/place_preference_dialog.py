from __future__ import annotations

import os
import sys
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QApplication, QFileDialog, QDialog

from annolid.data.videos import CV2Video
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_assay_profiles import available_assay_profiles
from annolid.utils.files import find_manual_labeled_json_files


class TrackingAnalyzerDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zone Analysis")
        self.resize(640, 520)

        self._assay_profiles = available_assay_profiles()
        self._mode_specs = self._build_mode_specs()

        self._build_ui()
        self._update_profile_hint()
        self._update_mode_hint()

    @staticmethod
    def _build_mode_specs() -> list[dict[str, object]]:
        return [
            {
                "key": "legacy_csv",
                "title": "Legacy Place-Preference CSV",
                "method": "save_all_instances_zone_time_to_csv",
                "status_prefix": "Legacy place-preference export",
                "description": "Historical one-column-per-zone export for older scripts.",
                "assay_profile": "generic",
                "requires_latency": False,
            },
            {
                "key": "zone_metrics",
                "title": "Generic Zone Metrics CSV",
                "method": "save_zone_metrics_to_csv",
                "status_prefix": "Generic zone metrics export",
                "description": "Exports occupancy, dwell, entry, transition, and barrier-adjacent metrics.",
                "assay_profile": "selected",
                "requires_latency": False,
            },
            {
                "key": "assay_summary",
                "title": "Assay Summary (Markdown + CSV)",
                "method": "save_assay_summary_report",
                "status_prefix": "Assay summary export",
                "description": "Profile-aware summary with included/blocked zones and computed metrics.",
                "assay_profile": "selected",
                "requires_latency": False,
            },
            {
                "key": "social_summary",
                "title": "Social Summary (Markdown + CSV)",
                "method": "save_social_summary_report",
                "status_prefix": "Social summary export",
                "description": "Adds latency, door proximity, and pairwise centroid-neighbor metrics.",
                "assay_profile": "selected",
                "requires_latency": True,
            },
        ]

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: palette(window);
                color: palette(window-text);
            }
            QFrame[card="true"],
            QFrame#zoneAnalysisHero,
            QFrame#zoneAnalysisCard {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 12px;
            }
            QLabel[analysisHint="true"] {
                color: palette(mid);
            }
            QLabel[analysisMetricTitle="true"] {
                color: palette(mid);
                font-size: 11px;
                font-weight: 600;
            }
            QLabel[analysisMetricValue="true"] {
                color: palette(text);
                font-size: 18px;
                font-weight: 700;
            }
            QLabel[analysisHeroTitle="true"] {
                color: palette(text);
                font-size: 22px;
                font-weight: 700;
            }
            QLabel[analysisHeroSubtitle="true"] {
                color: palette(mid);
            }
            QGroupBox {
                border: 1px solid palette(mid);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
                background: palette(base);
            }
            QGroupBox::title {
                left: 10px;
                top: -2px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 8px;
                padding: 6px 8px;
            }
            QPushButton {
                background: palette(button);
                border: 1px solid palette(mid);
                border-radius: 8px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton[primary="true"] {
                background: palette(highlight);
                color: palette(highlighted-text);
                border: none;
                font-weight: 700;
            }
            QTabWidget::pane {
                border: 1px solid palette(mid);
                border-radius: 12px;
                background: palette(base);
            }
            QTabBar::tab {
                background: palette(button);
                border: 1px solid palette(mid);
                padding: 6px 10px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                color: palette(window-text);
            }
            QTabBar::tab:selected {
                background: palette(base);
                border-bottom-color: palette(base);
                color: palette(text);
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setObjectName("zoneAnalysisScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        layout.addWidget(scroll)

        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)

        container = content

        hero = QtWidgets.QFrame(container)
        hero.setObjectName("zoneAnalysisHero")
        hero.setProperty("card", True)
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(14, 14, 14, 14)
        hero_layout.setSpacing(6)
        title = QtWidgets.QLabel("Zone Analysis Studio")
        title.setProperty("analysisHeroTitle", True)
        subtitle = QtWidgets.QLabel(
            "Run legacy, generic, assay-aware, or social summaries from the same zone file and profile settings."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("analysisHeroSubtitle", True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        content_layout.addWidget(hero)

        summary_strip = QtWidgets.QFrame(container)
        summary_strip.setProperty("card", True)
        summary_layout = QtWidgets.QHBoxLayout(summary_strip)
        summary_layout.setContentsMargins(10, 6, 10, 6)
        summary_layout.setSpacing(10)
        self.profile_card_value = self._create_inline_metric(
            summary_layout, "Profile", "Generic"
        )
        self.output_card_value = self._create_inline_metric(
            summary_layout, "Mode", "Zone metrics"
        )
        self.latency_card_value = self._create_inline_metric(
            summary_layout, "Latency", "Auto"
        )
        content_layout.addWidget(summary_strip)

        session_box = QtWidgets.QGroupBox("Session Inputs", container)
        session_form = QtWidgets.QFormLayout(session_box)
        self.video_path_edit = QtWidgets.QLineEdit()
        self.video_path_edit.setPlaceholderText("Select a video path")
        self.video_path_button = QtWidgets.QPushButton("Browse")
        self.video_path_button.clicked.connect(self.browse_video)
        video_row = QtWidgets.QHBoxLayout()
        video_row.addWidget(self.video_path_edit, 1)
        video_row.addWidget(self.video_path_button)

        self.zone_path_edit = QtWidgets.QLineEdit()
        self.zone_path_edit.setPlaceholderText("Optional: select a zone JSON path")
        self.zone_path_button = QtWidgets.QPushButton("Browse")
        self.zone_path_button.clicked.connect(self.browse_zone)
        zone_row = QtWidgets.QHBoxLayout()
        zone_row.addWidget(self.zone_path_edit, 1)
        zone_row.addWidget(self.zone_path_button)

        self.fps_edit = QtWidgets.QLineEdit()
        self.fps_edit.setPlaceholderText("Optional FPS (default 30)")
        self.fps_edit.setValidator(QtGui.QDoubleValidator(0.0, 1200.0, 3, self))

        self.latency_reference_edit = QtWidgets.QLineEdit()
        self.latency_reference_edit.setPlaceholderText(
            "Optional latency reference frame"
        )
        self.latency_reference_edit.setValidator(QtGui.QIntValidator(0, 2_147_483_647))
        self.latency_reference_edit.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.latency_reference_edit.textChanged.connect(self._update_latency_card)

        session_form.addRow("Video", video_row)
        session_form.addRow("Zone JSON", zone_row)
        session_form.addRow("FPS", self.fps_edit)
        session_form.addRow("Latency reference", self.latency_reference_edit)
        content_layout.addWidget(session_box)

        profile_box = QtWidgets.QGroupBox("Assay Profile", container)
        profile_layout = QtWidgets.QVBoxLayout(profile_box)
        self.profile_combo = QtWidgets.QComboBox()
        for profile in self._assay_profiles:
            self.profile_combo.addItem(profile.title, userData=profile.name)
        self.profile_combo.currentIndexChanged.connect(self._update_profile_hint)
        self.profile_combo.currentIndexChanged.connect(self._update_profile_card)
        self.profile_hint_label = QtWidgets.QLabel(
            "Selected profile controls how summaries filter zones."
        )
        self.profile_hint_label.setWordWrap(True)
        self.profile_hint_label.setProperty("analysisHint", True)
        profile_layout.addWidget(self.profile_combo)
        profile_layout.addWidget(self.profile_hint_label)
        content_layout.addWidget(profile_box)

        output_box = QtWidgets.QGroupBox("Output Mode", container)
        output_layout = QtWidgets.QVBoxLayout(output_box)
        self.output_mode_combo = QtWidgets.QComboBox()
        for spec in self._mode_specs:
            self.output_mode_combo.addItem(
                str(spec["title"]),
                userData=str(spec["key"]),
            )
        self.output_mode_combo.currentIndexChanged.connect(self._update_mode_hint)
        self.output_mode_combo.currentIndexChanged.connect(self._update_output_card)
        self.output_mode_combo.currentIndexChanged.connect(
            self._update_latency_requirement
        )
        self.mode_hint_label = QtWidgets.QLabel("")
        self.mode_hint_label.setWordWrap(True)
        self.mode_hint_label.setProperty("analysisHint", True)
        output_layout.addWidget(self.output_mode_combo)
        output_layout.addWidget(self.mode_hint_label)
        content_layout.addWidget(output_box)

        action_row = QtWidgets.QHBoxLayout()
        self.run_export_button = QtWidgets.QPushButton("Run Selected Export")
        self.run_export_button.clicked.connect(self._run_selected_export)
        self.run_export_button.setProperty("primary", True)
        self.legacy_export_button = QtWidgets.QPushButton("Legacy CSV")
        self.legacy_export_button.clicked.connect(self.export_legacy_csv)
        self.generic_export_button = QtWidgets.QPushButton("Zone Metrics")
        self.generic_export_button.clicked.connect(self.export_zone_metrics_csv)
        self.assay_summary_button = QtWidgets.QPushButton("Assay Summary")
        self.assay_summary_button.clicked.connect(self.export_assay_summary)
        self.social_summary_button = QtWidgets.QPushButton("Social Summary")
        self.social_summary_button.clicked.connect(self.export_social_summary)
        action_row.addWidget(self.run_export_button, 1)
        action_row.addWidget(self.legacy_export_button)
        action_row.addWidget(self.generic_export_button)
        action_row.addWidget(self.assay_summary_button)
        action_row.addWidget(self.social_summary_button)
        content_layout.addLayout(action_row)

        self.status_label = QtWidgets.QLabel(container)
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("analysisHint", True)
        content_layout.addWidget(self.status_label)

    def _create_inline_metric(
        self,
        parent_layout: QtWidgets.QHBoxLayout,
        title: str,
        value: str,
    ) -> QtWidgets.QLabel:
        box = QtWidgets.QWidget(self)
        box_layout = QtWidgets.QHBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(6)
        title_label = QtWidgets.QLabel(f"{title}:")
        title_label.setProperty("analysisMetricTitle", True)
        value_label = QtWidgets.QLabel(value)
        value_label.setProperty("analysisMetricValue", True)
        box_layout.addWidget(title_label)
        box_layout.addWidget(value_label)
        box_layout.addStretch(1)
        parent_layout.addWidget(box)
        return value_label

    def browse_video(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov,)"
        )
        if filename:
            self.video_path_edit.setText(filename)
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

    def _selected_mode_spec(self) -> dict[str, object]:
        mode_key = (
            self.output_mode_combo.currentData() or self.output_mode_combo.currentText()
        )
        for spec in self._mode_specs:
            if spec["key"] == mode_key:
                return spec
        return self._mode_specs[0]

    def _update_profile_card(self, *args) -> None:
        _ = args
        self.profile_card_value.setText(
            self.profile_combo.currentText().strip() or "Generic"
        )

    def _update_output_card(self, *args) -> None:
        _ = args
        self.output_card_value.setText(
            self.output_mode_combo.currentText().strip() or "Zone metrics"
        )

    def _update_latency_card(self, *args) -> None:
        _ = args
        value = self.latency_reference_edit.text().strip()
        self.latency_card_value.setText(value if value else "Auto")

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

    def _update_mode_hint(self, *args) -> None:
        _ = args
        spec = self._selected_mode_spec()
        self.mode_hint_label.setText(str(spec.get("description") or "").strip())

    def _update_latency_requirement(self, *args) -> None:
        _ = args
        spec = self._selected_mode_spec()
        requires_latency = bool(spec.get("requires_latency"))
        self.latency_reference_edit.setEnabled(requires_latency)
        if requires_latency:
            self.latency_reference_edit.setPlaceholderText(
                "Optional latency reference frame (used by social summary)"
            )
        else:
            self.latency_reference_edit.setPlaceholderText(
                "Not required for this export mode"
            )

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

    @staticmethod
    def _parse_integer_frame_text(text: str) -> int | None:
        value = str(text or "").strip()
        if not value:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _latency_reference_frame(self):
        text = self.latency_reference_edit.text().strip()
        if not text:
            return None
        frame = self._parse_integer_frame_text(text)
        if frame is None:
            raise ValueError("Latency reference frame must be an integer frame number.")
        return frame

    def extract_fps_and_find_zone_file(self, video_path):
        fps = None
        zone_file = None
        if os.path.exists(video_path):
            fps = CV2Video(video_path).get_fps()
            video_dir = Path(video_path).with_suffix("")
            json_files = find_manual_labeled_json_files(video_dir)
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

    def _run_selected_export(self):
        spec = self._selected_mode_spec()
        profile_mode = str(spec.get("assay_profile") or "selected")
        assay_profile = (
            "generic" if profile_mode == "generic" else self._selected_assay_profile()
        )
        latency_reference = (
            self._latency_reference_frame()
            if bool(spec.get("requires_latency"))
            else None
        )
        self._run_export(
            str(spec["method"]),
            status_prefix=str(spec["status_prefix"]),
            assay_profile=assay_profile,
            latency_reference_frame=latency_reference,
        )

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
