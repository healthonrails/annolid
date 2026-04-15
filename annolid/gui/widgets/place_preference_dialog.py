from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QApplication, QFileDialog, QDialog

from annolid.data.videos import CV2Video
from annolid.gui.widgets.zone_manager_utils import zone_file_for_source
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_assay_profiles import available_assay_profiles
from annolid.utils.files import find_manual_labeled_json_files


class _ZoneExportWorker(QtCore.QObject):
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(
        self,
        *,
        video_path: str,
        zone_path: str | None,
        fps: float | None,
        assay_profile: str,
        export_method_name: str,
        latency_reference_frame: int | None,
    ) -> None:
        super().__init__()
        self._video_path = str(video_path)
        self._zone_path = zone_path
        self._fps = fps
        self._assay_profile = str(assay_profile)
        self._export_method_name = str(export_method_name)
        self._latency_reference_frame = latency_reference_frame
        self._cancel_requested = False

    @QtCore.Slot()
    def request_cancel(self) -> None:
        self._cancel_requested = True

    @QtCore.Slot()
    def run(self) -> None:
        if self._cancel_requested:
            self.canceled.emit()
            return
        try:
            analyzer = TrackingResultsAnalyzer(
                self._video_path,
                zone_file=self._zone_path,
                fps=self._fps,
                assay_profile=self._assay_profile,
            )
            if self._cancel_requested:
                self.canceled.emit()
                return
            export_method = getattr(analyzer, self._export_method_name)
            if self._latency_reference_frame is None:
                output_path = export_method()
            else:
                output_path = export_method(
                    latency_reference_frame=self._latency_reference_frame
                )
            if self._cancel_requested:
                self.canceled.emit()
                return
            self.finished.emit(str(output_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class TrackingAnalyzerDialog(QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        video_path: str | None = None,
        zone_path: str | None = None,
        fps: float | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Zone Analysis")
        self.resize(640, 520)

        self._assay_profiles = available_assay_profiles()
        self._mode_specs = self._build_mode_specs()
        self._owner_window = parent
        self._export_thread: QtCore.QThread | None = None
        self._export_worker: _ZoneExportWorker | None = None
        self._active_status_prefix: str | None = None
        self._active_export_started: float | None = None
        self._cancel_requested = False

        self._build_ui()
        self._bind_live_updates()
        self._prefill_from_context(video_path=video_path, zone_path=zone_path, fps=fps)
        self._update_profile_hint()
        self._update_mode_hint()
        self._update_mode_button_state()
        self._update_latency_requirement()

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
        self.use_session_button = QtWidgets.QPushButton("Use Open Video")
        self.use_session_button.clicked.connect(self.apply_session_context)
        video_row = QtWidgets.QHBoxLayout()
        video_row.addWidget(self.video_path_edit, 1)
        video_row.addWidget(self.video_path_button)
        video_row.addWidget(self.use_session_button)

        self.zone_path_edit = QtWidgets.QLineEdit()
        self.zone_path_edit.setPlaceholderText("Optional: select a zone JSON path")
        self.zone_path_button = QtWidgets.QPushButton("Browse")
        self.zone_path_button.clicked.connect(self.browse_zone)
        self.autodetect_zone_button = QtWidgets.QPushButton("Auto")
        self.autodetect_zone_button.clicked.connect(self.autodetect_zone_file)
        zone_row = QtWidgets.QHBoxLayout()
        zone_row.addWidget(self.zone_path_edit, 1)
        zone_row.addWidget(self.zone_path_button)
        zone_row.addWidget(self.autodetect_zone_button)

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
        self.cancel_export_button = QtWidgets.QPushButton("Cancel Export")
        self.cancel_export_button.clicked.connect(self._cancel_running_export)
        self.cancel_export_button.setEnabled(False)
        self.mode_shortcut_button = QtWidgets.QPushButton("Set Mode")
        self.mode_shortcut_menu = QtWidgets.QMenu(self.mode_shortcut_button)
        for spec in self._mode_specs:
            action = self.mode_shortcut_menu.addAction(str(spec["title"]))
            action.triggered.connect(
                lambda _checked=False, key=str(spec["key"]): self._set_output_mode(key)
            )
        self.mode_shortcut_button.setMenu(self.mode_shortcut_menu)
        action_row.addWidget(self.run_export_button, 1)
        action_row.addWidget(self.cancel_export_button)
        action_row.addWidget(self.mode_shortcut_button)
        content_layout.addLayout(action_row)

        self.progress = QtWidgets.QProgressBar(container)
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        content_layout.addWidget(self.progress)

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
            self._refresh_input_defaults_from_video()
            self._set_status("Video selected. Updated FPS and suggested zone file.")

    def browse_zone(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Zone JSON File", "", "JSON Files (*.json)"
        )
        if filename:
            self.zone_path_edit.setText(filename)
            self._set_status(f"Using zone file: {filename}")

    def _bind_live_updates(self) -> None:
        self.video_path_edit.textChanged.connect(self._update_mode_button_state)
        self.video_path_edit.textChanged.connect(
            lambda *_: self._set_status(
                "Video path changed. Click Auto to refresh zone."
            )
        )
        self.zone_path_edit.textChanged.connect(self._update_mode_button_state)

    def _set_status(self, message: str, *, busy: bool = False) -> None:
        self.status_label.setText(str(message or ""))
        self.progress.setVisible(bool(busy))
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        QApplication.processEvents()

    def _set_output_mode(self, mode_key: str) -> None:
        for index in range(self.output_mode_combo.count()):
            if str(self.output_mode_combo.itemData(index) or "") == str(mode_key):
                self.output_mode_combo.setCurrentIndex(index)
                break

    def _owner_session_context(self) -> tuple[str | None, str | None, float | None]:
        owner = self._owner_window
        if owner is None:
            return (None, None, None)
        video_path = str(getattr(owner, "video_file", "") or "").strip() or None
        zone_path = str(getattr(owner, "zone_path", "") or "").strip() or None
        fps_value = getattr(owner, "fps", None)
        try:
            fps = float(fps_value) if fps_value else None
        except Exception:
            fps = None
        return (video_path, zone_path, fps)

    def _prefill_from_context(
        self,
        *,
        video_path: str | None,
        zone_path: str | None,
        fps: float | None,
    ) -> None:
        resolved_video = str(video_path or "").strip() or None
        resolved_zone = str(zone_path or "").strip() or None
        resolved_fps = fps
        if not resolved_video or not resolved_zone or not resolved_fps:
            owner_video, owner_zone, owner_fps = self._owner_session_context()
            if not resolved_video:
                resolved_video = owner_video
            if not resolved_zone:
                resolved_zone = owner_zone
            if not resolved_fps:
                resolved_fps = owner_fps
        if resolved_video:
            self.video_path_edit.setText(resolved_video)
            self._refresh_input_defaults_from_video(
                preferred_zone_file=resolved_zone, preferred_fps=resolved_fps
            )
            self._set_status("Loaded open session context for zone analysis.")
        else:
            self._set_status("Select a video to begin zone analysis.")
        if not resolved_video:
            self.use_session_button.setEnabled(bool(self._owner_window is not None))

    def apply_session_context(self) -> None:
        video_path, zone_path, fps = self._owner_session_context()
        if not video_path:
            self._set_status("No open video in the main window.", busy=False)
            return
        self.video_path_edit.setText(video_path)
        self._refresh_input_defaults_from_video(
            preferred_zone_file=zone_path, preferred_fps=fps
        )
        self._set_status("Applied open video/session zone settings.")

    def _update_mode_button_state(self, *args) -> None:
        _ = args
        has_video = bool(self.video_path_edit.text().strip())
        running = self._export_thread is not None
        self.run_export_button.setEnabled(has_video and not running)
        self.cancel_export_button.setEnabled(running and not self._cancel_requested)
        self.cancel_export_button.setText(
            "Canceling…" if self._cancel_requested and running else "Cancel Export"
        )
        self.mode_shortcut_button.setEnabled(not running)
        self.video_path_button.setEnabled(not running)
        self.use_session_button.setEnabled(not running)
        self.zone_path_button.setEnabled(not running)
        self.autodetect_zone_button.setEnabled(not running)
        self.video_path_edit.setEnabled(not running)
        self.zone_path_edit.setEnabled(not running)
        self.fps_edit.setEnabled(not running)
        self.profile_combo.setEnabled(not running)
        self.output_mode_combo.setEnabled(not running)
        self.latency_reference_edit.setEnabled(
            (not running) and bool(self._selected_mode_spec().get("requires_latency"))
        )

    def _refresh_input_defaults_from_video(
        self,
        *,
        preferred_zone_file: str | None = None,
        preferred_fps: float | None = None,
    ) -> None:
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            return
        fps, zone_file = self.extract_fps_and_find_zone_file(
            video_path, preferred_zone_file=preferred_zone_file
        )
        if preferred_fps:
            fps = preferred_fps
        if fps:
            self.fps_edit.setText(str(fps))
        if preferred_zone_file and str(preferred_zone_file).strip():
            self.zone_path_edit.setText(str(preferred_zone_file).strip())
        elif zone_file:
            self.zone_path_edit.setText(zone_file)

    def autodetect_zone_file(self) -> None:
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            self._set_status("Select a video first to auto-detect zone files.")
            return
        _, zone_file = self.extract_fps_and_find_zone_file(video_path)
        if zone_file:
            self.zone_path_edit.setText(zone_file)
            self._set_status(f"Detected zone file: {zone_file}")
        else:
            self._set_status("No zone JSON was auto-detected for this video.")

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

    def _build_export_job(
        self,
        *,
        export_method_name: str,
        assay_profile: str | None = None,
        latency_reference_frame: int | None = None,
    ) -> dict[str, object]:
        video_path = self.video_path_edit.text().strip()
        zone_path = self.zone_path_edit.text().strip() or None
        fps_text = self.fps_edit.text().strip()
        fps = float(fps_text) if fps_text else None
        if not video_path:
            raise ValueError("A video path is required.")
        profile = assay_profile or self._selected_assay_profile()
        return {
            "video_path": video_path,
            "zone_path": zone_path,
            "fps": fps,
            "assay_profile": str(profile),
            "export_method_name": str(export_method_name),
            "latency_reference_frame": latency_reference_frame,
        }

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

    def extract_fps_and_find_zone_file(
        self, video_path, *, preferred_zone_file: str | None = None
    ):
        fps = None
        zone_file = None
        if os.path.exists(video_path):
            try:
                fps = CV2Video(video_path).get_fps()
            except Exception:
                fps = None
            zone_file = self._infer_zone_file_for_video(
                video_path, preferred_zone_file=preferred_zone_file
            )
        return fps, (str(zone_file) if zone_file else None)

    @staticmethod
    def _is_valid_zone_file(path_value: str | Path | None) -> bool:
        if not path_value:
            return False
        try:
            path = Path(path_value).expanduser()
        except Exception:
            return False
        return path.exists() and path.is_file() and path.suffix.lower() == ".json"

    def _infer_zone_file_for_video(
        self, video_path: str, *, preferred_zone_file: str | None = None
    ) -> Path | None:
        candidates: list[Path] = []

        def _append(path_value: str | Path | None) -> None:
            if not path_value:
                return
            path = Path(path_value).expanduser()
            if path not in candidates:
                candidates.append(path)

        if self._is_valid_zone_file(preferred_zone_file):
            _append(preferred_zone_file)
        current_zone = self.zone_path_edit.text().strip()
        if self._is_valid_zone_file(current_zone):
            _append(current_zone)
        _append(zone_file_for_source(video_path))
        video_file = Path(video_path)
        _append(video_file.with_name(f"{video_file.stem}_zones.json"))

        video_dir = video_file.with_suffix("")
        if video_dir.exists():
            _append(zone_file_for_source(video_dir / f"{video_dir.name}_000000000.png"))
            for path in sorted(video_dir.glob("*_zones.json")):
                _append(path)
            json_files = find_manual_labeled_json_files(str(video_dir))
            for filename in json_files:
                _append(video_dir / filename)

        for candidate in candidates:
            if self._is_valid_zone_file(candidate):
                return candidate
        return None

    def _run_export_sync(
        self,
        export_method_name: str,
        *,
        status_prefix: str,
        assay_profile=None,
        latency_reference_frame=None,
    ) -> str | None:
        start = time.monotonic()
        try:
            self._set_status(
                f"{status_prefix}: preparing analyzer and inputs…",
                busy=True,
            )
            job = self._build_export_job(
                export_method_name=export_method_name,
                assay_profile=assay_profile,
                latency_reference_frame=latency_reference_frame,
            )
            analyzer = TrackingResultsAnalyzer(
                str(job["video_path"]),
                zone_file=job["zone_path"],
                fps=job["fps"],
                assay_profile=str(job["assay_profile"]),
            )
            export_method = getattr(analyzer, str(job["export_method_name"]))
            self._set_status(f"{status_prefix}: running export…", busy=True)
            if job["latency_reference_frame"] is None:
                output_path = export_method()
            else:
                output_path = export_method(
                    latency_reference_frame=int(job["latency_reference_frame"])
                )
        except Exception as exc:
            self._set_status(f"{status_prefix} failed: {exc}", busy=False)
            QtWidgets.QMessageBox.warning(self, f"{status_prefix} failed", str(exc))
            return None
        elapsed = time.monotonic() - start
        self._set_status(
            f"{status_prefix} saved to {output_path} ({elapsed:.1f}s).", busy=False
        )
        return output_path

    def _cleanup_export_worker(self) -> None:
        self._export_thread = None
        self._export_worker = None
        self._active_status_prefix = None
        self._active_export_started = None
        self._cancel_requested = False
        self._update_mode_button_state()

    def _on_export_finished(self, output_path: str) -> None:
        status_prefix = self._active_status_prefix or "Zone analysis export"
        started = self._active_export_started
        elapsed = (time.monotonic() - started) if started is not None else None
        elapsed_suffix = f" ({elapsed:.1f}s)" if elapsed is not None else ""
        if self._cancel_requested:
            self._set_status(
                f"{status_prefix}: cancellation requested, but current step completed and saved {output_path}{elapsed_suffix}.",
                busy=False,
            )
        else:
            self._set_status(
                f"{status_prefix} saved to {output_path}{elapsed_suffix}.",
                busy=False,
            )
        self._cleanup_export_worker()

    def _on_export_failed(self, message: str) -> None:
        status_prefix = self._active_status_prefix or "Zone analysis export"
        self._set_status(f"{status_prefix} failed: {message}", busy=False)
        self._cleanup_export_worker()
        QtWidgets.QMessageBox.warning(self, f"{status_prefix} failed", str(message))

    def _on_export_canceled(self) -> None:
        status_prefix = self._active_status_prefix or "Zone analysis export"
        self._set_status(f"{status_prefix} canceled.", busy=False)
        self._cleanup_export_worker()

    def _cancel_running_export(self) -> None:
        if self._export_worker is None or self._export_thread is None:
            self._set_status("No active export to cancel.")
            return
        self._cancel_requested = True
        self._update_mode_button_state()
        self._set_status(
            "Cancellation requested. Waiting for the current export step to stop…",
            busy=True,
        )
        try:
            QtCore.QMetaObject.invokeMethod(
                self._export_worker,
                "request_cancel",
                QtCore.Qt.QueuedConnection,
            )
        except Exception:
            self._set_status(
                "Cancellation request could not be delivered. Export is still running.",
                busy=True,
            )

    def _run_export_async(
        self,
        export_method_name: str,
        *,
        status_prefix: str,
        assay_profile=None,
        latency_reference_frame=None,
    ) -> None:
        if self._export_thread is not None:
            self._set_status("An export is already running. Please wait.", busy=True)
            return
        try:
            job = self._build_export_job(
                export_method_name=export_method_name,
                assay_profile=assay_profile,
                latency_reference_frame=latency_reference_frame,
            )
        except Exception as exc:
            self._set_status(f"{status_prefix} failed: {exc}", busy=False)
            QtWidgets.QMessageBox.warning(self, f"{status_prefix} failed", str(exc))
            return

        thread = QtCore.QThread(self)
        worker = _ZoneExportWorker(
            video_path=str(job["video_path"]),
            zone_path=job["zone_path"],
            fps=job["fps"],
            assay_profile=str(job["assay_profile"]),
            export_method_name=str(job["export_method_name"]),
            latency_reference_frame=job["latency_reference_frame"],
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        worker.canceled.connect(self._on_export_canceled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._export_thread = thread
        self._export_worker = worker
        self._active_status_prefix = status_prefix
        self._active_export_started = time.monotonic()
        self._cancel_requested = False
        self._set_status(f"{status_prefix}: preparing analyzer and inputs…", busy=True)
        self._update_mode_button_state()
        self._set_status(f"{status_prefix}: running export…", busy=True)
        thread.start()

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
        self._run_export_async(
            str(spec["method"]),
            status_prefix=str(spec["status_prefix"]),
            assay_profile=assay_profile,
            latency_reference_frame=latency_reference,
        )

    def export_legacy_csv(self):
        self._run_export_sync(
            "save_all_instances_zone_time_to_csv",
            status_prefix="Legacy place-preference export",
            assay_profile="generic",
        )

    def export_zone_metrics_csv(self):
        self._run_export_sync(
            "save_zone_metrics_to_csv",
            status_prefix="Generic zone metrics export",
            assay_profile=self._selected_assay_profile(),
        )

    def export_assay_summary(self):
        self._run_export_sync(
            "save_assay_summary_report",
            status_prefix="Assay summary export",
            assay_profile=self._selected_assay_profile(),
        )

    def export_social_summary(self):
        self._run_export_sync(
            "save_social_summary_report",
            status_prefix="Social summary export",
            assay_profile=self._selected_assay_profile(),
            latency_reference_frame=self._latency_reference_frame(),
        )

    def run_analysis_without_gui(self, video_path, zone_path=None, fps=None):
        self.video_path_edit.setText(video_path)
        resolved_fps, resolved_zone = self.extract_fps_and_find_zone_file(
            video_path, preferred_zone_file=zone_path
        )
        if zone_path:
            resolved_zone = zone_path
        if fps:
            resolved_fps = fps
        if resolved_zone:
            self.zone_path_edit.setText(str(resolved_zone))
        if resolved_fps:
            self.fps_edit.setText(str(resolved_fps))
        self.export_zone_metrics_csv()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._export_thread is not None:
            self._set_status(
                "Zone analysis export is still running. Wait for it to complete."
            )
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = TrackingAnalyzerDialog()
    dialog.exec_()
    sys.exit(app.exec_())
