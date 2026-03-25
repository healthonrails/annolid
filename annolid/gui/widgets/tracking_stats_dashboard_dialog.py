from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from qtpy import QtCore, QtGui, QtWidgets

from annolid.postprocessing.tracking_stats_dashboard import (
    TrackingStatsArtifacts,
    analyze_and_visualize_tracking_stats,
)
from annolid.utils.logger import logger


class TrackingStatsDashboardWidget(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        initial_root_dir: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._artifacts: Optional[TrackingStatsArtifacts] = None

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        controls_group = QtWidgets.QGroupBox("Data Source")
        controls_layout = QtWidgets.QGridLayout(controls_group)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)

        self.root_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.root_browse_btn = QtWidgets.QPushButton("Browse...")
        self.output_browse_btn = QtWidgets.QPushButton("Browse...")
        self.run_btn = QtWidgets.QPushButton("Run Analysis")
        self.open_output_btn = QtWidgets.QPushButton("Open Output Folder")
        self.open_output_btn.setEnabled(False)

        self.root_browse_btn.clicked.connect(self._choose_root_dir)
        self.output_browse_btn.clicked.connect(self._choose_output_dir)
        self.run_btn.clicked.connect(self.run_analysis)
        self.open_output_btn.clicked.connect(self._open_output_folder)

        controls_layout.addWidget(QtWidgets.QLabel("Root Directory"), 0, 0)
        controls_layout.addWidget(self.root_dir_edit, 0, 1)
        controls_layout.addWidget(self.root_browse_btn, 0, 2)
        controls_layout.addWidget(QtWidgets.QLabel("Output Directory"), 1, 0)
        controls_layout.addWidget(self.output_dir_edit, 1, 1)
        controls_layout.addWidget(self.output_browse_btn, 1, 2)
        controls_layout.addWidget(self.run_btn, 2, 1)
        controls_layout.addWidget(self.open_output_btn, 2, 2)
        main_layout.addWidget(controls_group)

        cards_row = QtWidgets.QHBoxLayout()
        cards_row.setSpacing(8)
        self.videos_card = self._make_stat_card("Videos", "0")
        self.manual_frames_card = self._make_stat_card("Manual Frames", "0")
        self.abnormal_card = self._make_stat_card("Abnormal Segments", "0")
        self.unresolved_bad_card = self._make_stat_card("Unresolved Bad Shapes", "0")
        cards_row.addWidget(self.videos_card)
        cards_row.addWidget(self.manual_frames_card)
        cards_row.addWidget(self.abnormal_card)
        cards_row.addWidget(self.unresolved_bad_card)
        main_layout.addLayout(cards_row)

        self.tabs = QtWidgets.QTabWidget()
        self.overview_table = QtWidgets.QTableWidget()
        self.abnormal_table = QtWidgets.QTableWidget()
        self.bad_shape_table = QtWidgets.QTableWidget()
        for table in (self.overview_table, self.abnormal_table, self.bad_shape_table):
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            table.setAlternatingRowColors(True)
            table.horizontalHeader().setStretchLastSection(True)

        self.tabs.addTab(self.overview_table, "Overview")
        self.tabs.addTab(self.abnormal_table, "Abnormal Segments")
        self.tabs.addTab(self.bad_shape_table, "Bad Shape Events")

        self.plot_panel = QtWidgets.QScrollArea()
        self.plot_panel.setWidgetResizable(True)
        plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.plot_layout.setContentsMargins(10, 10, 10, 10)
        self.plot_layout.setSpacing(10)
        self.plot_panel.setWidget(plot_container)
        self.tabs.addTab(self.plot_panel, "Plots")

        main_layout.addWidget(self.tabs, stretch=1)

        self.status_label = QtWidgets.QLabel(
            "Select a root directory and run analysis."
        )
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        default_root = (
            Path(initial_root_dir).expanduser().resolve()
            if initial_root_dir
            else Path.cwd().resolve()
        )
        self.root_dir_edit.setText(str(default_root))
        self.output_dir_edit.setText(str(default_root / "tracking_stats_dashboard"))

    @staticmethod
    def _make_stat_card(title: str, value: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("trackingStatsCard")
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setStyleSheet(
            "#trackingStatsCard {"
            "  border: 1px solid #2b3f56;"
            "  border-radius: 10px;"
            "  background-color: #13283b;"
            "  color: #e5edf5;"
            "}"
        )
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-size: 11px; color: #9ec0df;")
        value_label = QtWidgets.QLabel(value)
        value_label.setObjectName("valueLabel")
        value_label.setStyleSheet("font-size: 22px; font-weight: 700; color: #f7fbff;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        frame._value_label = value_label  # type: ignore[attr-defined]
        return frame

    @staticmethod
    def _set_card_value(card: QtWidgets.QFrame, value: int) -> None:
        label = getattr(card, "_value_label", None)
        if isinstance(label, QtWidgets.QLabel):
            label.setText(str(int(value)))

    def _choose_root_dir(self) -> None:
        start = self.root_dir_edit.text().strip() or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Root Directory", start
        )
        if not directory:
            return
        self.root_dir_edit.setText(directory)
        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(
                str(Path(directory) / "tracking_stats_dashboard")
            )

    def _choose_output_dir(self) -> None:
        start = self.output_dir_edit.text().strip() or self.root_dir_edit.text().strip()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start or str(Path.home())
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _open_output_folder(self) -> None:
        artifacts = self._artifacts
        if artifacts is None:
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(artifacts.output_dir))
        )

    @staticmethod
    def _populate_table(table: QtWidgets.QTableWidget, dataframe: pd.DataFrame) -> None:
        if dataframe is None:
            dataframe = pd.DataFrame()
        table.clear()
        columns = [str(c) for c in dataframe.columns]
        table.setColumnCount(len(columns))
        table.setRowCount(len(dataframe))
        if columns:
            table.setHorizontalHeaderLabels(columns)
        for row_idx in range(len(dataframe)):
            row = dataframe.iloc[row_idx]
            for col_idx, col_name in enumerate(columns):
                value = row.get(col_name, "")
                if pd.isna(value):
                    value = ""
                item = QtWidgets.QTableWidgetItem(str(value))
                table.setItem(row_idx, col_idx, item)
        table.resizeColumnsToContents()

    def _clear_plot_previews(self) -> None:
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_plot_preview(self, title: str, path: Optional[Path]) -> None:
        container = QtWidgets.QFrame()
        container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QtWidgets.QLabel(title))
        if path is None or not Path(path).exists():
            label = QtWidgets.QLabel("Plot unavailable (matplotlib may be missing).")
            label.setStyleSheet("color:#888;")
            layout.addWidget(label)
            self.plot_layout.addWidget(container)
            return
        pixmap = QtGui.QPixmap(str(path))
        image_label = QtWidgets.QLabel()
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.setPixmap(
            pixmap.scaledToWidth(900, QtCore.Qt.SmoothTransformation)
            if not pixmap.isNull()
            else pixmap
        )
        layout.addWidget(image_label)
        open_button = QtWidgets.QPushButton("Open Image")
        open_button.clicked.connect(
            lambda *_: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(path))
            )
        )
        layout.addWidget(open_button, alignment=QtCore.Qt.AlignRight)
        self.plot_layout.addWidget(container)

    def _refresh_from_artifacts(self, artifacts: TrackingStatsArtifacts) -> None:
        overview_df = pd.read_csv(artifacts.overview_csv)
        abnormal_df = pd.read_csv(artifacts.abnormal_segments_csv)
        bad_shape_df = pd.read_csv(artifacts.bad_shape_events_csv)
        self._populate_table(self.overview_table, overview_df)
        self._populate_table(self.abnormal_table, abnormal_df)
        self._populate_table(self.bad_shape_table, bad_shape_df)

        videos = int(len(overview_df))
        manual_frames = (
            int(overview_df["manual_frames"].sum()) if not overview_df.empty else 0
        )
        abnormal_segments = (
            int(overview_df["abnormal_segment_events"].sum())
            if not overview_df.empty
            else 0
        )
        unresolved_bad = (
            int(overview_df["bad_shape_events_unresolved"].sum())
            if not overview_df.empty
            else 0
        )
        self._set_card_value(self.videos_card, videos)
        self._set_card_value(self.manual_frames_card, manual_frames)
        self._set_card_value(self.abnormal_card, abnormal_segments)
        self._set_card_value(self.unresolved_bad_card, unresolved_bad)

        self._clear_plot_previews()
        self._add_plot_preview(
            "Manual Frames vs Bad Shape Frames", artifacts.manual_badshape_plot
        )
        self._add_plot_preview(
            "Abnormal Segment Events by Video", artifacts.abnormal_segments_plot
        )
        self._add_plot_preview(
            "Manual Frames vs Unresolved Bad Shapes",
            artifacts.unresolved_bad_shapes_plot,
        )
        self.plot_layout.addStretch(1)

    def run_analysis(self) -> None:
        root_text = self.root_dir_edit.text().strip()
        output_text = self.output_dir_edit.text().strip()
        if not root_text:
            QtWidgets.QMessageBox.warning(
                self, "Missing Root Directory", "Please select a root directory."
            )
            return

        root_dir = Path(root_text).expanduser()
        if not root_dir.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Directory Not Found",
                f"Root directory does not exist:\n{root_dir}",
            )
            return

        output_dir = Path(output_text).expanduser() if output_text else None
        self.run_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            artifacts = analyze_and_visualize_tracking_stats(
                root_dir=root_dir, output_dir=output_dir
            )
            self._artifacts = artifacts
            self.open_output_btn.setEnabled(True)
            self._refresh_from_artifacts(artifacts)
            self.status_label.setText(f"Dashboard generated in {artifacts.output_dir}.")
        except Exception as exc:
            logger.error(
                "Failed to generate tracking stats dashboard: %s", exc, exc_info=True
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Dashboard Error",
                f"Failed to generate tracking stats dashboard:\n\n{exc}",
            )
            self.status_label.setText("Dashboard generation failed.")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.run_btn.setEnabled(True)


class TrackingStatsDashboardDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        initial_root_dir: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tracking Stats Dashboard")
        self.resize(1240, 820)
        layout = QtWidgets.QVBoxLayout(self)
        self.dashboard = TrackingStatsDashboardWidget(
            initial_root_dir=initial_root_dir, parent=self
        )
        layout.addWidget(self.dashboard)
