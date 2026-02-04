import sys
import os
from qtpy.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.data.videos import CV2Video
from pathlib import Path
from annolid.utils.files import find_manual_labeled_json_files


class TrackingAnalyzerDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Place Preference Analyzer")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

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

        form_layout.addRow(QLabel("Video Path:"), self.video_path_edit)
        form_layout.addRow(QLabel(""), self.video_path_button)
        form_layout.addRow(QLabel("Zone JSON Path:"), self.zone_path_edit)
        form_layout.addRow(QLabel(""), self.zone_path_button)
        form_layout.addRow(QLabel("FPS:"), self.fps_edit)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze)

        layout.addLayout(form_layout)
        layout.addWidget(self.analyze_button)

        self.setLayout(layout)

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

    def analyze(self):
        video_path = self.video_path_edit.text()
        zone_path = self.zone_path_edit.text()
        fps = float(self.fps_edit.text()) if self.fps_edit.text() else None

        analyzer = TrackingResultsAnalyzer(video_path, zone_file=zone_path, fps=fps)
        try:
            analyzer.merge_and_calculate_distance()
            analyzer.save_all_instances_zone_time_to_csv()
        except Exception:
            pass

        # Optional: Notify the user with the path of the result CSV file
        # if output_csv_path:
        #     QMessageBox.information(
        #         self, "Analysis Complete",
        #         f"Analysis is complete. Result CSV file saved at:\n{output_csv_path}",
        #         QMessageBox.Ok)

    def run_analysis_without_gui(self, video_path, zone_path=None, fps=None):
        self.video_path_edit.setText(video_path)
        if zone_path is None:
            fps, zone_path = self.extract_fps_and_find_zone_file(video_path)
        if zone_path:
            self.zone_path_edit.setText(zone_path)
        if fps:
            self.fps_edit.setText(str(fps))
        self.analyze()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = TrackingAnalyzerDialog()
    dialog.exec_()
    sys.exit(app.exec_())
