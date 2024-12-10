import os
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTableWidget, QFileDialog,
    QTableWidgetItem, QProgressBar, QMessageBox, QAbstractItemView, QHBoxLayout
)
from qtpy.QtCore import Signal, Qt
from annolid.data.videos import extract_frames_from_videos
from qtpy.QtCore import QThread


class FrameExtractorWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, videos, output_folder, num_frames=5):
        super().__init__()
        self.videos = videos
        self.output_folder = output_folder
        self.num_frames = num_frames

    def run(self):
        try:
            for idx, video in enumerate(self.videos, 1):
                extract_frames_from_videos(
                    input_folder=os.path.dirname(video),
                    output_folder=self.output_folder,
                    num_frames=self.num_frames
                )
                progress_value = int((idx / len(self.videos)) * 100)
                self.progress.emit(progress_value)
                # Process videos and emit progress
                self.progress.emit(int((idx + 1) / len(self.videos) * 100))
            self.finished.emit(self.output_folder)
        except Exception as e:
            self.error.emit(str(e))


class VideoManagerWidget(QWidget):
    video_selected = Signal(str)  # Signal to send the selected video path
    close_video_requested = Signal()  # Signal to request closing the current video
    # Signal to notify the main window of the output folder
    output_folder_ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Layouts
        self.main_layout = QVBoxLayout(self)
        self.button_layout = QHBoxLayout()  # Horizontal layout for buttons

        # Set to track imported videos
        self.imported_videos = set()

        # Import Button
        self.import_button = QPushButton("Import Videos")
        self.import_button.clicked.connect(self.import_videos)
        self.button_layout.addWidget(self.import_button)

        # Close Video Button
        self.close_video_button = QPushButton("Close Video")
        self.close_video_button.clicked.connect(self.request_close_video)
        self.button_layout.addWidget(self.close_video_button)

        # Extract Frames Button
        self.extract_frames_button = QPushButton("Extract Frames")
        self.extract_frames_button.clicked.connect(self.extract_frames)
        self.button_layout.addWidget(self.extract_frames_button)

        self.main_layout.addLayout(self.button_layout)

        # Video Table
        self.video_table = QTableWidget(0, 4)
        self.video_table.setHorizontalHeaderLabels(
            ["Name", "Path", "Load", "Delete"])
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        self.main_layout.addWidget(self.video_table)

    def import_videos(self):
        # Open folder dialog
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Video Folder")
        if not folder_path:
            return

        # Get video files (recursively)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv',
                            '.mpg'}  # Add more extensions if needed
        video_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in video_extensions:
                    video_files.append(os.path.join(root, file))

        # Add videos to the table
        for video in video_files:
            # Check if video is already added
            if video not in self.imported_videos:
                self.add_video_to_table(video)

    def add_video_to_table(self, video_path):
        # Add the video to the imported videos set
        self.imported_videos.add(video_path)
        # Get video name
        video_name = os.path.basename(video_path)

        # Create a new row
        row_position = self.video_table.rowCount()
        self.video_table.insertRow(row_position)

        # Add Name and Path
        self.video_table.setItem(row_position, 0, QTableWidgetItem(video_name))
        self.video_table.setItem(row_position, 1, QTableWidgetItem(video_path))

        # Add Load Button
        load_button = QPushButton("Load")
        load_button.clicked.connect(
            lambda: self.video_selected.emit(video_path))
        self.video_table.setCellWidget(row_position, 2, load_button)

        # Add Delete Button
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(
            lambda: self.delete_video(row_position, video_path))
        self.video_table.setCellWidget(row_position, 3, delete_button)

    def delete_video(self, row, video_path):
        # Confirm deletion
        confirmation = QMessageBox.question(
            self, "Delete Video", "Are you sure you want to delete this video from the list?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirmation == QMessageBox.Yes:
            # Remove from the imported videos set
            self.imported_videos.discard(video_path)

            self.video_table.removeRow(row)

    def request_close_video(self):
        # Emit the signal to request closing the video
        self.close_video_requested.emit()

    def update_progress(self, value):
        """
        Update the progress bar or log progress.
        """
        self.progress_bar.setValue(value)

    def on_extraction_complete(self, output_folder):
        QMessageBox.information(
            self, "Extraction Complete", "Frame extraction finished successfully!")
        self.progress_bar.setValue(100)
        # Emit signal to notify the main window
        self.output_folder_ready.emit(output_folder)
        self.progress_bar.setVisible(False)

    def show_error(self, error_message):
        QMessageBox.critical(
            self, "Error", f"An error occurred: {error_message}")

    def extract_frames(self):
        if not self.imported_videos:
            QMessageBox.warning(
                self, "No Videos", "Please import videos before extracting frames.")
            return

        # Select output folder
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder")
        if not output_folder:
            return
        self.progress_bar.setVisible(True)

        self.worker = FrameExtractorWorker(self.imported_videos, output_folder)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_extraction_complete)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def run_frame_extraction(self, output_folder):
        try:
            total_videos = len(self.imported_videos)
            for idx, video_path in enumerate(self.imported_videos, start=1):
                # Extract frames for the current video
                video_folder = os.path.dirname(video_path)
                extract_frames_from_videos(
                    video_folder, output_folder, num_frames=5)

                # Update progress bar
                progress = int((idx / total_videos) * 100)
                self.progress_bar.setValue(progress)

            self.progress_bar.setValue(100)
            QMessageBox.information(
                self, "Success", "Frame extraction completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        finally:
            self.progress_bar.setVisible(False)
