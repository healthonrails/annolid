import os
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QAbstractItemView
)


class VideoManagerWidget(QWidget):
    video_selected = Signal(str)  # Signal to send the selected video path
    close_video_requested = Signal()  # Signal to request closing the current video

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

        self.main_layout.addLayout(self.button_layout)

        # Video Table
        self.video_table = QTableWidget(0, 4)
        self.video_table.setHorizontalHeaderLabels(
            ["Name", "Path", "Load", "Delete"])
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)
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
