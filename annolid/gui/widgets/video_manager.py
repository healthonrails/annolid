from pathlib import Path
import os
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTableWidget, QFileDialog,
    QTableWidgetItem, QProgressBar, QMessageBox, QAbstractItemView, QHBoxLayout, QTextEdit
)
from qtpy.QtCore import Signal, Qt, Slot
from annolid.data.videos import extract_frames_from_videos
from annolid.gui.workers import FrameExtractorWorker, ProcessVideosWorker, TrackAllWorker
from annolid.utils.files import find_most_recent_file
from annolid.utils.logger import logger


class VideoManagerWidget(QWidget):
    video_selected = Signal(str)
    close_video_requested = Signal()
    output_folder_ready = Signal(str)
    json_saved = Signal(str, str)
    track_all_worker_created = Signal(TrackAllWorker)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Layouts
        self.main_layout = QVBoxLayout(self)
        self.button_layout = QHBoxLayout()

        # Set to track imported videos
        self.imported_videos = set()

        # Import Button
        self.import_button = QPushButton("Import Videos")
        self.import_button.clicked.connect(self.import_videos)
        self.button_layout.addWidget(self.import_button)

        # Track All Button (replaces Batch Predict)
        self.track_all_button = QPushButton("Track All")
        self.track_all_button.clicked.connect(self.track_all_videos)
        self.button_layout.addWidget(self.track_all_button)

        self.stop_track_button = QPushButton("Stop Tracking")
        self.stop_track_button.clicked.connect(self.stop_tracking)
        self.stop_track_button.setVisible(False)
        self.button_layout.addWidget(self.stop_track_button)

        # Process All Videos Button
        self.process_all_button = QPushButton("Analyze All")
        self.process_all_button.clicked.connect(self.process_all_videos)
        self.button_layout.addWidget(self.process_all_button)

        # Extract Frames Button
        self.extract_frames_button = QPushButton("Extract Frames")
        self.extract_frames_button.clicked.connect(self.extract_frames)
        self.button_layout.addWidget(self.extract_frames_button)

        # Clear All Button
        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.clear_all_content)
        self.button_layout.addWidget(self.clear_all_button)

        self.main_layout.addLayout(self.button_layout)

        # Video Table
        # Video Table with new column for JSON status
        self.video_table = QTableWidget(0, 6)  # Increased column count to 6
        self.video_table.setHorizontalHeaderLabels(
            ["Name", "Path", "Load", "Close", "Delete", "JSON Labeled"]
        )
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)

        # Chat-Like Display
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #343a40;
                border: none;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
                padding: 10px;
                border-radius: 8px;
            }
        """)
        self.chat_display.setVisible(False)  # Initially hidden

        # Add widgets to layout
        self.main_layout.addWidget(self.video_table)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.chat_display)

    def import_videos(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Video Folder")
        if not folder_path:
            return

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mpg'}
        video_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files if os.path.splitext(file)[1].lower() in video_extensions
        ]

        for video in video_files:
            if video not in self.imported_videos:
                self.add_video_to_table(video)

    def check_json_exists(self, video_path):
        """Check if a JSON label file exists for the video's first frame."""
        video_name = Path(video_path).stem
        output_folder = Path(video_path).with_suffix('')  # e.g., video_name/
        json_file = find_most_recent_file(output_folder, '.json')
        if json_file and Path(json_file).exists():
            return "Yes"
        else:
            return "No"

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
        # Use a partial to pass video_path, row determination will be dynamic
        load_button.clicked.connect(
            lambda checked=False, vp=video_path: self.load_video_and_response(vp))  # Pass video_path
        self.video_table.setCellWidget(row_position, 2, load_button)

        # Add Close Button
        close_button = QPushButton("Close")
        # For close, knowing the row index might not be as critical if it just emits a global close
        close_button.clicked.connect(
            lambda checked=False, r=row_position: self.close_video_and_clear_by_sender_or_row(r))
        self.video_table.setCellWidget(row_position, 3, close_button)

        # Add Delete Button
        delete_button = QPushButton("Delete")
        # Connect to a new slot that dynamically finds the row
        delete_button.clicked.connect(self._handle_delete_button_clicked)
        # Store video_path on the button itself for easy retrieval
        delete_button.setProperty("video_path", video_path)
        self.video_table.setCellWidget(row_position, 4, delete_button)

        # Add JSON Labeled Status
        json_status = self.check_json_exists(video_path)
        json_item = QTableWidgetItem(json_status)
        json_item.setFlags(json_item.flags() & ~Qt.ItemIsEditable)
        self.video_table.setItem(row_position, 5, json_item)

    @Slot()
    def _handle_delete_button_clicked(self):
        button = self.sender()  # Get the button that was clicked
        if button:
            # Find the row of this button
            for row in range(self.video_table.rowCount()):
                # Check if cellWidget in column 4 (Delete button column) is our button
                if self.video_table.cellWidget(row, 4) == button:
                    video_path_to_delete = button.property(
                        "video_path")  # Retrieve stored path
                    # Now call the original delete_video logic with the correct current row
                    self.delete_video(row, video_path_to_delete)
                    break  # Found and processed, exit loop

    def close_video_and_clear_by_sender_or_row(self, fallback_row=None):
        button = self.sender()
        row_closed = -1
        video_name_closed = "Unknown"

        if button:
            for row in range(self.video_table.rowCount()):
                # Assuming close is in col 3
                if self.video_table.cellWidget(row, 3) == button:
                    row_closed = row
                    item = self.video_table.item(row, 0)  # Get name item
                    if item:
                        video_name_closed = item.text()
                    break
        elif fallback_row is not None:
            row_closed = fallback_row
            item = self.video_table.item(fallback_row, 0)
            if item:
                video_name_closed = item.text()

        self.close_video_requested.emit()
        self.chat_display.clear()
        self.chat_display.setVisible(False)
        if row_closed != -1:
            QMessageBox.information(self, "Video Closed",
                                    f"Video '{video_name_closed}' (formerly at row {row_closed + 1}) interface closed.")
        else:
            QMessageBox.information(
                self, "Video Closed", "Video interface closed.")

    def load_video_and_response(self, video_path):
        response_file = Path(video_path).with_suffix('.txt')

        # Check if response file exists
        if response_file.exists():
            try:
                # Read response content
                with open(response_file, 'r') as file:
                    content = file.read()

                # Emit signal to load the video (or pass to a player widget)
                self.video_selected.emit(video_path)

                # Display the response content in the chat-like display area
                self.chat_display.setVisible(True)
                self.chat_display.clear()
                self.chat_display.append(
                    f"<b>Response for:</b> {os.path.basename(video_path)}<br><br>")
                self.chat_display.append(content.replace('\n', '<br>'))

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load response: {e}")
        else:
            self.video_selected.emit(video_path)

    def track_all_videos(self):
        """Initiate Track All for all imported videos with JSON label files."""
        if not self.imported_videos:
            QMessageBox.warning(self, "No Videos",
                                "Please import videos before tracking.")
            return

        # Get all video paths
        video_paths = [self.video_table.item(row, 1).text()
                       for row in range(self.video_table.rowCount())]

        # Initialize worker
        self.worker = TrackAllWorker(video_paths=video_paths, parent=self)
        self.track_all_worker_created.emit(
            self.worker)  # Emit the worker instance
        self.worker.progress.connect(self.update_track_progress)
        self.worker.finished.connect(self.on_track_all_complete)
        self.worker.error.connect(self.show_error)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.chat_display.setVisible(True)
        self.stop_track_button.setVisible(True)
        self.worker.start()

    def stop_tracking(self):
        """Stop the Track All worker."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            # Also tell AnnolidWindow to finalize progress for the video that was interrupted
            if self.parent() and hasattr(self.parent(), '_handle_track_all_video_finished') and self.worker.current_processing_video_path:  # you'd need to store this in worker
                self.parent()._handle_track_all_video_finished(
                    self.worker.current_processing_video_path)
        self.stop_track_button.setVisible(False)
        self.progress_bar.setVisible(False)

    def update_track_progress(self, percentage, message):
        """Update progress bar and chat display."""
        self.progress_bar.setValue(percentage)
        self.chat_display.append(f"<b>{message}</b><br>")
        self.chat_display.ensureCursorVisible()

    def on_track_all_complete(self, message):
        """Handle completion of Track All."""
        QMessageBox.information(self, "Track All Complete", message)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.update_json_status()  # Refresh JSON Labeled column
        for video_path in self.imported_videos:
            output_folder = Path(video_path).with_suffix('')
            self.output_folder_ready.emit(str(output_folder))

    def show_error(self, error_message):
        """Display error message in chat display and as a warning."""
        self.chat_display.append(f"<b>Error:</b> {error_message}<br>")
        QMessageBox.warning(self, "Error", error_message)

    def delete_video(self, row, video_path):
        if QMessageBox.question(
            self, "Delete Video", "Are you sure you want to delete this video from the list?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self.imported_videos.discard(video_path)
            self.video_table.removeRow(row)

    def display_latest_response(self):
        # Get selected video
        selected_row = self.video_table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "No Selection", "Please select a video.")
            return

        video_path = self.video_table.item(selected_row, 1).text()
        response_file = Path(video_path).with_suffix('.txt')

        if not response_file.exists():
            QMessageBox.warning(self, "File Not Found",
                                "No response file found for the selected video.")
            return

        # Display response content
        try:
            with open(response_file, 'r') as file:
                content = file.read()

            self.chat_display.setVisible(True)
            self.chat_display.clear()
            self.chat_display.append(
                f"<b>Response for:</b> {response_file.name}<br><br>")
            self.chat_display.append(content.replace('\n', '<br>'))

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load response: {e}")

    def update_progress(self, value):
        """
        Update the progress bar or log progress.
        """
        self.progress_bar.setValue(value)

    def clear_all_content(self):
        """
        Clear all imported videos, reset the table, and clear the chat display.
        """
        # Emit the close video signal
        self.close_video_requested.emit()
        # Clear imported videos set
        self.imported_videos.clear()

        # Clear the video table
        self.video_table.setRowCount(0)

        # Clear chat display
        self.chat_display.clear()
        self.chat_display.setVisible(False)

        # Optionally, notify the user
        QMessageBox.information(
            self, "Clear All", "All videos and results have been cleared.")

    def update_json_status(self):
        """Update the JSON Labeled column for all videos in the table."""
        for row in range(self.video_table.rowCount()):
            video_path = self.video_table.item(row, 1).text()
            json_status = self.check_json_exists(video_path)
            json_item = QTableWidgetItem(json_status)
            json_item.setFlags(json_item.flags() & ~Qt.ItemIsEditable)
            self.video_table.setItem(row, 5, json_item)

    @Slot(str, str)
    def update_json_column(self, video_path, json_filename):
        """
        Update the 'JSON Labeled' column for the given video when a JSON is saved.

        Args:
            video_path (str): Path to the video file.
            json_filename (str): Path to the saved JSON file.
        """
        try:
            json_filename = Path(json_filename)
            # Validate JSON file
            if not json_filename.exists() or json_filename.suffix != '.json':
                logger.warning(f"Invalid JSON file: {json_filename}")
                return

            # Find the row for the video_path
            for row in range(self.video_table.rowCount()):
                if self.video_table.item(row, 1).text() == video_path:
                    # Update JSON Labeled column to "Yes"
                    json_item = QTableWidgetItem("Yes")
                    json_item.setFlags(json_item.flags() & ~Qt.ItemIsEditable)
                    self.video_table.setItem(row, 5, json_item)
                    logger.debug(
                        f"Updated JSON Labeled to 'Yes' for {video_path}")
                    break
            else:
                logger.warning(f"Video not found in table: {video_path}")

        except Exception as e:
            logger.error(
                f"Failed to update JSON Labeled for {video_path}: {str(e)}", exc_info=True)

    def on_extraction_complete(self, output_folder):
        QMessageBox.information(
            self, "Extraction Complete", "Frame extraction finished successfully!")
        self.progress_bar.setValue(100)
        # Emit signal to notify the main window
        self.output_folder_ready.emit(output_folder)
        self.progress_bar.setVisible(False)

    def on_processing_complete(self, message):
        """
        Handle completion of video processing.
        """
        QMessageBox.information(self, "Processing Complete", message)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)

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

    def process_all_videos(self):
        if not self.imported_videos:
            QMessageBox.warning(self, "No Videos",
                                "Please import videos before processing.")
            return

        # Initialize behavior agent
        try:
            from annolid.agents import behavior_agent
            agent = behavior_agent.initialize_agent()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to initialize agent: {str(e)}")
            return

        # Create and start the worker
        self.worker = ProcessVideosWorker(self.imported_videos, agent)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_complete)
        self.worker.error.connect(self.show_error)

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start the worker thread
        self.worker.start()
