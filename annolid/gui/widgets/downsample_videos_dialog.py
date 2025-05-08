import os
import subprocess
import cv2  # OpenCV to extract a video frame

from annolid.utils.videos import (
    collect_video_metadata,
    compress_and_rescale_video,
    save_metadata_to_csv,
)
from qtpy.QtWidgets import (
    QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QCheckBox, QSlider, QLineEdit, QMessageBox, QHBoxLayout, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
)
from qtpy.QtCore import Qt, QRectF, QPointF
from qtpy.QtGui import QPixmap, QPen, QColor
from annolid.data.videos import get_video_fps, get_video_files

# --- Cropping dialog classes ---


class CropFrameWidget(QGraphicsView):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Crop Region")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Load the image and add it to the scene.
        self.pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)

        self.crop_rect = None
        self.start_pos = QPointF()
        self.end_pos = QPointF()

        # Rectangle item to show user's selection.
        self.rect_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        self.rect_item.setPen(pen)
        self.scene.addItem(self.rect_item)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.end_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_pos, self.end_pos).normalized()
            self.rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_pos = self.mapToScene(event.pos())
            self.crop_rect = self.rect_item.rect()
        super().mouseReleaseEvent(event)

    def getCropRect(self):
        return self.crop_rect


class CropDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Frame")
        self.crop_widget = CropFrameWidget(image_path)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.crop_widget)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)

    def getCropCoordinates(self):
        rect = self.crop_widget.getCropRect()
        if rect:
            return int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        return None


# --- Main Widget ---

class VideoRescaleWidget(QDialog):
    """
    Widget for rescaling video files.

    Allows users to select input and output folders, specify a scale factor,
    adjust the FPS, optionally crop to an interesting region (using the first
    frame of the first video), choose to apply denoise, and then either rescale
    the videos or collect metadata.

    After processing:
      - A single CSV file (metadata.csv) is automatically saved in the output folder.
      - A Markdown README file is generated for each output video including video info,
        processing parameters (scale factor, FPS, crop parameters, denoise flag), and
        the executed FFmpeg command.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Rescaling')
        self.init_ui()

    def init_ui(self):
        # Input Folder
        self.input_folder_label = QLabel('Input Folder:')
        self.input_folder_button = QPushButton('Select Folder')
        self.input_folder_button.clicked.connect(self.select_input_folder)

        # Output Folder
        self.output_folder_label = QLabel('Output Folder:')
        self.output_folder_button = QPushButton('Select Folder')
        self.output_folder_button.clicked.connect(self.select_output_folder)

        # Scale Factor
        self.scale_factor_label = QLabel('Scale Factor:')
        self.scale_factor_slider = QSlider(Qt.Horizontal)
        self.scale_factor_slider.setMinimum(0)
        self.scale_factor_slider.setMaximum(100)
        self.scale_factor_slider.setValue(25)
        self.scale_factor_slider.setTickInterval(25)
        self.scale_factor_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_factor_slider.valueChanged.connect(
            self.update_scale_factor_from_slider)
        self.scale_factor_text = QLineEdit("0.5")
        self.scale_factor_text.editingFinished.connect(
            self.update_scale_factor_from_text)

        # FPS Option
        self.fps_label = QLabel('Frames Per Second (FPS):')
        self.fps_text = QLineEdit("FPS e.g. 29.97")
        self.override_fps_checkbox = QCheckBox(
            "Use specified FPS for all videos")

        # Codec Option (kept for future extension)
        self.codec_label = QLabel('Codec:')
        self.codec_text = QLineEdit("libx264")

        # Denoise Option
        self.denoise_checkbox = QCheckBox('Apply Denoise')

        # Crop Region Options
        self.crop_checkbox = QCheckBox('Enable Crop Region')
        self.crop_label = QLabel('Crop Region (x, y, width, height):')
        self.crop_x_text = QLineEdit()
        self.crop_x_text.setPlaceholderText("x")
        self.crop_y_text = QLineEdit()
        self.crop_y_text.setPlaceholderText("y")
        self.crop_width_text = QLineEdit()
        self.crop_width_text.setPlaceholderText("width")
        self.crop_height_text = QLineEdit()
        self.crop_height_text.setPlaceholderText("height")
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(self.crop_x_text)
        crop_layout.addWidget(self.crop_y_text)
        crop_layout.addWidget(self.crop_width_text)
        crop_layout.addWidget(self.crop_height_text)

        # New button: Extract first frame and crop interactively.
        self.crop_preview_button = QPushButton("Preview & Crop First Frame")
        self.crop_preview_button.clicked.connect(self.preview_and_crop)

        # Other Options
        self.rescale_checkbox = QCheckBox('Rescale Video')
        self.collect_only_checkbox = QCheckBox('Collect Metadata Only')

        # Run Button
        self.run_button = QPushButton('Run Rescaling')
        self.run_button.clicked.connect(self.run_rescaling)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.input_folder_label)
        layout.addWidget(self.input_folder_button)
        layout.addWidget(self.output_folder_label)
        layout.addWidget(self.output_folder_button)
        layout.addWidget(self.scale_factor_label)
        layout.addWidget(self.scale_factor_slider)
        layout.addWidget(self.scale_factor_text)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.fps_text)
        layout.addWidget(self.override_fps_checkbox)
        layout.addWidget(self.codec_label)
        layout.addWidget(self.codec_text)
        layout.addWidget(self.denoise_checkbox)
        layout.addWidget(self.crop_checkbox)
        layout.addWidget(self.crop_label)
        layout.addLayout(crop_layout)
        layout.addWidget(self.crop_preview_button)
        layout.addWidget(self.rescale_checkbox)
        layout.addWidget(self.collect_only_checkbox)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if folder:
            self.input_folder_label.setText(f'Input Folder: {folder}')
        video_files = get_video_files(folder)
        if video_files:
            first_video_path = os.path.join(folder, video_files[0])
            fps = get_video_fps(first_video_path)
            if fps:
                self.fps_text.setText(str(fps))
            else:
                print("[select_input_folder] Failed to extract FPS")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.output_folder_label.setText(f'Output Folder: {folder}')

    def update_scale_factor_from_slider(self):
        scale_factor = self.scale_factor_slider.value() / 100
        self.scale_factor_text.setText(str(scale_factor))

    def update_scale_factor_from_text(self):
        try:
            scale_factor = float(self.scale_factor_text.text())
            if 0.0 <= scale_factor <= 1.0:
                self.scale_factor_slider.setValue(int(scale_factor * 100))
            else:
                self.scale_factor_text.setText("Invalid Value")
        except ValueError:
            self.scale_factor_text.setText("Invalid Value")

    def preview_and_crop(self):
        """Extract the first frame from the first video in the input folder and open the crop dialog."""
        input_folder = self.input_folder_label.text().split(': ', 1)[-1]
        if not os.path.isdir(input_folder):
            QMessageBox.warning(
                self, 'Error', 'Please select a valid input folder.')
            return

        # Supported video extensions.
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
                            '.mpeg', '.mpg', '.m4v', '.mts')
        video_files = [f for f in os.listdir(input_folder)
                       if f.lower().endswith(video_extensions)]
        if not video_files:
            QMessageBox.warning(
                self, 'Error', 'No video files found in the input folder.')
            return

        first_video = os.path.join(input_folder, video_files[0])

        # Extract the first frame using OpenCV.
        cap = cv2.VideoCapture(first_video)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(
                self, 'Error', 'Unable to extract a frame from the first video.')
            return

        # Save the frame temporarily.
        temp_image_path = os.path.join(input_folder, "temp_crop_frame.jpg")
        cv2.imwrite(temp_image_path, frame)

        # Open the cropping dialog.
        crop_dialog = CropDialog(temp_image_path)
        if crop_dialog.exec_() == QDialog.Accepted:
            crop_coords = crop_dialog.getCropCoordinates()
            if crop_coords:
                crop_x, crop_y, crop_width, crop_height = crop_coords
                self.crop_x_text.setText(str(crop_x))
                self.crop_y_text.setText(str(crop_y))
                self.crop_width_text.setText(str(crop_width))
                self.crop_height_text.setText(str(crop_height))
                QMessageBox.information(
                    self, 'Crop Selected', f"Crop Region set to:\nx: {crop_x}, y: {crop_y}, width: {crop_width}, height: {crop_height}")
            else:
                QMessageBox.information(
                    self, 'No Crop', 'No crop region was selected.')
        # Clean up temporary image file.
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    def save_metadata_files(self, folder, scale_factor=None, fps=None,
                            apply_denoise=None, crop_params=None, command_log=None):
        """Save a CSV file (metadata.csv) and individual Markdown files for each video."""
        metadata_list = collect_video_metadata(folder)
        csv_all_path = os.path.join(folder, "metadata.csv")
        save_metadata_to_csv(metadata_list, csv_all_path)
        print(f"Saved metadata CSV: {csv_all_path}")

        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
                            '.mpeg', '.mpg', '.m4v', '.mts')
        for video_file in os.listdir(folder):
            if video_file.lower().endswith(video_extensions):
                base, _ = os.path.splitext(video_file)
                md_path = os.path.join(folder, base + '.md')
                with open(md_path, 'w') as f:
                    f.write(
                        f"# Metadata and Processing Info for {video_file}\n\n")
                    if scale_factor is not None:
                        f.write("**Processing Parameters:**\n")
                        f.write(f"- Scale Factor: {scale_factor}\n")
                        f.write(
                            f"- FPS: {fps if fps is not None else 'original per-video FPS'}\n")
                        f.write(f"- Apply Denoise: {apply_denoise}\n")
                        if crop_params is not None:
                            crop_x, crop_y, crop_width, crop_height = crop_params
                            f.write(
                                f"- Crop Region: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}\n")
                        f.write("\n")
                    if command_log is not None:
                        command_used = command_log.get(video_file, "N/A")
                        f.write("**FFmpeg Command:**\n")
                        f.write("```\n")
                        f.write(f"{command_used}\n")
                        f.write("```\n\n")
                    f.write("**Video Metadata:**\n")
                    video_metadata = [m for m in metadata_list if m.get(
                        'filename', '').lower() == video_file.lower()]
                    for entry in video_metadata:
                        for key, value in entry.items():
                            f.write(f"- **{key}**: {value}\n")
                        f.write("\n")
                print(f"Saved readme file: {md_path}")

    def run_rescaling(self):
        self.run_button.setEnabled(False)
        self.run_button.setText('Processing...')

        input_folder = self.input_folder_label.text().split(': ', 1)[-1]
        output_folder = self.output_folder_label.text().split(': ', 1)[-1]

        try:
            scale_factor = float(self.scale_factor_text.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid scale factor.')
            self.run_button.setEnabled(True)
            self.run_button.setText('Run Rescaling')
            return

        if self.override_fps_checkbox.isChecked():
            try:
                fps = float(self.fps_text.text())
            except ValueError:
                QMessageBox.warning(self, 'Error', 'Invalid FPS value.')
                self.run_button.setEnabled(True)
                self.run_button.setText('Run Rescaling')
                return
        else:
            fps = None  # Let each video retain its native FPS

        rescale = self.rescale_checkbox.isChecked()
        collect_only = self.collect_only_checkbox.isChecked()
        apply_denoise = self.denoise_checkbox.isChecked()

        crop_params = None
        if self.crop_checkbox.isChecked():
            try:
                crop_x = int(self.crop_x_text.text())
                crop_y = int(self.crop_y_text.text())
                crop_width = int(self.crop_width_text.text())
                crop_height = int(self.crop_height_text.text())
                crop_params = (crop_x, crop_y, crop_width, crop_height)
            except ValueError:
                QMessageBox.warning(
                    self, 'Error', 'Invalid crop parameters. Please enter integer values.')
                self.run_button.setEnabled(True)
                self.run_button.setText('Run Rescaling')
                return

        if collect_only:
            self.save_metadata_files(input_folder)
            QMessageBox.information(
                self, 'Done', 'Metadata collection is done.')
        elif rescale:
            command_log = compress_and_rescale_video(
                input_folder, output_folder, scale_factor, fps=fps,
                apply_denoise=apply_denoise,
                crop_x=crop_params[0] if crop_params else None,
                crop_y=crop_params[1] if crop_params else None,
                crop_width=crop_params[2] if crop_params else None,
                crop_height=crop_params[3] if crop_params else None)
            self.save_metadata_files(output_folder,
                                     scale_factor=scale_factor,
                                     fps=fps,
                                     apply_denoise=apply_denoise,
                                     crop_params=crop_params,
                                     command_log=command_log)
            QMessageBox.information(self, 'Done', 'Rescaling is done.')

        self.run_button.setEnabled(True)
        self.run_button.setText('Run Rescaling')


if __name__ == '__main__':
    app = QApplication([])
    widget = VideoRescaleWidget()
    widget.exec_()
    app.exec_()
