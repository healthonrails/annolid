import os
from annolid.utils.videos import (collect_video_metadata,
                                  compress_and_rescale_video,
                                  save_metadata_to_csv)
from qtpy.QtWidgets import (QApplication, QDialog,
                            QLabel, QPushButton,
                            QVBoxLayout, QFileDialog,
                            QCheckBox, QSlider,
                            QLineEdit, QMessageBox)
from qtpy.QtCore import Qt


class VideoRescaleWidget(QDialog):
    """
    Widget for rescaling video files.

    Allows users to select input and output folders, specify a scale factor,
    set the frames per second (FPS), choose whether to apply denoise, and
    either rescale the videos or collect metadata.

    After processing:
      - A single CSV file (metadata.csv) is automatically saved in the output folder.
      - A Markdown (readme) file is generated for each output video that includes:
            • Video metadata,
            • Processing parameters,
            • And the FFmpeg command executed.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Rescaling')
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
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

        self.scale_factor_text = QLineEdit()
        self.scale_factor_text.setText("0.25")  # Default scale factor
        self.scale_factor_text.editingFinished.connect(
            self.update_scale_factor_from_text)

        # FPS Option
        self.fps_label = QLabel('Frames Per Second (FPS):')
        self.fps_text = QLineEdit()
        self.fps_text.setText("30")  # Default FPS value

        # Codec Option (kept for future extension)
        self.codec_label = QLabel('Codec:')
        self.codec_text = QLineEdit()
        self.codec_text.setText("libx264")  # Default codec

        # Denoise Option
        self.denoise_checkbox = QCheckBox('Apply Denoise')

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
        layout.addWidget(self.codec_label)
        layout.addWidget(self.codec_text)
        layout.addWidget(self.denoise_checkbox)
        layout.addWidget(self.rescale_checkbox)
        layout.addWidget(self.collect_only_checkbox)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def select_input_folder(self):
        """Open a file dialog to select the input folder."""
        folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if folder:
            self.input_folder_label.setText(f'Input Folder: {folder}')

    def select_output_folder(self):
        """Open a file dialog to select the output folder."""
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.output_folder_label.setText(f'Output Folder: {folder}')

    def update_scale_factor_from_slider(self):
        """Update the scale factor when the slider is dragged."""
        scale_factor = self.scale_factor_slider.value() / 100
        self.scale_factor_text.setText(str(scale_factor))

    def update_scale_factor_from_text(self):
        """Update the slider when the text field is edited."""
        scale_factor_text = self.scale_factor_text.text()
        try:
            scale_factor = float(scale_factor_text)
            if 0.0 <= scale_factor <= 1.0:
                self.scale_factor_slider.setValue(int(scale_factor * 100))
            else:
                self.scale_factor_text.setText("Invalid Value")
        except ValueError:
            self.scale_factor_text.setText("Invalid Value")

    def save_metadata_files(self, folder, scale_factor=None, fps=None, apply_denoise=None, command_log=None):
        """
        Collect metadata from the given folder and automatically save:
          - A single CSV file ("metadata.csv") for all videos.
          - A Markdown file (readme) for each video containing video info,
            processing parameters, and the FFmpeg command executed.
        """
        # Get metadata for all videos in the folder.
        metadata_list = collect_video_metadata(folder)

        # Save one CSV file for all videos.
        csv_all_path = os.path.join(folder, "metadata.csv")
        save_metadata_to_csv(metadata_list, csv_all_path)
        print(f"Saved metadata CSV for all videos: {csv_all_path}")

        # Define supported video extensions.
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
                            '.mpeg', '.mpg', '.m4v', '.mts')

        # Create a Markdown readme file for each video.
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
                        f.write(f"- FPS: {fps}\n")
                        f.write(f"- Apply Denoise: {apply_denoise}\n\n")
                    if command_log is not None:
                        # Look up the command using the exact output filename.
                        command_used = command_log.get(video_file, "N/A")
                        f.write("**FFmpeg Command:**\n")
                        f.write("```\n")
                        f.write(f"{command_used}\n")
                        f.write("```\n\n")
                    f.write("**Video Metadata:**\n")
                    # Filter metadata entries matching this video file.
                    video_metadata = [m for m in metadata_list if m.get(
                        'filename', '').lower() == video_file.lower()]
                    for entry in video_metadata:
                        for key, value in entry.items():
                            f.write(f"- **{key}**: {value}\n")
                        f.write("\n")
                print(f"Saved readme file: {md_path}")

    def run_rescaling(self):
        """Run the rescaling process based on user inputs and auto-save metadata files."""
        # Disable the button during processing.
        self.run_button.setEnabled(False)
        self.run_button.setText('Processing...')

        # Get input and output folders.
        input_folder = self.input_folder_label.text().split(': ', 1)[-1]
        output_folder = self.output_folder_label.text().split(': ', 1)[-1]

        try:
            scale_factor = float(self.scale_factor_text.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid scale factor.')
            self.run_button.setEnabled(True)
            self.run_button.setText('Run Rescaling')
            return

        try:
            fps = int(self.fps_text.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid FPS value.')
            self.run_button.setEnabled(True)
            self.run_button.setText('Run Rescaling')
            return

        # Read processing options.
        rescale = self.rescale_checkbox.isChecked()
        collect_only = self.collect_only_checkbox.isChecked()
        apply_denoise = self.denoise_checkbox.isChecked()

        if collect_only:
            # Collect metadata from the input folder and automatically save readme files.
            self.save_metadata_files(input_folder)
            QMessageBox.information(
                self, 'Done', 'Metadata collection is done.')
        elif rescale:
            # Process videos with rescaling, FPS adjustment, and optional denoise.
            command_log = compress_and_rescale_video(
                input_folder, output_folder, scale_factor, fps=fps, apply_denoise=apply_denoise)
            # After processing, automatically save metadata (CSV and Markdown) in the output folder.
            self.save_metadata_files(output_folder,
                                     scale_factor=scale_factor,
                                     fps=fps,
                                     apply_denoise=apply_denoise,
                                     command_log=command_log)
            QMessageBox.information(self, 'Done', 'Rescaling is done.')

        # Re-enable the button.
        self.run_button.setEnabled(True)
        self.run_button.setText('Run Rescaling')


if __name__ == '__main__':
    app = QApplication([])
    widget = VideoRescaleWidget()
    widget.exec_()
    app.exec_()
