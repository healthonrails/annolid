from annolid.utils.videos import (collect_video_metadata,
                                  compress_and_rescale_video,
                                  save_metadata_to_csv)
from qtpy.QtWidgets import (QApplication, QDialog,
                            QLabel, QPushButton,
                            QVBoxLayout, QFileDialog,
                            QCheckBox, QSlider,
                            QLineEdit, QMessageBox
                            )
from qtpy.QtCore import Qt


class VideoRescaleWidget(QDialog):
    """
    Widget for rescaling video files.

    Allows users to select input and output folders, specify a scale factor,
    choose whether to rescale or downsample the video, and collect metadata.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Rescaling')

        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface.
        """
        self.input_folder_label = QLabel('Input Folder:')
        self.input_folder_button = QPushButton('Select Folder')
        self.input_folder_button.clicked.connect(self.select_input_folder)

        self.output_folder_label = QLabel('Output Folder:')
        self.output_folder_button = QPushButton('Select Folder')
        self.output_folder_button.clicked.connect(self.select_output_folder)

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

        self.rescale_checkbox = QCheckBox('Rescale Video')

        self.collect_only_checkbox = QCheckBox('Collect Metadata Only')

        self.run_button = QPushButton('Run Rescaling')
        self.run_button.clicked.connect(self.run_rescaling)

        layout = QVBoxLayout()
        layout.addWidget(self.input_folder_label)
        layout.addWidget(self.input_folder_button)
        layout.addWidget(self.output_folder_label)
        layout.addWidget(self.output_folder_button)
        layout.addWidget(self.scale_factor_label)
        layout.addWidget(self.scale_factor_slider)
        layout.addWidget(self.scale_factor_text)
        layout.addWidget(self.rescale_checkbox)
        layout.addWidget(self.collect_only_checkbox)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def select_input_folder(self):
        """
        Open a file dialog to select the input folder.
        """
        folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if folder:
            self.input_folder_label.setText(f'Input Folder: {folder}')

    def select_output_folder(self):
        """
        Open a file dialog to select the output folder.
        """
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.output_folder_label.setText(f'Output Folder: {folder}')

    def update_scale_factor_from_slider(self):
        """
        Update the scale factor when the slider is dragged.
        """
        scale_factor = self.scale_factor_slider.value() / 100
        self.scale_factor_text.setText(str(scale_factor))

    def update_scale_factor_from_text(self):
        """
        Update the slider when the text field is edited.
        """
        scale_factor_text = self.scale_factor_text.text()
        try:
            scale_factor = float(scale_factor_text)
            if 0.0 <= scale_factor <= 1.0:
                self.scale_factor_slider.setValue(int(scale_factor * 100))
            else:
                self.scale_factor_text.setText("Invalid Value")
        except ValueError:
            self.scale_factor_text.setText("Invalid Value")

    def run_rescaling(self):
        """
        Run the rescaling process based on user inputs.
        """
        # Disable the button during processing
        self.run_button.setEnabled(False)
        self.run_button.setText('Processing...')

        input_folder = self.input_folder_label.text().split(': ')[-1]
        output_folder = self.output_folder_label.text().split(': ')[-1]
        scale_factor = float(self.scale_factor_text.text())
        rescale = self.rescale_checkbox.isChecked()
        collect_only = self.collect_only_checkbox.isChecked()

        if collect_only:
            metadata = collect_video_metadata(input_folder)
            output_csv, _ = QFileDialog.getSaveFileName(
                self, 'Select Output CSV')
            if output_csv:
                save_metadata_to_csv(metadata, output_csv)
                QMessageBox.information(
                    self, 'Done', 'Metadata collection is done.')
        elif rescale:
            compress_and_rescale_video(
                input_folder, output_folder, scale_factor)
            if output_folder:
                metadata = collect_video_metadata(output_folder)
                output_csv, _ = QFileDialog.getSaveFileName(
                    self, 'Select Output CSV')
                if output_csv:
                    save_metadata_to_csv(metadata, output_csv)
                    QMessageBox.information(self, 'Done', 'Rescaling is done.')

        # Enable the button and change its text back to original
        self.run_button.setEnabled(True)
        self.run_button.setText('Run Rescaling')


if __name__ == '__main__':
    app = QApplication([])
    widget = VideoRescaleWidget()
    widget.exec_()
    app.exec_()
