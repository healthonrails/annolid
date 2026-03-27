from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from annolid.gui.widgets.video_rescale_workflow import VideoRescaleWorkflow


class VideoRescaleWidget(QDialog):
    """Layout-only dialog for video downsampling and metadata export."""

    def __init__(self, initial_video_path=None):
        super().__init__()
        self.setWindowTitle("Downsample / Rescale Video")
        self.input_folder_path = ""
        self.input_video_path = ""
        self.output_folder_path = ""
        self.workflow = VideoRescaleWorkflow(self)
        self.init_ui()
        self.workflow.apply_initial_video(initial_video_path)

    def init_ui(self):
        section_style = "font-weight: 600; color: #3a6ea5;"

        self.input_section_label = QLabel("1) Input / Output")
        self.input_section_label.setStyleSheet(section_style)
        self.input_source_label = QLabel("Input Source: Select one video or one folder")
        self.input_video_button = QPushButton("Select Video")
        self.input_video_button.clicked.connect(self.workflow.select_input_video)
        self.input_folder_button = QPushButton("Select Folder")
        self.input_folder_button.clicked.connect(self.workflow.select_input_folder)
        self.input_selection_label = QLabel("No input selected")
        self.input_selection_label.setWordWrap(True)

        self.output_folder_label = QLabel("Output Folder:")
        self.output_folder_button = QPushButton("Select Folder")
        self.output_folder_button.clicked.connect(self.workflow.select_output_folder)

        self.processing_section_label = QLabel("2) Processing")
        self.processing_section_label.setStyleSheet(section_style)
        self.scale_factor_label = QLabel("Scale Factor:")
        self.scale_factor_slider = QSlider(Qt.Horizontal)
        self.scale_factor_slider.setMinimum(0)
        self.scale_factor_slider.setMaximum(100)
        self.scale_factor_slider.setValue(25)
        self.scale_factor_slider.setTickInterval(25)
        self.scale_factor_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_factor_slider.valueChanged.connect(
            self.workflow.update_scale_factor_from_slider
        )
        self.scale_factor_text = QLineEdit("0.5")
        self.scale_factor_text.editingFinished.connect(
            self.workflow.update_scale_factor_from_text
        )

        self.fps_label = QLabel("Frames Per Second (FPS):")
        self.fps_text = QLineEdit("FPS e.g. 29.97")
        self.override_fps_checkbox = QCheckBox("Use specified FPS for all videos")

        self.codec_label = QLabel("Codec:")
        self.codec_text = QLineEdit("libx264")

        self.denoise_checkbox = QCheckBox("Apply Denoise")
        self.auto_contrast_checkbox = QCheckBox("Auto Contrast Enhancement")
        self.auto_contrast_strength_label = QLabel("Auto Contrast Strength:")
        self.auto_contrast_strength_slider = QSlider(Qt.Horizontal)
        self.auto_contrast_strength_slider.setMinimum(0)
        self.auto_contrast_strength_slider.setMaximum(200)
        self.auto_contrast_strength_slider.setValue(100)
        self.auto_contrast_strength_slider.setTickInterval(25)
        self.auto_contrast_strength_slider.setTickPosition(QSlider.TicksBelow)
        self.auto_contrast_strength_slider.valueChanged.connect(
            self.workflow.update_auto_contrast_strength_from_slider
        )
        self.auto_contrast_strength_text = QLineEdit("1.0")
        self.auto_contrast_strength_text.editingFinished.connect(
            self.workflow.update_auto_contrast_strength_from_text
        )
        self.auto_contrast_checkbox.toggled.connect(
            self.workflow.toggle_auto_contrast_controls
        )

        self.crop_section_label = QLabel("3) Region Selection")
        self.crop_section_label.setStyleSheet(section_style)
        self.crop_checkbox = QCheckBox("Enable Crop Region")
        self.crop_label = QLabel("Crop Region (x, y, width, height):")
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

        self.crop_preview_button = QPushButton("Preview & Crop First Frame")
        self.crop_preview_button.clicked.connect(self.workflow.preview_and_crop)

        self.run_section_label = QLabel("4) Run")
        self.run_section_label.setStyleSheet(section_style)
        self.rescale_checkbox = QCheckBox("Rescale Video")
        self.rescale_checkbox.setChecked(True)
        self.collect_only_checkbox = QCheckBox("Collect Metadata Only")

        self.run_button = QPushButton("Run Processing")
        self.run_button.clicked.connect(self.workflow.run_rescaling)

        layout = QVBoxLayout()
        layout.addWidget(self.input_section_label)
        layout.addWidget(self.input_source_label)
        input_select_layout = QHBoxLayout()
        input_select_layout.addWidget(self.input_video_button)
        input_select_layout.addWidget(self.input_folder_button)
        layout.addLayout(input_select_layout)
        layout.addWidget(self.input_selection_label)
        layout.addWidget(self.output_folder_label)
        layout.addWidget(self.output_folder_button)
        layout.addWidget(self.processing_section_label)
        layout.addWidget(self.scale_factor_label)
        layout.addWidget(self.scale_factor_slider)
        layout.addWidget(self.scale_factor_text)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.fps_text)
        layout.addWidget(self.override_fps_checkbox)
        layout.addWidget(self.codec_label)
        layout.addWidget(self.codec_text)
        layout.addWidget(self.denoise_checkbox)
        layout.addWidget(self.auto_contrast_checkbox)
        layout.addWidget(self.auto_contrast_strength_label)
        layout.addWidget(self.auto_contrast_strength_slider)
        layout.addWidget(self.auto_contrast_strength_text)
        layout.addWidget(self.crop_section_label)
        layout.addWidget(self.crop_checkbox)
        layout.addWidget(self.crop_label)
        layout.addLayout(crop_layout)
        layout.addWidget(self.crop_preview_button)
        layout.addWidget(self.run_section_label)
        layout.addWidget(self.rescale_checkbox)
        layout.addWidget(self.collect_only_checkbox)
        layout.addWidget(self.run_button)
        self.setLayout(layout)
        self.workflow.toggle_auto_contrast_controls(False)


if __name__ == "__main__":
    app = QApplication([])
    widget = VideoRescaleWidget()
    widget.exec_()
    app.exec_()
