from __future__ import annotations

import sys

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from annolid.gui.widgets.video_processing_settings_widget import (
    VideoProcessingSettingsWidget,
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

        self.input_source_label = QLabel("Input Source: Select one video or one folder")
        self.platform_note_label = QLabel(
            "Windows note: progress is streamed during ffmpeg encoding."
        )
        self.platform_note_label.setWordWrap(True)
        self.platform_note_label.setStyleSheet("color: #6b7280; font-style: italic;")
        self.platform_note_label.setVisible(sys.platform.startswith("win"))
        self.input_video_button = QPushButton("Select Video")
        self.input_video_button.clicked.connect(self.workflow.select_input_video)
        self.input_folder_button = QPushButton("Select Folder")
        self.input_folder_button.clicked.connect(self.workflow.select_input_folder)
        self.input_selection_label = QLabel("No input selected")
        self.input_selection_label.setWordWrap(True)

        self.output_folder_label = QLabel("Output Folder:")
        self.output_folder_button = QPushButton("Select Folder")
        self.output_folder_button.clicked.connect(self.workflow.select_output_folder)

        self.settings_widget = VideoProcessingSettingsWidget(
            self, crop_preview_label="Preview & Crop First Frame"
        )
        self.settings_widget.connect_signals(
            scale_slider_changed=self.workflow.update_scale_factor_from_slider,
            scale_text_finished=self.workflow.update_scale_factor_from_text,
            fps_text_finished=self.workflow.update_fps_from_text,
            override_fps_toggled=self.workflow.toggle_override_fps_controls,
            denoise_toggled=self.workflow.update_summary_tab,
            auto_contrast_toggled=self.workflow.toggle_auto_contrast_controls,
            auto_contrast_strength_slider_changed=self.workflow.update_auto_contrast_strength_from_slider,
            auto_contrast_strength_text_finished=self.workflow.update_auto_contrast_strength_from_text,
            crop_toggled=self.workflow._set_crop_section_active,
            crop_preview_clicked=self.workflow.preview_and_crop,
        )
        self._bind_settings_aliases()

        self.per_video_section_label = QLabel("3) Per-Video Review")
        self.per_video_section_label.setStyleSheet(section_style)
        self.per_video_review_button = QPushButton(
            "Review Videos One by One (Folder Input)"
        )
        self.per_video_review_button.clicked.connect(
            self.workflow.configure_per_video_review
        )
        self.per_video_review_label = QLabel("Per-video review: none")
        self.per_video_review_label.setWordWrap(True)

        self.run_section_label = QLabel("4) Run")
        self.run_section_label.setStyleSheet(section_style)
        self.rescale_checkbox = QCheckBox("Rescale Video")
        self.rescale_checkbox.setChecked(True)
        self.collect_only_checkbox = QCheckBox("Collect Metadata Only")
        self.rescale_checkbox.toggled.connect(self.workflow.update_summary_tab)
        self.collect_only_checkbox.toggled.connect(self.workflow.update_summary_tab)

        self.run_button = QPushButton("Run Processing")
        self.run_button.clicked.connect(self.workflow.run_rescaling)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.workflow.cancel_running_job)
        self.cancel_button.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.tabs = QTabWidget()

        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.addWidget(self.input_source_label)
        input_layout.addWidget(self.platform_note_label)
        input_select_layout = QHBoxLayout()
        input_select_layout.addWidget(self.input_video_button)
        input_select_layout.addWidget(self.input_folder_button)
        input_layout.addLayout(input_select_layout)
        input_layout.addWidget(self.input_selection_label)
        input_layout.addWidget(self.output_folder_label)
        input_layout.addWidget(self.output_folder_button)
        input_layout.addStretch(1)

        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        processing_layout.addWidget(self.settings_widget)
        processing_layout.addWidget(self.per_video_section_label)
        processing_layout.addWidget(self.per_video_review_button)
        processing_layout.addWidget(self.per_video_review_label)
        processing_layout.addStretch(1)

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.summary_header_label = QLabel(
            "Review the current batch before running it."
        )
        self.summary_header_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_header_label)
        self.summary_input_label = QLabel("")
        self.summary_output_label = QLabel("")
        self.summary_processing_label = QLabel("")
        self.summary_overrides_label = QLabel("")
        self.summary_run_label = QLabel("")
        for label in (
            self.summary_input_label,
            self.summary_output_label,
            self.summary_processing_label,
            self.summary_overrides_label,
            self.summary_run_label,
        ):
            label.setWordWrap(True)
            summary_layout.addWidget(label)
        self.summary_refresh_button = QPushButton("Refresh Summary")
        self.summary_refresh_button.clicked.connect(self.workflow.update_summary_tab)
        summary_layout.addWidget(self.summary_refresh_button)
        summary_layout.addStretch(1)

        run_tab = QWidget()
        run_layout = QVBoxLayout(run_tab)
        run_layout.addWidget(self.run_section_label)
        run_layout.addWidget(self.rescale_checkbox)
        run_layout.addWidget(self.collect_only_checkbox)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.cancel_button)
        run_layout.addWidget(self.progress_label)
        run_layout.addWidget(self.progress_bar)
        run_layout.addStretch(1)

        self.tabs.addTab(input_tab, "Input / Output")
        self.tabs.addTab(processing_tab, "Processing")
        self.tabs.addTab(summary_tab, "Summary")
        self.tabs.addTab(run_tab, "Run")

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.workflow.toggle_auto_contrast_controls(False)
        self.workflow._set_crop_section_active(False)
        self.workflow.update_per_video_review_label()
        self.workflow.update_summary_tab()

    def _bind_settings_aliases(self) -> None:
        settings = self.settings_widget
        self.scale_factor_label = settings.scale_factor_label
        self.scale_factor_slider = settings.scale_factor_slider
        self.scale_factor_text = settings.scale_factor_text
        self.fps_label = settings.fps_label
        self.fps_text = settings.fps_text
        self.override_fps_checkbox = settings.override_fps_checkbox
        self.denoise_checkbox = settings.denoise_checkbox
        self.auto_contrast_checkbox = settings.auto_contrast_checkbox
        self.auto_contrast_strength_label = settings.auto_contrast_strength_label
        self.auto_contrast_strength_slider = settings.auto_contrast_strength_slider
        self.auto_contrast_strength_text = settings.auto_contrast_strength_text
        self.crop_section_label = settings.crop_section_label
        self.crop_checkbox = settings.crop_checkbox
        self.crop_label = settings.crop_label
        self.crop_x_text = settings.crop_x_text
        self.crop_y_text = settings.crop_y_text
        self.crop_width_text = settings.crop_width_text
        self.crop_height_text = settings.crop_height_text
        self.crop_preview_button = settings.crop_preview_button

    def update_summary_labels(self, lines: dict[str, str]) -> None:
        self.summary_input_label.setText(lines.get("input", ""))
        self.summary_output_label.setText(lines.get("output", ""))
        self.summary_processing_label.setText(lines.get("processing", ""))
        self.summary_overrides_label.setText(lines.get("overrides", ""))
        self.summary_run_label.setText(lines.get("run", ""))

    def closeEvent(self, event):  # noqa: N802
        if getattr(self.workflow, "_thread", None) is not None:
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication([])
    widget = VideoRescaleWidget()
    widget.exec_()
    app.exec_()
