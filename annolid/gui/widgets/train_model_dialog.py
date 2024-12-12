from pathlib import Path
from qtpy import QtCore, QtWidgets


class TrainModelDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs):
        super(TrainModelDialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Train Models")

        # Initialize default parameters
        self.batch_size = 8
        self.algo = 'MaskRCNN'
        self.config_file = None
        self.out_dir = None
        self.max_iterations = 2000
        self.trained_model = None
        self.image_size = 640
        self.epochs = 100
        self.yolo_model_file = 'yolo11n-seg.pt'  # Generalized for all models

        # UI Components
        self.create_radio_buttons()
        self.create_sliders()
        self.create_file_selection_widgets()
        self.create_button_box()

        # Layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.groupBox)
        main_layout.addWidget(self.label1)
        main_layout.addWidget(self.slider)
        main_layout.addWidget(self.image_size_groupbox)
        main_layout.addWidget(self.epoch_groupbox)
        main_layout.addWidget(self.groupBoxFiles)
        main_layout.addWidget(self.max_iter_label)
        main_layout.addWidget(self.max_iter_slider)
        main_layout.addWidget(self.groupBoxModelFiles)
        main_layout.addWidget(self.groupBoxOutDir)
        main_layout.addWidget(self.buttonbox)
        self.setLayout(main_layout)

        # Initial setup based on default algorithm
        self.update_ui_for_algorithm()
        self.show()

    def create_radio_buttons(self):
        self.groupBox = QtWidgets.QGroupBox("Please choose a model")
        hbox_layout = QtWidgets.QHBoxLayout()

        self.radio_btn1 = QtWidgets.QRadioButton("MaskRCNN")
        self.radio_btn1.setChecked(True)
        self.radio_btn1.toggled.connect(self.on_radio_button_checked)
        hbox_layout.addWidget(self.radio_btn1)

        self.radio_btn2 = QtWidgets.QRadioButton("YOLACT")
        self.radio_btn2.toggled.connect(self.on_radio_button_checked)
        hbox_layout.addWidget(self.radio_btn2)

        self.radio_btn3 = QtWidgets.QRadioButton("YOLO")
        self.radio_btn3.toggled.connect(self.on_radio_button_checked)
        hbox_layout.addWidget(self.radio_btn3)

        self.groupBox.setLayout(hbox_layout)

    def create_sliders(self):
        # Batch Size Slider
        self.label1 = QtWidgets.QLabel(f"Batch Size: {self.batch_size}")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(1, 128)
        self.slider.setValue(self.batch_size)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.on_batch_size_slider_change)

        # Max Iterations Slider
        self.max_iter_label = QtWidgets.QLabel(
            f"Max Iterations: {self.max_iterations} (Optional)")
        self.max_iter_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_iter_slider.setRange(100, 20000)
        self.max_iter_slider.setValue(self.max_iterations)
        self.max_iter_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.max_iter_slider.setTickInterval(100)
        self.max_iter_slider.setSingleStep(100)
        self.max_iter_slider.valueChanged.connect(
            self.on_max_iter_slider_change)

        # Image Size Slider
        self.image_size_groupbox = QtWidgets.QGroupBox("Image Size")
        self.image_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.image_size_slider.setRange(320, 1280)
        self.image_size_slider.setValue(self.image_size)
        self.image_size_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.image_size_slider.setTickInterval(64)
        self.image_size_slider.valueChanged.connect(
            self.on_image_size_slider_change)
        self.image_size_label = QtWidgets.QLabel(
            f"Image Size: {self.image_size} px")
        image_size_hbox_layout = QtWidgets.QHBoxLayout()
        image_size_hbox_layout.addWidget(self.image_size_slider)
        image_size_hbox_layout.addWidget(self.image_size_label)
        self.image_size_groupbox.setLayout(image_size_hbox_layout)

        # Epochs Slider
        self.epoch_groupbox = QtWidgets.QGroupBox("Epochs")
        self.epoch_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.epoch_slider.setRange(1, 300)
        self.epoch_slider.setValue(self.epochs)
        self.epoch_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.epoch_slider.setTickInterval(1)
        self.epoch_slider.valueChanged.connect(self.on_epoch_slider_change)
        self.epoch_label = QtWidgets.QLabel(f"Epochs: {self.epochs}")
        epoch_hbox_layout = QtWidgets.QHBoxLayout()
        epoch_hbox_layout.addWidget(self.epoch_slider)
        epoch_hbox_layout.addWidget(self.epoch_label)
        self.epoch_groupbox.setLayout(epoch_hbox_layout)

    def create_file_selection_widgets(self):
        # Config File Selection
        self.groupBoxFiles = QtWidgets.QGroupBox("Config File (YAML for YOLO)")
        self.configFileLineEdit = QtWidgets.QLineEdit(self)
        self.configFileButton = QtWidgets.QPushButton('Open', self)
        self.configFileButton.clicked.connect(
            self.on_config_file_button_clicked)
        config_hbox_layout = QtWidgets.QHBoxLayout()
        config_hbox_layout.addWidget(self.configFileLineEdit)
        config_hbox_layout.addWidget(self.configFileButton)
        self.groupBoxFiles.setLayout(config_hbox_layout)

        # Trained Model File Selection (Optional)
        self.groupBoxModelFiles = QtWidgets.QGroupBox(
            "Trained Model File (.pt, Optional)")
        self.trainedModelLineEdit = QtWidgets.QLineEdit(self)
        self.trainedModelButton = QtWidgets.QPushButton('Open', self)
        self.trainedModelButton.clicked.connect(
            self.on_trained_model_button_clicked)
        model_hbox_layout = QtWidgets.QHBoxLayout()
        model_hbox_layout.addWidget(self.trainedModelLineEdit)
        model_hbox_layout.addWidget(self.trainedModelButton)
        self.groupBoxModelFiles.setLayout(model_hbox_layout)

        # Output Directory Selection (Optional)
        self.groupBoxOutDir = QtWidgets.QGroupBox(
            "Output Directory (Optional)")
        self.outDirLineEdit = QtWidgets.QLineEdit(self)
        self.outDirButton = QtWidgets.QPushButton('Select', self)
        self.outDirButton.clicked.connect(self.on_out_dir_button_clicked)
        out_dir_hbox_layout = QtWidgets.QHBoxLayout()
        out_dir_hbox_layout.addWidget(self.outDirLineEdit)
        out_dir_hbox_layout.addWidget(self.outDirButton)
        self.groupBoxOutDir.setLayout(out_dir_hbox_layout)

    def create_button_box(self):
        qbtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonbox = QtWidgets.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

    def on_batch_size_slider_change(self):
        self.batch_size = self.slider.value()
        self.label1.setText(f"Batch Size: {self.batch_size}")

    def on_max_iter_slider_change(self):
        self.max_iterations = self.max_iter_slider.value()
        self.max_iter_label.setText(
            f"Max Iterations: {self.max_iterations} (Optional)")

    def on_image_size_slider_change(self):
        self.image_size = self.image_size_slider.value()
        self.image_size_label.setText(f"Image Size: {self.image_size} px")

    def on_epoch_slider_change(self):
        self.epochs = self.epoch_slider.value()
        self.epoch_label.setText(f"Epochs: {self.epochs}")

    def on_config_file_button_clicked(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if self.algo == "YOLO":
            file_dialog.setNameFilter("YAML files (*.yaml)")
        else:
            file_dialog.setNameFilter("All files (*)")

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.config_file = selected_files[0]
                self.configFileLineEdit.setText(self.config_file)

    def on_trained_model_button_clicked(self):
        self.trained_model, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Open Trained Model File",
            directory=str(Path()),
            filter="PyTorch Models (*.pt *.pth)"
        )
        if self.trained_model:
            self.trainedModelLineEdit.setText(self.trained_model)

    def on_out_dir_button_clicked(self):
        self.out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if self.out_dir:
            self.outDirLineEdit.setText(self.out_dir)

    def on_radio_button_checked(self):
        radio_btn = self.sender()
        if radio_btn.isChecked():
            self.algo = radio_btn.text()
            self.update_ui_for_algorithm()

    def update_ui_for_algorithm(self):
        # Set visibility and enabled state of UI elements based on selected algorithm
        yolo_selected = self.algo == "YOLO"
        mask_rcnn_selected = self.algo == "MaskRCNN"

        self.label1.setVisible(not yolo_selected)
        self.slider.setVisible(not yolo_selected)
        self.max_iter_label.setVisible(mask_rcnn_selected)
        self.max_iter_slider.setVisible(mask_rcnn_selected)
        self.groupBoxFiles.setTitle(
            "Config File (YAML for YOLO)" if yolo_selected else "Config File")
        self.configFileButton.setText("Open YAML" if yolo_selected else "Open")
        self.image_size_groupbox.setVisible(yolo_selected)
        self.epoch_groupbox.setVisible(yolo_selected)

    def accept(self):  # Override accept method for training logic
        # Validate inputs before accepting
        if self.algo == "YOLO":
            if not self.config_file:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "Please select a config file.")
                return
            super().accept()
            self.close()
        elif self.algo == "MaskRCNN":
            # Placeholder for Mask R-CNN training
            QtWidgets.QMessageBox.information(self, "Training Info",
                                              f"Mask R-CNN Training Selected:\n"
                                              f"Config: {self.config_file}\n"
                                              f"Batch Size: {self.batch_size}\n"
                                              f"Max Iterations: {self.max_iterations}\n"
                                              f"Trained Model: {self.trained_model}\n"
                                              f"Output Directory: {self.out_dir}")
            super().accept()
        elif self.algo == "YOLACT":
            # Placeholder for YOLACT Training
            QtWidgets.QMessageBox.information(self, "Training Info",
                                              f"YOLACT Training Selected:\n"
                                              f"Config: {self.config_file}\n"
                                              f"Batch Size: {self.batch_size}\n"
                                              f"Trained Model: {self.trained_model}\n"
                                              f"Output Directory: {self.out_dir}")
            super().accept()
