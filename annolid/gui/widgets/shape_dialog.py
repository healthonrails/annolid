from qtpy import QtWidgets, QtCore
from annolid.utils.logger import logger


class ShapePropagationDialog(QtWidgets.QDialog):
    """
    A dialog that allows the user to select one of the shapes from the current
    canvas and specify a target frame. When confirmed, it will copy that shape
    into the JSON annotation files for all frames from the current one to the target.
    """

    def __init__(self, canvas, main_window, current_frame, max_frame, parent=None):
        super().__init__(parent)
        self.canvas = canvas           # Reference to the Canvas widget
        # Explicit reference to the main window (AnnolidWindow)
        self.main_window = main_window
        self.current_frame = current_frame
        self.max_frame = max_frame

        self.setWindowTitle("Propagate Shape to Future Frames")

        # Create a list widget to display available shapes.
        self.shape_list = QtWidgets.QListWidget(self)
        for shape in self.canvas.shapes:
            # Filter for the types you want (e.g., only polygons or masks)
            if shape.shape_type in ["polygon", "mask"]:
                item = QtWidgets.QListWidgetItem(shape.label)
                item.setData(QtCore.Qt.UserRole, shape)
                self.shape_list.addItem(item)

        # Spin box for selecting the target frame.
        self.frame_spin = QtWidgets.QSpinBox(self)
        self.frame_spin.setMinimum(current_frame + 1)
        self.frame_spin.setMaximum(max_frame)
        self.frame_spin.setValue(current_frame + 1)

        # Buttons for accepting or cancelling.
        self.ok_btn = QtWidgets.QPushButton("Propagate", self)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.ok_btn.clicked.connect(self.do_fill)
        self.cancel_btn.clicked.connect(self.reject)

        # Layout the widgets.
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Select a shape to propagate:"))
        layout.addWidget(self.shape_list)
        layout.addWidget(QtWidgets.QLabel("Propagate until frame:"))
        layout.addWidget(self.frame_spin)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def do_fill(self):
        # Get the selected shape.
        item = self.shape_list.currentItem()
        if not item:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a shape.")
            return

        selected_shape = item.data(QtCore.Qt.UserRole)
        target_frame = self.frame_spin.value()
        main_window = self.main_window

        logger.info(
            f"Propagating shape '{selected_shape.label}' from frame {self.current_frame+1} to {target_frame}")

        # Save the current frame so we can restore it later.
        original_frame = self.current_frame

        # Loop over future frames.
        for frame in range(self.current_frame + 1, target_frame + 1):
            # Set the target frame
            main_window.set_frame_number(frame)
            label_file = main_window._getLabelFile(main_window.filename)

            # Load existing shapes from the JSON file for this frame.
            try:
                existing_shapes = main_window.loadShapesFromFile(label_file)
            except Exception:
                existing_shapes = []

            # Create a deep copy of the selected shape (if possible),
            # ensuring that only this shape is propagated.
            new_shape = selected_shape.copy() if hasattr(
                selected_shape, "copy") else selected_shape

            # Append the new shape while keeping any shapes that already exist.
            existing_shapes.append(new_shape)

            # Save the updated shapes list to the JSON annotation file.
            # This ensures that whatever is displayed or stored remains unaltered
            # except for the appended shape.
            main_window.saveLabels(label_file, save_image_data=False)
            logger.info(f"Frame {frame} updated with the new shape.")

        # Optionally, restore the original frame display after propagation.
        main_window.set_frame_number(original_frame)

        QtWidgets.QMessageBox.information(
            self, "Propagation Complete", "The shape has been propagated to future frames.")
        self.accept()
