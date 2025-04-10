import os.path as osp
from qtpy import QtWidgets, QtCore
from annolid.utils.logger import logger
from annolid.gui.label_file import LabelFile
from annolid.annotation.polygons import are_polygons_close_or_overlap
from annolid.utils.shapes import shape_to_dict
from annolid.utils.files import get_future_frame_from_mask


class ShapePropagationDialog(QtWidgets.QDialog):
    """
    A dialog that allows the user to select a shape from the current canvas and specify
    a target frame. Depending on the selected action, the dialog will either propagate
    (copy) the shape into the JSON annotation files for all frames from the current one
    to the target, delete the same shape from those frames, or define a proximity-based
    event rule that automatically flags frames where a spatial relationship between annotated
    shapes is met.
    """

    def __init__(self, canvas, main_window, current_frame, max_frame, parent=None):
        super().__init__(parent)
        self.canvas = canvas           # Reference to the Canvas widget
        self.main_window = main_window  # Reference to the main window
        self.current_frame = current_frame
        self.max_frame = max_frame

        self.setWindowTitle("Shape Action on Future Frames")

        # Create a list widget to display available shapes.
        self.shape_list = QtWidgets.QListWidget(self)
        for shape in self.canvas.shapes:
            if shape.shape_type in ["polygon", "mask"]:
                item = QtWidgets.QListWidgetItem(shape.label)
                item.setData(QtCore.Qt.UserRole, shape)
                self.shape_list.addItem(item)

        # Drop-down for selecting the action.
        # Now includes "Define Proximity Event" as an additional action.
        self.action_combo = QtWidgets.QComboBox(self)
        self.action_combo.addItems(
            ["Propagate", "Delete", "Define Proximity Event"])
        self.action_combo.currentIndexChanged.connect(
            self.update_action_fields)

        # Spin box for selecting the target frame (used in propagate and delete actions).
        self.frame_spin_label = QtWidgets.QLabel("Apply action until frame:")
        self.frame_spin = QtWidgets.QSpinBox(self)
        self.frame_spin.setMinimum(current_frame + 1)

        # Determine the default value from a mask file if one exists.
        default_future_frame = None
        folder_to_check = None
        if hasattr(main_window, "video_results_folder") and main_window.video_results_folder:
            folder_to_check = main_window.video_results_folder
        elif hasattr(main_window, "annotation_dir") and main_window.annotation_dir:
            folder_to_check = main_window.annotation_dir

        if folder_to_check:
            default_future_frame = get_future_frame_from_mask(
                folder_to_check, current_frame)

        if default_future_frame is None:
            default_future_frame = current_frame + 100

        if default_future_frame > max_frame:
            default_future_frame = max_frame

        self.frame_spin.setMaximum(max_frame)
        self.frame_spin.setToolTip(f"Maximum frame: {max_frame}")
        self.frame_spin.lineEdit().setPlaceholderText(str(max_frame))

        # --- New UI Elements for "Define Proximity Event" ---
        self.event_widget = QtWidgets.QWidget(self)
        event_layout = QtWidgets.QVBoxLayout(self.event_widget)

        self.target_group_combo = QtWidgets.QComboBox(self)
        shape_labels = sorted(
            {shape.label for shape in self.canvas.shapes if shape.shape_type in ["polygon", "mask"]})
        for label in shape_labels:
            self.target_group_combo.addItem(label)
        self.target_group_combo.addItem("All Others")
        event_layout.addWidget(QtWidgets.QLabel("Select Target Group:"))
        event_layout.addWidget(self.target_group_combo)

        self.event_name_line = QtWidgets.QLineEdit(self)
        event_layout.addWidget(QtWidgets.QLabel("Event Name:"))
        event_layout.addWidget(self.event_name_line)

        self.proximity_threshold_spin = QtWidgets.QSpinBox(self)
        self.proximity_threshold_spin.setMinimum(1)
        self.proximity_threshold_spin.setMaximum(10000)
        self.proximity_threshold_spin.setValue(50)
        event_layout.addWidget(QtWidgets.QLabel(
            "Proximity Threshold (pixels):"))
        event_layout.addWidget(self.proximity_threshold_spin)

        self.rule_type_combo = QtWidgets.QComboBox(self)
        self.rule_type_combo.addItems(["any", "all"])
        event_layout.addWidget(QtWidgets.QLabel("Rule Type:"))
        event_layout.addWidget(self.rule_type_combo)

        self.event_start_frame_spin = QtWidgets.QSpinBox(self)
        self.event_start_frame_spin.setMinimum(current_frame + 1)
        self.event_start_frame_spin.setMaximum(max_frame)
        self.event_start_frame_spin.setValue(current_frame + 1)
        self.event_end_frame_spin = QtWidgets.QSpinBox(self)
        self.event_end_frame_spin.setMinimum(current_frame + 1)
        self.event_end_frame_spin.setMaximum(max_frame)
        self.event_end_frame_spin.setValue(default_future_frame)
        event_layout.addWidget(QtWidgets.QLabel("Event Start Frame:"))
        event_layout.addWidget(self.event_start_frame_spin)
        event_layout.addWidget(QtWidgets.QLabel("Event End Frame:"))
        event_layout.addWidget(self.event_end_frame_spin)

        self.event_widget.hide()

        self.apply_btn = QtWidgets.QPushButton("Apply", self)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.apply_btn.clicked.connect(self.do_action)
        self.cancel_btn.clicked.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Select a shape:"))
        layout.addWidget(self.shape_list)
        layout.addWidget(QtWidgets.QLabel("Select action:"))
        layout.addWidget(self.action_combo)
        layout.addWidget(self.frame_spin_label)
        layout.addWidget(self.frame_spin)
        layout.addWidget(self.event_widget)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def update_action_fields(self):
        """Toggle the visibility of input fields based on the selected action."""
        current_action = self.action_combo.currentText().lower()
        if current_action == "define proximity event":
            self.frame_spin_label.hide()
            self.frame_spin.hide()
            self.event_widget.show()
        else:
            self.frame_spin_label.show()
            self.frame_spin.show()
            self.event_widget.hide()

    def compute_centroid(self, shape):
        """Compute the centroid of a shape (works for object or dict with 'points')."""
        if isinstance(shape, dict):
            pts = shape.get("points", [])
        else:
            pts = [(pt.x(), pt.y()) for pt in shape.points]
        if not pts:
            return (0, 0)
        xs, ys = zip(*pts)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def load_or_create_label_file(self, label_file):
        """
        Load a label file if it exists; otherwise, create a new one with default metadata.
        Returns a LabelFile instance.
        """
        main_window = self.main_window
        if not osp.exists(label_file):
            logger.info(
                f"Label file {label_file} not found. Creating a new one.")
            lf = LabelFile()
            lf.filename = label_file
            try:
                lf.imageHeight = main_window.image.height()
                lf.imageWidth = main_window.image.width()
            except Exception:
                lf.imageHeight = None
                lf.imageWidth = None
            lf.shapes = []
            lf.caption = ""
            lf.flags = {}
            lf.otherData = {}
        else:
            try:
                lf = LabelFile(label_file, is_video_frame=True)
            except Exception as e:
                logger.error(f"Error loading label file {label_file}: {e}")
                return None
        return lf

    def do_action(self):
        item = self.shape_list.currentItem()
        if not item:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a shape.")
            return

        selected_shape = item.data(QtCore.Qt.UserRole)
        # 'propagate', 'delete', or 'define proximity event'
        action = self.action_combo.currentText().lower()
        main_window = self.main_window

        # If action is "delete", warn the user about irreversibility.
        if action == "delete":
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm Deletion",
                "Are you sure you want to delete this shape? This action cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                # User decided not to delete, so exit the method.
                return

        if action in ["propagate", "delete"]:
            logger.info(
                f"{action.capitalize()} shape '{selected_shape.label}' from frame {self.current_frame + 1} to {self.frame_spin.value()}"
            )
            original_frame = self.current_frame
            # if delete start from the current frame
            if action == "delete":
                self.current_frame -= 1

            for frame in range(self.current_frame + 1, self.frame_spin.value() + 1):
                main_window.set_frame_number(frame)
                label_file = main_window._getLabelFile(main_window.filename)
                lf = self.load_or_create_label_file(label_file)
                if lf is None:
                    continue

                shapes = lf.shapes

                if action == "propagate":
                    new_shape = selected_shape.copy() if hasattr(
                        selected_shape, "copy") else selected_shape
                    new_shape_dict = shape_to_dict(new_shape)
                    shapes.append(new_shape_dict)
                elif action == "delete":
                    # Convert the shape to a dictionary for detailed logging.
                    shape_details = shape_to_dict(selected_shape) if hasattr(
                        selected_shape, "points") else selected_shape
                    logger.info(
                        f"Deleting shape with label: {selected_shape.label} | Details: {shape_details}")
                    shapes = [s for s in shapes if s.get(
                        "label") != selected_shape.label]

                lf.shapes = shapes

                lf.save(
                    label_file,
                    lf.shapes,
                    lf.imagePath,
                    getattr(lf, "imageHeight", None),
                    getattr(lf, "imageWidth", None),
                    lf.imageData,
                    lf.otherData,
                    lf.flags,
                    lf.caption,
                )
                logger.info(f"Frame {frame} updated with action: {action}.")

            main_window.set_frame_number(original_frame)
            QtWidgets.QMessageBox.information(
                self, f"{action.capitalize()} Complete",
                f"The shape has been {action}ed in future frames."
            )
            self.accept()

        elif action == "define proximity event":
            target_group = self.target_group_combo.currentText()
            event_name = self.event_name_line.text().strip()
            if not event_name:
                QtWidgets.QMessageBox.warning(
                    self, "Missing Input", "Please enter an event name.")
                return
            proximity_threshold = self.proximity_threshold_spin.value()
            rule_type = self.rule_type_combo.currentText().lower()  # "any" or "all"
            event_start_frame = self.event_start_frame_spin.value()
            event_end_frame = self.event_end_frame_spin.value()
            if event_start_frame > event_end_frame:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Frame Range", "Event start frame must be less than or equal to event end frame.")
                return

            frames_updated = 0
            # Evaluate the proximity event for each frame in the given range.
            for frame in range(event_start_frame, event_end_frame + 1):
                main_window.set_frame_number(frame)
                label_file = main_window._getLabelFile(main_window.filename)
                lf = self.load_or_create_label_file(label_file)
                if lf is None:
                    continue

                # Filter target shapes from the current label file.
                target_shapes = []
                for shape in lf.shapes:
                    if target_group.lower() == "all others":
                        if shape.get("label") != selected_shape.label:
                            target_shapes.append(shape)
                    else:
                        if shape.get("label") == target_group:
                            target_shapes.append(shape)

                # Use Shapely to check proximity or overlap.
                if target_shapes:
                    proximity_results = [
                        are_polygons_close_or_overlap(
                            selected_shape, target, proximity_threshold)
                        for target in target_shapes
                    ]
                    if rule_type == "any":
                        triggered = any(proximity_results)
                    else:  # rule_type == "all"
                        triggered = all(proximity_results)
                else:
                    triggered = False

                if triggered:
                    lf.flags[event_name] = True
                    # only keep the event flag is true
                    lf.flags = {k: v for k, v in lf.flags.items() if v}
                    frames_updated += 1
                    lf.save(
                        label_file,
                        lf.shapes,
                        lf.imagePath,
                        getattr(lf, "imageHeight", None),
                        getattr(lf, "imageWidth", None),
                        lf.imageData,
                        lf.otherData,
                        lf.flags,
                        lf.caption,
                    )
                    logger.info(
                        f"Frame {frame} updated with event flag: {event_name}.")

            main_window.set_frame_number(self.current_frame)
            QtWidgets.QMessageBox.information(
                self, "Define Proximity Event Complete",
                f"Event '{event_name}' applied to {frames_updated} frame(s) where the condition was met."
            )
            self.accept()
