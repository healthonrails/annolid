import re
import glob
import os.path as osp
from qtpy import QtWidgets, QtCore
from annolid.utils.logger import logger
from annolid.gui.label_file import LabelFile


def shape_to_dict(shape):
    """
    Convert a shape object to a dictionary representation.
    If the shape is already a dict, return it unmodified.
    """
    if isinstance(shape, dict):
        return shape
    return {
        "label": shape.label,
        "points": [(pt.x(), pt.y()) for pt in shape.points],
        "group_id": shape.group_id,
        "shape_type": shape.shape_type,
        "flags": shape.flags,
        "description": shape.description,
        "mask": None if shape.mask is None else shape.mask,  # Adjust conversion as needed
        "visible": shape.visible,
    }


def get_future_frame_from_mask(dir_path, current_frame):
    """
    Look in the provided directory for PNG files following the pattern:
      *_<9-digit-frame-number>_mask.png
    Returns the smallest future frame number (greater than current_frame)
    found in these filenames. If none is found, returns None.
    """
    mask_pattern = osp.join(dir_path, "*_mask.png")
    mask_files = glob.glob(mask_pattern)
    frames = []
    for file in mask_files:
        basename = osp.basename(file)
        m = re.search(r"_(\d{9})_mask\.png$", basename)
        if m:
            try:
                frame_num = int(m.group(1))
                if frame_num > current_frame:
                    frames.append(frame_num)
            except ValueError:
                continue
    if frames:
        # Return the smallest future frame number found.
        return min(frames)
    return None


class ShapePropagationDialog(QtWidgets.QDialog):
    """
    A dialog that allows the user to select a shape from the current canvas and specify 
    a target frame. Depending on the selected action, the dialog will either propagate 
    (copy) the shape into the JSON annotation files for all frames from the current one 
    to the target, or delete the same shape from those frames. This version updates 
    the JSON file on disk directly, creating new files for future frames if needed.
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
        self.action_combo = QtWidgets.QComboBox(self)
        self.action_combo.addItems(["Propagate", "Delete"])

        # Spin box for selecting the target frame.
        self.frame_spin = QtWidgets.QSpinBox(self)
        self.frame_spin.setMinimum(current_frame + 1)

        # Determine the default value from a mask file if one exists.
        default_future_frame = None
        # Check for mask files in a directory – for example, video_results_folder or annotation_dir.
        folder_to_check = None
        if hasattr(main_window, "video_results_folder") and main_window.video_results_folder:
            folder_to_check = main_window.video_results_folder
        elif hasattr(main_window, "annotation_dir") and main_window.annotation_dir:
            folder_to_check = main_window.annotation_dir

        if folder_to_check:
            default_future_frame = get_future_frame_from_mask(
                folder_to_check, current_frame)

        # Fall back if no mask file was found.
        if default_future_frame is None:
            default_future_frame = current_frame + 100

        # Also, ensure we do not exceed the provided max_frame.
        if default_future_frame > max_frame:
            default_future_frame = max_frame

        self.frame_spin.setMaximum(max_frame)
        self.frame_spin.setValue(default_future_frame)

        # Buttons for applying or canceling.
        self.apply_btn = QtWidgets.QPushButton("Apply", self)
        self.cancel_btn = QtWidgets.QPushButton("Cancel", self)
        self.apply_btn.clicked.connect(self.do_action)
        self.cancel_btn.clicked.connect(self.reject)

        # Layout the widgets.
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Select a shape:"))
        layout.addWidget(self.shape_list)
        layout.addWidget(QtWidgets.QLabel("Select action:"))
        layout.addWidget(self.action_combo)
        layout.addWidget(QtWidgets.QLabel("Apply action until frame:"))
        layout.addWidget(self.frame_spin)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def do_action(self):
        # Get the selected shape.
        item = self.shape_list.currentItem()
        if not item:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a shape.")
            return

        selected_shape = item.data(QtCore.Qt.UserRole)
        target_frame = self.frame_spin.value()
        action = self.action_combo.currentText().lower()  # 'propagate' or 'delete'
        main_window = self.main_window

        logger.info(
            f"{action.capitalize()} shape '{selected_shape.label}' from frame {self.current_frame + 1} to {target_frame}"
        )

        original_frame = self.current_frame  # Save the current frame

        # Loop over each future frame.
        for frame in range(self.current_frame + 1, target_frame + 1):
            main_window.set_frame_number(frame)
            label_file = main_window._getLabelFile(main_window.filename)

            # If the label file does not exist, create a new LabelFile instance with default metadata.
            if not osp.exists(label_file):
                logger.info(
                    f"Label file {label_file} not found. Creating a new one.")
                lf = LabelFile()
                lf.filename = label_file
                # Use available metadata from main window. Adjust these attributes as needed.
                lf.imagePath = main_window.imagePath
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
                    continue

            # Get current shapes (assumed to be list of dicts).
            shapes = lf.shapes

            if action == "propagate":
                new_shape = selected_shape.copy() if hasattr(
                    selected_shape, "copy") else selected_shape
                new_shape_dict = shape_to_dict(new_shape)
                shapes.append(new_shape_dict)
            elif action == "delete":
                logger.info(
                    f"Deleting shape with label: {selected_shape.label}")
                shapes = [s for s in shapes if s.get(
                    "label") != selected_shape.label]

            # Update the label file's shapes.
            lf.shapes = shapes

            # Save the updated file, passing along available metadata.
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
