from pathlib import Path
from qtpy import QtWidgets


class CanvasScreenshotWidget(QtWidgets.QWidget):
    """Widget responsible for saving a canvas screenshot."""

    def __init__(self, parent=None, canvas=None, here=None):
        super().__init__(parent)
        self.canvas = canvas
        self.here = here
        self.canvas_pixmap = None
        self.current_filename = None

    def save_canvas_screenshot(self, filename=None):
        """Handles saving the canvas content as a PNG image."""
        self.current_filename = filename
        self.canvas_pixmap = self.canvas.grab()
        if not self.canvas_pixmap:
            self._show_message_box(
                "No Canvas Content", "The canvas is empty. There is nothing to save."
            )
            return

        default_file_name = self._get_default_screenshot_filename()
        file_path = self._get_save_file_path(default_file_name)

        if file_path:
            self._save_pixmap_to_png(file_path)
            self._show_message_box(
                "Canvas Saved", f"Canvas image has been saved to: {file_path}"
            )

    def _get_default_screenshot_filename(self):
        """Generates a default filename for the screenshot."""
        if self.current_filename:
            file_path = Path(self.current_filename)
            return str(file_path.with_stem(f"{file_path.stem}_screenshot"))
        else:
            return str(Path(self.here.parent / "annotation") / "annolid_canvas.png")

    def _get_save_file_path(self, default_file_name):
        """Opens a file dialog for selecting the save location."""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Canvas Image", default_file_name, "PNG files (*.png)"
        )
        return file_path

    def _save_pixmap_to_png(self, file_path):
        """Saves the current canvas pixmap to a PNG file."""
        self.canvas_pixmap.save(file_path, "png")

    def _show_message_box(self, title, message):
        """Displays a message box."""
        QtWidgets.QMessageBox.information(self, title, message)
