from qtpy import QtWidgets
from qtpy.QtWidgets import QVBoxLayout, QTextEdit
from qtpy.QtCore import Signal


class CaptionWidget(QtWidgets.QWidget):
    """A widget for editing and displaying image captions."""
    captionChanged = Signal(str)  # Signal emitted when caption changes
    # Signal emitted when a character is inserted
    charInserted = Signal(str)
    charDeleted = Signal(str)      # Signal emitted when a character is deleted

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()  # Create a QTextEdit
        self.layout.addWidget(self.text_edit)

        # Store previous text for comparison
        self.previous_text = ""

        # Connect signals and slots
        self.text_edit.textChanged.connect(
            self.emit_caption_changed)  # Emit signal on text change
        self.text_edit.textChanged.connect(
            self.monitor_text_change)    # Monitor text changes

        self.setLayout(self.layout)

    def monitor_text_change(self):
        """Monitor for text changes and emit signals for insertions and deletions."""
        current_text = self.text_edit.toPlainText()
        if self.previous_text == "":
            self.previous_text = current_text
            return

        # Compare current and previous text to find insertions or deletions
        if len(current_text) > len(self.previous_text):
            # A character has been inserted
            inserted_char = current_text[-1]  # Get the last character added
            # Emit the inserted character signal
            self.charInserted.emit(inserted_char)

        elif len(current_text) < len(self.previous_text):
            # A character has been deleted
            # Get the deleted character(s)
            deleted_chars = self.previous_text[len(current_text):]
            # Emit the deleted character(s)
            self.charDeleted.emit(deleted_chars)

        # Update the previous text for the next comparison
        self.previous_text = current_text

    def set_caption(self, caption_text):
        """Sets the caption text in the QTextEdit without emitting signals."""
        self.previous_text = ""
        self.text_edit.setPlainText(caption_text)  # Set new caption text
        self.previous_text = caption_text  # Update previous text without emitting

    def emit_caption_changed(self):
        """Emits the captionChanged signal with the current caption."""
        self.captionChanged.emit(self.text_edit.toPlainText())

    def get_caption(self):
        """Returns the current caption text."""
        return self.text_edit.toPlainText()
