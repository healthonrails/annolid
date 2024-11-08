from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QLabel, QHBoxLayout
from qtpy.QtCore import Signal, Qt, QRunnable, QThreadPool
import threading


class CaptionWidget(QtWidgets.QWidget):
    """A widget for editing and displaying image captions."""
    captionChanged = Signal(str)  # Signal emitted when caption changes
    charInserted = Signal(str)    # Signal emitted when a character is inserted
    charDeleted = Signal(str)     # Signal emitted when a character is deleted

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.previous_text = ""
        self.image_path = ""
        self.is_recording = False
        self.thread_pool = QThreadPool()  # Thread pool for running background tasks

    def init_ui(self):
        """Initializes the UI components."""
        self.layout = QVBoxLayout(self)

        # Create a QTextEdit for editing captions
        self.text_edit = QTextEdit()
        self.layout.addWidget(self.text_edit)

        # Create a horizontal layout for the buttons and labels
        button_layout = QHBoxLayout()

        # Create the record button with its label below
        self.record_button = self.create_button(
            icon_name="microphone",
            color="#ff4d4d",
            hover_color="#e60000"
        )
        self.record_label = QLabel("Tap to record")
        self.record_label.setAlignment(Qt.AlignCenter)
        record_button_layout = QVBoxLayout()
        record_button_layout.addWidget(
            self.record_button, alignment=Qt.AlignCenter)
        record_button_layout.addWidget(
            self.record_label, alignment=Qt.AlignCenter)

        # Create the describe button with its label below
        self.describe_button = self.create_button(
            icon_name="view-preview",  # Adjust this icon as needed
            color="#4d94ff",
            hover_color="#0040ff"
        )
        self.describe_label = QLabel("Describe the image")
        self.describe_label.setAlignment(Qt.AlignCenter)
        describe_button_layout = QVBoxLayout()
        describe_button_layout.addWidget(
            self.describe_button, alignment=Qt.AlignCenter)
        describe_button_layout.addWidget(
            self.describe_label, alignment=Qt.AlignCenter)

        # Add the save caption button
        self.save_button = self.create_button(
            icon_name="document-save",  # Adjust icon as needed
            color="#66b3ff",
            hover_color="#3399ff"
        )

        # Add the clear caption button
        self.clear_button = self.create_button(
            icon_name="edit-clear",  # Adjust icon as needed
            color="#ffcc99",
            hover_color="#ff9900"
        )
        self.clear_label = QLabel("Clear caption")
        self.clear_label.setAlignment(Qt.AlignCenter)
        clear_button_layout = QVBoxLayout()
        clear_button_layout.addWidget(
            self.clear_button, alignment=Qt.AlignCenter)
        clear_button_layout.addWidget(
            self.clear_label, alignment=Qt.AlignCenter)

        # Connect the buttons to their respective methods
        self.clear_button.clicked.connect(self.clear_caption)

        # Add both button layouts to the horizontal layout
        button_layout.addLayout(record_button_layout)
        button_layout.addLayout(describe_button_layout)
        # Add the new button layouts to the main button layout
        button_layout.addLayout(clear_button_layout)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

        # Connect describe button signal
        self.describe_button.clicked.connect(self.on_describe_clicked)

        # Connect signals and slots
        self.text_edit.textChanged.connect(self.emit_caption_changed)
        self.text_edit.textChanged.connect(self.monitor_text_change)
        self.record_button.clicked.connect(self.toggle_recording)

        self.setLayout(self.layout)

    def create_button(self, icon_name, color, hover_color):
        """Creates and returns a styled button."""
        button = QPushButton()
        button.setFixedSize(20, 20)  # Smaller size
        button.setIcon(QtGui.QIcon.fromTheme(icon_name))
        button.setIconSize(QtCore.QSize(10, 10))
        button.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background-color: {color};
                border-radius: 20px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)
        return button

    def monitor_text_change(self):
        """Monitors text changes and emits appropriate signals."""
        current_text = self.text_edit.toPlainText()
        if self.previous_text == "":
            self.previous_text = current_text
            return

        if len(current_text) > len(self.previous_text):
            inserted_char = current_text[-1]
            self.charInserted.emit(inserted_char)
        elif len(current_text) < len(self.previous_text):
            deleted_chars = self.previous_text[len(current_text):]
            self.charDeleted.emit(deleted_chars)

        self.previous_text = current_text

    def set_caption(self, caption_text):
        """Sets the caption text in the QTextEdit without emitting signals."""
        self.previous_text = ""
        self.text_edit.setPlainText(caption_text)
        self.previous_text = caption_text

    def clear_caption(self):
        """Clears the caption."""
        self.set_caption("")
        self.clear_label.setText("Clear caption")
        self.clear_label.setStyleSheet("color: red;")

    def set_image_path(self, image_path):
        """Sets the image path."""
        self.image_path = image_path

    def get_image_path(self):
        """Returns the image path."""
        return self.image_path

    def on_describe_clicked(self):
        """Handles the Describe button click and starts the background task."""
        image_path = self.get_image_path()
        if image_path:
            self.describe_label.setText("Describing image...")
            self.describe_label.setStyleSheet("color: blue;")
            task = DescribeImageTask(image_path, self)
            self.thread_pool.start(task)
        else:
            self.text_edit.setPlainText("No image selected for description.")

    @QtCore.Slot(str, bool)
    def update_description_status(self, message, is_error):
        """Updates the description status in the UI."""
        if is_error:
            self.text_edit.setPlainText(message)
            self.describe_label.setText("Description failed.")
            self.describe_label.setStyleSheet("color: red;")
        else:
            self.text_edit.setPlainText(message)
            self.describe_label.setText("Describe the image")
            self.describe_label.setStyleSheet("color: green;")

    def emit_caption_changed(self):
        """Emits the captionChanged signal with the current caption."""
        self.captionChanged.emit(self.text_edit.toPlainText())

    def get_caption(self):
        """Returns the current caption text."""
        return self.text_edit.toPlainText()

    def toggle_recording(self):
        """Toggles the recording state and updates the UI."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Starts the recording process and updates UI accordingly."""
        self.is_recording = True
        self.record_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: red;
                border-radius: 20px;
            }
        """)
        self.record_label.setText("Recording...")
        self.record_label.setStyleSheet("color: red; font-size: 16px;")
        threading.Thread(target=self.record_voice, daemon=True).start()

    def stop_recording(self):
        """Stops recording and displays conversion status."""
        self.is_recording = False
        self.record_label.setText("Converting speech to text...")
        self.record_label.setStyleSheet("color: blue; font-size: 16px;")

    def stop_recording_ui_reset(self):
        """Resets the UI after recording ends."""
        self.record_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: #ff4d4d;
                border-radius: 20px;
            }
        """)
        self.record_label.setText("Tap to record")
        self.record_label.setStyleSheet("color: black;")

    def record_voice(self):
        """Records voice input and converts it to text continuously until stopped."""
        try:
            import speech_recognition as sr
            # import pyaudio  # PyAudio is needed for microphone access
        except ImportError as e:
            missing_package = e.name
            print(
                f"Error: The required package '{missing_package}' is not installed.")
            if missing_package == "speech_recognition":
                print(
                    "Please install SpeechRecognition using the command: pip install SpeechRecognition")
            elif missing_package == "pyaudio":
                print("Please install PyAudio using the command: pip install pyaudio")
            else:
                print("Please install the required packages by running:")
                print("pip install SpeechRecognition")
                print("pip install pyaudio")
            print("Note: The program may not function properly without these packages.")
        else:
            # If packages are available, proceed with the program
            print("Packages are installed. Proceeding with the program.")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = []  # List to store chunks of audio

            try:
                while self.is_recording:
                    audio_chunk = recognizer.listen(
                        source, timeout=None, phrase_time_limit=None)
                    audio_data.append(audio_chunk)

                # Combine audio chunks into one AudioData instance
                complete_audio = sr.AudioData(
                    b"".join([chunk.get_raw_data() for chunk in audio_data]),
                    source.SAMPLE_RATE,
                    source.SAMPLE_WIDTH
                )

                # Transcribe the combined audio
                text = recognizer.recognize_google(complete_audio)
                self.text_edit.moveCursor(QtGui.QTextCursor.End)
                self.text_edit.insertPlainText(text)

            except sr.UnknownValueError:
                self.record_label.setText("Could not understand audio.")
                self.record_label.setStyleSheet("color: orange;")
            except sr.RequestError:
                self.record_label.setText("Recognition service error.")
                self.record_label.setStyleSheet("color: orange;")
            finally:
                # Reset the button and label after transcription or error
                self.is_recording = False
                self.record_button.setStyleSheet("""
                    QPushButton {
                        border: none;
                        background-color: #ff4d4d;
                        border-radius: 25px;
                    }
                """)
                self.record_label.setText("Tap to record")
                self.record_label.setStyleSheet("color: black;")


class DescribeImageTask(QRunnable):
    """A task to describe an image in the background."""

    def __init__(self, image_path, widget):
        super().__init__()
        self.image_path = image_path
        self.widget = widget

    def run(self):
        """Runs the task in the background."""
        try:
            import ollama
            response = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [self.image_path]
                }]
            )

            # Access response content safely
            if "message" in response and "content" in response["message"]:
                description = response["message"]["content"]
                QtCore.QMetaObject.invokeMethod(
                    self.widget, "update_description_status", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, description),
                    QtCore.Q_ARG(bool, False)
                )
            else:
                raise ValueError(
                    "Unexpected response format: 'message' or 'content' key missing.")

        except Exception as e:
            error_message = f"Error describing image: {e}"
            QtCore.QMetaObject.invokeMethod(
                self.widget, "update_description_status", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_message),
                QtCore.Q_ARG(bool, True)
            )
