from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QLabel, QHBoxLayout
from qtpy.QtCore import Signal, Qt, QRunnable, QThreadPool, QMetaObject
import threading
import os
import tempfile


class CaptionWidget(QtWidgets.QWidget):
    """A widget for editing and displaying image captions."""
    captionChanged = Signal(str)  # Signal emitted when caption changes
    charInserted = Signal(str)    # Signal emitted when a character is inserted
    charDeleted = Signal(str)     # Signal emitted when a character is deleted
    readCaptionFinished = Signal()  # Define a custom signal

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

        # Add the read caption button
        self.read_button = self.create_button(
            icon_name="media-playback-start",  # Adjust icon as needed
            color="#99ff99",
            hover_color="#66ff66"
        )
        self.read_label = QLabel("Read caption")
        self.read_label.setAlignment(Qt.AlignCenter)
        read_button_layout = QVBoxLayout()
        read_button_layout.addWidget(
            self.read_button, alignment=Qt.AlignCenter)
        read_button_layout.addWidget(
            self.read_label, alignment=Qt.AlignCenter)

        # Add the improve caption button
        self.improve_button = self.create_button(
            icon_name="draw-arrow-forward",  # Example icon, adjust as needed
            color="#ccccff",
            hover_color="#9999ff"
        )
        self.improve_label = QLabel("Improve Caption")
        self.improve_label.setAlignment(Qt.AlignCenter)
        improve_button_layout = QVBoxLayout()
        improve_button_layout.addWidget(
            self.improve_button, alignment=Qt.AlignCenter)
        improve_button_layout.addWidget(
            self.improve_label, alignment=Qt.AlignCenter)

        # Connect improve button
        self.improve_button.clicked.connect(self.improve_caption_async)

        # Connect read button to the read_caption_async method
        self.read_button.clicked.connect(self.read_caption_async)

        # Add both button layouts to the horizontal layout
        button_layout.addLayout(record_button_layout)
        button_layout.addLayout(improve_button_layout)
        button_layout.addLayout(describe_button_layout)
        button_layout.addLayout(read_button_layout)
        # (Add read button layout to the main button layout)
        button_layout.addLayout(clear_button_layout)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

        # Connect describe button signal
        self.describe_button.clicked.connect(self.on_describe_clicked)

        # Connect signals and slots
        self.text_edit.textChanged.connect(self.emit_caption_changed)
        self.text_edit.textChanged.connect(self.monitor_text_change)
        self.record_button.clicked.connect(self.toggle_recording)
        # Connect the signal to the slot
        self.readCaptionFinished.connect(self.on_read_caption_finished)

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

    def set_image_path(self, image_path):
        """Sets the image path."""
        self.image_path = image_path

    def get_image_path(self):
        """Returns the image path."""
        return self.image_path

    def read_caption_async(self):
        """Reads the caption in a background thread."""
        self.read_label.setText("Reading...")
        self.read_button.setEnabled(False)  # Disable button
        self.thread_pool.start(ReadCaptionTask(self))

    @QtCore.Slot()
    def on_read_caption_finished(self):
        """Slot connected to the readCaptionFinished signal."""
        self.read_label.setText("Read Caption")
        self.read_button.setEnabled(True)  # Re-enable button

    def read_caption(self):  # This method is now used by the background task
        """Reads the current caption using gTTS and plays it."""
        from gtts import gTTS
        import sounddevice as sd
        from pydub import AudioSegment
        import numpy as np
        try:
            current_text = self.text_edit.toPlainText()
            if not current_text:
                print("Caption is empty. Nothing to read.")
                return

            tts = gTTS(text=current_text, lang='en')

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_file_path = os.path.join(
                    tmpdir, "temp_caption.mp3")  # mp3 file inside temp dir
                tts.save(temp_file_path)

                try:
                    audio = AudioSegment.from_file(
                        temp_file_path, format="mp3")
                    samples = np.array(audio.get_array_of_samples())
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2))
                    sd.play(samples, samplerate=audio.frame_rate)
                    sd.wait()

                except Exception as e:
                    # More specific error message
                    print(f"Error playing MP3: {e}")

        except Exception as e:
            print(f"Error in gTTS: {e}")
        finally:
            self.readCaptionFinished.emit()

    def on_describe_clicked(self):
        """Handles the Describe button click and starts the background task."""
        image_path = self.get_image_path()
        if image_path:
            self.describe_label.setText("Describing image...")
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
        else:
            self.text_edit.setPlainText(message)
            self.describe_label.setText("Describe the image")

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
        threading.Thread(target=self.record_voice, daemon=True).start()

    def stop_recording(self):
        """Stops recording and displays conversion status."""
        self.is_recording = False
        self.record_label.setText("Converting speech to text...")

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
            except sr.RequestError:
                self.record_label.setText("Recognition service error.")
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

    def improve_caption_async(self):
        """Improves the caption using Ollama in a background thread."""
        current_caption = self.text_edit.toPlainText()
        if not current_caption:
            print("Caption is empty. Nothing to improve.")
            return

        self.improve_label.setText("Improving...")
        self.improve_button.setEnabled(False)

        if self.image_path:
            task = ImproveCaptionTask(self.image_path, current_caption, self)
            self.thread_pool.start(task)
        else:
            self.update_improve_status(
                "No image selected for caption improvement.", True)

    @QtCore.Slot(str, bool)
    def update_improve_status(self, message, is_error):
        """Updates the improve caption status in the UI."""
        if is_error:
            self.improve_label.setText("Improvement failed.")
        else:
            # Append improved version
            self.text_edit.append("\n\nImproved Version:\n" + message)
            self.improve_label.setText("Improve Caption")

        self.improve_button.setEnabled(True)


class ImproveCaptionTask(QRunnable):
    def __init__(self, image_path, current_caption, widget):
        super().__init__()
        self.image_path = image_path
        self.current_caption = current_caption
        self.widget = widget

    def run(self):
        try:
            import ollama

            prompt = f"Improve or rewrite the following caption, considering the image:\n\n{self.current_caption}"
            response = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [self.image_path]
                }]
            )

            if "message" in response and "content" in response["message"]:
                improved_caption = response["message"]["content"]
                QMetaObject.invokeMethod(
                    self.widget, "update_improve_status", Qt.QueuedConnection,
                    QtCore.Q_ARG(str, improved_caption), QtCore.Q_ARG(
                        bool, False)
                )

            else:
                raise ValueError("Unexpected response format from Ollama.")

        except Exception as e:
            error_message = f"Error improving caption: {e}"
            QMetaObject.invokeMethod(
                self.widget, "update_improve_status", Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_message), QtCore.Q_ARG(bool, True)
            )


class DescribeImageTask(QRunnable):
    """A task to describe an image in the background."""

    def __init__(self, image_path,
                 widget,
                 prompt='Describe this image in detail.'
                 ):
        super().__init__()
        self.image_path = image_path
        self.widget = widget
        self.prompt = prompt

    def run(self):
        """Runs the task in the background."""
        try:
            import ollama
            response = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': self.prompt,
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


class ReadCaptionTask(QRunnable):
    """A task to read the caption in the background."""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def run(self):
        """Runs the read_caption method in the background."""
        self.widget.read_caption()
