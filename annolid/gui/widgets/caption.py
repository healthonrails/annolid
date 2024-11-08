from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QLabel
from qtpy.QtCore import Signal, Qt
import threading


class CaptionWidget(QtWidgets.QWidget):
    """A widget for editing and displaying image captions."""
    captionChanged = Signal(str)  # Signal emitted when caption changes
    charInserted = Signal(str)    # Signal emitted when a character is inserted
    charDeleted = Signal(str)     # Signal emitted when a character is deleted

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Create a QTextEdit for editing captions
        self.text_edit = QTextEdit()
        self.layout.addWidget(self.text_edit)

        # Create a circular button with a microphone icon for recording
        self.record_button = QPushButton()
        self.record_button.setFixedSize(50, 50)  # Make it a circle
        self.record_button.setIcon(QtGui.QIcon.fromTheme("microphone"))
        self.record_button.setIconSize(QtCore.QSize(30, 30))
        self.record_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: #ff4d4d;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #e60000;
            }
        """)
        self.layout.addWidget(self.record_button, alignment=Qt.AlignCenter)

        # Add a label below the record button to display the recording status
        self.record_label = QLabel("Tap to record")
        self.record_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.record_label)

        # Store previous text for comparison
        self.previous_text = ""

        # Connect signals and slots
        self.text_edit.textChanged.connect(self.emit_caption_changed)
        self.text_edit.textChanged.connect(self.monitor_text_change)
        self.record_button.clicked.connect(self.toggle_recording)

        # Initialize recording state
        self.is_recording = False
        self.setLayout(self.layout)

    def monitor_text_change(self):
        """Monitor for text changes and emit signals for insertions and deletions."""
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

    def emit_caption_changed(self):
        """Emits the captionChanged signal with the current caption."""
        self.captionChanged.emit(self.text_edit.toPlainText())

    def get_caption(self):
        """Returns the current caption text."""
        return self.text_edit.toPlainText()

    def toggle_recording(self):
        """Toggles the recording state and starts or stops recording."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: red;
                    border-radius: 25px;
                }
            """)
            self.record_label.setText("Recording...")
            self.record_label.setStyleSheet("color: red; font-size: 16px;")
            threading.Thread(target=self.record_voice, daemon=True).start()
        else:
            # Stop recording
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

    def toggle_recording(self):
        """Toggles the recording state and starts or stops recording."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setStyleSheet("""
                QPushButton {
                    border: none;
                    background-color: red;
                    border-radius: 25px;
                }
            """)
            self.record_label.setText("Recording...")
            self.record_label.setStyleSheet("color: red; font-size: 16px;")
            threading.Thread(target=self.record_voice, daemon=True).start()
        else:
            # Stop recording and show "Converting speech to text..."
            self.is_recording = False
            self.record_label.setText("Converting speech to text...")
            self.record_label.setStyleSheet("color: blue; font-size: 16px;")
            # No need to change button appearance here, it will reset after processing

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
