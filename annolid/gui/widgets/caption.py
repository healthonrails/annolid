from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QLabel, QHBoxLayout, QLineEdit, QComboBox
from qtpy.QtCore import Signal, Qt, QRunnable, QThreadPool, QMetaObject
import threading
import os
import tempfile
import matplotlib.pyplot as plt
import io
import base64

try:
    import ollama
except ImportError:
    print(
        "The 'ollama' module is not installed. Please install it by running:\n"
        "    pip install ollama\n"
        "For more information, visit the GitHub repository:\n"
        "    https://github.com/ollama/ollama-python"
    )


class CaptionWidget(QtWidgets.QWidget):
    """A widget for editing and displaying image captions, supporting LaTeX."""
    captionChanged = Signal(str)  # Signal emitted when caption changes
    charInserted = Signal(str)    # Signal emitted when a character is inserted
    charDeleted = Signal(str)     # Signal emitted when a character is deleted
    readCaptionFinished = Signal()  # Define a custom signal
    imageNotFound = Signal(str)
    # Signal to emit when voice recording is done
    voiceRecordingFinished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.available_models = self.get_available_models()
        self.selected_model = "llama3.2-vision:latest"  # Default model
        if len(self.available_models) > 0 and "llama3.2-vision:latest" not in self.available_models:
            if self.available_models:
                # Fallback to first available if default not present
                self.selected_model = self.available_models[0]
        elif len(self.available_models) == 0:
            self.available_models.append("llama3.2-vision:latest")

        self.init_ui()
        self.previous_text = ""
        self.image_path = ""
        self.is_recording = False
        self.thread_pool = QThreadPool()  # Thread pool for running background tasks
        self.voiceRecordingFinished.connect(
            self.on_voice_recording_finished)  # Connect signal
        self.is_streaming_chat = False  # Flag to indicate if chat is streaming

    def get_available_models(self):
        """Fetches the list of available Ollama models."""
        try:
            model_list = ollama.list()

            # Check if models are in the original format (list of dicts)
            if isinstance(model_list['models'], list) and all(isinstance(model, dict) for model in model_list['models']):
                return [model['name'] for model in model_list['models']]

            # Handle the case where models are returned as objects with detailed attributes
            elif isinstance(model_list['models'], list) and all(hasattr(model, 'model') for model in model_list['models']):
                return [model.model for model in model_list['models']]

            # If the format is unexpected, raise a descriptive error
            else:
                raise ValueError("Unexpected model format in response.")

        except Exception as e:
            print(f"Error fetching model list: {e}")
            return []

    def init_ui(self):
        """Initializes the UI components."""
        self.layout = QVBoxLayout(self)

        # Model selection dropdown
        self.model_label = QLabel("Select Model:")
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.available_models)
        if self.selected_model and self.selected_model in self.available_models:
            self.model_selector.setCurrentText(self.selected_model)
        elif self.available_models:
            # Ensure selected_model is updated if default was not found and list is not empty
            self.selected_model = self.available_models[0]
            # Select the first model if default is not available
            self.model_selector.setCurrentIndex(0)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_selector)
        self.layout.addLayout(model_layout)

        self.model_selector.currentIndexChanged.connect(self.on_model_changed)

        # Create a QTextEdit for editing captions
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(False)  # Make it read-only for chat display
        # Style the QTextEdit to have a light background
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;          // Very light gray background
                border: 1px solid #c0c0c0;          // Light gray border

                color: #e0e0e0;                   // Light blue text (or white if very light background)
                hover:color: #d0d0d0;              // Slightly darker on hover for better UX
                text-indent: -1em;               // Indent lines for better readability

                font-family: monospace;           // Consistent character spacing, especially useful for code
                line-height: 1.6;                // Comfortable line height for reading

                -|:after {
                    content: '';
                    color: #666666;                // Subtle gray line between lines
                    font-size: 0.7em;
                }
            }
        """)
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

        # Connect describe button signal
        self.describe_button.clicked.connect(self.on_describe_clicked)

        # Connect signals and slots
        self.text_edit.textChanged.connect(self.emit_caption_changed)
        self.text_edit.textChanged.connect(self.monitor_text_change)
        self.record_button.clicked.connect(self.toggle_recording)
        # Connect the signal to the slot
        self.readCaptionFinished.connect(self.on_read_caption_finished)

        # Horizontal layout for the prompt text edit and chat button
        self.input_layout = QtWidgets.QHBoxLayout()

        # Prompt text editor for user input
        self.prompt_text_edit = QtWidgets.QLineEdit(self)
        self.prompt_text_edit.setPlaceholderText(
            "Type your chat prompt here...")
        self.input_layout.addWidget(self.prompt_text_edit)

        # Chat button
        self.chat_button = QtWidgets.QPushButton("Chat", self)
        self.chat_button.clicked.connect(self.chat_with_ollama)
        self.input_layout.addWidget(self.chat_button)

        # Add the input layout to the main layout
        self.layout.addLayout(self.input_layout)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

        # Integrate existing layouts
        self.setLayout(self.layout)

    def on_model_changed(self, index):
        """Updates the selected model when the combo box changes."""
        self.selected_model = self.model_selector.itemText(index)
        print(f"Selected model changed to: {self.selected_model}")

    def chat_with_ollama(self):
        """Initiates a chat with the Ollama model and displays chat history."""
        user_input = self.prompt_text_edit.text()
        if not user_input:
            print("No input provided for chat.")
            return

        # Append user's input to the chat history
        self.append_to_chat_history(user_input, is_user=True)
        # Add Ollama prefix immediately, ready for streaming
        self.append_to_chat_history("Ollama:", is_user=False)

        # Update UI to indicate that a chat is in progress
        self.chat_button.setEnabled(False)
        self.is_streaming_chat = True  # Set the streaming flag

        # Start the chat task
        task = StreamingChatWithOllamaTask(  # Use Streaming Task Here
            user_input, self.image_path, self, model=self.selected_model)  # Pass selected model
        self.thread_pool.start(task)
        self.prompt_text_edit.clear()

    def append_to_chat_history(self, message, is_user=False):
        """
        Appends a message to the chat history display with styled ChatGPT-style message boxes.

        Args:
            message (str): The message to append.
            is_user (bool): Whether the message is from the user. Defaults to False (AI message).
        """
        # Define styles
        styles = {
            "text_color": "#202124",
            "background_user": "#E8F0FE",  # Light blue
            "background_model": "#F1F3F4",  # Light gray
            "border_radius": "12px",
            "box_shadow": "0 1px 3px rgba(0,0,0,0.1)",
            "max_width": "60%",
            "padding": "12px 16px",
        }

        # Determine message styles
        background_color = styles["background_user"] if is_user else styles["background_model"]
        sender_label = "User: " if is_user else "AI "
        alignment = "right" if is_user else "left"

        # Construct the HTML for the message
        message_html = f"""
            <div style="margin-bottom: 12px; text-align: {alignment};">
                <div style="
                    display: inline-block;
                    /*color: {styles['text_color']};
                    background-color: {background_color};*/
                    padding: {styles['padding']};
                    border-radius: {styles['border_radius']};
                    word-wrap: break-word;
                    max-width: {styles['max_width']};
                    box-shadow: {styles['box_shadow']};
                ">
                    <span background-color:{background_color};><strong>{sender_label}</strong></span>{self.escape_html(message)}
                </div>
            </div>
        """

        # Update the chat history HTML
        current_html = self.text_edit.toHtml()
        body_close_tag_index = current_html.lower().rfind("</body>")
        if body_close_tag_index != -1:
            new_html = (
                current_html[:body_close_tag_index]
                + message_html
                + current_html[body_close_tag_index:]
            )
        else:
            new_html = current_html + message_html

        self.text_edit.setHtml(new_html)

        # Ensure the scrollbar is at the bottom after updating the chat
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum()
        )

    @QtCore.Slot(str, bool)
    def update_chat_response(self, message, is_error):
        """Handles the chat response and appends to chat history."""
        if is_error:
            # Keep error message simple
            # Stream error message
            self.stream_chat_chunk("\nError: " + message)
        else:
            # No need to append full message here as we are streaming
            pass

        # Reset UI and streaming flag
        self.chat_button.setEnabled(True)
        self.is_streaming_chat = False

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

    def latex_to_image_base64(self, latex_string, fontsize=12, dpi=100):
        """Renders LaTeX string to a base64 encoded PNG image."""
        try:
            plt.clf()  # Clear previous plots
            # Adjust figure size as needed
            fig = plt.figure(figsize=(6, 2), dpi=dpi)
            # Use r'' for raw string and $..$ for inline math
            fig.text(0.5, 0.5, rf'${latex_string}$',
                     fontsize=fontsize, ha='center', va='center')
            plt.axis('off')

            buf = io.BytesIO()
            # Save with tight bbox and transparent background
            plt.savefig(buf, format='png', bbox_inches='tight',
                        pad_inches=0.1, transparent=True)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)  # Close the figure to free memory
            return image_base64
        except Exception as e:
            print(f"Error rendering LaTeX: {e}")
            return None

    def set_caption(self, caption_text):
        """Sets the caption text in the QTextEdit, rendering LaTeX formulas as images."""
        self.previous_text = ""
        html_content = ""
        parts = caption_text.split("$$")  # Split by LaTeX delimiters

        for i, part in enumerate(parts):
            if i % 2 == 0:  # Non-LaTeX part
                # Escape HTML special characters
                html_content += self.escape_html(part)
            else:  # LaTeX part
                base64_data = self.latex_to_image_base64(part)
                if base64_data:
                    # vertical-align:middle to align images with text
                    html_content += f'<img src="data:image/png;base64,{base64_data}" style="vertical-align:middle;"/>'
                else:
                    # Display error if rendering fails, escape LaTeX for display
                    html_content += f'<span style="color:red;">Error rendering LaTeX: $${self.escape_html(part)}$$</span>'

        # Enclose in body for consistent HTML structure
        self.text_edit.setHtml(f"<body>{html_content}</body>")
        self.previous_text = caption_text

    def escape_html(self, text):
        """Escapes HTML special characters to prevent rendering issues."""
        return text.replace('&', '&').replace('<', '<').replace('>', '>')

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
        """Reads the current caption using kokoro or gTTS and plays it."""
        try:
            from annolid.agents.kokoro_tts import text_to_speech
            from annolid.agents.kokoro_tts import play_audio
            text = self.text_edit.toPlainText()
            if not text:
                print("Caption is empty. Nothing to read.")
                return
            audio_data = text_to_speech(text)
            if audio_data:
                samples, sample_rate = audio_data
                print("\nText-to-speech conversion successful!")
                # Play the audio using sounddevice
                play_audio(samples, sample_rate)
            else:
                print("\nText-to-speech conversion failed.")
        except Exception as e:
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
            task = DescribeImageTask(
                image_path, self, model=self.selected_model)  # Pass selected model
            self.thread_pool.start(task)
        else:
            self.set_caption("No image selected for description.")

    @QtCore.Slot(str, bool)
    def update_description_status(self, message, is_error):
        """Updates the description status in the UI."""
        if is_error:
            self.set_caption(message)
            self.describe_label.setText("Description failed.")
        else:
            self.set_caption(message)
            self.describe_label.setText("Describe the image")

    def emit_caption_changed(self):
        """Emits the captionChanged signal with the current caption."""
        self.captionChanged.emit(
            self.text_edit.toPlainText())  # emit plain text caption

    def get_caption(self):
        """Returns the current caption text (plain text, not HTML)."""
        return self.text_edit.toPlainText()  # get plain text caption for other operations

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
                try:
                    from annolid.agents.speech_recognition import transcribe_audio
                    text = transcribe_audio(complete_audio.get_wav_data())[0]
                except Exception as e:
                    # Transcribe the combined audio
                    text = recognizer.recognize_google(complete_audio)
                current_plain_text = self.get_caption()  # Get current plain text
                # Emit the signal with the transcribed text to be handled in the main thread
                QMetaObject.invokeMethod(
                    self, "on_voice_recording_finished", Qt.QueuedConnection,
                    QtCore.Q_ARG(str, current_plain_text + text)
                )

            except sr.UnknownValueError:
                self.record_label.setText("Could not understand audio.")
                QMetaObject.invokeMethod(
                    self, "stop_recording_ui_reset", Qt.QueuedConnection
                )  # UI reset on main thread
            except sr.RequestError:
                self.record_label.setText("Recognition service error.")
                QMetaObject.invokeMethod(
                    self, "stop_recording_ui_reset", Qt.QueuedConnection
                )  # UI reset on main thread
            finally:
                # Reset the button and label after transcription or error
                self.is_recording = False

    # Make this a slot just in case, even though it's called via invokeMethod now
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

    @QtCore.Slot(str)
    def on_voice_recording_finished(self, transcribed_text):
        """Slot to handle voice recording finished signal and update caption."""
        self.stop_recording_ui_reset()  # Reset UI elements
        # Safely update caption in main thread
        self.set_caption(transcribed_text)

    def improve_caption_async(self):
        """Improves the caption using Ollama in a background thread."""
        current_caption = self.get_caption()  # get plain text caption
        if not current_caption:
            print("Caption is empty. Nothing to improve.")
            return

        self.improve_label.setText("Improving...")
        self.improve_button.setEnabled(False)

        if self.image_path:
            task = ImproveCaptionTask(
                self.image_path, current_caption, self, model=self.selected_model)  # Pass selected model
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
            current_plain_text = self.get_caption()
            # append to plain text and re-render
            self.set_caption(current_plain_text +
                             "\n\nImproved Version:\n" + message)
            self.improve_label.setText("Improve Caption")

        self.improve_button.setEnabled(True)

    @QtCore.Slot(str)
    def stream_chat_chunk(self, chunk):
        """Streams a chunk of the chat response by appending to the last Ollama message's span."""
        current_html = self.text_edit.toHtml()
        chunk_escaped = self.escape_html(chunk)

        # Very specific targeting of the last Ollama response span
        ollama_span_start_marker_lower = "<span id='ollama-response-area' style='color: #202124;'></span>"
        last_ollama_span_start_index = current_html.lower().rfind(
            ollama_span_start_marker_lower)

        if last_ollama_span_start_index != -1:
            # Insertion point is right *before* the closing </span> of the marker span
            insertion_index = current_html.lower().find(
                "</span>", last_ollama_span_start_index)
            if insertion_index != -1:
                # Reconstruct HTML by inserting chunk content
                new_html = current_html[:insertion_index] + \
                    chunk_escaped + current_html[insertion_index:]
                self.text_edit.setHtml(new_html)
            else:
                print(
                    "Error: Closing </span> tag for Ollama response area not found.")
                # Fallback: Append to end of QTextEdit if span structure is broken
                self.text_edit.moveCursor(QtGui.QTextCursor.End)
                self.text_edit.insertPlainText(chunk)
        else:
            # Fallback: Append to end of QTextEdit if marker span is missing
            self.text_edit.moveCursor(QtGui.QTextCursor.End)
            self.text_edit.insertPlainText(chunk)

        # Ensure scrollbar is at the bottom
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum())


class ImproveCaptionTask(QRunnable):
    def __init__(self, image_path, current_caption, widget, model="llama3.2-vision:latest"):
        super().__init__()
        self.image_path = image_path
        self.current_caption = current_caption
        self.widget = widget
        self.model = model  # Store model

    def run(self):
        try:
            import ollama

            prompt = f"Improve or rewrite the following caption, considering the image:\n\n{self.current_caption}"
            if self.image_path and os.path.exists(self.image_path):
                messages = [{
                    'role': 'user',
                    'content': prompt,
                    'images': [self.image_path]
                }]
            else:
                messages = [{
                    'role': 'user',
                    'content': prompt,
                }]

            response = ollama.chat(
                model=self.model,
                messages=messages
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
                 prompt='Describe this image in detail.',
                 model="llama3.2-vision:latest"
                 ):
        super().__init__()
        self.image_path = image_path
        self.widget = widget
        self.prompt = prompt
        self.model = model  # Store model

    def run(self):
        """Runs the task in the background."""
        try:
            import ollama
            response = ollama.chat(
                model=self.model,
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
            error_message = f"An error occurred while describing the image: {e}.\n"
            error_message += "Please save the video frame to disk by clicking the 'Save' button or pressing Ctrl/Cmd + S."
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


# Renamed to StreamingChatWithOllamaTask
class StreamingChatWithOllamaTask(QRunnable):
    """A task to chat with the Ollama model in the background, streaming results."""

    def __init__(self, prompt, image_path=None, widget=None, model="llama3.2-vision:latest"):  # Added model parameter
        super().__init__()
        self.prompt = prompt
        self.image_path = image_path
        self.widget = widget
        self.model = model  # Store model

    def run(self):
        """Sends a chat message to Ollama and processes the response."""
        try:
            import ollama

            messages = [{'role': 'user', 'content': self.prompt}]
            if self.image_path and os.path.exists(self.image_path):
                # Attach the image if provided
                messages[0]['images'] = [self.image_path]

            stream = ollama.chat(
                model=self.model,  # Use selected model
                messages=messages,
                stream=True  # Enable streaming
            )
            full_response = ""  # Accumulate full response

            for part in stream:
                if 'message' in part and 'content' in part['message']:
                    chunk = part['message']['content']
                    full_response += chunk  # Accumulate full response
                    QMetaObject.invokeMethod(
                        self.widget, "stream_chat_chunk", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, chunk)
                    )
                elif 'error' in part:
                    error_message = f"Stream error: {part['error']}"
                    QMetaObject.invokeMethod(
                        self.widget, "update_chat_response", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, error_message),
                        QtCore.Q_ARG(bool, True)
                    )
                    return  # Stop streaming on error

            # After stream ends, handle completion or empty response
            if not full_response.strip():  # Check if full response is empty or just whitespace
                error_message = "No response from Ollama."
                QMetaObject.invokeMethod(
                    self.widget, "update_chat_response", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, error_message),
                    QtCore.Q_ARG(bool, True)
                )
            else:
                QMetaObject.invokeMethod(
                    self.widget, "update_chat_response", QtCore.Qt.QueuedConnection,
                    # No message for success, signal completion
                    QtCore.Q_ARG(str, ""),
                    QtCore.Q_ARG(bool, False)
                )

        except Exception as e:
            error_message = f"Error in chat interaction: {e}"
            QtCore.QMetaObject.invokeMethod(
                self.widget, "update_chat_response", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_message),
                QtCore.Q_ARG(bool, True)
            )
