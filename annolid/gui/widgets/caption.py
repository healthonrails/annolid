from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtWidgets import (
    QVBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
)
from qtpy.QtCore import Signal, Qt, QRunnable, QThreadPool, QMetaObject
import threading
import os
import tempfile
import matplotlib.pyplot as plt
import io
import base64
import mimetypes
import uuid
from typing import Any, Dict, List

from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog
from annolid.utils.llm_settings import (
    load_llm_settings,
    save_llm_settings,
    resolve_llm_config,
    ensure_provider_env,
)

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
    imageGenerated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm_settings = load_llm_settings()
        self.llm_settings.setdefault("last_models", {})
        self.provider_labels: Dict[str, str] = {
            "ollama": "Ollama (local)",
            "openai": "OpenAI GPT",
            "gemini": "Google Gemini",
        }
        self.selected_provider = self.llm_settings.get("provider", "ollama")
        self._ollama_error_reported = False
        self.available_models = self.get_available_models(self.selected_provider)
        self.selected_model = self._resolve_initial_model(self.selected_provider)
        self._suppress_model_updates = False
        self._suppress_provider_updates = False

        self.init_ui()
        self.previous_text = ""
        self.image_path = ""
        self.is_recording = False
        self.thread_pool = QThreadPool()  # Thread pool for running background tasks
        self.voiceRecordingFinished.connect(
            self.on_voice_recording_finished)  # Connect signal
        self.is_streaming_chat = False  # Flag to indicate if chat is streaming
        self.current_ai_span_id = None  # Tracks current AI response span

    def _default_model_for(self, provider: str) -> str:
        defaults = {
            "ollama": "llama3.2-vision:latest",
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.5-pro",
        }
        return defaults.get(provider, "")

    def _resolve_initial_model(self, provider: str) -> str:
        last_models = self.llm_settings.get("last_models", {})
        last_model = last_models.get(provider)
        if last_model and last_model in self.available_models:
            return last_model

        if self.available_models:
            return self.available_models[0]

        fallback = self._default_model_for(provider)
        if fallback and fallback not in self.available_models:
            self.available_models.append(fallback)
        return fallback

    def get_available_models(self, provider: str) -> List[str]:
        """Return known models for the given provider."""
        if provider == "ollama":
            models = self._fetch_ollama_models()
            pinned = self.llm_settings.get("ollama", {}).get(
                "preferred_models", [])
            for model in pinned:
                if model and model not in models:
                    models.append(model)
            if not models:
                fallback = self._default_model_for("ollama")
                if fallback:
                    models.append(fallback)
            elif models != pinned:
                self.llm_settings.setdefault("ollama", {})[
                    "preferred_models"] = models
                save_llm_settings(self.llm_settings)
            return models

        provider_settings = self.llm_settings.get(provider, {})
        return provider_settings.get("preferred_models", [])

    def _fetch_ollama_models(self) -> List[str]:
        """Fetch available Ollama models respecting the configured host."""
        host = self.llm_settings.get("ollama", {}).get("host")
        prev_host_present = "OLLAMA_HOST" in os.environ
        prev_host_value = os.environ.get("OLLAMA_HOST")

        ollama_module = globals().get("ollama")
        if ollama_module is None:
            return []

        try:
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)

            model_list = ollama_module.list()

            # Check if models are in the original format (list of dicts)
            if isinstance(model_list['models'], list) and all(isinstance(model, dict) for model in model_list['models']):
                self._ollama_error_reported = False
                return [model['name'] for model in model_list['models']]

            # Handle the case where models are returned as objects with detailed attributes
            elif isinstance(model_list['models'], list) and all(hasattr(model, 'model') for model in model_list['models']):
                self._ollama_error_reported = False
                return [model.model for model in model_list['models']]

            # If the format is unexpected, raise a descriptive error
            else:
                raise ValueError("Unexpected model format in response.")

        except Exception as e:
            if not self._ollama_error_reported:
                friendly_host = host or prev_host_value or "http://localhost:11434"
                print(
                    "Unable to reach the Ollama server to list models. "
                    f"Check that Ollama is running at {friendly_host}. "
                    f"Original error: {e}"
                )
                self._ollama_error_reported = True
            return []
        finally:
            if prev_host_present:
                os.environ["OLLAMA_HOST"] = prev_host_value  # type: ignore[arg-type]
            else:
                os.environ.pop("OLLAMA_HOST", None)

    def _determine_image_model(self) -> str:
        """Choose an appropriate Gemini model for image generation."""
        if self.selected_provider == "gemini" and self.selected_model:
            return self.selected_model
        preferred = self.llm_settings.get("gemini", {}).get(
            "preferred_models", [])
        if preferred:
            return preferred[0]
        return "gemini-flash-latest"

    def _determine_openai_image_model(self) -> str:
        """Pick the best-suited OpenAI image model."""
        openai_settings = self.llm_settings.get("openai", {})
        preferred_images = openai_settings.get("preferred_image_models") or []
        fallback_order = [
            "gpt-image-1",
            "gpt-image-1-mini",
            "dall-e-3",
            "dall-e-2",
        ]

        for model in preferred_images:
            if model:
                return model

        for model in fallback_order:
            if model:
                return model

        return "gpt-image-1"

    def _has_gemini_api_key(self) -> bool:
        gemini_settings = self.llm_settings.get("gemini", {})
        return bool(
            gemini_settings.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    def _has_openai_api_key(self) -> bool:
        openai_settings = self.llm_settings.get("openai", {})
        return bool(
            openai_settings.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )

    def _should_generate_image(self, prompt: str) -> bool:
        prompt_lower = (prompt or "").strip().lower()
        model_lower = (self.selected_model or "").lower()

        if prompt_lower.startswith("image:"):
            return True

        if self.selected_provider == "gemini" and (
            "image" in model_lower or model_lower.endswith("-live")
        ):
            return True

        if self.selected_provider == "openai":
            if "gpt-image" in model_lower:
                return True
            if "gpt-5" in model_lower and "image" in prompt_lower:
                return True

        return False

    def _sanitize_image_prompt(self, prompt: str) -> str:
        stripped = (prompt or "").strip()
        if stripped.lower().startswith("image:"):
            return stripped.split(":", 1)[1].strip()
        return stripped

    def _persist_state(self) -> None:
        """Save the current provider and last selected models."""
        self.llm_settings["provider"] = self.selected_provider
        self.llm_settings.setdefault("last_models", {})
        self.llm_settings["last_models"][self.selected_provider] = self.selected_model
        save_llm_settings(self.llm_settings)

    def _update_model_selector(self) -> None:
        """Refresh the model selector combo box contents."""
        self._suppress_model_updates = True
        try:
            self.model_selector.blockSignals(True)
            self.model_selector.clear()
            if self.available_models:
                self.model_selector.addItems(self.available_models)
            if self.selected_model:
                self.model_selector.setCurrentText(self.selected_model)
        finally:
            self.model_selector.blockSignals(False)
            self._suppress_model_updates = False

    def _ensure_provider_ready(self) -> bool:
        """Validate that required credentials are present for the provider."""
        provider_config = self.llm_settings.get(self.selected_provider, {})
        if self.selected_provider == "openai" and not provider_config.get("api_key"):
            QtWidgets.QMessageBox.warning(
                self,
                "OpenAI API key required",
                "Please add your OpenAI API key in the AI Model Settings dialog.",
            )
            return False
        if self.selected_provider == "gemini" and not provider_config.get("api_key"):
            QtWidgets.QMessageBox.warning(
                self,
                "Gemini API key required",
                "Please add your Google Gemini API key in the AI Model Settings dialog.",
            )
            return False
        return True

    def init_ui(self):
        """Initializes the UI components."""
        self.layout = QVBoxLayout(self)

        # Provider selection dropdown
        provider_layout = QHBoxLayout()
        self.provider_label = QLabel("Provider:")
        self.provider_selector = QComboBox()
        for key, label in self.provider_labels.items():
            self.provider_selector.addItem(label, userData=key)
        provider_index = self.provider_selector.findData(
            self.selected_provider)
        if provider_index != -1:
            self.provider_selector.setCurrentIndex(provider_index)
        provider_layout.addWidget(self.provider_label)
        provider_layout.addWidget(self.provider_selector)

        self.configure_models_button = QPushButton("Configure…")
        self.configure_models_button.clicked.connect(
            self.open_llm_settings_dialog)
        provider_layout.addWidget(self.configure_models_button)
        provider_layout.addStretch(1)
        self.layout.addLayout(provider_layout)

        self.provider_selector.currentIndexChanged.connect(
            self.on_provider_changed)

        # Model selection dropdown
        self.model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        self.model_selector.setEditable(True)
        self.model_selector.setInsertPolicy(QComboBox.NoInsert)
        self._update_model_selector()

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_selector)
        self.layout.addLayout(model_layout)

        self.model_selector.currentIndexChanged.connect(self.on_model_changed)
        self.model_selector.editTextChanged.connect(
            self.on_model_text_edited)

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

        # Chat button (also triggers image generation when applicable)
        self.chat_button = QtWidgets.QPushButton("Chat", self)
        self.chat_button.clicked.connect(self.chat_with_model)
        self.input_layout.addWidget(self.chat_button)

        # Add the input layout to the main layout
        self.layout.addLayout(self.input_layout)

        # Add the button layout to the main layout
        self.layout.addLayout(button_layout)

        # Integrate existing layouts
        self.setLayout(self.layout)

    def on_model_changed(self, index):
        """Updates the selected model when the combo box changes."""
        if self._suppress_model_updates:
            return
        self.selected_model = self.model_selector.itemText(index).strip()
        self._persist_state()
        print(f"Selected model changed to: {self.selected_model}")

    def on_model_text_edited(self, text):
        """Capture manual edits to the model combo box."""
        if self._suppress_model_updates:
            return
        self.selected_model = text.strip()
        self._persist_state()

    def on_provider_changed(self, index):
        """Handle provider selection changes."""
        if self._suppress_provider_updates:
            return
        provider = self.provider_selector.itemData(index)
        if not provider:
            return
        self.selected_provider = provider
        self.available_models = self.get_available_models(provider)
        if self.selected_model not in self.available_models:
            self.selected_model = self._resolve_initial_model(provider)
        self._update_model_selector()
        self._persist_state()

    def open_llm_settings_dialog(self):
        """Open the settings dialog for configuring providers and models."""
        dialog = LLMSettingsDialog(self, settings=dict(self.llm_settings))
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            settings = dialog.get_settings()
            settings.setdefault("last_models", self.llm_settings.get(
                "last_models", {}))
            self.llm_settings = settings
            new_provider = self.llm_settings.get(
                "provider", self.selected_provider)
            self._suppress_provider_updates = True
            try:
                provider_index = self.provider_selector.findData(new_provider)
                if provider_index != -1:
                    self.provider_selector.setCurrentIndex(provider_index)
                else:
                    self.selected_provider = new_provider
            finally:
                self._suppress_provider_updates = False

            self.selected_provider = self.provider_selector.currentData()
            self.available_models = self.get_available_models(
                self.selected_provider)
            self.selected_model = self._resolve_initial_model(
                self.selected_provider)
            self._update_model_selector()
            self._persist_state()

    def chat_with_model(self):
        """Initiates a chat with the selected model and displays chat history."""
        raw_prompt = self.prompt_text_edit.text()
        if not raw_prompt:
            print("No input provided for chat.")
            return
        if not self._ensure_provider_ready():
            return

        if self._should_generate_image(raw_prompt):
            cleaned_prompt = self._sanitize_image_prompt(raw_prompt)
            provider_label = self.provider_labels.get(
                self.selected_provider, "AI")

            self.append_to_chat_history(raw_prompt, is_user=True)
            self.append_to_chat_history(
                f"{provider_label} is generating an image…",
                is_user=False,
            )

            self.chat_button.setEnabled(False)
            self.is_streaming_chat = False
            task = None

            if self.selected_provider == "gemini":
                if not self._has_gemini_api_key():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Gemini API key required",
                        "Add a Gemini API key in the AI Model Settings dialog.",
                    )
                    self.chat_button.setEnabled(True)
                    return
                model = self._determine_image_model()
                task = GeminiImageGenerationTask(cleaned_prompt, self, model)
            elif self.selected_provider == "openai":
                if not self._has_openai_api_key():
                    QtWidgets.QMessageBox.warning(
                        self,
                        "OpenAI API key required",
                        "Add an OpenAI API key in the AI Model Settings dialog.",
                    )
                    self.chat_button.setEnabled(True)
                    return
                model = self._determine_openai_image_model()
                task = OpenAIImageGenerationTask(
                    prompt=cleaned_prompt,
                    widget=self,
                    model=model,
                    settings=self.llm_settings,
                )
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "Image generation unavailable",
                    "Select a Gemini or OpenAI model that supports image generation.",
                )
                return

            if task:
                self.thread_pool.start(task)
                self.prompt_text_edit.clear()
            return

        # Text chat path
        self.append_to_chat_history(raw_prompt, is_user=True)
        provider_label = self.provider_labels.get(
            self.selected_provider, "AI")
        self.current_ai_span_id = f"ai-response-{uuid.uuid4().hex}"
        self.append_to_chat_history(
            f"{provider_label}:", is_user=False, message_id=self.current_ai_span_id)

        self.chat_button.setEnabled(False)
        self.is_streaming_chat = True

        task = StreamingChatTask(
            prompt=raw_prompt,
            image_path=self.image_path,
            widget=self,
            model=self.selected_model,
            provider=self.selected_provider,
            settings=self.llm_settings,
        )
        self.thread_pool.start(task)
        self.prompt_text_edit.clear()

    @QtCore.Slot(str, str)
    def on_image_generation_success(self, image_path: str, message: str):
        """Handle successful image generation."""
        self.chat_button.setEnabled(True)
        self.is_streaming_chat = False
        note = message.strip() if message else ""
        if note:
            self.append_to_chat_history(note, is_user=False)
        else:
            self.append_to_chat_history(
                f"Generated an image: {os.path.basename(image_path)}",
                is_user=False,
            )
        self.imageGenerated.emit(image_path)

    @QtCore.Slot(str)
    def on_image_generation_failed(self, error_message: str):
        """Handle image generation failures."""
        self.chat_button.setEnabled(True)
        self.is_streaming_chat = False
        self.append_to_chat_history(
            f"Image generation failed: {error_message}",
            is_user=False,
        )

    def append_to_chat_history(self, message, is_user=False, message_id=None):
        """
        Appends a message to the chat history display with styled ChatGPT-style message boxes.

        Args:
            message (str): The message to append.
            is_user (bool): Whether the message is from the user. Defaults to False (AI message).
            message_id (str | None): Optional span identifier for streaming updates.
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
        span_markup = (
            f"<span id='{message_id}' style='color: {styles['text_color']};'></span>"
            if message_id and not is_user
            else ""
        )

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
                    <span background-color:{background_color};><strong>{sender_label}</strong></span>{self.escape_html(message)}{span_markup}
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
        elif message:
            self.stream_chat_chunk(message)
        else:
            # No need to append full message here as we are streaming
            pass

        # Reset UI and streaming flag
        self.chat_button.setEnabled(True)
        self.is_streaming_chat = False
        self.current_ai_span_id = None

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
            if self.selected_provider != "ollama":
                self.update_description_status(
                    "Image description currently supports Ollama models. Please switch provider.",
                    True,
                )
                return
            self.describe_label.setText("Describing image...")
            task = DescribeImageTask(
                image_path,
                self,
                model=self.selected_model,
                provider=self.selected_provider,
                settings=self.llm_settings,
            )  # Pass selected model
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
        """Improves the caption in a background thread."""
        current_caption = self.get_caption()  # get plain text caption
        if not current_caption:
            print("Caption is empty. Nothing to improve.")
            return
        if self.selected_provider != "ollama":
            self.update_improve_status(
                "Caption improvement currently supports Ollama models. Please switch provider.",
                True,
            )
            return

        self.improve_label.setText("Improving...")
        self.improve_button.setEnabled(False)

        if self.image_path:
            task = ImproveCaptionTask(
                image_path=self.image_path,
                current_caption=current_caption,
                widget=self,
                model=self.selected_model,
                provider=self.selected_provider,
                settings=self.llm_settings,
            )
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

        span_id = self.current_ai_span_id
        if not span_id:
            # Fallback: append to end if span is unknown
            self.text_edit.moveCursor(QtGui.QTextCursor.End)
            self.text_edit.insertPlainText(chunk)
            return

        span_marker = f"id='{span_id}'"
        last_span_index = current_html.lower().rfind(span_marker.lower())

        if last_span_index != -1:
            insertion_index = current_html.lower().find(
                "</span>", last_span_index)
            if insertion_index != -1:
                # Reconstruct HTML by inserting chunk content
                new_html = current_html[:insertion_index] + \
                    chunk_escaped + current_html[insertion_index:]
                self.text_edit.setHtml(new_html)
            else:
                print("Error: Closing </span> tag for AI response span not found.")
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


class GeminiImageGenerationTask(QRunnable):
    """Generate an image using Gemini models in a background task."""

    def __init__(self, prompt: str, widget: "CaptionWidget", model: str):
        super().__init__()
        self.prompt = prompt
        self.widget = widget
        self.model = model

    def run(self) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError:
            message = (
                "The 'google-genai' package is not installed. "
                "Install it with `pip install google-genai`."
            )
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "on_image_generation_failed",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, message),
            )
            return

        try:
            config = resolve_llm_config(
                provider="gemini",
                model=self.model,
                persist=False,
            )
            ensure_provider_env(config)
            api_key = (
                config.api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            client = genai.Client(api_key=api_key)

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=self.prompt)],
                )
            ] if hasattr(types, "Content") else [
                {
                    "role": "user",
                    "parts": [{"text": self.prompt}],
                }
            ]

            image_config_cls = getattr(types, "ImageConfig", None)
            if image_config_cls is not None:
                image_config = image_config_cls(image_size="1K")
            else:
                image_config = None

            generate_config_cls = getattr(types, "GenerateContentConfig", None)
            if generate_config_cls is not None:
                if image_config is not None:
                    config_args = generate_config_cls(
                        response_modalities=["IMAGE", "TEXT"],
                        image_config=image_config,
                    )
                else:
                    config_args = generate_config_cls(
                        response_modalities=["IMAGE", "TEXT"],
                    )
            else:
                config_args = {
                    "response_modalities": ["IMAGE", "TEXT"],
                }
                if image_config:
                    config_args["image_config"] = image_config

            text_chunks: List[str] = []
            image_path: str = ""

            stream = client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config_args,
            )

            for chunk in stream:
                if getattr(chunk, "text", None):
                    text_chunks.append(chunk.text)

                try:
                    candidate = chunk.candidates[0]
                    parts = getattr(candidate.content, "parts", None)
                except (AttributeError, IndexError, TypeError):
                    continue

                if not parts:
                    continue

                for part in parts:
                    text_part = getattr(part, "text", None)
                    if text_part:
                        text_chunks.append(text_part)

                    inline = getattr(part, "inline_data", None)
                    if not inline or not getattr(inline, "data", None):
                        continue

                    data_buffer = inline.data
                    data_bytes: bytes
                    if isinstance(data_buffer, bytes):
                        try:
                            data_bytes = base64.b64decode(
                                data_buffer, validate=True)
                        except Exception:
                            data_bytes = data_buffer
                    elif isinstance(data_buffer, str):
                        try:
                            data_bytes = base64.b64decode(
                                data_buffer, validate=True)
                        except Exception:
                            data_bytes = data_buffer.encode("utf-8")
                    else:
                        data_bytes = bytes(data_buffer)

                    mime_type = getattr(
                        inline, "mime_type", None) or "image/png"
                    suffix = mimetypes.guess_extension(mime_type) or ".png"
                    with tempfile.NamedTemporaryFile(
                        prefix="annolid_gemini_",
                        suffix=suffix,
                        delete=False,
                    ) as tmp:
                        tmp.write(data_bytes)
                        image_path = tmp.name

            summary = "\n".join(text_chunks).strip()
            if image_path:
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "on_image_generation_success",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, image_path),
                    QtCore.Q_ARG(str, summary),
                )
                return

            raise RuntimeError(
                summary or "Gemini did not return image data.")

        except Exception as exc:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "on_image_generation_failed",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, str(exc)),
            )


class OpenAIImageGenerationTask(QRunnable):
    """Generate an image using OpenAI GPT-5 responses API."""

    def __init__(self, prompt: str, widget: "CaptionWidget", model: str, settings: Dict[str, Any]):
        super().__init__()
        self.prompt = prompt
        self.widget = widget
        self.model = model
        self.settings = settings

    def run(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            message = (
                "The 'openai' package is required for GPT image generation. "
                "Install it with `pip install openai`."
            )
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "on_image_generation_failed",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, message),
            )
            return

        try:
            config = self.settings.get("openai", {})
            api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is missing.")

            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            base_url = config.get("base_url")
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)

            has_responses = hasattr(client, "responses") and callable(
                getattr(client.responses, "create", None)
            )

            image_b64 = None
            summary = ""

            if has_responses:
                response = client.responses.create(
                    model=self.model,
                    input=self.prompt,
                    tools=[{"type": "image_generation"}],
                )

                outputs = (
                    getattr(response, "output", None)
                    or getattr(response, "outputs", None)
                    or []
                )
                text_chunks: List[str] = []

                for item in outputs:
                    item_type = getattr(item, "type", None) or (
                        item.get("type") if isinstance(item, dict) else None
                    )
                    if not item_type:
                        continue

                    if item_type == "message":
                        text = getattr(item, "content", None)
                        if isinstance(text, str):
                            text_chunks.append(text)
                    elif item_type == "image_generation_call":
                        result = getattr(item, "result", None)
                        if isinstance(result, list) and result:
                            result = result[0]
                        if isinstance(result, str):
                            image_b64 = result

                summary = getattr(response, "output_text", "") or "\n".join(text_chunks)

            else:
                image_b64 = None
                summary = ""
                images_api = getattr(client, "images", None)
                if images_api and hasattr(images_api, "generate"):
                    image_response = images_api.generate(
                        model=self.model,
                        prompt=self.prompt,
                    )
                    data = getattr(image_response, "data", None) or (
                        image_response.get("data", [])
                        if isinstance(image_response, dict)
                        else []
                    )
                    if data:
                        first = data[0]
                        image_b64 = first.get("b64_json") or first.get("b64")
                else:
                    try:
                        import openai as openai_legacy  # type: ignore
                    except ImportError:
                        openai_legacy = None
                    if openai_legacy is None:
                        raise AttributeError(
                            "OpenAI image generation not supported by installed SDK."
                        )
                    openai_legacy.api_key = api_key
                    if base_url:
                        openai_legacy.api_base = base_url
                    legacy_response = openai_legacy.Image.create(
                        model=self.model,
                        prompt=self.prompt,
                    )
                    data = legacy_response.get("data", [])
                    if data:
                        image_b64 = data[0].get("b64_json") or data[0].get("b64")

            if not image_b64:
                raise RuntimeError("OpenAI did not return image data.")

            image_bytes = base64.b64decode(image_b64)
            suffix = ".png"
            with tempfile.NamedTemporaryFile(
                prefix="annolid_openai_",
                suffix=suffix,
                delete=False,
            ) as tmp:
                tmp.write(image_bytes)
                image_path = tmp.name

            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "on_image_generation_success",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, image_path),
                QtCore.Q_ARG(str, summary or ""),
            )

        except Exception as exc:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "on_image_generation_failed",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, str(exc)),
            )


class ImproveCaptionTask(QRunnable):
    def __init__(self, image_path, current_caption, widget, model="llama3.2-vision:latest",
                 provider="ollama", settings=None):
        super().__init__()
        self.image_path = image_path
        self.current_caption = current_caption
        self.widget = widget
        self.model = model  # Store model
        self.provider = provider
        self.settings = settings or {}

    def run(self):
        try:
            if self.provider != "ollama":
                raise ValueError(
                    "Caption improvement is currently available only for Ollama providers.")

            ollama_module = globals().get("ollama")
            if ollama_module is None:
                raise ImportError(
                    "The python 'ollama' package is not installed.")

            host = self.settings.get("ollama", {}).get("host")
            if host:
                os.environ["OLLAMA_HOST"] = host

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

            response = ollama_module.chat(
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
                 model="llama3.2-vision:latest",
                 provider="ollama",
                 settings=None,
                 ):
        super().__init__()
        self.image_path = image_path
        self.widget = widget
        self.prompt = prompt
        self.model = model  # Store model
        self.provider = provider
        self.settings = settings or {}

    def run(self):
        """Runs the task in the background."""
        try:
            if self.provider != "ollama":
                raise ValueError(
                    "Image description is currently available only for Ollama providers.")

            ollama_module = globals().get("ollama")
            if ollama_module is None:
                raise ImportError(
                    "The python 'ollama' package is not installed.")

            host = self.settings.get("ollama", {}).get("host")
            if host:
                os.environ["OLLAMA_HOST"] = host

            response = ollama_module.chat(
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


# Generalised chat task supporting multiple providers
class StreamingChatTask(QRunnable):
    """A task to chat with the selected model in the background."""

    def __init__(
        self,
        prompt,
        image_path=None,
        widget=None,
        model="llama3.2-vision:latest",
        provider="ollama",
        settings=None,
    ):
        super().__init__()
        self.prompt = prompt
        self.image_path = image_path
        self.widget = widget
        self.model = model
        self.provider = provider
        self.settings = settings or {}

    def run(self):
        """Route chat request to the appropriate provider."""
        try:
            if self.provider == "ollama":
                self._run_ollama()
            elif self.provider == "openai":
                self._run_openai()
            elif self.provider == "gemini":
                self._run_gemini()
            else:
                raise ValueError(f"Unsupported provider '{self.provider}'.")
        except Exception as e:
            error_message = f"Error in chat interaction: {e}"
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "update_chat_response",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, str(error_message)),
                QtCore.Q_ARG(bool, True),
            )

    # ------------------------------------------------------------------ #
    # Provider handlers
    # ------------------------------------------------------------------ #
    def _run_ollama(self) -> None:
        ollama_module = globals().get("ollama")
        if ollama_module is None:
            raise ImportError("The python 'ollama' package is not installed.")

        host = self.settings.get("ollama", {}).get("host")
        if host:
            os.environ["OLLAMA_HOST"] = host

        messages = [{'role': 'user', 'content': self.prompt}]
        if self.image_path and os.path.exists(self.image_path):
            messages[0]['images'] = [self.image_path]

        stream = ollama_module.chat(
            model=self.model,
            messages=messages,
            stream=True
        )
        full_response = ""

        for part in stream:
            if 'message' in part and 'content' in part['message']:
                chunk = part['message']['content']
                full_response += chunk
                QMetaObject.invokeMethod(
                    self.widget,
                    "stream_chat_chunk",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, chunk),
                )
            elif 'error' in part:
                error_message = f"Stream error: {part['error']}"
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "update_chat_response",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, error_message),
                    QtCore.Q_ARG(bool, True),
                )
                return

        if not full_response.strip():
            error_message = "No response from Ollama."
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "update_chat_response",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, error_message),
                QtCore.Q_ARG(bool, True),
            )
        else:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "update_chat_response",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, ""),
                QtCore.Q_ARG(bool, False),
            )

    def _run_openai(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GPT providers."
            ) from exc

        config = self.settings.get("openai", {})
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Configure it in settings.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        user_prompt = self.prompt
        if self.image_path and os.path.exists(self.image_path):
            user_prompt += (
                f"\n\n[Note: Image context available at {self.image_path}. "
                "Upload handling is not automated; describe based on this reminder.]"
            )

        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        model_lower = (self.model or "").lower()
        if "gpt-5" not in model_lower:
            request_payload["temperature"] = 0.7

        response = client.chat.completions.create(**request_payload)
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""

        QtCore.QMetaObject.invokeMethod(
            self.widget,
            "update_chat_response",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
            QtCore.Q_ARG(bool, False),
        )

    def _run_gemini(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for Gemini providers."
            ) from exc

        config = self.settings.get("gemini", {})
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is missing. Configure it in settings.")

        genai.configure(api_key=api_key)
        model_name = self.model or "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)

        user_prompt = self.prompt
        if self.image_path and os.path.exists(self.image_path):
            user_prompt += (
                f"\n\n[Note: Image context available at {self.image_path}. "
                "Upload handling is not automated; describe based on this reminder.]"
            )

        result = model.generate_content(user_prompt)
        text = getattr(result, "text", "") or ""

        QtCore.QMetaObject.invokeMethod(
            self.widget,
            "update_chat_response",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, text),
            QtCore.Q_ARG(bool, False),
        )
