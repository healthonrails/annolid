from __future__ import annotations

import os
from typing import Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.utils.llm_settings import default_settings
from annolid.utils.tts_settings import default_tts_settings, load_tts_settings, save_tts_settings
from annolid.agents.kokoro_tts import get_available_voices, get_suggested_languages


def _list_to_text(items: List[str]) -> str:
    return "\n".join(items)


def _text_to_list(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class LLMSettingsDialog(QtWidgets.QDialog):
    """
    Dialog for configuring Large Language Model providers.

    Users can specify API keys, preferred model identifiers, and Ollama host
    settings without editing environment variables manually.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        settings: Optional[Dict] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Model Settings")
        self.resize(520, 420)

        self._settings = default_settings() if settings is None else settings
        self._settings = {**default_settings(), **self._settings}
        self._tts_settings = load_tts_settings()
        self._tts_defaults = default_tts_settings()

        main_layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "API keys are stored in plain text inside ~/.annolid/llm_settings.json.\n"
            "Ensure this device is secure before saving credentials."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        self._tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self._tabs, 1)

        self._build_ollama_tab()
        self._build_openai_tab()
        self._build_gemini_tab()
        self._build_tts_tab()

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    # ------------------------------------------------------------------ #
    # Tab builders
    # ------------------------------------------------------------------ #
    def _build_ollama_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.ollama_host_edit = QtWidgets.QLineEdit(
            self._settings["ollama"].get("host", "http://localhost:11434")
        )
        layout.addRow("Server URL:", self.ollama_host_edit)

        self.ollama_models_edit = QtWidgets.QPlainTextEdit()
        self.ollama_models_edit.setPlaceholderText(
            "Optional: pin model names (one per line) to show in the selector."
        )
        self.ollama_models_edit.setPlainText(
            _list_to_text(self._settings["ollama"].get("preferred_models", []))
        )
        layout.addRow("Preferred models:", self.ollama_models_edit)

        refresh_button = QtWidgets.QPushButton("Refresh from Ollama")
        refresh_button.clicked.connect(self._refresh_ollama_models)
        layout.addRow(refresh_button)

        self._tabs.addTab(widget, "Ollama")

    def _build_openai_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.openai_key_edit = QtWidgets.QLineEdit(
            self._settings["openai"].get("api_key", "")
        )
        self.openai_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addRow("API key:", self.openai_key_edit)

        toggle_button = QtWidgets.QPushButton("Show")
        toggle_button.setCheckable(True)
        toggle_button.toggled.connect(
            lambda checked: self.openai_key_edit.setEchoMode(
                QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
            )
        )
        layout.addRow("Reveal key:", toggle_button)

        self.openai_base_url_edit = QtWidgets.QLineEdit(
            self._settings["openai"].get(
                "base_url", "https://api.openai.com/v1")
        )
        layout.addRow("Base URL:", self.openai_base_url_edit)

        self.openai_models_edit = QtWidgets.QPlainTextEdit()
        self.openai_models_edit.setPlainText(
            _list_to_text(self._settings["openai"].get("preferred_models", []))
        )
        self.openai_models_edit.setPlaceholderText(
            "One model per line, e.g. gpt-4o-mini")
        layout.addRow("Preferred models:", self.openai_models_edit)

        self._tabs.addTab(widget, "OpenAI GPT")

    def _build_gemini_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.gemini_key_edit = QtWidgets.QLineEdit(
            self._settings["gemini"].get("api_key", "")
        )
        self.gemini_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addRow("API key:", self.gemini_key_edit)

        toggle_button = QtWidgets.QPushButton("Show")
        toggle_button.setCheckable(True)
        toggle_button.toggled.connect(
            lambda checked: self.gemini_key_edit.setEchoMode(
                QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
            )
        )
        layout.addRow("Reveal key:", toggle_button)

        self.gemini_models_edit = QtWidgets.QPlainTextEdit()
        self.gemini_models_edit.setPlainText(
            _list_to_text(self._settings["gemini"].get("preferred_models", []))
        )
        self.gemini_models_edit.setPlaceholderText(
            "One model per line, e.g. gemini-1.5-flash"
        )
        layout.addRow("Preferred models:", self.gemini_models_edit)

        self._tabs.addTab(widget, "Google Gemini")

    def _build_tts_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        info_label = QtWidgets.QLabel(
            "Text-to-speech settings are stored in ~/.annolid/tts_settings.json."
        )
        info_label.setWordWrap(True)
        layout.addRow(info_label)

        self.tts_voice_combo = QtWidgets.QComboBox()
        self.tts_voice_combo.setEditable(True)
        self.tts_voice_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        voices = get_available_voices()
        if voices:
            self.tts_voice_combo.addItems(voices)
        self.tts_voice_combo.setCurrentText(
            str(self._tts_settings.get("voice", self._tts_defaults["voice"]))
        )
        layout.addRow("Voice:", self.tts_voice_combo)

        self.tts_lang_combo = QtWidgets.QComboBox()
        self.tts_lang_combo.setEditable(True)
        self.tts_lang_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        languages = get_suggested_languages()
        if languages:
            self.tts_lang_combo.addItems(languages)
        self.tts_lang_combo.setCurrentText(
            str(self._tts_settings.get("lang", self._tts_defaults["lang"]))
        )
        layout.addRow("Language:", self.tts_lang_combo)

        self.tts_speed_spin = QtWidgets.QDoubleSpinBox()
        self.tts_speed_spin.setRange(0.5, 2.0)
        self.tts_speed_spin.setSingleStep(0.05)
        self.tts_speed_spin.setDecimals(2)
        self.tts_speed_spin.setValue(
            float(self._tts_settings.get("speed", self._tts_defaults["speed"]))
        )
        layout.addRow("Speed:", self.tts_speed_spin)

        self._tabs.addTab(widget, "Text-to-Speech")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @QtCore.Slot()
    def _refresh_ollama_models(self) -> None:
        """Query the Ollama server for available models and update the text box."""
        host = self.ollama_host_edit.text().strip() or "http://localhost:11434"
        try:
            import ollama  # type: ignore
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self,
                "Ollama Not Installed",
                "The python 'ollama' package is required to list models.",
            )
            return

        try:
            client_factory = getattr(ollama, "Client", None)
            if callable(client_factory):
                client = client_factory(host=host)
                response = client.list()
            else:
                prev_host = os.environ.get("OLLAMA_HOST")
                os.environ["OLLAMA_HOST"] = host
                try:
                    response = ollama.list()
                finally:
                    if prev_host is not None:
                        os.environ["OLLAMA_HOST"] = prev_host
                    else:
                        os.environ.pop("OLLAMA_HOST", None)

            models = []
            if isinstance(response, dict):
                models = [
                    model.get("name", "")
                    for model in response.get("models", [])
                    if isinstance(model, dict)
                ]
            if not models:
                QtWidgets.QMessageBox.information(
                    self,
                    "No Models Found",
                    "No models were returned by the Ollama server.",
                )
                return

            self.ollama_models_edit.setPlainText(_list_to_text(models))
        except Exception as exc:  # pragma: no cover - defensive UI
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to list models",
                f"Could not reach Ollama at {host}.\n\n{exc}",
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_settings(self) -> Dict:
        return self._settings

    def accept(self) -> None:  # type: ignore[override]
        updated = dict(self._settings)
        updated["ollama"] = {
            "host": self.ollama_host_edit.text().strip() or "http://localhost:11434",
            "preferred_models": _text_to_list(self.ollama_models_edit.toPlainText()),
        }
        updated["openai"] = {
            "api_key": self.openai_key_edit.text().strip(),
            "base_url": self.openai_base_url_edit.text().strip()
            or "https://api.openai.com/v1",
            "preferred_models": _text_to_list(self.openai_models_edit.toPlainText()),
        }
        updated["gemini"] = {
            "api_key": self.gemini_key_edit.text().strip(),
            "preferred_models": _text_to_list(self.gemini_models_edit.toPlainText()),
        }
        self._tts_settings = {
            "voice": self.tts_voice_combo.currentText().strip()
            or self._tts_defaults["voice"],
            "lang": self.tts_lang_combo.currentText().strip()
            or self._tts_defaults["lang"],
            "speed": float(self.tts_speed_spin.value()),
        }
        save_tts_settings(self._tts_settings)
        self._settings = updated
        super().accept()
