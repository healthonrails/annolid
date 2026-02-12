from __future__ import annotations

import os
from typing import Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.utils.llm_settings import default_settings
from annolid.utils.tts_settings import (
    default_tts_settings,
    load_tts_settings,
    save_tts_settings,
)
from annolid.agents.kokoro_tts import (
    get_available_voices as get_kokoro_voices,
    get_suggested_languages,
)
from annolid.agents.pocket_tts import (
    get_available_voices as get_pocket_voices,
)


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
            "API keys are not persisted in ~/.annolid/llm_settings.json.\n"
            "Set keys here for the current session and/or via environment variables."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        self._tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self._tabs, 1)

        self._build_ollama_tab()
        self._build_openai_tab()
        self._build_openrouter_tab()
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

        self.ollama_models_list = self._create_model_list_editor(
            self._settings["ollama"].get("preferred_models", []),
            placeholder="Type model name (e.g. qwen3-vl) and press Add",
        )
        layout.addRow("Preferred models:", self.ollama_models_list["container"])

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
            self._settings["openai"].get("base_url", "https://api.openai.com/v1")
        )
        layout.addRow("Base URL:", self.openai_base_url_edit)

        self.openai_models_list = self._create_model_list_editor(
            self._settings["openai"].get("preferred_models", []),
            placeholder="Type model name (e.g. gpt-4o-mini) and press Add",
        )
        layout.addRow("Preferred models:", self.openai_models_list["container"])

        self._tabs.addTab(widget, "OpenAI GPT")

    def _build_openrouter_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        openrouter_cfg = dict(self._settings.get("openrouter", {}) or {})

        self.openrouter_key_edit = QtWidgets.QLineEdit(
            openrouter_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY", "")
        )
        self.openrouter_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addRow("API key:", self.openrouter_key_edit)

        toggle_button = QtWidgets.QPushButton("Show")
        toggle_button.setCheckable(True)
        toggle_button.toggled.connect(
            lambda checked: self.openrouter_key_edit.setEchoMode(
                QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
            )
        )
        layout.addRow("Reveal key:", toggle_button)

        self.openrouter_base_url_edit = QtWidgets.QLineEdit(
            openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1")
        )
        layout.addRow("Base URL:", self.openrouter_base_url_edit)

        self.openrouter_models_list = self._create_model_list_editor(
            openrouter_cfg.get("preferred_models", []),
            placeholder="Type model name (e.g. openai/gpt-4o-mini) and press Add",
        )
        layout.addRow("Preferred models:", self.openrouter_models_list["container"])

        self._tabs.addTab(widget, "OpenRouter")

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

        self.gemini_models_list = self._create_model_list_editor(
            self._settings["gemini"].get("preferred_models", []),
            placeholder="Type model name (e.g. gemini-1.5-flash) and press Add",
        )
        layout.addRow("Preferred models:", self.gemini_models_list["container"])

        self._tabs.addTab(widget, "Google Gemini")

    def _build_tts_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        info_label = QtWidgets.QLabel(
            "Text-to-speech settings are stored in ~/.annolid/tts_settings.json."
        )
        info_label.setWordWrap(True)
        layout.addRow(info_label)

        self.tts_engine_combo = QtWidgets.QComboBox()
        self.tts_engine_combo.addItem(
            "Auto (Kokoro → Pocket → Chatterbox → gTTS)", "auto"
        )
        self.tts_engine_combo.addItem("Kokoro (local voices)", "kokoro")
        self.tts_engine_combo.addItem("Pocket (Kyutai)", "pocket")
        self.tts_engine_combo.addItem("Chatterbox Turbo (voice cloning)", "chatterbox")
        self.tts_engine_combo.addItem("gTTS (online)", "gtts")
        current_engine = str(
            self._tts_settings.get("engine", self._tts_defaults.get("engine", "auto"))
        ).strip()
        idx = self.tts_engine_combo.findData(current_engine)
        if idx >= 0:
            self.tts_engine_combo.setCurrentIndex(idx)
        layout.addRow("Engine:", self.tts_engine_combo)

        self.tts_voice_combo = QtWidgets.QComboBox()
        self.tts_voice_combo.setEditable(True)
        self.tts_voice_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        voices = get_kokoro_voices()
        if voices:
            self.tts_voice_combo.addItems(voices)
        self.tts_voice_combo.setCurrentText(
            str(self._tts_settings.get("voice", self._tts_defaults["voice"]))
        )
        layout.addRow("Kokoro voice:", self.tts_voice_combo)

        self.tts_pocket_voice_combo = QtWidgets.QComboBox()
        self.tts_pocket_voice_combo.setEditable(True)
        self.tts_pocket_voice_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        pocket_voices = get_pocket_voices()
        if pocket_voices:
            self.tts_pocket_voice_combo.addItems(pocket_voices)
        self.tts_pocket_voice_combo.setCurrentText(
            str(
                self._tts_settings.get(
                    "pocket_voice", self._tts_defaults["pocket_voice"]
                )
            )
        )
        layout.addRow("Pocket voice:", self.tts_pocket_voice_combo)

        self.tts_pocket_speed_spin = QtWidgets.QDoubleSpinBox()
        self.tts_pocket_speed_spin.setRange(0.5, 2.0)
        self.tts_pocket_speed_spin.setSingleStep(0.05)
        self.tts_pocket_speed_spin.setDecimals(2)
        self.tts_pocket_speed_spin.setValue(
            float(
                self._tts_settings.get(
                    "pocket_speed", self._tts_defaults["pocket_speed"]
                )
            )
        )
        layout.addRow("Pocket speed:", self.tts_pocket_speed_spin)

        self.tts_pocket_prompt_edit = QtWidgets.QLineEdit()
        self.tts_pocket_prompt_edit.setPlaceholderText("Optional voice prompt WAV")
        self.tts_pocket_prompt_edit.setText(
            str(self._tts_settings.get("pocket_prompt_path", ""))
        )
        pocket_prompt_browse = QtWidgets.QPushButton("Browse…")
        pocket_prompt_browse.clicked.connect(self._browse_pocket_prompt)
        pocket_prompt_clear = QtWidgets.QPushButton("Clear")
        pocket_prompt_clear.clicked.connect(self._clear_pocket_prompt)
        prompt_row = QtWidgets.QWidget()
        prompt_layout = QtWidgets.QHBoxLayout(prompt_row)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.addWidget(self.tts_pocket_prompt_edit, 1)
        prompt_layout.addWidget(pocket_prompt_browse)
        prompt_layout.addWidget(pocket_prompt_clear)
        layout.addRow("Pocket prompt:", prompt_row)

        voice_prompt_row = QtWidgets.QWidget()
        voice_prompt_layout = QtWidgets.QHBoxLayout(voice_prompt_row)
        voice_prompt_layout.setContentsMargins(0, 0, 0, 0)
        self.tts_chatterbox_voice_edit = QtWidgets.QLineEdit()
        self.tts_chatterbox_voice_edit.setPlaceholderText(
            "Path to a short voice prompt WAV"
        )
        self.tts_chatterbox_voice_edit.setText(
            str(self._tts_settings.get("chatterbox_voice_path", ""))
        )
        browse_button = QtWidgets.QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_chatterbox_voice_prompt)
        voice_prompt_layout.addWidget(self.tts_chatterbox_voice_edit, 1)
        voice_prompt_layout.addWidget(browse_button)
        layout.addRow("Chatterbox voice:", voice_prompt_row)

        self.tts_chatterbox_dtype_combo = QtWidgets.QComboBox()
        self.tts_chatterbox_dtype_combo.addItems(["fp32", "fp16", "q8", "q4", "q4f16"])
        self.tts_chatterbox_dtype_combo.setCurrentText(
            str(self._tts_settings.get("chatterbox_dtype", "fp32"))
        )
        layout.addRow("Chatterbox dtype:", self.tts_chatterbox_dtype_combo)

        self.tts_chatterbox_max_tokens_spin = QtWidgets.QSpinBox()
        self.tts_chatterbox_max_tokens_spin.setRange(128, 4096)
        self.tts_chatterbox_max_tokens_spin.setSingleStep(128)
        self.tts_chatterbox_max_tokens_spin.setValue(
            int(self._tts_settings.get("chatterbox_max_new_tokens", 1024))
        )
        layout.addRow("Chatterbox max tokens:", self.tts_chatterbox_max_tokens_spin)

        self.tts_chatterbox_rep_penalty_spin = QtWidgets.QDoubleSpinBox()
        self.tts_chatterbox_rep_penalty_spin.setRange(1.0, 3.0)
        self.tts_chatterbox_rep_penalty_spin.setSingleStep(0.05)
        self.tts_chatterbox_rep_penalty_spin.setDecimals(2)
        self.tts_chatterbox_rep_penalty_spin.setValue(
            float(self._tts_settings.get("chatterbox_repetition_penalty", 1.2))
        )
        layout.addRow("Chatterbox repetition:", self.tts_chatterbox_rep_penalty_spin)

        self.tts_chatterbox_watermark_check = QtWidgets.QCheckBox()
        self.tts_chatterbox_watermark_check.setChecked(
            bool(self._tts_settings.get("chatterbox_apply_watermark", False))
        )
        layout.addRow("Chatterbox watermark:", self.tts_chatterbox_watermark_check)

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

    @QtCore.Slot()
    def _browse_chatterbox_voice_prompt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Voice Prompt Audio",
            os.path.expanduser("~"),
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)",
        )
        if path:
            self.tts_chatterbox_voice_edit.setText(path)

    def _browse_pocket_prompt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Pocket TTS Voice Prompt",
            os.path.expanduser("~"),
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)",
        )
        if path:
            self.tts_pocket_prompt_edit.setText(path)

    def _clear_pocket_prompt(self) -> None:
        self.tts_pocket_prompt_edit.setText("")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _normalize_models(self, raw_models) -> List[str]:
        if isinstance(raw_models, str):
            raw_models = [raw_models]
        if not isinstance(raw_models, list):
            return []
        models: List[str] = []
        seen = set()
        for value in raw_models:
            model = str(value or "").strip()
            if not model or model in seen:
                continue
            seen.add(model)
            models.append(model)
        return models

    def _create_model_list_editor(
        self, initial_models, *, placeholder: str
    ) -> Dict[str, QtWidgets.QWidget]:
        container = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        model_input = QtWidgets.QLineEdit(container)
        model_input.setPlaceholderText(placeholder)
        add_button = QtWidgets.QPushButton("Add", container)
        row.addWidget(model_input, 1)
        row.addWidget(add_button)
        root.addLayout(row)

        models_list = QtWidgets.QListWidget(container)
        models_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        models_list.setAlternatingRowColors(True)
        models_list.setMinimumHeight(120)
        root.addWidget(models_list, 1)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        remove_button = QtWidgets.QPushButton("Remove", container)
        up_button = QtWidgets.QPushButton("Move Up", container)
        down_button = QtWidgets.QPushButton("Move Down", container)
        clear_button = QtWidgets.QPushButton("Clear", container)
        controls_row.addWidget(remove_button)
        controls_row.addWidget(up_button)
        controls_row.addWidget(down_button)
        controls_row.addWidget(clear_button)
        controls_row.addStretch(1)
        root.addLayout(controls_row)

        self._set_models_list(models_list, self._normalize_models(initial_models))

        add_button.clicked.connect(
            lambda: self._add_model_from_input(models_list, model_input)
        )
        model_input.returnPressed.connect(
            lambda: self._add_model_from_input(models_list, model_input)
        )
        remove_button.clicked.connect(lambda: self._remove_selected_models(models_list))
        up_button.clicked.connect(lambda: self._move_selected_models(models_list, -1))
        down_button.clicked.connect(lambda: self._move_selected_models(models_list, 1))
        clear_button.clicked.connect(models_list.clear)

        return {
            "container": container,
            "list": models_list,
            "input": model_input,
        }

    def _set_models_list(
        self, models_list: QtWidgets.QListWidget, models: List[str]
    ) -> None:
        models_list.clear()
        for model in models:
            models_list.addItem(model)

    def _get_models_from_list(self, models_list: QtWidgets.QListWidget) -> List[str]:
        out: List[str] = []
        for idx in range(models_list.count()):
            text = models_list.item(idx).text().strip()
            if text and text not in out:
                out.append(text)
        return out

    def _add_model_from_input(
        self, models_list: QtWidgets.QListWidget, model_input: QtWidgets.QLineEdit
    ) -> None:
        model = model_input.text().strip()
        if not model:
            return
        for idx in range(models_list.count()):
            if models_list.item(idx).text().strip() == model:
                models_list.setCurrentRow(idx)
                model_input.clear()
                return
        models_list.addItem(model)
        models_list.setCurrentRow(models_list.count() - 1)
        model_input.clear()

    def _remove_selected_models(self, models_list: QtWidgets.QListWidget) -> None:
        rows = sorted(
            {idx.row() for idx in models_list.selectedIndexes()}, reverse=True
        )
        for row in rows:
            models_list.takeItem(row)

    def _move_selected_models(
        self, models_list: QtWidgets.QListWidget, offset: int
    ) -> None:
        if offset == 0:
            return
        count = models_list.count()
        if count <= 1:
            return
        selected_rows = sorted({idx.row() for idx in models_list.selectedIndexes()})
        if not selected_rows:
            return
        if offset < 0:
            if selected_rows[0] == 0:
                return
            iterator = selected_rows
        else:
            if selected_rows[-1] >= count - 1:
                return
            iterator = list(reversed(selected_rows))
        for row in iterator:
            item = models_list.takeItem(row)
            if item is None:
                continue
            new_row = row + offset
            models_list.insertItem(new_row, item)
            item.setSelected(True)

    @QtCore.Slot()
    def _refresh_ollama_models(self) -> None:
        """Query the Ollama server for available models and update the list."""
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

            self._set_models_list(self.ollama_models_list["list"], models)
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
        openai_base_url = (
            self.openai_base_url_edit.text().strip() or "https://api.openai.com/v1"
        )
        openai_key = self.openai_key_edit.text().strip()
        openrouter_key = self.openrouter_key_edit.text().strip()
        openrouter_base_url = (
            self.openrouter_base_url_edit.text().strip()
            or "https://openrouter.ai/api/v1"
        )

        updated["ollama"] = {
            "host": self.ollama_host_edit.text().strip() or "http://localhost:11434",
            "preferred_models": self._get_models_from_list(
                self.ollama_models_list["list"]
            ),
        }
        updated["openai"] = {
            "api_key": openai_key,
            "base_url": openai_base_url,
            "preferred_models": self._get_models_from_list(
                self.openai_models_list["list"]
            ),
        }
        updated["openrouter"] = {
            "api_key": openrouter_key,
            "base_url": openrouter_base_url,
            "preferred_models": self._get_models_from_list(
                self.openrouter_models_list["list"]
            ),
        }
        updated["gemini"] = {
            "api_key": self.gemini_key_edit.text().strip(),
            "preferred_models": self._get_models_from_list(
                self.gemini_models_list["list"]
            ),
        }
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if openai_base_url:
            os.environ["OPENAI_BASE_URL"] = openai_base_url

        self._tts_settings = {
            "engine": self.tts_engine_combo.currentData() or "auto",
            "voice": self.tts_voice_combo.currentText().strip()
            or self._tts_defaults["voice"],
            "pocket_voice": self.tts_pocket_voice_combo.currentText().strip()
            or self._tts_defaults["pocket_voice"],
            "pocket_speed": float(self.tts_pocket_speed_spin.value()),
            "pocket_prompt_path": self.tts_pocket_prompt_edit.text().strip(),
            "lang": self.tts_lang_combo.currentText().strip()
            or self._tts_defaults["lang"],
            "speed": float(self.tts_speed_spin.value()),
            "chatterbox_voice_path": self.tts_chatterbox_voice_edit.text().strip(),
            "chatterbox_dtype": self.tts_chatterbox_dtype_combo.currentText().strip()
            or "fp32",
            "chatterbox_max_new_tokens": int(
                self.tts_chatterbox_max_tokens_spin.value()
            ),
            "chatterbox_repetition_penalty": float(
                self.tts_chatterbox_rep_penalty_spin.value()
            ),
            "chatterbox_apply_watermark": bool(
                self.tts_chatterbox_watermark_check.isChecked()
            ),
        }
        save_tts_settings(self._tts_settings)
        self._settings = updated
        super().accept()
