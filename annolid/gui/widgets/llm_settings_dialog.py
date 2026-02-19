from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.core.agent.config import load_config, save_config
from annolid.utils.llm_settings import (
    default_settings,
    global_env_path,
    persist_global_env_vars,
)
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
        try:
            self._agent_config = load_config()
        except Exception:
            self._agent_config = None
        self._tts_settings = load_tts_settings()
        self._tts_defaults = default_tts_settings()
        self._provider_specs: Dict[str, Dict[str, Any]] = dict(
            self._settings.get("provider_definitions", {}) or {}
        )
        self._provider_specs.setdefault(
            "openai",
            {
                "label": "OpenAI GPT",
                "kind": "openai_compat",
                "env_keys": ["OPENAI_API_KEY"],
                "api_key_env": ["OPENAI_API_KEY"],
                "base_url_default": "https://api.openai.com/v1",
                "base_url_env": "OPENAI_BASE_URL",
                "model_placeholder": "Type model name (e.g. gpt-4o-mini) and press Add",
            },
        )
        self._provider_specs.setdefault(
            "openrouter",
            {
                "label": "OpenRouter",
                "kind": "openai_compat",
                "env_keys": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
                "api_key_env": ["OPENROUTER_API_KEY"],
                "base_url_default": "https://openrouter.ai/api/v1",
                "model_placeholder": (
                    "Type model name (e.g. openai/gpt-4o-mini) and press Add"
                ),
            },
        )
        self._provider_specs.setdefault(
            "gemini",
            {
                "label": "Google Gemini",
                "kind": "gemini",
                "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                "api_key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                "model_placeholder": (
                    "Type model name (e.g. gemini-1.5-flash) and press Add"
                ),
            },
        )
        self._provider_specs.setdefault(
            "ollama",
            {
                "label": "Ollama (local)",
                "kind": "ollama",
                "env_keys": ["OLLAMA_HOST"],
                "host_default": "http://localhost:11434",
                "model_placeholder": "Type model name (e.g. qwen3-vl) and press Add",
            },
        )
        self._provider_env_keys: Dict[str, List[str]] = {
            provider: [str(v).strip() for v in list(spec.get("env_keys", []) or [])]
            for provider, spec in self._provider_specs.items()
        }
        self._provider_widgets: Dict[str, Dict[str, Any]] = {}
        self._api_key_inputs: Dict[str, QtWidgets.QLineEdit] = {}
        self._api_key_status_labels: Dict[str, QtWidgets.QLabel] = {}

        main_layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "API keys entered here are session-only and are not persisted to "
            "~/.annolid/llm_settings.json.\n"
            "Use environment variables for durable credentials."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)
        add_provider_button = QtWidgets.QPushButton("Add Provider")
        add_provider_button.clicked.connect(self._add_provider_dialog)
        main_layout.addWidget(add_provider_button, 0, QtCore.Qt.AlignLeft)
        self.persist_env_checkbox = QtWidgets.QCheckBox(
            f"Persist entered credentials to {global_env_path()} (optional)"
        )
        self.persist_env_checkbox.setChecked(False)
        main_layout.addWidget(self.persist_env_checkbox, 0, QtCore.Qt.AlignLeft)

        self._tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self._tabs, 1)

        self._build_ollama_tab()
        for provider, spec in self._provider_specs.items():
            if str(spec.get("kind") or "").strip().lower() == "ollama":
                continue
            self._build_provider_tab(provider)
        self._build_agent_runtime_tab()
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
        self._ollama_tab_widget = widget
        layout = QtWidgets.QFormLayout(widget)
        spec = self._provider_specs.get("ollama", {})
        ollama_cfg = dict(self._settings.get("ollama", {}) or {})
        default_host = str(spec.get("host_default") or "http://localhost:11434")

        self.ollama_host_edit = QtWidgets.QLineEdit(
            str(ollama_cfg.get("host") or default_host)
        )
        layout.addRow("Server URL:", self.ollama_host_edit)

        self.ollama_models_list = self._create_model_list_editor(
            ollama_cfg.get("preferred_models", []),
            placeholder=str(
                spec.get("model_placeholder")
                or "Type model name (e.g. qwen3-vl) and press Add"
            ),
        )
        layout.addRow("Preferred models:", self.ollama_models_list["container"])

        refresh_button = QtWidgets.QPushButton("Refresh from Ollama")
        refresh_button.clicked.connect(self._refresh_ollama_models)
        layout.addRow(refresh_button)

        self._tabs.addTab(widget, str(spec.get("label") or "Ollama"))

    def _build_openai_tab(self) -> None:
        self._build_provider_tab("openai")

    def _build_openrouter_tab(self) -> None:
        self._build_provider_tab("openrouter")

    def _build_gemini_tab(self) -> None:
        self._build_provider_tab("gemini")

    def _build_provider_tab(self, provider: str) -> None:
        spec = self._provider_specs.get(provider)
        if not spec:
            return
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        provider_cfg = dict(self._settings.get(provider, {}) or {})
        kind = str(spec.get("kind") or "openai_compat").strip().lower()
        self._provider_widgets.setdefault(provider, {})["tab_widget"] = widget
        key_edit = self._add_api_key_controls(
            layout,
            provider=provider,
            initial_value=self._resolve_initial_api_key(provider, provider_cfg),
        )
        self._provider_widgets.setdefault(provider, {})["key_edit"] = key_edit

        base_url_edit: Optional[QtWidgets.QLineEdit] = None
        if kind == "openai_compat":
            default_base_url = str(spec.get("base_url_default") or "").strip()
            base_url_edit = QtWidgets.QLineEdit(
                str(provider_cfg.get("base_url") or default_base_url)
            )
            layout.addRow("Base URL:", base_url_edit)
        self._provider_widgets.setdefault(provider, {})["base_url_edit"] = base_url_edit

        models_list = self._create_model_list_editor(
            provider_cfg.get("preferred_models", []),
            placeholder=str(
                spec.get("model_placeholder", "Type model name and press Add")
            ),
        )
        layout.addRow("Preferred models:", models_list["container"])
        self._provider_widgets.setdefault(provider, {})["models_list"] = models_list

        # Backward-compatible widget aliases used in tests/callers.
        setattr(self, f"{provider}_key_edit", key_edit)
        setattr(self, f"{provider}_models_list", models_list)
        if base_url_edit is not None:
            setattr(self, f"{provider}_base_url_edit", base_url_edit)

        if provider not in {"openai", "openrouter", "gemini"}:
            remove_button = QtWidgets.QPushButton("Remove Provider")
            remove_button.clicked.connect(
                lambda _checked=False, p=provider: self._remove_provider_tab(p)
            )
            layout.addRow(remove_button)

        self._tabs.addTab(widget, str(spec.get("label") or provider.title()))

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

    def _build_agent_runtime_tab(self) -> None:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        agent_cfg = dict(self._settings.get("agent") or {})

        note = QtWidgets.QLabel(
            "Agent runtime timeout controls (seconds). Lower values fail faster."
        )
        note.setWordWrap(True)
        layout.addRow(note)

        self.enable_progress_stream_checkbox = QtWidgets.QCheckBox(
            "Enable intermediate progress stream", widget
        )
        self.enable_progress_stream_checkbox.setChecked(
            bool(agent_cfg.get("enable_progress_stream", True))
        )
        layout.addRow(self.enable_progress_stream_checkbox)

        self.browser_first_for_web_checkbox = QtWidgets.QCheckBox(
            "Prefer MCP browser for web tasks", widget
        )
        self.browser_first_for_web_checkbox.setChecked(
            bool(agent_cfg.get("browser_first_for_web", True))
        )
        layout.addRow(self.browser_first_for_web_checkbox)

        def _make_spin(
            value: float,
            *,
            minimum: float = 2.0,
            maximum: float = 300.0,
            step: float = 1.0,
            decimals: int = 0,
        ) -> QtWidgets.QDoubleSpinBox:
            spin = QtWidgets.QDoubleSpinBox(widget)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(float(value))
            return spin

        self.fast_mode_timeout_spin = _make_spin(
            agent_cfg.get("fast_mode_timeout_seconds", 60),
            minimum=10.0,
            maximum=300.0,
        )
        layout.addRow("Fast mode timeout:", self.fast_mode_timeout_spin)

        self.fallback_retry_timeout_spin = _make_spin(
            agent_cfg.get("fallback_retry_timeout_seconds", 20),
            minimum=5.0,
            maximum=60.0,
        )
        layout.addRow("Fallback retry timeout:", self.fallback_retry_timeout_spin)

        self.loop_llm_timeout_spin = _make_spin(
            agent_cfg.get("loop_llm_timeout_seconds", 60),
            minimum=5.0,
            maximum=300.0,
        )
        layout.addRow("Agent loop LLM timeout:", self.loop_llm_timeout_spin)

        self.loop_llm_timeout_no_tools_spin = _make_spin(
            agent_cfg.get("loop_llm_timeout_seconds_no_tools", 40),
            minimum=5.0,
            maximum=300.0,
        )
        layout.addRow(
            "Agent loop timeout (no tools):",
            self.loop_llm_timeout_no_tools_spin,
        )

        self.loop_tool_timeout_spin = _make_spin(
            agent_cfg.get("loop_tool_timeout_seconds", 20),
            minimum=3.0,
            maximum=120.0,
        )
        layout.addRow("Agent tool timeout:", self.loop_tool_timeout_spin)

        self.ollama_tool_timeout_spin = _make_spin(
            agent_cfg.get("ollama_tool_timeout_seconds", 45),
            minimum=5.0,
            maximum=180.0,
        )
        layout.addRow("Ollama tool request timeout:", self.ollama_tool_timeout_spin)

        self.ollama_plain_timeout_spin = _make_spin(
            agent_cfg.get("ollama_plain_timeout_seconds", 25),
            minimum=5.0,
            maximum=180.0,
        )
        layout.addRow("Ollama plain request timeout:", self.ollama_plain_timeout_spin)

        self.ollama_plain_recovery_timeout_spin = _make_spin(
            agent_cfg.get("ollama_plain_recovery_timeout_seconds", 12),
            minimum=3.0,
            maximum=90.0,
        )
        layout.addRow(
            "Ollama recovery timeout:",
            self.ollama_plain_recovery_timeout_spin,
        )

        self.ollama_plain_recovery_nudge_timeout_spin = _make_spin(
            agent_cfg.get("ollama_plain_recovery_nudge_timeout_seconds", 8),
            minimum=2.0,
            maximum=90.0,
        )
        layout.addRow(
            "Ollama recovery nudge timeout:",
            self.ollama_plain_recovery_nudge_timeout_spin,
        )

        email_note = QtWidgets.QLabel(
            "Email polling interval controls background IMAP checks."
        )
        email_note.setWordWrap(True)
        email_note.setStyleSheet("color: #6b7280;")
        layout.addRow(email_note)

        current_poll = 300
        try:
            if self._agent_config is not None:
                current_poll = int(self._agent_config.tools.email.polling_interval)
        except Exception:
            current_poll = 300
        self.email_poll_interval_spin = QtWidgets.QSpinBox(widget)
        self.email_poll_interval_spin.setRange(10, 3600)
        self.email_poll_interval_spin.setSingleStep(10)
        self.email_poll_interval_spin.setValue(max(10, current_poll))
        self.email_poll_interval_spin.setSuffix(" s")
        layout.addRow("Email IMAP poll interval:", self.email_poll_interval_spin)

        env_note = QtWidgets.QLabel(
            "Also exported to ANNOLID_EMAIL_POLL_INTERVAL_SECONDS for runtime override."
        )
        env_note.setWordWrap(True)
        env_note.setStyleSheet("color: #6b7280;")
        layout.addRow(env_note)

        self._tabs.addTab(widget, "Agent Runtime")

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

    def _add_provider_dialog(self) -> None:
        provider_id, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add Provider",
            "Provider ID (lowercase, e.g. nvidia):",
        )
        if not ok:
            return
        provider = str(provider_id or "").strip().lower().replace(" ", "_")
        if not provider:
            return
        if provider in self._provider_specs:
            QtWidgets.QMessageBox.information(
                self, "Provider Exists", f"Provider '{provider}' already exists."
            )
            return

        label, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add Provider",
            "Display name:",
            text=provider.title(),
        )
        if not ok:
            return
        base_url, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add Provider",
            "Base URL (OpenAI-compatible endpoint):",
            text="https://integrate.api.nvidia.com/v1",
        )
        if not ok:
            return
        env_var, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add Provider",
            "API key env var name:",
            text=f"{provider.upper()}_API_KEY",
        )
        if not ok:
            return
        env_name = str(env_var or "").strip().upper() or f"{provider.upper()}_API_KEY"
        self._provider_specs[provider] = {
            "label": str(label or provider.title()).strip(),
            "kind": "openai_compat",
            "env_keys": [env_name],
            "api_key_env": [env_name],
            "base_url_default": str(base_url or "").strip(),
            "base_url_env": "",
            "model_placeholder": "Type model name and press Add",
        }
        self._provider_env_keys[provider] = [env_name]
        self._settings.setdefault(provider, {})
        self._build_provider_tab(provider)
        self._tabs.setCurrentIndex(self._tabs.count() - 1)

    def _remove_provider_tab(self, provider: str) -> None:
        if provider in {"openai", "openrouter", "gemini", "ollama"}:
            return
        index_to_remove = -1
        for idx in range(self._tabs.count()):
            if self._tabs.widget(idx) is self._provider_widgets.get(provider, {}).get(
                "tab_widget"
            ):
                index_to_remove = idx
                break
        if index_to_remove >= 0:
            page = self._tabs.widget(index_to_remove)
            self._tabs.removeTab(index_to_remove)
            if page is not None:
                page.deleteLater()
        self._provider_widgets.pop(provider, None)
        self._provider_specs.pop(provider, None)
        self._provider_env_keys.pop(provider, None)
        self._settings.pop(provider, None)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _add_api_key_controls(
        self,
        layout: QtWidgets.QFormLayout,
        *,
        provider: str,
        initial_value: str = "",
    ) -> QtWidgets.QLineEdit:
        row_widget = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        key_edit = QtWidgets.QLineEdit(str(initial_value or ""))
        key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        key_edit.setPlaceholderText("Paste API key (session-only)")
        key_edit.setClearButtonEnabled(True)
        row_layout.addWidget(key_edit, 1)

        show_button = QtWidgets.QToolButton(row_widget)
        show_button.setText("Show")
        show_button.setCheckable(True)
        show_button.toggled.connect(
            lambda checked, edit=key_edit: edit.setEchoMode(
                QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password
            )
        )
        row_layout.addWidget(show_button, 0)

        env_button = QtWidgets.QToolButton(row_widget)
        env_button.setText("Use Env")
        env_button.clicked.connect(
            lambda _checked=False, p=provider, edit=key_edit: self._fill_key_from_env(
                p, edit
            )
        )
        row_layout.addWidget(env_button, 0)

        clear_button = QtWidgets.QToolButton(row_widget)
        clear_button.setText("Clear")
        clear_button.clicked.connect(key_edit.clear)
        row_layout.addWidget(clear_button, 0)

        layout.addRow("API key:", row_widget)
        status_label = QtWidgets.QLabel("")
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #6b7280;")
        layout.addRow("", status_label)

        self._api_key_inputs[provider] = key_edit
        self._api_key_status_labels[provider] = status_label
        key_edit.textChanged.connect(
            lambda _text, p=provider: self._update_api_key_status(p)
        )
        self._update_api_key_status(provider)
        return key_edit

    def _fill_key_from_env(self, provider: str, edit: QtWidgets.QLineEdit) -> None:
        for env_name in self._provider_env_keys.get(provider, []):
            value = str(os.getenv(env_name) or "").strip()
            if value:
                edit.setText(value)
                return

    def _resolve_initial_api_key(
        self, provider: str, provider_cfg: Dict[str, Any]
    ) -> str:
        key = str(provider_cfg.get("api_key") or "").strip()
        if key:
            return key
        for env_name in self._provider_env_keys.get(provider, []):
            env_value = str(os.getenv(env_name) or "").strip()
            if env_value:
                return env_value
        return ""

    def _update_api_key_status(self, provider: str) -> None:
        edit = self._api_key_inputs.get(provider)
        label = self._api_key_status_labels.get(provider)
        if edit is None or label is None:
            return
        if edit.text().strip():
            label.setText("Session key is set. It will not be persisted to disk.")
            return
        for env_name in self._provider_env_keys.get(provider, []):
            if str(os.getenv(env_name) or "").strip():
                label.setText(f"Using environment variable: {env_name}")
                return
        label.setText("No key configured. Paste one for this session or set env var.")

    @staticmethod
    def _set_env_if_present(name: str, value: str) -> None:
        text = str(value or "").strip()
        if text:
            os.environ[name] = text

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

    def _get_models_with_selected_first(
        self, models_list: QtWidgets.QListWidget
    ) -> List[str]:
        models = self._get_models_from_list(models_list)
        if not models:
            return models
        selected = models_list.selectedItems()
        if not selected:
            current = models_list.currentItem()
            selected = [current] if current is not None else []
        if not selected:
            return models
        selected_text = str(selected[0].text() or "").strip()
        if not selected_text or selected_text not in models:
            return models
        return [selected_text] + [m for m in models if m != selected_text]

    def _provider_for_current_tab(self) -> Optional[str]:
        current_tab = self._tabs.currentWidget()
        if current_tab is None:
            return None
        if current_tab is getattr(self, "_ollama_tab_widget", None):
            return "ollama"
        for provider, payload in self._provider_widgets.items():
            if payload.get("tab_widget") is current_tab:
                return str(provider or "").strip().lower() or None
        return None

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

        ollama_default_host = str(
            (self._provider_specs.get("ollama", {}) or {}).get("host_default")
            or "http://localhost:11434"
        )
        updated["ollama"] = {
            "host": self.ollama_host_edit.text().strip() or ollama_default_host,
            "preferred_models": self._get_models_with_selected_first(
                self.ollama_models_list["list"]
            ),
        }
        persistent_env_values: Dict[str, str] = {}
        provider_defs_out: Dict[str, Dict[str, Any]] = {}
        for provider, spec in self._provider_specs.items():
            kind = str(spec.get("kind") or "openai_compat").strip().lower()
            if kind not in {"ollama", "openai_compat", "gemini"}:
                kind = "openai_compat"
            if kind == "ollama":
                provider_defs_out[provider] = {
                    "label": str(spec.get("label") or "Ollama"),
                    "kind": "ollama",
                    "env_keys": [
                        str(v).strip() for v in list(spec.get("env_keys", []) or [])
                    ],
                    "api_key_env": [
                        str(v).strip() for v in list(spec.get("api_key_env", []) or [])
                    ],
                    "base_url_default": str(spec.get("base_url_default") or "").strip(),
                    "base_url_env": str(spec.get("base_url_env") or "").strip(),
                    "host_default": str(spec.get("host_default") or "").strip(),
                    "model_placeholder": str(
                        spec.get("model_placeholder") or "Type model name and press Add"
                    ),
                }
                continue
            provider_state = self._provider_widgets.get(provider, {})
            provider_payload: Dict[str, Any] = {}
            model_editor = provider_state.get("models_list")
            if isinstance(model_editor, dict):
                model_list = model_editor.get("list")
                if isinstance(model_list, QtWidgets.QListWidget):
                    provider_payload["preferred_models"] = (
                        self._get_models_with_selected_first(model_list)
                    )

            base_url_edit = provider_state.get("base_url_edit")
            if kind == "openai_compat" and isinstance(
                base_url_edit, QtWidgets.QLineEdit
            ):
                default_base_url = str(spec.get("base_url_default") or "").strip()
                provider_payload["base_url"] = (
                    base_url_edit.text().strip() or default_base_url
                )

            updated[provider] = provider_payload

            key_edit = provider_state.get("key_edit")
            key_value = (
                key_edit.text().strip()
                if isinstance(key_edit, QtWidgets.QLineEdit)
                else ""
            )
            for env_name in spec.get("api_key_env", []) or []:
                env_key = str(env_name).strip()
                self._set_env_if_present(env_key, key_value)
                if env_key and key_value and self.persist_env_checkbox.isChecked():
                    persistent_env_values[env_key] = key_value
            base_url_env = str(spec.get("base_url_env") or "").strip()
            if (
                kind == "openai_compat"
                and isinstance(base_url_edit, QtWidgets.QLineEdit)
                and base_url_env
            ):
                base_url_value = base_url_edit.text().strip() or str(
                    spec.get("base_url_default") or ""
                )
                self._set_env_if_present(base_url_env, base_url_value)
                if base_url_value and self.persist_env_checkbox.isChecked():
                    persistent_env_values[base_url_env] = base_url_value
            provider_defs_out[provider] = {
                "label": str(spec.get("label") or provider.title()),
                "kind": kind,
                "env_keys": [
                    str(v).strip() for v in list(spec.get("env_keys", []) or [])
                ],
                "api_key_env": [
                    str(v).strip() for v in list(spec.get("api_key_env", []) or [])
                ],
                "base_url_default": str(spec.get("base_url_default") or "").strip(),
                "base_url_env": base_url_env,
                "host_default": str(spec.get("host_default") or "").strip(),
                "model_placeholder": str(
                    spec.get("model_placeholder") or "Type model name and press Add"
                ),
            }
        updated["provider_definitions"] = provider_defs_out
        active_provider = self._provider_for_current_tab()
        if active_provider and active_provider in provider_defs_out:
            updated["provider"] = active_provider
            models_for_provider: List[str] = []
            if active_provider == "ollama":
                models_for_provider = list(
                    updated.get("ollama", {}).get("preferred_models", []) or []
                )
            else:
                models_for_provider = list(
                    updated.get(active_provider, {}).get("preferred_models", []) or []
                )
            if models_for_provider:
                updated.setdefault("last_models", {})
                updated["last_models"][active_provider] = str(
                    models_for_provider[0]
                ).strip()

        agent_block = dict(updated.get("agent") or {})
        agent_block["fast_mode_timeout_seconds"] = float(
            self.fast_mode_timeout_spin.value()
        )
        agent_block["fallback_retry_timeout_seconds"] = float(
            self.fallback_retry_timeout_spin.value()
        )
        agent_block["loop_llm_timeout_seconds"] = float(
            self.loop_llm_timeout_spin.value()
        )
        agent_block["loop_llm_timeout_seconds_no_tools"] = float(
            self.loop_llm_timeout_no_tools_spin.value()
        )
        agent_block["loop_tool_timeout_seconds"] = float(
            self.loop_tool_timeout_spin.value()
        )
        agent_block["ollama_tool_timeout_seconds"] = float(
            self.ollama_tool_timeout_spin.value()
        )
        agent_block["ollama_plain_timeout_seconds"] = float(
            self.ollama_plain_timeout_spin.value()
        )
        agent_block["ollama_plain_recovery_timeout_seconds"] = float(
            self.ollama_plain_recovery_timeout_spin.value()
        )
        agent_block["ollama_plain_recovery_nudge_timeout_seconds"] = float(
            self.ollama_plain_recovery_nudge_timeout_spin.value()
        )
        agent_block["enable_progress_stream"] = bool(
            self.enable_progress_stream_checkbox.isChecked()
        )
        agent_block["browser_first_for_web"] = bool(
            self.browser_first_for_web_checkbox.isChecked()
        )
        updated["agent"] = agent_block
        if self._agent_config is not None:
            try:
                poll_seconds = int(self.email_poll_interval_spin.value())
                self._agent_config.tools.email.polling_interval = max(10, poll_seconds)
                save_config(self._agent_config)
                os.environ["ANNOLID_EMAIL_POLL_INTERVAL_SECONDS"] = str(
                    self._agent_config.tools.email.polling_interval
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not persist email poll interval",
                    f"Failed to update agent config with email polling interval.\n\n{exc}",
                )

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
        if self.persist_env_checkbox.isChecked():
            try:
                persist_global_env_vars(persistent_env_values)
            except OSError as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not persist credentials",
                    f"Failed to update {global_env_path()}.\n\n{exc}",
                )
        self._settings = updated
        super().accept()
