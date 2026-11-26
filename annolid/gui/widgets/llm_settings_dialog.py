from __future__ import annotations

import os
import shutil
import threading
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.llm_provider_dialogs import (
    AddProviderDialog,
    ProviderProbeDispatcher,
)
from annolid.infrastructure.agent_config import (
    load_agent_config as load_config,
    save_agent_config as save_config,
)
from annolid.services.agent_update import (
    check_gui_agent_update,
    execute_gui_agent_rollback,
    run_agent_update,
)
from annolid.services.chat_runtime import get_chat_default_allowed_read_roots
from annolid.utils.llm_provider_catalog import (
    extract_ollama_model_names,
    inspect_provider_credential_status,
    validate_http_endpoint_url,
)
from annolid.utils.llm_settings import (
    clear_session_env_vars,
    detect_openai_codex_auth_state,
    default_settings,
    global_env_path,
    load_global_env_vars,
    mark_session_env_var_active,
    persist_global_env_vars,
    remove_global_env_vars,
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

# Backward-compatible module symbol imported by existing tests.
_extract_ollama_model_names = extract_ollama_model_names

_BUILTIN_PROVIDERS = frozenset(
    {
        "ollama",
        "openai",
        "anthropic",
        "openrouter",
        "gemini",
        "openai_codex",
        "codex_cli",
    }
)
_VALID_PROVIDER_KINDS = frozenset(
    {
        "ollama",
        "openai_compat",
        "anthropic",
        "gemini",
        "openai_codex",
        "codex_cli",
    }
)


class LLMSettingsDialog(QtWidgets.QDialog):
    boxAuthCompleted = QtCore.Signal(object)

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
        self.resize(620, 500)

        defaults = default_settings()
        self._default_agent_cfg: Dict[str, Any] = dict(defaults.get("agent", {}) or {})
        self._settings = defaults if settings is None else {**defaults, **settings}
        try:
            self._agent_config = load_config()
        except Exception:
            self._agent_config = None
        self._tts_settings = load_tts_settings()
        self._tts_defaults = default_tts_settings()
        self._dotenv_values: Dict[str, str] = load_global_env_vars()
        default_defs = dict(defaults.get("provider_definitions", {}) or {})
        passed_defs = dict(self._settings.get("provider_definitions", {}) or {})
        self._provider_specs: Dict[str, Dict[str, Any]] = {
            **default_defs,
            **passed_defs,
        }
        self._provider_env_keys: Dict[str, List[str]] = {
            provider: self._spec_api_key_envs(spec)
            for provider, spec in self._provider_specs.items()
        }
        self._provider_widgets: Dict[str, Dict[str, Any]] = {}
        self._api_key_inputs: Dict[str, QtWidgets.QLineEdit] = {}
        self._api_key_status_labels: Dict[str, QtWidgets.QLabel] = {}
        self._forget_key_buttons: Dict[str, QtWidgets.QToolButton] = {}
        self._provider_status_labels: Dict[str, QtWidgets.QLabel] = {}
        self._provider_probe_buttons: Dict[str, List[QtWidgets.QPushButton]] = {}
        self._initial_api_keys: Dict[str, str] = {}
        self._cleared_api_key_providers: set[str] = set()
        self._agent_runtime_tab_widget: Optional[QtWidgets.QWidget] = None
        self._tts_tab_widget: Optional[QtWidgets.QWidget] = None
        self._closing = False
        self._box_redirect_default = "http://localhost:8765/oauth/callback"
        self._box_auth_thread: Optional[threading.Thread] = None
        self.boxAuthCompleted.connect(self._on_box_auth_completed)
        self._probe_dispatcher = ProviderProbeDispatcher(self)
        self._probe_dispatcher.probeFinished.connect(self._on_provider_probe_finished)

        main_layout = QtWidgets.QVBoxLayout(self)
        info_label = QtWidgets.QLabel(
            "API keys entered here stay in memory for this session unless you opt in "
            "to saving them in ~/.annolid/.env (mode 0600). "
            "OpenAI Codex uses local OAuth and Codex CLI uses your local executable."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(8)
        top_bar.addWidget(QtWidgets.QLabel("Active provider:", self))
        self.active_provider_combo = QtWidgets.QComboBox(self)
        self.active_provider_combo.setMinimumWidth(180)
        self.active_provider_combo.currentIndexChanged.connect(
            self._on_active_provider_combo_changed
        )
        top_bar.addWidget(self.active_provider_combo, 1)
        add_provider_button = QtWidgets.QPushButton("Add Provider…", self)
        add_provider_button.clicked.connect(self._add_provider_dialog)
        top_bar.addWidget(add_provider_button, 0)
        main_layout.addLayout(top_bar)

        self.persist_env_checkbox = QtWidgets.QCheckBox(
            f"Persist entered credentials to {global_env_path()} (optional)"
        )
        self.persist_env_checkbox.setChecked(False)
        main_layout.addWidget(self.persist_env_checkbox, 0, QtCore.Qt.AlignLeft)

        self._tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self._tabs, 1)

        self._build_ollama_tab()
        for provider, spec in self._provider_specs.items():
            if provider == "ollama":
                continue
            self._build_provider_tab(provider)
        self._build_agent_runtime_tab()
        self._build_tts_tab()

        initial_provider = (
            str(self._settings.get("provider") or "ollama").strip().lower()
        )
        self._rebuild_active_provider_combo(selected_provider=initial_provider)
        self._focus_provider_tab(
            str(self.active_provider_combo.currentData() or initial_provider)
        )

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    @staticmethod
    def _spec_api_key_envs(spec: Dict[str, Any]) -> List[str]:
        raw = spec.get("api_key_env")
        if isinstance(raw, str):
            raw = [raw]
        names = [str(v).strip() for v in (raw or []) if str(v).strip()]
        if names:
            return names
        fallback = spec.get("env_keys")
        if isinstance(fallback, str):
            fallback = [fallback]
        return [str(v).strip() for v in (fallback or []) if str(v).strip()]

    def _rebuild_active_provider_combo(
        self, *, selected_provider: Optional[str] = None
    ) -> None:
        target = (
            str(
                selected_provider
                if selected_provider is not None
                else self.active_provider_combo.currentData()
                or self._settings.get("provider")
                or "ollama"
            )
            .strip()
            .lower()
        )
        self.active_provider_combo.blockSignals(True)
        try:
            self.active_provider_combo.clear()
            for provider, spec in self._provider_specs.items():
                label = str(spec.get("label") or provider.title())
                self.active_provider_combo.addItem(label, userData=provider)
            idx = self.active_provider_combo.findData(target)
            if idx < 0 and self.active_provider_combo.count() > 0:
                idx = 0
            if idx >= 0:
                self.active_provider_combo.setCurrentIndex(idx)
        finally:
            self.active_provider_combo.blockSignals(False)

    def _on_active_provider_combo_changed(self, _index: int) -> None:
        provider = str(self.active_provider_combo.currentData() or "").strip().lower()
        if provider:
            self._focus_provider_tab(provider)

    def _focus_provider_tab(self, provider: str) -> None:
        target_key = str(provider or "").strip().lower()
        target_widget: Optional[QtWidgets.QWidget] = None
        if target_key == "ollama":
            target_widget = getattr(self, "_ollama_tab_widget", None)
        else:
            target_widget = self._provider_widgets.get(target_key, {}).get("tab_widget")
        if target_widget is None:
            return
        idx = self._tabs.indexOf(target_widget)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._closing = True
        super().closeEvent(event)

    def reject(self) -> None:  # type: ignore[override]
        self._closing = True
        super().reject()

    # ------------------------------------------------------------------ #
    # Tab builders
    # ------------------------------------------------------------------ #
    def _create_scrollable_form_tab(
        self,
    ) -> tuple[QtWidgets.QScrollArea, QtWidgets.QWidget, QtWidgets.QFormLayout]:
        content_widget = QtWidgets.QWidget()
        content_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum
        )
        layout = QtWidgets.QFormLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidget(content_widget)
        return scroll, content_widget, layout

    def _build_ollama_tab(self) -> None:
        tab_widget, _content_widget, layout = self._create_scrollable_form_tab()
        self._ollama_tab_widget = tab_widget
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

        actions_row = QtWidgets.QWidget(self)
        actions_layout = QtWidgets.QHBoxLayout(actions_row)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        test_button = QtWidgets.QPushButton("Test Connection", actions_row)
        test_button.clicked.connect(
            lambda _checked=False: self._start_provider_probe("ollama", action="test")
        )
        refresh_button = QtWidgets.QPushButton("Refresh from Ollama", actions_row)
        refresh_button.clicked.connect(self._refresh_ollama_models)
        actions_layout.addWidget(test_button)
        actions_layout.addWidget(refresh_button)
        actions_layout.addStretch(1)
        layout.addRow("Connection:", actions_row)

        probe_status = QtWidgets.QLabel("", self)
        probe_status.setWordWrap(True)
        probe_status.setStyleSheet("color: #6b7280;")
        layout.addRow("", probe_status)
        self._provider_status_labels["ollama"] = probe_status
        self._provider_probe_buttons["ollama"] = [test_button, refresh_button]

        self._tabs.addTab(tab_widget, str(spec.get("label") or "Ollama"))

    def _build_openai_tab(self) -> None:
        self._build_provider_tab("openai")

    def _build_openrouter_tab(self) -> None:
        self._build_provider_tab("openrouter")

    def _build_gemini_tab(self) -> None:
        self._build_provider_tab("gemini")

    def _build_provider_tab(
        self, provider: str, *, insert_before_utility: bool = False
    ) -> None:
        spec = self._provider_specs.get(provider)
        if not spec:
            return
        tab_widget, _content_widget, layout = self._create_scrollable_form_tab()
        provider_cfg = dict(self._settings.get(provider, {}) or {})
        kind = str(spec.get("kind") or "openai_compat").strip().lower()
        if kind not in _VALID_PROVIDER_KINDS:
            kind = "openai_compat"
        self._provider_widgets.setdefault(provider, {})["tab_widget"] = tab_widget
        key_edit: Optional[QtWidgets.QLineEdit] = None
        host_edit: Optional[QtWidgets.QLineEdit] = None
        cli_path_edit: Optional[QtWidgets.QLineEdit] = None
        transport_combo: Optional[QtWidgets.QComboBox] = None

        if kind == "openai_codex":
            self._add_openai_codex_auth_controls(
                layout,
                provider=provider,
                provider_cfg=provider_cfg,
            )
        elif kind == "codex_cli":
            cli_path_edit = self._add_codex_cli_controls(
                layout,
                provider=provider,
                provider_cfg=provider_cfg,
            )
        elif kind == "ollama":
            default_host = str(spec.get("host_default") or "http://localhost:11434")
            host_edit = QtWidgets.QLineEdit(
                str(provider_cfg.get("host") or default_host)
            )
            layout.addRow("Server URL:", host_edit)
        else:
            key_edit = self._add_api_key_controls(
                layout,
                provider=provider,
                initial_value=self._resolve_initial_api_key(provider, provider_cfg),
            )
        self._provider_widgets.setdefault(provider, {})["key_edit"] = key_edit
        self._provider_widgets.setdefault(provider, {})["host_edit"] = host_edit
        self._provider_widgets.setdefault(provider, {})["cli_path_edit"] = cli_path_edit

        base_url_edit: Optional[QtWidgets.QLineEdit] = None
        if kind in {"openai_compat", "openai_codex", "anthropic"}:
            default_base_url = str(spec.get("base_url_default") or "").strip()
            base_url_edit = QtWidgets.QLineEdit(
                str(provider_cfg.get("base_url") or default_base_url)
            )
            layout.addRow("Base URL:", base_url_edit)
            if key_edit is not None:
                base_url_edit.textChanged.connect(
                    lambda _text, p=provider: self._update_api_key_status(p)
                )
        self._provider_widgets.setdefault(provider, {})["base_url_edit"] = base_url_edit
        if key_edit is not None:
            self._update_api_key_status(provider)

        if kind == "openai_codex":
            transport_combo = QtWidgets.QComboBox(self)
            transport_combo.addItem("Auto (default)", "auto")
            transport_combo.addItem("SSE (HTTP streaming)", "sse")
            transport_combo.addItem("WebSocket", "websocket")
            current_transport = (
                str(
                    provider_cfg.get("transport")
                    or spec.get("transport_default")
                    or "auto"
                )
                .strip()
                .lower()
            )
            t_idx = transport_combo.findData(current_transport)
            if t_idx < 0:
                t_idx = 0
            transport_combo.setCurrentIndex(t_idx)
            layout.addRow("Transport:", transport_combo)
        self._provider_widgets.setdefault(provider, {})["transport_combo"] = (
            transport_combo
        )

        models_list = self._create_model_list_editor(
            provider_cfg.get("preferred_models", []),
            placeholder=str(
                spec.get("model_placeholder", "Type model name and press Add")
            ),
        )
        layout.addRow("Preferred models:", models_list["container"])
        self._provider_widgets.setdefault(provider, {})["models_list"] = models_list

        # Probe / live model discovery controls (non-blocking background thread).
        probe_row = QtWidgets.QWidget(self)
        probe_layout = QtWidgets.QHBoxLayout(probe_row)
        probe_layout.setContentsMargins(0, 0, 0, 0)
        probe_layout.setSpacing(6)
        probe_buttons: List[QtWidgets.QPushButton] = []

        test_button = QtWidgets.QPushButton("Test Connection", probe_row)
        test_button.clicked.connect(
            lambda _checked=False, p=provider: self._start_provider_probe(
                p, action="test"
            )
        )
        probe_layout.addWidget(test_button)
        probe_buttons.append(test_button)

        if kind in {"openai_compat", "anthropic", "gemini", "ollama"}:
            fetch_button = QtWidgets.QPushButton("Fetch Models", probe_row)
            fetch_button.clicked.connect(
                lambda _checked=False, p=provider: self._start_provider_probe(
                    p, action="models"
                )
            )
            probe_layout.addWidget(fetch_button)
            probe_buttons.append(fetch_button)

        probe_layout.addStretch(1)
        layout.addRow("Connection:", probe_row)

        probe_status = QtWidgets.QLabel("", self)
        probe_status.setWordWrap(True)
        probe_status.setStyleSheet("color: #6b7280;")
        layout.addRow("", probe_status)
        self._provider_status_labels[provider] = probe_status
        self._provider_probe_buttons[provider] = probe_buttons

        # Backward-compatible widget aliases used in tests/callers.
        setattr(self, f"{provider}_key_edit", key_edit)
        setattr(self, f"{provider}_models_list", models_list)
        if base_url_edit is not None:
            setattr(self, f"{provider}_base_url_edit", base_url_edit)
        if transport_combo is not None:
            setattr(self, f"{provider}_transport_combo", transport_combo)

        if provider not in _BUILTIN_PROVIDERS:
            remove_button = QtWidgets.QPushButton("Remove Provider")
            remove_button.clicked.connect(
                lambda _checked=False, p=provider: self._remove_provider_tab(p)
            )
            layout.addRow(remove_button)

        label = str(spec.get("label") or provider.title())
        insert_idx = -1
        if insert_before_utility and self._agent_runtime_tab_widget is not None:
            insert_idx = self._tabs.indexOf(self._agent_runtime_tab_widget)
        if insert_idx >= 0:
            self._tabs.insertTab(insert_idx, tab_widget, label)
        else:
            self._tabs.addTab(tab_widget, label)

    def _build_tts_tab(self) -> None:
        tab_widget, _content_widget, layout = self._create_scrollable_form_tab()

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

        self._tts_tab_widget = tab_widget
        self._tabs.addTab(tab_widget, "Text-to-Speech")

    def _build_agent_runtime_tab(self) -> None:
        tab_widget, content_widget, layout = self._create_scrollable_form_tab()
        self._agent_runtime_tab_widget = tab_widget

        agent_cfg = dict(self._settings.get("agent") or {})
        default_agent = dict(self._default_agent_cfg or {})

        note = QtWidgets.QLabel(
            "Agent runtime timeout controls (seconds). Lower values fail faster."
        )
        note.setWordWrap(True)
        layout.addRow(note)

        self.enable_progress_stream_checkbox = QtWidgets.QCheckBox(
            "Enable intermediate progress stream", content_widget
        )
        self.enable_progress_stream_checkbox.setChecked(
            bool(agent_cfg.get("enable_progress_stream", True))
        )
        layout.addRow(self.enable_progress_stream_checkbox)

        self.browser_first_for_web_checkbox = QtWidgets.QCheckBox(
            "Prefer MCP browser for web tasks", content_widget
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
            spin = QtWidgets.QDoubleSpinBox(content_widget)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(float(value))
            return spin

        self.fast_mode_timeout_spin = _make_spin(
            agent_cfg.get(
                "fast_mode_timeout_seconds",
                default_agent.get("fast_mode_timeout_seconds", 60),
            ),
            minimum=10.0,
            maximum=600.0,
        )
        layout.addRow("Fast mode timeout:", self.fast_mode_timeout_spin)

        self.fallback_retry_timeout_spin = _make_spin(
            agent_cfg.get(
                "fallback_retry_timeout_seconds",
                default_agent.get("fallback_retry_timeout_seconds", 20),
            ),
            minimum=5.0,
            maximum=180.0,
        )
        layout.addRow("Fallback retry timeout:", self.fallback_retry_timeout_spin)

        self.loop_llm_timeout_spin = _make_spin(
            agent_cfg.get(
                "loop_llm_timeout_seconds",
                default_agent.get("loop_llm_timeout_seconds", 600),
            ),
            minimum=5.0,
            maximum=1800.0,
        )
        layout.addRow("Agent loop LLM timeout:", self.loop_llm_timeout_spin)

        self.loop_llm_timeout_no_tools_spin = _make_spin(
            agent_cfg.get(
                "loop_llm_timeout_seconds_no_tools",
                default_agent.get("loop_llm_timeout_seconds_no_tools", 180),
            ),
            minimum=5.0,
            maximum=900.0,
        )
        layout.addRow(
            "Agent loop timeout (no tools):",
            self.loop_llm_timeout_no_tools_spin,
        )

        self.loop_tool_timeout_spin = _make_spin(
            agent_cfg.get(
                "loop_tool_timeout_seconds",
                default_agent.get("loop_tool_timeout_seconds", 900),
            ),
            minimum=3.0,
            maximum=3600.0,
        )
        layout.addRow("Agent tool timeout:", self.loop_tool_timeout_spin)

        self.ollama_tool_timeout_spin = _make_spin(
            agent_cfg.get(
                "ollama_tool_timeout_seconds",
                default_agent.get("ollama_tool_timeout_seconds", 360),
            ),
            minimum=5.0,
            maximum=1800.0,
        )
        layout.addRow("Ollama tool request timeout:", self.ollama_tool_timeout_spin)

        self.ollama_plain_timeout_spin = _make_spin(
            agent_cfg.get(
                "ollama_plain_timeout_seconds",
                default_agent.get("ollama_plain_timeout_seconds", 90),
            ),
            minimum=5.0,
            maximum=900.0,
        )
        layout.addRow("Ollama plain request timeout:", self.ollama_plain_timeout_spin)

        self.ollama_plain_recovery_timeout_spin = _make_spin(
            agent_cfg.get(
                "ollama_plain_recovery_timeout_seconds",
                default_agent.get("ollama_plain_recovery_timeout_seconds", 45),
            ),
            minimum=3.0,
            maximum=300.0,
        )
        layout.addRow(
            "Ollama recovery timeout:",
            self.ollama_plain_recovery_timeout_spin,
        )

        self.ollama_plain_recovery_nudge_timeout_spin = _make_spin(
            agent_cfg.get(
                "ollama_plain_recovery_nudge_timeout_seconds",
                default_agent.get("ollama_plain_recovery_nudge_timeout_seconds", 20),
            ),
            minimum=2.0,
            maximum=300.0,
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
        self.email_poll_interval_spin = QtWidgets.QSpinBox(content_widget)
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

        bot_note = QtWidgets.QLabel(
            "Bot settings control skill loading and workspace memory retrieval behavior."
        )
        bot_note.setWordWrap(True)
        bot_note.setStyleSheet("color: #6b7280;")
        layout.addRow(bot_note)

        box_host = "https://account.box.com"
        box_redirect_uri = self._box_redirect_default
        box_client_id = ""
        box_client_secret = ""
        if self._agent_config is not None:
            box_cfg = getattr(getattr(self._agent_config, "tools", None), "box", None)
            if box_cfg is not None:
                box_host = str(
                    getattr(box_cfg, "authorize_base_url", box_host) or box_host
                )
                box_redirect_uri = str(
                    getattr(box_cfg, "redirect_uri", "") or self._box_redirect_default
                )
                box_client_id = str(getattr(box_cfg, "client_id", "") or "").strip()
                box_client_secret = str(
                    getattr(box_cfg, "client_secret", "") or ""
                ).strip()

        box_note = QtWidgets.QLabel(
            "Box OAuth can use your tenant host when Box routes users through org "
            "login or 2FA pages."
        )
        box_note.setWordWrap(True)
        box_note.setStyleSheet("color: #6b7280;")
        layout.addRow(box_note)

        self.box_authorize_base_url_edit = QtWidgets.QLineEdit(content_widget)
        self.box_authorize_base_url_edit.setPlaceholderText(
            "https://my_org_xxx.account.box.com"
        )
        self.box_authorize_base_url_edit.setText(box_host)
        layout.addRow("Box auth host:", self.box_authorize_base_url_edit)

        self.box_client_id_edit = QtWidgets.QLineEdit(content_widget)
        self.box_client_id_edit.setPlaceholderText("Box OAuth client id")
        self.box_client_id_edit.setText(box_client_id)
        layout.addRow("Box client ID:", self.box_client_id_edit)

        self.box_client_secret_edit = QtWidgets.QLineEdit(content_widget)
        self.box_client_secret_edit.setPlaceholderText("Box OAuth client secret")
        self.box_client_secret_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.box_client_secret_edit.setText(box_client_secret)
        layout.addRow("Box client secret:", self.box_client_secret_edit)

        self.box_redirect_uri_edit = QtWidgets.QLineEdit(content_widget)
        self.box_redirect_uri_edit.setPlaceholderText(self._box_redirect_default)
        self.box_redirect_uri_edit.setText(box_redirect_uri)
        box_redirect_row = QtWidgets.QWidget(content_widget)
        box_redirect_row_layout = QtWidgets.QHBoxLayout(box_redirect_row)
        box_redirect_row_layout.setContentsMargins(0, 0, 0, 0)
        box_redirect_row_layout.setSpacing(6)
        box_redirect_row_layout.addWidget(self.box_redirect_uri_edit, 1)
        self.box_copy_redirect_button = QtWidgets.QPushButton("Copy", box_redirect_row)
        self.box_copy_redirect_button.clicked.connect(self._copy_box_redirect_uri)
        box_redirect_row_layout.addWidget(self.box_copy_redirect_button)
        layout.addRow("Box redirect URI:", box_redirect_row)

        box_actions = QtWidgets.QWidget(content_widget)
        box_actions_layout = QtWidgets.QHBoxLayout(box_actions)
        box_actions_layout.setContentsMargins(0, 0, 0, 0)
        box_actions_layout.setSpacing(6)
        self.box_auth_button = QtWidgets.QPushButton("Grant Box Access", box_actions)
        self.box_auth_button.clicked.connect(self._open_box_auth_url)
        box_actions_layout.addWidget(self.box_auth_button)
        box_actions_layout.addStretch(1)
        layout.addRow("Box auth action:", box_actions)

        skills_watch_default = False
        memory_mode_default = "semantic_keyword"
        skills_extra_dirs_default: list[str] = []
        allowed_read_roots_default: list[str] = []
        if self._agent_config is not None:
            skills_cfg = getattr(self._agent_config, "skills", None)
            skills_load_cfg = getattr(skills_cfg, "load", None)
            skills_watch_default = bool(getattr(skills_load_cfg, "watch", False))
            memory_mode_default = str(
                getattr(
                    getattr(self._agent_config, "memory", None),
                    "mode",
                    "semantic_keyword",
                )
                or "semantic_keyword"
            ).strip()
            skills_extra_dirs_default = list(
                getattr(skills_load_cfg, "extra_dirs", []) or []
            )
            allowed_read_roots_default = list(
                getattr(
                    getattr(self._agent_config, "tools", None),
                    "allowed_read_roots",
                    [],
                )
                or []
            )

        self.skills_hot_reload_checkbox = QtWidgets.QCheckBox(
            "Enable skill hot reload (skills.load.watch)", content_widget
        )
        self.skills_hot_reload_checkbox.setChecked(bool(skills_watch_default))
        layout.addRow(self.skills_hot_reload_checkbox)

        self.memory_mode_combo = QtWidgets.QComboBox(content_widget)
        self.memory_mode_combo.addItem(
            "Semantic + keyword fallback", "semantic_keyword"
        )
        self.memory_mode_combo.addItem("Keyword only (lexical)", "lexical")
        mode_index = self.memory_mode_combo.findData(memory_mode_default)
        if mode_index < 0:
            mode_index = 0
        self.memory_mode_combo.setCurrentIndex(mode_index)
        layout.addRow("Memory mode:", self.memory_mode_combo)

        self.skill_source_locations_edit = QtWidgets.QLineEdit(content_widget)
        self.skill_source_locations_edit.setPlaceholderText(
            "Extra skill dirs (colon-separated)"
        )
        self.skill_source_locations_edit.setText(
            os.pathsep.join(skills_extra_dirs_default)
        )
        layout.addRow("Skill source locations:", self.skill_source_locations_edit)

        self._default_allowed_bot_roots = list(get_chat_default_allowed_read_roots())
        defaults_text = (
            "\n".join(self._default_allowed_bot_roots)
            if self._default_allowed_bot_roots
            else "(none detected)"
        )
        defaults_label = QtWidgets.QLabel(
            "Default allowed bot folder(s):\n" + defaults_text
        )
        defaults_label.setWordWrap(True)
        defaults_label.setStyleSheet("color: #6b7280;")
        layout.addRow(defaults_label)

        allowed_widget = QtWidgets.QWidget(content_widget)
        allowed_layout = QtWidgets.QVBoxLayout(allowed_widget)
        allowed_layout.setContentsMargins(0, 0, 0, 0)
        allowed_layout.setSpacing(6)

        self.allowed_bot_folders_list = QtWidgets.QListWidget(allowed_widget)
        self.allowed_bot_folders_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        default_roots_set = {str(p) for p in self._default_allowed_bot_roots}
        seen_custom: set[str] = set()
        for raw in allowed_read_roots_default:
            text = str(raw or "").strip()
            if not text:
                continue
            normalized = str(QtCore.QDir(text).absolutePath())
            if normalized in default_roots_set or normalized in seen_custom:
                continue
            seen_custom.add(normalized)
            self.allowed_bot_folders_list.addItem(normalized)
        allowed_layout.addWidget(self.allowed_bot_folders_list)

        allowed_actions = QtWidgets.QHBoxLayout()
        self.allowed_bot_add_button = QtWidgets.QPushButton(
            "Add Folder", allowed_widget
        )
        self.allowed_bot_add_button.clicked.connect(self._add_allowed_bot_folder)
        self.allowed_bot_remove_button = QtWidgets.QPushButton(
            "Remove Selected", allowed_widget
        )
        self.allowed_bot_remove_button.clicked.connect(
            self._remove_selected_allowed_bot_folders
        )
        allowed_actions.addWidget(self.allowed_bot_add_button)
        allowed_actions.addWidget(self.allowed_bot_remove_button)
        allowed_actions.addStretch(1)
        allowed_layout.addLayout(allowed_actions)
        layout.addRow("Extra allowed bot folders:", allowed_widget)

        update_note = QtWidgets.QLabel(
            "Update settings control automatic checks and release channel policy."
        )
        update_note.setWordWrap(True)
        update_note.setStyleSheet("color: #6b7280;")
        layout.addRow(update_note)

        update_auto_default = False
        update_channel_default = "stable"
        update_interval_default = 24 * 3600
        update_jitter_default = 15 * 60
        update_timeout_default = 4.0
        update_require_sig_default = False
        if self._agent_config is not None:
            auto_cfg = getattr(
                getattr(self._agent_config, "update", None), "auto", None
            )
            if auto_cfg is not None:
                update_auto_default = bool(getattr(auto_cfg, "enabled", False))
                update_channel_default = str(
                    getattr(auto_cfg, "channel", "stable") or "stable"
                ).strip()
                update_interval_default = int(
                    getattr(auto_cfg, "interval_seconds", 24 * 3600) or 24 * 3600
                )
                update_jitter_default = int(
                    getattr(auto_cfg, "jitter_seconds", 15 * 60) or 15 * 60
                )
                update_timeout_default = float(
                    getattr(auto_cfg, "timeout_s", 4.0) or 4.0
                )
                update_require_sig_default = bool(
                    getattr(auto_cfg, "require_signature", False)
                )

        self.auto_update_enabled_checkbox = QtWidgets.QCheckBox(
            "Enable auto-update", content_widget
        )
        self.auto_update_enabled_checkbox.setChecked(bool(update_auto_default))
        layout.addRow(self.auto_update_enabled_checkbox)

        self.auto_update_channel_combo = QtWidgets.QComboBox(content_widget)
        self.auto_update_channel_combo.addItem("Stable", "stable")
        self.auto_update_channel_combo.addItem("Beta", "beta")
        self.auto_update_channel_combo.addItem("Dev", "dev")
        channel_index = self.auto_update_channel_combo.findData(update_channel_default)
        if channel_index < 0:
            channel_index = 0
        self.auto_update_channel_combo.setCurrentIndex(channel_index)
        layout.addRow("Update channel:", self.auto_update_channel_combo)

        self.auto_update_interval_spin = QtWidgets.QSpinBox(content_widget)
        self.auto_update_interval_spin.setRange(300, 7 * 24 * 3600)
        self.auto_update_interval_spin.setSingleStep(300)
        self.auto_update_interval_spin.setValue(max(300, int(update_interval_default)))
        self.auto_update_interval_spin.setSuffix(" s")
        layout.addRow("Auto-update interval:", self.auto_update_interval_spin)

        self.auto_update_jitter_spin = QtWidgets.QSpinBox(content_widget)
        self.auto_update_jitter_spin.setRange(0, 3600)
        self.auto_update_jitter_spin.setSingleStep(60)
        self.auto_update_jitter_spin.setValue(max(0, int(update_jitter_default)))
        self.auto_update_jitter_spin.setSuffix(" s")
        layout.addRow("Auto-update jitter:", self.auto_update_jitter_spin)

        self.auto_update_timeout_spin = QtWidgets.QDoubleSpinBox(content_widget)
        self.auto_update_timeout_spin.setRange(1.0, 120.0)
        self.auto_update_timeout_spin.setSingleStep(0.5)
        self.auto_update_timeout_spin.setDecimals(1)
        self.auto_update_timeout_spin.setValue(max(1.0, float(update_timeout_default)))
        self.auto_update_timeout_spin.setSuffix(" s")
        layout.addRow("Update timeout:", self.auto_update_timeout_spin)

        self.auto_update_require_sig_checkbox = QtWidgets.QCheckBox(
            "Require signed manifest", content_widget
        )
        self.auto_update_require_sig_checkbox.setChecked(
            bool(update_require_sig_default)
        )
        layout.addRow(self.auto_update_require_sig_checkbox)

        update_actions = QtWidgets.QWidget(content_widget)
        update_actions_layout = QtWidgets.QHBoxLayout(update_actions)
        update_actions_layout.setContentsMargins(0, 0, 0, 0)
        update_actions_layout.setSpacing(6)
        self.update_check_now_button = QtWidgets.QPushButton(
            "Check now", update_actions
        )
        self.update_check_now_button.clicked.connect(self._check_now_update)
        self.update_run_now_button = QtWidgets.QPushButton("Update now", update_actions)
        self.update_run_now_button.clicked.connect(self._run_update_now)
        self.update_rollback_button = QtWidgets.QPushButton("Rollback", update_actions)
        self.update_rollback_button.clicked.connect(self._rollback_update)
        update_actions_layout.addWidget(self.update_check_now_button)
        update_actions_layout.addWidget(self.update_run_now_button)
        update_actions_layout.addWidget(self.update_rollback_button)
        update_actions_layout.addStretch(1)
        layout.addRow("Update actions:", update_actions)

        self._tabs.addTab(tab_widget, "Agent Runtime")

    def _check_now_update(self) -> None:
        channel = str(self.auto_update_channel_combo.currentData() or "stable")
        timeout_s = float(self.auto_update_timeout_spin.value())
        require_signature = bool(self.auto_update_require_sig_checkbox.isChecked())
        try:
            payload = check_gui_agent_update(
                project="annolid",
                channel=channel,
                timeout_s=timeout_s,
                require_signature=require_signature,
            )
            details = [
                f"Current version: {payload.get('current_version')}",
                f"Target version: {payload.get('target_version')}",
                f"Channel: {payload.get('channel')}",
                f"Update available: {payload.get('update_available')}",
                f"Verification: {payload.get('verification_reason')}",
            ]
            QtWidgets.QMessageBox.information(
                self,
                "Update Check",
                "\n".join(details),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Update Check Failed",
                f"Could not check updates.\n\n{exc}",
            )

    def _rollback_update(self) -> None:
        previous_version, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rollback",
            "Previous known-good version:",
        )
        if not ok:
            return
        version = str(previous_version or "").strip()
        if not version:
            QtWidgets.QMessageBox.information(
                self,
                "Rollback",
                "Rollback cancelled: previous version was empty.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Rollback",
            (
                "Run rollback now?\n\n"
                f"Target previous version: {version}\n"
                "This may run package manager commands."
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        try:
            payload = execute_gui_agent_rollback(
                project="annolid",
                previous_version=version,
            )
            if bool(payload.get("ok", False)):
                QtWidgets.QMessageBox.information(
                    self,
                    "Rollback Complete",
                    "Rollback completed successfully.",
                )
                return
            QtWidgets.QMessageBox.warning(
                self,
                "Rollback Result",
                str(payload.get("reason") or "Rollback failed."),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Rollback Failed",
                f"Could not execute rollback.\n\n{exc}",
            )

    def _run_update_now(self) -> None:
        channel = str(self.auto_update_channel_combo.currentData() or "stable")
        timeout_s = float(self.auto_update_timeout_spin.value())
        require_signature = bool(self.auto_update_require_sig_checkbox.isChecked())
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Update",
            (
                "Run update now?\n\n"
                f"Channel: {channel}\n"
                "This may run package manager commands and may require an app restart."
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        buttons = [
            self.update_check_now_button,
            self.update_run_now_button,
            self.update_rollback_button,
        ]
        try:
            for btn in buttons:
                btn.setEnabled(False)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            payload, exit_code = run_agent_update(
                project="annolid",
                channel=channel,
                timeout_s=timeout_s,
                require_signature=require_signature,
                execute=True,
                skip_post_check=False,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Update Failed",
                f"Could not execute update.\n\n{exc}",
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            for btn in buttons:
                btn.setEnabled(True)

        status = str((payload or {}).get("status") or "").strip().lower()
        if int(exit_code) == 0 and status in {"updated", "staged"}:
            restart_required = bool((payload or {}).get("restart_required", False))
            msg = "Update completed successfully."
            if restart_required:
                msg += "\n\nRestart Annolid to finish applying the update."
            QtWidgets.QMessageBox.information(self, "Update Complete", msg)
            return

        reason = str((payload or {}).get("reason") or "").strip()
        if not reason:
            reason = status or "unknown_error"
        QtWidgets.QMessageBox.warning(
            self,
            "Update Result",
            f"Update did not complete successfully.\n\nStatus: {status or 'unknown'}\nReason: {reason}",
        )

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

    def _add_allowed_bot_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Allowed Folder for Annolid Bot",
            os.path.expanduser("~"),
        )
        if not folder:
            return
        normalized = str(QtCore.QDir(folder).absolutePath())
        if not normalized:
            return
        existing = set(self._collect_allowed_bot_folders()) | set(
            str(p) for p in getattr(self, "_default_allowed_bot_roots", [])
        )
        if normalized in existing:
            return
        self.allowed_bot_folders_list.addItem(normalized)

    def _remove_selected_allowed_bot_folders(self) -> None:
        for item in list(self.allowed_bot_folders_list.selectedItems()):
            row = self.allowed_bot_folders_list.row(item)
            if row >= 0:
                self.allowed_bot_folders_list.takeItem(row)

    def _collect_allowed_bot_folders(self) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for idx in range(self.allowed_bot_folders_list.count()):
            item = self.allowed_bot_folders_list.item(idx)
            text = str(item.text() if item is not None else "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
        return values

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

    def _copy_box_redirect_uri(self) -> None:
        text = str(self.box_redirect_uri_edit.text() or "").strip()
        if not text:
            text = self._box_redirect_default
            self.box_redirect_uri_edit.setText(text)
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        QtWidgets.QMessageBox.information(
            self,
            "Box OAuth",
            "Box redirect URI copied to the clipboard.",
        )

    def _open_box_auth_url(self) -> None:
        from annolid.services.agent_box import complete_box_oauth_browser_flow

        client_id = str(self.box_client_id_edit.text() or "").strip()
        client_secret = str(self.box_client_secret_edit.text() or "").strip()
        authorize_base_url = str(self.box_authorize_base_url_edit.text() or "").strip()
        redirect_uri = str(self.box_redirect_uri_edit.text() or "").strip()

        if not client_id:
            QtWidgets.QMessageBox.information(
                self,
                "Box OAuth",
                "Box auth cancelled: client ID is empty. Fill in the Box fields first.",
            )
            return

        if not redirect_uri:
            QtWidgets.QMessageBox.information(
                self,
                "Box OAuth",
                "Box auth cancelled: redirect URI is empty. Fill in the Box fields first.",
            )
            return

        parsed = urlparse(redirect_uri)
        host = str(parsed.hostname or "").strip().lower()
        if parsed.scheme not in {"http", "https"} or host not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            QtWidgets.QMessageBox.warning(
                self,
                "Box OAuth",
                "Annolid can only capture Box auth automatically for a loopback "
                "redirect URI such as http://localhost:8765/oauth/callback.",
            )
            return

        if not client_secret:
            QtWidgets.QMessageBox.information(
                self,
                "Box OAuth",
                "Box auth cancelled: client secret is empty. Fill in the Box fields first.",
            )
            return

        self.box_auth_button.setEnabled(False)
        self.box_auth_button.setText("Connecting Box…")

        def _run_flow() -> None:
            try:
                payload, _exit_code = complete_box_oauth_browser_flow(
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=redirect_uri,
                    authorize_base_url=authorize_base_url or None,
                    persist=True,
                    open_browser=True,
                )
            except Exception as exc:
                payload = {"ok": False, "error": f"Failed Box OAuth flow: {exc}"}
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": "Unexpected Box OAuth response."}
            self.boxAuthCompleted.emit(payload)

        self._box_auth_thread = threading.Thread(
            target=_run_flow,
            name="BoxOAuthFlow",
            daemon=True,
        )
        self._box_auth_thread.start()

    def _on_box_auth_completed(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.box_auth_button.setEnabled(True)
        self.box_auth_button.setText("Grant Box Access")
        ok = bool(data.get("ok", False))
        if ok:
            try:
                self._agent_config = load_config()
            except Exception:
                pass
            QtWidgets.QMessageBox.information(
                self,
                "Box OAuth",
                "Box access granted and tokens saved.",
            )
            return

        error = str(data.get("error") or "Box OAuth failed.")
        details = str(data.get("error_description") or "").strip()
        if details:
            error = f"{error}\n\n{details}"
        QtWidgets.QMessageBox.warning(self, "Box OAuth", error)

    def _add_provider_dialog(self) -> None:
        dialog = AddProviderDialog(
            self,
            existing_providers=self._provider_specs.keys(),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        result = dialog.get_result()
        if result is None:
            return
        provider, spec, starter_models = result
        self._provider_specs[provider] = dict(spec)
        self._provider_env_keys[provider] = self._spec_api_key_envs(spec)
        provider_entry: Dict[str, Any] = {}
        if spec.get("kind") == "ollama":
            if spec.get("host_default"):
                provider_entry["host"] = str(spec["host_default"])
        elif spec.get("base_url_default"):
            provider_entry["base_url"] = str(spec["base_url_default"])
        if starter_models:
            provider_entry["preferred_models"] = list(starter_models)
        self._settings[provider] = provider_entry
        self._build_provider_tab(provider, insert_before_utility=True)
        self._rebuild_active_provider_combo()
        self._focus_provider_tab(provider)

    def _remove_provider_tab(self, provider: str) -> None:
        if provider in _BUILTIN_PROVIDERS:
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
        self._api_key_inputs.pop(provider, None)
        self._api_key_status_labels.pop(provider, None)
        self._forget_key_buttons.pop(provider, None)
        self._provider_status_labels.pop(provider, None)
        self._provider_probe_buttons.pop(provider, None)
        self._initial_api_keys.pop(provider, None)
        self._cleared_api_key_providers.discard(provider)
        self._settings.pop(provider, None)
        current_active = (
            str(self.active_provider_combo.currentData() or "").strip().lower()
        )
        next_active = "ollama" if current_active == provider else current_active
        self._rebuild_active_provider_combo(selected_provider=next_active)

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

        initial_text = str(initial_value or "").strip()
        self._initial_api_keys[provider] = initial_text
        key_edit = QtWidgets.QLineEdit(initial_text)
        key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        key_edit.setPlaceholderText("Paste API key (or leave blank to use env var)")
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
        env_button.setToolTip("Load the current environment variable into this field")
        env_button.clicked.connect(
            lambda _checked=False, p=provider, edit=key_edit: self._fill_key_from_env(
                p, edit
            )
        )
        row_layout.addWidget(env_button, 0)

        clear_button = QtWidgets.QToolButton(row_widget)
        clear_button.setText("Clear")
        clear_button.setToolTip("Clear this API key for the current session")
        clear_button.clicked.connect(
            lambda _checked=False, p=provider: self._clear_provider_api_key(p)
        )
        row_layout.addWidget(clear_button, 0)

        forget_button = QtWidgets.QToolButton(row_widget)
        forget_button.setText("Forget Saved")
        forget_button.setToolTip(
            f"Remove saved credentials for this provider from {global_env_path()}"
        )
        forget_button.clicked.connect(
            lambda _checked=False, p=provider: self._forget_saved_provider_api_key(p)
        )
        row_layout.addWidget(forget_button, 0)

        layout.addRow("API key:", row_widget)
        status_label = QtWidgets.QLabel("")
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #6b7280;")
        layout.addRow("", status_label)

        self._api_key_inputs[provider] = key_edit
        self._api_key_status_labels[provider] = status_label
        self._forget_key_buttons[provider] = forget_button
        key_edit.textChanged.connect(
            lambda text, p=provider: self._on_api_key_text_changed(p, text)
        )
        self._update_api_key_status(provider)
        return key_edit

    def _add_codex_cli_controls(
        self,
        layout: QtWidgets.QFormLayout,
        *,
        provider: str,
        provider_cfg: Dict[str, Any],
    ) -> QtWidgets.QLineEdit:
        cli_path = str(provider_cfg.get("cli_path") or "codex").strip() or "codex"
        cli_path_edit = QtWidgets.QLineEdit(cli_path, self)
        cli_path_edit.setPlaceholderText("codex")
        layout.addRow("CLI executable:", cli_path_edit)

        row_widget = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        status_label = QtWidgets.QLabel(row_widget)
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #6b7280;")
        row_layout.addWidget(status_label, 1)

        refresh_button = QtWidgets.QToolButton(row_widget)
        refresh_button.setText("Refresh")
        refresh_button.clicked.connect(
            lambda _checked=False, p=provider: self._refresh_codex_cli_status(p)
        )
        row_layout.addWidget(refresh_button, 0)

        layout.addRow("CLI status:", row_widget)
        self._provider_widgets.setdefault(provider, {})["cli_status_label"] = (
            status_label
        )
        self._provider_widgets.setdefault(provider, {})["cli_path_edit"] = cli_path_edit
        cli_path_edit.textChanged.connect(
            lambda _text, p=provider: self._refresh_codex_cli_status(p)
        )
        self._refresh_codex_cli_status(provider)
        return cli_path_edit

    def _refresh_codex_cli_status(self, provider: str) -> None:
        state = self._provider_widgets.get(provider, {})
        label = state.get("cli_status_label")
        edit = state.get("cli_path_edit")
        if not isinstance(label, QtWidgets.QLabel):
            return
        cli_cmd = (
            edit.text().strip() if isinstance(edit, QtWidgets.QLineEdit) else ""
        ) or "codex"
        resolved = shutil.which(cli_cmd)
        if resolved:
            label.setStyleSheet("color: #15803d;")
            label.setText(f"Codex CLI found at {resolved}")
        else:
            label.setStyleSheet("color: #b45309;")
            label.setText(
                f"Executable '{cli_cmd}' not found on PATH. Install Codex CLI or set full path."
            )

    def _add_openai_codex_auth_controls(
        self,
        layout: QtWidgets.QFormLayout,
        *,
        provider: str,
        provider_cfg: Dict[str, Any],
    ) -> None:
        row_widget = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        status_label = QtWidgets.QLabel(row_widget)
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #6b7280;")
        row_layout.addWidget(status_label, 1)

        refresh_button = QtWidgets.QToolButton(row_widget)
        refresh_button.setText("Refresh")
        refresh_button.clicked.connect(
            lambda _checked=False, p=provider: self._refresh_openai_codex_auth_status(p)
        )
        row_layout.addWidget(refresh_button, 0)

        layout.addRow("Authentication:", row_widget)

        note_label = QtWidgets.QLabel(
            "Annolid auto-detects local Codex OAuth credentials and stores only "
            "non-secret status metadata."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #6b7280;")
        layout.addRow("", note_label)

        self._provider_widgets.setdefault(provider, {})["auth_status_label"] = (
            status_label
        )
        cached_state = dict(provider_cfg.get("auth") or {})
        self._set_openai_codex_auth_status(provider, cached_state)

    def _refresh_openai_codex_auth_status(
        self,
        provider: str,
    ) -> None:
        state = detect_openai_codex_auth_state()
        provider_payload = self._settings.setdefault(provider, {})
        provider_payload["auth"] = dict(state)
        self._set_openai_codex_auth_status(provider, state)

    def _set_openai_codex_auth_status(
        self, provider: str, state: Optional[Dict[str, Any]]
    ) -> None:
        label = self._provider_widgets.get(provider, {}).get("auth_status_label")
        if not isinstance(label, QtWidgets.QLabel):
            return
        state = dict(state or {})
        if bool(state.get("authenticated")):
            suffix = str(state.get("account_id_suffix") or "").strip()
            detail = f" account …{suffix}" if suffix else ""
            label.setStyleSheet("color: #15803d;")
            label.setText(f"Local Codex OAuth detected.{detail}")
            return
        label.setStyleSheet("color: #6b7280;")
        error = str(state.get("error") or "").strip()
        if error == "oauth_cli_kit_not_installed":
            label.setText(
                "Codex OAuth helper is not installed. Install `oauth_cli_kit` to enable auto-detection."
            )
            return
        if error:
            label.setText(f"Codex OAuth not detected yet: {error}")
            return
        label.setText(
            "Codex OAuth status will be detected automatically on Save, or check it now with Refresh."
        )

    def _fill_key_from_env(self, provider: str, edit: QtWidgets.QLineEdit) -> None:
        for env_name in self._provider_env_keys.get(provider, []):
            value = str(os.getenv(env_name) or "").strip()
            if value:
                self._cleared_api_key_providers.discard(provider)
                edit.setText(value)
                return

    def _on_api_key_text_changed(self, provider: str, text: str) -> None:
        if str(text or "").strip():
            self._cleared_api_key_providers.discard(provider)
        self._update_api_key_status(provider)

    def _clear_provider_api_key(self, provider: str) -> None:
        edit = self._api_key_inputs.get(provider)
        self._cleared_api_key_providers.add(provider)
        if edit is not None:
            edit.clear()
        self._update_api_key_status(provider)

    def _forget_saved_provider_api_key(self, provider: str) -> None:
        env_names = list(self._provider_env_keys.get(provider, []))
        if not env_names:
            return
        try:
            remove_global_env_vars(env_names)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Could not remove saved credentials",
                f"Failed to update {global_env_path()}.\n\n{exc}",
            )
            return
        clear_session_env_vars(env_names)
        self._dotenv_values = load_global_env_vars()
        self._cleared_api_key_providers.add(provider)
        edit = self._api_key_inputs.get(provider)
        if edit is not None:
            edit.clear()
        self._update_api_key_status(provider)

    def _resolve_initial_api_key(
        self, provider: str, provider_cfg: Dict[str, Any]
    ) -> str:
        # Only pre-fill explicit in-memory session overrides; env/dotenv keys stay
        # external so credential source badges stay accurate and opt-in persistence
        # never copies unrelated process env vars into ~/.annolid/.env.
        return str(provider_cfg.get("api_key") or "").strip()

    def _build_provider_preview_settings(self, provider: str) -> Dict[str, Any]:
        preview = dict(self._settings)
        preview["provider_definitions"] = dict(self._provider_specs)
        block = dict(preview.get(provider) or {})
        state = self._provider_widgets.get(provider, {})
        base_url_edit = state.get("base_url_edit")
        if isinstance(base_url_edit, QtWidgets.QLineEdit):
            block["base_url"] = base_url_edit.text().strip()
        host_edit = (
            self.ollama_host_edit if provider == "ollama" else state.get("host_edit")
        )
        if isinstance(host_edit, QtWidgets.QLineEdit):
            block["host"] = host_edit.text().strip()
        cli_edit = state.get("cli_path_edit")
        if isinstance(cli_edit, QtWidgets.QLineEdit):
            block["cli_path"] = cli_edit.text().strip() or "codex"
        preview[provider] = block
        return preview

    def _update_api_key_status(self, provider: str) -> None:
        edit = self._api_key_inputs.get(provider)
        label = self._api_key_status_labels.get(provider)
        if edit is None or label is None:
            return
        entered = edit.text().strip()
        preview = self._build_provider_preview_settings(provider)
        status = inspect_provider_credential_status(
            preview,
            provider,
            dotenv_values=self._dotenv_values,
            session_key_override=entered,
        )
        forget_button = self._forget_key_buttons.get(provider)
        if forget_button is not None:
            forget_button.setVisible(bool(status.persisted_in_dotenv))
            forget_button.setEnabled(bool(status.persisted_in_dotenv))

        if not entered and provider in self._cleared_api_key_providers:
            label.setStyleSheet("color: #b45309;")
            if status.persisted_in_dotenv:
                label.setText(
                    "Cleared for this session (saved copy remains in ~/.annolid/.env; "
                    "use Forget Saved to delete permanently)."
                )
            else:
                label.setText("API key cleared for this session (click Save to apply).")
            return

        if status.configured:
            label.setStyleSheet("color: #15803d;")
        else:
            label.setStyleSheet("color: #6b7280;")
        label.setText(status.summary)

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
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum
        )
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

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
        models_list.setUniformItemSizes(True)
        models_list.setMinimumHeight(84)
        models_list.setMaximumHeight(132)
        models_list.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        root.addWidget(models_list)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        remove_button = QtWidgets.QPushButton("Remove", container)
        up_button = QtWidgets.QPushButton("Up", container)
        down_button = QtWidgets.QPushButton("Down", container)
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
        """Query the Ollama server asynchronously for available models."""
        self._start_provider_probe("ollama", action="models")

    def _start_provider_probe(self, provider: str, *, action: str) -> None:
        provider_key = str(provider or "").strip().lower()
        action_key = str(action or "test").strip().lower()
        preview = self._build_provider_preview_settings(provider_key)
        state = self._provider_widgets.get(provider_key, {})

        key_edit = state.get("key_edit")
        api_key: Optional[str] = None
        if isinstance(key_edit, QtWidgets.QLineEdit):
            typed = key_edit.text().strip()
            if typed:
                api_key = typed
            elif provider_key in self._cleared_api_key_providers:
                api_key = ""

        base_url: Optional[str] = None
        base_url_edit = state.get("base_url_edit")
        if isinstance(base_url_edit, QtWidgets.QLineEdit):
            base_url = base_url_edit.text().strip() or None

        host: Optional[str] = None
        if provider_key == "ollama":
            host = self.ollama_host_edit.text().strip() or "http://localhost:11434"
        else:
            host_edit = state.get("host_edit")
            if isinstance(host_edit, QtWidgets.QLineEdit):
                host = host_edit.text().strip() or None

        for btn in self._provider_probe_buttons.get(provider_key, []):
            btn.setEnabled(False)
        status_label = self._provider_status_labels.get(provider_key)
        if isinstance(status_label, QtWidgets.QLabel):
            status_label.setStyleSheet("color: #6b7280;")
            status_label.setText(
                "Fetching models…" if action_key == "models" else "Testing connection…"
            )

        self._probe_dispatcher.start_probe(
            provider=provider_key,
            action=action_key,
            settings=preview,
            api_key=api_key,
            base_url=base_url,
            host=host,
        )

    def _on_provider_probe_finished(
        self,
        request_id: int,
        provider: str,
        action: str,
        payload: object,
    ) -> None:
        if self._closing:
            return
        if not self._probe_dispatcher.is_latest(request_id, provider, action):
            return
        for btn in self._provider_probe_buttons.get(provider, []):
            btn.setEnabled(True)

        data = payload if isinstance(payload, dict) else {}
        ok = bool(data.get("ok", False))
        message = str(
            data.get("message") or ("Connected." if ok else "Failed.")
        ).strip()
        models = [
            str(m).strip() for m in list(data.get("models") or []) if str(m).strip()
        ]

        if provider == "openai_codex" and isinstance(data.get("auth"), dict):
            self._settings.setdefault("openai_codex", {})["auth"] = dict(data["auth"])
            self._set_openai_codex_auth_status("openai_codex", data["auth"])
        elif provider == "codex_cli":
            self._refresh_codex_cli_status("codex_cli")

        if ok and models and action == "models":
            models_list = self._models_list_widget_for(provider)
            if models_list is not None:
                if provider == "ollama":
                    merged = models
                else:
                    existing = self._get_models_from_list(models_list)
                    merged = list(existing)
                    for item in models:
                        if item not in merged:
                            merged.append(item)
                self._set_models_list(models_list, merged)
                self._settings.setdefault(provider, {})["preferred_models"] = list(
                    merged
                )
                added_count = (
                    len(models) if provider == "ollama" else len(merged) - len(existing)
                )
                if provider == "ollama":
                    message = f"Loaded {len(models)} model(s) from Ollama."
                elif added_count > 0:
                    message = f"Fetched {len(models)} model(s) ({added_count} added)."
                else:
                    message = f"Fetched {len(models)} model(s) (all already listed)."

        status_label = self._provider_status_labels.get(provider)
        if isinstance(status_label, QtWidgets.QLabel):
            status_label.setStyleSheet("color: #15803d;" if ok else "color: #dc2626;")
            status_label.setText(message)

    def _models_list_widget_for(self, provider: str) -> Optional[QtWidgets.QListWidget]:
        if provider == "ollama":
            editor = getattr(self, "ollama_models_list", None)
        else:
            editor = self._provider_widgets.get(provider, {}).get("models_list")
        if isinstance(editor, dict):
            widget = editor.get("list")
            if isinstance(widget, QtWidgets.QListWidget):
                return widget
        return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_settings(self) -> Dict:
        return self._settings

    def _validate_endpoint_fields(self) -> bool:
        """Validate user-edited HTTP(S) host/base URL fields before persisting."""
        raw_ollama_host = self.ollama_host_edit.text().strip()
        if raw_ollama_host:
            _clean, err = validate_http_endpoint_url(raw_ollama_host)
            if err:
                self._focus_provider_tab("ollama")
                self.ollama_host_edit.setFocus()
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Ollama Server URL",
                    f"Ollama Server URL is invalid: {err}",
                )
                return False

        for provider, spec in self._provider_specs.items():
            if provider == "ollama":
                continue
            state = self._provider_widgets.get(provider, {})
            label = str(spec.get("label") or provider.title())
            host_edit = state.get("host_edit")
            if isinstance(host_edit, QtWidgets.QLineEdit):
                raw_host = host_edit.text().strip()
                if raw_host:
                    _clean, err = validate_http_endpoint_url(raw_host)
                    if err:
                        self._focus_provider_tab(provider)
                        host_edit.setFocus()
                        QtWidgets.QMessageBox.warning(
                            self,
                            f"Invalid {label} Server URL",
                            f"{label} Server URL is invalid: {err}",
                        )
                        return False
            base_url_edit = state.get("base_url_edit")
            if isinstance(base_url_edit, QtWidgets.QLineEdit):
                raw_url = base_url_edit.text().strip()
                if raw_url:
                    _clean, err = validate_http_endpoint_url(raw_url)
                    if err:
                        self._focus_provider_tab(provider)
                        base_url_edit.setFocus()
                        QtWidgets.QMessageBox.warning(
                            self,
                            f"Invalid {label} Base URL",
                            f"{label} Base URL is invalid: {err}",
                        )
                        return False
        return True

    def accept(self) -> None:  # type: ignore[override]
        if not self._validate_endpoint_fields():
            return
        self._closing = True
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
        env_keys_to_remove: List[str] = []
        provider_defs_out: Dict[str, Dict[str, Any]] = {}
        for provider, spec in self._provider_specs.items():
            kind = str(spec.get("kind") or "openai_compat").strip().lower()
            if kind not in _VALID_PROVIDER_KINDS:
                kind = "openai_compat"
            if provider == "ollama":
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

            host_edit = provider_state.get("host_edit")
            if kind == "ollama" and isinstance(host_edit, QtWidgets.QLineEdit):
                default_host = str(spec.get("host_default") or "http://localhost:11434")
                provider_payload["host"] = host_edit.text().strip() or default_host

            base_url_edit = provider_state.get("base_url_edit")
            if kind in {"openai_compat", "openai_codex", "anthropic"} and isinstance(
                base_url_edit, QtWidgets.QLineEdit
            ):
                default_base_url = str(spec.get("base_url_default") or "").strip()
                provider_payload["base_url"] = (
                    base_url_edit.text().strip() or default_base_url
                )
            if kind == "openai_codex":
                transport_combo = provider_state.get("transport_combo")
                if isinstance(transport_combo, QtWidgets.QComboBox):
                    transport = (
                        str(transport_combo.currentData() or "auto").strip().lower()
                    )
                    if transport in {"auto", "sse", "websocket"}:
                        provider_payload["transport"] = transport
                auth_state = detect_openai_codex_auth_state()
                provider_payload["auth"] = auth_state
                self._set_openai_codex_auth_status(provider, auth_state)
            if kind == "codex_cli":
                cli_path_edit = provider_state.get("cli_path_edit")
                if isinstance(cli_path_edit, QtWidgets.QLineEdit):
                    cli_path = cli_path_edit.text().strip()
                    if cli_path and cli_path != "codex":
                        provider_payload["cli_path"] = cli_path

            updated[provider] = provider_payload

            key_edit = provider_state.get("key_edit")
            key_value = (
                key_edit.text().strip()
                if isinstance(key_edit, QtWidgets.QLineEdit)
                else ""
            )
            if kind not in {"openai_codex", "codex_cli", "ollama"}:
                key_envs = self._spec_api_key_envs(spec)
                was_cleared = (
                    provider in self._cleared_api_key_providers
                    or bool(self._initial_api_keys.get(provider))
                ) and not key_value
                for env_name in key_envs:
                    env_key = str(env_name).strip()
                    if not env_key:
                        continue
                    if key_value:
                        mark_session_env_var_active(env_key)
                        self._set_env_if_present(env_key, key_value)
                        if self.persist_env_checkbox.isChecked():
                            persistent_env_values[env_key] = key_value
                    elif was_cleared:
                        clear_session_env_vars([env_key])
                        if self.persist_env_checkbox.isChecked():
                            env_keys_to_remove.append(env_key)
            base_url_env = str(spec.get("base_url_env") or "").strip()
            if (
                kind in {"openai_compat", "openai_codex", "anthropic"}
                and isinstance(base_url_edit, QtWidgets.QLineEdit)
                and base_url_env
            ):
                default_url = str(spec.get("base_url_default") or "").strip()
                base_url_value = base_url_edit.text().strip() or default_url
                is_custom_base = bool(
                    base_url_value
                    and base_url_value.rstrip("/") != default_url.rstrip("/")
                )
                already_configured = bool(
                    os.getenv(base_url_env) or self._dotenv_values.get(base_url_env)
                )
                if is_custom_base or already_configured:
                    mark_session_env_var_active(base_url_env)
                    self._set_env_if_present(base_url_env, base_url_value)
                    if (
                        base_url_value
                        and is_custom_base
                        and self.persist_env_checkbox.isChecked()
                    ):
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
            if kind == "openai_codex" and spec.get("transport_default"):
                provider_defs_out[provider]["transport_default"] = str(
                    spec.get("transport_default")
                ).strip()
        updated["provider_definitions"] = provider_defs_out

        active_provider = (
            str(
                self.active_provider_combo.currentData()
                or updated.get("provider")
                or ""
            )
            .strip()
            .lower()
        )
        if active_provider and active_provider in provider_defs_out:
            updated["provider"] = active_provider
            models_for_provider = list(
                (updated.get(active_provider) or {}).get("preferred_models", []) or []
            )
            models_widget = self._models_list_widget_for(active_provider)
            user_selected_model = bool(
                models_widget is not None and models_widget.selectedItems()
            )
            prev_last = str(
                (updated.get("last_models") or {}).get(active_provider) or ""
            ).strip()
            if models_for_provider and (
                user_selected_model
                or not prev_last
                or prev_last not in models_for_provider
            ):
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
                self._agent_config.tools.box.authorize_base_url = (
                    self.box_authorize_base_url_edit.text().strip()
                    or "https://account.box.com"
                )
                self._agent_config.tools.box.client_id = (
                    self.box_client_id_edit.text().strip()
                )
                self._agent_config.tools.box.client_secret = (
                    self.box_client_secret_edit.text().strip()
                )
                self._agent_config.tools.box.redirect_uri = (
                    self.box_redirect_uri_edit.text().strip()
                )
                self._agent_config.skills.load.watch = bool(
                    self.skills_hot_reload_checkbox.isChecked()
                )
                raw_extra = str(self.skill_source_locations_edit.text() or "").strip()
                extra_dirs = [
                    part.strip() for part in raw_extra.split(os.pathsep) if part.strip()
                ]
                self._agent_config.skills.load.extra_dirs = extra_dirs
                self._agent_config.tools.allowed_read_roots = (
                    self._collect_allowed_bot_folders()
                )
                memory_mode = str(
                    self.memory_mode_combo.currentData() or "semantic_keyword"
                ).strip()
                self._agent_config.memory.mode = (
                    memory_mode
                    if memory_mode in {"semantic_keyword", "lexical"}
                    else "semantic_keyword"
                )
                channel = str(
                    self.auto_update_channel_combo.currentData() or "stable"
                ).strip()
                if channel not in {"stable", "beta", "dev"}:
                    channel = "stable"
                self._agent_config.update.auto.enabled = bool(
                    self.auto_update_enabled_checkbox.isChecked()
                )
                self._agent_config.update.auto.channel = channel
                self._agent_config.update.auto.interval_seconds = max(
                    300, int(self.auto_update_interval_spin.value())
                )
                self._agent_config.update.auto.jitter_seconds = max(
                    0, int(self.auto_update_jitter_spin.value())
                )
                self._agent_config.update.auto.timeout_s = max(
                    1.0, float(self.auto_update_timeout_spin.value())
                )
                self._agent_config.update.auto.require_signature = bool(
                    self.auto_update_require_sig_checkbox.isChecked()
                )
                save_config(self._agent_config)
                os.environ["ANNOLID_EMAIL_POLL_INTERVAL_SECONDS"] = str(
                    self._agent_config.tools.email.polling_interval
                )
                os.environ["ANNOLID_SKILLS_LOAD_WATCH"] = (
                    "1" if bool(self._agent_config.skills.load.watch) else "0"
                )
                os.environ["ANNOLID_SKILLS_EXTRA_DIRS"] = os.pathsep.join(
                    self._agent_config.skills.load.extra_dirs
                )
                os.environ["ANNOLID_MEMORY_RETRIEVAL_PLUGIN"] = (
                    "lexical"
                    if str(self._agent_config.memory.mode).strip() == "lexical"
                    else "workspace_semantic_keyword_v1"
                )
                os.environ["ANNOLID_AUTO_UPDATE_ENABLED"] = (
                    "1" if bool(self._agent_config.update.auto.enabled) else "0"
                )
                os.environ["ANNOLID_AUTO_UPDATE_CHANNEL"] = (
                    self._agent_config.update.auto.channel
                )
                os.environ["ANNOLID_AUTO_UPDATE_INTERVAL_SECONDS"] = str(
                    int(self._agent_config.update.auto.interval_seconds)
                )
                os.environ["ANNOLID_AUTO_UPDATE_JITTER_SECONDS"] = str(
                    int(self._agent_config.update.auto.jitter_seconds)
                )
                os.environ["ANNOLID_AUTO_UPDATE_TIMEOUT_S"] = str(
                    float(self._agent_config.update.auto.timeout_s)
                )
                os.environ["ANNOLID_AUTO_UPDATE_REQUIRE_SIGNATURE"] = (
                    "1"
                    if bool(self._agent_config.update.auto.require_signature)
                    else "0"
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not persist agent runtime settings",
                    f"Failed to update agent config runtime settings.\n\n{exc}",
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
                if env_keys_to_remove:
                    remove_global_env_vars(env_keys_to_remove)
                if persistent_env_values:
                    persist_global_env_vars(persistent_env_values)
            except OSError as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not persist credentials",
                    f"Failed to update {global_env_path()}.\n\n{exc}",
                )
        self._settings = updated
        super().accept()
