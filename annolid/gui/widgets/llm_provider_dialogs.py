from __future__ import annotations

import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qtpy import QtCore, QtWidgets

from annolid.utils.llm_provider_catalog import (
    ProviderPreset,
    fetch_provider_models_live,
    find_provider_preset,
    get_provider_presets,
    test_provider_connection,
    validate_http_endpoint_url,
)

_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_ADDABLE_PROVIDER_KINDS: Tuple[Tuple[str, str], ...] = (
    ("OpenAI-Compatible API", "openai_compat"),
    ("Anthropic Messages API", "anthropic"),
    ("Google Gemini API", "gemini"),
    ("Ollama Server", "ollama"),
)


class ProviderProbeDispatcher(QtCore.QObject):
    """Run non-blocking provider connection tests and model fetches off the UI thread."""

    probeFinished = QtCore.Signal(int, str, str, object)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._next_id = 0
        self._latest_request_by_key: Dict[Tuple[str, str], int] = {}

    def start_probe(
        self,
        *,
        provider: str,
        action: str,
        settings: Dict[str, Any],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        host: Optional[str] = None,
        timeout_s: float = 8.0,
    ) -> int:
        provider_key = str(provider or "").strip().lower()
        action_key = str(action or "test").strip().lower()
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._latest_request_by_key[(provider_key, action_key)] = request_id

        settings_snapshot = dict(settings or {})

        def _worker() -> None:
            try:
                if action_key == "models":
                    models, err = fetch_provider_models_live(
                        provider_key,
                        settings=settings_snapshot,
                        api_key=api_key,
                        base_url=base_url,
                        host=host,
                        timeout_s=timeout_s,
                    )
                    payload: Dict[str, Any] = {
                        "ok": not bool(err),
                        "provider": provider_key,
                        "models": list(models),
                        "message": (
                            f"Fetched {len(models)} model(s)."
                            if not err and models
                            else "Connected, but no models were returned."
                            if not err
                            else err
                        ),
                    }
                else:
                    payload = test_provider_connection(
                        provider_key,
                        settings=settings_snapshot,
                        api_key=api_key,
                        base_url=base_url,
                        host=host,
                        timeout_s=timeout_s,
                    )
            except Exception as exc:  # pragma: no cover - defensive worker guard
                payload = {
                    "ok": False,
                    "provider": provider_key,
                    "models": [],
                    "message": str(exc) or "Provider check failed.",
                }
            try:
                self.probeFinished.emit(request_id, provider_key, action_key, payload)
            except RuntimeError:
                # Dispatcher/dialog was already destroyed before worker completed.
                return

        thread = threading.Thread(
            target=_worker,
            name=f"LLMProbe-{provider_key}-{action_key}-{request_id}",
            daemon=True,
        )
        thread.start()
        return request_id

    def is_latest(self, request_id: int, provider: str, action: str) -> bool:
        key = (str(provider or "").strip().lower(), str(action or "").strip().lower())
        with self._lock:
            return self._latest_request_by_key.get(key) == request_id


class AddProviderDialog(QtWidgets.QDialog):
    """Single-form dialog for adding a preset or custom AI model provider."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        existing_providers: Iterable[str] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add AI Model Provider")
        self.resize(480, 340)

        self._existing_providers = {
            str(p or "").strip().lower()
            for p in existing_providers
            if str(p or "").strip()
        }
        self._result: Optional[Tuple[str, Dict[str, Any], List[str]]] = None

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.preset_combo = QtWidgets.QComboBox(self)
        presets = list(get_provider_presets())
        # Show presets that are not already configured first, then remaining presets.
        unconfigured = [p for p in presets if p.id not in self._existing_providers]
        configured = [p for p in presets if p.id in self._existing_providers]
        ordered_presets = unconfigured + configured
        for preset in ordered_presets:
            suffix = " (already added)" if preset.id in self._existing_providers else ""
            self.preset_combo.addItem(f"{preset.label}{suffix}", userData=preset.id)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Preset:", self.preset_combo)

        self.description_label = QtWidgets.QLabel("", self)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #6b7280;")
        form.addRow("", self.description_label)

        self.provider_id_edit = QtWidgets.QLineEdit(self)
        self.provider_id_edit.setPlaceholderText("lowercase id, e.g. deepseek or vllm")
        form.addRow("Provider ID:", self.provider_id_edit)

        self.label_edit = QtWidgets.QLineEdit(self)
        self.label_edit.setPlaceholderText("Display name, e.g. DeepSeek")
        form.addRow("Display name:", self.label_edit)

        self.kind_combo = QtWidgets.QComboBox(self)
        for kind_label, kind_value in _ADDABLE_PROVIDER_KINDS:
            self.kind_combo.addItem(kind_label, userData=kind_value)
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow("API protocol:", self.kind_combo)

        self.base_url_edit = QtWidgets.QLineEdit(self)
        self.base_url_edit.setPlaceholderText("https://api.example.com/v1")
        form.addRow("Base URL:", self.base_url_edit)

        self.env_var_edit = QtWidgets.QLineEdit(self)
        self.env_var_edit.setPlaceholderText(
            "e.g. DEEPSEEK_API_KEY (optional for local)"
        )
        form.addRow("API key env var:", self.env_var_edit)

        self.models_edit = QtWidgets.QLineEdit(self)
        self.models_edit.setPlaceholderText("Comma-separated starter models (optional)")
        form.addRow("Starter models:", self.models_edit)

        self.error_label = QtWidgets.QLabel("", self)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #dc2626;")
        self.error_label.setVisible(False)

        root.addLayout(form)
        root.addWidget(self.error_label)
        root.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        ok_btn = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Add Provider")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if self.preset_combo.count() > 0:
            self._apply_preset_by_id(str(self.preset_combo.currentData() or "custom"))

    def _on_preset_changed(self, _index: int) -> None:
        preset_id = str(self.preset_combo.currentData() or "").strip().lower()
        self._apply_preset_by_id(preset_id)

    def _apply_preset_by_id(self, preset_id: str) -> None:
        preset: Optional[ProviderPreset] = find_provider_preset(preset_id)
        if preset is None:
            return
        self.description_label.setText(preset.description)
        if preset.id == "custom":
            self.provider_id_edit.setText("")
            self.label_edit.setText("")
            self.env_var_edit.setText("")
        else:
            suggested_id = preset.id
            if suggested_id in self._existing_providers:
                suffix = 2
                while f"{preset.id}_{suffix}" in self._existing_providers:
                    suffix += 1
                suggested_id = f"{preset.id}_{suffix}"
            self.provider_id_edit.setText(suggested_id)
            self.label_edit.setText(preset.label)
            env_vars = ", ".join(preset.api_key_env or preset.env_keys)
            self.env_var_edit.setText(env_vars)

        kind_idx = self.kind_combo.findData(preset.kind)
        if kind_idx >= 0:
            self.kind_combo.setCurrentIndex(kind_idx)
        self.base_url_edit.setText(preset.base_url_default or preset.host_default)
        self.models_edit.setText(", ".join(preset.default_models))
        self.error_label.setVisible(False)
        self._on_kind_changed(self.kind_combo.currentIndex())

    def _on_kind_changed(self, _index: int) -> None:
        kind = str(self.kind_combo.currentData() or "openai_compat").strip().lower()
        needs_url = kind in {"openai_compat", "anthropic", "ollama"}
        self.base_url_edit.setEnabled(needs_url)
        needs_env = kind in {"openai_compat", "anthropic", "gemini"}
        self.env_var_edit.setEnabled(needs_env)

    def _show_validation_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(True)

    def accept(self) -> None:  # type: ignore[override]
        raw_id = (
            str(self.provider_id_edit.text() or "").strip().lower().replace(" ", "_")
        )
        if not raw_id:
            self._show_validation_error(
                "Enter a provider ID (for example: deepseek, vllm)."
            )
            self.provider_id_edit.setFocus()
            return
        if not _PROVIDER_ID_PATTERN.match(raw_id):
            self._show_validation_error(
                "Provider ID must start with a letter/digit and use only a-z, 0-9, '_' or '-' (max 32 chars)."
            )
            self.provider_id_edit.setFocus()
            return
        if raw_id in self._existing_providers:
            self._show_validation_error(f"Provider '{raw_id}' already exists.")
            self.provider_id_edit.setFocus()
            return

        kind = str(self.kind_combo.currentData() or "openai_compat").strip().lower()
        label = (
            str(self.label_edit.text() or "").strip()
            or raw_id.replace("_", " ").title()
        )
        raw_url = str(self.base_url_edit.text() or "").strip()
        clean_url = ""
        if kind in {"openai_compat", "ollama"} or (kind == "anthropic" and raw_url):
            if not raw_url and kind == "openai_compat":
                self._show_validation_error(
                    "Base URL is required for OpenAI-compatible providers (e.g. https://api.example.com/v1)."
                )
                self.base_url_edit.setFocus()
                return
            if raw_url:
                clean_url, url_err = validate_http_endpoint_url(raw_url)
                if url_err:
                    self._show_validation_error(url_err)
                    self.base_url_edit.setFocus()
                    return

        env_vars: List[str] = []
        if kind in {"openai_compat", "anthropic", "gemini"}:
            raw_envs = str(self.env_var_edit.text() or "").strip()
            for part in raw_envs.split(","):
                name = part.strip().upper()
                if name and name not in env_vars:
                    env_vars.append(name)
            if not env_vars:
                if kind == "anthropic":
                    env_vars = ["ANTHROPIC_API_KEY"]
                elif kind == "gemini":
                    env_vars = ["GEMINI_API_KEY"]
                else:
                    env_vars = [f"{raw_id.upper().replace('-', '_')}_API_KEY"]

        starter_models: List[str] = []
        for part in str(self.models_edit.text() or "").split(","):
            model = part.strip()
            if model and model not in starter_models:
                starter_models.append(model)

        spec: Dict[str, Any] = {
            "label": label,
            "kind": kind,
            "env_keys": list(env_vars) if kind != "ollama" else ["OLLAMA_HOST"],
            "api_key_env": list(env_vars) if kind != "ollama" else [],
            "base_url_default": clean_url if kind != "ollama" else "",
            "base_url_env": "",
            "host_default": clean_url if kind == "ollama" else "",
            "model_placeholder": (
                f"Type model name (e.g. {starter_models[0]}) and press Add"
                if starter_models
                else "Type model name and press Add"
            ),
        }
        self._result = (raw_id, spec, starter_models)
        super().accept()

    def get_result(self) -> Optional[Tuple[str, Dict[str, Any], List[str]]]:
        return self._result
