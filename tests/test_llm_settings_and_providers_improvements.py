from __future__ import annotations

import io
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest
from qtpy import QtWidgets

from annolid.core.agent.gui_backend.commands import parse_direct_gui_command
from annolid.core.agent.providers.background_chat import (
    _inject_openai_compat_env_defaults,
    dependency_error_for_kind,
    run_anthropic_chat,
    run_gemini_chat,
)
from annolid.core.agent.providers.call_runtime import sanitize_provider_error
from annolid.core.agent.providers.openai_compat import resolve_openai_compat
from annolid.gui.widgets.llm_provider_dialogs import AddProviderDialog
from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog
from annolid.gui.widgets.provider_registry import ProviderRegistry
from annolid.gui.widgets.provider_runtime_sync import refresh_runtime_llm_settings
from annolid.utils import llm_settings as llm_settings_module
from annolid.utils.llm_provider_catalog import (
    fetch_provider_models_live,
    find_provider_preset,
    get_provider_presets,
    inspect_provider_credential_status,
    is_loopback_base_url,
    mask_secret_preview,
    test_provider_connection as probe_provider_connection,
    validate_http_endpoint_url,
)
from annolid.utils.llm_settings import (
    LLMConfig,
    default_settings,
    ensure_provider_env,
    has_provider_api_key,
    load_global_env_vars,
    persist_global_env_vars,
    remove_global_env_vars,
    resolve_llm_config,
    save_llm_settings,
)


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _isolate_settings_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings_dir = tmp_path / ".annolid"
    monkeypatch.setattr(llm_settings_module, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(
        llm_settings_module, "_SETTINGS_FILE", settings_dir / "llm_settings.json"
    )
    monkeypatch.setattr(
        llm_settings_module, "_GLOBAL_DOTENV_FILE", settings_dir / ".env"
    )
    monkeypatch.setattr(llm_settings_module, "_SESSION_CLEARED_ENV_KEYS", set())


def test_atomic_private_settings_and_dotenv_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_paths(monkeypatch, tmp_path)

    settings = default_settings()
    settings["anthropic"]["api_key"] = "sk-ant-secret-9999"
    save_llm_settings(settings)

    settings_file = tmp_path / ".annolid" / "llm_settings.json"
    assert settings_file.exists()
    raw = settings_file.read_text(encoding="utf-8")
    assert "sk-ant-secret-9999" not in raw
    assert '"api_key_env"' in raw
    if os.name == "posix":
        assert stat.S_IMODE(settings_file.stat().st_mode) == 0o600

    persist_global_env_vars(
        {
            "ANTHROPIC_API_KEY": "sk-ant-saved-1234",
            "DEEPSEEK_API_KEY": "sk-ds-saved-5678",
        }
    )
    dotenv_file = tmp_path / ".annolid" / ".env"
    assert dotenv_file.exists()
    if os.name == "posix":
        assert stat.S_IMODE(dotenv_file.stat().st_mode) == 0o600
    assert load_global_env_vars() == {
        "ANTHROPIC_API_KEY": "sk-ant-saved-1234",
        "DEEPSEEK_API_KEY": "sk-ds-saved-5678",
    }

    assert remove_global_env_vars(["ANTHROPIC_API_KEY"]) is True
    assert load_global_env_vars() == {"DEEPSEEK_API_KEY": "sk-ds-saved-5678"}
    assert remove_global_env_vars(["NONEXISTENT_KEY"]) is False
    assert remove_global_env_vars(["DEEPSEEK_API_KEY"]) is True
    assert load_global_env_vars() == {}


def test_cross_provider_key_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-only-1111")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-only-2222")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-only-3333")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ACME_API_KEY", raising=False)

    settings = default_settings()
    settings["openai"]["api_key"] = ""
    settings["openrouter"]["api_key"] = ""
    assert has_provider_api_key(settings, "openrouter") is False
    resolved_or = resolve_llm_config(provider="openrouter", persist=False)
    assert not resolved_or.api_key

    # Custom OpenAI-compat provider must NOT inherit OpenAI/OpenRouter/NVIDIA/Moonshot keys.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-only-4444")
    settings["provider_definitions"]["acme"] = {
        "label": "Acme",
        "kind": "openai_compat",
        "env_keys": ["ACME_API_KEY"],
        "api_key_env": ["ACME_API_KEY"],
        "base_url_default": "https://api.acme.example/v1",
    }
    settings["acme"] = {"base_url": "https://api.acme.example/v1"}
    injected = _inject_openai_compat_env_defaults(
        settings=settings, provider_name="acme", params={}
    )
    assert not injected.get("api_key")


def test_anthropic_builtin_settings_env_and_dependency_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    settings = default_settings()
    assert "anthropic" in settings["provider_definitions"]
    assert settings["provider_definitions"]["anthropic"]["kind"] == "anthropic"
    assert has_provider_api_key(settings, "anthropic") is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-7777")
    assert has_provider_api_key(settings, "anthropic") is True

    cfg = resolve_llm_config(provider="anthropic", persist=False)
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-ant-test-7777"
    assert cfg.base_url == "https://api.anthropic.com"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ensure_provider_env(cfg)
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-7777"
    assert os.environ.get("ANTHROPIC_BASE_URL") == "https://api.anthropic.com"

    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    err = dependency_error_for_kind("anthropic")
    assert err is not None
    assert "anthropic" in err.lower()


def test_sanitize_provider_error_redacts_multi_provider_secret_formats() -> None:
    raw = (
        "Failed with keys sk-ant-api03-ABCDEF1234567890, "
        "sk-proj-ZXCVBN1234567890, sk-svcacct-QWER1234567890, "
        "gsk_1234567890abcdefghij, xai-1234567890abcdefghij, "
        "AIzaSyD1234567890abcdefghijklmno"
    )
    cleaned = sanitize_provider_error(raw)
    for leaked in (
        "ABCDEF1234567890",
        "ZXCVBN1234567890",
        "QWER1234567890",
        "1234567890abcdefghij",
        "AIzaSyD1234567890",
    ):
        assert leaked not in cleaned


def test_provider_catalog_presets_validation_and_credential_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset_ids = {p.id for p in get_provider_presets()}
    for expected in (
        "anthropic",
        "deepseek",
        "nvidia",
        "moonshot",
        "groq",
        "xai",
        "vllm",
        "lmstudio",
        "custom",
    ):
        assert expected in preset_ids
    assert find_provider_preset("deepseek") is not None

    clean, err = validate_http_endpoint_url("https://api.deepseek.com/v1/")
    assert clean == "https://api.deepseek.com/v1"
    assert err == ""
    assert validate_http_endpoint_url("file:///etc/passwd")[1] != ""
    assert validate_http_endpoint_url("ftp://example.com")[1] != ""
    assert validate_http_endpoint_url("")[1] != ""

    assert is_loopback_base_url("http://localhost:1234/v1") is True
    assert is_loopback_base_url("http://127.0.0.1:8000/v1") is True
    assert is_loopback_base_url("https://api.openai.com/v1") is False
    assert mask_secret_preview("sk-secret-abcd") == "••••abcd"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = default_settings()
    settings["openai"]["api_key"] = ""
    settings["anthropic"]["api_key"] = ""

    missing_status = inspect_provider_credential_status(settings, "anthropic")
    assert missing_status.configured is False
    assert missing_status.source == "missing"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-9876")
    dotenv_status = inspect_provider_credential_status(
        settings,
        "anthropic",
        dotenv_values={"ANTHROPIC_API_KEY": "sk-ant-env-9876"},
    )
    assert dotenv_status.configured is True
    assert dotenv_status.source == "dotenv"
    assert dotenv_status.persisted_in_dotenv is True
    assert dotenv_status.masked_key == "••••9876"
    assert "sk-ant-env-9876" not in dotenv_status.summary

    settings["provider_definitions"]["lmstudio"] = {
        "label": "LM Studio",
        "kind": "openai_compat",
        "env_keys": ["LMSTUDIO_API_KEY"],
        "api_key_env": ["LMSTUDIO_API_KEY"],
        "base_url_default": "http://localhost:1234/v1",
    }
    settings["lmstudio"] = {"base_url": "http://localhost:1234/v1"}
    local_status = inspect_provider_credential_status(settings, "lmstudio")
    assert local_status.configured is True
    assert local_status.source == "local"


def test_fetch_provider_models_live_and_probe_all_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[tuple[str, dict[str, str]]] = []

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._buf = io.BytesIO(payload)

        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    def _fake_urlopen(req: Any, timeout: float = 8.0) -> _FakeResponse:
        _ = timeout
        url = str(req.full_url)
        headers = {k.lower(): v for k, v in req.header_items()}
        captured_requests.append((url, headers))
        if url.endswith("/models") and "anthropic" in url:
            return _FakeResponse(
                b'{"data": [{"id": "claude-sonnet-4-5"}, {"id": "claude-3-5-haiku-latest"}]}'
            )
        if "generativelanguage.googleapis.com" in url:
            return _FakeResponse(
                b'{"models": ['
                b'{"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},'
                b'{"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}'
                b"]}"
            )
        if url.endswith("/models"):
            return _FakeResponse(
                b'{"data": [{"id": "deepseek-reasoner"}, {"id": "deepseek-chat"}]}'
            )
        if url.endswith("/api/tags"):
            return _FakeResponse(
                b'{"models": [{"name": "qwen3-vl:8b"}, {"model": "llama3.2:latest"}]}'
            )
        raise AssertionError(f"Unexpected probe URL: {url}")

    monkeypatch.setattr(
        "annolid.utils.llm_provider_catalog.urllib.request.urlopen",
        _fake_urlopen,
    )

    settings = default_settings()
    settings["provider_definitions"]["deepseek"] = {
        "label": "DeepSeek",
        "kind": "openai_compat",
        "env_keys": ["DEEPSEEK_API_KEY"],
        "api_key_env": ["DEEPSEEK_API_KEY"],
        "base_url_default": "https://api.deepseek.com/v1",
    }

    ds_models, ds_err = fetch_provider_models_live(
        "deepseek", settings=settings, api_key="sk-ds-test"
    )
    assert ds_err == ""
    assert ds_models == ["deepseek-chat", "deepseek-reasoner"]

    ant_models, ant_err = fetch_provider_models_live(
        "anthropic", settings=settings, api_key="sk-ant-test"
    )
    assert ant_err == ""
    assert ant_models == ["claude-sonnet-4-5", "claude-3-5-haiku-latest"]

    gem_dummy = "fake-gemini-key"
    gem_result = probe_provider_connection(
        "gemini", settings=settings, api_key=gem_dummy
    )
    assert gem_result["ok"] is True
    assert gem_result["models"] == ["gemini-2.5-pro"]
    gem_url, gem_headers = captured_requests[-1]
    assert gem_dummy not in gem_url
    assert gem_headers.get("x-goog-api-key") == gem_dummy


def test_resolve_openai_compat_custom_and_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)

    lmstudio_local = resolve_openai_compat(
        LLMConfig(
            provider="lmstudio",
            model="qwen2.5-7b-instruct",
            params={"base_url": "http://localhost:1234/v1"},
        )
    )
    assert lmstudio_local.base_url == "http://localhost:1234/v1"
    assert lmstudio_local.api_key == "dummy"

    custom_local = resolve_openai_compat(
        LLMConfig(
            provider="my_local_srv",
            model="custom-local",
            params={"base_url": "http://127.0.0.1:9000/v1"},
        )
    )
    assert custom_local.base_url == "http://127.0.0.1:9000/v1"
    assert custom_local.api_key == "local"

    remote = resolve_openai_compat(
        LLMConfig(
            provider="my_gateway",
            model="custom-model",
            params={"base_url": "https://gw.example.com/v1", "api_key": "sk-gw-123"},
        )
    )
    assert remote.base_url == "https://gw.example.com/v1"
    assert remote.api_key == "sk-gw-123"

    with pytest.raises(ValueError, match="API key"):
        resolve_openai_compat(
            LLMConfig(
                provider="my_gateway",
                model="custom-model",
                params={"base_url": "https://gw.example.com/v1"},
            )
        )


def test_run_anthropic_chat_and_gemini_env_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 1. Anthropic chat routing via UnifiedLLMProvider
    called: Dict[str, Any] = {}

    class _FakeUnifiedProvider:
        def __init__(self, **kwargs: Any) -> None:
            called.update(kwargs)

        async def chat_with_retry(self, **kwargs: Any) -> Any:
            called["chat_model"] = kwargs.get("model")
            return types.SimpleNamespace(
                content="Claude says hello",
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "annolid.core.agent.providers.background_chat.UnifiedLLMProvider",
        _FakeUnifiedProvider,
    )
    settings = default_settings()
    settings["anthropic"]["api_key"] = "sk-ant-chat-123"
    user_prompt, reply = run_anthropic_chat(
        prompt="Hi",
        image_path="",
        model="claude-sonnet-4-5",
        provider_name="anthropic",
        settings=settings,
        load_history_messages=lambda: [],
    )
    assert user_prompt == "Hi"
    assert reply == "Claude says hello"
    assert called["provider_name"] == "anthropic"
    assert called["api_key"] == "sk-ant-chat-123"
    assert called["chat_model"] == "claude-sonnet-4-5"

    # 2. Gemini chat resolves env API key and attaches inline image bytes
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaEnvKeyForGeminiTest12345")

    gemini_calls: Dict[str, Any] = {}

    class _FakeGenerativeModel:
        def __init__(self, model_name: str) -> None:
            gemini_calls["model"] = model_name

        def generate_content(self, content_input: Any) -> Any:
            gemini_calls["contents"] = content_input
            return types.SimpleNamespace(text="Saw the frame")

    fake_genai_mod = types.SimpleNamespace(
        configure=lambda *, api_key: gemini_calls.setdefault("api_key", api_key),
        GenerativeModel=_FakeGenerativeModel,
    )
    fake_google_mod = types.SimpleNamespace(generativeai=fake_genai_mod)
    monkeypatch.setitem(sys.modules, "google", fake_google_mod)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai_mod)

    gem_settings = default_settings()
    gem_settings["gemini"]["api_key"] = ""
    _gem_prompt, gem_reply = run_gemini_chat(
        prompt="Describe image",
        image_path=str(img_path),
        model="gemini-2.5-pro",
        provider_name="gemini",
        settings=gem_settings,
    )
    assert gem_reply == "Saw the frame"
    assert gemini_calls["api_key"] == "AIzaEnvKeyForGeminiTest12345"
    assert isinstance(gemini_calls["contents"], list)
    assert any(
        isinstance(part, dict)
        and part.get("mime_type") == "image/png"
        and part.get("data", b"").startswith(b"\x89PNG")
        for part in gemini_calls["contents"]
    )


def test_parse_direct_gui_command_accepts_anthropic_and_custom_providers() -> None:
    cmd1 = parse_direct_gui_command("switch model anthropic:claude-sonnet-4-5")
    assert cmd1 is not None
    assert cmd1["name"] == "set_chat_model"
    assert cmd1["args"]["provider"] == "anthropic"
    assert cmd1["args"]["model"] == "claude-sonnet-4-5"

    cmd2 = parse_direct_gui_command("set chat model deepseek:deepseek-reasoner")
    assert cmd2 is not None
    assert cmd2["name"] == "set_chat_model"
    assert cmd2["args"]["provider"] == "deepseek"
    assert cmd2["args"]["model"] == "deepseek-reasoner"


def test_llm_settings_dialog_preserves_timeouts_active_provider_and_codex_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.detect_openai_codex_auth_state",
        lambda: {"authenticated": False, "error": "not_signed_in"},
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.save_tts_settings",
        lambda _payload: None,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.save_config",
        lambda _cfg: None,
    )

    settings = default_settings()
    settings["provider"] = "anthropic"
    settings["openai_codex"]["transport"] = "websocket"

    dialog = LLMSettingsDialog(None, settings=settings)
    try:
        # Active provider combo and initial visible tab match saved active provider.
        assert dialog.active_provider_combo.currentData() == "anthropic"
        assert dialog._provider_for_current_tab() == "anthropic"

        # Codex CLI tab does not render a bogus API key edit.
        assert dialog._provider_widgets["codex_cli"]["key_edit"] is None
        assert dialog._provider_widgets["codex_cli"]["cli_path_edit"] is not None

        # OpenAI Codex exposes transport selector with saved value.
        transport_combo = dialog._provider_widgets["openai_codex"]["transport_combo"]
        assert transport_combo is not None
        assert transport_combo.currentData() == "websocket"

        # Switching visible tab to Ollama without changing Active provider must NOT
        # silently hijack the active provider on Save.
        dialog._focus_provider_tab("ollama")
        assert dialog._provider_for_current_tab() == "ollama"

        dialog.accept()
        updated = dialog.get_settings()
        assert updated["provider"] == "anthropic"
        assert updated["openai_codex"]["transport"] == "websocket"
        assert updated["agent"]["loop_llm_timeout_seconds"] == 600.0
        assert updated["agent"]["loop_llm_timeout_seconds_no_tools"] == 180.0
        assert updated["agent"]["loop_tool_timeout_seconds"] == 900.0
    finally:
        dialog.close()


def test_llm_settings_dialog_key_clear_and_forget_saved_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.detect_openai_codex_auth_state",
        lambda: {"authenticated": False},
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.save_tts_settings",
        lambda _payload: None,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.save_config",
        lambda _cfg: None,
    )

    persist_global_env_vars({"ANTHROPIC_API_KEY": "sk-ant-disk-4321"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-disk-4321")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-sess-8765")

    settings = default_settings()
    settings["anthropic"]["api_key"] = ""
    settings["openai"]["api_key"] = "sk-openai-sess-8765"

    dialog = LLMSettingsDialog(None, settings=settings)
    try:
        # Env-backed key is not copied into the line edit, but status shows dotenv + mask.
        assert dialog._api_key_inputs["anthropic"].text() == ""
        assert "••••4321" in dialog._api_key_status_labels["anthropic"].text()
        assert dialog._forget_key_buttons["anthropic"].isEnabled() is True

        # Forget Saved removes from ~/.annolid/.env and os.environ immediately.
        dialog._forget_saved_provider_api_key("anthropic")
        assert "ANTHROPIC_API_KEY" not in load_global_env_vars()
        assert "ANTHROPIC_API_KEY" not in os.environ

        # Clearing a session key removes it from os.environ upon Save.
        dialog._clear_provider_api_key("openai")
        dialog.accept()
        assert "OPENAI_API_KEY" not in os.environ
    finally:
        dialog.close()


def test_add_provider_dialog_presets_and_validation() -> None:
    _ensure_qapp()
    dialog = AddProviderDialog(
        None,
        existing_providers=["ollama", "openai", "anthropic"],
    )
    try:
        idx = dialog.preset_combo.findData("deepseek")
        assert idx >= 0
        dialog.preset_combo.setCurrentIndex(idx)
        assert dialog.provider_id_edit.text() == "deepseek"
        assert dialog.base_url_edit.text() == "https://api.deepseek.com/v1"
        assert dialog.env_var_edit.text() == "DEEPSEEK_API_KEY"

        # Reject invalid URL scheme.
        dialog.base_url_edit.setText("ftp://bad.example.com")
        dialog.accept()
        assert dialog.get_result() is None
        assert not dialog.error_label.isHidden()
        assert "http" in dialog.error_label.text().lower()

        # Accept valid URL.
        dialog.base_url_edit.setText("https://api.deepseek.com/v1")
        dialog.accept()
        result = dialog.get_result()
        assert result is not None
        provider_id, spec, models = result
        assert provider_id == "deepseek"
        assert spec["kind"] == "openai_compat"
        assert spec["api_key_env"] == ["DEEPSEEK_API_KEY"]
        assert "deepseek-chat" in models
    finally:
        dialog.close()


def test_provider_registry_ollama_uses_client_without_mutating_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    seen_hosts: list[str] = []

    class _FakeOllamaClient:
        def __init__(self, *, host: str) -> None:
            seen_hosts.append(host)

        def list(self) -> Dict[str, Any]:
            return {
                "models": [
                    {"model": "qwen3-vl:8b"},
                    {"name": "llama3.2-vision:latest"},
                ]
            }

    fake_ollama = types.SimpleNamespace(Client=_FakeOllamaClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    settings = {
        "provider": "ollama",
        "ollama": {
            "host": "http://127.0.0.1:11435",
            "preferred_models": [],
        },
        "last_models": {},
    }
    registry = ProviderRegistry(settings, lambda _s: None)
    models = registry.available_models("ollama")
    assert models == ["qwen3-vl:8b", "llama3.2-vision:latest"]
    assert seen_hosts == ["http://127.0.0.1:11435"]
    assert "OLLAMA_HOST" not in os.environ


def test_ai_chat_widget_visible_provider_selector_and_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "annolid.gui.widgets.provider_registry.ProviderRegistry._fetch_ollama_models",
        lambda self: ["qwen3-vl"],
    )
    from annolid.gui.widgets.ai_chat_widget import AIChatWidget

    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, _msg: warnings.append(str(title)),
    )

    widget = AIChatWidget()
    try:
        # Provider selector is now visible and synced to selected_provider.
        assert not widget.provider_selector.isHidden()
        assert widget.provider_selector.maximumWidth() > 0
        assert widget.provider_selector.currentData() == widget.selected_provider

        # Anthropic requires API key.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        widget.llm_settings["anthropic"]["api_key"] = ""
        widget.selected_provider = "anthropic"
        assert widget._ensure_provider_ready() is False
        assert warnings and "Anthropic" in warnings[-1]

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ready-123")
        assert widget._ensure_provider_ready() is True

        # Local loopback OpenAI-compatible provider works without API key.
        widget.llm_settings["provider_definitions"]["lmstudio"] = {
            "label": "LM Studio",
            "kind": "openai_compat",
            "env_keys": ["LMSTUDIO_API_KEY"],
            "api_key_env": ["LMSTUDIO_API_KEY"],
            "base_url_default": "http://localhost:1234/v1",
        }
        widget.llm_settings["lmstudio"] = {"base_url": "http://localhost:1234/v1"}
        widget.selected_provider = "lmstudio"
        assert widget._ensure_provider_ready() is True

        # Runtime sync must not trigger editTextChanged side effects.
        edit_events: list[str] = []
        widget.model_selector.editTextChanged.connect(edit_events.append)
        refresh_runtime_llm_settings(widget)
        assert edit_events == []
    finally:
        widget.close()


def test_agent_loop_anthropic_routing_and_missing_key_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio
    from annolid.core.agent.loop import AgentLoop
    from annolid.core.agent.tools import FunctionToolRegistry

    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loop_no_key = AgentLoop(
        tools=FunctionToolRegistry(),
        workspace=tmp_path,
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    err_callable, _model_name, _impl = loop_no_key._build_default_llm_callable(
        profile=None,
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    err_result = asyncio.run(err_callable([], [], "claude-sonnet-4-5", None))
    assert err_result["finish_reason"] == "error"
    assert "Anthropic provider requires API key" in str(err_result["content"])

    captured: Dict[str, Any] = {}

    class _FakeUnified:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def chat_with_retry(self, **kwargs: Any) -> Any:
            return types.SimpleNamespace(
                content="ok",
                tool_calls=[],
                finish_reason="stop",
            )

    monkeypatch.setattr("annolid.core.agent.loop.UnifiedLLMProvider", _FakeUnified)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-loop-999")
    loop_with_key = AgentLoop(
        tools=FunctionToolRegistry(),
        workspace=tmp_path,
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    llm_callable, resolved_model, _impl = loop_with_key._build_default_llm_callable(
        profile=None,
        provider="anthropic",
        model="claude-sonnet-4-5",
    )
    assert callable(llm_callable)
    assert resolved_model == "claude-sonnet-4-5"
    assert captured["provider_name"] == "anthropic"
    assert captured["api_key"] == "sk-ant-loop-999"


def test_llm_settings_dialog_probe_merges_models_and_handles_add_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.detect_openai_codex_auth_state",
        lambda: {"authenticated": False},
    )

    dialog = LLMSettingsDialog(None, settings=default_settings())
    try:
        # Simulate a live model fetch completion for anthropic.
        req_id = 1
        dialog._probe_dispatcher._latest_request_by_key[("anthropic", "models")] = (
            req_id
        )
        dialog._on_provider_probe_finished(
            req_id,
            "anthropic",
            "models",
            {
                "ok": True,
                "provider": "anthropic",
                "models": ["claude-sonnet-4-5", "claude-opus-4-1"],
                "message": "Fetched 2 model(s).",
            },
        )
        models = dialog._get_models_from_list(
            dialog._provider_widgets["anthropic"]["models_list"]["list"]
        )
        assert "claude-opus-4-1" in models
        assert "1 added" in dialog._provider_status_labels["anthropic"].text()

        # Adding a custom provider inserts it BEFORE Agent Runtime & TTS tabs.
        agent_idx_before = dialog._tabs.indexOf(dialog._agent_runtime_tab_widget)
        dialog._provider_specs["deepseek"] = {
            "label": "DeepSeek",
            "kind": "openai_compat",
            "env_keys": ["DEEPSEEK_API_KEY"],
            "api_key_env": ["DEEPSEEK_API_KEY"],
            "base_url_default": "https://api.deepseek.com/v1",
            "base_url_env": "",
            "model_placeholder": "Type model name",
        }
        dialog._provider_env_keys["deepseek"] = ["DEEPSEEK_API_KEY"]
        dialog._settings["deepseek"] = {
            "base_url": "https://api.deepseek.com/v1",
            "preferred_models": ["deepseek-chat"],
        }
        dialog._build_provider_tab("deepseek", insert_before_utility=True)
        dialog._rebuild_active_provider_combo()
        ds_idx = dialog._tabs.indexOf(
            dialog._provider_widgets["deepseek"]["tab_widget"]
        )
        agent_idx_after = dialog._tabs.indexOf(dialog._agent_runtime_tab_widget)
        assert ds_idx == agent_idx_before
        assert ds_idx < agent_idx_after
        assert dialog.active_provider_combo.findData("deepseek") >= 0
    finally:
        dialog.close()


def test_unified_provider_openrouter_does_not_hijack_anthropic_model_prefix() -> None:
    from annolid.core.agent.providers.unified_provider import UnifiedLLMProvider

    openrouter_provider = UnifiedLLMProvider(
        provider_name="openrouter",
        api_key="sk-or-v1-test",
        api_base="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-3.5-sonnet",
    )
    assert (
        openrouter_provider._resolved_provider_name("anthropic/claude-3.5-sonnet")
        == "openrouter"
    )

    native_anthropic = UnifiedLLMProvider(
        provider_name="anthropic",
        api_key="sk-ant-test",
        default_model="claude-sonnet-4-5",
    )
    assert native_anthropic._resolved_provider_name("claude-sonnet-4-5") == "anthropic"


def test_session_clear_prevents_dotenv_resurrection_and_export_dotenv_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from annolid.utils.llm_settings import (
        clear_session_env_vars,
        load_llm_settings,
        mark_session_env_var_active,
    )

    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    dotenv_path = tmp_path / ".annolid" / ".env"
    dotenv_path.parent.mkdir(parents=True, exist_ok=True)
    dotenv_path.write_text(
        'export OPENAI_API_KEY="dummy-openai-val"\n'
        "DEEPSEEK_API_KEY=dummy-deepseek-val\n",
        encoding="utf-8",
    )

    # export-prefixed .env lines are parsed cleanly.
    loaded = load_global_env_vars()
    assert loaded["OPENAI_API_KEY"] == "dummy-openai-val"
    assert loaded["DEEPSEEK_API_KEY"] == "dummy-deepseek-val"

    # Session-clearing a dotenv-backed key prevents load_llm_settings() from resurrecting it.
    clear_session_env_vars(["OPENAI_API_KEY"])
    try:
        settings = load_llm_settings()
        assert "OPENAI_API_KEY" not in os.environ
        assert not settings["openai"].get("api_key")
        assert os.environ.get("DEEPSEEK_API_KEY") == "dummy-deepseek-val"
    finally:
        mark_session_env_var_active("OPENAI_API_KEY")
        mark_session_env_var_active("DEEPSEEK_API_KEY")


def test_probe_redirect_handler_blocks_non_http_and_strips_cross_origin_secrets() -> (
    None
):
    import urllib.error
    import urllib.request
    from annolid.utils.llm_provider_catalog import _SafeProbeRedirectHandler

    handler = _SafeProbeRedirectHandler()
    req = urllib.request.Request(
        "https://api.example.com/v1/models",
        headers={
            "Authorization": "Bearer sk-secret-token",
            "x-api-key": "sk-ant-secret",
            "x-goog-api-key": "AIzaSecret",
            "Accept": "application/json",
        },
        method="GET",
    )

    # Non-http(s) redirect is blocked.
    with pytest.raises(urllib.error.URLError, match="unsupported endpoint scheme"):
        handler.redirect_request(req, None, 302, "Found", {}, "file:///etc/passwd")

    # Cross-origin redirect strips sensitive auth headers but preserves Accept.
    cross = handler.redirect_request(
        req, None, 302, "Found", {}, "https://evil.example.org/models"
    )
    assert cross is not None
    lower_headers = {
        k.lower(): v
        for k, v in list(cross.headers.items()) + list(cross.unredirected_hdrs.items())
    }
    assert "authorization" not in lower_headers
    assert "x-api-key" not in lower_headers
    assert "x-goog-api-key" not in lower_headers
    assert lower_headers.get("accept") == "application/json"

    # Same-origin redirect keeps auth headers.
    same = handler.redirect_request(
        req, None, 307, "Temp", {}, "https://api.example.com/v2/models"
    )
    assert same is not None
    same_headers = {
        k.lower(): v
        for k, v in list(same.headers.items()) + list(same.unredirected_hdrs.items())
    }
    assert same_headers.get("authorization") == "Bearer sk-secret-token"


def test_llm_settings_dialog_validates_urls_and_avoids_default_base_url_dotenv_pollution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.detect_openai_codex_auth_state",
        lambda: {"authenticated": False},
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.save_tts_settings",
        lambda _payload: None,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.llm_settings_dialog.save_config",
        lambda _cfg: None,
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, _msg: warnings.append(str(title)),
    )

    dialog = LLMSettingsDialog(None, settings=default_settings())
    try:
        # Invalid URL blocks accept() and keeps dialog open (not closing).
        dialog.ollama_host_edit.setText("file:///etc/passwd")
        dialog.accept()
        assert warnings and "Invalid Ollama" in warnings[-1]
        assert dialog._closing is False

        # Restore valid URL, set only Anthropic key with persist enabled.
        dialog.ollama_host_edit.setText("http://localhost:11434")
        dialog._api_key_inputs["anthropic"].setText("sk-ant-new-4567")
        dialog.persist_env_checkbox.setChecked(True)
        dialog.accept()

        saved_env = load_global_env_vars()
        assert saved_env.get("ANTHROPIC_API_KEY") == "sk-ant-new-4567"
        # Default base URLs were NOT customized and must NOT pollute ~/.annolid/.env.
        assert "OPENAI_BASE_URL" not in saved_env
        assert "ANTHROPIC_BASE_URL" not in saved_env
    finally:
        dialog.close()


def test_caption_widget_reenables_chat_button_when_image_gen_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "annolid.gui.widgets.provider_registry.ProviderRegistry._fetch_ollama_models",
        lambda self: ["qwen3-vl"],
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: None,
    )
    from annolid.gui.widgets.caption import CaptionWidget

    widget = CaptionWidget()
    try:
        widget.selected_provider = "ollama"
        widget.selected_model = "qwen3-vl"
        widget.prompt_text_edit.setText("image: a mouse in an arena")
        widget.chat_with_model()
        assert widget.chat_button.isEnabled() is True
    finally:
        widget.close()
