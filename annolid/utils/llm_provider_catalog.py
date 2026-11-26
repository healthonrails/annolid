from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

_ALLOWED_HTTP_SCHEMES = frozenset({"http", "https"})
_MAX_PROBE_RESPONSE_BYTES = 4 * 1024 * 1024
_DEFAULT_PROBE_TIMEOUT_S = 8.0
_MAX_PROBE_REDIRECTS = 5
_SENSITIVE_REDIRECT_HEADERS = frozenset(
    {"authorization", "x-api-key", "x-goog-api-key"}
)
_DEFAULT_STDLIB_URLOPEN = urllib.request.urlopen


class _SafeProbeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Restrict redirects to http(s) and strip auth headers on cross-origin hops."""

    max_redirections = _MAX_PROBE_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        new_parts = urlsplit(str(newurl or "").strip())
        new_scheme = str(new_parts.scheme or "").strip().lower()
        if (
            new_scheme not in _ALLOWED_HTTP_SCHEMES
            or not str(new_parts.netloc or "").strip()
        ):
            raise urllib.error.URLError(
                f"Blocked redirect to unsupported endpoint scheme {new_scheme or '(none)'!r}."
            )
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old_parts = urlsplit(req.full_url)
        same_origin = (
            str(old_parts.scheme or "").lower() == new_scheme
            and str(old_parts.hostname or "").lower()
            == str(new_parts.hostname or "").lower()
            and old_parts.port == new_parts.port
        )
        if not same_origin:
            for hdr in list(redirected.headers.keys()):
                if hdr.lower() in _SENSITIVE_REDIRECT_HEADERS:
                    redirected.remove_header(hdr)
            for hdr in list(redirected.unredirected_hdrs.keys()):
                if hdr.lower() in _SENSITIVE_REDIRECT_HEADERS:
                    redirected.unredirected_hdrs.pop(hdr, None)
        return redirected


_SAFE_PROBE_OPENER = urllib.request.build_opener(_SafeProbeRedirectHandler())


@dataclass(frozen=True)
class ProviderPreset:
    """Built-in template for adding an AI model provider with one click."""

    id: str
    label: str
    kind: str
    env_keys: Tuple[str, ...]
    api_key_env: Tuple[str, ...]
    base_url_default: str = ""
    base_url_env: str = ""
    host_default: str = ""
    default_models: Tuple[str, ...] = ()
    model_placeholder: str = "Type model name and press Add"
    description: str = ""

    def to_provider_definition(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "label": self.label,
            "kind": self.kind,
            "env_keys": list(self.env_keys),
            "api_key_env": list(self.api_key_env),
            "base_url_default": self.base_url_default,
            "base_url_env": self.base_url_env,
            "model_placeholder": self.model_placeholder,
        }
        if self.host_default:
            payload["host_default"] = self.host_default
        return payload


PROVIDER_PRESETS: Tuple[ProviderPreset, ...] = (
    ProviderPreset(
        id="anthropic",
        label="Anthropic Claude",
        kind="anthropic",
        env_keys=("ANTHROPIC_API_KEY",),
        api_key_env=("ANTHROPIC_API_KEY",),
        base_url_default="https://api.anthropic.com",
        base_url_env="ANTHROPIC_BASE_URL",
        default_models=(
            "claude-sonnet-4-5",
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
        ),
        model_placeholder="Type model name (e.g. claude-sonnet-4-5) and press Add",
        description="Native Anthropic Messages API for Claude models.",
    ),
    ProviderPreset(
        id="deepseek",
        label="DeepSeek",
        kind="openai_compat",
        env_keys=("DEEPSEEK_API_KEY",),
        api_key_env=("DEEPSEEK_API_KEY",),
        base_url_default="https://api.deepseek.com/v1",
        default_models=("deepseek-chat", "deepseek-reasoner"),
        model_placeholder="Type model name (e.g. deepseek-chat) and press Add",
        description="DeepSeek OpenAI-compatible API.",
    ),
    ProviderPreset(
        id="nvidia",
        label="NVIDIA NIM",
        kind="openai_compat",
        env_keys=("NVIDIA_API_KEY",),
        api_key_env=("NVIDIA_API_KEY",),
        base_url_default="https://integrate.api.nvidia.com/v1",
        default_models=(
            "meta/llama-3.3-70b-instruct",
            "moonshotai/kimi-k2.5",
            "qwen/qwen2.5-vl-72b-instruct",
        ),
        model_placeholder="Type model name (e.g. meta/llama-3.3-70b-instruct) and press Add",
        description="NVIDIA NIM OpenAI-compatible inference endpoints.",
    ),
    ProviderPreset(
        id="moonshot",
        label="Moonshot (Kimi)",
        kind="openai_compat",
        env_keys=("MOONSHOT_API_KEY",),
        api_key_env=("MOONSHOT_API_KEY",),
        base_url_default="https://api.moonshot.ai/v1",
        default_models=("kimi-k2.5", "moonshot-v1-auto"),
        model_placeholder="Type model name (e.g. kimi-k2.5) and press Add",
        description="Moonshot AI / Kimi OpenAI-compatible API.",
    ),
    ProviderPreset(
        id="groq",
        label="Groq",
        kind="openai_compat",
        env_keys=("GROQ_API_KEY",),
        api_key_env=("GROQ_API_KEY",),
        base_url_default="https://api.groq.com/openai/v1",
        default_models=(
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        model_placeholder="Type model name (e.g. llama-3.3-70b-versatile) and press Add",
        description="Groq fast OpenAI-compatible inference API.",
    ),
    ProviderPreset(
        id="xai",
        label="xAI (Grok)",
        kind="openai_compat",
        env_keys=("XAI_API_KEY",),
        api_key_env=("XAI_API_KEY",),
        base_url_default="https://api.x.ai/v1",
        default_models=("grok-3-mini", "grok-3"),
        model_placeholder="Type model name (e.g. grok-3-mini) and press Add",
        description="xAI Grok OpenAI-compatible API.",
    ),
    ProviderPreset(
        id="dashscope",
        label="Alibaba DashScope (Qwen)",
        kind="openai_compat",
        env_keys=("DASHSCOPE_API_KEY",),
        api_key_env=("DASHSCOPE_API_KEY",),
        base_url_default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_models=("qwen-vl-max", "qwen-plus"),
        model_placeholder="Type model name (e.g. qwen-vl-max) and press Add",
        description="DashScope OpenAI-compatible endpoint for Qwen models.",
    ),
    ProviderPreset(
        id="zhipu",
        label="Zhipu AI (GLM)",
        kind="openai_compat",
        env_keys=("ZAI_API_KEY", "ZHIPUAI_API_KEY"),
        api_key_env=("ZAI_API_KEY",),
        base_url_default="https://api.z.ai/api/paas/v4",
        default_models=("glm-4.5", "glm-4.5-air"),
        model_placeholder="Type model name (e.g. glm-4.5) and press Add",
        description="Zhipu / Z.ai OpenAI-compatible endpoint.",
    ),
    ProviderPreset(
        id="vllm",
        label="vLLM (Local / Self-hosted)",
        kind="openai_compat",
        env_keys=("HOSTED_VLLM_API_KEY",),
        api_key_env=("HOSTED_VLLM_API_KEY",),
        base_url_default="http://localhost:8000/v1",
        default_models=(),
        model_placeholder="Type served model name and press Add",
        description="Local or remote vLLM OpenAI-compatible server.",
    ),
    ProviderPreset(
        id="lmstudio",
        label="LM Studio (Local)",
        kind="openai_compat",
        env_keys=("LMSTUDIO_API_KEY",),
        api_key_env=("LMSTUDIO_API_KEY",),
        base_url_default="http://localhost:1234/v1",
        default_models=(),
        model_placeholder="Type loaded model name and press Add",
        description="Local LM Studio OpenAI-compatible server.",
    ),
    ProviderPreset(
        id="custom",
        label="Custom OpenAI-Compatible",
        kind="openai_compat",
        env_keys=(),
        api_key_env=(),
        base_url_default="https://api.example.com/v1",
        default_models=(),
        model_placeholder="Type model name and press Add",
        description="Any OpenAI-compatible API endpoint.",
    ),
)


@dataclass(frozen=True)
class ProviderCredentialStatus:
    """Non-secret diagnostic view of how a provider authenticates."""

    provider: str
    kind: str
    configured: bool
    source: str
    env_var: str = ""
    masked_key: str = ""
    persisted_in_dotenv: bool = False
    summary: str = ""


def mask_secret_preview(value: str) -> str:
    """Return a non-recoverable masked key suffix (e.g. '…1a2b') for UI display."""
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 4:
        return "••••"
    return f"••••{text[-4:]}"


def extract_ollama_model_names(response: Any) -> List[str]:
    """Normalize Ollama list() or /api/tags responses into unique model names."""
    if isinstance(response, Mapping):
        raw_models = response.get("models")
    else:
        raw_models = getattr(response, "models", None)

    if not isinstance(raw_models, list):
        return []

    names: List[str] = []
    seen: set[str] = set()
    for item in raw_models:
        if isinstance(item, Mapping):
            name = str(item.get("name") or item.get("model") or "").strip()
        else:
            name = str(
                getattr(item, "name", None) or getattr(item, "model", None) or ""
            ).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def is_loopback_base_url(url: str) -> bool:
    """Return True when `url` targets a loopback host (`localhost`, `127.0.0.1`, `::1`)."""
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlsplit(raw)
    except Exception:
        return False
    host = str(parsed.hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def validate_http_endpoint_url(url: str) -> Tuple[str, str]:
    """Validate an http(s) provider endpoint URL, returning `(clean_url, error)`."""
    clean = str(url or "").strip()
    if not clean:
        return "", "Endpoint URL is empty."
    try:
        parsed = urlsplit(clean)
    except Exception:
        return "", f"Invalid endpoint URL: {clean!r}"
    scheme = str(parsed.scheme or "").strip().lower()
    if scheme not in _ALLOWED_HTTP_SCHEMES:
        return "", f"Unsupported URL scheme {scheme or '(none)'!r}; use http or https."
    if not str(parsed.netloc or "").strip():
        return "", "Endpoint URL must include a hostname."
    return clean.rstrip("/"), ""


def _read_bounded_json(
    request: urllib.request.Request,
    *,
    timeout_s: float,
) -> Tuple[Optional[Any], str]:
    try:
        # Use our hardened redirect opener in production while still honoring
        # monkeypatches on urllib.request.urlopen in tests.
        opener = (
            _SAFE_PROBE_OPENER.open
            if urllib.request.urlopen is _DEFAULT_STDLIB_URLOPEN
            else urllib.request.urlopen
        )
        with opener(request, timeout=timeout_s) as resp:  # noqa: S310
            raw = resp.read(_MAX_PROBE_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_PROBE_RESPONSE_BYTES:
                return None, "Provider model list response exceeded size limit."
            return json.loads(raw.decode("utf-8", errors="replace")), ""
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read(512).decode("utf-8", errors="replace").strip()
        except Exception:
            body_text = ""
        detail = f"HTTP {exc.code}"
        if body_text:
            detail = f"{detail}: {body_text[:240]}"
        return None, detail
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _sanitize_error_text(text: str) -> str:
    try:
        from annolid.core.agent.providers.call_runtime import sanitize_provider_error

        return sanitize_provider_error(text, limit=500)
    except Exception:
        return str(text or "")[:500]


def inspect_provider_credential_status(
    settings: Mapping[str, Any],
    provider: str,
    *,
    dotenv_values: Optional[Mapping[str, str]] = None,
    session_key_override: Optional[str] = None,
) -> ProviderCredentialStatus:
    """Inspect where credentials for `provider` come from without exposing secrets."""
    from annolid.utils.llm_settings import (
        detect_openai_codex_auth_state,
        provider_definitions,
        provider_kind,
    )

    provider_key = str(provider or "").strip().lower()
    settings_dict = dict(settings or {})
    kind = provider_kind(settings_dict, provider_key)
    defs = provider_definitions(settings_dict)
    spec = defs.get(provider_key, {})
    block = settings_dict.get(provider_key)
    block_dict = dict(block) if isinstance(block, Mapping) else {}
    dotenv_map = dict(dotenv_values or {})

    if kind == "ollama":
        host = str(
            block_dict.get("host")
            or os.getenv("OLLAMA_HOST")
            or spec.get("host_default")
            or "http://localhost:11434"
        ).strip()
        return ProviderCredentialStatus(
            provider=provider_key,
            kind=kind,
            configured=True,
            source="local",
            summary=f"Local Ollama server ({host})",
        )

    if kind == "openai_codex":
        auth = block_dict.get("auth")
        if not isinstance(auth, Mapping) or "authenticated" not in auth:
            auth = detect_openai_codex_auth_state()
        authenticated = bool(auth.get("authenticated"))
        suffix = str(auth.get("account_id_suffix") or "").strip()
        if authenticated:
            detail = f" (account …{suffix})" if suffix else ""
            summary = f"Local Codex OAuth active{detail}"
        else:
            err = str(auth.get("error") or "").strip()
            summary = (
                f"Codex OAuth not detected ({err})"
                if err
                else "Codex OAuth not detected yet"
            )
        return ProviderCredentialStatus(
            provider=provider_key,
            kind=kind,
            configured=authenticated,
            source="oauth" if authenticated else "missing",
            summary=summary,
        )

    if kind == "codex_cli":
        cli_path = str(block_dict.get("cli_path") or "codex").strip() or "codex"
        resolved = shutil.which(cli_path)
        configured = bool(resolved)
        return ProviderCredentialStatus(
            provider=provider_key,
            kind=kind,
            configured=configured,
            source="cli" if configured else "missing",
            summary=(
                f"Codex CLI found at {resolved}"
                if resolved
                else f"Codex CLI executable '{cli_path}' not found on PATH"
            ),
        )

    candidate_envs: List[str] = []
    for field_name in ("api_key_env", "env_keys"):
        raw = spec.get(field_name) or []
        if isinstance(raw, str):
            raw = [raw]
        for env_name in raw:
            clean = str(env_name or "").strip()
            if clean and clean not in candidate_envs:
                candidate_envs.append(clean)
    if kind == "anthropic" and "ANTHROPIC_API_KEY" not in candidate_envs:
        candidate_envs.append("ANTHROPIC_API_KEY")
    elif kind == "gemini":
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name not in candidate_envs:
                candidate_envs.append(name)

    persisted_env = next(
        (name for name in candidate_envs if str(dotenv_map.get(name) or "").strip()),
        "",
    )
    session_val = (
        str(session_key_override).strip()
        if session_key_override is not None
        else str(block_dict.get("api_key") or "").strip()
    )
    if session_val:
        suffix = mask_secret_preview(session_val)
        extra = (
            f" (also saved in ~/.annolid/.env as {persisted_env})"
            if persisted_env
            else ""
        )
        return ProviderCredentialStatus(
            provider=provider_key,
            kind=kind,
            configured=True,
            source="session",
            env_var=candidate_envs[0] if candidate_envs else "",
            masked_key=suffix,
            persisted_in_dotenv=bool(persisted_env),
            summary=f"Session API key set ({suffix}){extra}",
        )

    for env_name in candidate_envs:
        env_val = str(os.getenv(env_name) or "").strip()
        if env_val:
            suffix = mask_secret_preview(env_val)
            from_dotenv = bool(str(dotenv_map.get(env_name) or "").strip())
            source = "dotenv" if from_dotenv else "env"
            location = "~/.annolid/.env" if from_dotenv else "environment"
            return ProviderCredentialStatus(
                provider=provider_key,
                kind=kind,
                configured=True,
                source=source,
                env_var=env_name,
                masked_key=suffix,
                persisted_in_dotenv=from_dotenv,
                summary=f"Using {env_name} from {location} ({suffix})",
            )

    if provider_key == "openai":
        base_url = str(block_dict.get("base_url") or "").lower()
        if "openrouter" in base_url:
            or_val = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
            if or_val:
                suffix = mask_secret_preview(or_val)
                return ProviderCredentialStatus(
                    provider=provider_key,
                    kind=kind,
                    configured=True,
                    source="env",
                    env_var="OPENROUTER_API_KEY",
                    masked_key=suffix,
                    persisted_in_dotenv=bool(dotenv_map.get("OPENROUTER_API_KEY")),
                    summary=f"Using OPENROUTER_API_KEY ({suffix})",
                )

    base_url = str(
        block_dict.get("base_url") or spec.get("base_url_default") or ""
    ).strip()
    if kind == "openai_compat" and is_loopback_base_url(base_url):
        return ProviderCredentialStatus(
            provider=provider_key,
            kind=kind,
            configured=True,
            source="local",
            env_var=candidate_envs[0] if candidate_envs else "",
            summary=f"Local OpenAI-compatible server ({base_url}) — API key optional",
        )

    env_hint = f" (env: {', '.join(candidate_envs)})" if candidate_envs else ""
    return ProviderCredentialStatus(
        provider=provider_key,
        kind=kind,
        configured=False,
        source="missing",
        env_var=candidate_envs[0] if candidate_envs else "",
        persisted_in_dotenv=bool(persisted_env),
        summary=f"No API key configured{env_hint}",
    )


def fetch_provider_models_live(
    provider: str,
    *,
    settings: Mapping[str, Any],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    host: Optional[str] = None,
    timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S,
) -> Tuple[List[str], str]:
    """Fetch available models from a live provider endpoint.

    Returns `(models, error_message)`. Safe to call from a background thread.
    """
    from annolid.utils.llm_settings import (
        _inject_env_defaults,
        provider_definitions,
        provider_kind,
    )

    provider_key = str(provider or "").strip().lower()
    settings_dict = dict(settings or {})
    kind = provider_kind(settings_dict, provider_key)
    defs = provider_definitions(settings_dict)
    spec = defs.get(provider_key, {})
    raw_block = settings_dict.get(provider_key)
    block = dict(raw_block) if isinstance(raw_block, Mapping) else {}
    block = _inject_env_defaults(settings_dict, provider_key, block)

    effective_key = str(
        api_key if api_key is not None else block.get("api_key") or ""
    ).strip()
    effective_base = str(
        base_url if base_url is not None else block.get("base_url") or ""
    ).strip()
    effective_host = str(
        host
        if host is not None
        else block.get("host") or spec.get("host_default") or ""
    ).strip()
    timeout = max(1.0, float(timeout_s or _DEFAULT_PROBE_TIMEOUT_S))

    if kind == "ollama":
        target_host = effective_host or "http://localhost:11434"
        clean_host, url_err = validate_http_endpoint_url(target_host)
        if url_err:
            return [], url_err
        try:
            import ollama  # type: ignore

            client_factory = getattr(ollama, "Client", None)
            if callable(client_factory):
                try:
                    client = client_factory(host=clean_host, timeout=timeout)
                except TypeError:
                    client = client_factory(host=clean_host)
                response = client.list()
                return extract_ollama_model_names(response), ""
        except ImportError:
            pass
        except Exception as exc:
            return [], _sanitize_error_text(str(exc))

        req = urllib.request.Request(  # noqa: S310 - scheme validated above
            f"{clean_host}/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        payload, err = _read_bounded_json(req, timeout_s=timeout)
        if err:
            return [], _sanitize_error_text(err)
        return extract_ollama_model_names(payload), ""

    if kind == "openai_compat":
        target_base = effective_base or str(spec.get("base_url_default") or "").strip()
        clean_base, url_err = validate_http_endpoint_url(target_base)
        if url_err:
            return [], url_err
        if not effective_key and not is_loopback_base_url(clean_base):
            return [], f"API key is required to query {provider_key} models."
        headers = {"Accept": "application/json", "User-Agent": "annolid"}
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"
        models_url = (
            f"{clean_base}/models" if not clean_base.endswith("/models") else clean_base
        )
        req = urllib.request.Request(  # noqa: S310 - scheme validated above
            models_url,
            headers=headers,
            method="GET",
        )
        payload, err = _read_bounded_json(req, timeout_s=timeout)
        if err:
            return [], _sanitize_error_text(err)
        raw_items = (
            payload.get("data")
            if isinstance(payload, Mapping)
            else payload
            if isinstance(payload, list)
            else []
        )
        models: List[str] = []
        seen: set[str] = set()
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, Mapping):
                    model_id = str(item.get("id") or item.get("name") or "").strip()
                elif isinstance(item, str):
                    model_id = item.strip()
                else:
                    model_id = ""
                if model_id and model_id not in seen:
                    seen.add(model_id)
                    models.append(model_id)
        return sorted(models), ""

    if kind == "anthropic":
        if not effective_key:
            return [], "Anthropic API key is required to query available Claude models."
        target_base = (
            effective_base
            or str(spec.get("base_url_default") or "").strip()
            or "https://api.anthropic.com"
        )
        clean_base, url_err = validate_http_endpoint_url(target_base)
        if url_err:
            return [], url_err
        models_url = (
            f"{clean_base}/models"
            if clean_base.endswith("/v1")
            else f"{clean_base}/v1/models"
        )
        req = urllib.request.Request(  # noqa: S310 - scheme validated above
            models_url,
            headers={
                "Accept": "application/json",
                "x-api-key": effective_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": "annolid",
            },
            method="GET",
        )
        payload, err = _read_bounded_json(req, timeout_s=timeout)
        if err:
            return [], _sanitize_error_text(err)
        models = []
        seen = set()
        for item in (
            list((payload or {}).get("data") or [])
            if isinstance(payload, Mapping)
            else []
        ):
            if isinstance(item, Mapping):
                mid = str(item.get("id") or "").strip()
                if mid and mid not in seen:
                    seen.add(mid)
                    models.append(mid)
        return models, ""

    if kind == "gemini":
        if not effective_key:
            return [], "Gemini API key is required to query available Gemini models."
        req = urllib.request.Request(  # noqa: S310 - fixed https endpoint
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=100",
            headers={
                "Accept": "application/json",
                "x-goog-api-key": effective_key,
                "User-Agent": "annolid",
            },
            method="GET",
        )
        payload, err = _read_bounded_json(req, timeout_s=timeout)
        if err:
            return [], _sanitize_error_text(err)
        models = []
        seen = set()
        for item in (
            list((payload or {}).get("models") or [])
            if isinstance(payload, Mapping)
            else []
        ):
            if not isinstance(item, Mapping):
                continue
            methods = list(item.get("supportedGenerationMethods") or [])
            if methods and "generateContent" not in methods:
                continue
            raw_name = str(item.get("name") or "").strip()
            if raw_name.startswith("models/"):
                raw_name = raw_name[len("models/") :]
            if raw_name and raw_name not in seen:
                seen.add(raw_name)
                models.append(raw_name)
        return models, ""

    if kind in {"openai_codex", "codex_cli"}:
        preferred = list(block.get("preferred_models") or [])
        return [str(m).strip() for m in preferred if str(m).strip()], ""

    return [], f"Unsupported provider kind: {kind}"


def test_provider_connection(
    provider: str,
    *,
    settings: Mapping[str, Any],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    host: Optional[str] = None,
    timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S,
) -> Dict[str, Any]:
    """Run a fast, non-destructive connectivity/auth probe for a provider."""
    from annolid.utils.llm_settings import (
        detect_openai_codex_auth_state,
        provider_kind,
    )

    started = time.perf_counter()
    provider_key = str(provider or "").strip().lower()
    settings_dict = dict(settings or {})
    kind = provider_kind(settings_dict, provider_key)

    if kind == "openai_codex":
        state = detect_openai_codex_auth_state()
        ok = bool(state.get("authenticated"))
        suffix = str(state.get("account_id_suffix") or "").strip()
        msg = (
            f"Codex OAuth verified (account …{suffix})."
            if ok and suffix
            else "Codex OAuth verified."
            if ok
            else f"Codex OAuth not authenticated: {state.get('error') or 'no token found'}."
        )
        return {
            "ok": ok,
            "provider": provider_key,
            "kind": kind,
            "message": msg,
            "models": [],
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "auth": state,
        }

    if kind == "codex_cli":
        block = settings_dict.get(provider_key)
        cli_path = (
            str((block or {}).get("cli_path") or "codex").strip()
            if isinstance(block, Mapping)
            else "codex"
        ) or "codex"
        resolved = shutil.which(cli_path)
        ok = bool(resolved)
        return {
            "ok": ok,
            "provider": provider_key,
            "kind": kind,
            "message": (
                f"Codex CLI executable ready at {resolved}."
                if resolved
                else f"Codex CLI executable '{cli_path}' was not found on PATH."
            ),
            "models": [],
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }

    models, err = fetch_provider_models_live(
        provider_key,
        settings=settings_dict,
        api_key=api_key,
        base_url=base_url,
        host=host,
        timeout_s=timeout_s,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if err:
        return {
            "ok": False,
            "provider": provider_key,
            "kind": kind,
            "message": f"Connection failed: {err}",
            "models": [],
            "latency_ms": elapsed_ms,
        }
    return {
        "ok": True,
        "provider": provider_key,
        "kind": kind,
        "message": (
            f"Connected successfully ({len(models)} model(s) available, {elapsed_ms:.0f} ms)."
        ),
        "models": models,
        "latency_ms": elapsed_ms,
    }


test_provider_connection.__test__ = False  # type: ignore[attr-defined]


def get_provider_presets() -> Tuple[ProviderPreset, ...]:
    return PROVIDER_PRESETS


def find_provider_preset(preset_id: str) -> Optional[ProviderPreset]:
    key = str(preset_id or "").strip().lower()
    for preset in PROVIDER_PRESETS:
        if preset.id == key:
            return preset
    return None


__all__ = [
    "PROVIDER_PRESETS",
    "ProviderCredentialStatus",
    "ProviderPreset",
    "extract_ollama_model_names",
    "fetch_provider_models_live",
    "find_provider_preset",
    "get_provider_presets",
    "inspect_provider_credential_status",
    "is_loopback_base_url",
    "mask_secret_preview",
    "test_provider_connection",
    "validate_http_endpoint_url",
]
