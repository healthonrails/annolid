from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from annolid.core.agent.providers.registry import PROVIDERS


@dataclass
class SessionRoutingConfig:
    dm_scope: str = "main"
    main_session_key: str = "main"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SessionRoutingConfig":
        payload = data or {}
        dm_scope = str(payload.get("dm_scope", payload.get("dmScope", "main"))).strip()
        if not dm_scope:
            dm_scope = "main"
        main_key = str(
            payload.get(
                "main_session_key",
                payload.get("mainSessionKey", payload.get("main_key", "main")),
            )
        ).strip()
        if not main_key:
            main_key = "main"
        return cls(dm_scope=dm_scope, main_session_key=main_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dm_scope": self.dm_scope,
            "main_session_key": self.main_session_key,
        }


@dataclass
class ProviderConfig:
    api_key: str = ""
    api_base: str = ""
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProviderConfig":
        payload = data or {}
        headers = payload.get("extra_headers") or payload.get("extraHeaders") or {}
        if not isinstance(headers, dict):
            headers = {}
        return cls(
            api_key=str(payload.get("api_key") or payload.get("apiKey") or ""),
            api_base=str(payload.get("api_base") or payload.get("apiBase") or ""),
            extra_headers={str(k): str(v) for k, v in headers.items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_base": self.api_base,
            "extra_headers": dict(self.extra_headers),
        }


@dataclass
class AgentDefaults:
    workspace: str = "~/.annolid/workspace"
    model: str = "qwen3-vl"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 12
    memory_window: int = 50
    session: SessionRoutingConfig = field(default_factory=SessionRoutingConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentDefaults":
        payload = data or {}
        session_payload = payload.get("session")
        if not isinstance(session_payload, dict):
            session_payload = {
                "dm_scope": payload.get(
                    "session_dm_scope",
                    payload.get("sessionDmScope"),
                ),
                "main_session_key": payload.get(
                    "main_session_key",
                    payload.get("mainSessionKey", payload.get("session_main_key")),
                ),
            }
        return cls(
            workspace=str(payload.get("workspace") or "~/.annolid/workspace"),
            model=str(payload.get("model") or "qwen3-vl"),
            max_tokens=int(payload.get("max_tokens", payload.get("maxTokens", 8192))),
            temperature=float(payload.get("temperature", 0.7)),
            max_tool_iterations=int(
                payload.get(
                    "max_tool_iterations",
                    payload.get("maxToolIterations", payload.get("max_iterations", 12)),
                )
            ),
            memory_window=int(
                payload.get(
                    "memory_window",
                    payload.get("memoryWindow", 50),
                )
            ),
            session=SessionRoutingConfig.from_dict(session_payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_tool_iterations": self.max_tool_iterations,
            "memory_window": self.memory_window,
            "session": self.session.to_dict(),
        }


@dataclass
class AgentsConfig:
    defaults: AgentDefaults = field(default_factory=AgentDefaults)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentsConfig":
        payload = data or {}
        return cls(defaults=AgentDefaults.from_dict(payload.get("defaults")))

    def to_dict(self) -> Dict[str, Any]:
        return {"defaults": self.defaults.to_dict()}


@dataclass
class ExecToolConfig:
    timeout: int = 60

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExecToolConfig":
        payload = data or {}
        return cls(timeout=int(payload.get("timeout", 60)))

    def to_dict(self) -> Dict[str, Any]:
        return {"timeout": self.timeout}


@dataclass
class ToolPolicyConfig:
    profile: str = ""
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ToolPolicyConfig":
        payload = data or {}
        allow_raw = payload.get("allow", [])
        deny_raw = payload.get("deny", [])
        allow = (
            [str(item).strip() for item in allow_raw if str(item).strip()]
            if isinstance(allow_raw, (list, tuple))
            else []
        )
        deny = (
            [str(item).strip() for item in deny_raw if str(item).strip()]
            if isinstance(deny_raw, (list, tuple))
            else []
        )
        return cls(
            profile=str(payload.get("profile") or "").strip().lower(),
            allow=allow,
            deny=deny,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "allow": list(self.allow),
            "deny": list(self.deny),
        }


@dataclass
class ToolsConfig:
    exec: ExecToolConfig = field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    allowed_read_roots: list[str] = field(default_factory=list)
    profile: str = "full"
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    by_provider: Dict[str, ToolPolicyConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ToolsConfig":
        payload = data or {}
        restrict_value = payload.get(
            "restrict_to_workspace",
            payload.get("restrictToWorkspace", False),
        )
        roots_raw = payload.get(
            "allowed_read_roots", payload.get("allowedReadRoots", [])
        )
        if isinstance(roots_raw, (list, tuple)):
            allowed_read_roots = [
                str(item).strip() for item in roots_raw if str(item).strip()
            ]
        else:
            allowed_read_roots = []
        allow_raw = payload.get("allow", [])
        deny_raw = payload.get("deny", [])
        allow = (
            [str(item).strip() for item in allow_raw if str(item).strip()]
            if isinstance(allow_raw, (list, tuple))
            else []
        )
        deny = (
            [str(item).strip() for item in deny_raw if str(item).strip()]
            if isinstance(deny_raw, (list, tuple))
            else []
        )
        by_provider_raw = payload.get("by_provider", payload.get("byProvider", {}))
        by_provider: Dict[str, ToolPolicyConfig] = {}
        if isinstance(by_provider_raw, dict):
            for key, value in by_provider_raw.items():
                by_provider[str(key)] = ToolPolicyConfig.from_dict(
                    value if isinstance(value, dict) else None
                )
        exec_cfg = ExecToolConfig.from_dict(payload.get("exec"))
        return cls(
            exec=exec_cfg,
            restrict_to_workspace=bool(restrict_value),
            allowed_read_roots=allowed_read_roots,
            profile=str(payload.get("profile") or "full").strip().lower() or "full",
            allow=allow,
            deny=deny,
            by_provider=by_provider,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exec": self.exec.to_dict(),
            "restrict_to_workspace": self.restrict_to_workspace,
            "allowed_read_roots": list(self.allowed_read_roots),
            "profile": self.profile,
            "allow": list(self.allow),
            "deny": list(self.deny),
            "by_provider": {
                key: value.to_dict() for key, value in self.by_provider.items()
            },
        }


@dataclass
class AgentConfig:
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        return Path(self.agents.defaults.workspace).expanduser()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentConfig":
        payload = data or {}
        provider_data = payload.get("providers") or {}
        providers: Dict[str, ProviderConfig] = {}
        if isinstance(provider_data, dict):
            for name, value in provider_data.items():
                providers[str(name)] = ProviderConfig.from_dict(
                    value if isinstance(value, dict) else None
                )
        return cls(
            agents=AgentsConfig.from_dict(payload.get("agents")),
            providers=providers,
            tools=ToolsConfig.from_dict(payload.get("tools")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": self.agents.to_dict(),
            "providers": {name: cfg.to_dict() for name, cfg in self.providers.items()},
            "tools": self.tools.to_dict(),
        }

    def _provider_candidates(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in PROVIDERS)

    def get_provider_name(self, model: Optional[str] = None) -> Optional[str]:
        model_lower = str(model or self.agents.defaults.model or "").lower()
        for spec in PROVIDERS:
            cfg = self.providers.get(spec.name)
            if not cfg or not cfg.api_key:
                continue
            if spec.is_gateway:
                continue
            if any(keyword in model_lower for keyword in spec.keywords):
                return spec.name
        for spec in PROVIDERS:
            cfg = self.providers.get(spec.name)
            if cfg and cfg.api_key:
                return spec.name
        return None

    def get_provider(self, model: Optional[str] = None) -> Optional[ProviderConfig]:
        name = self.get_provider_name(model=model)
        if name is None:
            return None
        return self.providers.get(name)
