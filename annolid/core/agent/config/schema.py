from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from annolid.core.agent.providers.registry import PROVIDERS


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

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentDefaults":
        payload = data or {}
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
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_tool_iterations": self.max_tool_iterations,
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
class ToolsConfig:
    exec: ExecToolConfig = field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ToolsConfig":
        payload = data or {}
        restrict_value = payload.get(
            "restrict_to_workspace",
            payload.get("restrictToWorkspace", False),
        )
        exec_cfg = ExecToolConfig.from_dict(payload.get("exec"))
        return cls(exec=exec_cfg, restrict_to_workspace=bool(restrict_value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exec": self.exec.to_dict(),
            "restrict_to_workspace": self.restrict_to_workspace,
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
