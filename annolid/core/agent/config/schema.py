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
class EmailChannelConfig:
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    user: str = ""
    password: str = ""
    polling_interval: int = 300
    allow_from: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EmailChannelConfig":
        payload = data or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            smtp_host=str(
                payload.get("smtp_host") or payload.get("smtpHost", "smtp.gmail.com")
            ),
            smtp_port=int(payload.get("smtp_port") or payload.get("smtpPort", 587)),
            imap_host=str(
                payload.get("imap_host") or payload.get("imapHost", "imap.gmail.com")
            ),
            imap_port=int(payload.get("imap_port") or payload.get("imapPort", 993)),
            user=str(payload.get("user") or ""),
            password=str(payload.get("password") or ""),
            polling_interval=int(
                payload.get("polling_interval") or payload.get("pollingInterval", 300)
            ),
            allow_from=list(payload.get("allow_from") or payload.get("allowFrom", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
            "user": self.user,
            "password": self.password,
            "polling_interval": self.polling_interval,
            "allow_from": list(self.allow_from),
        }


@dataclass
class WhatsAppChannelConfig:
    enabled: bool = False
    auto_start: bool = True
    bridge_mode: str = "python"
    bridge_url: str = ""
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 3001
    bridge_token: str = ""
    bridge_session_dir: str = "~/.annolid/whatsapp-web-session"
    bridge_headless: bool = False
    access_token: str = ""
    phone_number_id: str = ""
    verify_token: str = ""
    api_version: str = "v22.0"
    api_base: str = "https://graph.facebook.com"
    preview_url: bool = False
    webhook_enabled: bool = False
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 0
    webhook_path: str = "/whatsapp/webhook"
    ingest_outgoing_messages: bool = False
    allow_from: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WhatsAppChannelConfig":
        payload = data or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            auto_start=bool(payload.get("auto_start", payload.get("autoStart", True))),
            bridge_mode=str(
                payload.get("bridge_mode") or payload.get("bridgeMode") or "python"
            ),
            bridge_url=str(payload.get("bridge_url") or payload.get("bridgeUrl") or ""),
            bridge_host=str(
                payload.get("bridge_host") or payload.get("bridgeHost") or "127.0.0.1"
            ),
            bridge_port=int(
                payload.get("bridge_port", payload.get("bridgePort", 3001)) or 3001
            ),
            bridge_token=str(
                payload.get("bridge_token") or payload.get("bridgeToken") or ""
            ),
            bridge_session_dir=str(
                payload.get("bridge_session_dir")
                or payload.get("bridgeSessionDir")
                or "~/.annolid/whatsapp-web-session"
            ),
            bridge_headless=bool(
                payload.get("bridge_headless", payload.get("bridgeHeadless", False))
            ),
            access_token=str(
                payload.get("access_token") or payload.get("accessToken") or ""
            ),
            phone_number_id=str(
                payload.get("phone_number_id") or payload.get("phoneNumberId") or ""
            ),
            verify_token=str(
                payload.get("verify_token") or payload.get("verifyToken") or ""
            ),
            api_version=str(
                payload.get("api_version") or payload.get("apiVersion") or "v22.0"
            ),
            api_base=str(
                payload.get("api_base")
                or payload.get("apiBase")
                or "https://graph.facebook.com"
            ),
            preview_url=bool(
                payload.get("preview_url", payload.get("previewUrl", False))
            ),
            webhook_enabled=bool(
                payload.get("webhook_enabled", payload.get("webhookEnabled", False))
            ),
            webhook_host=str(
                payload.get("webhook_host") or payload.get("webhookHost") or "127.0.0.1"
            ),
            webhook_port=int(
                payload.get("webhook_port", payload.get("webhookPort", 0)) or 0
            ),
            webhook_path=str(
                payload.get("webhook_path")
                or payload.get("webhookPath")
                or "/whatsapp/webhook"
            ),
            ingest_outgoing_messages=bool(
                payload.get(
                    "ingest_outgoing_messages",
                    payload.get("ingestOutgoingMessages", False),
                )
            ),
            allow_from=list(payload.get("allow_from") or payload.get("allowFrom", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_start": self.auto_start,
            "bridge_mode": self.bridge_mode,
            "bridge_url": self.bridge_url,
            "bridge_host": self.bridge_host,
            "bridge_port": self.bridge_port,
            "bridge_token": self.bridge_token,
            "bridge_session_dir": self.bridge_session_dir,
            "bridge_headless": self.bridge_headless,
            "access_token": self.access_token,
            "phone_number_id": self.phone_number_id,
            "verify_token": self.verify_token,
            "api_version": self.api_version,
            "api_base": self.api_base,
            "preview_url": self.preview_url,
            "webhook_enabled": self.webhook_enabled,
            "webhook_host": self.webhook_host,
            "webhook_port": self.webhook_port,
            "webhook_path": self.webhook_path,
            "ingest_outgoing_messages": self.ingest_outgoing_messages,
            "allow_from": list(self.allow_from),
        }


@dataclass
class CalendarToolConfig:
    enabled: bool = False
    provider: str = "google"
    credentials_file: str = "~/.annolid/agent/google_calendar_credentials.json"
    token_file: str = "~/.annolid/agent/google_calendar_token.json"
    calendar_id: str = "primary"
    timezone: str = ""
    default_event_duration_minutes: int = 30

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CalendarToolConfig":
        payload = data or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            provider=str(payload.get("provider", "google") or "google"),
            credentials_file=str(
                payload.get("credentials_file")
                or payload.get("credentialsFile")
                or "~/.annolid/agent/google_calendar_credentials.json"
            ),
            token_file=str(
                payload.get("token_file")
                or payload.get("tokenFile")
                or "~/.annolid/agent/google_calendar_token.json"
            ),
            calendar_id=str(
                payload.get("calendar_id") or payload.get("calendarId") or "primary"
            ),
            timezone=str(payload.get("timezone", "") or ""),
            default_event_duration_minutes=int(
                payload.get(
                    "default_event_duration_minutes",
                    payload.get("defaultEventDurationMinutes", 30),
                )
                or 30
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "credentials_file": self.credentials_file,
            "token_file": self.token_file,
            "calendar_id": self.calendar_id,
            "timezone": self.timezone,
            "default_event_duration_minutes": self.default_event_duration_minutes,
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
    max_parallel_sessions: int = 1
    max_pending_messages: int = 2048
    collapse_superseded_pending: bool = True
    transient_retry_attempts: int = 2
    transient_retry_initial_backoff_s: float = 0.5
    transient_retry_max_backoff_s: float = 4.0
    strict_runtime_tool_guard: bool = True

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
            max_parallel_sessions=max(
                1,
                int(
                    payload.get(
                        "max_parallel_sessions",
                        payload.get("maxParallelSessions", 1),
                    )
                ),
            ),
            max_pending_messages=max(
                1,
                int(
                    payload.get(
                        "max_pending_messages",
                        payload.get("maxPendingMessages", 2048),
                    )
                ),
            ),
            collapse_superseded_pending=bool(
                payload.get(
                    "collapse_superseded_pending",
                    payload.get("collapseSupersededPending", True),
                )
            ),
            transient_retry_attempts=max(
                0,
                int(
                    payload.get(
                        "transient_retry_attempts",
                        payload.get("transientRetryAttempts", 2),
                    )
                ),
            ),
            transient_retry_initial_backoff_s=max(
                0.0,
                float(
                    payload.get(
                        "transient_retry_initial_backoff_s",
                        payload.get("transientRetryInitialBackoffS", 0.5),
                    )
                ),
            ),
            transient_retry_max_backoff_s=max(
                0.0,
                float(
                    payload.get(
                        "transient_retry_max_backoff_s",
                        payload.get("transientRetryMaxBackoffS", 4.0),
                    )
                ),
            ),
            strict_runtime_tool_guard=bool(
                payload.get(
                    "strict_runtime_tool_guard",
                    payload.get("strictRuntimeToolGuard", True),
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
            "memory_window": self.memory_window,
            "session": self.session.to_dict(),
            "max_parallel_sessions": self.max_parallel_sessions,
            "max_pending_messages": self.max_pending_messages,
            "collapse_superseded_pending": self.collapse_superseded_pending,
            "transient_retry_attempts": self.transient_retry_attempts,
            "transient_retry_initial_backoff_s": self.transient_retry_initial_backoff_s,
            "transient_retry_max_backoff_s": self.transient_retry_max_backoff_s,
            "strict_runtime_tool_guard": self.strict_runtime_tool_guard,
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
        return cls(timeout=int(payload.get("timeout", 180)))

    def to_dict(self) -> Dict[str, Any]:
        return {"timeout": self.timeout}


@dataclass
class MCPServerConfig:
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MCPServerConfig":
        payload = data or {}
        return cls(
            command=str(payload.get("command", "")),
            args=list(payload.get("args", [])),
            env=dict(payload.get("env", {})),
            url=str(payload.get("url", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "url": self.url,
        }


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
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    email: EmailChannelConfig = field(default_factory=EmailChannelConfig)
    whatsapp: WhatsAppChannelConfig = field(default_factory=WhatsAppChannelConfig)
    calendar: CalendarToolConfig = field(default_factory=CalendarToolConfig)
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
        mcp_raw = payload.get("mcp_servers", payload.get("mcpServers", {}))
        mcp_servers: Dict[str, MCPServerConfig] = {}
        if isinstance(mcp_raw, dict):
            for name, value in mcp_raw.items():
                mcp_servers[str(name)] = MCPServerConfig.from_dict(
                    value if isinstance(value, dict) else None
                )
        by_provider_raw = payload.get("by_provider", payload.get("byProvider", {}))
        by_provider: Dict[str, ToolPolicyConfig] = {}
        if isinstance(by_provider_raw, dict):
            for key, value in by_provider_raw.items():
                by_provider[str(key)] = ToolPolicyConfig.from_dict(
                    value if isinstance(value, dict) else None
                )
        email_cfg = EmailChannelConfig.from_dict(payload.get("email"))
        whatsapp_cfg = WhatsAppChannelConfig.from_dict(payload.get("whatsapp"))
        calendar_cfg = CalendarToolConfig.from_dict(payload.get("calendar"))
        exec_cfg = ExecToolConfig.from_dict(payload.get("exec"))
        return cls(
            exec=exec_cfg,
            restrict_to_workspace=bool(restrict_value),
            allowed_read_roots=allowed_read_roots,
            profile=str(payload.get("profile") or "full").strip().lower() or "full",
            allow=allow,
            deny=deny,
            mcp_servers=mcp_servers,
            email=email_cfg,
            whatsapp=whatsapp_cfg,
            calendar=calendar_cfg,
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
            "email": self.email.to_dict(),
            "whatsapp": self.whatsapp.to_dict(),
            "calendar": self.calendar.to_dict(),
            "mcp_servers": {
                name: cfg.to_dict() for name, cfg in self.mcp_servers.items()
            },
            "by_provider": {
                key: value.to_dict() for key, value in self.by_provider.items()
            },
        }


@dataclass
class SkillsLoadConfig:
    watch: bool = False
    poll_seconds: float = 1.0
    extra_dirs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SkillsLoadConfig":
        payload = data or {}
        dirs_raw = payload.get("extra_dirs", payload.get("extraDirs", []))
        if isinstance(dirs_raw, (list, tuple)):
            extra_dirs = [str(item).strip() for item in dirs_raw if str(item).strip()]
        else:
            extra_dirs = []
        poll_seconds = payload.get("poll_seconds", payload.get("pollSeconds", 1.0))
        try:
            poll_value = max(0.0, float(poll_seconds))
        except (TypeError, ValueError):
            poll_value = 1.0
        return cls(
            watch=bool(payload.get("watch", False)),
            poll_seconds=poll_value,
            extra_dirs=extra_dirs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "watch": bool(self.watch),
            "poll_seconds": float(self.poll_seconds),
            "extra_dirs": list(self.extra_dirs),
        }


@dataclass
class SkillsConfig:
    load: SkillsLoadConfig = field(default_factory=SkillsLoadConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SkillsConfig":
        payload = data or {}
        return cls(load=SkillsLoadConfig.from_dict(payload.get("load")))

    def to_dict(self) -> Dict[str, Any]:
        return {"load": self.load.to_dict()}


@dataclass
class MemoryConfig:
    mode: str = "semantic_keyword"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MemoryConfig":
        payload = data or {}
        mode = str(payload.get("mode") or "semantic_keyword").strip().lower()
        if mode not in {"semantic_keyword", "lexical"}:
            mode = "semantic_keyword"
        return cls(mode=mode)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode}


@dataclass
class AutoUpdateConfig:
    enabled: bool = False
    channel: str = "stable"
    interval_seconds: int = 24 * 3600
    jitter_seconds: int = 15 * 60
    timeout_s: float = 4.0
    require_signature: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AutoUpdateConfig":
        payload = data or {}
        channel = str(payload.get("channel") or "stable").strip().lower()
        if channel not in {"stable", "beta", "dev"}:
            channel = "stable"
        try:
            interval_seconds = max(
                300,
                int(
                    payload.get(
                        "interval_seconds", payload.get("intervalSeconds", 86400)
                    )
                ),
            )
        except (TypeError, ValueError):
            interval_seconds = 86400
        try:
            jitter_seconds = max(
                0, int(payload.get("jitter_seconds", payload.get("jitterSeconds", 900)))
            )
        except (TypeError, ValueError):
            jitter_seconds = 900
        try:
            timeout_s = max(
                1.0, float(payload.get("timeout_s", payload.get("timeoutS", 4.0)))
            )
        except (TypeError, ValueError):
            timeout_s = 4.0
        return cls(
            enabled=bool(payload.get("enabled", False)),
            channel=channel,
            interval_seconds=interval_seconds,
            jitter_seconds=jitter_seconds,
            timeout_s=timeout_s,
            require_signature=bool(
                payload.get("require_signature", payload.get("requireSignature", False))
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "channel": self.channel,
            "interval_seconds": int(self.interval_seconds),
            "jitter_seconds": int(self.jitter_seconds),
            "timeout_s": float(self.timeout_s),
            "require_signature": bool(self.require_signature),
        }


@dataclass
class UpdateConfig:
    auto: AutoUpdateConfig = field(default_factory=AutoUpdateConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UpdateConfig":
        payload = data or {}
        auto_payload = payload.get("auto", payload.get("auto_update"))
        return cls(auto=AutoUpdateConfig.from_dict(auto_payload))

    def to_dict(self) -> Dict[str, Any]:
        return {"auto": self.auto.to_dict()}


@dataclass
class AgentConfig:
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)

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
            skills=SkillsConfig.from_dict(payload.get("skills")),
            memory=MemoryConfig.from_dict(payload.get("memory")),
            update=UpdateConfig.from_dict(payload.get("update")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": self.agents.to_dict(),
            "providers": {name: cfg.to_dict() for name, cfg in self.providers.items()},
            "tools": self.tools.to_dict(),
            "skills": self.skills.to_dict(),
            "memory": self.memory.to_dict(),
            "update": self.update.to_dict(),
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
