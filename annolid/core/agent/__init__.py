"""Core (GUI-free) agent orchestration primitives.

Note: this module uses lazy imports so that lightweight subpackages like
`annolid.core.agent.tools` can be imported without pulling in video/ML
dependencies during module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .behavior_engine import (
        BehaviorEngine,
        BehaviorEngineConfig,
        BehaviorEvent,
        BehaviorUpdate,
    )
    from .frame_source import FrameSource
    from .context import AgentContextBuilder
    from .memory import AgentMemoryStore
    from .skills import AgentSkillsLoader
    from .workspace_bootstrap import bootstrap_workspace
    from .heartbeat import (
        DEFAULT_HEARTBEAT_INTERVAL_S,
        HEARTBEAT_OK_TOKEN,
        HEARTBEAT_PROMPT,
        HeartbeatResult,
        HeartbeatService,
        is_heartbeat_empty,
    )
    from .bus import InboundMessage, OutboundMessage, MessageBus, AgentBusService
    from .utils import (
        ensure_dir,
        get_agent_data_path,
        get_agent_workspace_path,
        get_sessions_path,
        get_memory_path,
        get_skills_path,
        today_date,
        timestamp,
        truncate_string,
        safe_filename,
        parse_session_key,
    )
    from .channels import (
        BaseChannel,
        ChannelManager,
        TelegramChannel,
        DiscordChannel,
        SlackChannel,
        EmailChannel,
        WhatsAppChannel,
        WhatsAppPythonBridge,
        WhatsAppWebhookServer,
        markdown_to_telegram_html,
    )
    from .cron import (
        CronService,
        compute_next_run,
        CronJob,
        CronJobState,
        CronPayload,
        CronSchedule,
        CronStore,
    )
    from .session_manager import (
        AgentSession,
        AgentSessionManager,
        PersistentSessionStore,
    )
    from .scheduler import ScheduledTask, TaskScheduler
    from .providers import (
        LLMProvider,
        LLMResponse,
        ToolCallRequest,
        LiteLLMProvider,
        OpenAICompatProvider,
        OpenAICompatResolved,
        OpenAICompatTranscriptionProvider,
        resolve_openai_compat,
    )
    from .config import (
        AgentConfig,
        AgentDefaults,
        AgentsConfig,
        ExecToolConfig,
        ProviderConfig,
        ToolsConfig,
        get_config_path,
        load_config,
        save_config,
    )
    from .subagent import SubagentManager, SubagentTask, build_subagent_tools_registry
    from .orchestrator import AnnolidAgent
    from .pipeline import AgentPipelineConfig
    from .runner import AgentRunConfig, AgentRunner
    from .service import AgentServiceResult, run_agent_to_results
    from .track_store import TrackStore
    from .loop import (
        AgentLoop,
        AgentLoopResult,
        AgentMemoryConfig,
        AgentToolRun,
        InMemorySessionStore,
        SessionStoreProtocol,
    )

__all__ = [
    "BehaviorEngine",
    "BehaviorEngineConfig",
    "BehaviorEvent",
    "BehaviorUpdate",
    "FrameSource",
    "AgentContextBuilder",
    "AgentMemoryStore",
    "AgentSkillsLoader",
    "bootstrap_workspace",
    "DEFAULT_HEARTBEAT_INTERVAL_S",
    "HEARTBEAT_PROMPT",
    "HEARTBEAT_OK_TOKEN",
    "HeartbeatResult",
    "HeartbeatService",
    "is_heartbeat_empty",
    "InboundMessage",
    "OutboundMessage",
    "MessageBus",
    "AgentBusService",
    "ensure_dir",
    "get_agent_data_path",
    "get_agent_workspace_path",
    "get_sessions_path",
    "get_memory_path",
    "get_skills_path",
    "today_date",
    "timestamp",
    "truncate_string",
    "safe_filename",
    "parse_session_key",
    "BaseChannel",
    "ChannelManager",
    "TelegramChannel",
    "DiscordChannel",
    "SlackChannel",
    "EmailChannel",
    "WhatsAppChannel",
    "WhatsAppPythonBridge",
    "WhatsAppWebhookServer",
    "markdown_to_telegram_html",
    "CronService",
    "compute_next_run",
    "CronJob",
    "CronJobState",
    "CronPayload",
    "CronSchedule",
    "CronStore",
    "AgentSession",
    "AgentSessionManager",
    "PersistentSessionStore",
    "ScheduledTask",
    "TaskScheduler",
    "LLMProvider",
    "LLMResponse",
    "ToolCallRequest",
    "LiteLLMProvider",
    "OpenAICompatProvider",
    "OpenAICompatResolved",
    "OpenAICompatTranscriptionProvider",
    "resolve_openai_compat",
    "AgentConfig",
    "ProviderConfig",
    "AgentDefaults",
    "AgentsConfig",
    "ExecToolConfig",
    "ToolsConfig",
    "get_config_path",
    "load_config",
    "save_config",
    "SubagentManager",
    "SubagentTask",
    "build_subagent_tools_registry",
    "AnnolidAgent",
    "AgentPipelineConfig",
    "AgentRunConfig",
    "AgentRunner",
    "AgentServiceResult",
    "run_agent_to_results",
    "TrackStore",
    "AgentLoop",
    "AgentLoopResult",
    "AgentMemoryConfig",
    "AgentToolRun",
    "InMemorySessionStore",
    "SessionStoreProtocol",
]


def __getattr__(name: str):  # noqa: ANN001
    if name in {"AgentPipelineConfig"}:
        from .pipeline import AgentPipelineConfig

        return {"AgentPipelineConfig": AgentPipelineConfig}[name]

    if name in {"AnnolidAgent"}:
        from .orchestrator import AnnolidAgent

        return {"AnnolidAgent": AnnolidAgent}[name]

    if name in {"FrameSource"}:
        from .frame_source import FrameSource

        return {"FrameSource": FrameSource}[name]

    if name in {"AgentContextBuilder"}:
        from .context import AgentContextBuilder

        return {"AgentContextBuilder": AgentContextBuilder}[name]

    if name in {"AgentMemoryStore"}:
        from .memory import AgentMemoryStore

        return {"AgentMemoryStore": AgentMemoryStore}[name]

    if name in {"AgentSkillsLoader"}:
        from .skills import AgentSkillsLoader

        return {"AgentSkillsLoader": AgentSkillsLoader}[name]

    if name in {"bootstrap_workspace"}:
        from .workspace_bootstrap import bootstrap_workspace

        return {"bootstrap_workspace": bootstrap_workspace}[name]

    if name in {
        "DEFAULT_HEARTBEAT_INTERVAL_S",
        "HEARTBEAT_PROMPT",
        "HEARTBEAT_OK_TOKEN",
        "HeartbeatResult",
        "HeartbeatService",
        "is_heartbeat_empty",
    }:
        from .heartbeat import (
            DEFAULT_HEARTBEAT_INTERVAL_S,
            HEARTBEAT_OK_TOKEN,
            HEARTBEAT_PROMPT,
            HeartbeatResult,
            HeartbeatService,
            is_heartbeat_empty,
        )

        return {
            "DEFAULT_HEARTBEAT_INTERVAL_S": DEFAULT_HEARTBEAT_INTERVAL_S,
            "HEARTBEAT_PROMPT": HEARTBEAT_PROMPT,
            "HEARTBEAT_OK_TOKEN": HEARTBEAT_OK_TOKEN,
            "HeartbeatResult": HeartbeatResult,
            "HeartbeatService": HeartbeatService,
            "is_heartbeat_empty": is_heartbeat_empty,
        }[name]

    if name in {"InboundMessage", "OutboundMessage", "MessageBus", "AgentBusService"}:
        from .bus import InboundMessage, OutboundMessage, MessageBus, AgentBusService

        return {
            "InboundMessage": InboundMessage,
            "OutboundMessage": OutboundMessage,
            "MessageBus": MessageBus,
            "AgentBusService": AgentBusService,
        }[name]

    if name in {
        "ensure_dir",
        "get_agent_data_path",
        "get_agent_workspace_path",
        "get_sessions_path",
        "get_memory_path",
        "get_skills_path",
        "today_date",
        "timestamp",
        "truncate_string",
        "safe_filename",
        "parse_session_key",
    }:
        from .utils import (
            ensure_dir,
            get_agent_data_path,
            get_agent_workspace_path,
            get_sessions_path,
            get_memory_path,
            get_skills_path,
            today_date,
            timestamp,
            truncate_string,
            safe_filename,
            parse_session_key,
        )

        return {
            "ensure_dir": ensure_dir,
            "get_agent_data_path": get_agent_data_path,
            "get_agent_workspace_path": get_agent_workspace_path,
            "get_sessions_path": get_sessions_path,
            "get_memory_path": get_memory_path,
            "get_skills_path": get_skills_path,
            "today_date": today_date,
            "timestamp": timestamp,
            "truncate_string": truncate_string,
            "safe_filename": safe_filename,
            "parse_session_key": parse_session_key,
        }[name]

    if name in {
        "BaseChannel",
        "ChannelManager",
        "TelegramChannel",
        "DiscordChannel",
        "SlackChannel",
        "EmailChannel",
        "WhatsAppChannel",
        "WhatsAppPythonBridge",
        "WhatsAppWebhookServer",
        "markdown_to_telegram_html",
    }:
        from .channels import (
            BaseChannel,
            ChannelManager,
            TelegramChannel,
            DiscordChannel,
            SlackChannel,
            EmailChannel,
            WhatsAppChannel,
            WhatsAppPythonBridge,
            WhatsAppWebhookServer,
            markdown_to_telegram_html,
        )

        return {
            "BaseChannel": BaseChannel,
            "ChannelManager": ChannelManager,
            "TelegramChannel": TelegramChannel,
            "DiscordChannel": DiscordChannel,
            "SlackChannel": SlackChannel,
            "EmailChannel": EmailChannel,
            "WhatsAppChannel": WhatsAppChannel,
            "WhatsAppPythonBridge": WhatsAppPythonBridge,
            "WhatsAppWebhookServer": WhatsAppWebhookServer,
            "markdown_to_telegram_html": markdown_to_telegram_html,
        }[name]

    if name in {
        "CronService",
        "compute_next_run",
        "CronJob",
        "CronJobState",
        "CronPayload",
        "CronSchedule",
        "CronStore",
    }:
        from .cron import (
            CronService,
            compute_next_run,
            CronJob,
            CronJobState,
            CronPayload,
            CronSchedule,
            CronStore,
        )

        return {
            "CronService": CronService,
            "compute_next_run": compute_next_run,
            "CronJob": CronJob,
            "CronJobState": CronJobState,
            "CronPayload": CronPayload,
            "CronSchedule": CronSchedule,
            "CronStore": CronStore,
        }[name]

    if name in {"AgentSession", "AgentSessionManager", "PersistentSessionStore"}:
        from .session_manager import (
            AgentSession,
            AgentSessionManager,
            PersistentSessionStore,
        )

        return {
            "AgentSession": AgentSession,
            "AgentSessionManager": AgentSessionManager,
            "PersistentSessionStore": PersistentSessionStore,
        }[name]

    if name in {"ScheduledTask", "TaskScheduler"}:
        from .scheduler import ScheduledTask, TaskScheduler

        return {
            "ScheduledTask": ScheduledTask,
            "TaskScheduler": TaskScheduler,
        }[name]

    if name in {
        "LLMProvider",
        "LLMResponse",
        "ToolCallRequest",
        "LiteLLMProvider",
        "OpenAICompatProvider",
        "OpenAICompatResolved",
        "OpenAICompatTranscriptionProvider",
        "resolve_openai_compat",
    }:
        from .providers import (
            LLMProvider,
            LLMResponse,
            ToolCallRequest,
            LiteLLMProvider,
            OpenAICompatProvider,
            OpenAICompatResolved,
            OpenAICompatTranscriptionProvider,
            resolve_openai_compat,
        )

        return {
            "LLMProvider": LLMProvider,
            "LLMResponse": LLMResponse,
            "ToolCallRequest": ToolCallRequest,
            "LiteLLMProvider": LiteLLMProvider,
            "OpenAICompatProvider": OpenAICompatProvider,
            "OpenAICompatResolved": OpenAICompatResolved,
            "OpenAICompatTranscriptionProvider": OpenAICompatTranscriptionProvider,
            "resolve_openai_compat": resolve_openai_compat,
        }[name]

    if name in {
        "AgentConfig",
        "ProviderConfig",
        "AgentDefaults",
        "AgentsConfig",
        "ExecToolConfig",
        "ToolsConfig",
        "get_config_path",
        "load_config",
        "save_config",
    }:
        from .config import (
            AgentConfig,
            AgentDefaults,
            AgentsConfig,
            ExecToolConfig,
            ProviderConfig,
            ToolsConfig,
            get_config_path,
            load_config,
            save_config,
        )

        return {
            "AgentConfig": AgentConfig,
            "ProviderConfig": ProviderConfig,
            "AgentDefaults": AgentDefaults,
            "AgentsConfig": AgentsConfig,
            "ExecToolConfig": ExecToolConfig,
            "ToolsConfig": ToolsConfig,
            "get_config_path": get_config_path,
            "load_config": load_config,
            "save_config": save_config,
        }[name]

    if name in {"SubagentManager", "SubagentTask", "build_subagent_tools_registry"}:
        from .subagent import (
            SubagentManager,
            SubagentTask,
            build_subagent_tools_registry,
        )

        return {
            "SubagentManager": SubagentManager,
            "SubagentTask": SubagentTask,
            "build_subagent_tools_registry": build_subagent_tools_registry,
        }[name]

    if name in {"TrackStore"}:
        from .track_store import TrackStore

        return {"TrackStore": TrackStore}[name]

    if name in {
        "BehaviorEngine",
        "BehaviorEngineConfig",
        "BehaviorEvent",
        "BehaviorUpdate",
    }:
        from .behavior_engine import (
            BehaviorEngine,
            BehaviorEngineConfig,
            BehaviorEvent,
            BehaviorUpdate,
        )

        return {
            "BehaviorEngine": BehaviorEngine,
            "BehaviorEngineConfig": BehaviorEngineConfig,
            "BehaviorEvent": BehaviorEvent,
            "BehaviorUpdate": BehaviorUpdate,
        }[name]

    if name in {"AgentRunConfig", "AgentRunner"}:
        from .runner import AgentRunConfig, AgentRunner

        return {"AgentRunConfig": AgentRunConfig, "AgentRunner": AgentRunner}[name]

    if name in {"AgentServiceResult", "run_agent_to_results"}:
        from .service import AgentServiceResult, run_agent_to_results

        return {
            "AgentServiceResult": AgentServiceResult,
            "run_agent_to_results": run_agent_to_results,
        }[name]

    if name in {
        "AgentLoop",
        "AgentLoopResult",
        "AgentMemoryConfig",
        "AgentToolRun",
        "InMemorySessionStore",
        "SessionStoreProtocol",
    }:
        from .loop import (
            AgentLoop,
            AgentLoopResult,
            AgentMemoryConfig,
            AgentToolRun,
            InMemorySessionStore,
            SessionStoreProtocol,
        )

        return {
            "AgentLoop": AgentLoop,
            "AgentLoopResult": AgentLoopResult,
            "AgentMemoryConfig": AgentMemoryConfig,
            "AgentToolRun": AgentToolRun,
            "InMemorySessionStore": InMemorySessionStore,
            "SessionStoreProtocol": SessionStoreProtocol,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
