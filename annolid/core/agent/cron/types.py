from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CronSchedule:
    """Schedule definition for cron jobs."""

    kind: Literal["at", "every", "cron"]
    at_ms: Optional[int] = None
    every_ms: Optional[int] = None
    expr: Optional[str] = None
    tz: Optional[str] = None


@dataclass
class CronPayload:
    """Payload executed by cron runner."""

    kind: Literal["agent_turn", "system_event"] = "agent_turn"
    message: str = ""
    deliver: bool = False
    channel: Optional[str] = None
    to: Optional[str] = None


@dataclass
class CronJobState:
    next_run_at_ms: Optional[int] = None
    last_run_at_ms: Optional[int] = None
    last_status: Optional[Literal["ok", "error", "skipped"]] = None
    last_error: Optional[str] = None


@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass
class CronStore:
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
