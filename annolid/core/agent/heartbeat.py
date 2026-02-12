from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60
HEARTBEAT_PROMPT = (
    "Read HEARTBEAT.md in your workspace (if it exists).\n"
    "Follow any instructions or tasks listed there.\n"
    "If nothing needs attention, reply with just: HEARTBEAT_OK"
)
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

HeartbeatHandler = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class HeartbeatResult:
    status: str
    message: str
    response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def is_heartbeat_empty(content: Optional[str]) -> bool:
    """Return True when HEARTBEAT.md has no actionable lines."""
    if not content:
        return True
    skip_lines = {"- [ ]", "* [ ]", "- [x]", "* [x]"}
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            continue
        if line in skip_lines:
            continue
        return False
    return True


class HeartbeatService:
    """Periodic workspace heartbeat that can wake an agent callback."""

    def __init__(
        self,
        workspace: Path,
        *,
        on_heartbeat: Optional[HeartbeatHandler] = None,
        interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S,
        enabled: bool = True,
        timeout_s: float = 120.0,
        jitter_ratio: float = 0.0,
        run_immediately: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser()
        self.on_heartbeat = on_heartbeat
        self.interval_s = max(1, int(interval_s))
        self.enabled = bool(enabled)
        self.timeout_s = max(0.01, float(timeout_s))
        self.jitter_ratio = min(0.5, max(0.0, float(jitter_ratio)))
        self.run_immediately = bool(run_immediately)
        self._logger = logger or logging.getLogger("annolid.agent.heartbeat")

        self._running = False
        self._task: Optional[asyncio.Task[Any]] = None
        self._last_result: Optional[HeartbeatResult] = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    @property
    def last_result(self) -> Optional[HeartbeatResult]:
        return self._last_result

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def _read_heartbeat_file(self) -> Optional[str]:
        if not self.heartbeat_file.exists():
            return None
        try:
            return self.heartbeat_file.read_text(encoding="utf-8")
        except Exception:
            return None

    async def start(self) -> None:
        if not self.enabled:
            self._last_result = HeartbeatResult(
                status="disabled", message="Heartbeat disabled."
            )
            return
        if self.is_running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self._logger.info("Heartbeat started: interval=%ss", self.interval_s)

    async def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def trigger_now(self) -> HeartbeatResult:
        return await self._tick()

    def _compute_sleep_seconds(self) -> float:
        if self.jitter_ratio <= 0.0:
            return float(self.interval_s)
        span = self.interval_s * self.jitter_ratio
        return max(1.0, float(self.interval_s) + random.uniform(-span, span))

    async def _run_loop(self) -> None:
        if self.run_immediately and self._running:
            await self._tick()
        while self._running:
            try:
                await asyncio.sleep(self._compute_sleep_seconds())
                if not self._running:
                    break
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._last_result = HeartbeatResult(
                    status="error",
                    message=f"Heartbeat loop error: {exc}",
                )
                self._logger.exception("Heartbeat loop error")

    async def _tick(self) -> HeartbeatResult:
        content = self._read_heartbeat_file()
        if is_heartbeat_empty(content):
            result = HeartbeatResult(
                status="idle",
                message="No actionable heartbeat tasks.",
            )
            self._last_result = result
            return result

        if self.on_heartbeat is None:
            result = HeartbeatResult(
                status="skipped",
                message="No heartbeat handler registered.",
            )
            self._last_result = result
            return result

        try:
            response = await asyncio.wait_for(
                self.on_heartbeat(HEARTBEAT_PROMPT),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            result = HeartbeatResult(
                status="timeout",
                message=f"Heartbeat timed out after {self.timeout_s:.0f}s.",
            )
            self._last_result = result
            return result
        except Exception as exc:
            result = HeartbeatResult(
                status="error",
                message=f"Heartbeat execution failed: {exc}",
            )
            self._last_result = result
            return result

        text = str(response or "")
        normalized = text.upper().replace("_", "").replace(" ", "")
        ok_token = HEARTBEAT_OK_TOKEN.upper().replace("_", "")
        if ok_token in normalized:
            result = HeartbeatResult(
                status="ok",
                message="Heartbeat completed with no action needed.",
                response=text,
            )
            self._last_result = result
            return result

        result = HeartbeatResult(
            status="action",
            message="Heartbeat completed with actionable output.",
            response=text,
        )
        self._last_result = result
        return result
