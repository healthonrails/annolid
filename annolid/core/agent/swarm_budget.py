from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

from annolid.core.agent.utils import get_agent_data_path


logger = logging.getLogger(__name__)


_SWARM_BRAINSTORM_HINTS = (
    "brainstorm",
    "roundtable",
    "debate",
    "collaborative review",
    "multi-agent",
    "swarm",
)

_SWARM_RESEARCH_HINTS = (
    "paper",
    "draft",
    "literature",
    "review",
    "research",
    "citation",
    "outline",
)

_SWARM_BUDGET_HISTORY_LOCK = Lock()
_SWARM_BUDGET_HISTORY: dict[str, deque["SwarmBudgetObservation"]] = defaultdict(
    lambda: deque(maxlen=24)
)
_SWARM_BUDGET_HISTORY_SOURCE: str | None = None
_SWARM_BUDGET_HISTORY_ENV = "ANNOLID_SWARM_BUDGET_HISTORY_PATH"
_SWARM_BUDGET_HISTORY_FILENAME = "swarm_budget_history.json"
_SWARM_BUDGET_HISTORY_BACKUP_SUFFIX = ".bak"


@dataclass(frozen=True)
class SwarmBudgetObservation:
    task_key: str
    requested_turns: int
    used_turns: int
    completed: bool
    timed_out: bool
    agent_count: int = 0
    paper_context: bool = False


def _get_swarm_budget_history_path() -> Path:
    raw_path = str(os.environ.get(_SWARM_BUDGET_HISTORY_ENV) or "").strip()
    if raw_path:
        return Path(raw_path).expanduser()
    return get_agent_data_path() / _SWARM_BUDGET_HISTORY_FILENAME


def _serialize_swarm_budget_history() -> dict[str, object]:
    observations = {}
    for key, entries in sorted(_SWARM_BUDGET_HISTORY.items()):
        if not entries:
            continue
        observations[key] = [asdict(entry) for entry in entries]
    return {
        "version": 1,
        "observations": observations,
    }


def _swarm_budget_history_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{_SWARM_BUDGET_HISTORY_BACKUP_SUFFIX}")


def _read_swarm_budget_history_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load swarm budget history from %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning(
            "Ignoring malformed swarm budget history at %s: root payload is not a mapping",
            path,
        )
        return None
    raw_observations = payload.get("observations") or {}
    if not isinstance(raw_observations, dict):
        logger.warning(
            "Ignoring malformed swarm budget history at %s: observations is not a mapping",
            path,
        )
        return None
    return payload


def _populate_swarm_budget_history_from_payload(payload: dict[str, object]) -> None:
    _SWARM_BUDGET_HISTORY.clear()
    raw_observations = payload.get("observations") or {}
    if not isinstance(raw_observations, dict):
        return
    for key, entries in raw_observations.items():
        if not isinstance(entries, list):
            continue
        bucket = deque(maxlen=24)
        for item in entries:
            if not isinstance(item, dict):
                continue
            try:
                bucket.append(
                    SwarmBudgetObservation(
                        task_key=str(item.get("task_key") or key),
                        requested_turns=max(1, int(item.get("requested_turns") or 1)),
                        used_turns=max(1, int(item.get("used_turns") or 1)),
                        completed=bool(item.get("completed", False)),
                        timed_out=bool(item.get("timed_out", False)),
                        agent_count=max(0, int(item.get("agent_count") or 0)),
                        paper_context=bool(item.get("paper_context", False)),
                    )
                )
            except Exception:
                continue
        if bucket:
            _SWARM_BUDGET_HISTORY[str(key)] = bucket


def _load_swarm_budget_history_locked() -> None:
    global _SWARM_BUDGET_HISTORY_SOURCE
    path = _get_swarm_budget_history_path()
    source = str(path)
    if _SWARM_BUDGET_HISTORY_SOURCE == source:
        return

    _SWARM_BUDGET_HISTORY_SOURCE = source
    payload = _read_swarm_budget_history_payload(path)
    recovered_from_backup = False
    if payload is None:
        backup_path = _swarm_budget_history_backup_path(path)
        payload = _read_swarm_budget_history_payload(backup_path)
        recovered_from_backup = payload is not None

    if payload is None:
        _SWARM_BUDGET_HISTORY.clear()
        return

    _populate_swarm_budget_history_from_payload(payload)
    if recovered_from_backup:
        _persist_swarm_budget_history_locked()


def _persist_swarm_budget_history_locked() -> None:
    path = _get_swarm_budget_history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f"{path.name}.tmp"
        serialized = json.dumps(
            _serialize_swarm_budget_history(), indent=2, sort_keys=True
        )
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(path)
        backup_path = _swarm_budget_history_backup_path(path)
        backup_tmp_path = backup_path.parent / f"{backup_path.name}.tmp"
        backup_tmp_path.write_text(serialized, encoding="utf-8")
        backup_tmp_path.replace(backup_path)
    except Exception as exc:
        logger.warning("Failed to persist swarm budget history to %s: %s", path, exc)


def _ensure_swarm_budget_history_loaded_locked() -> None:
    _load_swarm_budget_history_locked()


def _count_hits(text: str, hints: Sequence[str]) -> int:
    lower = str(text or "").lower()
    return sum(1 for hint in hints if hint in lower)


def classify_swarm_task(
    task: str,
    *,
    paper_context: bool = False,
    agent_count: int = 0,
) -> str:
    lower = str(task or "").lower()
    if paper_context or _count_hits(lower, _SWARM_RESEARCH_HINTS) > 0:
        return "paper"
    if _count_hits(lower, _SWARM_BRAINSTORM_HINTS) > 0:
        return "brainstorm"
    if any(token in lower for token in ("review", "critique", "audit", "check")):
        return "review"
    if agent_count >= 4:
        return "multi_agent"
    return "general"


def reset_swarm_budget_history() -> None:
    with _SWARM_BUDGET_HISTORY_LOCK:
        _ensure_swarm_budget_history_loaded_locked()
        _SWARM_BUDGET_HISTORY.clear()
        _persist_swarm_budget_history_locked()


def record_swarm_budget_observation(
    task: str,
    *,
    requested_turns: int,
    used_turns: int,
    completed: bool,
    timed_out: bool,
    agent_count: int = 0,
    paper_context: bool = False,
) -> None:
    key = classify_swarm_task(
        task,
        paper_context=paper_context,
        agent_count=agent_count,
    )
    observation = SwarmBudgetObservation(
        task_key=key,
        requested_turns=max(1, int(requested_turns)),
        used_turns=max(1, int(used_turns)),
        completed=bool(completed),
        timed_out=bool(timed_out),
        agent_count=max(0, int(agent_count)),
        paper_context=bool(paper_context),
    )
    with _SWARM_BUDGET_HISTORY_LOCK:
        _ensure_swarm_budget_history_loaded_locked()
        _SWARM_BUDGET_HISTORY[key].append(observation)
        _persist_swarm_budget_history_locked()


def _history_bonus(
    task: str,
    *,
    agent_count: int = 0,
    paper_context: bool = False,
) -> int:
    key = classify_swarm_task(
        task,
        paper_context=paper_context,
        agent_count=agent_count,
    )
    with _SWARM_BUDGET_HISTORY_LOCK:
        _ensure_swarm_budget_history_loaded_locked()
        observations = list(_SWARM_BUDGET_HISTORY.get(key, ()))
    if not observations:
        return 0

    weighted_total = 0.0
    weighted_timeouts = 0.0
    weighted_completions = 0.0
    weighted_requested = 0.0
    weighted_used = 0.0
    recent_pressure = 0.0
    count = len(observations)
    for index, obs in enumerate(observations, start=1):
        weight = 1.0 + (index / count)
        weighted_total += weight
        weighted_timeouts += weight if obs.timed_out else 0.0
        weighted_completions += weight if obs.completed else 0.0
        weighted_requested += weight * obs.requested_turns
        weighted_used += weight * obs.used_turns
        recent_pressure += obs.used_turns / max(1.0, obs.requested_turns)

    timeout_rate = weighted_timeouts / weighted_total
    completion_rate = weighted_completions / weighted_total
    avg_requested = weighted_requested / weighted_total
    avg_used = weighted_used / weighted_total
    pressure = avg_used / max(1.0, avg_requested)
    mean_pressure = recent_pressure / count

    score = 0
    if timeout_rate >= 0.5:
        score += 2
    elif timeout_rate > 0:
        score += 1
    if pressure >= 1.15 or mean_pressure >= 1.15:
        score += 2
    elif pressure >= 1.0:
        score += 1
    if completion_rate < 0.5 and avg_used >= 0.8 * avg_requested:
        score += 1
    if count >= 2 and observations[-1].timed_out and observations[-2].timed_out:
        score += 1
    return min(4, score)


def resolve_swarm_turn_budget(
    task: str,
    requested_max_turns: int = 8,
    *,
    agent_count: int = 0,
    paper_context: bool = False,
    default_floor: int = 8,
    max_cap: int = 14,
) -> int:
    """Resolve an adaptive swarm turn budget.

    The budget only increases above the requested value when the request is at
    or above the swarm default floor. Short explicit budgets are respected.
    """
    requested = max(1, int(requested_max_turns))
    if requested < default_floor:
        return requested

    score = 0
    lower = str(task or "").lower()
    score += _count_hits(lower, _SWARM_BRAINSTORM_HINTS)
    score += min(2, _count_hits(lower, _SWARM_RESEARCH_HINTS))
    if paper_context:
        score += 2
    if len(lower) > 1200:
        score += 2
    elif len(lower) > 500:
        score += 1
    if agent_count >= 4:
        score += 2
    elif agent_count >= 3:
        score += 1
    score += _history_bonus(
        task,
        agent_count=agent_count,
        paper_context=paper_context,
    )

    bonus = min(6, score)
    return min(max_cap, requested + bonus)
