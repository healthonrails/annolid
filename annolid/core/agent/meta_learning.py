from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Mapping, Sequence


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str, *, max_len: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    text = text.strip("-")
    if not text:
        return "unknown"
    return text[:max_len].strip("-") or "unknown"


def _extract_error_preview(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        for key in ("error", "message", "detail", "text", "content"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:220]
    return text[:220]


def _normalize_reason(reason: str) -> str:
    """
    Normalize volatile tokens so repeated failures cluster into stable signatures.
    """
    text = str(reason or "").strip().lower()
    if not text:
        return ""
    # normalize absolute/relative paths
    text = re.sub(r"(/[^ \n\t,;:]+)+", "<path>", text)
    text = re.sub(r"[A-Za-z]:\\[^ \n\t,;:]+", "<path>", text)
    # normalize integers/ids
    text = re.sub(r"\b\d+\b", "<n>", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


_PRM_SCORE_RE = re.compile(r"score\s*:\s*([-+]?\d)", re.IGNORECASE)
_SCHEDULER_STATES = {
    "disabled",
    "idle_wait",
    "window_open",
    "updating",
    "pausing",
}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _extract_frontmatter_fields(raw_text: str) -> dict[str, str]:
    text = str(raw_text or "")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    out: dict[str, str] = {}
    block = text[4:end].strip()
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        k = str(key).strip().lower()
        v = str(value).strip()
        if k and v:
            out[k] = v
    return out


@dataclass(frozen=True)
class FailurePatternHit:
    signature: str
    tool_name: str
    reason: str
    count: int


class AgentMetaLearner:
    """
    Lightweight meta-learning/evolution layer for Annolid agent loops.

    This module records turn outcomes, tracks repeated tool-failure patterns,
    and auto-generates remediation skills after failures recur.
    """

    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annolid-meta")

    def __init__(
        self,
        workspace: Path,
        *,
        enabled: bool | None = None,
        auto_evolve_skills: bool | None = None,
        failure_threshold: int | None = None,
        llm_evolver_enabled: bool | None = None,
        llm_client: Any | None = None,
        llm_model: str | None = None,
        prm_enabled: bool | None = None,
        prm_client: Any | None = None,
        prm_model: str | None = None,
        prm_votes: int | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self._io_lock = RLock()
        self.enabled = (
            _env_flag("ANNOLID_AGENT_META_LEARNING_ENABLED", False)
            if enabled is None
            else bool(enabled)
        )
        self.auto_evolve_skills = (
            _env_flag("ANNOLID_AGENT_META_LEARNING_AUTO_EVOLVE", True)
            if auto_evolve_skills is None
            else bool(auto_evolve_skills)
        )
        env_threshold = os.getenv("ANNOLID_AGENT_META_LEARNING_FAILURE_THRESHOLD", "")
        if failure_threshold is None:
            try:
                failure_threshold = int(env_threshold) if env_threshold else 3
            except Exception:
                failure_threshold = 3
        self.failure_threshold = max(2, int(failure_threshold))
        self.llm_evolver_enabled = (
            _env_flag("ANNOLID_AGENT_META_LEARNING_LLM_EVOLVER_ENABLED", False)
            if llm_evolver_enabled is None
            else bool(llm_evolver_enabled)
        )
        self._llm_client = llm_client
        self._llm_model = (
            str(llm_model or "").strip()
            or str(os.getenv("ANNOLID_AGENT_META_LEARNING_LLM_MODEL", "")).strip()
            or "gpt-5.2"
        )
        timeout_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_LLM_TIMEOUT_S", "")
        ).strip()
        try:
            self._llm_timeout_s = float(timeout_raw) if timeout_raw else 8.0
        except Exception:
            self._llm_timeout_s = 8.0
        self._llm_timeout_s = max(1.0, min(60.0, self._llm_timeout_s))
        self.prm_enabled = (
            _env_flag("ANNOLID_AGENT_META_LEARNING_PRM_ENABLED", False)
            if prm_enabled is None
            else bool(prm_enabled)
        )
        self._prm_client = prm_client
        self._prm_model = (
            str(prm_model or "").strip()
            or str(os.getenv("ANNOLID_AGENT_META_LEARNING_PRM_MODEL", "")).strip()
            or self._llm_model
        )
        prm_votes_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_PRM_VOTES", "")
        ).strip()
        if prm_votes is None:
            try:
                prm_votes = int(prm_votes_raw) if prm_votes_raw else 3
            except Exception:
                prm_votes = 3
        self._prm_votes = max(1, min(7, int(prm_votes)))
        prm_timeout_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_PRM_TIMEOUT_S", "")
        ).strip()
        try:
            self._prm_timeout_s = float(prm_timeout_raw) if prm_timeout_raw else 4.0
        except Exception:
            self._prm_timeout_s = 4.0
        self._prm_timeout_s = max(1.0, min(30.0, self._prm_timeout_s))
        self.reward_trigger_enabled = _env_flag(
            "ANNOLID_AGENT_META_LEARNING_REWARD_ENABLED", True
        )
        reward_window_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_REWARD_WINDOW", "")
        ).strip()
        try:
            reward_window = int(reward_window_raw) if reward_window_raw else 6
        except Exception:
            reward_window = 6
        self.reward_window = max(2, min(100, int(reward_window)))
        reward_threshold_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_REWARD_THRESHOLD", "")
        ).strip()
        reward_threshold = _coerce_float(reward_threshold_raw, 0.45)
        self.reward_threshold = _clamp(reward_threshold, 0.0, 1.0)
        self.idle_scheduler_enabled = _env_flag(
            "ANNOLID_AGENT_META_LEARNING_IDLE_SCHEDULER_ENABLED", False
        )
        idle_seconds_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_IDLE_MIN_SECONDS", "")
        ).strip()
        try:
            idle_seconds = int(idle_seconds_raw) if idle_seconds_raw else 900
        except Exception:
            idle_seconds = 900
        self.idle_min_seconds = max(30, min(86_400, int(idle_seconds)))
        self.idle_sleep_start = (
            str(os.getenv("ANNOLID_AGENT_META_LEARNING_IDLE_SLEEP_START", "")).strip()
            or "23:00"
        )
        self.idle_sleep_end = (
            str(os.getenv("ANNOLID_AGENT_META_LEARNING_IDLE_SLEEP_END", "")).strip()
            or "07:00"
        )
        idle_jobs_raw = str(
            os.getenv("ANNOLID_AGENT_META_LEARNING_IDLE_MAX_JOBS_PER_TICK", "")
        ).strip()
        try:
            idle_jobs = int(idle_jobs_raw) if idle_jobs_raw else 2
        except Exception:
            idle_jobs = 2
        self.idle_max_jobs_per_tick = max(1, min(20, int(idle_jobs)))

        root = self.workspace / "memory" / "meta_learning"
        self._events_path = root / "events.jsonl"
        self._patterns_path = root / "failure_patterns.json"
        self._reward_window_path = root / "reward_window.json"
        self._history_path = root / "evolution_history.jsonl"
        self._pending_jobs_path = root / "pending_evolution_jobs.json"
        self._last_activity_path = root / "last_activity.json"
        self._generation_state_path = root / "generation_state.json"
        self._scheduler_state_path = root / "scheduler_state.json"

    def record_turn(
        self,
        *,
        session_id: str,
        user_text: str,
        assistant_text: str,
        tool_runs: Sequence[Mapping[str, Any]] | Sequence[Any],
        stopped_reason: str,
        empty_repair_used: bool,
        llm_total_ms: float,
        total_ms: float,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "recorded": False, "evolved_skills": []}

        failures = list(self._extract_failure_hits(tool_runs))
        failure_count = len(failures)
        heuristic_score = self._compute_outcome_score(
            assistant_text=assistant_text,
            failure_count=failure_count,
            stopped_reason=stopped_reason,
            empty_repair_used=empty_repair_used,
        )
        prm_result = self._score_turn_with_prm(
            user_text=user_text,
            assistant_text=assistant_text,
            fallback_score=heuristic_score,
        )
        prm_score = prm_result.get("score")
        if prm_score is None:
            outcome_score = float(heuristic_score)
        else:
            outcome_score = float(prm_score)
        rolling_avg = outcome_score
        rolling_scores: list[float] = [outcome_score]
        if self.reward_trigger_enabled:
            rolling_avg, rolling_scores = self._append_reward_score(outcome_score)
        event = {
            "ts": _now_iso(),
            "session_id": str(session_id or ""),
            "user_chars": len(str(user_text or "")),
            "assistant_chars": len(str(assistant_text or "")),
            "tool_runs": len(list(tool_runs or [])),
            "stopped_reason": str(stopped_reason or "done"),
            "empty_repair_used": bool(empty_repair_used),
            "llm_total_ms": float(llm_total_ms),
            "total_ms": float(total_ms),
            "outcome_score": float(outcome_score),
            "reward_window_avg": float(rolling_avg),
            "score_source": str(prm_result.get("source") or "heuristic"),
            "prm_votes": list(prm_result.get("votes") or []),
            "prm_vote_count": int(prm_result.get("vote_count") or 0),
            "prm_representative_eval": str(prm_result.get("representative_eval") or "")[
                :300
            ],
            "failures": [
                {
                    "signature": hit.signature,
                    "tool": hit.tool_name,
                    "reason": hit.reason,
                    "count": hit.count,
                }
                for hit in failures
            ],
        }
        self._append_event(event)
        evolved_skills: list[str] = []
        queued_jobs = 0
        if failures:
            with self._io_lock:
                counts = self._load_pattern_counts()
                evolved_signatures: set[str] = set()
                for hit in failures:
                    counts[hit.signature] = int(counts.get(hit.signature, 0)) + 1
                    current_generation = self._get_current_generation()
                    if (
                        self.auto_evolve_skills
                        and counts[hit.signature] >= self.failure_threshold
                    ):
                        queued_jobs += self._enqueue_evolution_job(
                            signature=hit.signature,
                            tool_name=hit.tool_name,
                            reason=hit.reason,
                            trigger="failure_threshold",
                            signature_count=int(counts.get(hit.signature, 0)),
                            outcome_score=float(outcome_score),
                            reward_window_avg=float(rolling_avg),
                            queued_generation=current_generation,
                        )
                        evolved_signatures.add(hit.signature)
                        counts[hit.signature] = 0
                reward_trigger = (
                    self.auto_evolve_skills
                    and self.reward_trigger_enabled
                    and len(rolling_scores) >= min(3, self.reward_window)
                    and rolling_avg < self.reward_threshold
                )
                if reward_trigger:
                    for hit in failures:
                        if hit.signature in evolved_signatures:
                            continue
                        queued_jobs += self._enqueue_evolution_job(
                            signature=hit.signature,
                            tool_name=hit.tool_name,
                            reason=hit.reason,
                            trigger="reward_window",
                            signature_count=int(counts.get(hit.signature, 0)),
                            outcome_score=float(outcome_score),
                            reward_window_avg=float(rolling_avg),
                            reward_threshold=float(self.reward_threshold),
                            queued_generation=current_generation,
                        )
                        counts[hit.signature] = 0
                        break
                self._save_pattern_counts(counts)
        maintenance = self.run_idle_maintenance(
            force=not bool(self.idle_scheduler_enabled)
        )
        evolved_skills.extend(list(maintenance.get("evolved_skills") or []))
        self._save_last_activity()
        return {
            "enabled": True,
            "recorded": True,
            "queued_jobs": int(queued_jobs),
            "evolved_skills": evolved_skills,
            "maintenance": maintenance,
        }

    def _extract_failure_hits(
        self, tool_runs: Sequence[Mapping[str, Any]] | Sequence[Any]
    ) -> Iterable[FailurePatternHit]:
        seen: set[str] = set()
        for row in list(tool_runs or []):
            if isinstance(row, Mapping):
                tool_name = str(row.get("name") or "").strip().lower()
                result = str(row.get("result") or "")
            else:
                tool_name = str(getattr(row, "name", "") or "").strip().lower()
                result = str(getattr(row, "result", "") or "")
            if not tool_name or not result:
                continue
            reason = _extract_error_preview(result)
            if not reason:
                continue
            reason_l = _normalize_reason(reason)
            if (
                "error" not in reason_l
                and "failed" not in reason_l
                and "exception" not in reason_l
                and "timeout" not in reason_l
                and "not found" not in reason_l
                and "invalid" not in reason_l
            ):
                continue
            sig_source = f"{tool_name}:{reason_l[:120]}"
            sig_hash = hashlib.sha1(sig_source.encode("utf-8")).hexdigest()[:10]
            signature = f"{tool_name}:{sig_hash}"
            if signature in seen:
                continue
            seen.add(signature)
            yield FailurePatternHit(
                signature=signature,
                tool_name=tool_name,
                reason=reason,
                count=1,
            )

    def _append_event(self, payload: dict[str, Any]) -> None:
        with self._io_lock:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_history_event(self, payload: Mapping[str, Any]) -> None:
        with self._io_lock:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            with self._history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

    def _load_pending_jobs(self) -> dict[str, dict[str, Any]]:
        if not self._pending_jobs_path.exists():
            return {}
        try:
            loaded = json.loads(self._pending_jobs_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(loaded, Mapping):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for key, value in loaded.items():
            if not isinstance(value, Mapping):
                continue
            out[str(key)] = dict(value)
        return out

    def _save_pending_jobs(self, jobs: Mapping[str, Mapping[str, Any]]) -> None:
        with self._io_lock:
            self._pending_jobs_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, dict[str, Any]] = {}
            for key, value in jobs.items():
                payload[str(key)] = dict(value)
            self._pending_jobs_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _enqueue_evolution_job(
        self,
        *,
        signature: str,
        tool_name: str,
        reason: str,
        trigger: str,
        signature_count: int,
        outcome_score: float,
        reward_window_avg: float,
        queued_generation: int,
        reward_threshold: float | None = None,
    ) -> int:
        with self._io_lock:
            jobs = self._load_pending_jobs()
            existing = dict(jobs.get(signature) or {})
            payload: dict[str, Any] = {
                "signature": str(signature),
                "tool_name": str(tool_name),
                "reason": str(reason),
                "trigger": str(trigger),
                "signature_count": int(signature_count),
                "outcome_score": float(outcome_score),
                "reward_window_avg": float(reward_window_avg),
                "queued_generation": int(queued_generation),
                "queued_at": str(existing.get("queued_at") or _now_iso()),
                "updated_at": _now_iso(),
                "attempts": int(existing.get("attempts") or 0),
            }
            if reward_threshold is not None:
                payload["reward_threshold"] = float(reward_threshold)
            jobs[str(signature)] = payload
            self._save_pending_jobs(jobs)
            return 0 if existing else 1

    def _save_last_activity(self) -> None:
        with self._io_lock:
            self._last_activity_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"last_activity_at": _now_iso()}
            self._last_activity_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _load_last_activity_dt(self) -> datetime | None:
        if not self._last_activity_path.exists():
            return None
        try:
            loaded = json.loads(self._last_activity_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(loaded, Mapping):
            return None
        raw = str(loaded.get("last_activity_at") or "").strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _sleep_hours_active(self, now_local: datetime) -> bool:
        try:
            start = datetime.strptime(self.idle_sleep_start, "%H:%M").time()
            end = datetime.strptime(self.idle_sleep_end, "%H:%M").time()
        except Exception:
            return False
        now_t = now_local.timetz().replace(tzinfo=None)
        if start <= end:
            return start <= now_t <= end
        return now_t >= start or now_t <= end

    def _is_idle_window_open(self) -> tuple[bool, str]:
        now_local = datetime.now().astimezone()
        if self._sleep_hours_active(now_local):
            return True, "sleep_window"
        last_activity = self._load_last_activity_dt()
        if last_activity is None:
            return False, "no_activity_data"
        now_utc = datetime.now(timezone.utc)
        idle_seconds = max(0.0, (now_utc - last_activity).total_seconds())
        if idle_seconds >= float(self.idle_min_seconds):
            return True, "inactivity"
        return False, "active_recently"

    def _seconds_until_sleep_window(self, now_local: datetime) -> int | None:
        try:
            start = datetime.strptime(self.idle_sleep_start, "%H:%M").time()
            end = datetime.strptime(self.idle_sleep_end, "%H:%M").time()
        except Exception:
            return None
        if self._sleep_hours_active(now_local):
            return 0
        candidate = now_local.replace(
            hour=start.hour, minute=start.minute, second=0, microsecond=0
        )
        if candidate <= now_local:
            candidate = candidate + timedelta(days=1)
        if start <= end:
            # same-day window (non-wrapping), if before start then same day candidate
            if now_local.time() < start:
                candidate = now_local.replace(
                    hour=start.hour, minute=start.minute, second=0, microsecond=0
                )
        secs = int(max(0.0, (candidate - now_local).total_seconds()))
        return secs

    def _estimate_next_window_eta_seconds(self) -> int | None:
        open_window, _ = self._is_idle_window_open()
        if open_window:
            return 0
        now_local = datetime.now().astimezone()
        candidates: list[int] = []
        sleep_eta = self._seconds_until_sleep_window(now_local)
        if sleep_eta is not None:
            candidates.append(int(sleep_eta))
        last_activity = self._load_last_activity_dt()
        if last_activity is not None:
            idle_elapsed = max(
                0.0, (datetime.now(timezone.utc) - last_activity).total_seconds()
            )
            idle_eta = max(0.0, float(self.idle_min_seconds) - idle_elapsed)
            candidates.append(int(idle_eta))
        if not candidates:
            return None
        return min(candidates)

    def get_idle_maintenance_status(self) -> dict[str, Any]:
        with self._io_lock:
            jobs = self._load_pending_jobs()
            open_window, reason = self._is_idle_window_open()
            last_activity = self._load_last_activity_dt()
            self._sync_scheduler_state(open_window=open_window, window_reason=reason)
            scheduler_state = self._load_scheduler_state()
            now_utc = datetime.now(timezone.utc)
            last_activity_age_s: float | None = None
            if last_activity is not None:
                last_activity_age_s = max(
                    0.0, (now_utc - last_activity).total_seconds()
                )
            return {
                "scheduler_enabled": bool(self.idle_scheduler_enabled),
                "scheduler_state": str(scheduler_state.get("state") or "idle_wait"),
                "window_open": bool(open_window),
                "window_reason": str(reason),
                "pending_jobs_count": len(jobs),
                "next_window_eta_seconds": self._estimate_next_window_eta_seconds(),
                "idle_min_seconds": int(self.idle_min_seconds),
                "sleep_start": str(self.idle_sleep_start),
                "sleep_end": str(self.idle_sleep_end),
                "last_activity_age_seconds": (
                    float(last_activity_age_s)
                    if last_activity_age_s is not None
                    else None
                ),
            }

    def run_idle_maintenance(
        self,
        *,
        force: bool = False,
        max_jobs: int | None = None,
    ) -> dict[str, Any]:
        with self._io_lock:
            jobs = self._load_pending_jobs()
            pending_before = len(jobs)
            open_window, reason = self._is_idle_window_open()
            self._sync_scheduler_state(open_window=open_window, window_reason=reason)
            scheduler_state_before = str(
                self._load_scheduler_state().get("state") or "idle_wait"
            )
            if bool(force):
                open_window = True
                reason = "forced"
            if self.idle_scheduler_enabled and open_window:
                self._set_scheduler_state("window_open", reason=reason)
            if not jobs:
                scheduler_state_after = str(
                    self._load_scheduler_state().get("state") or "idle_wait"
                )
                return {
                    "ran": False,
                    "window_open": bool(open_window),
                    "window_reason": reason,
                    "pending_before": 0,
                    "pending_after": 0,
                    "processed_jobs": 0,
                    "evolved_skills": [],
                    "generation_before": self._get_current_generation(),
                    "generation_after": self._get_current_generation(),
                    "discarded_stale_jobs": 0,
                    "scheduler_state_before": scheduler_state_before,
                    "scheduler_state_after": scheduler_state_after,
                }
            if not self.idle_scheduler_enabled and not force:
                self._set_scheduler_state("disabled", reason="scheduler_disabled")
                return {
                    "ran": False,
                    "window_open": False,
                    "window_reason": "scheduler_disabled",
                    "pending_before": pending_before,
                    "pending_after": pending_before,
                    "processed_jobs": 0,
                    "evolved_skills": [],
                    "generation_before": self._get_current_generation(),
                    "generation_after": self._get_current_generation(),
                    "discarded_stale_jobs": 0,
                    "scheduler_state_before": scheduler_state_before,
                    "scheduler_state_after": "disabled",
                }
            if not open_window:
                self._set_scheduler_state("idle_wait", reason=reason)
                return {
                    "ran": False,
                    "window_open": False,
                    "window_reason": reason,
                    "pending_before": pending_before,
                    "pending_after": pending_before,
                    "processed_jobs": 0,
                    "evolved_skills": [],
                    "generation_before": self._get_current_generation(),
                    "generation_after": self._get_current_generation(),
                    "discarded_stale_jobs": 0,
                    "scheduler_state_before": scheduler_state_before,
                    "scheduler_state_after": "idle_wait",
                }
            generation_before = self._get_current_generation()
            current_generation = generation_before
            limit = (
                max(1, int(max_jobs))
                if max_jobs is not None
                else int(self.idle_max_jobs_per_tick)
            )
            selected_jobs: list[tuple[str, dict[str, Any], str]] = []
            for key, row in jobs.items():
                if isinstance(row, Mapping):
                    parsed = dict(row)
                    version_token = str(
                        parsed.get("updated_at") or parsed.get("queued_at") or ""
                    )
                    selected_jobs.append((str(key), parsed, version_token))
            selected_jobs.sort(key=lambda item: str(item[1].get("queued_at") or ""))
            selected_jobs = selected_jobs[:limit]
            if self.idle_scheduler_enabled:
                self._set_scheduler_state("updating", reason=reason)

        evolved_skills: list[str] = []
        processed = 0
        stale_discarded = 0
        paused_due_activity = False
        consumed_jobs: dict[str, str] = {}
        for job_key, row, version_token in selected_jobs:
            if self.idle_scheduler_enabled and not force:
                still_open, still_reason = self._is_idle_window_open()
                if not still_open:
                    self._set_scheduler_state("pausing", reason=still_reason)
                    reason = still_reason
                    paused_due_activity = True
                    break
            signature = str(row.get("signature") or job_key).strip()
            tool_name = str(row.get("tool_name") or "").strip()
            reason_text = str(row.get("reason") or "").strip()
            if not tool_name or not reason_text:
                consumed_jobs[job_key] = version_token
                continue
            queued_generation = int(row.get("queued_generation") or 0)
            if queued_generation < current_generation:
                stale_discarded += 1
                row["discarded_at"] = _now_iso()
                self._append_history_event(
                    {
                        "ts": _now_iso(),
                        "trigger": str(row.get("trigger") or "maintenance"),
                        "window_reason": reason,
                        "signature": signature,
                        "tool": tool_name,
                        "reason": reason_text,
                        "queued_generation": queued_generation,
                        "current_generation": current_generation,
                        "status": "discarded_stale",
                        "queued_at": str(row.get("queued_at") or ""),
                        "discarded_at": str(row.get("discarded_at") or ""),
                    }
                )
                consumed_jobs[job_key] = version_token
                continue
            skill_name = self._create_recovery_skill(
                signature=signature,
                tool_name=tool_name,
                reason=reason_text,
            )
            skill_snapshot = (
                self._read_skill_snapshot(skill_name) if skill_name else None
            )
            processed += 1
            if skill_name:
                current_generation = self._increment_generation()
            row["attempts"] = int(row.get("attempts") or 0) + 1
            row["executed_at"] = _now_iso()
            self._append_history_event(
                {
                    "ts": _now_iso(),
                    "trigger": str(row.get("trigger") or "maintenance"),
                    "window_reason": reason,
                    "signature": signature,
                    "tool": tool_name,
                    "reason": reason_text,
                    "signature_count": int(row.get("signature_count") or 0),
                    "outcome_score": float(row.get("outcome_score") or 0.0),
                    "reward_window_avg": float(row.get("reward_window_avg") or 0.0),
                    "reward_threshold": _coerce_float(
                        row.get("reward_threshold"), self.reward_threshold
                    ),
                    "skill_name": skill_name,
                    "skill": skill_snapshot,
                    "queued_generation": queued_generation,
                    "applied_generation": current_generation,
                    "queued_at": str(row.get("queued_at") or ""),
                    "executed_at": str(row.get("executed_at") or ""),
                    "attempts": int(row.get("attempts") or 1),
                }
            )
            if skill_name:
                evolved_skills.append(skill_name)
            consumed_jobs[job_key] = version_token

        with self._io_lock:
            jobs_after = self._load_pending_jobs()
            for key, token in consumed_jobs.items():
                current = jobs_after.get(key)
                if not isinstance(current, Mapping):
                    continue
                current_token = str(
                    current.get("updated_at") or current.get("queued_at") or ""
                )
                if token and current_token and current_token != token:
                    continue
                jobs_after.pop(key, None)
            self._save_pending_jobs(jobs_after)
            pending_after = len(jobs_after)
            if self.idle_scheduler_enabled:
                open_after, reason_after = self._is_idle_window_open()
                if paused_due_activity or not open_after:
                    self._set_scheduler_state("idle_wait", reason=reason_after)
                else:
                    self._set_scheduler_state("window_open", reason=reason_after)
            scheduler_state_after = str(
                self._load_scheduler_state().get("state") or "idle_wait"
            )

        return {
            "ran": True,
            "window_open": True,
            "window_reason": reason,
            "pending_before": pending_before,
            "pending_after": pending_after,
            "processed_jobs": processed,
            "evolved_skills": evolved_skills,
            "generation_before": generation_before,
            "generation_after": current_generation,
            "discarded_stale_jobs": stale_discarded,
            "paused_due_activity": bool(paused_due_activity),
            "scheduler_state_before": scheduler_state_before,
            "scheduler_state_after": scheduler_state_after,
        }

    def _read_skill_snapshot(self, skill_name: str) -> dict[str, Any] | None:
        name = str(skill_name or "").strip()
        if not name:
            return None
        skill_file = self.workspace / "skills" / name / "SKILL.md"
        if not skill_file.exists():
            return None
        try:
            raw = skill_file.read_text(encoding="utf-8")
        except Exception:
            return None
        fm = _extract_frontmatter_fields(raw)
        content = raw
        if raw.startswith("---\n"):
            end = raw.find("\n---", 4)
            if end >= 0:
                content = raw[end + 4 :].strip()
        description = str(fm.get("description") or "").strip()
        return {
            "name": name,
            "path": str(skill_file),
            "description": description[:240],
            "content_excerpt": content[:800],
            "content_length": len(content),
        }

    def _load_generation_state(self) -> dict[str, Any]:
        if not self._generation_state_path.exists():
            return {"current_generation": 0}
        try:
            loaded = json.loads(self._generation_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"current_generation": 0}
        if not isinstance(loaded, Mapping):
            return {"current_generation": 0}
        try:
            current = int(loaded.get("current_generation") or 0)
        except Exception:
            current = 0
        return {"current_generation": max(0, current)}

    def _save_generation_state(self, current_generation: int) -> None:
        with self._io_lock:
            self._generation_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "current_generation": max(0, int(current_generation)),
                "updated_at": _now_iso(),
            }
            self._generation_state_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _get_current_generation(self) -> int:
        return int(self._load_generation_state().get("current_generation") or 0)

    def _increment_generation(self) -> int:
        current = self._get_current_generation()
        nxt = current + 1
        self._save_generation_state(nxt)
        return nxt

    def _append_reward_score(self, score: float) -> tuple[float, list[float]]:
        with self._io_lock:
            window = self._load_reward_window()
            scores = [float(v) for v in list(window.get("scores") or [])]
            scores.append(float(score))
            if len(scores) > self.reward_window:
                scores = scores[-self.reward_window :]
            rolling_avg = sum(scores) / len(scores) if scores else float(score)
            self._save_reward_window(scores=scores, rolling_avg=rolling_avg)
            return float(rolling_avg), scores

    def _load_reward_window(self) -> dict[str, Any]:
        if not self._reward_window_path.exists():
            return {"scores": [], "rolling_avg": 0.0}
        try:
            loaded = json.loads(self._reward_window_path.read_text(encoding="utf-8"))
        except Exception:
            return {"scores": [], "rolling_avg": 0.0}
        if not isinstance(loaded, Mapping):
            return {"scores": [], "rolling_avg": 0.0}
        raw_scores = list(loaded.get("scores") or [])
        scores: list[float] = []
        for row in raw_scores:
            try:
                scores.append(float(row))
            except Exception:
                continue
        if len(scores) > self.reward_window:
            scores = scores[-self.reward_window :]
        rolling_avg = sum(scores) / len(scores) if scores else 0.0
        return {"scores": scores, "rolling_avg": rolling_avg}

    def _save_reward_window(
        self, *, scores: Sequence[float], rolling_avg: float
    ) -> None:
        with self._io_lock:
            self._reward_window_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "scores": [float(v) for v in list(scores)[-self.reward_window :]],
                "rolling_avg": float(rolling_avg),
                "updated_at": _now_iso(),
            }
            self._reward_window_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    @staticmethod
    def _compute_outcome_score(
        *,
        assistant_text: str,
        failure_count: int,
        stopped_reason: str,
        empty_repair_used: bool,
    ) -> float:
        score = 1.0
        if not str(assistant_text or "").strip():
            score -= 0.5
        if bool(empty_repair_used):
            score -= 0.2
        if str(stopped_reason or "").strip().lower() in {"max_iterations", "error"}:
            score -= 0.35
        score -= min(0.75, 0.25 * max(0, int(failure_count)))
        return _clamp(score, 0.0, 1.0)

    def _score_turn_with_prm(
        self,
        *,
        user_text: str,
        assistant_text: str,
        fallback_score: float,
    ) -> dict[str, Any]:
        if not self.prm_enabled:
            return {
                "score": float(fallback_score),
                "source": "heuristic",
                "votes": [],
                "vote_count": 0,
                "representative_eval": "",
            }
        prompt = (
            "You are scoring assistant quality for a single user turn.\n"
            "Return ONLY one line in format: Score: 1 or Score: 0 or Score: -1.\n"
            "Score guidelines:\n"
            "- 1: response clearly addresses user request with useful output.\n"
            "- 0: ambiguous/partial usefulness.\n"
            "-1: fails or irrelevant.\n\n"
            f"User:\n{str(user_text or '')[:800]}\n\n"
            f"Assistant:\n{str(assistant_text or '')[:1200]}\n"
        )
        votes: list[int] = []
        vote_details: list[int | str] = []
        representative_eval = ""
        for _ in range(int(self._prm_votes)):
            score, raw = self._query_prm_once(prompt)
            if score is not None:
                votes.append(score)
                vote_details.append(int(score))
            else:
                vote_details.append("fail")
            if not representative_eval and raw:
                representative_eval = str(raw).strip()[:300]
        if not votes:
            return {
                "score": float(fallback_score),
                "source": "heuristic_fallback",
                "votes": vote_details,
                "vote_count": 0,
                "representative_eval": representative_eval,
            }
        positive = sum(1 for v in votes if v > 0)
        negative = sum(1 for v in votes if v < 0)
        neutral = len(votes) - positive - negative
        score = 0.5
        if positive > max(negative, neutral):
            score = 1.0
        elif negative > max(positive, neutral):
            score = 0.0
        return {
            "score": float(score),
            "source": "prm",
            "votes": vote_details,
            "vote_count": len(votes),
            "representative_eval": representative_eval,
        }

    def _query_prm_once(self, prompt: str) -> tuple[int | None, str]:
        client = self._prm_client or self._llm_client
        if client is None:
            try:
                client = self._build_default_llm_client()
            except Exception:
                return None, ""

        def _invoke() -> str:
            if hasattr(client, "chat_complete"):
                return str(client.chat_complete(prompt))
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model=self._prm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=120,
                )
                return str(response.choices[0].message.content or "")
            raise RuntimeError("Unsupported PRM client")

        fut = self._executor.submit(_invoke)
        try:
            raw = str(fut.result(timeout=self._prm_timeout_s) or "")
        except FutureTimeoutError:
            fut.cancel()
            return None, ""
        except Exception:
            return None, ""
        match = _PRM_SCORE_RE.search(raw)
        if not match:
            return None, raw
        try:
            score = int(match.group(1))
        except Exception:
            return None, raw
        if score not in {-1, 0, 1}:
            return None, raw
        return score, raw

    def _load_scheduler_state(self) -> dict[str, Any]:
        if not self._scheduler_state_path.exists():
            return {"state": "idle_wait", "updated_at": _now_iso()}
        try:
            loaded = json.loads(self._scheduler_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"state": "idle_wait", "updated_at": _now_iso()}
        if not isinstance(loaded, Mapping):
            return {"state": "idle_wait", "updated_at": _now_iso()}
        state = str(loaded.get("state") or "idle_wait")
        if state not in _SCHEDULER_STATES:
            state = "idle_wait"
        return {
            "state": state,
            "updated_at": str(loaded.get("updated_at") or _now_iso()),
            "reason": str(loaded.get("reason") or ""),
        }

    def _set_scheduler_state(self, state: str, *, reason: str = "") -> None:
        normalized = str(state or "").strip().lower()
        if normalized not in _SCHEDULER_STATES:
            normalized = "idle_wait"
        with self._io_lock:
            self._scheduler_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "state": normalized,
                "reason": str(reason or ""),
                "updated_at": _now_iso(),
            }
            self._scheduler_state_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _sync_scheduler_state(self, *, open_window: bool, window_reason: str) -> None:
        if not self.idle_scheduler_enabled:
            self._set_scheduler_state("disabled", reason="scheduler_disabled")
            return
        current = str(self._load_scheduler_state().get("state") or "idle_wait")
        if current in {"updating", "pausing"}:
            return
        if open_window:
            self._set_scheduler_state("window_open", reason=window_reason)
            return
        self._set_scheduler_state("idle_wait", reason=window_reason)

    def _load_pattern_counts(self) -> dict[str, int]:
        if not self._patterns_path.exists():
            return {}
        try:
            loaded = json.loads(self._patterns_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(loaded, Mapping):
            return {}
        out: dict[str, int] = {}
        for key, value in loaded.items():
            try:
                out[str(key)] = max(0, int(value))
            except Exception:
                continue
        return out

    def _save_pattern_counts(self, counts: Mapping[str, int]) -> None:
        with self._io_lock:
            self._patterns_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {str(k): int(v) for k, v in counts.items()}
            self._patterns_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def _create_recovery_skill(
        self,
        *,
        signature: str,
        tool_name: str,
        reason: str,
    ) -> str:
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
        name = f"meta-recover-{_slug(tool_name, max_len=20)}-{digest}"
        skill_dir = self.workspace / "skills" / name
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            return ""
        skill_dir.mkdir(parents=True, exist_ok=True)
        title = f"Recover `{tool_name}` Failures"
        description = ""
        content = ""
        evolved = self._build_skill_with_llm(
            tool_name=tool_name,
            reason=reason,
            default_name=name,
        )
        if evolved:
            description = str(evolved.get("description") or "").strip()
            content = str(evolved.get("content") or "").strip()
            suggested = str(evolved.get("name") or "").strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9-]{5,80}", suggested):
                name = suggested
                skill_dir = self.workspace / "skills" / name
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    return ""
                skill_dir.mkdir(parents=True, exist_ok=True)
        if not description:
            description = (
                f"Use when `{tool_name}` repeatedly fails with similar errors. "
                "Apply a structured diagnose-and-retry flow."
            )
        if not content:
            content = (
                f"## {title}\n\n"
                "1. Re-read tool arguments and verify required fields are present and correctly typed.\n"
                "2. Validate path/session/URL preconditions before retrying.\n"
                "3. Retry once with corrected arguments and reduced scope.\n"
                "4. If it still fails, summarize the exact root error and propose the next best fallback tool.\n\n"
                f"Observed failure pattern: `{reason[:180]}`\n\n"
                "**Anti-pattern:** repeating identical tool calls without adjusting arguments or preconditions.\n"
            )
        skill_file.write_text(
            (
                "---\n"
                f"description: {description}\n"
                "metadata:\n"
                "  annolid:\n"
                "    user_invocable: false\n"
                "---\n\n"
                f"{content}"
            ),
            encoding="utf-8",
        )
        return name

    def _build_skill_with_llm(
        self,
        *,
        tool_name: str,
        reason: str,
        default_name: str,
    ) -> dict[str, str] | None:
        if not self.llm_evolver_enabled:
            return None
        raw = self._call_llm_evolver(
            tool_name=tool_name,
            reason=reason,
            default_name=default_name,
        )
        if not raw:
            return None
        payload = self._parse_llm_skill_payload(raw)
        if payload is None:
            return None
        out: dict[str, str] = {}
        for key in ("name", "description", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
        return out or None

    def _call_llm_evolver(
        self,
        *,
        tool_name: str,
        reason: str,
        default_name: str,
    ) -> str:
        client = self._llm_client
        if client is None:
            client = self._build_default_llm_client()
        prompt = (
            "You are generating one Annolid SKILL.md body for repeated tool failures.\n"
            "Return only valid JSON object with keys: name, description, content.\n"
            "Rules:\n"
            "- name: lowercase hyphen slug\n"
            "- description: one sentence\n"
            "- content: markdown with heading, 3-6 steps, anti-pattern section\n\n"
            f"Tool: {tool_name}\n"
            f"Failure: {reason[:220]}\n"
            f"Default name: {default_name}\n"
        )

        def _invoke() -> str:
            if hasattr(client, "chat_complete"):
                return str(client.chat_complete(prompt))
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    model=self._llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=800,
                )
                return str(response.choices[0].message.content or "")
            raise RuntimeError("Unsupported LLM evolver client")

        fut = self._executor.submit(_invoke)
        try:
            return str(fut.result(timeout=self._llm_timeout_s) or "")
        except FutureTimeoutError:
            fut.cancel()
            return ""
        except Exception:
            return ""

    @staticmethod
    def _parse_llm_skill_payload(raw: str) -> Mapping[str, Any] | None:
        text = str(raw or "").strip()
        if not text:
            return None
        candidates = [text]
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                candidates.insert(0, match.group(1).strip())
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0).strip())
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, Mapping):
                return parsed
        return None

    @staticmethod
    def _build_default_llm_client() -> Any:
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set for LLM evolver")
        from openai import OpenAI

        base_url = str(os.getenv("OPENAI_BASE_URL", "")).strip() or None
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
