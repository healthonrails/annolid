"""Service-layer orchestration for agent workspace admin commands."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def refresh_agent_skills(*, workspace: str | None = None) -> dict:
    from annolid.core.agent.observability import emit_governance_event
    from annolid.core.agent.skills import AgentSkillsLoader
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    loader = AgentSkillsLoader(workspace=resolved_workspace)
    before = loader.list_skills(filter_unavailable=False)
    before_names = sorted(str(s.get("name") or "") for s in before)
    loader.refresh_snapshot()
    skills = loader.list_skills(filter_unavailable=False)
    after_names = sorted(str(s.get("name") or "") for s in skills)
    before_set = set(before_names)
    after_set = set(after_names)
    added = [name for name in after_names if name not in before_set]
    removed = [name for name in before_names if name not in after_set]
    payload = {
        "workspace": str(resolved_workspace),
        "refreshed": True,
        "count": len(skills),
        "names": [str(s.get("name") or "") for s in skills],
        "added": added,
        "removed": removed,
    }
    emit_governance_event(
        event_type="skills",
        action="refresh",
        outcome="ok",
        actor="operator",
        details={
            "workspace": str(resolved_workspace),
            "count_before": len(before_names),
            "count_after": len(after_names),
            "added": added,
            "removed": removed,
        },
    )
    return payload


def inspect_agent_skills(*, workspace: str | None = None) -> dict:
    from annolid.core.agent.skills import AgentSkillsLoader
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    loader = AgentSkillsLoader(workspace=resolved_workspace)
    skills = loader.list_skills(filter_unavailable=False)
    workspace_skills = [s for s in skills if str(s.get("source") or "") == "workspace"]
    invalid = []
    for row in workspace_skills:
        if bool(row.get("manifest_valid", True)):
            continue
        invalid.append(
            {
                "name": str(row.get("name") or ""),
                "path": str(row.get("path") or ""),
                "manifest_errors": list(row.get("manifest_errors") or []),
            }
        )
    return {
        "workspace": str(resolved_workspace),
        "workspace_skill_count": len(workspace_skills),
        "invalid_manifest_count": len(invalid),
        "invalid_skills": invalid,
    }


def _slugify_skill_name(raw: str) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return "imported-skill"
    if not re.match(r"^[a-z]", text):
        text = f"skill-{text}"
    return text[:80].strip("-") or "imported-skill"


def _extract_frontmatter_description(raw: str) -> str:
    for line in str(raw or "").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Imported external skill for structured task execution."


def _render_annolid_skill_doc(
    *,
    description: str,
    content: str,
    original_name: str,
    source_tag: str,
    category: str,
) -> str:
    desc = json.dumps(str(description or "").strip()[:500], ensure_ascii=False)
    src = json.dumps(str(source_tag or "").strip()[:80], ensure_ascii=False)
    orig = json.dumps(str(original_name or "").strip()[:120], ensure_ascii=False)
    cat = json.dumps(str(category or "general").strip()[:80], ensure_ascii=False)
    return (
        "---\n"
        f"description: {desc}\n"
        "metadata:\n"
        "  annolid:\n"
        "    user_invocable: false\n"
        f"    source: {src}\n"
        f"    original_name: {orig}\n"
        f"    category: {cat}\n"
        "---\n\n"
        f"{str(content or '').strip()}\n"
    )


def import_agent_skills_pack(
    *,
    workspace: str | None = None,
    source_dir: str | Path,
    overwrite: bool = False,
) -> dict:
    from annolid.core.agent.skills import AgentSkillsLoader
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    source_root = Path(source_dir).expanduser()
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Skills source directory not found: {source_root}")
    workspace_skills = resolved_workspace / "skills"
    workspace_skills.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_root.glob("*/SKILL.md"))
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    renamed: list[dict[str, str]] = []
    for src in source_files:
        try:
            raw = src.read_text(encoding="utf-8")
        except OSError:
            skipped.append({"source": str(src), "reason": "unreadable"})
            continue
        meta = AgentSkillsLoader._read_frontmatter(raw)
        body = AgentSkillsLoader._strip_frontmatter(raw)
        original_name = str(meta.get("name") or src.parent.name).strip()
        description = str(meta.get("description") or "").strip()
        category = str(meta.get("category") or "general").strip()
        if not description:
            description = _extract_frontmatter_description(body)
        target_name = _slugify_skill_name(original_name)
        rendered = _render_annolid_skill_doc(
            description=description,
            content=body,
            original_name=original_name,
            source_tag="metaclaw",
            category=category,
        )
        target_dir = workspace_skills / target_name
        target_file = target_dir / "SKILL.md"

        if target_file.exists() and not overwrite:
            try:
                current = target_file.read_text(encoding="utf-8")
            except OSError:
                current = ""
            if current == rendered:
                skipped.append({"source": str(src), "reason": "already_imported"})
                continue
            suffix = 2
            alt_name = f"{target_name}-metaclaw"
            alt_dir = workspace_skills / alt_name
            alt_file = alt_dir / "SKILL.md"
            while alt_file.exists():
                alt_name = f"{target_name}-metaclaw-{suffix}"
                alt_dir = workspace_skills / alt_name
                alt_file = alt_dir / "SKILL.md"
                suffix += 1
            renamed.append({"source_name": target_name, "target_name": alt_name})
            target_name = alt_name
            target_dir = alt_dir
            target_file = alt_file

        target_dir.mkdir(parents=True, exist_ok=True)
        target_file.write_text(rendered, encoding="utf-8")
        imported.append(
            {
                "source": str(src),
                "target_name": target_name,
                "target_path": str(target_file),
            }
        )

    return {
        "workspace": str(resolved_workspace),
        "source_dir": str(source_root),
        "source_skill_count": len(source_files),
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "renamed_count": len(renamed),
        "imported": imported,
        "skipped": skipped,
        "renamed": renamed,
    }


def shadow_agent_skills(
    *, workspace: str | None = None, candidate_pack: str | Path
) -> dict:
    from annolid.core.agent.skill_registry import (
        compare_skill_pack_shadow,
        flatten_skills_by_name,
    )
    from annolid.core.agent.skills import AgentSkillsLoader
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    loader = AgentSkillsLoader(workspace=resolved_workspace)
    active = flatten_skills_by_name(loader.list_skills(filter_unavailable=False))
    payload = compare_skill_pack_shadow(
        active_skills=active,
        candidate_pack_dir=Path(candidate_pack),
    )
    payload["workspace"] = str(resolved_workspace)
    return payload


def add_agent_feedback(
    *,
    workspace: str | None = None,
    session_id: str = "default",
    rating: int,
    comment: str = "",
    trace_id: str = "",
    expected_substring: str = "",
) -> dict:
    from annolid.core.agent.eval.telemetry import RunTraceStore
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    store = RunTraceStore(resolved_workspace)
    row = store.capture_feedback(
        session_id=str(session_id or "default"),
        rating=int(rating),
        comment=str(comment or ""),
        trace_id=str(trace_id or ""),
        expected_substring=str(expected_substring or ""),
    )
    return {
        "workspace": str(resolved_workspace),
        "saved": bool(row),
        "feedback": row,
    }


def flush_agent_memory(
    *,
    workspace: str | None = None,
    session_id: str | None = None,
    note: str | None = None,
) -> dict:
    from annolid.core.agent.memory import AgentMemoryStore
    from annolid.core.agent.observability import emit_governance_event
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    store = AgentMemoryStore(resolved_workspace)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note_text = str(note or "").strip() or "operator memory flush"
    session_text = str(session_id or "").strip()
    entry = (
        f"[{stamp}] {note_text}"
        if not session_text
        else f"[{stamp}] {note_text} (session_id={session_text})"
    )
    store.append_today(entry)
    store.append_history(entry)
    payload = {
        "workspace": str(resolved_workspace),
        "flushed": True,
        "today_file": str(store.get_today_file()),
        "history_file": str(store.history_file),
        "entry": entry,
    }
    emit_governance_event(
        event_type="memory",
        action="operator_flush",
        outcome="ok",
        actor="operator",
        details={
            "workspace": str(resolved_workspace),
            "session_id": session_text,
            "entry_chars": len(entry),
        },
    )
    return payload


def inspect_agent_memory(*, workspace: str | None = None) -> dict:
    from annolid.core.agent.memory import AgentMemoryStore
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    store = AgentMemoryStore(resolved_workspace)
    return {
        "workspace": str(resolved_workspace),
        "memory_dir": str(store.memory_dir),
        "retrieval_plugin": store.retrieval_plugin_name,
        "today_file": str(store.get_today_file()),
        "long_term_file": str(store.memory_file),
        "history_file": str(store.history_file),
    }


def inspect_agent_meta_learning(
    *,
    workspace: str | None = None,
    limit: int = 20,
    brief: bool = False,
) -> dict:
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    meta_dir = resolved_workspace / "memory" / "meta_learning"
    events_path = meta_dir / "events.jsonl"
    patterns_path = meta_dir / "failure_patterns.json"
    history_path = meta_dir / "evolution_history.jsonl"
    pending_path = meta_dir / "pending_evolution_jobs.json"
    skills_dir = resolved_workspace / "skills"
    n = max(1, int(limit))

    events: list[dict] = []
    events_count = 0
    if events_path.exists():
        try:
            tail: list[dict] = []
            with events_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    row = line.strip()
                    if not row:
                        continue
                    events_count += 1
                    try:
                        parsed = json.loads(row)
                    except Exception:
                        continue
                    if isinstance(parsed, dict):
                        tail.append(parsed)
                        if len(tail) > n:
                            tail = tail[-n:]
            events = tail
        except OSError:
            events = []
            events_count = 0
    recent_events = events[-n:]

    patterns: dict[str, int] = {}
    if patterns_path.exists():
        try:
            loaded = json.loads(patterns_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    try:
                        patterns[str(key)] = int(value)
                    except Exception:
                        continue
        except Exception:
            patterns = {}
    top_patterns = [
        {"signature": key, "count": count}
        for key, count in sorted(patterns.items(), key=lambda row: (-row[1], row[0]))[
            :n
        ]
    ]

    evolved_skills: list[str] = []
    if skills_dir.exists():
        for p in sorted(skills_dir.iterdir()):
            if p.is_dir() and p.name.startswith("meta-recover-"):
                evolved_skills.append(p.name)
    evolved_recent = evolved_skills[-n:]

    tool_counter: Counter[str] = Counter()
    for evt in recent_events:
        failures = evt.get("failures")
        if not isinstance(failures, list):
            continue
        for row in failures:
            if not isinstance(row, dict):
                continue
            tool = str(row.get("tool") or "").strip()
            if tool:
                tool_counter[tool] += 1
    top_failure_tools = [
        {"tool": tool, "count": count} for tool, count in tool_counter.most_common(n)
    ]
    evolution_rows = _read_jsonl_tail(history_path, n)
    recent_evolution_events = evolution_rows["rows"]
    evolution_events_count = int(evolution_rows["count"])
    pending_jobs_count = 0
    if pending_path.exists():
        try:
            pending_loaded = json.loads(pending_path.read_text(encoding="utf-8"))
            if isinstance(pending_loaded, dict):
                pending_jobs_count = len(pending_loaded)
        except Exception:
            pending_jobs_count = 0

    payload = {
        "workspace": str(resolved_workspace),
        "meta_learning_dir": str(meta_dir),
        "events_path": str(events_path),
        "patterns_path": str(patterns_path),
        "evolution_history_path": str(history_path),
        "pending_jobs_path": str(pending_path),
        "events_count": events_count,
        "recent_events": recent_events,
        "top_patterns": top_patterns,
        "evolved_skill_count": len(evolved_skills),
        "evolved_skills": evolved_recent,
        "top_failure_tools": top_failure_tools,
        "evolution_events_count": evolution_events_count,
        "recent_evolution_events": recent_evolution_events,
        "pending_jobs_count": pending_jobs_count,
    }
    if brief:
        payload.pop("events_path", None)
        payload.pop("patterns_path", None)
        payload.pop("evolution_history_path", None)
        payload.pop("pending_jobs_path", None)
        payload.pop("recent_events", None)
        payload.pop("evolved_skills", None)
        payload.pop("recent_evolution_events", None)
    return payload


def inspect_agent_meta_learning_history(
    *,
    workspace: str | None = None,
    limit: int = 20,
    full: bool = False,
) -> dict:
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    history_path = (
        resolved_workspace / "memory" / "meta_learning" / "evolution_history.jsonl"
    )
    n = max(1, int(limit))
    rows = _read_jsonl_tail(history_path, n)
    events = list(rows["rows"])
    trigger_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    skills_generated = 0
    for row in events:
        if not isinstance(row, dict):
            continue
        trig = str(row.get("trigger") or "").strip()
        if trig:
            trigger_counter[trig] += 1
        tool = str(row.get("tool") or "").strip()
        if tool:
            tool_counter[tool] += 1
        if str(row.get("skill_name") or "").strip():
            skills_generated += 1
        if not bool(full):
            skill = row.get("skill")
            if isinstance(skill, dict):
                compact_skill = dict(skill)
                compact_skill.pop("content_excerpt", None)
                row["skill"] = compact_skill
        else:
            skill = row.get("skill")
            if isinstance(skill, dict):
                excerpt = str(skill.get("content_excerpt") or "")
                if len(excerpt) > 4000:
                    skill["content_excerpt"] = excerpt[:4000]
    status_counter: Counter[str] = Counter()
    generated_names: list[str] = []
    for row in events:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip()
        if status:
            status_counter[status] += 1
        skill_name = str(row.get("skill_name") or "").strip()
        if skill_name:
            generated_names.append(skill_name)
    return {
        "workspace": str(resolved_workspace),
        "history_path": str(history_path),
        "events_count": int(rows["count"]),
        "skills_generated_in_window": int(skills_generated),
        "generated_skill_names": generated_names,
        "top_triggers": [
            {"trigger": key, "count": count}
            for key, count in trigger_counter.most_common(n)
        ],
        "top_tools": [
            {"tool": key, "count": count} for key, count in tool_counter.most_common(n)
        ],
        "top_statuses": [
            {"status": key, "count": count}
            for key, count in status_counter.most_common(n)
        ],
        "events": events,
    }


def run_agent_meta_learning_maintenance(
    *,
    workspace: str | None = None,
    force: bool = False,
    max_jobs: int | None = None,
) -> dict:
    from annolid.core.agent.meta_learning import AgentMetaLearner
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    learner = AgentMetaLearner(
        resolved_workspace,
        enabled=True,
    )
    payload = learner.run_idle_maintenance(
        force=bool(force),
        max_jobs=max_jobs,
    )
    payload["workspace"] = str(resolved_workspace)
    return payload


def inspect_agent_meta_learning_maintenance_status(
    *,
    workspace: str | None = None,
) -> dict:
    from annolid.core.agent.meta_learning import AgentMetaLearner
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    learner = AgentMetaLearner(
        resolved_workspace,
        enabled=True,
    )
    payload = learner.get_idle_maintenance_status()
    payload["workspace"] = str(resolved_workspace)
    return payload


def inspect_agent_meta_learning_next_window(
    *,
    workspace: str | None = None,
) -> dict:
    from annolid.core.agent.meta_learning import AgentMetaLearner
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    learner = AgentMetaLearner(
        resolved_workspace,
        enabled=True,
    )
    status = learner.get_idle_maintenance_status()
    eta_seconds = status.get("next_window_eta_seconds")
    next_window_at = None
    if isinstance(eta_seconds, (int, float)):
        try:
            next_window_at = (
                datetime.now(timezone.utc) + timedelta(seconds=float(eta_seconds))
            ).isoformat(timespec="seconds")
        except Exception:
            next_window_at = None
    return {
        "workspace": str(resolved_workspace),
        "window_open": bool(status.get("window_open")),
        "window_reason": str(status.get("window_reason") or ""),
        "next_window_eta_seconds": eta_seconds,
        "next_window_at": next_window_at,
        "scheduler_enabled": bool(status.get("scheduler_enabled")),
        "scheduler_state": str(status.get("scheduler_state") or ""),
    }


def _read_jsonl_tail(path: Path, limit: int) -> dict[str, object]:
    n = max(1, int(limit))
    rows: list[dict] = []
    count = 0
    if not path.exists():
        return {"count": 0, "rows": []}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                count += 1
                try:
                    parsed = json.loads(text)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
                    if len(rows) > n:
                        rows = rows[-n:]
    except OSError:
        return {"count": 0, "rows": []}
    return {"count": count, "rows": rows}


__all__ = [
    "add_agent_feedback",
    "flush_agent_memory",
    "import_agent_skills_pack",
    "inspect_agent_meta_learning",
    "inspect_agent_meta_learning_history",
    "inspect_agent_meta_learning_maintenance_status",
    "inspect_agent_meta_learning_next_window",
    "inspect_agent_memory",
    "inspect_agent_skills",
    "refresh_agent_skills",
    "run_agent_meta_learning_maintenance",
    "shadow_agent_skills",
]
