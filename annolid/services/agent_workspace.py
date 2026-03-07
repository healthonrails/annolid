"""Service-layer orchestration for agent workspace admin commands."""

from __future__ import annotations

import datetime
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
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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


__all__ = [
    "add_agent_feedback",
    "flush_agent_memory",
    "inspect_agent_memory",
    "inspect_agent_skills",
    "refresh_agent_skills",
    "shadow_agent_skills",
]
