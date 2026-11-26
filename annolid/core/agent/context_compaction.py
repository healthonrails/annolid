from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class ContextCompactionDecision:
    messages: Sequence[Dict[str, Any]]
    trimmed_count: int = 0
    skipped_due_to_active_tasks: bool = False


def has_unresolved_tool_tasks(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return True if any assistant tool_call lacks a matching tool result message."""
    pending_ids: set[str] = set()
    saw_idless_tool_call = False
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, Sequence):
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    call_id = str(call.get("id") or "").strip()
                    if call_id:
                        pending_ids.add(call_id)
                    else:
                        # Conservative fallback for malformed/raw tool call records.
                        saw_idless_tool_call = True
            continue
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if tool_call_id:
                pending_ids.discard(tool_call_id)
    return bool(pending_ids) or saw_idless_tool_call


def _drop_orphaned_tool_indices(
    materialized: Sequence[Mapping[str, Any]],
    kept_non_system: List[int],
) -> List[int]:
    """Ensure compacted history never starts with or retains orphaned `tool` messages.

    OpenAI, Anthropic, and Codex APIs reject message histories where a `tool`
    result appears without its originating `assistant` message containing the
    matching `tool_calls` entry.
    """
    if not kept_non_system:
        return []

    # Drop any leading `tool` messages whose parent assistant message was trimmed.
    trimmed = list(kept_non_system)
    while trimmed:
        first_role = str(materialized[trimmed[0]].get("role") or "").strip().lower()
        if first_role != "tool":
            break
        trimmed.pop(0)

    if not trimmed:
        return []

    # Also filter any `tool` message whose tool_call_id has no preceding kept
    # assistant tool_call (only when IDs are present on the tool message).
    seen_assistant_call_ids: set[str] = set()
    saw_assistant_with_calls = False
    cleaned: List[int] = []
    for idx in trimmed:
        msg = materialized[idx]
        role = str(msg.get("role") or "").strip().lower()
        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, Sequence) and tool_calls:
                saw_assistant_with_calls = True
                for call in tool_calls:
                    if isinstance(call, Mapping):
                        call_id = str(call.get("id") or "").strip()
                        if call_id:
                            seen_assistant_call_ids.add(call_id)
            cleaned.append(idx)
            continue
        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "").strip()
            if call_id:
                if call_id not in seen_assistant_call_ids:
                    continue
            elif not saw_assistant_with_calls:
                continue
            cleaned.append(idx)
            continue
        cleaned.append(idx)
    return cleaned


def plan_context_compaction(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_messages: int = 48,
    max_non_system_messages: int = 36,
) -> ContextCompactionDecision:
    """Trim older non-system messages while keeping system prompts and tool pairs intact."""
    materialized = [dict(msg) for msg in messages]
    total_count = len(materialized)
    max_total = max(1, int(max_messages))
    max_non_system = max(1, int(max_non_system_messages))

    if total_count <= max_total:
        return ContextCompactionDecision(messages=materialized, trimmed_count=0)
    if has_unresolved_tool_tasks(materialized):
        # Do not compact while there are unresolved assistant->tool links.
        return ContextCompactionDecision(
            messages=materialized,
            trimmed_count=0,
            skipped_due_to_active_tasks=True,
        )

    system_indices = [
        idx
        for idx, msg in enumerate(materialized)
        if str(msg.get("role") or "").strip().lower() == "system"
    ]
    system_set = set(system_indices)
    non_system_indices = [idx for idx in range(total_count) if idx not in system_set]
    kept_non_system = non_system_indices[-max_non_system:]

    system_budget = max(0, max_total - len(kept_non_system))
    kept_system: List[int] = []
    if system_indices and system_budget > 0:
        if len(system_indices) <= system_budget:
            kept_system = list(system_indices)
        elif system_budget == 1:
            kept_system = [system_indices[0]]
        else:
            first = system_indices[0]
            tail = system_indices[-(system_budget - 1) :]
            kept_system = [first] + [idx for idx in tail if idx != first]
    elif system_indices and max_total > 1:
        # Preserve the first system message when possible, then keep newest context.
        kept_system = [system_indices[0]]
        kept_non_system = kept_non_system[-(max_total - 1) :]
    elif system_indices and max_total == 1:
        kept_system = [system_indices[0]]
        kept_non_system = []

    kept_non_system = _drop_orphaned_tool_indices(materialized, kept_non_system)
    if not kept_non_system and non_system_indices and max_total > len(kept_system):
        # Fallback if the entire kept window consisted of tool messages from one
        # huge call: keep the most recent non-tool message so the turn stays valid.
        for idx in reversed(non_system_indices):
            if str(materialized[idx].get("role") or "").strip().lower() != "tool":
                kept_non_system = [idx]
                break

    kept_indices = sorted(set(kept_system + kept_non_system))
    compacted = [materialized[idx] for idx in kept_indices]
    if len(compacted) > max_total:
        compacted = compacted[-max_total:]
    trimmed = max(0, total_count - len(compacted))
    return ContextCompactionDecision(
        messages=compacted,
        trimmed_count=trimmed,
        skipped_due_to_active_tasks=False,
    )


__all__ = [
    "ContextCompactionDecision",
    "has_unresolved_tool_tasks",
    "plan_context_compaction",
]
