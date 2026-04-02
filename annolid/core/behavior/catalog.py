from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .spec import BehaviorDefinition, ProjectSchema


def normalize_behavior_code(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized: List[str] = []
    previous_underscore = False
    for ch in text.lower():
        if ch.isalnum():
            normalized.append(ch)
            previous_underscore = False
        else:
            if not previous_underscore:
                normalized.append("_")
                previous_underscore = True
    return "".join(normalized).strip("_")


def _dedupe_strings(values: Iterable[Any]) -> List[str]:
    unique: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def behavior_catalog_entries(schema: Optional[ProjectSchema]) -> List[Dict[str, Any]]:
    if schema is None:
        return []
    entries: List[Dict[str, Any]] = []
    for behavior in schema.behaviors:
        entries.append(
            {
                "code": behavior.code,
                "name": behavior.name,
                "description": behavior.description or "",
                "category_id": behavior.category_id or "",
                "modifier_ids": list(behavior.modifier_ids or []),
                "key_binding": behavior.key_binding or "",
                "is_state": bool(behavior.is_state),
                "exclusive_with": list(behavior.exclusive_with or []),
            }
        )
    return entries


def find_behavior(
    schema: Optional[ProjectSchema], code: str
) -> Optional[BehaviorDefinition]:
    if schema is None:
        return None
    code_text = str(code or "").strip()
    if not code_text:
        return None
    behavior_map = schema.behavior_map()
    found = behavior_map.get(code_text)
    if found is not None:
        return found
    lowered = code_text.lower()
    for behavior in schema.behaviors:
        if behavior.code.lower() == lowered:
            return behavior
    return None


def upsert_behavior_definition(
    schema: ProjectSchema,
    *,
    code: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category_id: Optional[str] = None,
    modifier_ids: Optional[Sequence[str]] = None,
    key_binding: Optional[str] = None,
    is_state: Optional[bool] = None,
    exclusive_with: Optional[Sequence[str]] = None,
    replace_existing: bool = False,
) -> Tuple[ProjectSchema, bool, str]:
    normalized_code = normalize_behavior_code(code)
    if not normalized_code:
        return schema, False, "Behavior code is required."

    updated = deepcopy(schema)
    existing_index = next(
        (
            idx
            for idx, beh in enumerate(updated.behaviors)
            if beh.code == normalized_code
        ),
        None,
    )
    description_text = None if description is None else str(description).strip() or None
    category_text = None if category_id is None else str(category_id).strip() or None
    key_binding_text = None if key_binding is None else str(key_binding).strip() or None
    behavior = BehaviorDefinition(
        code=normalized_code,
        name=str(name or normalized_code).strip() or normalized_code,
        description=description_text,
        category_id=category_text,
        modifier_ids=_dedupe_strings(modifier_ids or []),
        key_binding=key_binding_text,
        is_state=bool(True if is_state is None else is_state),
        exclusive_with=_dedupe_strings(exclusive_with or []),
    )

    if existing_index is None:
        updated.behaviors.append(behavior)
        return updated, True, f"Created behavior '{normalized_code}'."

    if not replace_existing:
        return updated, False, f"Behavior '{normalized_code}' already exists."

    updated.behaviors[existing_index] = behavior
    return updated, True, f"Updated behavior '{normalized_code}'."


def update_behavior_definition(
    schema: ProjectSchema,
    *,
    code: str,
    updates: Dict[str, Any],
) -> Tuple[ProjectSchema, bool, str]:
    target = find_behavior(schema, code)
    if target is None:
        return schema, False, f"Behavior '{code}' not found."

    updated = deepcopy(schema)
    behavior_index = next(
        idx for idx, beh in enumerate(updated.behaviors) if beh.code == target.code
    )
    description = updates.get("description", target.description)
    category_id = updates.get("category_id", target.category_id)
    key_binding = updates.get("key_binding", target.key_binding)
    merged = asdict(target)
    merged.update(
        {
            "name": str(updates.get("name", target.name) or "").strip() or target.name,
            "description": None
            if description is None
            else str(description).strip() or None,
            "category_id": None
            if category_id is None
            else str(category_id).strip() or None,
            "modifier_ids": _dedupe_strings(
                updates.get("modifier_ids", target.modifier_ids) or []
            ),
            "key_binding": None
            if key_binding is None
            else str(key_binding).strip() or None,
            "is_state": bool(updates.get("is_state", target.is_state)),
            "exclusive_with": _dedupe_strings(
                updates.get("exclusive_with", target.exclusive_with) or []
            ),
        }
    )
    merged["code"] = target.code
    updated.behaviors[behavior_index] = BehaviorDefinition(**merged)
    return updated, True, f"Updated behavior '{target.code}'."


def delete_behavior_definition(
    schema: ProjectSchema,
    *,
    code: str,
) -> Tuple[ProjectSchema, bool, str]:
    target = find_behavior(schema, code)
    if target is None:
        return schema, False, f"Behavior '{code}' not found."
    updated = deepcopy(schema)
    updated.behaviors = [beh for beh in updated.behaviors if beh.code != target.code]
    return updated, True, f"Deleted behavior '{target.code}'."
