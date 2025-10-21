"""Project-level behavior schema shared across Annolid tools.

The schema captures behaviors, categories, modifiers, and subjects so that the
GUI and analytics can operate on a consistent definition set. This module
supports JSON (and optionally YAML) serialization for persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # YAML is optional; fall back to JSON-only when unavailable.
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore
    _HAS_YAML = False


SchemaVersion = "1.0"


@dataclass
class CategoryDefinition:
    id: str
    name: str
    color: Optional[str] = None  # CSS hex string
    description: Optional[str] = None


@dataclass
class ModifierDefinition:
    id: str
    name: str
    description: Optional[str] = None


@dataclass
class SubjectDefinition:
    id: str
    name: str
    description: Optional[str] = None


@dataclass
class BehaviorDefinition:
    code: str
    name: str
    description: Optional[str] = None
    category_id: Optional[str] = None
    modifier_ids: List[str] = field(default_factory=list)
    key_binding: Optional[str] = None
    is_state: bool = True
    exclusive_with: List[str] = field(default_factory=list)


@dataclass
class ProjectSchema:
    behaviors: List[BehaviorDefinition] = field(default_factory=list)
    categories: List[CategoryDefinition] = field(default_factory=list)
    modifiers: List[ModifierDefinition] = field(default_factory=list)
    subjects: List[SubjectDefinition] = field(default_factory=list)
    version: str = SchemaVersion

    def category_map(self) -> Dict[str, CategoryDefinition]:
        return {cat.id: cat for cat in self.categories}

    def modifier_map(self) -> Dict[str, ModifierDefinition]:
        return {mod.id: mod for mod in self.modifiers}

    def subject_map(self) -> Dict[str, SubjectDefinition]:
        return {sub.id: sub for sub in self.subjects}

    def behavior_map(self) -> Dict[str, BehaviorDefinition]:
        return {beh.code: beh for beh in self.behaviors}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def default_schema() -> ProjectSchema:
    """Return a minimal default schema."""
    default_category = CategoryDefinition(
        id="default", name="Default", color="#2E7D32")
    default_subject = SubjectDefinition(id="subject_1", name="Subject 1")
    return ProjectSchema(
        categories=[default_category],
        subjects=[default_subject],
        behaviors=[
            BehaviorDefinition(
                code="behavior_1",
                name="Behavior 1",
                description="Placeholder behavior",
                category_id=default_category.id,
                modifier_ids=[],
                key_binding="B",
            )
        ],
        modifiers=[],
    )


def _schema_to_dict(schema: ProjectSchema) -> Dict[str, object]:
    return {
        "version": schema.version,
        "categories": [vars(cat) for cat in schema.categories],
        "modifiers": [vars(mod) for mod in schema.modifiers],
        "subjects": [vars(subj) for subj in schema.subjects],
        "behaviors": [vars(beh) for beh in schema.behaviors],
    }


def _dict_to_schema(payload: Dict[str, object]) -> ProjectSchema:
    def _load_list(key: str, cls):
        items = payload.get(key, []) or []
        return [cls(**item) for item in items]

    schema = ProjectSchema(
        categories=_load_list("categories", CategoryDefinition),
        modifiers=_load_list("modifiers", ModifierDefinition),
        subjects=_load_list("subjects", SubjectDefinition),
        behaviors=_load_list("behaviors", BehaviorDefinition),
        version=str(payload.get("version", SchemaVersion)),
    )
    return schema


def load_schema(path: Path) -> ProjectSchema:
    """Load a schema from JSON or YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Project schema not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"} and _HAS_YAML:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid schema format in {path}")
    return _dict_to_schema(payload)


def save_schema(schema: ProjectSchema, path: Path) -> None:
    """Persist a schema to JSON or YAML based on file suffix."""
    payload = _schema_to_dict(schema)
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"} and _HAS_YAML:
        text = yaml.safe_dump(payload, sort_keys=False)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_schema(schema: ProjectSchema) -> List[str]:
    """Return warnings for schema inconsistencies (non-fatal)."""
    warnings: List[str] = []

    def _detect_duplicates(items: Iterable[str], label: str) -> None:
        seen = set()
        for item in items:
            if item in seen:
                warnings.append(f"Duplicate {label} identifier: {item}")
            else:
                seen.add(item)

    _detect_duplicates((cat.id for cat in schema.categories), "category")
    _detect_duplicates((mod.id for mod in schema.modifiers), "modifier")
    _detect_duplicates((subj.id for subj in schema.subjects), "subject")
    _detect_duplicates((beh.code for beh in schema.behaviors), "behavior")

    category_ids = {cat.id for cat in schema.categories}
    modifier_ids = {mod.id for mod in schema.modifiers}

    for behavior in schema.behaviors:
        if behavior.category_id and behavior.category_id not in category_ids:
            warnings.append(
                f"Behavior '{behavior.code}' references missing category '{behavior.category_id}'."
            )
        missing_mods = [
            mid for mid in behavior.modifier_ids if mid not in modifier_ids]
        if missing_mods:
            warnings.append(
                f"Behavior '{behavior.code}' references missing modifiers {missing_mods}."
            )

    return warnings


# ---------------------------------------------------------------------------
# Discovery utilities
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA_FILENAME = "project.annolid.json"


def find_schema_near_video(video_path: Path) -> Optional[Path]:
    """Search for a schema near a video file (video directory or output folder)."""
    video_path = Path(video_path)
    candidates: List[Path] = []
    # Search in the same directory as the video.
    candidates.append(video_path.parent / DEFAULT_SCHEMA_FILENAME)
    # Search in an output directory with same stem.
    candidates.append(video_path.with_suffix("") / DEFAULT_SCHEMA_FILENAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
