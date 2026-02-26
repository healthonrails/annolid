from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping


def _list_skill_files(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not root.exists() or not root.is_dir():
        return out
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        skill_file = child / "SKILL.md"
        if skill_file.exists():
            out[child.name] = str(skill_file)
    return out


def compare_skill_pack_shadow(
    *,
    active_skills: Mapping[str, str],
    candidate_pack_dir: Path,
) -> Dict[str, Any]:
    active = {str(k): str(v) for k, v in dict(active_skills).items()}
    candidate = _list_skill_files(Path(candidate_pack_dir).expanduser().resolve())
    active_names = set(active.keys())
    candidate_names = set(candidate.keys())
    return {
        "candidate_pack_dir": str(Path(candidate_pack_dir).expanduser().resolve()),
        "active_count": len(active_names),
        "candidate_count": len(candidate_names),
        "added": sorted(name for name in candidate_names if name not in active_names),
        "overridden": sorted(
            name
            for name in candidate_names
            if name in active_names and candidate.get(name) != active.get(name)
        ),
        "missing": sorted(name for name in active_names if name not in candidate_names),
    }


def flatten_skills_by_name(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in rows:
        name = str(row.get("name") or "").strip()
        path = str(row.get("path") or "").strip()
        if name and path:
            out[name] = path
    return out
