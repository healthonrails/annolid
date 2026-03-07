"""Migration helpers for memory import and re-embedding workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from annolid.domain.memory.models import MemoryRecord
from annolid.domain.memory.protocols import MemoryBackend
from annolid.domain.memory.scopes import MemoryCategory, MemoryScope, MemorySource


@dataclass(frozen=True)
class MigrationResult:
    imported: int = 0
    skipped: int = 0
    failed: int = 0


def import_records(
    backend: MemoryBackend, records: Iterable[MemoryRecord]
) -> MigrationResult:
    imported = 0
    failed = 0
    for record in records:
        try:
            memory_id = backend.add(record)
            if memory_id:
                imported += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    return MigrationResult(imported=imported, failed=failed)


def reembed_records(*, backend: MemoryBackend, scope: str | None = None) -> int:
    """Placeholder scaffold for explicit re-embedding command flows."""
    _ = (backend, scope)
    return 0


def collect_legacy_records(
    source_root: Path,
) -> tuple[list[MemoryRecord], dict[str, int]]:
    """
    Collect records from known legacy sources under a workspace/project root.

    Supported source patterns:
    - JSON files containing a top-level ``text`` field.
    - ``memory/MEMORY.md`` and ``memory/HISTORY.md`` markdown logs.
    - ``project.annolid.json|yaml|yml`` schema files.
    """
    records: list[MemoryRecord] = []
    stats = {"json": 0, "markdown": 0, "project_schema": 0}
    if not source_root.exists():
        return records, stats

    records.extend(_collect_json_memory_records(source_root, stats))
    records.extend(_collect_markdown_memory_records(source_root, stats))
    records.extend(_collect_project_schema_records(source_root, stats))
    return records, stats


def _collect_json_memory_records(
    source_root: Path, stats: dict[str, int]
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for file_path in source_root.rglob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or "text" not in data:
            continue
        text = str(data.get("text", "")).strip()
        if not text:
            continue
        records.append(
            MemoryRecord(
                text=text,
                scope=str(data.get("scope", MemoryScope.GLOBAL)),
                category=str(data.get("category", MemoryCategory.OTHER)),
                source=str(data.get("source", MemorySource.IMPORT)),
                importance=float(data.get("importance", 0.5)),
                tags=list(data.get("tags", []) or []),
                metadata={
                    "legacy_source": "json",
                    "legacy_path": str(file_path),
                    "metadata": data.get("metadata", {}),
                },
            )
        )
        stats["json"] += 1
    return records


def _collect_markdown_memory_records(
    source_root: Path, stats: dict[str, int]
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for md_path in source_root.rglob("memory/MEMORY.md"):
        records.extend(_extract_markdown_lines(md_path, MemoryCategory.FACT, stats))
    for md_path in source_root.rglob("memory/HISTORY.md"):
        records.extend(
            _extract_markdown_lines(md_path, MemoryCategory.TROUBLESHOOTING, stats)
        )
    return records


def _extract_markdown_lines(
    md_path: Path, category: str, stats: dict[str, int]
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    workspace_id = md_path.parent.parent.name
    scope = MemoryScope.workspace(workspace_id)
    for idx, raw_line in enumerate(
        md_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if len(line) < 8:
            continue
        records.append(
            MemoryRecord(
                text=line,
                scope=scope,
                category=category,
                source=MemorySource.IMPORT,
                importance=0.5,
                metadata={
                    "legacy_source": "markdown",
                    "legacy_path": str(md_path),
                    "line_number": idx,
                },
            )
        )
        stats["markdown"] += 1
    return records


def _collect_project_schema_records(
    source_root: Path, stats: dict[str, int]
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    patterns = ("project.annolid.json", "project.annolid.yaml", "project.annolid.yml")
    for pattern in patterns:
        for path in source_root.rglob(pattern):
            payload = _load_structured_file(path)
            if not isinstance(payload, dict):
                continue
            project_id = path.parent.name
            category_count = len(payload.get("categories", []) or [])
            behavior_count = len(payload.get("behaviors", []) or [])
            text = (
                f"Project schema for '{project_id}': "
                f"{category_count} categories, {behavior_count} behaviors."
            )
            records.append(
                MemoryRecord(
                    text=text,
                    scope=MemoryScope.project(project_id),
                    category=MemoryCategory.PROJECT_SCHEMA,
                    source=MemorySource.IMPORT,
                    importance=0.8,
                    metadata={
                        "legacy_source": "project_schema",
                        "legacy_path": str(path),
                        "project_schema_name": payload.get("name"),
                    },
                )
            )
            stats["project_schema"] += 1
    return records


def _load_structured_file(path: Path) -> object:
    if path.suffix.lower() == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
