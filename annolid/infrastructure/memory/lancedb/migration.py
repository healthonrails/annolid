"""Migration helpers for memory import and re-embedding workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from annolid.domain.memory.models import MemoryRecord
from annolid.domain.memory.protocols import MemoryBackend


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
