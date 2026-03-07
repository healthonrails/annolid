"""Conservative, explicit memory write helpers for workflow events."""

from __future__ import annotations

from typing import Any, Optional

from annolid.domain.memory.scopes import MemoryCategory, MemoryScope, MemorySource
from annolid.services.memory.memory_service import MemoryService


class PersistenceService:
    """Maps explicit UI/agent save actions into memory records."""

    def __init__(self, memory_service: MemoryService):
        self._memory_service = memory_service

    def save_project_note(
        self,
        project_id: str,
        text: str,
        *,
        importance: float = 0.8,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self._memory_service.store_memory(
            text=text,
            scope=MemoryScope.project(project_id),
            category=MemoryCategory.PROJECT_NOTE,
            source=MemorySource.PROJECT,
            importance=importance,
            metadata=metadata,
            dedupe=True,
        )

    def save_annotation_rule(
        self,
        dataset_id: str,
        text: str,
        *,
        importance: float = 0.9,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self._memory_service.store_memory(
            text=text,
            scope=MemoryScope.dataset(dataset_id),
            category=MemoryCategory.ANNOTATION_RULE,
            source=MemorySource.ANNOTATION,
            importance=importance,
            metadata=metadata,
            dedupe=True,
        )

    def save_settings_snapshot(
        self,
        *,
        scope: str,
        description: str,
        settings: dict[str, Any],
        context: str | None = None,
        importance: float = 0.6,
    ) -> str:
        metadata: dict[str, Any] = {"settings": settings}
        if context:
            metadata["context"] = context
        return self._memory_service.store_memory(
            text=description,
            scope=scope,
            category=MemoryCategory.SETTING,
            source=MemorySource.SETTINGS,
            importance=importance,
            metadata=metadata,
            dedupe=True,
        )
