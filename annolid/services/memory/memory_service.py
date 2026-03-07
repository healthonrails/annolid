import logging
import time
from typing import Any, Dict, List, Optional
from annolid.domain.memory.models import MemoryRecord
from annolid.domain.memory.protocols import MemoryBackend
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource


logger = logging.getLogger(__name__)


class MemoryService:
    """Orchestrates storing, updating, and deleting memory records."""

    def __init__(self, backend: MemoryBackend):
        self._backend = backend

    def store_memory(
        self,
        text: str,
        scope: str = MemoryScope.GLOBAL,
        category: str = MemoryCategory.OTHER,
        source: str = MemorySource.SYSTEM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dedupe: bool = True,
    ) -> str:
        """Stores a generic memory record. Optionally dedupes identical text within the same scope/category."""
        cleaned_text = text.strip()
        tags = tags or []
        metadata = metadata or {}
        if not cleaned_text:
            raise ValueError("Memory text must not be empty.")

        if dedupe:
            # Check for existing exact matches to avoid spamming the DB
            existing_hits = self._backend.search(
                query=cleaned_text, top_k=5, scope=scope, filters={"category": category}
            )
            for hit in existing_hits:
                # If the text is identical, we just bump the timestamp and importance instead of duplicating
                if hit.text == cleaned_text:
                    new_importance = max(hit.importance, importance)
                    self._backend.update(
                        memory_id=hit.id,
                        patch={
                            "timestamp_ms": int(time.time() * 1000),
                            "importance": new_importance,
                            "tags": list(dict.fromkeys([*hit.tags, *tags])),
                        },
                    )
                    return hit.id

        record = MemoryRecord(
            text=cleaned_text,
            scope=scope,
            category=category,
            source=source,
            importance=importance,
            tags=tags,
            metadata=metadata,
        )
        return self._backend.add(record)

    def store_project_note(
        self,
        project_id: str,
        text: str,
        importance: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method to store a project-specific note."""
        return self.store_memory(
            text=text,
            scope=MemoryScope.project(project_id),
            category=MemoryCategory.PROJECT_NOTE,
            source=MemorySource.PROJECT,
            importance=importance,
            metadata=metadata,
        )

    def store_annotation_rule(
        self,
        dataset_id: str,
        rule_text: str,
        importance: float = 0.9,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method to store an annotation rule for a dataset."""
        return self.store_memory(
            text=rule_text,
            scope=MemoryScope.dataset(dataset_id),
            category=MemoryCategory.ANNOTATION_RULE,
            source=MemorySource.ANNOTATION,
            importance=importance,
            metadata=metadata,
        )

    def store_setting_memory(
        self, scope: str, description: str, settings_dict: Dict[str, Any]
    ) -> str:
        """Stores a recoverable settings snapshot summary."""
        return self.store_memory(
            text=description,
            scope=scope,
            category=MemoryCategory.SETTING,
            source=MemorySource.SETTINGS,
            importance=0.6,
            metadata={"settings": settings_dict},
        )

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a memory by ID."""
        return self._backend.delete(memory_id)

    def update_memory(self, memory_id: str, patch: Dict[str, Any]) -> bool:
        """Updates fields of an existing memory."""
        return self._backend.update(memory_id, patch)

    def stats(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """Returns statistics for the memory backend."""
        return self._backend.stats(scope=scope)

    def health_check(self) -> Dict[str, Any]:
        """Returns health status of the memory backend."""
        return self._backend.health_check()
