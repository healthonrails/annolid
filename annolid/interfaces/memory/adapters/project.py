import logging
from typing import Optional
from annolid.interfaces.memory.registry import get_memory_service, get_context_service
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource

logger = logging.getLogger(__name__)


class ProjectMemoryAdapter:
    """Adapter for Annolid project-scoped memory."""

    def store_project_note(
        self, project_id: str, text: str, importance: float = 0.8
    ) -> Optional[str]:
        service = get_memory_service()
        if not service:
            return None

        return service.store_memory(
            text=text,
            scope=MemoryScope.project(project_id),
            category=MemoryCategory.PROJECT_NOTE,
            source=MemorySource.PROJECT,
            importance=importance,
        )

    def get_project_context(self, project_id: str, top_k: int = 5) -> str:
        service = get_context_service()
        if not service:
            return ""

        return service.build_project_context(project_id=project_id, top_k=top_k)
