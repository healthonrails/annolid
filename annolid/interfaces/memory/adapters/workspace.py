import logging
from typing import Any, Dict, List, Optional
from annolid.interfaces.memory.registry import get_memory_service, get_retrieval_service
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource
from annolid.interfaces.memory.adapters.settings_model import SettingsSnapshot

logger = logging.getLogger(__name__)


class WorkspaceMemoryAdapter:
    """Adapter for Annolid workspace-level settings memory."""

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id

    def store_settings_snapshot(
        self,
        description: str,
        settings_dict: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Stores a recoverable settings snapshot summary for the workspace."""
        service = get_memory_service()
        if not service:
            return None

        metadata = {"settings": settings_dict}
        if context:
            metadata["context"] = context

        return service.store_memory(
            text=description,
            scope=MemoryScope.workspace(self.workspace_id),
            category=MemoryCategory.SETTING,
            source=MemorySource.SETTINGS,
            importance=0.6,
            metadata=metadata,
            dedupe=True,
        )

    def retrieve_settings_snapshots(
        self, query: str = "", top_k: int = 5
    ) -> List[SettingsSnapshot]:
        """Retrieves past settings snapshots relevant to the query."""
        retrieval = get_retrieval_service()
        if not retrieval:
            return []

        hits = retrieval.search_memory(
            query=query,
            top_k=top_k,
            scope=MemoryScope.workspace(self.workspace_id),
            filters={"category": MemoryCategory.SETTING},
        )

        snapshots = []
        for hit in hits:
            settings_data = hit.metadata.get("settings", {})
            snapshot = SettingsSnapshot(
                id=hit.id,
                description=hit.text,
                settings=settings_data,
                timestamp_ms=hit.timestamp_ms,
                context=hit.metadata.get("context"),
                tags=hit.tags,
            )
            snapshots.append(snapshot)

        return snapshots
