import logging
from typing import List, Optional
from annolid.interfaces.memory.registry import get_memory_service, get_retrieval_service
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource
from annolid.domain.memory.models import MemoryHit

logger = logging.getLogger(__name__)


class WorkflowMemoryAdapter:
    """Adapter for storing and retrieving workflow intelligence (recipes, past errors)."""

    def __init__(self, workspace_id: str, project_id: Optional[str] = None):
        self.workspace_id = workspace_id
        self.project_id = project_id

    @property
    def _scope(self) -> str:
        if self.project_id:
            return MemoryScope.project(self.project_id)
        return MemoryScope.workspace(self.workspace_id)

    def store_recipe(
        self, name: str, steps: str, tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Stores a successful analysis or operation recipe for future recall."""
        service = get_memory_service()
        if not service:
            return None

        metadata = {"recipe_name": name, "type": "recipe"}
        return service.store_memory(
            text=f"Recipe '{name}':\n{steps}",
            scope=self._scope,
            category=MemoryCategory.WORKFLOW_RECIPE,
            source=MemorySource.WORKFLOW,
            importance=0.8,
            tags=tags,
            metadata=metadata,
            dedupe=True,
        )

    def store_troubleshooting_context(
        self, error_message: str, resolution: str, tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """Stores an error and its paired resolution to help the agent auto-fix future runs."""
        service = get_memory_service()
        if not service:
            return None

        metadata = {"type": "troubleshooting", "error": error_message}
        return service.store_memory(
            text=f"Error encountered:\n{error_message}\n\nResolution:\n{resolution}",
            scope=self._scope,
            category=MemoryCategory.TROUBLESHOOTING,
            source=MemorySource.WORKFLOW,
            importance=0.9,
            tags=tags,
            metadata=metadata,
            dedupe=True,
        )

    def retrieve_recipes(self, query: str = "", top_k: int = 3) -> List[MemoryHit]:
        """Finds relevant past workflow recipes based on a query."""
        retrieval = get_retrieval_service()
        if not retrieval:
            return []

        return retrieval.search_memory(
            query=query,
            top_k=top_k,
            scope=self._scope,
            filters={"category": MemoryCategory.WORKFLOW_RECIPE},
        )

    def check_past_errors(self, current_error: str, top_k: int = 1) -> List[MemoryHit]:
        """Searches past troubleshooting memory to find a resolution for the current error."""
        retrieval = get_retrieval_service()
        if not retrieval:
            return []

        return retrieval.search_memory(
            query=current_error,
            top_k=top_k,
            scope=self._scope,
            filters={"category": MemoryCategory.TROUBLESHOOTING},
        )

    def pre_run_hook(self, target_task: str) -> str:
        """
        Retrieval-aware startup hook.
        Fetches best recipes and any recent relevant gotchas before starting a workflow.
        """
        recipes = self.retrieve_recipes(query=target_task, top_k=2)
        gotchas = self.check_past_errors(
            current_error=target_task, top_k=2
        )  # Find errors related to this task name

        context_parts = []
        if recipes:
            context_parts.append("### Recommended Workflow Recipes")
            for hit in recipes:
                context_parts.append(f"- {hit.text}")

        if gotchas:
            context_parts.append("\n### Past Troubleshooting / Gotchas")
            for hit in gotchas:
                context_parts.append(f"- {hit.text}")

        return "\n".join(context_parts)
