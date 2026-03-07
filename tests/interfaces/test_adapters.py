import pytest
from annolid.domain.memory.scopes import MemoryCategory
from annolid.interfaces.memory.adapters.bot import BotMemoryAdapter
from annolid.interfaces.memory.adapters.annotation import AnnotationMemoryAdapter
from annolid.interfaces.memory.adapters.project import ProjectMemoryAdapter


@pytest.fixture(autouse=True)
def mock_registry(monkeypatch):
    class DummyHit:
        def __init__(self, text, score, category):
            self.text = text
            self.score = score
            self.category = category

    class DummyRetrieval:
        def list_memories(self):
            return []

        def search_memory(self, query, top_k, scope, filters=None):
            if filters and filters.get("category") == MemoryCategory.ANNOTATION_RULE:
                return [DummyHit("rule 1", 0.9, MemoryCategory.ANNOTATION_RULE)]
            return [DummyHit("mock result", 0.8, MemoryCategory.FACT)]

    class DummyContext:
        _retrieval_service = DummyRetrieval()

        def build_project_context(self, project_id, top_k):
            return f"Project {project_id} context"

        def build_annotation_context(self, dataset_id, top_k):
            return f"Annotation {dataset_id} context"

    class DummyService:
        def store_memory(self, *args, **kwargs):
            return "mem_123"

    import annolid.interfaces.memory.registry as reg

    # Must override the module-level variables since get_* functions use them directly
    # instead of calling _initialize_subsystem again if they are set
    reg._memory_service = DummyService()
    reg._context_service = DummyContext()
    reg._retrieval_service = DummyRetrieval()
    monkeypatch.setattr(reg, "_initialize_subsystem", lambda: None)


def test_bot_adapter():
    adapter = BotMemoryAdapter(agent_id="bot1")
    rid = adapter.store_chat_memory("Hello", importance=0.5)
    assert rid == "mem_123"
    ctx = adapter.get_chat_context("query")
    assert "mock result" in ctx


def test_annotation_adapter():
    adapter = AnnotationMemoryAdapter()
    rid = adapter.store_annotation_rule("ds1", "rule text")
    assert rid == "mem_123"
    ctx = adapter.get_annotation_context("ds1")
    assert "Annotation ds1 context" in ctx


def test_project_adapter():
    adapter = ProjectMemoryAdapter()
    rid = adapter.store_project_note("proj1", "note text")
    assert rid == "mem_123"
    ctx = adapter.get_project_context("proj1")
    assert "Project proj1 context" in ctx
