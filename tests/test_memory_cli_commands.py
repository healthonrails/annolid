from annolid.domain.memory.models import MemoryHit
from annolid.interfaces.memory import cli as memory_cli


def test_memory_cli_stats_command_prints_json(monkeypatch, capsys) -> None:
    class DummyService:
        def stats(self, scope=None):
            return {"count": 2, "scope": scope}

    monkeypatch.setattr(memory_cli, "get_memory_service", lambda: DummyService())
    rc = memory_cli.main(["stats", "--scope", "global"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"count": 2' in out


def test_memory_cli_search_command_prints_hits(monkeypatch, capsys) -> None:
    class DummyRetrieval:
        def search_memory(self, query, top_k, scope=None):
            _ = (query, top_k, scope)
            return [
                MemoryHit(
                    id="m1",
                    text="Use tail_base",
                    score=0.9,
                    scope="dataset:1",
                    category="annotation_rule",
                    source="annotation",
                    importance=0.8,
                    timestamp_ms=0,
                    tags=[],
                    metadata={},
                )
            ]

    monkeypatch.setattr(memory_cli, "get_retrieval_service", lambda: DummyRetrieval())
    rc = memory_cli.main(["search", "tail_base"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Use tail_base" in out


def test_memory_cli_cleanup_command_deletes_matching_hits(monkeypatch, capsys) -> None:
    class DummyBackend:
        def __init__(self):
            self.deleted = []

        def list_memories(self, top_k=10, scope=None):
            _ = (top_k, scope)
            return [
                MemoryHit(
                    id="old",
                    text="old note",
                    score=0.1,
                    scope="global",
                    category="project_note",
                    source="project",
                    importance=0.1,
                    timestamp_ms=0,
                    tags=[],
                    metadata={},
                )
            ]

        def delete(self, memory_id):
            self.deleted.append(memory_id)
            return True

    backend = DummyBackend()
    monkeypatch.setattr(memory_cli, "get_memory_backend", lambda: backend)
    rc = memory_cli.main(["cleanup", "global", "--older_than_days", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert backend.deleted == ["old"]
    assert "cleaned up 1" in out.lower()


def test_memory_cli_delete_command_returns_error_when_not_found(
    monkeypatch, capsys
) -> None:
    class DummyService:
        def delete_memory(self, memory_id):
            _ = memory_id
            return False

    monkeypatch.setattr(memory_cli, "get_memory_service", lambda: DummyService())
    rc = memory_cli.main(["delete", "missing"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed to delete" in out
