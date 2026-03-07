import json
import builtins

from scripts import export_memory, migrate_memory, reembed_memory


def test_export_memory_script_export_calls_backend(tmp_path, monkeypatch) -> None:
    captured = {}

    class DummyBackend:
        def export_jsonl(self, output_file):
            captured["output_file"] = output_file
            output_file.write_text('{"text":"hello"}\n', encoding="utf-8")
            return 1

    monkeypatch.setattr(export_memory, "get_memory_backend", lambda: DummyBackend())

    output = tmp_path / "memory.jsonl"
    assert export_memory.export_memories(output) == 1
    assert captured["output_file"] == output
    assert output.exists()


def test_export_memory_script_import_reads_jsonl_contract(
    tmp_path, monkeypatch
) -> None:
    stored = []

    class DummyService:
        def store_memory(self, **kwargs):
            stored.append(kwargs)
            return "mem_1"

    monkeypatch.setattr(export_memory, "get_memory_service", lambda: DummyService())

    input_file = tmp_path / "memory.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "legacy-id",
                "text": "Use Cutie",
                "scope": "project:1",
                "category": "project_note",
                "source": "project",
                "importance": 0.8,
                "tags": ["tracking"],
                "metadata_json": json.dumps({"project_id": "1"}),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert export_memory.import_memories(input_file) == 1
    assert stored[0]["text"] == "Use Cutie"
    assert stored[0]["scope"] == "project:1"
    assert stored[0]["category"] == "project_note"
    assert stored[0]["source"] == "project"
    assert stored[0]["tags"] == ["tracking"]
    assert stored[0]["metadata"] == {"project_id": "1"}


def test_migrate_memory_script_migrates_valid_json_only(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "legacy"
    source_dir.mkdir()
    (source_dir / "good.json").write_text(
        json.dumps({"text": "Project note", "scope": "project:1", "importance": 0.7}),
        encoding="utf-8",
    )
    (source_dir / "bad.json").write_text(
        json.dumps({"scope": "project:1"}), encoding="utf-8"
    )

    class DummyBackend:
        pass

    captured = {}

    def fake_import_records(backend, records):
        captured["backend"] = backend
        captured["records"] = records

        class Result:
            imported = 1
            failed = 0

        return Result()

    monkeypatch.setattr(migrate_memory, "get_memory_backend", lambda: DummyBackend())
    monkeypatch.setattr(migrate_memory, "import_records", fake_import_records)

    assert migrate_memory.migrate_legacy_memories(source_dir) == 0
    assert len(captured["records"]) == 1
    assert captured["records"][0].text == "Project note"
    assert captured["records"][0].metadata["legacy_source"] == "json"
    assert captured["records"][0].metadata["legacy_path"].endswith("good.json")


def test_reembed_memory_script_runs_backend_reembed(monkeypatch) -> None:
    class DummyBackend:
        def export_rows(self):
            return [{"id": "1", "text": "hello"}]

        def reembed_all(self, embedder):
            assert embedder is not None
            return {"success": 1, "failed": 0}

    class DummyConfig:
        embedding_provider = "none"
        embedding_model = "dummy"

        @classmethod
        def from_env(cls):
            return cls()

    monkeypatch.setattr(reembed_memory, "get_memory_backend", lambda: DummyBackend())
    monkeypatch.setattr(reembed_memory, "LanceDBConfig", DummyConfig)
    monkeypatch.setattr(reembed_memory, "Embedder", lambda provider, model: object())
    monkeypatch.setattr(builtins, "input", lambda _: "y")

    assert reembed_memory.main() == 0


def test_reembed_memory_script_aborts_cleanly(monkeypatch) -> None:
    class DummyBackend:
        def export_rows(self):
            return [{"id": "1", "text": "hello"}]

        def reembed_all(self, embedder):
            raise AssertionError("should not be called")

    class DummyConfig:
        embedding_provider = "none"
        embedding_model = "dummy"

        @classmethod
        def from_env(cls):
            return cls()

    monkeypatch.setattr(reembed_memory, "get_memory_backend", lambda: DummyBackend())
    monkeypatch.setattr(reembed_memory, "LanceDBConfig", DummyConfig)
    monkeypatch.setattr(reembed_memory, "Embedder", lambda provider, model: object())
    monkeypatch.setattr(builtins, "input", lambda _: "n")

    assert reembed_memory.main() == 0
