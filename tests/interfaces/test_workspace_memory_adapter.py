from annolid.interfaces.memory.adapters.workspace import WorkspaceMemoryAdapter


def test_workspace_adapter_stores_context_in_metadata(monkeypatch):
    captured = {}

    class DummyService:
        def store_memory(self, **kwargs):
            captured.update(kwargs)
            return "mem_1"

    import annolid.interfaces.memory.adapters.workspace as workspace_module

    monkeypatch.setattr(workspace_module, "get_memory_service", lambda: DummyService())

    adapter = WorkspaceMemoryAdapter("w1")
    memory_id = adapter.store_settings_snapshot(
        description="Known-good setup",
        settings_dict={"fps": 10},
        context="Infrared dataset",
    )

    assert memory_id == "mem_1"
    assert captured["metadata"]["context"] == "Infrared dataset"
    assert captured["metadata"]["settings"]["fps"] == 10
