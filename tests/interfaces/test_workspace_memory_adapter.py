from annolid.interfaces.memory.adapters.workspace import WorkspaceMemoryAdapter
from annolid.interfaces.memory.adapters.settings_model import SettingsProfile


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


def test_workspace_adapter_store_settings_profile(monkeypatch):
    captured = {}

    class DummyService:
        def store_memory(self, **kwargs):
            captured.update(kwargs)
            return "mem_profile_1"

    import annolid.interfaces.memory.adapters.workspace as workspace_module

    monkeypatch.setattr(workspace_module, "get_memory_service", lambda: DummyService())

    adapter = WorkspaceMemoryAdapter("w1")
    profile = SettingsProfile(
        name="IR Export",
        workflow="video_inference",
        settings={"fps": 10, "format": "csv"},
        workspace_id="w1",
        project_id="p1",
        tags=["infrared"],
    )
    memory_id = adapter.store_settings_profile(profile)
    assert memory_id == "mem_profile_1"
    assert captured["metadata"]["settings_profile"]["name"] == "IR Export"
    assert captured["metadata"]["settings"]["fps"] == 10


def test_workspace_adapter_retrieve_settings_profiles(monkeypatch):
    class DummyHit:
        id = "mem_profile_1"
        metadata = {
            "settings_profile": {
                "id": "",
                "name": "IR Export",
                "workflow": "video_inference",
                "workspace_id": "w1",
                "project_id": "p1",
                "settings": {"fps": 10},
                "created_ms": 1,
                "updated_ms": 2,
                "tags": [],
                "context": None,
            }
        }

    class DummyRetrieval:
        def search_memory(self, **kwargs):
            _ = kwargs
            return [DummyHit()]

    import annolid.interfaces.memory.adapters.workspace as workspace_module

    monkeypatch.setattr(
        workspace_module, "get_retrieval_service", lambda: DummyRetrieval()
    )

    adapter = WorkspaceMemoryAdapter("w1")
    profiles = adapter.retrieve_settings_profiles(workflow="video_inference")
    assert len(profiles) == 1
    assert profiles[0].name == "IR Export"
    assert profiles[0].id == "mem_profile_1"
