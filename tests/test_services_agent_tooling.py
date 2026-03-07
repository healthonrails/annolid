from __future__ import annotations

from annolid.services.agent_tooling import validate_agent_tools


def test_validate_agent_tools_returns_ok(monkeypatch) -> None:
    import annolid.core.agent.tools.artifacts as artifacts_mod
    import annolid.core.agent.tools.base as base_mod
    import annolid.core.agent.tools.function_builtin as builtin_mod
    import annolid.core.agent.tools.function_registry as fn_registry_mod
    import annolid.core.agent.tools.registry as registry_mod
    import annolid.core.agent.tools.sampling as sampling_mod
    import annolid.core.agent.tools.utility as utility_mod
    import annolid.core.agent.tools.vector_index as vector_mod

    class _Store:
        def __init__(self, **kwargs):
            pass

        def resolve(self, name, kind="cache"):
            return name

        def write_meta(self, path, payload):
            return None

        def should_reuse_cache(self, path, digest):
            return True

    monkeypatch.setattr(artifacts_mod, "FileArtifactStore", _Store)
    monkeypatch.setattr(artifacts_mod, "content_hash", lambda payload: "hash")

    class _Uniform:
        def __init__(self, step):
            pass

        def sample_indices(self, n, **kwargs):
            return [0, 2]

    class _FPS:
        def __init__(self, target_fps):
            pass

        def sample_indices(self, n, fps=None):
            return [0]

    class _Random:
        def __init__(self, count, seed):
            pass

        def sample_indices(self, n):
            return [1]

    monkeypatch.setattr(sampling_mod, "UniformSampler", _Uniform)
    monkeypatch.setattr(sampling_mod, "FPSampler", _FPS)
    monkeypatch.setattr(sampling_mod, "RandomSampler", _Random)

    class _Tool:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    class _Ctx:
        pass

    class _Registry:
        def __init__(self):
            self.names = set()
            self.classes = {}

        def register(self, name, cls):
            self.names.add(name)
            self.classes[name] = cls

        def has(self, name):
            return name in self.names

        def create(self, name):
            return self.classes[name]()

    class _FnRegistry:
        def __init__(self):
            self.names = set()

        def has(self, name):
            return name in self.names

    monkeypatch.setattr(base_mod, "Tool", _Tool)
    monkeypatch.setattr(base_mod, "ToolContext", _Ctx)
    monkeypatch.setattr(registry_mod, "ToolRegistry", _Registry)
    monkeypatch.setattr(fn_registry_mod, "FunctionToolRegistry", _FnRegistry)
    monkeypatch.setattr(
        utility_mod,
        "register_builtin_utility_tools",
        lambda registry: registry.names.add("calculator"),
    )

    async def _register_nanobot_style_tools(registry):
        registry.names.update({"read_file", "exec"})

    monkeypatch.setattr(
        builtin_mod, "register_nanobot_style_tools", _register_nanobot_style_tools
    )

    class _Index:
        def __init__(self, **kwargs):
            pass

        def search(self, vector, top_k=1):
            return [{"frame_index": 0}]

    monkeypatch.setattr(vector_mod, "NumpyEmbeddingIndex", _Index)

    summary, exit_code = validate_agent_tools()

    assert exit_code == 0
    assert summary["status"] == "ok"
