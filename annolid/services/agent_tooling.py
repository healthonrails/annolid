"""Service-layer validation helpers for agent tooling surfaces."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


def validate_agent_tools() -> tuple[dict[str, object], int]:
    summary: dict[str, object] = {"status": "ok", "checks": []}

    def _record(name: str, *, ok: bool, detail: str) -> None:
        summary["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            summary["status"] = "error"

    try:
        from annolid.core.agent.tools.artifacts import FileArtifactStore, content_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(base_dir=Path(tmpdir), run_id="validate")
            meta_path = store.resolve("agent_cache.json", kind="cache")
            payload = {"hello": "world"}
            store.write_meta(meta_path, {"content_hash": content_hash(payload)})
            ok = store.should_reuse_cache(meta_path, content_hash(payload))
            _record("artifacts", ok=bool(ok), detail="cache metadata round-trip")
    except Exception as exc:
        _record("artifacts", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.sampling import (
            FPSampler,
            RandomSampler,
            UniformSampler,
        )

        uniform = UniformSampler(step=2).sample_indices(10)
        fps = FPSampler(target_fps=5).sample_indices(30, fps=30)
        random = RandomSampler(count=2, seed=1).sample_indices(5)
        ok = bool(uniform) and bool(fps) and bool(random)
        _record("sampling", ok=ok, detail="uniform/fps/random sampling")
    except Exception as exc:
        _record("sampling", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.registry import ToolRegistry
        from annolid.core.agent.tools.base import Tool, ToolContext
        from annolid.core.agent.tools.function_registry import FunctionToolRegistry
        from annolid.core.agent.tools.function_builtin import (
            register_nanobot_style_tools,
        )
        from annolid.core.agent.tools.utility import register_builtin_utility_tools

        class _DummyTool(Tool[int, int]):
            name = "dummy"

            def run(self, ctx: ToolContext, payload: int) -> int:
                _ = ctx
                return payload + 1

        registry = ToolRegistry()
        registry.register("dummy", _DummyTool)
        register_builtin_utility_tools(registry)
        fn_registry = FunctionToolRegistry()
        asyncio.run(register_nanobot_style_tools(fn_registry))
        instance = registry.create("dummy")
        ok = (
            isinstance(instance, _DummyTool)
            and registry.has("calculator")
            and fn_registry.has("read_file")
            and fn_registry.has("exec")
        )
        _record(
            "registry",
            ok=ok,
            detail="register/create tool + utility + nanobot-style function tools",
        )
    except Exception as exc:
        _record("registry", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.vector_index import NumpyEmbeddingIndex
        from annolid.domain import FrameRef

        index = NumpyEmbeddingIndex(
            embeddings=[[0.1, 0.0], [0.0, 1.0]],
            frames=[FrameRef(frame_index=0), FrameRef(frame_index=1)],
        )
        results = index.search([0.2, 0.1], top_k=1)
        ok = bool(results)
        _record("vector_index", ok=ok, detail="numpy cosine search")
    except ImportError as exc:
        _record("vector_index", ok=True, detail=f"skipped: {exc}")
    except Exception as exc:
        _record("vector_index", ok=False, detail=str(exc))

    return summary, (0 if summary.get("status") == "ok" else 1)


__all__ = ["validate_agent_tools"]
