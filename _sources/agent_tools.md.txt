# Agent Tools Developer Guide

This guide covers how to add a new agent tool, how artifacts are laid out on disk,
and how caching works for agent runs.

## How to add a tool

1. **Define the tool** by extending the base class in `annolid/core/agent/tools/base.py`:

   - Implement `run(self, ctx, payload)` with your core logic.
   - Use `ctx.results_dir` and `ctx.run_id` to derive stable outputs.
   - Use `ctx.artifact_store` if you want to persist artifacts and participate in caching.

2. **Register the tool** in the registry:

   - Add a new tool wrapper in `annolid/core/agent/tools/`.
   - Export it from `annolid/core/agent/tools/__init__.py`.
   - Register it with `ToolRegistry` (see `annolid/core/agent/tools/registry.py`).

3. **Integrate with the runner** (Phase 4+):

   - Compose tools using the registry and a pipeline definition.
   - Ensure inputs/outputs follow the unified data models in `base.py`.

4. **Write a minimal test**:

   - Use tiny inputs and validate outputs.
   - Prefer tests under `tests/` that don’t require large external models.

## Artifact layout

Artifacts are stored per video results directory and organized as:

- `<results_dir>/`
  - `agent.ndjson` (default agent output)
  - `<video_name>_000000000.json` + per-frame LabelMe JSON
  - `.agent_runs/<run_id>/` (run-scoped artifacts)
  - `.cache/agent_cache.json` (cache metadata for re-run reuse)

The `FileArtifactStore` resolves paths relative to:

- **Run artifacts**: `.agent_runs/<run_id>/...`
- **Cache artifacts**: `.cache/...`

See `annolid/core/agent/tools/artifacts.py` for helpers.

## Caching semantics

Agent runs compute a **content hash** from:

- video path + filesystem stats (size/mtime),
- behavior spec (full schema),
- run config (stride, max frames, etc.),
- model identifiers,
- output NDJSON name.

If the cache hash matches and both the NDJSON and annotation store exist,
the service returns cached results without re-running the agent.

To disable reuse from the CLI, run:

```
annolid-run agent --no-cache ...
```
