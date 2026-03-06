# Architecture Layers

Annolid now defines four top-level layers with explicit intent:

1. `annolid.domain`
2. `annolid.services`
3. `annolid.interfaces`
4. `annolid.infrastructure`

## Layer responsibilities

- `annolid.domain`
  - Business schema/types and invariant-centric models (project schema, events, tracks, keypoints, timelines, datasets).
  - Concrete modules: `project_schema`, `behavior_events`, `tracks`, `keypoints`, `timelines`, `datasets`.
- `annolid.services`
  - Orchestration use-cases that interfaces call (`agent` pipeline, behavior `time_budget`, and future tracking/export/search workflows).
  - Concrete modules: `inference`, `training`, `export`, `search`, `tracking`, `agent`.
- `annolid.interfaces`
  - User-facing and transport-facing adapters (GUI, CLI, bots, background jobs).
  - Concrete modules: `gui`, `cli`, `background`, `bot`.
- `annolid.infrastructure`
  - Filesystem/persistence/runtime patching/model download/external API adapters.
  - Concrete modules: `filesystem`, `persistence`, `runtime`, `model_downloads`, `external_apis`.

## Rules

- Interfaces call services, not each other.
- Services may depend on domain and infrastructure.
- Domain must stay framework-agnostic.
- Infrastructure must not import GUI code.

## Migration status

- Shared service in place: `annolid.services.agent.run_agent_pipeline`
  - Used by both GUI (`AgentAnalysisMixin`) and CLI (`annolid-run agent`).
- Shared service in place: `annolid.services.time_budget.compute_behavior_time_budget_report`
  - Used by GUI behavior time-budget dialog/export path.
- Shared service in place: `annolid.services.embedding_search.run_embedding_search`
  - Used by GUI frame-similarity/embedding search worker.
- Shared service in place: `annolid.services.tracking.*`
  - Used by GUI and background tracking setup/execution paths (`prediction_execution_mixin`, `TrackAllWorker`, `jobs/tracking_worker.py`).
- Stable architecture wrappers in place:
  - `annolid.domain.*` re-exports canonical schema/event/track/keypoint/timeline/dataset types.
  - `annolid.services.*` exposes inference/training/export/search/tracking/agent APIs.
  - `annolid.interfaces.*` exposes GUI/CLI/background/bot entry points.
  - `annolid.infrastructure.*` exposes filesystem/persistence/runtime/download/API adapters.
