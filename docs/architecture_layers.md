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
- Shared service in place: `annolid.services.agent_admin.*`
  - Used by the CLI agent security/secret-management commands instead of direct `core.agent` imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.agent_workspace.*`
  - Used by the CLI agent skills/feedback/memory commands instead of direct `core.agent` imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.agent_eval.*`
  - Used by the CLI agent eval commands instead of direct `core.agent.eval.*` imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.agent_update.*`
  - Used by the CLI update commands instead of direct `core.agent.update_manager.*` imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.agent_bridge.run_agent_acp_bridge`
  - Used by the CLI ACP bridge command instead of direct `core.agent.acp_stdio_bridge` imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.agent_cron.*`
  - Used by the CLI agent onboard/status/cron commands instead of direct `core.agent.cron`, workspace bootstrap, and utils imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.agent_tooling.validate_agent_tools`
  - Used by the CLI agent tool validation command instead of direct `core.agent.tools.*` imports in `annolid/engine/cli.py`.
- Shared service in place: `annolid.services.chat_runtime.*`
  - Used by the GUI chat backend for workspace/config root resolution instead of embedding that bootstrap logic directly in `annolid/gui/widgets/ai_chat_backend.py`.
- Shared service in place: `annolid.services.chat_provider_runtime.*`
  - Used by the GUI chat backend for provider kind resolution, fast-mode execution, provider fallback, and direct provider chat calls instead of importing `core.agent.gui_backend.provider_*` modules directly.
- Shared service in place: `annolid.services.chat_session.*`
  - Used by the GUI chat backend for session-store bootstrap, outbound chat event emission, history loading, and turn persistence instead of importing `core.agent.session_manager` and `core.agent.gui_backend.session_io` directly.
- Shared service in place: `annolid.services.chat_widget_bridge.*`
  - Used by the GUI chat backend for widget slot invocation, GUI context payload assembly, direct-command dispatch, and sync-awaitable bridging instead of importing `core.agent.gui_backend.widget_bridge`, `direct_commands`, and the direct-command router directly.
- Shared service in place: `annolid.services.chat_web_pdf.*`
  - Used by the GUI chat backend for web view actions, PDF actions, and URL/PDF opener orchestration instead of importing `core.agent.gui_backend.tool_handlers_web_pdf` and `tool_handlers_openers` directly.
- Shared service in place: `annolid.services.chat_video.*`
  - Used by the GUI chat backend for video open/resolve flow and segment-track/behavior-label workflow orchestration instead of importing `core.agent.gui_backend.tool_handlers_video` and `tool_handlers_video_workflow` directly.
- Shared service in place: `annolid.services.chat_controls.*`
  - Used by the GUI chat backend for chat control actions such as prompt/model/frame updates, AI text segmentation, and next-frame tracking instead of importing `core.agent.gui_backend.tool_handlers_chat_controls` directly.
- Shared service in place: `annolid.services.chat_arxiv.*`
  - Used by the GUI chat backend for arXiv search and local PDF discovery instead of importing `core.agent.gui_backend.tool_handlers_arxiv` directly.
- Shared service in place: `annolid.services.chat_citations.*`
  - Used by the GUI chat backend for citation extraction, normalization, listing, and persistence orchestration instead of importing `core.agent.gui_backend.tool_handlers_citations` directly.
- Shared service in place: `annolid.services.chat_filesystem.*`
  - Used by the GUI chat backend for filesystem rename actions instead of importing `core.agent.gui_backend.tool_handlers_filesystem` directly.
- Shared service in place: `annolid.services.chat_realtime.*`
  - Used by the GUI chat backend for realtime stream/log actions instead of importing `core.agent.gui_backend.tool_handlers_realtime` directly.
- Shared service in place: `annolid.services.chat_shapes.*`
  - Used by the GUI chat backend for in-canvas shape selection/listing/edit actions instead of importing `core.agent.gui_backend.tool_handlers_shapes` directly.
- Shared service in place: `annolid.services.chat_shape_files.*`
  - Used by the GUI chat backend for shape-file listing/relabel/delete actions instead of importing `core.agent.gui_backend.tool_handlers_shape_files` directly.
- Shared service in place: `annolid.services.chat_agent_core.*`, `chat_backend_support.*`, `chat_devtools.*`, and `chat_manager_runtime.*`
  - Used by the GUI chat backend, chat widget, and chat manager for provider state, tool registry/policy, GUI backend helper logic, dev-tool execution, and background messaging/channel runtime instead of importing `core.agent` modules directly from GUI code.
- Infrastructure wrappers in place: `annolid.infrastructure.agent_config.*` and `agent_workspace.*`
  - Used by the chat backend, chat widget, settings dialog, web viewer, and chat manager for config/workspace access instead of importing `core.agent.config` and `core.agent.utils` directly from GUI code.
- GUI chat surfaces migrated:
  - `annolid/gui/widgets/ai_chat_backend.py`
  - `annolid/gui/widgets/ai_chat_widget.py`
  - `annolid/gui/widgets/ai_chat_manager.py`
  - `annolid/gui/widgets/ai_chat_session_dialog.py`
  - `annolid/gui/widgets/llm_settings_dialog.py`
  - `annolid/gui/widgets/web_viewer.py`
  - These now consume `annolid.services` and `annolid.infrastructure` wrappers rather than `annolid.core.agent` directly.
- Stable architecture wrappers in place:
  - `annolid.domain.*` re-exports canonical schema/event/track/keypoint/timeline/dataset types.
  - `annolid.services.*` exposes inference/training/export/search/tracking/agent APIs.
  - `annolid.interfaces.*` exposes GUI/CLI/background/bot entry points.
  - `annolid.infrastructure.*` exposes filesystem/persistence/runtime/download/API adapters.

## Next migration tranche

- Keep moving remaining GUI modules toward `annolid.domain`, `annolid.services`, and `annolid.infrastructure` wrappers instead of direct `core` or `utils` imports.
- When new chat/tooling features are added, land them in the service/infrastructure facades first rather than importing `annolid.core.agent` from widgets or dialogs.
- Keep `annolid/interfaces.*` as the public entry surfaces; avoid importing `annolid.interfaces` from the concrete GUI/CLI implementation modules themselves to prevent cycles.
