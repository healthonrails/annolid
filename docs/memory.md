# Annolid Memory

Annolid memory is an optional, local-first retrieval layer for reusable context:
project notes, annotation conventions, workflow recipes, settings snapshots, and
agent-facing memory.

Memory is support context, not source of truth. Canonical project files,
annotations, datasets, and configs remain authoritative.

## Current Scope

Implemented:

- Layered modules in `annolid/domain/memory`, `annolid/services/memory`,
  `annolid/infrastructure/memory/lancedb`, and `annolid/interfaces/memory`
- LanceDB-backed storage with optional semantic retrieval
- Scope/category/source taxonomy
- Persistence helpers and interface adapters
- `annolid-run memory ...` CLI commands (`stats`, `search`, `delete`, `distill`, `cleanup`)
- Maintenance scripts for export/import, migration, and re-embedding
- GUI Memory Manager with CRUD operations
- One-click settings profile apply actions for all Settings panels
- One-click "save current settings as profile" actions for all Settings panels
- Structured settings profiles (`SettingsProfile`) in addition to snapshots
- Automated multi-source migration collectors (JSON, markdown memory logs, project schemas)
- In-app migration report dashboard (scan + counts + import)

Not yet complete:

- Cross-workspace profile browsing/pinning in GUI
- Incremental migration history with rollback checkpoints

## GUI Workflow

Open the Memory Manager from the desktop app:

1. `Settings -> Memory Manager...`
2. Search by query and optional scope (`workspace:<id>`, `project:<id>`, `dataset:<id>`)
3. Use `Add`, `Edit`, and `Delete` for CRUD operations

Notes:

- edits are validated before write (required text/scope, metadata JSON validation)
- delete requires confirmation
- operations refresh the result table immediately

### Migration dashboard

In the same Memory Manager dialog, open `Migration Dashboard`:

1. Select a legacy source directory
2. Click `Scan Sources` to collect and preview import candidates
3. Review source counts and record previews
4. Click `Import Scanned` to ingest records into the active memory backend

## Quick Start

Install optional memory dependencies:

```bash
pip install -e ".[memory]"
```

For local development in this repository, activate `.venv` first:

```bash
source .venv/bin/activate
```

Enable memory via environment variables:

```bash
export ANNOLID_MEMORY_ENABLED=true
export ANNOLID_MEMORY_BACKEND=lancedb
export ANNOLID_MEMORY_LANCEDB_PATH=~/.annolid/memory/lancedb
export ANNOLID_MEMORY_EMBEDDING_PROVIDER=jina
export ANNOLID_MEMORY_EMBEDDING_MODEL=jina-embeddings-v3
export ANNOLID_MEMORY_VECTOR_WEIGHT=0.65
export ANNOLID_MEMORY_BM25_WEIGHT=0.35
export ANNOLID_MEMORY_RERANK_PROVIDER=none
```

Supported backend values today:

- `lancedb`
- `none`

## Architecture

```text
annolid/
  domain/memory/
  services/memory/
  infrastructure/memory/lancedb/
  interfaces/memory/
```

Layer responsibilities:

- `domain`: backend-agnostic models, protocols, scopes, taxonomy
- `services`: store/retrieve orchestration, context building, persistence workflows
- `infrastructure`: LanceDB config/backend/retriever/embedder/reranker/migration
- `interfaces`: registry, CLI, and adapters (bot/annotation/project/workflow/workspace)

## Data Contract

Primary models:

- `MemoryRecord`: write model for new records
- `MemoryHit`: read model returned by list/search

Key fields:

- `text`: short human-readable memory statement
- `scope`: for example `global`, `project:<id>`, `dataset:<id>`, `workspace:<id>`
- `category`: for example `project_note`, `annotation_rule`, `setting`, `workflow_recipe`
- `metadata`: structured references back to canonical artifacts and context

## Retrieval Behavior

The LanceDB retriever can combine:

- vector similarity
- full-text retrieval (when available)
- recency weighting
- length normalization
- optional reranking

Blank-query retrieval is supported for scoped context calls (for example,
"show project memory" with no semantic query text).

## CLI Usage

Use memory commands through the main runner:

```bash
annolid-run memory stats --scope global
annolid-run memory search "segmentation conventions" --top_k 5
annolid-run memory delete <memory_id>
annolid-run memory distill project:demo --max_records 50
annolid-run memory cleanup workspace:default --older_than_days 30 --min_importance 0.2
```

Notes:

- `search` uses semantic retrieval when embeddings are enabled
- `cleanup` removes low-importance/old records within a scope
- `distill` requires summarization service availability

## Developer API

Store memory directly:

```python
from annolid.domain.memory.scopes import MemoryCategory, MemoryScope, MemorySource
from annolid.interfaces.memory.registry import get_memory_service

service = get_memory_service()
service.store_memory(
    text="Use tail_base instead of tailroot for this dataset.",
    scope=MemoryScope.dataset("mouse-session-a"),
    category=MemoryCategory.ANNOTATION_RULE,
    source=MemorySource.ANNOTATION,
    importance=0.9,
    metadata={"species": "mouse"},
)
```

Build scoped context:

```python
from annolid.interfaces.memory.registry import get_context_service

context = get_context_service().build_project_context(project_id="demo-project")
```

Use persistence helpers:

```python
from annolid.domain.memory.scopes import MemoryScope
from annolid.interfaces.memory.registry import get_persistence_service

persistence = get_persistence_service()
persistence.save_settings_snapshot(
    scope=MemoryScope.workspace("lab-a"),
    description="Known-good export setup",
    settings={"fps": 10, "format": "csv"},
    context="Infrared videos",
)
```

## Adapters

Available interface adapters:

- `BotMemoryAdapter`
- `AnnotationMemoryAdapter`
- `ProjectMemoryAdapter`
- `WorkflowMemoryAdapter`
- `WorkspaceMemoryAdapter`

Adapters isolate callers from backend-specific implementation details.

`WorkspaceMemoryAdapter` now supports both:

- `SettingsSnapshot` for point-in-time captures
- `SettingsProfile` for reusable typed workflow configurations

Profile apply is available from `Settings` menu with one-click actions for:

- Advanced Parameters
- Optical Flow
- Depth
- SAM 3D
- Patch Similarity
- PCA Feature Map

Profile save actions are available in the same menu for the same workflows, so
you can save current panel state and later apply it with one click.

Minimal profile example:

```python
from annolid.interfaces.memory.adapters.settings_model import SettingsProfile
from annolid.interfaces.memory.adapters.workspace import WorkspaceMemoryAdapter

adapter = WorkspaceMemoryAdapter("default")
profile = SettingsProfile(
    name="Infrared tracking preset",
    workflow="tracking",
    settings={"tracker": "bytetrack", "confidence": 0.45},
    tags=["infrared", "tracking"],
)
adapter.store_settings_profile(profile)
```

## Maintenance Operations

Scripts:

```bash
source .venv/bin/activate
python scripts/export_memory.py export /tmp/annolid_memory.jsonl
python scripts/export_memory.py import /tmp/annolid_memory.jsonl
python scripts/migrate_memory.py --source-dir /path/to/legacy/json
python scripts/reembed_memory.py
```

Purpose:

- `export_memory.py`: export/import JSONL memory records
- `migrate_memory.py`: recursively collect and import legacy memory sources
- `reembed_memory.py`: regenerate vectors using current embedding settings

`migrate_memory.py` source coverage:

- JSON memory records with top-level `text`
- `memory/MEMORY.md` and `memory/HISTORY.md`
- `project.annolid.json|yaml|yml`

## JSONL Format

Export output is newline-delimited JSON. Each line is one record without vector
payloads.

Expected fields:

- `id`
- `text`
- `scope`
- `category`
- `source`
- `timestamp_ms`
- `importance`
- `token_count`
- `tags`
- `metadata_json`

Notes:

- vectors are omitted by design (recomputable)
- `metadata_json` is stored as JSON text for backend compatibility
- import regenerates embeddings via `store_memory(...)`
- duplicate handling follows service-layer dedupe behavior

## Validation

Run targeted memory tests:

```bash
source .venv/bin/activate
pytest tests/services/test_memory_service.py \
  tests/test_memory_cli.py \
  tests/test_memory_cli_commands.py \
  tests/test_memory_scripts.py \
  tests/services/test_persistence_service.py \
  tests/interfaces/test_adapters.py \
  tests/interfaces/test_workspace_memory_adapter.py \
  tests/memory/test_taxonomy.py \
  tests/memory/lancedb/test_migration.py \
  tests/memory/lancedb/test_backend.py \
  tests/memory/lancedb/test_backend_admin.py
```

## Design Constraints

The subsystem is intentionally conservative:

- writes are explicit, not implicit side effects
- stored memory should be compact and summary-oriented
- canonical project artifacts remain source of truth
- optional dependencies must degrade gracefully when unavailable
