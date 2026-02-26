# Agent Tools Developer Guide

This guide covers how to add a new agent tool, how artifacts are laid out on disk,
and how caching works for agent runs.

## Operational Model

Annolid agent operations are split into two layers:

- Self-improving:
  skills and memory evolve behavior without replacing installed code.
- Self-updating:
  signed update workflow stages and applies software updates with rollback plans.

### Self-improving

- Skills:
  loaded with precedence `workspace -> managed (~/.annolid/skills) -> bundled`.
- Hot reload:
  controlled by `skills.load.watch` and `skills.load.pollSeconds`.
- Skill manifest validation:
  frontmatter is validated at load time; invalid manifests are marked unavailable.
- Workspace memory:
  daily notes in `memory/YYYY-MM-DD.md` and curated long-term notes in `memory/MEMORY.md`.
- Pre-compaction flush:
  transcript snapshot can be appended before compaction via memory flush helpers.
- Memory retrieval plugin:
  default is local semantic ranking with keyword fallback (`workspace_semantic_keyword_v1`).

### Self-updating

- Channel-aware update manager supports `stable`, `beta`, and `dev`.
- Pipeline:
  `preflight -> stage -> verify -> apply -> restart marker -> post-check`.
- Rollback:
  rollback plan is generated for each run and executed on apply/post-check failures.
- Canary policy:
  rollout can enforce rollback thresholds using sample count, failure-rate, and regression limits.

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

## Citation management tools

Annolid includes built-in BibTeX tooling for paper citation workflows:

- CLI:
  - `annolid-run citations-list --bib-file refs.bib [--query ...]`
  - `annolid-run citations-upsert --bib-file refs.bib --key mykey --title ... --author ... --year ...`
  - `annolid-run citations-remove --bib-file refs.bib --key mykey`
  - `annolid-run citations-format --bib-file refs.bib`
- Agent function tools:
  - `bibtex_list_entries`
  - `bibtex_upsert_entry`
  - `bibtex_remove_entry`
  - `gui_save_citation` (save from active PDF/web viewer context)

Examples in Annolid Bot message input:

- `save citation`
- `list citations`
- `list citations from references.bib for annolid`
- `save citation from pdf as annolid2024 to references.bib`
- `save citation from web`
- `add citation @article{yang2024annolid, title={Annolid: Annotate, Segment, and Track Anything You Need}, author={Yang, Chen and Cleland, Thomas A}, journal={arXiv preprint arXiv:2403.18690}, year={2024}}`
- `save citation from web with strict validation`
- `save citation from pdf without validation`
- `open threejs example two mice`
- `open threejs example brain`
- `open threejs html /tmp/annolid_threejs_examples/two_mice.html`
- `open threejs https://example.org/viewer.html`

Default behavior:

- `save citation` first attempts Google Scholar BibTeX lookup from the active paper context, then falls back to Crossref/OpenAlex when needed, and saves the merged entry to `.bib`.

GUI workflow:

- In Annolid Bot input toolbar, click `📚` to open the citation manager.
- Manage a `.bib` file, save citations from active PDF/web context, choose auto-validation or strict mode, view/edit a `Source` column (URL or PDF path), edit rows inline with year/DOI checks, and remove selected entries.

See also: `docs/source/citations_tutorial.md` for a full user tutorial.

## Operator Commands

Use `annolid-run` commands for routine operations:

- `annolid-run agent skills refresh [--workspace <path>]`
- `annolid-run agent skills inspect [--workspace <path>]`
- `annolid-run agent memory flush [--workspace <path>] [--session-id <id>] [--note <text>]`
- `annolid-run agent memory inspect [--workspace <path>]`
- `annolid-run agent eval run --traces <jsonl> --candidate-responses <jsonl> --out <report.json>`
- `annolid-run agent eval build-regression --workspace <path> --out <traces.jsonl> [--min-abs-rating 1]`
- `annolid-run agent eval gate --changed-files <files.txt> --report <report.json> [--max-regressions 0] [--min-pass-rate 0.0]`
- `annolid-run agent feedback add --workspace <path> --rating -1|0|1 [--trace-id <id>] [--comment <text>] [--expected-substring <text>]`
- `annolid-run update check --channel stable|beta|dev [--require-signature]`
- `annolid-run update run --channel stable|beta|dev [--execute] [--require-signature] [--skip-post-check] [--canary-metrics <json>]`
- `annolid-run update rollback --install-mode package|source --previous-version <X.Y.Z> [--execute]`

## Improvement Quality Loop

- Anonymized run traces:
  `workspace/eval/run_traces.ndjson` captures hashed session/channel/chat IDs and redacted text previews.
- Explicit user feedback:
  `workspace/eval/feedback.ndjson` stores rating/comment/optional expected substring for promotion signals.
- Regression dataset build:
  combines traces + feedback into eval traces for CI and pre-promotion checks.
- Shadow mode:
  enable `ANNOLID_AGENT_SHADOW_MODE=1` to log alternative routing decisions to `workspace/eval/shadow_routing.ndjson`.
  use `annolid-run agent skills shadow --candidate-pack <dir>` to compare candidate skill packs before promotion.

## Governance and Audit

Governance events are stored as NDJSON with default path:

- `~/.annolid/governance/events.ndjson`

You can override it with:

- `ANNOLID_GOVERNANCE_EVENTS_PATH=/custom/path/events.ndjson`

Audited event categories include skill snapshot/refresh changes, memory writes/flushes,
update stage/run actions, and rollback outcomes.

## Three.js bot tools

Annolid Bot supports direct Three.js viewer control in GUI sessions.

- Function tools:
  - `gui_open_threejs(path_or_url)`
  - `gui_open_threejs_example(example_id)`
- Built-in example IDs:
  - `two_mice_html` (default)
  - `brain_viewer_html`
  - `helix_points_csv`
  - `wave_surface_obj`
  - `sphere_points_ply`

The bot recognizes natural-language commands such as `open threejs example ...`.

## Annolid code/docs Q&A and tutorials

Annolid Bot is optimized to answer Annolid-specific questions from local docs and code context.

- It can explain modules, workflows, and settings with file-path references.
- It can generate on-demand tutorials for requested topics and levels using the active chat model, grounded by Annolid docs/code evidence.
- When a tutorial is saved to Markdown, Annolid Bot auto-opens the generated `.md` in the embedded web viewer.
- Direct command examples:
  - `create on demand tutorial for realtime camera setup in annolid`
  - `create beginner tutorial for behavior analysis and save to markdown file`
  - `how do i use annolid for behavior analysis`

## Realtime camera snapshot + email

Annolid Bot can capture a snapshot from a camera stream and send it by email.

- Stream snapshot:
  - GUI sessions: use `gui_check_stream_source` with `save_snapshot=true`.
  - This GUI tool now runs a full camera mission pipeline:
    - `probe -> capture -> annotate -> notify/email`
    - returns explicit `camera_mission.steps` and `delivery` status objects.
  - Non-GUI channels (for example email/IM): use `camera_snapshot`.
  - Snapshot files are saved under `.annolid/workspace/camera_snapshots/`.
  - Outlook Safe Links camera URLs are automatically unwrapped to the original stream URL.
  - Source fallback policy is intent-aware:
    - eye-blink intent defaults to camera `0`
    - network camera intent prefers remembered network streams.
- Email with attachments:
  - Use the `email` tool with:
    - `to`
    - `subject`
    - `content`
    - optional `attachment_paths` (list of local file paths)

Example bot intent:

- `check wireless camera, save a snapshot, and email it to user@example.com`

Realtime email/report spam control:

- Realtime bot report interval controls report cadence.
- Email requests use an additional minimum interval (`bot_email_min_interval_sec`, default `60s`) to avoid repeated email requests.

## Security and policy hardening (Phase 2)

Adds stricter defaults for tool access and data handling:

- Capability-oriented tool profiles:
  - `gui`, `email`, `realtime`, `filesystem`
  - explicit capability expressions are supported, for example:
    - `capability:gui,email`
    - `capability:gui+realtime`
- Snapshot path hardening:
  - `camera_snapshot` writes only under workspace `camera_snapshots/`.
  - symlink escape paths are rejected.
- Redaction-at-source:
  - private/local stream endpoints are redacted in outbound content.
  - sensitive metadata keys (for example `peer_id`, `account_id`) are redacted before publish.
- Runtime high-risk guard:
  - deny-by-default blocks risky multi-tool chains unless explicit intent is provided.
  - config toggle: `agents.defaults.strict_runtime_tool_guard` (default `true`).

Example config:

```json
{
  "agents": {
    "defaults": {
      "strict_runtime_tool_guard": true
    }
  }
}
```

Explicit high-risk intent markers supported by policy/runtime guards:

- `intent:high-risk`
- `intent:high_risk`
- `allow:high-risk`
- `allow_high_risk`
- `unsafe:high-risk`

## Session memory and replay

Annolid agent sessions now keep separated memory layers and replayable event logs.

- Working memory:
  - short-horizon session summary derived from recent user/assistant turns.
  - stored in session metadata as `working_memory`.
  - bounded by a character quota in `PersistentSessionStore`.
- Long-term memory:
  - stable facts/notes derived from session facts and consolidation updates.
  - stored in session metadata as `long_term_memory`.
  - bounded by a character quota in `PersistentSessionStore`.

### Deterministic consolidation and telemetry

Memory consolidation now uses deterministic triggers based on:

- session turn counter (`turn_counter`)
- next scheduled consolidation turn (`next_consolidation_turn`)
- history length relative to memory window

Telemetry is persisted in session metadata as `memory_telemetry` with entries like:

- `timestamp`
- `outcome` (for example `llm_consolidated`, `skipped_short_transcript`, `not_due`)
- `history_len`, `archive_len`, `keep_len`
- `elapsed_ms`

### Memory mutation audit trail

Session metadata contains `memory_audit_trail` entries for memory changes, including:

- `timestamp`
- `scope` (`facts`, `working_memory`, `long_term_memory`)
- `mutation` (for example `set_fact`, `set_working_memory`)
- `reason`
- `turn_id`
- `before_chars` / `after_chars`

### Safe replay for debugging

Session event records are stored in metadata key `event_log`.

- Each entry includes:
  - `timestamp`
  - `direction` (`inbound`/`outbound`)
  - `kind` (for example `user`, `assistant`, `progress`, `final`)
  - optional `turn_id`, `event_id`, `idempotency_key`
  - `payload`

GUI/backend helpers:

- `replay_session_debug_events(session_store=..., session_id=..., direction=\"\", limit=200)`
- `format_replay_as_text(events)`

These helpers are implemented in:

- `annolid/core/agent/gui_backend/session_io.py`
