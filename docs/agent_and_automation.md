# Agent and Automation

Use this section when you are configuring Annolid Bot, connecting external tools,
or running repeatable workflows through the agent stack.

## Start Here

<div class="ann-grid">
  <article class="ann-card">
    <h3>Agent CLI</h3>
    <p>Run Annolid-native CLI flows through the typed <code>annolid_run</code> path.</p>
    <a href="../agent_annolid_run/">Open agent CLI guide</a>
  </article>
  <article class="ann-card">
    <h3>MCP</h3>
    <p>Connect external tools and resources through Model Context Protocol servers.</p>
    <a href="../mcp/">Open MCP guide</a>
  </article>
  <article class="ann-card">
    <h3>Codex and ACP</h3>
    <p>Bridge Annolid with Codex-style workflows and ACP-compatible runtime paths.</p>
    <a href="../codex_and_acp/">Open Codex and ACP guide</a>
  </article>
  <article class="ann-card">
    <h3>Calendar</h3>
    <p>Schedule and coordinate tasks with Google Calendar-aware agent flows.</p>
    <a href="../agent_calendar/">Open calendar guide</a>
  </article>
  <article class="ann-card">
    <h3>Workspace and Secrets</h3>
    <p>Configure Google Workspace, local secret storage, and channel-safe credentials.</p>
    <a href="../agent_workspace/">Open workspace guide</a>
  </article>
  <article class="ann-card">
    <h3>Memory and Security</h3>
    <p>Manage retrieval-backed memory and harden agent behavior before scaling up.</p>
    <a href="../agent_security/">Open security guide</a>
  </article>
</div>

## Recommended Sequence

1. Start with [Annolid Agent and annolid-run](agent_annolid_run.md) for safe CLI execution.
2. Configure [MCP](mcp.md) if you need external tools or browser/file bridges.
3. Set up [Google Workspace](agent_workspace.md) and [Agent Secrets](agent_secrets.md) before enabling integrations.
4. Review [Agent Security](agent_security.md) and [Memory Subsystem](memory.md) before turning on broader automation.

## What Lives Here

- typed agent tool execution,
- bot-assisted model discovery, training help, and background fine-tuning runs,
- MCP connectivity,
- Codex and ACP integration notes,
- calendar and workspace integrations,
- secret handling,
- memory-backed agent behavior,
- security and operational guardrails.

## Bot Training Workflows

Annolid Bot now exposes typed training tools for model discovery and launch:

- `annolid_dataset_inspect` inspects a dataset folder, summarizes raw LabelMe annotations, detects external formats such as DeepLabCut, COCO, and YOLO folders, distinguishes between saved trainable specs and inferred dataset layouts, and recommends the next prep/training step.
- `annolid_dataset_prepare` prepares a dataset folder for training by generating a LabelMe spec with train/val/test splits, inferring and writing a reusable COCO spec, importing DeepLabCut training data into LabelMe plus an Annolid index, or exporting a YOLO dataset from raw LabelMe or COCO annotations.
- `annolid_train_models` lists trainable model families, aliases, and task hints.
- `annolid_train_help` returns plugin-specific training help such as `dino_kpseg` flags.
- `annolid_train_start` launches long-running `annolid-run train ...` jobs in the background and returns a managed shell `session_id` for follow-up polling through `exec_process`. It can now accept `dataset_folder`, auto-resolve saved dataset configs, and for DinoKPSEG it can stage an inferred COCO spec into the workspace cache when the folder is structurally valid but `coco_spec.yaml` has not been written yet.

This is intended for workflows such as:

- DINOv3-based keypoint segmentation fine-tuning with `dino_kpseg`
- Ultralytics YOLO pose fine-tuning with the `pose` task preset
- Ultralytics YOLO segmentation or detection runs with the matching task preset

Typical dataset-to-training flow:

1. Inspect the folder with `annolid_dataset_inspect`
2. If the folder only has inferred COCO structure or raw LabelMe annotations, run `annolid_dataset_prepare`
3. Start training with `annolid_train_start` and pass `dataset_folder`

For external pose datasets, the first prep step can now stay inside the bot workflow. In particular, a DeepLabCut project with `labeled-data/**/CollectedData_*.csv` can be converted through `annolid_dataset_prepare(mode="deeplabcut_import")`, which writes LabelMe JSON sidecars, an optional pose schema, and an Annolid label index that can then feed the rest of the dataset/training pipeline. COCO folders can be turned into `coco_spec.yaml` with `mode="coco_spec"` or materialized into YOLO `data.yaml` datasets with `mode="coco_to_yolo"` when the target model family expects Ultralytics layout. If the target model is `dino_kpseg`, the bot can also stage an inferred COCO spec automatically under the workspace cache during `annolid_train_start`.

The training launcher prefers the workspace `.venv` interpreter when present so bot-initiated runs use the same dependency environment recommended for local validation.

## Bot Evaluation Reports

Annolid Bot also exposes a typed evaluation reporting tool:

- `annolid_eval_start` launches supported evaluation jobs such as DinoKPSEG evaluation or YOLO validation and returns a managed shell session
- `annolid_eval_report` reads saved evaluation artifacts such as DinoKPSEG eval JSON, YOLO `results.csv` and `predictions.json`, or behavior-classifier `metrics.json`
- it normalizes core metrics into a paper-style summary table
- it can write JSON, Markdown, CSV, and LaTeX report files when a report directory is provided

Typical bot evaluation flow:

1. Start the eval job with `annolid_eval_start`
2. Poll the background session with `exec_process`
3. Turn the produced artifacts into a paper-ready summary with `annolid_eval_report`

Supported eval launch families today:

- `dino_kpseg`
- `yolo`
- `behavior_classifier`

The report output is designed to support common ML reporting practice:

- explicit dataset/model/split metadata
- primary test metrics in a compact table
- confidence intervals when the source metrics support them
- explicit quality checks that flag missing CI coverage, weak sample size, or run-stability gaps
- artifact inventory for confusion matrices, curves, and raw metric files
- reproducibility notes so the bot reports from saved artifacts instead of handwritten numbers

For `yolo`, prefer launching evaluation with `save_json=true` so Ultralytics writes `predictions.json`. When the run directory also lets the report tool resolve a COCO-style annotation file from `args.yaml` and the dataset YAML, `annolid_eval_report` can add deterministic bootstrap confidence intervals for mAP@50 and mAP@50-95 instead of leaving those cells as `NA`.

For `behavior_classifier`, the eval launcher can now also write confusion-matrix and precision-recall curve figures directly during evaluation via `--plot-dir`, so the resulting run is closer to paper-ready without a separate plotting step.

## Research Paper Swarm Launcher

The GUI now includes **AI & Models → Draft Research Paper with Swarm…**, which opens Annolid Bot with a prefilled prompt that invokes `draft_paper_swarm`.

- If a PDF is open, the launcher passes the current PDF title and path into the prompt so the swarm can ground the draft in the active paper.
- The prompt asks the bot to search literature first when needed, then draft a structured paper with outline, sections, and citations.
- This is a thin GUI entry point over the same swarm-backed paper drafting workflow used by the agent tools.

## Citation Integrity Workflow

Annolid Bot and the Citation Manager now support citation verification reports for research-pipeline quality checks.

- `gui_save_citation` supports `verify_after_save=true` to emit a per-entry integrity result and report artifact.
- `gui_verify_citations` verifies an existing `.bib` file in batch and returns `verified/suspicious/hallucinated/skipped` counts plus an aggregate integrity score.
- Direct command example: `verify citations from refs.bib limit 200`
- Citation Manager dialog now includes:
  - `Verify after save` for context saves
  - `Verify .bib` for batch verification of the selected BibTeX file

Report artifacts are written under `.annolid_cache/citation_verification/` next to the selected `.bib` file.

`annolid_eval_report` can optionally enforce citation-quality gates when generating paper-ready model reports:

- `citation_gate=true` enables citation checks
- `citation_report_path` points to a batch verification report JSON (or auto-discovery can be used)
- `citation_hallucinated_max` sets a hard fail threshold
- `citation_suspicious_rate_warn` and `citation_integrity_min_warn` set warning thresholds
- `citation_gate_required=true` upgrades missing citation reports from warn to fail

## Novelty Preflight

For paper-drafting workflows, use `annolid_novelty_check` before writing claims.

- It scores lexical overlap between your proposed idea and related-work summaries.
- It reports coverage quality (`low/medium/high`) for the provided literature context.
- It returns a recommendation:
  - `proceed` when overlap is low and coverage is acceptable
  - `differentiate` when overlap is moderate or literature coverage is weak
  - `abort` when overlap crosses the configured high-risk threshold

You can provide related work inline (`related_work`) or via `related_work_json_path`.

## Unified Paper-Run Report

To consolidate paper-drafting signals in one artifact, use `annolid_paper_run_report`.

- It keeps `annolid_eval_report` unchanged and composes a new additive report.
- It merges:
  - model evaluation table from `annolid_eval_report`,
  - citation verification batch summary,
  - novelty preflight summary,
  - warnings and reproducibility checklist.
- Inputs can be passed as an in-memory eval report object or as `eval_report_json_path`, plus optional `citation_report_path` and `novelty_report_path`.
- It returns a unified markdown+JSON payload and can optionally write files when `report_dir` is provided with `allow_mutation=true`.

### Paper-Ready Quality Gates

For export-time paper readiness, `annolid_paper_run_report` supports configurable gates:

- `paper_ready_gate=true` enables gating checks.
- `citation_integrity_floor` sets the minimum allowed citation integrity score (`0.0` to `1.0`).
- `novelty_coverage_floor` sets the minimum allowed novelty coverage score (`idea_token_coverage`, `0.0` to `1.0`).
- `require_citation_summary` and `require_novelty_summary` control whether missing summaries are treated as blocking failures.

When `paper_ready_gate=true` and a gate check fails, file export to `report_dir` is blocked and the tool returns `ok=false` with the assembled report payload for inspection.
