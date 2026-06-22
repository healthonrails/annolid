# Workflows

This page summarizes the main workflows supported by Annolid today. Start with the standard GUI workflow unless you already know you need a CLI, agent, or specialized model path.

## Workflow Selection

| Goal | Start Here |
| --- | --- |
| Label and track animals in a video | [Standard GUI Video Workflow](#1-standard-gui-video-workflow) |
| Run model training, prediction, or evaluation from scripts | [CLI Model Workflow](#3-cli-model-workflow) |
| Score behavior events or time budgets | [Behavior Analysis Workflow](#4-behavior-analysis-workflow) |
| Repair identity switches after tracking | [Identity Repair Workflow](#identity-repair-workflow) |
| Use Annolid Bot or external tools | [Annolid Bot Workflow](#2-annolid-bot-workflow) |
| Estimate depth or build 3D outputs | [Depth Workflow](#5-depth-workflow) and [Optional 3D Workflows](#6-optional-3d-workflows) |

## 1. Standard GUI Video Workflow

This remains the core Annolid path for most users.

1. Launch the GUI with `annolid`.
2. Open a video directly in the GUI.
3. Move to a representative frame.
4. Label instances, zones, keypoints, or behavior events.
5. Run tracking or propagation from that frame.
6. Review difficult frames and correct identity or segmentation errors early.
7. Export annotations or use them for training/inference workflows.

Practical notes:

- Direct video labeling is supported; frame extraction is optional, not mandatory.
- Use stable instance labels when cross-frame identity matters.
- Save early and often; the LabelMe-compatible JSON files are the reviewable project contract.
- When using shape propagation, `Rename & Propagate` can switch a selected
  shape to another existing shape label from the dialog. If two shapes with
  different labels are selected, Annolid swaps those labels across the
  propagation range instead of renaming both identities to one label.
- Review overlap, occlusion, and high-motion segments before scaling to long runs.
- Validate a small representative segment before committing compute time to an entire recording.

## 2. Annolid Bot Workflow

Annolid Bot runs inside the GUI as a dedicated right-side dock.

Current bot workflow highlights:

- multimodal chat against the current GUI context,
- optional canvas or window sharing,
- speech input and reply read-aloud controls,
- optional web/MCP/browser tools depending on configuration,
- model/plugin execution against videos through bot tools,
- Box file operations through natural-language requests such as listing, searching, downloading, or uploading project files,
- draft-and-send support for configured Zulip targets from the bot UI.

Install optional Bot dependencies only when you need them. See [Installation](installation.md) for the `annolid_bot`, `bot`, `ai_chat`, and related extras.

When you use background integrations such as Zulip, WhatsApp, calendar services, or Box, the agent config is loaded from the Annolid agent config path, not the LLM settings file.

For Google Calendar, Annolid now prefers an existing cached token by default. If you want the agent to perform first-run OAuth authorization, set `tools.google_auth.allowInteractiveAuth=true` explicitly in the agent config and run that flow from an interactive session.

For Box, you can bootstrap OAuth with `agent-box-auth-url` and `agent-box-auth-exchange`; when Box uses an org-specific auth host, pass `--authorize-base-url` such as `https://ent.box.com` or `https://my_org_xxx.account.box.com`. The GUI also exposes editable Box auth fields for the same settings, including client secret, suggests `http://localhost:8765/oauth/callback` as the local redirect URI, persists the callback URL across sessions, and starts a temporary local callback listener so Box does not bounce the browser to a dead `localhost` page. Use `http://` for local redirects; `https://localhost/...` will not work with the local listener. When `tools.box.auto_refresh=true`, the Box tool will refresh expired access tokens using `tools.box.refresh_token`.

See:

- [MCP](mcp.md)
- [Agent Calendar](agent_calendar.md)
- [Tutorials](tutorials.md)

## 3. CLI Model Workflow

The current terminal entry point is `annolid-run`.

COCO dataset handling for model workflows is documented in
[COCO Data Flow](coco_data_flow.md).

Useful commands:

```bash
annolid-run --help
annolid-run list-models
annolid-run help train
annolid-run help predict
annolid-run help train <model>
annolid-run help predict <model>
```

Annolid supports plugin-oriented train/predict flows for multiple backends. Shared YAML run-config templates live under `annolid/configs/runs/`.

Built-in model help now starts with curated groups such as `Required inputs`,
`Model and runtime`, and `Training controls` before the full flag list.

Example:

```bash
annolid-run train dino_kpseg --run-config annolid/configs/runs/dino_kpseg_train.yaml
```

If you want the GUI agent to perform read-only CLI inspection or explicitly
invoke Annolid-native commands, use the dedicated typed tool flow described in
[Annolid Agent and annolid-run](agent_annolid_run.md).

Recommended practice:

- Run `annolid-run help <command> <model>` before a new job.
- Keep run configs under version control for repeated studies.
- Validate on a small dataset split before scaling training or inference.

## 4. Behavior Analysis Workflow

For behavior-centric projects, a typical path is:

1. annotate events or frame-level flags in the GUI,
2. review time ranges and fix labeling drift,
3. export event data,
4. summarize or post-process results with the CLI utilities.

For GUI-first behavior scoring with timeline playhead dragging, shared behavior catalog usage across Flags/Timeline, and Annolid Bot segment labeling prompts, see:

- [Behavior labeling with Timeline, Flags, and Annolid Bot](tutorials/behavior_timeline_flags_bot.md)
- [Behavior Agent User Guide](behavior_agent_user_guide.md)

Example time-budget command:

```bash
python -m annolid.behavior.time_budget exported_events.csv \
    --schema project.annolid.json \
    --bin-size 60 \
    -o time_budget.csv
```

### Polygon Classifier Workbench

For frame-level behavior classifiers based on two animal polygons, use:

- `AI & Models -> Polygon Classifier Workbench...`

The workbench has four tabs:

1. `Polygon CSV`: merges Annolid predicted polygon JSON or annotation-store
   NDJSON files with a manual one-hot label CSV and writes a polygon-points CSV.
2. `Legacy Dataset`: builds `train_dataset.csv` and `test_dataset.csv` from LabelMe
   frame folders.
3. `Train`: trains the temporal polygon-frame classifier and writes a run
   folder with checkpoint, metrics, and test predictions.
4. `Inference`: loads a checkpoint and writes per-frame predicted labels,
   confidence, and class probabilities to CSV.

Expected dataset layout:

```text
train/
  video_001/
    frame_000001.json
    frame_000002.json
test/
  video_002/
    frame_000001.json
```

Each LabelMe JSON needs one truthy behavior flag and polygon shapes labeled
`intruder` and `resident`. Optional `<video_name>_tracked.csv` files beside the
video folders can provide motion-index features; missing files default motion
features to `0`.

For predicted-shape workflows where labels live in a separate manual CSV, use
the `Polygon CSV` tab instead. It accepts:

- an Annolid frame folder containing predicted polygon JSON files and/or
  `*_annotations.ndjson` predicted polygon records,
- a manual one-hot behavior label CSV with frame numbers in `frame_number`,
  `frame`, or the first column,
- an output CSV path.

The generated polygon-points CSV contains `video`, `frame`, `frame_number`,
`label`, and one `<shape_label>_features` column per polygon label, plus area,
centroid, perimeter, and `<shape_label>_motion_index` columns when available.

## Identity Repair Workflow

For difficult multi-animal sessions where identity can flip after overlap/occlusion:

1. run tracking as usual,
2. define a policy for your assay (distance/zone/area thresholds),
3. run Identity Governor in dry-run mode,
4. inspect the generated correction report,
5. apply corrections and regenerate downstream summaries.

See [Identity Governor](identity_governor.md).

Dry-run the policy first. Review the generated correction report before applying changes to downstream analysis files.

## 5. Depth Workflow

The current GUI includes Video Depth Anything integration. For setup,
checkpoint handling, output formats, and troubleshooting, see
[Video Depth Anything](video_depth_anything.md).

Use:

- `View -> Video Depth Anything...`
- `View -> Depth Settings...`

The current implementation can auto-download checkpoints when needed and writes
a `depth.ndjson` sidecar plus optional rendered outputs depending on the
selected save options.

## 6. Optional 3D Workflows

Annolid supports two separate optional 3D paths:

- SAM 3D reconstruction from a video/object workflow, documented in [SAM 3D](sam3d.md).
- Brain 3D reconstruction from sagittal polygon annotations, documented in
  [Brain 3D Reconstruction](brain_3d_reconstruction.md).

Use the Brain 3D path when you need canonical sagittal-to-coronal reconstruction,
region presence controls per plane, and synchronized 2D/3D editing loops.

## 7. Simulation and FlyBody Workflow

Annolid now includes a simulation-oriented CLI path for converting pose tracks
into backend-specific targets and enriched NDJSON outputs.

Current capabilities:

- generate FlyBody mapping templates,
- validate mappings with `--dry-run`,
- lift 2D keypoints into 3D from `depth.ndjson`,
- smooth trajectories and fill small gaps before fitting.

Start here:

```bash
annolid-run help predict flybody
annolid-run predict flybody --pose-schema pose_schema.json --write-mapping-template flybody.yaml
annolid-run predict flybody --input pose.ndjson --depth-ndjson depth.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --dry-run
```

See [Simulation and FlyBody](simulation_flybody.md).

## Recommended Operating Pattern

- Keep annotation and model changes incremental.
- Prefer short label -> run -> review loops.
- Treat on-disk annotation files as the contract.
- Validate new model workflows on a small subset before scaling to a full dataset.
- Record model weights, run configs, prompts, and exported CSV locations with the project so results can be reproduced later.
