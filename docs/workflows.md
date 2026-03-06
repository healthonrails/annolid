# Workflows

This page summarizes the main workflows that are supported by the current Annolid codebase.

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
- Review overlap, occlusion, and high-motion segments before scaling to long runs.

## 2. Annolid Bot Workflow

Annolid Bot runs inside the GUI as a dedicated right-side dock.

Current bot workflow highlights:

- multimodal chat against the current GUI context,
- optional canvas or window sharing,
- speech input and reply read-aloud controls,
- optional web/MCP/browser tools depending on configuration,
- model/plugin execution against videos through bot tools,
- draft-and-send support for configured Zulip targets from the bot UI.

When you use background integrations such as Zulip, WhatsApp, or calendar services, the agent config is loaded from the Annolid agent config path, not the LLM settings file.

For Google Calendar, Annolid now prefers an existing cached token by default. If you want the agent to perform first-run OAuth authorization, set `tools.calendar.allow_interactive_auth=true` explicitly in the agent config and run that flow from an interactive session.

See:

- [MCP](mcp.md)
- [Agent Calendar](agent_calendar.md)
- [Tutorials](tutorials.md)

## 3. CLI Model Workflow

The current terminal entry point is `annolid-run`.

Useful commands:

```bash
annolid-run --help
annolid-run list-models
annolid-run train <model> --help-model
annolid-run predict <model> --help-model
```

Annolid supports plugin-oriented train/predict flows for multiple backends. Shared YAML run-config templates live under `annolid/configs/runs/`.

Example:

```bash
annolid-run train dino_kpseg --run-config annolid/configs/runs/dino_kpseg_train.yaml
```

## 4. Behavior Analysis Workflow

For behavior-centric projects, a typical path is:

1. annotate events or frame-level flags in the GUI,
2. review time ranges and fix labeling drift,
3. export event data,
4. summarize or post-process results with the CLI utilities.

Example time-budget command:

```bash
python -m annolid.behavior.time_budget exported_events.csv \
    --schema project.annolid.json \
    --bin-size 60 \
    -o time_budget.csv
```

## 5. Depth Workflow

The current GUI includes Video Depth Anything integration.

Use:

- `View -> Video Depth Anything...`
- `View -> Depth Settings...`

The current implementation can auto-download checkpoints when needed and writes a `depth.ndjson` sidecar plus optional rendered outputs depending on the selected save options.

## 6. Optional 3D Workflow

Annolid includes an optional SAM 3D reconstruction path from the GUI when configured.

Use cases:

- reconstruct a selected instance on the current frame,
- save a PLY result and sidecar metadata,
- optionally view the result in the bundled VTK-based viewer if available.

See [SAM 3D](sam3d.md).

## Recommended Operating Pattern

- Keep changes incremental.
- Prefer short label -> run -> review loops.
- Treat on-disk annotation files as the contract.
- Validate new model workflows on a small subset before scaling to a full dataset.
