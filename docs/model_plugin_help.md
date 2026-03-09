# Model Plugin Help

Use this page to understand the current `annolid-run` model help flow as both a
user and a plugin author.

## User Help Flow

Start at the broadest level, then narrow down only when you need more detail.

```bash
annolid-run --help
annolid-run help train
annolid-run help predict
annolid-run help train dino_kpseg
annolid-run help predict yolo
```

Equivalent older forms still work:

```bash
annolid-run train dino_kpseg --help-model
annolid-run predict yolo --help-model
```

## What Built-In Model Help Shows

Built-in model plugins now expose curated quick-reference groups before the full
flag list.

Typical groups include:

- `Required inputs`
- `Outputs and run location`
- `Model and runtime`
- `Training controls`
- `Inference controls`
- `Data and augmentation`
- `Saving and reporting`

That means a command like:

```bash
annolid-run help predict yolo_labelme
```

shows the important flags first, instead of forcing you to scan the full
argparse dump from top to bottom.

## Recommended User Pattern

1. Run `annolid-run list-models` to discover the available model names.
2. Use `annolid-run help train` or `annolid-run help predict` to understand the
   common entry pattern.
3. Use `annolid-run help train <model>` or `annolid-run help predict <model>`
   for the specific plugin you want.
4. Only then run the actual train or predict command.

## Plugin Author Contract

Built-in plugins now use explicit help metadata instead of relying on heuristic
flag grouping.

`annolid.engine.registry.ModelPluginBase` supports:

- `train_help_sections`
- `predict_help_sections`
- `get_help_sections(mode)`

Each section is declared as:

```python
(
    ("Required inputs", ("--data", "--weights")),
    ("Training controls", ("--epochs", "--batch")),
)
```

Annolid now also discovers third-party model plugins from the Python entry
point group `annolid.model_plugins`. This lets heavy optional backends stay in a
separate package while still appearing in:

```bash
annolid-run list-models
annolid-run help predict <model>
annolid-run predict <model> ...
```

Example `pyproject.toml` for an external plugin package:

```toml
[project.entry-points."annolid.model_plugins"]
flybody = "annolid_flybody.plugin:FlyBodyPlugin"
```

## Simulation Backend Pattern

For physics or kinematics integrations, prefer keeping backend-specific code in a
plugin package and reuse the core helpers under `annolid.simulation` for:

- typed pose frames (`Pose2DFrame`, `Pose3DFrame`),
- adapter contracts (`SimulationAdapter`),
- mapping/config loading (`load_simulation_mapping`),
- LabelMe or NDJSON pose ingestion (`read_pose_frames`),
- schema-valid simulation export (`write_simulation_ndjson`).

That split keeps Annolid core responsible for stable IO contracts while the
simulation package owns heavy runtime dependencies and backend behavior.

Current built-in examples:

```bash
annolid-run help predict simulation_runner
annolid-run predict simulation_runner --backend identity --input pose.json --mapping sim.json --out-ndjson sim.ndjson
annolid-run predict flybody --pose-schema pose_schema.json --write-mapping-template flybody.yaml
annolid-run predict flybody --input pose.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --dry-run
annolid-run predict flybody --input pose.ndjson --depth-ndjson depth.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --dry-run
annolid-run predict flybody --input pose.ndjson --depth-ndjson depth.ndjson --mapping flybody.yaml --out-ndjson flybody.ndjson --smooth-mode ema --max-gap-frames 2 --dry-run
```

If you install the optional FlyBody runtime into `.venv`, verify that stack
before the first non-`--dry-run` invocation:

```bash
uv pip install --python .venv/bin/python dm-control mujoco dm-tree mediapy h5py
uv pip install --python .venv/bin/python --no-deps -e /path/to/flybody
python scripts/check_flybody_runtime.py
```

Wrapper:

```bash
scripts/setup_flybody_uv.sh --flybody-path /path/to/flybody
```

Use `--venv-dir .venv311 --python 3.11` if you want an isolated FlyBody env
without modifying the default `.venv`.

Checked example template:

```bash
annolid/configs/flybody_template.yaml
```

Use it directly as a starting point, or generate a project-specific copy with:

```bash
annolid-run predict flybody --pose-schema pose_schema.json --write-mapping-template flybody.yaml
```

## Plugin Author Guidelines

- Add explicit help sections for every supported mode.
- Group flags by user intent, not implementation detail.
- Put the minimum viable path first:
  - required paths or inputs,
  - output location,
  - runtime/model choice,
  - train or predict tuning knobs.
- Keep the section names stable and human-readable.
- Treat the full parser output as the detailed reference, not the primary UX.
- Keep module imports light so discovery does not pull in heavy runtime
  dependencies before the user actually runs `train()` or `predict()`.

## Fallback Behavior

Third-party or older plugins that do not declare explicit help sections still
work. The CLI falls back to heuristic grouping for those cases.

Built-in plugins in this repository are expected to define explicit sections,
and the test suite enforces that contract.
