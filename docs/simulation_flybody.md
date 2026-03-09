# Simulation and FlyBody

This page documents the current Annolid simulation workflow, with FlyBody as
the first concrete backend.

Simulation support is additive and optional. Default Annolid GUI/CLI workflows
continue to work without FlyBody, MuJoCo, or `dm_control`.

## Workflow Diagram

```mermaid
flowchart LR
  A["Annolid pose input<br/>LabelMe JSON or pose.ndjson"] --> B["Optional preprocessing<br/>gap fill + smoothing"]
  B --> C{"depth.ndjson provided?"}
  C -- "no" --> D["2D path<br/>default_z fallback"]
  C -- "yes" --> E["3D lifting<br/>depth sampling + camera intrinsics"]
  D --> F["Simulation adapter<br/>identity or flybody"]
  E --> F
  F --> G["Dry run<br/>site target validation"]
  F --> H["Real backend run<br/>optional FlyBody IK"]
  G --> I["Annolid NDJSON output<br/>otherData.simulation"]
  H --> I
```

## What Exists Today

Annolid now supports:

- backend-neutral simulation IO through `annolid.simulation`,
- a built-in `simulation_runner` plugin for contract validation,
- a built-in `flybody` plugin for FlyBody-oriented mappings and outputs,
- optional depth-assisted 2D-to-3D lifting from Annolid `depth.ndjson`,
- temporal preprocessing before fitting:
  - gap filling,
  - EMA smoothing,
  - One Euro smoothing,
  - Kalman smoothing.

The current FlyBody path is designed to stay import-light. If FlyBody,
`dm_control`, or MuJoCo are not installed, `--dry-run` still works so you can
validate mappings and output contracts.

## Runtime Setup with `uv`

Use a repository-local `.venv` and install the optional FlyBody stack with
`uv pip`, not whatever global `pip` happens to resolve to:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install --python .venv/bin/python -e ".[gui]"
uv pip install --python .venv/bin/python dm-control mujoco dm-tree mediapy h5py
uv pip install --python .venv/bin/python --no-deps -e /path/to/flybody
python scripts/check_flybody_runtime.py
```

Or use the repo helper:

```bash
scripts/setup_flybody_uv.sh --flybody-path /path/to/flybody
```

To keep your current `.venv` untouched, point to a separate environment:

```bash
scripts/setup_flybody_uv.sh --venv-dir .venv311 --python 3.11 --flybody-path /path/to/flybody
```

Practical notes:

- Python 3.10 to 3.12 is the cleanest FlyBody target today.
- Python 3.13 can still require a local `labmaze` compatibility workaround.
- `--no-deps` on the editable FlyBody install keeps Annolid in control of the
  already-installed runtime packages inside `.venv`.
- `scripts/check_flybody_runtime.py` verifies import resolution and FlyBody
  environment creation before you run the Annolid plugin.
- `scripts/setup_flybody_uv.sh` wraps the same setup flow for a local FlyBody
  checkout.

## Core Commands

Discover the plugins:

```bash
annolid-run list-models
annolid-run help predict simulation_runner
annolid-run help predict flybody
```

## 1. Start From the Checked Template

Annolid ships a checked example config:

```bash
annolid/configs/flybody_template.yaml
```

Use it directly or copy it into your project and edit the site names to match
your FlyBody model.

Template fields:

- `keypoint_to_site`: Annolid keypoint label to FlyBody site name
- `site_to_joint`: optional site-to-joint lookup for diagnostics
- `coordinate_system.camera_intrinsics`: used for depth-assisted 3D lifting
- `metadata`: notes, provenance, and project-specific hints

## 2. Generate a Project-Specific Template

If you already have a pose schema, generate a mapping template from it:

```bash
annolid-run predict flybody \
  --pose-schema pose_schema.json \
  --write-mapping-template flybody.yaml
```

If you do not have a pose schema yet, provide keypoints directly:

```bash
annolid-run predict simulation_runner \
  --backend flybody \
  --template-keypoints nose,thorax,abdomen_tip,left_front_leg_tip,right_front_leg_tip \
  --write-mapping-template flybody.yaml
```

## 3. Validate Mapping Without FlyBody Installed

Use `--dry-run` to convert Annolid keypoints into FlyBody-style site targets and
write Annolid-compatible NDJSON output:

```bash
annolid-run predict flybody \
  --input pose.ndjson \
  --mapping flybody.yaml \
  --out-ndjson flybody.ndjson \
  --dry-run
```

This writes simulation metadata under:

```text
otherData.simulation
```

## 4. Add Depth for 3D Targets

If you have already generated Annolid depth sidecars, pass them in directly:

```bash
annolid-run predict flybody \
  --input pose.ndjson \
  --depth-ndjson depth.ndjson \
  --mapping flybody.yaml \
  --out-ndjson flybody.ndjson \
  --dry-run
```

When `coordinate_system.camera_intrinsics` is populated in the mapping file,
Annolid lifts image coordinates into 3D camera coordinates. If intrinsics are
missing, Annolid falls back to:

```text
(x_px, y_px, depth)
```

Recommended intrinsics fields:

- `fx`
- `fy`
- `cx`
- `cy`

## 5. Smooth and Fill Small Gaps

Preprocess pose tracks before lifting or fitting:

```bash
annolid-run predict flybody \
  --input pose.ndjson \
  --depth-ndjson depth.ndjson \
  --mapping flybody.yaml \
  --out-ndjson flybody.ndjson \
  --smooth-mode ema \
  --max-gap-frames 2 \
  --fps 30 \
  --dry-run
```

Supported smoothing modes:

- `none`
- `ema`
- `one_euro`
- `kalman`

Practical defaults:

- use `--max-gap-frames 1` or `2` for short occlusions,
- start with `--smooth-mode ema`,
- only move to `kalman` when tracks are noisy enough to justify prediction.

## 6. Run the Backend-Neutral Validation Path

Use the lightweight `identity` backend when you want to test data flow without
FlyBody-specific assumptions:

```bash
annolid-run predict simulation_runner \
  --backend identity \
  --input pose.json \
  --mapping sim.json \
  --out-ndjson sim.ndjson
```

## 7. Move to Real FlyBody Runtime

The non-`--dry-run` path is already wired for optional backend imports and has
been validated locally against a FlyBody checkout in `.venv`:

```bash
annolid-run predict flybody \
  --input pose.ndjson \
  --depth-ndjson depth.ndjson \
  --mapping flybody.yaml \
  --out-ndjson flybody.ndjson \
  --ik-max-steps 4000
```

Current expectation:

- FlyBody must be installed separately.
- `dm_control` and MuJoCo must be available in the same environment.
- Run `python scripts/check_flybody_runtime.py` before the first real backend
  invocation if you changed the environment.
- You may need to override callable locations if your FlyBody checkout differs:

```bash
annolid-run predict flybody \
  --input pose.ndjson \
  --mapping flybody.yaml \
  --out-ndjson flybody.ndjson \
  --env-factory flybody.fly_envs:walk_imitation \
  --ik-function flybody.inverse_kinematics:qpos_from_site_xpos
```

## Output Contract

Simulation runs preserve Annolid NDJSON compatibility and add backend data to:

```text
otherData.simulation.adapter
otherData.simulation.state
otherData.simulation.diagnostics
otherData.simulation.run_metadata
otherData.simulation.mapping_metadata
```

This keeps simulation results usable by existing Annolid tooling while exposing
backend-specific metadata for downstream analysis.

## Optional Runtime Test Coverage

Run the optional real-runtime smoke test locally:

```bash
source .venv/bin/activate
ANNOLID_RUN_FLYBODY_RUNTIME=1 pytest -m simulation tests/test_flybody_runtime_optional.py
```

This is intentionally excluded from default test runs and is covered by the
optional GitHub Actions workflow:

```text
.github/workflows/simulation-optional.yml
```

That workflow uses a headless MuJoCo configuration (`MUJOCO_GL=osmesa`) so
default Annolid CI does not need simulator graphics dependencies.

## Minimal End-to-End Example

```bash
source .venv/bin/activate

annolid-run predict flybody \
  --pose-schema pose_schema.json \
  --write-mapping-template flybody.yaml

annolid-run predict flybody \
  --input pose.ndjson \
  --depth-ndjson depth.ndjson \
  --mapping flybody.yaml \
  --out-ndjson flybody.ndjson \
  --smooth-mode ema \
  --max-gap-frames 2 \
  --dry-run
```

## Current Limits

- A clean Python 3.13 FlyBody install is still weaker than Python 3.10 to 3.12
  because upstream `labmaze` packaging may require a local workaround.
- Depth lifting currently uses nearest-pixel sampling.
- Multi-view lifting and camera extrinsics are not implemented yet.
- The current simulation frame model assumes one point per normalized label.

## Related Pages

- [Workflows](workflows.md)
- [Reference](reference.md)
- [Model Plugin Help](model_plugin_help.md)
