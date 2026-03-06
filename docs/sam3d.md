# SAM 3D Objects

Annolid includes an optional SAM 3D reconstruction workflow in the GUI.

It is intentionally isolated from the base install because the upstream dependency stack is heavy and often needs its own environment.

## Current Integration Model

The GUI-side integration is managed by the SAM 3D manager in `annolid/gui/widgets/sam3d_manager.py`.

At runtime it:

- reads a `sam3d` config block from the GUI config/state,
- falls back to `SAM3D_HOME` and `SAM3D_PYTHON` environment variables,
- checks availability only when you invoke the workflow,
- runs either in-process or through a separate Python executable.

## Typical Setup

Create a dedicated environment for upstream SAM 3D dependencies:

```bash
conda create -n sam3d-env python=3.10
conda activate sam3d-env
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
pip install -e .
```

Then record:

- the SAM 3D repository path
- the Python executable for that environment, for example `~/miniconda3/envs/sam3d-env/bin/python`

## Configuration Options

The current integration supports values such as:

```yaml
sam3d:
  repo_path: /path/to/sam-3d-objects
  checkpoints_dir: /path/to/sam-3d-objects/checkpoints
  checkpoint_tag: hf
  python_executable: /path/to/sam3d-env/bin/python
  output_dir: /path/to/sam3d_outputs
  seed: 42
  compile: false
  timeout_s: 600
```

Environment-variable fallbacks:

- `SAM3D_HOME`
- `SAM3D_PYTHON`

## GUI Workflow

1. Open a video in Annolid.
2. Select an instance shape on the current frame.
3. Run the SAM 3D reconstruction action from the GUI.
4. Annolid generates the mask/frame job, runs SAM 3D, and writes the output directory.
5. On success, Annolid reports the saved PLY path and may open the bundled VTK-based viewer if available.

## Output Behavior

The current code writes:

- a PLY reconstruction result
- sidecar metadata describing the video, frame index, and label
- output under either the configured `output_dir` or a default `<video_stem>_sam3d` folder next to the source video

## Operational Notes

- Normal Annolid use does not require SAM 3D.
- Availability is checked lazily when you click the action.
- If the external stack is missing, the GUI should report what is unavailable rather than failing at startup.
