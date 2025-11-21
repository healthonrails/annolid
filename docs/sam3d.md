# SAM 3D Objects (Optional Integration)

SAM 3D Objects is heavy and gated. Annolid keeps it optional and can run it from a separate conda env.

## Install SAM 3D in its own env

```bash
conda create -n sam3d-env python=3.10
conda activate sam3d-env
git clone https://github.com/facebookresearch/sam-3d-objects.git
cd sam-3d-objects
# follow their README; typical steps:
pip install -e .  # plus any extras they require (p3d/inference)
# download checkpoints (HuggingFace access required)
```

Record the SAM3D env Python path (e.g. `~/miniconda3/envs/sam3d-env/bin/python`).

## Configure Annolid

In `~/.labelmerc` or your Annolid config, add the optional `sam3d` block:

```yaml
sam3d:
  repo_path: /path/to/sam-3d-objects
  checkpoints_dir: /path/to/sam-3d-objects/checkpoints   # optional if default
  checkpoint_tag: hf                                      # matches checkpoints/<tag>/pipeline.yaml
  python_executable: /path/to/sam3d-env/bin/python        # run in separate env; omit to run in-process
  output_dir: /path/to/sam3d_outputs                      # optional; defaults near the video
  seed: 42                                                # optional
  compile: false                                          # optional
```

You can also set environment variables:
- `SAM3D_HOME` for the repo path
- `SAM3D_PYTHON` for the Python executable (separate env)

Annolid only checks availability when you click the action; normal operation does not require SAM3D.

## Usage in the GUI

1. Open a video and select an instance (polygon) on the current frame.
2. Choose **View → Reconstruct 3D (SAM 3D)…** on the toolbar/menu.
3. Annolid extracts the frame + mask, runs SAM3D (in the chosen env), and saves:
   - PLY Gaussian splat
   - Sidecar JSON with video/frame/label and SAM3D config info
4. The status bar and a dialog show the saved path so you can open it in your preferred 3D viewer.

If availability fails, Annolid will show the missing piece (Python path, repo, checkpoints, or import errors).
