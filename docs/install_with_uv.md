# Installing Annolid with uv

This guide shows how to set up Annolid using [uv](https://docs.astral.sh/uv/), Astral's fast drop-in replacement for `pip` and `virtualenv`. The workflow lets you create lightweight environments, install the package editable for development, and reproduce dependencies with a lock file.

## Prerequisites
- Python 3.9–3.11 available on your PATH (Annolid targets 3.6+, but newer releases are tested on 3.11).
- `git` for cloning this repository.
- `ffmpeg` (recommended) from your system package manager or Conda.
- `uv` installed. You can grab the standalone binary from Astral or install it into your user site-packages:

```bash
pip install --user uv
```

> **Tip:** On macOS and Linux, place the downloaded binary or the `~/.local/bin` directory on your PATH so the `uv` command is available globally.

## 1. Clone Annolid
```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
```

## 2. Create an isolated environment
```bash
uv venv .venv --python 3.11
```

Activate it with:

- macOS/Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

You can reuse this environment on future shells with the same activate command.

## 3. Install Annolid and its dependencies
With the environment active, let `uv` resolve and install everything declared in `pyproject.toml` (including editable mode for development):

```bash
uv pip install -e .
```

This step pulls PyPI wheels plus the bundled Git dependency for Segment Anything HQ. The install may take a few minutes on the first run.

## 4. (Optional) Generate a lock file for reproducible installs
To capture the exact versions resolved on your machine, write them to `uv.lock`:

```bash
uv pip compile pyproject.toml -o uv.lock
```

Share the lock file with teammates, then they can reproduce the environment exactly via:

```bash
uv pip sync uv.lock
```

## 5. Launch Annolid
```bash
annolid
```

If you need behavior logging or segmentation models, drop your trained weights under `annolid/models` or point the GUI to them through the **Model Manager**.

## Updating or rebuilding
- Upgrade dependencies in-place: `uv pip install -U annolid`.
- Recreate a clean environment: remove `.venv`, repeat the steps above, or run `uv pip sync uv.lock` inside the activated venv to match the locked versions.

## Troubleshooting
- **CUDA / GPU builds:** Ensure the appropriate PyTorch build is available. You can pin versions with `uv pip install "torch==2.2.0" "torchvision==0.17.0"` after the main install.
- **Apple Silicon MKL errors:** Use Python 3.11 and let `uv` fetch arm64 wheels. If MKL still loads, set `export OMP_NUM_THREADS=1` or reinstall with `UV_INDEX_URL=https://pypi.org/simple`.
- **FFmpeg / ffprobe missing:** Install FFmpeg (which bundles `ffprobe`) via Homebrew (`brew install ffmpeg`), apt (`sudo apt install ffmpeg`), or conda-forge (`conda install -c conda-forge ffmpeg`). Make sure the binaries are on your PATH (`which ffprobe` should resolve).
- **`ModuleNotFoundError: No module named 'zmq'`:** Ensure you are installing with the updated `pyproject.toml` that depends on `pyzmq`. If you already have an environment, run `uv pip install pyzmq` inside it and retry `annolid`.
- **`ModuleNotFoundError: No module named 'ffpyplayer'`:** The GUI’s real-time widgets use ffpyplayer’s video frame utils. Install it with `uv pip install ffpyplayer` or recreate the environment after pulling the updated dependency list.
- **`ModuleNotFoundError: No module named 'tree_config'`:** Realtime configs rely on `tree-config`. Install it with `uv pip install tree-config` or rebuild the environment after syncing the latest dependencies.
- **Polygon edges look noisy:** The mask-to-polygon conversion now uses OpenCV’s Douglas–Peucker simplifier. Adjust `approxpoly_epsilon` in `configs/default_config.yaml` if you need smoother (higher value) or more detailed (lower value) outlines.

You're now ready to track behaviors using the Annolid GUI or the CLI tools (`annolid-track`, `annolid-train`) from the activated `uv` environment.
