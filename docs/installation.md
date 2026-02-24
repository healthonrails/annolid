# Annolid Installation Guide

This document covers detailed installation instructions for Annolid, including alternative methods (Conda, Pip, Docker) and troubleshooting tips.

## Table of Contents
- [Quick Start (Anaconda)](#quick-start-anaconda)
- [Alternative Installation Methods](#alternative-installation-methods)
  - [Pip Only](#pip-only-installation)
  - [uv (High Performance)](#uv-lightweight-venv--installer)
  - [MediaPipe Integration (Optional)](#mediapipe-integration-optional)
  - [Docker](#docker)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Anaconda)

If you prefer using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) to manage environments:

```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
conda install git ffmpeg
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e ".[gui]"
annolid  # launches the GUI
```

This method is reliable on most systems. If you need specific CUDA versions for GPU acceleration, see the "Conda environment" section below.

---

## Alternative Installation Methods

### Conda environment (GPU-ready, Ubuntu 20.04 tested)
Use the provided `environment.yml` for a reproducible environment:

```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
conda env create -f environment.yml
conda activate annolid-env
annolid
```

**Note:** If you see `CUDA capability sm_86 is not compatible with the current PyTorch installation`, install a matching build:
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cudatoolkit=12.1 -c pytorch -c nvidia
```

### Pip-only installation
Works well on machines without Conda. You are responsible for installing system dependencies like `ffmpeg`.

```bash
python -m venv annolid-env
source annolid-env/bin/activate
pip install --upgrade pip
pip install "annolid[gui]"
pip install "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
annolid
```

### uv (lightweight venv + installer)
Use [uv](https://docs.astral.sh/uv/) for extremely fast environment creation and dependency resolution:

```bash
pip install --user uv  # or grab the standalone binary
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[gui]"
annolid
```

### Apple Silicon (macOS M1/M2)
Some Intel-specific libraries can trigger MKL errors on Apple Silicon (`Intel MKL FATAL ERROR`). To fix this, recreate the environment with native wheels:

```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
source .venv/bin/activate
pip install -e ".[gui]"
annolid
```

### MediaPipe Integration (Optional)

MediaPipe provides high-performance real-time pose, hand, and face landmark detection. It is an optional dependency.

To install MediaPipe support:

```bash
# Recommended: Install as an extra
pip install "annolid[mediapipe]"

# Alternatively, the GUI will prompt to auto-install it
# the first time you select a MediaPipe model.
```

---

## Docker

Ensure [Docker](https://www.docker.com/) is installed.

```bash
cd annolid/docker
docker build .
# Linux only; allows GUI forwarding
xhost +local:docker
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY <IMAGE_ID>
```
Replace `<IMAGE_ID>` with the identifier printed by `docker build`.

---

## Troubleshooting

- **Video playback errors** (`OpenCV: FFMPEG: tag ...` or missing codecs):
  Install FFmpeg via your package manager or `conda install -c conda-forge ffmpeg`.
- **Windows: ONNX Runtime import error on launch** (`ImportError: DLL load failed while importing onnxruntime_pybind11_state`):
  This can be triggered by an OpenMP runtime conflict between PyTorch and ONNX Runtime. Set `KMP_DUPLICATE_LIB_OK=TRUE` in your environment and retry launching. Newer Annolid versions set this automatically when launching the GUI.
- **macOS Qt warning** (`Class QCocoaPageLayoutDelegate is implemented in both ...`):
  `conda install qtpy` normally resolves the conflict between OpenCV and PyQt.
- **`qtpy.QtBindingsNotFoundError: No Qt bindings could be found`:**
  install GUI extras in your active environment: `pip install -e ".[gui]"` (from source) or `pip install "annolid[gui]"` (from PyPI).
- **Launch failures**:
  Confirm the correct environment is active (`conda activate annolid-env` or `source .venv/bin/activate`) and run `annolid --help`.
- **Advanced Training/Inference**:
  Use `annolid-run list-models`, `annolid-run train`, and `annolid-run predict` for CLI operations.
