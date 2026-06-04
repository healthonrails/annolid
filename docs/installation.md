# Installation

Annolid currently exposes two main commands after install:

- `annolid`: launch the desktop GUI
- `annolid-run`: run model and workflow plugins from the terminal

Annolid requires Python 3.10 or newer and supports Python 3.10 through 3.14 for the default GUI/core workflow. In practice, Python 3.11 or 3.12 remains the safest default for new shared lab environments.

## Recommended Paths

### One-line installer

For most users, start with the project installers:

- macOS and Linux:

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
```

- Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
```

See [One-Line Installer](one_line_install_choices.md) for flags, extras, CPU/GPU choices, and non-interactive installs.

The one-line installers also validate ONNX Runtime provider selection. On Linux and Windows, `onnxruntime-gpu` is installed when an NVIDIA driver reports CUDA 12.x or newer, including CUDA 13.x driver reports; otherwise Annolid uses CPU `onnxruntime`. On macOS, ONNX acceleration uses CoreML when available rather than a CUDA GPU wheel.

For a non-interactive CPU baseline, use:

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash -s -- --no-gpu --no-interactive
```

On Windows PowerShell:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -NoGpu -NoInteractive
```

### Local `.venv` with `uv`

For development or reproducible local work, use `uv` and a repository-local `.venv`:

```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[gui]"
```

The dedicated guide is [uv Setup](install_with_uv.md).

If you want ONNX Runtime GPU support in a manual `uv` environment on Linux or Windows, install Annolid first and then add the GPU wheel:

```bash
uv pip install --upgrade --force-reinstall onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### pip / venv

If you prefer standard Python tooling:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "annolid[gui]"
```

To install from source:

```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e ".[gui]"
```

### Conda

Conda still works well when you want environment management plus FFmpeg and native libraries from conda-forge:

```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
conda install -c conda-forge ffmpeg git
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e ".[gui]"
```

## Optional Extras

Useful extras currently defined in `pyproject.toml` include:

- `gui`: Qt bindings for the desktop application
- `audio`: audio loading/playback helpers (`librosa`, `sounddevice`)
- `ai_chat`: OpenAI-compatible and Anthropic SDKs for hosted LLM providers
- `training`: TensorBoard support for training dashboards and projector views
- `yolo`: Ultralytics YOLO/YOLOE workflows and tracker matching helpers
- `realtime`: serial and ZMQ dependencies for realtime/hardware integrations
- `onnx_gpu`: Windows/Linux ONNX Runtime CUDA provider
- `large_image`: TIFF/OME-TIFF metadata and optional streaming backends (`tifffile`, `pyvips`, `openslide-python`)
- `sam3`: SAM3-related dependencies
- `image_editing`: diffusion/image-editing features
- `text_to_speech`: read-aloud and narration features
- `qwen3_embedding`: embedding-related utilities
- `mediapipe`: MediaPipe-based workflows
- `cowtracker`: CowTracker backend dependency
- `remote_video`: network/remote video decoding through `ffpyplayer`
- `annolid_bot`: Annolid Bot integrations such as MCP, Playwright, WhatsApp bridge support, and Google Drive/Calendar dependencies
- `memory`: vector database dependencies for fast Annolid Bot memory
- `all`: convenience profile for full-feature workstations

Example:

```bash
pip install -e ".[gui,sam3,yolo,training,ai_chat]"
```

Annolid follows the same practical split used by mature annotation tools:
the default install keeps annotation, local video, tracking, and ONNX CPU paths
usable, while cloud providers, YOLO, audio, realtime hardware, large-image
backends, and heavyweight integrations are explicit extras. If a feature is
not installed, Annolid should start normally and show an install hint when that
feature is opened.

### Note on `large_image`

`large_image` is optional. A normal `annolid[gui]` install does not require these dependencies, and Annolid's standard image/video annotation workflows continue to work without them.

The `large_image` extra installs the Python packages, but some platforms also need native runtimes for the fastest backends:

- `pyvips` needs a working `libvips` runtime
- `openslide-python` needs a working OpenSlide runtime

If those native libraries are missing, Annolid falls back to `tifffile` when available. If the full `large_image` extra is not installed, Annolid still starts normally and falls back to the standard Qt/Pillow image path instead of requiring large-image packages at startup.

### Note on `remote_video`

`remote_video` is optional. Annolid's default install does not require `ffpyplayer`, which keeps Python 3.14 installs from failing when no compatible `ffpyplayer` wheel is available. Install it only if you use Annolid's network/remote video decoding path:

```bash
pip install -e ".[remote_video]"
```

On Python 3.14, `ffpyplayer` may build from source and require native FFmpeg development headers such as `libpostproc`.

## Verify the Install

Run:

```bash
annolid --help
annolid-run --help
```

Launch the GUI:

```bash
annolid
```

List available model plugins:

```bash
annolid-run list-models
```

## Common Post-install Notes

- Install FFmpeg if video import/export or codec support is incomplete.
- If ONNX Runtime GPU validation fails, activate the installer-created environment and rerun the repair command printed by the installer.
- If YOLO/YOLOE commands are missing, install `annolid[yolo]`.
- If hosted LLM providers are unavailable, install `annolid[ai_chat]`.
- If audio recording/playback is unavailable, install `annolid[audio]` or `annolid[text_to_speech]`.
- If you use Annolid Bot with MCP or browser automation, install the `annolid_bot` extra.
- If `qtpy.QtBindingsNotFoundError` appears, install the `gui` extra in the active environment.
- The `gui` extra now installs `PySide6` as the default Qt binding.

## Next Steps

- [Workflows](workflows.md) for the main GUI and CLI paths
- [MCP](mcp.md) for Annolid Bot integrations
- [SAM 3D](sam3d.md) for the optional 3D reconstruction setup
