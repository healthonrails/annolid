# One-Line Installer Choices (Detailed Guide)

This guide explains how to choose the right one-line installer command and flags for your setup.

## Default one-liner (most users)

Use this first unless you have a specific requirement.

### macOS / Linux

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
```

What the installer does:

- Clones Annolid.
- Creates an isolated environment (`.venv` by default).
- Installs dependencies and Annolid.
- Detects GPU and installs CUDA-enabled PyTorch when appropriate.
- Installs and validates ONNX Runtime providers.
- Prompts for optional extras and launch.

## Quick decision table

Choose the flag set that matches your goal.

| Goal | macOS / Linux | Windows (PowerShell) |
| --- | --- | --- |
| Default install | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash` | `irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 \| iex` |
| Skip GPU detection | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --no-gpu` | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -NoGpu` |
| Non-interactive install | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --no-interactive` | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -NoInteractive` |
| Named profile | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --profile workstation` | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -Profile workstation` |
| Install to custom folder | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --install-dir /path/to/annolid` | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -InstallDir C:\\path\\to\\annolid` |
| Custom venv location | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --venv-dir /path/to/.venv` | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -VenvDir C:\\path\\to\\.venv` |
| Use Conda env | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --use-conda` | `Not supported in install.ps1` |
| Enable optional extras | `curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh \| bash -s -- --extras sam3,text_to_speech` | `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -Extras sam3,text_to_speech` |

## Linux/macOS options in detail

`install.sh` supports:

- `--install-dir DIR`
- `--venv-dir DIR`
- `--profile minimal|gui|workstation|full`
- `--extras EXTRAS`
- `--no-gpu`
- `--use-conda`
- `--no-interactive`
- `--help`

Examples:

```bash
# Non-interactive CPU-only install to a custom directory
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | \
  bash -s -- --install-dir ~/tools/annolid --no-gpu --no-interactive

# Use Conda instead of venv
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | \
  bash -s -- --use-conda

# Install with optional features
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | \
  bash -s -- --extras sam3,image_editing,text_to_speech
```

## Windows options in detail

`install.ps1` supports:

- `-InstallDir DIR`
- `-VenvDir DIR`
- `-Profile minimal|gui|workstation|full`
- `-Extras EXTRAS`
- `-NoGpu`
- `-NoInteractive`
- `-Help`

For extra options, run:

```powershell
.\install.ps1 -Help
```

Examples:

```powershell
# Run installer from local script with explicit options
.\install.ps1 -InstallDir C:\annolid -NoGpu -NoInteractive

# Run remote installer with explicit options
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1))) -NoGpu -NoInteractive

# Install optional features
.\install.ps1 -Extras sam3,image_editing,text_to_speech
```

## Optional extras (`--extras` / `-Extras`)

`gui` is installed by default by the one-line installers, so `annolid` launches without additional flags.

Named profiles provide stable defaults for common machines:

| Profile | Installed optional extras | Use this for |
|---|---|---|
| `minimal` | none beyond the required GUI extra | Fastest default GUI/core annotation setup. |
| `gui` | none beyond the required GUI extra | Default profile; same behavior as omitting `--profile` / `-Profile`. |
| `workstation` | `tracking,sam3,training` | Maintained research workstations that need common tracking runtimes, promptable segmentation, and training dashboards. |
| `full` | `all` | Fully provisioned lab machines where dependency size is less important than breadth. |

Explicit extras are merged with the selected profile. For example,
`--profile workstation --extras ai_chat,text_to_speech` installs
`gui,tracking,sam3,training,ai_chat,text_to_speech`.

Current supported extras:

- `audio`
- `ai_chat`
- `ml`
- `tracking`
- `training`
- `yolo`
- `cutie`
- `realtime`
- `large_image`
- `remote_video`
- `sam3`
- `image_editing`
- `text_to_speech`
- `qwen3_embedding`
- `annolid_bot`
- `bot`
- `memory`
- `all`

### What each extra is for (with example use cases)

| Extra | Install this when you need... | Example use case |
|---|---|---|
| `audio` | audio decoding/playback helpers | You want waveform-derived behavior features, microphone recording, or sound playback without installing audio stacks in every lab workstation. |
| `ai_chat` | hosted OpenAI-compatible or Anthropic provider SDKs | You use GPT/OpenRouter/Claude-style providers instead of only local/Ollama-compatible paths. |
| `ml` | general ML/model runtime dependencies | You need PyTorch, Transformers, Hugging Face Hub, ONNX Runtime, Hydra/OmegaConf, and model tooling but do not want every optional workflow. |
| `tracking` | common tracking workstation dependencies | You need PyTorch, ONNX Runtime, Cutie-style config dependencies, and YOLO matching helpers on a maintained workstation. |
| `training` | TensorBoard dashboards and projector views | You train models from Annolid and want live training curves or embedding projector output. |
| `yolo` | Ultralytics YOLO/YOLOE workflows | You train YOLO models, run YOLOE prompt inference, or export YOLO detections to LabelMe JSON. |
| `cutie` | Cutie video object segmentation tracking runtime | You run Cutie tracking on minimal installs and want the required Python packages installed up front. Minimal/default installs can still auto-install missing Cutie packages on first use. |
| `realtime` | serial/ZMQ realtime hardware integrations | You use Bpod, realtime streams, or ZMQ control paths. |
| `large_image` | TIFF/OME-TIFF metadata and optional region-streaming backends | You want large TIFF, BigTIFF, OME-TIFF, or virtual-slide style viewing without forcing those native libraries into every install. Leave it out for the default GUI install. |
| `remote_video` | network/remote video decoding through `ffpyplayer` | You receive frames from Annolid's remote video player path. Leave it out for normal local video files and Python 3.14 default installs. |
| `sam3` | the SAM3-related segmentation workflow/features in Annolid | You want stronger promptable segmentation on difficult frames and plan to use SAM3 tools in the GUI/CLI pipeline. The installer also attempts the optional SAM-HQ dependency for this profile. |
| `image_editing` | diffusion-based image editing/generation dependencies | You are preparing augmented training images (inpainting/background edits) as part of annotation or data curation. |
| `text_to_speech` | built-in narration/read-aloud features | You want captions/notes read aloud during review, accessibility workflows, or hands-free labeling sessions. |
| `qwen3_embedding` | Qwen3-based embedding/multimodal utilities | You plan to run embedding-powered retrieval/comparison flows that rely on the Qwen3 stack. |
| `annolid_bot` | bundled Annolid Bot integrations (WhatsApp + Google Drive/Calendar + MCP) | You want background services/integrations without installing each Bot extra individually. |
| `bot` | Annolid Bot integrations plus memory dependencies | You want provider SDKs, channel integrations, MCP, browser automation, and LanceDB-backed memory in one extra. |
| `memory` | vector database dependencies for fast Annolid Bot memory | You want LanceDB-backed semantic memory rather than the lightweight lexical path. |
| `all` | full-feature workstation profile | You are setting up a maintained lab machine where install size is less important than having every optional integration ready. |

### Practical install suggestions

- Minimal install (fastest, lowest dependency footprint): no extras.
- Most common research annotation setup: start without extras; add `cutie` or `sam3` only when those workflows are needed.
- Common training workstation: add `yolo,training`.
- Hosted AI chat providers: add `ai_chat`.
- Accessibility or narrated review: add `text_to_speech`.
- Data augmentation/image synthesis workflows: add `image_editing`.
- Remote network video decoding: add `remote_video`.
- Realtime hardware/streaming workflows: add `realtime`.
- Only install `qwen3_embedding` if you explicitly use those embedding features.
- If you use Annolid Bot integrations, add `annolid_bot`.

Use comma-separated values with no spaces, for example:

```bash
--extras sam3,text_to_speech
```

### `large_image` runtime note

The `large_image` extra installs the Python bindings, but `pyvips` and `openslide-python` may still require native system runtimes on your machine. This extra is optional; Annolid's normal GUI workflows do not depend on it.

If those runtimes are not available, Annolid will still open the file and will fall back to `tifffile` when available, but navigation can be slower on very large TIFF files.

### `remote_video` runtime note

The `remote_video` extra installs `ffpyplayer`. On Python 3.14, `ffpyplayer` may need to build from source and require native FFmpeg development headers such as `libpostproc`. The default installer leaves this extra out so core GUI and local video workflows can install cleanly on Python 3.14.

### SAM-HQ runtime note

The one-line installers no longer install SAM-HQ during the default GUI setup. SAM-HQ is attempted only when a SAM-related profile such as `--extras sam3` / `-Extras sam3` or `all` is selected. This keeps the default installer independent of an extra GitHub source install and makes locked-down lab workstations easier to bootstrap.

### Cutie runtime note

The `cutie` extra installs the Python runtime packages used by Cutie tracking.
If a minimal/default install tries to run Cutie tracking and one of those
packages is missing, Annolid attempts to install the missing packages into the
active environment before loading the tracker. Set
`ANNOLID_AUTO_INSTALL_CUTIE_DEPS=0` to disable that automatic repair and show
the exact `python -m pip install ...` command instead.

## Recommended patterns

- Stable workstation install:
  use default one-liner first, then rerun with flags only if needed.
- Headless or CI environment:
  use `--no-interactive` (and `--no-gpu` if GPU drivers are unavailable).
- Shared lab server:
  set explicit `--install-dir` and `--venv-dir` to avoid confusion.

## GPU and device suggestions

Use this as a practical hardware guide before choosing flags.

| Device / Platform | Recommendation | Suggested installer choice |
|---|---|---|
| NVIDIA GPU workstation (Linux/Windows) | Best performance for training + large-batch inference. Keep CUDA drivers current. | Use default installer (do **not** pass `--no-gpu` / `-NoGpu`). |
| Apple Silicon (M1/M2/M3) | Good local inference performance via MPS; stable for many annotation/tracking tasks. | Use default installer on macOS. |
| CPU-only laptop/VM | Works for annotation and light inference, but slower for heavy models. | Use `--no-gpu` (Linux/macOS) or `-NoGpu` (Windows). |
| Shared HPC/server node | Prefer reproducibility and explicit paths/env control. | Use `--no-interactive`, explicit `--install-dir`/`--venv-dir`, optionally `--use-conda`. |

## ONNX Runtime CPU/GPU behavior

Annolid uses ONNX Runtime for several model paths. The one-line installers keep the provider choice explicit:

- Linux and Windows with an NVIDIA driver reporting CUDA 12.x or newer: install `onnxruntime-gpu` from the CUDA 12 package feed and validate that `CUDAExecutionProvider` is available.
- Linux and Windows with CUDA 13.x driver reports: use the same stable CUDA 12 ONNX Runtime GPU wheel because NVIDIA drivers are backward compatible with CUDA 12 runtimes.
- Linux and Windows without a CUDA 12-compatible NVIDIA driver: install CPU `onnxruntime` for compatibility.
- macOS: install `onnxruntime`; ONNX acceleration uses `CoreMLExecutionProvider` when the installed wheel and macOS runtime expose it. There is no CUDA ONNX Runtime wheel for macOS.
- `--no-gpu` / `-NoGpu`: skip GPU detection and install CPU ONNX Runtime.

If a CUDA 12-compatible NVIDIA driver is detected but `CUDAExecutionProvider` is missing after installation, the installer exits with a repair command instead of silently accepting a CPU-only ONNX Runtime install.

Manual repair inside the installer-created environment:

```bash
uv pip install --upgrade --force-reinstall onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Device-specific tips

- If you have an NVIDIA GPU, verify drivers and CUDA runtime before install (`nvidia-smi` should work).
- If GPU install fails or is unstable, rerun in CPU mode first (`--no-gpu` / `-NoGpu`) to get a working baseline quickly.
- For remote servers without display, combine `--no-interactive` and CPU mode unless GPU runtime is already validated.
- On Apple Silicon, use native arm64 Python/environment tools for best compatibility and speed.

## Verify installation

After install:

```bash
annolid --help
```

Launch GUI:

```bash
annolid
```

If `annolid` is not found, activate the environment printed by the installer and retry.

## Security note

Piping scripts from the internet is convenient but trust-based. For stricter security:

- download script first,
- inspect it,
- then run locally with explicit options.
