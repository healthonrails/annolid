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
- Prompts for optional extras and launch.

## Quick decision table

Choose the flag set that matches your goal.

| Goal | macOS / Linux | Windows (PowerShell) |
|---|---|---|
| Default install | `... | bash` | `... | iex` |
| Skip GPU detection | `... | bash -s -- --no-gpu` | `...; install.ps1 -NoGpu` |
| Non-interactive install | `... | bash -s -- --no-interactive` | `...; install.ps1 -NoInteractive` |
| Install to custom folder | `... | bash -s -- --install-dir /path/to/annolid` | `...; install.ps1 -InstallDir C:\path\to\annolid` |
| Custom venv location | `... | bash -s -- --venv-dir /path/to/.venv` | `...; install.ps1 -VenvDir C:\path\to\.venv` |
| Use Conda env (Linux/macOS only) | `... | bash -s -- --use-conda` | Not supported in `install.ps1` |
| Enable optional extras | `... | bash -s -- --extras sam3,text_to_speech` | `...; install.ps1 -Extras sam3,text_to_speech` |

## Linux/macOS options in detail

`install.sh` supports:
- `--install-dir DIR`
- `--venv-dir DIR`
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
- `-Extras EXTRAS`
- `-NoGpu`
- `-NoInteractive`
- `-Help`

For extra options, run:

```powershell
Get-Help .\install.ps1 -Detailed
```

Examples:

```powershell
# Run installer from local script with explicit options
.\install.ps1 -InstallDir C:\annolid -NoGpu -NoInteractive

# Install optional features
.\install.ps1 -Extras sam3,image_editing,text_to_speech
```

## Optional extras (`--extras` / `-Extras`)

`gui` is installed by default by the one-line installers, so `annolid` launches without additional flags.

Current supported extras:
- `sam3`
- `image_editing`
- `text_to_speech`
- `qwen3_embedding`
- `annolid_bot`

### What each extra is for (with example use cases)

| Extra | Install this when you need... | Example use case |
|---|---|---|
| `sam3` | the SAM3-related segmentation workflow/features in Annolid | You want stronger promptable segmentation on difficult frames and plan to use SAM3 tools in the GUI/CLI pipeline. |
| `image_editing` | diffusion-based image editing/generation dependencies | You are preparing augmented training images (inpainting/background edits) as part of annotation or data curation. |
| `text_to_speech` | built-in narration/read-aloud features | You want captions/notes read aloud during review, accessibility workflows, or hands-free labeling sessions. |
| `qwen3_embedding` | Qwen3-based embedding/multimodal utilities | You plan to run embedding-powered retrieval/comparison flows that rely on the Qwen3 stack. |
| `annolid_bot` | bundled Annolid Bot integrations (WhatsApp + Google Calendar + MCP) | You want background services/integrations without installing each Bot extra individually. |

### Practical install suggestions

- Minimal install (fastest, lowest dependency footprint): no extras.
- Most common research annotation setup: start with `sam3`.
- Accessibility or narrated review: add `text_to_speech`.
- Data augmentation/image synthesis workflows: add `image_editing`.
- Only install `qwen3_embedding` if you explicitly use those embedding features.
- If you use Annolid Bot integrations, add `annolid_bot`.

Use comma-separated values with no spaces, for example:

```bash
--extras sam3,text_to_speech
```

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
