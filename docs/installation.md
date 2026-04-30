# Installation

Annolid currently exposes two main commands after install:

- `annolid`: launch the desktop GUI
- `annolid-run`: run model and workflow plugins from the terminal

Annolid requires Python 3.10 or newer. In practice, Python 3.11 is the safest default for new environments.

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
- `large_image`: TIFF/OME-TIFF metadata and optional streaming backends (`tifffile`, `pyvips`, `openslide-python`)
- `sam3`: SAM3-related dependencies
- `image_editing`: diffusion/image-editing features
- `text_to_speech`: read-aloud and narration features
- `qwen3_embedding`: embedding-related utilities
- `mediapipe`: MediaPipe-based workflows
- `cowtracker`: CowTracker backend dependency
- `annolid_bot`: Annolid Bot integrations such as MCP, Playwright, WhatsApp bridge support, and Google Drive/Calendar dependencies

Example:

```bash
pip install -e ".[gui,large_image,annolid_bot,text_to_speech]"
```

### Note on `large_image`

`large_image` is optional. A normal `annolid[gui]` install does not require these dependencies, and Annolid's standard image/video annotation workflows continue to work without them.

The `large_image` extra installs the Python packages, but some platforms also need native runtimes for the fastest backends:

- `pyvips` needs a working `libvips` runtime
- `openslide-python` needs a working OpenSlide runtime

If those native libraries are missing, Annolid falls back to `tifffile` when available. If the full `large_image` extra is not installed, Annolid still starts normally and falls back to the standard Qt/Pillow image path instead of requiring large-image packages at startup.

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
- If you use Annolid Bot with MCP or browser automation, install the `annolid_bot` extra.
- If `qtpy.QtBindingsNotFoundError` appears, install the `gui` extra in the active environment.
- The `gui` extra now installs `PySide6` as the default Qt binding.

## Next Steps

- [Workflows](workflows.md) for the main GUI and CLI paths
- [MCP](mcp.md) for Annolid Bot integrations
- [SAM 3D](sam3d.md) for the optional 3D reconstruction setup
