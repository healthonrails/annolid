# Packaging and Distribution

This document defines Annolid's install tiers, optional runtime contract, frozen app artifact policy, and release validation expectations.

## Install Tiers

Annolid keeps the default one-line installer useful for common tracking/model workflows while preserving a `minimal` profile for smaller GUI-only setup.

| Tier | Command | Contract |
|---|---|---|
| Core | `pip install annolid` | Import-safe core package for non-GUI workflows and library use. Includes `pycocotools` so mask/RLE polygon paths are available by default. |
| GUI | `pip install "annolid[gui]"` | Desktop annotation GUI binding. The one-line installer's default `gui` profile also installs `ml`, `tracking`, and `cutie`. |
| ML | `pip install "annolid[ml]"` | General model runtime tier: PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, Hydra/OmegaConf, COCO tooling, and shared model helpers. |
| Tracking | `pip install "annolid[tracking]"` | Maintained tracking workstation tier with PyTorch, ONNX Runtime, Cutie-style config dependencies, YOLO matching helpers, and related image/model utilities. |
| Training | `pip install "annolid[training]"` | Training dashboard support such as TensorBoard. |
| Bot | `pip install "annolid[bot]"` | Annolid Bot providers, channel integrations, MCP/browser automation helpers, and memory dependencies. |
| Full | `pip install "annolid[all]"` | Convenience tier for fully provisioned lab machines where install size is less important than breadth. |

Feature-specific extras such as `cutie`, `sam3`, `yolo`, `large_image`, `remote_video`, `audio`, `text_to_speech`, `image_editing`, `ai_chat`, and `realtime` remain available for narrower installs.

## One-Line Installer Profiles

The macOS/Linux and Windows one-line installers expose stable profiles:

| Profile | Extras |
|---|---|
| `minimal` | `gui` only |
| `gui` | `gui,ml,tracking,cutie` |
| `workstation` | `gui,tracking,sam3,training` |
| `full` | `gui,all` |

The default profile is `gui`. SAM-HQ is attempted only when a SAM-related extra such as `sam`, `sam3`, `segment_anything`, or `all` is selected. This keeps default installs independent of optional GitHub source installs.

Each successful installer run writes `annolid-install-report.json` in the install directory. The report includes the selected profile, resolved extras, Python version, OS, architecture, package manager, GPU decision, ONNX Runtime providers, SAM-HQ status, and failed optional steps.

## Desktop Bundle Contract

The frozen desktop archives are intentionally lean Unix-style annotation
bundles. Current release binaries are built for macOS and Linux only, and their
baseline contract is opening media/annotations and drawing/editing polygons.
They must not bundle ML frameworks, training stacks, model checkpoints,
generated runs, or optional service clients.

When a user opens a model, tracking, bot, large-image, remote-video, or other
optional workflow, Annolid checks that workflow's runtime at the boundary where
the feature is launched. In a normal Python environment, missing optional
packages may be installed into that active environment. In a frozen desktop
bundle, Annolid reports the missing imports and the matching `pip install`
command instead of mutating the bundle.

## Optional Runtime Behavior

Frozen apps and lean installs intentionally exclude heavy optional runtimes. Code that opens optional workflows should use the central capability checker in `annolid.infrastructure.capabilities` to report:

- whether the runtime is available,
- which imports are missing,
- the exact install command,
- and the documentation link.

Optional runtime repair is controlled by `ANNOLID_AUTO_INSTALL_OPTIONAL_DEPS`.
Set `ANNOLID_AUTO_INSTALL_OPTIONAL_DEPS=0` to make optional workflows fail fast
with an install command instead of trying to install missing packages.

Cutie tracking keeps its narrower compatibility switch as well. If Cutie
packages are missing and `ANNOLID_AUTO_INSTALL_CUTIE_DEPS` is not disabled,
Annolid attempts to install the required packages into the active environment
before loading the tracker. Set `ANNOLID_AUTO_INSTALL_CUTIE_DEPS=0` to make the
workflow fail fast with an install command instead.

## Artifact Policy

Source distributions, wheels, PyInstaller folders, and final release archives must not include local checkpoints, model weights, generated runs, or heavyweight optional runtimes. The shared guard is:

```bash
python scripts/check_distribution_artifacts.py dist/*
python scripts/check_distribution_artifacts.py --kind bundle dist
```

The same policy rejects forbidden model suffixes such as `.pt`, `.pth`, `.onnx`, `.safetensors`, `.engine`, `.weights`, and known generated/runtime folders such as `checkpoints`, `weights`, `runs`, `torch`, `torchvision`, `transformers`, `onnxruntime`, `ultralytics`, and bundled model paths.

## Desktop Release Status

Current GitHub release binaries are PyInstaller onedir archives:

- macOS: `Annolid-macOS.zip`
- Linux: `Annolid-Linux.tar.gz`

Each archive is uploaded with:

- a `.sha256` checksum file,
- a release manifest JSON,
- and artifact guard validation before upload.

Signing and native installer packaging are staged but not yet treated as complete:

- macOS target: signed `.app`, notarized `.dmg`, with an explicit arm64/x86_64 or universal binary decision.
- Windows target: source/venv installation remains supported, but frozen
  Windows desktop archives are not published by the current release workflow.
- Linux target: AppImage first, then `.deb`/`.rpm` if demand justifies maintaining native packages.

Until signing identities and native packaging jobs are configured, release manifests mark CI-built desktop archives as `unsigned-ci-build`.

## Update and Uninstall

For pip/venv installs:

```bash
python -m pip install --upgrade "annolid[gui]"
python -m pip uninstall annolid
```

For one-line installer source checkouts, activate the printed environment and update from the install directory:

```bash
git pull --recurse-submodules
python -m pip install --upgrade -e ".[gui]"
```

To remove a one-line installer setup, delete the install directory after saving any user data stored inside it.

## Release Validation

Before publishing a release, run the smallest checks that cover the changed surface:

```bash
source .venv/bin/activate
pytest
python -m build
python -m twine check dist/*
python scripts/check_distribution_artifacts.py dist/*
```

For desktop archives, also run:

```bash
python scripts/check_distribution_artifacts.py --kind bundle dist
```

Release artifacts should include SHA256 checksums and a manifest JSON. If signing is not available for a given release, the release notes and manifest must say so explicitly.
