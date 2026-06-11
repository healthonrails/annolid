# Packaging and Distribution

This document defines Annolid's install tiers, optional runtime contract, frozen app artifact policy, and release validation expectations.

## Install Tiers

Annolid keeps the default install useful without forcing every model runtime onto every workstation.

| Tier | Command | Contract |
|---|---|---|
| Core | `pip install annolid` | Import-safe core package for non-GUI workflows and library use. Heavy model runtimes are not base dependencies. |
| GUI | `pip install "annolid[gui]"` | Default desktop annotation and local video baseline. The one-line installers install this tier by default. |
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
| `gui` | `gui` only |
| `workstation` | `gui,tracking,sam3,training` |
| `full` | `gui,all` |

The default profile is `gui`. SAM-HQ is attempted only when a SAM-related extra such as `sam`, `sam3`, `segment_anything`, or `all` is selected. This keeps default installs independent of optional GitHub source installs.

Each successful installer run writes `annolid-install-report.json` in the install directory. The report includes the selected profile, resolved extras, Python version, OS, architecture, package manager, GPU decision, ONNX Runtime providers, SAM-HQ status, and failed optional steps.

## Optional Runtime Behavior

Frozen apps and lean installs intentionally exclude heavy optional runtimes. Code that opens optional workflows should use the central capability checker in `annolid.infrastructure.capabilities` to report:

- whether the runtime is available,
- which imports are missing,
- the exact install command,
- and the documentation link.

Cutie tracking also has first-use dependency repair. If Cutie packages are missing and `ANNOLID_AUTO_INSTALL_CUTIE_DEPS` is not disabled, Annolid attempts to install the required packages into the active environment before loading the tracker. Set `ANNOLID_AUTO_INSTALL_CUTIE_DEPS=0` to make the workflow fail fast with an install command instead.

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
- Windows: `Annolid-Windows.zip`
- Linux: `Annolid-Linux.tar.gz`

Each archive is uploaded with:

- a `.sha256` checksum file,
- a release manifest JSON,
- and artifact guard validation before upload.

Signing and native installer packaging are staged but not yet treated as complete:

- macOS target: signed `.app`, notarized `.dmg`, with an explicit arm64/x86_64 or universal binary decision.
- Windows target: signed `.msi` or MSIX, plus zip fallback.
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
