# Installation Options

> Canonical installation docs now live in the Annolid Docs Portal:
> <https://annolid.com/installation/>

This page is a quick-start summary.
The canonical, detailed install instructions live in:

- [Detailed Installation Guide](../../docs/installation.md)
- [One-Line Installer Choices](../../docs/one_line_install_choices.md)
- [Install with uv](../../docs/install_with_uv.md)

## Recommended: One-Line Installer

- macOS / Linux:

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
```

- Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
```

## Conda Quick Start

```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
conda install -c conda-forge git ffmpeg
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e ".[gui]"
annolid
```

## Optional Components

- Detectron2 is optional and only needed for Detectron2-based training/inference workflows.
- For most modern workflows (AI polygons, Cutie/EfficientTAM tracking, YOLO-based inference, exports, analyses), Detectron2 is not required.

For Detectron2 guidance and compatibility notes, see:

- [Detailed Installation Guide](../../docs/installation.md)

## Docker

```bash
cd annolid/docker
docker build .
# Linux only; allows GUI forwarding
xhost +local:docker
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY <IMAGE_ID>
```

For troubleshooting and platform-specific notes, see:

- [Detailed Installation Guide](../../docs/installation.md)
- [Annolid Installation and Quick Start (PDF)](https://annolid.com/assets/pdfs/install_annolid.pdf)
