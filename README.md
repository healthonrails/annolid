# Annolid

[![Annolid Build](https://github.com/healthonrails/annolid/workflows/Annolid%20CI/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![Annolid Release](https://img.shields.io/github/v/release/healthonrails/annolid?display_name=tag)](https://github.com/healthonrails/annolid/releases/latest)
[![DOI](https://zenodo.org/badge/290017987.svg)](https://zenodo.org/badge/latestdoi/290017987)
[![Downloads](https://pepy.tech/badge/annolid)](https://pepy.tech/project/annolid)
[![Arxiv](https://img.shields.io/badge/cs.CV-2403.18690-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2403.18690)

> Annotate, segment, track, and analyze animals or other research targets in video with one reproducible toolchain.

## Table of Contents
- [Overview](#overview)
- [What Annolid Is For](#what-annolid-is-for)
- [Documentation & Support](#documentation--support)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Using Annolid](#using-annolid)
- [Core Workflows](#core-workflows)
- [Annotation Guide](#annotation-guide)
- [Labeling Best Practices](#labeling-best-practices)
- [Tutorials & Examples](#tutorials--examples)
- [Troubleshooting](#troubleshooting)
- [Docker](#docker)
- [Citing Annolid](#citing-annolid)
- [Publications](#publications)
- [Additional Resources](#additional-resources)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

## Overview
Annolid is a deep learning toolkit for behavior analysis and video annotation. It brings annotation, instance segmentation, tracking, keypoint workflows, behavior scoring, and downstream analysis into one GUI and CLI environment.

The common path is practical and iterative: label a representative frame, propagate or track instances, review difficult frames, repair identities, and export annotations or metrics for analysis. Annolid is designed for real lab data, including overlap, occlusion, variable lighting, long videos, and projects where saved annotations need to remain readable and reproducible.

> **Python support:** Annolid runs on Python 3.10–3.14 for the default GUI/core workflow. The optional remote network video path uses `ffpyplayer`; install `annolid[remote_video]` only when you need that feature, especially on Python 3.14 where native FFmpeg development libraries may be required.

## What Annolid Is For

- Markerless multi-animal tracking from a small number of labeled frames.
- LabelMe-compatible image and video annotation with polygons, keypoints, zones, and behavior events.
- Foundation-model assisted segmentation and tracking workflows, including Cutie, SAM-family workflows, Grounding DINO, CoTracker-style point tracking, TAPNext ONNX, EfficientTAM, and CowTracker where installed.
- Behavior scoring, timeline flags, zone analysis, time-budget summaries, and classifier workflows.
- GUI-first review and correction, plus `annolid-run` CLI commands for reproducible model training, prediction, evaluation, and automation.
- Optional Annolid Bot workflows for multimodal assistance, model/plugin execution, MCP tools, and lab automation integrations.
- Large TIFF and atlas-overlay work with optional tiled backends for OME-TIFF, BigTIFF, SVG, and Illustrator/PDF-compatible overlays.

Annolid keeps heavier runtime features behind extras so a standard GUI install stays usable on common lab machines. See [Installation](docs/installation.md) for the maintained extras and installer profiles.

## Documentation & Support
- Latest documentation and user guide: [https://annolid.com](https://annolid.com) (mirror: [https://cplab.science/annolid](https://cplab.science/annolid))
- Community updates and tutorials are shared on the [Annolid YouTube channel](https://www.youtube.com/@annolid).
- Sample datasets, posters, and publications are available in the `docs/` folder of this repository.
- Join the discussion on the [Annolid Google Group](https://groups.google.com/g/annolid).

## Featured Use Case
- **Tracking Four Interacting Mice with One Labeled Frame | 10-Minute Experiment**
  See how Annolid bootstraps multi-animal tracking from a single labeled frame in a fast end-to-end workflow:
  [https://youtu.be/PNbPA649r78](https://youtu.be/PNbPA649r78)
- For more practical examples and walkthroughs, visit the [Annolid YouTube channel](https://www.youtube.com/@annolid).

## Quick Start

The fastest maintained path is the one-line installer:

**macOS / Linux:**

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
```

**Windows PowerShell:**

```powershell
irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
```

After installation:

```bash
annolid --help
annolid-run --help
annolid
```

If you prefer [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
conda install git ffmpeg
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e ".[gui]"
annolid  # launches the GUI
```

For source development, use a repository-local `.venv`:

```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[gui]"
annolid
```

## Installation

### One-Line Installation (Recommended)

Get Annolid running in minutes with the automated installer. It clones the repository, creates an isolated environment, bootstraps `uv` when needed, installs GUI dependencies, and validates the ONNX Runtime provider setup.

**macOS / Linux:**

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash
```

**Windows PowerShell:**

```powershell
irm https://raw.githubusercontent.com/healthonrails/annolid/main/install.ps1 | iex
```

The script will:
- Clone the repository.
- Detect your OS and hardware.
- Create an isolated virtual environment.
- Install and validate ONNX Runtime CPU/GPU providers.
- Prompt for optional features such as SAM3 and text-to-speech when requested.
- Offer to launch Annolid immediately.

For a full breakdown of one-line installer choices, including GPU vs CPU, interactive vs non-interactive, custom paths, Conda, and extras, see [One-Line Installer Choices](docs/one_line_install_choices.md).

Common maintained workstation profile:

```bash
curl -sSL https://raw.githubusercontent.com/healthonrails/annolid/main/install.sh | bash -s -- --profile workstation
```

### Other Installation Methods

For advanced users, Docker, Conda, or manual Pip installation, please see the [Detailed Installation Guide](docs/installation.md).

## Using Annolid
- Launch the GUI:
  ```bash
  conda activate annolid-env
  annolid
  ```
- Provide custom labels:
  ```bash
  annolid --labels=/path/to/labels_custom.txt
  ```
- Draw shapes on a seed frame, often frame `0`, and use stable instance names when cross-frame identity matters.
- Mark zones directly in the label popup with **Zone type**, or use **Video Tools → Zones** for bulk zone management, presets, and zone JSON save/load.
- Use **View → Show Zones On All Frames** to control whether saved zone overlays are displayed across the full timeline.
- Open **Video Tools → Zone Analysis** to export legacy place-preference CSVs, generic zone metrics, or profile-aware assay summaries. See [Zone Analysis](docs/zone_analysis.md) and [Zone Analysis Workflow](docs/tutorials/zone_analysis_workflow.md).
- For behavior scoring with shared behavior names across Flags, Timeline, and Annolid Bot, see [Behavior labeling with Timeline, Flags, and Annolid Bot](docs/tutorials/behavior_timeline_flags_bot.md).
- Use `annolid-run list-models`, `annolid-run help train`, and `annolid-run help predict` for CLI model workflows.
- Open **AI & Models → Annolid Bot…** when you need multimodal chat, typed model/plugin execution, MCP integrations, or optional lab-automation channels. See [Agent and Automation](docs/agent_and_automation.md), [MCP](docs/mcp.md), and [Annolid Agent and annolid-run](docs/agent_annolid_run.md).
- Summarize annotated behavior events into a time-budget report (GUI: *File → Behavior Time Budget*; CLI example with 60 s bins and a project schema):
  ```bash
  python -m annolid.behavior.time_budget exported_events.csv \
      --schema project.annolid.json \
      --bin-size 60 \
      -o time_budget.csv
  ```
- Compute aggression-bout counts (for example `slap_in_face`, `run_away`, and `fight_initiation`) and export a `_bouts.csv` sidecar:
  ```bash
  python -m annolid.behavior.time_budget exported_events.csv \
      --bout-profile aggression \
      --bout-gap-seconds 2 \
      -o time_budget.csv
  ```
- Compress videos when storage is limited:
  ```bash
  ffmpeg -i input.mp4 -vcodec libx264 output_compressed.mp4
  ```

## Core Workflows

- [Getting Started](docs/getting_started.md): shortest path from install to a working GUI session.
- [Workflows](docs/workflows.md): supported GUI, Bot, CLI, behavior, depth, 3D, identity-repair, and simulation paths.
- [Tutorials](docs/tutorials.md): maintained walkthroughs and notebooks.
- [TAPNext ONNX point tracking](docs/tapnext.md): point-seeded tracking workflow and model-cache behavior.
- [SAM3 guide](docs/sam3.md): SAM3 tracking and agent-assisted long-video tracking.
- [Large TIFF and Atlas Overlay Workflow](docs/atlas_overlay_workflow.md): large image and vector overlay workflow.

## Video Depth Anything

- Run depth estimation on the currently loaded video via **View → Video Depth Anything…**. Use **View → Depth Settings…** to pick the encoder, resolution, FPS downsampling, and which outputs to save.
- Pretrained weights belong under `annolid/depth/checkpoints`. Download just what you need via the bundled Python helper (uses `huggingface-hub`, already listed in dependencies):
  ```bash
  cd annolid
  python -m annolid.depth.download_weights --model video_depth_anything_vitl
  ```
  Pass `--all` to fetch every checkpoint, or run `python -m annolid.depth.download_weights --list` for the full menu. Existing files are never re-downloaded.
- The GUI runner now auto-downloads whichever checkpoint you select in the dialog, so you only need to invoke the helper when you want to prefetch models ahead of time.
- Depth run now streams inference frame-by-frame, emits a single `depth.ndjson` record alongside the video (with per-frame base64 depth PNGs plus scale metadata) instead of writing separate JSON files per frame, and still shows a live blended overlay while processing. Enable `save_depth_video` or `save_depth_frames` only if you also need rendered outputs.
- Optional exports include `depth_frames/`, `<video_stem>_vis.mp4`, point cloud CSVs, `*.npz`, and `*_depths_exr/` (EXR requires `OpenEXR`/`Imath`).
- Full walkthrough and updates: <https://annolid.com/portal/workflows/>.

## CoWTracker Setup

If you want to run the CoWTracker backend, install Annolid with the CowTracker extra:

```bash
pip install "annolid[cowtracker]"
# or from source
pip install -e ".[cowtracker]"
```

If Annolid is already installed, add only the CowTracker dependency:

```bash
pip install "safetensors>=0.4.0"
```

Then select **CoWTracker** in the model dropdown. The model checkpoint is fetched
from Hugging Face on first use.

CowTracker uses a minimal vendored VGGT runtime subset under:

`annolid/tracker/cowtracker/cowtracker/thirdparty/vggt`

If that vendored subset is not present, CowTracker can also use an externally
installed `vggt` package. See `annolid/tracker/cowtracker/README.md` for the
required vendored file list and packaging notes.

## TAPNext ONNX Setup

Select **TAPNext (ONNX)** in the model dropdown. Annolid downloads the official
TAPNext ONNX model on first use, verifies its SHA256 checksum, and caches it at:

```bash
~/.annolid/workspace/downloads/tapnext.onnx
```

The published model URL is:

```bash
https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx
```

Expected checksum:

```bash
sha256:4fca0951802f0b745de254930c880938a74bf8b54b10786fc68d0ab4ba5c5300
```

Annolid runs the model in fixed-length clips, pads or batches query points to
match the exported `Q` capacity, and writes tracked points as LabelMe JSON files
like the CoTracker backend.

See the [TAPNext ONNX point tracking guide](docs/tapnext.md) for the full GUI
workflow, prompt tips, output format, and troubleshooting notes.

## Annotation Guide
![Annolid UI based on LabelMe](docs/imgs/annolid_ui.png)

## Atlas Overlay Workflow

Annolid now supports a practical atlas workflow for large TIFF-family images and Illustrator-exported overlays.

- Open `.tif`, `.tiff`, `.ome.tif`, or `.ome.tiff` images.
- Import Illustrator-exported `SVG` overlays or PDF-compatible `.ai` files.
- Use the `Vector Overlays` dock for visibility, opacity, transform, landmark pairing, and affine alignment.
- Edit polygons and other supported shapes directly inside the tiled large-image viewer, and create native point/line/linestrip/polygon/rectangle/circle annotations there.
- Keep immutable source provenance for imported overlays while editing a derived correction layer.
- Export corrected overlays as `SVG`, overlay `JSON`, or `*.labelme.json`.

Imported vector overlays now skip definition-only geometry such as clip paths, PDF-compatible Illustrator `.ai` files can be opened directly, small atlas drawings can auto-fit to the current image on import when their coordinates are clearly not already in image space, and PDF-compatible `.ai`/`.pdf` overlays preserve the source art box when available, then the page box and image origin during auto-fit so the imported geometry stays aligned to the underlying raster coordinate system. Explicit landmark pairs are stored with the overlay record instead of only existing as transient point metadata.

For step-by-step instructions, install notes, and landmark pairing details, see [Large TIFF and Atlas Overlay Workflow](docs/atlas_overlay_workflow.md).
For a broader overview of large TIFF support, viewing backends, caches, tile-native editing, and canvas fallback behavior, see [Large Image Guide](docs/large_image_guide.md).

Large-image support is optional by design. A standard `annolid[gui]` install still works normally for the usual GUI and annotation workflows. If you need large TIFF / OME-TIFF metadata, optimized cache generation, or faster whole-slide style navigation, install `annolid[large_image]`.

When Annolid opens a large TIFF, it reports which backend it used. If it says `tifffile` for a very large file, installing the native `libvips` or OpenSlide runtime can improve pan/zoom responsiveness.

For large flat TIFF files, `File -> Optimize Large TIFF for Fast Viewing...` builds a pyramidal cache that later opens can reuse. The same `File` menu also exposes cache info, cache-folder access, configurable cache limits, and safe cache cleanup actions, and Annolid prunes old optimized TIFF caches automatically so disk usage does not grow without bound.

During large-image work, the status bar is kept minimal so it mainly carries the page controls. Backend, page, zoom, tile, cache, and mode information are shown in the tiled viewer’s debug/status overlay instead.

Large TIFF sessions also expose a unified `Layers` dock built from the shared large-image layer model, so label-image overlays, vector overlays, and manual annotations can be inspected and toggled from one place instead of only through specialized overlay actions.

- **Label polygons and keypoints clearly.** Give each animal a unique instance name when tracking across frames (for example, `vole_1`, `mouse_2`). Use descriptive behavior names (`rearing`, `grooming`) for polygons dedicated to behavioral events, and name body-part keypoints (`nose`, `tail_base`) consistently.
- **Tune instance colors for review.** In the GUI, right-click a label in **Labels** or a shape row in **Label Instances**, then choose **Change color**. Annolid applies the color to every visible instance with that label and remembers the preference in app settings without changing LabelMe JSON files. Use **Reset color** to return to the automatic palette or project-schema color.
- **Accelerate timestamp annotation.** While scoring behaviors, press `s` to mark the start, `e` to mark the end, `f`/`b` to step ±10 frames, and `r` to remove events directly from the video slider.
- **Enable frame-level flags.** Launch Annolid with `--flags "digging,rearing,grooming"` to open a multi-select list of behaviors. Save selections with `Ctrl+S` or the **Save** button; remove events by pressing `R`.
- **Customize configuration.** The first run creates `~/.labelmerc` (or `C:\Users\<username>\.labelmerc` on Windows). Edit this file to change defaults such as `auto_save: true`, or supply an alternative path via `annolid --config /path/to/file`.
- **Learn more.** Additional annotation tips live in `annolid/annotation/labelme.md`.

## Labeling Best Practices
- Label 20–100 frames per video to reach strong performance; the curve in `docs/imgs/AP_across_labeled_frames.png` shows how accuracy scales with annotation volume.
- Close the loop with human-in-the-loop training (see `docs/imgs/human_in_the_loop.png`): train on initial annotations, auto-label, correct, and retrain until predictions align with human expectations.
- Draft labeling guidelines up front—start with [this template](https://docs.google.com/document/d/1fjgRSni7PNzMCSKw7NqVfGAp29phcf3NzrAojUhpVUY/edit#) and adapt it to your species and behaviors.
- Treat each animal instance as its own class when you need cross-frame identity. Use generic class names only when identity consistency is unnecessary, or when you are aggregating across many individuals.
- To generalize to new animals or videos, include diverse examples of each behavior and adjust the training set iteratively.

## Tutorials & Examples
- Featured demo: [Tracking Four Interacting Mice with One Labeled Frame | 10-Minute Experiment](https://youtu.be/PNbPA649r78)
- Behavior workflow tutorial: [Behavior labeling with Timeline, Flags, and Annolid Bot](docs/tutorials/behavior_timeline_flags_bot.md)
- DINOv3 Keypoint Tracking tutorial: book/tutorials/DINOv3_keypoint_tracking.md
- DINOv3 model selection/download helper: `annolid-run dinov3-models --list`
[![Effortless Multiple Instance Tracking using Annolid: Beginner's Tutorial](https://annolid.com/assets/images/annolid_gui.png)](https://www.youtube.com/embed/ry9bnaajKCs?si=o_rdLobKeKb4-LWX)

![Effortlessly Create Polygon Labels for Objects using Segment Anything Models](docs/imgs/annolid_with_segment_anything.gif)

[![Annolid Youtube playlist](docs/imgs/00002895_7.jpg)](https://www.youtube.com/embed/videoseries?list=PLYp4D9Y-8_dRXPOtfGu48W5ENtfKn-Owc "Annolid Youtube playlist")
|   YouTube Channel | Annolid documentations|
| :-------------------------------------: | :----------------------------: |
| ![](docs/imgs/annolid_youtube.png) | ![](docs/imgs/annolid_qr.png) |

[![Multiple Animal Tracking](docs/imgs/mutiple_animal_tracking.png)](https://youtu.be/lTlycRAzAnI)

| Instance segmentations | Behavior prediction |
| :--------------------: | :-----------------: |
| ![](docs/imgs/example_segmentation.png) | ![](docs/imgs/example_vis.png) |

[![Mouse behavior analysis with instance segmentation based deep learning networks](http://img.youtube.com/vi/op3A4_LuVj8/0.jpg)](http://www.youtube.com/watch?v=op3A4_LuVj8)

## Troubleshooting
- Video playback errors (`OpenCV: FFMPEG: tag ...` or missing codecs):
  Install FFmpeg via your package manager or `conda install -c conda-forge ffmpeg` to extend codec support.
- macOS Qt warning (`Class QCocoaPageLayoutDelegate is implemented in both ...`):
  `conda install qtpy` resolves the conflict between OpenCV and PyQt.
- If the GUI does not launch, confirm the correct environment is active and run `annolid --help` for CLI usage.
- If you see `qtpy.QtBindingsNotFoundError`, install GUI dependencies in the active environment: `pip install -e ".[gui]"` (source) or `pip install "annolid[gui]"` (PyPI).
- For model training/inference from the terminal, use `annolid-run list-models`, `annolid-run help train`, `annolid-run help predict`, `annolid-run help train <model>`, and `annolid-run help predict <model>`. Older `--help-model` forms still work.
- Built-in model plugins now show curated quick-reference groups such as `Required inputs`, `Model and runtime`, and `Training controls` before the full flag list.
- Shared YAML run-configs are supported for multiple training plugins (for example `dino_kpseg`, `maskrcnn_detectron2`, `yolo`, `behavior_classifier`): `annolid-run train <model> --run-config annolid/configs/runs/<template>.yaml` (CLI flags still override YAML fields).
- YOLOE-26 prompting (text, visual, prompt-free) is available via `annolid-run predict yolo_labelme` and in the GUI video inference workflow (see <https://annolid.com/portal/workflows/>).
- For an interactive TensorBoard embedding projector view of DinoKPSEG DINOv3 patch features, run `annolid-run dino-kpseg-embeddings --data /path/to/data.yaml [--weights /path/to/best.pt]` and then `tensorboard --logdir <run_dir>/tensorboard` (some DINOv3 checkpoints require a Hugging Face token).

## Docker
Ensure [Docker](https://www.docker.com/) is installed, then run:
```bash
cd annolid/docker
docker build .
xhost +local:docker  # Linux only; allows GUI forwarding
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY <IMAGE_ID>
```
Replace `<IMAGE_ID>` with the identifier printed by `docker build`.

## Citing Annolid
If you use Annolid in your research, please cite:
- **Preprint:** [Annolid: Annotation, Instance Segmentation, and Tracking Toolkit](https://arxiv.org/abs/2403.18690)
- **Zenodo:** Find the latest release DOI via the badge at the top of this README.
```bibtex
@misc{yang2024annolid,
      title={Annolid: Annotate, Segment, and Track Anything You Need},
      author={Chen Yang and Thomas A. Cleland},
      year={2024},
      eprint={2403.18690},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{yang2023automated,
  title={Automated Behavioral Analysis Using Instance Segmentation},
  author={Yang, Chen and Forest, Jeremy and Einhorn, Matthew and Cleland, Thomas A},
  journal={arXiv preprint arXiv:2312.07723},
  year={2023}
}

@misc{yang2020annolid,
  author = {Chen Yang and Jeremy Forest and Matthew Einhorn and Thomas Cleland},
  title = {Annolid: an instance segmentation-based multiple animal tracking and behavior analysis package},
  howpublished = {\url{https://github.com/healthonrails/annolid}},
  year = {2020}
}
```

## Publications
- **2022 – Ultrasonic vocalization study.** Pranic *et al.* relate mouse pup vocalizations to non-vocal behaviors ([bioRxiv](https://doi.org/10.1101/2022.10.14.512301)).
- **2022 – Digging and pain behavior.** Pattison *et al.* link digging behaviors to wellbeing in mice (*Pain*, 2022).
- **SfN Posters:**
  - [2021: Annolid — instance segmentation-based multiple-animal tracking](https://youtu.be/tVIE6vG9Gao)
  - 2023: PSTR512.01 *Scoring rodent digging behavior with Annolid*
  - 2023: PSTR512.02 *Annolid: Annotate, Segment, and Track Anything You Need*
- For more applications and datasets, visit [https://cplab.science/annolid](https://cplab.science/annolid).

## Additional Resources
- **Example dataset (COCO format):** [Download from Google Drive](https://drive.google.com/file/d/1fUXCLnoJ5SwXg54mj0NBKGzidsV8ALVR/view?usp=sharing).
- **Pretrained models:** Available in the [shared Google Drive folder](https://drive.google.com/drive/folders/1t1eXxoSN2irKRBJ8I7i3LHkjdGev7whF?usp=sharing).
- **Feature requests & bug reports:** Open an issue at [github.com/healthonrails/annolid/issues](https://github.com/healthonrails/annolid/issues).
- **Additional videos:** Visit the [Annolid YouTube channel](https://www.youtube.com/@annolid) for demonstrations and talks.

## Acknowledgements
Annolid's tracking module integrates **Cutie** for enhanced video object segmentation. If you use this feature, please cite *Putting the Object Back into Video Object Segmentation* (Cheng *et al.*, 2023) and the [Cutie repository](https://github.com/hkchengrex/Cutie).

The counting tool integrates **CountGD**; cite the original CountGD publication and repository when you rely on this module in your research.

## Contributing
Contributions are welcome! Review the guidelines in `CONTRIBUTING.md`, open an issue to discuss major changes, and run relevant tests before submitting a pull request.

## License
Annolid is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).
