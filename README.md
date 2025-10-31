# annolid

[![Annolid Build](https://github.com/healthonrails/annolid/workflows/Annolid%20CI/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![Annolid Release](https://github.com/healthonrails/annolid/workflows/Upload%20Python%20Package/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![DOI](https://zenodo.org/badge/290017987.svg)](https://zenodo.org/badge/latestdoi/290017987)
[![Downloads](https://pepy.tech/badge/annolid)](https://pepy.tech/project/annolid)
[![Arxiv](https://img.shields.io/badge/cs.CV-2403.18690-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2403.18690)

> Annotate, segment, and track multiple animals (or any research target) in video with a single toolchain.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Documentation & Support](#documentation--support)
- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [Using Annolid](#using-annolid)
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
Annolid is a deep learning toolkit for animal behavior analysis that brings annotation, instance segmentation, tracking, and behavior classification into a single workflow. It combines state-of-the-art models—including Cutie for video object segmentation, Segment Anything, and Grounding DINO—to deliver resilient, markerless tracking even when animals overlap, occlude each other, or are partially hidden by the environment.

Use Annolid to classify behavioral states such as freezing, digging, pup huddling, or social interaction while maintaining fine-grained tracking of individuals and body parts across long video sessions.

> **Python support:** Annolid runs on Python 3.8–3.13. The toolkit is not yet validated on Python 3.14, where several binary wheels (PyQt, Pillow) are still pending upstream releases.

## Key Features
- Markerless multiple-animal tracking from a single annotated frame.
- Instance segmentation powered by modern foundation models and transfer learning.
- Interactive GUI for rapid annotation (LabelMe-based) plus automation with text prompts.
- Behavioral state classification, keypoint tracking, and downstream analytics.
- Works with pre-recorded video or real-time streams; supports GPU acceleration.

## Documentation & Support
- Latest documentation and user guide: [https://annolid.com](https://annolid.com) (mirror: [https://cplab.science/annolid](https://cplab.science/annolid))
- Community updates and tutorials are shared on the [Annolid YouTube channel](https://www.youtube.com/@annolid/videos).
- Sample datasets, posters, and publications are available in the `docs/` folder of this repository.
- Join the discussion on the [Annolid Google Group](https://groups.google.com/g/annolid).

## Quick Start
Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), then set up Annolid in a new environment:

```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
conda install git ffmpeg
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e .
annolid  # launches the GUI
```

The `annolid` command detects your hardware automatically. If you need tighter control (for example, to target a specific CUDA toolkit), use the environment files described below.

## Installation Options

### Conda environment (GPU-ready, Ubuntu 20.04 tested)
```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
conda env create -f environment.yml
conda activate annolid-env
annolid
```

If you see `CUDA capability sm_86 is not compatible with the current PyTorch installation`, install a matching build:
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cudatoolkit=12.1 -c pytorch -c nvidia
```

### Pip-only installation
```bash
python -m venv annolid-env
source annolid-env/bin/activate
pip install --upgrade pip
pip install annolid
pip install "segment-anything @ git+https://github.com/SysCV/sam-hq.git"
annolid
```
This route works well on machines without Conda, but you remain responsible for installing system dependencies such as `ffmpeg`.

### uv (lightweight venv + installer)
Use [uv](https://docs.astral.sh/uv/) if you prefer fast virtualenv creation and dependency resolution:
```bash
pip install --user uv  # or grab the standalone binary
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
annolid
```
Generate a lock file for reproducible installs with `uv pip compile pyproject.toml -o uv.lock`, then reproduce the environment elsewhere via `uv pip sync uv.lock`. Ensure `ffmpeg`/`ffprobe` is available on your PATH (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu) so timestamp tools work correctly.

### Apple Silicon (macOS M1/M2)
Some Intel-specific libraries can trigger MKL errors on Apple Silicon. If you observe messages such as:
```
Intel MKL FATAL ERROR: This system does not meet the minimum requirements...
```
recreate the environment with native wheels:
```bash
conda create -n annolid-env python=3.11
conda activate annolid-env
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e .
annolid
```

### Optional dependencies
- **Detectron2** (required for training new instance segmentation models):
  ```bash
  conda activate annolid-env
  python -m pip install --user "git+https://github.com/facebookresearch/detectron2.git"
  ```
- **Segment Anything 2 (SAM2)** for object tracking:
  ```bash
  cd segmentation/SAM/segment-anything-2
  pip install -e .
  ```
- **FFmpeg** is recommended for format conversion and improved compatibility with OpenCV-based video I/O:
  ```bash
  conda install -c conda-forge ffmpeg
  ```

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
- Summarise annotated behavior events into a time-budget report (GUI: *File → Behavior Time Budget*; CLI example with 60 s bins and a project schema):
  ```bash
  python -m annolid.behavior.time_budget exported_events.csv \
      --schema project.annolid.json \
      --bin-size 60 \
      -o time_budget.csv
  ```
- Compress videos when storage is limited:
  ```bash
  ffmpeg -i input.mp4 -vcodec libx264 output_compressed.mp4
  ```

## Annotation Guide
![Annolid UI based on LabelMe](docs/imgs/annolid_ui.png)

- **Label polygons and keypoints clearly.** Give each animal a unique instance name when tracking across frames (for example, `vole_1`, `mouse_2`). Use descriptive behavior names (`rearing`, `grooming`) for polygons dedicated to behavioral events, and name body-part keypoints (`nose`, `tail_base`) consistently.
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
- DINOv3 Keypoint Tracking tutorial: book/tutorials/DINOv3_keypoint_tracking.md
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
- **Additional videos:** Visit the [Annolid YouTube channel](https://www.youtube.com/@annolid/videos) for demonstrations and talks.

## Acknowledgements
Annolid's tracking module integrates **Cutie** for enhanced video object segmentation. If you use this feature, please cite *Putting the Object Back into Video Object Segmentation* (Cheng *et al.*, 2023) and the [Cutie repository](https://github.com/hkchengrex/Cutie).

The counting tool integrates **CountGD**; cite the original CountGD publication and repository when you rely on this module in your research.

## Contributing
Contributions are welcome! Review the guidelines in `CONTRIBUTING.md`, open an issue to discuss major changes, and run relevant tests before submitting a pull request.

## License
Annolid is distributed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).
