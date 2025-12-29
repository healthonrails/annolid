# Installation Options

To run Annolid, we suggest using Anaconda package/environment manager for Python. Download and install the [Anaconda](https://www.anaconda.com/products/individual) environment first. Then do the following, using the bash shell in Linux or the conda command line (= Anaconda Prompt) in Windows.

We also provide a PyPI version of Annolid that you can use, but it may not be as up-to-date as the codebase on GitHub.

## Requirements
- Ubuntu / macOS / Windows
- Python >= 3.10 (recommended: 3.11)
- Qt bindings (installed automatically via Annolid’s dependencies)
- Optional: CUDA/MPS GPU for faster inference/training

## Install Annolid locally

We create a virtual environment called _annolid-env_ into which we will install Annolid and all of its dependencies, along with whatever other Python tools we need. Python 3.11 is recommended, as it is the version being used for Annolid development.

### Clone Annolid repository and change into the directory
```
conda create -n annolid-env python=3.11
conda activate annolid-env
conda install git
conda install ffmpeg
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e .
```
```{note}
Install [ffmpeg](https://ffmpeg.org/) (for example with `conda install -c conda-forge ffmpeg`) to ensure OpenCV can decode a wide range of video formats during playback.
```

```{note}
On Windows, if you encounter errors related to pycocotools, please download and install [Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16). Then, run the following command in your terminal: `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
```

```{note}
To fix the error `“Failed to load platform plugin “xcb”`, while launching the Qt5 app on Linux, run `sudo apt install --reinstall libxcb-xinerama0`.
```

We then activate the virtual environment we just created.
```
conda activate annolid-env
```
```{note}
Be sure to activate the Annolid virtual environment every time you restart Anaconda or your computer; the shell prompt should read "(annolid-env)".
```
Finally, to open the Annolid GUI, just type the following:
```
annolid
```

For detailed installation instructions, please check [Annolid Installation and Quick Start (PDF)](https://annolid.com/assets/pdfs/install_annolid.pdf).

# The following sections are optional.
## Detectron2 (optional): train Mask R-CNN / batch inference

::::{important}
Detectron2 is **not required** for the core Annolid GUI workflow (AI polygons, Cutie/EfficientTAM tracking, YOLO-based inference, exports, and analyses).

Install Detectron2 only if you specifically need **Mask R-CNN training/inference** through Detectron2.
::::


### Installation guidance
Detectron2 wheels depend on **your exact** Python / PyTorch / CUDA combination, and the recommended installation method changes over time.

- Official instructions: https://detectron2.readthedocs.io/tutorials/install.html
- If you want the simplest path, use the Annolid Colab notebook (below), which comes with a compatible GPU runtime.


```{note}
If you encounter an error on windows with message says:
`in _run_ninja_build raise RuntimeError(message) RuntimeError: Error compiling objects for extension` , please go to the link https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0 and download x64: `vc_redist.x64.exe`. Please click and install it. After restart, you can cd to detectron2 folder and run the following command: `pip install -e .` .
```

# Using Detectron2 on Google Colab
```{note}
If you installed Detectron2 locally you can skip this section.
```

This step is only if you did not install Detectron2 locally and intend to train/run Detectron2 models on Google Colab.

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb)

## YOLO (recommended for many custom models)
Annolid includes Ultralytics YOLO support (segmentation and pose) and a GUI training workflow. A typical path is:
1. Label frames in Annolid (LabelMe JSONs).
2. Convert to YOLO dataset format (Annolid menu: *File → Convert Labelme to YOLO format*).
3. Train from Annolid (menu: *File → Train models* → select **YOLO**).


# Alternative installation
## Get stable release from PyPI
```
pip install annolid
```

# Docker

We provide a script to build a docker container for Annolid to make it easier to access the package without the need to install anything besides Docker.

You need to make sure that [Docker](https://docs.docker.com/engine/install/ubuntu/) is installed on your system (or a similar software capable of building containerized applications)


```{note}
Currently this has only been tested on Ubuntu 20.04 LTS.
```


```
cd annolid/docker
docker build .
```

Now if you want to access Annolid through the GUI, you will need to connect the image through your computer's display using

```
xhost +local:docker
```

Finally, we will need the Image ID of your Annolid image. To get that use: 

```
docker image ls
```

and write down the IMAGE ID associated with annolid repository. 

Finally to launch annolid, run the following:

```
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY <Image ID>
```
