# Installation Options

To run Annolid, we suggest using Anaconda package/environment manager for Python. Download and install the [Anaconda](https://www.anaconda.com/products/individual) environment first. Then do the following, using the bash shell in Linux or the conda command line (= Anaconda Prompt) in Windows.

We also provide a PyPI version of Annolid that you can use, but it may not be as up-to-date as the codebase on GitHub.

## Requirements
- Ubuntu / macOS / Windows
- Python >= 3.10
- [PyQt4 / PyQt5]

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
Note: If you encounter this error: `ERROR: Could not find a version that satisfies the requirement decord>=0.4.0`, try to install [ffmpeg](https://ffmpeg.org/) or you can install it in conda with `conda install -c conda-forge ffmpeg`, and then install [decord](https://github.com/dmlc/decord) from source.
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

# The following section is optional.
## Install Detectron2 locally only for training custom Mask R-CNN models and video inferences.

::::{Important}
If you intend to process your tagged videos using Google Colab (which you should do unless you are using a workstation with a higher-end GPU), then you do not need to install Detectron2 on your local machine, and you can ignore this section.
::::


### Requirements:

Windows, Linux or MacOS with Python ≥ 3.7, PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation. Install them together at [pytorch.org](http://pytorch.org) to make sure of this. Presently, the combination of torch 1.8 and torchvision 0.9.1 works well, along with pyyaml 5.3, as shown below.
For purposes of using Annolid, it is OK to downgrade pyyaml from its current version to 5.3.

### Install Detectron2 dependencies:
```
pip install pyyaml==5.3
pip install pycocotools>=2.0.1
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
### Install Detectron2
```
import torch
assert torch.__version__.startswith("1.9")    
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
```
See https://detectron2.readthedocs.io/tutorials/install.html for further information.


### Install Detectron2 on Windows 10

```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```


```{note}
If you encounter an error on windows with message says:
`in _run_ninja_build raise RuntimeError(message) RuntimeError: Error compiling objects for extension` , please go to the link https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0 and download x64: `vc_redist.x64.exe`. Please click and install it. After restart, you can cd to detectron2 folder and run the following command: `pip install -e .` .
```

# Using Detectron2 on Google Colab
```{note}
If you installed Detectron2 locally you can skip this section.
```

This step is only if you did not install Detectron2 locally and intend to process your tagged videos using Google Colab.
Google Colab uses CUDA 10.2 + torch 1.9.0.

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/master/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb)

# Using YOLACT instead of Detectron2:
```{note}
YOLACT models are less accurate comparing to Mask-RCNN in Detectron2. However, it is faster in terms of inference.
```
DCNv2 will not work if Pytorch is greater than 1.4.0

```
!pip install torchvision==0.5.0
!pip install torch==1.4.0
```

For more information, please check https://github.com/healthonrails/annolid/blob/master/docs/tutorials/Train_networks_tutorial_v1.0.1.ipynb and https://github.com/healthonrails/yolac


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
