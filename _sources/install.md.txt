## Install Annolid

### Requirements
- Ubuntu / macOS / Windows
- Python >= 3.6
- [PyQt4 / PyQt5]

### Set up Anaconda virtual environment
To run Annolid, we suggest using the Anaconda package/environment manager for Python (free for personal use).  Download and install the [Anaconda](https://www.anaconda.com/products/individual) environment first.  Then do the following, using the bash shell in Linux or the conda command line (= Anaconda Prompt) in Windows:
```
# For MacOS or Linux
conda create --name=annolid python=3.7
source activate annolid
# conda install -c conda-forge pyside2
# conda install pyqt
# pip install pyqt5  # pyqt5 can be installed via pip on python3
# conda install git
```
Commented-out lines are optional; use them if necessary.

```
# For Windows
conda create --name=annolid python=3.7
conda activate annolid
conda install -c conda-forge pyside2
conda install git
```
This creates a virtual environment called "annolid" into which you will install Annolid and all of its dependencies, along with whatever other Python tools you may need.  Python 3.7 is recommended, as it is the version being used for Annolid development.    

### Install ffmpeg

`conda install -c conda-forge ffmpeg`

### Install Annolid
Clone the code repo from Github, change into the new directory, and install
```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid 

# install the package (be sure to include the space and the period after the -e)
pip install -e .
```
Install [ffmpeg](https://ffmpeg.org/) (for example with `conda install -c conda-forge ffmpeg`) to ensure OpenCV can decode a wide range of video formats during playback.

### If you encounter errors On Windows for pycocotools, please download and install Visual studio 2019 from the link below. 
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16

Then please run the following command in your terminal. 
`pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`

### To fix error,  "Failed to load platform plugin "xcb" while launching qt5 app on Linux

`sudo apt install --reinstall libxcb-xinerama0`

### Launch Annolid 
Typing "annolid" will bring up the Annolid GUI, which is based on the excellent [LabelMe](https://github.com/wkentaro/labelme) package.  
```bash
# be sure to activate the annolid virtual environment every time you restart Anaconda; the conda shell prompt should read "(annolid)"
source activate annolid   # on Linux/MacOS
conda activate annolid    # on Windows
# then start Annolid (a GUI window will appear)
annolid

# Optionally, you can use a predefined label.txt file as follows.
annolid --labels=/path/to/labels_custom.txt 
```

## Install Detectron2 locally

**Note**: if you intend to process your tagged videos using Google Colab (which you should do unless you are using a workstation with a higher-end GPU), then **you do not need to install Detectron2 on your local machine**, and you can ignore the following steps.  

### Requirements
- Windows, Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this.  
  Presently, the combination of torch 1.8 and torchvision 0.9.1 works well, along with pyyaml 5.3, as shown below.  
  For purposes of using Annolid, it is OK to downgrade pyyaml from its current version to 5.3.  

### Install Detectron2 dependencies: 
```
pip install pyyaml==5.3 
pip install pycocotools>=2.0.1
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

```
### Install Detectron2: 
See [https://detectron2.readthedocs.io/tutorials/install.html](https://detectron2.readthedocs.io/tutorials/install.html) for further information.  
``` 
import torch
assert torch.__version__.startswith("1.9")    
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

### Install Detectron2 on Windows 10

`git clone https://github.com/facebookresearch/detectron2.git`
`cd detectron2`
`pip install -e .`

```
Note. If you encounter an error  on windows with message says: 

`
in _run_ninja_build
raise RuntimeError(message)
RuntimeError: Error compiling objects for extension
`
Please go to the link https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0
and  download `x64: vc_redist.x64.exe`. 
Please click and install it. 
After restart, you can cd to detectron2 folder and run the following command. 
`pip install -e .`. 

## Install Detectron2 on Google Colab
Instructions will be posted here presently.  (Colab uses CUDA 10.2 + torch 1.9.0).  
<a href="https://colab.research.google.com/github/healthonrails/annolid/blob/master/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Optional: Install older version of Pytorch for YOLACT
Note: YOLACT models are less accurate comparing to Mask-RCNN in Detectron2. However, it is faster in terms of inference. 
```
# DCNv2 will not work if Pytorch is greater than 1.4.0
!pip install torchvision==0.5.0
!pip install torch==1.4.0
```
For more information, please check https://github.com/healthonrails/annolid/blob/master/docs/tutorials/Train_networks_tutorial_v1.0.1.ipynb and https://github.com/healthonrails/yolac. 