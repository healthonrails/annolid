## Install Annolid

### Requirements
- Ubuntu / macOS / Windows
- Python >= 3.6
- [PyQt4 / PyQt5]

### Anaconda
You need install [Anaconda](https://www.continuum.io/downloads), then run below:
```
# python3
conda create --name=annolid python=3.7
source activate annolid
# conda install -c conda-forge pyside2
# conda install pyqt
# pip install pyqt5  # pyqt5 can be installed via pip on python3
```
### Annolid
* Clone the code repo and change into the directory
```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid 

# install the package
pip install -e .
```
Note: if you got this error: 
```
ERROR: Could not find a version that satisfies the requirement decord>=0.4.0
```
, please try to install [ffmpeg](https://ffmpeg.org/) and then install decord from source as described [here](https://github.com/dmlc/decord).

## Launch annolid user interface based on labelme
```bash
source activate your_env_name
annolid
#or you can provide a label.txt file as follows.
annolid --labels=/path/to/labels_custom.txt 

```

## Install Detectron2

### Requirements
- Windows, Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this

### install dependencies: 
```
pip install pyyaml==5.3 
pip install pycocotools>=2.0.1
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

```
### install detectron2: (Colab has CUDA 10.1 + torch 1.8)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
``` 
import torch
assert torch.__version__.startswith("1.8")
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
```