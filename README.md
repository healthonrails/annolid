# annolid

[![Annolid Build](https://github.com/healthonrails/annolid/workflows/Annolid%20CI/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![Annolid Release](https://github.com/healthonrails/annolid/workflows/Upload%20Python%20Package/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![DOI](https://zenodo.org/badge/290017987.svg)](https://zenodo.org/badge/latestdoi/290017987)
[![Downloads](https://pepy.tech/badge/annolid)](https://pepy.tech/project/annolid)


An annotation and instance segmentation-based multiple animal tracking and behavior analysis package.

## Overview of Annolid workflow

![Overview of Annolid workflow](docs/imgs/annolid_workflow.png)

* Labeling of frames (annotation)
* COCO formatting
* Training and inference (local or Colab)
* Post-processing and analysis

## Annolid video tutorials

[![Annolid Youtube playlist](docs/imgs/00002895_7.jpg)](https://www.youtube.com/embed/videoseries?list=PLYp4D9Y-8_dRXPOtfGu48W5ENtfKn-Owc "Annolid Youtube playlist")

## User Guide

A basic user guide with installation instructions and recent documentation can be found at [https://cplab.science/annolid or https://annolid.com](https://annolid.com).

## Examples
[![Multiple Animal Tracking](docs/imgs/mutiple_animal_tracking.png)](https://youtu.be/lTlycRAzAnI)

|         Instance segmentations          |      Behavior prediction       |
| :-------------------------------------: | :----------------------------: |
| ![](docs/imgs/example_segmentation.png) | ![](docs/imgs/example_vis.png) |

[![Mouse behavior analysis with instance segmentation based deep learning networks](http://img.youtube.com/vi/op3A4_LuVj8/0.jpg)](http://www.youtube.com/watch?v=op3A4_LuVj8)

Mouse behavior analysis with instance segmentation based deep learning networks

## Local Installation
### First you need to install [anaconda](https://docs.anaconda.com/anaconda/install/index.html).
create a conda env
```
conda create -n annolid-env python=3.7
conda activate annolid-env 
```
* Clone the code repo and change into the directory
```bash
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
```

* Install the package with pip
```bash
pip install -e .
```
The pip install command will attempt to determine your computer's resources (like a GPU) automatically.  To control this directly, you alternatively can use the conda env command and the appropriate environment file (.yml).
For alternative installation methods, see the Annolid documentation ([https://cplab.science/annolid or https://annolid.com](https://cplab.science/annolid)).

### Recommended steps for Ubuntu 20.04 machine with GPUs
* Open a terminal window and navigate to the directory where the Annolid source code was downloaded.

* Create a Conda environment based on the specifications in the environment.yml file located in the Annolid source code directory using the following command:

```bash
conda env create -f environment.yml
```
This command will create a new Conda environment with the required packages and dependencies needed to run Annolid on an Ubuntu 20.04 machine with GPUs.

* Activate the new Conda environment using the following command:
```bash
conda activate annolid-env 
```

* Verify that the installation was successful by running the annolid
```bash
annolid
```
Note: For error NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation,
please try the following command.
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

```

That's it! You should now have Annolid installed on your Ubuntu 20.04 machine with GPUs and be able to use it for video analysis and annotation tasks.
### Note for Mac M1 Chip users
If you encounter the folloing errors,
```
Intel MKL FATAL ERROR: This system does not meet the minimum requirements for use of the Intel(R) Math Kernel Library.
The processor must support the Intel(R) Supplemental Streaming SIMD Extensions 3 (Intel(R) SSSE3) instructions.
The processor must support the Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) instructions.
The processor must support the Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
[end of output]
```
Please try the following commands.
```
conda create -n annolid-env python=3.7
conda activate annolid-env 
# Please skip this git clone step, if you have already done it in the previous step
git clone --recurse-submodules https://github.com/healthonrails/annolid.git
cd annolid
pip install -e .
annolid
```

Note: if you got this error:
`ERROR: Could not find a version that satisfies the requirement decord>=0.4.0`
try to install [ffmpeg](https://ffmpeg.org/) or you can install it in conda with `conda install -c conda-forge ffmpeg`.

FYI: it is helpful to compress videos to reduce diskspace using ffmpeg using `ffmpeg -i my_video.mp4 -vcodec libx264 my_video_compressed.mp4`

Tip: to fix the error like `objc[13977]: Class QCocoaPageLayoutDelegate is implemented in both /Users/xxx/anaconda3/envs/annolid-env/lib/python3.7/site-packages/cv2/.dylibs/QtGui (0x10ebd85c0) and /Users/xxx/anaconda3/envs/annolid-env/lib/python3.7/site-packages/PyQt5/Qt/lib/QtPrintSupport.framework/Versions/5/QtPrintSupport (0x10fc9d540`, please try the command `conda install qtpy`. 
## Launch annolid user interface based on labelme
```bash
source activate annolid-env
annolid
#or you can provide a label.txt file as follows.
annolid --labels=/path/to/labels_custom.txt

```
![Annolid UI based on labelme](docs/imgs/annolid_ui.png)

If you want to learn more about labelme, please check the following link.

[Read more about annotations](annolid/annotation/labelme.md)


## How to label animals and behaviors?
### Polygons & keypoints(e.g. Vole_1, nose, tail_base, rearing…...)

* To train models for tracking animals and assigning IDs, please
label each instance with a unique name or ID (e.g. vole_1, mouse_2, or frog_femal_01).

* For instances without needing to assign IDs across different frames or videos, please label instances with a generic name or ID (e.g vole, mouse, or frog).

* For encoding behaviors, please name the ploygon with the behavior name (e.g. rearing, object_investigation, or grooming)

* For body parts, please use keypoint with names like nose, tail_base, or left_ear.

## How many frames do you need to label?

* 20 to 100 frames per video
![Auto-labeling](docs/imgs/AP_across_labeled_frames.png)

* For autolabeling, you can label 20 frames and train a model. Then you can use the trained model to predict on the video and add the corrected predictions to the training set to train a better model. Repeat the process until the model is matching human performance.
  ![Auto-labeling](docs/imgs/human_in_the_loop.png)

* Write labeling guidelines by starting with this [template](https://docs.google.com/document/d/1fjgRSni7PNzMCSKw7NqVfGAp29phcf3NzrAojUhpVUY/edit#).
## Tracking and re-identification

* To link instances across frames, we treat each instance as its own class across frames.
* To track multiple animals, label each animal as a separate instance.
* To generalize across animals or videos, label multiple animals as examples of the same instance.  

# Docker

Please make sure that [Docker](https://www.docker.com/) is installed on your system.
```
# on Linux
cd annolid/docker
docker build .
xhost +local:docker
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix/ -e DISPLAY=$DISPLAY  <Image ID>

```

# Citing Annolid
If you use Annolid in your research, please use the following BibTeX entry.
```
@misc{yang2020Annolid,
  author =       {Chen Yang, Jeremy Forest, Matthew Einhorn, Thomas Cleland},
  title =        {Annolid:  an instance segmentation-based multiple animal tracking and behavior analysis package},
  howpublished = {\url{https://github.com/healthonrails/annolid}},
  year =         {2020}
}
```

# Other open-access pre-prints related to Annolid
```
@article{pranic2022rates,
  title={Rates but not acoustic features of ultrasonic vocalizations are related to non-vocal behaviors in mouse pups},
  author={Pranic, Nicole M and Kornbrek, Caroline and Yang, Chen and Cleland, Thomas A and Tschida, Katherine A},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

[For more information:](https://cplab.science/annolid) https://cplab.science/annolid
.

# SfN 2021 poster
[Annolid: an instance segmentation-based multiple-animal tracking
and behavior analysis package
](https://youtu.be/tVIE6vG9Gao)

# Datasets
**New** An example dataset annotated by Annolid and converted to COCO format dataset is now available from
this Google Drive link https://drive.google.com/file/d/1fUXCLnoJ5SwXg54mj0NBKGzidsV8ALVR/view?usp=sharing.

# Pretrained models
The pretrained models will be shared to the [Google Drive folder](https://drive.google.com/drive/folders/1t1eXxoSN2irKRBJ8I7i3LHkjdGev7whF?usp=sharing).

# Feature requests and bug reports

To request a new feature or report bugs, please use the link https://github.com/healthonrails/annolid/issues here.

# Annolid Google groups
[annolid@googlegroups.com](https://groups.google.com/g/annolid)

