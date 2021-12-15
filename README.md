# annolid

[![Annolid Build](https://github.com/healthonrails/annolid/workflows/Annolid%20CI/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![Annolid Release](https://github.com/healthonrails/annolid/workflows/Upload%20Python%20Package/badge.svg)](https://github.com/healthonrails/annolid/actions)
[![DOI](https://zenodo.org/badge/290017987.svg)](https://zenodo.org/badge/latestdoi/290017987)


An annotation and instance segmentation-based multiple animal tracking and behavior analysis package.

## Overview of Annolid workflow

![Overview of Annolid workflow](docs/imgs/annolid_workflow.png)

* Labeling of frames (annotation)
* COCO formatting
* Training and inference (local or Colab)
* Post-processing and analysis

## Annolid video tutorials

[![Annolid Youtube playlist](https://i9.ytimg.com/vi/pb8X4bqLRZ0/mq2.jpg?sqp=CKzjp4sG&rs=AOn4CLDiKkwHnt7dBr1GBVrNZJQXd0wbJw)](https://www.youtube.com/embed/videoseries?list=PLYp4D9Y-8_dRXPOtfGu48W5ENtfKn-Owc "Annolid Youtube playlist")

## User Guide

A basic user guide with installation instructions and recent documentation can be found at [](https://healthonrails.github.io/annolid/) https://healthonrails.github.io/annolid/.

## Examples
![Multiple Animal Tracking](docs/imgs/mutiple_animal_tracking.png)

Instance segmentations             |  Behavior prediction
:-------------------------:|:-------------------------:
![](docs/imgs/example_segmentation.png) | ![](docs/imgs/example_vis.png)

[![Mouse behavior analysis with instance segmentation based deep learning networks](http://img.youtube.com/vi/op3A4_LuVj8/0.jpg)](http://www.youtube.com/watch?v=op3A4_LuVj8 "Mouse behavior analysis with instance segmentation based deep learning networks")

## Local Installation

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
, please try to install [ffmpeg](https://ffmpeg.org/) or you can install it in conda `conda install -c conda-forge ffmpeg`.

FYI: it is helpful to compress videos to reduce diskspace using ffmpeg. 
```bash
ffmpeg -i my_video.mp4 -vcodec libx264 my_video_compressed.mp4
```

## Launch annolid user interface based on labelme
```bash
source activate your_env_name
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
  author =       {Chen Yang and Thomas Cleland},
  title =        {Annolid:  an instance segmentation-based multiple animal tracking and behavior analysis package},
  howpublished = {\url{https://github.com/healthonrails/annolid}},
  year =         {2020}
}
```
[For more information:](https://healthonrails.github.io/annolid/) https://healthonrails.github.io/annolid/
.

# SfN 2021 poster
[Annolid: an instance segmentation-based multiple-animal tracking 
and behavior analysis package
](https://youtu.be/tVIE6vG9Gao)


# Pretrained models 
The pretrained models will be shared to the [Google Drive folder](https://drive.google.com/drive/u/1/folders/1t1eXxoSN2irKRBJ8I7i3LHkjdGev7whF). 

# Feature requests and bug reports

To request a new feature or report bugs, please use the link https://github.com/healthonrails/annolid/issues here. 
