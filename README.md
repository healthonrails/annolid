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


## Examples
![Multiple Animal Tracking](docs/imgs/mutiple_animal_tracking.png)

Instance segmentations             |  Behavior prediction
:-------------------------:|:-------------------------:
![](docs/imgs/example_segmentation.png) | ![](docs/imgs/example_vis.png)

[![Mouse behavior analysis with instance segmentation based deep learning networks](http://img.youtube.com/vi/op3A4_LuVj8/0.jpg)](http://www.youtube.com/watch?v=op3A4_LuVj8 "Mouse behavior analysis with instance segmentation based deep learning networks")

## Installation

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

* For instances without needing to assign IDs across different frames, please 
label instances with a generic name or ID (e.g vole, mouse, or frog). 

* For encoding behaviors, please name the ploygon with the behavior name (e.g. rearing, object_investigation, or grooming)

* For body parts, please use keypoint with names like nose, tail_base, or left_ear.


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