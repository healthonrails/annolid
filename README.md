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