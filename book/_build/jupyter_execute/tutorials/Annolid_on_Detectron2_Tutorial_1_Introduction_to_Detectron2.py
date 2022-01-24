#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Annolid on Detectron2 Tutorial 1 : Introduction to Detectron2
# 
# <img src="https://dl.fbaipublicfiles.com/detectron2/Detectron2-Logo-Horz.png" width="500">
# 
# Welcome to Annolid on detectron2! This is modified from the official colab tutorial of detectron2. Here, we will go through some basics usage of detectron2, including the following:
# * Run inference on images or videos, with an existing detectron2 model
# * Train a detectron2 model on a new dataset
# 
# You can make a copy of this tutorial by "File -> Open in playground mode" and play with it yourself. __DO NOT__ request access to this tutorial.
# 

# # Install detectron2

# In[1]:


# Is running in colab or in jupyter-notebook
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


# In[2]:


# install dependencies: 
get_ipython().system('pip install pyyaml==5.3')
import torch, torchvision
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html')
# If there is not yet a detectron2 release that matches the given torch + CUDA version, you need to install a different pytorch.

# exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime


# In[3]:


# import some common libraries
import json
import os
import cv2
import random
import glob
import numpy as np
if IN_COLAB:
  from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[5]:


# is there a gpu
if torch.cuda.is_available():
    GPU = True
    print('gpu available')
else:
    GPU = False
    print('no gpu')


# ## Upload a labeled dataset.
# The following code is expecting the dataset in the COCO format to be in a ***.zip*** file. For example: ```sample_dataset.zip``` \
# 

# In[6]:


get_ipython().system('pip install gdown ')
get_ipython().system('gdown --id 1fUXCLnoJ5SwXg54mj0NBKGzidsV8ALVR')


# In[7]:


if IN_COLAB:
    dataset = '/content/novelctrlk6_8_coco_dataset.zip'
else:
    dataset = 'novelctrlk6_8_coco_dataset.zip'


# ### Note1: If you want to use your own dataset instead of the demo one, please uncomment and edit the following code.
# 
# ### Note2: please make sure there is no white space in your file path if you encounter file not found issues.

# In[8]:


# if IN_COLAB:
#     from google.colab import files
#     uploaded = files.upload()
# else:
#     from ipywidgets import FileUpload
#     from IPython.display import display
#     !jupyter nbextension enable --py widgetsnbextension
#     uploaded = FileUpload()

# display(uploaded)


# In[9]:


# if IN_COLAB:
#     dataset =  list(uploaded.keys())[0]
# else:
#     dataset = list(uploaded.value.keys())[0]


# If your dataset has the same name as the file you uploaded, you do not need to manually input the name (just run the next cells). **Otherwise, you need to replace DATASET_NAME and DATASET_DIR with your own strings like `DATASET_NAME = "NameOfMyDataset"` and `DATASETDIR="NameOfMyDatasetDirectory"`**. To do that, uncomment the commented out cell below and replace the strings with the appropriate names

# In[10]:


# DATASET_NAME = 'NameOfMyDataset' 
# DATASET_DIR = 'NameOfMyDatasetDirectory'


# In[11]:


if IN_COLAB:
    get_ipython().system('unzip $dataset -d /content/')
else:
    get_ipython().system('unzip -o $dataset -d .')


# In[12]:


DATASET_NAME = DATASET_DIR = f"{dataset.replace('.zip','')}"


# # Run a pre-trained detectron2 model

# First, we check a random selected image from our training dataset:

# In[13]:


# select and display one random image from the training set
img_file = random.choice(glob.glob(f"{DATASET_DIR}/train/JPEGImages/*.*"))
im = cv2.imread(img_file)
if IN_COLAB:
    cv2_imshow(im)
else:
    plt.imshow(im)


# Then, we create a Detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image.

# In[14]:


cfg = get_cfg()


# In[15]:


if GPU:
    pass
else:
    cfg.MODEL.DEVICE='cpu'


# In[16]:


# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
# Find a model from Detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


# In[17]:


# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)


# In[18]:


outputs['instances'].pred_masks


# In[19]:


MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


# In[20]:


# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
if IN_COLAB:
    cv2_imshow(out.get_image()[:, :, ::-1])
else:
    plt.imshow(out.get_image()[:, :, ::-1])


# As we can see, the network doesn't detect what we want. That is expected as we have not fine-tuned the network with our custom dataset. We are going to do that in the next steps.
