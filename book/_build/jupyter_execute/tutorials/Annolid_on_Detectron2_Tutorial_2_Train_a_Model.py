#!/usr/bin/env python
# coding: utf-8

# # Annolid on Detectron2 Tutorial 2 : Train a Model
# 
# This is modified from the official colab tutorial of detectron2. Here, we will 
# 
# * Train (fine-tune) a detectron2 model on our dataset.
# 
# You can make a copy of this tutorial by "File -> Open in playground mode" and play with it yourself. __DO NOT__ request access to this tutorial.
# 

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


# # Train on a custom dataset

# In this section, we show how to train an existing detectron2 model on a custom dataset in COCO format.
# 
# ## Prepare the dataset

# ## Upload a labeled dataset.
# The following code is expecting the dataset in the COCO format to be in a ***.zip*** file. For example: ```sample_dataset.zip``` \
# Note: please make sure there is no white space in your file path if you encounter file not found issues.

# In[6]:


get_ipython().system('pip install gdown ')
get_ipython().system('gdown --id 1fUXCLnoJ5SwXg54mj0NBKGzidsV8ALVR')


# In[7]:


if IN_COLAB:
    dataset = '/content/novelctrlk6_8_coco_dataset.zip'
else:
    dataset = 'novelctrlk6_8_coco_dataset.zip'


# In[8]:


if IN_COLAB:
    get_ipython().system('unzip $dataset -d /content/')
else:
    #TODO generalize this
    get_ipython().system('unzip -o $dataset -d .')


# In[9]:


DATASET_NAME = DATASET_DIR = f"{dataset.replace('.zip','')}"


# Register the custom dataset to Detectron2, following the [detectron2 custom dataset tutorial](https://detectron2.readthedocs.io/tutorials/datasets.html).
# Here, the dataset is in COCO format, therefore we register  into Detectron2's standard format. User should write such a function when using a dataset in custom format. See the tutorial for more details.
# 

# In[10]:


from detectron2.data.datasets import register_coco_instances
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import  builtin_meta


# In[11]:


register_coco_instances(f"{DATASET_NAME}_train", {}, f"{DATASET_DIR}/train/annotations.json", f"{DATASET_DIR}/train/")
register_coco_instances(f"{DATASET_NAME}_valid", {}, f"{DATASET_DIR}/valid/annotations.json", f"{DATASET_DIR}/valid/")


# In[12]:


dataset_dicts = get_detection_dataset_dicts([f"{DATASET_NAME}_train"])


# In[13]:


_dataset_metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")
_dataset_metadata.thing_colors = [cc['color'] for cc in builtin_meta.COCO_CATEGORIES]


# In[14]:


_dataset_metadata


# In[15]:


NUM_CLASSES = len(_dataset_metadata.thing_classes)
print(f"{NUM_CLASSES} Number of classes in the dataset")


# To verify the data loading is correct, let's visualize the annotations of a randomly selected sample in the training set:
# 
# 

# In[16]:


for d in random.sample(dataset_dicts, 2):
    if '\\' in d['file_name']:
        d['file_name'] = d['file_name'].replace('\\','/')
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=_dataset_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    if IN_COLAB:
        cv2_imshow(out.get_image()[:, :, ::-1])
    else:
        plt.imshow(out.get_image()[:, :, ::-1])
        


# ## Train!
# 
# Now, let's fine-tune the COCO-pretrained R50-FPN Mask R-CNN model with our custom dataset. It takes ~2 hours to train 3000 iterations on Colab's K80 GPU, or ~1.5 hours on a P100 GPU.
# 

# In[17]:


if GPU:
    get_ipython().system('nvidia-smi')


# In[18]:


from detectron2.engine import DefaultTrainer


# In[19]:


cfg = get_cfg()


# In[20]:


if GPU:
    pass
else:
    cfg.MODEL.DEVICE='cpu'


# In[21]:


cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (f"{DATASET_NAME}_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2 #@param
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
cfg.DATALOADER.REPEAT_THRESHOLD = 0.3
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH =  4 #@param
cfg.SOLVER.BASE_LR = 0.0025 #@param # pick a good LR
cfg.SOLVER.MAX_ITER = 3000 #@param    # 300 iterations seems good enough for 100 frames dataset; you will need to train longer for a practical dataset
cfg.SOLVER.CHECKPOINT_PERIOD = 1000 #@param 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32 #@param   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES  #  (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)


# In[22]:


# Look at training curves in tensorboard:
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir output')


# In[23]:


trainer.train()


# ## Save and download the trained model weights

# In[24]:


final_model_file = os.path.join(cfg.OUTPUT_DIR,'model_final.pth')


# In[25]:


if IN_COLAB:
    from google.colab import files
    files.download(final_model_file)
else:
    from IPython.display import FileLink
    FileLink(final_model_file)

