#!/usr/bin/env python
# coding: utf-8

# # Annolid on Detectron2 Tutorial 3 : Evaluating the model
#
# This is modified from the official colab tutorial of detectron2. Here, we will
#
# * Evaluate our previously trained model.
#
# You can make a copy of this tutorial by "File -> Open in playground mode" and play with it yourself. __DO NOT__ request access to this tutorial.
#

# In[1]:


# Is running in colab or in jupyter-notebook
import torchvision
import torch
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import builtin_meta
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
import detectron2
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import cv2
import os
import json
from IPython import get_ipython
from IPython import display
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False


# In[2]:


# install dependencies:
get_ipython().system('pip install pyyaml==5.3')
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
if IN_COLAB:
    from google.colab.patches import cv2_imshow
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Setup detectron2 logger
setup_logger()

# import some common detectron2 utilities


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
    # TODO generalize this
    get_ipython().system('unzip -o $dataset -d .')


# In[9]:


DATASET_NAME = DATASET_DIR = f"{dataset.replace('.zip','')}"


# In[10]:


# In[11]:


register_coco_instances(f"{DATASET_NAME}_train", {
}, f"{DATASET_DIR}/train/annotations.json", f"{DATASET_DIR}/train/")
register_coco_instances(f"{DATASET_NAME}_valid", {
}, f"{DATASET_DIR}/valid/annotations.json", f"{DATASET_DIR}/valid/")


# In[12]:


_dataset_metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")
_dataset_metadata.thing_colors = [cc['color']
                                  for cc in builtin_meta.COCO_CATEGORIES]


# In[13]:


dataset_dicts = get_detection_dataset_dicts([f"{DATASET_NAME}_train"])


# In[14]:


NUM_CLASSES = len(_dataset_metadata.thing_classes)
print(f"{NUM_CLASSES} Number of classes in the dataset")


# ## Inference & evaluation using the trained model
# Now, let's run inference with the trained model on the validation dataset. First, let's create a predictor using the model we just trained:

# In[15]:


if GPU:
    get_ipython().system('nvidia-smi')


# In[16]:


# In[17]:


cfg = get_cfg()


# In[18]:


if GPU:
    pass
else:
    cfg.MODEL.DEVICE = 'cpu'


# In[19]:


cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (f"{DATASET_NAME}_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2  # @param
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
cfg.DATALOADER.REPEAT_THRESHOLD = 0.3
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4  # @param
cfg.SOLVER.BASE_LR = 0.0025  # @param # pick a good LR
# @param    # 300 iterations seems good enough for 100 frames dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # @param
# @param   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
# (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)


# In[20]:


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously.
# We simply update the weights with the newly trained ones to perform inference:
# path to the model we just trained
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# set a custom testing threshold
# @param {type: "slider", min:0.0, max:1.0, step: 0.01}
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15
predictor = DefaultPredictor(cfg)


# Then, we randomly select several samples to visualize the prediction results.

# In[21]:


# In[22]:


dataset_dicts = get_detection_dataset_dicts([f"{DATASET_NAME}_valid"])
for d in random.sample(dataset_dicts, 4):
    im = cv2.imread(d["file_name"])
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=_dataset_metadata,
                   scale=0.5,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.SEGMENTATION
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    if IN_COLAB:
        cv2_imshow(out.get_image()[:, :, ::-1])
    else:
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()


# A more robust way to evaluate the model is to use a metric called Average Precision (AP) already implemented in the detectron2 package. If you want more precision on what the AP is, you can take a look [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) and [here](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision).

# ### #TODO: expand on  how to interpret AP

# In[23]:


# In[24]:


if IN_COLAB:
    evaluator = COCOEvaluator(
        f"{DATASET_NAME}_valid", cfg, False, output_dir="/content/eval_output/")
else:
    evaluator = COCOEvaluator(
        f"{DATASET_NAME}_valid", cfg, False, output_dir="eval_output/")

val_loader = build_detection_test_loader(cfg, f"{DATASET_NAME}_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`


# In[ ]:
