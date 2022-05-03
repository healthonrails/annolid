#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Annolid on Detectron2 Tutorial
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


# In[10]:


if IN_COLAB:
    get_ipython().system('unzip $dataset -d /content/')
else:
    get_ipython().system('unzip -o $dataset -d .')


# If your dataset has the same name as the file you uploaded, you do not need to manually input the name (just run the next cells). **Otherwise, you need to replace DATASET_NAME and DATASET_DIR with your own strings like `DATASET_NAME = "NameOfMyDataset"` and `DATASETDIR="NameOfMyDatasetDirectory"`**. To do that, uncomment the commented out cell below and replace the strings with the appropriate names

# In[11]:


DATASET_NAME = DATASET_DIR = f"{dataset.replace('.zip','')}"


# In[12]:


# DATASET_NAME = 'NameOfMyDataset' 
# DATASET_DIR = 'NameOfMyDatasetDirectory'


# In[13]:


DATASET_NAME


# In[14]:


DATASET_DIR


# # Run a pre-trained detectron2 model

# First, we check a random selected image from our training dataset:

# In[15]:


# select and display one random image from the training set
img_file = random.choice(glob.glob(f"{DATASET_DIR}/train/JPEGImages/*.*"))
im = cv2.imread(img_file)
if IN_COLAB:
    cv2_imshow(im)
else:
    plt.imshow(im)


# Then, we create a Detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image.

# In[16]:


cfg = get_cfg()


# In[17]:


if GPU:
    pass
else:
    cfg.MODEL.DEVICE='cpu'


# In[18]:


# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
# Find a model from Detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


# In[19]:


# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)


# In[20]:


outputs['instances'].pred_masks


# In[21]:


MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


# In[22]:


# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
if IN_COLAB:
    cv2_imshow(out.get_image()[:, :, ::-1])
else:
    plt.imshow(out.get_image()[:, :, ::-1])


# As we can see, the network doesn't detect what we want. That is expected as we have not fine-tuned the network with our custom dataset. We are going to do that in the next steps.

# # Train on a custom dataset

# In this section, we show how to train an existing detectron2 model on a custom dataset in COCO format.
# 
# ## Prepare the dataset

# Register the custom dataset to Detectron2, following the [detectron2 custom dataset tutorial](https://detectron2.readthedocs.io/tutorials/datasets.html).
# Here, the dataset is in COCO format, therefore we register  into Detectron2's standard format. User should write such a function when using a dataset in custom format. See the tutorial for more details.
# 

# In[23]:


from detectron2.data.datasets import register_coco_instances
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import  builtin_meta


# In[24]:


register_coco_instances(f"{DATASET_NAME}_train", {}, f"{DATASET_DIR}/train/annotations.json", f"{DATASET_DIR}/train/")
register_coco_instances(f"{DATASET_NAME}_valid", {}, f"{DATASET_DIR}/valid/annotations.json", f"{DATASET_DIR}/valid/")


# In[25]:


dataset_dicts = get_detection_dataset_dicts([f"{DATASET_NAME}_train"])


# In[26]:


_dataset_metadata = MetadataCatalog.get(f"{DATASET_NAME}_train")
_dataset_metadata.thing_colors = [cc['color'] for cc in builtin_meta.COCO_CATEGORIES]


# In[27]:


_dataset_metadata


# In[28]:


NUM_CLASSES = len(_dataset_metadata.thing_classes)
print(f"{NUM_CLASSES} Number of classes in the dataset")


# To verify the data loading is correct, let's visualize the annotations of a randomly selected sample in the training set:
# 
# 

# In[29]:


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

# In[30]:


if GPU:
    get_ipython().system('nvidia-smi')


# In[31]:


from detectron2.engine import DefaultTrainer


# In[32]:


cfg = get_cfg()


# In[33]:


if GPU:
    pass
else:
    cfg.MODEL.DEVICE='cpu'


# In[34]:


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


# In[35]:


# Look at training curves in tensorboard:
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir output')


# In[36]:


trainer.train()


# ## Inference & evaluation using the trained model
# Now, let's run inference with the trained model on the validation dataset. First, let's create a predictor using the model we just trained:
# 
# 

# In[37]:


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. 
# We simply update the weights with the newly trained ones to perform inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# set a custom testing threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15   #@param {type: "slider", min:0.0, max:1.0, step: 0.01}
predictor = DefaultPredictor(cfg)


# Then, we randomly select several samples to visualize the prediction results.

# In[38]:


from detectron2.utils.visualizer import ColorMode


# In[39]:


dataset_dicts = get_detection_dataset_dicts([f"{DATASET_NAME}_valid"])
for d in random.sample(dataset_dicts, 4):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=_dataset_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    if IN_COLAB:
        cv2_imshow(out.get_image()[:, :, ::-1])
    else:
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
        


# A more robust way to evaluate the model is to use a metric called Average Precision (AP) already implemented in the detectron2 package. If you want more precision on what the AP is, you can take a look [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) and [here](https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision). 

# ### #TODO: expand on  how to interpret AP

# In[40]:


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# In[41]:


if IN_COLAB:
    evaluator = COCOEvaluator(f"{DATASET_NAME}_valid", cfg, False, output_dir="/content/eval_output/")
else:
    evaluator = COCOEvaluator(f"{DATASET_NAME}_valid", cfg, False, output_dir="eval_output/")

val_loader = build_detection_test_loader(cfg, f"{DATASET_NAME}_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`


# # Let's test our newly trained model on a new video

# In[42]:


get_ipython().system('gdown --id 1aMCeFWng0JkRbw9LOytXXrv4I2qCiH0h')


# ### You can also download your own videos

# In[43]:


#!wget https://hosting-website.com/your-video.mp4


# In[44]:


if IN_COLAB:
    VIDEO_INPUT="/content/novelctrl.mkv"
    OUTPUT_DIR = "/content/eval_output"
else:
    VIDEO_INPUT="novelctrl.mkv"
    OUTPUT_DIR = "eval_output"


# ### If you use your own video / dataset you need to update the VIDEO_INPUT name

# In[45]:


# VIDEO_INPUT="YOUR_VIDEO_NAME"


# In[46]:


import cv2


# In[47]:


video = cv2.VideoCapture(VIDEO_INPUT)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
basename = os.path.basename(VIDEO_INPUT)


# In[48]:


import os 
os.makedirs(OUTPUT_DIR,exist_ok=True)


# In[49]:


def _frame_from_video(video):
  attempt = 0
  for i in range(num_frames):
      success, frame = video.read()
      if success:
          yield frame
      else:
          attempt += 1
          if attempt >= 2000:
              break
          else:
              video.set(cv2.CAP_PROP_POS_FRAMES, i+1)
              print('Cannot read this frame:', i)
              continue


# In[50]:


import pandas as pd
import pycocotools.mask as mask_util


# In[51]:


class_names = _dataset_metadata.thing_classes
print(class_names)


# In[52]:


frame_number = 0
tracking_results = []
VIS = True
for frame in _frame_from_video(video): 
    im = frame
    outputs = predictor(im)
    out_dict = {}  
    instances = outputs["instances"].to("cpu")
    num_instance = len(instances)
    if num_instance == 0:
        out_dict['frame_number'] = frame_number
        out_dict['x1'] = None
        out_dict['y1'] = None
        out_dict['x2'] = None
        out_dict['y2'] = None
        out_dict['instance_name'] = None
        out_dict['class_score'] = None
        out_dict['segmentation'] = None
        tracking_results.append(out_dict)
        out_dict = {}
    else:
        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_mask = instances.has("pred_masks")

        if has_mask:
            rles =[
                   mask_util.encode(np.array(mask[:,:,None], order="F", dtype="uint8"))[0]
                   for mask in instances.pred_masks
            ]
            for rle in rles:
              rle["counts"] = rle["counts"].decode("utf-8")

        assert len(rles) == len(boxes)
        for k in range(num_instance):
            box = boxes[k]
            out_dict['frame_number'] = frame_number
            out_dict['x1'] = box[0]
            out_dict['y1'] = box[1]
            out_dict['x2'] = box[2]
            out_dict['y2'] = box[3]
            out_dict['instance_name'] = class_names[classes[k]]
            out_dict['class_score'] = scores[k]
            out_dict['segmentation'] = rles[k]
            if frame_number % 1000 == 0:
              print(f"Frame number {frame_number}: {out_dict}")
            tracking_results.append(out_dict)
            out_dict = {}
        
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    if VIS:
        v = Visualizer(im[:, :, ::-1],
                    metadata=_dataset_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
         )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_image = out.get_image()[:, :, ::-1]
        if frame_number % 1000 == 0:
            if IN_COLAB:
                cv2_imshow(out_image)
            else:
                plt.imshow(out_image)
                plt.show()
            #Trun off the visulization to save time after the first frame
            VIS = False
    frame_number += 1
    print(f"Processing frame number {frame_number}")

video.release()


# ## All the tracking results will be saved to this Pandas dataframe. 
# 
# 

# In[53]:


df = pd.DataFrame(tracking_results).dropna()


# In[54]:


df.head()


# ## Calculate the bbox center point x, y locations

# In[55]:


cx = (df.x1 + df.x2)/2
cy = (df.y1 + df.y2)/2
df['cx'] = cx
df['cy'] = cy


# In[56]:


df.head()


# ## Only save the top 1 prediction for each frame for each class
# Note: You can change the number to save top n predictions for each frame and an instance name. head(2), head(5), or head(n)
# To save all the predictions, please use `df.to_csv('my_tracking_results.csv')`.

# In[57]:


df_top = df.groupby(['frame_number','instance_name'],sort=False).head(1)


# In[58]:


df_top.head()


# ## Visualize the center points with plotly scatter plot

# In[59]:


df_vis = df_top[df_top.instance_name != 'Text'][['frame_number','cx','cy','instance_name']]


# In[60]:


import plotly.express as px
import plotly.graph_objects as go
import numpy as np

fig = px.scatter(df_vis, 
                 x="cx",
                 y="cy", 
                 color="instance_name",
                 hover_data=['frame_number','cx','cy'])
fig.show()


# In[61]:


from pathlib import  Path
tracking_results_csv = f"{Path(dataset).stem}_{Path(VIDEO_INPUT).stem}_{cfg.SOLVER.MAX_ITER}_iters_mask_rcnn_tracking_results_with_segmenation.csv"
df_top.to_csv(tracking_results_csv)


# ## Download the tracking result CSV file to your local device

# In[62]:


if IN_COLAB:
    from google.colab import files
    files.download(tracking_results_csv)


# ## Save and download the trained model weights

# In[ ]:


final_model_file = os.path.join(cfg.OUTPUT_DIR,'model_final.pth')
files.download(final_model_file)

