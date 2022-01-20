#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from IPython import get_ipython
from ray import serve
import numpy as np
import cv2
import glob
import requests
import pandas as pd
from pathlib import Path
from annolid.annotation.keypoints import save_labels
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.data.videos import frame_from_video
from annolid.inference.predict import Segmentor


# In[ ]:


import ray
ray.init()


# In[ ]:


DATASET_DIR = "labeled_frames_coco_dataset"
MODEL_PATH = "model_final.pth"
MODEL_SERVER_URL = "http://localhost:8000/image_predict"
# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


# In[ ]:


@serve.deployment(route_prefix="/image_predict", num_replicas=2)
class AnnolidModel:
    def __init__(self):
        self.model = Segmentor(DATASET_DIR,
                               MODEL_PATH)

    async def __call__(self, starlette_request):
        image_payload_bytes = await starlette_request.body()
        # convert string of image data to uint8
        nparr = np.fromstring(image_payload_bytes, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        preds = self.model.predictor(img)
        instances = preds["instances"].to('cpu')
        results = self.model._process_instances(instances)
        return results


# In[ ]:

serve.start()
AnnolidModel.deploy()


# In[ ]:


video_file = "myvideo.mp4"

cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for frame in frame_from_video(cap, num_frames):
    img = frame
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(
        MODEL_SERVER_URL, data=img_encoded.tostring(), headers=headers)
    # decode response

    print(response.json())
    break


# In[ ]:


def instances_to_labelme(results,
                         image_path,
                         height,
                         width):

    df_res = pd.DataFrame(results)
    df_res = df_res.groupby(['instance_name'], sort=False).head(1)
    results = df_res.to_dict(orient='records')
    frame_label_list = []
    for res in results:
        label_list = pred_dict_to_labelme(res, 1, 0.05)
        frame_label_list += label_list
    img_ext = Path(image_path).suffix
    json_path = image_path.replace(img_ext, ".json")
    save_labels(json_path,
                str(Path(image_path).name),
                frame_label_list,
                height,
                width,
                imageData=None,
                save_image_to_json=False
                )
    return json_path


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimg_folder = "~/JPEGImages"\n\nfor img_file in glob.glob(img_folder+\'/*.jpg\'):\n    img = cv2.imread(img_file)\n    height, width, c = img.shape\n    # encode image as jpeg\n    _, img_encoded = cv2.imencode(\'.jpg\', img)\n    # send http request with image and receive response\n    response = requests.post(MODEL_SERVER_URL, data=img_encoded.tostring(), headers=headers)\n    # decode response\n    \n    instances = response.json()\n    json_path = instances_to_labelme(instances,img_file,height,width)\n    print(json_path)')
