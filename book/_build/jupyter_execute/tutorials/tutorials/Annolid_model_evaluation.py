#!/usr/bin/env python
# coding: utf-8

# # Explore and evaluate Annolid models

# In[ ]:


#This is modified from https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html
#!pip install fiftyone


# In[ ]:


import torch
import cv2
import torchvision
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util
from torchvision.transforms import functional as func


# In[ ]:


import fiftyone as fo
from annolid.inference.predict import Segmentor


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# The directory containing the source images
data_path = "/path/to/my_coco_dataset/valid"


# In[ ]:


# The path to the COCO labels JSON file
labels_path = "/path/to/my_coco_dataset/valid/annotations.json"


# In[ ]:


segmentor = Segmentor("/path/to/my_coco_dataset",
        "/path/to/model_final.pth")


# In[ ]:


segmentor.class_names


# In[ ]:


# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    label_types="segmentations"
)


# In[ ]:


sample = dataset.first()


# In[ ]:


print(sample.ground_truth.detections[0])


# In[ ]:


session = fo.launch_app(dataset)


# In[ ]:


# Choose a random subset of 100 samples to add predictions to
predictions_view = dataset.take(100, seed=51)


# In[ ]:


# Get class list
classes = dataset.default_classes[1:]

# Add predictions to samples
with fo.ProgressBar() as pb:
    for sample in pb(predictions_view):
        # Load image
        image = Image.open(sample.filepath)
        _image = cv2.imread(sample.filepath)
        image = func.to_tensor(image).to(device)
        c, h, w = image.shape

        # Perform inference
        preds = segmentor.predictor(_image)
        instances = preds["instances"]
        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        labels = instances.pred_classes.tolist()
        has_mask = instances.has("pred_masks")

        if has_mask:
            rles = [
                mask_util.encode(
                    np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

        # Convert detections to FiftyOne format
        detections = []
        for label, score, box in zip(labels, scores, boxes):
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x1, y1, x2, y2 = box
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

            detections.append(
                fo.Detection(
                    label=classes[label],
                    bounding_box=rel_box,
                    confidence=score,
                    masks=rles
                )
            )

        # Save predictions to dataset
        sample["mask_rcnn"] = fo.Detections(detections=detections)
        sample.save()

print("Finished adding predictions")


# In[ ]:


session.view = predictions_view


# In[ ]:


from fiftyone import ViewField as F

# Only contains detections with confidence >= 0.15
high_conf_view = predictions_view.filter_labels("mask_rcnn", F("confidence") > 0.15)


# In[ ]:


# Evaluate the predictions in the `faster_rcnn` field of our `high_conf_view`
# with respect to the objects in the `ground_truth` field
results = high_conf_view.evaluate_detections(
    "mask_rcnn",
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
)


# In[ ]:


counts = dataset.count_values("ground_truth.detections.label")
classes_all = sorted(counts, key=counts.get, reverse=True)

# Print a classification report for the top-10 classes
results.print_report(classes=classes_all)


# In[ ]:


print(results.mAP())


# In[ ]:


plot = results.plot_pr_curves(classes=["RightInteract"])
plot.show()


# In[ ]:


session.view = high_conf_view.sort_by("eval_fp", reverse=True)


# In[ ]:


session.show()


# In[ ]:


session.freeze()  # screenshot the active App for sharing


# In[ ]:


# Tag all highly confident false positives as "possibly-missing"
(
    high_conf_view
        .filter_labels("mask_rcnn", F("eval") == "fp")
        .select_fields("mask_rcnn")
        .tag_labels("possibly-missing")
)


# In[ ]:


# Export all labels with the `possibly-missing` tag in CVAT format
(
    dataset
        .select_labels(tags=["possibly-missing"])
        .export("~/Downloads/possoible-missing-dataset", fo.types.COCODetectionDataset)
)


# In[ ]:


# Compute metadata so we can reference image height/width in our view
dataset.compute_metadata()


# In[ ]:


#
# Create an expression that will match objects whose bounding boxes have
# area less than 32^2 pixels
#
# Bounding box format is [top-left-x, top-left-y, width, height]
# with relative coordinates in [0, 1], so we multiply by image
# dimensions to get pixel area
#
bbox_area = (
    F("$metadata.width") * F("bounding_box")[2] *
    F("$metadata.height") * F("bounding_box")[3]
)
small_boxes = bbox_area < 32 ** 2

# Create a view that contains only small (and high confidence) predictions
small_boxes_view = high_conf_view.filter_labels("mask_rcnn", small_boxes)

session.view = small_boxes_view


# In[ ]:


# Create a view that contains only small GT and predicted boxes
small_boxes_eval_view = (
    high_conf_view
    .filter_labels("ground_truth", small_boxes)
    .filter_labels("mask_rcnn", small_boxes)
)

# Run evaluation
small_boxes_results = small_boxes_eval_view.evaluate_detections(
    "mask_rcnn",
    gt_field="ground_truth",
)


# In[ ]:



# Get the 10 most common small object classes
small_counts = small_boxes_eval_view.count_values("ground_truth.detections.label")
classes_top10_small = sorted(small_counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 small object classes
small_boxes_results.print_report(classes=classes_top10_small)


# In[ ]:




