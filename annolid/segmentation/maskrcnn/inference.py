import cv2
import torch
import numpy as np
from annolid.segmentation.maskrcnn.model import predict_coco
from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_to_polygons
from labelme import label_file
from labelme.shape import Shape
from pycocotools import mask as coco_mask


def predict_mask_to_labelme(img_path=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ny, nx, nc = img.shape

    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    preds = predict_coco(img_tensor)
    labels = preds[0]["labels"]
    labels = labels.numpy()
    label_list = []

    for i, i_mask in enumerate((preds[0]['masks'])):
        i_mask = i_mask[0].mul(255).cpu().numpy()
        polys, has_holes = mask_to_polygons(i_mask)
        polys = polys[0]

        shape = Shape(label=str(labels[i]), shape_type='polygon', flags={})
        for x, y in zip(polys[0::2], polys[1::2]):
            shape.addPoint((x, y))
        label_list.append(shape)
    save_labels(img_path.replace('.jpg', '.json'),
                img_path, label_list, ny, nx,)
