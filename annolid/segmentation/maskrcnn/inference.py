import cv2
import torch
from annolid.segmentation.maskrcnn.model import predict_coco
from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_to_polygons
from annolid.gui.shape import Shape
import yaml
from pathlib import Path

here = Path(__file__).parent


def get_coco_labels(config_file=str(here.parent.parent / "configs/coco_labels.yaml")):
    with open(config_file, "r") as cfg:
        coco_labels = yaml.load(cfg, Loader=yaml.FullLoader)
    return coco_labels.split()


def predict_mask_to_labelme(img_path=None, score_threshold=0.29):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ny, nx, nc = img.shape

    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    preds = predict_coco(img_tensor)
    labels = preds[0]["labels"]
    scores = preds[0]["scores"]
    labels = labels.numpy()
    label_list = []

    coco_labels = get_coco_labels()

    for i, i_mask in enumerate((preds[0]["masks"])):
        i_mask = i_mask[0].mul(255).cpu().numpy()
        score = scores[i].cpu().numpy()
        if score >= score_threshold:
            polys, has_holes = mask_to_polygons(i_mask)
            polys = polys[0]

            shape = Shape(label=coco_labels[labels[i]], shape_type="polygon", flags={})
            for x, y in zip(polys[0::2], polys[1::2]):
                shape.addPoint((x, y))
            label_list.append(shape)
    save_labels(
        img_path.replace(".jpg", ".json"),
        img_path,
        label_list,
        ny,
        nx,
    )
