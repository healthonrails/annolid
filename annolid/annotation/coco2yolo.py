import os
import json
from pathlib import Path
import shutil


def xywh2cxcywh(box, img_size):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]

    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0

    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def create_dataset(json_file='annotation.json',
                   results_dir='yolov5_dataset',
                   dataset_type='train'
                   ):
    categories = []
    images_path = Path(f"{results_dir}/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok=True)

    labels_path = Path(f"{results_dir}/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exist_ok=True)

    with open(json_file, 'r') as jf:
        data = json.load(jf)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    img_file_path = Path(json_file).parent

    for img in data['images']:
        file_name = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]

        img_id = img["id"]

        file_name = file_name.replace("\\", "/")
        shutil.copy(img_file_path / file_name, images_path)
        anno_txt_name = os.path.basename(file_name).split(".")[0] + ".txt"
        anno_txt_flie = labels_path / anno_txt_name
        with open(anno_txt_flie, 'w') as atf:
            for ann in data['annotations']:
                if ann["image_id"] == img_id:
                    box = xywh2cxcywh(ann["bbox"], (img_width, img_height))
                    atf.write("%s %s %s %s %s\n" % (ann["category_id"], box[0],
                                                    box[1], box[2], box[3]))

    for c in data["categories"]:
        # exclude backgroud with id 0
        if not c['id'] == 0:
            categories.append(c['name'])

    data_yaml = Path(f"{results_dir}/data.yaml")
    names = list(set(categories))
    # dataset folder is in same dir as the yolov5 folder
    with open(data_yaml, 'w') as dy:
        dy.write(f"train: ../{results_dir}/images/train\n")
        dy.write(f"val: ../{results_dir}/images/val\n")
        dy.write(f"nc: {len(names)}\n")
        dy.write(f"names: {names}")

    return names
