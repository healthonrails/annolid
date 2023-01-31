import os
import json
from pathlib import Path
import shutil


def xywh2cxcywh(box, img_size):
    """convert COCO bounding box format to YOLO format.
    The YOLO format bounding box values were normalized with width and height values


    Args:
        box (list): COCO bounding box format (x_top_left, y_top_left, width, height)
        img_size (tuple): (img_width, img_height)

    Returns:
        tuple: normalized YOLO format bounding box (x_center,y_center, width, height)
    """
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
                   results_dir='yolo_dataset',
                   dataset_type='train',
                   class_id=None,
                   is_segmentation=True
                   ):
    """Convert COCO format dataset to YOLOV format

    Args:
        json_file (str, optional): file path for annotation.json. Defaults to 'annotation.json'.
        results_dir (str, optional):  result directory. Defaults to 'yolov5_dataset'.
        dataset_type (str, optional): train or val or test. Defaults to 'train'.
        class_id (int, optional): class id. Defaults to None.
        is_segmentation (bool, optional): segmentation labeling. Defaults to True.

    Returns:
        list: a list of labeled class names
    """
    categories = []
    images_path = Path(f"{results_dir}/{dataset_type}/images")
    images_path.mkdir(parents=True, exist_ok=True)

    labels_path = Path(f"{results_dir}/{dataset_type}/labels")
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
                    if ann["segmentation"] and is_segmentation:
                        points = []
                        i = 0
                        while i <= len(ann['segmentation'][0])/2:
                            points.append(str(ann['segmentation']
                                              [0][i] / img_width))
                            points.append(str(ann['segmentation'][0]
                                              [i+1] / img_height))
                            i += 2
                        if class_id is not None:
                            atf.write(
                                f"{class_id} {ann['category_id']} {' '.join(points)}\n")
                        else:
                            atf.write(
                                f"{ann['category_id']} {' '.join(points)}\n")
                    elif not is_segmentation:
                        if ann['bbox']:
                            box = xywh2cxcywh(
                                ann["bbox"], (img_width, img_height))
                            if class_id is not None:
                                atf.write("%s %s %s %s %s %s\n" % (class_id, ann["category_id"], box[0],
                                                                   box[1], box[2], box[3]))
                            else:
                                atf.write("%s %s %s %s %s\n" % (ann["category_id"], box[0],
                                                                box[1], box[2], box[3]))

    for c in data["categories"]:
        # exclude backgroud with id 0
        if not c['id'] == 0:
            categories.append(c['name'])

    data_yaml = Path(f"{results_dir}/data.yaml")
    names = list(categories)
    # dataset folder is in same dir as the yolov5 folder
    with open(data_yaml, 'w') as dy:
        dy.write(f"train: {os.path.basename(results_dir)}/train/images\n")
        dy.write(f"val: {os.path.basename(results_dir)}/val/images\n")
        dy.write(f"nc: {len(names)}\n")
        dy.write(f"names: {names}")

    return names
