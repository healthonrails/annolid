import json


def mot_to_coco(mot_anno_file, image_dir):
    """Convert the MOT-17 annotation file to the COCO format.

    Args:
    - mot_anno_file (str): Path to the MOT-17 annotation file.
    - image_dir (str): Path to the directory containing the images.

    Returns:
    - coco_format (dict): A dictionary in the COCO format, containing the annotations and meta information.
    """
    coco_format = {
        "images": [],
        "categories": [],
        "annotations": []
    }

    # Load the MOT-17 annotation file
    with open(mot_anno_file, "r") as f:
        anno_data = f.readlines()

    # Split the data into header and annotations
    header = anno_data[0].strip().split(",")
    annotations = anno_data[1:]

    # Get the unique image names and add them to the images list
    image_names = list(set([x.strip().split(",")[0] for x in annotations]))
    for idx, image_name in enumerate(image_names):
        coco_format["images"].append({
            "id": idx + 1,
            "file_name": image_name,
            "width": None,
            "height": None
        })

    # Add the categories
    coco_format["categories"].append({
        "id": 1,
        "name": "person",
        "supercategory": "person"
    })

    # Add the annotations
    for idx, annotation in enumerate(annotations):
        anno_data = annotation.strip().split(",")
        image_id = image_names.index(anno_data[0]) + 1
        xmin, ymin, w, h = map(
            float, [anno_data[2], anno_data[3], anno_data[4], anno_data[5]])
        xmax = xmin + w
        ymax = ymin + h

        coco_format["annotations"].append({
            "id": idx + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [xmin, ymin, w, h],
            "area": w * h,
            "iscrowd": 0
        })

    return coco_format
