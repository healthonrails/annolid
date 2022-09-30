"""
    This module converts COCO format dataset to labelme JSON files.
"""

import glob
import os
import logging
import json
from pathlib import Path
from labelme.shape import Shape
from keypoints import save_labels

logger = logging.getLogger(__name__)


class COCO2Labeme():
    """
    Given COCO format dataset with annotation.json file and the parent 
    folder that contains the images as provided in annotion images dict.
    Usage:
    cl = COCO2Labeme('/path/to/annotations_jsons',
                     'path/to/images_dir')
    cl.convert()
    """

    def __init__(self, annotions, images_dir) -> None:
        self.annotations = annotions
        self.images_dir = images_dir

    def get_annos(self):
        """find all anon json files

        Returns:
            list: json files
        """
        annos = glob.glob(self.annotations + '/*.json')
        return annos

    def parse_coco_json(self, json_file):
        if not os.path.exists(json_file):
            logger.info("{json_file} does not exist")

        with open(json_file, 'r') as jf:
            data = json.load(jf)
        return data

    def to_labelme(self, json_data):
        class_names = json_data['categories'][0]['keypoints']
        images = json_data['images']
        annotations = json_data['annotations']
        for img in images:
            img_id = img['id']
            img_file_name = os.path.join(self.images_dir, img['file_name'])
            if not os.path.exists(img_file_name):
                logger.info(f"Image file {img_file_name} does not exist!")
                continue
            img_file_json = Path(img_file_name).with_suffix('.json')
            img_height = img['height']
            img_width = img['width']
            try:
                img_kpts = annotations[img_id]['keypoints']
            except IndexError:
                logger.info(f"No keypoints for  img {img_file_name}")
                continue
            label_list = []
            for i, label in enumerate(class_names):
                i *= 3
                x, y, v = img_kpts[i:i+3]
                if not (x <= 0 and y <= 0):
                    s = Shape(label=label, shape_type='point', flags={})
                    s.addPoint((x, y))
                    label_list.append(s)
            save_labels(img_file_json,
                        img_file_name, label_list, img_height, img_width)

    def convert(self):
        json_files = self.get_annos()
        for jf in json_files:
            json_data = self.parse_coco_json(jf)
            self.to_labelme(json_data)
            logger.info(f"Finished {jf} .")
