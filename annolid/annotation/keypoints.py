import os
import cv2
import numpy as np
import pandas as pd
from labelme import label_file
from labelme.shape import Shape


def save_labels(filename,
                imagePath,
                label_list,
                height,
                width,
                imageData=None,
                otherData=None,
                save_image_to_json=False
                ):
    lf = label_file.LabelFile()

    def format_shape(s):
        data = s.other_data.copy()
        data.update(
            dict(
                label=s.label,
                points=s.points,
                group_id=s.group_id,
                shape_type=s.shape_type,
                flags=s.flags,
            )
        )
        return data

    shapes = [format_shape(item) for item in label_list]
    flags = {}

    if imageData is None and save_image_to_json:
        imageData = label_file.LabelFile.load_image_file(
            imagePath)

    if otherData is None:
        otherData = {}

    lf.save(
        filename=filename,
        shapes=shapes,
        imagePath=imagePath,
        imageData=imageData,
        imageHeight=height,
        imageWidth=width,
        otherData=otherData,
        flags=flags,
    )


def to_labelme(img_folder,
               anno_file):
    """Convert keypoints format to labelme json files.

    Args:
        img_folder (path): folder where images are located
        anno_file (h5): keypoints annotation file.
    """
    df = pd.read_hdf(os.path.join(img_folder, anno_file))
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = df.columns.get_level_values(1)

    for ind, imname in enumerate(df.index):
        img_path = os.path.join(img_folder, imname)
        image = cv2.imread(img_path)
        ny, nx, nc = np.shape(image)
        image_height = ny
        image_width = nx
        label_list = []
        for b in set(bodyparts):
            s = Shape(label=b, shape_type='point', flags={})
            x = df.iloc[ind][scorer][b]['x']
            y = df.iloc[ind][scorer][b]['y']
            if np.isfinite(x) and np.isfinite(y):
                s.addPoint((x, y))
                label_list.append(s)
        save_labels(img_path.replace('.png', '.json'),
                    img_path, label_list, image_height,
                    image_width)
