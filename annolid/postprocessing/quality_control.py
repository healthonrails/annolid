import pandas as pd
import os
from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_area, mask_to_polygons
from labelme import label_file
from labelme.shape import Shape
from pathlib import Path
import cv2
import pycocotools.mask as mask_util
import ast
import PIL.Image
import numpy as np
from labelme.utils.image import img_pil_to_data


class TracksResults():
    """Representing the tracking results.
    """

    def __init__(self, video_file: str = None,
                 tracking_csv: str = None) -> None:
        self.tracking_csv = tracking_csv
        self.video_file = video_file
        if Path(self.tracking_csv).exists:
            self.df = pd.read_csv(self.tracking_csv)
        else:
            self.df = None
        if Path(self.video_file).exists:
            self.cap = cv2.VideoCapture(self.video_file)
        else:
            self.cap = None

    def to_labelme_json(self,
                        output_dir: str,
                        keypoint_area_threshold: int = 265
                        ) -> str:

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        while self.cap.isOpened():
            label_list = []

            frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            img_path = f"{output_dir}/{Path(self.video_file).stem}_{frame_number:09}.png"

            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            PIL.Image.fromarray(frame).save(img_path)
            df_cur = self.df[self.df.frame_number == frame_number]
            for row in tuple(df_cur.values):
                _, _frame_number, x1, y1, x2, y2, instance_name, score, segmetnation = row
                _mask = ast.literal_eval(segmetnation)
                mask_area = mask_util.area(_mask)
                if mask_area >= keypoint_area_threshold:
                    _mask = mask_util.decode(_mask)[:, :]
                    polys, has_holes = mask_to_polygons(_mask)
                    try:
                        polys = polys[0]

                        shape = Shape(label=instance_name,
                                      shape_type='polygon',
                                      flags={}
                                      )
                        all_points = np.array(
                            list(zip(polys[0::2], polys[1::2])))
                        for x, y in all_points:
                            shape.addPoint((x, y))
                        label_list.append(shape)
                    except IndexError:
                        continue
                else:
                    shape = Shape(label=instance_name,
                                  shape_type='point',
                                  flags={}
                                  )
                    cx = round((x1 + x2) / 2, 2)
                    cy = round((y1 + y2) / 2, 2)
                    shape.addPoint((cx, cy))
                    label_list.append(shape)
                save_labels(img_path.replace(".png", ".json"),
                            img_path,
                            label_list,
                            height,
                            width,
                            imageData=None,
                            save_image_to_json=False

                            )

        self.clean_up()
        return output_dir

    def get_labels(self):
        return list(self.df.instance_name.unique())

    def array_to_image_data(self, img_arr):
        img_pil = PIL.Image.fromarray(img_arr)
        _img_data = img_pil_to_data(img_pil)
        return _img_data

    def clean_up(self):
        self.cap.release()
