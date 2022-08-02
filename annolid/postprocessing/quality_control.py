import pandas as pd
import os
from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_area, mask_to_polygons
from labelme import label_file
from labelme.shape import Shape
from pathlib import Path
import cv2
import decord as de
import pycocotools.mask as mask_util
import ast
import PIL.Image
import numpy as np
from labelme.utils.image import img_pil_to_data


def pred_dict_to_labelme(pred_row,
                         keypoint_area_threshold=256,
                         score_threshold=0.5
                         ):
    """[summary]

    Args:
        pred_row ([list[dict]]): [List of predict dicts]
        keypoint_area_threshold (int, optional): 
        [area less than the threshold will be treated as keypoints ]. 
        Defaults to 256.

    Returns:
        [type]: [description]
    """
    label_list = []
    try:
        _frame_number = pred_row['frame_number']
        x1 = pred_row['x1']
        y1 = pred_row['y1']
        x2 = pred_row['x2']
        y2 = pred_row['y2']
        instance_name = pred_row['instance_name']
        score = pred_row['class_score']
        segmentation = pred_row["segmentation"]
    except ValueError:
        return

    if segmentation and segmentation != 'nan' and score >= score_threshold:
        try:
            _mask = ast.literal_eval(segmentation)
        except ValueError:
            _mask = segmentation
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
                pass
        else:
            shape = Shape(label=instance_name,
                          shape_type='point',
                          flags={}
                          )
            cx = round((x1 + x2) / 2, 2)
            cy = round((y1 + y2) / 2, 2)
            shape.addPoint((cx, cy))
            label_list.append(shape)
    if segmentation is None:
        shape = Shape(label=instance_name,
                      shape_type='point',
                      flags={}
                      )
        cx = round((x1 + x2) / 2, 2)
        cy = round((y1 + y2) / 2, 2)
        shape.addPoint((cx, cy))
        label_list.append(shape)

    return label_list


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
        self._is_running = True

    def video_width(self):
        if self.cap is not None:
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            video_width = None
        return video_width

    def to_labelme_json(self,
                        output_dir: str,
                        keypoint_area_threshold: int = 265,
                        key_frames=False,
                        skip_frames=30,
                        ) -> str:

        width = self.video_width()
        # Fix the left right based on the middle of
        # lenght of the video width
        if self.df is not None:
            self.df['instance_name'] = self.df.apply(
                lambda row: self.switch_left_right(row,
                                                   width),
                axis=1)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if key_frames is not None and key_frames:
            with open(self.video_file, 'rb') as f:
                self.vr = de.VideoReader(f)
            de.bridge.set_bridge('native')
            self.key_frame_inices = self.vr.get_key_indices()
            for kf in self.key_frame_inices:
                frame = self.vr[kf]
                frame = frame.asnumpy()
                frame_label_list = []
                img_path = f"{output_dir}/{Path(self.video_file).stem}_{kf:09}.png"
                PIL.Image.fromarray(frame).save(img_path)
                df_cur = self.df[self.df.frame_number == kf]
                for row in df_cur.to_dict(orient='records'):
                    try:
                        label_list = pred_dict_to_labelme(
                            row,
                            keypoint_area_threshold
                        )
                        # each row is a dict of the single instance prediction
                        frame_label_list += label_list
                        save_labels(img_path.replace(".png", ".json"),
                                    img_path,
                                    frame_label_list,
                                    height,
                                    width,
                                    imageData=None,
                                    save_image_to_json=False

                                    )
                        yield (kf / num_frames) * 100 + 1, Path(img_path).stem + '.json'
                    except ValueError:
                        yield 0, 'No predictions'
                        continue

        else:
            frame_numbers = self.df.frame_number.unique()
            for frame_number in frame_numbers:
                if frame_number % skip_frames == 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_label_list = []
                        img_path = f"{output_dir}/{Path(self.video_file).stem}_{frame_number:09}.png"
                        PIL.Image.fromarray(frame).save(img_path)
                        df_cur = self.df[self.df.frame_number == frame_number]
                        for row in df_cur.to_dict(orient='records'):
                            try:
                                label_list = pred_dict_to_labelme(
                                    row,
                                    keypoint_area_threshold
                                )
                                # each row is a dict of the single instance prediction
                                frame_label_list += label_list
                                save_labels(img_path.replace(".png", ".json"),
                                            img_path,
                                            frame_label_list,
                                            height,
                                            width,
                                            imageData=None,
                                            save_image_to_json=False

                                            )
                                yield (frame_number / num_frames) * 100 + 1, Path(img_path).stem + '.json'
                            except ValueError:
                                yield 0, 'No predictions'
                                continue
                else:
                    yield (frame_number / num_frames) * 100 + 1, 'Skipped'
            yield 100, 'Done'

        self.clean_up()

    def get_labels(self):
        return list(self.df.instance_name.unique())

    def find_last_show_position(self,
                                instance_name='Female_52',
                                frame_number=0):
        """Find the last detection location and mask info the given instance and frame number

        Args:
            instance_name (str, optional): Instance name. Defaults to 'Female_52'.
            frame_number (int, optional): frame number. Defaults to 0.

        Returns:
            pd.DataFrame: dataframe row
        """
        return self.df[(self.df.instance_name == instance_name) &
                       (self.df.frame_number < frame_number)].sort_values(by='frame_number',
                                                                          ascending=False).head(1)

    def array_to_image_data(self, img_arr):
        img_pil = PIL.Image.fromarray(img_arr)
        _img_data = img_pil_to_data(img_pil)
        return _img_data

    @classmethod
    def switch_left_right(self, row, width=800):
        instance_name = row['instance_name']
        if 'cx' in row:
            x_val = row['cx']
        else:
            x_val = row['x1']
        if 'Left' in str(instance_name) and x_val >= width / 2:
            return instance_name.replace('Left', 'Right')
        elif 'Right' in str(instance_name) and x_val < width / 2:
            return instance_name.replace('Right', 'Left')
        return instance_name

    def clean_up(self):
        self.cap.release()
