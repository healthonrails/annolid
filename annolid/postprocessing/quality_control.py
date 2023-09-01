import pandas as pd
import os
from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_to_polygons
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
                         keypoint_area_threshold=512,
                         score_threshold=0.01
                         ):
    """Converted predict instance to labelme json format.

    Args:
        pred_row ([list[dict]]): [List of predict dicts]
        keypoint_area_threshold (int, optional): 
        [area less than the threshold will be treated as keypoints ]. 
        Defaults to 512.
        score_threshold (float): class score threshold, default 0.01

    Returns:
        [List]: [A list of labeme shapes]
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
        try:
            tracking_id = pred_row["tracking_id"]
        except KeyError:
            tracking_id = -1
    except ValueError:
        return

    if segmentation and segmentation != 'None' and segmentation != 'nan' and score >= score_threshold:
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
                label_name = instance_name
                if tracking_id and tracking_id >= 0:
                    label_name += f"_{tracking_id}"

                shape = Shape(label=label_name,
                              shape_type='polygon',
                              flags={}
                              )
                all_points = np.array(
                    list(zip(polys[0::2], polys[1::2])))
                for x, y in all_points:
                    # do not add 0,0 to the list
                    if x >= 1 and y >= 1:
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
            if cx >= 1 and cy >= 1:
                shape.addPoint((cx, cy))
                label_list.append(shape)
    if segmentation is None or segmentation == 'None':
        rect_shape = Shape(label=f"{instance_name}_{tracking_id}",
                           shape_type='rectangle',
                           flags={}
                           )
        rect_shape.addPoint((x1, y1))
        rect_shape.addPoint((x2, y2))
        label_list.append(rect_shape)

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
                                frame_number=0,
                                frames_backward=30
                                ):
        """Find the last detection location and mask info the given instance and frame number

        Args:
            instance_name (str, optional): Instance name. Defaults to 'Female_52'.
            frame_number (int, optional): frame number. Defaults to 0.
            frames_backword (int, optional): number of frames back. Defaults to 30.

        Returns:
            pd.DataFrame: dataframe row
        """
        return self.df[(self.df.instance_name == instance_name) &
                       (self.df.frame_number < frame_number) &
                       (self.df.frame_number > frame_number - frames_backward)
                       ].sort_values(by='frame_number',
                                     ascending=False).head(1)

    def find_future_show_position(self,
                                  instance_name='Female_52',
                                  frame_number=0,
                                  frames_forward=30
                                  ):
        """Find the next detection location and mask info the given instance and frame number

        Args:
            instance_name (str, optional): Instance name. Defaults to 'Female_52'.
            frame_number (int, optional): frame number. Defaults to 0.
            frames_forword (int, optional): number of frames forward. Defaults to 30.

        Returns:
            pd.DataFrame: dataframe row
        """
        return self.df[(self.df.instance_name == instance_name) &
                       (self.df.frame_number > frame_number) &
                       (self.df.frame_number <= frame_number + frames_forward)
                       ].sort_values(by='frame_number',
                                     ascending=True).head(1)

    def get_missing_instances_names(self,
                                    frame_number,
                                    expected_instance_names=None):
        """Find the missing instance names in the current frame not in the expected list
        Args:
            frame_number (int): current video frame number
            expected_instance_names (list): a list of expected instances e.g.[mouse_1,mouse_2]
        """
        instance_names = self.df[self.df.frame_number ==
                                 frame_number].instance_name
        unique_names_in_current_frame = set(instance_names.to_list())
        return set(expected_instance_names) - unique_names_in_current_frame

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

    def instance_center_distances(self, old_instances, cur_instances):
        """calculate the center distance between instances in the previous and current frames.

        Args:
            old_instances (pd.DataFrame): instances in the previous frame
            cur_instances (pd.DataFrame): instances in  the current frame

        Returns:
            dict: key: (prev frame_number, prev int(center_x), prev int(center_y),
                        current frame_number, current int(center_x),curent int(center_y)
                  val: (dist, old instance name, current instance name)
        """
        dists = {}
        for cidx, ci in cur_instances.iterrows():
            for oidx, oi in old_instances.iterrows():
                if (ci['frame_number'] == oi['frame_number']
                        and int(ci['cx']) == int(oi['cx'])
                        and int(ci['cy']) == int(oi['cy'])
                        ):
                    continue
                dist = np.sqrt((ci['cx'] - oi['cx'])**2 +
                               (ci['cy']-oi['cy']) ** 2)
                key = (oi['frame_number'], int(oi['cx']), int(oi['cy']),
                       ci['frame_number'], int(ci['cx']), int(ci['cy'])
                       )
                dists[key] = (dist, oi['instance_name'], ci['instance_name'])
        return dists

    def get_missing_instance_frames(self,
                                    instance_name='mouse_1'):
        """Get the frame numbers that do not have a prediction for instance with the 
        provided instance name

        Args:
            instance_name (str, optional): instance name. Defaults to 'mouse_1'.

        Returns:
            set: frame numbers
        """

        _df = self.df[self.df.instance_name == instance_name]
        max_frame_number = max(_df.frame_number)
        all_frames = set(range(0, max_frame_number+1))
        frames_with_preds = set(_df.frame_number)
        del _df
        return all_frames - frames_with_preds

    def fill_missing_instances(self, instance_name='mouse_2'):
        """Fill the given missing instance with the nearest exsiting predicitons

        Args:
            instance_name (str, optional): predicted instance name. Defaults to 'mouse_2'.

        Returns:
            pd.Dataframe: dataframe with filled instances
        """
        fill_rows = []
        missing_frames = list(self.get_missing_instance_frames(
            instance_name=instance_name))
        for frame_number in sorted(missing_frames):
            fp = self.find_future_show_position(instance_name, frame_number)
            lp = self.find_last_show_position(instance_name, frame_number)
            if frame_number - lp.frame_number.values[0] > fp.frame_number.values[0] - frame_number:
                fp.frame_number = frame_number
                fill_rows.append(fp)
            else:
                lp.frame_number = frame_number
                fill_rows.append(lp)
        self.df = self.df.append(fill_rows, ignore_index=True)
        del fill_rows
        return self.df
