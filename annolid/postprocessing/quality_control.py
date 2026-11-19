import ast
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import PIL.Image
import pycocotools.mask as mask_util

from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_to_polygons
from annolid.gui.shape import Shape
from annolid.utils.annotation_compat import img_pil_to_data

if TYPE_CHECKING:
    import pandas as pd


class _PandasProxy:
    """Import pandas lazily so GUI startup does not require optional analysis runtimes."""

    def __getattr__(self, name: str):
        import pandas as pandas

        return getattr(pandas, name)


pd = _PandasProxy()


def pred_dict_to_labelme(
    pred_row,
    keypoint_area_threshold=512,
    score_threshold=0.01,
    decode_segmentation=True,
):
    """Converted predict instance to labelme json format.

    Args:
        pred_row ([list[dict]]): [List of predict dicts]
        keypoint_area_threshold (int, optional):
        [area less than the threshold will be treated as keypoints ].
        Defaults to 512.
        score_threshold (float): class score threshold, default 0.01
        decode_segmentation (bool): decode segmentation masks into polygon shapes.
            Disable for interactive playback to reduce per-frame CPU cost.

    Returns:
        [List]: [A list of labeme shapes]
    """
    label_list = []
    try:
        _frame_number = pred_row["frame_number"]
        x1 = pred_row["x1"]
        y1 = pred_row["y1"]
        x2 = pred_row["x2"]
        y2 = pred_row["y2"]
        instance_name = pred_row["instance_name"]
        score = pred_row["class_score"]
        segmentation = pred_row["segmentation"]
        try:
            tracking_id = pred_row["tracking_id"]
        except KeyError:
            tracking_id = -1
    except (KeyError, TypeError, ValueError):
        return label_list

    if decode_segmentation and (
        segmentation
        and segmentation != "None"
        and segmentation != "nan"
        and score >= score_threshold
    ):
        try:
            _mask = ast.literal_eval(segmentation)
        except (ValueError, SyntaxError):
            _mask = segmentation
        mask_area = mask_util.area(_mask)
        if mask_area >= keypoint_area_threshold:
            _mask = mask_util.decode(_mask)[:, :]
            polys, has_holes = mask_to_polygons(_mask)
            if polys:
                polys = polys[0]
                label_name = instance_name
                if tracking_id is not None and tracking_id >= 0:
                    label_name += f"_{tracking_id}"

                shape = Shape(
                    label=label_name,
                    shape_type="polygon",
                    flags={},
                    visible=True,
                )
                all_points = np.array(list(zip(polys[0::2], polys[1::2])))
                for x, y in all_points:
                    if x >= 0 and y >= 0:
                        shape.addPoint((x, y))
                if shape.points:
                    # To close the polygon
                    shape.points.append(shape.points[0])
                    label_list.append(shape)
        else:
            shape = Shape(label=instance_name, shape_type="point", flags={})
            cx = round((x1 + x2) / 2, 2)
            cy = round((y1 + y2) / 2, 2)
            if cx >= 0 and cy >= 0:
                shape.addPoint((cx, cy))
                label_list.append(shape)
    if segmentation is None or segmentation == "None":
        rect_shape = Shape(
            label=f"{instance_name}_{tracking_id}", shape_type="rectangle", flags={}
        )
        rect_shape.addPoint((x1, y1))
        rect_shape.addPoint((x2, y2))
        label_list.append(rect_shape)

    return label_list


class TracksResults:
    """Representing the tracking results."""

    def __init__(self, video_file: str = None, tracking_csv: str = None) -> None:
        self.tracking_csv = tracking_csv
        self.video_file = video_file
        if self.tracking_csv and Path(self.tracking_csv).exists():
            self.df = pd.read_csv(self.tracking_csv)
        else:
            self.df = None
        if self.video_file and Path(self.video_file).exists():
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

    def to_labelme_json(
        self,
        output_dir: str,
        keypoint_area_threshold: int = 265,
        key_frames=False,
        skip_frames=30,
    ) -> str:
        if self.df is None or self.df.empty or self.cap is None:
            yield 100, "No detections available"
            self.clean_up()
            return

        width = self.video_width()
        # Fix the left right based on the middle of
        # length of the video width
        if width is not None and width > 0 and "instance_name" in self.df.columns:
            x_col = "cx" if "cx" in self.df.columns else "x1"
            if x_col in self.df.columns:
                inst_str = self.df["instance_name"].astype(str)
                x_vals = pd.to_numeric(self.df[x_col], errors="coerce")
                half_w = width / 2.0
                to_right = inst_str.str.contains("Left", regex=False) & (
                    x_vals >= half_w
                )
                to_left = inst_str.str.contains("Right", regex=False) & (
                    x_vals < half_w
                )
                if to_right.any() or to_left.any():
                    updated = inst_str.copy()
                    if to_right.any():
                        updated.loc[to_right] = inst_str.loc[to_right].str.replace(
                            "Left", "Right", regex=False
                        )
                    if to_left.any():
                        updated.loc[to_left] = inst_str.loc[to_left].str.replace(
                            "Right", "Left", regex=False
                        )
                    self.df["instance_name"] = updated
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        unique_frames = sorted(self.df.frame_number.unique())
        if not unique_frames:
            yield 100, "No detections available"
            self.clean_up()
            return

        if key_frames:
            if skip_frames and skip_frames > 1:
                selected_frames = set(unique_frames[::skip_frames])
            else:
                selected_frames = set(unique_frames)
        else:
            if skip_frames and skip_frames > 1:
                selected_frames = {fn for fn in unique_frames if fn % skip_frames == 0}
            else:
                selected_frames = set(unique_frames)

        for frame_number in unique_frames:
            progress = (frame_number / max(num_frames, 1)) * 100 + 1
            if frame_number not in selected_frames:
                yield progress, "Skipped"
                continue

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = self.cap.read()
            if not ret or frame is None:
                yield progress, "Failed to decode frame"
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_label_list = []
            img_path = (
                f"{output_dir}/{Path(self.video_file).stem}_{int(frame_number):09}.png"
            )
            PIL.Image.fromarray(frame).save(img_path)
            df_cur = self.df[self.df.frame_number == frame_number]
            for row in df_cur.to_dict(orient="records"):
                try:
                    label_list = pred_dict_to_labelme(row, keypoint_area_threshold)
                    frame_label_list += label_list
                    save_labels(
                        img_path.replace(".png", ".json"),
                        img_path,
                        frame_label_list,
                        height,
                        width,
                        imageData=None,
                        save_image_to_json=False,
                    )
                    yield progress, Path(img_path).stem + ".json"
                except ValueError:
                    yield 0, "No predictions"
                    continue

        yield 100, "Done"

        self.clean_up()

    def get_labels(self):
        return list(self.df.instance_name.unique())

    def find_last_show_position(
        self, instance_name="Female_52", frame_number=0, frames_backward=30
    ):
        """Find the last detection location and mask info the given instance and frame number

        Args:
            instance_name (str, optional): Instance name. Defaults to 'Female_52'.
            frame_number (int, optional): frame number. Defaults to 0.
            frames_backword (int, optional): number of frames back. Defaults to 30.

        Returns:
            pd.DataFrame: dataframe row
        """
        return (
            self.df[
                (self.df.instance_name == instance_name)
                & (self.df.frame_number < frame_number)
                & (self.df.frame_number > frame_number - frames_backward)
            ]
            .sort_values(by="frame_number", ascending=False)
            .head(1)
        )

    def find_future_show_position(
        self, instance_name="Female_52", frame_number=0, frames_forward=30
    ):
        """Find the next detection location and mask info the given instance and frame number

        Args:
            instance_name (str, optional): Instance name. Defaults to 'Female_52'.
            frame_number (int, optional): frame number. Defaults to 0.
            frames_forword (int, optional): number of frames forward. Defaults to 30.

        Returns:
            pd.DataFrame: dataframe row
        """
        return (
            self.df[
                (self.df.instance_name == instance_name)
                & (self.df.frame_number > frame_number)
                & (self.df.frame_number <= frame_number + frames_forward)
            ]
            .sort_values(by="frame_number", ascending=True)
            .head(1)
        )

    def get_missing_instances_names(self, frame_number, expected_instance_names=None):
        """Find the missing instance names in the current frame not in the expected list
        Args:
            frame_number (int): current video frame number
            expected_instance_names (list): a list of expected instances e.g.[mouse_1,mouse_2]
        """
        instance_names = self.df[self.df.frame_number == frame_number].instance_name
        unique_names_in_current_frame = set(instance_names.to_list())
        return set(expected_instance_names) - unique_names_in_current_frame

    def array_to_image_data(self, img_arr):
        img_pil = PIL.Image.fromarray(img_arr)
        _img_data = img_pil_to_data(img_pil)
        return _img_data

    @classmethod
    def switch_left_right(cls, row, width=800):
        instance_name = row["instance_name"]
        try:
            width_val = float(width)
        except (TypeError, ValueError):
            return instance_name
        if not np.isfinite(width_val) or width_val <= 0:
            return instance_name
        if "cx" in row:
            x_raw = row["cx"]
        elif "x1" in row:
            x_raw = row["x1"]
        else:
            return instance_name
        try:
            x_val = float(x_raw)
        except (TypeError, ValueError):
            return instance_name
        if not np.isfinite(x_val):
            return instance_name
        if "Left" in str(instance_name) and x_val >= width_val / 2:
            return str(instance_name).replace("Left", "Right")
        elif "Right" in str(instance_name) and x_val < width_val / 2:
            return str(instance_name).replace("Right", "Left")
        return instance_name

    def clean_up(self):
        if self.cap is not None:
            self.cap.release()

    def instance_center_distances(self, old_instances, cur_instances):
        """calculate the center distance between instances in the previous and current frames.

        Args:
            old_instances (pd.DataFrame): instances in the previous frame
            cur_instances (pd.DataFrame): instances in  the current frame

        Returns:
            dict: key: (prev frame_number, prev int(center_x), prev int(center_y),
                        current frame_number, current int(center_x), current int(center_y)
                  val: (dist, old instance name, current instance name)
        """
        dists = {}
        if (
            old_instances is None
            or cur_instances is None
            or old_instances.empty
            or cur_instances.empty
        ):
            return dists

        c_fn = cur_instances["frame_number"].to_numpy()
        c_cx = pd.to_numeric(cur_instances["cx"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        c_cy = pd.to_numeric(cur_instances["cy"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        c_names = cur_instances["instance_name"].to_numpy()

        o_fn = old_instances["frame_number"].to_numpy()
        o_cx = pd.to_numeric(old_instances["cx"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        o_cy = pd.to_numeric(old_instances["cy"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        o_names = old_instances["instance_name"].to_numpy()

        c_valid = np.isfinite(c_cx) & np.isfinite(c_cy)
        o_valid = np.isfinite(o_cx) & np.isfinite(o_cy)
        if not np.any(c_valid) or not np.any(o_valid):
            return dists

        d_mat = np.hypot(c_cx[:, None] - o_cx[None, :], c_cy[:, None] - o_cy[None, :])
        c_icx = np.where(c_valid, np.trunc(c_cx), 0).astype(int)
        c_icy = np.where(c_valid, np.trunc(c_cy), 0).astype(int)
        o_icx = np.where(o_valid, np.trunc(o_cx), 0).astype(int)
        o_icy = np.where(o_valid, np.trunc(o_cy), 0).astype(int)

        for ci_pos in np.flatnonzero(c_valid).tolist():
            cf = c_fn[ci_pos]
            ccx = int(c_icx[ci_pos])
            ccy = int(c_icy[ci_pos])
            cname = c_names[ci_pos]
            for oi_pos in np.flatnonzero(o_valid).tolist():
                of = o_fn[oi_pos]
                ocx = int(o_icx[oi_pos])
                ocy = int(o_icy[oi_pos])
                if cf == of and ccx == ocx and ccy == ocy:
                    continue
                key = (of, ocx, ocy, cf, ccx, ccy)
                dists[key] = (d_mat[ci_pos, oi_pos], o_names[oi_pos], cname)
        return dists

    def get_missing_instance_frames(self, instance_name="mouse_1"):
        """Get the frame numbers that do not have a prediction for instance with the
        provided instance name

        Args:
            instance_name (str, optional): instance name. Defaults to 'mouse_1'.

        Returns:
            set: frame numbers
        """
        if self.df is None or self.df.empty:
            return set()
        _df = self.df[self.df.instance_name == instance_name]
        if _df.empty:
            return set()
        valid_frames = pd.to_numeric(_df.frame_number, errors="coerce").dropna()
        if valid_frames.empty:
            return set()
        frame_ints = valid_frames.astype(int)
        max_frame_number = int(frame_ints.max())
        if max_frame_number < 0:
            return set()
        all_frames = set(range(0, max_frame_number + 1))
        frames_with_preds = set(frame_ints.tolist())
        return all_frames - frames_with_preds

    def fill_missing_instances(self, instance_name="mouse_2"):
        """Fill the given missing instance with the nearest existing predictions.

        Args:
            instance_name (str, optional): predicted instance name. Defaults to 'mouse_2'.

        Returns:
            pd.Dataframe: dataframe with filled instances
        """
        fill_rows = []
        missing_frames = list(
            self.get_missing_instance_frames(instance_name=instance_name)
        )
        for frame_number in sorted(missing_frames):
            fp = self.find_future_show_position(instance_name, frame_number).copy()
            lp = self.find_last_show_position(instance_name, frame_number).copy()
            if lp.empty and fp.empty:
                continue
            if lp.empty:
                fp["frame_number"] = frame_number
                fill_rows.append(fp)
            elif fp.empty:
                lp["frame_number"] = frame_number
                fill_rows.append(lp)
            elif (
                frame_number - lp.frame_number.values[0]
                > fp.frame_number.values[0] - frame_number
            ):
                fp["frame_number"] = frame_number
                fill_rows.append(fp)
            else:
                lp["frame_number"] = frame_number
                fill_rows.append(lp)
        if fill_rows:
            self.df = pd.concat([self.df, *fill_rows], ignore_index=True)
        del fill_rows
        return self.df
