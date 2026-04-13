import os
import cv2
import pandas as pd
import math
import ast
import json
import functools
import operator
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from annolid.utils import draw
from annolid.data.videos import frame_from_video
from collections import deque
import pycocotools.mask as mask_util
from annolid.postprocessing.freezing_analyzer import FreezingAnalyzer
from annolid.postprocessing.quality_control import TracksResults


class BehaviorAnalyzer:
    """Analyzes behaviors such as interactions, freezing, and zones for a given tracking dataframe."""

    def __init__(
        self,
        df,
        df_motion,
        subject_animal_name,
        left_interact_object,
        right_interact_object,
    ):
        self.df = df
        self.df_motion = df_motion
        self.subject_animal_name = subject_animal_name
        self.left_interact_object = left_interact_object
        self.right_interact_object = right_interact_object

    def get_bbox(self, frame_number):
        _df = self.df[self.df.frame_number == frame_number]
        try:
            res = _df.to_dict(orient="records")
        except Exception:
            res = []
        return res

    def is_freezing(self, frame_number, instance_name):
        if self.df_motion is not None:
            res = self.df_motion[
                (self.df_motion.frame_number == frame_number)
                & (self.df_motion.instance_name == instance_name)
            ]
            if not res.empty:
                freezing = res.freezing.values[0]
                return freezing > 0
        return False

    @staticmethod
    def animal_in_zone(animal_mask, zone_mask, threshold):
        if animal_mask is not None and zone_mask is not None:
            overlap = mask_util.iou([animal_mask], [zone_mask], [0]).flatten()[0]
            return overlap > threshold
        return False

    def keypoint_in_body_mask(self, frame_number, keypoint_name, animal_name=None):
        if animal_name is None:
            animal_name = self.subject_animal_name

        _df_k_b = self.df[self.df.frame_number == frame_number]
        try:
            body_seg = _df_k_b[_df_k_b.instance_name == animal_name][
                "segmentation"
            ].values[0]
            body_seg = ast.literal_eval(body_seg)
        except IndexError:
            return False

        try:
            keypoint_seg = _df_k_b[_df_k_b.instance_name == keypoint_name][
                "segmentation"
            ].values[0]
            keypoint_seg = ast.literal_eval(keypoint_seg)
        except IndexError:
            return False

        if keypoint_seg and body_seg:
            overlap = mask_util.iou([body_seg], [keypoint_seg], [0]).flatten()[0]
            return overlap > 0
        return False

    def left_right_interact(
        self,
        fn,
        subject_instance="subject_vole",
        left_instance="left_vole",
        right_instance="right_vole",
    ):
        _df_top = self.df[self.df.frame_number == fn]
        right_interact = 0.0
        left_interact = 0.0
        try:
            subject_instance_seg = _df_top[_df_top.instance_name == subject_instance][
                "segmentation"
            ].values[0]
            subject_instance_seg = ast.literal_eval(subject_instance_seg)
        except IndexError:
            return 0.0, 0.0

        try:
            left_instance_seg = _df_top[_df_top.instance_name == left_instance][
                "segmentation"
            ].values[0]
            left_instance_seg = ast.literal_eval(left_instance_seg)
            if left_instance_seg and subject_instance_seg:
                left_interact = mask_util.iou(
                    [left_instance_seg], [subject_instance_seg], [0]
                ).flatten()[0]
        except IndexError:
            pass

        try:
            right_instance_seg = _df_top[_df_top.instance_name == right_instance][
                "segmentation"
            ].values[0]
            right_instance_seg = ast.literal_eval(right_instance_seg)
            if right_instance_seg and subject_instance_seg:
                right_interact = mask_util.iou(
                    [right_instance_seg], [subject_instance_seg], [0]
                ).flatten()[0]
        except IndexError:
            pass

        return left_interact, right_interact


class GlitterNixExporter:
    """Handles accumulating metadata, timestamps, and zone properties for NIX CSV export."""

    def __init__(self, video_file, width, height, zone_info, df):
        self.metadata_dict = {
            "filename": video_file,
            "pixels_per_meter": 0,
            "video_width": f"{width}",
            "video_height": f"{height}",
            "saw_all_timestamps": "TRUE",
        }
        self.zone_dict = {}
        self.zone_background_dict = None
        self.zone_info = zone_info

        self.timestamps = {}

        if (
            self.zone_info is not None
            and getattr(self.zone_info, "suffix", "") != ".json"
            and isinstance(self.zone_info, Path)
        ):
            self.zone_background_dict = {
                "zone:background:property": ["type", "points"],
                "zone:background:value": ["polygon", str(self.zone_info)],
            }
            for isn in df["instance_name"].dropna().unique():
                if isn != "nan" and "object" in isn:
                    self.zone_dict[f"zone:{isn}:property"] = [
                        "type",
                        "center",
                        "radius",
                    ]
                    self.zone_dict[f"zone:{isn}:value"] = ["circle", "0, 0", 0]

    def init_frame(self, frame_timestamp, frame_number):
        self.timestamps.setdefault(frame_timestamp, {})
        self.timestamps[frame_timestamp].setdefault("event:Grooming", 0)
        self.timestamps[frame_timestamp].setdefault("event:Rearing", 0)
        self.timestamps[frame_timestamp].setdefault("event:Object_investigation", 0)
        self.timestamps[frame_timestamp].setdefault("event:RightInteract", 0)
        self.timestamps[frame_timestamp].setdefault("event:LeftInteract", 0)
        self.timestamps[frame_timestamp].setdefault("event:Freezing", 0)
        self.timestamps[frame_timestamp].setdefault("pos:animal_center:x", -1)
        self.timestamps[frame_timestamp].setdefault("pos:animal_center:y", -1)
        self.timestamps[frame_timestamp].setdefault("pos:interact_center:x", -1)
        self.timestamps[frame_timestamp].setdefault("pos:interact_center:y", -1)
        self.timestamps[frame_timestamp].setdefault("pos:animal_nose:x", -1)
        self.timestamps[frame_timestamp].setdefault("pos:animal_nose:y", -1)
        self.timestamps[frame_timestamp].setdefault("pos:animal_:x", -1)
        self.timestamps[frame_timestamp].setdefault("pos:animal_:y", -1)
        self.timestamps[frame_timestamp]["frame_number"] = frame_number

    def record_event(self, frame_timestamp, event_key, value=1):
        self.timestamps[frame_timestamp][event_key] = value

    def record_position(self, frame_timestamp, pos_type, x, y):
        self.timestamps[frame_timestamp][f"pos:{pos_type}:x"] = x
        self.timestamps[frame_timestamp][f"pos:{pos_type}:y"] = y

    def update_zone_object(self, obj_class, cx, cy_glitter, radius):
        self.zone_dict[f"zone:{obj_class}:value"] = [
            "circle",
            [cx, cy_glitter],
            radius,
        ]

    def export(self, out_nix_csv_file):
        df_res = pd.DataFrame.from_dict(self.timestamps, orient="index")
        df_res.index.rename("timestamps", inplace=True)
        df_meta = pd.DataFrame.from_dict(self.metadata_dict, orient="index")

        df_res.reset_index(inplace=True)
        df_meta.reset_index(inplace=True)
        df_meta.columns = ["metadata", "value"]
        df_res.insert(0, "metadata", df_meta["metadata"])
        df_res.insert(1, "value", df_meta["value"])

        if self.zone_background_dict is not None:
            df_zone_background = pd.DataFrame.from_dict(self.zone_background_dict)
            df_res = pd.concat([df_res, df_zone_background], axis=1)

        if self.zone_dict:
            df_zone = pd.DataFrame.from_dict(self.zone_dict)
            df_res = pd.concat([df_res, df_zone], axis=1)

        df_res.to_csv(out_nix_csv_file, index=False)


class TrackingVisualizer:
    """Handles drawing bounding boxes, masks, and overlays uniformly."""

    def __init__(self, out_video_file, target_fps, width, height):
        self.video_writer = cv2.VideoWriter(
            out_video_file, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (width, height)
        )
        self.points = [deque(maxlen=30) for _ in range(1000)]

    def draw_zone_masks(self, frame, height, width, zones):
        zone_masks = []
        right_zone_box = None
        if zones:
            for zs in zones:
                zone_box = zs["points"]
                zone_label = zs["label"]
                zone_box = functools.reduce(operator.iconcat, zone_box, [])
                if "right" in zone_label.lower():
                    right_zone_box = zone_box

                try:
                    rles = mask_util.frPyObjects([zone_box], height, width)
                    rle = mask_util.merge(rles)
                    m = mask_util.decode(rle)
                    frame = draw.draw_binary_masks(frame, [m], [zone_label])
                    zone_masks.append((zone_label, rle))
                except Exception:
                    continue
        return frame, zone_masks, right_zone_box

    def write_frame(self, frame):
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()
        cv2.destroyAllWindows()


def _normalize_label_names(values):
    """Normalize label inputs into a stable list of names.

    Accepts both legacy comma-separated strings and iterable inputs.
    """
    if values is None:
        return []

    if isinstance(values, str):
        tokens = re.split(r"[,\s]+", values)
        return [token for token in tokens if token]

    if isinstance(values, (list, tuple, set)):
        normalized = []
        for value in values:
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                normalized.extend(_normalize_label_names(value_str))
        return normalized

    value_str = str(values).strip()
    return [value_str] if value_str else []


def _extend_unique(base_values, extra_values):
    """Append names from extra_values to base_values while preserving order."""
    seen = {str(value).lower() for value in base_values}
    for value in extra_values:
        key = str(value).lower()
        if key not in seen:
            base_values.append(value)
            seen.add(key)
    return base_values


def _label_in_collection(label, names):
    """Case-insensitive exact label check."""
    if label is None:
        return False
    label_str = str(label)
    if label_str in names:
        return True
    return label_str.lower() in {str(name).lower() for name in names}


def _probe_sample_aspect_ratio(video_file):
    """Return the source sample aspect ratio as a ``num:den`` string."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=sample_aspect_ratio",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    sar = result.stdout.strip()
    if not sar or sar == "N/A":
        return None
    return sar


def _parse_sample_aspect_ratio(sample_aspect_ratio):
    """Parse a SAR string like ``29:18`` and return integer tuple."""
    if not sample_aspect_ratio or sample_aspect_ratio in {"1:1", "N/A"}:
        return None
    try:
        sar_num, sar_den = sample_aspect_ratio.split(":", 1)
        sar_num = int(sar_num)
        sar_den = int(sar_den)
        if sar_num <= 0 or sar_den <= 0:
            return None
        return sar_num, sar_den
    except Exception:
        return None


def _apply_sample_aspect_ratio_to_frame(frame, sample_aspect_ratio):
    """Resize frame for display using SAR and output square pixels."""
    parsed_sar = _parse_sample_aspect_ratio(sample_aspect_ratio)
    if parsed_sar is None:
        return frame

    sar_num, sar_den = parsed_sar
    height, width = frame.shape[:2]
    scaled_width = int(width * sar_num / sar_den)
    # Keep display width even and non-zero for stable rendering.
    scaled_width = max(2, (scaled_width // 2) * 2)
    if scaled_width == width:
        return frame

    return cv2.resize(frame, (scaled_width, height), interpolation=cv2.INTER_LINEAR)


def _finalize_tracked_video(temp_video_file, final_video_file, sample_aspect_ratio):
    """Write a square-pixel tracked video with the source display geometry."""
    temp_video_file = Path(temp_video_file)
    final_video_file = Path(final_video_file)

    if not temp_video_file.exists():
        raise FileNotFoundError(f"Tracked video was not written: {temp_video_file}")

    if not sample_aspect_ratio or sample_aspect_ratio in {"1:1", "N/A"}:
        temp_video_file.replace(final_video_file)
        return final_video_file

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temp_video_file.replace(final_video_file)
        return final_video_file

    parsed_sar = _parse_sample_aspect_ratio(sample_aspect_ratio)
    if parsed_sar is None:
        temp_video_file.replace(final_video_file)
        return final_video_file
    sar_num, sar_den = parsed_sar

    # Bake display aspect into actual pixels so playback stays correct
    # even in players that ignore non-square SAR metadata.
    sar_scale_expr = f"{sar_num}/{sar_den}"
    vf_filter = f"scale=trunc(iw*{sar_scale_expr}/2)*2:ih,setsar=1"

    with tempfile.NamedTemporaryFile(
        suffix=final_video_file.suffix or ".mp4",
        delete=False,
        dir=str(final_video_file.parent),
    ) as tmp_out:
        tmp_out_path = Path(tmp_out.name)

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(temp_video_file),
                "-vf",
                vf_filter,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(tmp_out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        tmp_out_path.replace(final_video_file)
        temp_video_file.unlink(missing_ok=True)
    except Exception:
        if tmp_out_path.exists():
            tmp_out_path.unlink(missing_ok=True)
        temp_video_file.replace(final_video_file)
    return final_video_file


def tracks2nix(
    video_file=None,
    tracking_results="tracking.csv",
    out_nix_csv_file="my_glitter_format.csv",
    zone_info="zone_info.json",
    overlay_mask=True,
    score_threshold=None,
    motion_threshold=None,
    deep=False,
    pretrained_model=None,
    subject_names=None,
    behavior_names=None,
    overlap_threshold=0.00001,
):
    """
    Args:
        video_file (str): video file path. Defaults to None.
        tracking_results (str, optional): the tracking results csv file from a model.
         Defaults to 'tracking.csv'.
        out_nix_csv_file (str, optional): [description]. Defaults to 'my_glitter_format.csv'.
        zone_info ([type], optional): a comma separated string e.g.
           "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0". Defaults to None.
        overlay_mask (bool): Overall mask or not. Defaults to True.
        score_threshold (float): the class score threshold between 0.0 to 1.0 to display the segmentation.
        motion_threshold (float): threshold for motion between frames. defaults 0.
        deep (bool): use deep learning based motion model. defaults to False.
        pretrained_model (str): path to the trained motion model. defaults to None.
        subject_name (str): a list of comma separated subject names like vole_01,vole_02,...
        behavior_names (str): a list of comma separated behavior names like rearing,walking,...
        overlay_threshold (float): overlay threshold between two masks e.g. between animal and a zone default 0.1

    Create a nix format csv file and annotated video
    """
    if score_threshold is None:
        score_threshold = 0.15
    else:
        score_threshold = float(score_threshold)

    print(f"Class or Instance score threshold is: {score_threshold}.")
    print("Please update the definitions of keypoints, instances, and events")
    keypoint_cfg_file = Path(__file__).parent.parent / "configs" / "keypoints.yaml"
    print(
        f"Please check and edit keypoints config file {keypoint_cfg_file} for your projects."
    )

    if zone_info and ".json" in zone_info:
        zone_info = Path(zone_info)
    elif zone_info == "zone_info.json":
        zone_info = Path(__file__).parent / zone_info

    _class_meta_data = draw.get_keypoint_connection_rules()
    (
        keypoints_connection_rules,
        animal_names_raw,
        behaviors_raw,
        zones_names_raw,
    ) = _class_meta_data
    animal_names = _normalize_label_names(animal_names_raw)
    behaviors = _normalize_label_names(behaviors_raw)
    zones_names = _normalize_label_names(zones_names_raw)

    if subject_names is not None:
        animal_names = _extend_unique(
            animal_names, _normalize_label_names(subject_names)
        )
    if behavior_names is not None:
        behaviors = _extend_unique(behaviors, _normalize_label_names(behavior_names))

    _animal_object_list = list(animal_names)
    subject_animal_name = (
        _animal_object_list[0] if len(_animal_object_list) > 0 else "subject_vole"
    )
    left_interact_object = (
        _animal_object_list[1] if len(_animal_object_list) > 1 else "left_vole"
    )
    right_interact_object = (
        _animal_object_list[2] if len(_animal_object_list) > 2 else "right_vole"
    )
    body_parts = (
        [bp for bp in keypoints_connection_rules[0]]
        if keypoints_connection_rules and len(keypoints_connection_rules) > 0
        else []
    )

    df_motion = None
    if motion_threshold is not None and motion_threshold > 0:
        fa = FreezingAnalyzer(
            video_file, tracking_results, motion_threshold=motion_threshold
        )
        if pretrained_model is not None and Path(pretrained_model).exists():
            deep = True
        df_motion = fa.run(deep=deep, pretrained_model=pretrained_model)

    df = pd.read_csv(tracking_results)
    try:
        df = df.drop(columns=["Unnamed: 0"])
    except KeyError:
        print("data frame does not have a column named Unnmaed: 0")

    analyzer = BehaviorAnalyzer(
        df, df_motion, subject_animal_name, left_interact_object, right_interact_object
    )

    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df["instance_name"] = df.apply(
        lambda row: TracksResults.switch_left_right(row, width), axis=1
    )

    instance_names = df["instance_name"].dropna().unique()
    for instance_name in instance_names:
        if (
            not _label_in_collection(instance_name, animal_names)
            and not _label_in_collection(instance_name, behaviors)
            and not _label_in_collection(instance_name, zones_names)
        ):
            animal_names = _extend_unique(animal_names, [instance_name])

    exporter = GlitterNixExporter(video_file, width, height, zone_info, df)

    if (
        zone_info
        and isinstance(zone_info, Path)
        and zone_info.exists()
        and zone_info.suffix == ".json"
    ):
        zone_file = json.loads(zone_info.read_bytes())
        zones = zone_file.get("shapes", None)
    else:
        zones = None

    num_grooming = 0
    num_rearing = 0
    num_object_investigation = 0
    num_left_interact = 0
    num_right_interact = 0

    out_video_file = f"{os.path.splitext(video_file)[0]}_tracked.mp4"
    temp_out_video_file = f"{os.path.splitext(out_video_file)[0]}_opencv_tmp.mp4"
    source_sample_aspect_ratio = _probe_sample_aspect_ratio(video_file)
    visualizer = TrackingVisualizer(temp_out_video_file, target_fps, width, height)

    for frame_number, frame in enumerate(frame_from_video(cap, num_frames)):
        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        bbox_info = analyzer.get_bbox(frame_number)
        left_interact, right_interact = analyzer.left_right_interact(
            frame_number,
            subject_animal_name,
            left_interact_object,
            right_interact_object,
        )

        exporter.init_frame(frame_timestamp, frame_number)

        frame, zone_masks, right_zone_box = visualizer.draw_zone_masks(
            frame, height, width, zones
        )
        parts_locations = {}

        for bf in bbox_info:
            _frame_num = bf["frame_number"]
            x1 = bf["x1"]
            y1 = bf["y1"]
            x2 = bf["x2"]
            y2 = bf["y2"]
            _class = bf["instance_name"]
            score = bf["class_score"]
            _mask = bf.get("segmentation", None)

            try:
                tracking_id = int(bf.get("tracking_id", ""))
            except Exception:
                tracking_id = ""

            if isinstance(_frame_num, tuple):
                _frame_num = _frame_num[0]
                x1 = x1[0]
                y1 = y1[0]
                x2 = x2[0]
                y2 = y2[0]
                _class = _class[0]
                score = score[0]

            bbox = [[x1, y1, x2, y2]]

            if not pd.isnull(_mask) and overlay_mask:
                if not _label_in_collection(_class, zones_names):
                    if score >= score_threshold and (
                        _label_in_collection(_class, animal_names)
                    ):
                        _mask_parsed = ast.literal_eval(_mask)
                        for zm in zone_masks:
                            if analyzer.animal_in_zone(
                                _mask_parsed, zm[1], overlap_threshold
                            ):
                                event = f"{_class}_in_{zm[0]}"
                                exporter.record_event(
                                    frame_timestamp, f"event:{event}", 1
                                )
                                draw.draw_boxes(
                                    frame,
                                    bbox,
                                    identities=[event],
                                    draw_track=False,
                                    points=visualizer.points,
                                )

                        _mask_decoded = mask_util.decode(_mask_parsed)[:, :]
                        frame = draw.draw_binary_masks(
                            frame, [_mask_decoded], [f"{_class}-{tracking_id}"]
                        )

            glitter_y1 = height - y1
            glitter_y2 = height - y2

            if "right" in str(_class).lower() and "interact" in _class.lower():
                _class = "RightInteract"
            elif "left" in str(_class).lower() and "interact" in _class.lower():
                _class = "LeftInteract"

            is_draw = True
            if _class == "RightInteract" and (
                right_zone_box is not None and x1 < right_zone_box[0]
            ):
                is_draw = False

            if not math.isnan(x1) and _frame_num == frame_number:
                cx = int((x1 + x2) / 2) if "cx" not in bf else int(bf["cx"])
                cy_glitter = int((glitter_y1 + glitter_y2) / 2)
                cy = int((y1 + y2) / 2) if "cy" not in bf else int(bf["cy"])
                _, color = draw.get_label_color(_class)

                if analyzer.keypoint_in_body_mask(
                    _frame_num, _class, subject_animal_name
                ):
                    parts_locations[_class] = (cx, cy, color)

                if _class == "nose" or "nose" in _class.lower():
                    exporter.record_position(
                        frame_timestamp, "animal_nose", cx, cy_glitter
                    )
                elif (
                    _class == "centroid"
                    or _class.lower().endswith("mouse")
                    or _class.lower().endswith("vole")
                ):
                    exporter.record_position(
                        frame_timestamp, "animal_center", cx, cy_glitter
                    )
                elif _class == "grooming":
                    exporter.record_event(frame_timestamp, "event:Grooming", 1)
                    exporter.record_position(frame_timestamp, "animal_", cx, cy_glitter)
                    num_grooming += 1
                elif "rearing" in _class:
                    exporter.record_event(frame_timestamp, "event:Rearing", 1)
                    exporter.record_position(frame_timestamp, "animal_", cx, cy_glitter)
                    num_rearing += 1
                elif _class == "object_investigation":
                    exporter.record_event(
                        frame_timestamp, "event:Object_investigation", 1
                    )
                    exporter.record_position(frame_timestamp, "animal_", cx, cy_glitter)
                    num_object_investigation += 1
                elif _class == "LeftInteract" and left_interact > 0:
                    exporter.record_position(
                        frame_timestamp, "interact_center", cx, cy_glitter
                    )
                    if cx > width / 2:
                        exporter.record_event(frame_timestamp, "event:RightInteract", 1)
                        num_right_interact += 1
                        _class = "RightInteract"
                    else:
                        exporter.record_event(frame_timestamp, "event:LeftInteract", 1)
                        num_left_interact += 1
                elif (
                    is_draw
                    and _class == "RightInteract"
                    and score >= score_threshold
                    and right_interact > 0
                ):
                    exporter.record_event(frame_timestamp, "event:RightInteract", 1)
                    exporter.record_position(
                        frame_timestamp, "interact_center_", cx, cy_glitter
                    )
                    num_right_interact += 1
                elif "object" in _class.lower() and _class != "object_investigation":
                    exporter.update_zone_object(
                        _class,
                        cx,
                        cy_glitter,
                        min(int((x2 - x1) / 2), int(glitter_y2 - glitter_y1)),
                    )

                if (
                    is_draw
                    and _label_in_collection(_class, behaviors)
                    and score >= score_threshold
                    and _class not in body_parts
                    and not _label_in_collection(_class, animal_names)
                ):
                    if _class == "grooming":
                        label = f"{_class}: {num_grooming} times"
                    elif _class == "rearing":
                        label = f"{_class}: {num_rearing} times"
                    elif _class == "object_investigation":
                        label = f"{_class}: {num_object_investigation} times"
                    elif _class == "LeftInteract":
                        label = f"{_class}: {num_left_interact} times"
                    elif _class == "RightInteract":
                        label = f"{_class}: {num_right_interact} times"
                    elif "rearing" in _class:
                        label = "rearing"
                    else:
                        label = f"{_class}-{tracking_id}-{round(score * 100, 2)}%"

                    if (_class == "RightInteract" and right_interact <= 0) or (
                        _class == "LeftInteract" and left_interact <= 0
                    ):
                        pass
                    else:
                        draw.draw_boxes(
                            frame,
                            bbox,
                            identities=[label],
                            draw_track=False,
                            points=visualizer.points,
                        )
                elif score >= score_threshold:
                    is_keypoint_in_mask = analyzer.keypoint_in_body_mask(
                        _frame_num, _class, subject_animal_name
                    )
                    if (
                        is_keypoint_in_mask
                        or any(map(str.isdigit, _class))
                        or _label_in_collection(_class, _animal_object_list)
                    ):
                        if "zone" not in _class.lower():
                            cv2.circle(frame, (cx, cy), 6, color, -1)
                        if (
                            _label_in_collection(_class, animal_names)
                            and "zone" not in _class.lower()
                        ):
                            mask_label = (
                                f"-{_class}{tracking_id if tracking_id != 0 else ''}"
                                if len(bbox_info) < 10
                                else f"-{_class.split('_')[-1]}"
                            )
                            cv2.putText(
                                frame,
                                mask_label,
                                (cx + 3, cy + 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                color,
                                2,
                            )

                if (
                    left_interact > 0
                    and "left" in _class.lower()
                    and "interact" in _class.lower()
                ):
                    num_left_interact += 1
                    exporter.record_event(frame_timestamp, "event:LeftInteract", 1)
                    exporter.record_position(
                        frame_timestamp, "interact_center_", cx, cy_glitter
                    )
                    label = f"left interact:{num_left_interact} times"
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=visualizer.points,
                    )

                if (
                    right_interact > 0
                    and "right" in _class.lower()
                    and "interact" in _class.lower()
                ):
                    num_right_interact += 1
                    exporter.record_event(frame_timestamp, "event:RightInteract", 1)
                    exporter.record_position(
                        frame_timestamp, "interact_center_", cx, cy_glitter
                    )
                    label = f"right interact:{num_right_interact} times"
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=visualizer.points,
                    )

                freezing = analyzer.is_freezing(_frame_num, _class)
                if freezing:
                    exporter.record_event(frame_timestamp, "event:Freezing", 1)
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=["freezing"],
                        draw_track=False,
                        points=visualizer.points,
                    )

                if _label_in_collection(_class, behaviors) and _class not in body_parts:
                    cv2.rectangle(frame, (5, 35), (5 + 140, 35 + 35), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"{_class}",
                        (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                if _label_in_collection(_class, zones_names):
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[_class],
                        draw_track=False,
                        points=visualizer.points,
                    )

        draw.draw_keypoint_connections(
            frame, parts_locations, keypoints_connection_rules
        )
        cv2.putText(
            frame,
            f"Timestamp: {frame_timestamp}",
            (25, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        display_frame = _apply_sample_aspect_ratio_to_frame(
            frame, source_sample_aspect_ratio
        )
        cv2.imshow("Frame", display_frame)
        visualizer.write_frame(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    visualizer.release()
    cap.release()
    _finalize_tracked_video(
        temp_out_video_file, out_video_file, source_sample_aspect_ratio
    )
    exporter.export(out_nix_csv_file)
