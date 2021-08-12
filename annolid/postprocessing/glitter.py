import os
import cv2
import pandas as pd
import numpy as np
import math
import ast
import json
import functools
import operator
from pathlib import Path
from annolid.utils import draw
from annolid.data.videos import frame_from_video
from collections import deque
import pycocotools.mask as mask_util
from annolid.postprocessing.freezing_analyzer import FreezingAnalyzer
from annolid.postprocessing.quality_control import TracksResults

points = [deque(maxlen=30) for _ in range(1000)]


def tracks2nix(video_file=None,
               tracking_results='tracking.csv',
               out_nix_csv_file='my_glitter_format.csv',
               zone_info='zone_info.json',
               overlay_mask=True,
               score_threshold=None,
               motion_threshold=None,
               deep=False,
               pretrained_model=None
               ):
    """
    Args:
        video_file (str): video file path. Defaults to None.
        tracking_results (str, optional): the tracking results csv file froma a model.
         Defaults to 'tracking.csv'.
        out_nix_csv_file (str, optional): [description]. Defaults to 'my_glitter_format.csv'.
        zone_info ([type], optional): a comma seperated string e.g.
           "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0". Defaults to None.
        overlay_mask (bool): Overal mask or not. Defaults to True. 
        score_threshold (float): the class score threshold between 0.0 to 1.0 to display the segmentation. 
        motion_threshold (float): threshold for motion between frames. defaults 0. 
        deep (bool): use deep learning based motion model. defaults to False.
        pretrained_model (str): path to the trained motion model. defaults to None.  

    Create a nix format csv file and annotated video
    """

    print(f"Class or Instance score threshold is: {score_threshold}.")

    if zone_info and '.json' in zone_info:
        zone_info = Path(zone_info)
    elif zone_info == 'zone_info.json':
        zone_info = Path(__file__).parent / zone_info

    keypoints_connection_rules, animal_names, behaviors = draw.get_keypoint_connection_rules()

    _animal_object_list = animal_names.split()
    subject_animal_name = _animal_object_list[0]
    left_interact_object = _animal_object_list[1]
    right_interact_object = _animal_object_list[2]
    body_parts = [bp for bp in keypoints_connection_rules[0]]

    df_motion = None
    if motion_threshold > 0:
        fa = FreezingAnalyzer(video_file,
                              tracking_results,
                              motion_threshold=motion_threshold)
        if pretrained_model is not None and Path(pretrained_model).exists():
            deep = True
        df_motion = fa.run(deep=deep,
                           pretrained_model=pretrained_model)

    df = pd.read_csv(tracking_results)
    try:
        df = df.drop(columns=['Unnamed: 0'])
    except KeyError:
        return

    def get_bbox(frame_number):
        _df = df[df.frame_number == frame_number]
        try:
            res = _df.to_dict(orient='records')
        except:
            res = []
        return res

    def is_freezing(frame_number, instance_name):
        if df_motion is not None:
            freezing = df_motion[(df_motion.frame_number == frame_number) & (
                df_motion.instance_name == instance_name)].freezing.values[0]
            return freezing > 0
        else:
            return False

    def keypoint_in_body_mask(
            frame_number,
            keypoint_name,
            animal_name=None):

        if animal_name is None:
            animal_name = subject_animal_name

        _df_k_b = df[df.frame_number == frame_number]
        try:
            body_seg = _df_k_b[_df_k_b.instance_name ==
                               animal_name]['segmentation'].values[0]
            body_seg = ast.literal_eval(body_seg)
        except IndexError:
            return False

        try:
            keypoint_seg = _df_k_b[_df_k_b.instance_name ==
                                   keypoint_name]['segmentation'].values[0]
            keypoint_seg = ast.literal_eval(keypoint_seg)
        except IndexError:
            return False

        if keypoint_seg and body_seg:
            overlap = mask_util.iou([body_seg], [keypoint_seg], [
                False, False]).flatten()[0]
            return overlap > 0
        else:
            return False

    def left_right_interact(fn,
                            subject_instance='subject_vole',
                            left_instance='left_vole',
                            right_instance='right_vole'
                            ):
        _df_top = df[df.frame_number == fn]
        right_interact = None
        left_interact = None
        try:
            subject_instance_seg = _df_top[_df_top.instance_name ==
                                           subject_instance]['segmentation'].values[0]
            subject_instance_seg = ast.literal_eval(subject_instance_seg)
        except IndexError:
            return 0.0, 0.0
        try:
            left_instance_seg = _df_top[_df_top.instance_name ==
                                        left_instance]['segmentation'].values[0]
            left_instance_seg = ast.literal_eval(left_instance_seg)
            left_interact = mask_util.iou([left_instance_seg], [subject_instance_seg], [
                False, False]).flatten()[0]
        except IndexError:
            left_interact = 0.0
        try:
            right_instance_seg = _df_top[_df_top.instance_name ==
                                         right_instance]['segmentation'].values[0]
            right_instance_seg = ast.literal_eval(right_instance_seg)
            right_interact = mask_util.iou([right_instance_seg], [subject_instance_seg], [
                False, False]).flatten()[0]
        except IndexError:
            right_interact = 0.0

        return left_interact, right_interact

    cap = cv2.VideoCapture(video_file)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fix left right swtich by checking the middle
    # point of the video frame width
    df['instance_name'] = df.apply(lambda row:
                                   TracksResults.switch_left_right(
                                       row,
                                       width
                                   ),
                                   axis=1
                                   )

    metadata_dict = {}
    metadata_dict['filename'] = video_file
    metadata_dict['pixels_per_meter'] = 0
    metadata_dict['video_width'] = f"{width}"
    metadata_dict['video_height'] = f"{height}"
    metadata_dict['saw_all_timestamps'] = 'TRUE'

    zone_dict = {}

    if zone_info is not None and zone_info.suffix != '.json':
        zone_background_dict = {}

        zone_background_dict['zone:background:property'] = ['type', 'points']

        zone_background_dict['zone:background:value'] = [
            'polygon',
            zone_info
        ]

        for isn in df['instance_name'].dropna().unique():

            if isn != 'nan' and 'object' in isn:
                zone_dict[f"zone:{isn}:property"] = [
                    'type',
                    'center',
                    'radius'
                ]
                zone_dict[f'zone:{isn}:value'] = [
                    'circle',
                    "0, 0",
                    0
                ]
    elif zone_info and zone_info.exists():
        zone_file = json.loads(
            zone_info.read_bytes())
        zones = zone_file['shapes']
    else:
        zones = None

    timestamps = {}

    num_grooming = 0
    num_rearing = 0
    num_object_investigation = 0
    num_left_interact = 0
    num_right_interact = 0

    out_video_file = f"{os.path.splitext(video_file)[0]}_tracked.mp4"

    video_writer = cv2.VideoWriter(out_video_file,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   target_fps,
                                   (width, height))

    # try mutlple times if opencv cannot read a frame
    for frame_number, frame in enumerate(frame_from_video(cap, num_frames)):
        # timestamp in seconds
        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        bbox_info = get_bbox(frame_number)

        # calculate left or rgiht interact in the frame
        left_interact, right_interact = left_right_interact(
            frame_number,
            subject_animal_name,
            left_interact_object,
            right_interact_object
        )

        timestamps.setdefault(frame_timestamp, {})
        timestamps[frame_timestamp].setdefault('event:Grooming', 0)
        timestamps[frame_timestamp].setdefault('event:Rearing', 0)
        timestamps[frame_timestamp].setdefault('event:Object_investigation', 0)
        timestamps[frame_timestamp].setdefault('event:RightInteract', 0)
        timestamps[frame_timestamp].setdefault('event:LeftInteract', 0)
        timestamps[frame_timestamp].setdefault('event:Freezing', 0)

        timestamps[frame_timestamp].setdefault('pos:animal_center:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_center:y', -1)

        timestamps[frame_timestamp].setdefault('pos:interact_center:x', -1)
        timestamps[frame_timestamp].setdefault('pos:interact_center:y', -1)

        timestamps[frame_timestamp].setdefault('pos:animal_nose:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_nose:y', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_:y', -1)
        timestamps[frame_timestamp].setdefault('frame_number', frame_number)

        right_zone_box = None
        left_zone_box = None

        if zones:
            for zs in zones:
                zone_box = zs['points']
                zone_label = zs['label']
                zone_box = functools.reduce(operator.iconcat, zone_box, [])
                if 'right' in zone_label.lower():
                    right_zone_box = zone_box
                elif 'left' in zone_label.lower():
                    left_zone_box = zone_box

                # draw masks labeled as zones
                # encode and merge polygons with format [[x1,y1,x2,y2,x3,y3....]]
                try:
                    rles = mask_util.frPyObjects([zone_box], height, width)
                    rle = mask_util.merge(rles)

                    # convert the polygons to mask
                    m = mask_util.decode(rle)
                    frame = draw.draw_binary_masks(
                        frame, [m], [zone_label])
                except:
                    # skip non polygon zones
                    continue

        parts_locations = {}

        timestamps[frame_timestamp]['frame_number'] = frame_number

        for bf in bbox_info:
            _frame_num = bf['frame_number'],
            x1 = bf['x1'],
            y1 = bf['y1'],
            x2 = bf['x2'],
            y2 = bf['y2'],
            _class = bf['instance_name'],
            score = bf['class_score'],
            _mask = bf['segmentation']
            if isinstance(_frame_num, tuple):
                _frame_num = _frame_num[0]
                x1 = x1[0]
                y1 = y1[0]
                x2 = x2[0]
                y2 = y2[0]
                _class = _class[0]
                score = score[0]

            if not pd.isnull(_mask) and overlay_mask:
                if score >= score_threshold and (_class in animal_names
                                                 or _class.lower() in animal_names):
                    _mask = ast.literal_eval(_mask)
                    mask_area = mask_util.area(_mask)
                    _mask = mask_util.decode(_mask)[:, :]
                    frame = draw.draw_binary_masks(
                        frame, [_mask], [_class])

            # In glitter, the y-axis is such that the bottom is zero and the top is height.
            # i.e. origin is bottom left
            glitter_y1 = height - y1
            glitter_y2 = height - y2

            if 'right' in str(_class).lower() and 'interact' in _class.lower():
                _class = 'RightInteract'
            elif 'left' in str(_class).lower() and 'interact' in _class.lower():
                _class = 'LeftInteract'

            is_draw = True
            if _class == "RightInteract" and (right_zone_box is not None
                                              and x1 < right_zone_box[0]):
                is_draw = False

            # draw bbox if model predicted with interact and their masks overlaps
            is_draw = is_draw  # and (left_interact > 0 or right_interact > 0)

            if not math.isnan(x1) and _frame_num == frame_number:
                cx = int((x1 + x2) / 2)
                cy_glitter = int((glitter_y1 + glitter_y2) / 2)
                cy = int((y1 + y2) / 2)
                _, color = draw.get_label_color(
                    _class)

                # the first animal
                if keypoint_in_body_mask(_frame_num, _class, subject_animal_name):
                    parts_locations[_class] = (cx, cy, color)

                if _class == 'nose' or 'nose' in _class.lower():
                    timestamps[frame_timestamp]['pos:animal_nose:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_nose:y'] = cy_glitter
                elif _class == 'centroid' or _class.lower().endswith('mouse') \
                        or _class.lower().endswith('vole'):
                    timestamps[frame_timestamp]['pos:animal_center:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_center:y'] = cy_glitter
                elif _class == 'grooming':
                    timestamps[frame_timestamp]['event:Grooming'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy_glitter
                    num_grooming += 1
                elif 'rearing' in _class:
                    timestamps[frame_timestamp]['event:Rearing'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy_glitter
                    num_rearing += 1
                elif _class == 'object_investigation':
                    timestamps[frame_timestamp]['event:Object_investigation'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy_glitter
                    num_object_investigation += 1
                elif _class == 'LeftInteract' and left_interact > 0:

                    timestamps[frame_timestamp]['pos:interact_center:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center:y'] = cy_glitter

                    if cx > width / 2:
                        timestamps[frame_timestamp]['event:RightInteract'] = 1
                        num_right_interact += 1
                        _class = "RightInteract"
                    else:
                        timestamps[frame_timestamp]['event:LeftInteract'] = 1
                        num_left_interact += 1
                elif (is_draw and _class == 'RightInteract'
                      and score >= score_threshold
                      and right_interact > 0):
                    timestamps[frame_timestamp]['event:RightInteract'] = 1
                    timestamps[frame_timestamp]['pos:interact_center_:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center_:y'] = cy_glitter
                    num_right_interact += 1

                elif 'object' in _class.lower() and _class != 'object_investigation':
                    zone_dict[f'zone:{_class}:value'] = [
                        'circle',
                        [cx, cy_glitter],
                        min(int((x2-x1)/2), int(glitter_y2-glitter_y1))
                    ]
                bbox = [[x1, y1, x2, y2]]

                # only draw behavior with bbox not body parts
                if (is_draw and _class in behaviors and
                    score >= score_threshold
                    and _class not in body_parts
                    and _class not in animal_names
                    ):

                    if _class == 'grooming':
                        label = f"{_class}: {num_grooming} times"
                    elif _class == 'rearing':
                        label = f"{_class}: {num_rearing} times"
                    elif _class == "object_investigation":
                        label = f"{_class}: {num_object_investigation} times"
                    elif _class == "LeftInteract":
                        label = f"{_class}: {num_left_interact} times"
                    elif _class == "RightInteract":
                        label = f"{_class}: {num_right_interact} times"
                    elif "rearing" in _class:
                        label = 'rearing'
                    else:
                        label = f"{_class}:{round(score * 100,2)}%"

                    if _class == 'RightInteract' and right_interact <= 0:
                        pass
                    elif _class == 'LeftInteract' and left_interact <= 0:
                        pass
                    else:
                        draw.draw_boxes(
                            frame,
                            bbox,
                            identities=[label],
                            draw_track=False,
                            points=points
                        )
                elif score >= score_threshold:
                    # draw box center as keypoints
                    # do not draw point center for zones
                    is_keypoint_in_mask = keypoint_in_body_mask(
                        _frame_num, _class, subject_animal_name)
                    if (is_keypoint_in_mask
                        or any(map(str.isdigit, _class))
                        or _class in _animal_object_list
                        ):
                        if 'zone' not in _class.lower():
                            cv2.circle(frame, (cx, cy),
                                       6,
                                       color,
                                       -1)
                        if _class in animal_names and 'zone' not in _class.lower():
                            cv2.putText(frame, f"-{_class}:{score*100:.2f}%",
                                        (cx+3, cy+3), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, color, 2)

                if (left_interact > 0
                        and 'left' in _class.lower()
                        and 'interact' in _class.lower()
                        ):
                    num_left_interact += 1
                    timestamps[frame_timestamp]['event:LeftInteract'] = 1
                    timestamps[frame_timestamp]['pos:interact_center_:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center_:y'] = cy_glitter
                    label = f"left interact:{num_left_interact} times"
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=points
                    )
                if (right_interact > 0
                        and 'right' in _class.lower() and
                        'interact' in _class.lower()
                    ):
                    num_right_interact += 1
                    timestamps[frame_timestamp]['event:RightInteract'] = 1
                    timestamps[frame_timestamp]['pos:interact_center_:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center_:y'] = cy_glitter
                    label = f"right interact:{num_right_interact} times"
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=points
                    )

                freezing = is_freezing(_frame_num, _class)
                if freezing:
                    timestamps[frame_timestamp]['event:Freezing'] = 1
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=['freezing'],
                        draw_track=False,
                        points=points
                    )

        # draw the lines between predefined keypoints
        draw.draw_keypoint_connections(frame,
                                       parts_locations,
                                       keypoints_connection_rules
                                       )

        cv2.putText(frame, f"Timestamp: {frame_timestamp}",
                    (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)
        video_writer.write(frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    video_writer.release()

    # save a NIX format CSV file for Glitter2
    df_res = pd.DataFrame.from_dict(timestamps,
                                    orient='index')
    df_res.index.rename('timestamps', inplace=True)

    df_meta = pd.DataFrame.from_dict(metadata_dict,
                                     orient='index'
                                     )

    if zone_info is not None and zone_info.suffix != '.json':
        df_zone_background = pd.DataFrame.from_dict(
            zone_background_dict
        )

    if zone_dict:
        df_zone = pd.DataFrame.from_dict(
            zone_dict
        )

    df_res.reset_index(inplace=True)
    df_meta.reset_index(inplace=True)
    df_meta.columns = ['metadata', 'value']
    df_res.insert(0, "metadata", df_meta['metadata'])
    df_res.insert(1, "value", df_meta['value'])

    if zone_info is not None and zone_info.suffix != '.json':
        df_res = pd.concat([df_res, df_zone_background], axis=1)

    if zone_dict:
        df_res = pd.concat([df_res, df_zone], axis=1)

    df_res.to_csv(out_nix_csv_file, index=False)
