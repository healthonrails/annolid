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
from collections import deque
import pycocotools.mask as mask_util
points = [deque(maxlen=30) for _ in range(1000)]


def tracks2nix(vidoe_file=None,
               tracking_results='tracking.csv',
               out_nix_csv_file='my_glitter_format.csv',
               zone_info='zone_info.json',
               overlay_mask=True,
               score_threshold=0.6
               ):
    """
    Args:
        vidoe_file (str): video file path. Defaults to None.
        tracking_results (str, optional): the tracking results csv file froma a model.
         Defaults to 'tracking.csv'.
        out_nix_csv_file (str, optional): [description]. Defaults to 'my_glitter_format.csv'.
        zone_info ([type], optional): a comma seperated string e.g.
           "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0". Defaults to None.

    Create a nix format csv file and annotated video
    """

    print(f"Class or Instance score threshold is: {score_threshold}.")

    if zone_info and '.json' in zone_info:
        zone_info = Path(zone_info)
    elif zone_info == 'zone_info.json':
        zone_info = Path(__file__).parent / zone_info

    df = pd.read_csv(tracking_results)
    try:
        df = df.drop(columns=['Unnamed: 0'])
    except KeyError:
        return

    def get_bbox(frame_number):
        _df = df[df.frame_number == frame_number]
        try:
            res = tuple(_df.values)
        except:
            res = []
        return res

    def left_right_interact(fn):
        _df_top = df[df.frame_number == fn]
        right_interact = None
        left_interact = None
        try:
            subject_vole_seg = _df_top[_df_top.instance_name ==
                                       'subject_vole']['segmentation'].values[0]
            subject_vole_seg = ast.literal_eval(subject_vole_seg)
        except IndexError:
            return 0.0, 0.0
        try:
            left_vole_seg = _df_top[_df_top.instance_name ==
                                    'left_vole']['segmentation'].values[0]
            left_vole_seg = ast.literal_eval(left_vole_seg)
            left_interact = mask_util.iou([left_vole_seg], [subject_vole_seg], [
                False, False]).flatten()[0]
        except IndexError:
            left_interact = 0.0
        try:
            right_vole_seg = _df_top[_df_top.instance_name ==
                                     'right_vole']['segmentation'].values[0]
            right_vole_seg = ast.literal_eval(right_vole_seg)
            right_interact = mask_util.iou([right_vole_seg], [subject_vole_seg], [
                False, False]).flatten()[0]
        except IndexError:
            right_interact = 0.0

        return left_interact, right_interact

    cap = cv2.VideoCapture(vidoe_file)
    ret, frame = cap.read()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_fps = int(cap.get(cv2.CAP_PROP_FPS))

    metadata_dict = {}
    metadata_dict['filename'] = vidoe_file
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

    out_video_file = f"{os.path.splitext(vidoe_file)[0]}_tracked.mp4"

    video_writer = cv2.VideoWriter(out_video_file,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   target_fps,
                                   (width, height))

    while ret:
        ret, frame = cap.read()

        if not ret:
            break
        # timestamp in seconds
        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        bbox_info = get_bbox(frame_number)

        # calculate left or rgiht interact in the frame
        left_interact, right_interact = left_right_interact(
            frame_number
        )

        timestamps.setdefault(frame_timestamp, {})
        timestamps[frame_timestamp].setdefault('event:Grooming', 0)
        timestamps[frame_timestamp].setdefault('event:Rearing', 0)
        timestamps[frame_timestamp].setdefault('event:Object_investigation', 0)
        timestamps[frame_timestamp].setdefault('event:RightInteract', 0)
        timestamps[frame_timestamp].setdefault('event:LeftInteract', 0)
        timestamps[frame_timestamp].setdefault('pos:animal_center:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_center:y', -1)

        timestamps[frame_timestamp].setdefault('pos:interact_center:x', -1)
        timestamps[frame_timestamp].setdefault('pos:interact_center:y', -1)

        timestamps[frame_timestamp].setdefault('pos:animal_nose:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_nose:y', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_:y', -1)

        right_zone_box = None
        left_zone_box = None

        if zones:
            for zs in zones:
                zone_box = zs['points']
                zone_label = zs['label']
                zone_box = functools.reduce(operator.iconcat, zone_box, [])
                if zone_label == 'right_zone':
                    right_zone_box = zone_box
                elif zone_label == 'left_zone':
                    left_zone_box = zone_box
                draw.draw_boxes(
                    frame,
                    [zone_box],
                    identities=[zone_label],
                    draw_track=False,
                )

        for bf in bbox_info:
            if len(bf) >= 8:
                _frame_num, x1, y1, x2, y2, _class, score, _mask = bf
                if not pd.isnull(_mask) and overlay_mask:
                    _mask = ast.literal_eval(_mask)
                    _mask = mask_util.decode(_mask)[:, :]
                    frame = draw.draw_binary_masks(frame, [_mask], [_class])
            else:
                _frame_num, x1, y1, x2, y2, _class, score = bf

            # In glitter, the y-axis is such that the bottom is zero and the top is height.
            # i.e. origin is bottom left
            glitter_y1 = height - y1
            glitter_y2 = height - y2

            if 'right' in _class.lower() and 'interact' in _class.lower():
                _class = 'RightInteract'
            elif 'left' in _class.lower() and 'interact' in _class.lower():
                _class = 'LeftInteract'

            is_draw = True
            if _class == "RightInteract" and (right_zone_box is not None
                                              and x1 < right_zone_box[0]):
                is_draw = False

            if not math.isnan(x1) and _frame_num == frame_number:
                cx = int((x1 + x2) / 2)
                cy_glitter = int((glitter_y1 + glitter_y2) / 2)
                cy = int((y1 + y2) / 2)
                color = draw.compute_color_for_labels(
                    hash(_class) % 100)
                if _class == 'nose' or 'nose' in _class.lower():
                    timestamps[frame_timestamp]['pos:animal_nose:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_nose:y'] = cy_glitter
                elif _class == 'centroid' or _class == 'Mouse':
                    timestamps[frame_timestamp]['pos:animal_center:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_center:y'] = cy_glitter
                elif _class == 'grooming':
                    timestamps[frame_timestamp]['event:Grooming'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy_glitter
                    num_grooming += 1
                elif _class == 'rearing':
                    timestamps[frame_timestamp]['event:Rearing'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy_glitter
                    num_rearing += 1
                elif _class == 'object_investigation':
                    timestamps[frame_timestamp]['event:Object_investigation'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy_glitter
                    num_object_investigation += 1
                elif _class == 'LeftInteract':

                    timestamps[frame_timestamp]['pos:interact_center:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center:y'] = cy_glitter

                    if cx > width / 2:
                        timestamps[frame_timestamp]['event:RightInteract'] = 1
                        num_right_interact += 1
                        _class = "RightInteract"
                    else:
                        timestamps[frame_timestamp]['event:LeftInteract'] = 1
                        num_left_interact += 1
                elif is_draw and _class == 'RightInteract' and score >= score_threshold:
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

                if is_draw and _class in ['grooming', 'rearing',
                                          'object_investigation',
                                          'LeftInteract', 'RightInteract'] and score >= score_threshold:

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

                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=points
                    )
                elif _class == 'mouse':
                    pass
                else:
                    cv2.circle(frame, (cx, cy),
                               8,
                               color,
                               -1)

                if left_interact > 0 and 'left' in _class.lower():
                    num_left_interact += 1
                    timestamps[frame_timestamp]['event:LeftInteract'] = 1
                    timestamps[frame_timestamp]['pos:interact_center_:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center_:y'] = cy_glitter
                    label = f"interact:{num_left_interact} times"
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=points
                    )
                if right_interact > 0 and 'right' in _class:
                    num_right_interact += 1
                    timestamps[frame_timestamp]['event:RightInteract'] = 1
                    timestamps[frame_timestamp]['pos:interact_center_:x'] = cx
                    timestamps[frame_timestamp]['pos:interact_center_:y'] = cy_glitter
                    label = f"interact:{num_right_interact} times"
                    draw.draw_boxes(
                        frame,
                        bbox,
                        identities=[label],
                        draw_track=False,
                        points=points
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
