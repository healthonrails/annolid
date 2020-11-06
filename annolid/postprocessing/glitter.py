import os
import cv2
import pandas as pd
import math
from annolid.utils import draw
from collections import deque

points = [deque(maxlen=30) for _ in range(1000)]


def tracks2nix(vidoe_file=None,
               tracking_results='tracking.csv',
               out_nix_csv_file='my_glitter_format.csv'
               ):

    df = pd.read_csv(tracking_results)
    df = df.drop(columns=['Unnamed: 0'])

    def get_bbox(frame_number):
        _df = df[df.frame_number == frame_number]
        try:
            res = tuple(_df.values)
        except:
            res = []
        return res

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

    zone_background_dict = {}
    zone_dict = {}
    zone_background_dict['zone:background:property'] = ['type', 'points']

    zone_background_dict['zone:background:value'] = [
        'polygon',
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
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
                "-1, -1",
                -1
            ]

    timestamps = {}

    num_grooming = 0
    num_rearing = 0
    num_object_investigation = 0

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
        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) // 1000
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        bbox_info = get_bbox(frame_number)

        timestamps.setdefault(frame_timestamp, {})
        timestamps[frame_timestamp].setdefault('event:Grooming', 0)
        timestamps[frame_timestamp].setdefault('event:Rearing', 0)
        timestamps[frame_timestamp].setdefault('event:Object_investigation', 0)
        timestamps[frame_timestamp].setdefault('pos:animal_center:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_center:y', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_nose:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_nose:y', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_:x', -1)
        timestamps[frame_timestamp].setdefault('pos:animal_:y', -1)
        for bf in bbox_info:
            _frame_num, x1, y1, x2, y2, _class, score = bf
            if not math.isnan(x1) and _frame_num == frame_number:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                color = draw.compute_color_for_labels(
                    hash(_class) % 100)

                if _class == 'nose':
                    timestamps[frame_timestamp]['pos:animal_nose:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_nose:y'] = cy
                elif _class == 'centroid':
                    timestamps[frame_timestamp]['pos:animal_center:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_center:y'] = cy
                elif _class == 'grooming':
                    timestamps[frame_timestamp]['event:Grooming'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy
                    num_grooming += 1
                elif _class == 'rearing':
                    timestamps[frame_timestamp]['event:Rearing'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy
                    num_rearing += 1
                elif _class == 'object_investigation':
                    timestamps[frame_timestamp]['event:Object_investigation'] = 1
                    timestamps[frame_timestamp]['pos:animal_:x'] = cx
                    timestamps[frame_timestamp]['pos:animal_:y'] = cy
                    num_object_investigation += 1
                elif 'object' in _class.lower() and _class != 'object_investigation':
                    zone_dict[f'zone:{_class}:value'] = [
                        'circle',
                        [cx, cy],
                        min(int((x2-x1)/2), int(y2-y1))
                    ]

                if _class in ['grooming', 'rearing', 'object_investigation']:
                    bbox = [[x1, y1, x2, y2]]
                    if _class == 'grooming':
                        label = f"{_class}: {num_grooming} times"
                    elif _class == 'rearing':
                        label = f"{_class}: {num_rearing} times"
                    elif _class == "object_investigation":
                        label = f"{_class}: {num_object_investigation} times"
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
                               10,
                               color,
                               -1)

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
    df_zone_background = pd.DataFrame.from_dict(
        zone_background_dict
    )

    df_zone = pd.DataFrame.from_dict(
        zone_dict
    )

    df_res.reset_index(inplace=True)
    df_meta.reset_index(inplace=True)
    df_meta.columns = ['metadata', 'value']
    df_res.insert(0, "metadata", df_meta['metadata'])
    df_res.insert(1, "value", df_meta['value'])

    df_res = pd.concat([df_res, df_zone_background], axis=1)
    df_res = pd.concat([df_res, df_zone], axis=1)

    df_res.to_csv(out_nix_csv_file, index=False)
