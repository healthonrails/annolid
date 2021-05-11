"""
Modified from here: 
https://github.com/ZQPei/deep_sort_pytorch/blob/master/utils/draw.py

"""
import numpy as np
import cv2
from pathlib import Path
from annolid.utils.config import get_config


def get_keypoint_connection_rules(keypoint_cfg_file=None):
    """Keypoint connection rules defined in the config file. 

    Args:
        keypoint_cfg_file (str, optional): a yaml config file. Defaults to None.

    Returns:
        [(tuples),]: [(body_part_1,body_part2,(225,255,0))]
    """
    if keypoint_cfg_file is None:
        keypoint_cfg_file = Path(__file__).parent.parent / \
            'configs' / 'keypoints.yaml'

    keypoints_connection_rules = []
    if keypoint_cfg_file.exists():
        key_points_rules = get_config(
            str(keypoint_cfg_file)
        )
        # color is a placehold for future customization
        for k, v in key_points_rules['HEAD'].items():
            v, r, g, b = v.split(',')
            color = (int(r), int(g), int(b))
            keypoints_connection_rules.append((k, v, color))
        for k, v in key_points_rules['BODY'].items():
            v, r, g, b = v.split(',')
            color = (int(r), int(g), int(b))
            keypoints_connection_rules.append((k, v, color))
    return keypoints_connection_rules


def _keypoint_color():
    keypoint_connection_rules = get_keypoint_connection_rules()
    keypoint_colors = {}
    for kcr in keypoint_connection_rules:
        keypoint_colors[kcr[0]] = kcr[-1]
    return keypoint_colors


KEYPOINT_COLORS = _keypoint_color()


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255)
             for p in palette]
    return tuple(color)


def get_label_color(label_id):
    if label_id in KEYPOINT_COLORS:
        return label_id, KEYPOINT_COLORS[label_id]
    else:
        try:
            _id = int(label_id) if label_id is not None else 0
            color = compute_color_for_labels(_id)
            label = '{}{:d}'.format("", _id)
        except:
            color = compute_color_for_labels(hash(label_id) % 100)
            label = f'{label_id}'
        return label, color


def draw_binary_masks(img,
                      masks,
                      identities=None
                      ):

    img_alpha = np.zeros(img.shape, img.dtype)
    for i, _mask in enumerate(masks):
        label, color = get_label_color(identities[i])
        img_alpha[:, :] = color
        img_alpha = cv2.bitwise_and(img_alpha, img_alpha, mask=_mask)
        img = cv2.addWeighted(img_alpha, 0.4, img, 1, 0, img)
    return img


def draw_boxes(img,
               bbox,
               identities=None,
               offset=(0, 0),
               draw_track=False,
               points=None
               ):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        _id = identities[i]

        label, color = get_label_color(_id)

        t_size = cv2.getTextSize(label,
                                 cv2.FONT_HERSHEY_PLAIN,
                                 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4),
            color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4),
            cv2.FONT_HERSHEY_PLAIN, 2,
            [255, 255, 255], 2)

        if draw_track:
            if isinstance(_id, str):
                _id = hash(_id) % 100
            center = (int((x2 + x1)/2), int((y2+y1)/2))
            points[_id].append(center)
            thickness = 2
            for j in range(len(points[_id])):
                if points[_id][j-1] is None or points[_id] is None:
                    continue
                cv2.line(
                    img,
                    points[_id][j-1],
                    points[_id][j],
                    color,
                    thickness
                )

    return img


def draw_flow(img,
              flow,
              step=16,
              quiver=(0, 100, 0)
              ):
    """
    Modified from here 
    https://gist.github.com/RodolfoFerro/11d39fad57e21b5e85fe4d4a906cf098

    """

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img
    cv2.polylines(vis, lines, 0, quiver, 3)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_keypoint_connections(frame,
                              keypoints,
                              keypoint_connection_rules=None,
                              max_dist=250):
    """draw the lines between defined keypoints
    """

    if keypoint_connection_rules is None:
        # # rules for drawing a line for a pair of keypoints.
        # keypoint_connection_rules = [
        #     # head
        #     ("left_ear", "right_ear", (102, 204, 255)),
        #     ("nose", "left_ear", (102, 0, 204)),
        #     ("right_ear", "nose", (51, 102, 255)),
        #     # body
        #     ("left_ear", "tail_base", (255, 128, 0)),
        #     ("tail_base", "right_ear", (153, 255, 204)),
        # ]
        return frame

    for kp0, kp1, color in keypoint_connection_rules:
        if kp0 in keypoints and kp1 in keypoints:
            kp0_point = keypoints[kp0][0:2]
            #kp0_color = keypoints[kp0][-1]
            kp1_point = keypoints[kp1][0:2]
            dist = np.linalg.norm(np.array(kp0_point)-np.array(kp1_point))
            if dist < max_dist:
                cv2.line(frame, kp0_point, kp1_point, color, 3)
            else:
                print(f"Dist between two points is {dist}")
