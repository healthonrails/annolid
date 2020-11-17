"""
Modified from here: 
https://github.com/ZQPei/deep_sort_pytorch/blob/master/utils/draw.py

"""
import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255)
             for p in palette]
    return tuple(color)


def get_label_color(label_id):

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
                      identities=None):

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
