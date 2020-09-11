"""
Reference:
https://github.com/nwojke/deep_sort/tree/master/deep_sort
https://github.com/ZQPei/deep_sort_pytorch
"""
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment

INF_COST = 1e+6

# Table for the 0.95 quantile of the chi-square distribution
# Table for the 0.95 quantile of the chi-square distribution
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


def iou(bbox, candidates):
    """
    Intersection over union
    bbox in top left x,top, left y
    width,height format
    """

    top_left, bottom_right = bbox[:2], bbox[:2] + bbox[2:]
    candidates_top_left = candidates[:, :2]
    candidates_bottom_right = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(top_left[0],
                          candidates_top_left[:, 0]
                          )[:, np.newaxis],
               np.maximum(top_left[1],
                          candidates_top_left[:, 1]
                          )[:, np.newaxis],
               ]
    br = np.c_[np.maximum(bottom_right[0],
                          candidates_bottom_right[:, 0]
                          )[:, np.newaxis],
               np.maximum(bottom_right[1],
                          candidates_bottom_right[:, 1]
                          )[:, np.newaxis],
               ]

    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)

    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks,
             detections,
             track_indices=None,
             detection_indices=None
             ):
    if track_indices is None:
        track_indices = np.arange(len(tracks))

    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((
        len(track_indices),
        len(detection_indices)
    ))

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INF_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([
            detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def gate_cost_matrix(kf,
                     cost_matrix,
                     tracks,
                     detections,
                     track_indices,
                     detection_indices,
                     gated_cost=INF_COST,
                     only_position=False
                     ):

    if only_position:
        gating_dim = 2
    else:
        gating_dim = 4

    gating_threshold = CHI2INV95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices]
    )
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position
        )
        cost_matrix[row, gating_distance >
                    gating_threshold] = gated_cost
    return cost_matrix


def min_cost_matching(distance_metric,
                      max_distance,
                      tracks,
                      detections,
                      track_indices=None,
                      detection_indices=None
                      ):
    """
    Linear assignment
    """

    if track_indices is None:
        track_indices = np.arange(len(tracks))

    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(
        tracks,
        detections,
        track_indices,
        detection_indices
    )

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_inices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in row_inices:
            unmatched_tracks.append(track_idx)

    for row, col in zip(row_inices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric,
                     max_distance,
                     cascade_depth,
                     tracks,
                     detections,
                     track_indices=None,
                     detection_indices=None
                     ):

    if track_indices is None:
        track_indices = list(range(len(tracks)))

    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]

        if len(track_indices_l) == 0:
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric,
                max_distance,
                tracks,
                detections,
                track_indices_l,
                unmatched_detections
            )
        matches += matches_l
    unmatched_tracks = list(set(track_indices) -
                            set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def non_max_suppression(boxes,
                        max_bbox_overlap,
                        scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick


def _pair_wise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2 = np.square(a).sum(axis=1)
    b2 = np.square(b).sum(axis=1)
    r2 = -2 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def _cosine(a, b, normalized=False):
    if not normalized:
        a = np.asarray(a) / np.linalg.norm(a,
                                            axis=1,
                                            keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b,
                                            axis=1,
                                            keepdims=True)
    return 1. - np.dot(a, b.T)

def _nn_euclidean(x, y):
    distances = _pair_wise(x, y)
    return np.maximum(0.0,
                        distances.min(axis=0))

def _nn_cosine(x, y):
    """
    nn: nearest neighbor
    """
    distances = _cosine(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric():

    def __init__(self,
                 metric,
                 matching_threshold,
                 budget=None
                 ):

        if metric == "euclidean":
            self.metric = _nn_euclidean
        elif metric == "cosine":
            self.metric = _nn_cosine
        else:
            raise ValueError(
                "Invalid metric"
            )
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self,
                    features,
                    targets,
                    active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))

        for i, target in enumerate(targets):
            cost_matrix[i, :] = self.metric(
                self.samples[target],
                features
            )
        return cost_matrix
