"""
Modified from https://github.com/nwojke/deep_sort
Reference: 
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
"""
from annolid.tracker import kalman_filter
from annolid.tracker import match
from enum import Enum
import numpy as np


class Detection(object):
    """
    A bounding box detection in (x1,y1,x2,y2) format.
    """

    def __init__(self,
                 bbox,
                 score,
                 feature,
                 flow=None
                 ):
        self.bbox = np.asarray(bbox, dtype=np.float)
        self.score = float(score)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.flow = np.asarray(flow, dtype=np.float32)

    def to_xyah(self):
        box = self.bbox.copy()
        cx = (box[2] - box[0]) / 2
        cy = (box[3] - box[1]) / 2
        w = (box[2] - box[0]) 
        h = (box[3] - box[1])
        a = w / h
        return np.asarray([cx,cy,a, h])



class TrackState(Enum):
    """
    A track status. 
    For the first detection, the track is tentative. 

    """
    TENTATIVE = 1
    CONFIRMED = 2
    DELTED = 3


class Track(object):
    """
    A track with (x1,y1,x2,y2)
    """

    def __init__(self,
                 track_id,
                 mean,
                 covariance,
                 n_init,
                 max_age,
                 feature=None,
                 flow=None
                 ):

        self.track_id = track_id
        self.mean = mean
        self.covariance = covariance
        self.hits = 1
        self.age = 1
        self.frames_since_update = 0

        self.state = TrackState.TENTATIVE
        self.features = []
        self.flows = []

        if feature is not None:
            self.features.append(feature)

        if flow is not None:
            self.flows.append(flow)

        self.n_init = n_init
        self.max_age = max_age

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(
            self.mean,
            self.covariance
        )
        self.age += 1
        self.frames_since_update += 1

    def update(self,
               kf,
               detection):
        self.mean, self.covariance = kf.update(
            self.mean,
            self.covariance,
            detection
        )
        self.features.append(detection.feature)
        self.flows.append(detection.flow)

        self.hits += 1
        self.frames_since_update = 0

        if (self.state == TrackState.TENTATIVE and
                self.hits >= self.n_init):
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELTED
        elif self.frames_since_update > self.max_age:
            self.state = TrackState.DELTED

    def is_confirmed(self):
        return self.state == TrackState.TENTATIVE

    def is_tentative(self):
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        return self.state == TrackState.DELTED


class Tracker():
    """
    Mutiple aninal tracker.
    """

    def __init__(self,
                 metric,
                 max_iou_distance=0.7,
                 max_age=100,
                 n_init=1
                 ):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = \
            self.match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx]
            )
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self.init_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id
                          for t in self.tracks if t.is_confirmed()]

        features, targets, flows, _targets = [], [], [], []

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            flows += track.flows
            targets += [track.track_id for _ in track.features]
            _targets += [track.track_id for _ in track.flows]
            track.features = []
            track.flows = []
        self.metric.partial_fit(
            np.asarray(features),
            np.asarray(targets),
            active_targets
        )

    def match(self, detections):

        def gated_matrix(tracks,
                         detections,
                         track_indices,
                         detection_indices
                         ):
            features = np.array([sorted(detections[i].flow.reshape(-1),
                                        reverse=True)[:256]
                                 + sorted(detections[i].feature,
                                          reverse=True)[:256]
                                 for i in detection_indices])
            features = np.abs(features)
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = match.gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                detections,
                track_indices,
                detection_indices
            )
            return cost_matrix

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks)
            if not t.is_confirmed()
        ]

        matches_a, unmatched_tracks_a, unmatched_detections = \
            match.matching_cascade(
                gated_matrix,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks
            )

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].frames_since_update == 1
        ]

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].frames_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            match.min_cost_matching(
                match.iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections

            )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a +
                                    unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def init_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance,
                  self.next_id,
                  self.next_id,
                  self.max_age,
                  detection.feature
                  )
        )
        self.next_id += 1
