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
