# Importing necessary libraries
# Modify from BoT-SORT
# https://github.com/NirAharon/BoT-SORT/blob/main/tracker/basetrack.py
import numpy as np
from collections import OrderedDict


# Defining the TrackState class to represent the possible states of a track
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


# Defining the BaseTrack class to represent a generic track
class BaseTrack:
    # Class variables shared by all instances of the class
    _count = 0  # counter to assign unique IDs to tracks

    # Instance variables (attributes) of the class
    track_id = 0  # unique ID of the track
    is_activated = False  # whether the track is currently active or not
    state = TrackState.New  # current state of the track

    # A dictionary to keep track of the track history
    history = OrderedDict()

    # A list to store the features extracted from the track
    features = []

    # The current feature being tracked
    curr_feature = None

    # A score to evaluate the quality of the track
    score = 0

    # The frame where the track started
    start_frame = 0

    # The current frame being processed
    frame_id = 0

    # The number of frames since the last successful update
    time_since_update = 0

    # Multi-camera support: the location of the track in the current camera view
    location = (np.inf, np.inf)

    # A property that returns the last frame where the track was seen
    @property
    def end_frame(self):
        return self.frame_id

    # A static method to generate a unique ID for a new track
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    # Abstract methods that must be implemented by subclasses
    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    # Methods to change the state of the track
    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    # A static method to reset the ID counter
    @staticmethod
    def clear_count():
        BaseTrack._count = 0
