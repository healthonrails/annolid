import numpy as np
from annolid.realtime.mediapipe_engine import MediaPipeEngine, MediaPipeResult


def test_mediapipe_pose_engine():
    engine = MediaPipeEngine("mediapipe_pose")
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = engine(frame)

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, MediaPipeResult)
    assert res.model_type == "pose"
    assert hasattr(res, "norm_landmarks")
    assert res.names[0] == "person"


def test_mediapipe_hands_engine():
    engine = MediaPipeEngine("mediapipe_hands")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = engine(frame)

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert res.model_type == "hands"
    assert res.names[0] == "hand"


def test_mediapipe_face_engine():
    engine = MediaPipeEngine("mediapipe_face")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = engine(frame)

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert res.model_type == "face"
    assert res.names[0] == "face"
