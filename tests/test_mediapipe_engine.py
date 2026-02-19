import importlib.util
import pytest
import numpy as np
from annolid.realtime.mediapipe_engine import MediaPipeEngine, MediaPipeResult

# Check if mediapipe is available
HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None


def _safe_engine(model_id: str) -> MediaPipeEngine:
    try:
        return MediaPipeEngine(model_id)
    except RuntimeError as exc:
        text = str(exc)
        if "kGpuService" in text or "NSOpenGLPixelFormat" in text:
            pytest.skip("mediapipe runtime unavailable (GPU/GL context not available)")
        raise


@pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")
def test_mediapipe_pose_engine():
    engine = _safe_engine("mediapipe_pose")
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


@pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")
def test_mediapipe_hands_engine():
    engine = _safe_engine("mediapipe_hands")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = engine(frame)

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert res.model_type == "hands"
    assert res.names[0] == "hand"


@pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")
def test_mediapipe_face_engine():
    engine = _safe_engine("mediapipe_face")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = engine(frame)

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert res.model_type == "face"
    assert res.names[0] == "face"
    assert hasattr(res, "distance_cm")
    if (
        res.landmarks
        and hasattr(res.landmarks, "face_landmarks")
        and res.landmarks.face_landmarks
    ):
        assert "gaze_avg" in res.metadata
        assert "gaze_left" in res.metadata
        assert "gaze_right" in res.metadata
