from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2


@contextmanager
def temporary_first_frame_image(video_path: str):
    """Yield a temporary image file containing the first frame of a video."""
    with TemporaryDirectory(prefix="annolid_crop_frame_") as tempdir:
        temp_path = Path(tempdir) / "frame.jpg"
        cap = cv2.VideoCapture(video_path)
        try:
            ret, frame = cap.read()
        finally:
            cap.release()
        if not ret:
            raise RuntimeError(f"Unable to extract a frame from {video_path}")
        if not cv2.imwrite(str(temp_path), frame):
            raise RuntimeError(
                f"Failed to write temporary frame image for {video_path}"
            )
        yield str(temp_path)
