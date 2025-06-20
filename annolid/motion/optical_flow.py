import cv2
import numpy as np


def compute_optical_flow(prev_frame, current_frame, scale=1.0):
    # Optionally downscale frames to reduce computational cost
    if scale != 1.0:
        prev_frame = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
        current_frame = cv2.resize(current_frame, (0, 0), fx=scale, fy=scale)

    # Convert frames to grayscale (consider using cv2.UMat for GPU acceleration)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow with tuned parameters (adjust these as needed)
    # flow = cv2.calcOpticalFlowFarneback(
    #     prev_gray, current_gray, None,
    #     0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        pyr_scale=0.5,     # This value is ignored when levels=1
        levels=1,          # Force calculation on full-resolution image only
        winsize=1,         # Sharply reduced window to focus on local details
        iterations=3,      # Keep iterations the same for now
        poly_n=3,          # Smaller polynomial neighborhood, proportional to winsize
        poly_sigma=1.1,    # Slightly smaller sigma for a sharper polynomial fit
        flags=0
    )

    # Convert flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sanitize and normalize magnitude in one go
    magnitude = np.nan_to_num(magnitude)
    magnitude_normalized = np.clip(cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)
    magnitude_float = magnitude.astype(np.float32)
    # Convert angle to degrees (scale and convert to uint8)
    angle_degrees = (angle * 180 / np.pi / 2).astype(np.uint8)

    # Preallocate HSV image and assign channels
    flow_hsv = np.empty_like(prev_frame)
    flow_hsv[..., 0] = angle_degrees
    flow_hsv[..., 1] = magnitude_normalized
    flow_hsv[..., 2] = magnitude_float

    return flow_hsv, flow
