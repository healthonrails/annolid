import cv2
import numpy as np


def compute_optical_flow(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Convert magnitude to [0, 255] range for visualization
    magnitude_normalized = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert angle to HSV for color mapping
    angle_degrees = (angle * 180 / np.pi / 2).astype(np.uint8)
    flow_hsv = np.zeros_like(prev_frame)
    flow_hsv[..., 0] = angle_degrees
    flow_hsv[..., 1] = magnitude_normalized
    flow_hsv[..., 2] = magnitude

    return flow_hsv, flow
