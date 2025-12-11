from typing import Optional, Tuple

import cv2
import numpy as np


def _cuda_flow_available() -> bool:
    """Check whether CUDA Farneback is available."""
    return bool(
        getattr(cv2, "cuda", None)
        and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
        and hasattr(cv2.cuda, "FarnebackOpticalFlow_create")
    )


def compute_optical_flow(prev_frame: np.ndarray,
                         current_frame: np.ndarray,
                         scale: float = 1.0,
                         max_dim: Optional[int] = None,
                         use_umat: Optional[bool] = None,
                         prefer_cuda: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense Farneback optical flow with automatic device selection.

    Tries CUDA (if available) for maximum speed, then OpenCL/UMat, then CPU.
    Frames can be optionally downscaled for speed via `scale` and/or `max_dim`.
    The returned flow is always resized back to the original resolution and
    scaled so downstream consumers do not need to adjust.
    """
    if prev_frame is None or current_frame is None:
        raise ValueError(
            "prev_frame and current_frame must be non-None arrays.")
    if prev_frame.shape != current_frame.shape:
        raise ValueError(
            f"prev_frame and current_frame must share the same shape, "
            f"got {prev_frame.shape} vs {current_frame.shape}"
        )

    orig_h, orig_w = prev_frame.shape[:2]

    # Combine scale and max_dim into an effective scale
    scale = float(max(scale, 1e-3))
    if max_dim is not None:
        dynamic_scale = min(1.0, float(max_dim) / float(max(orig_h, orig_w)))
        scale = min(scale, dynamic_scale)

    if scale != 1.0:
        prev_frame = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
        current_frame = cv2.resize(current_frame, (0, 0), fx=scale, fy=scale)

    prefer_cuda = True if prefer_cuda is None else bool(prefer_cuda)
    opencl_available = cv2.ocl.haveOpenCL()
    if use_umat is None:
        use_umat = opencl_available and cv2.ocl.useOpenCL()
    elif use_umat and opencl_available:
        cv2.ocl.setUseOpenCL(True)
        use_umat = cv2.ocl.useOpenCL()
    else:
        use_umat = False

    flow: Optional[np.ndarray] = None

    # Fastest path: CUDA Farneback
    if prefer_cuda and _cuda_flow_available():
        try:
            cuda_flow = cv2.cuda.FarnebackOpticalFlow_create(
                numLevels=1,
                pyrScale=0.5,
                fastPyramids=False,
                winSize=1,
                numIters=3,
                polyN=3,
                polySigma=1.1,
                flags=0,
            )
            prev_gpu = cv2.cuda_GpuMat()
            curr_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(prev_frame)
            curr_gpu.upload(current_frame)
            prev_gray_gpu = cv2.cuda.cvtColor(prev_gpu, cv2.COLOR_BGR2GRAY)
            curr_gray_gpu = cv2.cuda.cvtColor(curr_gpu, cv2.COLOR_BGR2GRAY)
            flow_gpu = cuda_flow.calc(prev_gray_gpu, curr_gray_gpu, None)
            flow = flow_gpu.download()
        except Exception:
            flow = None  # Fall back to OpenCL/CPU

    # Next: OpenCL via UMat
    if flow is None:
        used_umat = False
        if use_umat and opencl_available:
            try:
                prev_gray = cv2.cvtColor(
                    cv2.UMat(prev_frame), cv2.COLOR_BGR2GRAY)
                current_gray = cv2.cvtColor(
                    cv2.UMat(current_frame), cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, current_gray, None,
                    pyr_scale=0.5,
                    levels=1,
                    winsize=1,
                    iterations=3,
                    poly_n=3,
                    poly_sigma=1.1,
                    flags=0
                )
                used_umat = True
            except Exception:
                flow = None
        if flow is None and not used_umat:
            # CPU fallback
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, current_gray, None,
                pyr_scale=0.5,
                levels=1,
                winsize=1,
                iterations=3,
                poly_n=3,
                poly_sigma=1.1,
                flags=0
            )

    # Ensure ndarray regardless of UMat path
    if hasattr(flow, "get"):
        flow = flow.get()

    # Upscale flow to original resolution if downscaled
    if scale != 1.0:
        flow = cv2.resize(flow, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR)
        flow /= scale  # compensate for downscaling

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.nan_to_num(magnitude)
    magnitude_normalized = np.clip(
        cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX), 0, 255
    ).astype(np.uint8)
    angle_degrees = (angle * 180 / np.pi / 2).astype(np.uint8)

    # Store both the display-friendly normalized magnitude (channel 1) and the raw
    # float magnitude (channel 2) so motion index computations can use unscaled values.
    flow_hsv = np.empty((orig_h, orig_w, 3), dtype=np.float32)
    flow_hsv[..., 0] = angle_degrees
    flow_hsv[..., 1] = magnitude_normalized.astype(np.float32)
    flow_hsv[..., 2] = magnitude

    return flow_hsv, flow
