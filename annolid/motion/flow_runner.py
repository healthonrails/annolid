"""
Standalone optical-flow runner (Farneback, Torch Farneback, or RAFT) for videos.

Usage (CLI):
    python -m annolid.motion.flow_runner --video /path/to/video.mp4 \
        --backend farneback --csv flow_stats.csv --ndjson flow.ndjson \
        --viz quiver

Notes:
- Torch Farneback uses --backend farneback_torch.
- RAFT requires torch+torchvision optical_flow models; pass --backend raft to opt in.
- Overlays use annolid.utils.draw.draw_flow for visualization and can be streamed
  via a preview callback instead of saving to disk.
- NDJSON output mirrors depth-anything metadata: one record per frame with
  base64-encoded uint16 flow components and scales for reconstruction.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Callable, Optional, Tuple, List
import gzip

import cv2
import numpy as np

from annolid.motion.optical_flow import compute_optical_flow
from annolid.utils import draw


def flow_to_color(flow: np.ndarray, max_mag: Optional[float] = None) -> np.ndarray:
    """
    Convert flow (H, W, 2) to an RGB visualization similar to classic flow color wheels.
    Hue encodes direction; saturation encodes magnitude; value is fixed bright.
    """
    dx = flow[..., 0].astype(np.float32)
    dy = flow[..., 1].astype(np.float32)
    mag, ang = cv2.cartToPolar(dx, dy)
    if max_mag is None:
        finite = mag[np.isfinite(mag)]
        if finite.size:
            max_mag = float(np.percentile(finite, 95))
        else:
            max_mag = 1.0
    max_mag = max(max_mag, 1e-6)
    hue = ang * 180.0 / np.pi / 2.0  # [0,180)
    sat = np.clip(mag / max_mag, 0.0, 1.0) * 255.0
    val = np.full_like(hue, 255.0)
    hsv = np.stack(
        [np.clip(hue, 0, 179), sat, val],
        axis=-1,
    ).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def _compute_mean_flow(flow: np.ndarray) -> Tuple[float, float, float]:
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_dx = float(flow[..., 0].mean())
    mean_dy = float(flow[..., 1].mean())
    mean_mag = float(mag.mean())
    return mean_dx, mean_dy, mean_mag


def _build_flow_record(
    video_name: str,
    frame_index: int,
    flow: np.ndarray,
    image_height: int,
    image_width: int,
    quantize: bool = True,
) -> dict:
    """Create a depth-like NDJSON record for optical flow."""
    dx = flow[..., 0].astype(np.float16)
    dy = flow[..., 1].astype(np.float16)
    mag = np.sqrt(dx**2 + dy**2).astype(np.float16)

    return {
        "version": "OpticalFlow",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
        "otherData": {
            "flow_dx_raw": _encode_array(dx, quantize=quantize),
            "flow_dy_raw": _encode_array(dy, quantize=quantize),
            "flow_mag_raw": _encode_array(mag, quantize=quantize),
        },
        "frame_index": int(frame_index),
        "video_name": video_name,
    }


def _encode_array(arr: np.ndarray, quantize: bool = True) -> dict:
    """Encode an array with base64 + gzip compression."""
    data = np.asarray(arr, dtype=np.float32)
    scale = {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
    }
    if quantize:
        d_min, d_max = scale["min"], scale["max"]
        if abs(d_max - d_min) < 1e-6:
            d_max = d_min + 1e-6
        normalized = (data - d_min) / (d_max - d_min)
        quantized = np.clip(np.round(normalized * 65535.0), 0, 65535).astype(np.uint16)
        raw_bytes = quantized.tobytes()
        dtype = "uint16"
    else:
        arr16 = data.astype(np.float16)
        raw_bytes = arr16.tobytes()
        dtype = "float16"

    compressed = gzip.compress(raw_bytes)
    b64 = base64.b64encode(compressed).decode("ascii")
    return {
        "dtype": dtype,
        "shape": list(data.shape),
        "scale": scale,
        "compressed": True,
        "data": b64,
    }


def _append_flow_ndjson(ndjson_path: Path, record: dict) -> None:
    ndjson_path.parent.mkdir(parents=True, exist_ok=True)
    with ndjson_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record))
        fh.write("\n")


def process_video_flow(
    video_path: str,
    backend: str = "farneback",
    save_csv: Optional[str] = None,
    save_ndjson: Optional[str] = None,
    sample_stride: int = 1,
    visualization: str = "quiver",
    raft_model: str = "small",
    opacity: int = 70,
    quiver_step: int = 16,
    quiver_gain: float = 1.0,
    stable_hsv: bool = True,
    smooth: bool = False,
    smooth_kernel: int = 3,
    quantize: bool = True,
    use_torch_farneback: bool = False,
    farneback_pyr_scale: float = 0.5,
    farneback_levels: int = 1,
    farneback_winsize: int = 1,
    farneback_iterations: int = 3,
    farneback_poly_n: int = 3,
    farneback_poly_sigma: float = 1.1,
    progress_callback: Optional[Callable[[int], None]] = None,
    preview_callback: Optional[Callable[[dict], None]] = None,
) -> None:
    """
    Compute optical flow for a video and optionally stream overlays/metrics.

    Args:
        video_path: input video file path.
        backend: 'farneback' (default), 'farneback_torch', or 'raft'.
        raft_model: 'small' or 'large' RAFT variant when backend='raft'.
        save_csv: output csv path with mean dx, dy, magnitude per frame (optional).
        save_ndjson: optional NDJSON path (depth-anything style) storing flow components.
        sample_stride: process every Nth frame (>=1).
        visualization: 'quiver' or 'hsv' for color-coded magnitude/direction.
        opacity: overlay opacity percent (0-100).
        quiver_step: arrow sampling density for quiver visualization.
        quiver_gain: arrow length gain for quiver visualization.
        stable_hsv: if True, stabilize HSV brightness across frames.
        smooth: spatially smooth flow before encoding/preview.
        smooth_kernel: odd kernel size for smoothing when enabled.
        quantize: store ndjson values quantized to uint16 (smaller); if False store float16.
        use_torch_farneback: attempt torch Farneback (for verification) before cv CUDA/UMat/CPU.
        farneback_*: parameters forwarded to Farneback computation when backend='farneback'.
        progress_callback: optional callable receiving integer percent progress.
        preview_callback: optional callable receiving overlay previews.
    """
    backend_val = str(backend).lower()
    use_raft_backend = "raft" in backend_val
    use_torch_backend = ("torch" in backend_val) and not use_raft_backend
    use_torch = bool(use_torch_farneback) or use_torch_backend

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    stats: List[Tuple[int, float, float, float]] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_flows = (
        max(1, (total_frames - 1) // max(sample_stride, 1)) if total_frames > 1 else 0
    )

    ret, prev_frame = cap.read()
    frame_idx = 0
    processed = 0
    ndjson_path = Path(save_ndjson) if save_ndjson else None
    mag_scale: Optional[float] = None
    ema_alpha = 0.9  # higher -> slower adaptation

    while ret:
        ret, curr_frame = cap.read()
        frame_idx += 1
        if not ret or curr_frame is None:
            break
        if frame_idx % sample_stride != 0:
            prev_frame = curr_frame
            continue

        flow_hsv, flow = compute_optical_flow(
            prev_frame,
            curr_frame,
            use_raft=use_raft_backend,
            raft_model=raft_model,
            use_torch_farneback=use_torch,
            farneback_pyr_scale=farneback_pyr_scale,
            farneback_levels=farneback_levels,
            farneback_winsize=farneback_winsize,
            farneback_iterations=farneback_iterations,
            farneback_poly_n=farneback_poly_n,
            farneback_poly_sigma=farneback_poly_sigma,
        )
        if smooth and smooth_kernel > 1:
            k = int(max(1, smooth_kernel))
            if k % 2 == 0:
                k += 1
            flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (k, k), 0)
            flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (k, k), 0)
        if visualization.lower() == "hsv":
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
            finite = mag[np.isfinite(mag)]
            if stable_hsv and finite.size:
                current_p95 = float(np.percentile(finite, 95))
                if mag_scale is None:
                    mag_scale = max(current_p95, 1e-6)
                else:
                    mag_scale = ema_alpha * mag_scale + (1.0 - ema_alpha) * current_p95
            overlay_out = flow_to_color(flow, max_mag=mag_scale if stable_hsv else None)
        else:
            flow_scaled = flow * float(quiver_gain)
            blank = np.zeros_like(curr_frame)
            arrows_bgr = draw.draw_flow(blank, flow_scaled, step=int(quiver_step))
            arrows_rgb = cv2.cvtColor(arrows_bgr, cv2.COLOR_BGR2RGB)
            mask = np.any(arrows_bgr != 0, axis=2)
            alpha_val = int(np.clip(opacity, 0, 100) / 100.0 * 255.0)
            alpha = np.zeros((arrows_rgb.shape[0], arrows_rgb.shape[1]), dtype=np.uint8)
            alpha[mask] = alpha_val
            overlay_out = np.ascontiguousarray(np.dstack([arrows_rgb, alpha]))
        mean_dx, mean_dy, mean_mag = _compute_mean_flow(flow)
        stats.append((frame_idx, mean_dx, mean_dy, mean_mag))
        processed += 1
        if preview_callback:
            preview_callback(
                {
                    "overlay": overlay_out,
                    "frame_index": frame_idx,
                    "mean_flow": {
                        "dx": mean_dx,
                        "dy": mean_dy,
                        "magnitude": mean_mag,
                    },
                }
            )
        if progress_callback and total_flows > 0:
            percent = min(int(processed / total_flows * 100), 100)
            progress_callback(percent)
        if ndjson_path:
            record = _build_flow_record(
                Path(video_path).name,
                frame_idx,
                flow,
                curr_frame.shape[0],
                curr_frame.shape[1],
                quantize=quantize,
            )
            _append_flow_ndjson(ndjson_path, record)
        prev_frame = curr_frame

    cap.release()

    if save_csv and stats:
        import csv

        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "mean_dx", "mean_dy", "mean_magnitude"])
            writer.writerows(stats)
    if progress_callback and processed > 0:
        progress_callback(100)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optical flow runner")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument(
        "--backend",
        default="farneback",
        choices=["farneback", "farneback_torch", "raft"],
        help="Flow backend",
    )
    p.add_argument("--csv", help="Path to save CSV of mean flow stats")
    p.add_argument(
        "--stride", type=int, default=1, help="Process every Nth frame (default 1)"
    )
    p.add_argument("--ndjson", help="Path to save NDJSON of per-frame flow maps")
    p.add_argument(
        "--viz",
        default="quiver",
        choices=["quiver", "hsv"],
        help="Overlay visualization style",
    )
    p.add_argument(
        "--raft-model",
        default="small",
        choices=["small", "large"],
        help="RAFT model size",
    )
    p.add_argument(
        "--opacity", type=int, default=70, help="Overlay opacity percent (0-100)"
    )
    p.add_argument("--quiver-step", type=int, default=16, help="Quiver sampling step")
    p.add_argument("--quiver-gain", type=float, default=1.0, help="Quiver arrow gain")
    p.add_argument(
        "--stable-hsv",
        dest="stable_hsv",
        action="store_true",
        help="Stabilize HSV magnitude across frames",
    )
    p.add_argument(
        "--no-stable-hsv",
        dest="stable_hsv",
        action="store_false",
        help="Disable stabilized HSV magnitude",
    )
    p.add_argument(
        "--smooth", action="store_true", help="Spatially smooth flow before saving"
    )
    p.add_argument(
        "--smooth-kernel",
        type=int,
        default=3,
        help="Gaussian kernel size when smoothing",
    )
    p.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_false",
        help="Store raw float16 instead of quantized uint16",
    )
    p.set_defaults(stable_hsv=True)
    p.set_defaults(quantize=True)
    return p.parse_args()


def main():
    args = _parse_args()
    process_video_flow(
        video_path=args.video,
        backend=args.backend,
        save_csv=args.csv,
        save_ndjson=args.ndjson,
        sample_stride=max(1, int(args.stride)),
        visualization=args.viz,
        raft_model=args.raft_model,
        opacity=int(args.opacity),
        quiver_step=int(args.quiver_step),
        quiver_gain=float(args.quiver_gain),
        stable_hsv=bool(args.stable_hsv),
        smooth=bool(args.smooth),
        smooth_kernel=int(args.smooth_kernel),
        quantize=bool(args.quantize),
        use_torch_farneback=bool(args.torch_farneback),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
