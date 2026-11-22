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
import csv
import gzip
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from annolid.motion.optical_flow import (
    DEFAULT_FARNEBACK_ITERATIONS,
    DEFAULT_FARNEBACK_LEVELS,
    DEFAULT_FARNEBACK_POLY_N,
    DEFAULT_FARNEBACK_POLY_SIGMA,
    DEFAULT_FARNEBACK_PYR_SCALE,
    DEFAULT_FARNEBACK_WINSIZE,
    compute_optical_flow_field,
    resolve_optical_flow_backend_flags,
)
from annolid.utils import draw

DEFAULT_FLOW_GZIP_COMPRESSLEVEL = 3


def flow_to_color(
    flow: np.ndarray,
    max_mag: Optional[float] = None,
    *,
    mag: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert flow (H, W, 2) to an RGB visualization similar to classic flow color wheels.
    Hue encodes direction; saturation encodes magnitude; value is fixed bright.
    """
    flow_f32 = np.asarray(flow, dtype=np.float32)
    dx = flow_f32[..., 0]
    dy = flow_f32[..., 1]
    h, w = dx.shape[:2]

    computed_mag, ang_deg = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    mag_arr = computed_mag if mag is None else np.asarray(mag, dtype=np.float32)

    if max_mag is None:
        if np.isfinite(mag_arr).all():
            max_mag = float(np.percentile(mag_arr, 95)) if mag_arr.size else 1.0
        else:
            finite = mag_arr[np.isfinite(mag_arr)]
            max_mag = float(np.percentile(finite, 95)) if finite.size else 1.0
    max_mag = max(float(max_mag), 1e-6)

    hsv = np.empty((h, w, 3), dtype=np.uint8)
    if not np.isfinite(ang_deg).all():
        ang_deg = np.nan_to_num(ang_deg, nan=0.0, posinf=0.0, neginf=0.0)
    np.multiply(ang_deg, 0.5, out=ang_deg)
    np.clip(ang_deg, 0.0, 179.0, out=ang_deg)
    hsv[..., 0] = ang_deg

    sat = (
        mag_arr.copy()
        if np.isfinite(mag_arr).all()
        else np.nan_to_num(mag_arr, nan=0.0, posinf=max_mag, neginf=0.0)
    )
    np.multiply(sat, 255.0 / max_mag, out=sat)
    np.clip(sat, 0.0, 255.0, out=sat)
    hsv[..., 1] = sat
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flow_to_quiver_rgba(
    flow: np.ndarray,
    *,
    opacity: int = 70,
    step: int = 16,
    gain: float = 1.0,
) -> np.ndarray:
    """Render flow vectors onto a transparent RGBA overlay."""
    flow_f32 = np.asarray(flow, dtype=np.float32)
    h, w = flow_f32.shape[:2]
    gain_f = float(gain)
    flow_scaled = flow_f32 if gain_f == 1.0 else (flow_f32 * gain_f)
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    arrows_bgr = draw.draw_flow(blank, flow_scaled, step=max(1, int(step)))
    arrows_rgb = cv2.cvtColor(arrows_bgr, cv2.COLOR_BGR2RGB)
    mask = (
        (arrows_bgr[..., 0] != 0)
        | (arrows_bgr[..., 1] != 0)
        | (arrows_bgr[..., 2] != 0)
    )
    alpha_val = int(np.clip(float(opacity), 0.0, 100.0) / 100.0 * 255.0)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = arrows_rgb
    rgba[..., 3] = 0
    rgba[mask, 3] = alpha_val
    return np.ascontiguousarray(rgba)


def render_flow_overlay(
    flow: np.ndarray,
    *,
    visualization: str = "quiver",
    opacity: int = 70,
    quiver_step: int = 16,
    quiver_gain: float = 1.0,
    max_mag: Optional[float] = None,
    mag: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render either an HSV RGB overlay or a quiver RGBA overlay for `flow`."""
    if str(visualization or "").strip().lower() == "hsv":
        return flow_to_color(flow, max_mag=max_mag, mag=mag)
    return flow_to_quiver_rgba(
        flow,
        opacity=opacity,
        step=quiver_step,
        gain=quiver_gain,
    )


def _compute_mean_flow(
    flow: np.ndarray, *, mag: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag_arr = np.hypot(dx, dy) if mag is None else mag
    mean_dx = float(np.mean(dx, dtype=np.float64))
    mean_dy = float(np.mean(dy, dtype=np.float64))
    mean_mag = float(np.mean(mag_arr, dtype=np.float64))
    return mean_dx, mean_dy, mean_mag


def encode_flow_array(
    arr: np.ndarray,
    quantize: bool = True,
    *,
    compresslevel: int = DEFAULT_FLOW_GZIP_COMPRESSLEVEL,
) -> dict:
    """Encode a 2D float array with gzip + base64 (uint16 quantized or raw float16)."""
    data = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    d_min = float(data.min()) if data.size else 0.0
    d_max = float(data.max()) if data.size else 0.0
    scale = {"min": d_min, "max": d_max}

    if quantize:
        span = d_max - d_min
        if abs(span) < 1e-6:
            quantized = np.zeros(data.shape, dtype=np.uint16)
        else:
            normalized = (data - d_min) * (65535.0 / span)
            quantized = np.clip(np.rint(normalized), 0.0, 65535.0).astype(np.uint16)
        raw_bytes = np.ascontiguousarray(quantized).tobytes()
        dtype = "uint16"
    else:
        arr16 = np.ascontiguousarray(data, dtype=np.float16)
        raw_bytes = arr16.tobytes()
        dtype = "float16"

    compressed = gzip.compress(raw_bytes, compresslevel=int(compresslevel))
    b64 = base64.b64encode(compressed).decode("ascii")
    return {
        "dtype": dtype,
        "shape": list(data.shape),
        "scale": scale,
        "compressed": True,
        "data": b64,
    }


# Backward-compatible alias for internal callers/tests.
_encode_array = encode_flow_array


def decode_flow_component(
    comp: Optional[Mapping[str, Any]],
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    """Decode one encoded flow component (compressed base64 or legacy values list)."""
    if not comp or not isinstance(comp, Mapping):
        return None

    dtype_name = str(comp.get("dtype") or "uint16").lower()
    np_dtype = np.float16 if dtype_name == "float16" else np.uint16

    values = comp.get("values")
    if values is None and comp.get("compressed"):
        b64 = comp.get("data")
        if not b64:
            return None
        try:
            raw = gzip.decompress(base64.b64decode(b64))
            arr = np.frombuffer(raw, dtype=np_dtype)
            shape = comp.get("shape") or []
            if shape:
                arr = arr.reshape(tuple(int(s) for s in shape))
        except Exception:
            return None
    else:
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=np_dtype)
        except Exception:
            return None

    if arr.ndim != 2 or arr.size == 0:
        return None

    scale = comp.get("scale")
    scale_map = scale if isinstance(scale, Mapping) else {}
    has_min = "min" in scale_map and scale_map.get("min") is not None
    has_max = "max" in scale_map and scale_map.get("max") is not None

    try:
        if dtype_name == "uint16":
            v_min = float(scale_map.get("min", 0.0))
            v_max = float(scale_map.get("max", 0.0))
            if not (np.isfinite(v_min) and np.isfinite(v_max)):
                v_min, v_max = 0.0, 0.0
            span = v_max - v_min
            if abs(span) < 1e-6:
                out = np.full(arr.shape, v_min, dtype=np.float32)
            else:
                out = arr.astype(np.float32) * (span / 65535.0) + v_min
                out = np.clip(out, v_min, v_max)
        else:
            out = arr.astype(np.float32, copy=False)
            if has_min and has_max:
                v_min = float(scale_map["min"])
                v_max = float(scale_map["max"])
                if np.isfinite(v_min) and np.isfinite(v_max) and v_max >= v_min:
                    out = np.clip(out, v_min, v_max)

        if out.shape != (int(height), int(width)):
            out = cv2.resize(
                out, (int(width), int(height)), interpolation=cv2.INTER_LINEAR
            )
        return np.ascontiguousarray(out, dtype=np.float32)
    except Exception:
        return None


def decode_flow_record(record: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Decode `(H, W, 2)` float32 flow from an OpticalFlow NDJSON record."""
    try:
        height = int(record.get("imageHeight") or 0)
        width = int(record.get("imageWidth") or 0)
        if height <= 0 or width <= 0:
            return None
        other = record.get("otherData")
        if not isinstance(other, Mapping):
            return None
        dx_comp = other.get("flow_dx_raw") or other.get("flow_dx")
        dy_comp = other.get("flow_dy_raw") or other.get("flow_dy")
        dx = decode_flow_component(dx_comp, height, width)
        dy = decode_flow_component(dy_comp, height, width)
        if dx is None or dy is None:
            return None
        return np.stack((dx, dy), axis=-1).astype(np.float32, copy=False)
    except Exception:
        return None


def _build_flow_record(
    video_name: str,
    frame_index: int,
    flow: np.ndarray,
    image_height: int,
    image_width: int,
    quantize: bool = True,
    *,
    mag: Optional[np.ndarray] = None,
    compresslevel: int = DEFAULT_FLOW_GZIP_COMPRESSLEVEL,
) -> dict:
    """Create a depth-like NDJSON record for optical flow without float16 overflow."""
    dx = np.asarray(flow[..., 0], dtype=np.float32)
    dy = np.asarray(flow[..., 1], dtype=np.float32)
    mag_f32 = np.hypot(dx, dy) if mag is None else np.asarray(mag, dtype=np.float32)

    return {
        "version": "OpticalFlow",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
        "otherData": {
            "flow_dx_raw": encode_flow_array(
                dx, quantize=quantize, compresslevel=compresslevel
            ),
            "flow_dy_raw": encode_flow_array(
                dy, quantize=quantize, compresslevel=compresslevel
            ),
            "flow_mag_raw": encode_flow_array(
                mag_f32, quantize=quantize, compresslevel=compresslevel
            ),
        },
        "frame_index": int(frame_index),
        "video_name": video_name,
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
    farneback_pyr_scale: float = DEFAULT_FARNEBACK_PYR_SCALE,
    farneback_levels: int = DEFAULT_FARNEBACK_LEVELS,
    farneback_winsize: int = DEFAULT_FARNEBACK_WINSIZE,
    farneback_iterations: int = DEFAULT_FARNEBACK_ITERATIONS,
    farneback_poly_n: int = DEFAULT_FARNEBACK_POLY_N,
    farneback_poly_sigma: float = DEFAULT_FARNEBACK_POLY_SIGMA,
    scale: float = 1.0,
    max_dim: Optional[int] = None,
    torch_device: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    preview_callback: Optional[Callable[[dict], None]] = None,
    stop_event: Optional[threading.Event] = None,
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
        use_torch_farneback: attempt torch Farneback before cv CUDA/UMat/CPU.
        farneback_*: parameters forwarded to Farneback computation.
        scale: optional downscale factor in (0, 1] for faster computation.
        max_dim: optional max frame dimension cap for faster computation.
        torch_device: optional explicit torch device override ('cuda', 'mps', 'cpu').
        progress_callback: optional callable receiving integer percent progress.
        preview_callback: optional callable receiving overlay previews.
        stop_event: optional cooperative cancellation event.
    """
    sample_stride = max(1, int(sample_stride))
    use_raft_backend, use_torch = resolve_optical_flow_backend_flags(
        backend,
        use_torch_farneback=use_torch_farneback,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    stats: List[Tuple[int, float, float, float]] = []
    ndjson_path = Path(save_ndjson) if save_ndjson else None
    ndjson_tmp_path: Optional[Path] = None
    ndjson_fh = None
    completed_cleanly = False
    cancelled = False
    processed = 0

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_flows = (
            max(1, (total_frames - 1) // sample_stride) if total_frames > 1 else 0
        )

        if ndjson_path is not None:
            ndjson_path.parent.mkdir(parents=True, exist_ok=True)
            ndjson_tmp_path = ndjson_path.with_name(
                f".{ndjson_path.name}.{os.getpid()}.tmp"
            )
            ndjson_fh = ndjson_tmp_path.open("w", encoding="utf-8")

        ret, prev_frame = cap.read()
        frame_idx = 0
        mag_scale: Optional[float] = None
        ema_alpha = 0.9  # higher -> slower adaptation
        video_name = Path(video_path).name
        viz_mode = str(visualization or "quiver").strip().lower()
        smooth_k = int(max(1, smooth_kernel))
        if smooth_k % 2 == 0:
            smooth_k += 1

        while ret and prev_frame is not None:
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break

            ret, curr_frame = cap.read()
            frame_idx += 1
            if not ret or curr_frame is None:
                break
            if frame_idx % sample_stride != 0:
                prev_frame = curr_frame
                continue

            flow = compute_optical_flow_field(
                prev_frame,
                curr_frame,
                scale=scale,
                max_dim=max_dim,
                use_raft=use_raft_backend,
                raft_model=raft_model,
                use_torch_farneback=use_torch,
                torch_device=torch_device,
                farneback_pyr_scale=farneback_pyr_scale,
                farneback_levels=farneback_levels,
                farneback_winsize=farneback_winsize,
                farneback_iterations=farneback_iterations,
                farneback_poly_n=farneback_poly_n,
                farneback_poly_sigma=farneback_poly_sigma,
            )

            if smooth and smooth_k > 1:
                flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (smooth_k, smooth_k), 0)
                flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (smooth_k, smooth_k), 0)

            # Compute magnitude once and reuse across stats, HSV scaling, and NDJSON export.
            mag = np.hypot(flow[..., 0], flow[..., 1]).astype(np.float32, copy=False)
            mean_dx, mean_dy, mean_mag = _compute_mean_flow(flow, mag=mag)
            stats.append((frame_idx, mean_dx, mean_dy, mean_mag))
            processed += 1

            if preview_callback is not None:
                hsv_max_mag: Optional[float] = None
                if viz_mode == "hsv" and stable_hsv and mag.size:
                    if np.isfinite(mag).all():
                        current_p95 = float(np.percentile(mag, 95))
                    else:
                        finite = mag[np.isfinite(mag)]
                        current_p95 = (
                            float(np.percentile(finite, 95)) if finite.size else 1.0
                        )
                    if mag_scale is None:
                        mag_scale = max(current_p95, 1e-6)
                    else:
                        mag_scale = (
                            ema_alpha * mag_scale + (1.0 - ema_alpha) * current_p95
                        )
                    hsv_max_mag = mag_scale

                overlay_out = render_flow_overlay(
                    flow,
                    visualization=viz_mode,
                    opacity=opacity,
                    quiver_step=quiver_step,
                    quiver_gain=quiver_gain,
                    max_mag=hsv_max_mag,
                    mag=mag,
                )
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

            if ndjson_fh is not None:
                record = _build_flow_record(
                    video_name,
                    frame_idx,
                    flow,
                    curr_frame.shape[0],
                    curr_frame.shape[1],
                    quantize=quantize,
                    mag=mag,
                )
                ndjson_fh.write(json.dumps(record))
                ndjson_fh.write("\n")

            prev_frame = curr_frame

        completed_cleanly = not cancelled
    finally:
        cap.release()
        if ndjson_fh is not None:
            ndjson_fh.close()
        if ndjson_tmp_path is not None:
            if completed_cleanly and ndjson_path is not None:
                os.replace(ndjson_tmp_path, ndjson_path)
            else:
                try:
                    ndjson_tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    if completed_cleanly and save_csv and stats:
        csv_path = Path(save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "mean_dx", "mean_dy", "mean_magnitude"])
            writer.writerows(stats)
    if completed_cleanly and progress_callback and processed > 0:
        progress_callback(100)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optical flow runner")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument(
        "--backend",
        default="farneback",
        choices=["farneback", "farneback_torch", "raft"],
        help="Flow backend",
    )
    p.add_argument(
        "--torch-farneback",
        action="store_true",
        default=False,
        help="Attempt Torch Farneback before OpenCV CUDA/UMat/CPU",
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
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Optional frame downscale factor in (0, 1] for faster optical flow",
    )
    p.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Optional maximum frame dimension cap before computing flow",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Optional Torch device override (e.g. 'cuda', 'mps', 'cpu')",
    )
    p.set_defaults(stable_hsv=True)
    p.set_defaults(quantize=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
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
        scale=float(args.scale),
        max_dim=args.max_dim,
        torch_device=args.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
