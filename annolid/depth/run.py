"""Glue code that combines the bundled Video-Depth-Anything model with the Annolid UI."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import json
import base64
import io

import cv2
import imageio
import numpy as np
import torch
from matplotlib import cm

from annolid.core.media.video import CV2Video
from annolid.utils.logger import logger
from annolid.utils.annotation_store import AnnotationStore

from .download_weights import ensure_checkpoints
from .utils import save_video
from .video_depth_anything import VideoDepthAnything as BatchVideoDepthAnything
from .video_depth_anything.video_depth_stream import (
    VideoDepthAnything as StreamingVideoDepthAnything,
)

_INFERNO_PALETTE = (
    cm.get_cmap("inferno", 256)(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0
).astype(np.uint8)


def _depth_display_range(depth: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(depth, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi - lo < 1e-6:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo < 1e-6:
            hi = lo + 1e-6
    return lo, hi


def _colorize_depth_map(
    depth_map: np.ndarray, grayscale: bool
) -> Tuple[np.ndarray, float, float]:
    depth = np.asarray(depth_map, dtype=np.float32)
    d_min, d_max = _depth_display_range(depth)
    depth_clipped = np.clip(depth, d_min, d_max)
    normalized = (depth_clipped - d_min) / (d_max - d_min + 1e-9)
    if grayscale:
        vis = (normalized * 255.0).astype(np.uint8)
        return np.repeat(vis[..., None], 3, axis=2), d_min, d_max
    indices = np.clip((normalized * 255.0).round().astype(np.int16), 0, 255).astype(
        np.uint8
    )
    colored = _INFERNO_PALETTE[indices]
    return colored, d_min, d_max


def _quantize_depth(depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
    depth = depth.astype(np.float32)
    d_min = float(depth.min())
    d_max = float(depth.max())
    if abs(d_max - d_min) < 1e-6:
        d_max = d_min + 1e-6
    normalized = (depth - d_min) / (d_max - d_min)
    quantized = np.clip(np.round(normalized * 65535.0), 0, 65535).astype(np.uint16)
    return quantized, d_min, d_max


def _append_depth_ndjson(output_dir: Path, record: Dict[str, object]) -> Path:
    ndjson_path = output_dir / "depth.ndjson"
    with ndjson_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record))
        fh.write("\n")
    return ndjson_path


def _frame_positions(total_frames: int, stride: int, max_len: int) -> List[int]:
    positions = list(range(0, total_frames, stride))
    if max_len > 0 and len(positions) > max_len:
        return positions[:max_len]
    return positions


def _prepare_frame_sequence(
    video_loader: CV2Video, max_len: int, target_fps: int
) -> Tuple[List[int], float]:
    orig_fps = video_loader.get_fps()
    if not orig_fps or orig_fps <= 0:
        orig_fps = float(target_fps if target_fps > 0 else 30.0)
    fps = orig_fps if target_fps < 0 else target_fps
    stride = max(int(round(orig_fps / fps)) if fps > 0 else 1, 1)
    total_frames = video_loader.total_frames()
    positions = _frame_positions(total_frames, stride, max_len)
    return positions, fps if fps > 0 else 30.0


def _load_frames(
    video_path: Path, max_len: int, target_fps: int, max_res: int
) -> Tuple[np.ndarray, float, List[int]]:
    generator, fps, positions = _iterate_frames(
        video_path, max_len, target_fps, max_res
    )
    frames = [frame for frame in generator]
    if not frames:
        return np.zeros((0, 0, 0, 0)), fps, []
    return np.stack(frames), fps, positions


def _iterate_frames(
    video_path: Path, max_len: int, target_fps: int, max_res: int
) -> Tuple[Iterator[np.ndarray], float, List[int]]:
    video_loader = CV2Video(str(video_path))
    positions, fps = _prepare_frame_sequence(video_loader, max_len, target_fps)

    def frame_generator():
        try:
            for idx in positions:
                frame = video_loader.load_frame(idx)
                if max_res > 0:
                    height, width = frame.shape[:2]
                    if max(height, width) > max_res:
                        scale = max_res / max(height, width)
                        frame = cv2.resize(
                            frame,
                            (max(int(width * scale), 1), max(int(height * scale), 1)),
                            interpolation=cv2.INTER_AREA,
                        )
                yield frame
        finally:
            video_loader.release()

    return frame_generator(), fps, positions


def _save_depth_frames(
    depths: np.ndarray, output_dir: Path, grayscale: bool
) -> List[str]:
    depth_frames_dir = output_dir / "depth_frames"
    depth_frames_dir.mkdir(parents=True, exist_ok=True)
    flattened = depths.reshape(-1)
    d_min = float(flattened.min()) if flattened.size else 0.0
    d_max = float(flattened.max()) if flattened.size else 1.0
    result_paths: List[str] = []
    for idx, depth in enumerate(depths):
        normalized = (depth - d_min) / (d_max - d_min + 1e-9)
        if grayscale:
            vis = (normalized * 255).astype(np.uint8)
            imageio.imwrite(str(depth_frames_dir / f"depth_{idx:05d}.png"), vis)
        else:
            palette = cm.get_cmap("inferno")
            colored = (palette(normalized)[:, :, :3] * 255).astype(np.uint8)
            imageio.imwrite(str(depth_frames_dir / f"depth_{idx:05d}.png"), colored)
        result_paths.append(str(depth_frames_dir / f"depth_{idx:05d}.png"))
    return result_paths


def _safe_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tracking_frame_json_path(results_dir: Path, frame_index: int) -> Path:
    return results_dir / f"{results_dir.name}_{frame_index:09d}.json"


def _read_shapes_from_json(
    frame_path: Path,
) -> Tuple[Optional[List[Dict[str, object]]], Optional[int], Optional[int]]:
    if not frame_path.is_file():
        return None, None, None
    try:
        payload = json.loads(frame_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read tracking JSON %s: %s", frame_path, exc)
        return None, None, None
    shapes = payload.get("shapes") or []
    width = _safe_int(payload.get("imageWidth"))
    height = _safe_int(payload.get("imageHeight"))
    return shapes, width, height


def _load_shapes_for_frame(
    results_dir: Path, frame_index: int
) -> Tuple[List[Dict[str, object]], Optional[int], Optional[int]]:
    frame_path = _tracking_frame_json_path(results_dir, frame_index)
    shapes, width, height = _read_shapes_from_json(frame_path)
    if shapes is not None:
        return shapes, width, height
    try:
        store = AnnotationStore.for_frame_path(frame_path)
        record = store.get_frame(frame_index)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unable to read annotation store for frame %d: %s", frame_index, exc
        )
        return [], None, None
    if not record:
        return [], None, None
    shapes = record.get("shapes") or []
    width = _safe_int(record.get("imageWidth"))
    height = _safe_int(record.get("imageHeight"))
    return shapes, width, height


def _scaled_polygon_from_shape(
    shape: Dict[str, object], scale_x: float, scale_y: float
) -> Optional[np.ndarray]:
    points = shape.get("points")
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    shape_type = shape.get("shape_type")
    if shape_type == "rectangle" and arr.shape[0] >= 2:
        top_left, bottom_right = arr[0], arr[1]
        polygon = np.array(
            [
                [top_left[0], top_left[1]],
                [bottom_right[0], top_left[1]],
                [bottom_right[0], bottom_right[1]],
                [top_left[0], bottom_right[1]],
            ],
            dtype=np.float32,
        )
    else:
        if arr.shape[0] < 3:
            return None
        polygon = arr.copy()
    polygon[:, 0] *= scale_x
    polygon[:, 1] *= scale_y
    return np.round(polygon).astype(np.int32)


def _rasterize_shape_labels(
    shapes: List[Dict[str, object]],
    height: int,
    width: int,
    source_width: Optional[int],
    source_height: Optional[int],
) -> np.ndarray:
    num_pixels = height * width
    labels = np.full(num_pixels, "", dtype=object)
    if num_pixels == 0:
        return labels
    occupied = np.zeros((height, width), dtype=bool)
    scale_x = float(width) / float(source_width) if source_width else 1.0
    scale_y = float(height) / float(source_height) if source_height else 1.0
    for shape in shapes:
        label = shape.get("label")
        if not label:
            continue
        polygon = _scaled_polygon_from_shape(shape, scale_x, scale_y)
        if polygon is None or polygon.shape[0] < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        mask_bool = mask.astype(bool)
        assign_mask = mask_bool & ~occupied
        if not assign_mask.any():
            continue
        assigned_indices = assign_mask.reshape(-1)
        labels[assigned_indices] = str(label)
        occupied |= assign_mask
    return labels


def _frame_region_labels(
    tracking_dir: Path,
    frame_index: int,
    height: int,
    width: int,
    cache: Dict[int, Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    frame_key = int(frame_index)
    if frame_key in cache:
        return cache[frame_key]
    shapes, src_width, src_height = _load_shapes_for_frame(tracking_dir, frame_key)
    if not shapes:
        cache[frame_key] = None
        return None
    labels = _rasterize_shape_labels(shapes, height, width, src_width, src_height)
    cache[frame_key] = labels
    return labels


def _write_point_cloud_csv(
    path: Path,
    points: np.ndarray,
    intensity: np.ndarray,
    colors: Optional[np.ndarray] = None,
    region_labels: Optional[np.ndarray] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = intensity.reshape(-1)
    with path.open("w", encoding="utf-8") as fh:
        header_parts = ["x", "y", "z", "intensity"]
        color_arr = None
        if colors is not None:
            color_arr = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
            if color_arr.size and color_arr.max() <= 1.0:
                color_arr = color_arr * 255.0
            color_arr = np.clip(color_arr, 0.0, 255.0).astype(np.uint8)
        include_region_column = False
        region_arr = None
        if region_labels is not None:
            region_arr = np.asarray(region_labels).reshape(-1)
            if region_arr.shape[0] != points.shape[0]:
                logger.warning(
                    "Region labels count %d differs from point count %d for %s",
                    region_arr.shape[0],
                    points.shape[0],
                    path,
                )
                region_arr = None
            else:
                include_region_column = True
                header_parts.append("region")
        if color_arr is not None:
            header_parts.extend(["red", "green", "blue"])
        fh.write(",".join(header_parts) + "\n")
        for idx, ((x, y, z), val) in enumerate(zip(points, flattened)):
            row_items = [
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{float(val):.6f}",
            ]
            if include_region_column and region_arr is not None:
                raw_label = region_arr[idx]
                region_text = ""
                if raw_label not in (None, ""):
                    escaped = str(raw_label).replace('"', '""')
                    region_text = f'"{escaped}"'
                row_items.append(region_text)
            if color_arr is not None:
                r, g, b = color_arr[idx]
                row_items.extend([str(int(r)), str(int(g)), str(int(b))])
            fh.write(",".join(row_items) + "\n")


def _create_depth_preview_overlay(
    depth_map: np.ndarray, grayscale: bool
) -> Tuple[np.ndarray, float, float]:
    visualization, d_min, d_max = _colorize_depth_map(depth_map, grayscale)
    return np.ascontiguousarray(visualization), d_min, d_max


def _depth_map_to_base64_png(depth: np.ndarray, height: int, width: int) -> str:
    image = depth.reshape((height, width))
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="png")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded


def _build_depth_record(
    video_name: str,
    frame_index: int,
    depth_map: np.ndarray,
    image_height: int,
    image_width: int,
) -> Dict[str, object]:
    quantized, d_min, d_max = _quantize_depth(depth_map)
    depth_png = _depth_map_to_base64_png(quantized, image_height, image_width)
    return {
        "version": "Depth-Anything",
        "flags": {},
        "shapes": [],
        "imagePath": "",
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
        "otherData": {
            "depth_map": {
                "image_data": depth_png,
                "scale": {"min": d_min, "max": d_max},
                "dtype": "uint16",
            }
        },
        "frame_index": int(frame_index),
        "video_name": video_name,
    }


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(
    encoder: str,
    metric: bool,
    checkpoint_path: Path,
    device: torch.device,
    streaming: bool = True,
) -> Union[BatchVideoDepthAnything, StreamingVideoDepthAnything]:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }
    if encoder not in model_configs:
        raise ValueError(
            f"Unknown encoder {encoder!r}; expected one of {list(model_configs)}."
        )

    checkpoint_suffix = (
        "metric_video_depth_anything" if metric else "video_depth_anything"
    )
    expected_checkpoint = checkpoint_path / f"{checkpoint_suffix}_{encoder}.pth"
    if not expected_checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint {expected_checkpoint} not found. "
            "Download the Video-Depth-Anything weights and place them under annolid/depth/checkpoints."
        )

    cfg = model_configs[encoder]
    model_cls = StreamingVideoDepthAnything if streaming else BatchVideoDepthAnything
    model = model_cls(**cfg, metric=metric)
    state = torch.load(str(expected_checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model


def run_video_depth_anything(
    input_video: str,
    output_dir: str,
    encoder: str = "vits",
    input_size: int = 518,
    max_res: int = 1280,
    max_len: int = -1,
    target_fps: int = -1,
    metric: bool = False,
    fp32: bool = False,
    grayscale: bool = False,
    save_npz: bool = False,
    save_exr: bool = False,
    focal_length_x: float = 470.4,
    focal_length_y: float = 470.4,
    checkpoint_root: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    preview_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    streaming: bool = True,
    save_depth_video: bool = False,
    save_depth_frames: bool = False,
    save_point_clouds: bool = False,
    include_region_labels: bool = False,
) -> Dict[str, Optional[List[str]]]:
    """Run depth estimation on ``input_video`` and save the outputs."""

    input_path = Path(input_video).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Video '{input_path}' does not exist.")

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    tracking_results_dir_path: Optional[Path] = None
    region_cache: Dict[int, Optional[np.ndarray]] = {}
    if include_region_labels:
        tracking_candidate = input_path.with_suffix("")
        if tracking_candidate.is_dir():
            tracking_results_dir_path = tracking_candidate

    base_name = input_path.stem
    depth_vis_path = output_dir_path / f"{base_name}_vis.mp4"
    checkpoint_base = (
        Path(checkpoint_root).expanduser().resolve()
        if checkpoint_root
        else Path(__file__).resolve().parent / "checkpoints"
    )
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    model_key = f"{'metric_' if metric else ''}video_depth_anything_{encoder}"
    ensure_checkpoints([model_key], dest=checkpoint_base)

    device = _select_device()
    if device.type != "cuda" and not fp32:
        logger.info(
            "Video Depth Anything: forcing FP32 inference on %s device for stable depth values.",
            device.type,
        )
        fp32 = True
    model = _load_model(encoder, metric, checkpoint_base, device, streaming=streaming)
    infer_device = device.type

    if streaming:
        frame_iter, actual_fps, frame_positions = _iterate_frames(
            input_path, max_len=max_len, target_fps=target_fps, max_res=max_res
        )
        total_positions = len(frame_positions)
        frames_collected: List[np.ndarray] = []
        depths_collected: List[np.ndarray] = []
        for idx, frame in enumerate(frame_iter, start=1):
            depth = model.infer_video_depth_one(
                frame, input_size=input_size, device=infer_device, fp32=fp32
            )
            frames_collected.append(frame)
            depths_collected.append(depth)
            if preview_callback:
                overlay_rgb, display_min, display_max = _create_depth_preview_overlay(
                    depth, grayscale=grayscale
                )
                preview_callback(
                    {
                        "frame_index": frame_positions[idx - 1]
                        if idx - 1 < len(frame_positions)
                        else idx - 1,
                        "overlay": overlay_rgb,
                        "depth_stats": {
                            "min": float(np.nanmin(depth)),
                            "max": float(np.nanmax(depth)),
                            "mean": float(np.nanmean(depth)),
                            "display_min": display_min,
                            "display_max": display_max,
                        },
                    }
                )
            if progress_callback and total_positions:
                progress_callback(min(int(idx / total_positions * 100), 100))
            record = _build_depth_record(
                input_path.name,
                frame_positions[idx - 1] if idx - 1 < len(frame_positions) else idx - 1,
                depth,
                frame.shape[0],
                frame.shape[1],
            )
            _append_depth_ndjson(output_dir_path, record)
        if not frames_collected:
            raise RuntimeError("No frames were processed for streaming inference.")
        if progress_callback:
            progress_callback(100)
        frames = np.stack(frames_collected)
        depths = np.stack(depths_collected)
        fps = actual_fps
        frame_indices = frame_positions[: len(depths_collected)]
    else:
        frames, actual_fps, frame_indices = _load_frames(
            input_path, max_len=max_len, target_fps=target_fps, max_res=max_res
        )
        depths, fps = model.infer_video_depth(
            frames=frames,
            target_fps=actual_fps,
            input_size=input_size,
            device=infer_device,
            fp32=fp32,
            progress_callback=progress_callback,
        )
        for frame_idx, depth_map, frame_img in zip(frame_indices, depths, frames):
            record = _build_depth_record(
                input_path.name,
                frame_idx,
                depth_map,
                frame_img.shape[0],
                frame_img.shape[1],
            )
            _append_depth_ndjson(output_dir_path, record)

    depth_vis_output: Optional[str] = None
    if save_depth_video:
        save_video(
            depths,
            str(depth_vis_path),
            fps=float(fps) if fps else 30.0,
            is_depths=True,
            grayscale=grayscale,
        )
        depth_vis_output = str(depth_vis_path)

    depth_frame_paths: List[str] = []
    if save_depth_frames:
        depth_frame_paths = _save_depth_frames(depths, output_dir_path, grayscale)

    result: Dict[str, Optional[List[str]]] = {
        "output_dir": [str(output_dir_path)],
    }
    if depth_vis_output:
        result["depth_visualization"] = [depth_vis_output]
    if depth_frame_paths:
        result["depth_frames"] = depth_frame_paths
    ndjson_path = output_dir_path / "depth.ndjson"
    if ndjson_path.exists():
        result["depth_ndjson"] = [str(ndjson_path)]

    if save_npz:
        depth_npz_path = output_dir_path / f"{base_name}_depths.npz"
        np.savez_compressed(depth_npz_path, depths=depths)
        result["npz"] = [str(depth_npz_path)]

    if save_exr:
        try:
            import OpenEXR  # noqa: F401
            import Imath  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Saving EXR requires OpenEXR and Imath.") from exc
        depth_exr_dir = output_dir_path / f"{base_name}_depths_exr"
        depth_exr_dir.mkdir(exist_ok=True)
        header = None
        for idx, depth in enumerate(depths):
            if header is None:
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
            output_exr = depth_exr_dir / f"frame_{idx:05d}.exr"
            exr_file = OpenEXR.OutputFile(str(output_exr), header)
            exr_file.writePixels({"Z": depth.astype(np.float32).tobytes()})
            exr_file.close()
        result["exr"] = [str(depth_exr_dir)]

    # always write point cloud csv files alongside depth results
    if depths is not None and len(depths):
        width, height = depths[0].shape[-1], depths[0].shape[-2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
    if save_point_clouds:
        if depths is not None and len(depths):
            width, height = depths[0].shape[-1], depths[0].shape[-2]
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / focal_length_x
            y = (y - height / 2) / focal_length_y
            point_csv_paths: List[str] = []
            ply_dir = output_dir_path / "point_clouds"
            for idx, (color_image, depth_map) in enumerate(zip(frames, depths)):
                z = depth_map.astype(np.float32)
                points = np.stack(
                    (
                        np.multiply(x, z),
                        np.multiply(y, z),
                        z,
                    ),
                    axis=-1,
                ).reshape(-1, 3)
                colors = np.array(color_image).reshape(-1, 3)
                csv_path = ply_dir / f"point_{idx:04d}.csv"
                frame_index = (
                    int(frame_indices[idx]) if idx < len(frame_indices) else idx
                )
                region_labels = None
                if include_region_labels and tracking_results_dir_path is not None:
                    region_labels = _frame_region_labels(
                        tracking_results_dir_path,
                        frame_index,
                        depth_map.shape[0],
                        depth_map.shape[1],
                        region_cache,
                    )
                _write_point_cloud_csv(
                    csv_path,
                    points,
                    z.reshape(-1, 1),
                    colors=colors,
                    region_labels=region_labels,
                )
                point_csv_paths.append(str(csv_path))
            if point_csv_paths:
                result["point_cloud_csv"] = point_csv_paths

    if fp32 and device.type == "cuda":
        torch.cuda.empty_cache()

    return result
