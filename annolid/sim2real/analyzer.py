from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .models import SimDatasetConfig, VideoEnvironmentProfile

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
)


def _sample_frames_from_source(
    source_path: Path, max_samples: int = 24
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Sample evenly spaced BGR frames from a video file or directory of images."""
    if source_path.is_dir():
        img_paths = sorted(
            p
            for p in source_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not img_paths:
            raise ValueError(f"No image files found in directory: {source_path}")
        step = max(1, len(img_paths) // max(1, max_samples))
        chosen = img_paths[::step][:max_samples]
        frames: list[np.ndarray] = []
        for p in chosen:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        if not frames:
            raise ValueError(f"Failed to decode images in directory: {source_path}")
        h, w = frames[0].shape[:2]
        return frames, {
            "width": float(w),
            "height": float(h),
            "fps": 10.0,
            "total_frames": float(len(img_paths)),
        }

    if not source_path.is_file():
        raise FileNotFoundError(f"Example video/image source not found: {source_path}")

    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        img = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unable to read image file: {source_path}")
        h, w = img.shape[:2]
        return [img], {
            "width": float(w),
            "height": float(h),
            "fps": 1.0,
            "total_frames": 1.0,
        }

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {source_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not math.isfinite(fps) or fps <= 0.1:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frames = []
        if total > 0:
            n_pick = min(max(1, max_samples), total)
            indices = np.linspace(0, max(0, total - 1), num=n_pick, dtype=int)
            seen: set[int] = set()
            for idx in indices.tolist():
                if idx in seen:
                    continue
                seen.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        if not frames:
            # Fallback sequential read for containers with unknown frame count.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            read_count = 0
            while len(frames) < max_samples and read_count < max_samples * 10:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if read_count % 2 == 0:
                    frames.append(frame)
                read_count += 1
            if total <= 0:
                total = max(len(frames), read_count)
        if not frames:
            raise ValueError(f"No readable frames decoded from video: {source_path}")
        if width <= 0 or height <= 0:
            height, width = frames[0].shape[:2]
        return frames, {
            "width": float(width),
            "height": float(height),
            "fps": float(fps),
            "total_frames": float(max(total, len(frames))),
        }
    finally:
        cap.release()


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Robust MAD noise estimate using a 3x3 high-pass Laplacian kernel (Immerkaer 1996)."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
    sigma = float(np.median(np.abs(conv))) * math.sqrt(math.pi / 2.0) / (6.0 * 0.6745)
    return max(0.0, min(40.0, sigma))


def _estimate_vignette_strength(gray: np.ndarray) -> float:
    """Estimate radial illumination dropoff between image center and corners in [0, 1]."""
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    cy0, cy1 = int(h * 0.30), int(h * 0.70)
    cx0, cx1 = int(w * 0.30), int(w * 0.70)
    center_mean = float(np.mean(gray[cy0:cy1, cx0:cx1]))
    ch, cw = max(4, h // 8), max(4, w // 8)
    corners = np.concatenate(
        [
            gray[:ch, :cw].ravel(),
            gray[:ch, -cw:].ravel(),
            gray[-ch:, :cw].ravel(),
            gray[-ch:, -cw:].ravel(),
        ]
    )
    corner_mean = float(np.mean(corners))
    if center_mean <= 1.0:
        return 0.0
    dropoff = max(0.0, (center_mean - corner_mean) / max(center_mean, 1.0))
    return round(min(1.0, dropoff), 4)


def _detect_circular_arena(bg_gray: np.ndarray) -> bool:
    """Detect prominent circular arena enclosure via Hough circles."""
    h, w = bg_gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 48:
        return False
    blurred = cv2.GaussianBlur(bg_gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim * 0.4,
        param1=90,
        param2=42,
        minRadius=int(min_dim * 0.28),
        maxRadius=int(min_dim * 0.52),
    )
    if circles is None or len(circles) == 0:
        return False
    cx, cy, r = circles[0][0]
    near_center = abs(cx - w * 0.5) < w * 0.18 and abs(cy - h * 0.5) < h * 0.18
    return bool(near_center and r >= min_dim * 0.28)


def _detect_light_dark_box(bg_gray: np.ndarray) -> bool:
    """Detect split light/dark two-compartment box along horizontal or vertical axis."""
    h, w = bg_gray.shape[:2]
    if h < 24 or w < 24:
        return False
    left = float(np.mean(bg_gray[:, : w // 2]))
    right = float(np.mean(bg_gray[:, w // 2 :]))
    top = float(np.mean(bg_gray[: h // 2, :]))
    bottom = float(np.mean(bg_gray[h // 2 :, :]))
    split_h = abs(left - right)
    split_v = abs(top - bottom)
    # Require one side to be genuinely dark (<65) and the other bright (>135)
    if split_h > 75.0 and min(left, right) < 65.0 and max(left, right) > 135.0:
        return True
    if split_v > 75.0 and min(top, bottom) < 65.0 and max(top, bottom) > 135.0:
        return True
    return False


def _detect_wire_bars(bg_gray: np.ndarray) -> bool:
    """Detect repeating parallel high-contrast overhead wire-bar occluders."""
    h, w = bg_gray.shape[:2]
    if h < 48 or w < 48:
        return False
    # Contrast-normalize before Canny so dark IR wire-bar lids are detected reliably.
    norm = cv2.normalize(bg_gray, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(norm, 50, 140)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(40, int(min(h, w) * 0.35)),
        minLineLength=int(min(h, w) * 0.45),
        maxLineGap=12,
    )
    if lines is None:
        return False
    horiz_ys: list[float] = []
    vert_xs: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > dy * 6.0:
            horiz_ys.append(0.5 * (y1 + y2))
        elif dy > dx * 6.0:
            vert_xs.append(0.5 * (x1 + x2))

    def _has_periodic_bars(coords: list[float], span: float) -> bool:
        if len(coords) < 5:
            return False
        coords_sorted = sorted(coords)
        merged: list[float] = [coords_sorted[0]]
        for c in coords_sorted[1:]:
            if c - merged[-1] > max(6.0, span * 0.03):
                merged.append(c)
        if len(merged) < 5:
            return False
        gaps = np.diff(merged)
        med = float(np.median(gaps))
        if med <= 0:
            return False
        cv_gap = float(np.std(gaps) / med)
        return cv_gap < 0.35

    return _has_periodic_bars(horiz_ys, float(h)) or _has_periodic_bars(
        vert_xs, float(w)
    )


def _analyze_foreground_subjects(
    frames_bgr: Sequence[np.ndarray], bg_bgr: np.ndarray
) -> dict[str, float]:
    """Estimate subject count, relative footprint, and subject luminance from frame diffs."""
    h, w = bg_bgr.shape[:2]
    frame_area = float(max(1, h * w))
    bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    blob_counts: list[int] = []
    blob_area_ratios: list[float] = []
    blob_lumas: list[float] = []

    min_blob_px = max(12, int(frame_area * 0.00025))
    max_blob_px = int(frame_area * 0.45)

    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(gray - bg_gray)
        thr = max(16.0, float(np.percentile(diff, 96.5)) * 0.55)
        mask = (diff >= thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates: list[tuple[int, int]] = []
        for lbl_idx in range(1, n_labels):
            area = int(stats[lbl_idx, cv2.CC_STAT_AREA])
            if min_blob_px <= area <= max_blob_px:
                candidates.append((area, lbl_idx))
        if candidates:
            # Drop tiny fragmented limb/reflection specks (< 15% of largest moving blob)
            max_area = max(a for a, _ in candidates)
            area_floor = max(min_blob_px, int(max_area * 0.15))
            kept = [(a, idx) for a, idx in candidates if a >= area_floor]
            blob_counts.append(len(kept))
            for area, lbl_idx in kept:
                blob_area_ratios.append(area / frame_area)
                blob_lumas.append(float(np.mean(gray[labels == lbl_idx])))

    est_count = int(round(float(np.median(blob_counts)))) if blob_counts else 1
    est_count = max(1, min(12, est_count))
    med_area = float(np.median(blob_area_ratios)) if blob_area_ratios else 0.025
    luma_mean = float(np.mean(blob_lumas)) if blob_lumas else 85.0
    luma_spread = float(np.std(blob_lumas)) if len(blob_lumas) >= 2 else 0.0
    return {
        "subject_count": float(est_count),
        "median_area_ratio": med_area,
        "subject_luma_mean": luma_mean,
        "subject_luma_spread": luma_spread,
    }


def _save_sample_montage(
    frames_bgr: Sequence[np.ndarray], output_path: Path, max_tiles: int = 4
) -> str:
    if not frames_bgr:
        return ""
    indices = np.linspace(
        0, len(frames_bgr) - 1, num=min(max_tiles, len(frames_bgr)), dtype=int
    )
    tiles: list[np.ndarray] = []
    target_w, target_h = 320, 240
    for idx in indices.tolist():
        tile = cv2.resize(
            frames_bgr[idx], (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            tile,
            f"Sample {len(tiles) + 1}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if len(tiles) == 1:
        montage = tiles[0]
    elif len(tiles) == 2:
        montage = np.concatenate(tiles, axis=1)
    else:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        top = np.concatenate(tiles[:2], axis=1)
        bot = np.concatenate(tiles[2:4], axis=1)
        montage = np.concatenate([top, bot], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), montage)
    return str(output_path)


def analyze_example_videos(
    video_paths: Iterable[str | Path],
    *,
    species_hint: str = "auto",
    output_dir: Optional[str | Path] = None,
    max_samples_per_video: int = 24,
) -> VideoEnvironmentProfile:
    """Inspect real example videos/image folders and infer matching 3D simulation settings."""
    resolved_paths = [Path(p).expanduser().resolve() for p in video_paths]
    if not resolved_paths:
        raise ValueError("Provide at least one example video or image directory.")

    all_frames: list[np.ndarray] = []
    ref_width = 0
    ref_height = 0
    ref_fps = 30.0
    total_frames_sum = 0

    for idx, src in enumerate(resolved_paths):
        frames, meta = _sample_frames_from_source(
            src, max_samples=max_samples_per_video
        )
        if idx == 0:
            ref_width = int(meta["width"])
            ref_height = int(meta["height"])
            ref_fps = float(meta["fps"])
        total_frames_sum += int(meta["total_frames"])
        for f in frames:
            if f.shape[1] != ref_width or f.shape[0] != ref_height:
                f = cv2.resize(f, (ref_width, ref_height), interpolation=cv2.INTER_AREA)
            all_frames.append(f)

    # Downscale analysis stack for fast, low-memory median background & stats.
    analysis_w = min(ref_width, 480)
    analysis_h = max(16, int(round(ref_height * (analysis_w / max(1, ref_width)))))
    resized = [
        cv2.resize(f, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA)
        for f in all_frames
    ]
    stack = np.stack(resized, axis=0)
    bg_small = np.median(stack, axis=0).astype(np.uint8)
    bg_full = cv2.resize(
        bg_small, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR
    )

    # Color, luminance, sharpness, noise, and vignette measurements
    lab_means: list[np.ndarray] = []
    lab_stds: list[np.ndarray] = []
    rgb_means: list[np.ndarray] = []
    rgb_stds: list[np.ndarray] = []
    sats: list[float] = []
    lumas: list[float] = []
    luma_stds: list[float] = []
    sharpnesses: list[float] = []
    noises: list[float] = []
    vignettes: list[float] = []
    channel_diffs: list[float] = []

    for frame in resized:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rgb_f = rgb.astype(np.float32)
        rgb_means.append(rgb_f.reshape(-1, 3).mean(axis=0))
        rgb_stds.append(rgb_f.reshape(-1, 3).std(axis=0))
        lab_means.append(lab.reshape(-1, 3).mean(axis=0))
        lab_stds.append(lab.reshape(-1, 3).std(axis=0))
        sats.append(float(np.mean(hsv[:, :, 1])))
        lumas.append(float(np.mean(gray)))
        luma_stds.append(float(np.std(gray)))
        sharpnesses.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        noises.append(_estimate_noise_sigma(gray))
        vignettes.append(_estimate_vignette_strength(gray))
        ch_diff = (
            np.mean(np.abs(rgb_f[:, :, 0] - rgb_f[:, :, 1]))
            + np.mean(np.abs(rgb_f[:, :, 1] - rgb_f[:, :, 2]))
        ) * 0.5
        channel_diffs.append(float(ch_diff))

    mean_luma = float(np.mean(lumas))
    std_luma = float(np.mean(luma_stds))
    mean_sat = float(np.mean(sats))
    mean_ch_diff = float(np.mean(channel_diffs))
    mean_sharpness = float(np.mean(sharpnesses))
    mean_noise = float(np.mean(noises))
    mean_vignette = float(np.mean(vignettes))
    rgb_mean_vec = np.mean(rgb_means, axis=0)
    rgb_std_vec = np.mean(rgb_stds, axis=0)
    lab_mean_vec = np.mean(lab_means, axis=0)
    lab_std_vec = np.mean(lab_stds, axis=0)

    # Robust monochrome/IR check: tolerates YUV420 chroma quantization drift on dark MP4s
    # where Lab a/b channels remain tightly centered near neutral 128 with near-zero std.
    lab_chroma_offset = max(
        abs(float(lab_mean_vec[1]) - 128.0), abs(float(lab_mean_vec[2]) - 128.0)
    )
    lab_chroma_std = max(float(lab_std_vec[1]), float(lab_std_vec[2]))
    is_grayscale = bool(
        mean_sat < 14.0
        or mean_ch_diff < 4.5
        or (lab_chroma_offset < 5.0 and lab_chroma_std < 3.0)
    )

    # Estimate blur sigma inversely from Laplacian variance
    if mean_sharpness < 25.0:
        est_blur = 1.6
    elif mean_sharpness < 60.0:
        est_blur = 0.9
    elif mean_sharpness < 120.0:
        est_blur = 0.4
    else:
        est_blur = 0.0

    bg_gray = cv2.cvtColor(bg_small, cv2.COLOR_BGR2GRAY)
    fg_stats = _analyze_foreground_subjects(resized, bg_small)
    est_count = int(fg_stats["subject_count"])
    med_area = float(fg_stats["median_area_ratio"])
    subj_luma = float(fg_stats["subject_luma_mean"])
    subj_spread = float(fg_stats["subject_luma_spread"])

    rationale: dict[str, str] = {}

    # Species classification (honoring explicit user hint)
    hint = str(species_hint or "auto").strip().lower()
    if hint in {"mouse", "fly"}:
        species = hint
        species_conf = 1.0
        rationale["species"] = f"Explicitly set by user (`{species}`)."
    else:
        # Tiny dark subjects on bright/circular arena strongly suggest Drosophila / fly assays.
        joined_names = " ".join(p.name.lower() for p in resolved_paths)
        if any(tok in joined_names for tok in ("fly", "drosoph", "wing", "flybody")):
            species = "fly"
            species_conf = 0.92
            rationale["species"] = "Matched fly/Drosophila cue in example filename."
        elif any(
            tok in joined_names
            for tok in ("mouse", "mice", "rodent", "calms", "ventral")
        ):
            species = "mouse"
            species_conf = 0.92
            rationale["species"] = "Matched mouse/rodent cue in example filename."
        elif med_area < 0.0045 and mean_luma > 135.0:
            species = "fly"
            species_conf = 0.72
            rationale["species"] = (
                f"Small subject footprint ({med_area:.4f} of frame) on bright arena "
                f"(luminance {mean_luma:.1f}) suggests fly/insect assay."
            )
        else:
            species = "mouse"
            species_conf = 0.68
            rationale["species"] = (
                f"Subject footprint ({med_area:.4f} of frame) aligns with rodent/mouse assay."
            )

    # Lighting preset inference
    if is_grayscale:
        if mean_luma < 85.0:
            lighting = "ir_night"
            rationale["lighting"] = (
                f"Low-luminance monochrome footage (mean={mean_luma:.1f}, sat={mean_sat:.1f}) -> `ir_night`."
            )
        else:
            lighting = "calms_gray"
            rationale["lighting"] = (
                f"Monochrome/grayscale footage (mean={mean_luma:.1f}, sat={mean_sat:.1f}) -> `calms_gray`."
            )
    else:
        warm_bias = float(rgb_mean_vec[0] - rgb_mean_vec[2])
        if mean_luma < 92.0 and warm_bias > 12.0:
            lighting = "dim_warm"
            rationale["lighting"] = (
                f"Dim warm illumination (mean={mean_luma:.1f}, R-B={warm_bias:.1f}) -> `dim_warm`."
            )
        elif mean_luma < 65.0:
            lighting = "ir_night"
            rationale["lighting"] = (
                f"Very dark lighting (mean={mean_luma:.1f}) -> `ir_night`."
            )
        else:
            lighting = "default"
            rationale["lighting"] = (
                f"Standard color illumination (mean={mean_luma:.1f}, sat={mean_sat:.1f}) -> `default`."
            )

    # Arena, camera, coat, occluder, and scenarios
    joined_names = " ".join(p.name.lower() for p in resolved_paths)
    has_wire_bars = _detect_wire_bars(bg_gray)
    occluder = "wire_bars" if has_wire_bars else "none"
    if has_wire_bars:
        rationale["occluder"] = (
            "Detected periodic high-contrast linear bars -> `wire_bars`."
        )
    else:
        rationale["occluder"] = "No periodic wire-bar occluder detected -> `none`."

    if species == "fly":
        arena = (
            "circular_open_field" if _detect_circular_arena(bg_gray) else "open_field"
        )
        if "side" in joined_names or "lateral" in joined_names:
            camera_view = "walker/side"
            rationale["camera_view"] = (
                "Filename indicates lateral/side recording -> `walker/side`."
            )
        elif "back" in joined_names or "rear" in joined_names:
            camera_view = "walker/back"
            rationale["camera_view"] = "Filename indicates rear view -> `walker/back`."
        elif med_area > 0.04:
            camera_view = "walker/hero"
            rationale["camera_view"] = (
                f"Large close-up fly footprint ({med_area:.3f}) -> `walker/hero`."
            )
        else:
            camera_view = "walker/track1"
            rationale["camera_view"] = (
                f"Arena/tracking fly footprint ({med_area:.4f}) -> `walker/track1`."
            )
        coat_pair = "default"
        headstage = "none"
        if "flight" in joined_names or "takeoff" in joined_names:
            scenarios = ["escape_takeoff", "flight_imitation"]
        elif "groom" in joined_names:
            scenarios = ["grooming", "walk_imitation"]
        else:
            scenarios = ["walk_imitation", "forage_walk", "grooming"]
        rationale["arena"] = f"Inferred `{arena}` fly arena geometry."
    else:
        # Mouse arena geometry
        if "home" in joined_names or "cage" in joined_names or has_wire_bars:
            arena = "home_cage"
            rationale["arena"] = "Home-cage / wire-bar cues detected -> `home_cage`."
        elif "plus" in joined_names or "epm" in joined_names:
            arena = "elevated_plus_maze"
            rationale["arena"] = (
                "Elevated plus maze cue detected -> `elevated_plus_maze`."
            )
        elif "y_maze" in joined_names or "ymaze" in joined_names:
            arena = "y_maze"
            rationale["arena"] = "Y-maze cue detected -> `y_maze`."
        elif "t_maze" in joined_names or "tmaze" in joined_names:
            arena = "t_maze"
            rationale["arena"] = "T-maze cue detected -> `t_maze`."
        elif "three_chamber" in joined_names:
            arena = "three_chamber"
            rationale["arena"] = "Three-chamber cue detected -> `three_chamber`."
        elif "barnes" in joined_names:
            arena = "barnes_maze"
            rationale["arena"] = "Barnes maze cue detected -> `barnes_maze`."
        elif _detect_light_dark_box(bg_gray):
            arena = "light_dark_box"
            rationale["arena"] = (
                "Detected strong split dark/bright compartments -> `light_dark_box`."
            )
        elif _detect_circular_arena(bg_gray):
            arena = "circular_open_field"
            rationale["arena"] = (
                "Detected circular enclosure boundary -> `circular_open_field`."
            )
        else:
            arena = "open_field"
            rationale["arena"] = "Rectangular arena profile -> `open_field`."

        # Mouse camera angle
        if (
            "ventral" in joined_names
            or "bottom" in joined_names
            or "gait" in joined_names
        ):
            camera_view = "bottom-up"
            rationale["camera_view"] = "Ventral/bottom-up gait cue -> `bottom-up`."
        elif (
            "oblique" in joined_names
            or "rear" in joined_names
            or "side" in joined_names
        ):
            camera_view = "oblique-close" if med_area > 0.03 else "oblique"
            rationale["camera_view"] = f"Oblique/rearing cue -> `{camera_view}`."
        elif arena == "home_cage":
            camera_view = "home-cage-top"
            rationale["camera_view"] = (
                "Matched home-cage overhead camera -> `home-cage-top`."
            )
        elif med_area > 0.035:
            camera_view = "top-close"
            rationale["camera_view"] = (
                f"Larger subject footprint ({med_area:.3f}) -> `top-close`."
            )
        else:
            camera_view = "top"
            rationale["camera_view"] = (
                f"Wider arena framing (footprint {med_area:.3f}) -> `top`."
            )

        # Mouse coat pair inference from foreground luminance distribution
        if subj_spread > 32.0:
            coat_pair = "default"
            rationale["coat_pair"] = (
                f"Bimodal subject brightness (spread={subj_spread:.1f}) -> mixed black+white (`default`)."
            )
        elif subj_luma < 72.0:
            coat_pair = "c57_pair"
            rationale["coat_pair"] = (
                f"Dark subject luminance (mean={subj_luma:.1f}) -> `c57_pair` (C57BL/6)."
            )
        elif subj_luma > 165.0:
            coat_pair = "balbc_pair"
            rationale["coat_pair"] = (
                f"Bright subject luminance (mean={subj_luma:.1f}) -> `balbc_pair` (BALB/c)."
            )
        else:
            coat_pair = "default"
            rationale["coat_pair"] = (
                f"Mid-tone subject luminance (mean={subj_luma:.1f}) -> `default` (mixed strains)."
            )

        headstage = (
            "resident"
            if any(
                k in joined_names
                for k in ("miniscope", "headstage", "implant", "fiber")
            )
            else "none"
        )
        if camera_view == "bottom-up":
            scenarios = ["ventral_gait", "arena_explore"]
        elif lighting == "calms_gray" or arena == "home_cage":
            scenarios = ["calms_resident_intruder", "social_huddle", "curled_groom"]
        elif camera_view.startswith("oblique"):
            scenarios = ["rearing_social", "arena_explore", "social_huddle"]
        else:
            scenarios = ["arena_explore", "social_huddle", "curled_groom"]

    ref_bg_path = ""
    montage_path = ""
    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        bg_file = out / "reference_background.png"
        cv2.imwrite(str(bg_file), bg_full)
        ref_bg_path = str(bg_file)
        montage_path = _save_sample_montage(
            all_frames, out / "example_samples.png", max_tiles=4
        )

    return VideoEnvironmentProfile(
        example_videos=[str(p) for p in resolved_paths],
        species=species,
        species_confidence=round(species_conf, 3),
        width=ref_width,
        height=ref_height,
        fps=round(ref_fps, 2),
        total_frames=total_frames_sum,
        sampled_frames=len(all_frames),
        is_grayscale=is_grayscale,
        mean_luminance=round(mean_luma, 2),
        std_luminance=round(std_luma, 2),
        mean_saturation=round(mean_sat, 2),
        lab_mean=[round(float(x), 2) for x in lab_mean_vec.tolist()],
        lab_std=[round(float(x), 2) for x in lab_std_vec.tolist()],
        rgb_mean=[round(float(x), 2) for x in rgb_mean_vec.tolist()],
        rgb_std=[round(float(x), 2) for x in rgb_std_vec.tolist()],
        sharpness_laplacian_var=round(mean_sharpness, 2),
        estimated_blur_sigma=round(est_blur, 3),
        estimated_noise_sigma=round(mean_noise, 3),
        vignette_strength=round(mean_vignette, 4),
        estimated_subject_count=est_count,
        median_subject_area_ratio=round(med_area, 5),
        subject_luminance_mean=round(subj_luma, 2),
        subject_luminance_spread=round(subj_spread, 2),
        recommended_arena=arena,
        recommended_camera_view=camera_view,
        recommended_lighting=lighting,
        recommended_coat_pair=coat_pair,
        recommended_occluder=occluder,
        recommended_headstage=headstage,
        recommended_scenarios=scenarios,
        reference_background_path=ref_bg_path,
        sample_montage_path=montage_path,
        match_rationale=rationale,
    )


def build_default_sim_config(
    profile: VideoEnvironmentProfile,
    *,
    task: str = "yolo_seg",
    export_level: str = "instances",
    frames_per_scenario: int = 24,
    seed: int = 42,
) -> SimDatasetConfig:
    """Create an initial SimDatasetConfig aligned to an analyzed VideoEnvironmentProfile."""
    # Cap simulation capture size to reasonable bounds while preserving aspect ratio.
    target_w = min(max(320, int(profile.width)), 960)
    aspect = float(profile.height) / float(max(1, profile.width))
    target_h = max(240, int(round(target_w * aspect)))
    # Ensure even dimensions for H.264/MP4V video encoding.
    target_w += target_w % 2
    target_h += target_h % 2

    return SimDatasetConfig(
        species=profile.species,
        task=task,
        export_level=export_level,
        arena=profile.recommended_arena,
        camera_view=profile.recommended_camera_view,
        lighting=profile.recommended_lighting,
        coat_pair=profile.recommended_coat_pair,
        occluder=profile.recommended_occluder,
        headstage=profile.recommended_headstage,
        scenarios=list(profile.recommended_scenarios),
        frames_per_scenario=max(6, int(frames_per_scenario)),
        width=target_w,
        height=target_h,
        fps=8,
        seed=int(seed),
        match_example_photometry=True,
        photometry_strength=0.75,
        composite_on_real_background=bool(
            profile.species == "fly" and bool(profile.reference_background_path)
        ),
        domain_randomization=0.25,
        fly_subject_count=max(1, min(6, int(profile.estimated_subject_count))),
    )
