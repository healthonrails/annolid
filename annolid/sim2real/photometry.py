from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import VideoEnvironmentProfile


def composite_foreground_onto_background(
    sim_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    background_bgr: np.ndarray,
    *,
    feather_radius: int = 3,
    match_fg_lighting: bool = True,
) -> np.ndarray:
    """Alpha-composite rendered simulation subject(s) cleanly onto a real arena background.

    Preserves exact 2D spatial alignment of masks and keypoints while replacing the
    synthetic arena background with the user's actual recording environment.
    """
    h, w = sim_bgr.shape[:2]
    bg = background_bgr
    if bg.shape[0] != h or bg.shape[1] != w:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

    fg_bool = np.asarray(foreground_mask, dtype=bool)
    if not np.any(fg_bool):
        return bg.copy()

    fg_bgr = sim_bgr.copy()
    if match_fg_lighting:
        # Gently nudge foreground luminance toward background scale without crushing contrast.
        bg_luma = float(np.mean(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)))
        sim_bg_mask = ~fg_bool
        if np.any(sim_bg_mask):
            sim_gray = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2GRAY)
            sim_bg_luma = max(15.0, float(np.mean(sim_gray[sim_bg_mask])))
            gain = float(np.clip(bg_luma / sim_bg_luma, 0.55, 1.55))
            blended_gain = 0.65 + 0.35 * gain
            fg_bgr = np.clip(
                fg_bgr.astype(np.float32) * blended_gain, 0.0, 255.0
            ).astype(np.uint8)

    alpha = fg_bool.astype(np.float32)
    k = max(1, int(feather_radius))
    if k % 2 == 0:
        k += 1
    if k >= 3:
        alpha = cv2.GaussianBlur(alpha, (k, k), 0.7)
        # Keep interior opaque so semi-transparent body edges don't bleed background through.
        eroded = cv2.erode(
            fg_bool.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(bool)
        alpha[eroded] = 1.0

    # Add a subtle soft contact shadow beneath the foreground silhouette for realism.
    shadow_k = max(5, k * 2 + 1)
    shadow = cv2.GaussianBlur(fg_bool.astype(np.float32), (shadow_k, shadow_k), 2.0)
    shadow = np.clip(shadow * (1.0 - alpha), 0.0, 1.0)
    bg_shadowed = bg.astype(np.float32) * (1.0 - 0.22 * shadow[:, :, None])

    out = (
        alpha[:, :, None] * fg_bgr.astype(np.float32)
        + (1.0 - alpha[:, :, None]) * bg_shadowed
    )
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_environment_photometry(
    bgr: np.ndarray,
    profile: VideoEnvironmentProfile,
    *,
    strength: float = 0.75,
    domain_randomization: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Match a rendered synthetic BGR image to an example video's photometric profile.

    Pure pixel-intensity transformation: geometry, masks, and keypoints remain exact.
    """
    alpha = float(np.clip(strength, 0.0, 1.0))
    dr = float(np.clip(domain_randomization, 0.0, 1.0))
    if alpha <= 1e-4 and dr <= 1e-4:
        return bgr.copy()

    gen = rng if rng is not None else np.random.default_rng(0)
    out = bgr.copy()

    # 1. Reinhard Lab color/contrast transfer toward the real example video.
    if alpha > 1e-4 and len(profile.lab_mean) == 3 and len(profile.lab_std) == 3:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        src_mean = lab.reshape(-1, 3).mean(axis=0)
        src_std = np.maximum(lab.reshape(-1, 3).std(axis=0), 4.0)
        tgt_mean = np.asarray(profile.lab_mean, dtype=np.float32)
        tgt_std = np.maximum(np.asarray(profile.lab_std, dtype=np.float32), 4.0)

        if dr > 0.0:
            tgt_mean = tgt_mean + gen.normal(0.0, 10.0 * dr, size=3).astype(np.float32)
            tgt_std = tgt_std * (
                1.0 + gen.uniform(-0.18 * dr, 0.18 * dr, size=3)
            ).astype(np.float32)

        scale = np.clip(tgt_std / src_std, 0.45, 2.2)
        matched_lab = (lab - src_mean) * scale + tgt_mean
        blended_lab = (1.0 - alpha) * lab + alpha * matched_lab
        blended_lab = np.clip(blended_lab, 0.0, 255.0).astype(np.uint8)
        out = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)

    # 2. Match monochrome / IR desaturation if the example recording is grayscale.
    if profile.is_grayscale and alpha > 0.15:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        desat_w = min(1.0, alpha * 1.15)
        out = cv2.addWeighted(gray_bgr, desat_w, out, 1.0 - desat_w, 0.0)

    # 3. Match radial vignette dropoff.
    vig = float(profile.vignette_strength) * alpha
    if vig > 0.04:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ny = (yy - 0.5 * h) / max(1.0, 0.5 * h)
        nx = (xx - 0.5 * w) / max(1.0, 0.5 * w)
        r2 = np.clip(nx * nx + ny * ny, 0.0, 2.0) / 2.0
        gain = np.clip(1.0 - vig * 0.75 * r2, 0.35, 1.0)
        out = np.clip(out.astype(np.float32) * gain[:, :, None], 0.0, 255.0).astype(
            np.uint8
        )

    # 4. Mild optical defocus matching.
    blur_sigma = float(profile.estimated_blur_sigma) * alpha
    if dr > 0.0:
        blur_sigma = max(0.0, blur_sigma + float(gen.uniform(-0.2, 0.35)) * dr)
    if blur_sigma >= 0.35:
        k = max(3, int(round(blur_sigma * 3.0)) | 1)
        out = cv2.GaussianBlur(out, (k, k), blur_sigma)

    # 5. Sensor noise matching.
    noise_sigma = float(profile.estimated_noise_sigma) * alpha
    if dr > 0.0:
        noise_sigma = max(0.0, noise_sigma + float(gen.uniform(0.0, 3.0)) * dr)
    if noise_sigma >= 0.8:
        noise = gen.normal(0.0, min(14.0, noise_sigma), size=out.shape).astype(
            np.float32
        )
        out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    return out


def load_reference_background(
    profile: Optional[VideoEnvironmentProfile],
) -> Optional[np.ndarray]:
    if profile is None or not profile.reference_background_path:
        return None
    p = Path(profile.reference_background_path)
    if not p.is_file():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)
