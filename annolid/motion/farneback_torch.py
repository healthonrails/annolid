"""
Minimal PyTorch Farneback optical flow (CPU/GPU) with optional OpenCV fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    """
    Convert input image to single-channel float32 (OpenCV-compatible).
    Handles 2D, (H,W,1), (H,W,3) BGR, and (H,W,4) BGRA by dropping alpha.
    """
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            gray = img[..., 0]
        elif c >= 3:
            b, g, r = img[..., 0], img[..., 1], img[..., 2]
            gray = b * 0.114 + g * 0.587 + r * 0.299
        else:
            raise ValueError(
                f"Unexpected channel count for gray conversion: {c}")
    else:
        raise ValueError(
            f"Unexpected image shape for gray conversion: {img.shape}")
    return np.ascontiguousarray(gray, dtype=np.float32)


def _torch_gray(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H,W) float32 numpy -> (1,1,H,W) float32 tensor."""
    t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    if t.ndim != 2:
        raise ValueError("Expected (H,W) gray image.")
    return t.unsqueeze(0).unsqueeze(0)


def _replicate_pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    return F.pad(x, (pad, pad, pad, pad), mode="replicate")


@dataclass(frozen=True)
class FarnebackGaussKernels:
    n: int
    sigma: float
    g: torch.Tensor    # (2n+1) for offsets [-n..n]
    xg: torch.Tensor   # (2n+1)
    xxg: torch.Tensor  # (2n+1)
    ig11: float
    ig03: float
    ig33: float
    ig55: float


def farneback_prepare_gaussian(n: int, sigma: float, device: torch.device) -> FarnebackGaussKernels:
    """
    Matches OpenCV FarnebackPrepareGaussian (optflowgf.cpp).
    """
    if sigma < 1e-7:
        sigma = float(n) * 0.3

    # Compute kernels and scalar inverse constants in float64 on CPU for stability.
    xs = np.arange(-n, n + 1, dtype=np.float64)
    g_np = np.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    g_np /= g_np.sum()
    xg_np = xs * g_np
    xxg_np = xs * xs * g_np

    # Build 6x6 moment matrix G as in OpenCV, then invert.
    m0 = float(g_np.sum())  # 1.0
    m2 = float((xxg_np).sum())
    m4 = float(((xs ** 4) * g_np).sum())
    g00 = m0 * m0
    g11 = m0 * m2
    g33 = m0 * m4
    g55 = m2 * m2

    G = np.zeros((6, 6), dtype=np.float64)
    G[0, 0] = g00
    G[1, 1] = g11
    G[3, 3] = g33
    G[5, 5] = g55

    G[2, 2] = g11
    G[0, 3] = G[0, 4] = g11
    G[3, 0] = G[4, 0] = g11
    G[4, 4] = g33
    G[3, 4] = G[4, 3] = g55

    invG = np.linalg.inv(G)
    ig11 = float(invG[1, 1])
    ig03 = float(invG[0, 3])
    ig33 = float(invG[3, 3])
    ig55 = float(invG[5, 5])

    g = torch.from_numpy(g_np.astype(np.float32)).to(device)
    xg = torch.from_numpy(xg_np.astype(np.float32)).to(device)
    xxg = torch.from_numpy(xxg_np.astype(np.float32)).to(device)

    return FarnebackGaussKernels(
        n=n,
        sigma=float(sigma),
        g=g,
        xg=xg,
        xxg=xxg,
        ig11=ig11,
        ig03=ig03,
        ig33=ig33,
        ig55=ig55,
    )


def farneback_polyexp(img_1x1hw: torch.Tensor, kernels: FarnebackGaussKernels) -> torch.Tensor:
    """
    Matches OpenCV FarnebackPolyExp output layout: (H,W,5).
    """
    if img_1x1hw.ndim != 4 or img_1x1hw.shape[1] != 1:
        raise ValueError("Expected (1,1,H,W)")

    n = kernels.n
    g = kernels.g
    xg = kernels.xg
    xxg = kernels.xxg
    ig11 = kernels.ig11
    ig03 = kernels.ig03
    ig33 = kernels.ig33
    ig55 = kernels.ig55

    device = img_1x1hw.device
    dtype = img_1x1hw.dtype
    assert dtype == torch.float32

    pad = n  # replicate border to match OpenCV

    def ky(v: torch.Tensor) -> torch.Tensor:
        return v.view(1, 1, -1, 1).to(device=device, dtype=dtype)

    def kx(v: torch.Tensor) -> torch.Tensor:
        return v.view(1, 1, 1, -1).to(device=device, dtype=dtype)

    imgp = _replicate_pad2d(img_1x1hw, pad)
    row0 = F.conv2d(imgp, ky(g))
    row1 = F.conv2d(imgp, ky(xg))
    row2 = F.conv2d(imgp, ky(xxg))

    row0p = _replicate_pad2d(row0, pad)
    row1p = _replicate_pad2d(row1, pad)
    row2p = _replicate_pad2d(row2, pad)

    b1 = F.conv2d(row0p, kx(g))[0, 0]       # even-even
    b2 = F.conv2d(row0p, kx(xg))[0, 0]      # even-odd
    b4 = F.conv2d(row0p, kx(xxg))[0, 0]     # even-even (x^2)
    b3 = F.conv2d(row1p, kx(g))[0, 0]       # odd-even
    b6 = F.conv2d(row1p, kx(xg))[0, 0]      # odd-odd (xy)
    b5 = F.conv2d(row2p, kx(g))[0, 0]       # even-even (y^2)

    # OpenCV stores 5 channels (no r1):
    # dst[0]=b3*ig11, dst[1]=b2*ig11,
    # dst[2]=b1*ig03 + b5*ig33, dst[3]=b1*ig03 + b4*ig33, dst[4]=b6*ig55
    c0 = b3 * ig11
    c1 = b2 * ig11
    c2 = b1 * ig03 + b5 * ig33
    c3 = b1 * ig03 + b4 * ig33
    c4 = b6 * ig55
    return torch.stack([c0, c1, c2, c3, c4], dim=-1).contiguous()


def farneback_update_matrices(R0_hw5: torch.Tensor, R1_hw5: torch.Tensor, flow_hw2: torch.Tensor) -> torch.Tensor:
    device = R0_hw5.device
    dtype = R0_hw5.dtype
    assert dtype == torch.float32
    H, W, C = R0_hw5.shape
    assert C == 5
    if flow_hw2.ndim == 4 and flow_hw2.shape[0] == 1:
        flow_hw2 = flow_hw2.squeeze(0)
    if flow_hw2.ndim == 3 and flow_hw2.shape[0] == 2 and flow_hw2.shape[-1] != 2:
        flow_hw2 = flow_hw2.permute(1, 2, 0)
    fH, fW = flow_hw2.shape[:2]
    if (H, W) != (fH, fW):
        Hc = min(H, fH)
        Wc = min(W, fW)
        R0_hw5 = R0_hw5[:Hc, :Wc, :]
        R1_hw5 = R1_hw5[:Hc, :Wc, :]
        flow_hw2 = flow_hw2[:Hc, :Wc, :]
        H, W = Hc, Wc
    if flow_hw2.shape != (H, W, 2):
        raise ValueError(
            f"Expected flow shape {(H, W, 2)}, got {tuple(flow_hw2.shape)}")

    ys = torch.arange(H, device=device, dtype=torch.float32).view(
        H, 1).expand(H, W)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(
        1, W).expand(H, W)

    dx = flow_hw2[..., 0]
    dy = flow_hw2[..., 1]
    fx = xs + dx
    fy = ys + dy

    x1 = torch.floor(fx).to(torch.int64)
    y1 = torch.floor(fy).to(torch.int64)

    valid = (x1 >= 0) & (x1 < W - 1) & (y1 >= 0) & (y1 < H - 1)

    x1c = torch.clamp(x1, 0, W - 2)
    y1c = torch.clamp(y1, 0, H - 2)

    fx_frac = (fx - x1.to(torch.float32))
    fy_frac = (fy - y1.to(torch.float32))
    a00 = (1.0 - fx_frac) * (1.0 - fy_frac)
    a01 = fx_frac * (1.0 - fy_frac)
    a10 = (1.0 - fx_frac) * fy_frac
    a11 = fx_frac * fy_frac

    flat = R1_hw5.reshape(H * W, 5)
    idx00 = (y1c * W + x1c).view(-1)
    idx01 = (y1c * W + (x1c + 1)).view(-1)
    idx10 = ((y1c + 1) * W + x1c).view(-1)
    idx11 = ((y1c + 1) * W + (x1c + 1)).view(-1)

    R00 = flat[idx00].view(H, W, 5)
    R01 = flat[idx01].view(H, W, 5)
    R10 = flat[idx10].view(H, W, 5)
    R11 = flat[idx11].view(H, W, 5)

    R1w = (
        R00 * a00.unsqueeze(-1)
        + R01 * a01.unsqueeze(-1)
        + R10 * a10.unsqueeze(-1)
        + R11 * a11.unsqueeze(-1)
    )
    R1w = R1w * valid.unsqueeze(-1)

    # Match OpenCV FarnebackUpdateMatrices exactly (optflowgf.cpp).
    r2 = R1w[..., 0]
    r3 = R1w[..., 1]
    r4 = R1w[..., 2]
    r5 = R1w[..., 3]
    r6 = R1w[..., 4]

    in_bounds = valid
    # When in bounds: average quadratic terms with R0; r6 has 0.25 factor.
    r4 = torch.where(in_bounds, (R0_hw5[..., 2] + r4) * 0.5, R0_hw5[..., 2])
    r5 = torch.where(in_bounds, (R0_hw5[..., 3] + r5) * 0.5, R0_hw5[..., 3])
    r6 = torch.where(in_bounds, (R0_hw5[..., 4] + r6) * 0.25,
                     R0_hw5[..., 4] * 0.5)

    r2 = torch.where(in_bounds, r2, torch.zeros_like(r2))
    r3 = torch.where(in_bounds, r3, torch.zeros_like(r3))

    r2 = (R0_hw5[..., 0] - r2) * 0.5
    r3 = (R0_hw5[..., 1] - r3) * 0.5

    r2 = r2 + r4 * dy + r6 * dx
    r3 = r3 + r6 * dy + r5 * dx

    BORDER = 5
    border = torch.tensor(
        [0.14, 0.14, 0.4472, 0.4472, 0.4472],
        device=device,
        dtype=torch.float32,
    )
    if H > BORDER * 2 and W > BORDER * 2:
        sx = torch.ones((W,), device=device, dtype=torch.float32)
        sy = torch.ones((H,), device=device, dtype=torch.float32)
        sx[:BORDER] = border
        sx[-BORDER:] = torch.flip(border, dims=[0])
        sy[:BORDER] = border
        sy[-BORDER:] = torch.flip(border, dims=[0])
        scale = sy.view(H, 1) * sx.view(1, W)
        r2 = r2 * scale
        r3 = r3 * scale
        r4 = r4 * scale
        r5 = r5 * scale
        r6 = r6 * scale

    matM = torch.stack(
        [
            r4 * r4 + r6 * r6,
            (r4 + r5) * r6,
            r5 * r5 + r6 * r6,
            r4 * r2 + r6 * r3,
            r6 * r2 + r5 * r3,
        ],
        dim=-1,
    )
    return matM.contiguous()


def farneback_update_flow_blur(matM_hw5: torch.Tensor) -> torch.Tensor:
    g11 = matM_hw5[..., 0]
    g12 = matM_hw5[..., 1]
    g22 = matM_hw5[..., 2]
    h1 = matM_hw5[..., 3]
    h2 = matM_hw5[..., 4]
    idet = 1.0 / (g11 * g22 - g12 * g12 + 1e-3)
    # Match OpenCV FarnebackUpdateFlow_Blur:
    # flow_x = (g11*h2 - g12*h1)/det
    # flow_y = (g22*h1 - g12*h2)/det
    fx = (g11 * h2 - g12 * h1) * idet
    fy = (g22 * h1 - g12 * h2) * idet
    return torch.stack([fx, fy], dim=-1).contiguous()


OPTFLOW_FARNEBACK_GAUSSIAN = 256


def calc_optical_flow_farneback_torch(
    prev: np.ndarray,
    nxt: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 1,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.1,
    flags: int = 0,
    device: Optional[str] = None,
    clip_percentile: float = 99.0,
    max_magnitude: Optional[float] = None,
    outlier_ksize: int = 7,
    outlier_ratio: float = 25.0,
    outlier_min_magnitude: float = 10.0,
) -> np.ndarray:
    """
    PyTorch Farneback that aims to numerically match OpenCV's CPU implementation.
    Optionally clamps extreme flow magnitudes by percentile to suppress outliers.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    prev_f = _to_gray_f32(prev)
    nxt_f = _to_gray_f32(nxt)

    # OpenCV builds each level by Gaussian-blurring the original image (with a level-dependent
    # sigma) and then resizing; do the same for closer numerical behavior.
    pyr_prev: list[np.ndarray] = []
    pyr_nxt: list[np.ndarray] = []
    H0, W0 = prev_f.shape[:2]
    for lvl in range(0, levels + 1):
        scale = pyr_scale ** lvl
        sigma = (1.0 / scale - 1.0) * 0.5 if scale > 0 else 0.0
        smooth_sz = int(round(sigma * 5.0)) | 1
        smooth_sz = max(smooth_sz, 3)
        w = max(1, int(round(W0 * scale)))
        h = max(1, int(round(H0 * scale)))
        prev_blur = cv2.GaussianBlur(
            prev_f, (smooth_sz, smooth_sz), sigmaX=sigma, sigmaY=sigma)
        nxt_blur = cv2.GaussianBlur(
            nxt_f, (smooth_sz, smooth_sz), sigmaX=sigma, sigmaY=sigma)
        pyr_prev.append(cv2.resize(
            prev_blur, (w, h), interpolation=cv2.INTER_LINEAR))
        pyr_nxt.append(cv2.resize(
            nxt_blur, (w, h), interpolation=cv2.INTER_LINEAR))

    kernels = farneback_prepare_gaussian(poly_n, poly_sigma, dev)

    flow_t: Optional[torch.Tensor] = None

    for lvl in reversed(range(levels + 1)):
        I0 = pyr_prev[lvl]
        I1 = pyr_nxt[lvl]

        t0 = _torch_gray(I0, dev)
        t1 = _torch_gray(I1, dev)
        H, W = I0.shape[:2]

        if flow_t is None:
            flow_t = torch.zeros((H, W, 2), device=dev, dtype=torch.float32)
        else:
            flow_t = flow_t.permute(2, 0, 1).unsqueeze(0)
            flow_t = F.interpolate(flow_t, size=(H, W),
                                   mode="bilinear", align_corners=False)
            flow_t = flow_t.squeeze(0).permute(1, 2, 0).contiguous()
            flow_t = flow_t * (1.0 / pyr_scale)

        R0 = farneback_polyexp(t0, kernels)
        R1 = farneback_polyexp(t1, kernels)

        use_gaussian = (flags & OPTFLOW_FARNEBACK_GAUSSIAN) != 0
        if use_gaussian:
            raise NotImplementedError(
                "Gaussian pyramid not implemented in this script.")

        matM = farneback_update_matrices(R0, R1, flow_t)
        for _ in range(iterations):
            matM_blur = _box_blur5(matM, winsize)
            flow_t = farneback_update_flow_blur(matM_blur)
            matM = farneback_update_matrices(R0, R1, flow_t)

    if flow_t is None:
        raise RuntimeError("Torch Farneback produced no flow.")

    # Spatial outlier suppression: replace isolated spikes with local mean flow.
    if outlier_ksize and outlier_ksize > 1:
        k = int(outlier_ksize)
        if k % 2 == 0:
            k += 1
        flow_chw = flow_t.permute(2, 0, 1).unsqueeze(0)  # (1,2,H,W)
        flow_mean = F.avg_pool2d(flow_chw, kernel_size=k,
                                 stride=1, padding=k // 2)
        mag = torch.linalg.vector_norm(flow_chw, dim=1, keepdim=True)
        mag_mean = F.avg_pool2d(mag, kernel_size=k, stride=1, padding=k // 2)
        ratio = mag / (mag_mean + 1e-6)
        mask = (ratio > float(outlier_ratio)) & (
            mag > float(outlier_min_magnitude))
        flow_chw = torch.where(mask, flow_mean, flow_chw)
        flow_t = flow_chw.squeeze(0).permute(1, 2, 0).contiguous()

    flow = flow_t.detach().cpu().numpy().astype(np.float32)

    # Global clipping (optional) to bound remaining extremes.
    if clip_percentile is not None and 0 < clip_percentile < 100:
        mag_np = np.linalg.norm(flow, axis=2)
        thresh = float(np.percentile(mag_np, clip_percentile))
        if thresh > 0:
            scale_np = np.minimum(mag_np, thresh) / (mag_np + 1e-9)
            flow = flow * scale_np[..., None]

    if max_magnitude is not None:
        mag_np = np.linalg.norm(flow, axis=2)
        scale_np = np.minimum(mag_np, float(max_magnitude)) / (mag_np + 1e-9)
        flow = flow * scale_np[..., None]

    return flow


def _box_blur5(matM_hw5: torch.Tensor, winsize: int) -> torch.Tensor:
    """5-channel box blur over matM (H,W,5)."""
    if winsize <= 1:
        return matM_hw5
    pad = winsize // 2
    weight = torch.ones(
        (5, 1, winsize, winsize),
        device=matM_hw5.device,
        dtype=matM_hw5.dtype,
    ) / (winsize * winsize)
    m = matM_hw5.permute(2, 0, 1).unsqueeze(0)  # (1,5,H,W)
    m_blur = F.conv2d(_replicate_pad2d(m, pad), weight, groups=5)
    return m_blur.squeeze(0).permute(1, 2, 0).contiguous()
