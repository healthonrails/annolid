"""
Minimal PyTorch Farneback optical flow (CPU/GPU) with optional OpenCV fallback.
- farneback_prepare_gaussian: matches OpenCV FarnebackPrepareGaussian (moment matrix + Cholesky inverse)
- farneback_polyexp: matches OpenCV FarnebackPolyExp structure (even/odd vertical + horizontal)
- farneback_update_matrices: includes OpenCV-exact border attenuation behavior
- _box_blur5: matches OpenCV FarnebackUpdateFlow_Blur blur stage (running sums, float64 accumulators)
"""

from __future__ import annotations
from dataclasses import dataclass
import warnings
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from annolid.utils.logger import logger


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


def _acc_dtype_for_device(device: torch.device) -> torch.dtype:
    """
    Prefer float64 accumulators (to mirror OpenCV) but fall back to float32 on
    devices that do not support float64, e.g., MPS.
    """
    return torch.float32 if device.type == "mps" else torch.float64


_BORDER_SCALE_CACHE: dict[tuple[str, int, int, int], torch.Tensor] = {}
_BASE_COORDS_CACHE: dict[tuple[str, int, int, int],
                         tuple[torch.Tensor, torch.Tensor]] = {}


def _device_key(device: torch.device) -> tuple[str, int]:
    return (device.type, device.index if device.index is not None else -1)


def _to_device_safe(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type != "mps":
        return x.to(device=device)
    x = x.contiguous()
    dst = torch.empty_like(x, device=device)
    dst.copy_(x, non_blocking=False)
    torch.mps.synchronize()
    return dst


def _get_border_scale_numpy(H: int, W: int) -> np.ndarray:
    BORDER = 5
    border = np.array([0.14, 0.14, 0.4472, 0.4472, 0.4472], dtype=np.float32)
    xs_i = np.arange(W, dtype=np.int64)
    ys_i = np.arange(H, dtype=np.int64)

    fx = np.ones((W,), dtype=np.float32)
    fy = np.ones((H,), dtype=np.float32)

    mxL = xs_i < BORDER
    if np.any(mxL):
        fx[mxL] *= border[xs_i[mxL]]
    mxR = xs_i >= (W - BORDER)
    if np.any(mxR):
        fx[mxR] *= border[np.clip(W - xs_i[mxR] - 1, 0, BORDER - 1)]

    myT = ys_i < BORDER
    if np.any(myT):
        fy[myT] *= border[ys_i[myT]]
    myB = ys_i >= (H - BORDER)
    if np.any(myB):
        fy[myB] *= border[np.clip(H - ys_i[myB] - 1, 0, BORDER - 1)]

    return (fy.reshape(H, 1) * fx.reshape(1, W)).astype(np.float32)


def _get_border_scale(H: int, W: int, device: torch.device) -> torch.Tensor:
    key = (*_device_key(device), H, W)
    scale = _BORDER_SCALE_CACHE.get(key)
    if scale is not None:
        return scale

    BORDER = 5
    if device.type == "mps":
        scale_np = _get_border_scale_numpy(H, W)
        scale = torch.from_numpy(scale_np).to(device=device)
        _BORDER_SCALE_CACHE[key] = scale
        return scale

    border = torch.tensor([0.14, 0.14, 0.4472, 0.4472, 0.4472],
                          device=device, dtype=torch.float32)

    xs_i = torch.arange(W, device=device, dtype=torch.int64)
    ys_i = torch.arange(H, device=device, dtype=torch.int64)

    fx = torch.ones((W,), device=device, dtype=torch.float32)
    fy = torch.ones((H,), device=device, dtype=torch.float32)

    mxL = xs_i < BORDER
    if mxL.any():
        fx[mxL] *= border[xs_i[mxL]]
    mxR = xs_i >= (W - BORDER)
    if mxR.any():
        fx[mxR] *= border[(W - xs_i[mxR] - 1).clamp(min=0, max=BORDER - 1)]

    myT = ys_i < BORDER
    if myT.any():
        fy[myT] *= border[ys_i[myT]]
    myB = ys_i >= (H - BORDER)
    if myB.any():
        fy[myB] *= border[(H - ys_i[myB] - 1).clamp(min=0, max=BORDER - 1)]

    scale = (fy.view(H, 1) * fx.view(1, W)).contiguous()
    _BORDER_SCALE_CACHE[key] = scale
    return scale


def _get_base_coords(H: int, W: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (*_device_key(device), H, W)
    coords = _BASE_COORDS_CACHE.get(key)
    if coords is not None:
        return coords

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    coords = (ys, xs)
    _BASE_COORDS_CACHE[key] = coords
    return coords


@dataclass(frozen=True)
class FarnebackGaussKernels:
    n: int
    sigma: float
    g: torch.Tensor    # (2n+1) for offsets [-n..n], indexed by offset+n
    xg: torch.Tensor   # (2n+1)
    xxg: torch.Tensor  # (2n+1)
    ig11: float
    ig03: float
    ig33: float
    ig55: float
    kv_f32: torch.Tensor  # (3,1,2n+1,1)
    kh_f32: torch.Tensor  # (6,3,1,2n+1)
    kv_f64: Optional[torch.Tensor]
    kh_f64: Optional[torch.Tensor]


# ---------------------------
# OpenCV-parity core
# ---------------------------

_GAUSS_KERNEL_CACHE: dict[tuple[str, int, int, float],
                          FarnebackGaussKernels] = {}


def farneback_prepare_gaussian(n: int, sigma: float, device: torch.device) -> FarnebackGaussKernels:
    """
    Matches OpenCV FarnebackPrepareGaussian (optflowgf.cpp):
    - build g/xg/xxg over [-n..n]
    - build moment sums with nested loops
    - invert analytically using the matrix's block structure (avoids LAPACK/Accelerate)
    """
    if sigma < 1e-7:
        sigma = float(n) * 0.3
    sigma = float(sigma)

    cache_key = (*_device_key(device), n, sigma)
    cached = _GAUSS_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    xs = np.arange(-n, n + 1, dtype=np.float64)
    g = np.exp(-(xs * xs) / (2.0 * sigma * sigma)).astype(np.float64)
    g /= g.sum()
    xg = (xs * g).astype(np.float64)
    xxg = (xs * xs * g).astype(np.float64)

    # OpenCV builds a structured 6x6 moment matrix G but only a few unique sums
    # are needed. The matrix is block diagonal except for a symmetric 3x3 block
    # over indices (0,3,4); we compute the required inverse entries analytically.
    G00 = 0.0
    G11 = 0.0
    G33 = 0.0
    G55 = 0.0
    for y in range(-n, n + 1):
        gy = g[y + n]
        yy = float(y * y)
        for x in range(-n, n + 1):
            gx = g[x + n]
            xx = float(x * x)
            w = gy * gx
            G00 += w
            G11 += w * xx
            G33 += w * xx * xx
            G55 += w * xx * yy

    A = float(G00)
    B = float(G11)
    C = float(G33)
    D = float(G55)

    # Indices 1 and 2 are independent 1x1 blocks (value B), and index 5 is an
    # independent 1x1 block (value D).
    ig11 = 1.0 / B
    ig55 = 1.0 / D

    # For the (0,3,4) 3x3 block:
    #   [[A, B, B],
    #    [B, C, D],
    #    [B, D, C]]
    # In a symmetric/antisymmetric basis it becomes (C-D) ⊕ [[A, √2 B],[√2 B, C+D]].
    det2 = A * (C + D) - 2.0 * (B * B)
    ig03 = -B / det2
    ig33 = 0.5 * (A / det2 + 1.0 / (C - D))

    g_cpu = torch.from_numpy(g.astype(np.float32))
    xg_cpu = torch.from_numpy(xg.astype(np.float32))
    xxg_cpu = torch.from_numpy(xxg.astype(np.float32))

    kv_cpu = torch.stack([g_cpu, xg_cpu, xxg_cpu], dim=0).view(
        3, 1, 2 * n + 1, 1)
    kh_cpu = torch.zeros((6, 3, 1, 2 * n + 1), dtype=torch.float32)
    kh_cpu[0, 0, 0, :] = g_cpu
    kh_cpu[1, 0, 0, :] = xg_cpu
    kh_cpu[2, 1, 0, :] = g_cpu
    kh_cpu[3, 0, 0, :] = xxg_cpu
    kh_cpu[4, 2, 0, :] = g_cpu
    kh_cpu[5, 1, 0, :] = xg_cpu

    g_t = _to_device_safe(g_cpu, device)
    xg_t = _to_device_safe(xg_cpu, device)
    xxg_t = _to_device_safe(xxg_cpu, device)
    kv_f32 = _to_device_safe(kv_cpu, device)
    kh_f32 = _to_device_safe(kh_cpu, device)

    kv_f64: Optional[torch.Tensor] = None
    kh_f64: Optional[torch.Tensor] = None
    if _acc_dtype_for_device(device) == torch.float64:
        kv_f64 = kv_f32.to(torch.float64)
        kh_f64 = kh_f32.to(torch.float64)

    kernels = FarnebackGaussKernels(
        n=n,
        sigma=float(sigma),
        g=g_t,
        xg=xg_t,
        xxg=xxg_t,
        ig11=ig11,
        ig03=ig03,
        ig33=ig33,
        ig55=ig55,
        kv_f32=kv_f32,
        kh_f32=kh_f32,
        kv_f64=kv_f64,
        kh_f64=kh_f64,
    )
    _GAUSS_KERNEL_CACHE[cache_key] = kernels
    return kernels


def farneback_polyexp(img_1x1hw: torch.Tensor, kernels: FarnebackGaussKernels) -> torch.Tensor:
    """
    Matches OpenCV FarnebackPolyExp output layout: (H,W,5).

    Important: OpenCV's implementation is not a simple separable conv.
    It uses even/odd pairing in vertical and horizontal accumulation.
    This function translates that structure closely.
    """
    if img_1x1hw.ndim != 4 or img_1x1hw.shape[0] != 1 or img_1x1hw.shape[1] != 1:
        raise ValueError("Expected (1,1,H,W)")
    if img_1x1hw.dtype != torch.float32:
        raise ValueError("Expected float32")

    n = kernels.n
    ig11 = kernels.ig11
    ig03 = kernels.ig03
    ig33 = kernels.ig33
    ig55 = kernels.ig55

    H, W = img_1x1hw.shape[2:]
    dev = img_1x1hw.device
    acc_dtype = _acc_dtype_for_device(dev)

    # --- vertical stage: replicate padding then 1D conv over Y ---
    if acc_dtype == torch.float32:
        src_4d = img_1x1hw
        kv = kernels.kv_f32
        kh = kernels.kh_f32
    else:
        src_4d = img_1x1hw.to(dtype=acc_dtype)
        kv = kernels.kv_f64 if kernels.kv_f64 is not None else kernels.kv_f32.to(
            dtype=acc_dtype)
        kh = kernels.kh_f64 if kernels.kh_f64 is not None else kernels.kh_f32.to(
            dtype=acc_dtype)

    src_pad_v = F.pad(src_4d, (0, 0, n, n), mode="replicate")
    row = F.conv2d(src_pad_v, kv)  # (1,3,H,W)

    # --- horizontal stage: replicate padding then 1D conv over X ---
    row_pad = F.pad(row, (n, n, 0, 0), mode="replicate")
    b = F.conv2d(row_pad, kh)  # (1,6,H,W)
    b1, b2, b3, b4, b5, b6 = torch.unbind(b, dim=1)

    dst = torch.empty((H, W, 5), device=dev, dtype=torch.float32)
    dst[..., 0] = (b3 * ig11).float()
    dst[..., 1] = (b2 * ig11).float()
    dst[..., 2] = (b1 * ig03 + b5 * ig33).float()
    dst[..., 3] = (b1 * ig03 + b4 * ig33).float()
    dst[..., 4] = (b6 * ig55).float()

    return dst.contiguous()


def farneback_update_matrices(R0_hw5: torch.Tensor, R1_hw5: torch.Tensor, flow_hw2: torch.Tensor) -> torch.Tensor:
    """
    Builds matM (H,W,5) like OpenCV FarnebackUpdateMatrices.

    Includes OpenCV-exact border attenuation:
        scale = (x<B ? border[x] : 1) * (x>=W-B ? border[W-x-1] : 1) * ...
    """
    R0_hw5, R1_hw5, flow_hw2, H, W = _normalize_flow_hw2(
        R0_hw5, R1_hw5, flow_hw2)
    device = R0_hw5.device
    assert R0_hw5.dtype == torch.float32

    # Grid
    ys_1d, xs_1d = _get_base_coords(H, W, device)
    ys = ys_1d.view(H, 1)
    xs = xs_1d.view(1, W)

    dx = flow_hw2[..., 0]
    dy = flow_hw2[..., 1]
    fx = xs + dx
    fy = ys + dy
    if device.type == "mps":
        # Avoid int64 indexing on MPS by using grid_sample.
        valid = (fx >= 0) & (fx < W - 1) & (fy >= 0) & (fy < H - 1)
        R1w = _grid_sample_R1(R1_hw5, fx, fy)
    else:
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

    # When OOB, OpenCV uses r2=r3=0 and uses R0 terms for r4/r5, and r6=R0*0.5
    # We'll compute r2..r6 in vectorized form.
    r2 = R1w[..., 0]
    r3 = R1w[..., 1]
    r4 = R1w[..., 2]
    r5 = R1w[..., 3]
    r6 = R1w[..., 4]

    inb = valid

    r4 = torch.where(inb, (R0_hw5[..., 2] + r4) * 0.5, R0_hw5[..., 2])
    r5 = torch.where(inb, (R0_hw5[..., 3] + r5) * 0.5, R0_hw5[..., 3])
    r6 = torch.where(inb, (R0_hw5[..., 4] + r6) * 0.25, R0_hw5[..., 4] * 0.5)

    r2 = torch.where(inb, r2, torch.zeros_like(r2))
    r3 = torch.where(inb, r3, torch.zeros_like(r3))

    r2 = (R0_hw5[..., 0] - r2) * 0.5
    r3 = (R0_hw5[..., 1] - r3) * 0.5

    r2 = r2 + r4 * dy + r6 * dx
    r3 = r3 + r6 * dy + r5 * dx

    # ---- OpenCV-exact border attenuation ----
    if H > 0 and W > 0:
        scale = _get_border_scale(H, W, device)

        # OpenCV applies only near borders; interior scale == 1 so multiplying everywhere is ok.
        r2 = r2 * scale
        r3 = r3 * scale
        r4 = r4 * scale
        r5 = r5 * scale
        r6 = r6 * scale
    # ---- end border attenuation ----

    matM = torch.stack(
        [
            r4 * r4 + r6 * r6,          # G(1,1)
            (r4 + r5) * r6,             # G(1,2)
            r5 * r5 + r6 * r6,          # G(2,2)
            r4 * r2 + r6 * r3,          # h(1)
            r6 * r2 + r5 * r3,          # h(2)
        ],
        dim=-1,
    )
    return matM.contiguous()


def _normalize_flow_hw2(
    R0_hw5: torch.Tensor,
    R1_hw5: torch.Tensor,
    flow_hw2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    H, W, C = R0_hw5.shape
    if C != 5:
        raise ValueError("Expected R0_hw5 with last dim=5")

    # Normalize flow to (H,W,2)
    if flow_hw2.ndim == 4 and flow_hw2.shape[0] == 1:
        flow_hw2 = flow_hw2.squeeze(0)
    if flow_hw2.ndim == 3 and flow_hw2.shape[0] == 2 and flow_hw2.shape[-1] != 2:
        flow_hw2 = flow_hw2.permute(1, 2, 0).contiguous()
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
    return R0_hw5, R1_hw5, flow_hw2, H, W


def _grid_sample_R1(R1_hw5: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """MPS-safe bilinear sampling of R1 without int64 indexing."""
    H, W = fx.shape
    if W > 1:
        x_norm = fx / (W - 1) * 2.0 - 1.0
    else:
        x_norm = torch.zeros_like(fx)
    if H > 1:
        y_norm = fy / (H - 1) * 2.0 - 1.0
    else:
        y_norm = torch.zeros_like(fy)
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
    R1_t = R1_hw5.permute(2, 0, 1).unsqueeze(0)
    R1w = F.grid_sample(
        R1_t,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return R1w.squeeze(0).permute(1, 2, 0).contiguous()


def farneback_update_flow_blur(matM_hw5: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel solve (same as OpenCV after blur):
        flow_x = (g11*h2 - g12*h1) / det
        flow_y = (g22*h1 - g12*h2) / det
    """
    g11 = matM_hw5[..., 0]
    g12 = matM_hw5[..., 1]
    g22 = matM_hw5[..., 2]
    h1 = matM_hw5[..., 3]
    h2 = matM_hw5[..., 4]
    idet = 1.0 / (g11 * g22 - g12 * g12 + 1e-3)
    fx = (g11 * h2 - g12 * h1) * idet
    fy = (g22 * h1 - g12 * h2) * idet
    return torch.stack([fx, fy], dim=-1).contiguous()


def _vector_norm_chw(flow_chw: torch.Tensor) -> torch.Tensor:
    """
    MPS has had stability issues with torch.linalg; use a float32-safe norm there.
    """
    if flow_chw.device.type == "mps":
        return torch.sqrt((flow_chw * flow_chw).sum(dim=1, keepdim=True))
    return torch.linalg.vector_norm(flow_chw, dim=1, keepdim=True)


# Flag constant
OPTFLOW_FARNEBACK_GAUSSIAN = 256


def _box_blur5(matM_hw5: torch.Tensor, winsize: int) -> torch.Tensor:
    """
    5-channel OpenCV-style box blur over matM (H,W,5), matching FarnebackUpdateFlow_Blur's blur stage.
    Uses replicate padding + avg pooling, accumulating in float64 when available.
    Returns blurred M as float32.
    """
    if winsize <= 1:
        return matM_hw5
    _, _, C = matM_hw5.shape
    if C != 5:
        raise ValueError("Expected matM_hw5 with last dim=5")

    # OpenCV accumulates in double; fall back to float32 on backends without float64.
    acc_dtype = _acc_dtype_for_device(matM_hw5.device)
    mat = matM_hw5.permute(2, 0, 1).unsqueeze(0).to(acc_dtype)  # (1,5,H,W)
    pad = winsize // 2
    mat_pad = F.pad(mat, (pad, pad, pad, pad), mode="replicate")
    out = F.avg_pool2d(mat_pad, kernel_size=winsize, stride=1)
    return out.squeeze(0).permute(1, 2, 0).contiguous().to(torch.float32)


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
    cpu_num_threads: Optional[int] = None,
) -> np.ndarray:
    """
    PyTorch Farneback that aims to numerically match OpenCV's CPU implementation.
    Optionally clamps extreme flow magnitudes by percentile to suppress outliers.

    If you want strict OpenCV parity, set:
      outlier_ksize=0, clip_percentile=None, max_magnitude=None

    Note: The custom kernels here are heavy on small, per-row/per-pixel ops that
    launch many tiny kernels.

    cpu_num_threads: Optional override for torch CPU thread count (None keeps default).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    req_dev = torch.device(device)
    dev = req_dev
    if dev.type == "mps":
        _ = torch.zeros(1, device=dev)
        torch.mps.synchronize()

    prev_f = _to_gray_f32(prev)
    nxt_f = _to_gray_f32(nxt)
    # OpenCV pyramid: blur original with sigma(level), then resize
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
        pyr_prev.append(cv2.resize(prev_blur, (w, h),
                        interpolation=cv2.INTER_LINEAR))
        pyr_nxt.append(cv2.resize(nxt_blur, (w, h),
                       interpolation=cv2.INTER_LINEAR))

    def _run_torch(dev_run: torch.device) -> torch.Tensor:
        prev_threads: Optional[int] = None
        if dev_run.type == "cpu" and cpu_num_threads is not None:
            try:
                threads = int(cpu_num_threads)
                if threads > 0:
                    prev_threads = int(torch.get_num_threads())
                    torch.set_num_threads(threads)
            except Exception:
                prev_threads = None

        try:
            with torch.inference_mode():
                kernels = farneback_prepare_gaussian(
                    poly_n, poly_sigma, dev_run)

                flow_t: Optional[torch.Tensor] = None

                for lvl in reversed(range(levels + 1)):
                    I0 = pyr_prev[lvl]
                    I1 = pyr_nxt[lvl]

                    t0 = _torch_gray(I0, dev_run)
                    t1 = _torch_gray(I1, dev_run)
                    H, W = I0.shape[:2]

                    if flow_t is None:
                        flow_t = torch.zeros(
                            (H, W, 2), device=dev_run, dtype=torch.float32)
                    else:
                        flow_t = flow_t.permute(2, 0, 1).unsqueeze(0)
                        flow_t = F.interpolate(flow_t, size=(
                            H, W), mode="bilinear", align_corners=False)
                        flow_t = flow_t.squeeze(0).permute(
                            1, 2, 0).contiguous()
                        flow_t = flow_t * (1.0 / pyr_scale)

                    R0 = farneback_polyexp(t0, kernels)
                    R1 = farneback_polyexp(t1, kernels)

                    use_gaussian = (flags & OPTFLOW_FARNEBACK_GAUSSIAN) != 0
                    if use_gaussian:
                        raise NotImplementedError(
                            "Gaussian window update path not implemented in this script.")

                    matM = farneback_update_matrices(R0, R1, flow_t)
                    for it in range(iterations):
                        matM_blur = _box_blur5(matM, winsize)
                        flow_t = farneback_update_flow_blur(matM_blur)
                        if it < iterations - 1:
                            matM = farneback_update_matrices(R0, R1, flow_t)

                if flow_t is None:
                    raise RuntimeError("Torch Farneback produced no flow.")

                # Spatial outlier suppression: replace isolated spikes with local mean flow.
                if outlier_ksize and outlier_ksize > 1:
                    k = int(outlier_ksize)
                    if k % 2 == 0:
                        k += 1
                    flow_chw = flow_t.permute(
                        2, 0, 1).unsqueeze(0)  # (1,2,H,W)
                    flow_mean = F.avg_pool2d(
                        flow_chw, kernel_size=k, stride=1, padding=k // 2)
                    mag = _vector_norm_chw(flow_chw)
                    mag_mean = F.avg_pool2d(
                        mag, kernel_size=k, stride=1, padding=k // 2)
                    ratio = mag / (mag_mean + 1e-6)
                    mask = (ratio > float(outlier_ratio)) & (
                        mag > float(outlier_min_magnitude))
                    flow_chw = torch.where(mask, flow_mean, flow_chw)
                    flow_t = flow_chw.squeeze(
                        0).permute(1, 2, 0).contiguous()

                return flow_t
        finally:
            if prev_threads is not None and dev_run.type == "cpu":
                try:
                    torch.set_num_threads(prev_threads)
                except Exception:
                    pass

    if dev.type == "mps":
        try:
            flow_t = _run_torch(dev)
        except Exception as exc:
            logger.exception("MPS Farneback failed; retrying on CPU.")
            warnings.warn(
                f"MPS Farneback failed ({exc}); retrying on CPU.",
                RuntimeWarning,
            )
            flow_t = _run_torch(torch.device("cpu"))
    else:
        flow_t = _run_torch(dev)

    flow = flow_t.cpu().numpy().astype(np.float32)

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
