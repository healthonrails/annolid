# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

_COLORWHEEL_CACHE = None


def _build_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def _get_cached_colorwheel():
    global _COLORWHEEL_CACHE
    if _COLORWHEEL_CACHE is None:
        _COLORWHEEL_CACHE = _build_colorwheel()
    return _COLORWHEEL_CACHE


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """
    return _get_cached_colorwheel().copy()


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    u = np.nan_to_num(np.asarray(u, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    flow_image = np.empty((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = _get_cached_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.hypot(u, v)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1.0) * 0.5 * (ncols - 1)
    k0 = np.clip(np.floor(fk).astype(np.int32), 0, ncols - 1)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    one_minus_f = 1.0 - f
    in_range = rad <= 1.0
    for i in range(3):
        tmp = colorwheel[:, i] * (1.0 / 255.0)
        col = one_minus_f * tmp[k0] + f * tmp[k1]
        col = np.where(in_range, 1.0 - rad * (1.0 - col), col * 0.75)
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.clip(np.floor(255.0 * col), 0.0, 255.0).astype(
            np.uint8
        )
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Symmetric clip bound for flow components. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    flow_uv = np.nan_to_num(
        np.asarray(flow_uv, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if clip_flow is not None:
        bound = abs(float(clip_flow))
        flow_uv = np.clip(flow_uv, -bound, bound)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.hypot(u, v)
    rad_max = float(np.max(rad)) if rad.size else 0.0
    epsilon = 1e-5
    inv_scale = 1.0 / (rad_max + epsilon)
    return flow_uv_to_colors(u * inv_scale, v * inv_scale, convert_to_bgr)
