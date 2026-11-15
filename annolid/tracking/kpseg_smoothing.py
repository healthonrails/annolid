from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np


class _OneEuroFilter:
    def __init__(
        self,
        *,
        freq: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ) -> None:
        self.freq = max(1e-6, float(freq))
        self.min_cutoff = max(1e-6, float(min_cutoff))
        self.beta = float(beta)
        self.d_cutoff = max(1e-6, float(d_cutoff))
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    def _alpha(self, cutoff: float) -> float:
        cutoff = max(1e-6, float(cutoff))
        tau = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / self.freq
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def update(self, x: float) -> float:
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


@dataclass
class _KalmanState:
    state: np.ndarray
    cov: np.ndarray


class KeypointSmoother:
    def __init__(
        self,
        *,
        mode: str = "none",
        fps: float = 30.0,
        ema_alpha: float = 0.7,
        min_score: float = 0.25,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.0,
        one_euro_d_cutoff: float = 1.0,
        kalman_process_noise: float = 1e-2,
        kalman_measurement_noise: float = 1e-1,
        kalman_gate_sigma: float = 0.0,
        max_jump_px: float = 0.0,
    ) -> None:
        self.mode = str(mode or "none").strip().lower()
        self.fps = max(1e-3, float(fps))
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.min_score = float(max(0.0, min_score))
        self.one_euro_min_cutoff = float(max(1e-6, one_euro_min_cutoff))
        self.one_euro_beta = float(one_euro_beta)
        self.one_euro_d_cutoff = float(max(1e-6, one_euro_d_cutoff))
        self.kalman_process_noise = float(max(1e-8, kalman_process_noise))
        self.kalman_measurement_noise = float(max(1e-8, kalman_measurement_noise))
        self.kalman_gate_sigma = float(max(0.0, kalman_gate_sigma))
        self.max_jump_px = float(max(0.0, max_jump_px))

        self._ema_state: Dict[str, Tuple[float, float]] = {}
        self._one_euro_state: Dict[str, Tuple[_OneEuroFilter, _OneEuroFilter]] = {}
        self._kalman_state: Dict[str, _KalmanState] = {}

    def reset(self) -> None:
        self._ema_state.clear()
        for fx, fy in self._one_euro_state.values():
            fx.reset()
            fy.reset()
        self._one_euro_state.clear()
        self._kalman_state.clear()

    def _ensure_one_euro(self, key: str) -> Tuple[_OneEuroFilter, _OneEuroFilter]:
        state = self._one_euro_state.get(key)
        if state is not None:
            return state
        fx = _OneEuroFilter(
            freq=self.fps,
            min_cutoff=self.one_euro_min_cutoff,
            beta=self.one_euro_beta,
            d_cutoff=self.one_euro_d_cutoff,
        )
        fy = _OneEuroFilter(
            freq=self.fps,
            min_cutoff=self.one_euro_min_cutoff,
            beta=self.one_euro_beta,
            d_cutoff=self.one_euro_d_cutoff,
        )
        self._one_euro_state[key] = (fx, fy)
        return fx, fy

    def _ensure_kalman(self, key: str, coord: Tuple[float, float]) -> _KalmanState:
        state = self._kalman_state.get(key)
        if state is not None:
            return state
        x, y = float(coord[0]), float(coord[1])
        vec = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        cov = np.eye(4, dtype=np.float32)
        state = _KalmanState(state=vec, cov=cov)
        self._kalman_state[key] = state
        return state

    @staticmethod
    def _clamp_jump(
        target: Tuple[float, float],
        reference: Optional[Tuple[float, float]],
        max_jump_px: float,
    ) -> Tuple[float, float]:
        if reference is None or max_jump_px <= 0.0:
            return (float(target[0]), float(target[1]))
        dx = float(target[0]) - float(reference[0])
        dy = float(target[1]) - float(reference[1])
        dist = math.hypot(dx, dy)
        if dist <= max_jump_px or dist <= 1e-9:
            return (float(target[0]), float(target[1]))
        scale = float(max_jump_px) / dist
        return (
            float(reference[0]) + dx * scale,
            float(reference[1]) + dy * scale,
        )

    def _kalman_predict(self, state: _KalmanState) -> Tuple[float, float]:
        dt = 1.0 / max(1e-3, self.fps)
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        q = self.kalman_process_noise * np.eye(4, dtype=np.float32)
        state.state = f @ state.state
        state.cov = f @ state.cov @ f.T + q
        return float(state.state[0]), float(state.state[1])

    def _kalman_update(
        self, state: _KalmanState, coord: Tuple[float, float]
    ) -> Tuple[float, float]:
        z = np.array([float(coord[0]), float(coord[1])], dtype=np.float32)
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        r = self.kalman_measurement_noise * np.eye(2, dtype=np.float32)
        y = z - (h @ state.state)
        s = h @ state.cov @ h.T + r
        s_inv = np.linalg.inv(s)
        if self.kalman_gate_sigma > 0.0:
            mahal_sq = float(y.T @ s_inv @ y)
            if math.isfinite(mahal_sq) and mahal_sq > (
                self.kalman_gate_sigma * self.kalman_gate_sigma
            ):
                return float(state.state[0]), float(state.state[1])
        k = state.cov @ h.T @ s_inv
        state.state = state.state + (k @ y)
        i = np.eye(4, dtype=np.float32)
        state.cov = (i - k @ h) @ state.cov
        return float(state.state[0]), float(state.state[1])

    def smooth(
        self,
        key: str,
        coord: Tuple[float, float],
        *,
        score: float,
        mask_ok: bool,
    ) -> Tuple[float, float]:
        raw_x, raw_y = float(coord[0]), float(coord[1])
        finite_coord = math.isfinite(raw_x) and math.isfinite(raw_y)
        mode = self.mode
        if mode in ("none", ""):
            return (raw_x, raw_y)

        finite_score = float(score) if math.isfinite(float(score)) else 0.0
        gated = finite_coord and bool(mask_ok) and finite_score >= float(self.min_score)

        if mode == "ema":
            prev = self._ema_state.get(key)
            if not gated:
                if prev is not None:
                    return (float(prev[0]), float(prev[1]))
                return (raw_x, raw_y) if finite_coord else (0.0, 0.0)
            clamped = self._clamp_jump((raw_x, raw_y), prev, self.max_jump_px)
            if prev is None:
                new = clamped
            else:
                alpha = self.ema_alpha
                new = (
                    alpha * float(clamped[0]) + (1.0 - alpha) * float(prev[0]),
                    alpha * float(clamped[1]) + (1.0 - alpha) * float(prev[1]),
                )
            self._ema_state[key] = new
            return new

        if mode == "one_euro":
            prev = self._ema_state.get(key)
            if not gated:
                if prev is not None:
                    return (float(prev[0]), float(prev[1]))
                return (raw_x, raw_y) if finite_coord else (0.0, 0.0)
            fx, fy = self._ensure_one_euro(key)
            clamped = self._clamp_jump((raw_x, raw_y), prev, self.max_jump_px)
            x = fx.update(float(clamped[0]))
            y = fy.update(float(clamped[1]))
            self._ema_state[key] = (float(x), float(y))
            return (float(x), float(y))

        if mode == "kalman":
            state = self._kalman_state.get(key)
            if state is None:
                if not gated:
                    return (raw_x, raw_y) if finite_coord else (0.0, 0.0)
                self._ensure_kalman(key, (raw_x, raw_y))
                return (raw_x, raw_y)
            prev_xy = (float(state.state[0]), float(state.state[1]))
            pred_x, pred_y = self._kalman_predict(state)
            if not gated:
                return self._clamp_jump((pred_x, pred_y), prev_xy, self.max_jump_px)
            clamped_meas = self._clamp_jump(
                (raw_x, raw_y), (pred_x, pred_y), self.max_jump_px
            )
            updated = self._kalman_update(state, clamped_meas)
            return self._clamp_jump(updated, prev_xy, self.max_jump_px)

        return (raw_x, raw_y)
