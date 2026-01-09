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
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

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
    ) -> None:
        self.mode = str(mode or "none").strip().lower()
        self.fps = max(1e-3, float(fps))
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.min_score = float(max(0.0, min_score))
        self.one_euro_min_cutoff = float(max(1e-6, one_euro_min_cutoff))
        self.one_euro_beta = float(one_euro_beta)
        self.one_euro_d_cutoff = float(max(1e-6, one_euro_d_cutoff))
        self.kalman_process_noise = float(max(1e-8, kalman_process_noise))
        self.kalman_measurement_noise = float(
            max(1e-8, kalman_measurement_noise))

        self._ema_state: Dict[str, Tuple[float, float]] = {}
        self._one_euro_state: Dict[str,
                                   Tuple[_OneEuroFilter, _OneEuroFilter]] = {}
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

    def _kalman_update(self, state: _KalmanState, coord: Tuple[float, float]) -> Tuple[float, float]:
        z = np.array([float(coord[0]), float(coord[1])], dtype=np.float32)
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                     dtype=np.float32)
        r = self.kalman_measurement_noise * np.eye(2, dtype=np.float32)
        y = z - (h @ state.state)
        s = h @ state.cov @ h.T + r
        k = state.cov @ h.T @ np.linalg.inv(s)
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
        mode = self.mode
        if mode in ("none", ""):
            return (float(coord[0]), float(coord[1]))

        gated = bool(mask_ok) and float(score) >= float(self.min_score)

        if mode == "ema":
            prev = self._ema_state.get(key)
            if not gated:
                if prev is not None:
                    return (float(prev[0]), float(prev[1]))
                return (float(coord[0]), float(coord[1]))
            if prev is None:
                new = (float(coord[0]), float(coord[1]))
            else:
                alpha = self.ema_alpha
                new = (
                    alpha * float(coord[0]) + (1.0 - alpha) * float(prev[0]),
                    alpha * float(coord[1]) + (1.0 - alpha) * float(prev[1]),
                )
            self._ema_state[key] = new
            return new

        if mode == "one_euro":
            fx, fy = self._ensure_one_euro(key)
            if not gated:
                prev = self._ema_state.get(key)
                if prev is not None:
                    return (float(prev[0]), float(prev[1]))
                return (float(coord[0]), float(coord[1]))
            x = fx.update(float(coord[0]))
            y = fy.update(float(coord[1]))
            self._ema_state[key] = (float(x), float(y))
            return (float(x), float(y))

        if mode == "kalman":
            state = self._ensure_kalman(key, coord)
            pred_x, pred_y = self._kalman_predict(state)
            if not gated:
                return (float(pred_x), float(pred_y))
            return self._kalman_update(state, coord)

        return (float(coord[0]), float(coord[1]))
