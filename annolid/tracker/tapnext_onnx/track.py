"""TAPNext ONNX point tracking backend for Annolid.

The local ``torch_tapnext_demo.ipynb`` exports TAPNext as a stateless ONNX clip
model:

* ``video``: ``[B, T, H, W, C]``
* ``query_points``: ``[B, Q, 3]``, with ``[t, y, x]`` coordinates
* ``tracks``: ``[B, T, Q, 2]``, with ``[y, x]`` coordinates
* ``visible_logits``: ``[B, T, Q, 1]``

This module keeps that ONNX-specific runtime isolated behind the shared
point-tracking processor contract used by CoTracker and CoWTracker.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from annolid.segmentation.videomt_onnx import (
    _preflight_validate_onnx_file,
    _require_onnxruntime,
)
from annolid.tracker.point_tracking_processor import BasePointTrackingProcessor
from annolid.utils.logger import logger
from annolid.utils.model_assets import (
    ensure_cached_model_asset,
    model_downloads_dir,
    resolve_existing_model_path,
)


TAPNEXT_ONNX_BACKEND = "tapnext"
TAPNEXT_ONNX_FILE_NAME = "tapnext.onnx"
TAPNEXT_ONNX_RELATIVE_PATH = f"downloads/{TAPNEXT_ONNX_FILE_NAME}"
TAPNEXT_ONNX_URL = (
    "https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx"
)
TAPNEXT_ONNX_SHA256 = "4fca0951802f0b745de254930c880938a74bf8b54b10786fc68d0ab4ba5c5300"

_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}


@dataclass(frozen=True)
class _InputSpec:
    name: str
    shape: list[Any]
    rank: int


@dataclass(frozen=True)
class _OutputSpec:
    name: str
    shape: list[Any]
    rank: int


def _clean_dim(dim: Any) -> int | None:
    return dim if isinstance(dim, int) and dim > 0 else None


def _select_tapnext_providers(available_providers: list[str]) -> list[str]:
    available = set(available_providers or [])
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    use_coreml = str(os.environ.get("ANNOLID_TAPNEXT_USE_COREML", "")).lower() in {
        "1",
        "true",
        "yes",
    }
    if use_coreml and "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers or ["CPUExecutionProvider"]


def _tensor_has_last_xy(shape: list[Any]) -> bool:
    return len(shape) >= 3 and _clean_dim(shape[-1]) == 2


def _is_official_tapnext_ref(model_ref: str, resolved: Path | None = None) -> bool:
    normalized = model_ref.strip().replace("\\", "/").lower()
    if normalized in {"", "tapnext", "tapnext.onnx", TAPNEXT_ONNX_RELATIVE_PATH}:
        return True
    if resolved is None:
        return False
    try:
        return (
            resolved.resolve()
            == (model_downloads_dir() / TAPNEXT_ONNX_FILE_NAME).resolve()
        )
    except Exception:
        return resolved == model_downloads_dir() / TAPNEXT_ONNX_FILE_NAME


class TapNextOnnxProcessor(BasePointTrackingProcessor):
    """Stateless clip inference for TAPNext/TAPNext++ ONNX exports."""

    supports_online = True

    def __init__(
        self,
        video_path: str,
        json_path: Optional[str] = None,
        is_online: bool = True,
        should_stop: Optional[Callable[[], bool]] = None,
        model_name: Optional[str] = None,
        tapnext_model_path: Optional[str] = None,
        tapnext_input_height: int = 0,
        tapnext_input_width: int = 0,
        tapnext_visibility_threshold: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            video_path=video_path,
            json_path=json_path,
            is_online=is_online,
            should_stop=should_stop,
            model_name=model_name,
            **kwargs,
        )
        self.model_path = self._resolve_model_path(
            tapnext_model_path
            or kwargs.get("model_path")
            or kwargs.get("model_weight")
            or model_name
        )
        self.input_height_override = int(tapnext_input_height or 0)
        self.input_width_override = int(tapnext_input_width or 0)
        self.visibility_threshold = float(tapnext_visibility_threshold)

        self._input_specs: list[_InputSpec] = []
        self._output_specs: list[_OutputSpec] = []
        self._video_input: _InputSpec | None = None
        self._query_input: _InputSpec | None = None
        self._track_output: _OutputSpec | None = None
        self._visibility_output: _OutputSpec | None = None
        self._input_height = 0
        self._input_width = 0
        self._temporal_window = 1
        self._query_capacity: int | None = None

    def _resolve_model_path(self, model_ref: str | None) -> Path:
        model_ref_text = str(model_ref or "").strip()
        candidates: list[str] = []
        if model_ref_text and model_ref_text.lower().endswith(".onnx"):
            candidates.append(model_ref_text)

        for candidate in candidates:
            resolved = resolve_existing_model_path(candidate)
            if (
                resolved is not None
                and resolved.suffix.lower() == ".onnx"
                and not _is_official_tapnext_ref(candidate, resolved)
            ):
                return resolved

        try:
            cached = ensure_cached_model_asset(
                file_name=TAPNEXT_ONNX_FILE_NAME,
                url=TAPNEXT_ONNX_URL,
                expected_sha256=TAPNEXT_ONNX_SHA256,
            )
            logger.info(
                "TAPNext ONNX model downloaded and cached at %s from %s",
                cached,
                TAPNEXT_ONNX_URL,
            )
            return cached
        except Exception as exc:
            tried = list(dict.fromkeys(candidates + [TAPNEXT_ONNX_RELATIVE_PATH]))
            raise FileNotFoundError(
                "TAPNext ONNX model not found locally and auto-download failed. "
                f"Tried candidates: {tried}. Download URL: {TAPNEXT_ONNX_URL}. "
                f"Expected sha256: {TAPNEXT_ONNX_SHA256}. Error: {exc}"
            ) from exc

    def load_model(self):
        _preflight_validate_onnx_file(self.model_path)
        ort = _require_onnxruntime()
        providers = _select_tapnext_providers(ort.get_available_providers())
        cache_key = (str(self.model_path.resolve()), tuple(providers))
        if cache_key in _SESSION_CACHE:
            session = _SESSION_CACHE[cache_key]
        else:
            sess_opts = ort.SessionOptions()
            session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
                sess_options=sess_opts,
            )
            _SESSION_CACHE[cache_key] = session
        self._inspect_session(session)
        logger.info(
            "Loaded TAPNext ONNX model '%s' with providers %s",
            self.model_path,
            session.get_providers(),
        )
        return session

    def _inspect_session(self, session: Any) -> None:
        self._input_specs = [
            _InputSpec(inp.name, list(inp.shape or []), len(inp.shape or []))
            for inp in session.get_inputs()
        ]
        self._output_specs = [
            _OutputSpec(out.name, list(out.shape or []), len(out.shape or []))
            for out in session.get_outputs()
        ]
        self._video_input = self._find_video_input()
        self._query_input = self._find_query_input()
        self._track_output = self._find_track_output()
        self._visibility_output = self._find_visibility_output()

        if self._video_input is None:
            raise RuntimeError(
                "TAPNext ONNX model has no recognizable video/image input."
            )
        if self._query_input is None:
            raise RuntimeError(
                "TAPNext ONNX model has no recognizable query-points input."
            )
        if self._track_output is None:
            raise RuntimeError(
                "TAPNext ONNX model has no recognizable track output with xy coordinates."
            )

        self._input_height, self._input_width = self._resolve_input_size()
        self._temporal_window = self._resolve_temporal_window()
        self._query_capacity = self._resolve_query_capacity()

    def _find_video_input(self) -> _InputSpec | None:
        name_matches = [
            spec
            for spec in self._input_specs
            if any(token in spec.name.lower() for token in ("video", "image", "frame"))
        ]
        for spec in name_matches + self._input_specs:
            if spec.rank in {4, 5} and 3 in {_clean_dim(dim) for dim in spec.shape}:
                return spec
        return None

    def _find_query_input(self) -> _InputSpec | None:
        name_matches = [
            spec
            for spec in self._input_specs
            if any(token in spec.name.lower() for token in ("query", "point", "coord"))
        ]
        for spec in name_matches + self._input_specs:
            if (
                spec.rank in {2, 3, 4}
                and _clean_dim(spec.shape[-1]) in {2, 3}
                and spec != self._video_input
            ):
                return spec
        return None

    def _find_track_output(self) -> _OutputSpec | None:
        name_matches = [
            spec
            for spec in self._output_specs
            if "track" in spec.name.lower() and _tensor_has_last_xy(spec.shape)
        ]
        if name_matches:
            return name_matches[0]
        for spec in self._output_specs:
            if _tensor_has_last_xy(spec.shape):
                return spec
        return None

    def _find_visibility_output(self) -> _OutputSpec | None:
        for spec in self._output_specs:
            name = spec.name.lower()
            if any(token in name for token in ("visible", "visibility", "occlusion")):
                return spec
        return None

    def _resolve_input_size(self) -> tuple[int, int]:
        if self.input_height_override > 0 and self.input_width_override > 0:
            return self.input_height_override, self.input_width_override
        assert self._video_input is not None
        shape = self._video_input.shape
        if self._is_channels_last(shape):
            height = _clean_dim(shape[-3])
            width = _clean_dim(shape[-2])
        else:
            height = _clean_dim(shape[-2])
            width = _clean_dim(shape[-1])
        return height or self.video_height, width or self.video_width

    @staticmethod
    def _is_channels_last(shape: list[Any]) -> bool:
        return bool(shape and _clean_dim(shape[-1]) == 3)

    def _resolve_temporal_window(self) -> int:
        assert self._video_input is not None
        shape = self._video_input.shape
        if self._video_input.rank != 5:
            return 1
        if self._is_channels_last(shape):
            return _clean_dim(shape[1]) or 1
        if _clean_dim(shape[2]) == 3:
            return _clean_dim(shape[1]) or 1
        return _clean_dim(shape[2]) or 1

    def _resolve_query_capacity(self) -> int | None:
        assert self._query_input is not None
        shape = self._query_input.shape
        if self._query_input.rank == 2:
            return _clean_dim(shape[0])
        if self._query_input.rank >= 3:
            return _clean_dim(shape[1])
        return None

    def _format_video_clip(self, frames: list[np.ndarray]) -> np.ndarray:
        assert self._video_input is not None
        if not frames:
            raise RuntimeError(
                "TAPNext requires at least one frame per inference clip."
            )

        target_len = max(1, int(self._temporal_window))
        padded_frames = list(frames)
        while len(padded_frames) < target_len:
            padded_frames.append(padded_frames[-1])

        prepared = []
        for frame in padded_frames[:target_len]:
            resized = cv2.resize(
                frame,
                (int(self._input_width), int(self._input_height)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            if resized.max(initial=0.0) > 1.5:
                resized = resized / 255.0
            prepared.append(resized)
        clip = np.stack(prepared, axis=0)

        shape = self._video_input.shape
        rank = self._video_input.rank
        if rank == 5:
            if self._is_channels_last(shape):
                return clip[None, ...]
            return clip.transpose(0, 3, 1, 2)[None, ...]
        if rank == 4:
            frame0 = clip[0]
            if self._is_channels_last(shape):
                return frame0[None, ...]
            return frame0.transpose(2, 0, 1)[None, ...]
        raise RuntimeError(
            f"Unsupported TAPNext video input rank {rank}; expected rank 4 or 5."
        )

    def _format_queries(
        self,
        queries: torch.Tensor,
        *,
        chunk_start_frame: int,
        clip_len: int,
        query_capacity: int | None,
    ) -> np.ndarray:
        assert self._query_input is not None
        q = queries.detach().cpu().numpy().astype(np.float32).copy()
        scale_x = float(self._input_width) / float(self.video_width)
        scale_y = float(self._input_height) / float(self.video_height)
        x = q[:, 1].copy()
        y = q[:, 2].copy()
        q[:, 0] = np.clip(
            q[:, 0] - float(chunk_start_frame),
            0.0,
            float(max(0, clip_len - 1)),
        )
        # TAPNext query points use [t, y, x]; Annolid points are [t, x, y].
        q[:, 1] = y * scale_y
        q[:, 2] = x * scale_x

        if _clean_dim(self._query_input.shape[-1]) == 2:
            q = q[:, 1:3]
        if query_capacity is not None:
            if q.shape[0] > query_capacity:
                raise RuntimeError(
                    "TAPNext query batch exceeds exported ONNX query capacity: "
                    f"{q.shape[0]} > {query_capacity}."
                )
            if q.shape[0] < query_capacity:
                pad = np.zeros((query_capacity - q.shape[0], q.shape[1]), dtype=q.dtype)
                if q.shape[0] > 0:
                    pad[:] = q[-1]
                q = np.concatenate([q, pad], axis=0)

        if self._query_input.rank == 2:
            return q
        if self._query_input.rank == 3:
            return q[None, ...]
        if self._query_input.rank == 4:
            return q[None, :, None, :]
        raise RuntimeError(
            f"Unsupported TAPNext query input rank {self._query_input.rank}."
        )

    def _query_batches(self, queries: torch.Tensor) -> list[torch.Tensor]:
        capacity = self._query_capacity
        if capacity is None or capacity <= 0 or int(queries.shape[0]) <= capacity:
            return [queries]
        return [
            queries[idx : idx + capacity]
            for idx in range(0, queries.shape[0], capacity)
        ]

    def _run_clip(
        self,
        session: Any,
        frames: list[np.ndarray],
        *,
        queries: torch.Tensor,
        chunk_start_frame: int,
        query_capacity: int | None,
    ) -> dict[str, np.ndarray]:
        assert self._video_input is not None
        assert self._query_input is not None
        feed = {
            self._video_input.name: self._format_video_clip(frames),
            self._query_input.name: self._format_queries(
                queries,
                chunk_start_frame=chunk_start_frame,
                clip_len=len(frames),
                query_capacity=query_capacity,
            ),
        }
        output_names = [spec.name for spec in self._output_specs]
        raw_outputs = session.run(output_names, feed)
        return dict(zip(output_names, raw_outputs))

    def _extract_clip_tracks(
        self,
        outputs: dict[str, np.ndarray],
        *,
        point_count: int,
        frame_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._track_output is not None
        tracks = np.asarray(outputs[self._track_output.name], dtype=np.float32)
        if tracks.ndim == 4:
            tracks = tracks[0]
        if tracks.ndim == 2 and frame_count == 1:
            tracks = tracks[None, ...]
        if (
            tracks.ndim == 3
            and tracks.shape[0] == point_count
            and tracks.shape[1] == frame_count
            and tracks.shape[0] != frame_count
        ):
            tracks = np.transpose(tracks, (1, 0, 2))
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise RuntimeError(
                f"Unsupported TAPNext track output shape: {outputs[self._track_output.name].shape}"
            )
        if tracks.shape[1] < point_count:
            raise RuntimeError(
                "TAPNext track output point count mismatch: "
                f"expected at least {point_count}, got {tracks.shape[1]}."
            )
        tracks = tracks[:frame_count, :point_count, :]
        # TAPNext returns [y, x] model coordinates. Annolid stores [x, y].
        tracks_y = tracks[..., 0].copy()
        tracks_x = tracks[..., 1].copy()
        annolid_tracks = np.empty_like(tracks)
        annolid_tracks[..., 0] = (
            tracks_x * float(self.video_width) / float(self._input_width)
        )
        annolid_tracks[..., 1] = (
            tracks_y * float(self.video_height) / float(self._input_height)
        )

        visibility = np.ones((frame_count, point_count), dtype=bool)
        if self._visibility_output is not None:
            logits = np.asarray(outputs[self._visibility_output.name])
            if logits.ndim == 4:
                logits = logits[0]
            if logits.ndim == 3 and logits.shape[-1] == 1:
                logits = logits[..., 0]
            if logits.ndim == 1 and frame_count == 1:
                logits = logits[None, ...]
            if (
                logits.ndim == 2
                and logits.shape[0] == point_count
                and logits.shape[1] == frame_count
                and logits.shape[0] != frame_count
            ):
                logits = logits.T
            if logits.ndim == 2 and logits.shape[1] >= point_count:
                visibility = (
                    logits[:frame_count, :point_count].astype(np.float32)
                    > self.visibility_threshold
                )
        return annolid_tracks, visibility

    def _run_query_batches(
        self,
        session: Any,
        frames: list[np.ndarray],
        *,
        queries: torch.Tensor,
        chunk_start_frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        track_chunks: list[np.ndarray] = []
        visibility_chunks: list[np.ndarray] = []
        for query_batch in self._query_batches(queries):
            outputs = self._run_clip(
                session,
                frames,
                queries=query_batch,
                chunk_start_frame=chunk_start_frame,
                query_capacity=self._query_capacity,
            )
            tracks, visibility = self._extract_clip_tracks(
                outputs,
                point_count=int(query_batch.shape[0]),
                frame_count=len(frames),
            )
            track_chunks.append(tracks)
            visibility_chunks.append(visibility)
        return (
            np.concatenate(track_chunks, axis=1),
            np.concatenate(visibility_chunks, axis=1),
        )

    def _process_video_bidirectional(
        self,
        start_frame: int = 0,
        end_frame: int = -1,
        grid_size: int = 10,
        grid_query_frame: int = 0,
    ):
        del grid_size, grid_query_frame
        if self._should_stop():
            self._stop_triggered = True
            return None, None, None

        session = self._ensure_model()
        total_frames = int(self.video_loader.total_frames())
        actual_end = int(end_frame) if int(end_frame) >= 0 else total_frames - 1
        actual_end = min(actual_end, total_frames - 1)
        if start_frame > actual_end:
            return None, None, None

        tracks_by_frame: list[np.ndarray] = []
        visibility_by_frame: list[np.ndarray] = []
        video_source: list[np.ndarray] = []
        chunk_size = max(1, int(self._temporal_window))

        for chunk_start in range(int(start_frame), actual_end + 1, chunk_size):
            if self._should_stop():
                self._stop_triggered = True
                break
            chunk_end = min(chunk_start + chunk_size - 1, actual_end)
            frames: list[np.ndarray] = []
            for frame_number in range(chunk_start, chunk_end + 1):
                frame = self.video_loader.load_frame(frame_number)
                if frame is None:
                    logger.warning(
                        "Failed to load frame %s, stopping TAPNext", frame_number
                    )
                    break
                frames.append(frame)
            if not frames:
                break

            tracks, visibility = self._run_query_batches(
                session,
                frames,
                queries=self.queries,
                chunk_start_frame=chunk_start,
            )
            tracks_by_frame.extend(tracks)
            visibility_by_frame.extend(visibility)
            video_source.extend(frames)

        if not tracks_by_frame:
            return None, None, None

        tracks_np = np.stack(tracks_by_frame, axis=0)[None, ...]
        visibility_np = np.stack(visibility_by_frame, axis=0)[None, ...]
        pred_tracks = torch.from_numpy(tracks_np).to(self.device)
        pred_visibility = torch.from_numpy(visibility_np).to(self.device)
        return pred_tracks, pred_visibility, video_source

    def _process_video_online(
        self,
        grid_size: int,
        grid_query_frame: int,
        need_visualize: bool,
    ) -> str:
        del grid_size, grid_query_frame
        if self._should_stop():
            self._stop_triggered = True
            return self._stop_message(self.start_frame)

        if need_visualize:
            logger.warning(
                "TAPNext visualization is skipped during incremental online tracking "
                "to avoid retaining the full video in memory."
            )

        session = self._ensure_model()
        total_frames = int(self.video_loader.total_frames())
        actual_end = (
            int(self.end_frame) if int(self.end_frame) >= 0 else total_frames - 1
        )
        actual_end = min(actual_end, total_frames - 1)
        if self.start_frame > actual_end:
            return "No frames saved"

        chunk_size = max(1, int(self._temporal_window))
        saved_until: int | None = None
        total_to_process = max(1, actual_end - int(self.start_frame) + 1)
        current_queries = self.queries.detach().clone()

        query_times = current_queries[:, 0].detach().cpu().numpy()
        finite_query_times = query_times[np.isfinite(query_times)]
        seed_frame = (
            int(np.min(finite_query_times))
            if finite_query_times.size
            else int(self.start_frame)
        )
        chunk_start_iter = min(int(self.start_frame), max(0, seed_frame))

        while chunk_start_iter <= actual_end:
            if self._should_stop():
                self._stop_triggered = True
                logger.info("TAPNext stop requested at frame %s", chunk_start_iter)
                break

            chunk_start = int(chunk_start_iter)
            chunk_end = min(chunk_start + chunk_size - 1, actual_end)
            frames: list[np.ndarray] = []
            for frame_number in range(chunk_start, chunk_end + 1):
                frame = self.video_loader.load_frame(frame_number)
                if frame is None:
                    logger.warning(
                        "Failed to load frame %s, stopping TAPNext", frame_number
                    )
                    break
                frames.append(frame)
            if not frames:
                break

            anchor_overlap = saved_until is not None and chunk_start == int(saved_until)
            tracks, visibility = self._run_query_batches(
                session,
                frames,
                queries=current_queries,
                chunk_start_frame=chunk_start,
            )
            pred_tracks = torch.from_numpy(tracks[None, ...]).to(self.device)
            pred_visibility = torch.from_numpy(visibility[None, ...]).to(self.device)
            local_frame_indices = [
                idx
                for idx in range(len(frames))
                if chunk_start + idx >= int(self.start_frame)
                and not (anchor_overlap and idx == 0)
            ]
            self.extract_frame_points(
                pred_tracks,
                pred_visibility,
                chunk_start_frame=chunk_start,
                local_frame_indices=local_frame_indices,
                description="TAPNext ONNX",
            )
            saved_until = chunk_start + len(frames) - 1
            current_queries = torch.as_tensor(
                np.column_stack(
                    [
                        np.full(
                            (tracks.shape[1],), float(saved_until), dtype=np.float32
                        ),
                        tracks[-1, :, 0].astype(np.float32),
                        tracks[-1, :, 1].astype(np.float32),
                    ]
                ),
                dtype=torch.float32,
                device=self.device,
            )

            if self.pred_worker is not None:
                completed = max(0, int(saved_until) - int(self.start_frame) + 1)
                pct = int(min(100, max(0, round(completed * 100 / total_to_process))))
                try:
                    if hasattr(self.pred_worker, "report_progress"):
                        self.pred_worker.report_progress(pct)
                    elif hasattr(self.pred_worker, "progress_signal"):
                        self.pred_worker.progress_signal.emit(pct)
                except Exception:
                    logger.debug("Failed to emit TAPNext progress.", exc_info=True)

            logger.info("TAPNext: saved frames %d-%d", chunk_start, saved_until)
            if int(saved_until) >= actual_end:
                break
            chunk_start_iter = int(saved_until)
            if chunk_start_iter < int(self.start_frame):
                chunk_start_iter = int(self.start_frame)
            if chunk_start_iter == chunk_start:
                chunk_start_iter += chunk_size

        if self._stop_triggered:
            return self._stop_message(saved_until or self.start_frame)

        message = (
            f"Completed. Saved frames {self.start_frame}-{saved_until}"
            if saved_until is not None
            else "No frames saved"
        )
        logger.info(message)
        return message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--json_path", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    args = parser.parse_args()

    processor = TapNextOnnxProcessor(
        args.video_path,
        args.json_path,
        tapnext_model_path=args.model_path,
    )
    print(processor.process_video_frames(args.start_frame, args.end_frame))
