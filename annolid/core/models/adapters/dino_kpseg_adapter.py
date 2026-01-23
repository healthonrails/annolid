from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..base import ModelCapabilities, ModelRequest, ModelResponse, RuntimeModel


@dataclass(frozen=True)
class KeypointDetection:
    bbox_xyxy: Tuple[float, float, float, float]
    label_id: int
    score: float
    keypoints_xy: Sequence[Sequence[float]]
    keypoint_scores: Sequence[float]
    keypoint_visible: Optional[Sequence[bool]] = None
    keypoint_names: Optional[Sequence[str]] = None

    def to_dict(self) -> Dict[str, object]:
        x1, y1, x2, y2 = self.bbox_xyxy
        payload: Dict[str, object] = {
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "label_id": int(self.label_id),
            "score": float(self.score),
            "keypoints_xy": [[float(x), float(y)] for x, y in self.keypoints_xy],
            "keypoint_scores": [float(s) for s in self.keypoint_scores],
        }
        if self.keypoint_visible is not None:
            payload["keypoint_visible"] = [bool(v) for v in self.keypoint_visible]
        if self.keypoint_names:
            payload["keypoint_names"] = list(self.keypoint_names)
        return payload


class DinoKPSEGAdapter(RuntimeModel):
    """Stateful DinoV3 keypoint tracker seeded by DinoKPSEG weights.

    This adapter is intentionally stateful: it learns a descriptor tracker from
    seed keypoints (manual labels) and then updates keypoints frame-by-frame.
    """

    def __init__(
        self,
        *,
        weight_path: str | Path,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        bbox_padding_px: float = 4.0,
        label_id: int = 0,
    ) -> None:
        self._weight_path = str(weight_path)
        self._device = device
        self._score_threshold = float(score_threshold)
        self._bbox_padding_px = float(bbox_padding_px)
        self._label_id = int(label_id)

        self._predictor: Any = None
        self._tracker: Any = None
        self._registry: Any = None
        self._keypoint_names: Optional[List[str]] = None
        self._instance_label = "subject_0"
        self._started = False
        self._started_frame: Optional[int] = None

    @property
    def model_id(self) -> str:
        weight_name = Path(self._weight_path).name
        return f"dino_kpseg:{weight_name}" if weight_name else "dino_kpseg"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("detect",),
            input_modalities=("image",),
            output_modalities=("detections", "keypoints"),
            streaming=False,
        )

    def load(self) -> None:
        if self._predictor is not None:
            return
        try:
            from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor
            from annolid.tracking.configuration import CutieDinoTrackerConfig
            from annolid.tracking.dino_keypoint_tracker import DinoKeypointTracker
            from annolid.tracking.domain import InstanceRegistry
        except ImportError as exc:
            raise ImportError(
                "DinoKPSEGAdapter requires DinoKPSEG + DinoKeypointTracker dependencies."
            ) from exc

        predictor = DinoKPSEGPredictor(self._weight_path, device=self._device)
        self._predictor = predictor
        self._keypoint_names = list(getattr(predictor, "keypoint_names", []) or [])

        runtime_cfg = CutieDinoTrackerConfig()
        runtime_cfg.normalize()
        tracker_device = (
            str(self._device)
            if self._device
            else str(getattr(predictor, "device", None) or "")
        ).strip() or None
        self._tracker = DinoKeypointTracker(
            model_name=str(getattr(predictor.meta, "model_name", "")),
            short_side=int(getattr(predictor.meta, "short_side", 768)),
            device=tracker_device,
            runtime_config=runtime_cfg,
        )
        self._registry = InstanceRegistry()

    def predict(self, request: ModelRequest) -> ModelResponse:
        self.load()
        if self._predictor is None or self._tracker is None or self._registry is None:
            raise RuntimeError("DinoKPSEGAdapter not loaded.")

        task = str(request.task).strip().lower()
        if task != "detect":
            raise ValueError(f"Unsupported task for DinoKPSEGAdapter: {request.task!r}")

        params = request.params or {}
        frame_index = None
        try:
            frame_index = int(params.get("frame_index"))
        except Exception:
            frame_index = None
        seed_keypoints = self._parse_seed_keypoints(params)

        frame_bgr = self._load_image_bgr(request)
        frame_rgb = frame_bgr[..., ::-1].copy()

        if not self._started:
            initial = self._bootstrap_keypoints(
                frame_bgr, seed_keypoints=seed_keypoints
            )
            self._start_tracker(frame_rgb, initial_keypoints=initial)
            self._started = True
            self._started_frame = frame_index
            detections = self._detections_from_keypoints(initial, frame_rgb.shape[:2])
        else:
            detections = self._update_and_detect(
                frame_rgb,
                frame_hw=frame_rgb.shape[:2],
                seed_keypoints=seed_keypoints,
            )

        payload = [det.to_dict() for det in detections]
        return ModelResponse(
            task=request.task,
            output={"detections": payload},
            meta={"num_detections": len(payload)},
        )

    def close(self) -> None:
        self._predictor = None
        self._tracker = None
        self._registry = None
        self._keypoint_names = None
        self._started = False
        self._started_frame = None

    def _load_image_bgr(self, request: ModelRequest) -> np.ndarray:
        if request.image is not None:
            image_rgb = np.asarray(request.image)
            if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
                raise ValueError("Expected RGB image array with shape HxWx3.")
            return image_rgb[..., ::-1].copy()

        if request.image_path:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OpenCV (cv2) is required to load image_path for DinoKPSEGAdapter."
                ) from exc
            path = Path(request.image_path)
            frame_bgr = cv2.imread(str(path))
            if frame_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {path}")
            return frame_bgr

        raise ValueError(
            "DinoKPSEGAdapter requires request.image or request.image_path."
        )

    def _bootstrap_keypoints(
        self,
        frame_bgr: np.ndarray,
        *,
        seed_keypoints: Optional[Dict[str, Tuple[float, float, bool]]],
    ) -> Dict[str, Tuple[float, float, bool, float]]:
        prediction = self._predictor.predict(frame_bgr, threshold=self._score_threshold)
        names = list(self._keypoint_names or [])
        coords = list(getattr(prediction, "keypoints_xy", []) or [])
        scores = list(getattr(prediction, "keypoint_scores", []) or [])

        out: Dict[str, Tuple[float, float, bool, float]] = {}
        for idx, name in enumerate(names):
            if idx < len(coords) and len(coords[idx]) >= 2:
                x, y = float(coords[idx][0]), float(coords[idx][1])
            else:
                x, y = 0.0, 0.0
            conf = float(scores[idx]) if idx < len(scores) else 0.0
            visible = conf >= float(self._score_threshold)
            out[str(name)] = (x, y, visible, conf)

        for name, (x, y, visible) in (seed_keypoints or {}).items():
            if name not in out:
                continue
            out[name] = (float(x), float(y), bool(visible), 1.0)
        return out

    def _start_tracker(
        self,
        frame_rgb: np.ndarray,
        *,
        initial_keypoints: Dict[str, Tuple[float, float, bool, float]],
    ) -> None:
        from PIL import Image
        from annolid.tracking.domain import KeypointState

        self._registry.instances.clear()
        for name, (x, y, visible, conf) in initial_keypoints.items():
            self._registry.register_keypoint(
                KeypointState(
                    key=str(name),
                    instance_label=str(self._instance_label),
                    label=str(name),
                    x=float(x),
                    y=float(y),
                    visible=bool(visible),
                    confidence=float(conf),
                )
            )
        self._tracker.start(Image.fromarray(frame_rgb), self._registry)

    def _update_and_detect(
        self,
        frame_rgb: np.ndarray,
        *,
        frame_hw: Tuple[int, int],
        seed_keypoints: Optional[Dict[str, Tuple[float, float, bool]]],
    ) -> List[KeypointDetection]:
        from PIL import Image

        results = self._tracker.update(Image.fromarray(frame_rgb))
        self._registry.apply_tracker_results(results)

        if seed_keypoints:
            corrections = {k: (v[0], v[1]) for k, v in seed_keypoints.items()}
            self._tracker.apply_external_corrections(corrections)
            for name, (x, y, visible) in seed_keypoints.items():
                kp = self._registry.get_keypoint(str(name))
                if kp is None:
                    continue
                kp.update(x=float(x), y=float(y), visible=bool(visible), confidence=1.0)

        current = self._current_keypoints()
        return self._detections_from_keypoints(current, frame_hw)

    def _current_keypoints(self) -> Dict[str, Tuple[float, float, bool, float]]:
        out: Dict[str, Tuple[float, float, bool, float]] = {}
        for name in self._keypoint_names or []:
            kp = self._registry.get_keypoint(str(name))
            if kp is None:
                continue
            out[str(name)] = (
                float(kp.x),
                float(kp.y),
                bool(kp.visible),
                float(kp.confidence),
            )
        return out

    def _detections_from_keypoints(
        self,
        keypoints: Dict[str, Tuple[float, float, bool, float]],
        image_hw: Tuple[int, int],
    ) -> List[KeypointDetection]:
        names = list(self._keypoint_names or [])
        if not names:
            return []

        coords: List[List[float]] = []
        scores: List[float] = []
        visible: List[bool] = []
        xs: List[float] = []
        ys: List[float] = []

        for name in names:
            x, y, v, conf = keypoints.get(str(name), (0.0, 0.0, False, 0.0))
            coords.append([float(x), float(y)])
            scores.append(float(conf))
            visible.append(bool(v))
            xs.append(float(x))
            ys.append(float(y))

        h, w = int(image_hw[0]), int(image_hw[1])
        pad = max(0.0, self._bbox_padding_px)
        x1 = max(0.0, min(xs) - pad) if xs else 0.0
        y1 = max(0.0, min(ys) - pad) if ys else 0.0
        x2 = min(float(w - 1), max(xs) + pad) if xs else 0.0
        y2 = min(float(h - 1), max(ys) + pad) if ys else 0.0

        vis_scores = [s for s, v in zip(scores, visible) if v]
        if vis_scores:
            bbox_score = float(np.mean(vis_scores))
        elif scores:
            bbox_score = float(np.mean(scores))
        else:
            bbox_score = 0.0

        return [
            KeypointDetection(
                bbox_xyxy=(x1, y1, x2, y2),
                label_id=self._label_id,
                score=bbox_score,
                keypoints_xy=coords,
                keypoint_scores=scores,
                keypoint_visible=visible,
                keypoint_names=names,
            )
        ]

    @staticmethod
    def _parse_seed_keypoints(
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Tuple[float, float, bool]]]:
        raw = params.get("seed_keypoints")
        if not isinstance(raw, dict) or not raw:
            return None
        parsed: Dict[str, Tuple[float, float, bool]] = {}
        for key, val in raw.items():
            name = DinoKPSEGAdapter._canonical_keypoint_name(str(key))
            if not name:
                continue
            if isinstance(val, dict):
                xy = val.get("xy")
                if isinstance(xy, list) and len(xy) >= 2:
                    x, y = float(xy[0]), float(xy[1])
                else:
                    continue
                visible = val.get("visible")
                if visible is None:
                    visible = True
                parsed[name] = (x, y, bool(visible))
                continue
            if isinstance(val, list) and len(val) >= 2:
                parsed[name] = (float(val[0]), float(val[1]), True)
        return parsed or None

    @staticmethod
    def _canonical_keypoint_name(name: str) -> str:
        candidate = str(name or "").strip()
        if not candidate:
            return ""
        for sep in (":", "/", "\\", "|"):
            if sep in candidate:
                candidate = candidate.split(sep)[-1].strip()
        return candidate
