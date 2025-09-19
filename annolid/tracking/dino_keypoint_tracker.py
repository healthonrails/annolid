"""DINO feature-based keypoint tracking over video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import torch
import cv2
from PIL import Image

from annolid.annotation.keypoints import save_labels
from annolid.data.videos import CV2Video
from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.gui.shape import Shape
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger


@dataclass
class KeypointTrack:
    identifier: str
    label: str
    patch_rc: Tuple[int, int]
    descriptor: torch.Tensor
    velocity: Tuple[float, float] = (0.0, 0.0)
    misses: int = 0


class DinoKeypointTracker:
    """Lightweight DINO patch tracker for a set of keypoints on a single frame."""

    def __init__(
        self,
        model_name: str,
        *,
        short_side: int = 768,
        device: Optional[str] = None,
        search_radius: int = 2,
        min_similarity: float = 0.2,
        momentum: float = 0.2,
    ) -> None:
        cfg = Dinov3Config(
            model_name=model_name,
            short_side=short_side,
            device=device,
        )
        self.extractor = Dinov3FeatureExtractor(cfg)
        self.search_radius = max(1, int(search_radius))
        self.min_similarity = float(min_similarity)
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.tracks: Dict[str, KeypointTrack] = {}
        self.patch_size = self.extractor.patch_size
        self.max_misses = 8
        self.prev_gray: Optional[np.ndarray] = None

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        feats = self.extractor.extract(
            image, return_layer="all", normalize=True)
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.dim() == 4:  # [L, D, H, W]
            feats = feats[-2:].mean(dim=0)
        return feats

    def _pixel_to_patch(self, x: float, y: float, scale_x: float, scale_y: float, grid_hw: Tuple[int, int]) -> Tuple[int, int]:
        grid_h, grid_w = grid_hw
        resized_x = x * scale_x
        resized_y = y * scale_y
        patch_c = int(resized_x / self.patch_size)
        patch_r = int(resized_y / self.patch_size)
        patch_c = max(0, min(grid_w - 1, patch_c))
        patch_r = max(0, min(grid_h - 1, patch_r))
        return patch_r, patch_c

    def _patch_to_pixel(self, patch_rc: Tuple[int, int], scale_x: float, scale_y: float) -> Tuple[float, float]:
        r, c = patch_rc
        center_resized_x = (c + 0.5) * self.patch_size
        center_resized_y = (r + 0.5) * self.patch_size
        return center_resized_x / scale_x, center_resized_y / scale_y

    def start(self, image: Image.Image, keypoints: List[Dict[str, float]]) -> None:
        feats = self._extract_features(image)
        new_h, new_w = self.extractor._compute_resized_hw(*image.size)
        scale_x = new_w / image.width
        scale_y = new_h / image.height
        grid_hw = feats.shape[1:]
        self.prev_scale = (scale_x, scale_y)

        self.tracks.clear()
        for idx, kp in enumerate(keypoints):
            x = float(kp["x"])
            y = float(kp["y"])
            identifier = kp.get("id") or kp.get("label") or f"kp_{idx}"
            label = kp.get("label") or identifier
            patch_rc = self._pixel_to_patch(x, y, scale_x, scale_y, grid_hw)
            desc = feats[:, patch_rc[0], patch_rc[1]].detach().clone()
            desc = desc / (desc.norm() + 1e-12)
            self.tracks[identifier] = KeypointTrack(
                identifier, label, patch_rc, desc)

        self.prev_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    def update(self, image: Image.Image) -> List[Dict[str, float]]:
        if not self.tracks:
            return []
        feats = self._extract_features(image)
        new_h, new_w = self.extractor._compute_resized_hw(*image.size)
        scale_x = new_w / image.width
        scale_y = new_h / image.height
        grid_h, grid_w = feats.shape[1:]

        frame_array = np.array(image)
        frame_gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        flow = None
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray,
                frame_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        self.prev_gray = frame_gray

        results: List[Dict[str, float]] = []
        for track in self.tracks.values():
            prev_r, prev_c = track.patch_rc
            base_x, base_y = self._patch_to_pixel(
                (prev_r, prev_c), scale_x, scale_y)
            flow_dx = flow_dy = 0.0
            if flow is not None:
                fy = int(round(base_y))
                fx = int(round(base_x))
                if 0 <= fy < flow.shape[0] and 0 <= fx < flow.shape[1]:
                    flow_vec = flow[fy, fx]
                    flow_dx, flow_dy = float(flow_vec[0]), float(flow_vec[1])

            predicted_x = base_x + flow_dx + track.velocity[0]
            predicted_y = base_y + flow_dy + track.velocity[1]
            predicted_r, predicted_c = self._pixel_to_patch(
                predicted_x,
                predicted_y,
                scale_x,
                scale_y,
                feats.shape[1:],
            )

            radius = self.search_radius + track.misses
            r_min = max(0, predicted_r - radius)
            r_max = min(grid_h - 1, predicted_r + radius)
            c_min = max(0, predicted_c - radius)
            c_max = min(grid_w - 1, predicted_c + radius)
            best_sim = -1.0
            best_rc = (prev_r, prev_c)

            for r in range(r_min, r_max + 1):
                row_vecs = feats[:, r, c_min:c_max + 1]
                sims = torch.matmul(row_vecs.transpose(0, 1), track.descriptor)
                sims_np = sims.cpu().numpy()
                idx = np.argmax(sims_np)
                candidate_sim = float(sims_np[idx])
                candidate_c = c_min + idx
                if candidate_sim > best_sim:
                    best_sim = candidate_sim
                    best_rc = (r, candidate_c)

            if best_sim < self.min_similarity:
                track.misses += 1
                visible = False
                x, y = base_x, base_y
                if track.misses > self.max_misses:
                    track.velocity = (0.0, 0.0)
            else:
                track.misses = 0
                track.patch_rc = best_rc
                new_desc = feats[:, best_rc[0], best_rc[1]].detach()
                new_desc = new_desc / (new_desc.norm() + 1e-12)
                blended = (1.0 - self.momentum) * \
                    track.descriptor + self.momentum * new_desc
                track.descriptor = blended / (blended.norm() + 1e-12)
                x, y = self._patch_to_pixel(best_rc, scale_x, scale_y)
                delta_x = x - base_x
                delta_y = y - base_y
                track.velocity = (
                    0.6 * track.velocity[0] + 0.4 * delta_x,
                    0.6 * track.velocity[1] + 0.4 * delta_y,
                )
                visible = True

            results.append(
                {
                    "id": track.identifier,
                    "label": track.label,
                    "x": x,
                    "y": y,
                    "visible": visible,
                }
            )
        return results


class DinoKeypointVideoProcessor:
    """Video-level runner that reads annotations and writes per-frame keypoints."""

    def __init__(
        self,
        video_path: str,
        *,
        result_folder: Optional[Path],
        model_name: str,
        short_side: int = 768,
        device: Optional[str] = None,
    ) -> None:
        self.video_path = str(video_path)
        self.video_loader = CV2Video(self.video_path)
        first_frame = self.video_loader.get_first_frame()
        self.video_height, self.video_width = first_frame.shape[:2]
        self.total_frames = self.video_loader.total_frames()

        self.video_result_folder = Path(result_folder) if result_folder else Path(
            self.video_path).with_suffix("")
        self.video_result_folder.mkdir(parents=True, exist_ok=True)

        self.tracker = DinoKeypointTracker(
            model_name=model_name,
            short_side=short_side,
            device=device,
        )

    @staticmethod
    def _load_initial_annotations(folder: Path) -> Tuple[int, List[Dict[str, float]]]:
        json_files = sorted(find_manual_labeled_json_files(str(folder)))
        if not json_files:
            raise RuntimeError(
                "No labeled JSON files found. Label and save the first frame before running DINO tracking.")

        json_path = folder / json_files[0]
        with open(json_path, "r") as fp:
            data = json.load(fp)

        keypoints: List[Dict[str, float]] = []
        for idx, shape in enumerate(data.get("shapes", [])):
            if shape.get("shape_type") != "point":
                continue
            pts = shape.get("points", [])
            if not pts:
                continue
            x, y = pts[0]
            label = shape.get("label") or f"kp_{idx}"
            keypoints.append({
                "id": label,
                "label": label,
                "x": float(x),
                "y": float(y),
            })

        if not keypoints:
            raise RuntimeError(
                "The labeled frame has no point annotations. Add at least one point and save again.")

        frame_number = get_frame_number_from_json(json_files[0])
        return frame_number, keypoints

    def _write_keypoints(self, frame_number: int, keypoints: List[Dict[str, float]]) -> None:
        json_path = self.video_result_folder / \
            f"{self.video_result_folder.name}_{frame_number:09d}.json"
        label_list: List[Shape] = []
        for kp in keypoints:
            shape = Shape(
                label=kp.get("label") or kp.get("id"),
                flags={},
                description="DINOv3",
                shape_type='point',
                visible=bool(kp.get("visible", True)),
            )
            shape.points = [[float(kp["x"]), float(kp["y"])]]
            label_list.append(shape)

        save_labels(
            filename=json_path,
            imagePath="",
            label_list=label_list,
            height=self.video_height,
            width=self.video_width,
            save_image_to_json=False,
        )

    def set_pred_worker(self, pred_worker) -> None:
        self.pred_worker = pred_worker

    def process_video(
        self,
        *,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        step: int = 1,
        pred_worker=None,
    ) -> str:
        if self.total_frames <= 0:
            return "Video contains no frames."

        initial_frame, keypoints = self._load_initial_annotations(
            self.video_result_folder)
        start_frame = initial_frame if start_frame is None else max(
            initial_frame, start_frame)
        if end_frame is None or end_frame < 0:
            end_frame = self.total_frames - 1
        end_frame = min(end_frame, self.total_frames - 1)
        if start_frame >= end_frame:
            end_frame = min(self.total_frames - 1, start_frame + 1)
        step = max(1, abs(step))

        initial_frame_array = self.video_loader.load_frame(initial_frame)
        self.tracker.start(Image.fromarray(initial_frame_array), keypoints)

        frame_numbers = list(range(start_frame, end_frame + 1, step))
        total_steps = max(1, len(frame_numbers) - 1)
        processed = 0
        stopped_early = False

        for frame_number in frame_numbers:
            if frame_number == initial_frame:
                if pred_worker is not None and pred_worker.is_stopped():
                    stopped_early = True
                    break
                continue

            if pred_worker is not None and pred_worker.is_stopped():
                stopped_early = True
                break

            frame = self.video_loader.load_frame(frame_number)
            points = self.tracker.update(Image.fromarray(frame))
            if points:
                self._write_keypoints(frame_number, points)

            if pred_worker is not None and total_steps > 0:
                processed += 1
                progress = int(
                    min(100, max(0, (processed / total_steps) * 100)))
                pred_worker.report_progress(progress)

        message = "DINO keypoint tracking completed."
        if stopped_early:
            message = "DINO keypoint tracking stopped early."
        logger.info(message)
        return message
