from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Iterable

import numpy as np
import torch

from annolid.annotation.keypoints import save_labels
from annolid.data.videos import CV2Video
from annolid.gui.shape import Shape
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger
from annolid.utils.shapes import sample_grid_in_polygon
from annolid.utils.annotation_compat import shape_to_mask


class BasePointTracker(ABC):
    """Abstract base class for point-based trackers in Annolid."""

    def __init__(
        self,
        video_path: str,
        json_path: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self.video_path = video_path
        self.manual_json_path = Path(json_path) if json_path else None
        self._should_stop_callback = should_stop or (lambda: False)
        self.pred_worker = None
        self._stop_triggered = False

        self.video_loader = CV2Video(self.video_path)
        self.video_result_folder = Path(video_path).with_suffix("")
        self.video_result_folder.mkdir(exist_ok=True, parents=True)

        first_frame = self.video_loader.get_first_frame()
        if first_frame is None:
            raise RuntimeError(f"Video contains no frames: {video_path}")
        self.video_height, self.video_width, _ = first_frame.shape

        self.device = self._select_device()
        self.point_labels: list[str] = []
        self.start_frame = 0
        self.end_frame = 0
        self.mask = None
        self.mask_label = None
        self.track_polygon_grid_points = bool(
            kwargs.get("track_polygon_grid_points", True)
        )
        self.polygon_grid_size = kwargs.get("polygon_grid_size")

    @staticmethod
    def _select_device() -> torch.device:
        """Pick the best available accelerator in priority order."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @abstractmethod
    def load_model(self) -> Any:
        """Load the tracking model."""
        pass

    @abstractmethod
    def process_video_frames(
        self,
        start_frame: int = 0,
        end_frame: int = -1,
        grid_size: int = 10,
        grid_query_frame: int = 0,
        need_visualize: bool = False,
        **kwargs: Any,
    ) -> str:
        """Run tracking on the designated video range."""
        pass

    def set_pred_worker(self, pred_worker: Any) -> None:
        """Link a GUI worker to this processor for cancellation and progress."""
        self.pred_worker = pred_worker

    def request_stop(self) -> None:
        """Request a cooperative stop."""
        self._stop_triggered = True

    def _should_stop(self) -> bool:
        """Check if processing should be terminated."""
        if self._stop_triggered:
            return True
        if self._should_stop_callback():
            return True
        if (
            self.pred_worker
            and getattr(self.pred_worker, "is_stopped", lambda: False)()
        ):
            return True
        return False

    def load_queries(self) -> torch.Tensor:
        """Gather point prompts from labeled JSON files."""
        self.point_labels = []
        self.mask = None
        self.mask_label = None

        discovered = []
        if self.manual_json_path and self.manual_json_path.exists():
            discovered.append(self.manual_json_path)

        for filename in find_manual_labeled_json_files(self.video_result_folder):
            candidate = self.video_result_folder / filename
            if candidate.exists():
                discovered.append(candidate)

        unique_files = []
        seen = set()
        for path in sorted(discovered):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(path)

        if not unique_files:
            raise RuntimeError(
                f"No manually labeled frames found for {self.__class__.__name__}."
            )

        queries = []
        for json_path in unique_files:
            with open(json_path, "r") as file:
                data = json.load(file)

            frame_number = get_frame_number_from_json(json_path.name)
            queries.extend(self._process_shapes(data.get("shapes", []), frame_number))

        if not queries:
            raise RuntimeError(
                f"No point prompts found in labeled frames for {self.__class__.__name__}."
            )

        return torch.as_tensor(queries, dtype=torch.float32, device=self.device)

    def _process_shapes(
        self, shapes: list[dict], frame_number: int
    ) -> list[list[float]]:
        processed_queries = []
        for shape in shapes:
            label = shape.get("label")
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type == "point" and shape.get("points"):
                processed_queries.append(
                    self._process_point(shape["points"][0], frame_number, label)
                )
            elif shape_type == "polygon" and shape.get("points"):
                description = str(shape.get("description") or "").lower()
                polygon_prompt_enabled = self.track_polygon_grid_points or any(
                    key in description for key in ("grid", "point")
                )
                if not polygon_prompt_enabled:
                    continue
                processed_queries.extend(
                    self._process_polygon(shape.get("points", []), frame_number, label)
                )
        return processed_queries

    def _process_point(
        self, point: list[float], frame_number: int, label: str
    ) -> list[float]:
        self.point_labels.append(label)
        return [frame_number] + list(point)

    def _process_polygon(
        self, points: list[list[float]], frame_number: int, label: str
    ) -> list[list[float]]:
        queries = []
        if not points:
            return queries
        if self.mask is None:
            self.mask_label = label
            img_shape = (self.video_height, self.video_width)
            self.mask = shape_to_mask(img_shape, points, shape_type="polygon").astype(
                np.uint8
            )
            self.mask = torch.from_numpy(self.mask)[None, None].to(self.device)

        points_in_polygon = sample_grid_in_polygon(
            points, grid_size=self.polygon_grid_size
        )
        if points_in_polygon is None or len(points_in_polygon) == 0:
            # Very small/thin polygons can yield zero strict "inside" samples.
            # Fallback to polygon centroid so tracking can still bootstrap.
            pts = np.asarray(points, dtype=np.float32)
            if pts.size == 0:
                return queries
            centroid = pts.mean(axis=0)
            points_in_polygon = np.asarray([centroid], dtype=np.float32)

        for point in points_in_polygon:
            self.point_labels.append(label)
            queries.append([frame_number] + list(point))
        return queries

    def save_frame_json(
        self,
        frame_number: int,
        points: list[tuple[np.ndarray, bool]],
        description: str = "",
    ):
        """Save tracked points for a single frame to JSON."""
        json_file_path = self.video_result_folder / (
            self.video_result_folder.name + f"_{frame_number:0>{9}}.json"
        )
        label_list = []
        for label, (coord, visible) in zip(self.point_labels, points):
            cur_shape = Shape(
                label=label,
                flags={},
                description=description or self.__class__.__name__,
                shape_type="point",
                visible=visible,
            )
            cur_shape.points = [coord.tolist()]
            label_list.append(cur_shape)

        save_labels(
            json_file_path,
            imagePath="",
            label_list=label_list,
            width=self.video_width,
            height=self.video_height,
            persist_json=False,
        )

    def extract_frame_points(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor | None = None,
        *,
        chunk_start_frame: int = 0,
        local_frame_indices: Optional[Iterable[int]] = None,
        description: str = "",
    ) -> str:
        """Standardized method to persist tracks to JSON files."""
        # tracks: [B, T, N, 2] -> we assume B=1
        tracks_np = tracks[0].float().detach().cpu().numpy()
        num_local_frames = tracks_np.shape[0]

        if local_frame_indices is None:
            local_frame_indices = range(num_local_frames)

        last_saved_frame = None
        for local_t in local_frame_indices:
            global_frame = chunk_start_frame + int(local_t)
            if global_frame < self.start_frame or (
                self.end_frame >= 0 and global_frame > self.end_frame
            ):
                continue

            points = []
            for point_idx in range(tracks_np.shape[1]):
                coord = tracks_np[local_t, point_idx]
                visible = True
                if visibility is not None:
                    visible = bool(visibility[0, local_t, point_idx].item())
                points.append((coord, visible))

            self.save_frame_json(global_frame, points, description)
            last_saved_frame = global_frame

        msg = (
            f"Saved tracks up to frame #{last_saved_frame}"
            if last_saved_frame is not None
            else "No frames saved"
        )
        logger.info(msg)
        return msg

    def _stop_message(self, frame_number: int) -> str:
        """Standardized stop message."""
        return f"{self.__class__.__name__} stopped by user#{frame_number}"
