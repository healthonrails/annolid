from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    set_visibility_flags,
    visibility_from_labelme_shape,
)
from annolid.annotation.keypoints import save_labels
from annolid.data.videos import CV2Video
from annolid.gui.shape import Shape
from annolid.utils.annotation_compat import shape_to_mask
from annolid.utils.annotation_store import AnnotationStoreError, load_labelme_json
from annolid.utils.cancellation import is_stop_requested
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.utils.logger import logger
from annolid.utils.shapes import sample_grid_in_polygon
from annolid.utils.zone_shapes import is_zone_shape_payload


@dataclass(frozen=True)
class PointQuerySpec:
    """Metadata preserved for each point query across point-tracking backends."""

    label: str
    group_id: Optional[int | str] = None
    instance_label: str = ""
    display_label: str = ""
    query_index: Optional[int] = None


class BasePointTracker(ABC):
    """Abstract base class for point-based trackers in Annolid."""

    allow_force_thread_terminate = False

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
        self._stop_event = kwargs.get("stop_event")
        self._stop_triggered = False

        self.video_loader = CV2Video(self.video_path)
        custom_folder = kwargs.get("results_folder") or kwargs.get("result_folder")
        self.video_result_folder = (
            Path(custom_folder) if custom_folder else Path(video_path).with_suffix("")
        )
        self.video_result_folder.mkdir(exist_ok=True, parents=True)
        self.persist_json = bool(kwargs.get("persist_json", False))

        first_frame = self.video_loader.get_first_frame()
        if first_frame is None:
            raise RuntimeError(f"Video contains no frames: {video_path}")
        self.video_height, self.video_width, _ = first_frame.shape

        self.device = self._select_device()
        self.point_labels: list[str] = []
        self.point_specs: list[PointQuerySpec] = []
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
        stop_event = self._resolve_stop_event()
        if stop_event is not None and callable(getattr(stop_event, "set", None)):
            try:
                stop_event.set()
            except Exception:
                pass

    def _resolve_stop_event(self) -> Any:
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            return stop_event
        pred_worker = getattr(self, "pred_worker", None)
        if pred_worker is not None:
            return getattr(pred_worker, "stop_event", None)
        return None

    def _should_stop(self) -> bool:
        """Check if processing should be terminated."""
        if self._stop_triggered:
            return True
        if is_stop_requested(self._resolve_stop_event()):
            return True
        if self._should_stop_callback():
            return True
        if (
            self.pred_worker
            and getattr(self.pred_worker, "is_stopped", lambda: False)()
        ):
            return True
        return False

    def _ensure_point_specs_list(self) -> list[PointQuerySpec]:
        specs = getattr(self, "point_specs", None)
        if specs is None:
            specs = []
            self.point_specs = specs
        return specs

    def _frame_seed_priority(
        self, frame_number: int, is_explicit: bool
    ) -> tuple[int, int, int]:
        """Rank candidate seed frames so the closest relevant manual seed wins on duplicates."""
        start = int(getattr(self, "start_frame", 0) or 0)
        end = int(
            getattr(self, "end_frame", -1)
            if getattr(self, "end_frame", -1) is not None
            else -1
        )
        in_window = frame_number >= start and (end < 0 or frame_number <= end)
        at_or_before_start = frame_number <= start
        if in_window:
            bucket = 0
            distance = abs(frame_number - start)
        elif at_or_before_start:
            bucket = 1
            distance = start - frame_number
        else:
            bucket = 2
            distance = frame_number - (end if end >= 0 else start)
        return (0 if is_explicit else 1, bucket, distance)

    @staticmethod
    def _load_annotation_payload(json_path: Path) -> dict[str, Any]:
        try:
            payload = load_labelme_json(json_path)
            if isinstance(payload, dict):
                return payload
        except (
            AnnotationStoreError,
            FileNotFoundError,
            ValueError,
            json.JSONDecodeError,
        ):
            pass
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def load_queries(self) -> torch.Tensor:
        """Gather point prompts from labeled JSON files and AnnotationStore records."""
        self.point_labels = []
        self.point_specs = []
        self.mask = None
        self.mask_label = None

        explicit_resolved = (
            self.manual_json_path.resolve()
            if self.manual_json_path and self.manual_json_path.exists()
            else None
        )
        discovered: list[Path] = []
        if self.manual_json_path and self.manual_json_path.exists():
            discovered.append(self.manual_json_path)

        for filename in find_manual_labeled_json_files(str(self.video_result_folder)):
            candidate = self.video_result_folder / filename
            if candidate.exists():
                discovered.append(candidate)

        unique_files: list[Path] = []
        seen: set[Path] = set()
        for path in sorted(discovered):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(path)

        if not unique_files:
            raise RuntimeError(
                f"No manually labeled frames found for {self.__class__.__name__}."
            )

        # Rank manual files so if the same named keypoint appears in multiple manual
        # frames, we keep the seed closest to the active tracking range rather than
        # emitting duplicate tracks for the same landmark.
        ranked_files = sorted(
            unique_files,
            key=lambda p: (
                *self._frame_seed_priority(
                    get_frame_number_from_json(p.name),
                    explicit_resolved is not None and p.resolve() == explicit_resolved,
                ),
                get_frame_number_from_json(p.name),
            ),
        )

        selected_entries: list[tuple[int, list[float], PointQuerySpec]] = []
        seen_identities: set[tuple[str, Optional[int | str], str, Optional[int]]] = (
            set()
        )

        for json_path in ranked_files:
            data = self._load_annotation_payload(json_path)
            frame_number = get_frame_number_from_json(json_path.name)

            prev_count = len(self.point_specs)
            frame_queries = self._process_shapes(data.get("shapes", []), frame_number)
            frame_specs = self.point_specs[prev_count:]
            # Roll back provisional append so we can filter duplicate identities cleanly.
            del self.point_specs[prev_count:]
            del self.point_labels[prev_count:]

            for q, spec in zip(frame_queries, frame_specs):
                identity = (
                    spec.label,
                    spec.group_id,
                    spec.instance_label,
                    spec.query_index,
                )
                if identity in seen_identities:
                    continue
                seen_identities.add(identity)
                selected_entries.append((frame_number, q, spec))

        # Preserve deterministic chronological order of selected queries.
        selected_entries.sort(key=lambda item: item[0])
        queries: list[list[float]] = []
        for _, q, spec in selected_entries:
            self.point_labels.append(spec.label)
            self.point_specs.append(spec)
            queries.append(q)

        if not queries:
            raise RuntimeError(
                f"No point prompts found in labeled frames for {self.__class__.__name__}."
            )

        return torch.as_tensor(queries, dtype=torch.float32, device=self.device)

    @staticmethod
    def _extract_shape_metadata(
        shape: Mapping[str, Any],
    ) -> tuple[Optional[int | str], str, str]:
        raw_gid = shape.get("group_id")
        group_id: Optional[int | str] = None
        if raw_gid is not None and not isinstance(raw_gid, bool):
            if isinstance(raw_gid, int):
                group_id = int(raw_gid)
            elif isinstance(raw_gid, float) and math.isfinite(raw_gid):
                group_id = int(raw_gid) if raw_gid.is_integer() else raw_gid
            else:
                text = str(raw_gid).strip()
                if text:
                    group_id = int(text) if text.lstrip("+-").isdigit() else text

        instance_label = str(shape.get("instance_label") or "").strip()
        display_label = str(shape.get("display_label") or "").strip()
        for key in ("flags", "other_data", "otherData"):
            container = shape.get(key)
            if isinstance(container, Mapping):
                if not instance_label:
                    instance_label = str(container.get("instance_label") or "").strip()
                if not display_label:
                    display_label = str(container.get("display_label") or "").strip()
        return group_id, instance_label, display_label

    def _process_shapes(
        self, shapes: list[dict], frame_number: int
    ) -> list[list[float]]:
        processed_queries: list[list[float]] = []
        for shape in shapes:
            if is_zone_shape_payload(shape):
                continue
            label = str(shape.get("label") or "")
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            group_id, instance_label, display_label = self._extract_shape_metadata(
                shape
            )
            if shape_type == "point" and shape.get("points"):
                kp_vis = visibility_from_labelme_shape(shape)
                if kp_vis == int(KeypointVisibility.MISSING):
                    continue
                pt = shape["points"][0]
                if (
                    pt is None
                    or len(pt) < 2
                    or pt[0] is None
                    or pt[1] is None
                    or not (math.isfinite(float(pt[0])) and math.isfinite(float(pt[1])))
                ):
                    continue
                processed_queries.append(
                    self._process_point(
                        pt,
                        frame_number,
                        label,
                        group_id=group_id,
                        instance_label=instance_label,
                        display_label=display_label,
                    )
                )
            elif shape_type == "polygon" and shape.get("points"):
                description = str(shape.get("description") or "").lower()
                polygon_prompt_enabled = self.track_polygon_grid_points or any(
                    key in description for key in ("grid", "point")
                )
                if not polygon_prompt_enabled:
                    continue
                processed_queries.extend(
                    self._process_polygon(
                        shape.get("points", []),
                        frame_number,
                        label,
                        group_id=group_id,
                        instance_label=instance_label,
                    )
                )
        return processed_queries

    def _process_point(
        self,
        point: list[float],
        frame_number: int,
        label: str,
        *,
        group_id: Optional[int | str] = None,
        instance_label: str = "",
        display_label: str = "",
    ) -> list[float]:
        self.point_labels.append(label)
        self._ensure_point_specs_list().append(
            PointQuerySpec(
                label=str(label or ""),
                group_id=group_id,
                instance_label=instance_label,
                display_label=display_label,
                query_index=None,
            )
        )
        return [float(frame_number), float(point[0]), float(point[1])]

    def _process_polygon(
        self,
        points: list[list[float]],
        frame_number: int,
        label: str,
        *,
        group_id: Optional[int | str] = None,
        instance_label: str = "",
    ) -> list[list[float]]:
        queries: list[list[float]] = []
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

        specs = self._ensure_point_specs_list()
        for idx, point in enumerate(points_in_polygon):
            self.point_labels.append(label)
            specs.append(
                PointQuerySpec(
                    label=str(label or ""),
                    group_id=group_id,
                    instance_label=instance_label,
                    display_label="",
                    query_index=int(idx),
                )
            )
            queries.append([float(frame_number), float(point[0]), float(point[1])])
        return queries

    def _resolved_specs_for_save(self, count: int) -> list[PointQuerySpec]:
        specs = getattr(self, "point_specs", None)
        if isinstance(specs, list) and len(specs) == count:
            return specs
        labels = list(getattr(self, "point_labels", []) or [])
        resolved: list[PointQuerySpec] = []
        for idx in range(min(count, len(labels))):
            lbl = str(labels[idx] or "")
            # Disambiguate repeated bare labels when point_specs wasn't populated
            # (e.g., direct point_labels setup in tests).
            q_idx = idx if labels.count(labels[idx]) > 1 else None
            resolved.append(PointQuerySpec(label=lbl, query_index=q_idx))
        return resolved

    def save_frame_json(
        self,
        frame_number: int,
        points: list[tuple[np.ndarray, bool]],
        description: str = "",
    ):
        """Save tracked points for a single frame to AnnotationStore / JSON."""
        json_file_path = self.video_result_folder / (
            self.video_result_folder.name + f"_{frame_number:0>{9}}.json"
        )
        specs = self._resolved_specs_for_save(len(points))
        label_list: list[Shape] = []
        for spec, (coord, visible) in zip(specs, points):
            coord_arr = np.asarray(coord, dtype=np.float32).reshape(-1)
            if coord_arr.size < 2 or not (
                math.isfinite(float(coord_arr[0]))
                and math.isfinite(float(coord_arr[1]))
            ):
                continue
            flags: dict[str, Any] = {}
            if spec.instance_label:
                flags["instance_label"] = spec.instance_label
            if spec.display_label:
                flags["display_label"] = spec.display_label
            if spec.query_index is not None:
                flags["query_index"] = int(spec.query_index)
            set_visibility_flags(
                flags,
                int(
                    KeypointVisibility.VISIBLE
                    if visible
                    else KeypointVisibility.OCCLUDED
                ),
            )
            cur_shape = Shape(
                label=spec.label,
                group_id=spec.group_id,
                flags=flags,
                description=description or self.__class__.__name__,
                shape_type="point",
                visible=bool(visible),
            )
            cur_shape.points = [[float(coord_arr[0]), float(coord_arr[1])]]
            label_list.append(cur_shape)

        save_labels(
            json_file_path,
            imagePath="",
            label_list=label_list,
            width=self.video_width,
            height=self.video_height,
            persist_json=bool(getattr(self, "persist_json", False)),
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
