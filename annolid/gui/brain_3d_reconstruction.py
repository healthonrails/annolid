from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from annolid.gui.brain_3d_model import (
    Brain3DConfig,
    Brain3DModel,
    PlanePolygonSet,
    apply_coronal_polygon_edit,
    build_brain_3d_model,
    export_brain_model_mesh,
    reslice_brain_model,
    set_region_presence_on_plane,
)


@dataclass(slots=True)
class Brain3DReconstructionService:
    """Reconstruction and reslicing pipeline for sagittal->3D->coronal workflows."""

    def reconstruct(
        self,
        sagittal_pages: Sequence[Any],
        config: Brain3DConfig | dict[str, Any] | None,
    ) -> Brain3DModel:
        return build_brain_3d_model(sagittal_pages, config)

    def reslice_coronal(
        self,
        model: Brain3DModel,
        *,
        spacing: float | None = None,
        plane_count: int | None = None,
    ) -> list[PlanePolygonSet]:
        return reslice_brain_model(
            model,
            orientation="coronal",
            spacing=spacing,
            plane_count=plane_count,
        )

    def apply_coronal_edit(
        self,
        model: Brain3DModel,
        *,
        plane_index: int,
        region_id: str,
        edited_shape: Any,
        guide_points=None,
        snapping_strength: float | None = None,
        snapping_max_distance: float = 8.0,
    ) -> Brain3DModel:
        return apply_coronal_polygon_edit(
            model,
            plane_index,
            region_id,
            edited_shape,
            guide_points=guide_points,
            snapping_strength=snapping_strength,
            snapping_max_distance=snapping_max_distance,
        )

    def set_presence_state(
        self,
        model: Brain3DModel,
        *,
        plane_index: int,
        region_id: str,
        state: str,
    ) -> Brain3DModel:
        return set_region_presence_on_plane(model, plane_index, region_id, state)

    def mesh_preview(
        self,
        model: Brain3DModel,
        *,
        smoothing: float | None = None,
        region_ids: set[str] | None = None,
    ):
        return export_brain_model_mesh(
            model,
            smoothing=smoothing,
            region_ids=region_ids,
        )
