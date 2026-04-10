from __future__ import annotations

from annolid.gui.brain_3d_model import Brain3DConfig
from annolid.gui.brain_3d_reconstruction import Brain3DReconstructionService


def _shape(label: str, points):
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {},
        "description": "",
        "visible": True,
    }


def test_reconstruction_service_pipeline() -> None:
    service = Brain3DReconstructionService()
    pages = [
        {
            "page_index": 0,
            "shapes": [_shape("region_k", [[0, 0], [8, 0], [8, 8], [0, 8]])],
        },
        {
            "page_index": 2,
            "shapes": [_shape("region_k", [[2, 0], [10, 0], [10, 8], [2, 8]])],
        },
    ]
    model = service.reconstruct(
        pages,
        Brain3DConfig(
            point_count=12, smoothing_longitudinal=0.4, smoothing_inplane=0.2
        ),
    )
    planes = service.reslice_coronal(model, plane_count=4)
    assert len(planes) == 4
    region_id = next(iter(model.regions.keys()))
    service.set_presence_state(
        model,
        plane_index=1,
        region_id=region_id,
        state="zero_area",
    )
    planes_after = service.reslice_coronal(model, plane_count=4)
    region = next(r for r in planes_after[1].regions if r.region_id == region_id)
    assert region.state == "zero_area"
