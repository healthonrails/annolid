from __future__ import annotations

from qtpy import QtCore

from annolid.gui.shape import Shape
from annolid.gui.brain_3d_model import (
    Brain3DConfig,
    build_brain_3d_model,
    materialize_coronal_plane_shapes,
    reslice_brain_model,
    set_region_presence_on_plane,
    store_brain_model_in_other_data,
)
from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin


class _CanvasStub:
    def __init__(self) -> None:
        self.shapes = []
        self.selectedShapes = []

    def selectShapes(self, shapes) -> None:
        self.selectedShapes = list(shapes or [])

    def update(self) -> None:
        return None


class _Brain3DHost(AnnotationLoadingMixin):
    def __init__(self) -> None:
        self._config = {"label_flags": {}}
        self.canvas = _CanvasStub()
        self.large_image_view = None
        self.otherData = {}
        self.frame_number = 0
        self._planes = []
        self._has_large_image = True
        self.status_messages: list[str] = []

    def _has_large_image_page_navigation(self) -> bool:
        return bool(self._has_large_image)

    def loadShapes(self, shapes, replace=True) -> None:
        _ = replace
        self.canvas.shapes = list(shapes or [])

    def setLargeImagePageNumber(self, page_index: int) -> None:
        self.frame_number = int(page_index)
        plane = self._planes[int(page_index)]
        self.loadShapes(materialize_coronal_plane_shapes(plane, include_hidden=False))

    def shapeSelectionChanged(self, _shapes) -> None:
        return None

    def status(self, message, *_args, **_kwargs) -> None:
        self.status_messages.append(str(message))

    def tr(self, text: str) -> str:
        return str(text)


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


def test_brain3d_region_selection_moves_to_nearest_visible_plane() -> None:
    pages = [
        {
            "page_index": 0,
            "shapes": [_shape("region_l", [[0, 0], [8, 0], [8, 8], [0, 8]])],
        },
        {
            "page_index": 1,
            "shapes": [_shape("region_l", [[1, 0], [9, 0], [9, 8], [1, 8]])],
        },
        {
            "page_index": 2,
            "shapes": [_shape("region_l", [[2, 0], [10, 0], [10, 8], [2, 8]])],
        },
    ]
    model = build_brain_3d_model(
        pages, Brain3DConfig(point_count=8, coronal_plane_count=3)
    )
    region_id = next(iter(model.regions.keys()))
    set_region_presence_on_plane(model, 0, region_id, "hidden")

    host = _Brain3DHost()
    host.otherData = store_brain_model_in_other_data({}, model)
    host._planes = reslice_brain_model(model, plane_count=3)
    host.frame_number = 0
    host.loadShapes(
        materialize_coronal_plane_shapes(host._planes[0], include_hidden=False)
    )
    assert host.canvas.shapes == []

    host._onBrain3DRegionSelectionChanged(region_id)

    assert host.frame_number == 1
    assert host.canvas.selectedShapes
    selected = host.canvas.selectedShapes[0]
    region = host._brain3d_region_id_from_shape(selected)
    assert region == region_id


def test_brain3d_best_plane_returns_available_non_hidden_plane() -> None:
    pages = [
        {
            "page_index": 0,
            "shapes": [_shape("region_m", [[0, 0], [8, 0], [8, 8], [0, 8]])],
        },
        {
            "page_index": 1,
            "shapes": [_shape("region_m", [[1, 0], [9, 0], [9, 8], [1, 8]])],
        },
        {
            "page_index": 2,
            "shapes": [_shape("region_m", [[2, 0], [10, 0], [10, 8], [2, 8]])],
        },
    ]
    model = build_brain_3d_model(
        pages, Brain3DConfig(point_count=8, coronal_plane_count=3)
    )
    region_id = next(iter(model.regions.keys()))
    set_region_presence_on_plane(model, 0, region_id, "hidden")
    host = _Brain3DHost()
    host.otherData = store_brain_model_in_other_data({}, model)
    host.frame_number = 2
    best = host._brain3d_best_plane_for_region(model, region_id)
    assert best is not None
    assert best in {1, 2}


def test_brain3d_mesh_pick_falls_back_without_session_dock() -> None:
    host = _Brain3DHost()
    host.brain3d_session_dock = None
    captured: list[str] = []

    def _capture(region_id: str) -> None:
        captured.append(str(region_id or ""))

    host._onBrain3DRegionSelectionChanged = _capture  # type: ignore[method-assign]
    host._onBrain3DMeshRegionPicked("region_n||")

    assert captured == ["region_n||"]
    assert getattr(host, host._BRAIN3D_PENDING_REGION_KEY, "") == "region_n||"


def test_annotation_dirty_invalidates_brain3d_after_sagittal_source_edit() -> None:
    pages = [
        {
            "page_index": 0,
            "shapes": [_shape("region_o", [[0, 0], [8, 0], [8, 8], [0, 8]])],
        },
        {
            "page_index": 1,
            "shapes": [_shape("region_o", [[1, 0], [9, 0], [9, 8], [1, 8]])],
        },
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    host = _Brain3DHost()
    host.otherData = store_brain_model_in_other_data({}, model)
    host.frame_number = 0
    # Edit source polygon.
    mutated = Shape(label="region_o", shape_type="polygon")
    for x, y in [(2.0, 0.0), (10.0, 0.0), (10.0, 8.0), (2.0, 8.0)]:
        mutated.addPoint(QtCore.QPointF(float(x), float(y)))
    mutated.close()
    host.canvas.shapes = [mutated]

    host._onAnnotationDirty()

    assert "brain_3d_model" not in host.otherData
    sync = dict(host.otherData.get("brain3d_sync") or {})
    assert sync.get("invalidated") is True
    assert sync.get("reason") == "source_page_changed"
    assert host.status_messages


def test_brain3d_prepare_save_syncs_generated_coronal_edits_first() -> None:
    pages = [
        {
            "page_index": 0,
            "shapes": [_shape("region_p", [[0, 0], [8, 0], [8, 8], [0, 8]])],
        },
        {
            "page_index": 1,
            "shapes": [_shape("region_p", [[1, 0], [9, 0], [9, 8], [1, 8]])],
        },
    ]
    model = build_brain_3d_model(pages, Brain3DConfig(point_count=8))
    host = _Brain3DHost()
    host.otherData = store_brain_model_in_other_data(
        {
            "large_image_page": {
                "page_index": 0,
                "brain3d_generated": True,
                "brain3d_orientation": "coronal",
            }
        },
        model,
    )
    calls: list[bool] = []

    def _sync(_value=False) -> bool:
        calls.append(True)
        return True

    host.applyCurrentCoronalEditsToBrain3DModel = _sync  # type: ignore[method-assign]
    host._brain3d_prepare_save("dummy.json")
    assert calls == [True]


def test_brain3d_actions_create_hide_restore_selected_region_on_plane() -> None:
    host = _Brain3DHost()
    states: list[str] = []

    def _apply(state: str) -> bool:
        states.append(str(state or ""))
        return True

    host._applyBrain3DSelectedRegionState = _apply  # type: ignore[method-assign]

    assert host.createBrain3DRegionOnPlane() is True
    assert host.hideBrain3DRegionOnPlane() is True
    assert host.restoreBrain3DRegionOnPlane() is True
    assert states == ["created", "hidden", "present"]


def test_brain3d_config_overrides_include_interpolation_and_snapping_controls() -> None:
    host = _Brain3DHost()
    config = host._brain3d_config_from_overrides(
        {
            "point_count": 72,
            "interpolation_density": 4,
            "coronal_plane_count": 20,
            "coronal_spacing": None,
            "smoothing_longitudinal": 0.35,
            "smoothing_inplane": 0.2,
            "snapping_enabled": True,
            "snapping_strength": 0.5,
            "snapping_max_distance": 14.0,
        }
    )
    assert config.point_count == 72
    assert config.interpolation_density == 4
    assert config.coronal_plane_count == 20
    assert config.coronal_spacing is None
    assert abs(config.smoothing_longitudinal - 0.35) < 1e-9
    assert abs(config.smoothing_inplane - 0.2) < 1e-9
    assert config.snapping_enabled is True
    assert abs(config.snapping_strength - 0.5) < 1e-9
    assert abs(config.snapping_max_distance - 14.0) < 1e-9
