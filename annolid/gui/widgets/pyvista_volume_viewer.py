from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

from annolid.utils.logger import logger
from annolid.gui.widgets.pyvista_qt_backend import (
    RenderQtBackend,
    select_render_qt_backend,
)
from annolid.gui.widgets import pyvista_runtime as pv_rt
from annolid.gui.widgets.volume_loader import VolumeSourceLoader
from annolid.gui.widgets.volume_types import (
    VolumeData,
    _OverlayVolumeEntry,
)
from annolid.gui.widgets.volume_readers import (
    VolumeReaderConfig,
    VolumeReaders,
)
from annolid.gui.widgets.volume_viewer_controls import (
    VolumeControlsGroup,
    SliceViewerControls,
)

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui

try:
    import pyvista as pv
except Exception:  # pragma: no cover - optional dependency path
    pv = None  # type: ignore[assignment]

# Re-export runtime symbols locally for readability.
_HAS_VTK = pv_rt.HAS_VTK
_RENDER_IMPORT_ERROR = pv_rt.VTK_IMPORT_ERROR
_HAS_PLANE_WIDGET = pv_rt.HAS_PLANE_WIDGET
_HAS_GDCM = getattr(pv_rt, "HAS_GDCM", False)
vtkRenderer = pv_rt.vtkRenderer
vtkVolume = pv_rt.vtkVolume
vtkVolumeProperty = pv_rt.vtkVolumeProperty
vtkSmartVolumeMapper = pv_rt.vtkSmartVolumeMapper
vtkImageData = pv_rt.vtkImageData
vtkPiecewiseFunction = pv_rt.vtkPiecewiseFunction
vtkColorTransferFunction = pv_rt.vtkColorTransferFunction
vtkPolyDataMapper = pv_rt.vtkPolyDataMapper
vtkActor = pv_rt.vtkActor
vtkInteractorStyleTrackballCamera = pv_rt.vtkInteractorStyleTrackballCamera
vtkInteractorStyleUser = pv_rt.vtkInteractorStyleUser
vtkImagePlaneWidget = pv_rt.vtkImagePlaneWidget
QVTKRenderWindowInteractor = pv_rt.QVTKRenderWindowInteractor

POINT_CLOUD_EXTS = (".ply", ".csv", ".xyz")
MESH_EXTS = (".stl", ".obj")
DICOM_EXTS = (".dcm", ".dicom", ".ima")
TIFF_SUFFIXES = (".tif", ".tiff")
OME_TIFF_SUFFIXES = (".ome.tif", ".ome.tiff")
AUTO_OUT_OF_CORE_MB = 2048.0
MAX_VOLUME_VOXELS = 1_073_741_824
SLICE_MODE_BYTES = 1_073_741_824.0
VOLUME_SOURCE_FILTERS = (
    "3D sources (*.tif *.tiff *.ome.tif *.ome.tiff "
    "*.nii *.nii.gz *.hdr *.img *.dcm *.dicom *.ima *.IMA *.zarr *.zarr.json *.zgroup);;All files (*.*)"
)

PLANE_DEFS = (
    (0, "Axial (Z)", "SetPlaneOrientationToZAxes"),
    (1, "Coronal (Y)", "SetPlaneOrientationToYAxes"),
    (2, "Sagittal (X)", "SetPlaneOrientationToXAxes"),
)
PLANE_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.9, 0.2, 0.2),
    1: (0.2, 0.9, 0.2),
    2: (0.2, 0.2, 0.9),
}


class PyVistaVolumeViewerDialog(QtWidgets.QDialog):
    """
    Main dialog for 3D Volume interaction using PyVista and VTK.
    Supports 3D volume rendering,Dedicated 2D slice viewing,
    point clouds, meshes, and overlayers.
    """

    def __init__(
        self, path: Optional[str] = None, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        if pv is None:
            raise ModuleNotFoundError(
                "PyVista is not installed. Install optional dependencies to enable the 3D PyVista viewer."
            )
        if not _HAS_VTK:
            detail = (
                f": {_RENDER_IMPORT_ERROR}" if _RENDER_IMPORT_ERROR is not None else ""
            )
            raise ModuleNotFoundError(f"PyVista/VTK runtime is unavailable{detail}")
        if QVTKRenderWindowInteractor is None:
            raise ModuleNotFoundError(
                "pyvistaqt is not installed. Install optional dependencies to enable the 3D PyVista viewer."
            )
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Annolid 3D Volume Viewer")
        self.resize(1200, 900)

        self._path = Path(path) if path else None
        self._volume: Optional[VolumeData] = None
        self._overlays: list[_OverlayVolumeEntry] = []
        self._slice_mode = False

        # Core rendering components
        self._scene_backend: RenderQtBackend = select_render_qt_backend(
            widget_cls=QVTKRenderWindowInteractor,
            prefer_pyvista=True,
        ).backend
        self.renderer = vtkRenderer()
        self.render_widget: Optional[QtWidgets.QWidget] = None
        self.interactor: Any = None

        self._scene_initialized = False
        self._load_in_progress = False
        self._render_reset_camera_pending = False
        self._initial_source_loaded = False
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._run_render_cycle)

        # Readers and Loaders
        reader_config = VolumeReaderConfig(
            dicom_exts=DICOM_EXTS,
            tiff_suffixes=TIFF_SUFFIXES,
            ome_tiff_suffixes=OME_TIFF_SUFFIXES,
            auto_out_of_core_mb=AUTO_OUT_OF_CORE_MB,
            max_volume_voxels=MAX_VOLUME_VOXELS,
            slice_mode_bytes=SLICE_MODE_BYTES,
        )
        self._readers = VolumeReaders(
            config=reader_config,
            use_gdcm=bool(_HAS_GDCM),
        )
        self._loader = VolumeSourceLoader(readers=self._readers)

        # UI Components
        self._init_ui()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._set_cursor_safe(QtCore.Qt.ArrowCursor)
        if self._path and not self._initial_source_loaded:
            self._initial_source_loaded = True
            QtCore.QTimer.singleShot(0, self._load_initial_source)

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout()

        # Left Panel (Controls)
        self.controls_stack = QtWidgets.QStackedWidget()

        # 3D Controls
        self.volume_controls = VolumeControlsGroup(
            parent=self,
            on_load_volume=self._on_load_volume_clicked,
            on_reload_volume=self._on_reload_volume_clicked,
            on_load_points=self._on_load_points_clicked,
            on_load_mesh=self._on_load_mesh_clicked,
            on_save_snapshot=self._on_save_snapshot_clicked,
            on_reset_view=self._on_reset_view_clicked,
            on_show_help=self._on_show_help_clicked,
            on_opacity_changed=self._on_opacity_changed,
            on_shading_toggled=self._on_shading_toggled,
            on_visibility_toggled=self._on_visibility_toggled,
            on_intensity_changed=self._on_intensity_changed,
            on_plane_toggled=self._on_plane_visibility_changed,
            on_plane_moved=self._on_plane_position_changed,
        )
        self.controls_stack.addWidget(self.volume_controls)

        # Slice Controls
        self.slice_viewer_controls = SliceViewerControls(
            parent=self,
            on_index_changed=self._on_slice_index_changed,
            on_axis_changed=self._on_slice_axis_changed,
            on_window_changed=self._on_slice_window_changed,
            on_gamma_changed=self._on_slice_gamma_changed,
            on_auto_window=self._slice_auto_window,
        )
        self.controls_stack.addWidget(self.slice_viewer_controls)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls_stack)
        scroll.setFixedWidth(300)
        layout.addWidget(scroll)

        self.render_area = QtWidgets.QWidget(self)
        self.render_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.render_area.setLayout(QtWidgets.QVBoxLayout())
        self.render_area.layout().setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.render_area, 1)

        # Status area
        self.status_bar = QtWidgets.QStatusBar()
        status_layout = QtWidgets.QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status_bar)

        main_v_layout = QtWidgets.QVBoxLayout()
        main_v_layout.addLayout(layout, 1)
        main_v_layout.addWidget(self.status_bar)
        self.setLayout(main_v_layout)

        # Info Label
        self.info_label = QtWidgets.QLabel("Ready")
        self.status_bar.addPermanentWidget(self.info_label)

    def _ensure_scene_initialized(self) -> bool:
        if self._scene_initialized:
            return True
        try:
            if self.render_widget is None:
                self.render_widget = self._scene_backend.create_widget(self.render_area)
                self.render_area.layout().addWidget(self.render_widget)

            render_window = self._scene_backend.get_render_window(self.render_widget)
            if render_window is None:
                return False

            render_window.AddRenderer(self.renderer)
            self.interactor = self._scene_backend.get_interactor(
                self.render_widget, render_window
            )
            try:
                init_fn = getattr(self.interactor, "Initialize", None)
                if callable(init_fn):
                    init_fn()
            except Exception:
                pass

            # Default style
            if hasattr(pv_rt, "vtkInteractorStyleTrackballCamera"):
                style = pv_rt.vtkInteractorStyleTrackballCamera()
                self.interactor.SetInteractorStyle(style)

            self._scene_initialized = True
            return True
        except Exception as exc:
            logger.exception("Failed to initialize 3D scene: %s", exc)
            return False

    def _present_scene(self) -> None:
        if not self._scene_initialized:
            logger.info("3D viewer render: skipped because scene is not initialized.")
            return
        rw = self._scene_backend.get_render_window(self.render_widget)
        if rw:
            logger.info("3D viewer render: calling render window Render().")
            self._scene_backend.render(self.render_widget, rw)
        else:
            logger.warning("3D viewer render: no render window available.")

    def _scene_add_mesh(self, mesh: Any, **kwargs: Any) -> Any:
        """Add mesh through the active render backend with VTK fallback."""
        widget = getattr(self, "render_widget", None)
        if widget is not None:
            add_mesh = getattr(widget, "add_mesh", None)
            if callable(add_mesh):
                try:
                    return add_mesh(mesh, render=False, reset_camera=False, **kwargs)
                except Exception:
                    logger.debug(
                        "3D viewer mesh: backend add_mesh failed, trying VTK fallback.",
                        exc_info=True,
                    )
        mapper_cls = vtkPolyDataMapper
        actor_cls = vtkActor
        if mapper_cls is None or actor_cls is None:
            raise RuntimeError("VTK mesh classes are unavailable.")
        poly = mesh
        cast_to_polydata = getattr(mesh, "cast_to_polydata", None)
        if callable(cast_to_polydata):
            poly = cast_to_polydata()
        mapper = mapper_cls()
        mapper.SetInputData(poly)
        actor = actor_cls()
        actor.SetMapper(mapper)
        self._scene_backend.add_actor(self.renderer, widget, actor)
        return actor

    def _request_scene_render(self, *, reset_camera: bool = False) -> None:
        lifecycle = getattr(self, "_lifecycle_controller", None)
        if lifecycle is not None:
            request_render = getattr(lifecycle, "request_render", None)
            if callable(request_render):
                request_render(reset_camera=reset_camera)
                return
        if reset_camera:
            self._render_reset_camera_pending = True
        timer = getattr(self, "_render_timer", None)
        if timer is None:
            return
        if timer.isActive():
            return
        timer.start(0)

    def _run_render_cycle(self) -> None:
        if not self._scene_initialized:
            logger.info(
                "3D viewer render: cycle skipped because scene is not initialized."
            )
            return
        logger.info(
            "3D viewer render: cycle start reset_camera=%s has_volume=%s",
            bool(self._render_reset_camera_pending),
            bool(getattr(self, "_volume", None)),
        )
        if self._render_reset_camera_pending:
            try:
                self._scene_backend.reset_camera(self.renderer, self.render_widget)
            except Exception:
                pass
            self._render_reset_camera_pending = False
        self._present_scene()

    def _load_initial_source(self) -> None:
        if self._path:
            self._load_source_path(self._path)

    def _load_source_path(self, path: Path) -> bool:
        if not self._ensure_scene_initialized():
            return False

        self._load_in_progress = True
        if getattr(self, "info_label", None) is not None:
            self.info_label.setText(f"Loading: {path.name}...")
        self._set_cursor_safe(QtCore.Qt.WaitCursor)

        try:
            # Check for volume
            loader_fn = getattr(self._loader, "read_volume_any", None)
            if not callable(loader_fn):
                loader_fn = getattr(self._loader, "load_any_volume", None)
            if not callable(loader_fn):
                raise AttributeError(
                    "Volume loader must provide read_volume_any(path) or load_any_volume(path)."
                )
            vol_data = loader_fn(path)
            if vol_data:
                self._apply_volume_data(vol_data)
                return True

            # Check for points/mesh
            # TODO: Implement point cloud and mesh loading

            return False
        except Exception as exc:
            logger.exception("Error loading source: %s", exc)
            return False
        finally:
            self._load_in_progress = False
            self._set_cursor_safe(QtCore.Qt.ArrowCursor)
            if getattr(self, "info_label", None) is not None:
                self.info_label.setText(f"Loaded: {path.name}")

    def _set_cursor_safe(self, cursor_shape: QtCore.Qt.CursorShape) -> None:
        """Best effort cursor update that tolerates test doubles created via __new__."""
        app = QtWidgets.QApplication.instance()
        if app is not None and cursor_shape == QtCore.Qt.ArrowCursor:
            try:
                while app.overrideCursor() is not None:
                    app.restoreOverrideCursor()
            except Exception:
                pass
        try:
            self.setCursor(cursor_shape)
        except Exception:
            pass
        try:
            if getattr(self, "render_widget", None) is not None:
                self.render_widget.setCursor(cursor_shape)
        except Exception:
            pass

    def _apply_volume_data(self, data: VolumeData) -> None:
        self._volume = data
        if data.slice_mode:
            self._init_slice_mode()
        else:
            self._init_3d_mode()

    def _init_3d_mode(self) -> None:
        self._slice_mode = False
        self.controls_stack.setCurrentWidget(self.volume_controls)

        volume = self._volume
        if volume is None:
            return

        # Prepare VTK image if not already there
        if volume.vtk_image is None and volume.array is not None:
            from pyvista import wrap

            volume.vtk_image = wrap(volume.array)
            if volume.spacing:
                spacing = tuple(float(s) for s in volume.spacing)
                try:
                    volume.vtk_image.spacing = spacing
                except Exception:
                    try:
                        volume.vtk_image.SetSpacing(spacing)
                    except Exception:
                        pass

        if volume.vtk_image:
            self._setup_3d_volume_actor(volume)
            logger.info(
                "3D viewer load: 3D actor setup complete shape=%s spacing=%s vmin=%s vmax=%s",
                getattr(volume, "shape", None) or getattr(volume.array, "shape", None),
                volume.spacing,
                volume.vmin,
                volume.vmax,
            )

        self._request_scene_render(reset_camera=True)

    def _setup_3d_volume_actor(self, volume: VolumeData) -> None:
        if not self._scene_initialized:
            return

        # Clean up existing actors
        self.renderer.RemoveAllViewProps()

        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(volume.vtk_image)

        prop = vtkVolumeProperty()
        prop.ShadeOn()
        prop.SetInterpolationTypeToLinear()

        # Configure transfer functions
        self._update_volume_property(prop, volume)

        actor = vtkVolume()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)

        self._scene_backend.add_volume(self.renderer, self.render_widget, actor)
        self._volume_actor = actor
        logger.info("3D viewer load: vtk volume actor added to renderer.")
        try:
            self._scene_backend.reset_camera(self.renderer, self.render_widget)
        except Exception:
            pass
        try:
            self._present_scene()
        except Exception:
            pass

    def _update_volume_property(
        self, prop: vtkVolumeProperty, volume: VolumeData
    ) -> None:
        # Gray scale transfer functions
        opacity = vtkPiecewiseFunction()
        color = vtkColorTransferFunction()

        vmin, vmax = volume.vmin, volume.vmax
        if volume.is_label_map:
            # Step function for labels
            opacity.AddPoint(vmin, 0.0)
            opacity.AddPoint(vmin + 0.1, 1.0)
            opacity.AddPoint(vmax, 1.0)
            # TODO: Color mapping for labels
        else:
            # Linear ramp for grayscale
            opacity.AddPoint(vmin, 0.0)
            opacity.AddPoint(vmax, 1.0)
            color.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            color.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

        prop.SetScalarOpacity(opacity)
        prop.SetColor(color)

    def _init_slice_mode(self) -> None:
        self._slice_mode = True
        self.controls_stack.setCurrentWidget(self.slice_viewer_controls)

        volume = self._volume
        if volume is None or volume.slice_loader is None:
            return

        # Setup 2D view orientation
        self.renderer.RemoveAllViewProps()

        # Configure slice navigator
        total = volume.slice_loader.total_slices()
        self.slice_viewer_controls.slice_slider.setRange(0, total - 1)
        self.slice_viewer_controls.slice_slider.setValue(volume.initial_slice_index)

        # Configure intensity spins
        self.slice_viewer_controls.min_spin.setValue(volume.vmin)
        self.slice_viewer_controls.max_spin.setValue(volume.vmax)

        self._load_slice_image()
        self._request_scene_render(reset_camera=True)

    def _load_slice_image(self) -> None:
        volume = self._volume
        if volume is None or volume.slice_loader is None:
            return

        try:
            slice_data = volume.slice_loader.read_slice(
                volume.slice_axis, volume.current_slice_index
            )

            # Apply windowing and gamma
            vmin, vmax = volume.vmin, volume.vmax
            if not volume.window_override:
                # Use current slider/spin values if override is off
                vmin = self.slice_viewer_controls.min_spin.value()
                vmax = self.slice_viewer_controls.max_spin.value()

            # Simple conversion to uint8 for display if needed,
            # but PyVista can handle raw if we use a mapper.
            # In 2D slice mode, we often use ImageActor or a Plane with texture.

            self._update_slice_display(slice_data, vmin, vmax, volume.gamma)

            self.slice_viewer_controls.status_label.setText(
                f"Slice: {volume.current_slice_index + 1} / {volume.slice_loader.total_slices()}"
            )
        except Exception as exc:
            logger.exception("Failed to load slice: %s", exc)

    def _update_slice_display(
        self, data: np.ndarray, vmin: float, vmax: float, gamma: float
    ) -> None:
        if not self._scene_initialized:
            return

        # For simplicity, we create a PyVista Plane and update its scalars
        # In a real 10x version, we'd use a persistent Actor and just update the mapper input.

        grid = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),  # Axial view
            i_size=data.shape[1],
            j_size=data.shape[0],
            i_resolution=1,
            j_resolution=1,
        )

        # Normalize/Clip for visual consistency if gamma > 1
        # data_norm = ((data - vmin) / (vmax - vmin)).clip(0, 1)
        # if gamma != 1.0: data_norm = data_norm ** (1.0 / gamma)

        self.renderer.RemoveAllViewProps()
        actor = self._scene_add_mesh(
            grid,
            scalars=data.flatten(order="F"),
            clim=[vmin, vmax],
            cmap="gray",
            show_scalar_bar=False,
        )
        self._slice_actor = actor
        try:
            self.renderer.ResetCamera()
        except Exception:
            pass
        try:
            self._present_scene()
        except Exception:
            pass

    def _on_slice_index_changed(self, val: int) -> None:
        volume = self._volume
        if volume:
            volume.current_slice_index = val
            self._load_slice_image()
            self._request_scene_render()

    def _on_slice_axis_changed(self, val: int) -> None:
        volume = self._volume
        if volume:
            volume.slice_axis = val
            # Reset range if axis changed (may have different depth)
            # TODO: Implement multi-axis slice loading
            self._load_slice_image()
            self._request_scene_render(reset_camera=True)

    def _on_slice_window_changed(self) -> None:
        self._load_slice_image()
        self._request_scene_render()

    def _on_slice_gamma_changed(self, val: int) -> None:
        volume = self._volume
        if volume:
            volume.gamma = val / 100.0
            self.slice_viewer_controls.gamma_label.setText(f"Gamma: {volume.gamma:.2f}")
            self._load_slice_image()
            self._request_scene_render()

    def _slice_auto_window(self) -> None:
        volume = self._volume
        if volume and volume.array is not None:
            v_min, v_max = np.min(volume.array), np.max(volume.array)
            self.slice_viewer_controls.min_spin.setValue(float(v_min))
            self.slice_viewer_controls.max_spin.setValue(float(v_max))
            # Triggering the spin change will reload the slice.

    # UI slots
    def _on_load_volume_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Volume Source", "", VOLUME_SOURCE_FILTERS
        )
        if path:
            self._load_source_path(Path(path))

    def _on_reload_volume_clicked(self) -> None:
        volume = self._volume
        if volume and volume.backing_path:
            self._load_source_path(volume.backing_path)

    def _on_load_points_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Point Cloud", "", "Points (*.ply *.csv *.xyz);;All files (*.*)"
        )
        if path:
            self._load_point_cloud(Path(path))

    def _load_point_cloud(self, path: Path) -> None:
        if not self._ensure_scene_initialized():
            return
        try:
            poly = pv.read(str(path))
            # If it's a CSV, it might need manual parsing, but pv.read(csv) often works if it has XYZ.
            self._scene_add_mesh(
                poly,
                color="cyan",
                point_size=5,
                render_points_as_spheres=True,
            )
            self._request_scene_render(reset_camera=True)
        except Exception as exc:
            logger.error("Failed to load point cloud: %s", exc)

    def _on_load_mesh_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Mesh", "", "Mesh (*.stl *.obj *.ply);;All files (*.*)"
        )
        if path:
            self._load_mesh(Path(path))

    def _load_mesh(self, path: Path) -> None:
        if not self._ensure_scene_initialized():
            return
        try:
            mesh = pv.read(str(path))
            self._scene_add_mesh(mesh, color="tan", show_edges=True)
            self._request_scene_render(reset_camera=True)
        except Exception as exc:
            logger.error("Failed to load mesh: %s", exc)

    def _on_save_snapshot_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Snapshot", "snapshot.png", "Images (*.png *.jpg *.tif)"
        )
        if path and self._scene_initialized:
            try:
                # Capture current view
                rw = self._scene_backend.get_render_window(self.render_widget)
                if rw:
                    # PyVista might have a helper if we use its Plotter,
                    # but here we use the raw backend.
                    pass
            except Exception as exc:
                logger.error("Failed to save snapshot: %s", exc)

    def _on_opacity_changed(self, val: int) -> None:
        if hasattr(self, "_volume_actor") and self._volume_actor:
            _ = val  # Placeholder until transfer-function-driven opacity is wired.
            # This is global opacity. Real volume rendering uses transfer functions.
            # We can scale the transfer function points.
            self._request_scene_render()

    def _on_shading_toggled(self, state: int) -> None:
        if hasattr(self, "_volume_actor") and self._volume_actor:
            self._volume_actor.GetProperty().SetShade(state == QtCore.Qt.Checked)
            self._request_scene_render()

    def _on_visibility_toggled(self, state: int) -> None:
        if hasattr(self, "_volume_actor") and self._volume_actor:
            self._volume_actor.SetVisibility(state == QtCore.Qt.Checked)
            self._request_scene_render()

    def _on_intensity_changed(self, val: int) -> None:
        # Update lights intensity
        if self._scene_initialized:
            # TODO: Update renderer lights
            self._request_scene_render()

    def _on_plane_visibility_changed(self, axis: int, state: int) -> None:
        volume = self._volume
        if volume is None:
            return

        vtk_image = volume.vtk_image
        if vtk_image is None:
            return

        if not hasattr(self, "_plane_widgets"):
            self._plane_widgets: dict[int, Any] = {}

        visible = state == QtCore.Qt.Checked

        if axis not in self._plane_widgets:
            if not _HAS_PLANE_WIDGET:
                logger.warning(
                    "Plane widgets not supported by current VTK/PyVista version."
                )
                return

            pw = vtkImagePlaneWidget()
            pw.SetInteractor(self.interactor)
            pw.SetInputData(vtk_image)
            pw.SetPlaneOrientation(axis)
            # Center the plane
            bounds = vtk_image.GetBounds()
            mid = (bounds[axis * 2] + bounds[axis * 2 + 1]) / 2.0
            pw.SetSlicePosition(mid)

            # Formatting
            color = PLANE_COLORS.get(axis, (1, 1, 1))
            pw.GetPlaneProperty().SetColor(*color)
            pw.DisplayTextOn()

            self._plane_widgets[axis] = pw

        pw = self._plane_widgets[axis]
        if visible:
            pw.On()
        else:
            pw.Off()

        self._request_scene_render()

    def _on_plane_position_changed(self, axis: int, val: int) -> None:
        if hasattr(self, "_plane_widgets") and axis in self._plane_widgets:
            pw = self._plane_widgets[axis]
            # val is index 0..N-1? No, we need to map it to physical space if possible,
            # or use SetSliceIndex if available.
            if hasattr(pw, "SetSliceIndex"):
                pw.SetSliceIndex(val)
            self._request_scene_render()

    def _on_reset_view_clicked(self) -> None:
        if self._scene_initialized:
            self.renderer.ResetCamera()
            self._request_scene_render()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Clean up resources on close."""
        volume = self._volume
        if volume and volume.slice_loader:
            try:
                volume.slice_loader.close()
            except Exception:
                pass

        # Disable plane widgets
        if hasattr(self, "_plane_widgets"):
            for pw in self._plane_widgets.values():
                pw.Off()
                pw.SetInteractor(None)
            self._plane_widgets.clear()

        super().closeEvent(event)

    def _on_show_help_clicked(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Help",
            "Left Click + Drag: Rotate\n"
            "Right Click + Drag: Zoom\n"
            "Middle Click + Drag: Pan\n"
            "Shift + Left Click: Roll\n"
            "R: Reset Camera",
        )
