from __future__ import annotations

import base64
import json
import math
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.threejs_support import supports_threejs_canvas
from annolid.gui.widgets.threejs_viewer import ThreeJsViewerWidget
from annolid.simulation import export_simulation_view_payload
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class ThreeJsManager(QtCore.QObject):
    """Manage an embedded Three.js viewer in the shared stacked viewer area."""

    _ZARR_TARGET_SAMPLED_VOXELS = 1_000_000
    _ZARR_MAX_POINTS = 120_000
    _ZARR_MULTISCALE_MAX_VOXELS = 16_000_000
    _TIFF_TARGET_SAMPLED_VOXELS = 1_000_000

    def __init__(
        self, window: "AnnolidWindow", viewer_stack: QtWidgets.QStackedWidget
    ) -> None:
        super().__init__(window)
        self.window = window
        self.viewer_stack = viewer_stack
        self.threejs_viewer: Optional[ThreeJsViewerWidget] = None
        self._zarr_payload_cache: dict[str, tuple[int, Path]] = {}
        self._tiff_payload_cache: dict[str, tuple[int, Path]] = {}

    def ensure_threejs_viewer(self) -> ThreeJsViewerWidget:
        if self.threejs_viewer is None:
            viewer = ThreeJsViewerWidget(self.window)
            viewer.status_changed.connect(
                lambda msg: self.window.statusBar().showMessage(msg, 3000)
            )
            viewer.flybody_command_requested.connect(
                self._handle_flybody_viewer_command
            )
            viewer.region_picked.connect(self._handle_region_picked)
            self.viewer_stack.addWidget(viewer)
            self.threejs_viewer = viewer
        return self.threejs_viewer

    def is_supported(self, path: str | Path) -> bool:
        return supports_threejs_canvas(path)

    def show_model_in_viewer(
        self,
        model_path: str | Path,
        *,
        pick_mode: str = "",
        object_region_map: dict[str, str] | None = None,
    ) -> bool:
        path = Path(model_path)
        if not self.is_supported(path):
            return False
        viewer = self.ensure_threejs_viewer()
        is_zarr_source = path.suffix.lower() == ".zarr"
        is_tiff_source = path.suffix.lower() in {".tif", ".tiff"}
        try:
            if is_zarr_source:
                payload_path = self._resolve_zarr_simulation_payload(path)
                viewer.load_simulation_payload(payload_path, title=path.stem)
            elif is_tiff_source:
                payload_path = self._resolve_tiff_simulation_payload(path)
                viewer.load_simulation_payload(payload_path, title=path.stem)
            else:
                viewer.load_model(
                    path,
                    pick_mode=pick_mode,
                    object_region_map=object_region_map,
                )
        except Exception as exc:
            logger.warning("Failed to load model in Three.js viewer: %s", exc)
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Three.js Viewer"),
                self.window.tr("Unable to load model in Three.js canvas:\n%1").replace(
                    "%1", str(exc)
                ),
            )
            return False
        # Hide unrelated docks when switching to the 3D canvas.
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("threejs")
        try:
            close_action = getattr(getattr(self.window, "actions", None), "close", None)
            if close_action is not None:
                close_action.setEnabled(True)
        except Exception:
            pass
        if is_zarr_source:
            self.window.statusBar().showMessage(
                self.window.tr("Loaded Zarr volume %1").replace("%1", path.name), 3000
            )
        elif is_tiff_source:
            self.window.statusBar().showMessage(
                self.window.tr("Loaded TIFF volume %1").replace("%1", path.name), 3000
            )
        else:
            self.window.statusBar().showMessage(
                self.window.tr("Loaded 3D model %1").replace("%1", path.name), 3000
            )
        return True

    def _resolve_zarr_simulation_payload(self, path: Path) -> Path:
        cache_key = str(path.resolve())
        mtime_ns = int(path.stat().st_mtime_ns)
        cached = self._zarr_payload_cache.get(cache_key)
        if cached is not None and int(cached[0]) == mtime_ns and cached[1].exists():
            return cached[1]
        payload = self._build_zarr_simulation_payload(path)
        out_dir = Path(tempfile.gettempdir()) / "annolid_threejs_zarr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_threejs_payload.json"
        out_path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        self._zarr_payload_cache[cache_key] = (mtime_ns, out_path)
        return out_path

    def _resolve_tiff_simulation_payload(self, path: Path) -> Path:
        cache_key = str(path.resolve())
        mtime_ns = int(path.stat().st_mtime_ns)
        cached = self._tiff_payload_cache.get(cache_key)
        if cached is not None and int(cached[0]) == mtime_ns and cached[1].exists():
            return cached[1]
        payload = self._build_tiff_simulation_payload(path)
        out_dir = Path(tempfile.gettempdir()) / "annolid_threejs_tiff"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_threejs_payload.json"
        out_path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        self._tiff_payload_cache[cache_key] = (mtime_ns, out_path)
        return out_path

    def _build_tiff_simulation_payload(self, path: Path) -> dict[str, Any]:
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("TIFF volume support requires `numpy` package.") from exc

        volume, spacing_zyx = self._read_tiff_volume(path)
        if volume.ndim != 3:
            raise RuntimeError(
                f"Unsupported TIFF dimensionality for 3D viewer: ndim={int(volume.ndim)}"
            )
        if volume.size <= 0:
            raise RuntimeError(f"TIFF volume is empty: {path}")

        z_len_full, y_len_full, x_len_full = [int(v) for v in volume.shape]
        spatial_voxels = float(
            np.prod(np.asarray([z_len_full, y_len_full, x_len_full], dtype=np.float64))
        )
        stride = max(
            1,
            int(
                math.ceil(
                    (spatial_voxels / float(self._TIFF_TARGET_SAMPLED_VOXELS))
                    ** (1.0 / 3.0)
                )
            ),
        )
        sampled = volume[::stride, ::stride, ::stride]
        if sampled.size <= 0:
            sampled = volume
            stride = 1

        finite = sampled[np.isfinite(sampled)]
        if finite.size <= 0:
            raise RuntimeError(f"TIFF volume has no finite values: {path}")
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        scale = hi - lo
        if scale <= 1e-8:
            normalized = np.ones_like(sampled, dtype=np.float32)
        else:
            normalized = (sampled - lo) / scale

        intensity_inverted = self._should_invert_zarr_signal(normalized)
        if intensity_inverted:
            normalized = 1.0 - normalized
        dense_volume = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)

        threshold_quantile = 0.70 if intensity_inverted else 0.92
        threshold_floor = 0.10 if intensity_inverted else 0.18
        threshold = max(
            threshold_floor,
            float(np.quantile(normalized[np.isfinite(normalized)], threshold_quantile)),
        )
        mask = np.isfinite(normalized) & (normalized >= threshold)
        if int(mask.sum()) < 2000:
            threshold = max(
                0.05 if intensity_inverted else 0.45,
                float(
                    np.quantile(
                        normalized[np.isfinite(normalized)],
                        0.52 if intensity_inverted else 0.80,
                    )
                ),
            )
            mask = np.isfinite(normalized) & (normalized >= threshold)
        if int(mask.sum()) <= 0:
            mask = np.isfinite(normalized)

        coords = np.argwhere(mask)
        values = normalized[mask]
        max_points = int(self._ZARR_MAX_POINTS)
        if coords.shape[0] > max_points:
            keep = np.argpartition(values, -max_points)[-max_points:]
            coords = coords[keep]
            values = values[keep]

        histogram = self._build_zarr_histogram(values)
        render_defaults = self._build_zarr_render_defaults(
            values=values,
            stride=int(stride),
            source_path=path,
            intensity_inverted=bool(intensity_inverted),
        )
        if str(render_defaults.get("render_style", "")).lower() in {"points", "hybrid"}:
            render_defaults["render_style"] = "raymarch"
        z_len, y_len, x_len = [int(v) for v in dense_volume.shape]
        z_scale = float(max(1, int(stride)) * float(spacing_zyx[0]))
        y_scale = float(max(1, int(stride)) * float(spacing_zyx[1]))
        x_scale = float(max(1, int(stride)) * float(spacing_zyx[2]))
        render_profile = self._classify_zarr_render_profile(
            source_path=path,
            intensity_inverted=bool(intensity_inverted),
        )
        render_defaults.setdefault(
            "section_emphasis", "auto" if render_profile != "cinematic" else "neutral"
        )
        interleaved_detected = "interleaved" in str(path.stem or "").lower()

        points: list[dict[str, Any]] = []
        for idx, (z, y, x) in enumerate(coords):
            points.append(
                {
                    "label": f"v{idx}",
                    "x": float(min(int(x), x_len - 1) * x_scale),
                    "y": float(-min(int(y), y_len - 1) * y_scale),
                    "z": float(-min(int(z), z_len - 1) * z_scale),
                    "confidence": float(values[idx]),
                }
            )

        payload: dict[str, Any] = {
            "kind": "annolid-simulation-v1",
            "adapter": "tiff-volume",
            "metadata": {
                "source_path": str(path),
                "shape": [int(v) for v in volume.shape],
                "source_dataset_path": "",
                "axes": ["z", "y", "x"],
                "downsample_stride": int(stride),
                "threshold": float(threshold),
                "channel_index": 0,
                "render_mode": "gaussian_splatting",
                "render_profile": render_profile,
                "interleaved_detected": bool(interleaved_detected),
                "section_axis": "z",
                "section_step_world": float(z_scale),
                "voxel_spacing_zyx": [float(v) for v in spacing_zyx],
                "voxel_spacing_xyz": [
                    float(spacing_zyx[2]),
                    float(spacing_zyx[1]),
                    float(spacing_zyx[0]),
                ],
                "signal_polarity": "dark_on_light"
                if intensity_inverted
                else "light_on_dark",
                "intensity_inverted": bool(intensity_inverted),
                "splat_size": float(render_defaults["size"]),
                "splat_opacity": float(render_defaults["opacity"]),
                "point_count": int(len(points)),
                "confidence_range": [
                    float(np.min(values)) if values.size > 0 else 0.0,
                    float(np.max(values)) if values.size > 0 else 1.0,
                ],
                "volume_grid_shape": [int(v) for v in dense_volume.shape],
                "volume_grid_base64": self._encode_zarr_volume_grid(dense_volume),
                "volume_histogram": histogram,
                "volume_bounds": {
                    "x": [0.0, float(max(0, x_len - 1) * x_scale)],
                    "y": [float(-max(0, y_len - 1) * y_scale), 0.0],
                    "z": [float(-max(0, z_len - 1) * z_scale), 0.0],
                },
                "volume_render_defaults": render_defaults,
            },
            "edges": [],
            "display": {
                "show_points": True,
                "show_labels": False,
                "show_edges": False,
                "show_trails": False,
            },
            "playback": {
                "autoplay": False,
                "loop": False,
                "interval_ms": 200,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "points": points,
                }
            ],
        }
        return payload

    @staticmethod
    def _read_tiff_volume(path: Path) -> tuple[Any, list[float]]:
        try:
            import numpy as np
            import tifffile
        except Exception as exc:
            raise RuntimeError(
                "TIFF volume support requires `tifffile` and `numpy` packages."
            ) from exc

        with tifffile.TiffFile(str(path)) as tif:
            volume = np.asarray(tif.asarray(), dtype=np.float32)
            spacing_zyx = ThreeJsManager._resolve_tiff_spacing_zyx(tif=tif)

        volume = np.squeeze(volume)
        if volume.ndim == 2:
            volume = volume[np.newaxis, :, :]
        elif volume.ndim >= 4:
            # Collapse likely color channels, then select the first sample on extra axes.
            if int(volume.shape[-1]) <= 4:
                volume = np.mean(volume, axis=-1)
            while volume.ndim > 3:
                volume = volume[0]
        if volume.ndim != 3:
            raise RuntimeError(
                f"Unsupported TIFF shape for volume rendering: {tuple(int(v) for v in volume.shape)}"
            )
        return np.asarray(volume, dtype=np.float32), spacing_zyx

    @staticmethod
    def _resolve_tiff_spacing_zyx(*, tif: Any) -> list[float]:
        spacing_zyx = [1.0, 1.0, 1.0]
        try:
            page0 = tif.pages[0]
        except Exception:
            return spacing_zyx

        def _resolution_to_spacing(value: Any) -> float:
            try:
                num, den = value
                num_f = float(num)
                den_f = float(den)
                if num_f > 0.0:
                    return den_f / num_f
            except Exception:
                return 1.0
            return 1.0

        try:
            xres_tag = page0.tags.get("XResolution")
            if xres_tag is not None:
                spacing_zyx[2] = float(
                    _resolution_to_spacing(getattr(xres_tag, "value", None))
                )
        except Exception:
            pass
        try:
            yres_tag = page0.tags.get("YResolution")
            if yres_tag is not None:
                spacing_zyx[1] = float(
                    _resolution_to_spacing(getattr(yres_tag, "value", None))
                )
        except Exception:
            pass

        try:
            imagej = getattr(tif, "imagej_metadata", None)
            z_spacing = imagej.get("spacing") if isinstance(imagej, dict) else None
            if z_spacing is not None:
                spacing_zyx[0] = float(z_spacing)
        except Exception:
            pass
        return [max(1e-6, float(v)) for v in spacing_zyx]

    def _build_zarr_simulation_payload(self, path: Path) -> dict[str, Any]:
        try:
            import numpy as np
            import zarr
        except Exception as exc:
            raise RuntimeError(
                "Zarr support requires `zarr` and `numpy` packages."
            ) from exc

        source = zarr.open(str(path), mode="r")
        array, source_dataset_path, axis_names = self._resolve_zarr_array_for_viewer(
            source
        )
        shape = tuple(int(v) for v in getattr(array, "shape", ()) or ())
        if not shape:
            raise RuntimeError(f"Zarr store has invalid shape metadata: {path}")
        if any(int(v) <= 0 for v in shape):
            raise RuntimeError(f"Zarr store has non-positive shape dimensions: {path}")

        axis_map = self._resolve_zarr_axis_map(shape=shape, axis_names=axis_names)
        spatial_axes = axis_map["spatial_axes"]
        if len(spatial_axes) < 2:
            raise RuntimeError(
                f"Unsupported Zarr axis layout for 3D viewer: shape={shape}, axes={axis_names}"
            )

        spatial_spacing = self._resolve_zarr_spatial_spacing(
            source=source,
            source_dataset_path=source_dataset_path,
            axis_names=axis_names,
            shape=shape,
            spatial_axes=spatial_axes,
        )
        spatial_shape = [int(shape[idx]) for idx in spatial_axes]
        spatial_voxels = float(np.prod(np.asarray(spatial_shape, dtype=np.float64)))
        stride = max(
            1,
            int(
                math.ceil(
                    (spatial_voxels / float(self._ZARR_TARGET_SAMPLED_VOXELS))
                    ** (1.0 / max(1, len(spatial_axes)))
                )
            ),
        )
        channel_axis = axis_map.get("channel_axis")
        channel_index = self._choose_zarr_channel_index(
            array=array,
            shape=shape,
            channel_axis=channel_axis,
            spatial_axes=spatial_axes,
            stride=stride,
        )
        sampled, sampled_shape = self._sample_zarr_spatial_data(
            array=array,
            shape=shape,
            spatial_axes=spatial_axes,
            stride=stride,
            fixed_channel_axis=channel_axis,
            fixed_channel_index=channel_index,
        )
        if sampled.ndim == 2:
            sampled = sampled[np.newaxis, :, :]
        if sampled.ndim != 3:
            raise RuntimeError(
                f"Unsupported sampled Zarr dimensionality for 3D viewer: ndim={int(sampled.ndim)}"
            )
        if sampled.size <= 0:
            raise RuntimeError(f"Zarr volume is empty: {path}")

        finite = sampled[np.isfinite(sampled)]
        if finite.size <= 0:
            raise RuntimeError(f"Zarr volume has no finite values: {path}")
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        scale = hi - lo
        if scale <= 1e-8:
            normalized = np.ones_like(sampled, dtype=np.float32)
        else:
            normalized = (sampled - lo) / scale

        intensity_inverted = self._should_invert_zarr_signal(normalized)
        if intensity_inverted:
            normalized = 1.0 - normalized
        dense_volume = np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)

        threshold_quantile = 0.70 if intensity_inverted else 0.92
        threshold_floor = 0.10 if intensity_inverted else 0.18
        threshold = max(
            threshold_floor,
            float(np.quantile(normalized[np.isfinite(normalized)], threshold_quantile)),
        )
        mask = np.isfinite(normalized) & (normalized >= threshold)
        if int(mask.sum()) < 2000:
            threshold = max(
                0.05 if intensity_inverted else 0.45,
                float(
                    np.quantile(
                        normalized[np.isfinite(normalized)],
                        0.52 if intensity_inverted else 0.80,
                    )
                ),
            )
            mask = np.isfinite(normalized) & (normalized >= threshold)
        if int(mask.sum()) <= 0:
            mask = np.isfinite(normalized)

        coords = np.argwhere(mask)
        values = normalized[mask]
        max_points = int(self._ZARR_MAX_POINTS)
        if coords.shape[0] > max_points:
            keep = np.argpartition(values, -max_points)[-max_points:]
            coords = coords[keep]
            values = values[keep]

        histogram = self._build_zarr_histogram(values)
        render_defaults = self._build_zarr_render_defaults(
            values=values,
            stride=int(stride),
            source_path=path,
            intensity_inverted=bool(intensity_inverted),
        )
        points: list[dict[str, Any]] = []
        sampled_spatial_shape = [
            int(sampled_shape[idx] if idx < len(sampled_shape) else 1)
            for idx in spatial_axes
        ]
        z_len = int(sampled_spatial_shape[-3]) if len(sampled_spatial_shape) >= 3 else 1
        y_len = int(sampled_spatial_shape[-2]) if len(sampled_spatial_shape) >= 2 else 1
        x_len = int(sampled_spatial_shape[-1]) if len(sampled_spatial_shape) >= 1 else 1
        z_scale = float(max(1, int(stride)) * float(spatial_spacing[0]))
        y_scale = float(max(1, int(stride)) * float(spatial_spacing[1]))
        x_scale = float(max(1, int(stride)) * float(spatial_spacing[2]))
        render_profile = self._classify_zarr_render_profile(
            source_path=path,
            intensity_inverted=bool(intensity_inverted),
        )
        interleaved_detected = "interleaved" in str(path.stem or "").lower()
        for idx, (z, y, x) in enumerate(coords):
            points.append(
                {
                    "label": f"v{idx}",
                    "x": float(min(int(x), x_len - 1) * x_scale),
                    "y": float(-min(int(y), y_len - 1) * y_scale),
                    "z": float(-min(int(z), z_len - 1) * z_scale),
                    "confidence": float(values[idx]),
                }
            )

        payload: dict[str, Any] = {
            "kind": "annolid-simulation-v1",
            "adapter": "zarr-volume",
            "metadata": {
                "source_path": str(path),
                "shape": [int(v) for v in shape],
                "source_dataset_path": str(source_dataset_path or ""),
                "axes": [str(v) for v in axis_names],
                "downsample_stride": int(stride),
                "threshold": float(threshold),
                "channel_index": int(channel_index) if channel_index is not None else 0,
                "render_mode": "gaussian_splatting",
                "render_profile": render_profile,
                "interleaved_detected": bool(interleaved_detected),
                "section_axis": "z",
                "section_step_world": float(z_scale),
                "voxel_spacing_zyx": [float(v) for v in spatial_spacing],
                "voxel_spacing_xyz": [
                    float(spatial_spacing[2]),
                    float(spatial_spacing[1]),
                    float(spatial_spacing[0]),
                ],
                "signal_polarity": "dark_on_light"
                if intensity_inverted
                else "light_on_dark",
                "intensity_inverted": bool(intensity_inverted),
                "splat_size": float(render_defaults["size"]),
                "splat_opacity": float(render_defaults["opacity"]),
                "point_count": int(len(points)),
                "confidence_range": [
                    float(np.min(values)) if values.size > 0 else 0.0,
                    float(np.max(values)) if values.size > 0 else 1.0,
                ],
                "volume_grid_shape": [int(v) for v in dense_volume.shape],
                "volume_grid_base64": self._encode_zarr_volume_grid(dense_volume),
                "volume_histogram": histogram,
                "volume_bounds": {
                    "x": [0.0, float(max(0, x_len - 1) * x_scale)],
                    "y": [float(-max(0, y_len - 1) * y_scale), 0.0],
                    "z": [float(-max(0, z_len - 1) * z_scale), 0.0],
                },
                "volume_render_defaults": render_defaults,
            },
            "edges": [],
            "display": {
                "show_points": True,
                "show_labels": False,
                "show_edges": False,
                "show_trails": False,
            },
            "playback": {
                "autoplay": False,
                "loop": False,
                "interval_ms": 200,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "points": points,
                }
            ],
        }
        return payload

    @staticmethod
    def _build_zarr_render_defaults(
        *, values: Any, stride: int, source_path: Path, intensity_inverted: bool
    ) -> dict[str, Any]:
        try:
            import numpy as np
        except Exception:
            np = None

        mean_value = 0.78
        peak_value = 1.0
        if np is not None and getattr(values, "size", 0):
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                mean_value = float(np.mean(finite))
                peak_value = float(np.max(finite))

        profile = ThreeJsManager._classify_zarr_render_profile(
            source_path=source_path,
            intensity_inverted=bool(intensity_inverted),
        )
        if profile == "nissl_sections":
            return ThreeJsManager._with_volume_renderer_defaults(
                {
                    "preset": "nissl_sections",
                    "intensity": 1.26,
                    "contrast": 1.62,
                    "gamma": 0.82,
                    "opacity": 0.66,
                    "size": 0.048,
                    "threshold": 0.03,
                    "density": 1.0,
                    "saturation": 0.96,
                    "tf_low": 0.02,
                    "tf_mid": 0.34,
                    "tf_high": 0.84,
                    "clip_axis": "none",
                    "clip_center": 0.5,
                    "clip_thickness": 1.0,
                    "clip_invert": False,
                    "palette": "nissl",
                    "blend_mode": "normal",
                    "point_texture": "section",
                    "background_theme": "light",
                },
                profile=profile,
            )
        if profile == "myelin_sections":
            return ThreeJsManager._with_volume_renderer_defaults(
                {
                    "preset": "myelin_sections",
                    "intensity": 1.14,
                    "contrast": 1.78,
                    "gamma": 0.88,
                    "opacity": 0.72,
                    "size": 0.05,
                    "threshold": 0.04,
                    "density": 1.0,
                    "saturation": 0.22,
                    "tf_low": 0.03,
                    "tf_mid": 0.3,
                    "tf_high": 0.82,
                    "clip_axis": "none",
                    "clip_center": 0.5,
                    "clip_thickness": 1.0,
                    "clip_invert": False,
                    "palette": "myelin",
                    "blend_mode": "normal",
                    "point_texture": "section",
                    "background_theme": "light",
                },
                profile=profile,
            )
        if profile == "section_stack":
            return ThreeJsManager._with_volume_renderer_defaults(
                {
                    "preset": "section_stack",
                    "intensity": 1.22,
                    "contrast": 1.56,
                    "gamma": 0.84,
                    "opacity": 0.68,
                    "size": 0.05,
                    "threshold": 0.03,
                    "density": 1.0,
                    "saturation": 0.78,
                    "tf_low": 0.02,
                    "tf_mid": 0.32,
                    "tf_high": 0.84,
                    "clip_axis": "none",
                    "clip_center": 0.5,
                    "clip_thickness": 1.0,
                    "clip_invert": False,
                    "palette": "section_ink",
                    "blend_mode": "normal",
                    "point_texture": "section",
                    "background_theme": "light",
                },
                profile=profile,
            )

        size = 0.024 + min(max(int(stride), 1), 6) * 0.004
        density = 0.9 if mean_value >= 0.68 else 0.82
        intensity = 1.1 if peak_value >= 0.95 else 1.2
        contrast = 1.28 if mean_value >= 0.72 else 1.42
        gamma = 0.9 if mean_value >= 0.72 else 0.82
        return ThreeJsManager._with_volume_renderer_defaults(
            {
                "preset": "cinematic",
                "intensity": float(intensity),
                "contrast": float(contrast),
                "gamma": float(gamma),
                "opacity": 0.42,
                "size": float(size),
                "threshold": 0.16,
                "density": float(density),
                "saturation": 1.08,
                "tf_low": 0.06,
                "tf_mid": 0.48,
                "tf_high": 0.96,
                "clip_axis": "none",
                "clip_center": 0.5,
                "clip_thickness": 1.0,
                "clip_invert": False,
                "palette": "ice_fire",
                "blend_mode": "additive",
                "point_texture": "glow",
                "background_theme": "dark",
            },
            profile=profile,
        )

    @staticmethod
    def _with_volume_renderer_defaults(
        defaults: dict[str, Any], *, profile: str
    ) -> dict[str, Any]:
        merged = dict(defaults)
        is_histology = profile in {"nissl_sections", "myelin_sections", "section_stack"}
        merged.setdefault("render_style", "raymarch" if is_histology else "hybrid")
        merged.setdefault(
            "section_emphasis",
            "nissl"
            if profile == "nissl_sections"
            else "myelin"
            if profile == "myelin_sections"
            else "auto",
        )
        merged.setdefault("raymarch_steps", 256 if is_histology else 208)
        merged.setdefault("raymarch_step_scale", 1.0)
        merged.setdefault("raymarch_jitter", 0.45)
        merged.setdefault("gradient_opacity", bool(is_histology))
        merged.setdefault("gradient_opacity_factor", 3.6 if is_histology else 2.6)
        merged.setdefault("use_shading", True)
        merged.setdefault("ambient_strength", 0.34)
        merged.setdefault("diffuse_strength", 0.86)
        merged.setdefault("specular_strength", 0.22)
        merged.setdefault("specular_power", 24.0)
        merged.setdefault("light_direction", [0.38, 0.52, 0.76])
        return merged

    @staticmethod
    def _classify_zarr_render_profile(
        *, source_path: Path, intensity_inverted: bool
    ) -> str:
        stem = str(source_path.stem or "").lower()
        if "myelin" in stem:
            return "myelin_sections"
        if "nissl" in stem:
            return "nissl_sections"
        if intensity_inverted or "interleaved" in stem or "atlas" in stem:
            return "section_stack"
        return "cinematic"

    @staticmethod
    def _should_invert_zarr_signal(normalized: Any) -> bool:
        try:
            import numpy as np
        except Exception:
            return False

        if getattr(normalized, "ndim", 0) != 3:
            return False
        if normalized.size <= 0:
            return False
        z_len, y_len, x_len = [int(v) for v in normalized.shape]
        z_margin = max(1, z_len // 10)
        y_margin = max(1, y_len // 10)
        x_margin = max(1, x_len // 10)
        center = normalized[
            z_margin : max(z_margin + 1, z_len - z_margin),
            y_margin : max(y_margin + 1, y_len - y_margin),
            x_margin : max(x_margin + 1, x_len - x_margin),
        ]
        border_mask = np.ones_like(normalized, dtype=bool)
        border_mask[
            z_margin : max(z_margin + 1, z_len - z_margin),
            y_margin : max(y_margin + 1, y_len - y_margin),
            x_margin : max(x_margin + 1, x_len - x_margin),
        ] = False
        border = normalized[border_mask]
        center_finite = center[np.isfinite(center)]
        border_finite = border[np.isfinite(border)]
        if center_finite.size <= 0 or border_finite.size <= 0:
            return False
        center_med = float(np.median(center_finite))
        border_med = float(np.median(border_finite))
        return border_med > center_med + 0.08

    @staticmethod
    def _build_zarr_histogram(values: Any, bins: int = 32) -> dict[str, Any]:
        try:
            import numpy as np
        except Exception:
            return {"bins": [], "counts": []}

        finite = values[np.isfinite(values)]
        if finite.size <= 0:
            return {"bins": [], "counts": []}
        counts, edges = np.histogram(finite, bins=max(4, int(bins)), range=(0.0, 1.0))
        max_count = int(np.max(counts)) if counts.size > 0 else 0
        normalized = (
            [float(v / max_count) for v in counts.tolist()] if max_count > 0 else []
        )
        return {
            "bins": [float(v) for v in edges[:-1].tolist()],
            "counts": [int(v) for v in counts.tolist()],
            "normalized_counts": normalized,
        }

    @staticmethod
    def _encode_zarr_volume_grid(volume: Any) -> str:
        return base64.b64encode(memoryview(volume).tobytes()).decode("ascii")

    @staticmethod
    def _select_zarr_array(node: Any) -> Any:
        if hasattr(node, "shape") and hasattr(node, "__getitem__"):
            return node
        if hasattr(node, "arrays"):
            candidates: list[Any] = []
            try:
                for _name, arr in node.arrays():
                    if hasattr(arr, "shape"):
                        candidates.append(arr)
            except Exception:
                candidates = []
            if candidates:
                return max(
                    candidates, key=lambda arr: int(getattr(arr, "size", 0) or 0)
                )
        if hasattr(node, "groups"):
            nested: list[Any] = []
            try:
                for _name, grp in node.groups():
                    nested.append(grp)
            except Exception:
                nested = []
            for grp in nested:
                selected = ThreeJsManager._select_zarr_array(grp)
                if selected is not None:
                    return selected
        return None

    def _resolve_zarr_array_for_viewer(self, source: Any) -> tuple[Any, str, list[str]]:
        shape = tuple(int(v) for v in getattr(source, "shape", ()) or ())
        if shape:
            axis_names = self._normalize_zarr_axes(
                None,
                ndim=len(shape),
                array=source,
            )
            return source, "", axis_names

        multiscale_candidates = self._zarr_multiscale_candidates(source)
        if multiscale_candidates:
            array, path, axes = self._choose_zarr_multiscale_candidate(
                multiscale_candidates
            )
            axis_names = self._normalize_zarr_axes(
                axes, ndim=len(array.shape), array=array
            )
            return array, path, axis_names

        selected = self._select_zarr_array(source)
        if selected is None:
            raise RuntimeError("No readable array found in Zarr store.")
        axis_names = self._normalize_zarr_axes(
            None,
            ndim=len(getattr(selected, "shape", ()) or ()),
            array=selected,
        )
        return selected, "", axis_names

    @staticmethod
    def _normalize_zarr_axes(axes_raw: Any, *, ndim: int, array: Any) -> list[str]:
        names: list[str] = []
        candidate = axes_raw
        if candidate is None and hasattr(array, "attrs"):
            with_names = array.attrs.get("_ARRAY_DIMENSIONS")
            if isinstance(with_names, list):
                candidate = with_names
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict):
                    names.append(str(item.get("name", "") or "").strip().lower())
                else:
                    names.append(str(item or "").strip().lower())
        if len(names) != int(ndim):
            names = []
        return names

    def _zarr_multiscale_candidates(self, root: Any) -> list[tuple[Any, str, Any]]:
        attrs = getattr(root, "attrs", None)
        if attrs is None:
            return []
        multiscales = attrs.get("multiscales")
        if not isinstance(multiscales, list) or not multiscales:
            return []
        primary = multiscales[0]
        if not isinstance(primary, dict):
            return []
        datasets = primary.get("datasets")
        axes = primary.get("axes")
        if not isinstance(datasets, list):
            return []
        resolved: list[tuple[Any, str, Any]] = []
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            path = str(dataset.get("path", "") or "").strip()
            node = root
            try:
                if path:
                    node = root[path]
            except Exception:
                continue
            arr = self._select_zarr_array(node)
            if arr is None:
                continue
            resolved.append((arr, path, axes))
        return resolved

    @staticmethod
    def _resolve_zarr_spatial_spacing(
        *,
        source: Any,
        source_dataset_path: str,
        axis_names: list[str],
        shape: tuple[int, ...],
        spatial_axes: list[int],
    ) -> list[float]:
        spacing_by_axis: list[float] = [1.0 for _ in range(len(shape))]
        attrs = getattr(source, "attrs", None)
        multiscales = attrs.get("multiscales") if attrs is not None else None
        if isinstance(multiscales, list) and multiscales:
            primary = multiscales[0]
            if isinstance(primary, dict):
                datasets = primary.get("datasets")
                if isinstance(datasets, list):
                    target = None
                    for dataset in datasets:
                        if not isinstance(dataset, dict):
                            continue
                        if str(dataset.get("path", "") or "").strip() == str(
                            source_dataset_path or ""
                        ):
                            target = dataset
                            break
                    transforms = (
                        target.get("coordinateTransformations")
                        if isinstance(target, dict)
                        else None
                    )
                    if isinstance(transforms, list):
                        for transform in transforms:
                            if not isinstance(transform, dict):
                                continue
                            if str(transform.get("type", "") or "").lower() != "scale":
                                continue
                            scale = transform.get("scale")
                            if isinstance(scale, list) and len(scale) == len(shape):
                                parsed: list[float] = []
                                for value in scale:
                                    try:
                                        parsed.append(float(value))
                                    except Exception:
                                        parsed.append(1.0)
                                spacing_by_axis = parsed
                                break
        if axis_names and len(axis_names) == len(shape):
            resolved: dict[str, float] = {}
            for axis_index, axis_name in enumerate(axis_names):
                if axis_name in {"z", "y", "x"}:
                    resolved[axis_name] = float(spacing_by_axis[axis_index] or 1.0)
            if resolved:
                return [
                    float(resolved.get("z", 1.0)),
                    float(resolved.get("y", 1.0)),
                    float(resolved.get("x", 1.0)),
                ]
        spatial_values = [
            float(spacing_by_axis[idx] or 1.0) for idx in spatial_axes[-3:]
        ]
        while len(spatial_values) < 3:
            spatial_values.insert(0, 1.0)
        return spatial_values[-3:]

    def _choose_zarr_multiscale_candidate(
        self, candidates: list[tuple[Any, str, Any]]
    ) -> tuple[Any, str, Any]:
        if len(candidates) == 1:
            return candidates[0]
        chosen = candidates[-1]
        for candidate in candidates:
            arr = candidate[0]
            size = int(getattr(arr, "size", 0) or 0)
            if size <= int(self._ZARR_MULTISCALE_MAX_VOXELS):
                chosen = candidate
                break
        return chosen

    @staticmethod
    def _resolve_zarr_axis_map(
        *, shape: tuple[int, ...], axis_names: list[str]
    ) -> dict[str, Any]:
        ndim = len(shape)
        if ndim <= 0:
            return {"spatial_axes": [], "channel_axis": None}
        if axis_names and len(axis_names) == ndim:
            z_axis = axis_names.index("z") if "z" in axis_names else None
            y_axis = axis_names.index("y") if "y" in axis_names else None
            x_axis = axis_names.index("x") if "x" in axis_names else None
            channel_axis = axis_names.index("c") if "c" in axis_names else None
            spatial: list[int] = [v for v in [z_axis, y_axis, x_axis] if v is not None]
            if len(spatial) >= 2:
                return {
                    "spatial_axes": spatial,
                    "channel_axis": channel_axis,
                }
        if ndim >= 3:
            return {
                "spatial_axes": [ndim - 3, ndim - 2, ndim - 1],
                "channel_axis": 1 if ndim >= 4 and int(shape[1]) <= 8 else None,
            }
        if ndim == 2:
            return {"spatial_axes": [0, 1], "channel_axis": None}
        return {"spatial_axes": [0], "channel_axis": None}

    def _choose_zarr_channel_index(
        self,
        *,
        array: Any,
        shape: tuple[int, ...],
        channel_axis: Optional[int],
        spatial_axes: list[int],
        stride: int,
    ) -> Optional[int]:
        if channel_axis is None:
            return None
        channel_count = int(shape[channel_axis] or 0)
        if channel_count <= 1:
            return 0
        try:
            import numpy as np
        except Exception:
            return 0
        best_idx = 0
        best_score = -1.0
        test_count = min(4, channel_count)
        for candidate in range(test_count):
            sampled, _shape = self._sample_zarr_spatial_data(
                array=array,
                shape=shape,
                spatial_axes=spatial_axes,
                stride=stride,
                fixed_channel_axis=channel_axis,
                fixed_channel_index=candidate,
            )
            finite = sampled[np.isfinite(sampled)]
            if finite.size <= 0:
                score = -1.0
            else:
                score = float(np.max(finite) - np.min(finite))
            if score > best_score:
                best_score = score
                best_idx = candidate
        return int(best_idx)

    @staticmethod
    def _sample_zarr_spatial_data(
        *,
        array: Any,
        shape: tuple[int, ...],
        spatial_axes: list[int],
        stride: int,
        fixed_channel_axis: Optional[int],
        fixed_channel_index: Optional[int],
    ) -> tuple[Any, tuple[int, ...]]:
        import numpy as np

        ndim = len(shape)
        indexers: list[Any] = []
        for axis in range(ndim):
            if axis in spatial_axes:
                indexers.append(slice(0, int(shape[axis]), max(1, int(stride))))
                continue
            if fixed_channel_axis is not None and axis == int(fixed_channel_axis):
                indexers.append(int(fixed_channel_index or 0))
                continue
            indexers.append(0)

        sampled = np.asarray(array[tuple(indexers)], dtype=np.float32)
        sampled_shape = tuple(int(v) for v in sampled.shape)

        kept_axes: list[int] = []
        for axis, indexer in enumerate(indexers):
            if isinstance(indexer, int):
                continue
            kept_axes.append(axis)
        remapped: list[int] = []
        for axis in spatial_axes:
            if axis in kept_axes:
                remapped.append(kept_axes.index(axis))
        if not remapped:
            return sampled, sampled_shape

        sampled = np.moveaxis(sampled, remapped, range(len(remapped)))
        if len(remapped) == 2:
            sampled = sampled[np.newaxis, :, :]
        return sampled, sampled_shape

    def show_url_in_viewer(self, url: str) -> bool:
        viewer = self.ensure_threejs_viewer()
        try:
            viewer.load_url(url)
        except Exception as exc:
            logger.warning("Failed to load URL in Three.js viewer: %s", exc)
            return False
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("threejs")
        try:
            close_action = getattr(getattr(self.window, "actions", None), "close", None)
            if close_action is not None:
                close_action.setEnabled(True)
        except Exception:
            pass
        return True

    def show_simulation_in_viewer(self, simulation_path: str | Path) -> bool:
        path = Path(simulation_path)
        if not path.exists():
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Simulation Viewer"),
                self.window.tr("Simulation file not found:\n%1").replace(
                    "%1", str(path)
                ),
            )
            return False
        viewer = self.ensure_threejs_viewer()
        started = time.perf_counter()
        payload_path = path
        try:
            payload_path = self._resolve_simulation_payload_path(path)
            viewer.load_simulation_payload(payload_path, title=path.stem)
        except Exception as exc:
            logger.warning("Failed to load simulation in Three.js viewer: %s", exc)
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Simulation Viewer"),
                self.window.tr(
                    "Unable to open FlyBody/simulation output in the 3D viewer:\n%1"
                ).replace("%1", str(exc)),
            )
            return False
        logger.info(
            "Prepared Three.js simulation view for %s using %s in %.1fms",
            path,
            payload_path,
            (time.perf_counter() - started) * 1000.0,
        )
        self.window.set_unrelated_docks_visible(False)
        self.window._set_active_view("threejs")
        self.window.statusBar().showMessage(
            self.window.tr("Loaded simulation view %1").replace("%1", path.name), 3000
        )
        return True

    def update_simulation_in_viewer(
        self, simulation_path: str | Path, *, title: str | None = None
    ) -> bool:
        path = Path(simulation_path)
        if not path.exists():
            return False
        viewer = self.ensure_threejs_viewer()
        started = time.perf_counter()
        payload_path = path
        try:
            payload_path = self._resolve_simulation_payload_path(path)
            viewer.update_simulation_payload(payload_path, title=title or path.stem)
        except Exception as exc:
            logger.warning("Failed to update simulation in Three.js viewer: %s", exc)
            return False
        logger.info(
            "Updated Three.js simulation view for %s using %s in %.1fms",
            path,
            payload_path,
            (time.perf_counter() - started) * 1000.0,
        )
        return True

    def _resolve_simulation_payload_path(self, path: Path) -> Path:
        if self._is_prebuilt_simulation_payload(path):
            return path
        return export_simulation_view_payload(path)

    @staticmethod
    def _is_prebuilt_simulation_payload(path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            isinstance(payload, dict) and payload.get("kind") == "annolid-simulation-v1"
        )

    def viewer_widget(self) -> Optional[ThreeJsViewerWidget]:
        return self.threejs_viewer

    def close_threejs(self) -> None:
        """Close Three.js 3D view and return to canvas."""
        # Switch back to canvas.
        try:
            self.window._set_active_view("canvas")
        except Exception:
            pass

    def _handle_flybody_viewer_command(self, action: str, behavior: str) -> None:
        handler = getattr(self.window, "handle_flybody_viewer_command", None)
        if callable(handler):
            handler(action, behavior)

    def _handle_region_picked(self, region_id: str) -> None:
        handler = getattr(self.window, "_onBrain3DMeshRegionPicked", None)
        if callable(handler):
            handler(str(region_id or ""))
