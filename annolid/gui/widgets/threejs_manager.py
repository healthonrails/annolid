from __future__ import annotations

import base64
import csv
import json
import math
import re
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

        sampled = np.asarray(sampled)
        finite = sampled[np.isfinite(sampled)]
        if finite.size <= 0:
            raise RuntimeError(f"TIFF volume has no finite values: {path}")
        (
            is_label_volume,
            label_indices,
            label_id_lut,
            label_id_total,
            label_id_truncated,
        ) = self._prepare_tiff_label_volume(sampled=sampled, source_path=path)
        threshold = 0.0
        intensity_inverted = False
        label_indices_i32: Any | None = None
        if is_label_volume:
            dense_volume = label_indices
            label_indices_i32 = np.asarray(dense_volume, dtype=np.int32)
            mask = label_indices_i32 > 0
            coords = np.argwhere(mask)
            values = np.ones(coords.shape[0], dtype=np.float32)
            if coords.shape[0] > int(self._ZARR_MAX_POINTS):
                step = max(
                    1, int(math.ceil(coords.shape[0] / float(self._ZARR_MAX_POINTS)))
                )
                coords = coords[::step]
                if coords.shape[0] > int(self._ZARR_MAX_POINTS):
                    coords = coords[: int(self._ZARR_MAX_POINTS)]
                values = np.ones(coords.shape[0], dtype=np.float32)
            histogram = self._build_zarr_histogram(
                dense_volume.astype(np.float32) / 255.0
            )
            render_defaults = self._build_label_volume_render_defaults()
            render_defaults["label_color_seed"] = int(
                self._stable_label_color_seed(path)
            )
            render_profile = "label_ids"
        else:
            sampled_f32 = sampled.astype(np.float32, copy=False)
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            scale = hi - lo
            if scale <= 1e-8:
                normalized = np.ones_like(sampled_f32, dtype=np.float32)
            else:
                normalized = (sampled_f32 - lo) / scale

            intensity_inverted = self._should_invert_zarr_signal(normalized)
            if intensity_inverted:
                normalized = 1.0 - normalized
            dense_volume = np.clip(np.round(normalized * 255.0), 0, 255).astype(
                np.uint8
            )

            threshold_quantile = 0.70 if intensity_inverted else 0.92
            threshold_floor = 0.10 if intensity_inverted else 0.18
            threshold = max(
                threshold_floor,
                float(
                    np.quantile(normalized[np.isfinite(normalized)], threshold_quantile)
                ),
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
            if str(render_defaults.get("render_style", "")).lower() in {
                "points",
                "hybrid",
            }:
                render_defaults["render_style"] = "raymarch"
            render_profile = self._classify_zarr_render_profile(
                source_path=path,
                intensity_inverted=bool(intensity_inverted),
            )
        z_len, y_len, x_len = [int(v) for v in dense_volume.shape]
        z_scale = float(max(1, int(stride)) * float(spacing_zyx[0]))
        y_scale = float(max(1, int(stride)) * float(spacing_zyx[1]))
        x_scale = float(max(1, int(stride)) * float(spacing_zyx[2]))
        render_defaults.setdefault(
            "section_emphasis", "auto" if render_profile != "cinematic" else "neutral"
        )
        if is_label_volume:
            render_defaults["section_emphasis"] = "neutral"
        interleaved_detected = "interleaved" in str(path.stem or "").lower()

        points: list[dict[str, Any]] = []
        for idx, (z, y, x) in enumerate(coords):
            point = {
                "label": f"v{idx}",
                "x": float(min(int(x), x_len - 1) * x_scale),
                "y": float(-min(int(y), y_len - 1) * y_scale),
                "z": float(-min(int(z), z_len - 1) * z_scale),
                "confidence": float(values[idx]),
            }
            if is_label_volume and label_indices_i32 is not None:
                label_index = int(label_indices_i32[int(z), int(y), int(x)])
                point["label_index"] = int(label_index)
                point["label_id"] = (
                    int(label_id_lut[label_index - 1])
                    if label_index > 0 and label_index <= len(label_id_lut)
                    else int(label_index)
                )
            points.append(point)

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
                "render_mode": "label_ids" if is_label_volume else "gaussian_splatting",
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
                "label_volume": bool(is_label_volume),
                "label_mode": "categorical_ids" if is_label_volume else "continuous",
                "label_id_count": int(label_id_total) if is_label_volume else 0,
                "label_id_lut_count": int(len(label_id_lut)) if is_label_volume else 0,
                "label_id_lut_truncated": bool(label_id_truncated)
                if is_label_volume
                else False,
                "volume_label_id_lut": [int(v) for v in label_id_lut]
                if is_label_volume
                else [],
                "volume_label_color_seed": int(self._stable_label_color_seed(path))
                if is_label_volume
                else 1337,
                "volume_label_colors": self._load_label_color_overrides(
                    source_path=path,
                    label_ids=label_id_lut,
                )
                if is_label_volume
                else {},
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
            volume = np.asarray(tif.asarray())
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
        return np.asarray(volume), spacing_zyx

    @staticmethod
    def _build_label_volume_render_defaults() -> dict[str, Any]:
        defaults = ThreeJsManager._with_volume_renderer_defaults(
            {
                "preset": "label_ids",
                "intensity": 1.0,
                "contrast": 1.0,
                "gamma": 1.0,
                "opacity": 0.9,
                "size": 0.032,
                "threshold": 0.0,
                "density": 1.0,
                "saturation": 1.0,
                "tf_low": 0.0,
                "tf_mid": 0.5,
                "tf_high": 1.0,
                "clip_axis": "none",
                "clip_center": 0.5,
                "clip_thickness": 1.0,
                "clip_invert": False,
                "palette": "allen_labels",
                "blend_mode": "normal",
                "point_texture": "section",
                "background_theme": "dark",
            },
            profile="cinematic",
        )
        defaults["render_style"] = "slab"
        defaults["gradient_opacity"] = False
        defaults["use_shading"] = False
        defaults["raymarch_steps"] = 180
        defaults["raymarch_step_scale"] = 1.0
        defaults["raymarch_jitter"] = 0.0
        defaults["section_emphasis"] = "neutral"
        return defaults

    @staticmethod
    def _stable_label_color_seed(source_path: Path) -> int:
        stem = str(source_path.stem or source_path.name or "")
        seed = 1337
        for ch in stem:
            seed = ((seed * 33) ^ ord(ch)) & 0xFFFFFFFF
        return int(seed % 100000)

    @staticmethod
    def _normalize_mapping_header(value: str) -> str:
        return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _parse_rgb_triplet(value: str) -> list[int] | None:
        text = str(value or "").strip()
        if not text:
            return None
        parts = [p for p in re.split(r"[\s,;|]+", text) if p]
        if len(parts) < 3:
            return None
        try:
            rgb = [int(float(parts[i])) for i in range(3)]
        except Exception:
            return None
        if not all(0 <= ch <= 255 for ch in rgb):
            return None
        return [rgb[0], rgb[1], rgb[2], 255]

    @staticmethod
    def _parse_hex_color(value: str) -> list[int] | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text.lower().startswith("0x"):
            text = text[2:]
        if text.startswith("#"):
            text = text[1:]
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6:
            return None
        try:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
        except Exception:
            return None
        return [r, g, b, 255]

    @staticmethod
    def _candidate_ontology_color_tables(source_path: Path) -> list[Path]:
        base = source_path.parent
        stem = source_path.stem
        stem_root = re.sub(
            r"(?i)(_labels?|_annotation|_annot|_seg(mentation)?|_atlas)$", "", stem
        ).strip("_-")
        names = [
            "structure_tree_safe_2017.csv",
            "structure_tree_safe_2017.json",
            "structure_tree.csv",
            "structure_tree.json",
            "allen_structure_tree.csv",
            "allen_structure_tree.json",
            "ontology.csv",
            "ontology.json",
            "structures.csv",
            "structures.json",
            f"{stem}_structure_tree.csv",
            f"{stem}_structure_tree.json",
            f"{stem}_ontology.csv",
            f"{stem}_ontology.json",
            f"{stem}_structures.csv",
            f"{stem}_structures.json",
        ]
        if stem_root and stem_root != stem:
            names.extend(
                [
                    f"{stem_root}_structure_tree.csv",
                    f"{stem_root}_structure_tree.json",
                    f"{stem_root}_ontology.csv",
                    f"{stem_root}_ontology.json",
                    f"{stem_root}_structures.csv",
                    f"{stem_root}_structures.json",
                ]
            )
        candidates: list[Path] = []
        for name in names:
            p = base / name
            if p.exists() and p.is_file():
                candidates.append(p)
        patterns = (
            "*structure*tree*.csv",
            "*structure*tree*.json",
            "*ontology*.csv",
            "*ontology*.json",
            "*atlas*structure*.csv",
            "*atlas*structure*.json",
            "*atlas*label*.csv",
            "*atlas*label*.json",
            "*label*map*.csv",
            "*label*map*.json",
        )
        for pattern in patterns:
            for p in sorted(base.glob(pattern)):
                if not p.is_file():
                    continue
                if p not in candidates:
                    candidates.append(p)
                if len(candidates) >= 24:
                    return candidates
        return candidates

    @staticmethod
    def _extract_label_color_from_record(record: dict[str, Any]) -> list[int] | None:
        if not isinstance(record, dict):
            return None
        hex_keys = (
            "color_hex_triplet",
            "hex_triplet",
            "color_hex",
            "hex_color",
            "hex",
            "color",
        )
        rgb_keys = (
            "rgb",
            "rgb_triplet",
            "rgb_color",
            "color_rgb",
        )
        for key in hex_keys:
            if key in record:
                rgba = ThreeJsManager._parse_hex_color(str(record.get(key, "")))
                if rgba is not None:
                    return rgba
        for key in rgb_keys:
            if key in record:
                rgba = ThreeJsManager._parse_rgb_triplet(str(record.get(key, "")))
                if rgba is not None:
                    return rgba
        if all(k in record for k in ("r", "g", "b")):
            rgba = ThreeJsManager._parse_rgb_triplet(
                f"{record.get('r', '')},{record.get('g', '')},{record.get('b', '')}"
            )
            if rgba is not None:
                return rgba
        if all(k in record for k in ("red", "green", "blue")):
            rgba = ThreeJsManager._parse_rgb_triplet(
                f"{record.get('red', '')},{record.get('green', '')},{record.get('blue', '')}"
            )
            if rgba is not None:
                return rgba
        return None

    @staticmethod
    def _extract_label_id_from_record(record: dict[str, Any]) -> int | None:
        if not isinstance(record, dict):
            return None
        for key in (
            "id",
            "structure_id",
            "atlas_id",
            "label_id",
            "region_id",
            "value",
            "index",
        ):
            if key not in record:
                continue
            try:
                return int(float(str(record.get(key, "")).strip()))
            except Exception:
                continue
        return None

    @staticmethod
    def _collect_color_records_from_json(value: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if isinstance(value, dict):
            normalized = {
                ThreeJsManager._normalize_mapping_header(str(k)): v
                for k, v in value.items()
            }
            candidate: dict[str, Any] = {}
            for k, v in normalized.items():
                candidate[k] = v
            if ThreeJsManager._extract_label_id_from_record(candidate) is not None:
                records.append(candidate)
            for child in value.values():
                records.extend(ThreeJsManager._collect_color_records_from_json(child))
            return records
        if isinstance(value, list):
            for item in value:
                records.extend(ThreeJsManager._collect_color_records_from_json(item))
        return records

    @staticmethod
    def _load_label_color_overrides_from_json(
        *, json_path: Path, wanted: set[int]
    ) -> dict[str, list[int]]:
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        discovered: dict[str, list[int]] = {}
        for record in ThreeJsManager._collect_color_records_from_json(parsed):
            if len(discovered) >= len(wanted):
                break
            label_id = ThreeJsManager._extract_label_id_from_record(record)
            if label_id is None or label_id not in wanted:
                continue
            key = str(label_id)
            if key in discovered:
                continue
            rgba = ThreeJsManager._extract_label_color_from_record(record)
            if rgba is None:
                continue
            discovered[key] = rgba
        return discovered

    @staticmethod
    def _load_label_color_overrides_from_csv(
        *, csv_path: Path, wanted: set[int]
    ) -> dict[str, list[int]]:
        id_column_candidates = (
            "id",
            "structure_id",
            "atlas_id",
            "label_id",
            "region_id",
            "value",
            "index",
        )
        hex_color_candidates = (
            "color_hex_triplet",
            "hex_triplet",
            "color_hex",
            "hex_color",
            "hex",
            "color",
        )
        rgb_compact_candidates = (
            "rgb",
            "rgb_triplet",
            "rgb_color",
            "color_rgb",
        )
        rgb_r_candidates = ("r", "red")
        rgb_g_candidates = ("g", "green")
        rgb_b_candidates = ("b", "blue")

        def _select_column(
            fieldnames: list[str], candidates: tuple[str, ...]
        ) -> str | None:
            normalized = {
                ThreeJsManager._normalize_mapping_header(name): name
                for name in fieldnames
                if name
            }
            for candidate in candidates:
                if candidate in normalized:
                    return normalized[candidate]
            return None

        try:
            raw = csv_path.read_text(encoding="utf-8-sig")
        except Exception:
            return {}
        if not raw.strip():
            return {}
        try:
            dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",\t;")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(raw.splitlines(), dialect=dialect)
        fieldnames = list(reader.fieldnames or [])
        id_col = _select_column(fieldnames, id_column_candidates)
        if id_col is None:
            return {}
        hex_col = _select_column(fieldnames, hex_color_candidates)
        rgb_compact_col = _select_column(fieldnames, rgb_compact_candidates)
        r_col = _select_column(fieldnames, rgb_r_candidates)
        g_col = _select_column(fieldnames, rgb_g_candidates)
        b_col = _select_column(fieldnames, rgb_b_candidates)
        if (
            hex_col is None
            and rgb_compact_col is None
            and not (r_col and g_col and b_col)
        ):
            return {}
        discovered: dict[str, list[int]] = {}
        for row in reader:
            try:
                label_id = int(float(str(row.get(id_col, "")).strip()))
            except Exception:
                continue
            if label_id not in wanted:
                continue
            if str(label_id) in discovered:
                continue
            rgba = None
            if hex_col:
                rgba = ThreeJsManager._parse_hex_color(str(row.get(hex_col, "")))
            if rgba is None and rgb_compact_col:
                rgba = ThreeJsManager._parse_rgb_triplet(
                    str(row.get(rgb_compact_col, ""))
                )
            if rgba is None and r_col and g_col and b_col:
                rgba = ThreeJsManager._parse_rgb_triplet(
                    f"{row.get(r_col, '')},{row.get(g_col, '')},{row.get(b_col, '')}"
                )
            if rgba is None:
                continue
            discovered[str(label_id)] = rgba
        return discovered

    @staticmethod
    def _load_label_color_overrides(
        *, source_path: Path, label_ids: list[int]
    ) -> dict[str, list[int]]:
        wanted = {int(v) for v in label_ids if int(v) > 0}
        if not wanted:
            return {}
        discovered: dict[str, list[int]] = {}
        for table_path in ThreeJsManager._candidate_ontology_color_tables(source_path):
            if len(discovered) >= len(wanted):
                break
            current: dict[str, list[int]]
            if table_path.suffix.lower() == ".json":
                current = ThreeJsManager._load_label_color_overrides_from_json(
                    json_path=table_path,
                    wanted=wanted,
                )
            else:
                current = ThreeJsManager._load_label_color_overrides_from_csv(
                    csv_path=table_path,
                    wanted=wanted,
                )
            for key, value in current.items():
                if key not in discovered:
                    discovered[key] = value
                if len(discovered) >= len(wanted):
                    break
        return discovered

    @staticmethod
    def _prepare_tiff_label_volume(
        *, sampled: Any, source_path: Path
    ) -> tuple[bool, Any, list[int], int, bool]:
        try:
            import numpy as np
        except Exception:
            return False, sampled, [], 0, False

        finite = sampled[np.isfinite(sampled)]
        if finite.size <= 0:
            return False, sampled, [], 0, False

        integer_dtype = np.issubdtype(sampled.dtype, np.integer)
        if integer_dtype:
            labels_raw = sampled.astype(np.int64, copy=False)
        else:
            rounded = np.rint(finite)
            fractional = np.abs(finite - rounded)
            if float(np.quantile(fractional, 0.995)) > 1e-4:
                return False, sampled, [], 0, False
            labels_raw = np.rint(sampled).astype(np.int64)

        labels_nonzero = labels_raw[labels_raw > 0]
        if labels_nonzero.size <= 0:
            return False, sampled, [], 0, False
        label_ids, label_counts = np.unique(labels_nonzero, return_counts=True)
        if label_ids.size <= 1:
            return False, sampled, [], 0, False

        label_span = max(1, int(label_ids[-1] - label_ids[0] + 1))
        fill_ratio = float(label_ids.size) / float(label_span)
        hint = str(source_path.stem or "").lower()
        hinted = any(k in hint for k in ("label", "annotation", "atlas", "seg", "mask"))
        if not integer_dtype and not hinted:
            return False, sampled, [], 0, False
        # Detect categorical label volumes conservatively.
        # Continuous integer-valued microscopy stacks (e.g. 12/16-bit intensity)
        # can have max values >255 and should not be treated as label IDs.
        unique_count = int(label_ids.size)
        nonzero_count = int(labels_nonzero.size)
        unique_ratio = float(unique_count) / float(max(1, nonzero_count))
        is_sparse_id_space = fill_ratio < 0.45
        is_small_vocab = unique_count <= 2048
        is_low_unique_ratio = unique_ratio <= 0.01

        is_label_volume = bool(
            hinted or (is_sparse_id_space and is_small_vocab and is_low_unique_ratio)
        )
        if not is_label_volume:
            return False, sampled, [], 0, False

        label_ids_total = int(label_ids.size)
        max_lut = 255
        if label_ids.size > max_lut:
            keep = np.argpartition(label_counts, -max_lut)[-max_lut:]
            label_ids = label_ids[keep]
        selected = np.sort(label_ids.astype(np.int64))
        lut = [int(v) for v in selected.tolist()]
        selected_arr = selected
        if selected_arr.size <= 0:
            return False, sampled, [], 0, False

        flat = labels_raw.reshape(-1)
        indices = np.searchsorted(selected_arr, flat)
        valid = (indices >= 0) & (indices < selected_arr.size)
        matched = np.zeros(flat.shape, dtype=bool)
        if np.any(valid):
            matched[valid] = selected_arr[indices[valid]] == flat[valid]
        mapped = np.zeros(flat.shape, dtype=np.uint8)
        if np.any(matched):
            mapped[matched] = (indices[matched] + 1).astype(np.uint8)
        mapped_volume = mapped.reshape(labels_raw.shape)

        return (
            True,
            mapped_volume,
            lut,
            label_ids_total,
            bool(label_ids_total > len(lut)),
        )

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
        candidates = self._resolve_zarr_array_candidates_for_viewer(source)
        failures: list[tuple[str, Exception]] = []
        for array, source_dataset_path, axis_names in candidates:
            try:
                return self._build_zarr_payload_from_array(
                    source=source,
                    source_path=path,
                    array=array,
                    source_dataset_path=source_dataset_path,
                    axis_names=axis_names,
                    np_module=np,
                )
            except Exception as exc:
                failures.append((str(source_dataset_path or ""), exc))
                logger.warning(
                    "Failed to read Zarr dataset '%s' for Three.js viewer: %s",
                    source_dataset_path or "<root>",
                    exc,
                )
        if failures:
            last_dataset, last_exc = failures[-1]
            if len(failures) == 1:
                raise RuntimeError(str(last_exc)) from last_exc
            failed_paths = ", ".join(
                f"'{dataset or '<root>'}'" for dataset, _exc in failures
            )
            raise RuntimeError(
                f"Unable to read any Zarr dataset for 3D viewer; tried {failed_paths}. "
                f"Last error from '{last_dataset or '<root>'}': {last_exc}"
            ) from last_exc
        raise RuntimeError("No readable array found in Zarr store.")

    def _build_zarr_payload_from_array(
        self,
        *,
        source: Any,
        source_path: Path,
        array: Any,
        source_dataset_path: str,
        axis_names: list[str],
        np_module: Any,
    ) -> dict[str, Any]:
        np = np_module
        shape = tuple(int(v) for v in getattr(array, "shape", ()) or ())
        if not shape:
            raise RuntimeError(f"Zarr store has invalid shape metadata: {source_path}")
        if any(int(v) <= 0 for v in shape):
            raise RuntimeError(
                f"Zarr store has non-positive shape dimensions: {source_path}"
            )

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
            raise RuntimeError(f"Zarr volume is empty: {source_path}")

        finite = sampled[np.isfinite(sampled)]
        if finite.size <= 0:
            raise RuntimeError(f"Zarr volume has no finite values: {source_path}")
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
            source_path=source_path,
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
            source_path=source_path,
            intensity_inverted=bool(intensity_inverted),
        )
        interleaved_detected = "interleaved" in str(source_path.stem or "").lower()
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
                "source_path": str(source_path),
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
        return self._resolve_zarr_array_candidates_for_viewer(source)[0]

    def _resolve_zarr_array_candidates_for_viewer(
        self, source: Any
    ) -> list[tuple[Any, str, list[str]]]:
        shape = tuple(int(v) for v in getattr(source, "shape", ()) or ())
        if shape:
            axis_names = self._normalize_zarr_axes(
                None,
                ndim=len(shape),
                array=source,
            )
            return [(source, "", axis_names)]

        multiscale_candidates = self._zarr_multiscale_candidates(source)
        if multiscale_candidates:
            preferred = self._choose_zarr_multiscale_candidate(multiscale_candidates)
            ordered: list[tuple[Any, str, Any]] = [preferred]
            for candidate in multiscale_candidates:
                if candidate is preferred:
                    continue
                ordered.append(candidate)
            resolved: list[tuple[Any, str, list[str]]] = []
            for array, path, axes in ordered:
                axis_names = self._normalize_zarr_axes(
                    axes, ndim=len(array.shape), array=array
                )
                resolved.append((array, path, axis_names))
            return resolved

        selected = self._select_zarr_array(source)
        if selected is None:
            raise RuntimeError("No readable array found in Zarr store.")
        axis_names = self._normalize_zarr_axes(
            None,
            ndim=len(getattr(selected, "shape", ()) or ()),
            array=selected,
        )
        return [(selected, "", axis_names)]

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
