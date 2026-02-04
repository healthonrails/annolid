from __future__ import annotations

import base64
import io
import math
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


PY2 = False
QT4 = False
QT5 = True


try:
    __version__ = metadata.version("annolid")
except Exception:
    __version__ = "unknown"


def get_ai_models() -> list[Any]:
    """Return AI model classes for point-based polygon/mask prediction.

    Annolid historically relied on `labelme.ai.MODELS`. To avoid requiring the
    external `labelme` package, we provide an in-tree fallback backed by the
    ONNX SegmentAnything implementation shipped in `annolid/segmentation/SAM`.
    """
    try:
        from labelme.ai import MODELS as labelme_models  # type: ignore

        return list(labelme_models)
    except Exception:
        pass

    # -------------------------------
    # In-tree fallback (no labelme).
    # -------------------------------
    sam_dir = Path(__file__).resolve().parents[1] / "segmentation" / "SAM"
    edge_encoder = sam_dir / "edge_sam_3x_encoder.onnx"
    edge_decoder = sam_dir / "edge_sam_3x_decoder.onnx"

    class _AnnolidSegmentAnythingONNX:
        """Thin wrapper that matches the LabelMe AI model API used by Canvas."""

        def __init__(self, name: str, encoder_path: Path, decoder_path: Path):
            self.name = name
            self._encoder_path = Path(encoder_path)
            self._decoder_path = Path(decoder_path)
            self._model = None

        def _ensure_model(self):
            if self._model is not None:
                return
            if not self._encoder_path.exists() or not self._decoder_path.exists():
                raise FileNotFoundError(
                    f"Missing SAM ONNX weights for '{self.name}': "
                    f"{self._encoder_path.name}, {self._decoder_path.name}"
                )
            try:
                from annolid.segmentation.SAM.segment_anything import (
                    SegmentAnythingModel,
                )

                self._model = SegmentAnythingModel(
                    self.name, str(self._encoder_path), str(self._decoder_path)
                )
            except ModuleNotFoundError as exc:
                # Keep draw-mode UX/testability when optional ONNX runtime is missing.
                if getattr(exc, "name", "") == "onnxruntime":
                    self._model = _PointPromptFallbackModel(self.name)
                else:
                    raise

        def set_image(self, image: np.ndarray):
            self._ensure_model()
            return self._model.set_image(image)

        def predict_polygon_from_points(self, points, point_labels):
            self._ensure_model()
            return self._model.predict_polygon_from_points(points, point_labels)

        def predict_mask_from_points(self, points, point_labels):
            self._ensure_model()
            return self._model.predict_mask_from_points(points, point_labels)

    class _PointPromptFallbackModel:
        """Lightweight point-prompt model used when onnxruntime is unavailable."""

        def __init__(self, name: str):
            self.name = name
            self._image: np.ndarray | None = None

        def set_image(self, image: np.ndarray):
            self._image = np.asarray(image)

        def predict_mask_from_points(self, points, point_labels):
            if self._image is None:
                raise RuntimeError("Image must be set before prediction.")
            height, width = self._image.shape[:2]
            mask = np.zeros((height, width), dtype=bool)
            if points is None or point_labels is None:
                return mask

            radius = max(2, int(round(min(height, width) * 0.08)))
            yy, xx = np.ogrid[:height, :width]
            for (x, y), label in zip(points, point_labels):
                px = int(round(float(x)))
                py = int(round(float(y)))
                disk = (xx - px) ** 2 + (yy - py) ** 2 <= radius * radius
                if int(label) > 0:
                    mask |= disk
                else:
                    mask &= ~disk
            return mask

        def predict_polygon_from_points(self, points, point_labels):
            mask = self.predict_mask_from_points(points, point_labels)
            ys, xs = np.where(mask)
            if ys.size == 0:
                return np.empty((0, 2), dtype=np.float32)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            return np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )

    def _make_model_class(display_name: str):
        class _Model(_AnnolidSegmentAnythingONNX):
            name = display_name

            def __init__(self):
                super().__init__(display_name, edge_encoder, edge_decoder)

        _Model.__name__ = f"AnnolidSAMONNX_{display_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        return _Model

    # Keep display names aligned with existing Annolid defaults/config.
    # For now, all entries map to the shipped EdgeSAM ONNX weights.
    # (Optional) future: add additional ONNX weights/download support.
    return [
        _make_model_class("SegmentAnything (Edge)"),
        _make_model_class("SegmentAnything (speed)"),
        _make_model_class("SegmentAnything (balanced)"),
        _make_model_class("SegmentAnything (accuracy)"),
        _make_model_class("EfficientSam (speed)"),
        _make_model_class("EfficientSam (accuracy)"),
    ]


AI_MODELS = get_ai_models()


def apply_exif_orientation(image_pil: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image_pil)


def img_data_to_png_data(image_data: bytes) -> bytes:
    with Image.open(io.BytesIO(image_data)) as image:
        with io.BytesIO() as out:
            image.save(out, format="PNG")
            return out.getvalue()


def img_data_to_arr(image_data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_data)) as image:
        return np.asarray(image)


def img_arr_to_data(image_arr: np.ndarray) -> bytes:
    image = Image.fromarray(np.asarray(image_arr))
    with io.BytesIO() as out:
        image.save(out, format="PNG")
        return out.getvalue()


def img_arr_to_b64(image_arr: np.ndarray) -> str:
    return base64.b64encode(img_arr_to_data(image_arr)).decode("utf-8")


def img_b64_to_arr(image_b64: str | bytes) -> np.ndarray:
    if isinstance(image_b64, str):
        payload = image_b64.encode("utf-8")
    else:
        payload = image_b64
    return img_data_to_arr(base64.b64decode(payload))


def img_pil_to_data(image_pil: Image.Image) -> bytes:
    with io.BytesIO() as out:
        image_pil.save(out, format="PNG")
        return out.getvalue()


def img_qt_to_arr(qimage: Any) -> np.ndarray:
    try:
        from qimage2ndarray import rgb_view

        return np.asarray(rgb_view(qimage))
    except Exception:
        from qtpy import QtGui

        qt_format = getattr(QtGui.QImage, "Format_RGBA8888", None)
        if qt_format is None:
            qt_format = QtGui.QImage.Format_ARGB32
        image = qimage.convertToFormat(qt_format)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr[..., :3]


def _points_to_xy(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in points]


def shape_to_mask(
    img_shape: Sequence[int],
    points: Sequence[Sequence[float]],
    shape_type: Optional[str] = None,
    line_width: int = 10,
    point_size: int = 5,
) -> np.ndarray:
    shape_type = shape_type or "polygon"
    mask_image = Image.new("L", (int(img_shape[1]), int(img_shape[0])), 0)
    draw = ImageDraw.Draw(mask_image)
    xy = _points_to_xy(points)

    if shape_type == "circle" and len(xy) >= 2:
        (x1, y1), (x2, y2) = xy[0], xy[1]
        radius = math.hypot(x2 - x1, y2 - y1)
        draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=1)
    elif shape_type == "rectangle" and len(xy) >= 2:
        (x1, y1), (x2, y2) = xy[0], xy[1]
        draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=1)
    elif shape_type == "line" and len(xy) >= 2:
        draw.line(xy=xy, fill=1, width=int(line_width))
    elif shape_type == "linestrip" and len(xy) >= 2:
        draw.line(xy=xy, fill=1, width=int(line_width))
    elif shape_type in {"point", "points"} and xy:
        x, y = xy[0]
        draw.ellipse(
            (x - point_size, y - point_size, x + point_size, y + point_size), fill=1
        )
    else:
        if len(xy) >= 3:
            draw.polygon(xy=xy, outline=1, fill=1)

    return np.asarray(mask_image, dtype=bool)


def shapes_to_label(
    img_shape: Sequence[int],
    shapes: Sequence[dict[str, Any]],
    label_name_to_value: dict[str, int],
    *,
    label_name_list: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros(img_shape[:2], dtype=np.int32)
    instances: list[tuple[str, Any]] = []

    for shape in shapes:
        points = shape.get("points")
        if not points:
            continue
        label = str(shape.get("label", ""))
        if not label:
            continue

        if label_name_list is not None and label not in label_name_list:
            continue

        group_id = shape.get("group_id")
        if group_id is None:
            group_id = id(shape)

        instance_key = (label, group_id)
        if instance_key not in instances:
            instances.append(instance_key)
        ins_id = instances.index(instance_key) + 1

        cls_id = label_name_to_value.get(label)
        if cls_id is None:
            continue

        mask = shape_to_mask(
            img_shape,
            points,
            shape_type=shape.get("shape_type", "polygon"),
        )
        cls[mask] = int(cls_id)
        ins[mask] = ins_id

    return cls, ins


def lblsave(filename: str, lbl: np.ndarray) -> None:
    arr = np.asarray(lbl)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    image = Image.fromarray(arr, mode="P")
    try:
        import imgviz

        colormap = imgviz.label_colormap()
        image.putpalette(colormap.flatten())
    except Exception:
        pass
    image.save(filename)


def distance(point: Any) -> float:
    try:
        x = float(point.x())
        y = float(point.y())
    except Exception:
        arr = np.asarray(point, dtype=float)
        x, y = float(arr[0]), float(arr[1])
    return math.hypot(x, y)


def distancetoline(point: Any, line: Sequence[Any]) -> float:
    p1, p2 = line

    def _xy(obj: Any) -> tuple[float, float]:
        try:
            return float(obj.x()), float(obj.y())
        except Exception:
            arr = np.asarray(obj, dtype=float)
            return float(arr[0]), float(arr[1])

    x0, y0 = _xy(point)
    x1, y1 = _xy(p1)
    x2, y2 = _xy(p2)

    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return math.hypot(x0 - x1, y0 - y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / denom
    t = max(0.0, min(1.0, t))
    px = x1 + t * dx
    py = y1 + t * dy
    return math.hypot(x0 - px, y0 - py)


def addActions(widget: Any, actions: Iterable[Any]) -> None:
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif hasattr(action, "menuAction"):
            widget.addAction(action.menuAction())
        else:
            widget.addAction(action)


def newAction(
    parent: Any,
    text: str,
    slot: Any = None,
    shortcut: Any = None,
    icon: Optional[str] = None,
    tip: Optional[str] = None,
    checkable: bool = False,
    enabled: bool = True,
    checked: bool = False,
    iconText: Optional[str] = None,
) -> Any:
    from qtpy import QtGui, QtWidgets

    action = QtWidgets.QAction(text, parent)
    if icon:
        action.setIcon(QtGui.QIcon.fromTheme(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            action.setShortcuts(shortcut)
        else:
            action.setShortcut(shortcut)
    if tip:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        action.triggered.connect(slot)
    action.setCheckable(checkable)
    action.setEnabled(enabled)
    if checkable:
        action.setChecked(checked)
    if iconText:
        action.setIconText(iconText)
    return action


class _UtilsNamespace:
    apply_exif_orientation = staticmethod(apply_exif_orientation)
    img_data_to_png_data = staticmethod(img_data_to_png_data)
    img_data_to_arr = staticmethod(img_data_to_arr)
    img_arr_to_data = staticmethod(img_arr_to_data)
    img_arr_to_b64 = staticmethod(img_arr_to_b64)
    img_b64_to_arr = staticmethod(img_b64_to_arr)
    img_pil_to_data = staticmethod(img_pil_to_data)
    img_qt_to_arr = staticmethod(img_qt_to_arr)
    shape_to_mask = staticmethod(shape_to_mask)
    shapes_to_label = staticmethod(shapes_to_label)
    lblsave = staticmethod(lblsave)
    distance = staticmethod(distance)
    distancetoline = staticmethod(distancetoline)
    addActions = staticmethod(addActions)
    newAction = staticmethod(newAction)


utils = _UtilsNamespace()
