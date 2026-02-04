from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image
from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.image_editing.errors import ImageEditingError
from annolid.image_editing.factory import (
    StableDiffusionCppPresetConfig,
    create_backend,
)
from annolid.image_editing.presets import list_stable_diffusion_cpp_presets
from annolid.image_editing.types import ImageEditMode, ImageEditRequest, ImageEditResult

try:
    from annolid.utils.annotation_compat import shape_to_mask  # type: ignore
except Exception:
    shape_to_mask = None  # type: ignore


@dataclass(frozen=True)
class ImageEditingConfig:
    backend: Literal["diffusers", "sdcpp", "pillow"]
    mode: ImageEditMode
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: Optional[int]
    strength: float
    num_images: int

    # Diffusers
    model_id: str = "Qwen/Qwen-Image-2512"
    device: str = "auto"
    dtype: str = "auto"

    # stable-diffusion.cpp preset
    sd_cli_path: str = ""
    preset: Optional[str] = None
    quant: Optional[str] = None
    llm_quant: Optional[str] = None
    extra_args: Tuple[str, ...] = ()


class ImageEditingWidget(QtWidgets.QWidget):
    runRequested = QtCore.Signal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItem("Diffusers (local)", userData="diffusers")
        self.backend_combo.addItem("stable-diffusion.cpp (sd-cli)", userData="sdcpp")
        self.backend_combo.addItem("Pillow (demo)", userData="pillow")

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Text → Image", userData="text_to_image")
        self.mode_combo.addItem("Frame → Image (img2img)", userData="image_to_image")
        self.mode_combo.addItem("Inpaint Selected Shapes", userData="inpaint")

        self.model_id_edit = QtWidgets.QLineEdit("Qwen/Qwen-Image-2512")
        self.model_id_edit.setPlaceholderText("Hugging Face model id (Diffusers)")

        self.device_edit = QtWidgets.QLineEdit("auto")
        self.device_edit.setPlaceholderText("auto|cpu|cuda|mps")

        self.dtype_edit = QtWidgets.QLineEdit("auto")
        self.dtype_edit.setPlaceholderText("auto|float32|float16|bfloat16")

        self.sd_cli_edit = QtWidgets.QLineEdit()
        self.sd_cli_edit.setPlaceholderText("Path to sd-cli (stable-diffusion.cpp)")

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("(none)", userData=None)
        for preset in list_stable_diffusion_cpp_presets():
            self.preset_combo.addItem(preset, userData=preset)

        self.quant_edit = QtWidgets.QLineEdit()
        self.quant_edit.setPlaceholderText("Preset quant (e.g. Q4_K_M)")

        self.llm_quant_edit = QtWidgets.QLineEdit()
        self.llm_quant_edit.setPlaceholderText("LLM quant (e.g. Q8_0)")

        self.extra_args_edit = QtWidgets.QLineEdit()
        self.extra_args_edit.setPlaceholderText("Extra sd-cli args (space-separated)")

        self.prompt_edit = QtWidgets.QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Prompt / edit instruction (required)")
        self.prompt_edit.setMaximumHeight(90)

        self.negative_prompt_edit = QtWidgets.QLineEdit()
        self.negative_prompt_edit.setPlaceholderText("Negative prompt")

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setValue(1024)

        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setValue(1024)

        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(1, 200)
        self.steps_spin.setValue(20)

        self.cfg_spin = QtWidgets.QDoubleSpinBox()
        self.cfg_spin.setRange(0.1, 30.0)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setValue(2.5)

        self.seed_edit = QtWidgets.QLineEdit()
        self.seed_edit.setPlaceholderText("Seed (blank = random)")

        self.strength_spin = QtWidgets.QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setValue(0.75)

        self.num_images_spin = QtWidgets.QSpinBox()
        self.num_images_spin.setRange(1, 16)
        self.num_images_spin.setValue(1)

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.clicked.connect(self._emit_request)

        self.status_label = QtWidgets.QLabel("")

        form = QtWidgets.QFormLayout()
        form.addRow("Backend", self.backend_combo)
        form.addRow("Mode", self.mode_combo)
        form.addRow("Prompt", self.prompt_edit)
        form.addRow("Negative", self.negative_prompt_edit)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("Steps", self.steps_spin)
        form.addRow("CFG scale", self.cfg_spin)
        form.addRow("Seed", self.seed_edit)
        form.addRow("Strength", self.strength_spin)
        form.addRow("# Images", self.num_images_spin)
        form.addRow("Diffusers model", self.model_id_edit)
        form.addRow("Diffusers device", self.device_edit)
        form.addRow("Diffusers dtype", self.dtype_edit)
        form.addRow("sd-cli", self.sd_cli_edit)
        form.addRow("sd-cli preset", self.preset_combo)
        form.addRow("Preset quant", self.quant_edit)
        form.addRow("LLM quant", self.llm_quant_edit)
        form.addRow("sd-cli extra", self.extra_args_edit)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.run_button)
        layout.addWidget(self.status_label)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setLayout(layout)

    def _emit_request(self) -> None:
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QtWidgets.QMessageBox.warning(
                self, "Missing prompt", "Please enter a prompt."
            )
            return

        backend = self.backend_combo.currentData()
        mode = self.mode_combo.currentData()

        seed = self.seed_edit.text().strip()
        seed_value = int(seed) if seed else None

        extra_args = tuple(self.extra_args_edit.text().strip().split())

        cfg = ImageEditingConfig(
            backend=str(backend),
            mode=mode,
            prompt=prompt,
            negative_prompt=self.negative_prompt_edit.text().strip(),
            width=int(self.width_spin.value()),
            height=int(self.height_spin.value()),
            steps=int(self.steps_spin.value()),
            cfg_scale=float(self.cfg_spin.value()),
            seed=seed_value,
            strength=float(self.strength_spin.value()),
            num_images=int(self.num_images_spin.value()),
            model_id=self.model_id_edit.text().strip() or "Qwen/Qwen-Image-2512",
            device=self.device_edit.text().strip() or "auto",
            dtype=self.dtype_edit.text().strip() or "auto",
            sd_cli_path=self.sd_cli_edit.text().strip(),
            preset=self.preset_combo.currentData(),
            quant=self.quant_edit.text().strip() or None,
            llm_quant=self.llm_quant_edit.text().strip() or None,
            extra_args=extra_args,
        )
        self.runRequested.emit(cfg)


class ImageEditingDockWidget(QtWidgets.QDockWidget):
    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window.tr("Image Editing"), window)
        self.window = window
        self.setObjectName("ImageEditingDock")

        self.editor_widget = ImageEditingWidget(self)
        self.editor_widget.runRequested.connect(self._handle_request)
        self.setWidget(self.editor_widget)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[FlexibleWorker] = None
        self._diffusers_backend_cache = {}

        self.destroyed.connect(lambda *_: self._clear_worker())

    def show_or_raise(self) -> None:
        if self.isHidden():
            self.show()
        self.raise_()

    def _clear_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.request_stop()
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception:
                pass
        self._worker = None
        self._thread = None

    def _handle_request(self, cfg: ImageEditingConfig) -> None:
        if self._thread is not None and self._thread.isRunning():
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("Busy"),
                self.window.tr("An image editing job is already running. Please wait."),
            )
            return

        try:
            request = self._build_request(cfg)
            backend = self._get_backend(cfg)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Invalid configuration"),
                str(exc),
            )
            return

        self.editor_widget.run_button.setEnabled(False)
        self.editor_widget.status_label.setText("Running…")

        self._worker = FlexibleWorker(_run_backend, backend, request)
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished_signal.connect(self._handle_finished)
        self._thread.start()

    def _build_request(self, cfg: ImageEditingConfig) -> ImageEditRequest:
        mode = cfg.mode
        init_image: Optional[Image.Image] = None
        mask_image: Optional[Image.Image] = None

        if mode in ("image_to_image", "inpaint"):
            getter = getattr(self.window, "_get_pil_image_from_state", None)
            init_image = getter() if callable(getter) else None
            if init_image is None:
                raise ValueError("Load a frame before using frame-based modes.")

        if mode == "inpaint":
            mask_image = self._mask_from_selected_shapes(init_image)

        return ImageEditRequest(
            mode=mode,
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.width,
            height=cfg.height,
            steps=cfg.steps,
            cfg_scale=cfg.cfg_scale,
            seed=cfg.seed,
            strength=cfg.strength,
            init_image=init_image,
            mask_image=mask_image,
            num_images=cfg.num_images,
        )

    def _get_backend(self, cfg: ImageEditingConfig):
        if cfg.backend == "diffusers":
            key = (cfg.model_id, cfg.device, cfg.dtype)
            backend = self._diffusers_backend_cache.get(key)
            if backend is None:
                backend = create_backend(
                    "diffusers",
                    diffusers_model_id=cfg.model_id,
                    diffusers_device=cfg.device,
                    diffusers_dtype=cfg.dtype,
                )
                self._diffusers_backend_cache[key] = backend
            return backend

        if cfg.backend == "sdcpp":
            preset_cfg = None
            if cfg.preset:
                preset_cfg = StableDiffusionCppPresetConfig(
                    name=str(cfg.preset),
                    quant=cfg.quant,
                    llm_quant=cfg.llm_quant,
                )
            return create_backend(
                "sdcpp",
                sd_cli_path=cfg.sd_cli_path,
                sdcpp_preset=preset_cfg,
                sdcpp_extra_args=tuple(cfg.extra_args),
            )

        return create_backend("pillow")

    def _mask_from_selected_shapes(self, image: Optional[Image.Image]) -> Image.Image:
        if image is None:
            raise ValueError("No image available for mask generation.")
        if shape_to_mask is None:
            raise ValueError(
                "Mask generation requires Annolid's annotation helpers (shape_to_mask missing)."
            )
        canvas = getattr(self.window, "canvas", None)
        shapes = getattr(canvas, "selectedShapes", []) if canvas is not None else []
        if not shapes:
            raise ValueError("Select one or more shapes to use as an inpaint mask.")

        h, w = image.height, image.width
        mask = np.zeros((h, w), dtype=np.uint8)
        for shape in shapes:
            pts = [_point_to_xy(p) for p in getattr(shape, "points", [])]
            shape_type = getattr(shape, "shape_type", None) or "polygon"
            min_pts = 3 if shape_type == "polygon" else 2
            if len(pts) < min_pts:
                continue
            shape_mask = shape_to_mask((h, w), pts, shape_type)
            mask[shape_mask] = 255
        if mask.max() == 0:
            raise ValueError("Selected shapes produced an empty mask.")
        return Image.fromarray(mask, mode="L")

    def _handle_finished(self, payload: object) -> None:
        self.editor_widget.run_button.setEnabled(True)
        self.editor_widget.status_label.setText("")

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None

        if isinstance(payload, Exception):
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Image editing failed"),
                str(payload),
            )
            return

        if not isinstance(payload, ImageEditResult) or not payload.images:
            QtWidgets.QMessageBox.warning(
                self.window,
                self.window.tr("Image editing failed"),
                self.window.tr("No image was produced."),
            )
            return

        first = payload.images[0]
        out_dir = Path(tempfile.gettempdir()) / "annolid_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annolid_image_edit_{uuid.uuid4().hex}.png"
        first.save(out_path)

        display = getattr(self.window, "display_generated_image", None)
        if callable(display):
            display(str(out_path))
        else:
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("Image generated"),
                self.window.tr("Saved: %s") % str(out_path),
            )


def _point_to_xy(point) -> Tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        try:
            return (float(point.x()), float(point.y()))
        except Exception:
            pass
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return (float(point[0]), float(point[1]))
    raise ValueError(f"Unsupported point type: {point!r}")


def _run_backend(backend, request: ImageEditRequest) -> ImageEditResult:
    try:
        return backend.run(request)
    except ImageEditingError:
        raise
    except Exception as exc:
        raise ImageEditingError(str(exc)) from exc
