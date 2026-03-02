from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from qtpy import QtWidgets

from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS
from annolid.segmentation.dino_kpseg import defaults as dino_defaults


DINO_HEAD_TYPE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Relational attention head", "relational"),
    ("Conv head (fast)", "conv"),
    ("Multitask head", "multitask"),
)

DINO_BCE_TYPE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Standard BCE", "bce"),
    ("Focal BCE", "focal"),
)


def _set_combo_data_options(
    combo: QtWidgets.QComboBox,
    options: Sequence[Tuple[str, str]],
    *,
    default: str,
) -> None:
    combo.clear()
    for label, value in options:
        combo.addItem(str(label), userData=str(value))
    idx = combo.findData(str(default))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def configure_dino_model_combo(
    combo: QtWidgets.QComboBox,
    *,
    default_identifier: str,
) -> None:
    combo.clear()
    for cfg in PATCH_SIMILARITY_MODELS:
        combo.addItem(cfg.display_name, cfg.identifier)
    idx = combo.findData(str(default_identifier))
    if idx >= 0:
        combo.setCurrentIndex(idx)


def configure_dino_head_type_combo(
    combo: QtWidgets.QComboBox,
    *,
    default_head_type: str = dino_defaults.HEAD_TYPE,
) -> None:
    _set_combo_data_options(
        combo,
        DINO_HEAD_TYPE_OPTIONS,
        default=str(default_head_type or dino_defaults.HEAD_TYPE),
    )


def configure_dino_bce_type_combo(
    combo: QtWidgets.QComboBox,
    *,
    default_bce_type: str = dino_defaults.BCE_TYPE,
) -> None:
    _set_combo_data_options(
        combo,
        DINO_BCE_TYPE_OPTIONS,
        default=str(default_bce_type or dino_defaults.BCE_TYPE),
    )


def apply_dino_head_control_state(
    *,
    head_type: str,
    relational_controls: Iterable[QtWidgets.QWidget],
    multitask_controls: Iterable[QtWidgets.QWidget],
) -> None:
    head_type_norm = str(head_type or "").strip().lower()
    relational_enabled = head_type_norm == "relational"
    multitask_enabled = head_type_norm == "multitask"
    for widget in relational_controls:
        try:
            widget.setEnabled(relational_enabled)
        except Exception:
            pass
    for widget in multitask_controls:
        try:
            widget.setEnabled(multitask_enabled)
        except Exception:
            pass


def apply_dino_bce_control_state(
    *,
    bce_type: str,
    focal_controls: Iterable[QtWidgets.QWidget],
) -> None:
    enabled = str(bce_type or "").strip().lower() == "focal"
    for widget in focal_controls:
        try:
            widget.setEnabled(enabled)
        except Exception:
            pass


@dataclass(frozen=True)
class DinoHeadLossControls:
    head_type_combo: QtWidgets.QComboBox
    attn_heads_spin: QtWidgets.QSpinBox
    attn_layers_spin: QtWidgets.QSpinBox
    hidden_dim_spin: QtWidgets.QSpinBox
    threshold_spin: QtWidgets.QDoubleSpinBox
    bce_type_combo: QtWidgets.QComboBox
    focal_alpha_spin: QtWidgets.QDoubleSpinBox
    focal_gamma_spin: QtWidgets.QDoubleSpinBox
    obj_loss_weight_spin: QtWidgets.QDoubleSpinBox
    box_loss_weight_spin: QtWidgets.QDoubleSpinBox
    inst_loss_weight_spin: QtWidgets.QDoubleSpinBox
    multitask_aux_warmup_spin: QtWidgets.QSpinBox

    @property
    def relational_controls(self) -> Tuple[QtWidgets.QWidget, ...]:
        return (self.attn_heads_spin, self.attn_layers_spin)

    @property
    def multitask_controls(self) -> Tuple[QtWidgets.QWidget, ...]:
        return (
            self.obj_loss_weight_spin,
            self.box_loss_weight_spin,
            self.inst_loss_weight_spin,
            self.multitask_aux_warmup_spin,
        )

    @property
    def focal_controls(self) -> Tuple[QtWidgets.QWidget, ...]:
        return (self.focal_alpha_spin, self.focal_gamma_spin)


def create_dino_head_loss_controls(
    parent: QtWidgets.QWidget,
    *,
    head_type: str = dino_defaults.HEAD_TYPE,
    attn_heads: int = dino_defaults.ATTN_HEADS,
    attn_layers: int = dino_defaults.ATTN_LAYERS,
    hidden_dim: int = dino_defaults.HIDDEN_DIM,
    threshold: float = dino_defaults.THRESHOLD,
    bce_type: str = dino_defaults.BCE_TYPE,
    focal_alpha: float = dino_defaults.FOCAL_ALPHA,
    focal_gamma: float = dino_defaults.FOCAL_GAMMA,
    obj_loss_weight: float = dino_defaults.OBJ_LOSS_WEIGHT,
    box_loss_weight: float = dino_defaults.BOX_LOSS_WEIGHT,
    inst_loss_weight: float = dino_defaults.INST_LOSS_WEIGHT,
    multitask_aux_warmup_epochs: int = dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
) -> DinoHeadLossControls:
    head_combo = QtWidgets.QComboBox(parent)
    configure_dino_head_type_combo(head_combo, default_head_type=str(head_type))

    attn_heads_spin = QtWidgets.QSpinBox(parent)
    attn_heads_spin.setRange(1, 32)
    attn_heads_spin.setValue(int(attn_heads))

    attn_layers_spin = QtWidgets.QSpinBox(parent)
    attn_layers_spin.setRange(1, 8)
    attn_layers_spin.setValue(int(attn_layers))

    hidden_dim_spin = QtWidgets.QSpinBox(parent)
    hidden_dim_spin.setRange(16, 2048)
    hidden_dim_spin.setSingleStep(16)
    hidden_dim_spin.setValue(int(hidden_dim))

    threshold_spin = QtWidgets.QDoubleSpinBox(parent)
    threshold_spin.setDecimals(3)
    threshold_spin.setRange(0.01, 0.99)
    threshold_spin.setSingleStep(0.01)
    threshold_spin.setValue(float(threshold))

    bce_combo = QtWidgets.QComboBox(parent)
    configure_dino_bce_type_combo(bce_combo, default_bce_type=str(bce_type))

    focal_alpha_spin = QtWidgets.QDoubleSpinBox(parent)
    focal_alpha_spin.setDecimals(3)
    focal_alpha_spin.setRange(0.0, 1.0)
    focal_alpha_spin.setSingleStep(0.05)
    focal_alpha_spin.setValue(float(focal_alpha))

    focal_gamma_spin = QtWidgets.QDoubleSpinBox(parent)
    focal_gamma_spin.setDecimals(3)
    focal_gamma_spin.setRange(0.0, 10.0)
    focal_gamma_spin.setSingleStep(0.1)
    focal_gamma_spin.setValue(float(focal_gamma))

    obj_w = QtWidgets.QDoubleSpinBox(parent)
    obj_w.setDecimals(6)
    obj_w.setRange(0.0, 10.0)
    obj_w.setSingleStep(0.01)
    obj_w.setValue(float(obj_loss_weight))

    box_w = QtWidgets.QDoubleSpinBox(parent)
    box_w.setDecimals(6)
    box_w.setRange(0.0, 10.0)
    box_w.setSingleStep(0.01)
    box_w.setValue(float(box_loss_weight))

    inst_w = QtWidgets.QDoubleSpinBox(parent)
    inst_w.setDecimals(6)
    inst_w.setRange(0.0, 10.0)
    inst_w.setSingleStep(0.01)
    inst_w.setValue(float(inst_loss_weight))

    multitask_aux = QtWidgets.QSpinBox(parent)
    multitask_aux.setRange(0, 1000)
    multitask_aux.setValue(int(multitask_aux_warmup_epochs))

    return DinoHeadLossControls(
        head_type_combo=head_combo,
        attn_heads_spin=attn_heads_spin,
        attn_layers_spin=attn_layers_spin,
        hidden_dim_spin=hidden_dim_spin,
        threshold_spin=threshold_spin,
        bce_type_combo=bce_combo,
        focal_alpha_spin=focal_alpha_spin,
        focal_gamma_spin=focal_gamma_spin,
        obj_loss_weight_spin=obj_w,
        box_loss_weight_spin=box_w,
        inst_loss_weight_spin=inst_w,
        multitask_aux_warmup_spin=multitask_aux,
    )
