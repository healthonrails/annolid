from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np


@dataclass
class VolumeData:
    """Core data structure for a loaded 3D volume."""

    array: Optional[np.ndarray]
    spacing: Optional[Tuple[float, float, float]]
    vmin: float
    vmax: float
    min_val: float = 0.0
    max_val: float = 1.0
    is_grayscale: bool = True
    is_out_of_core: bool = False
    backing_path: Optional[Path] = None
    vtk_image: Any = None
    slice_mode: bool = False
    slice_loader: Any = None
    slice_axis: int = 0
    current_slice_index: int = 0
    initial_slice_index: int = 0
    gamma: float = 1.0
    window_override: bool = False
    volume_shape: Optional[tuple[int, int, int]] = None
    shape: tuple[int, int, int] = (0, 0, 0)
    is_label_map: bool = False


@dataclass
class _OverlayVolumeEntry:
    """Entry for an overlay volume."""

    data: VolumeData
    name: str
    visible: bool = True
    opacity: float = 0.5
    actor: Any = None
