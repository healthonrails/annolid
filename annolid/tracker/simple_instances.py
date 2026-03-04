"""
Lightweight drop-in replacement for ``detectron2.structures.Instances``.

Provides the subset of the D2 Instances API that Annolid's tracker and
inference code actually uses, without any detectron2 dependency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class SimpleInstances:
    """Minimal container for per-image detection results.

    Attributes match the fields consumed by Annolid's ``_process_instances``
    and the bundled IOU tracker:

    * ``pred_boxes`` — ``Tensor[N, 4]`` in ``(x1, y1, x2, y2)`` format
    * ``pred_classes`` — ``Tensor[N]`` integer class indices
    * ``scores`` — ``Tensor[N]`` confidence scores
    * ``pred_masks`` — ``Tensor[N, H, W]`` binary masks (optional)
    * ``ID`` — ``list[int | None]`` tracking IDs (optional)
    * ``ID_period`` — ``list[int | None]`` consecutive-frame count
    * ``lost_frame_count`` — ``list[int | None]``
    """

    image_size: Tuple[int, int]  # (H, W)
    _fields: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Field access — attribute-style  (instances.pred_boxes, etc.)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "image_size":
            raise AttributeError(name)
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no field '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("image_size", "_fields"):
            super().__setattr__(name, value)
        else:
            self._fields[name] = value

    # ------------------------------------------------------------------
    # D2-compatible helpers
    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self._fields

    def set(self, name: str, value: Any) -> None:
        self._fields[name] = value

    def get(self, name: str) -> Any:
        return self._fields[name]

    def remove(self, name: str) -> None:
        del self._fields[name]

    def __len__(self) -> int:
        for v in self._fields.values():
            if isinstance(v, (torch.Tensor, list)):
                return len(v)
        return 0

    def __getitem__(self, item):
        """Support indexing / boolean-mask selection like D2 Instances."""
        ret = SimpleInstances(image_size=self.image_size)
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                ret._fields[k] = v[item]
            elif isinstance(v, list):
                if isinstance(item, (int, slice)):
                    ret._fields[k] = v[item] if isinstance(item, slice) else [v[item]]
                else:
                    # boolean / index tensor
                    indices = item
                    if isinstance(indices, torch.Tensor):
                        indices = indices.tolist()
                    ret._fields[k] = [v[i] for i in range(len(v)) if indices[i]]
            else:
                ret._fields[k] = v
        return ret

    # ------------------------------------------------------------------
    # Concatenation (used by the tracker to merge untracked instances)
    # ------------------------------------------------------------------

    @classmethod
    def cat(cls, instances_list: List["SimpleInstances"]) -> "SimpleInstances":
        """Concatenate a list of ``SimpleInstances`` along the instance axis."""
        assert len(instances_list) > 0
        image_size = instances_list[0].image_size
        ret = cls(image_size=image_size)

        all_keys = set()
        for inst in instances_list:
            all_keys.update(inst._fields.keys())

        for key in all_keys:
            values = []
            for inst in instances_list:
                if not inst.has(key):
                    continue
                values.append(inst._fields[key])
            if not values:
                continue
            if all(isinstance(v, torch.Tensor) for v in values):
                ret._fields[key] = torch.cat(values, dim=0)
            elif all(isinstance(v, list) for v in values):
                merged: list = []
                for v in values:
                    merged.extend(v)
                ret._fields[key] = merged
            else:
                # Mixed or scalar — keep first
                ret._fields[key] = values[0]

        return ret

    def to(self, device: str) -> "SimpleInstances":
        """Move all tensor fields to *device* (no-op for lists)."""
        ret = SimpleInstances(image_size=self.image_size)
        for k, v in self._fields.items():
            if isinstance(v, torch.Tensor):
                ret._fields[k] = v.to(device)
            else:
                ret._fields[k] = v
        return ret

    def __deepcopy__(self, memo):
        new = SimpleInstances(image_size=self.image_size)
        new._fields = copy.deepcopy(self._fields, memo)
        return new
