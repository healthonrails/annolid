from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from qtpy import QtWidgets

from annolid.utils.logger import logger


class Sam2Manager:
    """Encapsulate SAM2 model detection and processor setup."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        self.window = window

    @staticmethod
    def is_sam2_model(identifier: str, weight: str) -> bool:
        identifier = (identifier or "").lower()
        weight = (weight or "").lower()
        return "sam2_hiera" in identifier or "sam2_hiera" in weight

    @staticmethod
    def resolve_model_config(identifier: str, weight: str) -> str:
        """
        Resolve the SAM2 config file name based on the selected identifier or weight.
        Falls back to the small hierarchy config if nothing matches.
        """
        key = f"{identifier or ''}|{weight or ''}".lower()
        if "hiera_l" in key:
            return "sam2.1_hiera_l.yaml"
        if "hiera_s" in key:
            return "sam2.1_hiera_s.yaml"
        return "sam2.1_hiera_s.yaml"

    @staticmethod
    def resolve_checkpoint_path(weight: str) -> Optional[str]:
        """
        Try to resolve the absolute checkpoint path for SAM2 models.
        Returns None to use the default download location when the file is not found.
        """
        if not weight:
            return None

        weight = weight.strip()
        if not weight:
            return None

        weight_path = Path(weight)
        if weight_path.exists():
            return str(weight_path)

        checkpoints_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "segmentation"
            / "SAM"
            / "segment-anything-2"
            / "checkpoints"
        )

        candidate = checkpoints_dir / weight_path.name
        if candidate.exists():
            return str(candidate)

        lower_name = weight_path.name.lower()
        fallback_names = []
        if "hiera_l" in lower_name:
            fallback_names.extend(["sam2_hiera_large.pt", "sam2.1_hiera_large.pt"])
        elif "hiera_s" in lower_name:
            fallback_names.extend(["sam2_hiera_small.pt", "sam2.1_hiera_small.pt"])

        for fallback_name in fallback_names:
            fallback_candidate = checkpoints_dir / fallback_name
            if fallback_candidate.exists():
                return str(fallback_candidate)

        return None

    def build_video_processor(
        self, model_name: str, model_weight: str, epsilon_for_polygon: float
    ):
        """Return a callable SAM2 video processor or None on failure."""
        try:
            from annolid.segmentation.SAM.sam_v2 import process_video
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window,
                "SAM2 import error",
                f"Failed to load SAM2 packages.\n{exc}",
            )
            return None

        sam2_config = self.resolve_model_config(model_name, model_weight)
        sam2_checkpoint = self.resolve_checkpoint_path(model_weight)
        logger.info(
            "Using SAM2 config '%s' with checkpoint '%s'",
            sam2_config,
            sam2_checkpoint if sam2_checkpoint else "auto-download",
        )
        return functools.partial(
            process_video,
            video_path=self.window.video_file,
            checkpoint_path=sam2_checkpoint,
            model_config=sam2_config,
            epsilon_for_polygon=epsilon_for_polygon,
        )
