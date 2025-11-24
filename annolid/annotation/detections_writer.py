import json
from pathlib import Path
from typing import Iterable, Optional

from annolid.annotation.keypoints import save_labels, format_shape
from annolid.utils.logger import logger
import numpy as np


def _json_safe(value):
    """
    Recursively convert numpy types to native Python types to ensure
    json.dumps succeeds.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


class DetectionsWriter:
    """
    Persist detection outputs to the annotation store and optional NDJSON file.

    This utility consolidates the persistence logic used by both offline YOLO
    batch processing and realtime pipelines that need to materialize detections.
    """

    def __init__(self,
                 output_dir: Path,
                 *,
                 enable_annotation_store: bool = True,
                 ndjson_filename: str = "predictions.ndjson"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_annotation_store = enable_annotation_store
        self.ndjson_path: Optional[Path] = None

        if ndjson_filename:
            self.ndjson_path = self.output_dir / ndjson_filename
            try:
                if self.ndjson_path.exists():
                    self.ndjson_path.unlink()
                self.ndjson_path.touch()
            except OSError as exc:
                logger.error("Unable to initialise NDJSON output at %s: %s",
                             self.ndjson_path, exc)
                self.ndjson_path = None

    def write(self,
              frame_index: int,
              frame_shape: tuple,
              shapes: Iterable,
              frame_other_data: Optional[dict] = None) -> None:
        shapes = list(shapes)
        if not shapes:
            return
        height, width, _ = frame_shape
        json_filename = f"{frame_index:09d}.json"
        json_path = self.output_dir / json_filename

        if self.enable_annotation_store:
            save_labels(
                filename=str(json_path),
                imagePath="",
                label_list=shapes,
                height=height,
                width=width,
                save_image_to_json=False,
                persist_json=False,
            )

        if self.ndjson_path is not None:
            record = {
                "frame_number": frame_index,
                "imageHeight": height,
                "imageWidth": width,
                "shapes": [format_shape(shape) for shape in shapes],
            }
            if frame_other_data:
                record["frame_other_data"] = frame_other_data
            try:
                record = _json_safe(record)
                with self.ndjson_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False))
                    fh.write("\n")
            except OSError as exc:
                logger.error(
                    "Failed to append frame %d to NDJSON file %s: %s",
                    frame_index,
                    self.ndjson_path,
                    exc,
                )
