#!/usr/bin/env python
"""CalMS21-style polygon dataset builder for two-instance polygons."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from annolid.datasets.polygon_utils import list_polygons, load_annotation, resample_polygon
from annolid.utils.logger import logger

__all__ = [
    "extract_polygons_from_frame",
    "process_sequence",
    "build_group_sequences",
    "save_group_dataset",
    "main",
]


def extract_polygons_from_frame(json_path: Path, num_points: int) -> Optional[List[List[List[float]]]]:
    """Extract and resample up to two polygons from a LabelMe JSON frame."""
    data = load_annotation(json_path)
    if data is None:
        return None

    polygons = list_polygons(
        data.get("shapes", []),
        max_polygons=2,
        context=str(json_path),
    )
    while len(polygons) < 2:
        polygons.append([])

    processed: List[List[List[float]]] = []
    for polygon in polygons[:2]:
        resampled = resample_polygon(polygon, num_points)
        processed.append(np.asarray(resampled, dtype=float).T.tolist())
    return processed


def process_sequence(sequence_path: Path, num_points: int) -> Dict[str, List[List[List[float]]]]:
    """Process every frame JSON in a sequence directory."""
    frames_keypoints: List[List[List[List[float]]]] = []
    for json_file in sorted(sequence_path.glob("*.json")):
        keypoints = extract_polygons_from_frame(json_file, num_points)
        if keypoints is None:
            logger.warning(
                "Skipping frame %s due to extraction error.", json_file)
            continue
        frames_keypoints.append(keypoints)
    return {"keypoints": frames_keypoints}


def build_group_sequences(group_folder: Path, num_points: int) -> Dict[str, Dict[str, object]]:
    """Convert a group folder (e.g., train or test) into a mapping of sequences -> keypoints."""
    sequences: Dict[str, Dict[str, object]] = {}
    for seq_path in sorted([p for p in group_folder.iterdir() if p.is_dir()]):
        logger.info("Processing sequence '%s' in '%s'",
                    seq_path.name, group_folder)
        sequences[seq_path.name] = process_sequence(seq_path, num_points)
    return sequences


def save_group_dataset(group_label: str, sequences: Dict[str, Dict[str, object]], output_folder: Path) -> None:
    """Persist each sequence dictionary as a .npy file following the CalMS21 naming convention."""
    for seq, sequence_data in sequences.items():
        file_name = f"calms21_{group_label}_{seq}.npy"
        output_path = output_folder / file_name
        np.save(output_path, {group_label: {
                seq: sequence_data}}, allow_pickle=True)
        logger.info("Saved sequence '%s' (group: '%s') to '%s'",
                    seq, group_label, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a CalMS21-compatible dataset from JSON polygon files and save as .npy files. "
            "Each frame is represented as a 2x2xnum_points array (2 instances, 2 coordinates, fixed number of points)."
        )
    )
    parser.add_argument("--train_folder", type=str,
                        required=True, help="Path to the train folder")
    parser.add_argument("--test_folder", type=str,
                        required=True, help="Path to the test folder")
    parser.add_argument("--output_folder", type=str,
                        default="data", help="Directory to output npy files")
    parser.add_argument("--num_points", type=int, default=10,
                        help="Number of points to sample each polygon to")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, etc.)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    train_sequences = build_group_sequences(
        Path(args.train_folder), args.num_points)
    test_sequences = build_group_sequences(
        Path(args.test_folder), args.num_points)

    save_group_dataset("train", train_sequences, output_folder)
    save_group_dataset("test", test_sequences, output_folder)


if __name__ == "__main__":
    main()
