import json
from pathlib import Path
import numpy as np
from typing import Dict


class CalMS21ToLabelme:
    """Convert CalMS21 dataset to LabelMe format.
    Reference: https://data.caltech.edu/records/s0vdx-0k302
    """

    def __init__(self, calms21_json_path: str, output_dir: str):
        """Initialize the converter.

        Args:
            calms21_json_path: Path to CalMS21 JSON file
            output_dir: Directory to save LabelMe JSON files
        """
        self.calms21_json_path = Path(calms21_json_path)
        self.output_dir = Path(output_dir)
        self.image_width = 1024  # Original image dimensions
        self.image_height = 570
        self.body_parts = [
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "left_hip",
            "right_hip",
            "tail_base",
        ]
        self.video_name = None
        self.behavior_mapping = {
            0: "attack",
            1: "investigation",
            2: "mount",
            3: "other",
        }

    def convert(self):
        """Convert CalMS21 JSON to LabelMe format."""
        # Load CalMS21 JSON
        with open(self.calms21_json_path) as f:
            calms21_data = json.load(f)

        # Process each video
        for annotator_id, videos in calms21_data.items():
            for video_path, video_data in videos.items():
                # Create output directory for this video
                video_name = Path(video_path).name
                self.video_name = video_name
                video_dir = self.output_dir / video_name
                video_dir.mkdir(parents=True, exist_ok=True)

                keypoints = video_data.get("keypoints", [])
                scores = video_data.get("scores", [])
                annotations = video_data.get("annotations", [])

                # Process each frame
                for frame_idx, (frame_keypoints, frame_scores) in enumerate(
                    zip(keypoints, scores)
                ):
                    labelme_json = self._create_labelme_json(
                        frame_keypoints,
                        frame_scores,
                        annotations[frame_idx] if annotations else None,
                        video_data.get("metadata", {}),
                        frame_idx,
                    )

                    # Save LabelMe JSON
                    json_path = video_dir / f"{self.video_name}_{frame_idx:09d}.json"
                    with open(json_path, "w") as f:
                        json.dump(labelme_json, f, indent=2)

    def _create_labelme_json(
        self,
        frame_keypoints: np.ndarray,
        frame_scores: np.ndarray,
        behavior_id: int = None,
        metadata: Dict = None,
        frame_idx: int = 0,
    ) -> Dict:
        """Create LabelMe JSON for a single frame."""
        shapes = []

        # Process each mouse (0: resident/black, 1: intruder/white)
        for mouse_idx in range(2):
            mouse_name = "resident" if mouse_idx == 0 else "intruder"

            # Add keypoints for each body part
            for part_idx, part_name in enumerate(self.body_parts):
                x, y = (
                    frame_keypoints[mouse_idx][0][part_idx],
                    frame_keypoints[mouse_idx][1][part_idx],
                )
                confidence = frame_scores[mouse_idx][part_idx]

                shape = {
                    "label": f"{mouse_name}_{part_name}",
                    "points": [[float(x), float(y)]],
                    "group_id": mouse_idx,
                    "shape_type": "point",
                    "flags": {},
                    "description": f"{confidence:.4f}",
                    "visible": True,
                }
                shapes.append(shape)

        return {
            "version": "5.5.0",
            "flags": {self.behavior_mapping.get(behavior_id, "unknown"): True}
            if behavior_id is not None
            else {},
            "shapes": shapes,
            "imagePath": "",
            "imageData": None,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CalMS21 to LabelMe format")
    parser.add_argument("calms21_json", help="Path to CalMS21 JSON file")
    parser.add_argument("output_dir", help="Output directory for LabelMe JSON files")
    args = parser.parse_args()

    converter = CalMS21ToLabelme(args.calms21_json, args.output_dir)
    converter.convert()
