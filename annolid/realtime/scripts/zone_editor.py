# scripts/zone_editor.py
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
import time

# --- Configuration ---
CONFIG_FILE_PATH = Path("../configs/events_config.yaml")
WINDOW_NAME = "Zone Editor"
INSTRUCTIONS = [
    "CONTROLS:",
    "  'n' - Start new zone",
    "  'd' - Finish current zone (check console for name prompt)",
    "  'c' - Clear last point of current zone",
    "  'r' - Reset all unsaved zones",
    "  's' - Save zones to config file",
    "  'q' - Quit without saving",
    "Left-Click: Add point to current zone",
]


class ZoneEditor:
    """
    An interactive OpenCV tool to draw polygonal zones on a video feed
    and save them to a YAML configuration file.
    """

    def __init__(self, camera_index: int, config_path: Path):
        self.camera_index = camera_index
        self.config_path = config_path

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_index}")

        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # State variables
        self.is_drawing = False
        self.current_polygon_pixels = []
        self.zones = []  # List of {'name': str, 'polygon': List[List[float]]}

        self.full_config = {}  # To preserve other config keys like 'events'

        # Load existing zones if the config file exists
        self._load_zones()

    def _load_zones(self):
        """Loads existing zones from the YAML file to allow for editing/adding."""
        if self.config_path.exists():
            print(f"Loading existing configuration from {self.config_path}...")
            with open(self.config_path, "r") as f:
                self.full_config = yaml.safe_load(f) or {}
                self.zones = self.full_config.get("zones", [])
            print(f"Loaded {len(self.zones)} existing zones.")
        else:
            print("No existing config file found. Starting fresh.")
            self.full_config = {"zones": [], "events": []}

    def _save_zones(self):
        """Saves the current zones to the YAML file, preserving other keys."""
        print(f"Saving {len(self.zones)} zones to {self.config_path}...")
        # Ensure the parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Update the 'zones' key in our loaded config
        self.full_config["zones"] = self.zones

        with open(self.config_path, "w") as f:
            yaml.dump(self.full_config, f, sort_keys=False, default_flow_style=False)
        print("Save complete!")
        time.sleep(1)  # Give user time to see the message

    def _pixel_to_normalized(self, point: tuple) -> list:
        """Converts (x, y) pixel coordinates to normalized [x, y] format."""
        x, y = point
        norm_x = round(x / self.frame_width, 4)
        norm_y = round(y / self.frame_height, 4)
        return [norm_x, norm_y]

    def _normalized_to_pixel(self, norm_point: list) -> tuple:
        """Converts normalized [x, y] to pixel (x, y) format."""
        norm_x, norm_y = norm_point
        x = int(norm_x * self.frame_width)
        y = int(norm_y * self.frame_height)
        return (x, y)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles mouse clicks to add points to the current polygon."""
        if event == cv2.EVENT_LBUTTONDOWN and self.is_drawing:
            self.current_polygon_pixels.append((x, y))
            print(f"Added point: ({x}, {y})")

    def run(self):
        """Main loop to display the video feed and handle user input."""
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Create a copy to draw on
            display_frame = frame.copy()

            # Draw existing saved zones
            for zone in self.zones:
                pixel_points = np.array(
                    [self._normalized_to_pixel(p) for p in zone["polygon"]], np.int32
                )
                cv2.polylines(
                    display_frame,
                    [pixel_points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                # Put zone name
                text_pos = pixel_points[0]
                cv2.putText(
                    display_frame,
                    zone["name"],
                    (text_pos[0], text_pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Draw the current polygon being created
            if self.is_drawing and self.current_polygon_pixels:
                # Draw points
                for point in self.current_polygon_pixels:
                    cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
                # Draw connecting lines
                if len(self.current_polygon_pixels) > 1:
                    pts = np.array(self.current_polygon_pixels, np.int32)
                    cv2.polylines(
                        display_frame,
                        [pts],
                        isClosed=False,
                        color=(0, 255, 255),
                        thickness=2,
                    )

            # Display instructions
            for i, line in enumerate(INSTRUCTIONS):
                cv2.putText(
                    display_frame,
                    line,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            # --- Handle Keyboard Input ---
            if key == ord("q"):
                break

            elif key == ord("n"):  # Start new zone
                if not self.is_drawing:
                    self.is_drawing = True
                    self.current_polygon_pixels = []
                    print("\n--- Starting new zone. Click points on the image. ---")
                else:
                    print("Already drawing a zone. Press 'd' to finish it first.")

            elif key == ord("d"):  # Done drawing zone
                if self.is_drawing and len(self.current_polygon_pixels) > 2:
                    self.is_drawing = False
                    print("\n--- Zone shape finished. ---")
                    # Prompt for name in the console
                    zone_name = input(">>> Enter a name for this zone: ").strip()
                    if zone_name:
                        normalized_points = [
                            self._pixel_to_normalized(p)
                            for p in self.current_polygon_pixels
                        ]
                        self.zones.append(
                            {"name": zone_name, "polygon": normalized_points}
                        )
                        self.current_polygon_pixels = []
                        print(f"Zone '{zone_name}' added.")
                    else:
                        print("Invalid name. Zone not saved.")
                elif self.is_drawing:
                    print("A zone must have at least 3 points.")

            elif key == ord("c"):  # Clear last point
                if self.is_drawing and self.current_polygon_pixels:
                    self.current_polygon_pixels.pop()
                    print("Removed last point.")

            elif key == ord("r"):  # Reset
                self.is_drawing = False
                self.current_polygon_pixels = []
                self.zones = self.full_config.get(
                    "zones", []
                )  # Revert to last saved state
                print("Reset all unsaved changes.")

            elif key == ord("s"):  # Save
                self._save_zones()

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Editor closed.")


def main():
    parser = argparse.ArgumentParser(description="Interactive Zone Editor for Annolid")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the camera to use for the feed.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_FILE_PATH),
        help="Path to the events configuration YAML file.",
    )
    args = parser.parse_args()

    editor = ZoneEditor(camera_index=args.camera_index, config_path=Path(args.config))
    editor.run()


if __name__ == "__main__":
    main()
