import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import urllib.request
from pathlib import Path
import subprocess
import sys
import importlib.util

from annolid.utils.logger import logger


@dataclass
class MediaPipeResult:
    """Native MediaPipe result container for Annolid."""

    model_type: str  # 'pose', 'hands', 'face'
    landmarks: List[Any]  # Native landmark objects
    norm_landmarks: List[List[List[float]]]  # [ [ [x, y, z/vis], ... ], ... ]
    world_landmarks: Optional[List[Any]] = None
    segmentation_mask: Optional[np.ndarray] = None
    names: Dict[int, str] = field(default_factory=dict)
    orig_img: Optional[np.ndarray] = None
    distance_cm: Optional[float] = None  # Estimated distance to subject
    metadata: Dict[str, Any] = field(default_factory=dict)

    def plot(self, **kwargs) -> np.ndarray:
        """Draw detections on the original image and return the annotated frame."""
        if self.orig_img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        annotated_frame = self.orig_img.copy()
        h, w = annotated_frame.shape[:2]

        # Try to use native drawing utils if available
        try:
            from mediapipe.tasks.python.vision import drawing_utils
            from mediapipe.tasks.python.vision import face_landmarker
            from mediapipe.tasks.python.vision import hand_landmarker
            from mediapipe.tasks.python.vision import pose_landmarker

            if self.model_type == "hands" and hasattr(self.landmarks, "hand_landmarks"):
                for hand_lms in self.landmarks.hand_landmarks:
                    drawing_utils.draw_landmarks(
                        annotated_frame,
                        hand_lms,
                        hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
                    )
            elif self.model_type == "face" and hasattr(
                self.landmarks, "face_landmarks"
            ):
                for face_lms in self.landmarks.face_landmarks:
                    drawing_utils.draw_landmarks(
                        annotated_frame,
                        face_lms,
                        face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    )

                    # Highlight Iris and eye contours
                    if len(face_lms) > 477:
                        # Landmarks for eyes and iris
                        # Right eye: 33+
                        RIGHT_EYE_CONTOUR = [
                            33,
                            7,
                            163,
                            144,
                            145,
                            153,
                            154,
                            155,
                            133,
                            173,
                            157,
                            158,
                            159,
                            160,
                            161,
                            246,
                        ]
                        # Left eye: 362+
                        LEFT_EYE_CONTOUR = [
                            362,
                            382,
                            381,
                            380,
                            374,
                            373,
                            390,
                            249,
                            263,
                            466,
                            388,
                            387,
                            386,
                            385,
                            384,
                            398,
                        ]
                        LEFT_IRIS = [468, 469, 470, 471, 472]
                        RIGHT_IRIS = [473, 474, 475, 476, 477]

                        def to_pixels(idx):
                            lm = face_lms[idx]
                            return (int(lm.x * w), int(lm.y * h))

                        # 1. Draw eye contours in red
                        for contour in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
                            pts = np.array([to_pixels(i) for i in contour], np.int32)
                            cv2.polylines(
                                annotated_frame,
                                [pts],
                                True,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        # 2. Draw Iris blue circle and green dots
                        for iris_indices in [LEFT_IRIS, RIGHT_IRIS]:
                            center = to_pixels(iris_indices[0])

                            # Draw all key iris points (center, top, bottom, left, right) as green dots
                            for idx in iris_indices:
                                cv2.circle(
                                    annotated_frame, to_pixels(idx), 2, (0, 255, 0), -1
                                )

                            # Draw iris boundary as a blue circle centered at the pupil
                            iris_pts = [to_pixels(i) for i in iris_indices[1:]]
                            if iris_pts:
                                # Average distance from center to boundary points for radius
                                radius = np.mean(
                                    [
                                        np.linalg.norm(np.array(pt) - np.array(center))
                                        for pt in iris_pts
                                    ]
                                )
                                cv2.circle(
                                    annotated_frame,
                                    center,
                                    int(radius),
                                    (255, 0, 0),
                                    1,
                                    cv2.LINE_AA,
                                )

                        if self.distance_cm:
                            text = f"Dist: {self.distance_cm:.1f} cm"
                            cv2.putText(
                                annotated_frame,
                                text,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
            elif self.model_type == "pose" and hasattr(
                self.landmarks, "pose_landmarks"
            ):
                for pose_lms in self.landmarks.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        annotated_frame,
                        pose_lms,
                        pose_landmarker.PoseLandmarker.POSE_CONNECTIONS,
                    )
            else:
                self._fallback_plot(annotated_frame, h, w)
        except Exception as e:
            logger.debug(f"Native drawing failed, using fallback: {e}")
            self._fallback_plot(annotated_frame, h, w)

        return annotated_frame

    def _fallback_plot(self, annotated_frame, h, w):
        for obj_norm_lms in self.norm_landmarks:
            # Draw landmarks
            for pt in obj_norm_lms:
                x, y = int(pt[0] * w), int(pt[1] * h)
                vis = pt[2] if len(pt) > 2 and pt[2] is not None else 1.0
                if vis > 0.3:
                    cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)

            # Draw bounding box
            if obj_norm_lms:
                xs = [p[0] for p in obj_norm_lms]
                ys = [p[1] for p in obj_norm_lms]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                label = self.names.get(0, self.model_type)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )


class MediaPipeEngine:
    """
    Inference engine using MediaPipe Tasks API.
    Provides native MediaPipe results to the perception pipeline.
    """

    MODEL_URLS = {
        "mediapipe_pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "mediapipe_hands": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "mediapipe_face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    }

    def __init__(self, model_identifier: str = "mediapipe_pose"):
        # Lazy import MediaPipe
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp
            self.python = python
            self.vision = vision
        except ImportError:
            raise ImportError(
                "MediaPipe is not installed. Please install it with 'pip install mediapipe' "
                "or use the MediaPipeEngine.install() helper."
            )

        self.model_identifier = model_identifier.lower()
        self.model_path = self._ensure_model_exists()

        base_options = self.python.BaseOptions(model_asset_path=str(self.model_path))

        if "hands" in self.model_identifier:
            options = self.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=self.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
            self.detector = self.vision.HandLandmarker.create_from_options(options)
            self.names = {0: "hand"}
            self.landmark_names = [f"hand_{i}" for i in range(21)]
            self.type = "hands"
        elif "face" in self.model_identifier:
            options = self.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=self.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            self.detector = self.vision.FaceLandmarker.create_from_options(options)
            self.names = {0: "face"}
            self.landmark_names = [f"face_{i}" for i in range(478)]
            self.type = "face"
        else:
            options = self.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=self.vision.RunningMode.IMAGE,
                output_segmentation_masks=True,
                min_pose_detection_confidence=0.5,
            )
            self.detector = self.vision.PoseLandmarker.create_from_options(options)
            self.names = {0: "person"}
            self.landmark_names = [f"pose_{i}" for i in range(33)]
            self.type = "pose"

    @staticmethod
    def is_installed() -> bool:
        """Check if MediaPipe is installed."""
        return importlib.util.find_spec("mediapipe") is not None

    @staticmethod
    def install():
        """Install MediaPipe using pip."""
        logger.info("Attempting to install MediaPipe...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
            logger.info("MediaPipe installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install MediaPipe: {e}")
            return False

    def _ensure_model_exists(self) -> Path:
        """Download the .task model file if it doesn't exist."""
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)

        if "hands" in self.model_identifier:
            model_name = "hand_landmarker.task"
        elif "face" in self.model_identifier:
            model_name = "face_landmarker.task"
        else:
            model_name = "pose_landmarker_heavy.task"

        local_path = models_dir / model_name

        if not local_path.exists():
            url = self.MODEL_URLS.get(
                self.model_identifier, self.MODEL_URLS["mediapipe_pose"]
            )
            logger.info("Downloading MediaPipe model from %s to %s", url, local_path)
            urllib.request.urlretrieve(url, local_path)

        return local_path

    def __call__(self, frame: np.ndarray, **kwargs) -> List[MediaPipeResult]:
        return self.process(frame)

    def process(self, frame: np.ndarray) -> List[MediaPipeResult]:
        """Process a BGR frame and return a MediaPipeResult."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)

        results = self.detector.detect(mp_image)

        norm_landmarks = []
        mask = None
        distance_cm = None

        if self.type == "hands" and results.hand_landmarks:
            metadata_hands = {"hands": []}
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                hand_info = {
                    "index": i,
                    "landmarks": [
                        [lm.x, lm.y, getattr(lm, "presence", 1.0)]
                        for lm in hand_landmarks
                    ],
                }

                # Get handedness
                if hasattr(results, "handedness") and i < len(results.handedness):
                    hand_info["label"] = results.handedness[i][
                        0
                    ].category_name  # "Left" or "Right"
                    hand_info["score"] = results.handedness[i][0].score

                # Basic gesture: Pinch (Thumb tip 4 to Index tip 8)
                p4 = hand_landmarks[4]
                p8 = hand_landmarks[8]
                dist = np.sqrt(
                    (p4.x - p8.x) ** 2 + (p4.y - p8.y) ** 2 + (p4.z - p8.z) ** 2
                )
                # Threshold for pinch can be around 0.05 normalized distance
                hand_info["pinch_score"] = float(dist)
                hand_info["is_pinching"] = bool(dist < 0.05)

                metadata_hands["hands"].append(hand_info)

                # Keep backward compatibility with norm_landmarks
                norm_landmarks.append(hand_info["landmarks"])

            metadata_gaze = metadata_hands  # Reusing variable name for final assignment
        elif self.type == "face" and results.face_landmarks:
            metadata_gaze = {}
            for face_landmarks in results.face_landmarks:
                face_lms_norm = [[lm.x, lm.y, 1.0] for lm in face_landmarks]
                norm_landmarks.append(face_lms_norm)

                # Iris depth estimation
                # Indices: 468-472 (left), 473-477 (right)
                if len(face_landmarks) > 477:
                    try:
                        h, w = frame.shape[:2]

                        # Calculate iris diameter in pixels (average of both eyes)
                        def get_diameter(p1_idx, p3_idx, p2_idx, p4_idx):
                            # Horizontal diameter
                            dx = (
                                face_landmarks[p1_idx].x - face_landmarks[p3_idx].x
                            ) * w
                            dy = (
                                face_landmarks[p1_idx].y - face_landmarks[p3_idx].y
                            ) * h
                            d1 = np.sqrt(dx**2 + dy**2)
                            # Vertical diameter
                            dx2 = (
                                face_landmarks[p2_idx].x - face_landmarks[p4_idx].x
                            ) * w
                            dy2 = (
                                face_landmarks[p2_idx].y - face_landmarks[p4_idx].y
                            ) * h
                            d2 = np.sqrt(dx2**2 + dy2**2)
                            return (d1 + d2) / 2.0

                        # Left iris: 469, 471 (horiz), 470, 472 (vert)
                        d_left = get_diameter(469, 471, 470, 472)
                        # Right iris: 474, 476 (horiz), 475, 477 (vert)
                        d_right = get_diameter(474, 476, 475, 477)

                        diameters = [d for d in [d_left, d_right] if d > 0]
                        if diameters:
                            avg_diameter_px = sum(diameters) / len(diameters)

                            # Focal length approximation (f_px = image_width * 0.75)
                            focal_length_px = w * 0.75
                            # Average human iris is 11.7mm
                            real_iris_diameter_mm = 11.7

                            # Distance D = (F * RealSize) / PixelSize
                            distance_mm = (
                                focal_length_px * real_iris_diameter_mm
                            ) / avg_diameter_px
                            distance_cm = distance_mm / 10.0

                            # Gaze estimation
                            def get_gaze(eye_indices, iris_center_idx):
                                eye_pts = np.array(
                                    [
                                        (face_landmarks[i].x, face_landmarks[i].y)
                                        for i in eye_indices
                                    ]
                                )
                                eye_center = np.mean(eye_pts, axis=0)
                                iris_center = np.array(
                                    [
                                        face_landmarks[iris_center_idx].x,
                                        face_landmarks[iris_center_idx].y,
                                    ]
                                )
                                width = np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])
                                height = np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])
                                gaze_x = (
                                    (iris_center[0] - eye_center[0]) / (width / 2.0)
                                    if width > 0
                                    else 0
                                )
                                gaze_y = (
                                    (iris_center[1] - eye_center[1]) / (height / 2.0)
                                    if height > 0
                                    else 0
                                )
                                return [float(gaze_x), float(gaze_y)]

                            # Indices for eye contours (same as in plot)
                            RIGHT_EYE = [
                                33,
                                7,
                                163,
                                144,
                                145,
                                153,
                                154,
                                155,
                                133,
                                173,
                                157,
                                158,
                                159,
                                160,
                                161,
                                246,
                            ]
                            LEFT_EYE = [
                                362,
                                382,
                                381,
                                380,
                                374,
                                373,
                                390,
                                249,
                                263,
                                466,
                                388,
                                387,
                                386,
                                385,
                                384,
                                398,
                            ]

                            gaze_right = get_gaze(RIGHT_EYE, 473)
                            gaze_left = get_gaze(LEFT_EYE, 468)

                            metadata_gaze = {
                                "gaze_left": gaze_left,
                                "gaze_right": gaze_right,
                                "gaze_avg": [
                                    (gaze_left[0] + gaze_right[0]) / 2.0,
                                    (gaze_left[1] + gaze_right[1]) / 2.0,
                                ],
                            }
                    except Exception as e:
                        logger.debug(f"Iris depth/gaze estimation error: {e}")

        elif self.type == "pose" and results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                norm_landmarks.append(
                    [
                        [lm.x, lm.y, getattr(lm, "presence", lm.visibility)]
                        for lm in pose_landmarks
                    ]
                )
            if results.segmentation_masks:
                mask = results.segmentation_masks[0].numpy_view()

        res = MediaPipeResult(
            model_type=self.type,
            landmarks=results,
            norm_landmarks=norm_landmarks,
            segmentation_mask=mask,
            names=self.names,
            orig_img=frame,
            distance_cm=distance_cm,
            metadata=metadata_gaze if "metadata_gaze" in locals() else {},
        )
        return [res]

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector:
            try:
                self.detector.close()
            except Exception as e:
                logger.debug(f"Error closing MediaPipe detector: {e}")
            finally:
                self.detector = None
