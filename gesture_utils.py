from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandDetection:
    box: Optional[Tuple[int, int, int, int]]
    landmarks: Optional[list[Tuple[int, int]]]
    vector: Optional[np.ndarray]


class HandCropper:
    """Detect one hand and return its box and normalized landmark vector."""

    def __init__(self, static_image_mode: bool = False) -> None:
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not HAND_MODEL_PATH.exists():
            print("Downloading MediaPipe hand landmark model...")
            urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)

        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if static_image_mode
            else mp.tasks.vision.RunningMode.VIDEO
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self._static_image_mode = static_image_mode
        self._timestamp_ms = 0
        self._hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> HandDetection:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self._static_image_mode:
            result = self._hands.detect(image)
        else:
            self._timestamp_ms += 33
            result = self._hands.detect_for_video(image, self._timestamp_ms)

        if not result.hand_landmarks:
            return HandDetection(None, None, None)

        landmarks = result.hand_landmarks[0]
        raw_points = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
            dtype=np.float32,
        )
        relative_points = raw_points - raw_points[0]
        scale = np.max(np.linalg.norm(relative_points[:, :2], axis=1))
        if scale <= 1e-6:
            return HandDetection(None, None, None)
        landmark_vector = (relative_points / scale).reshape(-1)
        landmark_points = [
            (
                min(width - 1, max(0, int(landmark.x * width))),
                min(height - 1, max(0, int(landmark.y * height))),
            )
            for landmark in landmarks
        ]
        x_values = [landmark.x * width for landmark in landmarks]
        y_values = [landmark.y * height for landmark in landmarks]

        left = max(0, int(min(x_values)))
        right = min(width, int(max(x_values)))
        top = max(0, int(min(y_values)))
        bottom = min(height, int(max(y_values)))

        return HandDetection(
            (left, top, right, bottom),
            landmark_points,
            landmark_vector,
        )

    def close(self) -> None:
        self._hands.close()
