"""
hand_tracker.py
MediaPipe Tasks-based hand tracker (Python 3.12 compatible).

Uses a local .task model file:
assets/models/hand_landmarker.task
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


@dataclass
class HandTrackingResult:
    landmarks: List[List[Dict[str, float]]]
    handedness: List[str]


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)

        # Resolve model path
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "assets" / "models" / "hand_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model file:\n{model_path}\n\n"
                "Download 'hand_landmarker.task' and place it there."
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame: np.ndarray) -> HandTrackingResult:
        if bgr_frame is None or bgr_frame.size == 0:
            return HandTrackingResult([], [])

        rgb = bgr_frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        landmarks_out: List[List[Dict[str, float]]] = []
        handedness_out: List[str] = []

        if result.hand_landmarks:
            for i, hand_lms in enumerate(result.hand_landmarks):
                one_hand = []
                for lm in hand_lms:
                    one_hand.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
                landmarks_out.append(one_hand)

                label = "Right"
                try:
                    if result.handedness and len(result.handedness) > i and len(result.handedness[i]) > 0:
                        label = result.handedness[i][0].category_name or "Right"
                except Exception:
                    label = "Right"

                handedness_out.append(label)

        return HandTrackingResult(landmarks_out, handedness_out)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
