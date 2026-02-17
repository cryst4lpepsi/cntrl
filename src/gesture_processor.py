"""
gesture_processor.py
Processes hand landmarks and converts to control values.

Fixes:
- Handles JSON mapping keys (string -> int)
- More stable smoothing
- Safer landmark access
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class GestureProcessor:
    def __init__(self, smoothing_window: int = 6):
        self.smoothing_window = int(max(1, smoothing_window))
        self.value_history: Dict[str, deque] = {}
        self.smoothing_factor = 0.55

    def process(
        self,
        landmarks_list: List[List[Dict]],
        handedness_list: List[str],
        mappings: Dict,
    ) -> Dict[int, int]:
        output_values: Dict[int, int] = {}

        # mapping keys might come as strings from JSON
        normalized_mappings = {}
        for k, v in mappings.items():
            try:
                normalized_mappings[int(k)] = v
            except Exception:
                normalized_mappings[k] = v

        for hand_idx, landmarks in enumerate(landmarks_list):
            handedness = handedness_list[hand_idx] if hand_idx < len(handedness_list) else "Right"

            for point_id, config in normalized_mappings.items():
                try:
                    gesture_type = config.get("gesture_type", "Y Position")
                    cc_number = int(config.get("cc_number", 1))
                    enabled = bool(config.get("enabled", True))

                    if not enabled:
                        continue

                    # if mapping is for specific hand
                    target_hand = config.get("hand", "Any")
                    if target_hand != "Any" and target_hand != handedness:
                        continue

                    raw_value = self._get_gesture_value(landmarks, gesture_type, point_id)
                    midi_value = self._map_to_midi(raw_value, config)

                    # smoothing per (hand, cc)
                    key = f"{handedness}_{cc_number}"
                    smoothed = self._smooth_value(key, midi_value)

                    output_values[cc_number] = smoothed

                except Exception:
                    continue

        return output_values

    def _get_gesture_value(self, landmarks: List[Dict], gesture_type: str, point_id: int) -> float:
        if not landmarks or point_id is None:
            return 0.0

        pid = int(point_id)
        if pid < 0 or pid >= len(landmarks):
            return 0.0

        lm = landmarks[pid]
        x = float(lm.get("x", 0.0))
        y = float(lm.get("y", 0.0))
        z = float(lm.get("z", 0.0))

        if gesture_type == "X Position":
            return x
        if gesture_type == "Y Position":
            return 1.0 - y  # invert so up = higher
        if gesture_type == "Z Depth":
            return (z + 0.1) / 0.2  # normalize approx

        if gesture_type == "Pinch":
            # pinch = distance thumb tip (4) to index tip (8)
            if len(landmarks) < 9:
                return 0.0
            t = landmarks[4]
            i = landmarks[8]
            dx = float(t.get("x", 0.0)) - float(i.get("x", 0.0))
            dy = float(t.get("y", 0.0)) - float(i.get("y", 0.0))
            dist = (dx * dx + dy * dy) ** 0.5
            # invert so close pinch -> 1
            val = 1.0 - np.clip(dist / 0.2, 0.0, 1.0)
            return float(val)

        # default
        return 1.0 - y

    def _map_to_midi(self, value: float, config: Dict) -> int:
        value = float(np.clip(value, 0.0, 1.0))

        invert = bool(config.get("invert", False))
        if invert:
            value = 1.0 - value

        # range scaling
        min_val = int(config.get("min_value", 0))
        max_val = int(config.get("max_value", 127))
        min_val = max(0, min(127, min_val))
        max_val = max(0, min(127, max_val))

        if max_val < min_val:
            min_val, max_val = max_val, min_val

        midi = int(min_val + value * (max_val - min_val))
        return max(0, min(127, midi))

    def _smooth_value(self, key: str, value: int) -> int:
        if key not in self.value_history:
            self.value_history[key] = deque(maxlen=self.smoothing_window)

        hist = self.value_history[key]
        hist.append(int(value))

        if len(hist) < 2:
            return int(value)

        # weighted average: recent values matter more
        weights = np.linspace(0.3, 1.0, len(hist))
        arr = np.array(hist, dtype=np.float32)
        smoothed = float(np.sum(arr * weights) / np.sum(weights))
        return int(round(smoothed))
