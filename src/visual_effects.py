from __future__ import annotations

import time
import numpy as np
import cv2


class VisualEffectsRenderer:
    """
    Optimized CPU effects.
    Main goal: look cool while staying fast.

    Supported:
      - Particle Cloud (CPU)   (fast + pretty)
      - Neon Trails (CPU)
      - Wireframe Mesh (CPU)
      - Holographic (CPU)
    """

    def __init__(self):
        self.enabled = True
        self.effect_style = "Particle Cloud"

        # --- Particle system (CPU) ---
        self.particle_count = 500
        self.dot_radius = 1
        self.pos = np.random.rand(self.particle_count, 2).astype(np.float32)
        self.vel = (np.random.randn(self.particle_count, 2) * 0.002).astype(np.float32)

        self.last_time = time.time()
        self.prev_pinch = 0.0

        # Trail buffer (low-res)
        self.trail_w = 640
        self.trail_h = 360
        self.trail = np.zeros((self.trail_h, self.trail_w, 3), dtype=np.float32)

    # ---------------------------
    # Public API
    # ---------------------------

    def render(self, frame_bgr: np.ndarray, hand_data: dict) -> np.ndarray:
        if not self.enabled:
            return frame_bgr

        style = (self.effect_style or "").strip()

        if style in ["Particle Cloud", "Particle Cloud (CPU)"]:
            return self._render_particle_cloud(frame_bgr, hand_data)

        if style in ["Neon Trails", "Neon Trails (CPU)"]:
            return self._render_neon_trails(frame_bgr, hand_data)

        if style in ["Wireframe Mesh", "Wireframe Mesh (CPU)"]:
            return self._render_wireframe(frame_bgr, hand_data)

        if style in ["Holographic", "Holographic (CPU)"]:
            return self._render_holographic(frame_bgr, hand_data)

        return frame_bgr

    # ---------------------------
    # Helpers: hand info
    # ---------------------------

    def _extract_fingertips_and_pinch(self, hand_data: dict):
        """
        Returns:
          fingertips_norm: list of (x,y) normalized 0..1
          pinch: 0..1
        """
        landmarks = hand_data.get("landmarks", [])
        if not landmarks:
            return [], 0.0

        lms = landmarks[0]
        tip_ids = [4, 8, 12, 16, 20]

        tips = []
        for tid in tip_ids:
            if tid < len(lms):
                tips.append((float(lms[tid]["x"]), float(lms[tid]["y"])))

        pinch = 0.0
        if len(lms) > 8:
            dx = float(lms[4]["x"]) - float(lms[8]["x"])
            dy = float(lms[4]["y"]) - float(lms[8]["y"])
            dist = (dx * dx + dy * dy) ** 0.5
            pinch = float(1.0 - np.clip(dist / 0.2, 0.0, 1.0))

        return tips, pinch

    # ---------------------------
    # Effect 1: Particle Cloud (FAST)
    # ---------------------------

    def _render_particle_cloud(self, frame_bgr: np.ndarray, hand_data: dict) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # Work on a small overlay (massive speedup)
        ow, oh = self.trail_w, self.trail_h

        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        dt = float(np.clip(dt, 0.001, 0.05))

        fingertips, pinch = self._extract_fingertips_and_pinch(hand_data)

        # Pinch burst trigger
        burst = (pinch > 0.78 and self.prev_pinch <= 0.78)
        self.prev_pinch = pinch

        # Fade trails slowly
        self.trail *= 0.32

        # Base drift noise
        self.vel += (np.random.randn(self.particle_count, 2).astype(np.float32) * 0.0006)

        # Attraction to index fingertip (stable)
        if len(fingertips) > 0:
            ax, ay = fingertips[1] if len(fingertips) > 1 else fingertips[0]
            attract = np.array([ax, ay], dtype=np.float32)

            dirv = attract - self.pos
            dist = np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-6
            dn = dirv / dist

            # attraction strength
            self.vel += dn * (0.010 + pinch * 0.018) * dt

            # swirl
            swirl = np.concatenate([-dn[:, 1:2], dn[:, 0:1]], axis=1)
            self.vel += swirl * (0.008 + pinch * 0.012) * dt

        # Burst
        if burst:
            rand_dir = np.random.randn(self.particle_count, 2).astype(np.float32)
            rand_dir /= (np.linalg.norm(rand_dir, axis=1, keepdims=True) + 1e-6)
            self.vel += rand_dir * 0.08

        # Damping
        self.vel *= (0.92 + pinch * 0.04)

        # Update
        self.pos += self.vel * (60.0 * dt)

        # Wrap around screen
        self.pos %= 1.0

        # Draw particles on overlay
        px = (self.pos[:, 0] * (ow - 1)).astype(np.int32)
        py = (self.pos[:, 1] * (oh - 1)).astype(np.int32)

        # Clamp safety
        px = np.clip(px, 0, ow - 1)
        py = np.clip(py, 0, oh - 1)

        # Speed controls brightness
        sp = np.linalg.norm(self.vel, axis=1)
        sp = np.clip(sp * 60.0, 0.0, 1.0)

        # Neon colors: cyan -> magenta
        # Create RGB values
        r = (0.6 + 0.8 * sp).astype(np.float32)
        g = (0.2 + 0.9 * (1.0 - sp)).astype(np.float32)
        b = (0.8 + 0.7 * sp).astype(np.float32)
# Draw bigger opaque dots using circles (still fast because low-res canvas)
        overlay_img = np.clip(self.trail * 255.0, 0, 255).astype(np.uint8)

        for i in range(self.particle_count):
            x = int(px[i])
            y = int(py[i])

            # solid bright colors
            col = (
                int(255 * b[i]),
                int(255 * g[i]),
                int(255 * r[i]),
            )

            cv2.circle(overlay_img, (x, y), self.dot_radius, col, -1, cv2.LINE_AA)

        # convert back to float trail buffer for fading
        self.trail = overlay_img.astype(np.float32) / 255.0


        # Convert overlay to 0..255
        overlay = np.clip(self.trail * 255.0, 0, 255).astype(np.uint8)

        # Upscale overlay to full frame
        overlay_big = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blend (screen-like)
        out = cv2.addWeighted(frame_bgr, 1.0, overlay_big, 1.00, 0.0)

        return out

    # ---------------------------
    # Effect 2: Neon Trails (simple)
    # ---------------------------

    def _render_neon_trails(self, frame_bgr: np.ndarray, hand_data: dict) -> np.ndarray:
        out = frame_bgr.copy()
        landmarks = hand_data.get("landmarks", [])

        if not landmarks:
            return out

        lms = landmarks[0]
        pts = []
        for i in [4, 8, 12, 16, 20]:
            if i < len(lms):
                pts.append((int(lms[i]["x"] * out.shape[1]), int(lms[i]["y"] * out.shape[0])))

        for p in pts:
            cv2.circle(out, p, 18, (255, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(out, p, 6, (255, 255, 255), -1, cv2.LINE_AA)

        return out

    # ---------------------------
    # Effect 3: Wireframe Mesh
    # ---------------------------

    def _render_wireframe(self, frame_bgr: np.ndarray, hand_data: dict) -> np.ndarray:
        out = frame_bgr.copy()
        landmarks = hand_data.get("landmarks", [])
        if not landmarks:
            return out

        lms = landmarks[0]
        h, w = out.shape[:2]

        # Simple hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        pts = [(int(p["x"] * w), int(p["y"] * h)) for p in lms]

        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)

        return out

    # ---------------------------
    # Effect 4: Holographic
    # ---------------------------

    def _render_holographic(self, frame_bgr: np.ndarray, hand_data: dict) -> np.ndarray:
        b, g, r = cv2.split(frame_bgr)

        # chromatic shift
        r2 = np.roll(r, 6, axis=1)
        b2 = np.roll(b, -6, axis=0)

        out = cv2.merge([b2, g, r2])

        # add scanlines
        h, w = out.shape[:2]
        scan = np.zeros((h, w), dtype=np.uint8)
        scan[::3, :] = 20
        out = cv2.subtract(out, cv2.merge([scan, scan, scan]))

        return out
