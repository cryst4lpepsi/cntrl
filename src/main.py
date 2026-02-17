from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QSpinBox,
    QTabWidget, QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap

from midi_controller import MIDIController
from gesture_processor import GestureProcessor
from visual_effects import VisualEffectsRenderer
from hand_tracker import HandTracker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)


# ----------------------------
# Camera Worker Thread
# ----------------------------

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = int(camera_index)
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        # stable defaults
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            print("[Camera] Failed to open camera.")
            return

        self.running = True

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            self.frame_ready.emit(frame)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(700)


# ----------------------------
# Tracking Worker Thread
# ----------------------------

class TrackingThread(QThread):
    """
    Runs hand tracking at limited FPS on downscaled frames.
    Keeps UI smooth.
    """
    tracking_ready = Signal(object)  # (landmarks, handedness, fingertips, pinch)

    def __init__(self, hand_tracker: HandTracker, fps: int = 12, size=(320, 180)):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.running = False

        self.interval = 1.0 / float(max(1, fps))
        self.size = (int(size[0]), int(size[1]))

        self.latest_frame: Optional[np.ndarray] = None
        self._last_time = 0.0

    def set_frame(self, frame: np.ndarray):
        self.latest_frame = frame

    def run(self):
        self.running = True
        while self.running:
            now = time.time()
            if (now - self._last_time) < self.interval:
                time.sleep(0.002)
                continue

            self._last_time = now

            if self.latest_frame is None:
                continue

            # Downscale for tracking (massive speedup)
            small = cv2.resize(self.latest_frame, self.size, interpolation=cv2.INTER_AREA)

            # Detect
            res = self.hand_tracker.detect(small)

            fingertips: List[Tuple[float, float]] = []
            pinch = 0.0

            if res.landmarks:
                lms = res.landmarks[0]
                tip_ids = [4, 8, 12, 16, 20]

                for tid in tip_ids:
                    if tid < len(lms):
                        fingertips.append((float(lms[tid]["x"]), float(lms[tid]["y"])))

                # pinch strength
                if len(lms) > 8:
                    dx = float(lms[4]["x"]) - float(lms[8]["x"])
                    dy = float(lms[4]["y"]) - float(lms[8]["y"])
                    dist = (dx * dx + dy * dy) ** 0.5
                    pinch = float(1.0 - np.clip(dist / 0.2, 0.0, 1.0))

            self.tracking_ready.emit((res.landmarks, res.handedness, fingertips, pinch))

    def stop(self):
        self.running = False
        self.wait(700)


# ----------------------------
# Main Window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Gesture Knob Controller (FL Studio)")
        self.setMinimumSize(1280, 780)

        # modules
        self.midi = MIDIController()
        self.gesture_processor = GestureProcessor()
        self.effects = VisualEffectsRenderer()
        self.hand_tracker = HandTracker(max_num_hands=2)

        # state
        self.tracking_enabled = False

        # tracking data cache
        self.latest_landmarks = []
        self.latest_handedness = []
        self.latest_fingertips: List[Tuple[float, float]] = []
        self.latest_pinch = 0.0

        # camera
        self.camera_index = 0
        self.camera_thread: Optional[CameraThread] = None
        self.latest_frame: Optional[np.ndarray] = None

        # tracking thread
        self.tracking_thread = TrackingThread(self.hand_tracker, fps=12, size=(320, 180))
        self.tracking_thread.tracking_ready.connect(self.on_tracking_data)
        self.tracking_thread.start()

        # mappings
        self.mappings: Dict[str, Any] = self.load_mappings()

        # UI
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_live = QWidget()
        self.tab_midi = QWidget()
        self.tab_mapping = QWidget()
        self.tab_visuals = QWidget()

        self.tabs.addTab(self.tab_live, "Live")
        self.tabs.addTab(self.tab_midi, "MIDI + Camera")
        self.tabs.addTab(self.tab_mapping, "Mapping")
        self.tabs.addTab(self.tab_visuals, "Visuals")

        self.build_live_tab()
        self.build_midi_tab()
        self.build_mapping_tab()
        self.build_visuals_tab()

        # startup
        self.detect_cameras()
        self.refresh_midi_ports()
        self.start_camera_preview()

        # status updater
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_status_text)
        self.ui_timer.start(250)

    # ----------------------------
    # UI Tabs
    # ----------------------------

    def build_live_tab(self):
        layout = QVBoxLayout()

        self.video_label = QLabel("Camera preview...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background: #111; color: #ddd; border-radius: 14px; padding: 12px;"
        )
        layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()

        self.btn_start = QPushButton("Start Tracking")
        self.btn_stop = QPushButton("Stop Tracking")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_tracking)
        self.btn_stop.clicked.connect(self.stop_tracking)

        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)

        self.status_label = QLabel("Status: Preview Mode")
        controls.addWidget(self.status_label, stretch=1)

        layout.addLayout(controls)
        self.tab_live.setLayout(layout)

    def build_midi_tab(self):
        layout = QVBoxLayout()

        # Camera group
        cam_group = QGroupBox("Camera Selection")
        cam_layout = QVBoxLayout()

        self.camera_combo = QComboBox()
        self.btn_refresh_cams = QPushButton("Refresh Cameras")
        self.btn_refresh_cams.clicked.connect(self.detect_cameras)

        cam_layout.addWidget(QLabel("Select camera (preview starts automatically):"))
        cam_layout.addWidget(self.camera_combo)
        cam_layout.addWidget(self.btn_refresh_cams)
        cam_group.setLayout(cam_layout)

        # MIDI group
        midi_group = QGroupBox("MIDI Output")
        midi_layout = QVBoxLayout()

        self.midi_port_combo = QComboBox()
        self.btn_refresh_midi = QPushButton("Refresh MIDI Ports")
        self.btn_refresh_midi.clicked.connect(self.refresh_midi_ports)

        midi_layout.addWidget(QLabel("Select MIDI Output Port:"))
        midi_layout.addWidget(self.midi_port_combo)
        midi_layout.addWidget(self.btn_refresh_midi)

        midi_group.setLayout(midi_layout)

        layout.addWidget(cam_group)
        layout.addWidget(midi_group)
        layout.addStretch(1)

        self.tab_midi.setLayout(layout)

        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

    def build_mapping_tab(self):
        layout = QVBoxLayout()

        info = QLabel(
            "Mapping controls what gesture drives which MIDI CC.\n"
            "Recommended: Index Tip (8) Y Position = knob up/down."
        )
        layout.addWidget(info)

        grid = QGridLayout()

        self.mapping_rows = []
        default_points = [8, 4, 12, 16, 20]
        default_ccs = [14, 74, 71, 73, 10]

        for row in range(5):
            point_spin = QSpinBox()
            point_spin.setRange(0, 20)
            point_spin.setValue(default_points[row])

            gesture_combo = QComboBox()
            gesture_combo.addItems(["Y Position", "X Position", "Pinch"])

            cc_spin = QSpinBox()
            cc_spin.setRange(0, 127)
            cc_spin.setValue(default_ccs[row])

            enable_check = QCheckBox("Enabled")
            enable_check.setChecked(True)

            invert_check = QCheckBox("Invert")
            invert_check.setChecked(False)

            grid.addWidget(QLabel(f"Slot {row+1}"), row, 0)
            grid.addWidget(QLabel("Point"), row, 1)
            grid.addWidget(point_spin, row, 2)
            grid.addWidget(QLabel("Gesture"), row, 3)
            grid.addWidget(gesture_combo, row, 4)
            grid.addWidget(QLabel("CC"), row, 5)
            grid.addWidget(cc_spin, row, 6)
            grid.addWidget(enable_check, row, 7)
            grid.addWidget(invert_check, row, 8)

            self.mapping_rows.append((point_spin, gesture_combo, cc_spin, enable_check, invert_check))

        layout.addLayout(grid)

        btns = QHBoxLayout()
        self.btn_save_map = QPushButton("Save Mapping")
        self.btn_load_map = QPushButton("Reload Mapping")
        self.btn_save_map.clicked.connect(self.save_mappings_from_ui)
        self.btn_load_map.clicked.connect(self.load_mappings_into_ui)

        btns.addWidget(self.btn_save_map)
        btns.addWidget(self.btn_load_map)
        btns.addStretch(1)

        layout.addLayout(btns)
        layout.addStretch(1)

        self.tab_mapping.setLayout(layout)
        self.load_mappings_into_ui()

    def build_visuals_tab(self):
        layout = QVBoxLayout()

        self.effects_combo = QComboBox()
        self.effects_combo.addItems([
            "Off",
            "Particle Cloud (CPU)",
            "Neon Trails (CPU)",
            "Wireframe Mesh (CPU)",
            "Holographic (CPU)"
        ])
        self.effects_combo.currentIndexChanged.connect(self.on_effect_changed)

        layout.addWidget(QLabel("Choose Visual Effect:"))
        layout.addWidget(self.effects_combo)

        layout.addStretch(1)
        self.tab_visuals.setLayout(layout)

    # ----------------------------
    # Visuals handlers
    # ----------------------------

    def on_effect_changed(self):
        mode = self.effects_combo.currentText()

        if mode == "Off":
            self.effects.enabled = False
            return

        self.effects.enabled = True
        self.effects.effect_style = mode.replace(" (CPU)", "")

    # ----------------------------
    # Camera / MIDI helpers
    # ----------------------------

    def detect_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        found = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                label = f"Camera {i}"
                if w and h:
                    label += f" ({w}x{h})"
                found.append((i, label))

        if not found:
            self.camera_combo.addItem("No cameras found", 0)
        else:
            for idx, label in found:
                self.camera_combo.addItem(label, idx)

        self.camera_combo.blockSignals(False)

    def refresh_midi_ports(self):
        self.midi_port_combo.clear()
        ports = self.midi.get_available_ports()
        for p in ports:
            self.midi_port_combo.addItem(p)

    def on_camera_changed(self):
        idx = self.camera_combo.currentData()
        if idx is None:
            return
        idx = int(idx)
        if idx == self.camera_index:
            return

        self.camera_index = idx
        self.restart_camera_preview()

    def start_camera_preview(self):
        if self.camera_thread and self.camera_thread.isRunning():
            return

        self.camera_thread = CameraThread(self.camera_index)
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.start()

    def restart_camera_preview(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.start_camera_preview()

    # ----------------------------
    # Tracking data callback
    # ----------------------------

    def on_tracking_data(self, payload):
        landmarks, handedness, fingertips, pinch = payload
        self.latest_landmarks = landmarks
        self.latest_handedness = handedness
        self.latest_fingertips = fingertips
        self.latest_pinch = pinch

    # ----------------------------
    # Tracking control
    # ----------------------------

    def start_tracking(self):
        port = self.midi_port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "MIDI", "No MIDI port selected.")
            return

        if not self.midi.is_connected():
            ok = self.midi.connect(port)
            if not ok:
                QMessageBox.warning(
                    self,
                    "MIDI",
                    "Failed to connect to MIDI port.\n\n"
                    "Recommended: install loopMIDI and create a port named 'GestureKnob'."
                )
                return

        self.tracking_enabled = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_tracking(self):
        self.tracking_enabled = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ----------------------------
    # Frame processing
    # ----------------------------

    def on_frame(self, frame: np.ndarray):
        """
        Called from camera thread.
        """
        self.latest_frame = frame

        # feed tracking thread
        self.tracking_thread.set_frame(frame)

        # MIDI output only when tracking enabled
        if self.tracking_enabled and self.latest_landmarks:
            cc_values = self.gesture_processor.process(
                self.latest_landmarks,
                self.latest_handedness,
                self.mappings
            )
            for cc, val in cc_values.items():
                self.midi.send_cc(cc, val)

        # CPU effects
        mode = self.effects_combo.currentText()

        if mode == "Off":
            out = frame
        else:
            hand_data = {
                "landmarks": self.latest_landmarks if self.tracking_enabled else [],
                "handedness": self.latest_handedness
            }
            out = self.effects.render(frame, hand_data)

        self.display_frame(out)

    def display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(pix)

    # ----------------------------
    # Mapping storage
    # ----------------------------

    def mappings_path(self) -> Path:
        return CONFIG_DIR / "mapping.json"

    def load_mappings(self) -> Dict[str, Any]:
        path = self.mappings_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_mappings_from_ui(self):
        mappings = {}

        for row in self.mapping_rows:
            point_spin, gesture_combo, cc_spin, enable_check, invert_check = row
            point_id = int(point_spin.value())

            mappings[str(point_id)] = {
                "gesture_type": gesture_combo.currentText(),
                "cc_number": int(cc_spin.value()),
                "enabled": bool(enable_check.isChecked()),
                "invert": bool(invert_check.isChecked()),
                "min_value": 0,
                "max_value": 127,
                "hand": "Any"
            }

        self.mappings = mappings
        self.mappings_path().write_text(json.dumps(mappings, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Saved", "Mapping saved to config/mapping.json")

    def load_mappings_into_ui(self):
        self.mappings = self.load_mappings()

        for row in self.mapping_rows:
            point_spin, gesture_combo, cc_spin, enable_check, invert_check = row
            pid = str(point_spin.value())

            if pid in self.mappings:
                cfg = self.mappings[pid]
                gesture_combo.setCurrentText(cfg.get("gesture_type", "Y Position"))
                cc_spin.setValue(int(cfg.get("cc_number", 14)))
                enable_check.setChecked(bool(cfg.get("enabled", True)))
                invert_check.setChecked(bool(cfg.get("invert", False)))

    # ----------------------------
    # Status / close
    # ----------------------------

    def update_status_text(self):
        cam = self.camera_index
        track = "Tracking ON" if self.tracking_enabled else "Preview"
        midi = "Connected" if self.midi.is_connected() else "Not connected"
        fx = self.effects_combo.currentText()
        self.status_label.setText(f"Status: {track} | Cam {cam} | MIDI: {midi} | FX: {fx}")

    def closeEvent(self, event):
        try:
            self.tracking_enabled = False

            if self.camera_thread:
                self.camera_thread.stop()

            if self.tracking_thread:
                self.tracking_thread.stop()

            self.hand_tracker.close()
            self.midi.close()
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
