# Hand Gesture MIDI Knob Controller for FL Studio
MediaPipe + PySide6 + Real-time Visualizers + MIDI CC Output

A standalone desktop application that turns hand gestures into MIDI CC knob controls for FL Studio.  
It includes a modern UI, real-time camera preview, gesture-to-knob mapping, and multiple visualizer modes (particle cloud, wireframe, holographic, etc.).

---

## Features

### Live Camera Preview
- Camera preview starts automatically on startup
- Camera selection dropdown (supports multiple webcams)

### Hand Tracking (MediaPipe)
- Tracks up to 2 hands
- Pinch detection support
- Optimized tracking thread (runs at lower FPS on downscaled frames)

### MIDI Knob Control for FL Studio
- Sends MIDI CC values in real-time
- Works using virtual MIDI ports (recommended: loopMIDI on Windows)
- Compatible with FL Studio “Link to Controller”

### Gesture Mapping UI
- Configure multiple mapping slots from UI
- Each slot can map:
  - Landmark point ID (0–20)
  - Gesture type (X position / Y position / Pinch)
  - MIDI CC number (0–127)
  - Invert toggle
  - Enable toggle
- Mapping saved to `config/mapping.json`

### Visualizers (CPU Optimized)
- Particle Cloud (sharp dots / opaque mode)
- Neon Trails
- Wireframe Mesh
- Holographic mode

All effects are optimized using a low-resolution overlay pipeline for smooth performance.

---

## How It Works (Pipeline)

1. Camera thread captures frames continuously
2. Tracking thread runs MediaPipe on a downscaled frame (fast)
3. Landmarks → gesture features → CC values
4. CC values are sent through a MIDI output port
5. Visual effects are rendered on top of the camera feed

---

## Requirements

- Python 3.12+
- Windows 10/11 recommended (tested)
- Webcam (USB or built-in)
- FL Studio
- loopMIDI (recommended for virtual MIDI routing)

---

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/<your-username>/hand-gesture-flstudio-controller.git
cd hand-gesture-flstudio-controller
```
2) Create and activate virtual environment
Windows PowerShell:
```
python -m venv venv
venv\Scripts\activate
```
3) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
Run the App
```
python src/main.py
```
On startup:
camera preview starts automatically
tracking is OFF until you click Start Tracking


FL Studio Setup (MIDI)
Step 1: Install loopMIDI
Download and install loopMIDI.

Create a port named:
```
GestureKnob
```
Step 2: Select MIDI port in the app
Go to the MIDI + Camera tab and select GestureKnob.

Step 3: Connect in FL Studio
In FL Studio:
```
Open Options → MIDI Settings

Enable the loopMIDI port

Right-click any knob → Link to controller

Move your hand gesture (or pinch) to bind it
```
Gesture Mapping Tips
Recommended default mapping:
```
Point 8 (Index Tip) → Y Position → CC 14
This gives the most natural “knob up/down” motion.

Pinch is best for:

filter cutoff snap

FX toggle style controls

burst-based effects
```
Visualizer Modes
Go to the Visuals tab and select:
```
Off

Particle Cloud (CPU)

Neon Trails (CPU)

Wireframe Mesh (CPU)

Holographic (CPU)

Particle Cloud is optimized for:

sharp, larger dots

opaque visibility

stable FPS
