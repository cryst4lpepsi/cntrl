#!/usr/bin/env python3
"""
Demo Script - Test Hand Gesture Controller without FL Studio
Shows tracking and MIDI output in terminal
"""

import cv2
import mediapipe as mp
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from midi_controller import MIDIController
from gesture_processor import GestureProcessor
from visual_effects import VisualEffectsRenderer


def main():
    """Run demo with visual feedback"""
    print("=" * 60)
    print("Hand Gesture Knob Controller - Demo Mode")
    print("=" * 60)
    print()
    print("This demo shows hand tracking and MIDI output")
    print("Press 'Q' to quit")
    print("Press 'E' to cycle through effects")
    print("Press 'S' to save frame")
    print()
    
    # Initialize components
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    
    # Initialize controllers
    midi = MIDIController()
    processor = GestureProcessor()
    effects = VisualEffectsRenderer()
    
    # Try to connect to MIDI
    ports = midi.get_available_ports()
    if ports:
        print(f"Available MIDI ports: {ports}")
        midi.connect(ports[0])
        print(f"Connected to: {ports[0]}")
    else:
        print("No MIDI ports available - running in demo mode only")
    
    # Sample mapping for demo
    demo_mapping = {
        "8": {
            "gesture_type": "Y-Axis",
            "cc_number": 74,
            "min_value": 0,
            "max_value": 127
        },
        "pinch_index": {
            "gesture_type": "Distance",
            "cc_number": 7,
            "min_value": 0,
            "max_value": 127
        }
    }
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    effect_styles = ["Neon Trails", "Particle System", "Wireframe Mesh", 
                     "Energy Field", "Holographic"]
    current_effect = 0
    
    frame_count = 0
    
    print("\nStarting camera feed...\n")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Prepare hand data
        hand_data = {
            'landmarks': [],
            'handedness': []
        }
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )
                
                # Convert to our format
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    })
                hand_data['landmarks'].append(landmarks)
                
                if results.multi_handedness:
                    hand_data['handedness'].append(
                        results.multi_handedness[idx].classification[0].label
                    )
            
            # Process gestures
            gesture_values = processor.process(
                hand_data['landmarks'],
                hand_data['handedness'],
                demo_mapping
            )
            
            # Send MIDI and display values
            for cc_num, value in gesture_values.items():
                midi.send_cc(cc_num, value)
                
                # Display on frame
                text = f"CC {cc_num}: {value}"
                cv2.putText(frame, text, (10, 30 + (cc_num * 30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Apply visual effects
        frame = effects.render(frame, hand_data)
        
        # Add info overlay
        cv2.putText(frame, f"Effect: {effect_styles[current_effect]}", 
                   (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Q: Quit | E: Change Effect | S: Save", 
                   (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Hand Gesture Controller Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('e') or key == ord('E'):
            current_effect = (current_effect + 1) % len(effect_styles)
            effects.set_effect_style(effect_styles[current_effect])
            print(f"Switched to: {effect_styles[current_effect]}")
        elif key == ord('s') or key == ord('S'):
            filename = f"demo_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")
            frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    midi.disconnect()
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()